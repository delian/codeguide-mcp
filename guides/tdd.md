# Test-Driven Development (TDD) Guidelines

Mandatory coding standards and development practices for test-driven development. Language-agnostic testing frameworks, CI/CD pipelines, coverage analysis tools.

---

**Agent Profile**: The TDD Practitioner
**Role**: Senior Software Engineer & Quality Advocate
**Objective**: Generate production-ready, thoroughly tested, and maintainable code through test-first development.
**Tools**: Language-agnostic testing frameworks, CI/CD pipelines, coverage analysis tools

---

## 1. Core Philosophies: TDD-FIRST

The agent must adhere to the **TDD-FIRST** principles for every implementation:

**Red-Green-Refactor**: ALWAYS write failing test → make it pass → refactor. Never skip this cycle.
**Test Before Code**: Write the test before writing implementation code, without exception.
**Regression Shield**: Every bug discovered MUST receive a test before fixing to prevent regression.
**Fast Feedback**: Tests must run quickly to enable rapid iteration.
**Isolated Tests**: Each test must be independent and not rely on execution order.
**Readable Tests**: Tests are documentation; they must be clear and expressive.
**Coverage as Proof**: Aim for >90% code coverage, 100% for critical paths.
**Verified Code**: All tests MUST pass before code delivery; agent-generated code must be verified by running the test suite.

---

## 2. The TDD Cycle (MANDATORY)

### A. The Red-Green-Refactor Loop

**CRITICAL: This cycle is MANDATORY for ALL new code. Never write production code without a failing test first.**

```
1. 🔴 RED: Write a failing test
   ↓
2. 🟢 GREEN: Write minimal code to make it pass
   ↓
3. 🔵 REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### B. Detailed TDD Workflow

#### Step 1: RED - Write a Failing Test

- **MUST**: Write the test FIRST, before any implementation code
- **MUST**: Run the test and verify it FAILS
- **MUST**: Verify it fails for the RIGHT reason (not syntax error, wrong assertion)
- **SHOULD**: Start with the simplest possible test case

```pseudocode
// Example: Testing a calculator add function

TEST "calculator add returns sum of two numbers"
  calculator = new Calculator()
  result = calculator.add(2, 3)
  ASSERT result EQUALS 5
END TEST

// Run test → FAILS (add method doesn't exist yet)
```

#### Step 2: GREEN - Make it Pass

- **MUST**: Write the MINIMUM code to make the test pass
- **MUST**: Not add features or code not required by the current test
- **SHOULD**: Accept "ugly" code initially; refactoring comes next
- **NEVER**: Skip tests or write multiple features at once

```pseudocode
// Minimal implementation to pass the test

CLASS Calculator
  METHOD add(a, b)
    RETURN a + b
  END METHOD
END CLASS

// Run test → PASSES ✓
```

#### Step 3: REFACTOR - Improve the Code

- **MUST**: Keep all tests passing (green) while refactoring
- **MUST**: Run tests after each refactoring step
- **SHOULD**: Improve code structure, remove duplication, enhance readability
- **SHOULD**: Apply design patterns and SOLID principles
- **NEVER**: Add new functionality during refactoring

```pseudocode
// Refactoring: Extract validation, add documentation

CLASS Calculator
  /// Adds two numbers and returns the sum
  METHOD add(a, b)
    validateNumber(a)
    validateNumber(b)
    RETURN a + b
  END METHOD
  
  PRIVATE METHOD validateNumber(value)
    IF NOT isNumber(value)
      THROW TypeError("Value must be a number")
    END IF
  END METHOD
END CLASS

// Run all tests → PASSES ✓
```

### C. When to Write Multiple Tests

- **MUST**: Write one test at a time
- **MUST**: Complete the full Red-Green-Refactor cycle before the next test
- **SHOULD**: List additional test cases as comments or TODOs for clarity

```pseudocode
TEST "calculator add with positive numbers"
  // Test implementation
END TEST

// TODO: Test negative numbers
// TODO: Test zero
// TODO: Test decimal numbers
// TODO: Test large numbers
// TODO: Test type validation
```

---

## 3. Regression Testing (MANDATORY)

### A. The Golden Rule: Bug = Test

**CRITICAL: Every bug discovered MUST be captured in a test BEFORE fixing.**

#### Workflow for Bug Fixes

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
6. 📝 Document the bug and test in commit message
   ↓
7. 🚀 Deploy with confidence (regression prevented)
```

### B. Regression Test Requirements

- **MUST**: Write a test that reproduces the bug before fixing
- **MUST**: Verify the test fails without the fix
- **MUST**: Verify the test passes with the fix
- **MUST**: Keep the regression test in the test suite permanently
- **SHOULD**: Add a comment explaining the bug being prevented
- **SHOULD**: Reference the bug ticket/issue number in the test

```pseudocode
// Regression test for bug #1234

TEST "calculator handles division by zero - Bug #1234"
  // Bug: Division by zero returned Infinity instead of throwing error
  // Discovered: 2026-01-18
  // This test prevents regression of the fix
  
  calculator = new Calculator()
  
  ASSERT THROWS Error WHEN calculator.divide(10, 0)
  ASSERT error.message CONTAINS "Cannot divide by zero"
END TEST
```

### C. Types of Regression Tests

#### 1. Edge Case Bugs

```pseudocode
TEST "handles empty string input - Bug #4567"
  // Bug: Empty strings caused null pointer exception
  parser = new Parser()
  result = parser.parse("")
  ASSERT result EQUALS empty_result_object
END TEST
```

#### 2. Boundary Condition Bugs

```pseudocode
TEST "handles maximum integer value - Bug #7890"
  // Bug: Integer overflow caused incorrect results
  calculator = new Calculator()
  MAX_INT = system.getMaxInt()
  
  ASSERT calculator.isValid(MAX_INT) EQUALS true
  ASSERT THROWS OverflowError WHEN calculator.add(MAX_INT, 1)
END TEST
```

#### 3. Race Condition Bugs

```pseudocode
TEST "concurrent access to shared resource - Bug #3456"
  // Bug: Race condition caused data corruption
  cache = new ThreadSafeCache()
  
  threads = []
  FOR i FROM 1 TO 100
    threads.append(spawn_thread(() => cache.set("key", i)))
  END FOR
  
  wait_for_all(threads)
  
  // Verify no corruption occurred
  value = cache.get("key")
  ASSERT value IS_IN range(1, 100)
  ASSERT cache.size() EQUALS 1
END TEST
```

#### 4. Integration Bugs

```pseudocode
TEST "API handles malformed JSON - Bug #2345"
  // Bug: Server crashed on malformed JSON input
  server = new TestServer()
  
  response = server.post("/api/users", body: "{invalid json")
  
  ASSERT response.status EQUALS 400
  ASSERT response.body CONTAINS "Invalid JSON"
  ASSERT server.isRunning() EQUALS true  // Didn't crash
END TEST
```

### D. Regression Test Documentation

- **MUST**: Include bug ID/ticket number in test name or comment
- **MUST**: Document when the bug was discovered
- **SHOULD**: Describe the bug behavior and expected behavior
- **SHOULD**: Link to the bug report or issue tracker

```pseudocode
/**
 * Regression test for Bug #12345
 * 
 * Issue: User authentication failed when password contained special characters
 * Discovered: 2026-01-15
 * Root Cause: Password validation regex didn't escape special characters
 * 
 * This test ensures that passwords with special characters are properly handled.
 * 
 * @see https://bugtracker.example.com/issues/12345
 */
TEST "authenticate with special characters in password - Bug #12345"
  auth = new AuthService()
  password = "P@ssw0rd!#$%"
  
  result = auth.authenticate("user@example.com", password)
  
  ASSERT result.success EQUALS true
  ASSERT result.user IS_NOT null
END TEST
```

---

## 4. Test Structure & Organization

### A. Test Naming Conventions

- **MUST**: Use descriptive test names that explain behavior
- **SHOULD**: Follow pattern: "method/feature WHEN condition THEN expected result"
- **SHOULD**: Make test names readable as documentation

```pseudocode
// ✅ GOOD - Clear and descriptive
TEST "add returns sum when given two positive numbers"
TEST "divide throws error when divisor is zero"
TEST "authenticate returns null when credentials are invalid"

// ❌ BAD - Vague or unclear
TEST "test1"
TEST "add test"
TEST "it works"
```

### B. Arrange-Act-Assert Pattern (AAA)

**MUST**: Structure all tests using the AAA pattern:

```pseudocode
TEST "transfer funds between accounts successfully"
  // ARRANGE - Set up test data and conditions
  source_account = new Account(balance: 1000)
  target_account = new Account(balance: 500)
  bank = new Bank()
  amount = 200
  
  // ACT - Execute the behavior being tested
  result = bank.transfer(source_account, target_account, amount)
  
  // ASSERT - Verify the expected outcome
  ASSERT result.success EQUALS true
  ASSERT source_account.balance EQUALS 800
  ASSERT target_account.balance EQUALS 700
  ASSERT bank.transaction_log.length EQUALS 1
END TEST
```

### C. Given-When-Then Pattern (BDD Style)

**SHOULD**: Use Given-When-Then for behavior-driven tests:

```pseudocode
TEST "user can purchase item with sufficient balance"
  // GIVEN - Initial state
  user = new User(balance: 100)
  item = new Item(price: 50)
  shop = new Shop()
  
  // WHEN - Action occurs
  result = shop.purchase(user, item)
  
  // THEN - Expected outcome
  ASSERT result.success EQUALS true
  ASSERT user.balance EQUALS 50
  ASSERT user.inventory CONTAINS item
END TEST
```

### D. Test Organization

```
tests/
├── unit/                      # Unit tests (fast, isolated)
│   ├── models/
│   │   ├── user.test
│   │   └── order.test
│   ├── services/
│   │   └── payment.test
│   └── utils/
│       └── validation.test
│
├── integration/               # Integration tests (slower, multiple components)
│   ├── api/
│   │   └── user_api.test
│   └── database/
│       └── user_repository.test
│
├── e2e/                       # End-to-end tests (slowest, full system)
│   ├── user_journey.test
│   └── checkout_flow.test
│
├── regression/                # Regression tests (bug fixes)
│   ├── bug_1234.test
│   └── bug_5678.test
│
└── fixtures/                  # Test data and helpers
    ├── test_data.json
    └── helpers.test
```

---

## 5. Test Types & Coverage

### A. Test Pyramid

```
        /\
       /  \     E2E Tests (Few, Slow, Expensive)
      /    \    Focus: Complete user workflows
     /------\
    /        \  Integration Tests (Some, Medium Speed)
   /          \ Focus: Component interaction
  /------------\
 /              \ Unit Tests (Many, Fast, Cheap)
/________________\ Focus: Individual functions/methods

MUST: Follow the pyramid - many unit tests, fewer integration, few E2E
```

### B. Unit Tests (MANDATORY)

- **MUST**: Test individual functions/methods in isolation
- **MUST**: Run in milliseconds
- **MUST**: Not depend on external systems (databases, APIs, file system)
- **SHOULD**: Mock/stub dependencies

```pseudocode
TEST "calculateTax returns correct tax amount"
  // Pure function, no dependencies
  amount = 100
  tax_rate = 0.15
  
  result = calculateTax(amount, tax_rate)
  
  ASSERT result EQUALS 15
END TEST

TEST "UserService validates email format"
  // Mock email validator
  email_validator = new MockEmailValidator()
  user_service = new UserService(email_validator)
  
  is_valid = user_service.validateEmail("test@example.com")
  
  ASSERT is_valid EQUALS true
  ASSERT email_validator.was_called_with("test@example.com")
END TEST
```

### C. Integration Tests

- **MUST**: Test interaction between components
- **SHOULD**: Test with real dependencies (database, cache, etc.)
- **SHOULD**: Use test databases/environments
- **MUST**: Clean up after each test

```pseudocode
TEST "UserRepository saves and retrieves user from database"
  // ARRANGE - Use test database
  db = new TestDatabase()
  repo = new UserRepository(db)
  user = new User(name: "John", email: "john@example.com")
  
  // ACT
  saved_id = repo.save(user)
  retrieved_user = repo.findById(saved_id)
  
  // ASSERT
  ASSERT retrieved_user.name EQUALS "John"
  ASSERT retrieved_user.email EQUALS "john@example.com"
  
  // CLEANUP
  db.cleanup()
END TEST
```

### D. End-to-End Tests

- **SHOULD**: Test complete user workflows
- **SHOULD**: Run against staging environment
- **MUST**: Be fewer in number (slow and fragile)

```pseudocode
TEST "user completes full checkout process"
  // ARRANGE
  browser = new TestBrowser()
  browser.goto("https://shop.example.com")
  
  // ACT - Complete workflow
  browser.click("Add to Cart")
  browser.goto("/checkout")
  browser.fill("card_number", "4111111111111111")
  browser.fill("expiry", "12/28")
  browser.click("Place Order")
  
  // ASSERT
  success_message = browser.find(".success-message").text
  ASSERT success_message CONTAINS "Order placed successfully"
  
  // Verify in database
  order = database.orders.findLatest()
  ASSERT order.status EQUALS "completed"
END TEST
```

### E. Test Coverage Requirements

- **MUST**: Achieve minimum 90% code coverage
- **MUST**: Achieve 100% coverage for critical paths (payment, security, data loss)
- **MUST**: Test all public APIs/methods
- **SHOULD**: Test private methods indirectly through public API
- **NEVER**: Write code to satisfy coverage without meaningful assertions

```pseudocode
// Coverage report example

File: payment_processor.js
Lines: 98% (245/250)
Branches: 92% (88/96)
Functions: 100% (24/24)

Critical Paths:
- Payment processing: 100% ✓
- Refund handling: 100% ✓
- Error recovery: 95% ⚠️  (Missing: rare edge case)

Status: PASS (meets 90% threshold)
```

---

## 6. Testing Best Practices (MANDATORY)

### A. Test Independence

- **MUST**: Tests must not depend on execution order
- **MUST**: Each test must set up its own data
- **MUST**: Clean up after each test
- **NEVER**: Share mutable state between tests

```pseudocode
// ❌ BAD - Tests share state
shared_counter = 0

TEST "increment counter"
  shared_counter = shared_counter + 1
  ASSERT shared_counter EQUALS 1  // Fails if run after other tests
END TEST

// ✅ GOOD - Tests are independent
TEST "increment counter"
  counter = 0  // Fresh state for each test
  counter = counter + 1
  ASSERT counter EQUALS 1  // Always passes
END TEST
```

### B. Test Data Management

- **MUST**: Use test fixtures for complex data
- **SHOULD**: Use factories or builders for test objects
- **SHOULD**: Keep test data minimal and relevant

```pseudocode
// Factory pattern for test data
FUNCTION createTestUser(overrides = {})
  defaults = {
    id: generate_uuid(),
    name: "Test User",
    email: "test@example.com",
    created_at: now()
  }
  RETURN merge(defaults, overrides)
END FUNCTION

TEST "user can update profile"
  // Use factory with custom data
  user = createTestUser({ name: "John Doe" })
  
  user.updateProfile({ name: "Jane Doe" })
  
  ASSERT user.name EQUALS "Jane Doe"
END TEST
```

### C. Test Speed

- **MUST**: Unit tests run in < 10ms each
- **SHOULD**: Full unit test suite completes in < 10 seconds
- **SHOULD**: Integration tests run in < 1 minute
- **MUST**: Parallelize tests when possible

```pseudocode
// Configuration for parallel test execution

TEST_CONFIG
  parallel: true
  max_workers: 4
  timeout: 5000  // 5 seconds per test
  
  test_groups:
    unit: { pattern: "**/*.unit.test", timeout: 100 }
    integration: { pattern: "**/*.integration.test", timeout: 5000 }
    e2e: { pattern: "**/*.e2e.test", timeout: 30000, parallel: false }
END CONFIG
```

### D. Assertions

- **MUST**: Use specific assertions, not generic ones
- **MUST**: Include meaningful assertion messages
- **SHOULD**: Use one logical assertion per test (may be multiple assert statements)

```pseudocode
// ❌ BAD - Generic assertion
TEST "user creation"
  user = createUser("John", "john@example.com")
  ASSERT user IS_NOT null  // Too vague
END TEST

// ✅ GOOD - Specific assertions
TEST "createUser returns user with correct properties"
  user = createUser("John", "john@example.com")
  
  ASSERT user.name EQUALS "John", "User name should match input"
  ASSERT user.email EQUALS "john@example.com", "Email should match input"
  ASSERT user.id IS_NOT null, "User should have generated ID"
  ASSERT user.created_at <= now(), "Created timestamp should be valid"
END TEST
```

### E. Test Doubles (Mocks, Stubs, Fakes)

- **MUST**: Use test doubles to isolate units under test
- **SHOULD**: Prefer stubs over mocks when possible (less brittle)
- **MUST**: Verify mock expectations

```pseudocode
// Stub - Returns predefined data
STUB EmailService
  METHOD send(to, subject, body)
    RETURN { success: true, message_id: "12345" }
  END METHOD
END STUB

// Mock - Verifies interactions
MOCK EmailService
  EXPECT send(to: "user@example.com", subject: ANY, body: ANY)
    RETURN { success: true, message_id: "12345" }
  END EXPECT
END MOCK

TEST "password reset sends email"
  email_mock = new MockEmailService()
  user_service = new UserService(email_mock)
  
  user_service.resetPassword("user@example.com")
  
  // Verify mock was called correctly
  VERIFY email_mock.send WAS_CALLED_ONCE
  VERIFY email_mock.send WAS_CALLED_WITH(
    to: "user@example.com",
    subject: CONTAINS("Password Reset")
  )
END TEST

// Fake - Working implementation for testing
FAKE Database
  storage = {}
  
  METHOD save(key, value)
    storage[key] = value
    RETURN true
  END METHOD
  
  METHOD find(key)
    RETURN storage[key]
  END METHOD
END FAKE
```

---

## 7. Agent TDD Requirements (MANDATORY)

### A. Code Generation Protocol

**CRITICAL: Agents MUST follow this workflow for ALL code generation:**

```
1. 📝 Analyze Requirements
   - Understand what needs to be built
   - Break into testable behaviors
   
2. ✍️ Write Test First
   - Write failing test for first behavior
   - Verify test compiles/runs (and fails)
   
3. 💻 Write Minimal Implementation
   - Write just enough code to pass test
   - Run test → verify it passes
   
4. 🔄 Refactor
   - Improve code quality
   - Keep tests green
   
5. 🔁 Repeat
   - Next behavior/test
   - Continue until feature complete
   
6. ✅ Verify
   - All tests pass
   - Coverage > 90%
   - No skipped tests
```

### B. Agent Verification Checklist

**Before delivering ANY code, agent MUST verify:**

- [ ] **Test written first**: Every function has a test written before implementation
- [ ] **All tests pass**: Exit code 0 from test runner
- [ ] **Coverage threshold met**: Minimum 90% coverage (100% for critical paths)
- [ ] **No skipped tests**: All tests enabled and running
- [ ] **Fast execution**: Unit tests complete in < 10 seconds
- [ ] **Independent tests**: Tests pass when run in any order or individually
- [ ] **Regression tests exist**: All fixed bugs have permanent regression tests
- [ ] **Test documentation**: Test names are clear and descriptive
- [ ] **AAA pattern used**: All tests follow Arrange-Act-Assert structure
- [ ] **No dead code**: All code is exercised by tests

### C. Test-First Code Example

```pseudocode
// Step 1: Write failing test
TEST "Stack push adds item to top"
  stack = new Stack()
  
  stack.push(42)
  
  ASSERT stack.peek() EQUALS 42
  ASSERT stack.size() EQUALS 1
END TEST

// Run test → FAILS (Stack class doesn't exist)

// Step 2: Write minimal implementation
CLASS Stack
  items = []
  
  METHOD push(item)
    items.append(item)
  END METHOD
  
  METHOD peek()
    RETURN items[items.length - 1]
  END METHOD
  
  METHOD size()
    RETURN items.length
  END METHOD
END CLASS

// Run test → PASSES ✓

// Step 3: Write next test
TEST "Stack pop removes and returns top item"
  stack = new Stack()
  stack.push(42)
  stack.push(100)
  
  popped = stack.pop()
  
  ASSERT popped EQUALS 100
  ASSERT stack.size() EQUALS 1
  ASSERT stack.peek() EQUALS 42
END TEST

// Run test → FAILS (pop method doesn't exist)

// Step 4: Implement pop
CLASS Stack
  // ... existing methods ...
  
  METHOD pop()
    IF items.length EQUALS 0
      THROW Error("Stack is empty")
    END IF
    RETURN items.remove_last()
  END METHOD
END CLASS

// Run all tests → PASSES ✓

// Step 5: Test edge cases
TEST "Stack pop throws error when empty"
  stack = new Stack()
  
  ASSERT THROWS Error WHEN stack.pop()
  ASSERT error.message EQUALS "Stack is empty"
END TEST

// Continue cycle...
```

### D. Prohibited Practices

**NEVER deliver code that:**
- ❌ Has implementation written before tests
- ❌ Has tests that don't fail first
- ❌ Has skipped or commented-out tests
- ❌ Has tests without assertions
- ❌ Has tests that test implementation details
- ❌ Has flaky tests (pass/fail randomly)
- ❌ Has tests dependent on execution order
- ❌ Has tests dependent on external state
- ❌ Has bugs without corresponding regression tests
- ❌ Has coverage < 90% without justification

---

## 8. TDD Anti-Patterns

### A. Testing Anti-Patterns to AVOID

#### 1. Testing Implementation Details

```pseudocode
// ❌ BAD - Tests internal implementation
TEST "user service calls database twice"
  db_mock = new MockDatabase()
  user_service = new UserService(db_mock)
  
  user_service.getUser("123")
  
  ASSERT db_mock.query WAS_CALLED 2 TIMES  // Brittle!
END TEST

// ✅ GOOD - Tests behavior, not implementation
TEST "user service returns user data"
  db_stub = new StubDatabase({ "123": { name: "John" } })
  user_service = new UserService(db_stub)
  
  user = user_service.getUser("123")
  
  ASSERT user.name EQUALS "John"
END TEST
```

#### 2. Test Interdependence

```pseudocode
// ❌ BAD - Tests depend on order
shared_state = null

TEST "create user"  // Must run first
  shared_state = createUser("John")
  ASSERT shared_state IS_NOT null
END TEST

TEST "update user"  // Depends on previous test
  updateUser(shared_state, { name: "Jane" })
  ASSERT shared_state.name EQUALS "Jane"
END TEST

// ✅ GOOD - Independent tests
TEST "create user"
  user = createUser("John")
  ASSERT user.name EQUALS "John"
END TEST

TEST "update user"
  user = createUser("John")  // Own setup
  updateUser(user, { name: "Jane" })
  ASSERT user.name EQUALS "Jane"
END TEST
```

#### 3. Excessive Setup

```pseudocode
// ❌ BAD - Complex setup obscures intent
TEST "user can place order"
  db = new Database()
  db.migrate()
  user = db.createUser({ name: "John", email: "john@test.com" })
  category = db.createCategory({ name: "Books" })
  product = db.createProduct({
    name: "TDD Book",
    price: 29.99,
    category: category,
    inventory: 100
  })
  cart = new Cart(user)
  // ... 20 more lines of setup ...
  
  cart.addItem(product)
  
  ASSERT cart.items.length EQUALS 1
END TEST

// ✅ GOOD - Use helpers and factories
TEST "user can place order"
  user = createTestUser()
  product = createTestProduct()
  cart = new Cart(user)
  
  cart.addItem(product)
  
  ASSERT cart.items.length EQUALS 1
END TEST
```

#### 4. Multiple Assertions on Different Behaviors

```pseudocode
// ❌ BAD - Tests multiple behaviors
TEST "user service"
  service = new UserService()
  
  // Testing creation
  user = service.create("John", "john@test.com")
  ASSERT user.name EQUALS "John"
  
  // Testing update (different behavior!)
  service.update(user.id, { name: "Jane" })
  updated = service.get(user.id)
  ASSERT updated.name EQUALS "Jane"
  
  // Testing deletion (yet another behavior!)
  service.delete(user.id)
  deleted = service.get(user.id)
  ASSERT deleted IS null
END TEST

// ✅ GOOD - One behavior per test
TEST "user service creates user with correct name"
  service = new UserService()
  user = service.create("John", "john@test.com")
  ASSERT user.name EQUALS "John"
END TEST

TEST "user service updates user name"
  service = new UserService()
  user = service.create("John", "john@test.com")
  
  service.update(user.id, { name: "Jane" })
  
  updated = service.get(user.id)
  ASSERT updated.name EQUALS "Jane"
END TEST
```

### B. Design Smells Revealed by Tests

Tests reveal design problems. If tests are hard to write, the design needs improvement:

| Test Problem | Design Issue | Solution |
|-------------|--------------|----------|
| Too much setup required | Too many dependencies | Apply dependency injection |
| Can't test without real database | Tight coupling to infrastructure | Use repository pattern |
| Need to test private methods | Missing abstraction | Extract to separate class |
| Tests are slow | Doing too much | Break into smaller units |
| Hard to mock dependencies | Concrete dependencies | Depend on interfaces |
| Tests break with refactoring | Testing implementation | Test behavior, not details |

---

## 9. CI/CD Integration

### A. Continuous Testing Requirements

- **MUST**: Run tests on every commit
- **MUST**: Block merges if tests fail
- **SHOULD**: Run different test types in parallel
- **SHOULD**: Generate coverage reports

```yaml
# CI/CD Pipeline Example (language-agnostic)

pipeline:
  stages:
    - fast_tests
    - slow_tests
    - coverage_check
    - deploy
  
  fast_tests:
    - name: "Unit Tests"
      command: "run-unit-tests"
      timeout: 60s
      fail_fast: true
      parallel: 4
      
  slow_tests:
    - name: "Integration Tests"
      command: "run-integration-tests"
      timeout: 5m
      depends_on: [fast_tests]
      
    - name: "E2E Tests"
      command: "run-e2e-tests"
      timeout: 15m
      depends_on: [fast_tests]
      
  coverage_check:
    - name: "Coverage Report"
      command: "generate-coverage-report"
      min_coverage: 90%
      fail_below_threshold: true
      depends_on: [fast_tests, slow_tests]
      
  deploy:
    - name: "Deploy to Production"
      command: "deploy-prod"
      depends_on: [coverage_check]
      only_branch: main
```

### B. Quality Gates

- **MUST**: Enforce minimum coverage threshold (90%)
- **MUST**: Fail build if any test fails
- **SHOULD**: Track coverage trends
- **SHOULD**: Fail if coverage decreases

```pseudocode
// Quality gate configuration

quality_gates:
  coverage:
    minimum: 90%
    critical_paths: 100%
    fail_below: true
    allow_decrease: false
    
  tests:
    all_passing: required
    no_skipped: required
    max_duration: 600s  // 10 minutes
    
  code_quality:
    max_complexity: 10
    no_code_smells: true
    
  regression:
    all_bugs_have_tests: required
```

---

## 10. Language-Specific Examples

### A. Python Example

```python
# test_calculator.py
import pytest
from calculator import Calculator

# RED: Write failing test first
def test_add_returns_sum_of_two_numbers():
    calc = Calculator()
    result = calc.add(2, 3)
    assert result == 5

# GREEN: Minimal implementation
# calculator.py
class Calculator:
    def add(self, a, b):
        return a + b

# Regression test for bug
def test_add_handles_negative_numbers_bug_1234():
    """
    Bug #1234: Calculator.add returned incorrect results for negative numbers
    Discovered: 2026-01-15
    Root cause: Missing handling of negative operands
    """
    calc = Calculator()
    
    result = calc.add(-5, -3)
    
    assert result == -8, "Should correctly add negative numbers"
```

### B. JavaScript/TypeScript Example

```javascript
// calculator.test.ts
import { describe, it, expect } from 'vitest';
import { Calculator } from './calculator';

// RED: Write failing test first
describe('Calculator', () => {
  it('should add two numbers correctly', () => {
    const calc = new Calculator();
    const result = calc.add(2, 3);
    expect(result).toBe(5);
  });
  
  // Regression test
  it('should handle division by zero - Bug #5678', () => {
    // Bug: Division by zero returned Infinity instead of throwing
    // Discovered: 2026-01-16
    const calc = new Calculator();
    
    expect(() => calc.divide(10, 0)).toThrow('Cannot divide by zero');
  });
});

// GREEN: Minimal implementation
// calculator.ts
export class Calculator {
  add(a: number, b: number): number {
    return a + b;
  }
  
  divide(a: number, b: number): number {
    if (b === 0) {
      throw new Error('Cannot divide by zero');
    }
    return a / b;
  }
}
```

### C. Java Example

```java
// CalculatorTest.java
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CalculatorTest {
    // RED: Write failing test first
    @Test
    void testAddReturnsSumOfTwoNumbers() {
        Calculator calc = new Calculator();
        int result = calc.add(2, 3);
        assertEquals(5, result);
    }
    
    // Regression test
    @Test
    void testHandlesNullInput_Bug4567() {
        // Bug #4567: NullPointerException when input is null
        // Discovered: 2026-01-17
        // This test prevents regression
        
        Calculator calc = new Calculator();
        
        assertThrows(IllegalArgumentException.class, () -> {
            calc.add(null, 5);
        });
    }
}

// GREEN: Minimal implementation
// Calculator.java
public class Calculator {
    public Integer add(Integer a, Integer b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("Operands cannot be null");
        }
        return a + b;
    }
}
```

### D. Go Example

```go
// calculator_test.go
package calculator

import "testing"

// RED: Write failing test first
func TestAddReturnsSumOfTwoNumbers(t *testing.T) {
    calc := NewCalculator()
    result := calc.Add(2, 3)
    
    if result != 5 {
        t.Errorf("Add(2, 3) = %d; want 5", result)
    }
}

// Regression test
func TestHandlesOverflow_Bug9012(t *testing.T) {
    // Bug #9012: Integer overflow not detected
    // Discovered: 2026-01-18
    // This test ensures overflow is properly handled
    
    calc := NewCalculator()
    
    _, err := calc.Add(MaxInt, 1)
    
    if err == nil {
        t.Error("Expected overflow error, got nil")
    }
}

// GREEN: Minimal implementation
// calculator.go
package calculator

import (
    "errors"
    "math"
)

type Calculator struct{}

func NewCalculator() *Calculator {
    return &Calculator{}
}

func (c *Calculator) Add(a, b int) (int, error) {
    if a > 0 && b > math.MaxInt-a {
        return 0, errors.New("integer overflow")
    }
    return a + b, nil
}
```

### E. Rust Example

```rust
// calculator.rs
pub struct Calculator;

impl Calculator {
    pub fn add(&self, a: i32, b: i32) -> i32 {
        a + b
    }
    
    pub fn divide(&self, a: i32, b: i32) -> Result<i32, String> {
        if b == 0 {
            Err("Cannot divide by zero".to_string())
        } else {
            Ok(a / b)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // RED: Write failing test first
    #[test]
    fn test_add_returns_sum_of_two_numbers() {
        let calc = Calculator;
        let result = calc.add(2, 3);
        assert_eq!(result, 5);
    }
    
    // Regression test
    #[test]
    fn test_divide_by_zero_bug_7890() {
        // Bug #7890: Division by zero panicked instead of returning error
        // Discovered: 2026-01-19
        // This test ensures proper error handling
        
        let calc = Calculator;
        let result = calc.divide(10, 0);
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Cannot divide by zero");
    }
}
```

---

## 11. Deployment Checklist

### Code Quality
- [ ] **All tests pass**: Exit code 0 from test runner
- [ ] **No skipped tests**: All tests are enabled and running
- [ ] **Fast execution**: Unit tests complete in < 10 seconds
- [ ] **Test independence**: Tests pass when run in any order

### Coverage
- [ ] **Minimum coverage met**: ≥ 90% overall coverage
- [ ] **Critical paths covered**: 100% coverage for payment, security, data loss scenarios
- [ ] **Branch coverage**: ≥ 85% of branches covered
- [ ] **No untested code**: All public methods have tests

### TDD Process
- [ ] **Tests written first**: All new code has tests written before implementation
- [ ] **Red-Green-Refactor followed**: Each feature went through full TDD cycle
- [ ] **Regression tests exist**: All bugs have corresponding tests
- [ ] **Bug tests documented**: Regression tests reference bug ID/ticket

### Test Quality
- [ ] **AAA pattern used**: All tests follow Arrange-Act-Assert
- [ ] **Descriptive names**: Test names clearly describe behavior
- [ ] **Single responsibility**: Each test verifies one behavior
- [ ] **No test smells**: No interdependence, excessive setup, or implementation testing

### CI/CD Integration
- [ ] **Tests run on commit**: CI pipeline runs tests automatically
- [ ] **Build blocks on failure**: Failing tests prevent merge
- [ ] **Coverage tracked**: Coverage reports generated and monitored
- [ ] **Quality gates enforced**: Minimum thresholds are enforced

---

## Why TDD Works

1. **Prevents Regressions**: Every bug gets a test, ensuring it never returns (data-backed: 40-80% reduction in production defects).

2. **Better Design**: Writing tests first forces clean interfaces and loose coupling (testable code is maintainable code).

3. **Living Documentation**: Tests document how code should behave better than comments (executable documentation that can't go stale).

4. **Faster Debugging**: When tests fail, you know immediately what broke and where (reduces debugging time by 50-70%).

5. **Confidence to Refactor**: Comprehensive tests enable safe refactoring without fear of breaking functionality.

6. **Reduced Integration Issues**: Early test failures catch integration problems before they cascade (shift-left testing).

7. **Minimal Debugging**: Writing code to pass a specific test means less guesswork and debugging.

8. **Production Stability**: TDD practitioners report 40-90% fewer production incidents compared to test-after approaches.

9. **Faster Development**: Despite seeming slower initially, TDD reduces overall development time by catching issues early.

10. **Better Coverage**: TDD naturally achieves high coverage (typically >90%) without trying, as every line has a reason to exist.

---

## 12. Quick Reference

```
TDD CYCLE:
  1. RED    - Write a failing test (verify it fails for the right reason)
  2. GREEN  - Write minimal code to pass (no extra features)
  3. REFACTOR - Improve code, keep tests green (no new behavior)
  Repeat.

TEST NAMING:
  test_<unit>_<scenario>_<expected_result>
  "should <behavior> when <condition>"

AAA PATTERN (Arrange-Act-Assert):
  // Arrange - Set up test data and dependencies
  // Act     - Execute the behavior under test
  // Assert  - Verify the expected outcome

COVERAGE TARGETS:
  Overall:        >= 90%
  Critical paths: 100% (payments, auth, data loss scenarios)
  Branch:         >= 85%

TEST DOUBLES:
  Stub    - Returns canned data (no assertions on calls)
  Mock    - Verifies interactions (asserts method was called)
  Spy     - Records calls for later assertion
  Fake    - Simplified working implementation (in-memory DB)
  Dummy   - Placeholder, never actually used

BUG FIX PROTOCOL:
  1. Write a test that reproduces the bug (must FAIL)
  2. Verify the test fails for the correct reason
  3. Fix the bug (minimal change)
  4. Verify the test PASSES
  5. Reference bug ID in test name/comment

COMMON ASSERTIONS:
  assertEqual(expected, actual)
  assertTrue(condition)
  assertRaises(ExceptionType, callable)
  assertContains(collection, element)
  assertNull / assertNotNull

ANTI-PATTERNS TO AVOID:
  - Testing implementation details (not behavior)
  - Shared mutable state between tests
  - Tests that depend on execution order
  - Excessive mocking (test the integration)
  - Testing private methods directly
  - Ignoring/skipping failing tests
```

---

## References

- [Test Driven Development: By Example](https://www.pearson.com/en-us/subject-catalog/p/test-driven-development-by-example/P200000009421) - Kent Beck
- [Growing Object-Oriented Software, Guided by Tests](http://www.growing-object-oriented-software.com/) - Steve Freeman & Nat Pryce
- [xUnit Test Patterns](http://xunitpatterns.com/) - Gerard Meszaros
- [The Art of Unit Testing](https://www.manning.com/books/the-art-of-unit-testing-third-edition) - Roy Osherove
- [Working Effectively with Legacy Code](https://www.pearson.com/en-us/subject-catalog/p/working-effectively-with-legacy-code/P200000009462) - Michael Feathers

---

**Last Updated:** 2026-01-18
**Version:** 1.0
**Maintainer:** Development Team

---

**End of Test-Driven Development (TDD) Guidelines**
