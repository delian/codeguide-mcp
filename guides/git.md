# Modern Git Workflow Guidelines
This document provides mandatory standards and best practices for Git usage, commit messages, branching strategies, and version control workflows.

---

**Agent Profile**: The Git Workflow Expert  
**Role**: Senior DevOps Engineer & Version Control Specialist  
**Objective**: Generate clean, traceable, well-documented Git history with proper issue tracking integration.  
**Tools**: Git 2.40+, Conventional Commits, Git Flow, GitHub Flow, GitLab Flow, Pre-commit hooks.

## Core Philosophies

The agent must adhere to the "GIT-FIRST" principles for every Git operation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation, commit tests with implementation.
**Regression Shield**: EVERY bug fix commit MUST reference the bug ID and include regression test.
**Clean History**: Atomic commits, logical grouping, readable history, no "WIP" commits in main branches.
**Linear History**: Prefer rebase over merge for feature branches, squash merge for pull requests.
**Explicit References**: ALL commits MUST reference issues/tickets (bug fixes, features, chores).
**Atomic Commits**: One logical change per commit, easy to revert, easy to review.
**Never Break Build**: Every commit MUST pass tests, compile successfully, maintain working state.

**Conventional Commits**: Follow strict format: `type(scope): description [#issue]`
**Issue Integration**: Every commit references issue ID for traceability and automation.
**Tested Changes**: Commits include both implementation AND tests, never separate.

**Meaningful Messages**: Clear, concise, explains WHY not just WHAT.
**Easy Rollback**: Atomic commits enable safe revert operations.
**Sign Your Work**: Use GPG signatures for commit authentication.
**Safe Rewrites**: Interactive rebase for cleanup, never rewrite public history.
**Automated Checks**: Pre-commit hooks verify format, tests, linting before commit.
**Git Flow Patterns**: Use appropriate branching strategy for project type.
**Efficient Operations**: Shallow clones, sparse checkout, Git LFS for large files.

---

## 1. Commit Message Standards (MANDATORY)

### A. Conventional Commits Format

**ALL commits MUST follow Conventional Commits 1.0.0 specification.**

```
<type>(<scope>): <description> [#<issue-id>]

[optional body]

[optional footer(s)]
```

### B. Commit Types (MANDATORY)

**Use EXACTLY these types:**

- **`feat`**: New feature (user-facing functionality)
- **`fix`**: Bug fix (correlates with PATCH in SemVer)
- **`docs`**: Documentation changes only
- **`style`**: Code style changes (formatting, missing semicolons, no code change)
- **`refactor`**: Code refactoring (no feature change, no bug fix)
- **`perf`**: Performance improvements
- **`test`**: Adding or updating tests (including regression tests)
- **`build`**: Build system changes (Gradle, npm, Webpack, etc.)
- **`ci`**: CI/CD pipeline changes (GitHub Actions, GitLab CI, Jenkins)
- **`chore`**: Other changes (dependency updates, tooling)
- **`revert`**: Reverting a previous commit

### C. Issue Reference Requirements (MANDATORY)

**CRITICAL: ALL commits MUST reference an issue/ticket ID.**

```bash
# ✅ CORRECT - Bug fix with issue reference
git commit -m "fix(auth): prevent null pointer in login handler [#1234]

Fixes #1234

Added null check before accessing user object.
Includes regression test to prevent future occurrence."

# ✅ CORRECT - Feature with issue reference
git commit -m "feat(api): add user search endpoint [#5678]

Implements #5678

- Add GET /api/users/search endpoint
- Support query params: name, email, role
- Include pagination support
- Add integration tests for all query combinations"

# ✅ CORRECT - Multiple issues
git commit -m "feat(dashboard): add analytics widgets [#123, #124]

Implements #123, #124

- Add user activity widget (Fixes #123)
- Add revenue chart widget (Fixes #124)
- Both widgets use shared data service"

# ❌ WRONG - No issue reference
git commit -m "fix: bug in login"

# ❌ WRONG - Vague description
git commit -m "fix(auth): fix bug [#1234]"

# ❌ WRONG - Missing type
git commit -m "update login page [#1234]"
```

### D. Commit Message Template

Create `.gitmessage` template:

```bash
# ~/.gitmessage
<type>(<scope>): <description> [#<issue>]
# |<----  Using a Maximum Of 72 Characters  ---->|

# Explain why this change is being made
# |<----   Try To Limit Each Line to a Maximum Of 72 Characters   ---->|

# Provide links or keys to any relevant tickets, articles or other resources
# Example: Fixes #1234, Implements #5678

# --- COMMIT END ---
# Type can be:
#    feat     (new feature)
#    fix      (bug fix)
#    docs     (documentation)
#    style    (formatting, missing semi colons, etc; no code change)
#    refactor (refactoring production code)
#    test     (adding tests, refactoring test; no production code change)
#    chore    (updating build tasks, package manager configs, etc)
# --------------------
# Remember to:
#   - Reference issue ID with [#issue]
#   - Capitalize the subject line
#   - Use the imperative mood in the subject line
#   - Do not end the subject line with a period
#   - Separate subject from body with a blank line
#   - Use the body to explain what and why vs. how
#   - Can use multiple lines with "-" or "*" for bullet points in body
# --------------------
```

Configure Git to use template:

```bash
git config --global commit.template ~/.gitmessage
```

### E. Bug Fix Commit Requirements (MANDATORY)

**CRITICAL: Bug fix commits have additional requirements.**

```bash
# Bug fix commit MUST include:
# 1. Type: fix
# 2. Issue ID reference
# 3. Regression test in same commit
# 4. Clear description of the bug
# 5. "Fixes #issue" in footer

# ✅ CORRECT - Complete bug fix commit
git commit -m "fix(api): prevent race condition in user creation [#2341]

Fixes #2341

Bug: User creation endpoint would fail when multiple requests
were made simultaneously for the same email, causing duplicate
user records.

Solution: Added distributed lock using Redis before user creation.
Lock is released after transaction commit.

Changes:
- Add Redis distributed lock service
- Wrap user creation in lock acquisition
- Add regression test with concurrent requests
- Update user service tests

Test: test/api/user_creation_concurrent_test.js reproduces
the race condition and verifies the fix."

# Files changed:
# - src/services/user-service.js (implementation + lock)
# - src/services/lock-service.js (new file)
# - test/services/user-service.test.js (regression test)
```

---

## 2. Branching Strategy (MANDATORY)

### A. Branch Naming Convention

**ALL branches MUST follow this naming pattern:**

```
<type>/<issue-id>-<short-description>
```

**Examples:**

```bash
# ✅ CORRECT
feature/1234-user-authentication
bugfix/5678-login-crash
hotfix/9012-security-patch
refactor/3456-api-cleanup
test/7890-integration-tests

# ❌ WRONG
fix-login
user-auth
my-feature
test
```

### B. Git Flow Pattern (Recommended for Libraries/Products)

```
main (production)
  ├── develop (integration)
  │   ├── feature/123-new-widget
  │   ├── feature/456-api-update
  │   └── bugfix/789-form-validation
  ├── release/v1.2.0
  └── hotfix/999-critical-bug
```

**Branch Types:**

- **`main`**: Production-ready code, tagged releases only
- **`develop`**: Integration branch, latest development
- **`feature/*`**: New features, branch from `develop`
- **`bugfix/*`**: Non-critical bugs, branch from `develop`
- **`hotfix/*`**: Critical fixes, branch from `main`
- **`release/*`**: Release preparation, branch from `develop`

**Workflow:**

```bash
# Start new feature
git checkout develop
git pull origin develop
git checkout -b feature/1234-user-search

# Work on feature (multiple commits)
git add src/search.js test/search.test.js
git commit -m "feat(search): add user search endpoint [#1234]"

# Keep up to date
git fetch origin
git rebase origin/develop

# Push feature
git push -u origin feature/1234-user-search

# Create Pull Request: feature/1234-user-search → develop
# After review and approval, squash merge to develop

# Release process
git checkout develop
git checkout -b release/v1.2.0
# Final testing, version bumps, changelog
git checkout main
git merge --no-ff release/v1.2.0
git tag -a v1.2.0 -m "Release version 1.2.0"
git push origin main --tags
```

### C. GitHub Flow Pattern (Recommended for Web Apps/Continuous Deployment)

```
main (production)
  ├── feature/123-new-widget
  ├── bugfix/456-login-fix
  └── hotfix/789-security-patch
```

**Simpler workflow:**

```bash
# All branches from main
git checkout main
git pull origin main
git checkout -b feature/1234-dashboard-widget

# Work and commit
git add src/widget.js test/widget.test.js
git commit -m "feat(dashboard): add analytics widget [#1234]"

# Push and create PR
git push -u origin feature/1234-dashboard-widget

# Create Pull Request: feature/1234-dashboard-widget → main
# After CI passes and review approved, squash merge to main
# Automatic deployment triggered
```

### D. Branch Protection Rules (MANDATORY)

**Configure these protections for `main` and `develop`:**

```yaml
# GitHub branch protection settings
branch_protection:
  main:
    required_status_checks:
      strict: true
      contexts:
        - ci/tests
        - ci/lint
        - ci/security-scan
    required_pull_request_reviews:
      required_approving_review_count: 2
      dismiss_stale_reviews: true
      require_code_owner_reviews: true
    restrictions: null
    enforce_admins: true
    required_linear_history: true
    allow_force_pushes: false
    allow_deletions: false
    required_signatures: true  # GPG signing required
```

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Integrate TDD practices with Git workflows for reliable, traceable development.**

### TDD Cycle with Git Integration

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TDD-GIT INTEGRATED WORKFLOW                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   1. RED PHASE                    2. GREEN PHASE                    │
│   ┌─────────────────┐            ┌─────────────────┐               │
│   │ Write failing   │            │ Write minimal   │               │
│   │ test first      │───────────>│ implementation  │               │
│   │                 │            │                 │               │
│   │ git add tests   │            │ git add src+    │               │
│   │ (stage only)    │            │ tests (atomic)  │               │
│   └─────────────────┘            └────────┬────────┘               │
│          │                                │                         │
│          │ verify test fails              │ verify all tests pass   │
│          │ for right reason               │                         │
│          ↓                                ↓                         │
│   ┌─────────────────┐            ┌─────────────────┐               │
│   │ DO NOT COMMIT   │            │ COMMIT:         │               │
│   │ failing tests   │            │ feat(scope):    │               │
│   │ alone           │            │ description     │               │
│   └─────────────────┘            │ [#issue]        │               │
│                                  └────────┬────────┘               │
│                                           │                         │
│   3. REFACTOR PHASE                       │                         │
│   ┌─────────────────┐                     │                         │
│   │ Improve code    │<────────────────────┘                         │
│   │ quality         │                                               │
│   │                 │                                               │
│   │ Tests must      │                                               │
│   │ remain GREEN    │                                               │
│   └────────┬────────┘                                               │
│            │                                                        │
│            ↓                                                        │
│   ┌─────────────────┐                                               │
│   │ COMMIT:         │                                               │
│   │ refactor(scope):│───────> Push to feature branch               │
│   │ description     │         Create PR                             │
│   │ [#issue]        │                                               │
│   └─────────────────┘                                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Branch Naming for TDD Workflow

**Use descriptive branch names that reflect TDD stages:**

```bash
# Feature development with TDD
feature/<issue-id>-<feature-description>
# Example: feature/1234-user-authentication

# Experimental/spike branches (exploratory TDD)
spike/<issue-id>-<exploration-topic>
# Example: spike/5678-payment-gateway-integration

# Refactoring branches
refactor/<issue-id>-<refactor-description>
# Example: refactor/9012-extract-validation-service
```

### TDD Commit Strategy

**Option 1: Single Atomic Commit (Recommended for small changes)**

```bash
# Complete TDD cycle in one atomic commit
# Tests + Implementation together
git add src/user-service.js test/user-service.test.js
git commit -m "feat(auth): implement password reset flow [#1234]

Implements #1234

TDD Implementation:
- Added failing tests for password reset functionality
- Implemented reset token generation with 24hr expiry
- Added email service integration for reset links
- All tests pass (12 new tests added)

Test coverage: 94% for user-service module"
```

**Option 2: Multi-Commit TDD (For complex features)**

```bash
# Commit 1: RED - Tests that define behavior (may include stub)
git checkout -b feature/1234-payment-processing
git add test/payment.test.js src/payment.js  # stub with NotImplementedError
git commit -m "test(payment): define payment processing behavior [#1234]

Part 1/3: RED phase - Define expected behavior

- Add unit tests for payment validation
- Add integration tests for payment gateway
- Add edge case tests (insufficient funds, invalid card)
- Implementation stubs throw NotImplementedError
- Tests currently fail as expected

Related to #1234"

# Commit 2: GREEN - Minimal implementation
git add src/payment.js src/payment-gateway.js test/payment.test.js
git commit -m "feat(payment): implement payment processing [#1234]

Part 2/3: GREEN phase - Minimal implementation

Implements #1234

- Implement PaymentService with validation
- Add Stripe gateway integration
- Handle error cases with proper exceptions
- All 15 tests now pass

Related to #1234"

# Commit 3: REFACTOR - Clean up
git add src/payment.js src/payment-validator.js
git commit -m "refactor(payment): extract validation logic [#1234]

Part 3/3: REFACTOR phase - Code improvement

- Extract PaymentValidator to separate module
- Add builder pattern for PaymentRequest
- Reduce cyclomatic complexity
- All tests still pass (no behavior change)

Completes #1234"
```

### TDD Branch Workflow Example

```bash
# 1. Create feature branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/2345-user-profile-api

# 2. RED: Write failing tests
# Edit test/user-profile.test.js
npm test -- --grep "UserProfile"  # Verify tests fail
# DO NOT commit failing tests alone

# 3. GREEN: Implement feature
# Edit src/user-profile.js
npm test -- --grep "UserProfile"  # Verify tests pass

# 4. Stage both files atomically
git add src/user-profile.js test/user-profile.test.js

# 5. Commit with proper message
git commit -m "feat(api): add user profile CRUD endpoints [#2345]

Implements #2345

- GET /api/users/:id/profile - Retrieve profile
- PUT /api/users/:id/profile - Update profile
- Profile validation with JSON Schema
- Rate limiting (100 req/min per user)

Tests: 8 unit tests, 4 integration tests
Coverage: 96% for user-profile module"

# 6. REFACTOR: Improve code quality
# Edit src/user-profile.js, src/profile-validator.js
npm test  # Verify all tests still pass

git add src/user-profile.js src/profile-validator.js
git commit -m "refactor(api): extract profile validation [#2345]

- Extract ProfileValidator for reuse
- Add custom validation decorators
- Improve error messages
- All tests pass, no behavior change

Related to #2345"

# 7. Push and create PR
git push -u origin feature/2345-user-profile-api
```

### TDD Commit Message Patterns

```bash
# For RED+GREEN combined (single commit)
git commit -m "feat(<scope>): <description> [#<issue>]

Implements #<issue>

TDD Implementation:
- <test 1 description>
- <test 2 description>
- <implementation summary>
- All tests pass (<N> tests)"

# For REFACTOR phase
git commit -m "refactor(<scope>): <description> [#<issue>]

- <refactoring change 1>
- <refactoring change 2>
- All tests pass (no behavior change)

Related to #<issue>"
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug fix MUST include a regression test. Test BEFORE fix.**

### Bug Fix Git Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     BUG FIX GIT WORKFLOW                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   1. BUG REPORTED                                                   │
│   ┌─────────────────┐                                               │
│   │ Issue created   │                                               │
│   │ #<bug-id>       │                                               │
│   │                 │                                               │
│   │ Reproduce bug   │                                               │
│   │ locally         │                                               │
│   └────────┬────────┘                                               │
│            │                                                        │
│            ↓                                                        │
│   2. CREATE BUG FIX BRANCH                                          │
│   ┌─────────────────────────────────────────────────────┐           │
│   │ git checkout main  # or develop                     │           │
│   │ git pull origin main                                │           │
│   │ git checkout -b bugfix/<bug-id>-<description>       │           │
│   │                                                     │           │
│   │ For critical bugs (production):                     │           │
│   │ git checkout -b hotfix/<bug-id>-<description>       │           │
│   └────────┬────────────────────────────────────────────┘           │
│            │                                                        │
│            ↓                                                        │
│   3. WRITE REGRESSION TEST FIRST                                    │
│   ┌─────────────────────────────────────────────────────┐           │
│   │ // test/regression/bug-<id>.test.js                 │           │
│   │ describe('Bug #<id>: <description>', () => {        │           │
│   │   it('should <expected behavior>', () => {          │           │
│   │     // Reproduce bug scenario                       │           │
│   │     // Assert correct behavior                      │           │
│   │   });                                               │           │
│   │ });                                                 │           │
│   │                                                     │           │
│   │ npm test -- test will FAIL (expected)               │           │
│   └────────┬────────────────────────────────────────────┘           │
│            │                                                        │
│            ↓                                                        │
│   4. FIX THE BUG                                                    │
│   ┌─────────────────────────────────────────────────────┐           │
│   │ // Implement fix in source code                     │           │
│   │                                                     │           │
│   │ npm test -- test now PASSES                         │           │
│   │ npm test -- ALL tests still pass                    │           │
│   └────────┬────────────────────────────────────────────┘           │
│            │                                                        │
│            ↓                                                        │
│   5. ATOMIC COMMIT (Test + Fix together)                            │
│   ┌─────────────────────────────────────────────────────┐           │
│   │ git add src/<file>.js test/regression/bug-<id>.js   │           │
│   │ git commit -m "fix(<scope>): <description> [#<id>]  │           │
│   │                                                     │           │
│   │ Fixes #<id>                                         │           │
│   │                                                     │           │
│   │ Bug: <what was happening>                           │           │
│   │ Root Cause: <why it was happening>                  │           │
│   │ Solution: <how it was fixed>                        │           │
│   │                                                     │           │
│   │ Regression test: test/regression/bug-<id>.test.js"  │           │
│   └────────┬────────────────────────────────────────────┘           │
│            │                                                        │
│            ↓                                                        │
│   6. MERGE STRATEGY                                                 │
│   ┌─────────────────────────────────────────────────────┐           │
│   │ For bugfix/* branches:                              │           │
│   │   → PR to develop → squash merge                    │           │
│   │                                                     │           │
│   │ For hotfix/* branches (critical):                   │           │
│   │   → Merge to main with --no-ff                      │           │
│   │   → Tag release (v1.2.3)                            │           │
│   │   → Cherry-pick to develop                          │           │
│   └─────────────────────────────────────────────────────┘           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Bug Fix Branch Naming

```bash
# Regular bug fixes (non-critical)
bugfix/<bug-id>-<short-description>
# Examples:
bugfix/3456-login-validation-error
bugfix/7890-null-pointer-user-service
bugfix/1122-timezone-calculation

# Critical/production bugs (hotfix)
hotfix/<bug-id>-<short-description>
# Examples:
hotfix/9999-payment-data-corruption
hotfix/8888-security-vulnerability
hotfix/7777-database-connection-leak
```

### Complete Bug Fix Example

```bash
# Bug Report #4567: Users cannot login with email containing '+'
# Example: user+tag@example.com fails validation

# 1. Create bug fix branch
git checkout develop
git pull origin develop
git checkout -b bugfix/4567-plus-sign-email-validation

# 2. Write regression test FIRST
cat > test/regression/bug-4567-email-plus-sign.test.js << 'EOF'
/**
 * Regression test for Bug #4567
 * Issue: Users cannot login with email containing '+' character
 * Example: user+tag@example.com fails validation incorrectly
 */
describe('Bug #4567: Email validation with + character', () => {
  it('should accept valid email with + in local part', async () => {
    const email = 'user+tag@example.com';
    const result = await authService.validateEmail(email);
    expect(result.valid).toBe(true);
  });

  it('should allow login with + email', async () => {
    const user = await userService.create({
      email: 'test+login@example.com',
      password: 'SecurePass123!'
    });

    const loginResult = await authService.login({
      email: 'test+login@example.com',
      password: 'SecurePass123!'
    });

    expect(loginResult.success).toBe(true);
    expect(loginResult.user.id).toBe(user.id);
  });
});
EOF

# 3. Run test - verify it FAILS
npm test -- --grep "Bug #4567"
# ✗ should accept valid email with + in local part
# ✗ should allow login with + email

# 4. Fix the bug
# Edit src/validators/email-validator.js
# Change: /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/
# The regex was missing proper escaping

# 5. Run test - verify it PASSES
npm test -- --grep "Bug #4567"
# ✓ should accept valid email with + in local part
# ✓ should allow login with + email

# 6. Run ALL tests - verify no regressions
npm test
# All 234 tests pass

# 7. Atomic commit - test AND fix together
git add src/validators/email-validator.js \
        test/regression/bug-4567-email-plus-sign.test.js

git commit -m "fix(auth): accept plus sign in email validation [#4567]

Fixes #4567

Bug: Users with '+' in their email (e.g., user+tag@example.com)
could not register or login. The validation incorrectly rejected
these valid email addresses.

Root Cause: Email validation regex did not properly handle the '+'
character in the local part of email addresses, despite '+' being
a valid character per RFC 5321.

Solution: Updated email validation regex to correctly allow '+'
character: /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/

Changes:
- Update EMAIL_REGEX in email-validator.js
- Add regression tests for + character handling

Regression test: test/regression/bug-4567-email-plus-sign.test.js
Test verifies both validation and full login flow with + emails."

# 8. Push and create PR
git push -u origin bugfix/4567-plus-sign-email-validation

# Create PR: bugfix/4567-plus-sign-email-validation → develop
```

### Hotfix Workflow (Critical Bugs)

```bash
# CRITICAL: Payment processing corrupting transaction amounts
# Bug #9999 - Production impact

# 1. Create hotfix from main (production)
git checkout main
git pull origin main
git checkout -b hotfix/9999-payment-amount-corruption

# 2. Write regression test
git add test/regression/bug-9999-payment-corruption.test.js
# Test reproduces the corruption scenario

# 3. Fix the bug
git add src/payment/transaction-processor.js \
        test/regression/bug-9999-payment-corruption.test.js

git commit -m "fix(payment): prevent amount corruption in transactions [#9999]

Fixes #9999 - CRITICAL HOTFIX

Bug: Transaction amounts were being corrupted during currency
conversion, resulting in incorrect charges (e.g., $10.00 became $1000).

Root Cause: Integer overflow when multiplying amount by 100 for
cents conversion on amounts > $21,474,836.47.

Solution: Use BigInt for all currency calculations, add overflow
protection, validate amounts before processing.

Impact: Affected ~50 transactions over past 24 hours.
Remediation: Finance team to issue refunds for affected users.

Changes:
- Convert transaction-processor to use BigInt
- Add amount overflow validation
- Add regression test with large amounts

Regression test: test/regression/bug-9999-payment-corruption.test.js"

# 4. Merge to main (production)
git checkout main
git merge --no-ff hotfix/9999-payment-amount-corruption

# 5. Tag the release
git tag -a v2.3.1 -m "Hotfix v2.3.1: Fix payment amount corruption [#9999]"
git push origin main --tags

# 6. Cherry-pick to develop
git checkout develop
git cherry-pick <hotfix-commit-sha>
git push origin develop

# 7. Delete hotfix branch
git branch -d hotfix/9999-payment-amount-corruption
git push origin --delete hotfix/9999-payment-amount-corruption
```

### Bug Fix Commit Message Template

```bash
git commit -m "fix(<scope>): <imperative description> [#<bug-id>]

Fixes #<bug-id>

Bug: <What was the observable problem?>

Root Cause: <Why was this happening? What was the underlying issue?>

Solution: <How did you fix it? What approach was taken?>

Changes:
- <Specific change 1>
- <Specific change 2>
- <Specific change 3>

Regression test: <path/to/test/file.js>
<Brief description of what the test verifies>"
```

---

## 3. Atomic Commits & TDD Integration (MANDATORY)

### A. Atomic Commit Principles

**Each commit MUST:**

1. Contain ONE logical change
2. Include both implementation AND tests
3. Pass all tests
4. Compile/build successfully
5. Be independently revertible
6. Have a clear, single purpose

### B. TDD Commit Pattern

**For new features (3-commit pattern):**

```bash
# Commit 1: RED - Add failing tests
git add test/feature.test.js
git commit -m "test(feature): add tests for new search feature [#1234]

Part 1/3: Add failing tests

- Add unit tests for search query parsing
- Add integration tests for search endpoint
- Tests currently fail (expected behavior)

Related to #1234"

# Commit 2: GREEN - Add minimal implementation
git add src/search.js test/feature.test.js
git commit -m "feat(search): implement user search feature [#1234]

Part 2/3: Minimal implementation

- Add search endpoint
- Implement query parsing
- All tests now pass

Implements #1234"

# Commit 3: REFACTOR - Improve code
git add src/search.js src/search-helper.js
git commit -m "refactor(search): optimize search query performance [#1234]

Part 3/3: Refactor for performance

- Extract query builder to separate module
- Add query result caching
- Reduce database queries by 60%
- All tests still pass

Related to #1234"

# ✅ ALTERNATIVE: Single atomic commit (preferred for small changes)
git add src/search.js test/feature.test.js
git commit -m "feat(search): implement user search feature [#1234]

Implements #1234

- Add search endpoint GET /api/users/search
- Support query params: name, email, role
- Add pagination (limit, offset)
- Include comprehensive tests (unit + integration)
- All tests pass"
```

### C. Bug Fix Commit Pattern (MANDATORY)

**Bug fixes MUST include regression test in same commit:**

```bash
# ✅ CORRECT - Single atomic commit with test
git add src/auth.js test/auth.test.js
git commit -m "fix(auth): prevent null pointer in token validation [#2341]

Fixes #2341

Bug: Application crashed when validating expired tokens due to
null pointer exception when accessing user.roles.

Root Cause: Token validation didn't check for null user before
accessing nested properties.

Solution: Added null check and early return with appropriate
error message.

Changes:
- Add null check in validateToken()
- Return 401 with 'Invalid token' message
- Add regression test: test/auth.test.js:142-158
  (reproduces the null pointer scenario)

The regression test ensures this bug cannot reoccur."

# ❌ WRONG - Separate commits for test and fix
git add test/auth.test.js
git commit -m "test: add test for auth bug [#2341]"

git add src/auth.js
git commit -m "fix(auth): fix null pointer [#2341]"
```

---

## 4. Hexagonal Architecture & Repository Structure

### A. Repository Organization

**Structure repositories following hexagonal architecture principles:**

```
project-root/
├── .git/
├── .github/                    # GitHub workflows, templates
│   ├── workflows/
│   │   ├── ci.yml
│   │   └── cd.yml
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── pull_request_template.md
│
├── src/
│   ├── domain/                 # Core business logic (no dependencies)
│   │   ├── entities/
│   │   ├── value-objects/
│   │   └── services/
│   │
│   ├── application/            # Use cases, application services
│   │   ├── commands/
│   │   ├── queries/
│   │   └── handlers/
│   │
│   ├── infrastructure/         # External dependencies
│   │   ├── database/
│   │   ├── http/
│   │   └── messaging/
│   │
│   └── adapters/               # Ports implementation
│       ├── api/                # REST/GraphQL controllers
│       ├── cli/                # CLI commands
│       └── events/             # Event handlers
│
├── test/
│   ├── unit/                   # Unit tests (domain, application)
│   ├── integration/            # Integration tests (infrastructure)
│   └── e2e/                    # End-to-end tests
│
├── docs/
│   ├── architecture/           # Architecture decision records
│   │   └── adr-001-hexagonal.md
│   ├── api/                    # API documentation
│   └── guides/                 # Development guides
│
├── .gitignore
├── .gitattributes
├── .gitmessage                 # Commit message template
└── README.md
```

### B. Hexagonal Architecture Commit Patterns

**Organize commits by architecture layer:**

```bash
# Domain layer (core business logic)
git commit -m "feat(domain): add User entity with validation [#123]

Implements #123

- Add User entity with email validation
- Add UserRepository port interface
- Include unit tests for validation rules"

# Application layer (use cases)
git commit -m "feat(application): add CreateUser use case [#123]

Related to #123

- Add CreateUserCommand
- Add CreateUserHandler
- Include unit tests with mock repository"

# Infrastructure layer (external dependencies)
git commit -m "feat(infrastructure): implement PostgreSQL user repository [#123]

Related to #123

- Implement UserRepository for PostgreSQL
- Add database migrations
- Include integration tests with test database"

# Adapter layer (ports)
git commit -m "feat(api): add user registration endpoint [#123]

Completes #123

- Add POST /api/users endpoint
- Wire CreateUser use case
- Include E2E tests"
```

---

## 5. Git Hooks & Automation (MANDATORY)

### A. Pre-commit Hook

**Prevent bad commits with automated checks:**

```bash
# .git/hooks/pre-commit (or use pre-commit framework)
#!/bin/bash

echo "Running pre-commit checks..."

# 1. Check commit message format
if ! head -1 .git/COMMIT_EDITMSG | grep -qE '^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)\(.+\): .+ \[#[0-9]+\]$'; then
    echo "❌ ERROR: Commit message doesn't follow Conventional Commits format"
    echo "   Format: <type>(<scope>): <description> [#<issue>]"
    echo "   Example: feat(auth): add login endpoint [#1234]"
    exit 1
fi

# 2. Verify issue reference exists
if ! head -1 .git/COMMIT_EDITMSG | grep -qE '\[#[0-9]+\]'; then
    echo "❌ ERROR: Commit message must reference an issue"
    echo "   Format: [#<issue-number>]"
    exit 1
fi

# 3. Run linter
echo "Running linter..."
npm run lint
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Linting failed"
    exit 1
fi

# 4. Run tests
echo "Running tests..."
npm test
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Tests failed"
    exit 1
fi

# 5. Check for debugging code
if git diff --cached | grep -E 'console\.log|debugger|TODO|FIXME'; then
    echo "⚠️  WARNING: Found debugging code or TODOs"
    echo "   Remove or commit with --no-verify if intentional"
    exit 1
fi

echo "✅ All pre-commit checks passed"
exit 0
```

### B. Commit Message Validation

**Use commitlint for automated validation:**

```javascript
// commitlint.config.js
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      [
        'feat',
        'fix',
        'docs',
        'style',
        'refactor',
        'perf',
        'test',
        'build',
        'ci',
        'chore',
        'revert'
      ]
    ],
    'subject-case': [2, 'always', 'sentence-case'],
    'subject-full-stop': [2, 'never', '.'],
    'header-max-length': [2, 'always', 72],
    'body-leading-blank': [2, 'always'],
    'footer-leading-blank': [2, 'always'],
    // Custom rule: require issue reference
    'references-empty': [2, 'never']
  },
  parserPreset: {
    parserOpts: {
      issuePrefixes: ['#']
    }
  }
};
```

### C. Pre-push Hook

**Prevent pushing untested code:**

```bash
#!/bin/bash
# .git/hooks/pre-push

echo "Running pre-push checks..."

# Get the current branch
branch=$(git symbolic-ref HEAD | sed -e 's,.*/\(.*\),\1,')

# Don't allow direct push to main/develop
if [ "$branch" = "main" ] || [ "$branch" = "develop" ]; then
    echo "❌ ERROR: Direct push to $branch is not allowed"
    echo "   Create a feature branch and open a Pull Request"
    exit 1
fi

# Run full test suite
echo "Running full test suite..."
npm run test:all
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Tests failed"
    exit 1
fi

# Check build
echo "Verifying build..."
npm run build
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Build failed"
    exit 1
fi

echo "✅ All pre-push checks passed"
exit 0
```

---

## 6. Pull Request Best Practices (MANDATORY)

### A. Pull Request Template

Create `.github/pull_request_template.md`:

```markdown
## Description
<!-- Provide a clear description of the changes -->

Closes #<issue-number>

## Type of Change
<!-- Mark with an 'x' -->
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📝 Documentation update
- [ ] ♻️ Refactoring (no functional changes)
- [ ] 🎨 Style changes (formatting, missing semi-colons, etc)
- [ ] ✅ Test updates

## Changes Made
<!-- Bullet list of changes -->
- 
- 
- 

## Testing
<!-- Describe the tests you added or ran -->
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally
- [ ] Test coverage maintained/increased

**Test Coverage:**
- Before: XX%
- After: XX%

## Regression Testing (for bug fixes)
<!-- Required for all bug fixes -->
- [ ] Regression test added that reproduces the bug
- [ ] Test fails before fix, passes after fix
- [ ] Test is included in this PR

**Test Location:** `test/path/to/regression_test.js:LINE`

## Checklist
- [ ] Code follows the style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Dependent changes merged
- [ ] Issue referenced in commit messages with [#issue]
- [ ] Commit messages follow Conventional Commits format

## Screenshots (if applicable)
<!-- Add screenshots for UI changes -->

## Additional Notes
<!-- Any additional information -->
```

### B. Pull Request Size Guidelines

**Keep PRs small and focused:**

```
✅ GOOD PR Sizes:
- 1-3 files changed
- < 400 lines of code
- Single feature or bug fix
- Reviewable in < 30 minutes

⚠️ WARNING PR Sizes:
- 4-10 files changed
- 400-800 lines of code
- Multiple related changes
- Reviewable in 30-60 minutes

❌ BAD PR Sizes:
- > 10 files changed
- > 800 lines of code
- Multiple unrelated changes
- Difficult to review

**If PR is too large, split into:**
1. Refactoring/cleanup (separate PR)
2. Infrastructure/foundation (separate PR)
3. Feature implementation (main PR)
4. Tests and documentation (included in each)
```

### C. Code Review Checklist

**Reviewers MUST verify:**

```markdown
## Code Quality
- [ ] Code is clean and readable
- [ ] Follows project style guide
- [ ] No code duplication
- [ ] Proper error handling
- [ ] No hardcoded values
- [ ] Appropriate abstractions

## Tests
- [ ] Tests are included
- [ ] Tests are comprehensive
- [ ] Tests pass in CI
- [ ] Edge cases covered
- [ ] Regression tests for bug fixes

## Commits
- [ ] All commits follow Conventional Commits
- [ ] All commits reference issues
- [ ] Commits are atomic
- [ ] Commit history is clean
- [ ] No merge commits (rebased)

## Documentation
- [ ] README updated if needed
- [ ] API docs updated
- [ ] Comments explain WHY, not WHAT
- [ ] Complex logic documented

## Security
- [ ] No sensitive data exposed
- [ ] Input validation present
- [ ] No SQL injection risks
- [ ] No XSS vulnerabilities
```

---

## 7. Advanced Git Techniques

### A. Interactive Rebase for Clean History

**Clean up commits before pushing:**

```bash
# Start interactive rebase
git rebase -i HEAD~5

# Editor opens with commits:
pick 1a2b3c4 feat(api): add endpoint [#123]
pick 5d6e7f8 fix typo
pick 9g0h1i2 test: add tests [#123]
pick 3j4k5l6 fix: bug [#123]
pick 7m8n9o0 refactor: cleanup [#123]

# Reorder and squash:
pick 1a2b3c4 feat(api): add endpoint [#123]
squash 9g0h1i2 test: add tests [#123]
squash 7m8n9o0 refactor: cleanup [#123]
pick 3j4k5l6 fix: bug in endpoint [#123]
fixup 5d6e7f8 fix typo

# Result: Clean, logical commits
# 1. feat(api): add endpoint [#123] (includes tests and cleanup)
# 2. fix(api): correct endpoint bug [#123]
```

### B. Cherry-picking for Hotfixes

```bash
# Bug found in production (main)
git checkout main
git checkout -b hotfix/1234-critical-bug

# Fix bug and commit
git add src/fix.js test/fix.test.js
git commit -m "fix(api): prevent data corruption [#1234]

Fixes #1234 - CRITICAL

Bug: Race condition in data update endpoint caused data corruption.
Solution: Added transaction-level locking.
Includes regression test."

# Merge to main
git checkout main
git merge --no-ff hotfix/1234-critical-bug
git tag -a v1.2.3 -m "Hotfix v1.2.3"
git push origin main --tags

# Cherry-pick to develop
git checkout develop
git cherry-pick <hotfix-commit-sha>
git push origin develop
```

### C. Bisect for Bug Hunting

```bash
# Find the commit that introduced a bug
git bisect start
git bisect bad HEAD
git bisect good v1.2.0

# Git checks out commits, you test each
npm test
git bisect good  # or bad

# Git finds the culprit commit
# Bisecting: 5 revisions left to test after this
# ...
# <commit-sha> is the first bad commit
# fix(auth): update token validation [#5678]

# Now you know which commit introduced the bug
git bisect reset
```

### D. Worktrees for Parallel Work

```bash
# Work on multiple branches simultaneously
git worktree add ../project-feature feature/1234-new-api
git worktree add ../project-hotfix hotfix/5678-bug-fix

# Now you have:
# /project/ (main worktree on develop)
# /project-feature/ (worktree on feature/1234-new-api)
# /project-hotfix/ (worktree on hotfix/5678-bug-fix)

# Work in parallel, each worktree is independent
cd ../project-feature
# edit, commit, push

cd ../project-hotfix
# edit, commit, push

# Clean up when done
git worktree remove ../project-feature
git worktree remove ../project-hotfix
```

---

## 8. GitOps & CI/CD Integration

### A. Automated Deployments

**Tag-based deployment workflow:**

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*.*.*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Extract version
        id: version
        run: echo "VERSION=${GITHUB_REF#refs/tags/v}" >> $GITHUB_OUTPUT
      
      - name: Run tests
        run: npm test
      
      - name: Build
        run: npm run build
      
      - name: Create Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ steps.version.outputs.VERSION }}
          draft: false
          prerelease: false
      
      - name: Deploy to Production
        run: |
          # Deploy commands here
          echo "Deploying version ${{ steps.version.outputs.VERSION }}"
```

### B. Conventional Commits Automation

**Auto-generate changelog and version bumps:**

```yaml
# .github/workflows/release-please.yml
name: Release Please

on:
  push:
    branches:
      - main

jobs:
  release-please:
    runs-on: ubuntu-latest
    steps:
      - uses: google-github-actions/release-please-action@v3
        with:
          release-type: node
          package-name: my-package
          # Automatically:
          # - Parses conventional commits
          # - Generates CHANGELOG.md
          # - Creates release PR
          # - Bumps version (major/minor/patch)
```

---

## 9. Security Best Practices

### A. GPG Commit Signing (MANDATORY for Production)

```bash
# Generate GPG key
gpg --full-generate-key

# List keys
gpg --list-secret-keys --keyid-format LONG

# Configure Git
git config --global user.signingkey <key-id>
git config --global commit.gpgsign true
git config --global tag.gpgsign true

# Signed commit
git commit -S -m "feat(auth): add 2FA support [#1234]"

# Verify signatures
git log --show-signature
```

### B. Secrets Management

**.gitignore patterns (MANDATORY):**

```gitignore
# Secrets and credentials
.env
.env.local
.env.*.local
*.key
*.pem
*.p12
secrets.yml
credentials.json

# IDE
.vscode/
.idea/
*.swp
*.swo

# Dependencies
node_modules/
vendor/

# Build artifacts
dist/
build/
*.log

# OS files
.DS_Store
Thumbs.db
```

### C. Sensitive Data Detection

**Pre-commit hook for secrets:**

```bash
# Use tools like git-secrets or gitleaks
git secrets --install
git secrets --register-aws

# Or use gitleaks
gitleaks detect --source . --verbose
```

---

## 10. Deployment Checklist

### Pre-Commit Checklist
- [ ] **Tests pass**: `npm test` succeeds
- [ ] **Linting passes**: `npm run lint` succeeds
- [ ] **Build succeeds**: `npm run build` succeeds
- [ ] **Commit message**: Follows Conventional Commits format
- [ ] **Issue reference**: Includes `[#issue]` reference
- [ ] **Atomic commit**: Single logical change
- [ ] **Tests included**: Implementation + tests in same commit
- [ ] **For bug fixes**: Regression test included

### Pre-Push Checklist
- [ ] **All tests pass**: Full test suite runs
- [ ] **No merge conflicts**: Rebased on latest target branch
- [ ] **Clean history**: No "WIP" or "temp" commits
- [ ] **Branch named correctly**: `type/issue-description`
- [ ] **No secrets**: No credentials or API keys
- [ ] **No debugging code**: No console.logs or debuggers

### Pull Request Checklist
- [ ] **PR template filled**: All sections completed
- [ ] **Issue linked**: Closes #issue in description
- [ ] **Tests documented**: Test approach explained
- [ ] **Screenshots included**: For UI changes
- [ ] **Breaking changes noted**: Migration guide if needed
- [ ] **Reviewers assigned**: Appropriate team members
- [ ] **CI passing**: All checks green
- [ ] **Squash merge ready**: Commits ready to be squashed

### Post-Merge Checklist
- [ ] **Delete feature branch**: Cleanup after merge
- [ ] **Close issue**: Mark issue as resolved
- [ ] **Update documentation**: If API changed
- [ ] **Monitor deployment**: Check logs and metrics
- [ ] **Notify team**: Share in communication channels

---

## 11. Common Patterns & Examples

### A. Feature Development Flow

```bash
# 1. Create feature branch
git checkout develop
git pull origin develop
git checkout -b feature/1234-user-dashboard

# 2. TDD: Write tests first
# Create test/user-dashboard.test.js
git add test/user-dashboard.test.js
git commit -m "test(dashboard): add user dashboard tests [#1234]

Part 1/2: Add failing tests

- Add component tests for dashboard layout
- Add data loading tests
- Tests currently fail (expected)

Related to #1234"

# 3. Implement feature
# Create src/components/UserDashboard.jsx
git add src/components/UserDashboard.jsx test/user-dashboard.test.js
git commit -m "feat(dashboard): implement user dashboard [#1234]

Part 2/2: Implementation

Implements #1234

- Add UserDashboard component
- Display user stats and recent activity
- Implement real-time updates
- All tests now pass

Changes:
- New UserDashboard component with Material-UI
- WebSocket connection for live updates
- Responsive grid layout
- Loading and error states"

# 4. Keep up to date
git fetch origin
git rebase origin/develop

# 5. Push and create PR
git push -u origin feature/1234-user-dashboard

# 6. After PR approved, squash merge to develop
# Branch automatically deleted
```

### B. Hotfix Flow

```bash
# 1. Critical bug in production
git checkout main
git pull origin main
git checkout -b hotfix/5678-payment-failure

# 2. Write regression test
git add test/payment.test.js
git commit -m "test(payment): add regression test for payment failure [#5678]

Reproduces bug #5678

Test scenario: Payment fails when amount has 3 decimal places.
Currently fails (expected behavior)."

# 3. Fix bug
git add src/payment.js test/payment.test.js
git commit -m "fix(payment): handle amounts with 3+ decimal places [#5678]

Fixes #5678 - CRITICAL

Bug: Payment processing failed when amount had 3+ decimal places,
causing transaction to be rejected by payment gateway.

Root Cause: Amount formatting used toFixed(2) which rounded 3+
decimals incorrectly for some currencies.

Solution: Properly handle decimal precision based on currency.
Round to currency-specific decimal places before sending to gateway.

Changes:
- Update formatAmount() to use currency decimal places
- Add decimal place mapping for all supported currencies
- Regression test included (test/payment.test.js:234-256)

Impact: Prevents payment failures, estimated 50 failed transactions/day."

# 4. Merge to main
git checkout main
git merge --no-ff hotfix/5678-payment-failure
git tag -a v1.2.3 -m "Hotfix v1.2.3: Fix payment decimal handling"
git push origin main --tags

# 5. Cherry-pick to develop
git checkout develop
git cherry-pick <hotfix-commit-sha>
git push origin develop
```

---

## 12. Why This Configuration Works

1. **Conventional Commits**: Enables automated changelog generation, semantic versioning, and clear history.

2. **Issue Integration**: Provides full traceability from feature request → commit → deployment → issue closure.

3. **Atomic Commits**: Each commit is independently revertible, making rollbacks safe and precise.

4. **TDD Integration**: Tests committed with implementation ensures working code at every commit.

5. **Regression Tests for Bugs**: Every bug fix includes a test that prevents recurrence, building a safety net over time.

6. **Clean History**: Rebasing and squashing creates linear, readable history that's easy to navigate.

7. **Branch Protection**: Prevents accidental force pushes, requires reviews, enforces CI checks.

8. **Pre-commit Hooks**: Catches issues before they enter version control, maintaining code quality.

9. **Hexagonal Architecture**: Clear separation of concerns reflected in commit organization and file structure.

10. **GitOps Ready**: Tag-based deployments and automated workflows enable true GitOps practices.

11. **Security First**: GPG signing, secrets detection, and proper .gitignore prevent security issues.

12. **Traceable Deployments**: Every production change linked to issue, PR, and responsible developer.

---

## Quick Reference

### Common Git Commands

```bash
# ═══════════════════════════════════════════════════════════════════
# BRANCH OPERATIONS
# ═══════════════════════════════════════════════════════════════════

# Create feature branch
git checkout develop && git pull origin develop
git checkout -b feature/<issue-id>-<description>

# Create bugfix branch
git checkout develop && git pull origin develop
git checkout -b bugfix/<issue-id>-<description>

# Create hotfix branch (from main)
git checkout main && git pull origin main
git checkout -b hotfix/<issue-id>-<description>

# Update branch with latest changes
git fetch origin
git rebase origin/develop  # or origin/main

# Delete local branch
git branch -d <branch-name>

# Delete remote branch
git push origin --delete <branch-name>

# ═══════════════════════════════════════════════════════════════════
# COMMIT OPERATIONS
# ═══════════════════════════════════════════════════════════════════

# Stage specific files (atomic commit)
git add src/file.js test/file.test.js

# Commit with conventional format
git commit -m "feat(scope): description [#123]"

# Commit with body (use heredoc for multiline)
git commit -m "$(cat <<'EOF'
fix(auth): prevent null pointer in login [#456]

Fixes #456

Bug: Application crashed on login with expired token.
Root Cause: Missing null check on user object.
Solution: Added null check with appropriate error.

Regression test: test/auth/login.test.js:142
EOF
)"

# Amend last commit (ONLY if not pushed)
git commit --amend -m "new message"

# ═══════════════════════════════════════════════════════════════════
# HISTORY OPERATIONS
# ═══════════════════════════════════════════════════════════════════

# Interactive rebase (clean up commits)
git rebase -i HEAD~<n>

# Squash merge (for PRs)
git checkout develop
git merge --squash feature/123-feature
git commit -m "feat(scope): complete feature [#123]"

# Cherry-pick commit
git cherry-pick <commit-sha>

# Find bug-introducing commit
git bisect start
git bisect bad HEAD
git bisect good <known-good-commit>
# Test each checkout, mark good/bad
git bisect reset

# ═══════════════════════════════════════════════════════════════════
# RELEASE OPERATIONS
# ═══════════════════════════════════════════════════════════════════

# Create annotated tag
git tag -a v1.2.3 -m "Release v1.2.3: Description"

# Push with tags
git push origin main --tags

# Create release branch
git checkout -b release/v1.2.0 develop

# Merge release to main
git checkout main
git merge --no-ff release/v1.2.0
git tag -a v1.2.0 -m "Release v1.2.0"
```

### Branch Naming Patterns

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/<issue>-<description>` | `feature/1234-user-auth` |
| Bug Fix | `bugfix/<issue>-<description>` | `bugfix/5678-login-crash` |
| Hotfix | `hotfix/<issue>-<description>` | `hotfix/9012-security-fix` |
| Release | `release/v<version>` | `release/v1.2.0` |
| Refactor | `refactor/<issue>-<description>` | `refactor/3456-api-cleanup` |

### Commit Message Format

```
<type>(<scope>): <description> [#<issue>]

[optional body]

[optional footer]
```

| Type | Description | SemVer |
|------|-------------|--------|
| `feat` | New feature | MINOR |
| `fix` | Bug fix | PATCH |
| `docs` | Documentation only | - |
| `style` | Formatting, no code change | - |
| `refactor` | Code change, no feature/fix | - |
| `perf` | Performance improvement | PATCH |
| `test` | Adding/updating tests | - |
| `build` | Build system changes | - |
| `ci` | CI/CD changes | - |
| `chore` | Maintenance tasks | - |
| `revert` | Reverting commits | - |

### TDD Commit Patterns

```bash
# Single atomic commit (recommended for small changes)
git add src/feature.js test/feature.test.js
git commit -m "feat(scope): implement feature [#123]

Implements #123

- Test coverage for all scenarios
- Implementation passes all tests"

# Multi-commit TDD (for complex features)
# Commit 1: Tests + stubs
git commit -m "test(scope): define feature behavior [#123]

Part 1/3: RED phase - Tests defined"

# Commit 2: Implementation
git commit -m "feat(scope): implement feature [#123]

Part 2/3: GREEN phase - All tests pass"

# Commit 3: Refactor
git commit -m "refactor(scope): improve code quality [#123]

Part 3/3: REFACTOR phase - No behavior change"
```

### Bug Fix Commit Template

```bash
git commit -m "fix(scope): description [#<bug-id>]

Fixes #<bug-id>

Bug: <observable problem>
Root Cause: <underlying issue>
Solution: <how it was fixed>

Regression test: <test/path/file.test.js>"
```

### Pre-Commit Checklist

```bash
# Before every commit, verify:
npm test                    # All tests pass
npm run lint                # No linting errors
npm run build               # Build succeeds
git diff --staged           # Review changes
```

### Git Aliases (Recommended)

```bash
# Add to ~/.gitconfig
[alias]
    # Branch operations
    co = checkout
    cb = checkout -b
    bd = branch -d

    # Status and log
    st = status -sb
    lg = log --oneline --graph --decorate -20

    # Commit operations
    cm = commit -m
    ca = commit --amend

    # Sync operations
    fp = fetch --prune
    rb = rebase

    # TDD workflow
    wip = "!git add -A && git commit -m 'WIP: work in progress [skip ci]'"
    unwip = reset HEAD~1

    # Feature workflow
    feature = "!f() { git checkout develop && git pull && git checkout -b feature/$1; }; f"
    bugfix = "!f() { git checkout develop && git pull && git checkout -b bugfix/$1; }; f"
    hotfix = "!f() { git checkout main && git pull && git checkout -b hotfix/$1; }; f"
```

### Workflow Decision Tree

```
Is it a new feature?
├── YES → feature/<issue>-<description> branch
│         └── TDD: Write tests → Implement → Refactor → PR to develop
│
├── Is it a bug fix?
│   ├── Critical (production)? → hotfix/<issue>-<description> from main
│   │                           └── Regression test → Fix → Merge main → Tag → Cherry-pick develop
│   │
│   └── Non-critical? → bugfix/<issue>-<description> from develop
│                       └── Regression test → Fix → PR to develop
│
└── Is it a refactor?
    └── refactor/<issue>-<description> branch
        └── No behavior change → All tests must pass → PR to develop
```

---

## References

- [Conventional Commits 1.0.0](https://www.conventionalcommits.org/)
- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)
- [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Semantic Versioning](https://semver.org/)
- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [Git Best Practices](https://git-scm.com/book/en/v2)

---

**Last Updated:** 2026-01-18  
**Version:** 1.0  
**Maintainer:** Development Team
