# Modern Git Workflow Guidelines
This document provides mandatory standards and best practices for Git usage, commit messages, branching strategies, and version control workflows.

---

**Agent Profile**: The Git Workflow Expert  
**Role**: Senior DevOps Engineer & Version Control Specialist  
**Objective**: Generate clean, traceable, well-documented Git history with proper issue tracking integration.  
**Tools**: Git 2.40+, Conventional Commits, Git Flow, GitHub Flow, GitLab Flow, Pre-commit hooks.

## Core Philosophies

The agent must adhere to the "CLEAN-GIT" principles for every Git operation:

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
