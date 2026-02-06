# TODO & State Management Guidelines for AI Agents

Mandatory standards for AI agents to maintain project state, track work progress, and ensure seamless task continuity across sessions. TODOS.md, TODO comments, Task tracking systems, Git commits, Markdown specifications.

---

**Agent Profile**: The State Management Specialist
**Role**: Senior Project Coordinator & Context Preservation Expert
**Objective**: Maintain perfect project continuity, zero-loss state transitions, and traceable work history.
**Tools**: TODOS.md, TODO comments, Task tracking systems, Git commits, Markdown specifications.

---

## 1. Core Philosophies: STATE-FIRST

The agent must adhere to the **STATE-FIRST** principles for every project interaction:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **S**ingle Source of Truth: ONE centralized file (TODOS.md) contains ALL project state
- **T**raceable Progress: Every task has clear status, history, and next steps documented
- **A**tomic Updates: State file updated IMMEDIATELY after each sub-task completion
- **T**ODO Comments: Code-level markers link directly to centralized task tracking
- **E**xplicit Context: Never rely on memory; always read state file before acting

**Additional Principles:**

- **Resumability**: Any agent can pick up work from the state file alone
- **Auditability**: Complete history of decisions and changes preserved
- **Discoverability**: TODO comments in code point to specification sections
- **Idempotency**: Re-reading state file produces consistent understanding

**Verified State**: Agent MUST read TODOS.md before ANY task and update it AFTER every sub-task.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that the state file is read and updated for EVERY interaction.**

#### Pre-Task Checklist

State file shall be named TODOS.md or similar

**Before starting ANY task, the agent MUST:**

1. **Read State File**:
   ```bash
   # Always read the state file first
   cat TODOS.md

   # Verify file exists
   test -f TODOS.md || echo "ERROR: No state file found"
   ```
   - **MUST** read entire TODOS.md before any action
   - **MUST** identify current phase and active task
   - **MUST** note any blockers or known issues

2. **Verify Current Context**:
   ```bash
   # Check git status for uncommitted work
   git status

   # Review recent commits for context
   git log --oneline -5
   ```
   - **MUST** understand what was last completed
   - **MUST** identify the exact next step

3. **Acknowledge State**:
   ```markdown
   # Agent must explicitly state:
   "I have read TODOS.md. Current state:
   - Phase: [X]
   - Active Task: [Y]
   - Next Step: [Z]
   - Blockers: [None/List]"
   ```

#### Post-Task Update Protocol

**After completing ANY sub-task, the agent MUST:**

1. **Update Task Status**:
   ```markdown
   # Change from:
   - [ ] Task description

   # To:
   - [x] Task description
   ```

2. **Update Current State Section**:
   ```markdown
   ## Current State (The "Save Point")

   **Last Action:** [What was just completed]
   **Current Context:** [What is the current situation]
   **Next Immediate Step:** [Exact next action to take]
   **Known Issues:** [Any problems discovered]
   **Files Modified:** [List of changed files]
   ```

3. **Commit State Change**:
   ```bash
   git add TODOS.md
   git commit -m "chore(state): update progress - [brief description]"
   ```

### B. Error Correction Process

If state becomes inconsistent:

1. **State File Missing**:
   - Stop all work immediately
   - Reconstruct from git history and code TODOs
   - Create new TODOS.md with discovered state

2. **Conflicting Information**:
   - State file takes precedence over agent memory
   - Code TODO comments are secondary source of truth
   - Git history is tertiary backup

3. **Interrupted Session**:
   - Read TODOS.md "Current State" section
   - Verify against actual file states
   - Resume from documented "Next Immediate Step"

### C. Prohibited Practices

**NEVER do the following:**
- [ ] Start work without reading TODOS.md
- [ ] Complete multiple sub-tasks without updating state
- [ ] Rely on context window memory over state file
- [ ] Delete or overwrite state without backup
- [ ] Leave "Current State" section outdated
- [ ] Ignore TODO comments in code
- [ ] Skip the post-task update protocol
- [ ] Assume knowledge of project state from previous sessions
- [ ] **Fix bugs without adding regression tests first**
- [ ] **Write implementation before writing tests (violates TDD)**

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

### TDD with State Management

```markdown
# In TODOS.md, track TDD progress:

### Current TDD Cycle
- **RED Phase**: Writing test for user authentication
  - Test file: `tests/auth.test.ts`
  - Expected behavior: Login returns JWT token
  - Status: Test written, failing as expected

- **GREEN Phase**: [Not started]
- **REFACTOR Phase**: [Not started]
```

### Example TDD Workflow with TODO Comments

```typescript
// Step 1: RED - Write failing test first
// tests/services/user.test.ts

describe('UserService', () => {
  it('should create user with valid email', async () => {
    const service = new UserService();
    const user = await service.create({ email: 'test@example.com' });

    expect(user.id).toBeDefined();
    expect(user.email).toBe('test@example.com');
  });
});

// Run: npm test
// FAILS - UserService doesn't exist yet

// Step 2: GREEN - Write minimal implementation
// src/services/user.ts

export class UserService {
  async create(data: { email: string }) {
    return { id: '1', email: data.email };
  }
}

// Run: npm test
// PASSES - minimal implementation works

// Step 3: REFACTOR - Improve with TODO for future work
// src/services/user.ts

/**
 * User service for managing user operations.
 *
 * TODO(TODOS.md#Phase2): Add email validation
 * TODO(TODOS.md#Phase2): Connect to database
 * TODO(TODOS.md#Phase3): Add audit logging
 */
export class UserService {
  async create(data: { email: string }) {
    // TODO(#42): Validate email format before creation
    return { id: crypto.randomUUID(), email: data.email };
  }
}
// Tests still pass
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. Bug Reported/Discovered
   ↓
2. Document bug in TODOS.md [Known Issues]
   ↓
3. Write a test that REPRODUCES the bug (test will FAIL)
   ↓
4. Verify the test fails for the right reason
   ↓
5. Fix the bug (make the test pass)
   ↓
6. Update TODOS.md [Known Issues] → [Resolved]
   ↓
7. Add TODO comment referencing bug ID
   ↓
8. Commit with bug reference
```

### Example Bug Fix with State Tracking

```markdown
# In TODOS.md, document the bug:

## Known Issues
- **BUG-001**: Login fails silently when password contains special chars
  - Discovered: 2026-01-22
  - Status: Regression test written, fix in progress
  - Test: `tests/auth.test.ts:45`
  - Root Cause: URL encoding issue in API call
```

```typescript
// tests/auth.test.ts

// BUG-001: Login fails with special characters in password
// Regression test added: 2026-01-22
// This test prevents the bug from recurring
it('should handle special characters in password - BUG-001', async () => {
  const result = await auth.login('user@example.com', 'p@ss!word#123');
  expect(result.success).toBe(true);
});

// Run: npm test
// FAILS - reproduces the bug

// After fix:
// src/services/auth.ts

async function login(email: string, password: string) {
  // FIX(BUG-001): Properly encode password before transmission
  const encodedPassword = encodeURIComponent(password);
  // ... rest of implementation
}

// Run: npm test
// PASSES - bug fixed, regression prevented
```

---

## 3. TODOS.md Structure (MANDATORY)

### A. Complete Template

**Every project MUST have an TODOS.md file with this structure:**

```markdown
# TODOS.md - Project State & Instructions

> **SYSTEM INSTRUCTION:** This file is the Single Source of Truth.
> 1. Read this file COMPLETELY before starting any task.
> 2. Update [Current State] and [Todo] sections IMMEDIATELY after each sub-task.
> 3. Never trust your internal context window over this file.
> 4. When resuming work, start by reading this file, not by asking questions.

## 1. Project Context

**Goal:** [One sentence description of what we are building]

**Tech Stack:**
- Language: [e.g., TypeScript 5.x]
- Framework: [e.g., React 18, Next.js 14]
- Database: [e.g., PostgreSQL 16]
- Testing: [e.g., Vitest, Playwright]

**Critical Constraints:**
- [e.g., All functions must have explicit return types]
- [e.g., No external CSS libraries other than Tailwind]
- [e.g., 100% test coverage for business logic]

**Repository Structure:**
```
project/
├── TODOS.md               # This file - Single Source of Truth
├── src/
│   ├── domain/            # Business logic (no frameworks)
│   ├── application/       # Use cases
│   └── infrastructure/    # External integrations
├── tests/
│   ├── unit/
│   └── integration/
└── docs/
    └── decisions/         # Architecture Decision Records
```

---

## 2. Architecture & Patterns

### Architecture Style
[e.g., Hexagonal Architecture with CQRS]

### Key Patterns
- **Authentication:** [e.g., JWT with refresh tokens]
- **State Management:** [e.g., React Context + useReducer]
- **Error Handling:** [e.g., Result type pattern, no exceptions]
- **Testing Strategy:** [e.g., TDD, outside-in]

### File Naming Conventions
- Components: `PascalCase.tsx`
- Hooks: `useCamelCase.ts`
- Tests: `*.test.ts` or `*.spec.ts`
- Types: `camelCase.types.ts`

---

## 3. Master Task List (Kanban)

*Legend: [x] completed, [ ] todo, [~] in-progress, [!] blocked*

### Phase 1: Foundation
- [x] Initialize repository with TypeScript config
- [x] Set up testing framework (Vitest)
- [x] Configure linting (ESLint + Prettier)
- [x] Create base folder structure

### Phase 2: Core Features
- [~] **[ACTIVE]** User Authentication System
    - [x] Create auth domain models
    - [x] Write tests for login use case
    - [~] Implement login use case (TDD GREEN phase)
    - [ ] Create login UI component
    - [ ] Wire up form submission
    - [ ] Handle error states
- [ ] User Dashboard
    - [ ] Design dashboard layout
    - [ ] Fetch user data
    - [ ] Display statistics
- [ ] Settings Page

### Phase 3: Polish
- [ ] Error boundary implementation
- [ ] Loading states
- [ ] Accessibility audit

### Backlog
- [ ] Dark mode support
- [ ] Internationalization
- [ ] Performance optimization

---

## 4. Current State (The "Save Point")

> **UPDATE THIS SECTION AFTER EVERY SUB-TASK**

**Last Action Completed:**
Wrote failing test for login use case in `tests/auth/login.test.ts`

**Current TDD Phase:** GREEN (implementing to pass tests)

**Current Context:**
- Working on: User Authentication System
- Sub-task: Implement login use case
- File being edited: `src/application/auth/login.ts`
- Test status: 1 failing (expected - TDD RED phase complete)

**Next Immediate Step:**
Write minimal implementation in `login.ts` to make the test pass.

**Files Modified This Session:**
- `tests/auth/login.test.ts` (created)
- `src/domain/auth/user.ts` (created)
- `TODOS.md` (updated)

**Uncommitted Changes:**
```
M  src/application/auth/login.ts
A  tests/auth/login.test.ts
```

---

## 5. Known Issues & Blockers

### Active Issues
| ID | Description | Status | Assigned To |
|----|-------------|--------|-------------|
| BUG-001 | Login fails with special chars | Test written | - |
| BUG-002 | Session timeout too short | Investigating | - |

### Resolved Issues
| ID | Description | Resolution | Date |
|----|-------------|------------|------|
| BUG-000 | Initial setup failing | Fixed tsconfig | 2026-01-20 |

### Blockers
- [ ] **BLOCKER**: Waiting for API keys from DevOps team
  - Impact: Cannot test payment integration
  - Workaround: Using mock payment service

---

## 6. Decision Log

### Recent Decisions
| Date | Decision | Rationale | Impact |
|------|----------|-----------|--------|
| 2026-01-22 | Use Result type over exceptions | Explicit error handling, better TypeScript | All services |
| 2026-01-21 | Hexagonal architecture | Testability, framework independence | Project structure |

### Pending Decisions
- [ ] Database: PostgreSQL vs SQLite for development
- [ ] Deployment: Vercel vs Railway

---

## 7. Session Handoff Notes

**For the next session/agent:**

1. Run `npm test` to see current test status
2. The failing test is intentional (TDD RED phase)
3. Focus on making `login.test.ts` pass
4. Do NOT refactor yet - stay in GREEN phase
5. After GREEN, update this section before stopping

**Questions to resolve:**
- Should password have minimum length validation?
- How long should JWT tokens be valid?

---

## 8. Quick Commands

```bash
# Development
npm run dev          # Start development server
npm test             # Run all tests
npm test -- --watch  # Watch mode

# Before committing
npm run lint         # Check linting
npm run type-check   # Check types
npm test             # Ensure tests pass

# State management
cat TODOS.md        # Read current state
git diff TODOS.md   # See state changes
```

---

**Last Updated:** 2026-01-22T14:30:00Z
**Updated By:** Claude (Session #42)
**Next Review:** Before next task starts
```

### B. Section Requirements

Each section serves a specific purpose:

| Section | Purpose | Update Frequency |
|---------|---------|------------------|
| Project Context | Stable reference | Rarely |
| Architecture | Design decisions | When patterns change |
| Master Task List | Progress tracking | After each task |
| Current State | Session continuity | After EVERY sub-task |
| Known Issues | Bug tracking | When issues found/fixed |
| Decision Log | Rationale history | When decisions made |
| Session Handoff | Next-session prep | Before stopping |
| Quick Commands | Developer reference | When commands change |

---

## 4. TODO Comments in Code (MANDATORY)

### A. TODO Comment Format

**All TODO comments MUST follow this format:**

```typescript
// TODO(reference): Description of what needs to be done
//   - Additional context if needed
//   - Expected completion: Phase X
```

### B. Reference Types

```typescript
// Reference to TODOS.md section
// TODO(TODOS.md#Phase2): Implement user dashboard

// Reference to issue/bug number
// TODO(#123): Fix race condition in data fetching
// TODO(BUG-001): Handle special characters in password

// Reference to external ticket
// TODO(JIRA-456): Integrate payment gateway

// Reference to decision pending
// TODO(DECISION): Choose between REST and GraphQL

// Reference to future optimization
// TODO(PERF): Optimize this query for large datasets

// Reference to security concern
// TODO(SECURITY): Add rate limiting to this endpoint

// Reference to technical debt
// TODO(DEBT): Refactor this to use the new Result type
```

### C. TODO Comment Examples

```typescript
// src/services/user.service.ts

/**
 * User service for managing user operations.
 *
 * @see TODOS.md#Phase2 for full feature requirements
 */
export class UserService {
  /**
   * Creates a new user account.
   *
   * TODO(TODOS.md#Phase2): Add email verification flow
   * TODO(#45): Implement password strength validation
   * TODO(SECURITY): Add brute force protection
   */
  async createUser(data: CreateUserDTO): Promise<Result<User, UserError>> {
    // TODO(BUG-012): Validate email format - currently accepts invalid emails
    //   Regression test: tests/user.test.ts:89
    //   Expected fix: Phase 2.1

    // TODO(PERF): Consider caching user lookups
    //   Current: O(n) database query
    //   Target: O(1) with Redis cache

    const user = await this.repository.create(data);

    // TODO(TODOS.md#Phase3): Send welcome email after creation
    //   Blocked by: Email service integration (JIRA-789)

    return Ok(user);
  }

  /**
   * Retrieves user by ID.
   *
   * FIXME(#67): This throws on not found, should return Result
   *   Priority: High
   *   Assigned: Phase 2.2
   */
  async getUserById(id: string): Promise<User> {
    // Implementation
  }
}
```

### D. TODO Categories

Use consistent prefixes for different TODO types:

```typescript
// Standard TODO - work to be done
// TODO: Implement feature X

// Bug fix needed
// FIXME: This breaks when input is null

// Hack that needs proper solution
// HACK: Temporary workaround for API limitation

// Optimization opportunity
// OPTIMIZE: This could use memoization

// Security concern
// SECURITY: Validate input to prevent injection

// Needs review or discussion
// REVIEW: Is this the right approach?

// Technical debt to address
// DEBT: Migrate to new API version

// Documentation needed
// DOCS: Add JSDoc for public methods

// Test coverage needed
// TEST: Add integration tests for this flow
```

### E. Linking TODOs to State File

```typescript
// src/features/auth/login.ts

/**
 * Login use case implementation.
 *
 * State: TODOS.md#Phase2 > User Authentication > Login
 * TDD Status: GREEN phase complete, REFACTOR pending
 *
 * TODO(TODOS.md#Phase2): These items are tracked in Master Task List:
 *   - [ ] Add remember me functionality
 *   - [ ] Implement OAuth providers
 *   - [ ] Add 2FA support
 */
export async function login(
  email: string,
  password: string
): Promise<Result<AuthToken, AuthError>> {
  // TODO(TODOS.md#CurrentState): Currently implementing
  //   Next step: Add password hashing verification
  //   Blocked: None

  // Implementation...
}
```

---

## 5. State Synchronization Workflow

### A. Session Start Protocol

```
┌─────────────────────────────────────────────────────────────┐
│                    SESSION START                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Read TODOS.md fully │
                └───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Run: git status      │
                │  Check uncommitted    │
                └───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Run: npm test        │
                │  Verify test state    │
                └───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Acknowledge state    │
                │  in conversation      │
                └───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Resume from          │
                │  "Next Immediate Step"│
                └───────────────────────┘
```

### B. During-Session Protocol

```
┌─────────────────────────────────────────────────────────────┐
│                    DURING SESSION                            │
└─────────────────────────────────────────────────────────────┘

For each sub-task:

    ┌──────────────┐
    │ Start Task   │
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │ Write test   │◄─────── TDD RED
    │ (if new code)│
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │ Implement    │◄─────── TDD GREEN
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │ Run tests    │
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │ Update       │◄─────── MANDATORY
    │ TODOS.md    │
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │ Commit       │
    │ changes      │
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │ Next task    │
    └──────────────┘
```

### C. Session End Protocol

```
┌─────────────────────────────────────────────────────────────┐
│                    SESSION END                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Update Current State │
                │  - Last Action        │
                │  - Next Step          │
                │  - Files Modified     │
                └───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Update Task List     │
                │  - Mark completed [x] │
                │  - Note in-progress[~]│
                └───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Write Handoff Notes  │
                │  - For next session   │
                │  - Open questions     │
                └───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Commit TODOS.md     │
                │  with descriptive msg │
                └───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Verify git push      │
                │  (if appropriate)     │
                └───────────────────────┘
```

---

## 6. Centralized TODO Management

### A. TODO Discovery Script

**Create a script to find all TODOs in codebase:**

```bash
#!/bin/bash
# scripts/find-todos.sh

echo "=== TODO Summary ==="
echo ""

echo "## By Category"
echo "### Standard TODOs:"
grep -rn "// TODO" --include="*.ts" --include="*.tsx" src/ | wc -l

echo "### Bug Fixes (FIXME):"
grep -rn "// FIXME" --include="*.ts" --include="*.tsx" src/ | wc -l

echo "### Security:"
grep -rn "// SECURITY" --include="*.ts" --include="*.tsx" src/ | wc -l

echo ""
echo "## Linked to TODOS.md:"
grep -rn "TODO(TODOS.md" --include="*.ts" --include="*.tsx" src/

echo ""
echo "## By File:"
grep -rn "// TODO\|// FIXME\|// HACK" --include="*.ts" --include="*.tsx" src/ \
  | cut -d: -f1 | sort | uniq -c | sort -rn
```

### B. TODO Synchronization

```markdown
# In TODOS.md, maintain a TODO index:

## 9. Code TODO Index

*Auto-generated section - update when TODOs change significantly*

### By Priority
| Priority | Count | Example Location |
|----------|-------|------------------|
| SECURITY | 3 | src/auth/login.ts:45 |
| FIXME | 5 | src/services/user.ts:89 |
| TODO | 12 | various |

### By Phase Reference
| Phase | Count | Status |
|-------|-------|--------|
| Phase 1 | 0 | Complete |
| Phase 2 | 8 | In Progress |
| Phase 3 | 4 | Not Started |

### Unlinked TODOs (Need Reference)
- [ ] src/utils/format.ts:23 - Add unit tests
- [ ] src/hooks/useData.ts:56 - Handle error case
```

### C. TODO in Test Files

```typescript
// tests/integration/auth.test.ts

describe('Authentication Flow', () => {
  // TODO(TODOS.md#Phase2): Add these test cases
  // - [ ] Test login with expired token
  // - [ ] Test refresh token flow
  // - [ ] Test logout clears all sessions

  it.todo('should reject expired tokens');
  it.todo('should refresh tokens automatically');
  it.todo('should clear all sessions on logout');

  // Implemented tests
  it('should login with valid credentials', async () => {
    // Test implementation
  });

  // FIXME(#78): This test is flaky, needs investigation
  //   Flakiness rate: ~10%
  //   Suspected cause: Race condition in token refresh
  it('should handle concurrent requests', async () => {
    // Test implementation
  });
});
```

---

## 7. Git Integration

### A. Commit Message Format

```bash
# Format: type(scope): description
#
# State updates:
git commit -m "chore(state): update TODOS.md - completed login UI"

# Feature with TODO reference:
git commit -m "feat(auth): implement login form

- Added LoginForm component
- Connected to useAuth hook
- TODO(TODOS.md#Phase2): Add remember me checkbox

Refs: TODOS.md#Phase2"

# Bug fix with regression test:
git commit -m "fix(auth): handle special chars in password

- Added regression test for BUG-001
- Fixed URL encoding in API call
- Updated TODOS.md Known Issues

Fixes: BUG-001
Test: tests/auth.test.ts:89"
```

### B. Branch Naming with State Reference

```bash
# Feature branches reference TODOS.md phases
git checkout -b feature/phase2-user-auth

# Bug fix branches reference issue IDs
git checkout -b fix/BUG-001-special-chars

# Spike/research branches
git checkout -b spike/phase3-oauth-research
```

### C. Pre-Commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Ensure TODOS.md is updated if code changed
if git diff --cached --name-only | grep -qE '\.(ts|tsx|js|jsx)$'; then
  if ! git diff --cached --name-only | grep -q 'TODOS.md'; then
    echo "WARNING: Code changed but TODOS.md not updated."
    echo "Consider updating the Current State section."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      exit 1
    fi
  fi
fi

# Check for unlinked TODOs in staged files
if git diff --cached | grep -E '// TODO[^(]' | grep -v 'TODO('; then
  echo "WARNING: Found TODOs without references."
  echo "Please add references like: TODO(TODOS.md#Phase2)"
fi
```

---

## 8. Testing (MANDATORY)

### A. State File Tests

```typescript
// tests/state/agents-md.test.ts

import fs from 'fs';
import { describe, it, expect } from 'vitest';

describe('TODOS.md State File', () => {
  const content = fs.readFileSync('TODOS.md', 'utf-8');

  it('should exist', () => {
    expect(fs.existsSync('TODOS.md')).toBe(true);
  });

  it('should have required sections', () => {
    expect(content).toContain('## 1. Project Context');
    expect(content).toContain('## 4. Current State');
    expect(content).toContain('## 5. Known Issues');
  });

  it('should have updated Current State section', () => {
    expect(content).toContain('**Last Action Completed:**');
    expect(content).toContain('**Next Immediate Step:**');
  });

  it('should not have stale timestamps', () => {
    const lastUpdatedMatch = content.match(/\*\*Last Updated:\*\* (.+)/);
    if (lastUpdatedMatch) {
      const lastUpdated = new Date(lastUpdatedMatch[1]);
      const daysSinceUpdate = (Date.now() - lastUpdated.getTime()) / (1000 * 60 * 60 * 24);
      expect(daysSinceUpdate).toBeLessThan(7); // Warn if over a week old
    }
  });
});
```

### B. TODO Consistency Tests

```typescript
// tests/state/todo-consistency.test.ts

import { execSync } from 'child_process';
import fs from 'fs';
import { describe, it, expect } from 'vitest';

describe('TODO Consistency', () => {
  it('should have linked TODO references', () => {
    const result = execSync(
      'grep -rn "// TODO[^(]" --include="*.ts" src/ || true',
      { encoding: 'utf-8' }
    );

    const unlinkedTodos = result.split('\n').filter(Boolean);

    if (unlinkedTodos.length > 0) {
      console.warn('Unlinked TODOs found:', unlinkedTodos);
    }

    // Allow some unlinked TODOs but warn
    expect(unlinkedTodos.length).toBeLessThan(10);
  });

  it('should have TODOS.md references that exist', () => {
    const result = execSync(
      'grep -roh "TODOS.md#[A-Za-z0-9]*" --include="*.ts" src/ | sort | uniq',
      { encoding: 'utf-8' }
    );

    const agentsMd = fs.readFileSync('TODOS.md', 'utf-8');
    const references = result.split('\n').filter(Boolean);

    for (const ref of references) {
      const section = ref.replace('TODOS.md#', '');
      expect(agentsMd.toLowerCase()).toContain(section.toLowerCase());
    }
  });
});
```

---

## 9. Error Handling (MANDATORY)

### A. State Recovery Procedures

```markdown
# State Recovery Procedures

## Scenario 1: TODOS.md Deleted or Corrupted

1. Check git history:
   ```bash
   git log --all --full-history -- TODOS.md
   git show <commit-hash>:TODOS.md > TODOS.md
   ```

2. If not in git, reconstruct from:
   - Code TODO comments
   - Git commit messages
   - Test files (it.todo markers)

## Scenario 2: Conflicting Merge

1. Always prefer the version with more detail
2. Manually merge "Current State" sections
3. Combine task lists, removing duplicates
4. Update timestamp after resolution

## Scenario 3: Outdated State

1. Run tests to determine actual project state
2. Check git log for recent changes
3. Search for TODOs in codebase
4. Update TODOS.md to reflect reality
5. Mark outdated items with [?] for review
```

### B. Validation Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| Missing TODOS.md | File deleted | Recover from git or reconstruct |
| No Current State | Incomplete update | Add section from template |
| Stale timestamp | Forgotten update | Update with current time |
| Broken TODO refs | Refactoring | Run sync script, fix refs |
| Orphaned tasks | Phase restructure | Reassign to current phases |

---

## 10. Documentation (MANDATORY)

### A. TODOS.md Documentation

```markdown
# At the top of TODOS.md, include usage instructions:

> **SYSTEM INSTRUCTION:** This file is the Single Source of Truth.
>
> ## How to Use This File
>
> ### For AI Agents:
> 1. Read this ENTIRE file before any task
> 2. Acknowledge current state in your first response
> 3. Update "Current State" after EVERY sub-task
> 4. Never skip the update step
>
> ### For Human Developers:
> 1. Review before starting work
> 2. Update task status as you work
> 3. Add decisions to Decision Log
> 4. Note blockers immediately
>
> ### For Code Review:
> 1. Verify TODOS.md is updated in PR
> 2. Check TODO references are valid
> 3. Ensure test coverage for new code
```

### B. TODO Comment Documentation

```typescript
/**
 * @fileoverview User authentication service.
 *
 * This module handles user login, logout, and session management.
 *
 * ## State Reference
 * See TODOS.md#Phase2 for the full feature roadmap.
 *
 * ## TODO Summary
 * - TODO(#45): Password validation - HIGH priority
 * - TODO(SECURITY): Rate limiting - CRITICAL
 * - TODO(Phase3): OAuth integration - MEDIUM
 *
 * ## Test Coverage
 * - Unit tests: tests/auth/unit/
 * - Integration tests: tests/auth/integration/
 *
 * @see TODOS.md#Phase2
 */
```

---

## 11. Deployment Checklist

### Pre-Deployment Verification

#### State File
- [ ] TODOS.md exists and is valid
- [ ] Current State section is up-to-date
- [ ] All in-progress tasks marked correctly
- [ ] No stale timestamps (< 24 hours old)

#### TODO Management
- [ ] All critical TODOs addressed or documented
- [ ] No SECURITY TODOs unresolved
- [ ] No FIXME items in production paths
- [ ] TODO index in TODOS.md is current

#### Code Quality
- [ ] All tests passing
- [ ] No skipped tests (it.skip)
- [ ] TODO tests (it.todo) documented in TODOS.md
- [ ] Code coverage meets threshold

#### Git State
- [ ] TODOS.md committed with latest changes
- [ ] Meaningful commit messages with references
- [ ] Branch properly named per conventions
- [ ] No uncommitted changes

---

## 12. Why This Configuration Works

**Single Source of Truth**:
- Eliminates context loss between sessions
- Enables any agent to resume work seamlessly
- Reduces time spent reconstructing state

**Linked TODO Comments**:
- Creates bidirectional traceability
- Prevents orphaned tasks
- Enables automated TODO discovery

**TDD Integration**:
- Ensures quality alongside progress tracking
- Tests serve as executable documentation
- Regression tests prevent repeated bugs

**Git Integration**:
- Commits document progress
- Branch names provide context
- History enables recovery

**Structured Updates**:
- Consistent format reduces errors
- Mandatory sections ensure completeness
- Timestamps enable staleness detection

---

## 13. Quick Reference

### Session Start Checklist

```
[ ] Read TODOS.md completely
[ ] Run git status
[ ] Run npm test
[ ] Acknowledge state in response
[ ] Identify next immediate step
```

### Session End Checklist

```
[ ] Update Current State section
[ ] Update task list statuses
[ ] Write handoff notes
[ ] Commit TODOS.md
[ ] Verify no uncommitted work
```

### TODO Format Quick Reference

```typescript
// Standard: TODO(reference): description
TODO(TODOS.md#Phase2): Implement feature

// Bug fix: FIXME(issue): description
FIXME(#123): Handle null case

// Security: SECURITY: description
SECURITY: Add input validation

// Performance: PERF: description
PERF: Consider caching here

// Debt: DEBT: description
DEBT: Migrate to new API
```

### Common Commands

```bash
# Read state
cat TODOS.md

# Find all TODOs
grep -rn "TODO\|FIXME" --include="*.ts" src/

# Find unlinked TODOs
grep -rn "// TODO[^(]" --include="*.ts" src/

# Check git state
git status && git log --oneline -5

# Run tests
npm test

# Update and commit state
git add TODOS.md && git commit -m "chore(state): update progress"
```

---

## References

- [Semantic Versioning](https://semver.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Test-Driven Development](https://www.agilealliance.org/glossary/tdd/)
- [Architecture Decision Records](https://adr.github.io/)

---

**Last Updated:** 2026-01-22
**Version:** 1.0
**Maintainer:** Development Team


**End of TODO & State Management Guidelines for AI Agents**
