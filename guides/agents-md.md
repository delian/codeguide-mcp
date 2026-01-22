# AGENTS.md Creation Guidelines for AI Agents

This document provides mandatory standards for AI agents to create and maintain AGENTS.md files as the Single Source of Truth for every project.

---

**Agent Profile**: The Project State Architect
**Role**: Senior Context Management Specialist & Project Documentation Expert
**Objective**: Create comprehensive, self-documenting project state files that enable seamless AI-human collaboration.
**Tools**: AGENTS.md, MCP servers, Code Guide references, Git, Testing frameworks, Documentation generators.

---

## 1. Core Philosophies: AGENTS-FIRST

The agent must adhere to the **AGENTS-FIRST** principles when creating or maintaining AGENTS.md:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **A**utomatic Creation: Create AGENTS.md if it doesn't exist at project root
- **G**uide Compliance: Follow applicable code-guide MCP instructions for the tech stack
- **E**xhaustive State: Capture ALL project context, constraints, and progress
- **N**ever Assume: Document everything; assume no prior knowledge
- **T**est Tracking: Maintain test state, coverage, and TDD progress
- **S**pec Priority: Follow spec-coding instructions when provided (with MCP compliance)

**Additional Principles:**

- **MCP Integration**: Reference and follow applicable MCP server guidelines
- **STATE-FIRST Compliance**: Follow [todo.md](./todo.md) STATE-FIRST principles for task tracking
- **TODOS.md Integration**: Use TODOS.md for centralized task management (see todo.md guide)
- **Documentation Living**: Keep docs state synchronized with code state
- **Handoff Ready**: Any agent should resume work from AGENTS.md alone
- **Decision Audit**: Log all architectural and technical decisions

**Verified Creation**: Agent MUST verify AGENTS.md exists before ANY task; create if missing.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST check for AGENTS.md existence at the start of EVERY session.**

#### Pre-Task Checklist

**Before starting ANY task, the agent MUST:**

1. **Check for AGENTS.md**:
   ```bash
   # Check if AGENTS.md exists
   if [ -f "AGENTS.md" ]; then
     echo "AGENTS.md found - reading state..."
     cat AGENTS.md
   else
     echo "AGENTS.md not found - must create before proceeding"
   fi
   ```
   - **MUST** check project root for AGENTS.md
   - **MUST** create AGENTS.md if it doesn't exist
   - **MUST** read entire file if it exists

2. **Identify Tech Stack and MCPs**:
   ```bash
   # Detect project type
   ls -la package.json pyproject.toml Cargo.toml go.mod *.csproj 2>/dev/null

   # Check for existing configs
   ls -la .eslintrc* tsconfig.json .prettierrc* 2>/dev/null
   ```
   - **MUST** identify applicable code-guide MCPs
   - **MUST** note which guidelines apply to the project

3. **Verify MCP Compliance**:
   ```markdown
   # Agent must identify:
   "Applicable MCPs for this project:
   - Language: [e.g., typescript.md, python.md, rust.md]
   - Framework: [e.g., reactjs.md, nodejs.md, flutter.md]
   - Architecture: [e.g., hexagonal.md, cleanarch.md]
   - CI/CD: [e.g., github.md, gitlab.md]
   - Infrastructure: [e.g., kubernetes.md, docker-compose.md]"
   ```

#### AGENTS.md Creation Protocol

**If AGENTS.md does not exist, the agent MUST create it immediately:**

1. **Analyze Project Structure**:
   - Scan directory structure
   - Identify existing code patterns
   - Detect testing frameworks
   - Find documentation

2. **Populate from Detection**:
   - Auto-fill tech stack from config files
   - Identify architectural patterns in use
   - Discover existing conventions

3. **Apply MCP Guidelines**:
   - Reference applicable code-guide MCPs
   - Include MCP-specific requirements
   - Note deviations from guidelines

### B. Error Correction Process

If AGENTS.md is missing or incomplete:

1. **Missing File**:
   - Stop current task
   - Create AGENTS.md from template
   - Populate with discovered project state
   - Resume original task

2. **Incomplete Sections**:
   - Add missing sections from template
   - Preserve existing content
   - Update timestamps

3. **Outdated Information**:
   - Verify against actual project state
   - Update stale sections
   - Note changes in decision log

### C. Prohibited Practices

**NEVER do the following:**
- [ ] Start work without checking for AGENTS.md
- [ ] Proceed without creating AGENTS.md if missing
- [ ] Ignore applicable MCP guidelines
- [ ] Skip test state documentation
- [ ] Leave MCP references section empty
- [ ] Omit spec-coding instructions when provided
- [ ] Forget to update after completing tasks
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

### TDD State in AGENTS.md

```markdown
## Test State

### Current TDD Cycle
- **Feature**: User Authentication
- **Phase**: GREEN (implementing)
- **Test File**: `tests/auth/login.test.ts`
- **Status**: 3 passing, 1 failing (expected)

### Test Coverage
| Module | Coverage | Target | Status |
|--------|----------|--------|--------|
| auth/ | 87% | 80% | ✅ |
| api/ | 72% | 80% | ⚠️ |
| utils/ | 95% | 80% | ✅ |

### Pending Tests (it.todo)
- [ ] `auth.test.ts`: OAuth provider integration
- [ ] `api.test.ts`: Rate limiting behavior
- [ ] `user.test.ts`: Profile update validation
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. Bug Reported/Discovered
   ↓
2. Document in AGENTS.md [Known Issues]
   ↓
3. Write test that REPRODUCES bug (will FAIL)
   ↓
4. Verify test fails correctly
   ↓
5. Fix the bug (make test pass)
   ↓
6. Update AGENTS.md [Known Issues] → [Resolved]
   ↓
7. Commit with bug reference
```

### Bug Tracking in AGENTS.md

```markdown
## Known Issues

### Active
| ID | Description | Severity | Test | Status |
|----|-------------|----------|------|--------|
| BUG-001 | Login fails with special chars | High | `auth.test.ts:45` | Fix in progress |
| BUG-002 | Session timeout too short | Medium | - | Investigating |

### Resolved
| ID | Description | Resolution | Test | Date |
|----|-------------|------------|------|------|
| BUG-000 | CORS error on API calls | Added headers | `api.test.ts:12` | 2026-01-20 |
```

---

## 3. AGENTS.md Complete Template (MANDATORY)

### A. Full Template Structure

**Every AGENTS.md MUST follow this structure:**

```markdown
# AGENTS.md - Project State & Instructions

> **SYSTEM INSTRUCTION:** This file is the Single Source of Truth.
> 1. Read this file COMPLETELY before starting any task.
> 2. Update [Current State] and [Todo] sections IMMEDIATELY after each sub-task.
> 3. Never trust your internal context window over this file.
> 4. When resuming work, start by reading this file, not by asking questions.
> 5. Follow all applicable MCP guidelines referenced in this file.

---

## 1. Project Context

**Goal:** [One sentence description of what we are building]

**Tech Stack:**
- Language: [e.g., TypeScript, Python, Rust, Go, Java, etc.]
- Framework: [e.g., React, Django, Actix, Gin, Spring, etc.]
- Runtime: [e.g., Node.js, JVM, .NET, etc.]
- Database: [e.g., PostgreSQL, MongoDB, Redis, etc.]
- Testing: [e.g., pytest, vitest, cargo test, go test, JUnit, etc.]
- CI/CD: [e.g., GitHub Actions, GitLab CI, Jenkins, etc.]

**Critical Constraints:**
- [e.g., All functions must have explicit return types]
- [e.g., No external CSS libraries other than Tailwind]
- [e.g., 100% test coverage for business logic]
- [e.g., Must follow hexagonal architecture]

---

## 2. MCP & Code Guide References

> **IMPORTANT:** Follow these guidelines for all code generation.

### Applicable MCPs
| Category | Guide | Status | Notes |
|----------|-------|--------|-------|
| Language | `[language].md` | Active | [language-specific notes] |
| Framework | `[framework].md` | Active | [framework-specific patterns] |
| Architecture | `[arch-pattern].md` | Active | [architecture notes] |
| Testing | `tdd.md` | Active | TDD mandatory for all languages |
| CI/CD | `[ci-platform].md` | Active | [CI/CD workflow notes] |
| State Mgmt | `todo.md` | Active | TODOS.md format and STATE-FIRST |
| Agents | `agents-md.md` | Active | This file creation guide |

**Example MCP configurations by stack:**
- Python: `python.md`, `rest.md`, `tdd.md`, `github.md`
- Rust: `rust.md`, `hexagonal.md`, `tdd.md`, `gitlab.md`
- Go: `go.md`, `cleanarch.md`, `tdd.md`, `github.md`
- TypeScript: `typescript.md`, `reactjs.md`, `tdd.md`, `github.md`
- Java: `java.md`, `spring.md`, `tdd.md`, `jenkins.md`

### MCP Compliance Checklist
- [ ] Code follows language guide conventions
- [ ] Architecture matches specified pattern
- [ ] Tests follow TDD protocol
- [ ] CI/CD pipelines follow guide
- [ ] Documentation meets standards

### Spec-Coding Instructions
> If specific coding instructions were provided, they take precedence
> over default MCP guidelines (while remaining MCP-compliant).

```
[Paste any spec-coding instructions here]
[These override defaults but must still follow MCP structure]
```

---

## 3. Architecture & Patterns

### Architecture Style
[e.g., Hexagonal Architecture with CQRS]

```
┌─────────────────────────────────────────────────────────────┐
│                        PRESENTATION                          │
│                    (React Components)                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                        APPLICATION                           │
│                      (Use Cases)                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                          DOMAIN                              │
│                   (Business Logic)                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      INFRASTRUCTURE                          │
│                (Database, APIs, etc.)                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Patterns
- **Authentication:** [e.g., JWT with refresh tokens]
- **State Management:** [e.g., React Context + useReducer]
- **Error Handling:** [e.g., Result type pattern]
- **API Design:** [e.g., REST with OpenAPI]

### File Structure Conventions
```
project/
├── AGENTS.md              # This file
├── src/
│   ├── domain/            # Business logic (no framework deps)
│   │   ├── entities/
│   │   ├── value-objects/
│   │   └── services/
│   ├── application/       # Use cases
│   │   ├── commands/
│   │   └── queries/
│   ├── infrastructure/    # External integrations
│   │   ├── database/
│   │   ├── api/
│   │   └── external/
│   └── presentation/      # UI layer
│       ├── components/
│       ├── hooks/
│       └── pages/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── docs/
    ├── api/
    └── decisions/
```

---

## 4. Master Task List (Kanban)

Master tasks lists shall be kept in TODOS.md and shall follow the requirements of the MCP for the TODOS.md file.

*Legend: [x] completed, [ ] todo, [~] in-progress, [!] blocked*

### Phase 1: Foundation
- [x] Initialize repository with TypeScript config
- [x] Set up testing framework (Vitest)
- [x] Configure linting (ESLint + Prettier)
- [x] Create base folder structure
- [x] Set up CI/CD pipeline

### Phase 2: Core Features
- [~] **[ACTIVE]** User Authentication System
    - [x] Define auth domain models
    - [x] Write login use case tests (TDD RED)
    - [~] Implement login use case (TDD GREEN)
    - [ ] Create login UI component
    - [ ] Wire up form submission
    - [ ] Handle error states
    - [ ] Add session management
- [ ] User Dashboard
    - [ ] Design dashboard layout
    - [ ] Create dashboard components
    - [ ] Fetch user data
- [ ] Settings Page

### Phase 3: Polish & Deploy
- [ ] Error boundary implementation
- [ ] Loading states and skeletons
- [ ] Accessibility audit (WCAG 2.1 AA)
- [ ] Performance optimization
- [ ] Production deployment

### Backlog
- [ ] Dark mode support
- [ ] Internationalization (i18n)
- [ ] Analytics integration

---

## 5. Test State

**CRITICAL: All new code must have tests. No exceptions.**
The tests state shall follow the requirements of the MCP for the TODOS.md file.

### Current TDD Status
| Feature | Phase | Test File | Passing | Failing |
|---------|-------|-----------|---------|---------|
| Auth Login | GREEN | `auth/login.test.ts` | 3 | 1 |
| Auth Logout | RED | `auth/logout.test.ts` | 0 | 2 |
| User Profile | - | Not started | - | - |

### Test Coverage Report
```
----------------------|---------|----------|---------|---------|
File                  | % Stmts | % Branch | % Funcs | % Lines |
----------------------|---------|----------|---------|---------|
All files             |   78.5  |    72.3  |   81.2  |   79.1  |
 src/domain/          |   92.1  |    88.5  |   95.0  |   92.8  |
 src/application/     |   85.3  |    79.2  |   88.0  |   86.1  |
 src/infrastructure/  |   65.2  |    58.9  |   70.5  |   64.8  |
----------------------|---------|----------|---------|---------|
```

### Pending Tests (it.todo)
- `auth.test.ts:45` - OAuth provider integration
- `auth.test.ts:67` - Password reset flow
- `api.test.ts:23` - Rate limiting behavior

### Test Commands
```bash
# Adapt to your test framework:
# Node:   npm test | npm test -- --watch | npm run test:coverage
# Python: pytest | pytest-watch | pytest --cov
# Rust:   cargo test | cargo watch -x test | cargo tarpaulin
# Go:     go test ./... | gowatch | go test -cover ./...
# Java:   mvn test | gradle test | jacoco:report
# Flutter: flutter test | flutter test --coverage
```

---

## 6. Documentation State

### Generated Documentation
| Type | Generator | Output | Last Updated | Status |
|------|-----------|--------|--------------|--------|
| API Docs | TypeDoc | `docs/api/` | 2026-01-22 | ✅ Current |
| OpenAPI | Swagger | `docs/openapi.yaml` | 2026-01-21 | ⚠️ Needs update |
| ADRs | Manual | `docs/decisions/` | 2026-01-22 | ✅ Current |

### Documentation Commands
```bash
# Adapt to your documentation generator:
# Node/TS: npm run docs | typedoc
# Python:  sphinx-build | pdoc | mkdocs build
# Rust:    cargo doc --open
# Go:      godoc | go doc
# Java:    mvn javadoc:javadoc | dokka
```

### Pending Documentation
- [ ] README.md: Add deployment section
- [ ] API: Document new auth endpoints
- [ ] ADR: Document state management decision

---

## 7. Current State (The "Save Point")

> **UPDATE THIS SECTION AFTER EVERY SUB-TASK**

**Last Action Completed:**
Wrote failing tests for login use case in `tests/auth/login.test.ts`

**Current TDD Phase:** GREEN (implementing to pass tests)

**Current Context:**
- Working on: User Authentication System
- Sub-task: Implement login use case
- File being edited: `src/application/auth/login.ts`
- Test status: 3 passing, 1 failing (expected)

**Next Immediate Step:**
Implement password verification in `login.ts` to make the failing test pass.

**Files Modified This Session:**
- `tests/auth/login.test.ts` (created)
- `src/domain/auth/user.ts` (created)
- `src/domain/auth/credentials.ts` (created)
- `AGENTS.md` (updated)

**Uncommitted Changes:**
```
M  src/application/auth/login.ts
A  tests/auth/login.test.ts
M  AGENTS.md
```

**Active MCP Guidelines Being Followed:**
- `[language].md`: [Language-specific conventions being followed]
- `[architecture].md`: [Architecture pattern being applied]
- `tdd.md`: RED-GREEN-REFACTOR cycle
- `todo.md`: State tracking via TODOS.md (STATE-FIRST principles)
- code guide MCP for applicable technologies
- any other MCP specified or requested by the user

---

## 8. Known Issues & Blockers

### Active Issues
| ID | Description | Severity | Test | Status |
|----|-------------|----------|------|--------|
| BUG-001 | Login fails with special chars | High | `auth.test.ts:89` | Test written |
| BUG-002 | Session expires too quickly | Medium | - | Investigating |

### Blockers
| ID | Description | Impact | Workaround | Owner |
|----|-------------|--------|------------|-------|
| BLOCK-001 | Waiting for API keys | Cannot test payment | Using mock | DevOps |

### Resolved Issues
| ID | Description | Resolution | Test | Date |
|----|-------------|------------|------|------|
| BUG-000 | CORS errors | Added headers in API | `api.test.ts:12` | 2026-01-20 |

---

## 9. Decision Log

### Recent Decisions
| Date | Decision | Rationale | ADR | Impact |
|------|----------|-----------|-----|--------|
| 2026-01-22 | Use Result type | Explicit errors | ADR-003 | All services |
| 2026-01-21 | Hexagonal arch | Testability | ADR-002 | Structure |
| 2026-01-20 | TypeScript strict | Fewer bugs | ADR-001 | All code |

### Pending Decisions
- [ ] Database: PostgreSQL vs SQLite for dev
- [ ] Auth: Session-based vs JWT
- [ ] Deploy: Vercel vs Railway

---

## 10. Session Handoff Notes

**For the next session/agent:**

1. Read this entire file first
2. Run `npm test` to see current test state (1 failing is expected)
3. The failing test is intentional (TDD GREEN phase)
4. Focus on making `login.test.ts:67` pass
5. Do NOT refactor yet - stay in GREEN phase
6. After passing, update Test State section
7. Follow `typescript.md` and `hexagonal.md` MCPs

**Open Questions for User:**
- Should password have minimum length validation?
- Preferred JWT expiration time?
- Need OAuth providers (Google, GitHub)?

**Dependencies Needed:**
- `bcrypt` for password hashing (not yet installed)
- `jsonwebtoken` for JWT (not yet installed)

---

## 11. Quick Commands

```bash
# Development (adapt to your runtime/language)
# Node: npm run dev | Python: python manage.py runserver | Go: go run .
# Rust: cargo run | Java: mvn spring-boot:run | Flutter: flutter run

# Testing (TDD workflow - adapt to your test framework)
# Node: npm test | Python: pytest | Go: go test ./...
# Rust: cargo test | Java: mvn test | Flutter: flutter test

# Watch mode for TDD
# Node: npm test -- --watch | Python: pytest-watch | Go: gowatch
# Rust: cargo watch -x test

# Code Quality (adapt to your linter/formatter)
# Node: npm run lint | Python: ruff check . | Go: golangci-lint run
# Rust: cargo clippy | Java: mvn checkstyle:check

# State Management
cat AGENTS.md         # Read current state
cat TODOS.md          # Read task state (see todo.md)
git diff AGENTS.md    # See state changes
git diff TODOS.md     # See task changes
```

---

## 12. Environment & Secrets

### Required Environment Variables
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `DATABASE_URL` | PostgreSQL connection | Yes | - |
| `JWT_SECRET` | Token signing key | Yes | - |
| `NODE_ENV` | Environment mode | No | development |

### Environment Files
- `.env.example` - Template (committed)
- `.env.local` - Local development (gitignored)
- `.env.test` - Test environment (gitignored)

---

**Last Updated:** 2026-01-22T15:30:00Z
**Updated By:** Claude (Session #1)
**Next Review:** Before next task starts
**MCP Compliance:** typescript.md, reactjs.md, hexagonal.md, tdd.md, github.md
```

### B. Section Requirements Matrix

| Section | Purpose | When to Update | Priority |
|---------|---------|----------------|----------|
| Project Context | Stable reference | Project changes | High |
| MCP References | Guide compliance | New MCPs added | High |
| Architecture | Design decisions | Pattern changes | Medium |
| Master Task List | Progress tracking | After each task | Critical |
| Test State | TDD tracking | After each test | Critical |
| Documentation State | Docs sync | After doc changes | Medium |
| Current State | Session continuity | After EVERY sub-task | Critical |
| Known Issues | Bug tracking | Issues found/fixed | High |
| Decision Log | Audit trail | Decisions made | Medium |
| Session Handoff | Next session prep | Before stopping | Critical |

---

## 4. Creating AGENTS.md for Existing Projects

### A. Discovery Protocol

**When encountering a project without AGENTS.md:**

```bash
#!/bin/bash
# AGENTS.md Discovery Script (Language-Agnostic)

echo "=== Project Analysis ==="

# 1. Detect package managers and languages
echo "## Tech Stack Detection"

# JavaScript/TypeScript
[ -f "package.json" ] && echo "- Node.js/JavaScript/TypeScript project detected"
[ -f "tsconfig.json" ] && echo "  - TypeScript configuration found"
[ -f "bun.lockb" ] && echo "  - Bun runtime detected"
[ -f "deno.json" ] && echo "  - Deno runtime detected"

# Python
[ -f "pyproject.toml" ] && echo "- Python project detected (pyproject.toml)"
[ -f "setup.py" ] && echo "- Python project detected (setup.py)"
[ -f "requirements.txt" ] && echo "- Python dependencies found"
[ -f "Pipfile" ] && echo "  - Pipenv detected"
[ -f "poetry.lock" ] && echo "  - Poetry detected"

# Rust
[ -f "Cargo.toml" ] && echo "- Rust project detected"

# Go
[ -f "go.mod" ] && echo "- Go project detected"

# Java/Kotlin
[ -f "pom.xml" ] && echo "- Maven/Java project detected"
[ -f "build.gradle" ] && echo "- Gradle project detected"
[ -f "build.gradle.kts" ] && echo "- Gradle/Kotlin project detected"

# C/C++
[ -f "CMakeLists.txt" ] && echo "- CMake/C++ project detected"
[ -f "Makefile" ] && echo "- Make-based project detected"
[ -f "meson.build" ] && echo "- Meson project detected"

# .NET
ls *.csproj *.fsproj 2>/dev/null && echo "- .NET project detected"
[ -f "*.sln" ] && echo "  - Solution file found"

# Dart/Flutter
[ -f "pubspec.yaml" ] && echo "- Dart/Flutter project detected"

# Ruby
[ -f "Gemfile" ] && echo "- Ruby project detected"

# Haskell
[ -f "stack.yaml" ] && echo "- Haskell/Stack project detected"
[ -f "cabal.project" ] && echo "- Haskell/Cabal project detected"

# Elixir
[ -f "mix.exs" ] && echo "- Elixir project detected"

# 2. Detect frameworks (sampling across languages)
echo ""
echo "## Framework Detection"
grep -q "react\|\"next\"" package.json 2>/dev/null && echo "- React/Next.js detected"
grep -q "vue" package.json 2>/dev/null && echo "- Vue.js detected"
grep -q "angular" package.json 2>/dev/null && echo "- Angular detected"
grep -q "svelte" package.json 2>/dev/null && echo "- Svelte detected"
grep -q "express" package.json 2>/dev/null && echo "- Express detected"
grep -q "fastapi\|flask\|django" pyproject.toml requirements.txt 2>/dev/null && echo "- Python web framework detected"
grep -q "actix\|axum\|rocket" Cargo.toml 2>/dev/null && echo "- Rust web framework detected"
grep -q "gin\|echo\|fiber" go.mod 2>/dev/null && echo "- Go web framework detected"
grep -q "spring" pom.xml build.gradle 2>/dev/null && echo "- Spring framework detected"

# 3. Detect testing frameworks
echo ""
echo "## Testing Framework Detection"
grep -q "vitest\|jest\|mocha" package.json 2>/dev/null && echo "- JS/TS testing framework detected"
grep -q "pytest" pyproject.toml requirements.txt 2>/dev/null && echo "- pytest detected"
[ -d "tests" ] || [ -d "test" ] && echo "- Test directory found"
ls *_test.go 2>/dev/null && echo "- Go test files found"
ls *_test.rs 2>/dev/null && echo "- Rust test files found"

# 4. Detect CI/CD
echo ""
echo "## CI/CD Detection"
[ -d ".github/workflows" ] && echo "- GitHub Actions detected"
[ -f ".gitlab-ci.yml" ] && echo "- GitLab CI detected"
[ -f "Jenkinsfile" ] && echo "- Jenkins detected"
[ -f "azure-pipelines.yml" ] && echo "- Azure DevOps detected"
[ -f ".circleci/config.yml" ] && echo "- CircleCI detected"
[ -f ".travis.yml" ] && echo "- Travis CI detected"

# 5. Detect architecture patterns (language-agnostic folders)
echo ""
echo "## Architecture Detection"
[ -d "src/domain" ] || [ -d "domain" ] && echo "- Domain layer detected (DDD/Hexagonal)"
[ -d "src/infrastructure" ] || [ -d "infrastructure" ] && echo "- Infrastructure layer detected"
[ -d "src/application" ] || [ -d "application" ] && echo "- Application layer detected"
[ -d "src/adapters" ] || [ -d "adapters" ] && echo "- Adapters layer detected (Hexagonal)"
[ -d "src/ports" ] || [ -d "ports" ] && echo "- Ports layer detected (Hexagonal)"
[ -d "cmd" ] && echo "- Go cmd pattern detected"
[ -d "internal" ] && echo "- Go internal pattern detected"
[ -d "pkg" ] && echo "- Go pkg pattern detected"

# 6. Check for existing state files
echo ""
echo "## State Files"
[ -f "AGENTS.md" ] && echo "- AGENTS.md exists (good!)" || echo "- AGENTS.md NOT FOUND - needs creation"
[ -f "TODOS.md" ] && echo "- TODOS.md exists (good!)" || echo "- TODOS.md NOT FOUND - needs creation (see todo.md)"
```

### B. Auto-Population Strategy

```markdown
# When creating AGENTS.md for existing project:

## 1. Populate Tech Stack
- Read package.json/pyproject.toml for dependencies
- Identify primary language version
- List all frameworks in use

## 2. Identify Applicable MCPs
- Match detected tech to available guides:
  - Language detected → language.md guide
  - Framework detected → framework.md guide
  - CI/CD detected → ci-cd.md guide

## 3. Infer Architecture
- Check folder structure for patterns
- Look for interface/port definitions
- Identify layer separation

## 4. Discover Test State
- Run existing tests to get baseline
- Calculate current coverage
- Identify test framework configuration

## 5. Extract TODOs from Code
- Scan for TODO/FIXME comments
- Group by file/module
- Add to Master Task List backlog

## 6. Check Documentation
- List existing docs
- Identify generators in use
- Note what needs updating
```

### C. Example Creation Flow

```
# Pseudocode for AGENTS.md creation (language-agnostic)

function createAgentsMd(projectPath):
    # 1. Analyze project
    techStack = detectTechStack(projectPath)
    architecture = inferArchitecture(projectPath)
    testState = analyzeTests(projectPath)
    todos = extractTodos(projectPath)
    docs = analyzeDocs(projectPath)

    # 2. Match MCPs based on detected tech
    applicableMcps = matchMcps(techStack, architecture)

    # 3. Generate AGENTS.md content
    content = generateAgentsMd({
        techStack: techStack,
        architecture: architecture,
        testState: testState,
        todos: todos,
        docs: docs,
        mcps: applicableMcps
    })

    # 4. Write file to project root
    writeFile(projectPath + '/AGENTS.md', content)

    # 5. Also create or update TODOS.md per todo.md guidelines
    if not exists(projectPath + '/TODOS.md'):
        createTodosMd(projectPath, todos)

    # 6. Verify
    print('AGENTS.md created successfully')
    print('Applicable MCPs: ' + join(applicableMcps, ', '))
```

---

## 5. MCP Integration Requirements

### A. MCP Reference Section

**Every AGENTS.md MUST include MCP references:**

```markdown
## MCP & Code Guide References

### Primary Guides (Must Follow)
| Guide | Version | Compliance | Last Verified |
|-------|---------|------------|---------------|
| typescript.md | 1.0 | Required | 2026-01-22 |
| hexagonal.md | 1.0 | Required | 2026-01-22 |
| tdd.md | 1.0 | Required | 2026-01-22 |

### Secondary Guides (Recommended)
| Guide | Version | Compliance | Notes |
|-------|---------|------------|-------|
| github.md | 1.0 | Recommended | Using GitHub Actions |
| rest.md | 1.0 | Recommended | API design |

### Guide Deviations
| Guide | Section | Deviation | Rationale |
|-------|---------|-----------|-----------|
| typescript.md | 4.2 | Using `any` in tests | Legacy test utils |
```

### B. MCP Compliance Checks

```markdown
### MCP Compliance Status

#### typescript.md
- [x] Strict mode enabled
- [x] No implicit any
- [x] Explicit return types
- [ ] All exports documented (in progress)

#### hexagonal.md
- [x] Domain layer isolated
- [x] Ports defined as interfaces
- [x] Adapters in infrastructure/
- [ ] No framework in domain (1 violation)

#### tdd.md
- [x] Tests before implementation
- [x] Red-Green-Refactor cycle
- [x] Regression tests for bugs
- [x] Test state documented
```

---

## 6. Spec-Coding Integration

### A. When Spec-Coding Instructions Exist

**If user provides specific coding instructions:**

```markdown
## Spec-Coding Instructions

> **PRIORITY:** These instructions take precedence over default MCP
> guidelines, but MUST remain MCP-compliant in structure.

### User-Provided Specifications
```
[Paste exact spec-coding instructions here]
```

### Spec-to-MCP Mapping
| Spec Instruction | Applicable MCP | Compliance |
|------------------|----------------|------------|
| "Use functional components" | reactjs.md | ✅ Aligned |
| "No class syntax" | typescript.md | ✅ Aligned |
| "Custom error format" | rest.md | ⚠️ Deviation |

### Spec Deviations from MCP
| Spec | MCP Default | Resolution |
|------|-------------|------------|
| Error format | RFC 7807 | Use spec format (documented) |
```

### B. Spec Priority Rules

```markdown
### Priority Order for Instructions

1. **User Spec-Coding** (highest)
   - Direct user instructions
   - Project-specific requirements

2. **Project AGENTS.md**
   - Established project conventions
   - Team decisions

3. **MCP Guidelines**
   - Code guide standards
   - Best practices

4. **Language/Framework Defaults** (lowest)
   - Built-in conventions
   - Community standards

### Conflict Resolution
When specs conflict with MCPs:
1. Document the conflict in Decision Log
2. Follow spec if MCP-compliant in structure
3. Note deviation in MCP Compliance section
4. Ensure tests still pass
```

---

## 7. Testing (MANDATORY)

### A. AGENTS.md Validation Tests

**Note:** Adapt the test syntax to your project's testing framework.

#### Python (pytest) Example
```python
# tests/state/test_agents_md.py

import os
import re
from datetime import datetime, timedelta

def test_agents_md_exists():
    assert os.path.exists('AGENTS.md'), "AGENTS.md must exist at project root"

def test_required_sections():
    with open('AGENTS.md', 'r') as f:
        content = f.read()

    assert '## 1. Project Context' in content
    assert '## 2. MCP & Code Guide References' in content
    assert '## 7. Current State' in content
    assert '## 5. Test State' in content

def test_has_tech_stack():
    with open('AGENTS.md', 'r') as f:
        content = f.read()
    assert '**Tech Stack:**' in content

def test_mcp_references():
    with open('AGENTS.md', 'r') as f:
        content = f.read()
    assert re.search(r'\.md.*\|.*Active', content)
```

#### JavaScript/TypeScript (any test framework) Example
```javascript
// tests/state/agents-md.test.js

const fs = require('fs');

describe('AGENTS.md Validation', () => {
  let content;

  beforeAll(() => {
    content = fs.readFileSync('AGENTS.md', 'utf-8');
  });

  test('should have required sections', () => {
    expect(content).toContain('## 1. Project Context');
    expect(content).toContain('## 2. MCP & Code Guide References');
    expect(content).toContain('## 7. Current State');
    expect(content).toContain('## 5. Test State');
  });

  test('should have tech stack defined', () => {
    expect(content).toMatch(/\*\*Tech Stack:\*\*/);
  });

  test('should reference todo.md for state management', () => {
    expect(content).toMatch(/todo\.md/i);
  });
});
```

#### Bash Script Validation
```bash
#!/bin/bash
# scripts/validate-agents-md.sh

set -e

echo "Validating AGENTS.md..."

# Check file exists
[ -f "AGENTS.md" ] || { echo "ERROR: AGENTS.md not found"; exit 1; }

# Check required sections
for section in "Project Context" "MCP & Code Guide" "Current State" "Test State"; do
  grep -q "$section" AGENTS.md || { echo "ERROR: Missing section: $section"; exit 1; }
done

# Check for MCP references
grep -q "\.md.*|.*Active" AGENTS.md || { echo "WARNING: No active MCPs found"; }

# Check for todo.md reference
grep -qi "todo\.md" AGENTS.md || { echo "WARNING: No todo.md reference"; }

echo "AGENTS.md validation passed"
```

### B. Pre-Commit Validation

```bash
#!/bin/bash
# .git/hooks/pre-commit
# Language-agnostic AGENTS.md validation

# Ensure AGENTS.md exists
if [ ! -f "AGENTS.md" ]; then
  echo "ERROR: AGENTS.md not found!"
  echo "Create AGENTS.md before committing (see agents-md.md guide)"
  exit 1
fi

# Ensure TODOS.md exists (per todo.md guide)
if [ ! -f "TODOS.md" ]; then
  echo "WARNING: TODOS.md not found"
  echo "Consider creating TODOS.md for task tracking (see todo.md guide)"
fi

# Check for required sections
required_sections=(
  "Project Context"
  "MCP & Code Guide"
  "Current State"
  "Test State"
)

for section in "${required_sections[@]}"; do
  if ! grep -q "$section" AGENTS.md; then
    echo "ERROR: AGENTS.md missing section: $section"
    exit 1
  fi
done

# Check for todo.md reference
if ! grep -qi "todo\.md" AGENTS.md; then
  echo "WARNING: AGENTS.md should reference todo.md for state management"
fi

# Check timestamp freshness
last_updated=$(grep -oP 'Last Updated:\*\* \K[\d-T:Z]+' AGENTS.md 2>/dev/null || true)
if [ -n "$last_updated" ]; then
  last_epoch=$(date -d "$last_updated" +%s 2>/dev/null || echo 0)
  now_epoch=$(date +%s)
  days_old=$(( (now_epoch - last_epoch) / 86400 ))

  if [ "$days_old" -gt 7 ]; then
    echo "WARNING: AGENTS.md last updated $days_old days ago"
    echo "Consider updating the Current State section"
  fi
fi

echo "AGENTS.md validation passed"
```

---

## 8. Session Workflow

### A. Session Start

```
┌─────────────────────────────────────────────────────────────┐
│                    SESSION START                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Check AGENTS.md      │
                │  exists?              │
                └───────────────────────┘
                     │            │
                    Yes           No
                     │            │
                     ▼            ▼
        ┌────────────────┐  ┌────────────────┐
        │ Read AGENTS.md │  │ Create from    │
        │ completely     │  │ template       │
        └────────────────┘  └────────────────┘
                     │            │
                     └─────┬──────┘
                           ▼
                ┌───────────────────────┐
                │  Identify applicable  │
                │  MCPs                 │
                └───────────────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │  Acknowledge state    │
                │  in response          │
                └───────────────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │  Begin work from      │
                │  "Next Immediate Step"│
                └───────────────────────┘
```

### B. During Session

```
For each sub-task:

    ┌──────────────────┐
    │ Follow TDD cycle │
    └──────────────────┘
            │
            ▼
    ┌──────────────────┐
    │ Follow MCP       │
    │ guidelines       │
    └──────────────────┘
            │
            ▼
    ┌──────────────────┐
    │ Update Test State│
    └──────────────────┘
            │
            ▼
    ┌──────────────────┐
    │ Update Current   │
    │ State section    │
    └──────────────────┘
            │
            ▼
    ┌──────────────────┐
    │ Commit changes   │
    │ with AGENTS.md   │
    └──────────────────┘
```

### C. Session End

```
┌─────────────────────────────────────────────────────────────┐
│                    SESSION END                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Update ALL sections: │
                │  - Current State      │
                │  - Test State         │
                │  - Task List          │
                │  - Known Issues       │
                └───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Write Handoff Notes  │
                │  - For next session   │
                │  - Open questions     │
                │  - Dependencies       │
                └───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Update timestamp     │
                │  and session ID       │
                └───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Commit AGENTS.md     │
                └───────────────────────┘
```

---

## 9. Quick Reference

### AGENTS.md Creation Checklist

```
[ ] Check if AGENTS.md exists
[ ] If missing, create from template
[ ] Detect tech stack automatically
[ ] Identify applicable MCPs
[ ] Run existing tests for baseline
[ ] Scan for TODO comments
[ ] Document current state
[ ] Set up validation tests
```

### Required Sections Checklist

```
[ ] Project Context (goal, tech stack, constraints)
[ ] MCP References (applicable guides, compliance)
[ ] Architecture (patterns, file structure)
[ ] Master Task List (phases, tasks)
[ ] Test State (TDD phase, coverage, pending)
[ ] Documentation State (generators, status)
[ ] Current State (save point)
[ ] Known Issues (bugs, blockers)
[ ] Decision Log (rationale trail)
[ ] Session Handoff (next session prep)
[ ] Quick Commands (common operations)
```

### MCP Mapping Quick Reference

| Project Type | Primary MCPs |
|--------------|--------------|
| Python + Django/FastAPI | python.md, rest.md, tdd.md, todo.md |
| Rust + Actix/Axum | rust.md, rest.md, tdd.md, todo.md |
| Go + Gin/Echo | go.md, rest.md, tdd.md, todo.md |
| TypeScript + React/Node | typescript.md, nodejs.md, tdd.md, todo.md |
| Java + Spring | java.md, rest.md, tdd.md, todo.md |
| C# + .NET | csharp.md, rest.md, tdd.md, todo.md |
| Flutter/Dart | flutter.md, tdd.md, todo.md |
| C++ | cpp.md, cmake.md, tdd.md, todo.md |
| Kotlin | kotlin.md, tdd.md, todo.md |
| Haskell | haskell.md, tdd.md, todo.md |

**Note:** Always include `todo.md` for state management and `tdd.md` for testing standards.

### Common Commands

```bash
# Check for AGENTS.md and TODOS.md
test -f AGENTS.md && echo "AGENTS.md Found" || echo "AGENTS.md Missing"
test -f TODOS.md && echo "TODOS.md Found" || echo "TODOS.md Missing"

# Validate structure
grep -E "^## [0-9]+\." AGENTS.md

# Check MCP references
grep "\.md.*Active" AGENTS.md

# Check for todo.md compliance
grep -i "todo\.md\|TODOS\.md" AGENTS.md

# Find TODO comments in code (language-agnostic patterns)
grep -rn "TODO\|FIXME\|HACK" --include="*.py" --include="*.rs" --include="*.go" \
  --include="*.ts" --include="*.js" --include="*.java" --include="*.kt" \
  --include="*.cpp" --include="*.c" --include="*.hs" src/ || true

# Run project tests (adapt to your language)
# Python: pytest
# Rust: cargo test
# Go: go test ./...
# Node: npm test
# Java: mvn test / gradle test
```

---

## 10. Deployment Checklist

### Pre-Commit Verification

#### AGENTS.md
- [ ] File exists at project root
- [ ] All required sections present
- [ ] Current State is up-to-date
- [ ] Test State reflects actual coverage
- [ ] MCP references are current
- [ ] Timestamp updated

#### MCP Compliance
- [ ] All referenced MCPs followed
- [ ] Deviations documented
- [ ] Compliance checklist current

#### Test State
- [ ] TDD phase documented
- [ ] Coverage report current
- [ ] Pending tests listed
- [ ] No failing tests (unless TDD RED)

#### Documentation
- [ ] Doc state reflects reality
- [ ] Generated docs up-to-date
- [ ] ADRs current

---

## 11. Why This Configuration Works

**Single Source of Truth**:
- Eliminates context fragmentation
- Any agent can resume immediately
- Complete project understanding in one file

**AGENTS.md + TODOS.md Integration**:
- AGENTS.md: Project context, MCP compliance, architecture decisions
- TODOS.md: Task tracking, state management, TODO synchronization
- Together: Complete project state following STATE-FIRST principles (see [todo.md](./todo.md))

**Language-Agnostic Design**:
- Works with any programming language or framework
- MCP guides selected based on detected tech stack
- Consistent patterns regardless of language choice

**MCP Integration**:
- Consistent code quality
- Best practices enforced
- Clear guidelines for all decisions

**Test State Tracking**:
- TDD progress visible
- Coverage always known
- No surprise failures

**Documentation Sync**:
- Docs match code state
- No stale documentation
- Clear what needs updating

**Session Continuity**:
- Perfect handoffs between sessions
- No repeated questions
- Immediate productivity

---

## References

### Required Companion Guides
- [todo.md](./todo.md) - **REQUIRED**: TODO and state management guidelines (STATE-FIRST principles, TODOS.md format)
- [tdd.md](./tdd.md) - **REQUIRED**: Test-driven development guide (Red-Green-Refactor cycle)

### Architecture Guides
- [hexagonal.md](./hexagonal.md) - Hexagonal architecture guide
- [cleanarch.md](./cleanarch.md) - Clean architecture guide
- [microservices.md](./microservices.md) - Microservices patterns

### Language-Specific Guides
Select based on your project's tech stack:
- `python.md`, `rust.md`, `go.md`, `typescript.md`, `java.md`, `kotlin.md`, `cpp.md`, `haskell.md`, etc.

### Framework Guides
- `reactjs.md`, `nodejs.md`, `flutter.md`, `angular.md`, `svelte.md`, etc.

### CI/CD Guides
- `github.md`, `gitlab.md`, `jenkins.md`, `azuredevops.md`

---

**Last Updated:** 2026-01-22
**Version:** 1.0
**Maintainer:** Development Team
