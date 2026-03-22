# Coding AI Agent Guidelines
Mandatory standards and best practices for effective collaboration with AI coding agents. Context engineering, prompt design, verification workflows, security guardrails, human oversight, and quality gates. Claude Code, GitHub Copilot, Cursor, Windsurf, and any AI-assisted development tool.

---

**Agent Profile**: The AI-Augmented Developer
**Role**: Senior Software Engineer & AI Collaboration Specialist
**Objective**: Maximize the quality, security, and velocity of AI-assisted software development through disciplined workflows, context engineering, and verification guardrails.
**Tools**: AI coding agents (Claude Code, Copilot, Cursor, Windsurf, Cody, etc.), project context files (CLAUDE.md, AGENTS.md, .cursorrules, copilot-instructions.md), testing frameworks, linters, security scanners, code review platforms.

---

## 1. Core Philosophies: VERIFIED-FIRST

The developer must adhere to the **VERIFIED-FIRST** principles for every AI-assisted implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE asking the agent to implement — tests are the specification the agent codes against.
**Regression Shield**: EVERY bug discovered in AI-generated code MUST receive a test BEFORE fixing to prevent regression.
**Security-First**: AI-generated code is untrusted by default — mandatory review, scanning, and verification before merging.

- **V**erify Everything: Never blindly accept AI output. Every generated line must be reviewed, tested, and understood.
- **E**ngineer Context: Provide the right information in the right format — context engineering is the #1 lever for output quality.
- **R**eview Like Production Code: AI-generated code gets the same (or stricter) review as human-written code.
- **I**terate in Small Steps: Break work into focused, verifiable chunks — one function, one bug, one feature at a time.
- **F**ail Fast with Guardrails: Use type checkers, linters, and tests as immediate feedback loops the agent can iterate against.
- **I**nstruct, Don't Hope: Explicit instructions beat implicit expectations. If you want a behavior, write a rule for it.
- **E**scalate on Uncertainty: When the agent is unsure or the task is high-risk, pause and involve human judgment.
- **D**ocument Decisions: Record why the agent was given specific instructions, not just what it produced.

**Additional Principles:**

- **Human in the Loop**: Humans own architecture, design, and security decisions. Agents execute within approved boundaries.
- **Proportional Autonomy**: More autonomous agents require proportionally more guardrails and verification.
- **Context Over Cleverness**: Clear structure and context matter more than clever prompting — most failures come from ambiguity.
- **Tool Layering**: Agents don't compete — they layer. Editor assistants, coding agents, and review agents each have their role.
- **Reproducible Instructions**: Another developer (or a fresh agent session) should produce equivalent results from the same instructions.

**Verified Output**: AI-generated code MUST compile, pass tests, pass linting, and pass security scans before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: ALL AI-generated code MUST pass verification before acceptance. Treat every output as untrusted until proven correct.**

#### Pre-Acceptance Checklist

**Before accepting ANY AI-generated code, the developer MUST verify:**

1. **Compilation / Syntax Check**:
   ```bash
   # Verify code compiles / parses without errors
   # Use the appropriate tool for your language:
   tsc --noEmit                    # TypeScript
   python -m py_compile file.py    # Python
   go build ./...                  # Go
   cargo check                     # Rust
   ```
   - **MUST** compile with zero errors
   - **MUST** produce zero new warnings
   - All imports and dependencies MUST be real, existing packages

2. **Test Execution**:
   ```bash
   # Run the full test suite
   npm test                        # Node.js
   pytest                          # Python
   go test ./...                   # Go
   cargo test                      # Rust

   # Verify coverage hasn't decreased
   pytest --cov --cov-fail-under=80
   ```
   - **MUST** pass all existing tests (zero regressions)
   - **MUST** include tests for new functionality
   - Coverage MUST NOT decrease

3. **Lint & Format Check**:
   ```bash
   # Verify code style compliance
   eslint . && prettier --check .  # JavaScript/TypeScript
   ruff check . && ruff format --check .  # Python
   golangci-lint run               # Go
   cargo clippy && cargo fmt --check  # Rust
   ```
   - **MUST** pass all linting rules
   - **MUST** match project formatting standards

4. **Security Scan**:
   ```bash
   # Scan for vulnerabilities
   npm audit                       # Node.js
   pip-audit                       # Python
   govulncheck ./...               # Go
   cargo audit                     # Rust

   # Scan for secrets
   gitleaks detect --source .
   trufflehog filesystem .
   ```
   - **MUST** have zero high/critical vulnerabilities
   - **MUST** have zero hardcoded secrets
   - All new dependencies MUST be verified as real, maintained packages

5. **Dependency Verification (CRITICAL)**:
   ```bash
   # Verify all packages exist and are legitimate
   # AI agents hallucinate packages — ALWAYS verify
   npm info <package-name>         # Check npm
   pip show <package-name>         # Check pip
   ```
   - **MUST** verify every new dependency the agent introduces
   - Check for "slopsquatting" — AI-hallucinated package names that attackers register
   - Verify license compatibility
   - Check last publish date and maintainer activity

#### Error Correction Process

If verification fails:

1. **Compilation Errors**:
   - Feed the exact error message back to the agent
   - Provide surrounding context (file, function)
   - Let the agent iterate — but cap at 3 attempts before manual intervention

2. **Test Failures**:
   - Distinguish between broken existing tests (regression) and failing new tests
   - For regressions: revert and re-prompt with clearer constraints
   - For new test failures: let the agent iterate with test output

3. **Hallucinated Dependencies**:
   - Remove the fabricated package immediately
   - Ask the agent to solve the problem using only real, specified libraries
   - Provide a list of approved/available packages if possible

4. **Security Findings**:
   - Never override security findings without human review
   - Ask the agent to remediate, but verify the fix independently
   - Escalate critical findings to the security team

### B. Agent Workflow — The Verified Loop

**Complete AI-assisted development workflow:**

```
1. PLAN: Define the task with clear acceptance criteria
   ↓
2. CONTEXT: Provide the agent with relevant files, rules, and constraints
   ↓
3. TEST FIRST: Write (or have agent write) failing tests that define success
   ↓
4. GENERATE: Ask the agent to implement code that passes the tests
   ↓
5. VERIFY: Run compilation, tests, linting, security scans
   ↓
6. REVIEW: Human reads and understands every change
   ↓
7. ITERATE: If verification fails, feed errors back (max 3 cycles)
   ↓
8. ACCEPT: Only after ALL checks pass and human approves
```

### C. Prohibited Practices

**NEVER accept AI-generated code that:**
- [ ] Fails compilation or type checking
- [ ] Has failing tests or reduces coverage
- [ ] Introduces dependencies you haven't verified exist
- [ ] Contains hardcoded secrets, tokens, or credentials
- [ ] You don't understand — if you can't explain it, don't ship it
- [ ] Modifies files outside the requested scope ("scope creep")
- [ ] Bypasses security checks or disables safety mechanisms
- [ ] Was accepted without any human review ("rubber-stamping")
- [ ] **Fixes bugs without adding regression tests first**
- [ ] **Was generated from a vague, unscoped prompt**
- [ ] **Uses patterns that contradict project conventions**

---

## 2A. Test-Driven Development with AI Agents (MANDATORY)

**CRITICAL: TDD is the single most effective guardrail for AI-generated code quality.**

### Why TDD + AI Agents

Tests are unambiguous specifications. When you give an agent a failing test, you give it:
- A clear, verifiable goal
- An automated feedback loop
- A boundary it cannot silently violate

Without tests, you're relying on the agent's interpretation of natural language — which is where hallucinations thrive.

### TDD Cycle with AI

```
1. RED: Human (or agent) writes a failing test that defines the requirement
   ↓
2. GREEN: Agent writes minimal code to make the test pass
   ↓
3. VERIFY: Human reviews that the implementation is correct, not just passing
   ↓
4. REFACTOR: Agent improves code while tests stay green
   ↓
   Repeat
```

### Example: TDD Workflow with a Coding Agent

```markdown
## Prompt to Agent (Step 1 — RED):

Write a test for a `calculateDiscount` function that:
- Returns 0% discount for orders under $50
- Returns 10% discount for orders $50-$99.99
- Returns 20% discount for orders $100+
- Throws an error for negative amounts

Do NOT write the implementation yet — only the test.

## Prompt to Agent (Step 2 — GREEN):

The test is failing as expected. Now implement `calculateDiscount`
to make all tests pass. Use minimal code.

## Human Review (Step 3 — VERIFY):

Developer reviews:
- Does the implementation match business intent?
- Are there edge cases the tests missed?
- Is the code clear and maintainable?

## Prompt to Agent (Step 4 — REFACTOR):

The tests pass. Refactor for clarity — extract the discount
tiers into a configuration object. Keep all tests passing.
```

### Anti-Pattern: Implementation Before Tests

```
WRONG: "Write a calculateDiscount function and add some tests"
→ Agent writes implementation, then writes tests that match the implementation
→ Tests validate what the agent decided, not what you need
→ Bugs hide in the gap between intent and implementation

RIGHT: "Write tests for calculateDiscount with these rules: [...]"
→ Tests encode YOUR requirements
→ Agent must satisfy YOUR specification
→ Mismatches surface immediately
```

---

## 2B. Bug Fix Protocol for AI-Generated Code (MANDATORY)

**CRITICAL: Every bug in AI-generated code MUST get a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. Bug Discovered in AI-Generated Code
   ↓
2. Write a test that REPRODUCES the bug (test FAILS)
   ↓
3. Verify the test fails for the right reason
   ↓
4. Ask the agent to fix the bug (make test pass)
   ↓
5. Human verifies the fix doesn't introduce regressions
   ↓
6. Run full test suite
   ↓
7. Document: what went wrong and why the agent produced it
```

### Root Cause Analysis for Agent Bugs

When AI-generated code has bugs, diagnose **why** the agent produced the error:

| Root Cause | Symptom | Prevention |
|------------|---------|------------|
| Vague prompt | Agent guessed wrong behavior | Write specific acceptance criteria |
| Missing context | Agent didn't know about constraints | Include relevant files and rules |
| Hallucinated API | Agent used non-existent method | Provide API references or docs |
| Stale training data | Agent used deprecated patterns | Specify versions and current APIs |
| Context overflow | Agent forgot earlier instructions | Break into smaller tasks |
| Scope creep | Agent changed unrelated code | Set explicit file/scope boundaries |

---

## 3. Context Engineering (MANDATORY)

### A. Project Context Files

**Context engineering is the #1 lever for AI coding agent output quality.** Provide the right information in the right format so the agent can accomplish the task.

#### Standard Context File Locations

| Agent | File | Location |
|-------|------|----------|
| Claude Code | `CLAUDE.md` | Project root, or `.claude/` |
| GitHub Copilot | `copilot-instructions.md` | `.github/` |
| Cursor | `.cursorrules` or rules | `.cursor/rules/` |
| Universal | `AGENTS.md` | Project root |
| Any Agent | Inline prompt | Direct conversation |

#### AGENTS.md — Universal Agent Configuration

**Use AGENTS.md as the universal, agent-agnostic project configuration:**

```markdown
# AGENTS.md

## Project Overview
Brief description of the project, its purpose, and architecture.

## Tech Stack
- Language: TypeScript 5.x (strict mode)
- Framework: Next.js 15 (App Router)
- Database: PostgreSQL with Prisma ORM
- Testing: Vitest + Playwright
- CI: GitHub Actions

## Architecture
- Follow hexagonal architecture (ports & adapters)
- Business logic in `src/core/` — no framework imports
- API routes in `src/app/api/`
- Database adapters in `src/adapters/db/`

## Coding Standards
- No `any` types — use `unknown` and narrow
- All functions must have JSDoc with @param and @returns
- Error handling: use Result<T, E> pattern, no thrown exceptions
- Prefer named exports over default exports

## Testing Requirements
- Unit tests co-located: `foo.ts` → `foo.test.ts`
- Integration tests in `tests/integration/`
- Minimum 80% coverage for `src/core/`
- All API endpoints must have request/response validation tests

## Banned Patterns
- No `console.log` in production code (use structured logger)
- No `eval()` or `Function()` constructors
- No `npm install` without `--save-exact`
- No direct database queries outside adapter layer

## Build & Run
- `npm run dev` — start dev server
- `npm test` — run all tests
- `npm run lint` — lint and type-check
- `npm run build` — production build
```

### B. Context File Best Practices

**Keep context files effective:**

1. **Start Small, Iterate**:
   - Begin with 30-50 lines covering architecture and critical rules
   - Add rules only when the agent makes a mistake you want to prevent
   - Remove rules that are no longer relevant

2. **Stay Under 300 Lines**:
   - Frontier LLMs reliably follow 150-200 instructions
   - Beyond 300 lines, agents start ignoring or contradicting rules
   - Split into directory-level files for large projects

3. **Be Specific, Not Aspirational**:
   ```markdown
   # BAD — vague, aspirational
   Write clean, maintainable code following best practices.

   # GOOD — specific, actionable
   Use early returns to reduce nesting. Max function length: 40 lines.
   Name booleans with is/has/should prefix. No abbreviations in names.
   ```

4. **Include Build Commands**:
   - Agents fail when they don't know how to build, test, or lint
   - Document bootstrap steps, prerequisites, and environment variables
   - Include exact commands, not descriptions of commands

5. **Document Banned Patterns with Reasons**:
   ```markdown
   # BAD — rule without reason
   Don't use moment.js

   # GOOD — rule with reason
   Don't use moment.js — it's deprecated and 300KB.
   Use date-fns or Temporal API instead.
   ```

6. **Version and Maintain**:
   - Context files are code — commit them to version control
   - Review them in PRs alongside code changes
   - Stale instructions are worse than no instructions — they actively mislead

### C. Directory-Level Context

**Use nested context files for large projects:**

```
project/
├── AGENTS.md                    # Global rules
├── src/
│   ├── AGENTS.md                # Source-level rules
│   ├── core/
│   │   └── AGENTS.md            # "No framework imports in core/"
│   ├── api/
│   │   └── AGENTS.md            # "All endpoints need auth middleware"
│   └── adapters/
│       └── AGENTS.md            # "Adapters implement ports from core/"
└── tests/
    └── AGENTS.md                # "Use factories, not fixtures"
```

Rules are additive — nested files inherit from parents and add specificity.

### D. Providing Runtime Context

**Beyond static files, provide dynamic context in prompts:**

```markdown
## Effective Context in Prompts

### Include:
- The specific file(s) to modify (or @ reference them)
- The test file that defines expected behavior
- Error messages or stack traces (verbatim)
- Relevant type definitions or interfaces
- The acceptance criteria for "done"

### Exclude:
- Entire codebases (causes context overflow)
- Unrelated files (dilutes focus)
- Vague descriptions of what you want
- Multiple unrelated tasks in one prompt
```

---

## 4. Prompt Design Patterns (MANDATORY)

### A. The Specification-First Pattern

**ALWAYS define what you want before asking the agent to build it:**

```markdown
## BAD — Vague, open-ended
"Build a user authentication system"

## GOOD — Specification-first
"Implement JWT authentication with these requirements:

1. POST /api/auth/login accepts { email, password }
2. Returns { accessToken, refreshToken } on success
3. Access tokens expire in 15 minutes
4. Refresh tokens expire in 7 days
5. Store refresh tokens in HttpOnly cookies
6. Rate limit: 5 failed attempts per email per 15 minutes

Constraints:
- Use the existing User model in src/core/models/user.ts
- Implement in src/adapters/auth/jwt-auth.ts
- Tests go in src/adapters/auth/jwt-auth.test.ts
- Use jose library for JWT operations (already in package.json)

Write the tests first, then implement."
```

### B. The Scope-Bounded Pattern

**Set explicit boundaries for what the agent may and may not touch:**

```markdown
## Scope-Bounded Prompt

Task: Add pagination to the /api/products endpoint.

MODIFY ONLY:
- src/api/routes/products.ts
- src/core/services/product-service.ts
- tests/api/products.test.ts

DO NOT MODIFY:
- Database schema or migrations
- Other API endpoints
- Shared utilities or types
- package.json

Pagination spec:
- Query params: ?page=1&limit=20
- Response: { data: Product[], meta: { page, limit, total, totalPages } }
- Default limit: 20, max limit: 100
- Invalid page/limit returns 400 with error message
```

### C. The Plan-First Pattern

**For complex tasks, ask the agent to plan before coding:**

```markdown
## Plan-First Prompt

Before writing any code, create a plan:

1. List every file you will create or modify
2. For each file, describe what changes you'll make
3. Identify potential risks or edge cases
4. List any assumptions you're making

Wait for my approval of the plan before implementing.

Task: [description]
```

### D. The Incremental Pattern

**Break large tasks into small, verifiable steps:**

```markdown
## BAD — Monolithic prompt
"Build the entire checkout flow with cart, payment, shipping,
tax calculation, order confirmation, and email notifications"

## GOOD — Incremental steps
Step 1: "Add a Cart model with add/remove/total methods. Write tests first."
Step 2: "Add shipping cost calculation based on weight and zone. Tests first."
Step 3: "Add tax calculation by jurisdiction. Tests first."
Step 4: "Wire cart + shipping + tax into a CheckoutService. Tests first."
Step 5: "Add the POST /api/checkout endpoint. Integration tests."
```

### E. Anti-Patterns to Avoid

| Anti-Pattern | Problem | Fix |
|-------------|---------|-----|
| "Make it work" | No success criteria | Define acceptance tests |
| "Clean up this file" | Unbounded scope | Specify exact changes |
| "Use best practices" | Subjective, vague | Specify which practices |
| Dumping entire codebase | Context overflow | Include only relevant files |
| Multiple tasks per prompt | Agent loses focus | One task per prompt |
| No file boundaries | Scope creep | Explicit MODIFY ONLY list |
| "Fix all the bugs" | Undefined scope | One bug per prompt with repro |

---

## 5. Security Guardrails (MANDATORY)

### A. AI-Generated Code Is Untrusted

**Treat AI-generated code with the same scrutiny as third-party code submissions:**

Research shows 15-25% of AI-generated code suggestions contain security vulnerabilities. Common issues include:

| Vulnerability Category | AI Agent Tendency | Mitigation |
|-----------------------|-------------------|------------|
| SQL Injection | String concatenation over parameterized queries | Enforce ORM/parameterized queries in rules |
| XSS | Missing output encoding | Require template auto-escaping |
| Hardcoded Secrets | Placeholder values left in code | Secret scanner in CI + pre-commit |
| Insecure Defaults | `cors: *`, `debug: true` | Banned patterns in AGENTS.md |
| Dependency Confusion | Hallucinated packages | Verify every new dependency |
| Path Traversal | Unsanitized file paths | Input validation rules |
| Insecure Cryptography | MD5/SHA1 for hashing, weak random | Specify approved crypto in rules |
| Missing Auth Checks | Forgetting middleware | Require auth on all endpoints by default |

### B. Mandatory Security Checks

```bash
# 1. Static Application Security Testing (SAST)
semgrep --config=auto .
# or
bandit -r src/                     # Python
# or
brakeman                           # Ruby on Rails

# 2. Secret Detection (pre-commit + CI)
gitleaks detect --source .
trufflehog filesystem .

# 3. Dependency Vulnerability Scan
npm audit --audit-level=high       # Node.js
pip-audit                          # Python
govulncheck ./...                  # Go
cargo audit                        # Rust

# 4. Software Composition Analysis
syft . -o spdx-json > sbom.json   # Generate SBOM
grype sbom:sbom.json              # Scan SBOM for vulnerabilities

# 5. License Compliance
license-checker --failOn "GPL-3.0;AGPL-3.0"  # Node.js
pip-licenses --fail-on "GPLv3"               # Python
```

### C. Dependency Hygiene

**AI agents frequently hallucinate package names. This is a critical supply chain risk.**

```markdown
## Dependency Verification Protocol

Before accepting ANY new dependency from an agent:

1. Verify the package exists:
   - Check the package registry (npm, PyPI, crates.io, etc.)
   - If it doesn't exist, the agent hallucinated it — reject immediately

2. Verify the package is legitimate:
   - Check weekly download count (low downloads = suspicious)
   - Check publish date (very recent + low downloads = possible typosquat)
   - Check maintainer identity and history
   - Cross-reference with the library's official documentation

3. Verify the package is maintained:
   - Last release within 12 months (or declared stable)
   - Open issues are being triaged
   - No known unpatched vulnerabilities

4. Check license compatibility:
   - Verify license is compatible with your project
   - Flag copyleft licenses (GPL, AGPL) for legal review
```

### D. Sandboxing Agent Execution

**When agents can execute code (Claude Code, Cursor, Copilot Workspace):**

```markdown
## Execution Boundaries

1. Network access: Restrict to known domains (registries, internal APIs)
2. File system: Limit to project directory — no access to ~/.ssh, ~/.aws, etc.
3. Secrets: Never paste secrets into agent prompts or context
4. Permissions: Agents run with minimal privileges — no sudo, no production access
5. Environment: Use development/staging only — never production
6. Review commands: Inspect agent-proposed shell commands before execution
```

---

## 6. Human Oversight & Review (MANDATORY)

### A. Review Tiers

**Scale review depth based on risk:**

| Change Type | Risk Level | Review Requirements |
|------------|-----------|-------------------|
| Formatting / typos | Low | Automated checks sufficient |
| Bug fix with regression test | Low-Medium | Quick human review + test verification |
| New feature (isolated) | Medium | Full code review + test coverage check |
| API changes / public interface | High | Full review + API design discussion |
| Auth / security / payments | Critical | Security-focused review + penetration test |
| Database schema changes | Critical | DBA review + migration rollback plan |
| Infrastructure / CI changes | Critical | Platform team review + dry run |

### B. What to Watch For in AI-Generated Code

**Develop pattern recognition for common agent failure modes:**

1. **Hallucinated APIs**: Agent invents methods that don't exist
   - Fix: Read the actual library docs, verify method signatures

2. **Plausible but Wrong Logic**: Code looks correct but handles edge cases incorrectly
   - Fix: Write edge case tests before accepting

3. **Over-Engineering**: Agent adds abstractions, patterns, or features you didn't ask for
   - Fix: Compare diff against the specific request — reject scope creep

4. **Copy-Paste Drift**: Agent duplicates logic instead of reusing existing code
   - Fix: Ask agent to search for existing implementations first (see Cursor's reuse rule)

5. **Stale Patterns**: Agent uses deprecated APIs or old syntax
   - Fix: Specify versions in AGENTS.md; provide current API docs as context

6. **Silent Behavior Changes**: Agent "improves" existing code while making the requested change
   - Fix: Review the full diff, not just the new code

7. **Incomplete Error Handling**: Happy path works, error paths are wrong or missing
   - Fix: Write tests for error cases explicitly

### C. The Review Prompt

**Use this checklist when reviewing AI-generated changes:**

```markdown
## AI Code Review Checklist

### Correctness
- [ ] Does it do what was requested — and nothing more?
- [ ] Are edge cases handled?
- [ ] Are error paths correct (not just happy path)?
- [ ] Do all new and existing tests pass?

### Security
- [ ] No hardcoded secrets or credentials?
- [ ] Input validation on all external data?
- [ ] Output encoding where needed?
- [ ] Auth/authz checks present?
- [ ] All new dependencies verified as real and maintained?

### Scope
- [ ] Changes limited to requested files/modules?
- [ ] No unrelated "improvements" or refactors?
- [ ] No unnecessary new dependencies?
- [ ] No changes to shared utilities without discussion?

### Quality
- [ ] Code follows project conventions?
- [ ] Tests cover the new behavior?
- [ ] No duplicated logic (reuses existing code)?
- [ ] Names are clear and consistent?

### Understanding
- [ ] Can I explain every line to a colleague?
- [ ] If not → I don't accept it until I can
```

---

## 7. Workflow Patterns (MANDATORY)

### A. The Solo Developer Workflow

```
┌─────────────────────────────────────────────────┐
│              Solo AI-Assisted Workflow            │
│                                                  │
│  1. Define task with acceptance criteria          │
│  2. Write (or generate) failing tests             │
│  3. Prompt agent with context + tests + scope     │
│  4. Agent generates implementation                │
│  5. Run: build → test → lint → security scan     │
│  6. Review diff — understand every change         │
│  7. If issues: feed errors back (max 3 cycles)    │
│  8. Commit with conventional commit message       │
│                                                  │
│  Guardrails: pre-commit hooks, AGENTS.md rules    │
└─────────────────────────────────────────────────┘
```

### B. The Team Workflow

```
┌─────────────────────────────────────────────────┐
│              Team AI-Assisted Workflow            │
│                                                  │
│  1. Issue / ticket with clear requirements        │
│  2. Developer creates branch                      │
│  3. Developer uses agent with shared AGENTS.md    │
│  4. Agent generates code within defined scope     │
│  5. CI pipeline: build → test → lint → scan      │
│  6. AI code review (CodeRabbit, Copilot Review)  │
│  7. Human code review (focus: architecture,       │
│     security, business logic correctness)          │
│  8. Merge after all gates pass                    │
│                                                  │
│  Guardrails: CI gates, CODEOWNERS, branch protect │
└─────────────────────────────────────────────────┘
```

### C. The Autonomous Agent Workflow

**For agents that work asynchronously (Copilot Coding Agent, Claude Code /delegate):**

```
┌─────────────────────────────────────────────────┐
│           Autonomous Agent Workflow               │
│                                                  │
│  1. Assign issue to agent with detailed spec      │
│  2. Agent creates branch and works autonomously   │
│  3. Agent opens draft PR when done                │
│  4. CI pipeline runs automatically                │
│  5. AI reviewer runs automatically                │
│  6. Human reviewer receives notification          │
│  7. Human reviews: correctness + architecture     │
│  8. Request changes or approve                    │
│                                                  │
│  Guardrails: branch protection, CI required,      │
│  CODEOWNERS, no direct merge to main              │
└─────────────────────────────────────────────────┘
```

**Rules for autonomous agents:**

```markdown
## Autonomous Agent Boundaries

MUST:
- Create a new branch (never commit to main)
- Open a draft PR (never merge directly)
- Run all tests before opening PR
- Include a clear description of changes in PR body
- Stay within the scope of the assigned issue

MUST NOT:
- Modify CI/CD pipelines without human approval
- Change database schemas without human approval
- Delete files or data without human approval
- Add new external dependencies without justification
- Access production systems or secrets
- Merge their own PRs
```

### D. The Multi-Agent Workflow

**For complex tasks using multiple specialized agents:**

```
┌─────────────────────────────────────────────────┐
│           Multi-Agent Workflow                    │
│                                                  │
│  Coordinator (Human or Lead Agent)               │
│    ├── Planning Agent: breaks task into subtasks  │
│    ├── Implementation Agent: writes code          │
│    ├── Test Agent: writes and runs tests          │
│    ├── Review Agent: reviews for quality/security │
│    └── Verifier Agent: validates completed work   │
│                                                  │
│  Flow:                                            │
│  Plan → Implement → Test → Review → Verify       │
│  Each agent has read access; only Implementation  │
│  agent has write access to source files.          │
└─────────────────────────────────────────────────┘
```

Example agent definitions (Cursor/Claude format):

```markdown
---
name: verifier
description: Validates completed work. Use after tasks are done.
model: fast
tools: ["Read", "Grep", "Glob", "Bash"]
---

You are a skeptical validator. Test everything.

When invoked:
1. Identify what was claimed complete
2. Check implementation exists and is functional
3. Run relevant tests
4. Look for edge cases missed
5. Report what passed and what's broken

Do not accept claims at face value.
```

---

## 8. Automated Guardrails (MANDATORY)

### A. Pre-Commit Hooks

**Catch issues before they enter version control:**

```yaml
# .pre-commit-config.yaml
repos:
  # Secret detection
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.21.0
    hooks:
      - id: gitleaks

  # Language-specific linting (choose appropriate)
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
      - id: ruff-format

  # Conventional commits
  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v4.0.0
    hooks:
      - id: conventional-pre-commit
        stages: [commit-msg]
```

### B. CI Pipeline Quality Gates

**Mandatory CI checks for AI-assisted development:**

```yaml
# .github/workflows/ai-quality-gate.yml
name: AI Code Quality Gate

on: [pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build
        run: npm run build

      - name: Test
        run: npm test -- --coverage

      - name: Lint
        run: npm run lint

      - name: Security Scan
        run: |
          npm audit --audit-level=high
          npx gitleaks detect --source .

      - name: Dependency Review
        uses: actions/dependency-review-action@v4
        with:
          fail-on-severity: high
          deny-licenses: GPL-3.0, AGPL-3.0

      - name: SBOM Generation
        run: npx @cyclonedx/cyclonedx-npm --output-file sbom.json
```

### C. AI Code Review Automation

**Layer automated AI review with human review:**

```yaml
# .github/workflows/ai-review.yml
name: AI Code Review

on:
  pull_request:
    types: [opened, synchronize, ready_for_review]

jobs:
  ai-review:
    if: github.event.pull_request.draft == false
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      # Option 1: Claude Code review
      - name: AI Review
        run: |
          claude --print "Review this PR diff for:
          - Security vulnerabilities (injection, XSS, auth bypass)
          - Logic errors and edge cases
          - Performance anti-patterns
          - Scope creep beyond the PR description
          Focus on high-severity issues only. Max 10 comments."
```

### D. Branch Protection Rules

**Enforce quality gates at the repository level:**

```markdown
## Required Branch Protection (main)

- [x] Require pull request before merging
- [x] Require at least 1 human approval
- [x] Require status checks to pass (build, test, lint, security)
- [x] Require conversation resolution before merging
- [x] Require signed commits
- [x] Do not allow bypassing of these rules
- [x] Restrict who can push to main (no agents)
```

---

## 9. Context Management for Large Codebases (MANDATORY)

### A. The Context Budget

**AI agents have finite context windows. Manage them deliberately:**

```markdown
## Context Priority (highest to lowest)

1. AGENTS.md / project rules (always loaded)
2. The specific files being modified
3. Test files that define expected behavior
4. Type definitions and interfaces
5. Related files (imports, callers)
6. Error messages and stack traces
7. Documentation and API references

## Context Budget Rules
- Never dump the entire codebase
- Limit to 5-10 files per prompt
- Use @ references or file paths — let the agent read what it needs
- Prefer showing interfaces/types over full implementations
```

### B. Strategies for Large Projects

1. **Use directory-level context files** (see Section 3C)
2. **Reference, don't include**: Point to files; let the agent read them
3. **Provide architectural maps**: A high-level diagram is worth 1000 lines of code
4. **Use tags and markers**: `// AI:BOUNDARY — do not modify below this line`
5. **Create focused workspaces**: Work in a specific module, not the whole repo

### C. Memory and Session Management

**For agents with memory (Claude Code, Cursor):**

```markdown
## Memory Best Practices

DO store in memory:
- Project architecture decisions
- Recurring patterns the agent should follow
- Past mistakes and their corrections
- User preferences for code style

DO NOT store in memory:
- Specific file contents (they change)
- Task-specific details (ephemeral)
- Git history (use git log)
- Things already in AGENTS.md (redundant)
```

---

## 10. Measuring and Improving AI Effectiveness (MANDATORY)

### A. Quality Metrics

**Track these metrics to evaluate AI coding agent effectiveness:**

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| First-pass acceptance rate | How often AI code passes review without changes | >70% |
| Rework cycles | Times code bounces between agent and reviewer | <2 average |
| Test coverage delta | Coverage change after AI contribution | ≥0% (never decreases) |
| Security findings per PR | Vulnerabilities introduced by AI | 0 high/critical |
| Hallucinated dependencies | Fake packages introduced per sprint | 0 |
| Time to verify | Time spent reviewing AI output | Decreasing trend |
| CI pass rate on AI PRs | How often AI PRs pass CI on first push | >85% |

### B. Continuous Improvement Loop

```
1. Track which prompts produce the best results
   ↓
2. Identify recurring failures → add rules to AGENTS.md
   ↓
3. Review and prune AGENTS.md quarterly
   ↓
4. Share effective patterns across the team
   ↓
5. Update onboarding docs to include AI workflow
```

### C. When NOT to Use AI Agents

**AI agents are not the right tool for every task:**

- **Novel architecture decisions**: Agents follow patterns; they don't invent them
- **Security-critical cryptography**: Use vetted libraries, not AI-generated crypto
- **Performance-critical hot paths**: Benchmark manually; don't trust AI optimization claims
- **Regulatory compliance code**: Requires domain expertise and legal review
- **Undocumented legacy systems**: Agent lacks context; will make wrong assumptions
- **Highly ambiguous requirements**: Clarify with humans first, then involve the agent

---

## 11. Project Structure for AI-Assisted Development (MANDATORY)

### A. AI-Friendly Project Layout

**Structure your project to maximize AI agent effectiveness:**

```
project/
├── AGENTS.md                    # Universal agent instructions
├── CLAUDE.md                    # Claude Code specific (if used)
├── .github/
│   ├── copilot-instructions.md  # Copilot specific (if used)
│   └── workflows/
│       ├── ci.yml               # Standard CI pipeline
│       └── ai-review.yml        # AI review pipeline
├── .cursor/
│   └── rules/                   # Cursor rules (if used)
├── docs/
│   ├── architecture.md          # High-level architecture
│   ├── api-reference.md         # API documentation
│   └── decisions/               # Architecture Decision Records
├── src/
│   ├── core/                    # Business logic (agent focus area)
│   │   ├── AGENTS.md            # "No framework imports here"
│   │   ├── models/
│   │   ├── services/
│   │   └── ports/               # Interfaces for adapters
│   ├── adapters/                # External integrations
│   │   └── AGENTS.md            # "Implement ports from core/"
│   └── app/                     # Framework/entry point
├── tests/
│   ├── AGENTS.md                # "Use factories, no fixtures"
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── .pre-commit-config.yaml      # Pre-commit hooks
└── Makefile                     # Common commands
```

### B. Makefile for AI Agents

**Provide a Makefile so agents (and humans) have standardized commands:**

```makefile
.PHONY: help build test lint format security check all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

build: ## Build the project
	npm run build

test: ## Run all tests
	npm test

test-unit: ## Run unit tests only
	npm test -- --testPathPattern=unit

test-integration: ## Run integration tests only
	npm test -- --testPathPattern=integration

lint: ## Run linter and type checker
	npm run lint
	npx tsc --noEmit

format: ## Format code
	npx prettier --write .

security: ## Run security scans
	npm audit --audit-level=high
	npx gitleaks detect --source .

check: lint test security ## Run all checks (use before committing)

all: format check build ## Format, check, and build
```

---

## 12. Deployment Checklist

### AI-Assisted Code Verification (MANDATORY)

**Before merging ANY AI-generated code:**

#### Correctness
- [ ] All tests pass (existing + new)
- [ ] Edge cases covered by tests
- [ ] Error handling verified (not just happy path)
- [ ] No behavior changes outside requested scope

#### Security
- [ ] Zero hardcoded secrets or credentials
- [ ] All new dependencies verified as real and maintained
- [ ] Dependency vulnerability scan: 0 high/critical
- [ ] Input validation on all external data
- [ ] SAST scan passes with zero high findings
- [ ] No new permissions or access patterns without review

#### Quality
- [ ] Code follows project conventions (AGENTS.md rules)
- [ ] Linter passes with zero warnings
- [ ] Code formatted per project standards
- [ ] No duplicated logic — reuses existing code
- [ ] Test coverage maintained or improved

#### Human Review
- [ ] Developer can explain every line of the change
- [ ] Changes match the original task scope
- [ ] Architecture decisions are appropriate
- [ ] No over-engineering or unnecessary abstractions
- [ ] PR description accurately reflects the changes

#### Agent Hygiene
- [ ] Context file (AGENTS.md) updated if new patterns emerged
- [ ] Agent failures documented for team learning
- [ ] Effective prompts shared with the team
- [ ] No agent-specific artifacts left in code (TODO: AI, FIXME: agent, etc.)

---

## 13. Why This Configuration Works

**Verified Code**:
- AI-generated code passes through the same (or stricter) quality gates as human code, ensuring no regression in code quality as AI adoption increases.

**Context Engineering**:
- By engineering the context — not just the prompt — you give the agent structured, unambiguous instructions that produce consistent results across sessions and team members.

**Proportional Guardrails**:
- More autonomy requires more verification. Pre-commit hooks, CI gates, AI review, and human review form a defense-in-depth strategy that catches issues at multiple stages.

**Human-AI Symbiosis**:
- Humans focus on what they do best (architecture, design, security decisions) while agents focus on what they do best (boilerplate, iteration, pattern application). Neither replaces the other.

**Continuous Improvement**:
- Tracking metrics, updating rules, and sharing patterns creates a flywheel where every interaction makes the next one better.

---

## 14. Quick Reference

### Context File Cheat Sheet

| Tool | File | Location | Format |
|------|------|----------|--------|
| Universal | `AGENTS.md` | Project root + subdirs | Markdown |
| Claude Code | `CLAUDE.md` | Project root / `.claude/` | Markdown |
| Copilot | `copilot-instructions.md` | `.github/` | Markdown |
| Cursor | Rules files | `.cursor/rules/` | Markdown + YAML frontmatter |
| Windsurf | `.windsurfrules` | Project root | Markdown |
| Cody | `.sourcegraph/cody.json` | `.sourcegraph/` | JSON |

### Prompt Templates

```markdown
# Feature Implementation
Implement [feature] with these requirements:
1. [requirement 1]
2. [requirement 2]

Modify ONLY: [file list]
Do NOT modify: [exclusion list]
Write tests first in [test file].

# Bug Fix
Bug: [description]
Reproduction: [steps]
Expected: [behavior]
Actual: [behavior]

Write a failing test first, then fix.
Modify ONLY: [file list]

# Refactor
Refactor [target] to [goal].
Keep all tests passing.
Do not change external behavior.
Modify ONLY: [file list]
```

### Verification Commands

```bash
# The Universal Check Sequence
make check              # Or run individually:

# 1. Build
npm run build           # Verify compilation

# 2. Test
npm test                # Run all tests

# 3. Lint
npm run lint            # Check code quality

# 4. Security
npm audit               # Dependency vulnerabilities
gitleaks detect .       # Secret detection

# 5. Format
prettier --check .      # Formatting consistency
```

### Red Flags in AI Output

```markdown
## Immediate Rejection Triggers

- Package you can't find on npm/PyPI/crates.io → hallucinated
- API method not in official docs → hallucinated
- "I'll add error handling later" → incomplete
- Changes to files not in scope → scope creep
- New dependency without justification → unnecessary
- `// TODO` or `// FIXME` without ticket → unfinished
- `any` type in TypeScript → type safety bypass
- `eval()`, `exec()`, `Function()` → security risk
- `cors: '*'` or `debug: true` → insecure default
```

---

**End of Coding AI Agent Guidelines**
