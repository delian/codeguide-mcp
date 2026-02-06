# Pre-commit Framework Guidelines
Mandatory standards for using the pre-commit framework to enforce code quality, security, and consistency across all programming languages.

---

**Agent Profile**: The Quality Gate Enforcer
**Role**: Senior DevOps Engineer & Code Quality Specialist
**Objective**: Ensure all code passes comprehensive quality and security checks before entering version control.
**Tools**: pre-commit framework, language-specific linters, security scanners, formatters, type checkers, test runners.

---

## 1. Core Philosophies: PRECOMMIT-FIRST

The agent must adhere to the **PRECOMMIT-FIRST** principles:

**Test-Driven Development (TDD)**: Pre-commit hooks MUST include test execution to verify code correctness.
**Regression Shield**: Security and quality checks prevent regressions from entering the codebase.

- **P**revent Bad Commits: Block commits that fail quality, security, or formatting checks
- **R**un Checks Automatically: Automate all verification before code enters version control
- **E**nforce Consistency: Apply the same standards across all developers and CI/CD
- **C**atch Issues Early: Find problems at commit time, not in CI or production
- **O**ptimize for Speed: Fast hooks encourage developers to keep them enabled
- **M**andatory Verification: ALWAYS run `pre-commit run -a` before committing
- **M**CP Integration: Consult MCP servers and web resources for optimal configurations
- **I**nfrastructure Agnostic: Same hooks work locally and in CI/CD pipelines
- **T**horough Validation: Check security, dependencies, build, types, lint, tests, docs

**Agent Mandatory Behavior**: ALWAYS run `pre-commit run -a` before creating any commit, even if pre-commit is not installed as a git hook.

---

## 2. Agent Pre-Commit Requirements (MANDATORY)

### A. Agent Verification Protocol

**CRITICAL: Agents MUST run `pre-commit run -a` before EVERY commit operation.**

#### Pre-Commit Execution (MANDATORY)

**Before creating ANY commit, the agent MUST:**

```bash
# Step 1: Check if pre-commit is available
if command -v pre-commit &> /dev/null; then
    echo "pre-commit found, running all hooks..."
    pre-commit run -a
    if [ $? -ne 0 ]; then
        echo "ERROR: pre-commit checks failed"
        echo "Fix issues before committing"
        exit 1
    fi
else
    echo "WARNING: pre-commit not installed"
    echo "Installing pre-commit..."
    pip install pre-commit || pipx install pre-commit
fi

# Step 2: Verify .pre-commit-config.yaml exists
if [ ! -f ".pre-commit-config.yaml" ]; then
    echo "WARNING: .pre-commit-config.yaml not found"
    echo "Creating recommended configuration..."
    # Agent should create or suggest configuration
fi

# Step 3: Run all hooks on all files
pre-commit run -a

# Step 4: Only proceed with commit if all hooks pass
if [ $? -eq 0 ]; then
    echo "All pre-commit checks passed"
    git commit -m "..."
else
    echo "BLOCKED: Fix pre-commit issues first"
fi
```

### B. Agent Configuration Discovery Protocol

**CRITICAL: When setting up pre-commit, agents MUST consult available resources.**

#### Configuration Discovery Steps

```markdown
## Agent Configuration Discovery Protocol

1. **Check Project Type**
   - Detect language(s): package.json, pyproject.toml, Cargo.toml, go.mod, etc.
   - Detect frameworks: React, Django, FastAPI, Spring, etc.
   - Detect infrastructure: Docker, Kubernetes, Terraform, etc.

2. **Consult MCP Servers**
   - Query code-guide MCP for language-specific recommendations
   - Check for project-specific hook requirements
   - Gather security scanning tool recommendations

3. **Search Web Resources**
   - Search: "best pre-commit hooks for [language] 2026"
   - Search: "pre-commit config [framework] security"
   - Search: "[tool] pre-commit hook configuration"
   - Check GitHub for popular pre-commit-config.yaml examples
   - Reference: https://pre-commit.com/hooks.html

4. **Evaluate and Select Hooks**
   - Prioritize: security > build > types > lint > format > tests
   - Consider: speed (fast hooks first, slow hooks optional)
   - Validate: hooks work with project's language/framework versions

5. **Generate Configuration**
   - Create .pre-commit-config.yaml with selected hooks
   - Include comments explaining each hook's purpose
   - Test configuration with `pre-commit run -a`
```

### C. Prohibited Practices

**NEVER commit code that:**
- [ ] Has not been validated with `pre-commit run -a`
- [ ] Bypasses pre-commit with `--no-verify` (unless explicitly requested)
- [ ] Fails security scanning
- [ ] Has dependency vulnerabilities
- [ ] Fails to build/compile
- [ ] Has type errors (in typed languages)
- [ ] Fails linting
- [ ] Has formatting issues
- [ ] Has failing unit tests
- [ ] Lacks required documentation

---

## 3. Pre-commit Installation & Setup (MANDATORY)

### A. Installation

```bash
# Using pip (recommended)
pip install pre-commit

# Using pipx (isolated environment)
pipx install pre-commit

# Using Homebrew (macOS)
brew install pre-commit

# Using conda
conda install -c conda-forge pre-commit

# Verify installation
pre-commit --version
```

### B. Project Setup

```bash
# Initialize pre-commit in project
cd /path/to/project

# Create configuration file (or use agent-generated config)
touch .pre-commit-config.yaml

# Install hooks into git
pre-commit install

# Install additional hook types
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push

# Run on all files (first-time setup)
pre-commit run -a

# Update hooks to latest versions
pre-commit autoupdate
```

### C. Git Hook Installation

```bash
# Install all hook types for comprehensive coverage
pre-commit install                        # pre-commit hook
pre-commit install --hook-type commit-msg # commit message validation
pre-commit install --hook-type pre-push   # pre-push validation

# Verify hooks are installed
ls -la .git/hooks/
# Should show: pre-commit, commit-msg, pre-push -> pre-commit scripts
```

---

## 4. Comprehensive Hook Configuration (MANDATORY)

### A. Core Configuration Structure

```yaml
# .pre-commit-config.yaml
# Comprehensive pre-commit configuration for code quality and security
# Generated following pre-commit.md guidelines

# Global settings
default_language_version:
  python: python3
  node: "18.17.0"

# Run hooks in parallel for speed
default_stages: [commit]

# Fail fast on first failure (optional, for speed)
fail_fast: false

# CI-specific settings
ci:
  autofix_prs: true
  autoupdate_schedule: weekly
  autoupdate_commit_msg: "chore(deps): update pre-commit hooks"

repos:
  # ═══════════════════════════════════════════════════════════════════
  # SECTION 1: SECURITY SCANNING (HIGHEST PRIORITY)
  # These hooks detect secrets, vulnerabilities, and security issues
  # ═══════════════════════════════════════════════════════════════════

  # Secret Detection - MANDATORY
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        name: "Security: Detect secrets"
        args: ['--baseline', '.secrets.baseline']
        exclude: package-lock\.json|yarn\.lock|pnpm-lock\.yaml

  # Alternative: Gitleaks (more comprehensive)
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks
        name: "Security: Detect secrets (gitleaks)"

  # Security linting for Python
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        name: "Security: Python security linter"
        args: ['-r', '-ll', '-ii']
        types: [python]

  # Security linting for JavaScript/TypeScript
  - repo: local
    hooks:
      - id: npm-audit
        name: "Security: npm audit"
        entry: npm audit --audit-level=high
        language: system
        pass_filenames: false
        files: package\.json|package-lock\.json
        stages: [commit, push]

  # ═══════════════════════════════════════════════════════════════════
  # SECTION 2: DEPENDENCY VALIDATION
  # Ensure dependencies are secure and properly managed
  # ═══════════════════════════════════════════════════════════════════

  - repo: local
    hooks:
      # Python dependencies
      - id: pip-audit
        name: "Dependencies: Python vulnerability check"
        entry: pip-audit
        language: system
        pass_filenames: false
        files: requirements.*\.txt|pyproject\.toml|setup\.py
        stages: [commit, push]

      # Node.js dependencies
      - id: npm-outdated-check
        name: "Dependencies: Check for outdated packages"
        entry: bash -c 'npm outdated --json | jq -e "length == 0"'
        language: system
        pass_filenames: false
        files: package\.json
        stages: [push]  # Only on push (slower)

  # ═══════════════════════════════════════════════════════════════════
  # SECTION 3: BUILD & COMPILATION
  # Verify code compiles/builds successfully
  # ═══════════════════════════════════════════════════════════════════

  - repo: local
    hooks:
      # TypeScript compilation
      - id: typescript-compile
        name: "Build: TypeScript compilation"
        entry: npx tsc --noEmit
        language: system
        types: [ts, tsx]
        pass_filenames: false

      # Rust compilation
      - id: cargo-check
        name: "Build: Rust compilation"
        entry: cargo check
        language: system
        types: [rust]
        pass_filenames: false

      # Go compilation
      - id: go-build
        name: "Build: Go compilation"
        entry: go build ./..
        language: system
        types: [go]
        pass_filenames: false

      # Python syntax check
      - id: python-compile
        name: "Build: Python syntax check"
        entry: python -m py_compile
        language: system
        types: [python]

  # ═══════════════════════════════════════════════════════════════════
  # SECTION 4: TYPE CHECKING
  # Static type analysis for typed languages
  # ═══════════════════════════════════════════════════════════════════

  # Python type checking with mypy
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        name: "Types: Python type checking"
        additional_dependencies: [types-all]
        args: [--ignore-missing-imports, --strict]

  # Python type checking with pyright (faster)
  - repo: local
    hooks:
      - id: pyright
        name: "Types: Python type checking (pyright)"
        entry: pyright
        language: node
        types: [python]
        pass_filenames: false
        additional_dependencies: ['pyright@1.1.325']

  # TypeScript strict mode (already covered by tsc --noEmit above)

  # ═══════════════════════════════════════════════════════════════════
  # SECTION 5: LINTING
  # Code quality and style enforcement
  # ═══════════════════════════════════════════════════════════════════

  # General linters
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: check-yaml
        name: "Lint: YAML syntax"
      - id: check-json
        name: "Lint: JSON syntax"
      - id: check-toml
        name: "Lint: TOML syntax"
      - id: check-xml
        name: "Lint: XML syntax"
      - id: check-merge-conflict
        name: "Lint: Check merge conflicts"
      - id: check-case-conflict
        name: "Lint: Check case conflicts"
      - id: check-symlinks
        name: "Lint: Check symlinks"
      - id: check-added-large-files
        name: "Lint: Check large files"
        args: ['--maxkb=1000']
      - id: end-of-file-fixer
        name: "Lint: Fix end of file"
      - id: trailing-whitespace
        name: "Lint: Trailing whitespace"
      - id: mixed-line-ending
        name: "Lint: Mixed line endings"
        args: ['--fix=lf']
      - id: no-commit-to-branch
        name: "Lint: Prevent direct commits to main/develop"
        args: ['--branch', 'main', '--branch', 'develop']

  # Python linting with Ruff (fast, comprehensive)
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        name: "Lint: Python (ruff)"
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
        name: "Format: Python (ruff)"

  # JavaScript/TypeScript linting
  - repo: https://github.com/pre-commit/mirrors-eslint
    rev: v8.52.0
    hooks:
      - id: eslint
        name: "Lint: JavaScript/TypeScript"
        types: [javascript, tsx, ts]
        additional_dependencies:
          - eslint@8.52.0
          - '@typescript-eslint/parser@6.9.0'
          - '@typescript-eslint/eslint-plugin@6.9.0'

  # Go linting
  - repo: https://github.com/golangci/golangci-lint
    rev: v1.55.2
    hooks:
      - id: golangci-lint
        name: "Lint: Go"

  # Rust linting
  - repo: local
    hooks:
      - id: clippy
        name: "Lint: Rust (clippy)"
        entry: cargo clippy -- -D warnings
        language: system
        types: [rust]
        pass_filenames: false

  # Shell script linting
  - repo: https://github.com/shellcheck-py/shellcheck-py
    rev: v0.9.0.6
    hooks:
      - id: shellcheck
        name: "Lint: Shell scripts"

  # Dockerfile linting
  - repo: https://github.com/hadolint/hadolint
    rev: v2.12.0
    hooks:
      - id: hadolint
        name: "Lint: Dockerfile"

  # Markdown linting
  - repo: https://github.com/igorshubovych/markdownlint-cli
    rev: v0.37.0
    hooks:
      - id: markdownlint
        name: "Lint: Markdown"
        args: ['--fix']

  # ═══════════════════════════════════════════════════════════════════
  # SECTION 6: FORMATTING
  # Consistent code formatting
  # ═══════════════════════════════════════════════════════════════════

  # Prettier for web files
  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.0.3
    hooks:
      - id: prettier
        name: "Format: Prettier"
        types_or: [javascript, jsx, ts, tsx, css, scss, json, yaml, markdown, html]

  # Python formatting with Black
  - repo: https://github.com/psf/black
    rev: 23.10.1
    hooks:
      - id: black
        name: "Format: Python (black)"

  # Python import sorting
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        name: "Format: Python imports"
        args: ['--profile', 'black']

  # Go formatting
  - repo: local
    hooks:
      - id: go-fmt
        name: "Format: Go"
        entry: gofmt -w
        language: system
        types: [go]

  # Rust formatting
  - repo: local
    hooks:
      - id: rustfmt
        name: "Format: Rust"
        entry: cargo fmt --
        language: system
        types: [rust]

  # ═══════════════════════════════════════════════════════════════════
  # SECTION 7: UNIT TESTS (BASIC)
  # Run fast unit tests before commit
  # ═══════════════════════════════════════════════════════════════════

  - repo: local
    hooks:
      # Python tests
      - id: pytest
        name: "Test: Python unit tests"
        entry: pytest -x -q --tb=short
        language: system
        types: [python]
        pass_filenames: false
        stages: [commit]

      # JavaScript/TypeScript tests
      - id: jest
        name: "Test: JavaScript unit tests"
        entry: npm test -- --passWithNoTests --bail
        language: system
        pass_filenames: false
        files: \.(js|jsx|ts|tsx)$
        stages: [commit]

      # Go tests
      - id: go-test
        name: "Test: Go unit tests"
        entry: go test -short ./..
        language: system
        types: [go]
        pass_filenames: false
        stages: [commit]

      # Rust tests
      - id: cargo-test
        name: "Test: Rust unit tests"
        entry: cargo test --no-fail-fast
        language: system
        types: [rust]
        pass_filenames: false
        stages: [commit]

  # ═══════════════════════════════════════════════════════════════════
  # SECTION 8: API DOCUMENTATION
  # Ensure API documentation is included and up-to-date
  # ═══════════════════════════════════════════════════════════════════

  - repo: local
    hooks:
      # OpenAPI/Swagger validation
      - id: openapi-validate
        name: "Docs: Validate OpenAPI spec"
        entry: npx @openapitools/openapi-generator-cli validate -i
        language: system
        files: (openapi|swagger)\.(yaml|yml|json)$

      # Python docstring coverage
      - id: docstring-coverage
        name: "Docs: Python docstring coverage"
        entry: interrogate -vv --fail-under=80
        language: system
        types: [python]
        pass_filenames: false

      # TypeDoc validation
      - id: typedoc-check
        name: "Docs: TypeScript documentation"
        entry: npx typedoc --validation
        language: system
        files: \.tsx?$
        pass_filenames: false
        stages: [push]

  # ═══════════════════════════════════════════════════════════════════
  # SECTION 9: COMMIT MESSAGE VALIDATION
  # Enforce conventional commits format
  # ═══════════════════════════════════════════════════════════════════

  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v3.0.0
    hooks:
      - id: conventional-pre-commit
        name: "Commit: Conventional commit message"
        stages: [commit-msg]
        args:
          - feat
          - fix
          - docs
          - style
          - refactor
          - perf
          - test
          - build
          - ci
          - chore
          - revert

  # Alternative: commitlint
  - repo: https://github.com/alessandrojcm/commitlint-pre-commit-hook
    rev: v9.10.0
    hooks:
      - id: commitlint
        name: "Commit: Commitlint validation"
        stages: [commit-msg]
        additional_dependencies: ['@commitlint/config-conventional']

  # ═══════════════════════════════════════════════════════════════════
  # SECTION 10: ADDITIONAL QUALITY CHECKS
  # Miscellaneous quality and safety checks
  # ═══════════════════════════════════════════════════════════════════

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: check-executables-have-shebangs
        name: "Quality: Check executables have shebangs"
      - id: check-shebang-scripts-are-executable
        name: "Quality: Check shebang scripts are executable"
      - id: debug-statements
        name: "Quality: Check for debug statements"
      - id: detect-private-key
        name: "Security: Detect private keys"

  # Check for TODO/FIXME comments
  - repo: local
    hooks:
      - id: check-todo-fixme
        name: "Quality: Warn on TODO/FIXME"
        entry: bash -c 'git diff --cached --name-only | xargs grep -l "TODO\|FIXME" && echo "WARNING: Found TODO/FIXME comments" && exit 0 || exit 0'
        language: system
        pass_filenames: false
        verbose: true
```

### B. Language-Specific Configurations

#### Python Projects

```yaml
# .pre-commit-config.yaml for Python projects
repos:
  # Security
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-r', 'src/', '-ll']

  # Dependencies
  - repo: local
    hooks:
      - id: pip-audit
        name: pip-audit
        entry: pip-audit
        language: system
        pass_filenames: false

  # Build
  - repo: local
    hooks:
      - id: python-compile
        name: Compile Python
        entry: python -m py_compile
        language: system
        types: [python]

  # Types
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--strict]

  # Lint & Format
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ['--profile', 'black']

  # Tests
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest -x -q
        language: system
        pass_filenames: false
        types: [python]

  # Docs
  - repo: local
    hooks:
      - id: interrogate
        name: docstring-coverage
        entry: interrogate -vv --fail-under=80
        language: system
        pass_filenames: false

  # Commit
  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v3.0.0
    hooks:
      - id: conventional-pre-commit
        stages: [commit-msg]
```

#### JavaScript/TypeScript Projects

```yaml
# .pre-commit-config.yaml for JavaScript/TypeScript projects
repos:
  # Security
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks

  - repo: local
    hooks:
      - id: npm-audit
        name: npm audit
        entry: npm audit --audit-level=high
        language: system
        pass_filenames: false

  # Build
  - repo: local
    hooks:
      - id: tsc
        name: TypeScript compile
        entry: npx tsc --noEmit
        language: system
        pass_filenames: false
        types: [ts, tsx]

  # Lint
  - repo: local
    hooks:
      - id: eslint
        name: ESLint
        entry: npx eslint --fix
        language: system
        types: [javascript, jsx, ts, tsx]

  # Format
  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.0.3
    hooks:
      - id: prettier
        types_or: [javascript, jsx, ts, tsx, css, json, yaml, markdown]

  # Tests
  - repo: local
    hooks:
      - id: jest
        name: Jest tests
        entry: npm test -- --passWithNoTests --bail
        language: system
        pass_filenames: false

  # Commit
  - repo: https://github.com/alessandrojcm/commitlint-pre-commit-hook
    rev: v9.10.0
    hooks:
      - id: commitlint
        stages: [commit-msg]
        additional_dependencies: ['@commitlint/config-conventional']
```

#### Go Projects

```yaml
# .pre-commit-config.yaml for Go projects
repos:
  # Security
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks

  - repo: local
    hooks:
      - id: go-sec
        name: Go security scan
        entry: gosec ./..
        language: system
        pass_filenames: false
        types: [go]

  # Build
  - repo: local
    hooks:
      - id: go-build
        name: Go build
        entry: go build ./..
        language: system
        pass_filenames: false
        types: [go]

  # Lint
  - repo: https://github.com/golangci/golangci-lint
    rev: v1.55.2
    hooks:
      - id: golangci-lint

  # Format
  - repo: local
    hooks:
      - id: go-fmt
        name: Go format
        entry: gofmt -w
        language: system
        types: [go]

      - id: go-imports
        name: Go imports
        entry: goimports -w
        language: system
        types: [go]

  # Tests
  - repo: local
    hooks:
      - id: go-test
        name: Go tests
        entry: go test -short ./..
        language: system
        pass_filenames: false
        types: [go]

  # Commit
  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v3.0.0
    hooks:
      - id: conventional-pre-commit
        stages: [commit-msg]
```

#### Rust Projects

```yaml
# .pre-commit-config.yaml for Rust projects
repos:
  # Security
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks

  - repo: local
    hooks:
      - id: cargo-audit
        name: Cargo audit
        entry: cargo audit
        language: system
        pass_filenames: false
        types: [rust]

  # Build
  - repo: local
    hooks:
      - id: cargo-check
        name: Cargo check
        entry: cargo check
        language: system
        pass_filenames: false
        types: [rust]

  # Lint
  - repo: local
    hooks:
      - id: clippy
        name: Clippy
        entry: cargo clippy -- -D warnings
        language: system
        pass_filenames: false
        types: [rust]

  # Format
  - repo: local
    hooks:
      - id: rustfmt
        name: Rust format
        entry: cargo fmt --
        language: system
        types: [rust]

  # Tests
  - repo: local
    hooks:
      - id: cargo-test
        name: Cargo test
        entry: cargo test
        language: system
        pass_filenames: false
        types: [rust]

  # Commit
  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v3.0.0
    hooks:
      - id: conventional-pre-commit
        stages: [commit-msg]
```

---

## 5. Hook Categories & Priority (MANDATORY)

### A. Hook Execution Order

**Hooks should be ordered by priority (fastest and most critical first):**

```
Priority 1: SECURITY (Block commits with secrets/vulnerabilities)
    ├── detect-secrets
    ├── gitleaks
    ├── bandit (Python)
    ├── npm-audit (Node.js)
    └── cargo-audit (Rust)

Priority 2: BUILD (Ensure code compiles)
    ├── tsc --noEmit (TypeScript)
    ├── cargo check (Rust)
    ├── go build (Go)
    └── python -m py_compile (Python)

Priority 3: TYPES (Static type analysis)
    ├── mypy (Python)
    ├── pyright (Python)
    └── TypeScript strict mode

Priority 4: LINT (Code quality)
    ├── ruff/pylint (Python)
    ├── eslint (JavaScript/TypeScript)
    ├── golangci-lint (Go)
    ├── clippy (Rust)
    └── shellcheck (Shell)

Priority 5: FORMAT (Code style)
    ├── black/ruff-format (Python)
    ├── prettier (Web)
    ├── gofmt (Go)
    └── rustfmt (Rust)

Priority 6: TESTS (Verification)
    ├── pytest (Python)
    ├── jest/vitest (JavaScript/TypeScript)
    ├── go test (Go)
    └── cargo test (Rust)

Priority 7: DOCS (Documentation)
    ├── interrogate (Python docstrings)
    ├── typedoc (TypeScript)
    └── openapi-validate (API specs)

Priority 8: COMMIT (Message format)
    └── conventional-pre-commit
```

### B. Hook Stages

```yaml
# Configure different hooks for different stages

# pre-commit: Fast checks, run on every commit
stages: [commit]
# - Security scanning
# - Formatting
# - Fast linting
# - Syntax checks

# commit-msg: Message validation
stages: [commit-msg]
# - Conventional commits
# - Issue reference check

# pre-push: Slower, comprehensive checks
stages: [push]
# - Full test suite
# - Dependency audits
# - Documentation generation
# - Integration tests
```

---

## 6. Agent Configuration Discovery (MANDATORY)

### A. Web Search Queries

**When setting up pre-commit, agents MUST search for current best practices:**

```markdown
## Recommended Search Queries

### By Language
- "best pre-commit hooks python 2026 security"
- "pre-commit config typescript eslint prettier 2026"
- "golang pre-commit hooks golangci-lint 2026"
- "rust pre-commit cargo clippy rustfmt 2026"
- "java pre-commit checkstyle spotbugs 2026"

### By Framework
- "pre-commit hooks react next.js 2026"
- "pre-commit config django fastapi 2026"
- "spring boot pre-commit hooks 2026"
- "flutter dart pre-commit hooks 2026"

### By Use Case
- "pre-commit security scanning secrets detection"
- "pre-commit dependency vulnerability scanning"
- "pre-commit commit message validation conventional"
- "pre-commit api documentation validation openapi"

### By Infrastructure
- "pre-commit terraform tflint tfsec"
- "pre-commit dockerfile hadolint trivy"
- "pre-commit kubernetes yaml validation"
- "pre-commit helm chart linting"
```

### B. MCP Server Consultation

```markdown
## Agent MCP Consultation Protocol

1. **Query code-guide MCP**
   - Request language-specific recommendations
   - Get security hook configurations
   - Obtain formatting standards

2. **Query available tool MCPs**
   - Get current hook versions
   - Obtain configuration snippets
   - Verify compatibility

3. **Cross-reference with project requirements**
   - Match hooks to detected languages
   - Apply framework-specific hooks
   - Include infrastructure hooks (Docker, K8s, etc.)
```

### C. GitHub Repository Examples

**Reference these popular pre-commit configurations:**

```markdown
## Example Configurations to Reference

### Python
- https://github.com/python/cpython (Python itself)
- https://github.com/pandas-dev/pandas
- https://github.com/tiangolo/fastapi

### JavaScript/TypeScript
- https://github.com/facebook/react
- https://github.com/vercel/next.js
- https://github.com/microsoft/vscode

### Go
- https://github.com/kubernetes/kubernetes
- https://github.com/hashicorp/terraform

### Rust
- https://github.com/rust-lang/rust
- https://github.com/denoland/deno

### Multi-language
- https://github.com/pre-commit/pre-commit-hooks (official examples)
```

---

## 7. CI/CD Integration (MANDATORY)

### A. GitHub Actions

```yaml
# .github/workflows/pre-commit.yml
name: Pre-commit

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install pre-commit
        run: pip install pre-commit

      - name: Run pre-commit on all files
        run: pre-commit run -a --show-diff-on-failure

      - name: Run pre-commit for commit message (PR only)
        if: github.event_name == 'pull_request'
        run: |
          echo "${{ github.event.pull_request.title }}" | pre-commit run --hook-stage commit-msg --commit-msg-filename /dev/stdin
```

### B. GitLab CI

```yaml
# .gitlab-ci.yml
pre-commit:
  image: python:3.11
  stage: test
  script:
    - pip install pre-commit
    - pre-commit run -a --show-diff-on-failure
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == "main"
    - if: $CI_COMMIT_BRANCH == "develop"
```

### C. Jenkins

```groovy
// Jenkinsfile
pipeline {
    agent any

    stages {
        stage('Pre-commit') {
            steps {
                sh '''
                    pip install pre-commit
                    pre-commit run -a --show-diff-on-failure
                '''
            }
        }
    }
}
```

---

## 8. Troubleshooting & Performance

### A. Common Issues

```bash
# Issue: Hook takes too long
# Solution: Run slow hooks only on push
stages: [push]

# Issue: Hook fails to install
# Solution: Clear cache and reinstall
pre-commit clean
pre-commit install-hooks

# Issue: Hook version conflict
# Solution: Update to latest versions
pre-commit autoupdate

# Issue: Hook needs specific environment
# Solution: Use system language with proper PATH
language: system
entry: /path/to/tool

# Issue: False positives in secret detection
# Solution: Create baseline file
detect-secrets scan > .secrets.baseline
```

### B. Performance Optimization

```yaml
# Speed optimizations

# 1. Run hooks in parallel (default)
# No configuration needed - pre-commit runs in parallel by default

# 2. Fail fast on first failure
fail_fast: true

# 3. Only run on changed files (not -a)
# Default behavior for pre-commit (not pre-commit run -a)

# 4. Separate slow hooks to pre-push stage
- repo: local
  hooks:
    - id: full-test-suite
      name: Full test suite
      entry: npm run test:all
      stages: [push]  # Only on push, not every commit

# 5. Use faster tool alternatives
# - ruff instead of pylint/flake8 (10-100x faster)
# - pyright instead of mypy (faster)
# - biome instead of eslint+prettier (faster)

# 6. Exclude generated/vendored files
exclude: |
  (?x)^(
    node_modules/|
    vendor/|
    dist/|
    build/|
    \.min\.js$|
    package-lock\.json|
    yarn\.lock
  )$
```

### C. Debugging Hooks

```bash
# Run single hook
pre-commit run <hook-id>

# Run with verbose output
pre-commit run -a --verbose

# Run on specific files
pre-commit run --files path/to/file.py

# Show hook diff
pre-commit run -a --show-diff-on-failure

# Skip specific hook (use sparingly!)
SKIP=hook-id pre-commit run -a

# Clear cache for fresh install
pre-commit clean
pre-commit gc

# Update all hooks
pre-commit autoupdate
```

---

## 9. Agent Workflow (MANDATORY)

### A. Before Every Commit

```bash
#!/bin/bash
# Agent pre-commit workflow

echo "=== Agent Pre-Commit Workflow ==="

# Step 1: Ensure pre-commit is available
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    pip install pre-commit
fi

# Step 2: Check for configuration
if [ ! -f ".pre-commit-config.yaml" ]; then
    echo "WARNING: No .pre-commit-config.yaml found"
    echo "Agent should create configuration based on project type"
    # Agent creates configuration using discovery protocol
fi

# Step 3: ALWAYS run pre-commit on all files
echo "Running pre-commit on all files..."
pre-commit run -a

# Step 4: Check result
if [ $? -ne 0 ]; then
    echo "ERROR: Pre-commit checks failed"
    echo "Agent must fix issues before committing"
    exit 1
fi

# Step 5: Stage changes (including auto-fixes)
git add -A

# Step 6: Run pre-commit again (verify fixes)
pre-commit run -a

# Step 7: Only commit if all checks pass
if [ $? -eq 0 ]; then
    echo "All checks passed - proceeding with commit"
    git commit -m "..."
else
    echo "BLOCKED: Issues remain after auto-fix"
    exit 1
fi
```

### B. Creating New Project Configuration

```markdown
## Agent Configuration Creation Protocol

When no .pre-commit-config.yaml exists:

1. **Detect Project Type**
   ```bash
   # Check for language indicators
   ls package.json pyproject.toml Cargo.toml go.mod *.csproj
   ```

2. **Search for Best Practices**
   - Query web: "best pre-commit config [detected-language] 2026"
   - Query MCP servers for recommendations
   - Check project-specific requirements

3. **Generate Configuration**
   - Start with security hooks (ALWAYS)
   - Add build verification hooks
   - Add type checking (if applicable)
   - Add linting hooks
   - Add formatting hooks
   - Add test hooks
   - Add documentation hooks
   - Add commit message validation

4. **Validate Configuration**
   ```bash
   pre-commit run -a
   ```

5. **Document Configuration**
   - Add comments explaining each hook
   - Note any project-specific customizations
   - Include in commit message
```

---

## 10. Deployment Checklist

### Pre-Commit Configuration
- [ ] .pre-commit-config.yaml exists
- [ ] Configuration covers all project languages
- [ ] Security scanning hooks included
- [ ] Build verification hooks included
- [ ] Type checking hooks included (if applicable)
- [ ] Linting hooks included
- [ ] Formatting hooks included
- [ ] Test hooks included
- [ ] Documentation hooks included
- [ ] Commit message validation included

### Hook Installation
- [ ] `pre-commit install` executed
- [ ] `pre-commit install --hook-type commit-msg` executed
- [ ] `pre-commit install --hook-type pre-push` executed
- [ ] Hooks verified with `pre-commit run -a`

### CI/CD Integration
- [ ] Pre-commit runs in CI pipeline
- [ ] Failure blocks merge/deploy
- [ ] Same configuration as local development

### Agent Compliance
- [ ] Agent always runs `pre-commit run -a` before commit
- [ ] Agent does not bypass hooks without explicit user request
- [ ] Agent consults MCP/web for configuration recommendations
- [ ] Agent creates configuration if missing

---

## 11. Quick Reference

### Essential Commands

```bash
# Installation
pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push

# Running hooks
pre-commit run -a                    # Run all hooks on all files
pre-commit run                       # Run on staged files only
pre-commit run <hook-id>             # Run specific hook
pre-commit run --files file.py       # Run on specific files

# Maintenance
pre-commit autoupdate               # Update hooks to latest
pre-commit clean                    # Clear cache
pre-commit gc                       # Garbage collect

# Debugging
pre-commit run -a --verbose         # Verbose output
pre-commit run -a --show-diff-on-failure  # Show diffs
SKIP=hook-id pre-commit run -a      # Skip specific hook
```

### Minimal Configuration Template

```yaml
# .pre-commit-config.yaml - Minimal but comprehensive

repos:
  # Security (ALWAYS INCLUDE)
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks

  # General checks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: check-yaml
      - id: check-json
      - id: check-merge-conflict
      - id: check-added-large-files
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: no-commit-to-branch
        args: ['--branch', 'main']

  # Commit message
  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v3.0.0
    hooks:
      - id: conventional-pre-commit
        stages: [commit-msg]

  # ADD LANGUAGE-SPECIFIC HOOKS BELOW
  # (Agent should add based on project type)
```

### Search Queries for Agent

```
Pre-commit configuration:
- "pre-commit hooks [language] [year] best practices"
- "pre-commit config [framework] security linting"
- "[tool] pre-commit hook setup"

Security hooks:
- "pre-commit secret detection gitleaks detect-secrets"
- "pre-commit security scanning [language]"

Linting/formatting:
- "pre-commit [linter] [language] configuration"
- "pre-commit [formatter] setup"
```

---

## 12. Why This Configuration Works

**Prevent Issues at Source**:
- Catches problems before they enter version control
- Faster feedback than waiting for CI/CD

**Consistent Quality**:
- Same checks run for all developers
- Eliminates "works on my machine" issues

**Security First**:
- Secrets never make it to repository
- Vulnerabilities caught before commit

**Automated Enforcement**:
- No manual review for style/formatting
- Focus code review on logic and design

**CI/CD Alignment**:
- Same checks locally and in pipeline
- No surprises during deployment

**Developer Experience**:
- Fast hooks don't slow down development
- Auto-fixes reduce manual work

---

## References

- [Pre-commit Framework](https://pre-commit.com/)
- [Pre-commit Hooks Repository](https://github.com/pre-commit/pre-commit-hooks)
- [Supported Hooks List](https://pre-commit.com/hooks.html)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Gitleaks](https://github.com/gitleaks/gitleaks)
- [Ruff](https://github.com/astral-sh/ruff)
- [ESLint](https://eslint.org/)
- [golangci-lint](https://golangci-lint.run/)

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Development Team


**End of Pre-commit Framework Guidelines**
