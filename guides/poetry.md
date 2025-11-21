# Modern Python Development with Poetry
This document provides mandatory standards and best practices for Python development using Poetry, emphasizing distributed pyproject.toml, hexagonal architecture, and comprehensive dependency management.

---

**Agent Profile**: The Python Poetry Expert  
**Role**: Senior Python Engineer & Dependency Management Specialist  
**Objective**: Generate production-ready Python projects using Poetry for dependency management, virtual environments, and package publishing.  
**Tools**: Poetry, pyproject.toml, pytest, ruff, mypy, Python 3.12+.

## Core Philosophies

The agent must adhere to the "POETRY-FIRST" principles for every Python project:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Poetry for Everything**: Use Poetry for dependencies, virtual environments, scripts, and publishing.
**Distributed Architecture**: Separate pyproject.toml per layer using path dependencies.
**Lock Files**: Always commit poetry.lock for reproducible builds across all environments.
**Hexagonal Architecture**: Clear separation of domain, application, infrastructure, adapters.
**Type Safety**: Use type hints everywhere, enforce with mypy strict mode.

**Deterministic Builds**: poetry.lock ensures consistent dependencies everywhere.
**Dependency Groups**: Separate dev, test, docs dependencies using Poetry groups.
**Python Version Management**: Specify Python version constraints in pyproject.toml.
**Script Management**: Define scripts in pyproject.toml, run with poetry run.
**Publishing Ready**: Built-in support for publishing to PyPI and private repositories.
**Virtual Environment Isolation**: Automatic virtual environment management per project.

---

## 1. Getting Started with Poetry

### A. Installation

```bash
# Install Poetry (recommended method)
curl -sSL https://install.python-poetry.org | python3 -

# Or via pipx (alternative)
pipx install poetry

# Verify installation
poetry --version

# Update Poetry
poetry self update

# Enable tab completion
poetry completions bash >> ~/.bash_completion  # Bash
poetry completions zsh > ~/.zfunc/_poetry      # Zsh

# Configure Poetry
poetry config virtualenvs.in-project true  # Create .venv in project
poetry config virtualenvs.prefer-active-python true
```

### B. Project Initialization

```bash
# Create new project with Poetry
poetry new my-project
cd my-project

# Initialize existing project
cd existing-project
poetry init

# Creates:
# - pyproject.toml
# - README.md
# - my_project/
# - tests/
```

### C. Virtual Environment Management

```bash
# Create virtual environment (automatic with most commands)
poetry install

# Run commands in virtual environment (recommended - no activation needed)
poetry run python script.py
poetry run pytest
poetry run mypy .

# Manual activation (if needed)
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# Show virtual environment info
poetry env info
poetry env list

# Show virtual environment path
poetry env info --path

# Remove virtual environment
poetry env remove python

# Use specific Python version
poetry env use 3.12
poetry env use python3.11
```

---

## 2. Hexagonal Architecture with Path Dependencies (MANDATORY)

### A. Project Structure

```
project-root/
├── pyproject.toml                 # Root project (optional aggregator)
├── poetry.lock                    # Shared lock file (COMMIT THIS)
├── README.md
├── .gitignore
│
├── packages/
│   ├── domain/                    # Core domain (no external deps)
│   │   ├── pyproject.toml
│   │   ├── poetry.lock
│   │   ├── myapp_domain/
│   │   │   ├── __init__.py
│   │   │   ├── entities/
│   │   │   │   ├── __init__.py
│   │   │   │   └── user.py
│   │   │   ├── value_objects/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── email.py
│   │   │   │   └── user_id.py
│   │   │   └── repositories/
│   │   │       ├── __init__.py
│   │   │       └── user_repository.py  # Interface
│   │   └── tests/
│   │       ├── __init__.py
│   │       └── test_user.py
│   │
│   ├── application/               # Use cases
│   │   ├── pyproject.toml
│   │   ├── poetry.lock
│   │   ├── myapp_application/
│   │   │   ├── __init__.py
│   │   │   ├── commands/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── create_user.py
│   │   │   │   └── update_user.py
│   │   │   ├── queries/
│   │   │   │   ├── __init__.py
│   │   │   │   └── get_user.py
│   │   │   └── services/
│   │   │       ├── __init__.py
│   │   │       └── user_service.py
│   │   └── tests/
│   │       ├── __init__.py
│   │       └── test_user_service.py
│   │
│   ├── infrastructure/            # External dependencies
│   │   ├── pyproject.toml
│   │   ├── poetry.lock
│   │   ├── myapp_infrastructure/
│   │   │   ├── __init__.py
│   │   │   ├── persistence/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── sqlalchemy_user_repository.py
│   │   │   │   └── models.py
│   │   │   ├── cache/
│   │   │   │   ├── __init__.py
│   │   │   │   └── redis_cache.py
│   │   │   └── messaging/
│   │   │       ├── __init__.py
│   │   │       └── rabbitmq_publisher.py
│   │   └── tests/
│   │       ├── __init__.py
│   │       └── test_sqlalchemy_repository.py
│   │
│   └── adapters/                  # API/UI adapters
│       ├── api/
│       │   ├── pyproject.toml
│       │   ├── poetry.lock
│       │   ├── myapp_api/
│       │   │   ├── __init__.py
│       │   │   ├── main.py
│       │   │   ├── routes/
│       │   │   │   ├── __init__.py
│       │   │   │   └── users.py
│       │   │   └── dependencies.py
│       │   └── tests/
│       │       ├── __init__.py
│       │       └── test_api.py
│       │
│       └── cli/
│           ├── pyproject.toml
│           ├── poetry.lock
│           ├── myapp_cli/
│           │   ├── __init__.py
│           │   └── main.py
│           └── tests/
│               ├── __init__.py
│               └── test_cli.py
│
└── scripts/
    ├── setup.sh
    ├── lint-all.sh
    └── test-all.sh
```

### B. Domain Layer pyproject.toml

```toml
# packages/domain/pyproject.toml

[tool.poetry]
name = "myapp-domain"
version = "1.0.0"
description = "Core domain layer - no external dependencies"
authors = ["Your Name <you@example.com>"]
readme = "README.md"
packages = [{include = "myapp_domain"}]

[tool.poetry.dependencies]
python = "^3.12"
# Domain layer should have MINIMAL dependencies
# Prefer standard library only

[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"
pytest-cov = "^4.1.0"
ruff = "^0.3.0"
mypy = "^1.8.0"

[tool.poetry.group.test.dependencies]
pytest = "^8.0.0"
pytest-cov = "^4.1.0"
pytest-asyncio = "^0.23.0"
pytest-mock = "^3.12.0"
faker = "^22.0.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

# Ruff configuration
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "C4", "SIM", "TCH"]
ignore = ["E501"]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101"]

# Mypy configuration
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

# Pytest configuration
[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--strict-markers",
    "--strict-config",
    "--cov=myapp_domain",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-fail-under=80",
]

# Coverage configuration
[tool.coverage.run]
source = ["myapp_domain"]
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@abstractmethod",
]
```

### C. Application Layer pyproject.toml

```toml
# packages/application/pyproject.toml

[tool.poetry]
name = "myapp-application"
version = "1.0.0"
description = "Application layer - use cases and orchestration"
authors = ["Your Name <you@example.com>"]
readme = "README.md"
packages = [{include = "myapp_application"}]

[tool.poetry.dependencies]
python = "^3.12"
# Path dependency on domain layer
myapp-domain = {path = "../domain", develop = true}

[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"
pytest-cov = "^4.1.0"
ruff = "^0.3.0"
mypy = "^1.8.0"

[tool.poetry.group.test.dependencies]
pytest = "^8.0.0"
pytest-cov = "^4.1.0"
pytest-asyncio = "^0.23.0"
pytest-mock = "^3.12.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

# Ruff, Mypy, Pytest configuration (same as domain)
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "C4", "SIM", "TCH"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.12"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-v", "--cov=myapp_application", "--cov-fail-under=80"]
```

### D. Infrastructure Layer pyproject.toml

```toml
# packages/infrastructure/pyproject.toml

[tool.poetry]
name = "myapp-infrastructure"
version = "1.0.0"
description = "Infrastructure layer - external dependencies"
authors = ["Your Name <you@example.com>"]
readme = "README.md"
packages = [{include = "myapp_infrastructure"}]

[tool.poetry.dependencies]
python = "^3.12"
# Path dependency on domain layer
myapp-domain = {path = "../domain", develop = true}
# External dependencies allowed here
sqlalchemy = {extras = ["asyncio"], version = "^2.0.0"}
alembic = "^1.13.0"
asyncpg = "^0.29.0"
redis = "^5.0.0"
pika = "^1.3.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"
pytest-cov = "^4.1.0"
pytest-asyncio = "^0.23.0"
ruff = "^0.3.0"
mypy = "^1.8.0"

[tool.poetry.group.test.dependencies]
pytest = "^8.0.0"
pytest-asyncio = "^0.23.0"
pytest-mock = "^3.12.0"
# Test database
aiosqlite = "^0.19.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

# Configuration (same as above)
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-v", "--cov=myapp_infrastructure", "--cov-fail-under=80"]
```

### E. API Adapter pyproject.toml

```toml
# packages/adapters/api/pyproject.toml

[tool.poetry]
name = "myapp-api"
version = "1.0.0"
description = "FastAPI REST API adapter"
authors = ["Your Name <you@example.com>"]
readme = "README.md"
packages = [{include = "myapp_api"}]

[tool.poetry.dependencies]
python = "^3.12"
# Path dependencies on all layers
myapp-domain = {path = "../../domain", develop = true}
myapp-application = {path = "../../application", develop = true}
myapp-infrastructure = {path = "../../infrastructure", develop = true}
# API-specific dependencies
fastapi = "^0.109.0"
uvicorn = {extras = ["standard"], version = "^0.27.0"}
pydantic = "^2.6.0"
pydantic-settings = "^2.1.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"
pytest-cov = "^4.1.0"
pytest-asyncio = "^0.23.0"
httpx = "^0.26.0"  # For testing async HTTP
ruff = "^0.3.0"
mypy = "^1.8.0"

[tool.poetry.group.test.dependencies]
pytest = "^8.0.0"
pytest-asyncio = "^0.23.0"
httpx = "^0.26.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

# Scripts
[tool.poetry.scripts]
serve = "myapp_api.main:serve"
serve-dev = "myapp_api.main:serve_dev"

# Configuration
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-v", "--cov=myapp_api", "--cov-fail-under=80"]
```

### F. Root Makefile for Multi-Package Management

```makefile
# Makefile - Manage all packages

.PHONY: install test lint format typecheck clean help

# Install all packages
install:
	@echo "Installing all packages..."
	cd packages/domain && poetry install
	cd packages/application && poetry install
	cd packages/infrastructure && poetry install
	cd packages/adapters/api && poetry install
	cd packages/adapters/cli && poetry install

# Run tests in all packages
test:
	@echo "Running tests in all packages..."
	cd packages/domain && poetry run pytest
	cd packages/application && poetry run pytest
	cd packages/infrastructure && poetry run pytest
	cd packages/adapters/api && poetry run pytest
	cd packages/adapters/cli && poetry run pytest

# Run linter in all packages
lint:
	@echo "Linting all packages..."
	cd packages/domain && poetry run ruff check .
	cd packages/application && poetry run ruff check .
	cd packages/infrastructure && poetry run ruff check .
	cd packages/adapters/api && poetry run ruff check .
	cd packages/adapters/cli && poetry run ruff check .

# Format code in all packages
format:
	@echo "Formatting all packages..."
	cd packages/domain && poetry run ruff format .
	cd packages/application && poetry run ruff format .
	cd packages/infrastructure && poetry run ruff format .
	cd packages/adapters/api && poetry run ruff format .
	cd packages/adapters/cli && poetry run ruff format .

# Type check all packages
typecheck:
	@echo "Type checking all packages..."
	cd packages/domain && poetry run mypy .
	cd packages/application && poetry run mypy .
	cd packages/infrastructure && poetry run mypy .
	cd packages/adapters/api && poetry run mypy .
	cd packages/adapters/cli && poetry run mypy .

# Run all checks
check: lint typecheck test

# Clean all packages
clean:
	@echo "Cleaning all packages..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true

# Help
help:
	@echo "Available commands:"
	@echo "  make install    - Install all packages"
	@echo "  make test       - Run tests in all packages"
	@echo "  make lint       - Run linter in all packages"
	@echo "  make format     - Format code in all packages"
	@echo "  make typecheck  - Type check all packages"
	@echo "  make check      - Run all checks"
	@echo "  make clean      - Clean all build artifacts"
```

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### TDD Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    TDD Red-Green-Refactor Cycle                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    ┌──────────┐         ┌──────────┐         ┌──────────┐      │
│    │   RED    │         │  GREEN   │         │ REFACTOR │      │
│    │  Write   │ ──────► │  Write   │ ──────► │ Improve  │      │
│    │ Failing  │         │ Minimal  │         │  Code    │      │
│    │  Test    │         │   Code   │         │          │      │
│    └──────────┘         └──────────┘         └────┬─────┘      │
│         ▲                                         │            │
│         │                                         │            │
│         └─────────────────────────────────────────┘            │
│                         Repeat                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Step 1: RED    → Write a test that fails (test doesn't exist yet)
Step 2: GREEN  → Write minimal code to make the test pass
Step 3: REFACTOR → Improve code quality while keeping tests green
Step 4: REPEAT → Continue with next feature/requirement
```

### Example TDD Workflow for Python with Poetry

```python
# =============================================================================
# Step 1: RED - Write failing test first
# =============================================================================
# packages/domain/tests/test_calculator.py

import pytest
from myapp_domain.services.calculator import Calculator


def test_add_two_positive_numbers() -> None:
    """Test adding two positive numbers."""
    calc = Calculator()
    result = calc.add(2, 3)
    assert result == 5


def test_add_negative_numbers() -> None:
    """Test adding negative numbers."""
    calc = Calculator()
    result = calc.add(-1, -1)
    assert result == -2


def test_add_with_zero() -> None:
    """Test adding with zero."""
    calc = Calculator()
    assert calc.add(0, 5) == 5
    assert calc.add(5, 0) == 5
```

```bash
# Run: poetry run pytest tests/test_calculator.py
# ❌ FAILS - ModuleNotFoundError: No module named 'myapp_domain.services.calculator'
```

```python
# =============================================================================
# Step 2: GREEN - Write minimal implementation to pass
# =============================================================================
# packages/domain/myapp_domain/services/calculator.py

class Calculator:
    """Basic calculator service."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
```

```bash
# Run: poetry run pytest tests/test_calculator.py -v
# ✅ PASSES
# tests/test_calculator.py::test_add_two_positive_numbers PASSED
# tests/test_calculator.py::test_add_negative_numbers PASSED
# tests/test_calculator.py::test_add_with_zero PASSED
```

```python
# =============================================================================
# Step 3: REFACTOR - Improve while keeping tests green
# =============================================================================
# packages/domain/myapp_domain/services/calculator.py

from typing import Union

Number = Union[int, float]


class Calculator:
    """
    Basic calculator service.

    Provides arithmetic operations with type safety.
    """

    def add(self, a: Number, b: Number) -> Number:
        """
        Add two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            Sum of a and b
        """
        return a + b
```

```bash
# Run: poetry run pytest tests/test_calculator.py -v
# ✅ PASSES - Tests still pass after refactoring
```

### Visual TDD Step-by-Step Example

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TDD Example: Implementing Email Validator                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: RED - Write Failing Test                                          │
│  ────────────────────────────────────────────────────────────────────────   │
│  $ poetry run pytest tests/test_email.py                                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ def test_valid_email_returns_true():                                │   │
│  │     validator = EmailValidator()                                    │   │
│  │     assert validator.is_valid("user@example.com") is True          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Result: ❌ FAILED - ImportError: cannot import 'EmailValidator'            │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 2: GREEN - Write Minimal Code                                        │
│  ────────────────────────────────────────────────────────────────────────   │
│  $ poetry run pytest tests/test_email.py                                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ class EmailValidator:                                               │   │
│  │     def is_valid(self, email: str) -> bool:                        │   │
│  │         return "@" in email and "." in email                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Result: ✅ PASSED - 1 passed in 0.02s                                      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 3: REFACTOR - Improve Implementation                                  │
│  ────────────────────────────────────────────────────────────────────────   │
│  $ poetry run pytest tests/test_email.py && poetry run mypy .               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ import re                                                           │   │
│  │ from dataclasses import dataclass                                   │   │
│  │                                                                     │   │
│  │ @dataclass                                                          │   │
│  │ class EmailValidator:                                               │   │
│  │     _pattern: str = r'^[\w\.-]+@[\w\.-]+\.\w+$'                    │   │
│  │                                                                     │   │
│  │     def is_valid(self, email: str) -> bool:                        │   │
│  │         return bool(re.match(self._pattern, email))                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Result: ✅ PASSED - Tests pass, types check                                │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 4: REPEAT - Add More Tests, Continue Cycle                            │
│  ────────────────────────────────────────────────────────────────────────   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ def test_invalid_email_missing_at():                                │   │
│  │     assert validator.is_valid("userexample.com") is False          │   │
│  │                                                                     │   │
│  │ def test_invalid_email_missing_domain():                            │   │
│  │     assert validator.is_valid("user@") is False                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Continue the Red-Green-Refactor cycle...                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### TDD Commands with Poetry

```bash
# Watch mode for TDD (install pytest-watch)
poetry add --group dev pytest-watch
poetry run ptw tests/  # Auto-runs tests on file changes

# Run specific test during TDD
poetry run pytest tests/test_calculator.py::test_add_two_positive_numbers -v

# Run with coverage to track progress
poetry run pytest --cov=myapp_domain --cov-report=term-missing

# Run only tests matching pattern
poetry run pytest -k "test_add" -v

# Run tests with verbose failure output
poetry run pytest -vvs --tb=short
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Bug Fix Workflow Diagram                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐                                                          │
│   │ Bug Reported │                                                          │
│   │   (#123)     │                                                          │
│   └──────┬───────┘                                                          │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │  1. REPRODUCE: Write test that demonstrates the bug              │     │
│   │     poetry run pytest tests/test_bug_123.py                      │     │
│   │     Result: ❌ FAILS (confirms bug exists)                        │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │  2. VERIFY: Ensure test fails for the RIGHT reason               │     │
│   │     Check error message matches expected bug behavior            │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │  3. FIX: Implement the bug fix                                   │     │
│   │     Modify the code to correct the behavior                      │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │  4. VERIFY FIX: Run the regression test                          │     │
│   │     poetry run pytest tests/test_bug_123.py                      │     │
│   │     Result: ✅ PASSES (bug is fixed)                              │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │  5. FULL SUITE: Run all tests to prevent regressions             │     │
│   │     poetry run pytest                                            │     │
│   │     Result: ✅ ALL PASS (no regressions introduced)               │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │  6. DOCUMENT: Add bug ID to test docstring                       │     │
│   │     """Bug #123: Description of what was fixed"""                │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────┐                                                          │
│   │   Deploy     │  Regression test prevents bug from returning             │
│   │   Safely     │                                                          │
│   └──────────────┘                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Example Bug Fix with Regression Test

```python
# =============================================================================
# Bug Report #456: Division by zero not handled in Calculator.divide()
#
# Reported: 2026-01-20
# Severity: High
# Description: Calculator.divide(10, 0) crashes with unhandled ZeroDivisionError
# Expected: Should raise a descriptive ValueError
# =============================================================================

# -----------------------------------------------------------------------------
# Step 1: Write test that REPRODUCES the bug (test will FAIL)
# -----------------------------------------------------------------------------
# packages/domain/tests/test_calculator_bug_456.py

import pytest
from myapp_domain.services.calculator import Calculator


def test_divide_by_zero_raises_value_error_bug_456() -> None:
    """
    Bug #456: Division by zero should raise ValueError.

    Regression test to prevent bug from returning.
    Discovered: 2026-01-20
    """
    calc = Calculator()

    with pytest.raises(ValueError, match="Cannot divide by zero"):
        calc.divide(10, 0)


def test_divide_by_zero_with_float_bug_456() -> None:
    """
    Bug #456: Division by zero should also handle 0.0.

    Edge case discovered during bug investigation.
    """
    calc = Calculator()

    with pytest.raises(ValueError, match="Cannot divide by zero"):
        calc.divide(10.5, 0.0)
```

```bash
# Run: poetry run pytest tests/test_calculator_bug_456.py -v
# ❌ FAILS - ZeroDivisionError: division by zero
# (Confirms bug exists - this is expected!)
```

```python
# -----------------------------------------------------------------------------
# Step 2: Current broken implementation (before fix)
# -----------------------------------------------------------------------------
# packages/domain/myapp_domain/services/calculator.py

class Calculator:
    """Calculator service."""

    def divide(self, a: float, b: float) -> float:
        """Divide a by b."""
        return a / b  # BUG: No handling for b == 0


# -----------------------------------------------------------------------------
# Step 3: Fix the bug
# -----------------------------------------------------------------------------
# packages/domain/myapp_domain/services/calculator.py

class Calculator:
    """Calculator service."""

    def divide(self, a: float, b: float) -> float:
        """
        Divide a by b.

        Args:
            a: Dividend
            b: Divisor (must not be zero)

        Returns:
            Result of a / b

        Raises:
            ValueError: If b is zero
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
```

```bash
# Run: poetry run pytest tests/test_calculator_bug_456.py -v
# ✅ PASSES
# tests/test_calculator_bug_456.py::test_divide_by_zero_raises_value_error_bug_456 PASSED
# tests/test_calculator_bug_456.py::test_divide_by_zero_with_float_bug_456 PASSED

# Run full test suite to check for regressions
# poetry run pytest
# ✅ ALL TESTS PASS - Safe to deploy
```

### Bug Fix Checklist

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Bug Fix Checklist                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Before Fix:                                                                │
│  □ Bug ticket/issue created with clear description                          │
│  □ Bug reproduced locally                                                   │
│  □ Regression test written that FAILS                                       │
│  □ Test failure message matches bug description                             │
│                                                                             │
│  During Fix:                                                                │
│  □ Minimal code change to fix the issue                                     │
│  □ No unrelated changes mixed in                                            │
│  □ Code follows existing patterns                                           │
│                                                                             │
│  After Fix:                                                                 │
│  □ Regression test now PASSES                                               │
│  □ All existing tests still PASS                                            │
│  □ Type checking passes: poetry run mypy .                                  │
│  □ Linting passes: poetry run ruff check .                                  │
│  □ Bug ID documented in test docstring                                      │
│  □ Commit message references bug ID                                         │
│                                                                             │
│  Commands:                                                                  │
│  $ poetry run pytest tests/test_bug_XXX.py -v  # Run regression test        │
│  $ poetry run pytest                            # Full suite                │
│  $ poetry run mypy .                           # Type check                 │
│  $ poetry run ruff check .                     # Lint                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Poetry Workflow (MANDATORY)

### A. Installing Dependencies

```bash
# Install dependencies (creates poetry.lock)
poetry install

# Install without dev dependencies
poetry install --without dev

# Install specific groups
poetry install --with docs
poetry install --only test

# Add new dependency
poetry add requests

# Add dev dependency
poetry add --group dev pytest-xdist

# Add dependency with version constraint
poetry add "fastapi>=0.109.0"
poetry add "pydantic^2.6.0"

# Add path dependency (for hexagonal architecture)
cd packages/application
poetry add --editable ../domain

# Update dependencies
poetry update

# Update specific package
poetry update fastapi

# Show installed packages
poetry show
poetry show --tree

# Remove dependency
poetry remove requests
```

### B. Running Commands

```bash
# Run Python script with Poetry (recommended)
poetry run python script.py

# Run Python module
poetry run -m pytest

# Run defined script from pyproject.toml
poetry run serve              # Custom script
poetry run serve-dev          # Custom script

# Run commands without poetry run prefix (manual activation)
# First, activate the virtual environment:
source $(poetry env info --path)/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows (if virtualenvs.in-project = true)

# Once activated, commands run in venv:
python script.py
pytest
mypy .

# Deactivate
deactivate

# Run command with poetry run (no activation needed - RECOMMENDED)
poetry run pytest
poetry run ruff check .
poetry run mypy .
```

### C. Lock File Management

```bash
# Generate lock file (automatic with poetry install/add)
poetry lock

# Update lock file without upgrading dependencies
poetry lock --no-update

# Verify lock file is consistent
poetry check

# Export to requirements.txt (for compatibility)
poetry export -f requirements.txt -o requirements.txt
poetry export --without dev -o requirements-prod.txt

# ALWAYS commit poetry.lock
git add poetry.lock pyproject.toml
git commit -m "chore(deps): update dependencies"
```

### D. Publishing Packages

```bash
# Build package
poetry build

# Creates:
# - dist/myapp-1.0.0.tar.gz
# - dist/myapp-1.0.0-py3-none-any.whl

# Configure PyPI credentials
poetry config pypi-token.pypi <your-token>

# Publish to PyPI
poetry publish

# Build and publish in one command
poetry publish --build

# Publish to private repository
poetry config repositories.private https://private-repo.com/simple/
poetry config http-basic.private <username> <password>
poetry publish -r private
```

---

## 4. Test-Driven Development with Poetry (MANDATORY)

### A. TDD Workflow Example

```python
# Step 1: RED - Write failing test first
# packages/domain/tests/test_user.py

from myapp_domain.entities.user import User
from myapp_domain.value_objects.email import Email
from myapp_domain.value_objects.user_id import UserId
import pytest


def test_create_user_with_valid_data() -> None:
    """Test creating a user with valid data."""
    # This will fail - classes don't exist yet
    user = User.create(
        name="John Doe",
        email="john@example.com"
    )
    
    assert isinstance(user.id, UserId)
    assert user.name == "John Doe"
    assert user.email.value == "john@example.com"


def test_user_with_invalid_email_raises_error() -> None:
    """Test that invalid email raises ValueError."""
    with pytest.raises(ValueError, match="Invalid email"):
        User.create(
            name="John Doe",
            email="invalid-email"
        )


def test_user_is_immutable() -> None:
    """Test that user entity is immutable."""
    user = User.create(name="John Doe", email="john@example.com")
    
    # Should return new instance
    updated = user.with_name("Jane Doe")
    
    assert user.name == "John Doe"
    assert updated.name == "Jane Doe"
    assert user is not updated


# Run: poetry run pytest
# ❌ FAILS - Classes don't exist yet

# Step 2: GREEN - Write minimal implementation
# packages/domain/myapp_domain/value_objects/user_id.py

from dataclasses import dataclass
import uuid


@dataclass(frozen=True)
class UserId:
    """User identifier value object."""
    
    value: str
    
    @classmethod
    def generate(cls) -> "UserId":
        """Generate a new user ID."""
        return cls(value=str(uuid.uuid4()))
    
    @classmethod
    def from_string(cls, value: str) -> "UserId":
        """Create UserId from string."""
        if not value:
            raise ValueError("User ID cannot be empty")
        return cls(value=value)


# packages/domain/myapp_domain/value_objects/email.py

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Email:
    """Email address value object."""
    
    value: str
    
    def __post_init__(self) -> None:
        """Validate email format."""
        if not self._is_valid(self.value):
            raise ValueError(f"Invalid email: {self.value}")
    
    @staticmethod
    def _is_valid(email: str) -> bool:
        """Check if email is valid."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))


# packages/domain/myapp_domain/entities/user.py

from dataclasses import dataclass, replace
from datetime import datetime

from myapp_domain.value_objects.email import Email
from myapp_domain.value_objects.user_id import UserId


@dataclass(frozen=True)
class User:
    """User domain entity."""
    
    id: UserId
    name: str
    email: Email
    created_at: datetime
    
    @classmethod
    def create(cls, name: str, email: str) -> "User":
        """Create a new user."""
        if not name or len(name) < 2:
            raise ValueError("Name must be at least 2 characters")
        
        return cls(
            id=UserId.generate(),
            name=name,
            email=Email(email),
            created_at=datetime.now()
        )
    
    def with_name(self, name: str) -> "User":
        """Return a new user with updated name."""
        if not name or len(name) < 2:
            raise ValueError("Name must be at least 2 characters")
        return replace(self, name=name)


# Run: poetry run pytest
# ✅ PASSES - All tests pass

# Step 3: REFACTOR - Improve while keeping tests green
# Tests still pass ✓
```

### B. Bug Fix with Regression Test

```python
# Bug Report #123: User.create crashes with empty name

# Step 1: Write test that reproduces the bug
# packages/domain/tests/test_user.py

def test_user_create_with_empty_name_raises_error_bug_123() -> None:
    """
    Bug #123: User.create should reject empty names.
    
    Discovered: 2026-01-18
    This test prevents regression.
    """
    with pytest.raises(ValueError, match="Name must be at least 2 characters"):
        User.create(name="", email="john@example.com")


def test_user_create_with_whitespace_name_raises_error_bug_123() -> None:
    """Bug #123: User.create should reject whitespace-only names."""
    with pytest.raises(ValueError, match="Name must be at least 2 characters"):
        User.create(name="  ", email="john@example.com")


# Run: poetry run pytest
# ❌ FAILS if bug exists

# Step 2: Fix the bug (already fixed in implementation above)
# The validation in User.create already handles this

# Run: poetry run pytest
# ✅ PASSES - Bug fixed, regression prevented
```

### C. Running Tests with Poetry

```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov

# Run specific test file
poetry run pytest tests/test_user.py

# Run specific test
poetry run pytest tests/test_user.py::test_create_user_with_valid_data

# Run tests by marker
poetry run pytest -m unit
poetry run pytest -m integration

# Run tests in parallel
poetry add --group dev pytest-xdist
poetry run pytest -n auto

# Watch mode for TDD
poetry add --group dev pytest-watch
poetry run ptw tests/

# Generate HTML coverage report
poetry run pytest --cov --cov-report=html
```

---

## 5. Hexagonal Architecture Implementation

### A. Domain Layer (Core)

```python
# packages/domain/myapp_domain/repositories/user_repository.py

from abc import ABC, abstractmethod
from typing import Optional

from myapp_domain.entities.user import User
from myapp_domain.value_objects.email import Email
from myapp_domain.value_objects.user_id import UserId


class UserRepository(ABC):
    """
    User repository interface (port).
    
    This is the domain's contract - infrastructure provides implementation.
    """
    
    @abstractmethod
    async def find_by_id(self, user_id: UserId) -> Optional[User]:
        """Find user by ID."""
        ...
    
    @abstractmethod
    async def find_by_email(self, email: Email) -> Optional[User]:
        """Find user by email."""
        ...
    
    @abstractmethod
    async def save(self, user: User) -> None:
        """Save user."""
        ...
    
    @abstractmethod
    async def delete(self, user_id: UserId) -> None:
        """Delete user."""
        ...
```

### B. Application Layer (Use Cases)

```python
# packages/application/myapp_application/commands/create_user.py

from dataclasses import dataclass

from myapp_domain.entities.user import User
from myapp_domain.repositories.user_repository import UserRepository
from myapp_domain.value_objects.email import Email


@dataclass(frozen=True)
class CreateUserCommand:
    """Command to create a new user."""
    
    name: str
    email: str


class CreateUserHandler:
    """Handler for CreateUserCommand."""
    
    def __init__(self, user_repository: UserRepository) -> None:
        self._user_repository = user_repository
    
    async def handle(self, command: CreateUserCommand) -> User:
        """
        Handle user creation.
        
        Args:
            command: The create user command
            
        Returns:
            The created user
            
        Raises:
            ValueError: If user with email already exists
        """
        # Check if user exists
        existing = await self._user_repository.find_by_email(
            Email(command.email)
        )
        
        if existing:
            raise ValueError(f"User with email {command.email} already exists")
        
        # Create user
        user = User.create(name=command.name, email=command.email)
        
        # Save user
        await self._user_repository.save(user)
        
        return user
```

### C. Infrastructure Layer (Implementations)

```python
# packages/infrastructure/myapp_infrastructure/persistence/sqlalchemy_user_repository.py

from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from myapp_domain.entities.user import User
from myapp_domain.repositories.user_repository import UserRepository
from myapp_domain.value_objects.email import Email
from myapp_domain.value_objects.user_id import UserId

from .models import UserModel


class SQLAlchemyUserRepository(UserRepository):
    """SQLAlchemy implementation of UserRepository."""
    
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
    
    async def find_by_id(self, user_id: UserId) -> Optional[User]:
        """Find user by ID."""
        result = await self._session.execute(
            select(UserModel).where(UserModel.id == user_id.value)
        )
        model = result.scalar_one_or_none()
        
        if not model:
            return None
        
        return self._to_entity(model)
    
    async def find_by_email(self, email: Email) -> Optional[User]:
        """Find user by email."""
        result = await self._session.execute(
            select(UserModel).where(UserModel.email == email.value)
        )
        model = result.scalar_one_or_none()
        
        if not model:
            return None
        
        return self._to_entity(model)
    
    async def save(self, user: User) -> None:
        """Save user."""
        model = UserModel(
            id=user.id.value,
            name=user.name,
            email=user.email.value,
            created_at=user.created_at
        )
        
        self._session.add(model)
        await self._session.flush()
    
    async def delete(self, user_id: UserId) -> None:
        """Delete user."""
        result = await self._session.execute(
            select(UserModel).where(UserModel.id == user_id.value)
        )
        model = result.scalar_one_or_none()
        
        if model:
            await self._session.delete(model)
            await self._session.flush()
    
    def _to_entity(self, model: UserModel) -> User:
        """Convert model to entity."""
        return User(
            id=UserId.from_string(model.id),
            name=model.name,
            email=Email(model.email),
            created_at=model.created_at
        )
```

### D. API Adapter Layer

```python
# packages/adapters/api/myapp_api/routes/users.py

from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from myapp_application.commands.create_user import CreateUserCommand, CreateUserHandler
from myapp_application.queries.get_user import GetUserQuery, GetUserHandler
from myapp_api.dependencies import get_create_user_handler, get_get_user_handler


router = APIRouter(prefix="/users", tags=["users"])


class CreateUserRequest(BaseModel):
    """Request model for creating a user."""
    
    name: str
    email: EmailStr


class UserResponse(BaseModel):
    """Response model for user."""
    
    id: str
    name: str
    email: str
    created_at: str


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=UserResponse)
async def create_user(
    request: CreateUserRequest,
    handler: Annotated[CreateUserHandler, Depends(get_create_user_handler)]
) -> UserResponse:
    """
    Create a new user.
    
    Args:
        request: User creation request
        handler: Create user handler (injected)
        
    Returns:
        Created user
        
    Raises:
        HTTPException: If user already exists or validation fails
    """
    try:
        command = CreateUserCommand(name=request.name, email=request.email)
        user = await handler.handle(command)
        
        return UserResponse(
            id=user.id.value,
            name=user.name,
            email=user.email.value,
            created_at=user.created_at.isoformat()
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    handler: Annotated[GetUserHandler, Depends(get_get_user_handler)]
) -> UserResponse:
    """Get user by ID."""
    query = GetUserQuery(user_id=user_id)
    user = await handler.handle(query)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )
    
    return UserResponse(
        id=user.id.value,
        name=user.name,
        email=user.email.value,
        created_at=user.created_at.isoformat()
    )


# packages/adapters/api/myapp_api/main.py

from fastapi import FastAPI

from myapp_api.routes import users


app = FastAPI(
    title="MyApp API",
    description="REST API with hexagonal architecture",
    version="1.0.0"
)

app.include_router(users.router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


def serve() -> None:
    """Run production server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


def serve_dev() -> None:
    """Run development server with reload."""
    import uvicorn
    uvicorn.run("myapp_api.main:app", host="0.0.0.0", port=8000, reload=True)
```

---

## 6. CI/CD Integration with Poetry

### A. GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml

name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
        package: ["domain", "application", "infrastructure", "adapters/api"]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install Poetry
        uses: snok/install-poetry@v1
        with:
          version: latest
          virtualenvs-create: true
          virtualenvs-in-project: true
      
      - name: Load cached venv
        id: cached-poetry-dependencies
        uses: actions/cache@v4
        with:
          path: packages/${{ matrix.package }}/.venv
          key: venv-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('packages/${{ matrix.package }}/poetry.lock') }}
      
      - name: Install dependencies
        working-directory: packages/${{ matrix.package }}
        run: poetry install --no-interaction
      
      - name: Run linter
        working-directory: packages/${{ matrix.package }}
        run: poetry run ruff check .
      
      - name: Run type checker
        working-directory: packages/${{ matrix.package }}
        run: poetry run mypy .
      
      - name: Run tests
        working-directory: packages/${{ matrix.package }}
        run: poetry run pytest --cov --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: packages/${{ matrix.package }}/coverage.xml
          flags: ${{ matrix.package }}
```

### B. GitLab CI Pipeline

```yaml
# .gitlab-ci.yml

stages:
  - test
  - build

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

.install-poetry: &install-poetry
  - curl -sSL https://install.python-poetry.org | python3 -
  - export PATH="$HOME/.local/bin:$PATH"
  - poetry config virtualenvs.in-project true

test:domain:
  stage: test
  image: python:3.12-slim
  before_script:
    - *install-poetry
  cache:
    paths:
      - .cache/pip
      - packages/domain/.venv
  script:
    - cd packages/domain
    - poetry install
    - poetry run ruff check .
    - poetry run mypy .
    - poetry run pytest --cov --cov-report=xml
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: packages/domain/coverage.xml

test:application:
  stage: test
  image: python:3.12-slim
  before_script:
    - *install-poetry
  script:
    - cd packages/application
    - poetry install
    - poetry run pytest --cov

test:infrastructure:
  stage: test
  image: python:3.12-slim
  before_script:
    - *install-poetry
  script:
    - cd packages/infrastructure
    - poetry install
    - poetry run pytest --cov

test:api:
  stage: test
  image: python:3.12-slim
  before_script:
    - *install-poetry
  script:
    - cd packages/adapters/api
    - poetry install
    - poetry run pytest --cov
```

### C. Dockerfile with Poetry

```dockerfile
# Dockerfile - Multi-stage build with Poetry

FROM python:3.12-slim AS builder

# Install Poetry
RUN pip install poetry==1.8.0

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set working directory
WORKDIR /app

# Copy dependency files
COPY packages/domain/pyproject.toml packages/domain/poetry.lock packages/domain/
COPY packages/application/pyproject.toml packages/application/poetry.lock packages/application/
COPY packages/infrastructure/pyproject.toml packages/infrastructure/poetry.lock packages/infrastructure/
COPY packages/adapters/api/pyproject.toml packages/adapters/api/poetry.lock packages/adapters/api/

# Install dependencies for each package
RUN cd packages/domain && poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR
RUN cd packages/application && poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR
RUN cd packages/infrastructure && poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR
RUN cd packages/adapters/api && poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

# Copy application code
COPY packages packages/

# Install packages
RUN cd packages/domain && poetry install --without dev && rm -rf $POETRY_CACHE_DIR
RUN cd packages/application && poetry install --without dev && rm -rf $POETRY_CACHE_DIR
RUN cd packages/infrastructure && poetry install --without dev && rm -rf $POETRY_CACHE_DIR
RUN cd packages/adapters/api && poetry install --without dev && rm -rf $POETRY_CACHE_DIR

# Production stage
FROM python:3.12-slim AS runtime

# Set environment
ENV PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/app/packages/adapters/api/.venv \
    PATH="/app/packages/adapters/api/.venv/bin:$PATH"

# Copy application and virtual environment
COPY --from=builder /app /app

WORKDIR /app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run application
CMD ["python", "-m", "uvicorn", "myapp_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 7. Deployment Checklist

### Project Setup
- [ ] **Poetry installed**: Latest version via install script
- [ ] **pyproject.toml**: Configured for each package
- [ ] **Path dependencies**: Properly linked between packages
- [ ] **Python version**: Specified in each pyproject.toml
- [ ] **Lock files**: poetry.lock committed for each package
- [ ] **Dependencies**: All layers properly configured
- [ ] **.gitignore**: .venv, __pycache__, .pytest_cache excluded

### Hexagonal Architecture
- [ ] **Domain layer**: No external dependencies
- [ ] **Application layer**: Depends only on domain
- [ ] **Infrastructure layer**: Implements domain interfaces
- [ ] **Adapter layers**: API, CLI properly separated
- [ ] **Repository interfaces**: Defined in domain
- [ ] **Dependency direction**: All point inward to domain

### Testing
- [ ] **pytest configured**: In each pyproject.toml
- [ ] **Coverage threshold**: ≥80% enforced
- [ ] **TDD practiced**: Tests written before code
- [ ] **Regression tests**: For all bug fixes
- [ ] **Test markers**: unit, integration, slow defined
- [ ] **Test scripts**: poetry run pytest works

### Code Quality
- [ ] **Ruff configured**: Linting and formatting
- [ ] **Mypy configured**: Strict type checking
- [ ] **Type hints**: All functions typed
- [ ] **Scripts defined**: In [tool.poetry.scripts]
- [ ] **Pre-commit hooks**: Optional but recommended

### Dependencies
- [ ] **Dependency groups**: dev, test, docs separated
- [ ] **Path dependencies**: develop = true for local packages
- [ ] **Lock files**: Up to date with poetry lock
- [ ] **Version constraints**: Properly specified (^, ~, ==)
- [ ] **No pip usage**: Everything via Poetry

### CI/CD
- [ ] **Poetry in CI**: Installed in pipeline
- [ ] **Tests run**: All packages tested
- [ ] **Coverage reported**: To Codecov or similar
- [ ] **Linting enforced**: In CI pipeline
- [ ] **Type checking**: In CI pipeline
- [ ] **Matrix testing**: Multiple Python versions

### Publishing (if applicable)
- [ ] **Package metadata**: Complete in pyproject.toml
- [ ] **README**: Clear package description
- [ ] **License**: Specified
- [ ] **Keywords**: For PyPI discoverability
- [ ] **Classifiers**: Python versions, license, etc.
- [ ] **Build**: poetry build works
- [ ] **PyPI token**: Configured for publishing

### Documentation
- [ ] **README**: Project setup instructions with Poetry
- [ ] **Architecture docs**: Hexagonal architecture explained
- [ ] **API docs**: If applicable
- [ ] **CONTRIBUTING**: Development workflow with Poetry
- [ ] **Makefile**: For multi-package management

---

## 8. Why This Configuration Works

1. **Poetry Maturity**: Battle-tested, widely adopted Python packaging tool.
2. **Distributed Architecture**: Separate pyproject.toml per layer maintains boundaries.
3. **Path Dependencies**: develop = true provides editable installs for development.
4. **Lock Files**: poetry.lock ensures reproducible builds across all environments.
5. **Dependency Groups**: Clean separation of dev, test, and optional dependencies.
6. **Built-in Publishing**: First-class PyPI publishing support.
7. **Hexagonal Architecture**: Clear boundaries, testable code, changeable infrastructure.
8. **TDD**: Tests first ensures quality and prevents regressions.
9. **Type Safety**: Mypy strict mode catches bugs at compile time.
10. **Scripts**: pyproject.toml scripts provide consistent task running.
11. **Virtual Environments**: Automatic isolation per project.
12. **Community**: Large ecosystem, extensive plugin support.

---

## 9. Quick Reference

### Common Commands

```bash
# =============================================================================
# Project Initialization
# =============================================================================
poetry new my-project              # Create new project with standard structure
poetry init                        # Initialize poetry in existing project

# =============================================================================
# Dependency Management
# =============================================================================
poetry install                     # Install dependencies from pyproject.toml
poetry install --without dev       # Install without dev dependencies
poetry install --with docs         # Install with optional docs group
poetry install --only test         # Install only test dependencies

poetry add requests                # Add runtime dependency
poetry add --group dev pytest      # Add dev dependency
poetry add --group test faker      # Add test dependency
poetry add "fastapi>=0.109.0"      # Add with version constraint
poetry add "pydantic^2.6.0"        # Add with caret constraint

poetry remove requests             # Remove dependency
poetry update                      # Update all dependencies
poetry update fastapi              # Update specific package

poetry show                        # Show installed packages
poetry show --tree                 # Show dependency tree
poetry show --outdated             # Show outdated packages

# =============================================================================
# Running Commands (Recommended - no activation needed)
# =============================================================================
poetry run python script.py        # Run Python script
poetry run pytest                  # Run tests
poetry run pytest -v --cov         # Run tests with coverage
poetry run mypy .                  # Type checking
poetry run ruff check .            # Linting
poetry run ruff format .           # Formatting

# =============================================================================
# Virtual Environment
# =============================================================================
poetry env info                    # Show venv information
poetry env info --path             # Show venv path
poetry env list                    # List available environments
poetry env use 3.12                # Use specific Python version
poetry env remove python           # Remove virtual environment

# Manual activation (if needed)
source $(poetry env info --path)/bin/activate  # Linux/macOS
# .venv\Scripts\activate           # Windows

# =============================================================================
# Lock File & Export
# =============================================================================
poetry lock                        # Generate/update lock file
poetry lock --no-update            # Regenerate without updating deps
poetry check                       # Verify pyproject.toml and lock file
poetry export -f requirements.txt -o requirements.txt  # Export to requirements.txt

# =============================================================================
# Building & Publishing
# =============================================================================
poetry build                       # Build package (sdist + wheel)
poetry publish                     # Publish to PyPI
poetry publish --build             # Build and publish
poetry publish -r private          # Publish to private repo

# =============================================================================
# Configuration
# =============================================================================
poetry config --list                           # Show all config
poetry config virtualenvs.in-project true      # Create .venv in project
poetry config virtualenvs.prefer-active-python true
poetry config pypi-token.pypi <token>          # Set PyPI token
```

### Poetry Patterns Cheat Sheet

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Poetry Patterns Cheat Sheet                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Version Constraints:                                                       │
│  ────────────────────────────────────────────────────────────────────────   │
│  ^2.6.0    →  >=2.6.0, <3.0.0    (caret - recommended for most cases)      │
│  ~2.6.0    →  >=2.6.0, <2.7.0    (tilde - patch updates only)              │
│  >=2.6.0   →  >=2.6.0            (minimum version)                          │
│  ==2.6.0   →  exactly 2.6.0      (exact version - use sparingly)           │
│  >=2.6,<3  →  >=2.6.0, <3.0.0    (range constraint)                        │
│  *         →  any version         (wildcard - avoid in production)          │
│                                                                             │
│  Dependency Groups:                                                         │
│  ────────────────────────────────────────────────────────────────────────   │
│  [tool.poetry.dependencies]           # Runtime dependencies                │
│  [tool.poetry.group.dev.dependencies] # Development tools                   │
│  [tool.poetry.group.test.dependencies]# Testing libraries                   │
│  [tool.poetry.group.docs.dependencies]# Documentation tools                 │
│                                                                             │
│  Path Dependencies (Hexagonal Architecture):                                │
│  ────────────────────────────────────────────────────────────────────────   │
│  myapp-domain = {path = "../domain", develop = true}                       │
│  myapp-application = {path = "../application", develop = true}             │
│                                                                             │
│  Optional Dependencies:                                                     │
│  ────────────────────────────────────────────────────────────────────────   │
│  sqlalchemy = {extras = ["asyncio"], version = "^2.0.0"}                   │
│  uvicorn = {extras = ["standard"], version = "^0.27.0"}                    │
│                                                                             │
│  Scripts (Entry Points):                                                    │
│  ────────────────────────────────────────────────────────────────────────   │
│  [tool.poetry.scripts]                                                      │
│  serve = "myapp_api.main:serve"        # poetry run serve                  │
│  migrate = "myapp_db.migrations:run"   # poetry run migrate                │
│                                                                             │
│  TDD Workflow:                                                              │
│  ────────────────────────────────────────────────────────────────────────   │
│  1. poetry run pytest tests/test_new.py  →  ❌ RED (fails)                  │
│  2. Write implementation                                                    │
│  3. poetry run pytest tests/test_new.py  →  ✅ GREEN (passes)               │
│  4. Refactor, poetry run pytest          →  ✅ Still passes                 │
│                                                                             │
│  Bug Fix Workflow:                                                          │
│  ────────────────────────────────────────────────────────────────────────   │
│  1. poetry run pytest tests/test_bug_N.py  →  ❌ Reproduces bug             │
│  2. Fix the bug                                                             │
│  3. poetry run pytest tests/test_bug_N.py  →  ✅ Bug fixed                  │
│  4. poetry run pytest                      →  ✅ No regressions             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### pyproject.toml Template

```toml
# =============================================================================
# pyproject.toml - Complete Poetry Configuration Template
# =============================================================================

[tool.poetry]
name = "myapp"
version = "1.0.0"
description = "A production-ready Python application"
authors = ["Your Name <you@example.com>"]
readme = "README.md"
license = "MIT"
repository = "https://github.com/username/myapp"
documentation = "https://myapp.readthedocs.io"
keywords = ["python", "poetry", "hexagonal"]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.12",
]
packages = [{include = "myapp"}]

# =============================================================================
# Dependencies
# =============================================================================

[tool.poetry.dependencies]
python = "^3.12"
pydantic = "^2.6.0"
pydantic-settings = "^2.1.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"
pytest-cov = "^4.1.0"
pytest-asyncio = "^0.23.0"
pytest-mock = "^3.12.0"
pytest-watch = "^4.2.0"
pytest-xdist = "^3.5.0"
ruff = "^0.3.0"
mypy = "^1.8.0"
pre-commit = "^3.6.0"

[tool.poetry.group.test.dependencies]
faker = "^22.0.0"
httpx = "^0.26.0"
aiosqlite = "^0.19.0"

[tool.poetry.group.docs]
optional = true

[tool.poetry.group.docs.dependencies]
mkdocs = "^1.5.0"
mkdocs-material = "^9.5.0"

# =============================================================================
# Scripts (Entry Points)
# =============================================================================

[tool.poetry.scripts]
serve = "myapp.main:serve"
serve-dev = "myapp.main:serve_dev"
migrate = "myapp.db.migrations:run"

# =============================================================================
# Build System
# =============================================================================

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

# =============================================================================
# Ruff Configuration (Linting & Formatting)
# =============================================================================

[tool.ruff]
line-length = 100
target-version = "py312"
exclude = [".venv", "__pycache__", ".git", "dist", "build"]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # Pyflakes
    "I",      # isort
    "N",      # pep8-naming
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "SIM",    # flake8-simplify
    "TCH",    # flake8-type-checking
    "PTH",    # flake8-use-pathlib
    "RUF",    # Ruff-specific rules
]
ignore = ["E501"]  # Line too long (handled by formatter)

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101"]  # Allow assert in tests

[tool.ruff.lint.isort]
known-first-party = ["myapp"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
docstring-code-format = true

# =============================================================================
# Mypy Configuration (Type Checking)
# =============================================================================

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
show_error_codes = true
namespace_packages = true
explicit_package_bases = true

[[tool.mypy.overrides]]
module = ["tests.*"]
disallow_untyped_defs = false

# =============================================================================
# Pytest Configuration
# =============================================================================

[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
asyncio_mode = "auto"
filterwarnings = ["ignore::DeprecationWarning"]
markers = [
    "unit: Unit tests (fast, no I/O)",
    "integration: Integration tests (may require external services)",
    "slow: Slow tests (>1s)",
]
addopts = [
    "-v",
    "--strict-markers",
    "--strict-config",
    "--cov=myapp",
    "--cov-report=term-missing",
    "--cov-report=html:htmlcov",
    "--cov-fail-under=80",
]

# =============================================================================
# Coverage Configuration
# =============================================================================

[tool.coverage.run]
source = ["myapp"]
branch = true
omit = ["*/tests/*", "*/__pycache__/*", "*/.venv/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@abstractmethod",
    "@overload",
]
show_missing = true
fail_under = 80
```

---

## References

- [Poetry Documentation](https://python-poetry.org/docs/)
- [Poetry GitHub Repository](https://github.com/python-poetry/poetry)
- [Python Packaging Guide](https://packaging.python.org/)
- [pytest Documentation](https://docs.pytest.org/)
- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [PEP 621 - pyproject.toml](https://peps.python.org/pep-0621/)
- [Poetry Plugins](https://python-poetry.org/docs/plugins/)

---

**Last Updated:** 2026-01-18  
**Version:** 1.0  
**Maintainer:** Development Team
