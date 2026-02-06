# Modern Python Development with uv
Mandatory standards and best practices for Python development using uv. Workspaces, hexagonal architecture, dependency management. uv, pyproject.toml, pytest, ruff, mypy, Python 3.12+.

---

**Agent Profile**: The Python uv Expert  
**Role**: Senior Python Engineer & Dependency Management Specialist  
**Objective**: Generate production-ready Python projects using uv for dependency management, virtual environments, and workspace configuration.  
**Tools**: uv, pyproject.toml, pytest, ruff, mypy, Python 3.12+.

## Core Philosophies

The agent must adhere to the "UV-FIRST" principles for every Python project:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**uv for Everything**: Use uv for dependencies, virtual environments, scripts, tools, and task running.
**Workspaces**: Use hierarchical pyproject.toml with workspaces for hexagonal architecture.
**Lock Files**: Always commit uv.lock for reproducible builds across all environments.
**Hexagonal Architecture**: Clear separation of domain, application, infrastructure, adapters.
**Type Safety**: Use type hints everywhere, enforce with mypy strict mode.

**Fast and Reproducible**: uv's speed and determinism ensure consistent environments.
**No Global Installs**: All tools (pytest, ruff, mypy) managed via uv, no pip/pipx needed.
**Dependency Groups**: Separate dev, test, docs, and optional dependencies.
**Python Version Management**: uv manages Python versions, no pyenv needed.
**Script Management**: Define scripts in pyproject.toml, run with uv run.
**Monorepo Ready**: Workspaces support for multi-package projects.

---

## 1. Getting Started with uv

### A. Installation

```bash
# Install uv (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip (not recommended, use script above)
pip install uv

# Or via pipx
pipx install uv

# Verify installation
uv --version

# Update uv
uv self update
```

### B. Project Initialization

```bash
# Create new project with uv
uv init my-project
cd my-project

# Initialize existing project
cd existing-project
uv init

# Creates:
# - pyproject.toml
# - .python-version
# - src/my_project/
# - README.md
```

### C. Virtual Environment Management

```bash
# Create virtual environment (automatic with most commands)
uv venv

# Activate virtual environment (if needed manually)
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# uv commands automatically use .venv, no activation needed!
uv add requests     # Installs into .venv
uv run script.py     # Runs using .venv
uv run pytest              # Runs pytest from .venv

# Specify Python version
uv venv --python 3.12
uv venv --python python3.11

# Remove virtual environment
rm -rf .venv
```

---

## 2. Hexagonal Architecture with Workspaces (MANDATORY)

### A. Project Structure

```
project-root/
├── pyproject.toml                 # Root workspace configuration
├── uv.lock                        # Locked dependencies (COMMIT THIS)
├── .python-version                # Python version (e.g., 3.12)
├── README.md
├── .gitignore
│
├── packages/
│   ├── domain/                    # Core domain (no external deps)
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   │   └── myapp_domain/
│   │   │       ├── __init__.py
│   │   │       ├── entities/
│   │   │       │   ├── __init__.py
│   │   │       │   └── user.py
│   │   │       ├── value_objects/
│   │   │       │   ├── __init__.py
│   │   │       │   ├── email.py
│   │   │       │   └── user_id.py
│   │   │       └── repositories/
│   │   │           ├── __init__.py
│   │   │           └── user_repository.py  # Interface
│   │   └── tests/
│   │       └── test_user.py
│   │
│   ├── application/               # Use cases
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   │   └── myapp_application/
│   │   │       ├── __init__.py
│   │   │       ├── commands/
│   │   │       │   ├── __init__.py
│   │   │       │   ├── create_user.py
│   │   │       │   └── update_user.py
│   │   │       ├── queries/
│   │   │       │   ├── __init__.py
│   │   │       │   └── get_user.py
│   │   │       └── services/
│   │   │           ├── __init__.py
│   │   │           └── user_service.py
│   │   └── tests/
│   │       └── test_user_service.py
│   │
│   ├── infrastructure/            # External dependencies
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   │   └── myapp_infrastructure/
│   │   │       ├── __init__.py
│   │   │       ├── persistence/
│   │   │       │   ├── __init__.py
│   │   │       │   ├── sqlalchemy_user_repository.py
│   │   │       │   └── models.py
│   │   │       ├── cache/
│   │   │       │   ├── __init__.py
│   │   │       │   └── redis_cache.py
│   │   │       └── messaging/
│   │   │           ├── __init__.py
│   │   │           └── rabbitmq_publisher.py
│   │   └── tests/
│   │       └── test_sqlalchemy_repository.py
│   │
│   └── adapters/                  # API/UI adapters
│       ├── api/
│       │   ├── pyproject.toml
│       │   ├── src/
│       │   │   └── myapp_api/
│       │   │       ├── __init__.py
│       │   │       ├── main.py
│       │   │       ├── routes/
│       │   │       │   ├── __init__.py
│       │   │       │   └── users.py
│       │   │       └── dependencies.py
│       │   └── tests/
│       │       └── test_api.py
│       │
│       └── cli/
│           ├── pyproject.toml
│           ├── src/
│           │   └── myapp_cli/
│           │       ├── __init__.py
│           │       └── main.py
│           └── tests/
│               └── test_cli.py
│
└── scripts/
    ├── setup.sh
    └── lint.sh
```

### B. Root pyproject.toml (Workspace Configuration)

```toml
# pyproject.toml - Root workspace configuration

[project]
name = "myapp"
version = "1.0.0"
description = "My application with hexagonal architecture"
authors = [
    {name = "Your Name", email = "you@example.com"}
]
readme = "README.md"
requires-python = ">=3.12"
license = {text = "MIT"}

# Workspace configuration (MANDATORY for hexagonal architecture)
[tool.uv.workspace]
members = [
    "packages/domain",
    "packages/application",
    "packages/infrastructure",
    "packages/adapters/api",
    "packages/adapters/cli",
]

# Root dependencies (shared across workspace)
dependencies = []

# Development dependencies (available to all workspace members)
[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
]

test = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.23.0",
    "pytest-mock>=3.12.0",
    "faker>=22.0.0",
]

docs = [
    "sphinx>=7.2.0",
    "sphinx-rtd-theme>=2.0.0",
    "myst-parser>=2.0.0",
]

# Build system
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# Ruff configuration (linting and formatting)
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
    "TCH", # flake8-type-checking
]
ignore = ["E501"]  # Line too long (handled by formatter)

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101"]  # Allow assert in tests

# Mypy configuration (type checking)
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_generics = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
follow_imports = "normal"
ignore_missing_imports = false

# Pytest configuration
[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["packages/*/tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--strict-markers",
    "--strict-config",
    "--cov=packages",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-fail-under=80",
]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow tests",
]

# Coverage configuration
[tool.coverage.run]
source = ["packages"]
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
]

# Scripts (run with: uv run <script>)
[tool.uv.scripts]
test = "pytest"
test-unit = "pytest -m unit"
test-integration = "pytest -m integration"
test-cov = "pytest --cov --cov-report=html"
lint = "ruff check ."
format = "ruff format ."
format-check = "ruff format --check ."
typecheck = "mypy packages"
check-all = ["lint", "format-check", "typecheck", "test"]
```

### C. Domain Layer pyproject.toml

```toml
# packages/domain/pyproject.toml

[project]
name = "myapp-domain"
version = "1.0.0"
description = "Core domain layer - no external dependencies"
requires-python = ">=3.12"

# Domain layer should have MINIMAL dependencies
# Prefer standard library only
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/myapp_domain"]
```

### D. Application Layer pyproject.toml

```toml
# packages/application/pyproject.toml

[project]
name = "myapp-application"
version = "1.0.0"
description = "Application layer - use cases and orchestration"
requires-python = ">=3.12"

# Depends on domain layer only
dependencies = [
    "myapp-domain",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/myapp_application"]
```

### E. Infrastructure Layer pyproject.toml

```toml
# packages/infrastructure/pyproject.toml

[project]
name = "myapp-infrastructure"
version = "1.0.0"
description = "Infrastructure layer - external dependencies"
requires-python = ">=3.12"

# External dependencies allowed here
dependencies = [
    "myapp-domain",  # Implements domain interfaces
    "sqlalchemy>=2.0.0",
    "alembic>=1.13.0",
    "redis>=5.0.0",
    "pika>=1.3.0",  # RabbitMQ
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/myapp_infrastructure"]
```

### F. API Adapter pyproject.toml

```toml
# packages/adapters/api/pyproject.toml

[project]
name = "myapp-api"
version = "1.0.0"
description = "FastAPI REST API adapter"
requires-python = ">=3.12"

dependencies = [
    "myapp-domain",
    "myapp-application",
    "myapp-infrastructure",
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.6.0",
    "pydantic-settings>=2.1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/myapp_api"]

# API-specific scripts
[tool.uv.scripts]
serve = "uvicorn myapp_api.main:app --reload"
serve-prod = "uvicorn myapp_api.main:app --host 0.0.0.0 --port 8000"
```

---

## 3. uv Workflow (MANDATORY)

### A. Installing Dependencies

```bash
# Install all dependencies (creates/updates uv.lock)
uv sync

# Install only production dependencies
uv sync --no-dev

# Install specific dependency group
uv sync --group test

# Add new dependency to a specific package
cd packages/domain
uv add pydantic

# Add dev dependency to root workspace
uv add --dev pytest-xdist

# Add dependency to specific package from root
uv add --package myapp-infrastructure sqlalchemy

# Update all dependencies
uv sync --upgrade

# Update specific package
uv add --upgrade fastapi
```

### B. Running Commands

```bash
# Run Python script with uv (uses .venv automatically)
uv run script.py

# Run Python module
uv run -m pytest

# Run defined script from pyproject.toml
uv run test              # pytest
uv run lint              # ruff check
uv run format            # ruff format
uv run typecheck         # mypy
uv run check-all         # all checks

# Run command in specific package context
cd packages/api
uv run serve             # uvicorn myapp_api.main:app --reload

# Run with environment variables
DATABASE_URL=postgresql://... uv run python script.py
```

### C. Managing Python Versions

```bash
# Install Python version
uv python install 3.12
uv python install 3.11

# List installed Python versions
uv python list

# Set Python version for project
echo "3.12" > .python-version

# Use specific Python version
uv run --python 3.12 python script.py
uv venv --python 3.11
```

### D. Lock File Management

```bash
# Generate lock file (automatic with uv sync)
uv lock

# Update lock file
uv lock --upgrade

# Verify lock file is up to date
uv lock --check

# Export to requirements.txt (for compatibility)
uv export > requirements.txt
uv export --no-dev > requirements-prod.txt

# ALWAYS commit uv.lock
git add uv.lock pyproject.toml
git commit -m "chore(deps): update dependencies"
```

---

## 4. Test-Driven Development with uv (MANDATORY)

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


# Run: uv run pytest packages/domain/tests/
# ❌ FAILS - Classes don't exist yet

# Step 2: GREEN - Write minimal implementation
# packages/domain/src/myapp_domain/value_objects/user_id.py

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


# packages/domain/src/myapp_domain/value_objects/email.py

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


# packages/domain/src/myapp_domain/entities/user.py

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
    
    def with_email(self, email: str) -> "User":
        """Return a new user with updated email."""
        return replace(self, email=Email(email))


# Run: uv run pytest packages/domain/tests/
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


# Run: uv run pytest
# ❌ FAILS if bug exists

# Step 2: Fix the bug (already fixed in implementation above)
# The validation in User.create already handles this

# Run: uv run pytest
# ✅ PASSES - Bug fixed, regression prevented
```

### C. Running Tests with uv

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run test-cov

# Run specific test file
uv run pytest packages/domain/tests/test_user.py

# Run specific test
uv run pytest packages/domain/tests/test_user.py::test_create_user_with_valid_data

# Run tests by marker
uv run pytest -m unit
uv run pytest -m integration

# Run tests in parallel
uv add --dev pytest-xdist
uv run pytest -n auto

# Watch mode for TDD
uv add --dev pytest-watch
uv run ptw packages/domain/tests/
```

---

## 5. Hexagonal Architecture Implementation

### A. Domain Layer (Core)

```python
# packages/domain/src/myapp_domain/repositories/user_repository.py

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
        pass
    
    @abstractmethod
    async def find_by_email(self, email: Email) -> Optional[User]:
        """Find user by email."""
        pass
    
    @abstractmethod
    async def save(self, user: User) -> None:
        """Save user."""
        pass
    
    @abstractmethod
    async def delete(self, user_id: UserId) -> None:
        """Delete user."""
        pass
```

### B. Application Layer (Use Cases)

```python
# packages/application/src/myapp_application/commands/create_user.py

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
# packages/infrastructure/src/myapp_infrastructure/persistence/sqlalchemy_user_repository.py

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
# packages/adapters/api/src/myapp_api/routes/users.py

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
```

---

## 6. CI/CD Integration with uv

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
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v2
        with:
          version: "latest"
      
      - name: Set up Python ${{ matrix.python-version }}
        run: uv python install ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: uv sync --all-groups
      
      - name: Run linter
        run: uv run lint
      
      - name: Run type checker
        run: uv run typecheck
      
      - name: Run tests
        run: uv run test-cov
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

### B. GitLab CI Pipeline

```yaml
# .gitlab-ci.yml

stages:
  - test
  - build

variables:
  UV_CACHE_DIR: ${CI_PROJECT_DIR}/.uv-cache

before_script:
  - curl -LsSf https://astral.sh/uv/install.sh | sh
  - export PATH="$HOME/.cargo/bin:$PATH"
  - uv python install 3.12

test:
  stage: test
  image: python:3.12-slim
  cache:
    paths:
      - .uv-cache
      - .venv
  script:
    - uv sync --all-groups
    - uv run lint
    - uv run typecheck
    - uv run test-cov
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
```

### C. Dockerfile with uv

```dockerfile
# Dockerfile

FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./
COPY packages packages/

# Install dependencies
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.12-slim

# Copy uv
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy application and dependencies
COPY --from=builder /app/.venv .venv
COPY --from=builder /app/packages packages/

# Set environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Run application
CMD ["uv", "run", "--no-sync", "serve-prod"]
```

---

## 7. Deployment Checklist

### Project Setup
- [ ] **uv installed**: Latest version via install script
- [ ] **pyproject.toml**: Root workspace configuration
- [ ] **Workspaces**: Configured for hexagonal architecture
- [ ] **Python version**: Specified in .python-version
- [ ] **Lock file**: uv.lock committed to repository
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
- [ ] **pytest configured**: In root pyproject.toml
- [ ] **Coverage threshold**: ≥80% enforced
- [ ] **TDD practiced**: Tests written before code
- [ ] **Regression tests**: For all bug fixes
- [ ] **Test markers**: unit, integration, slow defined
- [ ] **Test scripts**: Defined in pyproject.toml

### Code Quality
- [ ] **Ruff configured**: Linting and formatting
- [ ] **Mypy configured**: Strict type checking
- [ ] **Type hints**: All functions typed
- [ ] **Scripts defined**: test, lint, format, typecheck
- [ ] **Pre-commit hooks**: Optional but recommended

### Dependencies
- [ ] **Dependency groups**: dev, test, docs separated
- [ ] **Layer dependencies**: Correctly specified
- [ ] **Lock file**: Up to date with uv lock
- [ ] **No pip usage**: Everything via uv
- [ ] **Version constraints**: Properly specified

### CI/CD
- [ ] **uv in CI**: Installed in pipeline
- [ ] **Tests run**: All tests passing
- [ ] **Coverage reported**: To Codecov or similar
- [ ] **Linting enforced**: In CI pipeline
- [ ] **Type checking**: In CI pipeline
- [ ] **Matrix testing**: Multiple Python versions

### Documentation
- [ ] **README**: Project setup instructions with uv
- [ ] **Architecture docs**: Hexagonal architecture explained
- [ ] **API docs**: If applicable
- [ ] **CONTRIBUTING**: Development workflow with uv

---

## 8. Why This Configuration Works

1. **uv Speed**: 10-100x faster than pip, instant dependency resolution.
2. **Workspaces**: Perfect for hexagonal architecture, manages inter-package dependencies.
3. **Lock Files**: uv.lock ensures reproducible builds across all environments.
4. **No pip Needed**: uv replaces pip, pip-tools, pipx, and virtualenv.
5. **Python Management**: uv manages Python versions, no pyenv needed.
6. **Scripts**: pyproject.toml scripts provide consistent task running.
7. **Hexagonal Architecture**: Clear boundaries, testable code, changeable infrastructure.
8. **TDD**: Tests first ensures quality and prevents regressions.
9. **Type Safety**: Mypy strict mode catches bugs at compile time.
10. **Fast CI**: uv's speed dramatically reduces CI/CD times.
11. **Monorepo Ready**: Workspaces support complex project structures.
12. **Zero Config**: Sensible defaults, works out of the box.

---

## References

- [uv Documentation](https://docs.astral.sh/uv/)
- [uv GitHub Repository](https://github.com/astral-sh/uv)
- [Python Packaging Guide](https://packaging.python.org/)
- [pytest Documentation](https://docs.pytest.org/)
- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [PEP 621 - pyproject.toml](https://peps.python.org/pep-0621/)

---

**Last Updated:** 2026-01-18  
**Version:** 1.0  
**Maintainer:** Development Team


**End of Modern Python Development with uv**
