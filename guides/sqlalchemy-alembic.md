# SQLAlchemy & Alembic Development Guidelines

This document provides comprehensive standards for using SQLAlchemy ORM and Alembic migrations in Python applications.

---

**Agent Profile**: The Python Database Expert
**Role**: Senior Python Backend Engineer & Database Schema Architect
**Objective**: Generate secure, performant, and maintainable database code using SQLAlchemy 2.0+ and Alembic.
**Tools**: SQLAlchemy 2.0+, Alembic, PostgreSQL, MySQL, SQLite, pytest, uv.
**Companion Guides**: sql.md, postgresql.md, python.md, testing.md, secure-coding.md

---

## 1. Core Philosophies: ORM-SMART

The agent must adhere to the **ORM-SMART** principles:

**Test-Driven Development (TDD)**: ALWAYS write database tests BEFORE implementation. Test migrations up AND down.
**Regression Shield**: EVERY database bug MUST receive a test BEFORE fixing to prevent data corruption.

### CRITICAL: Migration Verification Requirement

```
┌─────────────────────────────────────────────────────────────────────┐
│  ⚠️  MANDATORY: Every migration MUST be verified before delivery    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Agents MUST:                                                        │
│  1. Test upgrade() executes without errors                          │
│  2. Test downgrade() executes without errors                        │
│  3. Verify upgrade → downgrade → upgrade cycle works                │
│  4. Confirm no data loss occurs during rollback                     │
│  5. NEVER present untested migrations to users                      │
│                                                                      │
│  Migrations without verified rollback capability are REJECTED.      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### ORM-SMART Principles

- **O**RM for Business Logic - Use ORM for complex domain logic; raw SQL for performance-critical paths
- **R**eversible Migrations - Every migration MUST have a working, TESTED downgrade path
- **M**apped Types - Use SQLAlchemy 2.0 `Mapped` type annotations everywhere

- **S**ession Lifecycle - Explicit session management with context managers
- **M**igration Testing - Test EVERY migration (up AND down) in CI before production
- **A**sync-Aware - Design for async from the start when needed
- **R**eproducible State - Migrations must be idempotent and deterministic
- **T**ransaction Boundaries - Explicit commit/rollback, never rely on autocommit

**Additional Principles:**

- **Portability**: Test against SQLite in CI, production database in staging
- **Type Safety**: Full type annotations with `Mapped[]` for IDE support
- **Lazy by Default**: Relationships lazy-loaded unless explicitly eager
- **Connection Pooling**: Always configure appropriate pool settings
- **Zero Data Loss**: Rollbacks must preserve all data that existed before upgrade

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Pre-Generation Verification Protocol

**CRITICAL: Agents MUST verify database context before generating code.**

#### Pre-Task Checklist

**Before writing ANY SQLAlchemy code, the agent MUST:**

1. **Identify SQLAlchemy Version**:
   ```python
   import sqlalchemy
   print(sqlalchemy.__version__)  # Must be 2.0+
   ```

2. **Understand Existing Models**:
   ```bash
   # Find existing models
   find . -name "*.py" -exec grep -l "DeclarativeBase\|mapped_column" {} \;

   # Check for existing Base class
   grep -r "class Base" --include="*.py"
   ```

3. **Review Migration History**:
   ```bash
   # Check current migration state
   alembic current

   # View migration history
   alembic history --verbose
   ```

### B. Code Verification Checklist

- [ ] Using SQLAlchemy 2.0+ syntax with `Mapped[]` and `mapped_column()`
- [ ] All models inherit from shared `DeclarativeBase`
- [ ] Session management uses context managers
- [ ] Relationships define both sides with `back_populates`
- [ ] Foreign keys have appropriate `ondelete` actions
- [ ] Indexes defined for query patterns
- [ ] Type annotations on all model attributes
- [ ] Tests exist for all model operations

### C. Migration Verification Checklist (MANDATORY)

**CRITICAL: Agents MUST complete ALL checks before presenting a migration.**

#### Pre-Flight Checks
- [ ] Migration has both `upgrade()` and `downgrade()` functions
- [ ] Both functions have meaningful implementations (not just `pass`)
- [ ] Downgrade reverses ALL changes made by upgrade

#### Verification Execution (MUST RUN)
- [ ] `alembic upgrade head` - Upgrade succeeds without errors
- [ ] `alembic downgrade -1` - Downgrade succeeds without errors
- [ ] `alembic upgrade head` - Re-upgrade succeeds (cycle test)
- [ ] Data preserved after downgrade (row counts match)

#### Safety Checks
- [ ] No destructive operations without explicit user approval
- [ ] Data migrations handle NULL values and edge cases
- [ ] Long-running operations use batching (>10k rows)
- [ ] Index creation uses `CONCURRENTLY` where supported
- [ ] Migration tested against production-like data volume
- [ ] Rollback procedure documented in migration docstring

### D. Prohibited Practices (MANDATORY)

**NEVER do the following - violations will cause data loss or corruption:**

#### Migration Prohibitions
- [ ] Present a migration without testing BOTH upgrade AND downgrade
- [ ] Skip `downgrade()` implementation ("we'll never need to rollback")
- [ ] Use `pass` as the downgrade() body without explicit approval
- [ ] Deploy migrations without verifying the upgrade→downgrade→upgrade cycle
- [ ] Assume autogenerated downgrades are correct (ALWAYS verify)
- [ ] Drop columns/tables without data backup confirmation

#### Code Prohibitions
- [ ] Use SQLAlchemy 1.x `Column()` syntax in new code
- [ ] Create sessions without context managers
- [ ] Use `session.commit()` without error handling
- [ ] Use `autocommit=True` mode
- [ ] Share sessions across threads/async tasks
- [ ] Use `expire_on_commit=True` with async code
- [ ] Concatenate strings into queries (SQL injection risk)
- [ ] Ignore relationship loading strategies

---

## 3. Model Definition Standards (MANDATORY)

### A. Base Class Setup

```python
"""Database base configuration.

This module defines the shared declarative base and common mixins
for all SQLAlchemy models.
"""
from datetime import datetime
from typing import Any

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func

# Naming convention for constraints (required for Alembic autogenerate)
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models.

    Provides consistent metadata configuration and naming conventions.
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampMixin:
    """Mixin providing created_at and updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class SoftDeleteMixin:
    """Mixin providing soft delete functionality."""

    deleted_at: Mapped[datetime | None] = mapped_column(default=None)

    @property
    def is_deleted(self) -> bool:
        """Check if record is soft deleted."""
        return self.deleted_at is not None
```

### B. Model Definition Template

```python
"""User domain models.

This module defines the User and related models for authentication
and user management.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, SoftDeleteMixin, TimestampMixin

if TYPE_CHECKING:
    from .address import Address


class UserStatus(str, Enum):
    """User account status enumeration."""

    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"


class User(Base, TimestampMixin, SoftDeleteMixin):
    """User account model.

    Represents a user in the system with authentication credentials
    and profile information.

    Attributes:
        id: Primary key, auto-generated.
        email: Unique email address for login.
        password_hash: Hashed password (never store plaintext).
        first_name: User's first name.
        last_name: User's last name.
        status: Account status (pending, active, suspended).
        addresses: Related address records.
    """

    __tablename__ = "users"

    # Primary key - use BIGINT for scalability
    id: Mapped[int] = mapped_column(primary_key=True)

    # Authentication fields
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
    )
    password_hash: Mapped[str] = mapped_column(String(255))

    # Profile fields
    first_name: Mapped[str] = mapped_column(String(100))
    last_name: Mapped[str] = mapped_column(String(100))
    bio: Mapped[str | None] = mapped_column(Text, default=None)

    # Status with enum constraint
    status: Mapped[UserStatus] = mapped_column(
        default=UserStatus.PENDING,
    )

    # Relationships - always define both sides
    addresses: Mapped[list[Address]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",  # Explicit loading strategy
    )

    # Table-level constraints and indexes
    __table_args__ = (
        CheckConstraint(
            "length(email) >= 5",
            name="email_min_length",
        ),
        Index(
            "ix_users_status_created",
            "status",
            "created_at",
        ),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"User(id={self.id!r}, email={self.email!r})"

    @property
    def full_name(self) -> str:
        """Return user's full name."""
        return f"{self.first_name} {self.last_name}"
```

### C. Relationship Patterns

```python
"""Relationship pattern examples."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, Table
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .user import User

# Many-to-Many association table
user_roles = Table(
    "user_roles",
    Base.metadata,
    mapped_column("user_id", ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    mapped_column("role_id", ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
)


class Address(Base):
    """User address model - One-to-Many relationship."""

    __tablename__ = "addresses"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    street: Mapped[str] = mapped_column(String(255))
    city: Mapped[str] = mapped_column(String(100))

    # Back-reference to parent
    user: Mapped[User] = relationship(back_populates="addresses")


class Role(Base):
    """Role model - Many-to-Many with User."""

    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)

    # Many-to-many relationship
    users: Mapped[list[User]] = relationship(
        secondary=user_roles,
        back_populates="roles",
        lazy="selectin",
    )


class Profile(Base):
    """Profile model - One-to-One relationship."""

    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,  # Ensures one-to-one
    )
    avatar_url: Mapped[str | None] = mapped_column(String(500))

    # One-to-one relationship
    user: Mapped[User] = relationship(
        back_populates="profile",
        uselist=False,  # Returns single object, not list
    )
```

---

## 4. Session Management (MANDATORY)

### A. Synchronous Session Pattern

```python
"""Synchronous session management."""
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .base import Base

# Engine configuration
engine = create_engine(
    "postgresql://user:pass@localhost/dbname",
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,  # Recycle connections after 30 minutes
    pool_pre_ping=True,  # Verify connection health
    echo=False,  # Set True for debugging SQL
)

# Session factory
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=True,
)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide a transactional scope around operations.

    Yields:
        Session: Database session with automatic commit/rollback.

    Example:
        with get_session() as session:
            user = User(email="test@example.com")
            session.add(user)
            # Commits on successful exit, rolls back on exception
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Usage patterns
def create_user(email: str, name: str) -> User:
    """Create a new user."""
    with get_session() as session:
        user = User(email=email, first_name=name, last_name="")
        session.add(user)
        session.flush()  # Get ID before commit
        return user


def get_user_by_email(email: str) -> User | None:
    """Get user by email address."""
    with get_session() as session:
        return session.query(User).filter(User.email == email).first()
```

### B. Async Session Pattern

```python
"""Asynchronous session management."""
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .base import Base

# Async engine configuration
async_engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/dbname",
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
    echo=False,
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autoflush=False,
    expire_on_commit=False,  # IMPORTANT: Must be False for async
)


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async transactional scope.

    Yields:
        AsyncSession: Async database session.

    Example:
        async with get_async_session() as session:
            user = User(email="test@example.com")
            session.add(user)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# FastAPI dependency pattern
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    async with get_async_session() as session:
        yield session


# Usage with explicit transaction
async def transfer_funds(from_id: int, to_id: int, amount: float) -> None:
    """Transfer funds between accounts with explicit transaction."""
    async with get_async_session() as session:
        async with session.begin():  # Explicit transaction
            from_account = await session.get(Account, from_id)
            to_account = await session.get(Account, to_id)

            if from_account.balance < amount:
                raise ValueError("Insufficient funds")

            from_account.balance -= amount
            to_account.balance += amount
            # Commits on successful exit of inner context
```

### C. Query Patterns (SQLAlchemy 2.0 Style)

```python
"""Modern query patterns using SQLAlchemy 2.0 select()."""
from sqlalchemy import func, select
from sqlalchemy.orm import joinedload, selectinload

from .models import Address, User


async def get_users_with_addresses(session: AsyncSession) -> list[User]:
    """Fetch users with their addresses eagerly loaded."""
    stmt = (
        select(User)
        .options(selectinload(User.addresses))
        .where(User.deleted_at.is_(None))
        .order_by(User.created_at.desc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_user_with_profile(
    session: AsyncSession,
    user_id: int,
) -> User | None:
    """Fetch user with profile using joinedload."""
    stmt = (
        select(User)
        .options(joinedload(User.profile))
        .where(User.id == user_id)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def search_users(
    session: AsyncSession,
    search_term: str,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[User], int]:
    """Search users with pagination."""
    base_query = select(User).where(
        User.deleted_at.is_(None),
        User.email.ilike(f"%{search_term}%"),
    )

    # Count query
    count_stmt = select(func.count()).select_from(base_query.subquery())
    total = (await session.execute(count_stmt)).scalar() or 0

    # Data query with pagination
    data_stmt = (
        base_query
        .order_by(User.email)
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(data_stmt)
    users = list(result.scalars().all())

    return users, total


async def bulk_create_users(
    session: AsyncSession,
    users_data: list[dict],
) -> list[User]:
    """Efficiently create multiple users."""
    users = [User(**data) for data in users_data]
    session.add_all(users)
    await session.flush()  # Get IDs
    return users
```

---

## 5. Alembic Migration Standards (MANDATORY)

### A. Project Setup

```bash
# Initialize Alembic
alembic init alembic

# Project structure
project/
├── alembic/
│   ├── versions/           # Migration files
│   ├── env.py             # Migration environment
│   └── script.py.mako     # Migration template
├── alembic.ini            # Alembic configuration
├── src/
│   └── models/
│       ├── __init__.py
│       ├── base.py        # DeclarativeBase
│       └── user.py        # Models
└── pyproject.toml
```

### B. env.py Configuration

```python
"""Alembic environment configuration.

This module configures how Alembic runs migrations, supporting both
online (connected) and offline (SQL generation) modes.
"""
import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from src.models.base import Base
from src.config import settings

# Alembic Config object
config = context.config

# Configure logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set database URL from settings (never hardcode)
config.set_main_option("sqlalchemy.url", settings.database_url)

# Import all models for autogenerate detection
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Generates SQL scripts without connecting to the database.
    Useful for generating migration scripts for DBA review.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with database connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations for async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### C. Migration Template

```python
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

Migration Description:
    [Describe what this migration does and why]

Rollback Risk: [LOW/MEDIUM/HIGH]
    [Describe any data loss or risks during rollback]

Dependencies:
    [List any external dependencies or prerequisites]
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# Revision identifiers
revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    """Apply migration changes.

    IMPORTANT: Consider the following before running:
    - Estimated runtime for large tables
    - Lock implications for concurrent access
    - Index creation strategy (CONCURRENTLY if possible)
    """
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    """Revert migration changes.

    WARNING: Document any data loss that occurs during downgrade.
    """
    ${downgrades if downgrades else "pass"}
```

### D. Migration Best Practices

```python
"""Example migration with best practices."""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "abc123def456"
down_revision: Union[str, None] = "previous_rev"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add user preferences table and migrate existing data."""
    # 1. Create new table
    op.create_table(
        "user_preferences",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("theme", sa.String(50), nullable=False, server_default="light"),
        sa.Column("notifications_enabled", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_user_preferences")),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_user_preferences_user_id_users"),
            ondelete="CASCADE",
        ),
    )

    # 2. Create index CONCURRENTLY for large tables (PostgreSQL)
    # Note: CONCURRENTLY requires autocommit mode
    op.execute("COMMIT")  # End current transaction
    op.create_index(
        "ix_user_preferences_user_id",
        "user_preferences",
        ["user_id"],
        postgresql_concurrently=True,
    )
    op.execute("BEGIN")  # Start new transaction

    # 3. Migrate existing data in batches
    connection = op.get_bind()
    connection.execute(
        sa.text("""
            INSERT INTO user_preferences (user_id, theme, notifications_enabled)
            SELECT id, 'light', true
            FROM users
            WHERE id NOT IN (SELECT user_id FROM user_preferences)
        """)
    )


def downgrade() -> None:
    """Remove user preferences table.

    WARNING: This will delete all user preference data permanently.
    """
    op.drop_index("ix_user_preferences_user_id", table_name="user_preferences")
    op.drop_table("user_preferences")
```

### E. Safe Schema Changes

```python
"""Safe migration patterns for production databases."""
from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    """Demonstrate safe migration patterns."""

    # SAFE: Add nullable column (no table rewrite)
    op.add_column(
        "users",
        sa.Column("phone", sa.String(20), nullable=True),
    )

    # SAFE: Add column with default (PostgreSQL 11+, no table rewrite)
    op.add_column(
        "users",
        sa.Column("is_verified", sa.Boolean(), server_default="false", nullable=False),
    )

    # SAFE: Create index concurrently (doesn't block writes)
    op.execute("COMMIT")
    op.execute("""
        CREATE INDEX CONCURRENTLY ix_users_phone
        ON users (phone)
        WHERE phone IS NOT NULL
    """)
    op.execute("BEGIN")

    # DANGEROUS: Changing column type (may lock table)
    # Use batching for large tables:
    # op.alter_column(
    #     "users",
    #     "status",
    #     type_=sa.String(50),
    #     postgresql_using="status::varchar(50)",
    # )


def safe_add_not_null_constraint() -> None:
    """Safely add NOT NULL constraint to existing column.

    Three-phase approach to avoid blocking:
    1. Add column as nullable
    2. Backfill data
    3. Add NOT NULL constraint
    """
    # Phase 1: Already nullable column exists

    # Phase 2: Backfill in batches
    connection = op.get_bind()
    batch_size = 10000
    while True:
        result = connection.execute(
            sa.text("""
                UPDATE users
                SET phone = 'unknown'
                WHERE phone IS NULL
                AND id IN (
                    SELECT id FROM users
                    WHERE phone IS NULL
                    LIMIT :batch_size
                )
            """),
            {"batch_size": batch_size},
        )
        if result.rowcount == 0:
            break

    # Phase 3: Add constraint (fast since no NULLs exist)
    op.alter_column("users", "phone", nullable=False)
```

### F. SQLite Compatibility with Batch Mode

```python
"""Migration with SQLite batch mode support.

SQLite doesn't support many ALTER TABLE operations. Alembic's batch
mode recreates the table with modifications.
"""
from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    """Modify columns with SQLite compatibility."""
    # Use batch_alter_table for SQLite compatibility
    with op.batch_alter_table("users", schema=None) as batch_op:
        # Add column
        batch_op.add_column(
            sa.Column("middle_name", sa.String(100), nullable=True)
        )

        # Alter column type
        batch_op.alter_column(
            "first_name",
            existing_type=sa.String(50),
            type_=sa.String(100),
        )

        # Drop column
        batch_op.drop_column("legacy_field")

        # Add index
        batch_op.create_index(
            "ix_users_email_status",
            ["email", "status"],
        )


def downgrade() -> None:
    """Revert changes with SQLite compatibility."""
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_index("ix_users_email_status")
        batch_op.add_column(
            sa.Column("legacy_field", sa.String(255), nullable=True)
        )
        batch_op.alter_column(
            "first_name",
            existing_type=sa.String(100),
            type_=sa.String(50),
        )
        batch_op.drop_column("middle_name")
```

---

## 6. Migration Verification Protocol (MANDATORY)

**CRITICAL: Agents MUST verify every migration before presenting it to users.**

### A. Agent Migration Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MANDATORY MIGRATION WORKFLOW                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. GENERATE    →  Create migration with autogenerate or manually   │
│        ↓                                                             │
│  2. REVIEW      →  Inspect generated upgrade() and downgrade()      │
│        ↓                                                             │
│  3. TEST UP     →  Apply migration to test database                 │
│        ↓                                                             │
│  4. VERIFY DATA →  Confirm no data loss occurred                    │
│        ↓                                                             │
│  5. TEST DOWN   →  Rollback migration completely                    │
│        ↓                                                             │
│  6. VERIFY DATA →  Confirm data restored to original state          │
│        ↓                                                             │
│  7. TEST CYCLE  →  Run upgrade → downgrade → upgrade again          │
│        ↓                                                             │
│  8. PRESENT     →  Only then present migration to user              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Agents MUST NOT present migrations that:**
- Have not been tested for both upgrade AND downgrade
- Cause data loss that hasn't been explicitly approved
- Have incomplete or non-functional downgrade() functions
- Fail the upgrade → downgrade → upgrade cycle test

### B. Migration Test Script (MANDATORY)

Agents MUST run this verification script before presenting any migration:

```python
"""Migration verification script.

MANDATORY: Run this script to verify every migration before deployment.
This script tests upgrade, downgrade, and data preservation.
"""
import subprocess
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session


class MigrationVerifier:
    """Verifies migrations are safe and reversible."""

    def __init__(self, alembic_ini: str = "alembic.ini", test_db_url: str | None = None):
        """Initialize verifier with config path.

        Args:
            alembic_ini: Path to alembic.ini file.
            test_db_url: Test database URL. If None, uses SQLite in-memory.
        """
        self.config = Config(alembic_ini)
        self.test_db_url = test_db_url or "sqlite:///test_migration.db"
        self.config.set_main_option("sqlalchemy.url", self.test_db_url)
        self.engine = create_engine(self.test_db_url)

    def get_current_revision(self) -> str | None:
        """Get current database revision."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT version_num FROM alembic_version")
            )
            row = result.fetchone()
            return row[0] if row else None

    def get_table_checksums(self) -> dict[str, int]:
        """Get row counts for all tables (data preservation check)."""
        inspector = inspect(self.engine)
        checksums = {}
        with Session(self.engine) as session:
            for table_name in inspector.get_table_names():
                if table_name != "alembic_version":
                    result = session.execute(
                        text(f"SELECT COUNT(*) FROM {table_name}")
                    )
                    checksums[table_name] = result.scalar() or 0
        return checksums

    def verify_migration(self, revision: str = "head") -> dict:
        """Verify a migration is safe and reversible.

        Args:
            revision: Target revision to test (default: head).

        Returns:
            dict with verification results.
        """
        results = {
            "revision": revision,
            "upgrade_success": False,
            "downgrade_success": False,
            "cycle_success": False,
            "data_preserved": False,
            "errors": [],
        }

        try:
            # Step 1: Get baseline state
            print(f"[1/7] Recording baseline state...")
            baseline_revision = self.get_current_revision()
            baseline_checksums = self.get_table_checksums()

            # Step 2: Test upgrade
            print(f"[2/7] Testing upgrade to {revision}...")
            command.upgrade(self.config, revision)
            results["upgrade_success"] = True
            post_upgrade_checksums = self.get_table_checksums()

            # Step 3: Test downgrade
            print(f"[3/7] Testing downgrade to {baseline_revision or 'base'}...")
            target = baseline_revision or "base"
            command.downgrade(self.config, target)
            results["downgrade_success"] = True

            # Step 4: Verify data preservation after downgrade
            print(f"[4/7] Verifying data preservation...")
            post_downgrade_checksums = self.get_table_checksums()

            # Compare checksums (tables that existed before should have same counts)
            data_issues = []
            for table, count in baseline_checksums.items():
                if table in post_downgrade_checksums:
                    if post_downgrade_checksums[table] != count:
                        data_issues.append(
                            f"{table}: {count} → {post_downgrade_checksums[table]}"
                        )

            if data_issues:
                results["errors"].append(f"Data changed after rollback: {data_issues}")
            else:
                results["data_preserved"] = True

            # Step 5: Test upgrade again (cycle test)
            print(f"[5/7] Testing upgrade cycle...")
            command.upgrade(self.config, revision)

            # Step 6: Final downgrade
            print(f"[6/7] Final downgrade test...")
            command.downgrade(self.config, target)
            results["cycle_success"] = True

            # Step 7: Restore to target state
            print(f"[7/7] Restoring to target state...")
            command.upgrade(self.config, revision)

        except Exception as e:
            results["errors"].append(str(e))

        return results

    def print_report(self, results: dict) -> None:
        """Print verification report."""
        print("\n" + "=" * 60)
        print("MIGRATION VERIFICATION REPORT")
        print("=" * 60)
        print(f"Revision: {results['revision']}")
        print(f"Upgrade:      {'✅ PASS' if results['upgrade_success'] else '❌ FAIL'}")
        print(f"Downgrade:    {'✅ PASS' if results['downgrade_success'] else '❌ FAIL'}")
        print(f"Cycle Test:   {'✅ PASS' if results['cycle_success'] else '❌ FAIL'}")
        print(f"Data Safe:    {'✅ PASS' if results['data_preserved'] else '❌ FAIL'}")

        if results["errors"]:
            print("\nErrors:")
            for error in results["errors"]:
                print(f"  ❌ {error}")

        all_passed = all([
            results["upgrade_success"],
            results["downgrade_success"],
            results["cycle_success"],
            results["data_preserved"],
        ])

        print("\n" + "=" * 60)
        if all_passed:
            print("✅ MIGRATION VERIFIED - Safe to deploy")
        else:
            print("❌ MIGRATION FAILED - Do NOT deploy")
        print("=" * 60)

        return all_passed


def verify_migration(revision: str = "head") -> bool:
    """Convenience function to verify a migration.

    Args:
        revision: Target revision to verify.

    Returns:
        True if migration is safe, False otherwise.
    """
    verifier = MigrationVerifier()
    results = verifier.verify_migration(revision)
    return verifier.print_report(results)


if __name__ == "__main__":
    revision = sys.argv[1] if len(sys.argv) > 1 else "head"
    success = verify_migration(revision)
    sys.exit(0 if success else 1)
```

### C. CI/CD Integration (MANDATORY)

```yaml
# .github/workflows/migration-test.yml
name: Migration Tests

on:
  pull_request:
    paths:
      - 'alembic/versions/**'
      - 'src/models/**'

jobs:
  test-migrations:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test_db
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync

      - name: Test migration upgrade
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/test_db
        run: |
          uv run alembic upgrade head

      - name: Test migration downgrade
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/test_db
        run: |
          uv run alembic downgrade base

      - name: Test migration cycle
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/test_db
        run: |
          uv run alembic upgrade head
          uv run alembic downgrade -1
          uv run alembic upgrade head

      - name: Verify no pending migrations
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/test_db
        run: |
          # Check that models match database after migrations
          uv run alembic check
```

### D. Data Loss Prevention Checklist

**Before ANY migration that modifies or removes data:**

```python
"""Data loss prevention checklist for migrations."""

# ============================================================
# DESTRUCTIVE OPERATION CHECKLIST
# Complete ALL items before deploying migrations that:
# - DROP columns, tables, or constraints
# - ALTER column types that may truncate data
# - DELETE or UPDATE existing data
# ============================================================

DESTRUCTIVE_MIGRATION_CHECKLIST = """
□ 1. BACKUP EXISTS
    - Production backup taken within last 24 hours
    - Backup restoration tested and verified
    - Point-in-time recovery configured (if available)

□ 2. DATA EXPORTED
    - Affected data exported to recoverable format
    - Export verified (row counts match)
    - Export stored in secure location

□ 3. ROLLBACK TESTED
    - downgrade() function implemented
    - downgrade() tested on production-like data
    - Data restoration verified after downgrade

□ 4. STAKEHOLDERS NOTIFIED
    - Data owners informed of permanent deletion
    - Written approval obtained (link to ticket/email)
    - Retention policies verified (legal/compliance)

□ 5. DEPLOYMENT PLAN
    - Maintenance window scheduled
    - Rollback procedure documented
    - Monitoring alerts configured
    - Team members on-call identified
"""


def check_destructive_operations(migration_file: str) -> list[str]:
    """Scan migration for destructive operations.

    Returns list of warnings for destructive operations found.
    """
    destructive_patterns = [
        ("op.drop_table", "TABLE DROP - All data will be permanently deleted"),
        ("op.drop_column", "COLUMN DROP - Column data will be permanently deleted"),
        ("op.drop_constraint", "CONSTRAINT DROP - Data integrity may be affected"),
        ("op.drop_index", "INDEX DROP - Query performance may be affected"),
        ("batch_op.drop_column", "COLUMN DROP (batch) - Column data will be deleted"),
        ("op.execute.*DELETE", "DATA DELETE - Rows will be permanently removed"),
        ("op.execute.*TRUNCATE", "TABLE TRUNCATE - All rows will be removed"),
        ("alter_column.*type_=", "TYPE CHANGE - Data may be truncated or lost"),
        ("nullable=False", "NOT NULL - Existing NULL values must be handled"),
    ]

    warnings = []
    with open(migration_file) as f:
        content = f.read()
        for pattern, warning in destructive_patterns:
            import re
            if re.search(pattern, content, re.IGNORECASE):
                warnings.append(f"⚠️  {warning}")

    return warnings
```

### E. Migration Safety Levels

```python
"""Migration safety classification."""
from enum import Enum


class MigrationSafety(Enum):
    """Safety levels for migration operations."""

    # SAFE: Can be run without downtime, fully reversible
    SAFE = "safe"

    # CAUTION: May cause brief locks, reversible with data
    CAUTION = "caution"

    # DANGEROUS: May cause downtime or data loss, needs approval
    DANGEROUS = "dangerous"

    # IRREVERSIBLE: Cannot be automatically rolled back
    IRREVERSIBLE = "irreversible"


# Operation safety classification
OPERATION_SAFETY = {
    # SAFE operations
    "add_column_nullable": MigrationSafety.SAFE,
    "add_index_concurrently": MigrationSafety.SAFE,
    "create_table": MigrationSafety.SAFE,
    "add_constraint_check": MigrationSafety.SAFE,

    # CAUTION operations
    "add_column_with_default": MigrationSafety.CAUTION,  # PG11+ is safe
    "add_index": MigrationSafety.CAUTION,  # Locks table briefly
    "add_foreign_key": MigrationSafety.CAUTION,  # Validates existing data
    "alter_column_nullable_to_not_null": MigrationSafety.CAUTION,

    # DANGEROUS operations
    "alter_column_type": MigrationSafety.DANGEROUS,  # May lock, may truncate
    "drop_column": MigrationSafety.DANGEROUS,  # Data loss
    "drop_table": MigrationSafety.DANGEROUS,  # Data loss
    "drop_constraint": MigrationSafety.DANGEROUS,  # May allow invalid data

    # IRREVERSIBLE operations
    "data_migration_destructive": MigrationSafety.IRREVERSIBLE,
    "column_rename_with_data_copy": MigrationSafety.IRREVERSIBLE,
}
```

### F. Agent Verification Commands

**Agents MUST run these commands in sequence when creating migrations:**

```bash
# 1. Generate the migration
uv run alembic revision --autogenerate -m "description"

# 2. Review the generated migration file
cat alembic/versions/*_description.py

# 3. Test upgrade on fresh database
uv run alembic upgrade head

# 4. Verify tables created correctly
uv run python -c "
from sqlalchemy import create_engine, inspect
engine = create_engine('sqlite:///test.db')
inspector = inspect(engine)
for table in inspector.get_table_names():
    print(f'Table: {table}')
    for col in inspector.get_columns(table):
        print(f'  - {col[\"name\"]}: {col[\"type\"]}')
"

# 5. Test downgrade
uv run alembic downgrade -1

# 6. Verify downgrade worked
uv run alembic current

# 7. Test full cycle
uv run alembic upgrade head && uv run alembic downgrade base && uv run alembic upgrade head

# 8. Verify final state
uv run alembic current
```

### G. Migration Verification Report Template

When presenting a migration to users, agents MUST include this report:

```markdown
## Migration Verification Report

**Migration**: `add_user_preferences_table`
**Revision**: `abc123def456`
**Created**: 2026-01-15

### Safety Classification
- **Level**: ⚠️ CAUTION
- **Reason**: Adds foreign key constraint (validates existing data)

### Operations
| Operation | Safety | Reversible |
|-----------|--------|------------|
| CREATE TABLE user_preferences | ✅ Safe | ✅ Yes |
| ADD FOREIGN KEY | ⚠️ Caution | ✅ Yes |
| CREATE INDEX | ✅ Safe | ✅ Yes |

### Verification Results
- [x] Upgrade tested successfully
- [x] Downgrade tested successfully
- [x] Upgrade → Downgrade → Upgrade cycle passed
- [x] No data loss detected
- [x] Models match database schema

### Rollback Command
```bash
uv run alembic downgrade abc123def455  # Previous revision
```

### Estimated Impact
- **Lock Duration**: < 1 second (small table)
- **Downtime Required**: No
- **Data at Risk**: None
```

---

## 7. Migration Commands Reference

### A. Common Operations

```bash
# Generate migration from model changes
alembic revision --autogenerate -m "add user preferences table"

# Create empty migration for manual changes
alembic revision -m "custom data migration"

# Apply all pending migrations
alembic upgrade head

# Apply specific number of migrations
alembic upgrade +2

# Rollback last migration
alembic downgrade -1

# Rollback to specific revision
alembic downgrade abc123def456

# Rollback all migrations
alembic downgrade base

# View current revision
alembic current

# View migration history
alembic history --verbose

# Generate SQL without applying (for DBA review)
alembic upgrade head --sql > migration.sql

# Stamp database without running migrations
alembic stamp head
```

### B. Multi-Database Support

```python
"""Multi-database migration configuration."""
# alembic.ini
[alembic]
script_location = alembic
# Remove sqlalchemy.url - we'll set it per database

[alembic:main]
sqlalchemy.url = postgresql://user:pass@localhost/main_db

[alembic:analytics]
sqlalchemy.url = postgresql://user:pass@localhost/analytics_db
```

```bash
# Run migrations for specific database
alembic -n main upgrade head
alembic -n analytics upgrade head
```

---

## 8. Testing Database Code (MANDATORY)

### A. Test Configuration

```python
"""Database testing configuration."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.models.base import Base


@pytest.fixture(scope="session")
def engine():
    """Create test database engine."""
    # Use in-memory SQLite for fast tests
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    """Create test session with automatic rollback."""
    connection = engine.connect()
    transaction = connection.begin()

    session = Session(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def sample_user(session: Session) -> User:
    """Create a sample user for testing."""
    user = User(
        email="test@example.com",
        password_hash="hashed_password",
        first_name="Test",
        last_name="User",
    )
    session.add(user)
    session.flush()
    return user
```

### B. Model Tests

```python
"""User model tests."""
import pytest
from sqlalchemy.exc import IntegrityError

from src.models import User, UserStatus


class TestUserModel:
    """Tests for User model."""

    def test_create_user(self, session):
        """Test creating a new user."""
        user = User(
            email="new@example.com",
            password_hash="hashed",
            first_name="New",
            last_name="User",
        )
        session.add(user)
        session.flush()

        assert user.id is not None
        assert user.status == UserStatus.PENDING
        assert user.created_at is not None

    def test_unique_email_constraint(self, session, sample_user):
        """Test that duplicate emails are rejected."""
        duplicate = User(
            email=sample_user.email,
            password_hash="hashed",
            first_name="Dup",
            last_name="User",
        )
        session.add(duplicate)

        with pytest.raises(IntegrityError):
            session.flush()

    def test_full_name_property(self, sample_user):
        """Test full_name computed property."""
        assert sample_user.full_name == "Test User"

    def test_soft_delete(self, session, sample_user):
        """Test soft delete functionality."""
        from datetime import datetime, timezone

        assert not sample_user.is_deleted

        sample_user.deleted_at = datetime.now(timezone.utc)
        session.flush()

        assert sample_user.is_deleted
```

### C. Migration Tests (MANDATORY)

**CRITICAL: These tests MUST pass before any migration is deployed.**

```python
"""Migration tests - MANDATORY for all migration changes.

These tests verify that:
1. All migrations can be applied (upgrade)
2. All migrations can be reverted (downgrade)
3. The upgrade → downgrade → upgrade cycle works
4. Data is preserved during rollback
5. Models match the final database schema
"""
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session


class TestMigrationIntegrity:
    """Tests for migration integrity and reversibility."""

    @pytest.fixture
    def alembic_config(self, tmp_path):
        """Create Alembic config for testing."""
        db_path = tmp_path / "test.db"
        config = Config("alembic.ini")
        config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
        return config, db_path

    def test_upgrade_to_head(self, alembic_config):
        """Test all migrations can be applied."""
        config, db_path = alembic_config
        # MUST succeed without errors
        command.upgrade(config, "head")

        # Verify we're at head
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            assert result.fetchone() is not None

    def test_downgrade_to_base(self, alembic_config):
        """Test all migrations can be reverted."""
        config, db_path = alembic_config

        # First upgrade
        command.upgrade(config, "head")

        # Then downgrade - MUST succeed
        command.downgrade(config, "base")

        # Verify we're at base (no version)
        engine = create_engine(f"sqlite:///{db_path}")
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        # Only alembic_version should remain (or be empty)
        assert "alembic_version" in tables or len(tables) == 0

    def test_upgrade_downgrade_cycle(self, alembic_config):
        """Test the complete upgrade → downgrade → upgrade cycle."""
        config, db_path = alembic_config

        # Cycle 1: Fresh upgrade
        command.upgrade(config, "head")

        # Cycle 2: Downgrade
        command.downgrade(config, "base")

        # Cycle 3: Re-upgrade - MUST succeed
        command.upgrade(config, "head")

        # Verify final state
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            assert result.fetchone() is not None

    def test_single_head_no_branches(self, alembic_config):
        """Test no migration branches exist."""
        from alembic.script import ScriptDirectory

        config, _ = alembic_config
        script = ScriptDirectory.from_config(config)
        heads = list(script.get_heads())

        assert len(heads) == 1, f"Multiple heads detected: {heads}. Run 'alembic merge' to fix."


class TestDataPreservation:
    """Tests verifying data is preserved during rollback."""

    @pytest.fixture
    def populated_db(self, alembic_config):
        """Create database with test data."""
        config, db_path = alembic_config

        # Apply migrations
        command.upgrade(config, "head")

        # Insert test data
        engine = create_engine(f"sqlite:///{db_path}")
        with Session(engine) as session:
            # Add test records (adjust to your models)
            session.execute(text("""
                INSERT INTO users (email, password_hash, first_name, last_name, status)
                VALUES
                    ('user1@test.com', 'hash1', 'User', 'One', 'active'),
                    ('user2@test.com', 'hash2', 'User', 'Two', 'active'),
                    ('user3@test.com', 'hash3', 'User', 'Three', 'pending')
            """))
            session.commit()

        return config, db_path, engine

    def test_data_preserved_after_rollback(self, populated_db):
        """Test that data survives upgrade → downgrade → upgrade cycle."""
        config, db_path, engine = populated_db

        # Record initial state
        with Session(engine) as session:
            initial_count = session.execute(
                text("SELECT COUNT(*) FROM users")
            ).scalar()
            initial_emails = set(
                row[0] for row in session.execute(text("SELECT email FROM users"))
            )

        # Downgrade one migration
        command.downgrade(config, "-1")

        # Re-upgrade
        command.upgrade(config, "head")

        # Verify data preserved
        with Session(engine) as session:
            final_count = session.execute(
                text("SELECT COUNT(*) FROM users")
            ).scalar()
            final_emails = set(
                row[0] for row in session.execute(text("SELECT email FROM users"))
            )

        assert final_count == initial_count, (
            f"Data loss detected: {initial_count} → {final_count} rows"
        )
        assert final_emails == initial_emails, (
            f"Data changed: missing {initial_emails - final_emails}"
        )

    def test_rollback_does_not_corrupt_data(self, populated_db):
        """Test that rollback doesn't corrupt existing data."""
        config, db_path, engine = populated_db

        # Record detailed initial state
        with Session(engine) as session:
            initial_data = list(session.execute(text("""
                SELECT email, first_name, last_name, status
                FROM users ORDER BY email
            """)))

        # Full downgrade to base
        command.downgrade(config, "base")

        # Full upgrade back to head
        command.upgrade(config, "head")

        # Re-insert data (simulating what would happen in production)
        # Note: In real scenario, data would be backed up before destructive migration
        with Session(engine) as session:
            for email, first_name, last_name, status in initial_data:
                session.execute(text("""
                    INSERT OR IGNORE INTO users (email, password_hash, first_name, last_name, status)
                    VALUES (:email, 'hash', :first_name, :last_name, :status)
                """), {
                    "email": email,
                    "first_name": first_name,
                    "last_name": last_name,
                    "status": status,
                })
            session.commit()


class TestMigrationSafety:
    """Tests for migration safety checks."""

    def test_no_destructive_operations_without_flag(self):
        """Scan migrations for destructive operations."""
        from pathlib import Path
        import re

        migrations_dir = Path("alembic/versions")
        if not migrations_dir.exists():
            pytest.skip("No migrations directory")

        destructive_patterns = [
            (r"op\.drop_table", "DROP TABLE"),
            (r"op\.drop_column", "DROP COLUMN"),
            (r"\.execute\([^)]*DELETE", "DELETE statement"),
            (r"\.execute\([^)]*TRUNCATE", "TRUNCATE statement"),
        ]

        issues = []
        for migration_file in migrations_dir.glob("*.py"):
            content = migration_file.read_text()

            # Skip if explicitly marked as reviewed
            if "# DESTRUCTIVE_REVIEWED:" in content:
                continue

            for pattern, desc in destructive_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    issues.append(f"{migration_file.name}: {desc}")

        if issues:
            pytest.fail(
                f"Destructive operations found without review flag:\n"
                + "\n".join(f"  - {issue}" for issue in issues)
                + "\n\nAdd '# DESTRUCTIVE_REVIEWED: <reason>' to acknowledge."
            )

    def test_all_downgrades_implemented(self):
        """Verify all migrations have downgrade implementations."""
        from pathlib import Path
        import ast

        migrations_dir = Path("alembic/versions")
        if not migrations_dir.exists():
            pytest.skip("No migrations directory")

        empty_downgrades = []
        for migration_file in migrations_dir.glob("*.py"):
            content = migration_file.read_text()
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "downgrade":
                    # Check if body is just 'pass'
                    if (
                        len(node.body) == 1
                        and isinstance(node.body[0], ast.Pass)
                    ):
                        # Check for explicit approval comment
                        if "# DOWNGRADE_NOT_NEEDED:" not in content:
                            empty_downgrades.append(migration_file.name)

        if empty_downgrades:
            pytest.fail(
                f"Migrations with empty downgrade() functions:\n"
                + "\n".join(f"  - {f}" for f in empty_downgrades)
                + "\n\nEither implement downgrade or add "
                "'# DOWNGRADE_NOT_NEEDED: <reason>' comment."
            )
```

### D. Running Migration Tests

```bash
# Run all migration tests
uv run pytest tests/test_migrations.py -v

# Run with coverage
uv run pytest tests/test_migrations.py --cov=alembic --cov-report=term-missing

# Run specific test class
uv run pytest tests/test_migrations.py::TestDataPreservation -v

# Run before every PR that touches migrations
uv run pytest tests/test_migrations.py -v --tb=short
```

---

## 9. Security Best Practices (MANDATORY)

### A. SQL Injection Prevention

```python
"""Secure query patterns."""
from sqlalchemy import select, text

# ✅ CORRECT: Parameterized queries
async def get_user_by_id(session: AsyncSession, user_id: int) -> User | None:
    """Safe: uses ORM with bound parameters."""
    stmt = select(User).where(User.id == user_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


# ✅ CORRECT: Raw SQL with bound parameters
async def search_users_raw(
    session: AsyncSession,
    search_term: str,
) -> list[User]:
    """Safe: uses bound parameters with raw SQL."""
    stmt = text("""
        SELECT * FROM users
        WHERE email LIKE :search
        AND deleted_at IS NULL
    """)
    result = await session.execute(stmt, {"search": f"%{search_term}%"})
    return result.fetchall()


# ❌ WRONG: String concatenation (SQL INJECTION VULNERABILITY)
# NEVER do this:
# query = f"SELECT * FROM users WHERE email = '{email}'"
# session.execute(text(query))


# ❌ WRONG: String formatting in queries
# NEVER do this:
# stmt = text("SELECT * FROM users WHERE id = %s" % user_id)
```

### B. Connection String Security

```python
"""Secure database configuration."""
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration from environment."""

    # Load from environment variables
    db_host: str
    db_port: int = 5432
    db_name: str
    db_user: str
    db_password: str  # NEVER hardcode

    # SSL/TLS settings
    db_ssl_mode: str = "require"
    db_ssl_ca: str | None = None

    @property
    def database_url(self) -> str:
        """Build connection URL securely."""
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
            f"?ssl={self.db_ssl_mode}"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # NEVER commit .env files to version control
```

### C. Principle of Least Privilege

```sql
-- Create application user with minimal privileges
CREATE USER app_user WITH PASSWORD 'secure_password';

-- Grant only necessary permissions
GRANT CONNECT ON DATABASE myapp TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO app_user;

-- For migrations, use a separate user with more privileges
CREATE USER migration_user WITH PASSWORD 'different_secure_password';
GRANT ALL PRIVILEGES ON DATABASE myapp TO migration_user;

-- Revoke dangerous permissions
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
```

---

## 10. Performance Optimization

### A. Query Optimization

```python
"""Performance-optimized query patterns."""
from sqlalchemy import select
from sqlalchemy.orm import selectinload, joinedload, load_only


async def get_users_optimized(
    session: AsyncSession,
    page: int,
    per_page: int,
) -> list[User]:
    """Optimized user query with selective loading."""
    stmt = (
        select(User)
        # Load only needed columns
        .options(load_only(User.id, User.email, User.first_name))
        # Eager load relationships in separate query (N+1 prevention)
        .options(selectinload(User.addresses).load_only(Address.city))
        # Filter before loading
        .where(User.deleted_at.is_(None))
        # Keyset pagination (faster than OFFSET for large datasets)
        .order_by(User.id)
        .limit(per_page)
    )

    if page > 1:
        # Assume we have the last ID from previous page
        stmt = stmt.where(User.id > last_seen_id)

    result = await session.execute(stmt)
    return list(result.scalars().all())


async def bulk_update_status(
    session: AsyncSession,
    user_ids: list[int],
    new_status: UserStatus,
) -> int:
    """Efficient bulk update."""
    from sqlalchemy import update

    stmt = (
        update(User)
        .where(User.id.in_(user_ids))
        .values(status=new_status)
    )
    result = await session.execute(stmt)
    return result.rowcount
```

### B. Connection Pool Tuning

```python
"""Production-ready connection pool configuration."""
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine(
    database_url,
    # Pool size based on: (2 * num_cpu_cores) + spindle_count
    # For SSD: typically 10-20 connections per app instance
    pool_size=10,

    # Allow temporary overflow during traffic spikes
    max_overflow=20,

    # Wait time before giving up on getting connection
    pool_timeout=30,

    # Recycle connections to avoid stale connections
    # Set to less than DB's wait_timeout
    pool_recycle=1800,

    # Verify connection health before use
    # Small overhead but prevents errors from dead connections
    pool_pre_ping=True,

    # Echo SQL for debugging (disable in production)
    echo=False,
)
```

---

## 11. Scalability Patterns

### A. Read Replicas

```python
"""Read replica routing."""
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


class RoutingSession(Session):
    """Session that routes reads to replica."""

    def get_bind(self, mapper=None, clause=None, **kwargs):
        """Route SELECT queries to read replica."""
        if self._flushing or self._is_modifying:
            return engines["primary"]
        return engines["replica"]


engines = {
    "primary": create_engine("postgresql://primary-host/db"),
    "replica": create_engine("postgresql://replica-host/db"),
}
```

### B. Sharding Pattern

```python
"""Database sharding by tenant."""
from sqlalchemy import event
from sqlalchemy.orm import Session


def get_engine_for_tenant(tenant_id: str):
    """Get database engine for specific tenant."""
    shard = hash(tenant_id) % NUM_SHARDS
    return shard_engines[shard]


@event.listens_for(Session, "after_begin")
def set_search_path(session, transaction, connection):
    """Set schema for multi-tenant isolation."""
    tenant_id = get_current_tenant()
    if tenant_id:
        connection.execute(f"SET search_path TO tenant_{tenant_id}")
```

---

## 12. Monitoring and Observability

### A. Query Logging

```python
"""Query performance monitoring."""
import logging
import time

from sqlalchemy import event
from sqlalchemy.engine import Engine

logger = logging.getLogger("sqlalchemy.performance")


@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Record query start time."""
    conn.info["query_start_time"] = time.time()


@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Log slow queries."""
    total_time = time.time() - conn.info.get("query_start_time", 0)

    if total_time > 0.5:  # Log queries taking more than 500ms
        logger.warning(
            "Slow query detected",
            extra={
                "duration_ms": total_time * 1000,
                "statement": statement[:500],  # Truncate long queries
            },
        )
```

### B. Health Checks

```python
"""Database health check endpoint."""
from sqlalchemy import text


async def check_database_health(session: AsyncSession) -> dict:
    """Check database connectivity and performance."""
    try:
        start = time.time()
        await session.execute(text("SELECT 1"))
        latency_ms = (time.time() - start) * 1000

        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 2),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
```

---

## 13. Common Patterns Reference

### A. Soft Delete

```python
"""Soft delete pattern with automatic filtering."""
from sqlalchemy import event
from sqlalchemy.orm import Query


@event.listens_for(Query, "before_compile", retval=True)
def filter_soft_deleted(query):
    """Automatically exclude soft-deleted records."""
    for desc in query.column_descriptions:
        entity = desc.get("entity")
        if entity and hasattr(entity, "deleted_at"):
            query = query.filter(entity.deleted_at.is_(None))
    return query
```

### B. Audit Trail

```python
"""Audit trail mixin for tracking changes."""
from sqlalchemy import event
from sqlalchemy.orm import Session


class AuditMixin:
    """Mixin that tracks who created/modified records."""

    created_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
    )
    updated_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
    )


@event.listens_for(Session, "before_flush")
def set_audit_fields(session, flush_context, instances):
    """Automatically set audit fields."""
    current_user_id = get_current_user_id()

    for obj in session.new:
        if hasattr(obj, "created_by"):
            obj.created_by = current_user_id
            obj.updated_by = current_user_id

    for obj in session.dirty:
        if hasattr(obj, "updated_by"):
            obj.updated_by = current_user_id
```

### C. Optimistic Locking

```python
"""Optimistic locking with version column."""
from sqlalchemy.orm import Mapped, mapped_column


class VersionedMixin:
    """Mixin for optimistic locking."""

    version: Mapped[int] = mapped_column(default=1)


class Order(Base, VersionedMixin):
    """Order with optimistic locking."""

    __tablename__ = "orders"
    __mapper_args__ = {"version_id_col": version}

    id: Mapped[int] = mapped_column(primary_key=True)
    total: Mapped[float]


# Usage - will raise StaleDataError on conflict
async def update_order_total(session: AsyncSession, order_id: int, new_total: float):
    """Update order with optimistic locking."""
    order = await session.get(Order, order_id)
    order.total = new_total
    # If another transaction modified this row, commit will fail
    await session.commit()
```

---

## 14. Troubleshooting Guide

### A. Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `DetachedInstanceError` | Accessing lazy attribute after session closed | Use eager loading or keep session open |
| `ObjectDeletedError` | Accessing deleted object | Check `deleted_at` before access |
| `StaleDataError` | Optimistic lock conflict | Retry with fresh data |
| `IntegrityError: duplicate key` | Unique constraint violation | Check before insert or handle exception |
| `TimeoutError: QueuePool limit` | Connection pool exhausted | Increase pool_size or fix connection leaks |
| Migration "Multiple heads" | Parallel development | Merge heads with `alembic merge` |

### B. Debug Mode

```python
"""Enable SQL debugging."""
import logging

# Log all SQL statements
logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)

# Log connection pool events
logging.getLogger("sqlalchemy.pool").setLevel(logging.DEBUG)

# Or per-engine
engine = create_engine(url, echo=True, echo_pool=True)
```

---

## 15. Quick Reference

### A. Migration Commands

```bash
alembic revision --autogenerate -m "message"  # Create migration
alembic upgrade head                          # Apply all
alembic downgrade -1                          # Rollback one
alembic current                               # Show current
alembic history                               # Show history
alembic upgrade head --sql                    # Generate SQL
```

### B. Session Lifecycle

```python
# Sync
with Session(engine) as session:
    with session.begin():
        session.add(obj)
        # auto-commit on success, rollback on exception

# Async
async with AsyncSession(engine) as session:
    async with session.begin():
        session.add(obj)
```

### C. Query Patterns

```python
# Select with filter
stmt = select(User).where(User.email == email)

# Eager loading
stmt = select(User).options(selectinload(User.addresses))

# Pagination
stmt = select(User).limit(20).offset(40)

# Aggregation
stmt = select(func.count(User.id)).where(User.status == "active")
```

---

## Cross-References

- **SQL fundamentals**: See `sql.md` for query writing standards
- **PostgreSQL specifics**: See `postgresql.md` for PostgreSQL features
- **Python standards**: See `python.md` for code style requirements
- **Security practices**: See `secure-coding.md` for security guidelines
- **Testing practices**: See `testing.md` for test writing standards
