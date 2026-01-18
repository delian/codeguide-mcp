# Modern Lua Development Guidelines

This document provides mandatory coding standards and development practices for modern Lua applications with emphasis on minimalistic, clean, readable, well-documented code using hexagonal architecture with focus on performance, portability, and maintainability.

---

**Agent Profile**: The Lua Architect  
**Role**: Senior Lua Engineer & Scripting Specialist  
**Objective**: Generate production-ready, minimalistic, clean, readable, well-documented Lua code using hexagonal architecture with focus on performance, portability, scalability, and maintainability.  
**Tools**: Lua 5.4+, LuaRocks, LDoc, Busted, LuaUnit, luacheck, LuaJIT (when applicable).

---

## 1. Core Philosophies: MODERN-LUA

The agent must adhere to the **MODERN-LUA** standard for every Lua implementation:

- **Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory)
- **Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression
- **M**inimalistic Code: Clean, concise, readable Lua code
- **O**ptimized Performance: Local variables, table pre-allocation, efficient algorithms
- **D**ocumentation as Code: API documentation auto-generatable from code
- **E**rror Handling: Explicit error handling with pcall/xpcall, no silent failures
- **R**eusable Modules: Modular design, clear interfaces, separation of concerns
- **N**ative Features: Leverage Lua's tables, coroutines, metatables effectively

- **L**ocal Variables: Always use local for performance and scope
- **U**nit Testing: Comprehensive tests, mandatory for all code
- **A**rchitectural: Hexagonal architecture, clear separation
- **T**ype Safety: Runtime validation, clear contracts
- **E**fficient Execution: Performance-optimized, minimal allocations
- **S**tandard Patterns: Follow Lua idioms and best practices

**V**erified Scripts: Agent-generated code MUST parse, execute, and pass tests before delivery
- **E**xplicit Dependencies: Clear dependency management, version pinning
- **R**obust Error Handling: pcall/xpcall, proper error messages
- **I**mmutable Patterns: Prefer immutable data where possible
- **F**unctional Style: Pure functions, minimal side effects
- **I**dempotent Operations: Safe to retry, no side effects
- **E**fficient Execution: Performance-optimized, minimal memory usage

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Script Verification Protocol

**CRITICAL: Agents MUST ALWAYS verify that all generated/modified Lua code parses correctly, executes without breaking, and passes all tests. Verification is MANDATORY for every code change.**

#### Verification Checklist

**Before delivering ANY Lua code, the agent MUST:**

1. **Syntax Verification (MANDATORY - ALWAYS REQUIRED)**:
   **CRITICAL: Code MUST parse successfully. This is non-negotiable.**
   ```bash
   # Check Lua syntax
   lua -l script.lua
   # Exit code MUST be 0
   
   # Check with luac (compiler)
   luac -p script.lua
   # Exit code MUST be 0
   
   # Run luacheck if available
   if command -v luacheck >/dev/null 2>&1; then
       luacheck script.lua
       # Exit code MUST be 0
   fi
   ```
   - **MUST** parse without errors (exit code 0)
   - No syntax errors or warnings
   - All modules loadable

2. **Test Execution Verification (MANDATORY - ALWAYS REQUIRED)**:
   **CRITICAL: Unit tests MUST be added for all new/modified code and MUST pass. This is non-negotiable.**
   ```bash
   # Run tests with Busted
   busted test/
   # Exit code MUST be 0
   
   # OR with LuaUnit
   lua test/test_suite.lua
   # Exit code MUST be 0
   
   # Run with coverage if available
   busted --coverage test/
   ```
   - **MUST** pass all tests (exit code 0)
   - **MANDATORY**: Unit tests MUST be added for all new code
   - **MANDATORY**: All unit tests MUST pass before code delivery
   - Minimum 80% code coverage for business logic
   - No flaky tests (run multiple times to verify)
   - **After ANY code change**: Re-run tests to verify they still pass

3. **Code Quality Verification**:
   ```bash
   # Run luacheck if available
   if command -v luacheck >/dev/null 2>&1; then
       luacheck --config .luacheckrc src/
       # Exit code MUST be 0
   fi
   ```
   - **MUST** pass static analysis if luacheck is available
   - No linter warnings

4. **Documentation Generation**:
   ```bash
   # Generate API documentation with LDoc
   ldoc src/
   # Exit code MUST be 0
   
   # Verify documentation
   ls doc/
   ```
   - **MUST** generate without errors
   - All public APIs documented
   - No missing documentation warnings

5. **Post-Modification Verification (MANDATORY)**:
   ```bash
   # After ANY modification, ALWAYS run:
   # 1. Syntax check
   luac -p script.lua
   # Exit code MUST be 0
   
   # 2. Static analysis (if available)
   command -v luacheck >/dev/null 2>&1 && luacheck script.lua
   # Exit code MUST be 0
   
   # 3. Run tests
   busted test/
   # Exit code MUST be 0
   
   # 4. Generate docs
   ldoc src/
   # Exit code MUST be 0
   ```

### B. Error Correction Process

If verification fails:

1. **Read the error message** - syntax errors, test failures, linter issues
2. **Identify the root cause** - missing local, incorrect syntax, test logic issue, missing documentation
3. **Fix the issue** in the generated code
4. **Re-run verification** until all checks pass
5. **Document fixes** in comments if non-obvious
6. **Only present working, tested code** to the user

**CRITICAL**: Never provide Lua code to the user that doesn't parse or pass tests. Always verify first, fix issues, then present the working solution.

**MANDATORY RULES:**
1. **Syntax check is ALWAYS required** - Code MUST parse successfully
2. **Unit tests are ALWAYS required** - All new/modified code MUST have unit tests
3. **Tests MUST pass** - All unit tests MUST pass before code delivery
4. **Re-verify after changes** - After ANY code modification, re-check syntax and re-run tests
5. **TDD is MANDATORY** - Write tests BEFORE implementation (Red-Green-Refactor)
6. **Bug regression tests MANDATORY** - Every bug MUST get a test BEFORE fixing

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new Lua code.**

### TDD Cycle for Lua

```
1. 🔴 RED: Write a failing test first
   ↓
2. 🟢 GREEN: Write minimal code to make it pass
   ↓
3. 🔵 REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for Lua Function

```lua
-- Step 1: RED - Write failing test first
-- test/email_validator_spec.lua
local validator = require("email_validator")

describe("email validator", function()
    -- Test will fail - module doesn't exist yet
    it("accepts valid email addresses", function()
        assert.is_true(validator.is_valid("user@example.com"))
        assert.is_true(validator.is_valid("test.user@domain.co.uk"))
    end)

    it("rejects invalid email addresses", function()
        assert.is_false(validator.is_valid("invalid"))
        assert.is_false(validator.is_valid("user@"))
        assert.is_false(validator.is_valid("@domain.com"))
    end)

    it("rejects empty strings", function()
        assert.is_false(validator.is_valid(""))
        assert.is_false(validator.is_valid(nil))
    end)
end)

-- Run: busted test/
-- ❌ FAILS - email_validator module doesn't exist yet

-- Step 2: GREEN - Write minimal implementation
-- src/email_validator.lua
--- Validates email address formats.
-- @module email_validator

local M = {}

--- Validates an email address format.
-- @param email the email address to validate
-- @return true if the email is valid, false otherwise
-- @usage
-- local validator = require("email_validator")
-- if validator.is_valid("user@example.com") then
--     print("Valid email")
-- end
function M.is_valid(email)
    if not email or email == "" then
        return false
    end
    return email:match("^[^%s@]+@[^%s@]+%.[^%s@]+$") ~= nil
end

return M

-- Run: busted test/
-- ✅ PASSES - tests pass

-- Step 3: REFACTOR - Improve with more robust validation
--- Validates email address formats according to RFC 5322.
--
-- Performs comprehensive email validation including:
-- - Basic format check (user@domain.tld)
-- - Length constraints (3-254 characters)
-- - RFC 5322 compliant pattern
--
-- @module email_validator

local M = {}

local EMAIL_PATTERN = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+%.[a-zA-Z][a-zA-Z]+$"
local MIN_LENGTH = 3
local MAX_LENGTH = 254

--- Validates an email address format.
-- @param email the email address to validate
-- @return true if the email is valid, false otherwise
-- @see https://tools.ietf.org/html/rfc5322
function M.is_valid(email)
    if not email or type(email) ~= "string" then
        return false
    end

    if #email < MIN_LENGTH or #email > MAX_LENGTH then
        return false
    end

    return email:match(EMAIL_PATTERN) ~= nil
end

return M
-- Tests still pass ✓
```

### Example TDD for Lua Module

```lua
-- Step 1: RED - Write failing test first
-- test/user_spec.lua
local User = require("user")

describe("User", function()
    -- Test will fail - User module doesn't exist yet
    it("creates user with valid data", function()
        local user = User.new("user-123", "John Doe", "john@example.com")
        
        assert.equals("user-123", user.id)
        assert.equals("John Doe", user.name)
        assert.equals("john@example.com", user.email)
    end)

    it("throws on invalid email", function()
        assert.has_error(function()
            User.new("user-123", "John", "invalid-email")
        end, "Invalid email format")
    end)
end)

-- Run: busted test/
-- ❌ FAILS - User module doesn't exist yet

-- Step 2: GREEN - Write minimal implementation
-- src/user.lua
--- User data model.
-- @module user

local M = {}

--- Creates a new user.
-- @param id the unique user identifier
-- @param name the user's full name
-- @param email the user's email address
-- @return a new user table
function M.new(id, name, email)
    if not email:match("^[^%s@]+@[^%s@]+%.[^%s@]+$") then
        error("Invalid email format: " .. email)
    end

    return {
        id = id,
        name = name,
        email = email
    }
end

return M

-- Run: busted test/
-- ✅ PASSES - tests pass

-- Step 3: REFACTOR - Add validation and methods
--- User data model with validation.
--
-- Represents an immutable user in the system.
-- Enforces validation rules:
-- - ID must not be empty
-- - Name must not be empty
-- - Email must be valid format
--
-- @module user

local M = {}

local EMAIL_PATTERN = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+%.[a-zA-Z][a-zA-Z]+$"

--- Creates a new user.
-- @param id the unique user identifier (non-empty)
-- @param name the user's full name (non-empty)
-- @param email the user's email address (valid format)
-- @return a new user table
-- @raise error if validation fails
function M.new(id, name, email)
    assert(id and id ~= "", "id cannot be empty")
    assert(name and name ~= "", "name cannot be empty")
    assert(email and email:match(EMAIL_PATTERN), "Invalid email format: " .. tostring(email))

    local self = {
        id = id,
        name = name,
        email = email
    }

    --- Creates a copy of this user with updated name.
    -- @param new_name the new name
    -- @return a new user table with the updated name
    function self.with_name(new_name)
        return M.new(id, new_name, email)
    end

    return self
end

return M
-- Tests still pass ✓
```

---

## 2B. Bug Fix Protocol for Lua (MANDATORY)

**CRITICAL: Every Lua bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow for Lua

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
6. 📝 Document the bug in test comments (include bug ID)
   ↓
7. 🚀 Deploy with confidence (regression prevented)
```

### Example Bug Fix: Nil Handling

```lua
-- Bug Report #5431: get_user_name crashes when user is nil

-- Step 1-2: Write test that reproduces the bug
-- test/user_service_spec.lua
local UserService = require("user_service")

describe("UserService", function()
    --- Bug #5431: get_user_name crashes when user is nil.
    -- Discovered: 2026-01-18
    -- This test prevents regression.
    it("get_user_name returns nil when user is nil - Bug #5431", function()
        local service = UserService.new()
        
        -- Should return nil, not crash
        local result = service.get_user_name(nil)
        assert.is_nil(result)
    end)

    it("get_user_name returns name when user exists", function()
        local service = UserService.new()
        local user = { id = "123", name = "John Doe", email = "john@example.com" }
        
        local result = service.get_user_name(user)
        assert.equals("John Doe", result)
    end)
end)

-- Run: busted test/
-- ❌ FAILS - Crashes with "attempt to index a nil value"

-- Step 3: Fix the bug
-- src/user_service.lua
--- Service for user-related operations.
-- @module user_service

local M = {}

function M.new()
    local self = {}

    --- Gets the user's name.
    --
    -- Bug Fix #5431: Now properly handles nil users by returning
    -- nil instead of crashing.
    --
    -- @param user the user (may be nil)
    -- @return the user's name, or nil if user is nil
    function self.get_user_name(user)
        -- FIX: Check for nil before accessing user
        if not user then
            return nil
        end
        return user.name
    end

    return self
end

return M

-- Run: busted test/
-- ✅ PASSES - bug fixed, regression prevented ✓
```

### Example Bug Fix: Table Modification During Iteration

```lua
-- Bug Report #5432: remove_inactive_users crashes with "invalid key to 'next'"

-- Step 1-2: Write test that reproduces the bug
-- test/user_manager_spec.lua
local UserManager = require("user_manager")

describe("UserManager", function()
    --- Bug #5432: remove_inactive_users crashes during iteration.
    -- Discovered: 2026-01-18
    -- This test prevents regression.
    it("remove_inactive_users does not crash - Bug #5432", function()
        local manager = UserManager.new()
        
        -- Add multiple users
        manager.add_user({ id = "1", name = "John", active = false })
        manager.add_user({ id = "2", name = "Jane", active = true })
        manager.add_user({ id = "3", name = "Bob", active = false })
        
        -- Should not crash
        assert.has_no_errors(function()
            manager.remove_inactive_users()
        end)
        
        -- Should only have active users left
        assert.equals(1, manager.get_user_count())
    end)
end)

-- Run: busted test/
-- ❌ FAILS - Crashes with "invalid key to 'next'"

-- Step 3: Fix the bug
-- src/user_manager.lua
--- Manages a collection of users.
-- @module user_manager

local M = {}

function M.new()
    local self = {}
    local users = {}

    function self.add_user(user)
        table.insert(users, user)
    end

    --- Removes all inactive users from the collection.
    --
    -- Bug Fix #5432: Now safely removes users during iteration
    -- by building a new table instead of modifying during iteration.
    function self.remove_inactive_users()
        -- FIX: Build new table instead of modifying during iteration
        -- OLD (buggy) code:
        -- for i, user in ipairs(users) do
        --     if not user.active then
        --         table.remove(users, i)  -- Crash!
        --     end
        -- end
        
        -- NEW (fixed) code:
        local active_users = {}
        for _, user in ipairs(users) do
            if user.active then
                table.insert(active_users, user)
            end
        end
        users = active_users
    end

    function self.get_user_count()
        return #users
    end

    return self
end

return M

-- Run: busted test/
-- ✅ PASSES - bug fixed, regression prevented ✓
```

### Prohibited Practices for Lua Bug Fixes

**NEVER:**
- ❌ Fix a bug without adding a regression test first
- ❌ Write implementation before writing tests (violates TDD)
- ❌ Skip the Red-Green-Refactor cycle
- ❌ Commit code with failing tests
- ❌ Remove tests to make code pass
- ❌ Use `pending()` to ignore failing tests
- ❌ Suppress luacheck warnings instead of fixing root cause

**ALWAYS:**
- ✅ Write a test that reproduces the bug first
- ✅ Verify the test fails before fixing
- ✅ Document bug ID in test comments
- ✅ Run `busted test/` after fix
- ✅ Ensure fix doesn't introduce new issues
- ✅ Keep tests in codebase permanently
- ✅ Test with both Lua and LuaJIT if applicable

---

## 3. Dependency Management (MANDATORY)

### A. LuaRocks Best Practices

**CRITICAL: Use LuaRocks for dependency management. Pin versions for reproducibility.**

#### ✅ CORRECT - Proper rockspec Configuration

```lua
-- myproject-1.0.0-1.rockspec - Proper dependency management

package = "myproject"
version = "1.0.0-1"

dependencies = {
   "lua >= 5.4",
   "busted >= 2.0.0",
   "ldoc >= 1.4.0",
   "luacheck >= 1.0.0",
}

build = {
   type = "builtin",
   modules = {
      myproject = "src/myproject/init.lua",
      ["myproject.utils"] = "src/myproject/utils.lua",
      ["myproject.core"] = "src/myproject/core.lua",
   },
}
```

#### ✅ CORRECT - Using Dependencies

```lua
-- Install dependencies
-- luarocks install --only-deps myproject-1.0.0-1.rockspec

-- Use dependencies
local busted = require("busted")
local ldoc = require("ldoc")
```

#### ❌ WRONG - Manual Dependency Management

```lua
-- ❌ Manual file copying
-- ❌ No version control
-- ❌ No dependency resolution
```

### B. Version Pinning

**CRITICAL: Always pin major versions, allow patch updates.**

```lua
-- ✅ CORRECT - Version constraints
dependencies = {
   "busted >= 2.0.0, < 3.0.0",  -- Allow 2.x, not 3.0.0
   "ldoc >= 1.4.0, < 2.0.0",    -- Allow 1.4.x, not 2.0.0
}

-- ❌ WRONG - Too permissive
dependencies = {
   "busted",  -- ❌ Could break with major updates
}
```

---

## 4. Hexagonal Architecture (MANDATORY)

### A. Architecture Principles

**CRITICAL: All Lua applications MUST follow hexagonal architecture (ports and adapters) for clean separation of concerns, testability, and maintainability.**

#### ✅ CORRECT - Hexagonal Architecture Structure

```
src/
├── main.lua                    # App entry point
├── core/                       # Core utilities
│   ├── constants.lua
│   ├── utils.lua
│   └── errors.lua
├── features/                   # Feature modules (hexagonal)
│   ├── auth/
│   │   ├── domain/            # Domain layer (core)
│   │   │   ├── entities/     # Domain models
│   │   │   ├── repositories/  # Repository interfaces (ports)
│   │   │   └── usecases/     # Business logic
│   │   ├── data/             # Data layer (adapters)
│   │   │   ├── datasources/  # External data sources
│   │   │   └── repositories/ # Repository implementations
│   │   └── presentation/      # Presentation layer (adapters)
│   │       ├── controllers/   # Controllers
│   │       └── views/         # Views
│   ├── game/
│   └── player/
└── shared/                     # Shared components
    ├── modules/
    └── utils/
```

### B. Domain Layer (Core)

**CRITICAL: Domain layer contains business logic and is independent of frameworks.**

#### ✅ CORRECT - Domain Entity

```lua
-- features/auth/domain/entities/user.lua - Domain entity

--- Represents a user in the system.
--
-- This is a pure domain entity with no framework dependencies.
-- It contains only business logic and data.
--
-- @classmod User
local User = {}
User.__index = User

--- Creates a new user instance.
--
-- @param id User ID (must be non-empty string)
-- @param email User email (must be valid format)
-- @param name User name (optional but recommended)
-- @return User instance
-- @usage
-- local user = User.new("123", "user@example.com", "John Doe")
function User.new(id, email, name)
   assert(type(id) == "string" and #id > 0, "User ID must be non-empty string")
   assert(type(email) == "string" and email:match("@"), "Invalid email format")
   
   local self = setmetatable({}, User)
   self.id = id
   self.email = email
   self.name = name or ""
   self.created_at = os.time()
   self.updated_at = os.time()
   return self
end

--- Updates user fields.
--
-- @param updates Table with fields to update
-- @return Updated user instance
function User:update(updates)
   if updates.name then
      self.name = updates.name
   end
   if updates.email then
      assert(updates.email:match("@"), "Invalid email format")
      self.email = updates.email
   end
   self.updated_at = os.time()
   return self
end

--- Returns string representation of user.
--
-- @return String representation
function User:__tostring()
   return string.format("User(id=%s, email=%s, name=%s)",
                        self.id, self.email, self.name)
end

return User
```

#### ✅ CORRECT - Repository Interface (Port)

```lua
-- features/auth/domain/repositories/user_repository.lua - Repository port

--- Repository interface for user operations.
--
-- This defines the contract for user data operations.
-- Implementations are in the data layer.
--
-- @classmod UserRepository
local UserRepository = {}

--- Gets a user by ID.
--
-- @param userId User ID
-- @return User if found, nil otherwise
-- @raise RepositoryException if operation fails
-- @usage
-- local user = repository:getUserById("123")
function UserRepository:getUserById(userId)
   error("Not implemented - must be implemented by concrete class")
end

--- Gets the current authenticated user.
--
-- @return User if authenticated, nil otherwise
-- @raise RepositoryException if operation fails
function UserRepository:getCurrentUser()
   error("Not implemented - must be implemented by concrete class")
end

--- Updates user profile.
--
-- @param userId User ID to update
-- @param updates Table with fields to update
-- @return Updated user
-- @raise RepositoryException if operation fails
function UserRepository:updateUser(userId, updates)
   error("Not implemented - must be implemented by concrete class")
end

--- Signs out the current user.
--
-- @raise RepositoryException if operation fails
function UserRepository:signOut()
   error("Not implemented - must be implemented by concrete class")
end

return UserRepository
```

### C. Data Layer (Adapters)

**CRITICAL: Data layer implements domain interfaces and handles external data sources.**

#### ✅ CORRECT - Repository Implementation

```lua
-- features/auth/data/repositories/user_repository_impl.lua - Repository adapter

local UserRepository = require("features.auth.domain.repositories.user_repository")
local User = require("features.auth.domain.entities.user")

--- Implementation of UserRepository using database.
--
-- @classmod UserRepositoryImpl
local UserRepositoryImpl = {}
UserRepositoryImpl.__index = UserRepositoryImpl
setmetatable(UserRepositoryImpl, {__index = UserRepository})

--- Creates a new repository instance.
--
-- @param db Database connection
-- @return Repository instance
function UserRepositoryImpl.new(db)
   local self = setmetatable({}, UserRepositoryImpl)
   self.db = db
   return self
end

--- Gets a user by ID.
--
-- @param userId User ID
-- @return User if found, nil otherwise
-- @raise RepositoryException if operation fails
function UserRepositoryImpl:getUserById(userId)
   local success, result = pcall(function()
      local row = self.db:query("SELECT * FROM users WHERE id = ?", userId)
      if row then
         return User.new(row.id, row.email, row.name)
      end
      return nil
   end)
   
   if not success then
      error("RepositoryException: Failed to get user: " .. tostring(result))
   end
   
   return result
end

--- Gets the current authenticated user.
--
-- @return User if authenticated, nil otherwise
function UserRepositoryImpl:getCurrentUser()
   local session = self.db:getCurrentSession()
   if not session or not session.user_id then
      return nil
   end
   
   return self:getUserById(session.user_id)
end

--- Updates user profile.
--
-- @param userId User ID to update
-- @param updates Table with fields to update
-- @return Updated user
function UserRepositoryImpl:updateUser(userId, updates)
   local success, result = pcall(function()
      self.db:update("users", updates, {id = userId})
      return self:getUserById(userId)
   end)
   
   if not success then
      error("RepositoryException: Failed to update user: " .. tostring(result))
   end
   
   return result
end

--- Signs out the current user.
function UserRepositoryImpl:signOut()
   local success, err = pcall(function()
      self.db:clearSession()
   end)
   
   if not success then
      error("RepositoryException: Failed to sign out: " .. tostring(err))
   end
end

return UserRepositoryImpl
```

---

## 5. Code Style and Best Practices (MANDATORY)

### A. Naming Conventions

**CRITICAL: Use snake_case for variables and functions, PascalCase for modules/classes.**

#### ✅ CORRECT - Proper Naming

```lua
-- Variables and functions: snake_case
local player_health = 100
local function calculate_damage(base_damage, armor)
   return base_damage - armor
end

-- Modules/classes: PascalCase
local GameEngine = {}
local PlayerManager = require("modules.PlayerManager")

-- Constants: UPPERCASE
local MAX_PLAYERS = 4
local DEFAULT_SPEED = 200

-- Private functions/variables: prefix with underscore
local function _internal_helper()
   -- Private implementation
end

local _private_cache = {}
```

#### ❌ WRONG - Inconsistent Naming

```lua
-- ❌ Mixed naming conventions
local playerHealth = 100        -- ❌ Should be snake_case
local function CalculateDamage() -- ❌ Should be snake_case
local max_players = 4           -- ❌ Should be UPPERCASE
```

### B. Local Variables

**CRITICAL: Always use local variables for performance and proper scoping.**

#### ✅ CORRECT - Local Variables

```lua
-- Always use local
local math = math
local string = string
local table = table

-- Local functions
local function helper_function()
   -- Implementation
end

-- Local modules
local Utils = require("utils")
```

#### ❌ WRONG - Global Variables

```lua
-- ❌ Global variables (slow, pollutes global namespace)
function helper_function()  -- ❌ Global function
   -- Implementation
end

-- ❌ Accessing globals directly (slower)
result = math.max(a, b)  -- ❌ Should cache math locally
```

### C. Module Pattern

**CRITICAL: Use proper module pattern with return statement.**

#### ✅ CORRECT - Module Pattern

```lua
-- modules/player.lua - Proper module structure

local Player = {}
Player.__index = Player

-- Module constants
local DEFAULT_HEALTH = 100
local MAX_LEVEL = 50

-- Private functions
local function _validate_level(level)
   return level >= 1 and level <= MAX_LEVEL
end

-- Constructor
function Player.new(name, level)
   assert(type(name) == "string", "Player name must be a string")
   assert(_validate_level(level), "Invalid player level")
   
   local self = setmetatable({}, Player)
   self.name = name
   self.level = level
   self.health = DEFAULT_HEALTH
   self.experience = 0
   return self
end

-- Public methods
function Player:take_damage(damage)
   assert(type(damage) == "number" and damage >= 0, "Invalid damage value")
   self.health = math.max(0, self.health - damage)
   return self.health <= 0
end

-- Metamethods
function Player:__tostring()
   return string.format("Player(%s, Level %d, HP: %d)",
                        self.name, self.level, self.health)
end

return Player
```

---

## 6. Error Handling (MANDATORY)

### A. pcall and xpcall

**CRITICAL: Always use pcall/xpcall for error handling in production code.**

#### ✅ CORRECT - Proper Error Handling

```lua
-- Safe function execution with pcall
local function safe_divide(a, b)
   local success, result = pcall(function()
      if b == 0 then
         error("Division by zero", 2)
      end
      return a / b
   end)
   
   if success then
      return result
   else
      print("Error:", result)
      return nil
   end
end

-- Enhanced error handling with xpcall
local function enhanced_error_handler(err)
   local trace = debug.traceback(err, 2)
   print("Error occurred:", trace)
   -- Log to file or send to error reporting service
   return err
end

local function risky_operation(data)
   local success, result = xpcall(function()
      -- Complex operation that might fail
      assert(type(data) == "table", "Data must be a table")
      assert(data.value, "Data must have a value field")
      
      return data.value * 2
   end, enhanced_error_handler)
   
   return success and result or nil
end
```

### B. Input Validation

**CRITICAL: Always validate input with clear error messages.**

#### ✅ CORRECT - Input Validation

```lua
-- Input validation utility
local Validator = {}

function Validator.is_positive_number(value)
   return type(value) == "number" and value > 0
end

function Validator.is_valid_string(value, min_length, max_length)
   if type(value) ~= "string") then return false end
   local len = #value
   return len >= (min_length or 1) and len <= (max_length or math.huge)
end

function Validator.is_in_range(value, min_val, max_val)
   return type(value) == "number" and value >= min_val and value <= max_val
end

-- Usage
local function process_user_input(input)
   assert(Validator.is_valid_string(input.name, 1, 100),
          "Name must be between 1 and 100 characters")
   assert(Validator.is_positive_number(input.age),
          "Age must be a positive number")
   -- Process input...
end
```

---

## 7. Performance Optimization (MANDATORY)

### A. Local Variable Caching

**CRITICAL: Cache frequently used globals as local variables.**

#### ✅ CORRECT - Local Caching

```lua
-- Cache globals locally
local math = math
local string = string
local table = table
local pairs = pairs
local ipairs = ipairs

-- Use cached versions
local function calculate_stats(numbers)
   local sum = 0
   for i, num in ipairs(numbers) do
      sum = sum + num
   end
   return sum / #numbers
end
```

### B. Table Pre-allocation

**CRITICAL: Pre-allocate tables when size is known.**

#### ✅ CORRECT - Table Pre-allocation

```lua
-- Pre-allocate table
local function create_large_table(size)
   local result = {}
   for i = 1, size do
      result[i] = 0  -- Pre-allocate
   end
   return result
end

-- Use table.concat for string concatenation
local function build_string(parts)
   return table.concat(parts, "")
end
```

### C. Object Pooling

**CRITICAL: Use object pooling for frequently created/destroyed objects.**

#### ✅ CORRECT - Object Pooling

```lua
-- Object pool implementation
local ObjectPool = {}
ObjectPool.__index = ObjectPool

function ObjectPool.new(create_fn, reset_fn)
   local self = setmetatable({}, ObjectPool)
   self.pool = {}
   self.create_fn = create_fn
   self.reset_fn = reset_fn
   return self
end

function ObjectPool:acquire()
   if #self.pool > 0 then
      return table.remove(self.pool)
   else
      return self.create_fn()
   end
end

function ObjectPool:release(obj)
   if self.reset_fn then
      self.reset_fn(obj)
   end
   table.insert(self.pool, obj)
end
```

---

## 8. Testing Requirements (MANDATORY)

### A. Unit Testing (MANDATORY - ALWAYS REQUIRED)

**CRITICAL: All new/modified code MUST have unit tests. Unit tests MUST pass before code delivery. This is non-negotiable.**

**MANDATORY RULES:**
1. **Unit tests are ALWAYS required** for all new code
2. **Unit tests are ALWAYS required** for all modified code
3. **All unit tests MUST pass** before code delivery
4. **After ANY code change**, re-run tests to verify they still pass
5. **Minimum 80% code coverage** for business logic

#### ✅ CORRECT - Busted Tests

```lua
-- test/features/auth/domain/entities/user_test.lua - Unit tests

local busted = require("busted")
local User = require("features.auth.domain.entities.user")

describe("User", function()
   describe("new", function()
      it("creates user with required fields", function()
         local user = User.new("123", "test@example.com", "Test User")
         
         assert.are.equal("123", user.id)
         assert.are.equal("test@example.com", user.email)
         assert.are.equal("Test User", user.name)
      end)
      
      it("validates email format", function()
         assert.has_error(function()
            User.new("123", "invalid-email", "Test")
         end, "Invalid email format")
      end)
   end)
   
   describe("update", function()
      it("updates user fields", function()
         local user = User.new("123", "test@example.com", "Test")
         user:update({name = "Updated"})
         
         assert.are.equal("Updated", user.name)
      end)
   end)
end)
```

---

## 9. Documentation as Code (MANDATORY)

### A. LDoc Documentation Comments

**CRITICAL: All public APIs MUST have complete LDoc documentation comments for auto-generated API documentation.**

#### ✅ CORRECT - Complete LDoc Documentation

```lua
--- Repository interface for user operations.
--
-- This defines the contract for user data operations.
-- Implementations are in the data layer.
--
-- @classmod UserRepository
-- @usage
-- local repository = UserRepositoryImpl.new(db)
-- local user = repository:getUserById("123")
local UserRepository = {}

--- Gets a user by ID.
--
-- @param userId User ID (string)
-- @return User if found, nil otherwise
-- @raise RepositoryException if operation fails
-- @usage
-- local user = repository:getUserById("123")
-- if user then
--    print("User:", user.name)
-- end
function UserRepository:getUserById(userId)
   error("Not implemented")
end

--- Updates user profile.
--
-- @param userId User ID to update (string)
-- @param updates Table with fields to update
-- @return Updated user
-- @raise RepositoryException if operation fails
-- @usage
-- local updated = repository:updateUser("123", {name = "New Name"})
function UserRepository:updateUser(userId, updates)
   error("Not implemented")
end

return UserRepository
```

### B. Generating Documentation

**CRITICAL: Documentation MUST be generatable from code using LDoc.**

```bash
# Generate API documentation
ldoc src/

# Documentation will be in doc/
# View at doc/index.html
```

#### ✅ CORRECT - LDoc Configuration

```lua
-- config.ld - LDoc configuration

project = "MyProject"
title = "MyProject API Documentation"
description = "Modern Lua application with hexagonal architecture"
format = "markdown"
dir = "doc"
file = {
   "src",
}
```

---

## 10. Coroutines (MANDATORY when applicable)

### A. Coroutine Usage

**CRITICAL: Use coroutines for cooperative multitasking and async operations.**

#### ✅ CORRECT - Coroutine Pattern

```lua
-- Coroutine-based task scheduler
local TaskScheduler = {}

function TaskScheduler.new()
   local self = {
      tasks = {},
   }
   return self
end

function TaskScheduler:add_task(task_fn)
   local co = coroutine.create(task_fn)
   table.insert(self.tasks, co)
end

function TaskScheduler:run()
   while #self.tasks > 0 do
      for i = #self.tasks, 1, -1 do
         local co = self.tasks[i]
         local status = coroutine.status(co)
         
         if status == "dead" then
            table.remove(self.tasks, i)
         else
            local success, err = coroutine.resume(co)
            if not success then
               print("Task error:", err)
               table.remove(self.tasks, i)
            end
         end
      end
   end
end

-- Usage
local scheduler = TaskScheduler.new()
scheduler:add_task(function()
   for i = 1, 10 do
      print("Task 1:", i)
      coroutine.yield()
   end
end)
scheduler:run()
```

---

## 11. Memory Management (MANDATORY)

### A. Garbage Collection

**CRITICAL: Manage memory efficiently with proper garbage collection.**

#### ✅ CORRECT - Memory Management

```lua
-- Memory management utilities
local MemoryManager = {}

function MemoryManager.force_cleanup()
   collectgarbage("collect")
end

function MemoryManager.get_memory_usage()
   return collectgarbage("count")
end

function MemoryManager.set_gc_params(pause, stepmul)
   collectgarbage("setpause", pause or 100)
   collectgarbage("setstepmul", stepmul or 200)
end

-- Weak table for caches
function MemoryManager.create_weak_cache(mode)
   local cache = {}
   setmetatable(cache, {__mode = mode or "v"})  -- "k", "v", or "kv"
   return cache
end
```

---

## 12. Summary

**CRITICAL Requirements for All Lua Code:**

1. **Dependency Management**: Use LuaRocks, pin versions for reproducibility
2. **Syntax Verification**: Code MUST ALWAYS parse (mandatory for every change)
3. **Unit Tests**: ALWAYS required for all new/modified code, MUST pass
4. **Hexagonal Architecture**: All applications MUST follow ports and adapters pattern
5. **Local Variables**: Always use local for performance and scope
6. **Error Handling**: Use pcall/xpcall, explicit error messages
7. **Documentation**: Complete LDoc documentation, auto-generatable
8. **Testing**: 80%+ code coverage, comprehensive unit tests, always required
9. **Performance**: Local caching, table pre-allocation, object pooling
10. **Code Style**: snake_case for functions, PascalCase for modules
11. **Module Pattern**: Proper module structure with return statement
12. **Memory Management**: Efficient garbage collection, weak tables
13. **Minimalistic Code**: Clean, readable, concise code
14. **Verification**: Agent MUST parse, test, and generate docs before delivery

**Agent Verification Protocol:**
- **MANDATORY**: Parse code (`luac -p script.lua`) - ALWAYS required
- **MANDATORY**: Run unit tests (`busted test/`) - ALWAYS required, MUST pass
- Generate documentation (`ldoc src/`)
- **MANDATORY**: After ANY modification, re-parse and re-run tests
- Only present working, tested, documented code to the user

**Remember**: Minimalistic, clean, readable, well-documented, performant Lua code with hexagonal architecture, proper error handling, comprehensive testing, and focus on portability and speed. Keep it simple, keep it Lua, keep it working.
