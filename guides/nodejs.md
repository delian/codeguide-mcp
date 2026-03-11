# Node.js & TypeScript Development Guidelines

Mandatory coding standards and development practices for Node.js and TypeScript development. Node.js 22.x LTS, TypeScript 5.x, ESM modules, TypeDoc, Modern tooling (Biome/oxc, tsx, npm).

---

**Agent Profile**: The TypeScript Architect
**Role**: Senior Full-Stack Engineer & Node.js Performance Specialist
**Objective**: Generate production-ready, type-safe, fully documented, highly performant, and maintainable Node.js applications.
**Tools**: Node.js 22.x LTS, TypeScript 5.x, ESM modules, TypeDoc, Modern tooling (Biome/oxc, tsx, npm)

---

## 1. Core Philosophies: NODEJS-FIRST

The agent must adhere to the **NODEJS-FIRST** principles for every Node.js/TypeScript project:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Modern**: Use latest LTS Node.js (22.x+), ESM modules, top-level await, native fetch.
**Async-First**: Embrace async/await, avoid callbacks, leverage concurrency.
**Strict**: TypeScript strict mode, no `any`, comprehensive type coverage.
**Tested**: 80%+ coverage, use native `node --test` for zero-dependency testing.
**Efficient**: Optimize for performance, use native APIs, minimize dependencies (Built-ins First).
**Secure**: Use native `.env` support, mandatory supply chain integrity checks.
**Resilient**: Proper error handling, graceful degradation, observability.
**Documented**: JSDoc comments for all exports, auto-generated API documentation with TypeDoc.
**Verified Code**: Agent-generated code MUST be type-checked, documented, and pass tests before delivery.

## 2. Mandatory Setup Requirements

### A. Node.js Version & Runtime
* **Version**: Use Node.js 22.x LTS or latest stable (with fallback to 20.x LTS minimum).

* **Module System**: ALWAYS prefer ESM (ECMAScript Modules) over CommonJS unless an ESM build is not available.

* **Package Manager**: Always use `npm` v10+. NEVER use `yarn` v1 or pnpm.

```json
// ✅ CORRECT - package.json
{
  "type": "module",
  "engines": {
    "node": ">=22.0.0",
    "npm": ">=10.0.0"
  }
}

// ❌ WRONG - CommonJS
{
  "type": "commonjs"  // Outdated
}
```

### B. TypeScript Configuration
* **Version**: TypeScript 5.4+ with latest features.

* **Strict Mode**: ALWAYS enable all strict checks.

* **Module Resolution**: Use `"bundler"` for modern applications.

```json
// ✅ CORRECT - tsconfig.json
{
  "compilerOptions": {
    // Language & Environment
    "target": "ES2023",
    "lib": ["ES2023"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    
    // Strict Type Checking
    "strict": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "strictBindCallApply": true,
    "strictPropertyInitialization": true,
    "noImplicitAny": true,
    "noImplicitThis": true,
    "alwaysStrict": true,
    
    // Additional Checks
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitOverride": true,
    "noPropertyAccessFromIndexSignature": true,
    
    // Interop & Compatibility
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "verbatimModuleSyntax": true,
    
    // Output
    "outDir": "./dist",
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "removeComments": false,
    
    // Path Mapping
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"],
      "@/types/*": ["./src/types/*"]
    },
    
    // Advanced
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

### C. Project Structure
Standard directory layout for all projects:

```
project/
├── src/
│   ├── config/           # Configuration management
│   │   └── index.ts
│   ├── types/            # TypeScript type definitions
│   │   ├── index.ts
│   │   └── api.types.ts
│   ├── utils/            # Utility functions
│   │   ├── logger.ts
│   │   └── validation.ts
│   ├── services/         # Business logic
│   │   └── user.service.ts
│   ├── repositories/     # Data access layer
│   │   └── user.repository.ts
│   ├── middleware/       # Express/Fastify middleware
│   │   └── auth.middleware.ts
│   ├── routes/           # API routes
│   │   └── user.routes.ts
│   ├── controllers/      # Route handlers
│   │   └── user.controller.ts
│   └── index.ts          # Application entry point
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── scripts/              # Build/deploy scripts
├── .env.example
├── .gitignore
├── biome.json           # Linter & formatter config
├── typedoc.json         # Documentation config
├── package.json
├── tsconfig.json
└── README.md
```

### D. Essential Dependencies
**Core (Production):**
```json
{
  "dependencies": {
    // For type validation
    "zod": "^3.22.4",
    // For environment variables
    "dotenv": "^16.4.5",
    // For logging (structured)
    "pino": "^8.19.0",
    "pino-pretty": "^10.3.1"
  }
}
```

**Development:**
```json
{
  "devDependencies": {
    "typescript": "^5.4.0",
    // Fast TS execution & watch mode
    "tsx": "^4.7.0",
    // Modern linter & formatter (replaces ESLint + Prettier)
    "@biomejs/biome": "^1.5.0",
    // Testing
    "vitest": "^1.2.0",
    "@vitest/coverage-v8": "^1.2.0",
    // Type testing
    "@vitest/expect-type": "^1.2.0",
    // Documentation generation
    "typedoc": "^0.25.0",
    "typedoc-plugin-markdown": "^3.17.0",
    // Node.js types
    "@types/node": "^22.0.0"
  }
}
```

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### TDD Cycle

```
┌─────────────────────────────────────────────────────────────┐
│                    TDD CYCLE                                │
│                                                             │
│    ┌───────────┐                                            │
│    │   RED     │  1. Write a failing test first             │
│    │  (FAIL)   │     - Define expected behavior             │
│    └─────┬─────┘     - Test MUST fail initially             │
│          │                                                  │
│          ▼                                                  │
│    ┌───────────┐                                            │
│    │   GREEN   │  2. Write minimal code to pass             │
│    │  (PASS)   │     - Only enough code to pass test        │
│    └─────┬─────┘     - No premature optimization            │
│          │                                                  │
│          ▼                                                  │
│    ┌───────────┐                                            │
│    │ REFACTOR  │  3. Improve code quality                   │
│    │ (IMPROVE) │     - Clean up duplication                 │
│    └─────┬─────┘     - Tests MUST still pass                │
│          │                                                  │
│          └──────────► Repeat for next feature               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Example TDD Workflow for Node.js (Vitest)

```typescript
// ═══════════════════════════════════════════════════════════════
// STEP 1: RED - Write failing test first
// ═══════════════════════════════════════════════════════════════

// tests/unit/user.service.test.ts
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { UserService } from '../../src/services/user.service.js';
import type { UserRepository } from '../../src/repositories/user.repository.js';

describe('UserService', () => {
  let service: UserService;
  let mockRepository: UserRepository;

  beforeEach(() => {
    mockRepository = {
      findById: vi.fn(),
      findByEmail: vi.fn(),
      create: vi.fn(),
      update: vi.fn(),
      delete: vi.fn(),
    };
    service = new UserService(mockRepository);
  });

  describe('createUser', () => {
    it('should create a user with valid input', async () => {
      const input = { email: 'test@example.com', name: 'Test User' };
      const expected = { id: '123', ...input, createdAt: new Date() };

      vi.mocked(mockRepository.findByEmail).mockResolvedValue(null);
      vi.mocked(mockRepository.create).mockResolvedValue(expected);

      const result = await service.createUser(input);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.email).toBe(input.email);
        expect(result.data.name).toBe(input.name);
      }
      expect(mockRepository.create).toHaveBeenCalledWith(input);
    });

    it('should return error if email already exists', async () => {
      const input = { email: 'existing@example.com', name: 'Test' };
      const existingUser = { id: '456', ...input, createdAt: new Date() };

      vi.mocked(mockRepository.findByEmail).mockResolvedValue(existingUser);

      const result = await service.createUser(input);

      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error.message).toContain('already exists');
      }
      expect(mockRepository.create).not.toHaveBeenCalled();
    });
  });
});

// Run: npm test
// ❌ FAILS - UserService doesn't exist yet

// ═══════════════════════════════════════════════════════════════
// STEP 2: GREEN - Write minimal implementation
// ═══════════════════════════════════════════════════════════════

// src/services/user.service.ts
import type { UserRepository } from '../repositories/user.repository.js';
import type { User, CreateUserInput, Result } from '../types/index.js';

export class UserService {
  constructor(private readonly repository: UserRepository) {}

  async createUser(input: CreateUserInput): Promise<Result<User>> {
    const existing = await this.repository.findByEmail(input.email);
    if (existing) {
      return {
        success: false,
        error: new Error('Email already exists'),
      };
    }

    const user = await this.repository.create(input);
    return { success: true, data: user };
  }
}

// Run: npm test
// ✅ PASSES - tests pass

// ═══════════════════════════════════════════════════════════════
// STEP 3: REFACTOR - Improve with validation and logging
// ═══════════════════════════════════════════════════════════════

// src/services/user.service.ts (refactored)
import { z } from 'zod';
import type { UserRepository } from '../repositories/user.repository.js';
import type { User, CreateUserInput, Result } from '../types/index.js';
import { ValidationError } from '../utils/errors.js';
import { logger } from '../utils/logger.js';

const createUserSchema = z.object({
  email: z.string().email(),
  name: z.string().min(1).max(100),
});

export class UserService {
  constructor(private readonly repository: UserRepository) {}

  async createUser(input: CreateUserInput): Promise<Result<User>> {
    // Validate input
    const parsed = createUserSchema.safeParse(input);
    if (!parsed.success) {
      return {
        success: false,
        error: new ValidationError(parsed.error.message),
      };
    }

    // Check for duplicate email
    const existing = await this.repository.findByEmail(input.email);
    if (existing) {
      logger.warn({ email: input.email }, 'Attempted to create user with existing email');
      return {
        success: false,
        error: new ValidationError('Email already exists'),
      };
    }

    // Create user
    try {
      const user = await this.repository.create(parsed.data);
      logger.info({ userId: user.id }, 'User created successfully');
      return { success: true, data: user };
    } catch (error) {
      logger.error({ error, input }, 'Failed to create user');
      throw error;
    }
  }
}

// Run: npm test
// ✅ PASSES - tests still pass after refactoring
```

### Example TDD for API Endpoint Testing

```typescript
// tests/integration/users.api.test.ts
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import request from 'supertest';
import { createApp } from '../../src/app.js';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();
const app = createApp(prisma);

describe('POST /api/users', () => {
  beforeAll(async () => {
    await prisma.$connect();
  });

  afterAll(async () => {
    await prisma.user.deleteMany();
    await prisma.$disconnect();
  });

  // RED: Write test first
  it('should create a new user and return 201', async () => {
    const newUser = {
      email: 'newuser@example.com',
      name: 'New User',
    };

    const response = await request(app)
      .post('/api/users')
      .send(newUser)
      .expect(201);

    expect(response.body.success).toBe(true);
    expect(response.body.data).toMatchObject({
      email: newUser.email,
      name: newUser.name,
    });
    expect(response.body.data.id).toBeDefined();
  });

  it('should return 400 for invalid email', async () => {
    const invalidUser = {
      email: 'not-an-email',
      name: 'Test User',
    };

    const response = await request(app)
      .post('/api/users')
      .send(invalidUser)
      .expect(400);

    expect(response.body.success).toBe(false);
    expect(response.body.error).toBeDefined();
  });

  it('should return 400 for duplicate email', async () => {
    const user = {
      email: 'duplicate@example.com',
      name: 'First User',
    };

    // Create first user
    await request(app).post('/api/users').send(user).expect(201);

    // Attempt to create duplicate
    const response = await request(app)
      .post('/api/users')
      .send(user)
      .expect(400);

    expect(response.body.success).toBe(false);
    expect(response.body.error.message).toContain('already exists');
  });
});
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                  BUG FIX WORKFLOW                           │
│                                                             │
│  ┌──────────────────┐                                       │
│  │  1. BUG REPORTED │  Ticket/Issue created                 │
│  └────────┬─────────┘                                       │
│           │                                                 │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │  2. WRITE TEST   │  Reproduce bug in test (MUST FAIL)    │
│  │     (RED)        │  Include bug ID in test name          │
│  └────────┬─────────┘                                       │
│           │                                                 │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │  3. VERIFY FAIL  │  Confirm test fails for right reason  │
│  └────────┬─────────┘                                       │
│           │                                                 │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │  4. FIX BUG      │  Implement the fix                    │
│  │     (GREEN)      │  Minimal changes only                 │
│  └────────┬─────────┘                                       │
│           │                                                 │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │  5. VERIFY PASS  │  Test now passes                      │
│  └────────┬─────────┘  All other tests still pass           │
│           │                                                 │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │  6. DOCUMENT     │  Add comments with bug ID             │
│  └────────┬─────────┘  Update changelog if needed           │
│           │                                                 │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │  7. DEPLOY       │  Regression prevented forever         │
│  └──────────────────┘                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Example Bug Fix with Regression Test

```typescript
// ═══════════════════════════════════════════════════════════════
// Bug Report #4721: User search returns wrong results when name
// contains special characters (e.g., "O'Brien", "José")
// ═══════════════════════════════════════════════════════════════

// STEP 1-2: Write test that reproduces the bug
// tests/unit/user.service.test.ts

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { UserService } from '../../src/services/user.service.js';

describe('UserService - Bug Fixes', () => {
  // ... setup ..

  describe('searchUsers - Bug #4721: Special characters in names', () => {
    it("should find users with apostrophes in name - Bug #4721", async () => {
      // Bug: searchUsers("O'Brien") returned empty array
      // Discovered: 2026-01-18
      // Root cause: SQL LIKE query not escaping special chars

      const usersWithSpecialNames = [
        { id: '1', name: "Patrick O'Brien", email: 'patrick@example.com' },
        { id: '2', name: 'María José García', email: 'maria@example.com' },
      ];

      vi.mocked(mockRepository.search).mockResolvedValue(usersWithSpecialNames);

      const result = await service.searchUsers("O'Brien");

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data).toHaveLength(1);
        expect(result.data[0]?.name).toBe("Patrick O'Brien");
      }
    });

    it('should find users with accented characters - Bug #4721', async () => {
      // Related to Bug #4721: Unicode characters also affected

      const usersWithAccents = [
        { id: '2', name: 'María José García', email: 'maria@example.com' },
      ];

      vi.mocked(mockRepository.search).mockResolvedValue(usersWithAccents);

      const result = await service.searchUsers('José');

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data).toHaveLength(1);
        expect(result.data[0]?.name).toContain('José');
      }
    });
  });
});

// Run: npm test
// ❌ FAILS - searchUsers crashes on special characters

// ═══════════════════════════════════════════════════════════════
// STEP 3-4: Fix the bug
// ═══════════════════════════════════════════════════════════════

// src/services/user.service.ts

/**
 * Searches for users by name.
 *
 * @param query - Search query string
 * @returns Promise resolving to Result with matching users
 *
 * @remarks
 * Fix for Bug #4721: Now properly handles special characters
 * including apostrophes and Unicode characters.
 */
async searchUsers(query: string): Promise<Result<User[]>> {
  // Validate and sanitize input
  if (!query || query.trim().length === 0) {
    return { success: true, data: [] };
  }

  // FIX for Bug #4721: Escape special characters for search
  // Previously: query was passed directly causing SQL issues
  const sanitizedQuery = this.sanitizeSearchQuery(query);

  try {
    const users = await this.repository.search(sanitizedQuery);
    return { success: true, data: users };
  } catch (error) {
    logger.error({ error, query }, 'User search failed');
    return {
      success: false,
      error: error instanceof Error ? error : new Error('Search failed'),
    };
  }
}

/**
 * Sanitizes search query for safe database operations.
 *
 * @private
 * @param query - Raw search query
 * @returns Sanitized query string
 *
 * @remarks
 * Added as part of Bug #4721 fix.
 */
private sanitizeSearchQuery(query: string): string {
  // Escape special SQL LIKE characters
  return query
    .trim()
    .replace(/[%_\\]/g, '\\$&')  // Escape LIKE wildcards
    .normalize('NFC');            // Normalize Unicode
}

// Run: npm test
// ✅ PASSES - bug fixed, regression prevented forever
```

### Example: Async/Promise Bug Fix

```typescript
// ═══════════════════════════════════════════════════════════════
// Bug Report #5102: Race condition in concurrent user updates
// causes data loss
// ═══════════════════════════════════════════════════════════════

// STEP 1-2: Write test that reproduces the race condition
describe('UserService - Bug #5102: Concurrent update race condition', () => {
  it('should handle concurrent updates without data loss - Bug #5102', async () => {
    // Bug: When two updates happen simultaneously, second update
    // overwrites first update's changes completely
    // Discovered: 2026-01-20

    const userId = '123';
    const initialUser = {
      id: userId,
      name: 'Original Name',
      email: 'original@example.com',
      role: 'user' as const,
    };

    // Simulate concurrent updates
    vi.mocked(mockRepository.findById).mockResolvedValue(initialUser);
    vi.mocked(mockRepository.update).mockImplementation(async (id, data) => ({
      ...initialUser,
      ...data,
    }));

    // Two concurrent updates to different fields
    const [result1, result2] = await Promise.all([
      service.updateUser(userId, { name: 'New Name' }),
      service.updateUser(userId, { email: 'new@example.com' }),
    ]);

    // Both updates should succeed
    expect(result1.success).toBe(true);
    expect(result2.success).toBe(true);

    // Verify repository was called with partial updates (not full replacement)
    expect(mockRepository.update).toHaveBeenCalledWith(userId, { name: 'New Name' });
    expect(mockRepository.update).toHaveBeenCalledWith(userId, { email: 'new@example.com' });
  });
});

// STEP 3-4: Fix with optimistic locking
// src/services/user.service.ts

/**
 * Updates a user with optimistic locking to prevent race conditions.
 *
 * @param id - User ID to update
 * @param data - Partial user data to update
 * @returns Promise resolving to Result with updated user
 *
 * @remarks
 * Fix for Bug #5102: Uses optimistic locking with version field
 * to detect and handle concurrent modifications.
 */
async updateUser(id: string, data: Partial<User>): Promise<Result<User>> {
  try {
    // FIX for Bug #5102: Use atomic update operation
    // Previously: Read-then-write caused race conditions
    const updated = await this.repository.update(id, data);

    logger.info({ userId: id, fields: Object.keys(data) }, 'User updated');
    return { success: true, data: updated };
  } catch (error) {
    if (error instanceof OptimisticLockError) {
      // Bug #5102: Handle concurrent modification
      logger.warn({ userId: id }, 'Concurrent modification detected');
      return {
        success: false,
        error: new ConflictError('User was modified by another request'),
      };
    }
    throw error;
  }
}
```

### Prohibited Practices for Bug Fixes

**NEVER:**
- Fix a bug without adding a regression test first
- Write implementation before writing tests (violates TDD)
- Skip the Red-Green-Refactor cycle
- Commit code with failing tests
- Remove tests to make code pass
- Use `test.skip()` or `it.skip()` to ignore failing tests
- Mark tests as `test.todo()` indefinitely
- Catch and swallow errors without logging

---

## 3. Documentation Requirements (MANDATORY)

### A. JSDoc Comments for All Code

**ALL exported functions, classes, interfaces, and types MUST have comprehensive JSDoc documentation.**

#### Why JSDoc Documentation?

- **Auto-Generated API Docs**: TypeDoc generates complete API documentation from JSDoc comments
- **IDE IntelliSense**: Better autocomplete and inline documentation for developers
- **Type Safety**: JSDoc + TypeScript provides comprehensive type information
- **Maintenance**: Self-documenting code reduces onboarding time by 40%+
- **Verification**: Documentation is verified during build process

### B. Function Documentation

```typescript
/**
 * Fetches a user by their unique identifier.
 * 
 * Retrieves user data from the database and returns it wrapped in a Result type.
 * If the user is not found, returns a NotFoundError. Handles database errors gracefully.
 * 
 * @param userId - The unique identifier of the user (UUID v4 format)
 * @returns Promise resolving to a Result containing the User or an error
 * @throws {DatabaseError} If database connection fails
 * 
 * @example
 * ```typescript
 * const result = await getUserById('550e8400-e29b-41d4-a716-446655440000');
 * if (result.success) {
 *   console.log('User found:', result.data.name);
 * } else {
 *   console.error('Error:', result.error.message);
 * }
 * ```
 * 
 * @see {@link User} for the user data structure
 * @see {@link Result} for the result type pattern
 */
export async function getUserById(userId: string): Promise<Result<User>> {
  try {
    const user = await userRepository.findById(userId);
    if (!user) {
      return {
        success: false,
        error: new NotFoundError('User'),
      };
    }
    return { success: true, data: user };
  } catch (error) {
    logger.error({ error, userId }, 'Failed to fetch user');
    throw error;
  }
}

/**
 * Processes a batch of items concurrently with controlled parallelism.
 * 
 * Executes the processor function for each item in the batch, limiting
 * the number of concurrent operations to prevent resource exhaustion.
 * 
 * @template T - Type of items being processed
 * @template R - Type of the processing result
 * @param items - Array of items to process
 * @param processor - Async function that processes each item
 * @param concurrency - Maximum number of concurrent operations (default: 10)
 * @returns Promise resolving to an array of results
 * 
 * @example
 * ```typescript
 * const urls = ['https://api.example.com/1', 'https://api.example.com/2'];
 * const results = await processBatch(
 *   urls,
 *   async (url) => fetch(url).then(r => r.json()),
 *   5
 * );
 * ```
 */
export async function processBatch<T, R>(
  items: T[],
  processor: (item: T) => Promise<R>,
  concurrency: number = 10,
): Promise<R[]> {
  // Implementation
}
```

### C. Class Documentation

```typescript
/**
 * Service for managing user operations.
 * 
 * Provides high-level business logic for user management, including
 * creation, retrieval, updating, and deletion. Implements proper error
 * handling and logging for all operations.
 * 
 * @class
 * @example
 * ```typescript
 * const repository = new PrismaUserRepository(prisma);
 * const service = new UserService(repository);
 * 
 * const result = await service.createUser({
 *   email: 'user@example.com',
 *   name: 'John Doe'
 * });
 * ```
 */
export class UserService {
  /**
   * Creates a new UserService instance.
   * 
   * @param repository - Repository for user data access
   * @param logger - Logger instance for structured logging (optional)
   */
  constructor(
    private readonly repository: UserRepository,
    private readonly logger: Logger = defaultLogger,
  ) {}

  /**
   * Creates a new user in the system.
   * 
   * Validates the input, checks for duplicate emails, and creates
   * the user record. Returns a Result type to handle errors gracefully.
   * 
   * @param input - User creation data
   * @param input.email - User's email address (must be unique)
   * @param input.name - User's full name
   * @param input.role - Optional user role (defaults to 'user')
   * @returns Promise resolving to Result containing the created User or an error
   * 
   * @example
   * ```typescript
   * const result = await service.createUser({
   *   email: 'new@example.com',
   *   name: 'Jane Smith'
   * });
   * 
   * if (!result.success) {
   *   console.error('Failed to create user:', result.error.message);
   * }
   * ```
   */
  async createUser(input: CreateUserInput): Promise<Result<User>> {
    // Implementation
  }

  /**
   * Retrieves a user by their unique identifier.
   * 
   * @param id - User's unique identifier
   * @returns Promise resolving to Result containing the User or an error
   */
  async getUserById(id: string): Promise<Result<User>> {
    // Implementation
  }
}
```

### D. Interface and Type Documentation

```typescript
/**
 * Represents a user in the system.
 * 
 * Users have a unique identifier, authentication credentials,
 * and role-based permissions. All timestamps are in ISO 8601 format.
 * 
 * @interface
 * @property {string} id - Unique identifier (UUID v4)
 * @property {string} email - Email address (unique, validated)
 * @property {string} name - Full name (1-100 characters)
 * @property {UserRole} role - User's role in the system
 * @property {Date} createdAt - Account creation timestamp
 * @property {Date} updatedAt - Last modification timestamp
 * @property {Record<string, unknown>} [metadata] - Optional metadata
 */
export interface User {
  readonly id: string;
  email: string;
  name: string;
  role: UserRole;
  createdAt: Date;
  updatedAt: Date;
  metadata?: Record<string, unknown>;
}

/**
 * User role types available in the application.
 * 
 * - `admin`: Full system access, can manage all users
 * - `user`: Standard user with limited permissions
 * - `guest`: Read-only access, no modification rights
 * 
 * @typedef {('admin' | 'user' | 'guest')} UserRole
 */
export type UserRole = 'admin' | 'user' | 'guest';

/**
 * Result type for operations that can fail gracefully.
 * 
 * Provides a type-safe way to handle errors without throwing exceptions.
 * Success results contain data, failure results contain error information.
 * 
 * @template T - Type of the success data
 * @template E - Type of the error (defaults to Error)
 * @typedef {Object} Result
 * @property {boolean} success - Indicates operation success
 * 
 * @example Success result
 * ```typescript
 * const result: Result<User> = {
 *   success: true,
 *   data: { id: '123', name: 'John', email: 'john@example.com' }
 * };
 * ```
 * 
 * @example Error result
 * ```typescript
 * const result: Result<User> = {
 *   success: false,
 *   error: new NotFoundError('User not found')
 * };
 * ```
 */
export type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

/**
 * Input data for creating a new user.
 * 
 * Omits system-generated fields (id, timestamps) from the User type.
 * All fields are validated before user creation.
 * 
 * @interface
 */
export interface CreateUserInput {
  /** User's email address (must be unique) */
  email: string;
  /** User's full name (1-100 characters) */
  name: string;
  /** Optional role assignment (defaults to 'user') */
  role?: UserRole;
}
```

### E. Repository Pattern Documentation

```typescript
/**
 * Repository interface for user data access.
 * 
 * Defines the contract for user data operations, abstracting
 * the underlying data source. Implementations must be thread-safe
 * and handle database errors appropriately.
 * 
 * @interface
 * @see {@link PrismaUserRepository} for Prisma implementation
 */
export interface UserRepository {
  /**
   * Finds a user by their unique identifier.
   * 
   * @param id - User's unique identifier
   * @returns Promise resolving to User or null if not found
   */
  findById(id: string): Promise<User | null>;

  /**
   * Finds a user by their email address.
   * 
   * @param email - User's email address
   * @returns Promise resolving to User or null if not found
   */
  findByEmail(email: string): Promise<User | null>;

  /**
   * Creates a new user record.
   * 
   * @param data - User creation data
   * @returns Promise resolving to the created User
   * @throws {UniqueConstraintError} If email already exists
   */
  create(data: CreateUserInput): Promise<User>;

  /**
   * Updates an existing user record.
   * 
   * @param id - User's unique identifier
   * @param data - Partial user data to update
   * @returns Promise resolving to the updated User
   * @throws {NotFoundError} If user doesn't exist
   */
  update(id: string, data: Partial<User>): Promise<User>;

  /**
   * Deletes a user record.
   * 
   * @param id - User's unique identifier
   * @returns Promise resolving when deletion is complete
   * @throws {NotFoundError} If user doesn't exist
   */
  delete(id: string): Promise<void>;
}

/**
 * Prisma implementation of the UserRepository interface.
 * 
 * Provides type-safe database access using Prisma Client.
 * All operations are transactional and properly handle errors.
 * 
 * @class
 * @implements {UserRepository}
 */
export class PrismaUserRepository implements UserRepository {
  /**
   * Creates a new Prisma user repository.
   * 
   * @param db - Prisma client instance
   */
  constructor(private readonly db: PrismaClient) {}

  /** @inheritdoc */
  async findById(id: string): Promise<User | null> {
    return this.db.user.findUnique({ where: { id } });
  }

  /** @inheritdoc */
  async findByEmail(email: string): Promise<User | null> {
    return this.db.user.findUnique({ where: { email } });
  }

  /** @inheritdoc */
  async create(data: CreateUserInput): Promise<User> {
    return this.db.user.create({ data });
  }

  /** @inheritdoc */
  async update(id: string, data: Partial<User>): Promise<User> {
    return this.db.user.update({ where: { id }, data });
  }

  /** @inheritdoc */
  async delete(id: string): Promise<void> {
    await this.db.user.delete({ where: { id } });
  }
}
```

### F. Generating Documentation with TypeDoc

#### Installation and Setup

```bash
# Install TypeDoc and plugins
npm add --save-dev typedoc typedoc-plugin-markdown

# Add scripts to package.json
npm pkg set scripts.docs="typedoc --out docs src/"
npm pkg set scripts.docs:check="typedoc --emit none --validation.notDocumented true"
npm pkg set scripts.docs:serve="typedoc --out docs src/ && npx serve docs"
npm pkg set scripts.docs:json="typedoc --json docs/api.json src/"
```

#### TypeDoc Configuration

Create `typedoc.json` in project root:

```json
{
  "entryPoints": ["src/index.ts"],
  "entryPointStrategy": "expand",
  "out": "docs",
  "exclude": [
    "**/*.test.ts",
    "**/*.spec.ts",
    "**/test/**",
    "**/tests/**",
    "**/__tests__/**"
  ],
  "excludePrivate": true,
  "excludeProtected": false,
  "excludeInternal": false,
  "readme": "README.md",
  "plugin": ["typedoc-plugin-markdown"],
  "theme": "default",
  "categorizeByGroup": true,
  "categoryOrder": [
    "Services",
    "Repositories",
    "Controllers",
    "Middleware",
    "Utilities",
    "Types",
    "*"
  ],
  "navigation": {
    "includeCategories": true,
    "includeGroups": true
  },
  "sort": ["source-order"],
  "validation": {
    "notExported": true,
    "invalidLink": true,
    "notDocumented": true
  },
  "compilerOptions": {
    "moduleResolution": "bundler"
  }
}
```

#### Generating Documentation

```bash
# Generate HTML documentation
npm run docs

# Check documentation completeness
npm run docs:check

# Generate and serve documentation
npm run docs:serve

# Generate JSON documentation (for tooling)
npm run docs:json

# Open generated docs
open docs/index.html  # macOS
xdg-open docs/index.html  # Linux
```

#### Documentation Categories

Organize your code with JSDoc tags:

```typescript
/**
 * User service for business logic.
 * @category Services
 */
export class UserService {}

/**
 * User repository for data access.
 * @category Repositories
 */
export class UserRepository {}

/**
 * Authentication middleware.
 * @category Middleware
 */
export function authMiddleware() {}

/**
 * Logger utility.
 * @category Utilities
 */
export const logger = createLogger();

/**
 * User type definition.
 * @category Types
 */
export interface User {}
```

### G. Documentation Verification

**Add documentation checks to scripts:**

```json
// package.json
{
  "scripts": {
    "dev": "tsx watch src/index.ts",
    "build": "tsc",
    "start": "node dist/index.js",
    "test": "vitest",
    "test:coverage": "vitest run --coverage",
    "lint": "biome check .",
    "lint:fix": "biome check --apply .",
    "format": "biome format --write .",
    "typecheck": "tsc --noEmit",
    "docs": "typedoc --out docs src/",
    "docs:check": "typedoc --emit none --validation.notDocumented true",
    "docs:serve": "typedoc --out docs src/ && npx serve docs",
    "verify": "npm run typecheck && npm run docs:check && npm run lint && npm run test"
  }
}
```

### H. CI/CD Integration

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Type check
        run: npm run typecheck
      
      - name: Verify documentation
        run: npm run docs:check
      
      - name: Lint code
        run: npm run lint
      
      - name: Run tests
        run: npm run test:coverage
      
      - name: Generate documentation
        run: npm run docs
      
      - name: Upload documentation artifacts
        uses: actions/upload-artifact@v3
        with:
          name: api-documentation
          path: docs/
      
      - name: Deploy documentation to GitHub Pages
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
```

### I. Documentation Best Practices

**DO:**
- ✅ Document all public exports (functions, classes, interfaces, types)
- ✅ Include `@param` for all function parameters
- ✅ Include `@returns` for all return values
- ✅ Include `@throws` for functions that can throw
- ✅ Provide at least one `@example` for complex APIs
- ✅ Use `@template` for generic type parameters
- ✅ Include `@see` tags for related functions/types
- ✅ Keep examples up-to-date with implementation
- ✅ Generate docs as part of CI/CD pipeline
- ✅ Use `@category` to organize documentation

**DON'T:**
- ❌ Skip documentation for "obvious" functions
- ❌ Write vague descriptions ("Does stuff", "Helper function")
- ❌ Let examples become outdated
- ❌ Commit generated docs to git (add `docs/` to `.gitignore`)
- ❌ Use `@ts-ignore` to suppress documentation warnings
- ❌ Document private implementation details excessively

### J. Documentation Checklist

**Before committing code, verify:**
- [ ] All exported functions have JSDoc comments
- [ ] All classes have JSDoc comments
- [ ] All public interfaces and types have JSDoc comments
- [ ] All `@param` tags document parameter types and purpose
- [ ] All `@returns` tags document return types and values
- [ ] At least one `@example` provided for complex APIs
- [ ] `@throws` documented for functions that can throw errors
- [ ] TypeDoc can generate docs: `npm run docs:check` passes
- [ ] Generated documentation is readable and complete
- [ ] No "not documented" warnings from TypeDoc
- [ ] Examples compile and run correctly

### K. Complete Documentation Example

```typescript
/**
 * @fileoverview User authentication service with JWT token management.
 * Provides secure authentication, authorization, and session management.
 * @module services/auth
 */

import { SignJWT, jwtVerify } from 'jose';
import type { User } from '../types/user.types.js';
import type { UserRepository } from '../repositories/user.repository.js';
import { logger } from '../utils/logger.js';

/**
 * JWT payload structure for authentication tokens.
 * 
 * @interface
 * @property {string} userId - Unique user identifier
 * @property {string} email - User's email address
 * @property {UserRole} role - User's role for authorization
 * @property {number} iat - Issued at timestamp (Unix epoch)
 * @property {number} exp - Expiration timestamp (Unix epoch)
 */
export interface JWTPayload {
  userId: string;
  email: string;
  role: string;
  iat: number;
  exp: number;
}

/**
 * Authentication credentials for login.
 * 
 * @interface
 * @property {string} email - User's email address
 * @property {string} password - User's password (plain text, will be hashed)
 */
export interface LoginCredentials {
  email: string;
  password: string;
}

/**
 * Authentication result after successful login.
 * 
 * @interface
 * @property {User} user - Authenticated user data
 * @property {string} token - JWT access token
 * @property {string} refreshToken - Refresh token for obtaining new access tokens
 */
export interface AuthResult {
  user: User;
  token: string;
  refreshToken: string;
}

/**
 * Service for handling user authentication and authorization.
 * 
 * Manages JWT token generation, validation, and refresh operations.
 * Uses industry-standard security practices including password hashing
 * with bcrypt and secure token signing with HS256 algorithm.
 * 
 * @class
 * @category Services
 * 
 * @example
 * ```typescript
 * const authService = new AuthService(userRepository);
 * 
 * // Login
 * const result = await authService.login({
 *   email: 'user@example.com',
 *   password: 'secure-password'
 * });
 * 
 * if (result.success) {
 *   console.log('Token:', result.data.token);
 * }
 * 
 * // Verify token
 * const payload = await authService.verifyToken(result.data.token);
 * ```
 */
export class AuthService {
  private readonly jwtSecret: Uint8Array;
  private readonly tokenExpiry = '24h';
  private readonly refreshExpiry = '7d';

  /**
   * Creates a new authentication service.
   * 
   * @param repository - User repository for data access
   * @param jwtSecret - Secret key for JWT signing (min 32 bytes)
   * @throws {Error} If JWT secret is not provided or too short
   */
  constructor(
    private readonly repository: UserRepository,
    jwtSecret: string = process.env.JWT_SECRET || '',
  ) {
    if (!jwtSecret || jwtSecret.length < 32) {
      throw new Error('JWT_SECRET must be at least 32 characters');
    }
    this.jwtSecret = new TextEncoder().encode(jwtSecret);
  }

  /**
   * Authenticates a user with email and password.
   * 
   * Validates credentials, generates JWT tokens, and returns user data.
   * Uses bcrypt for secure password comparison. Failed attempts are logged.
   * 
   * @param credentials - User login credentials
   * @param credentials.email - User's email address
   * @param credentials.password - User's password (plain text)
   * @returns Promise resolving to Result with AuthResult or error
   * 
   * @example
   * ```typescript
   * const result = await authService.login({
   *   email: 'user@example.com',
   *   password: 'my-password'
   * });
   * 
   * if (result.success) {
   *   const { user, token } = result.data;
   *   // Store token and redirect to dashboard
   * } else {
   *   console.error('Login failed:', result.error.message);
   * }
   * ```
   * 
   * @see {@link verifyToken} for token validation
   * @see {@link refreshToken} for obtaining new tokens
   */
  async login(credentials: LoginCredentials): Promise<Result<AuthResult>> {
    const { email, password } = credentials;

    // Find user
    const user = await this.repository.findByEmail(email);
    if (!user) {
      logger.warn({ email }, 'Login attempt with non-existent email');
      return {
        success: false,
        error: new UnauthorizedError('Invalid credentials'),
      };
    }

    // Verify password
    const isValid = await this.verifyPassword(password, user.passwordHash);
    if (!isValid) {
      logger.warn({ userId: user.id }, 'Login attempt with invalid password');
      return {
        success: false,
        error: new UnauthorizedError('Invalid credentials'),
      };
    }

    // Generate tokens
    const token = await this.generateToken({
      userId: user.id,
      email: user.email,
      role: user.role,
    });

    const refreshToken = await this.generateRefreshToken(user.id);

    logger.info({ userId: user.id }, 'User logged in successfully');

    return {
      success: true,
      data: { user, token, refreshToken },
    };
  }

  /**
   * Verifies a JWT access token.
   * 
   * Validates token signature, expiration, and claims. Returns the
   * decoded payload if valid, or an error if verification fails.
   * 
   * @param token - JWT token string to verify
   * @returns Promise resolving to Result with JWTPayload or error
   * @throws {UnauthorizedError} If token is invalid or expired
   * 
   * @example
   * ```typescript
   * const result = await authService.verifyToken(
   *   'eyJhbGciOiJIUzI1NiIs...'
   * );
   * 
   * if (result.success) {
   *   const { userId, role } = result.data;
   *   // Check permissions based on role
   * }
   * ```
   */
  async verifyToken(token: string): Promise<Result<JWTPayload>> {
    try {
      const { payload } = await jwtVerify(token, this.jwtSecret, {
        issuer: 'myapp',
        audience: 'myapp-api',
      });

      return {
        success: true,
        data: payload as JWTPayload,
      };
    } catch (error) {
      logger.warn({ error }, 'Token verification failed');
      return {
        success: false,
        error: new UnauthorizedError('Invalid or expired token'),
      };
    }
  }

  /**
   * Generates a new JWT access token.
   * 
   * @private
   * @param payload - Token payload data
   * @returns Promise resolving to signed JWT string
   */
  private async generateToken(payload: Omit<JWTPayload, 'iat' | 'exp'>): Promise<string> {
    return await new SignJWT(payload)
      .setProtectedHeader({ alg: 'HS256' })
      .setIssuedAt()
      .setExpirationTime(this.tokenExpiry)
      .setIssuer('myapp')
      .setAudience('myapp-api')
      .sign(this.jwtSecret);
  }

  /**
   * Generates a refresh token for obtaining new access tokens.
   * 
   * @private
   * @param userId - User's unique identifier
   * @returns Promise resolving to signed refresh token string
   */
  private async generateRefreshToken(userId: string): Promise<string> {
    return await new SignJWT({ userId, type: 'refresh' })
      .setProtectedHeader({ alg: 'HS256' })
      .setIssuedAt()
      .setExpirationTime(this.refreshExpiry)
      .setIssuer('myapp')
      .setAudience('myapp-api')
      .sign(this.jwtSecret);
  }

  /**
   * Verifies a password against its hash.
   * 
   * @private
   * @param password - Plain text password
   * @param hash - Bcrypt password hash
   * @returns Promise resolving to true if password matches
   */
  private async verifyPassword(password: string, hash: string): Promise<boolean> {
    // Implementation with bcrypt
    return true; // Placeholder
  }
}
```

## 4. Mandatory Code Standards

### A. Type Safety
* **NEVER use `any`**. Use `unknown` and type guards instead.

* **ALWAYS define return types** explicitly for functions.

* **USE discriminated unions** for complex state.

```typescript
// ✅ CORRECT - Proper typing
interface User {
  readonly id: string;
  name: string;
  email: string;
  role: 'admin' | 'user' | 'guest';
}

function findUser(id: string): Promise<User | null> {
  // Implementation
}

// Type guard pattern
function isUser(value: unknown): value is User {
  return (
    typeof value === 'object' &&
    value !== null &&
    'id' in value &&
    typeof value.id === 'string'
  );
}

// Discriminated union for results
type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

async function getUserSafely(id: string): Promise<Result<User>> {
  try {
    const user = await findUser(id);
    if (!user) {
      return { success: false, error: new Error('User not found') };
    }
    return { success: true, data: user };
  } catch (error) {
    return { 
      success: false, 
      error: error instanceof Error ? error : new Error('Unknown error') 
    };
  }
}

// ❌ WRONG - Using any
function processData(data: any): any {
  return data.value;
}
```

### B. Modern ESM Patterns
* **Always use ESM imports/exports**.

* **Use file extensions** in imports for Node.js ESM.

* **Leverage top-level await** (Node.js 14.8+).

```typescript
// ✅ CORRECT - Modern ESM
import { readFile } from 'node:fs/promises';
import { z } from 'zod';
import type { User } from './types/index.js';  // .js extension required
import { logger } from '@/utils/logger.js';

// Top-level await
const config = await loadConfig();

export async function processUser(data: unknown): Promise<User> {
  // Implementation
}

export type { User };

// ❌ WRONG - CommonJS
const fs = require('fs');
module.exports = { processUser };

// ❌ WRONG - No file extension in relative imports
import { User } from './types';  // Missing .js
```

### C. Async/Await Patterns
* **ALWAYS use async/await** over callbacks or raw promises.

* **USE Promise.all()** for concurrent operations.

* **IMPLEMENT proper error boundaries** for async code.

```typescript
// ✅ CORRECT - Concurrent async operations
async function fetchUserData(userId: string) {
  const [user, posts, comments] = await Promise.all([
    fetchUser(userId),
    fetchUserPosts(userId),
    fetchUserComments(userId),
  ]);
  
  return { user, posts, comments };
}

// ✅ CORRECT - Error handling with Result type
async function safeFetch<T>(url: string): Promise<Result<T>> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      return {
        success: false,
        error: new Error(`HTTP ${response.status}: ${response.statusText}`),
      };
    }
    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error : new Error('Unknown error'),
    };
  }
}

// ✅ CORRECT - Async iteration
async function processLargeDataset(items: AsyncIterable<Item>) {
  for await (const item of items) {
    await processItem(item);
  }
}

// ❌ WRONG - Callback hell
function fetchData(callback: (err: Error | null, data?: any) => void) {
  getData((err, data) => {
    if (err) return callback(err);
    processData(data, (err, result) => {
      if (err) return callback(err);
      callback(null, result);
    });
  });
}
```

### D. Error Handling
* **CREATE custom error classes** with proper inheritance.

* **USE Result types** for expected errors.

* **IMPLEMENT global error handlers** for unexpected errors.

```typescript
// ✅ CORRECT - Custom error hierarchy
class AppError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly statusCode: number = 500,
    public readonly isOperational: boolean = true,
  ) {
    super(message);
    this.name = this.constructor.name;
    Error.captureStackTrace(this, this.constructor);
  }
}

class ValidationError extends AppError {
  constructor(message: string) {
    super(message, 'VALIDATION_ERROR', 400);
  }
}

class NotFoundError extends AppError {
  constructor(resource: string) {
    super(`${resource} not found`, 'NOT_FOUND', 404);
  }
}

class UnauthorizedError extends AppError {
  constructor(message: string = 'Unauthorized') {
    super(message, 'UNAUTHORIZED', 401);
  }
}

// ✅ CORRECT - Error handling middleware (Express)
function errorHandler(
  error: Error,
  req: Request,
  res: Response,
  next: NextFunction,
): void {
  if (error instanceof AppError) {
    res.status(error.statusCode).json({
      success: false,
      error: {
        code: error.code,
        message: error.message,
      },
    });
    return;
  }

  // Unexpected errors
  logger.error('Unexpected error', { error, url: req.url });
  res.status(500).json({
    success: false,
    error: {
      code: 'INTERNAL_ERROR',
      message: 'An unexpected error occurred',
    },
  });
}

// ✅ CORRECT - Using Result type
type Result<T, E = AppError> =
  | { success: true; data: T }
  | { success: false; error: E };

async function createUser(input: CreateUserInput): Promise<Result<User>> {
  // Validation
  const parsed = userSchema.safeParse(input);
  if (!parsed.success) {
    return {
      success: false,
      error: new ValidationError(parsed.error.message),
    };
  }

  // Business logic
  try {
    const user = await userRepository.create(parsed.data);
    return { success: true, data: user };
  } catch (error) {
    if (error instanceof UniqueConstraintError) {
      return {
        success: false,
        error: new ValidationError('Email already exists'),
      };
    }
    throw error; // Unexpected errors bubble up
  }
}
```

### E. Validation with Zod
* **ALWAYS validate external input** (API requests, env vars, external APIs).

* **USE Zod for runtime validation** and type inference.

* **DEFINE schemas co-located** with types.

```typescript
// ✅ CORRECT - Zod schemas
import { z } from 'zod';

// Schema definition
export const userSchema = z.object({
  id: z.string().uuid(),
  email: z.string().email(),
  name: z.string().min(1).max(100),
  age: z.number().int().positive().optional(),
  role: z.enum(['admin', 'user', 'guest']).default('user'),
  createdAt: z.date(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

// Infer TypeScript type from schema
export type User = z.infer<typeof userSchema>;

// API request validation
export const createUserSchema = userSchema.omit({ 
  id: true, 
  createdAt: true 
});

export type CreateUserInput = z.infer<typeof createUserSchema>;

// Environment variables validation
const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'staging', 'production']),
  PORT: z.coerce.number().int().positive().default(3000),
  DATABASE_URL: z.string().url(),
  API_KEY: z.string().min(32),
  LOG_LEVEL: z.enum(['debug', 'info', 'warn', 'error']).default('info'),
});

export const env = envSchema.parse(process.env);

// Usage in API handler
async function createUserHandler(req: Request, res: Response) {
  const result = createUserSchema.safeParse(req.body);
  
  if (!result.success) {
    return res.status(400).json({
      success: false,
      error: {
        code: 'VALIDATION_ERROR',
        details: result.error.format(),
      },
    });
  }
  
  const user = await createUser(result.data);
  return res.status(201).json({ success: true, data: user });
}
```

### F. Logging Standards
* **USE structured logging** with Pino or similar.

* **LOG at appropriate levels** (debug, info, warn, error).

* **INCLUDE correlation IDs** for request tracing.

```typescript
// ✅ CORRECT - Structured logging setup
import pino from 'pino';

export const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  formatters: {
    level: (label) => ({ level: label }),
  },
  timestamp: pino.stdTimeFunctions.isoTime,
  ...(process.env.NODE_ENV === 'development' && {
    transport: {
      target: 'pino-pretty',
      options: {
        colorize: true,
        translateTime: 'SYS:standard',
        ignore: 'pid,hostname',
      },
    },
  }),
});

// Usage with context
logger.info({ userId: '123', action: 'login' }, 'User logged in');

// Child logger with request context
app.use((req, res, next) => {
  req.log = logger.child({ 
    requestId: crypto.randomUUID(),
    method: req.method,
    url: req.url,
  });
  next();
});

// Logging errors
try {
  await riskyOperation();
} catch (error) {
  logger.error({ error, context: 'important-operation' }, 'Operation failed');
  throw error;
}
```

### G. Node.js 22+ Native Features (MANDATORY)

**Prefer built-in modules over external dependencies:**

```typescript
// 1. Native .env support (Node.js 20.6+)
// Run with: node --env-file=.env index.js
const apiKey = process.env.API_KEY;

// 2. Native SQLite (Node.js 22.5+)
import { DatabaseSync } from 'node:sqlite';
const db = new DatabaseSync('data.db');

// 3. Native Test Runner (Node.js 20+)
import { test, describe, it } from 'node:test';
import assert from 'node:assert';

test('synchronous test', (t) => {
  assert.strictEqual(1, 1);
});

// 4. Native Fetch (Node.js 18+)
const response = await fetch('https://api.example.com');
const data = await response.json();
```

## 5. Modern Framework Patterns

### A. Fastify (Recommended for APIs)
* **USE Fastify v4+** for high-performance APIs.

* **LEVERAGE built-in schema validation** with JSON Schema or Zod integration.

```typescript
// ✅ CORRECT - Fastify with TypeScript
import Fastify from 'fastify';
import { TypeBoxTypeProvider } from '@fastify/type-provider-typebox';
import { Type } from '@sinclair/typebox';

const server = Fastify({
  logger: true,
}).withTypeProvider<TypeBoxTypeProvider>();

// Route with schema
server.post(
  '/api/users',
  {
    schema: {
      body: Type.Object({
        email: Type.String({ format: 'email' }),
        name: Type.String({ minLength: 1, maxLength: 100 }),
      }),
      response: {
        201: Type.Object({
          id: Type.String({ format: 'uuid' }),
          email: Type.String(),
          name: Type.String(),
        }),
      },
    },
  },
  async (request, reply) => {
    const user = await createUser(request.body);
    return reply.code(201).send(user);
  },
);

// Start server
await server.listen({ port: 3000, host: '0.0.0.0' });
```

### B. Express (Legacy, but common)
If using Express, follow these modern patterns:

```typescript
// ✅ CORRECT - Modern Express with TypeScript
import express, { type Request, type Response, type NextFunction } from 'express';
import { z } from 'zod';

const app = express();
app.use(express.json());

// Type-safe request handler
interface TypedRequest<T> extends Request {
  body: T;
}

// Validation middleware factory
function validateBody<T extends z.ZodType>(schema: T) {
  return (req: Request, res: Response, next: NextFunction) => {
    const result = schema.safeParse(req.body);
    if (!result.success) {
      return res.status(400).json({
        success: false,
        error: result.error.format(),
      });
    }
    req.body = result.data;
    next();
  };
}

// Route with validation
app.post(
  '/api/users',
  validateBody(createUserSchema),
  async (req: TypedRequest<CreateUserInput>, res: Response) => {
    const user = await createUser(req.body);
    res.status(201).json({ success: true, data: user });
  },
);

// Async handler wrapper
function asyncHandler(
  fn: (req: Request, res: Response, next: NextFunction) => Promise<void>,
) {
  return (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
}
```

### C. Database Layer (Prisma)
* **USE Prisma for type-safe database access**.

* **IMPLEMENT repository pattern** for business logic separation.

```typescript
// ✅ CORRECT - Prisma with repository pattern
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient({
  log: ['query', 'error', 'warn'],
});

// Repository interface
interface UserRepository {
  findById(id: string): Promise<User | null>;
  findByEmail(email: string): Promise<User | null>;
  create(data: CreateUserInput): Promise<User>;
  update(id: string, data: Partial<User>): Promise<User>;
  delete(id: string): Promise<void>;
}

// Implementation
export class PrismaUserRepository implements UserRepository {
  constructor(private readonly db: PrismaClient) {}

  async findById(id: string): Promise<User | null> {
    return this.db.user.findUnique({ where: { id } });
  }

  async findByEmail(email: string): Promise<User | null> {
    return this.db.user.findUnique({ where: { email } });
  }

  async create(data: CreateUserInput): Promise<User> {
    return this.db.user.create({ data });
  }

  async update(id: string, data: Partial<User>): Promise<User> {
    return this.db.user.update({ where: { id }, data });
  }

  async delete(id: string): Promise<void> {
    await this.db.user.delete({ where: { id } });
  }
}

// Service layer
export class UserService {
  constructor(private readonly repository: UserRepository) {}

  async createUser(input: CreateUserInput): Promise<Result<User>> {
    // Validation
    const existing = await this.repository.findByEmail(input.email);
    if (existing) {
      return {
        success: false,
        error: new ValidationError('Email already exists'),
      };
    }

    // Create user
    try {
      const user = await this.repository.create(input);
      return { success: true, data: user };
    } catch (error) {
      logger.error({ error }, 'Failed to create user');
      throw error;
    }
  }
}
```

## 6. Testing Standards

### A. Unit Testing with Vitest
* **USE Vitest** for fast, modern testing.

* **AIM for 80%+ coverage** on business logic.

* **WRITE type tests** for complex types.

```typescript
// ✅ CORRECT - Vitest unit tests
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { UserService } from './user.service.js';
import type { UserRepository } from './user.repository.js';

describe('UserService', () => {
  let service: UserService;
  let mockRepository: UserRepository;

  beforeEach(() => {
    // Create mock repository
    mockRepository = {
      findById: vi.fn(),
      findByEmail: vi.fn(),
      create: vi.fn(),
      update: vi.fn(),
      delete: vi.fn(),
    };
    service = new UserService(mockRepository);
  });

  describe('createUser', () => {
    it('should create a user successfully', async () => {
      const input = { email: 'test@example.com', name: 'Test User' };
      const created = { id: '123', ...input };

      vi.mocked(mockRepository.findByEmail).mockResolvedValue(null);
      vi.mocked(mockRepository.create).mockResolvedValue(created);

      const result = await service.createUser(input);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data).toEqual(created);
      }
    });

    it('should fail if email already exists', async () => {
      const input = { email: 'existing@example.com', name: 'Test' };
      const existing = { id: '456', ...input };

      vi.mocked(mockRepository.findByEmail).mockResolvedValue(existing);

      const result = await service.createUser(input);

      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error.code).toBe('VALIDATION_ERROR');
      }
    });
  });
});

// Type testing
import { expectTypeOf } from 'vitest';
import type { User } from './types/index.js';

describe('Type tests', () => {
  it('should have correct User type structure', () => {
    expectTypeOf<User>().toHaveProperty('id');
    expectTypeOf<User>().toHaveProperty('email');
    expectTypeOf<User['id']>().toBeString();
    expectTypeOf<User['role']>().toEqualTypeOf<'admin' | 'user' | 'guest'>();
  });
});
```

### B. Integration Testing
```typescript
// ✅ CORRECT - Integration tests with test database
import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient({
  datasourceUrl: process.env.TEST_DATABASE_URL,
});

describe('User Integration Tests', () => {
  beforeAll(async () => {
    await prisma.$connect();
  });

  afterAll(async () => {
    await prisma.$disconnect();
  });

  beforeEach(async () => {
    // Clean database before each test
    await prisma.user.deleteMany();
  });

  it('should create and retrieve a user', async () => {
    const repository = new PrismaUserRepository(prisma);
    const input = { email: 'test@example.com', name: 'Test User' };

    const created = await repository.create(input);
    const retrieved = await repository.findById(created.id);

    expect(retrieved).toEqual(created);
  });
});
```

## 7. Performance Optimization

### A. Native Node.js APIs
* **USE native Node.js APIs** when available.

* **LEVERAGE Web Streams API** for data processing.

```typescript
// ✅ CORRECT - Using native APIs
import { pipeline } from 'node:stream/promises';
import { createReadStream, createWriteStream } from 'node:fs';
import { createGzip } from 'node:zlib';

// Native fetch (Node.js 18+)
const response = await fetch('https://api.example.com/data');
const data = await response.json();

// Web Crypto API
const hash = await crypto.subtle.digest(
  'SHA-256',
  new TextEncoder().encode('data'),
);

// Stream processing
await pipeline(
  createReadStream('input.txt'),
  createGzip(),
  createWriteStream('input.txt.gz'),
);

// ❌ WRONG - Using external libraries when native exists
import axios from 'axios';  // Use native fetch instead
import crypto from 'crypto-js';  // Use Web Crypto API
```

### B. Concurrency Patterns
```typescript
// ✅ CORRECT - Efficient concurrency
// Batch processing with concurrency limit
async function processBatch<T, R>(
  items: T[],
  processor: (item: T) => Promise<R>,
  concurrency: number = 10,
): Promise<R[]> {
  const results: R[] = [];
  const queue = [...items];

  async function worker() {
    while (queue.length > 0) {
      const item = queue.shift();
      if (item) {
        results.push(await processor(item));
      }
    }
  }

  await Promise.all(Array.from({ length: concurrency }, () => worker()));
  return results;
}

// Async generator for memory efficiency
async function* readLargeFile(path: string): AsyncGenerator<string> {
  const stream = createReadStream(path, { encoding: 'utf-8' });
  let buffer = '';

  for await (const chunk of stream) {
    buffer += chunk;
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      yield line;
    }
  }

  if (buffer) yield buffer;
}

// Usage
for await (const line of readLargeFile('large-file.txt')) {
  await processLine(line);
}
```

## 8. Security Best Practices

### A. Input Sanitization & Validation
```typescript
// ✅ CORRECT - Comprehensive validation
import { z } from 'zod';
import DOMPurify from 'isomorphic-dompurify';

const sanitizedStringSchema = z
  .string()
  .transform((val) => DOMPurify.sanitize(val));

const apiInputSchema = z.object({
  email: z.string().email().toLowerCase(),
  password: z.string().min(12).max(128),
  name: sanitizedStringSchema.min(1).max(100),
  url: z.string().url(),
  age: z.number().int().min(0).max(150),
});

// SQL injection prevention (Prisma handles this)
const user = await prisma.user.findUnique({
  where: { email: input.email },  // Safe, parameterized
});

// ❌ WRONG - Raw SQL without parameterization
await prisma.$executeRaw(`SELECT * FROM users WHERE email = '${email}'`);
```

### B. Authentication & Authorization
```typescript
// ✅ CORRECT - JWT with proper validation
import { SignJWT, jwtVerify } from 'jose';

const JWT_SECRET = new TextEncoder().encode(process.env.JWT_SECRET);

async function generateToken(payload: { userId: string; role: string }) {
  return await new SignJWT(payload)
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuedAt()
    .setExpirationTime('24h')
    .setIssuer('myapp')
    .setAudience('myapp-api')
    .sign(JWT_SECRET);
}

async function verifyToken(token: string) {
  try {
    const { payload } = await jwtVerify(token, JWT_SECRET, {
      issuer: 'myapp',
      audience: 'myapp-api',
    });
    return { success: true, data: payload };
  } catch (error) {
    return { success: false, error: new UnauthorizedError('Invalid token') };
  }
}

// Authorization middleware
function requireRole(...allowedRoles: string[]) {
  return async (req: Request, res: Response, next: NextFunction) => {
    const token = req.headers.authorization?.replace('Bearer ', '');
    if (!token) {
      return res.status(401).json({ error: 'No token provided' });
    }

    const result = await verifyToken(token);
    if (!result.success) {
      return res.status(401).json({ error: 'Invalid token' });
    }

    const { role } = result.data;
    if (!allowedRoles.includes(role)) {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }

    req.user = result.data;
    next();
  };
}
```

### C. Rate Limiting
```typescript
// ✅ CORRECT - Rate limiting with Redis
import rateLimit from 'express-rate-limit';
import RedisStore from 'rate-limit-redis';
import { createClient } from 'redis';

const redisClient = createClient({ url: process.env.REDIS_URL });
await redisClient.connect();

const limiter = rateLimit({
  store: new RedisStore({
    client: redisClient,
    prefix: 'rate-limit:',
  }),
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per window
  standardHeaders: true,
  legacyHeaders: false,
  handler: (req, res) => {
    res.status(429).json({
      error: 'Too many requests, please try again later.',
    });
  },
});

app.use('/api/', limiter);
```

## 9. Development Tools

### A. Biome Configuration (Modern Linter + Formatter)
```json
// ✅ CORRECT - biome.json
{
  "$schema": "https://biomejs.dev/schemas/1.5.0/schema.json",
  "organizeImports": {
    "enabled": true
  },
  "linter": {
    "enabled": true,
    "rules": {
      "recommended": true,
      "complexity": {
        "noExtraBooleanCast": "error",
        "noMultipleSpacesInRegularExpressionLiterals": "error",
        "noUselessCatch": "error",
        "noUselessConstructor": "error",
        "noUselessLoneBlockStatements": "error",
        "noUselessRename": "error",
        "noWith": "error",
        "useFlatMap": "error",
        "useOptionalChain": "error",
        "useSimplifiedLogicExpression": "error"
      },
      "suspicious": {
        "noArrayIndexKey": "warn",
        "noAssignInExpressions": "error",
        "noAsyncPromiseExecutor": "error",
        "noCatchAssign": "error",
        "noCommentText": "error",
        "noCompareNegZero": "error",
        "noDebugger": "error",
        "noDoubleEquals": "error",
        "noDuplicateCase": "error",
        "noExplicitAny": "error",
        "noFallthroughSwitchClause": "error",
        "noGlobalIsFinite": "error",
        "noGlobalIsNan": "error",
        "noShadowRestrictedNames": "error",
        "noUnsafeNegation": "error"
      },
      "style": {
        "noNegationElse": "error",
        "noShoutyConstants": "warn",
        "useBlockStatements": "error",
        "useCollapsedElseIf": "error",
        "useConsistentArrayType": {
          "level": "error",
          "options": { "syntax": "shorthand" }
        },
        "useForOf": "error",
        "useShorthandArrayType": "error",
        "useShorthandAssign": "error",
        "useSingleVarDeclarator": "error",
        "useTemplate": "error"
      }
    }
  },
  "formatter": {
    "enabled": true,
    "indentStyle": "space",
    "indentWidth": 2,
    "lineWidth": 100
  },
  "javascript": {
    "formatter": {
      "quoteStyle": "single",
      "trailingComma": "all",
      "semicolons": "always",
      "arrowParentheses": "always"
    }
  }
}
```

### B. Package.json Scripts
```json
{
  "name": "modern-node-app",
  "version": "1.0.0",
  "type": "module",
  "engines": {
    "node": ">=22.0.0",
    "npm": ">=10.0.0"
  },
  "scripts": {
    "dev": "tsx watch src/index.ts",
    "build": "tsc",
    "start": "node dist/index.js",
    "test": "vitest",
    "test:coverage": "vitest run --coverage",
    "test:ui": "vitest --ui",
    "lint": "biome check .",
    "lint:fix": "biome check --apply .",
    "format": "biome format --write .",
    "typecheck": "tsc --noEmit",
    "docs": "typedoc --out docs src/",
    "docs:check": "typedoc --emit none --validation.notDocumented true",
    "docs:serve": "typedoc --out docs src/ && npx serve docs",
    "docs:json": "typedoc --json docs/api.json src/",
    "verify": "npm run typecheck && npm run docs:check && npm run lint && npm run test",
    "db:migrate": "prisma migrate dev",
    "db:generate": "prisma generate",
    "db:studio": "prisma studio",
    "clean": "rm -rf dist docs"
  }
}
```

## 10. Complete Production Example

```typescript
// src/index.ts - Main application
import { serve } from '@hono/node-server';
import { Hono } from 'hono';
import { logger as loggerMiddleware } from 'hono/logger';
import { cors } from 'hono/cors';
import { PrismaClient } from '@prisma/client';
import { logger } from './utils/logger.js';
import { errorHandler } from './middleware/error-handler.js';
import { createUserRoutes } from './routes/user.routes.js';

// Initialize database
const prisma = new PrismaClient({
  log: ['query', 'error', 'warn'],
});

// Initialize app
const app = new Hono();

// Middleware
app.use('*', loggerMiddleware());
app.use('*', cors());

// Health check
app.get('/health', (c) => {
  return c.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Routes
app.route('/api/users', createUserRoutes(prisma));

// Error handling
app.onError(errorHandler);

// Graceful shutdown
const shutdown = async (signal: string) => {
  logger.info(`${signal} received, shutting down gracefully`);
  
  await prisma.$disconnect();
  logger.info('Database connection closed');
  
  process.exit(0);
};

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));

// Start server
const port = Number(process.env.PORT) || 3000;
serve({ fetch: app.fetch, port }, (info) => {
  logger.info(`Server running on http://localhost:${info.port}`);
});

// src/routes/user.routes.ts
import { Hono } from 'hono';
import type { PrismaClient } from '@prisma/client';
import { UserService } from '../services/user.service.js';
import { PrismaUserRepository } from '../repositories/user.repository.js';
import { createUserSchema } from '../types/user.types.js';

export function createUserRoutes(prisma: PrismaClient) {
  const app = new Hono();
  const repository = new PrismaUserRepository(prisma);
  const service = new UserService(repository);

  app.post('/', async (c) => {
    const body = await c.req.json();
    const parsed = createUserSchema.safeParse(body);

    if (!parsed.success) {
      return c.json(
        { success: false, error: parsed.error.format() },
        400,
      );
    }

    const result = await service.createUser(parsed.data);

    if (!result.success) {
      return c.json({ success: false, error: result.error.message }, 400);
    }

    return c.json({ success: true, data: result.data }, 201);
  });

  app.get('/:id', async (c) => {
    const { id } = c.req.param();
    const result = await service.getUserById(id);

    if (!result.success) {
      return c.json({ success: false, error: result.error.message }, 404);
    }

    return c.json({ success: true, data: result.data });
  });

  return app;
}

// src/utils/logger.ts
import pino from 'pino';

export const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  transport:
    process.env.NODE_ENV === 'development'
      ? {
          target: 'pino-pretty',
          options: {
            colorize: true,
            translateTime: 'SYS:standard',
            ignore: 'pid,hostname',
          },
        }
      : undefined,
});

// src/middleware/error-handler.ts
import type { Context } from 'hono';
import { AppError } from '../utils/errors.js';
import { logger } from '../utils/logger.js';

export function errorHandler(error: Error, c: Context) {
  if (error instanceof AppError) {
    return c.json(
      {
        success: false,
        error: {
          code: error.code,
          message: error.message,
        },
      },
      error.statusCode,
    );
  }

  logger.error({ error, path: c.req.path }, 'Unexpected error');

  return c.json(
    {
      success: false,
      error: {
        code: 'INTERNAL_ERROR',
        message: 'An unexpected error occurred',
      },
    },
    500,
  );
}
```

## 11. Deployment Checklist

### Pre-Production Validation
- [ ] All TypeScript strict checks enabled and passing
- [ ] **All exported functions/classes documented with JSDoc** (`npm run docs:check`)
- [ ] **API documentation generated successfully** (`npm run docs`)
- [ ] Test coverage ≥ 80% on business logic
- [ ] No `any` types in production code
- [ ] Environment variables validated with Zod
- [ ] All external inputs validated
- [ ] Error handling implemented for all async operations
- [ ] Logging configured with appropriate levels
- [ ] Database connection pooling configured
- [ ] Rate limiting enabled on public endpoints
- [ ] CORS configured appropriately
- [ ] Security headers set (Helmet.js or equivalent)
- [ ] Secrets not hardcoded (use environment variables)
- [ ] Graceful shutdown handlers implemented
- [ ] Health check endpoint implemented
- [ ] Monitoring/observability configured (OpenTelemetry, Sentry, etc.)

### Agent Code Generation Verification (MANDATORY)
**If code was generated by an agent, verify BEFORE delivery:**
- [ ] TypeScript compilation successful: `npm run typecheck` passes
- [ ] **JSDoc comments added for ALL new exports** (functions, classes, types)
- [ ] **Documentation check passes**: `npm run docs:check` returns exit code 0
- [ ] **Documentation can be generated**: `npm run docs` succeeds without errors
- [ ] All documentation includes `@param`, `@returns`, and `@example` tags
- [ ] Biome linter passes: `npm run lint` passes
- [ ] All tests passing: `npm run test` passes
- [ ] Test coverage ≥ 80%: `npm run test:coverage`
- [ ] No `any` types used as workarounds
- [ ] Agent has documented any complex fixes made during verification

### Environment Variables Template
```bash
# .env.example
NODE_ENV=production
PORT=3000
LOG_LEVEL=info

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/mydb

# Redis
REDIS_URL=redis://localhost:6379

# Authentication
JWT_SECRET=your-256-bit-secret-here

# External APIs
API_KEY=your-api-key-here

# Monitoring
SENTRY_DSN=https://..
```

### .gitignore Template
```gitignore
# Dependencies
node_modules/

# Build output
dist/
build/
*.js
*.js.map
*.d.ts
*.d.ts.map

# Generated documentation (regenerate during CI/CD)
docs/
api-docs/

# Environment
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# Testing
coverage/
.nyc_output/

# OS
.DS_Store
Thumbs.db

# Logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*
```

## 12. Why This Configuration Works

1. **ESM Modules**: Modern standard, tree-shaking support, better for performance and compatibility with web standards.

2. **TypeScript Strict Mode**: Catches errors at compile time, reduces runtime bugs by 15-30%.

3. **JSDoc + TypeDoc**: Auto-generated documentation from code, always in sync, reduces onboarding time by 40%+, better IDE IntelliSense, verified during build.

4. **Zod Validation**: Runtime type safety at API boundaries, automatic TypeScript type inference.

5. **Result Type Pattern**: Explicit error handling, makes error states visible in types, prevents uncaught exceptions.

6. **Structured Logging**: Essential for debugging in production, enables log aggregation and analysis.

7. **Repository Pattern**: Separates data access from business logic, easier testing, database agnostic.

8. **Vitest**: 10x faster than Jest, native ESM support, better TypeScript integration.

9. **Biome**: Single tool for linting and formatting, 100x faster than ESLint+Prettier.

10. **npm**: Standard tool working correctly under Linux, OSX and Windows.

11. **Native APIs**: Better performance, smaller bundle size, no external dependencies to maintain.

---

## 13. Quick Reference

### Common Commands

```bash
# ═══════════════════════════════════════════════════════════════
# DEVELOPMENT
# ═══════════════════════════════════════════════════════════════

# Run development server with hot reload
npm run dev                    # Uses tsx watch src/index.ts

# Run a single TypeScript file
npx tsx src/script.ts          # Execute TS directly

# Type check without emitting
npm run typecheck              # tsc --noEmit

# ═══════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════

# Run all tests
npm test                       # vitest

# Run tests in watch mode
npm run test:watch             # vitest --watch

# Run tests with coverage
npm run test:coverage          # vitest run --coverage

# Run specific test file
npx vitest src/services/user.service.test.ts

# Run tests matching pattern
npx vitest -t "should create user"

# ═══════════════════════════════════════════════════════════════
# BUILD & PRODUCTION
# ═══════════════════════════════════════════════════════════════

# Build TypeScript to JavaScript
npm run build                  # tsc

# Run production build
npm start                      # node dist/index.js

# Clean build artifacts
npm run clean                  # rm -rf dist docs

# ═══════════════════════════════════════════════════════════════
# CODE QUALITY
# ═══════════════════════════════════════════════════════════════

# Lint code
npm run lint                   # biome check .

# Fix lint issues
npm run lint:fix               # biome check --apply .

# Format code
npm run format                 # biome format --write .

# Full verification (typecheck + lint + test)
npm run verify                 # All checks in sequence

# ═══════════════════════════════════════════════════════════════
# DOCUMENTATION
# ═══════════════════════════════════════════════════════════════

# Generate API documentation
npm run docs                   # typedoc --out docs src/

# Check documentation completeness
npm run docs:check             # typedoc --emit none --validation.notDocumented true

# Serve documentation locally
npm run docs:serve             # Generate and serve at localhost

# ═══════════════════════════════════════════════════════════════
# DATABASE (Prisma)
# ═══════════════════════════════════════════════════════════════

# Run migrations in development
npm run db:migrate             # prisma migrate dev

# Generate Prisma Client
npm run db:generate            # prisma generate

# Open Prisma Studio (database GUI)
npm run db:studio              # prisma studio

# Reset database
npx prisma migrate reset       # Drop and recreate

# ═══════════════════════════════════════════════════════════════
# PACKAGE MANAGEMENT
# ═══════════════════════════════════════════════════════════════

# Install dependencies
npm install                    # Install from package.json

# Add production dependency
npm add zod                    # Add to dependencies

# Add dev dependency
npm add -D vitest              # Add to devDependencies

# Update all dependencies
npm update                     # Update within semver ranges

# Check for outdated packages
npm outdated                   # Show outdated deps

# Audit for vulnerabilities
npm audit                      # Security check
npm audit fix                  # Auto-fix vulnerabilities
```

### Node.js Patterns Cheat Sheet

```typescript
// ═══════════════════════════════════════════════════════════════
// ASYNC/AWAIT PATTERNS
// ═══════════════════════════════════════════════════════════════

// Parallel execution (independent operations)
const [users, posts, comments] = await Promise.all([
  fetchUsers(),
  fetchPosts(),
  fetchComments(),
]);

// Sequential execution (dependent operations)
const user = await fetchUser(id);
const posts = await fetchUserPosts(user.id);
const comments = await fetchPostComments(posts[0].id);

// Race condition (first to complete wins)
const result = await Promise.race([
  fetchFromPrimary(),
  fetchFromBackup(),
]);

// All settled (get results regardless of success/failure)
const results = await Promise.allSettled([
  riskyOperation1(),
  riskyOperation2(),
]);
results.forEach(result => {
  if (result.status === 'fulfilled') {
    console.log('Success:', result.value);
  } else {
    console.log('Failed:', result.reason);
  }
});

// Timeout wrapper
async function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  const timeout = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new Error('Timeout')), ms)
  );
  return Promise.race([promise, timeout]);
}

// Retry with exponential backoff
async function retry<T>(
  fn: () => Promise<T>,
  maxAttempts: number = 3,
  baseDelay: number = 1000,
): Promise<T> {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      if (attempt === maxAttempts) throw error;
      await new Promise(r => setTimeout(r, baseDelay * Math.pow(2, attempt - 1)));
    }
  }
  throw new Error('Unreachable');
}

// ═══════════════════════════════════════════════════════════════
// STREAM PATTERNS
// ═══════════════════════════════════════════════════════════════

import { pipeline } from 'node:stream/promises';
import { createReadStream, createWriteStream } from 'node:fs';
import { createGzip, createGunzip } from 'node:zlib';
import { Transform } from 'node:stream';

// Basic file compression
await pipeline(
  createReadStream('input.txt'),
  createGzip(),
  createWriteStream('input.txt.gz'),
);

// Transform stream (process line by line)
const upperCaseTransform = new Transform({
  transform(chunk, encoding, callback) {
    this.push(chunk.toString().toUpperCase());
    callback();
  },
});

// Async generator for memory-efficient processing
async function* readLines(path: string): AsyncGenerator<string> {
  const stream = createReadStream(path, { encoding: 'utf-8' });
  let buffer = '';

  for await (const chunk of stream) {
    buffer += chunk;
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      yield line;
    }
  }
  if (buffer) yield buffer;
}

// Usage
for await (const line of readLines('large-file.txt')) {
  await processLine(line);
}

// ═══════════════════════════════════════════════════════════════
// MODULE PATTERNS (ESM)
// ═══════════════════════════════════════════════════════════════

// Named exports (preferred)
export function doSomething() {}
export const CONFIG = { port: 3000 };
export type User = { id: string; name: string };

// Default export (use sparingly)
export default class MyService {}

// Re-exports (barrel files)
// src/services/index.ts
export { UserService } from './user.service.js';
export { AuthService } from './auth.service.js';
export type { User, CreateUserInput } from './types.js';

// Dynamic imports (code splitting)
const module = await import('./heavy-module.js');

// Import JSON (requires resolveJsonModule)
import config from './config.json' with { type: 'json' };

// Import with assertions
import data from './data.json' with { type: 'json' };

// ═══════════════════════════════════════════════════════════════
// ERROR HANDLING PATTERNS
// ═══════════════════════════════════════════════════════════════

// Result type pattern
type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

async function safeOperation(): Promise<Result<Data>> {
  try {
    const data = await riskyOperation();
    return { success: true, data };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error : new Error('Unknown error'),
    };
  }
}

// Custom error classes
class AppError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly statusCode: number = 500,
  ) {
    super(message);
    this.name = this.constructor.name;
    Error.captureStackTrace(this, this.constructor);
  }
}

class NotFoundError extends AppError {
  constructor(resource: string) {
    super(`${resource} not found`, 'NOT_FOUND', 404);
  }
}

class ValidationError extends AppError {
  constructor(message: string) {
    super(message, 'VALIDATION_ERROR', 400);
  }
}

// ═══════════════════════════════════════════════════════════════
// TYPE PATTERNS
// ═══════════════════════════════════════════════════════════════

// Type guards
function isUser(value: unknown): value is User {
  return (
    typeof value === 'object' &&
    value !== null &&
    'id' in value &&
    'email' in value &&
    typeof (value as User).id === 'string'
  );
}

// Discriminated unions
type ApiResponse<T> =
  | { status: 'success'; data: T }
  | { status: 'error'; error: string }
  | { status: 'loading' };

// Branded types (nominal typing)
type UserId = string & { readonly __brand: 'UserId' };
type PostId = string & { readonly __brand: 'PostId' };

function createUserId(id: string): UserId {
  return id as UserId;
}

// Utility types
type Readonly<T> = { readonly [P in keyof T]: T[P] };
type Partial<T> = { [P in keyof T]?: T[P] };
type Required<T> = { [P in keyof T]-?: T[P] };
type Pick<T, K extends keyof T> = { [P in K]: T[P] };
type Omit<T, K extends keyof T> = Pick<T, Exclude<keyof T, K>>;
```

### Project Structure

```
project/
├── src/
│   ├── config/               # Configuration management
│   │   ├── index.ts          # Config loader with Zod validation
│   │   └── env.ts            # Environment variable schema
│   │
│   ├── types/                # TypeScript type definitions
│   │   ├── index.ts          # Main type exports
│   │   ├── user.types.ts     # Domain-specific types
│   │   └── api.types.ts      # API request/response types
│   │
│   ├── utils/                # Utility functions
│   │   ├── logger.ts         # Pino logger setup
│   │   ├── errors.ts         # Custom error classes
│   │   └── validation.ts     # Zod schemas and validators
│   │
│   ├── repositories/         # Data access layer
│   │   ├── user.repository.ts
│   │   └── index.ts          # Repository interfaces
│   │
│   ├── services/             # Business logic layer
│   │   ├── user.service.ts
│   │   ├── auth.service.ts
│   │   └── index.ts
│   │
│   ├── middleware/           # Express/Fastify middleware
│   │   ├── auth.middleware.ts
│   │   ├── error.middleware.ts
│   │   └── validation.middleware.ts
│   │
│   ├── routes/               # API route definitions
│   │   ├── user.routes.ts
│   │   ├── auth.routes.ts
│   │   └── index.ts
│   │
│   ├── controllers/          # Route handlers (optional)
│   │   └── user.controller.ts
│   │
│   ├── app.ts                # Application factory
│   └── index.ts              # Entry point
│
├── tests/
│   ├── unit/                 # Unit tests (mirror src structure)
│   │   ├── services/
│   │   │   └── user.service.test.ts
│   │   └── utils/
│   │       └── validation.test.ts
│   │
│   ├── integration/          # Integration tests
│   │   ├── api/
│   │   │   └── users.api.test.ts
│   │   └── setup.ts          # Test database setup
│   │
│   └── e2e/                  # End-to-end tests
│       └── user-flow.e2e.test.ts
│
├── prisma/                   # Prisma ORM (if using)
│   ├── schema.prisma
│   └── migrations/
│
├── scripts/                  # Build/deploy scripts
│   ├── seed.ts               # Database seeding
│   └── migrate.ts            # Migration runner
│
├── docs/                     # Generated documentation (gitignored)
│
├── .env.example              # Environment template
├── .gitignore
├── biome.json                # Linter & formatter config
├── tsconfig.json             # TypeScript configuration
├── typedoc.json              # Documentation config
├── vitest.config.ts          # Test configuration
├── package.json
└── README.md
```

### Package.json Scripts Reference

```json
{
  "name": "modern-node-app",
  "version": "1.0.0",
  "type": "module",
  "engines": {
    "node": ">=22.0.0",
    "npm": ">=10.0.0"
  },
  "scripts": {
    "dev": "tsx watch src/index.ts",
    "build": "tsc",
    "start": "node dist/index.js",
    "test": "vitest",
    "test:watch": "vitest --watch",
    "test:coverage": "vitest run --coverage",
    "test:ui": "vitest --ui",
    "lint": "biome check .",
    "lint:fix": "biome check --apply .",
    "format": "biome format --write .",
    "typecheck": "tsc --noEmit",
    "docs": "typedoc --out docs src/",
    "docs:check": "typedoc --emit none --validation.notDocumented true",
    "docs:serve": "typedoc --out docs src/ && npx serve docs",
    "verify": "npm run typecheck && npm run docs:check && npm run lint && npm run test",
    "db:migrate": "prisma migrate dev",
    "db:generate": "prisma generate",
    "db:studio": "prisma studio",
    "db:seed": "tsx scripts/seed.ts",
    "clean": "rm -rf dist docs coverage"
  }
}
```

---

## References

- [Node.js Best Practices](https://github.com/goldbergyoni/nodebestpractices)
- [TypeScript Deep Dive](https://basarat.gitbook.io/typescript/)
- [Zod Documentation](https://zod.dev/)
- [Fastify Documentation](https://fastify.dev/)
- [Prisma Best Practices](https://www.prisma.io/docs/guides)
- [Biome](https://biomejs.dev/)
- [Vitest](https://vitest.dev/)

---

**End of Node.js & TypeScript Development Guidelines**
