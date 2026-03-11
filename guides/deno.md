# Deno Development Guidelines
Modern development practices for building secure, performant, and maintainable applications with Deno 2.1+.

---

**Agent Profile**: The Deno Security Architect
**Role**: Senior Full-Stack Engineer & Deno Performance Specialist
**Objective**: Generate production-ready, type-safe, secure-by-default, highly performant, and maintainable Deno applications.
**Tools**: Deno 2.1+, TypeScript 5.x, JSR, Fresh, Oak, Standard Library, Built-in tooling.
**Companion Guides**: typescript.md, nodejs.md, secure-coding.md, testing.md

---

## 1. Core Philosophies: DENO-SECURE

The agent must adhere to the **DENO-SECURE** principles for every Deno project:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation using Deno.test (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Security-First**: Mandatory vulnerability scanning, dependency auditing, and supply chain integrity checks using `deno audit`.

### DENO-SECURE Principles

- **D**efault Secure - Explicit permissions, no file/network access without flags.
- **E**SM Native - Native ES modules, no build step, direct TypeScript execution.
- **N**o Config - Zero configuration by default, works out of the box.
- **O**ptimized - Built on Rust + V8, optimized for modern JavaScript performance.

- **S**tandard Library - Curated, audited standard library modules.
- **E**xplicit Permissions - Granular security with `--allow-*` flags.
- **C**omplete Tooling - Built-in formatter, linter, test runner, bundler, LSP.
- **U**RL Imports - Direct URL imports, no package.json or node_modules.
- **R**ust-Powered - Fast, memory-safe runtime built with Rust.
- **E**dge-Ready - Deploy to edge with Deno Deploy, global distribution.

**Verified Code**: Agent-generated code MUST pass `deno check` and `deno audit` before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated Deno code compiles, passes tests, and is secure before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY Deno code, the agent MUST:**

1. **Syntax & Type Check**:
   ```bash
   # Verify TypeScript compiles without errors
   deno check main.ts
   # Exit code MUST be 0
   ```

2. **Linting Check**:
   ```bash
   # Run built-in linter
   deno lint
   # Fix all errors, address warnings
   ```

3. **Security & Dependency Verification (MANDATORY)**:
   ```bash
   # Scan for vulnerabilities in dependencies
   deno audit
   ```
   - **MUST** have 0 high/critical vulnerabilities.
   - Supply chain integrity (`deno.lock`) MUST be verified.

4. **Test Execution**:
   ```bash
   # Run all tests with frozen lockfile
   deno test --frozen --allow-read --allow-net
   # Exit code MUST be 0
   ```

5. **Documentation Verification**:
   - All public APIs have JSDoc comments.
   - Examples in documentation are tested with `deno test --doc`.

#### Error Correction Process

If verification fails:

1. **Identify the error**: Read the full Deno error message (often includes suggested fixes).
2. **Locate the source**: Identify which module or permission failed.
3. **Fix the root cause**:
   - Permission error? Add specific `--allow-*` flag to task.
   - Vulnerability? Update dependency in `deno.json`.
4. **Re-verify**: Run check, lint, and tests again.

---

## 3. Mandatory Setup Requirements

### A. Deno Installation & Version

**Version**: Use Deno 2.1+ (latest stable)

```bash
# Install Deno (official installer)
curl -fsSL https://deno.land/install.sh | sh

# Verify installation
deno --version
# deno 2.1.0 (release, x86_64-unknown-linux-gnu)
# v8 12.9
# typescript 5.6

# Upgrade to latest
deno upgrade
```

### B. Project Configuration (deno.json)

**Minimal Configuration** - Deno works without configuration, but use `deno.json` for customization:

```json
{
  "name": "@myorg/myproject",
  "version": "1.0.0",
  "exports": "./mod.ts",

  "tasks": {
    "dev": "deno run --watch --allow-net --allow-read --allow-env main.ts",
    "start": "deno run --allow-net --allow-read --allow-env main.ts",
    "test": "deno test --frozen --allow-read --allow-net --coverage=coverage",
    "audit": "deno audit",
    "lint": "deno lint",
    "fmt": "deno fmt"
  },

  "imports": {
    "@/": "./src/",
    "@std/": "jsr:@std/",
    "zod": "npm:zod@^3.23.0"
  },

  "compilerOptions": {
    "strict": true
  },

  "lock": true
}
```

---

## 4. Test-Driven Development (TDD) Protocol (MANDATORY)
    "@/": "./src/",
    "@std/": "https://deno.land/std@0.224.0/",
    "zod": "npm:zod@^3.22.4"
  },

  "compilerOptions": {
    "lib": ["deno.window", "dom", "dom.iterable"],
    "strict": true,
    "allowJs": false,
    "checkJs": false
  },

  "lint": {
    "rules": {
      "tags": ["recommended"],
      "include": ["ban-untagged-todo"],
      "exclude": ["no-unused-vars"]
    },
    "exclude": ["dist/", "coverage/"]
  },

  "fmt": {
    "useTabs": false,
    "lineWidth": 100,
    "indentWidth": 2,
    "semiColons": true,
    "singleQuote": true,
    "proseWrap": "preserve",
    "exclude": ["dist/", "coverage/"]
  },

  "test": {
    "include": ["**/*_test.ts"],
    "exclude": ["dist/"]
  },

  "publish": {
    "include": ["mod.ts", "src/", "README.md", "LICENSE"],
    "exclude": ["**/*_test.ts", "scripts/"]
  }
}
```

### C. Project Structure

**Standard directory layout** for Deno projects:

```
project/
├── src/
│   ├── config/           # Configuration management
│   │   └── mod.ts
│   ├── types/            # TypeScript type definitions
│   │   └── mod.ts
│   ├── utils/            # Utility functions
│   │   ├── logger.ts
│   │   └── validation.ts
│   ├── services/         # Business logic
│   │   └── user_service.ts
│   ├── repositories/     # Data access layer
│   │   └── user_repository.ts
│   ├── routes/           # API routes (Oak/Fresh)
│   │   └── user_routes.ts
│   └── middleware/       # HTTP middleware
│       └── auth_middleware.ts
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── benchmarks/
│   └── user_bench.ts
├── scripts/
│   └── seed.ts
├── main.ts               # Application entry point
├── mod.ts                # Library entry point (for JSR publishing)
├── deno.json             # Deno configuration
├── deno.lock             # Lock file (committed)
├── import_map.json       # Alternative to deno.json imports
├── .env.example
├── .gitignore
└── README.md
```

### D. Essential Dependencies

**Standard Library (No Installation Required)**:

```typescript
// ✅ CORRECT - Use Deno Standard Library
import { assertEquals } from '@std/assert';
import { serve } from '@std/http/server';
import { parse } from '@std/flags';
import { join } from '@std/path';
import { crypto } from '@std/crypto';
import { delay } from '@std/async/delay';
import { readLines } from '@std/io';

// Or with explicit version pinning
import { assertEquals } from 'https://deno.land/std@0.224.0/assert/mod.ts';
```

**Common External Dependencies (JSR/npm)**:

```typescript
// JSR packages (preferred for Deno)
import { z } from 'jsr:@std/zod@^3.22';
import { Hono } from 'jsr:@hono/hono@^4.0';

// npm packages (when needed)
import { z } from 'npm:zod@^3.22.4';
import Stripe from 'npm:stripe@^14.0.0';

// Import maps in deno.json (recommended)
// "imports": {
//   "zod": "npm:zod@^3.22.4",
//   "stripe": "npm:stripe@^14.0.0"
// }
```

---

## 3. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### A. TDD Cycle with Deno.test

```typescript
// ═══════════════════════════════════════════════════════════════
// STEP 1: RED - Write failing test first
// ═══════════════════════════════════════════════════════════════

// tests/unit/user_service_test.ts
import { assertEquals, assertRejects } from '@std/assert';
import { describe, it, beforeEach, afterEach } from '@std/testing/bdd';
import { stub, assertSpyCall, assertSpyCalls } from '@std/testing/mock';
import { UserService } from '../../src/services/user_service.ts';
import type { UserRepository } from '../../src/repositories/user_repository.ts';

describe('UserService', () => {
  let service: UserService;
  let mockRepository: UserRepository;

  beforeEach(() => {
    mockRepository = {
      findById: stub(),
      findByEmail: stub(),
      create: stub(),
      update: stub(),
      delete: stub(),
    } as unknown as UserRepository;

    service = new UserService(mockRepository);
  });

  describe('createUser', () => {
    it('should create a user with valid input', async () => {
      const input = { email: 'test@example.com', name: 'Test User' };
      const expected = { id: '123', ...input, createdAt: new Date() };

      stub(mockRepository, 'findByEmail').returns(Promise.resolve(null));
      stub(mockRepository, 'create').returns(Promise.resolve(expected));

      const result = await service.createUser(input);

      assertEquals(result.success, true);
      if (result.success) {
        assertEquals(result.data.email, input.email);
        assertEquals(result.data.name, input.name);
      }
      assertSpyCalls(mockRepository.create as any, 1);
    });

    it('should return error if email already exists', async () => {
      const input = { email: 'existing@example.com', name: 'Test' };
      const existingUser = { id: '456', ...input, createdAt: new Date() };

      stub(mockRepository, 'findByEmail').returns(Promise.resolve(existingUser));

      const result = await service.createUser(input);

      assertEquals(result.success, false);
      if (!result.success) {
        assertEquals(result.error.message.includes('already exists'), true);
      }
      assertSpyCalls(mockRepository.create as any, 0);
    });
  });
});

// Run: deno test
// ❌ FAILS - UserService doesn't exist yet

// ═══════════════════════════════════════════════════════════════
// STEP 2: GREEN - Write minimal implementation
// ═══════════════════════════════════════════════════════════════

// src/services/user_service.ts
import type { UserRepository } from '../repositories/user_repository.ts';
import type { User, CreateUserInput, Result } from '../types/mod.ts';

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

// Run: deno test
// ✅ PASSES - tests pass

// ═══════════════════════════════════════════════════════════════
// STEP 3: REFACTOR - Improve with validation and logging
// ═══════════════════════════════════════════════════════════════

// src/services/user_service.ts (refactored)
import { z } from 'npm:zod@^3.22.4';
import type { UserRepository } from '../repositories/user_repository.ts';
import type { User, CreateUserInput, Result } from '../types/mod.ts';
import { ValidationError } from '../utils/errors.ts';
import { logger } from '../utils/logger.ts';

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

// Run: deno test
// ✅ PASSES - tests still pass after refactoring
```

### B. Testing Patterns with Deno.test

```typescript
// tests/unit/calculator_test.ts
import { assertEquals, assertThrows } from '@std/assert';

// Simple test
Deno.test('add() should sum two numbers', () => {
  assertEquals(add(2, 3), 5);
  assertEquals(add(-1, 1), 0);
  assertEquals(add(0, 0), 0);
});

// Async test
Deno.test('fetchUser() should return user data', async () => {
  const user = await fetchUser('123');
  assertEquals(user.id, '123');
  assertEquals(typeof user.name, 'string');
});

// Test with permissions
Deno.test({
  name: 'readConfig() should load configuration',
  permissions: { read: true, env: true },
  fn: async () => {
    const config = await readConfig('./config.json');
    assertEquals(typeof config.port, 'number');
  },
});

// Test that should fail
Deno.test({
  name: 'unauthorized access should throw',
  permissions: { net: false },
  fn: async () => {
    await assertRejects(
      async () => await fetch('https://api.example.com'),
      Deno.errors.PermissionDenied,
    );
  },
});

// Ignored test (for work in progress)
Deno.test({
  name: 'complex feature not yet implemented',
  ignore: true,
  fn: () => {
    // Will be implemented later
  },
});

// Test with setup and teardown
Deno.test('database operations', async (t) => {
  const db = await setupDatabase();

  await t.step('insert record', async () => {
    const id = await db.insert({ name: 'Test' });
    assertEquals(typeof id, 'string');
  });

  await t.step('query record', async () => {
    const record = await db.find({ name: 'Test' });
    assertEquals(record.name, 'Test');
  });

  await db.close();
});
```

### C. BDD-Style Testing

```typescript
// tests/unit/user_service_bdd_test.ts
import { describe, it, beforeEach, afterEach } from '@std/testing/bdd';
import { assertEquals, assertExists } from '@std/assert';

describe('UserService', () => {
  let service: UserService;

  beforeEach(() => {
    service = new UserService();
  });

  afterEach(() => {
    // Cleanup
  });

  describe('createUser', () => {
    it('creates a user with valid input', async () => {
      const result = await service.createUser({
        email: 'test@example.com',
        name: 'Test User',
      });

      assertEquals(result.success, true);
      assertExists(result.data?.id);
    });

    it('returns error for invalid email', async () => {
      const result = await service.createUser({
        email: 'invalid',
        name: 'Test',
      });

      assertEquals(result.success, false);
      assertExists(result.error);
    });
  });
});
```

### D. Mocking and Stubbing

```typescript
// tests/unit/api_client_test.ts
import { stub, assertSpyCall } from '@std/testing/mock';
import { FakeTime } from '@std/testing/time';

Deno.test('API client with mocked fetch', async () => {
  const fetchStub = stub(
    globalThis,
    'fetch',
    () => Promise.resolve(new Response(JSON.stringify({ id: '123' }))),
  );

  try {
    const client = new ApiClient();
    const result = await client.getUser('123');

    assertEquals(result.id, '123');
    assertSpyCall(fetchStub, 0, {
      args: ['https://api.example.com/users/123'],
    });
  } finally {
    fetchStub.restore();
  }
});

Deno.test('time-dependent function with FakeTime', () => {
  using time = new FakeTime();

  const start = Date.now();
  time.tick(1000); // Advance time by 1 second
  const end = Date.now();

  assertEquals(end - start, 1000);
});
```

---

## 4. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow Example

```typescript
// ═══════════════════════════════════════════════════════════════
// Bug Report #1523: parseDate() fails with ISO strings containing
// timezone offsets like "2024-01-15T10:30:00+05:30"
// ═══════════════════════════════════════════════════════════════

// STEP 1-2: Write test that reproduces the bug
// tests/unit/date_utils_test.ts

import { assertEquals } from '@std/assert';

Deno.test('parseDate() should handle ISO strings with timezone - Bug #1523', () => {
  // Bug: parseDate() throws error on timezone offsets
  // Discovered: 2026-02-06
  // Root cause: Regex doesn't account for timezone format

  const isoWithTimezone = '2024-01-15T10:30:00+05:30';
  const result = parseDate(isoWithTimezone);

  assertEquals(result instanceof Date, true);
  assertEquals(result.getFullYear(), 2024);
  assertEquals(result.getMonth(), 0); // January
  assertEquals(result.getDate(), 15);
});

Deno.test('parseDate() should handle various timezone formats - Bug #1523', () => {
  const testCases = [
    '2024-01-15T10:30:00Z',           // UTC
    '2024-01-15T10:30:00+00:00',      // UTC explicit
    '2024-01-15T10:30:00-08:00',      // PST
    '2024-01-15T10:30:00+05:30',      // IST
  ];

  for (const isoString of testCases) {
    const result = parseDate(isoString);
    assertEquals(result instanceof Date, true, `Failed for: ${isoString}`);
  }
});

// Run: deno test
// ❌ FAILS - parseDate crashes on timezone offsets

// ═══════════════════════════════════════════════════════════════
// STEP 3-4: Fix the bug
// ═══════════════════════════════════════════════════════════════

// src/utils/date_utils.ts

/**
 * Parses an ISO 8601 date string into a Date object.
 *
 * Supports various ISO formats including timezone offsets.
 *
 * @param isoString - ISO 8601 formatted date string
 * @returns Parsed Date object
 * @throws {Error} If the date string is invalid
 *
 * @remarks
 * Fix for Bug #1523: Now properly handles timezone offsets
 * including formats like "+05:30", "-08:00", and "Z".
 *
 * @example
 * ```ts
 * const date1 = parseDate('2024-01-15T10:30:00Z');
 * const date2 = parseDate('2024-01-15T10:30:00+05:30');
 * ```
 */
export function parseDate(isoString: string): Date {
  // FIX for Bug #1523: Use native Date parsing which handles timezones
  // Previously: Custom regex didn't handle timezone offsets
  const date = new Date(isoString);

  if (isNaN(date.getTime())) {
    throw new Error(`Invalid ISO date string: ${isoString}`);
  }

  return date;
}

// Run: deno test
// ✅ PASSES - bug fixed, regression prevented forever
```

---

## 5. Permission Model (MANDATORY)

### A. Permission Best Practices

**CRITICAL: Always use the principle of least privilege.**

```typescript
// ✅ CORRECT - Minimal permissions
// deno run --allow-net=api.example.com --allow-read=./data main.ts

// ✅ CORRECT - Granular permissions in deno.json
{
  "tasks": {
    "start": "deno run --allow-net=:8000 --allow-read=./public main.ts",
    "dev": "deno run --watch --allow-net --allow-read --allow-env=PORT,DATABASE_URL main.ts"
  }
}

// ❌ WRONG - Overly permissive
// deno run --allow-all main.ts
// deno run -A main.ts

// ❌ WRONG - Unrestricted network access
// deno run --allow-net main.ts  // Allows access to any domain
```

### B. Programmatic Permission Checks

```typescript
// src/utils/permissions.ts

/**
 * Checks if the program has specific permissions and requests them if needed.
 */
export async function ensurePermissions(): Promise<void> {
  // Check read permission
  const readStatus = await Deno.permissions.query({ name: 'read', path: './data' });
  if (readStatus.state !== 'granted') {
    const result = await Deno.permissions.request({ name: 'read', path: './data' });
    if (result.state !== 'granted') {
      throw new Error('Read permission required for ./data');
    }
  }

  // Check network permission
  const netStatus = await Deno.permissions.query({
    name: 'net',
    host: 'api.example.com'
  });
  if (netStatus.state !== 'granted') {
    throw new Error('Network permission required for api.example.com');
  }
}

/**
 * Runs a function with temporary elevated permissions.
 */
export async function withPermissions<T>(
  permissions: Deno.PermissionDescriptor[],
  fn: () => Promise<T>,
): Promise<T> {
  // Request permissions
  const granted = await Promise.all(
    permissions.map(p => Deno.permissions.request(p))
  );

  if (granted.some(g => g.state !== 'granted')) {
    throw new Error('Required permissions not granted');
  }

  try {
    return await fn();
  } finally {
    // Optionally revoke permissions after use
    // Note: Deno doesn't support permission revocation yet
  }
}

// Usage
await withPermissions(
  [
    { name: 'read', path: './config.json' },
    { name: 'env', variable: 'DATABASE_URL' },
  ],
  async () => {
    const config = await Deno.readTextFile('./config.json');
    const dbUrl = Deno.env.get('DATABASE_URL');
    // Do work with config and env
  }
);
```

### C. Permission Matrix for Common Operations

| Operation | Required Permission | Example |
|-----------|-------------------|---------|
| Read file | `--allow-read=path` | `--allow-read=./data` |
| Write file | `--allow-write=path` | `--allow-write=./logs` |
| HTTP server | `--allow-net=:port` | `--allow-net=:8000` |
| HTTP client | `--allow-net=host` | `--allow-net=api.example.com` |
| Environment variables | `--allow-env=VAR` | `--allow-env=PORT,DB_URL` |
| Run subprocess | `--allow-run=cmd` | `--allow-run=git,npm` |
| FFI (unsafe) | `--allow-ffi=lib` | `--allow-ffi=./native.so` |
| High-resolution time | `--allow-hrtime` | `--allow-hrtime` |
| System info | `--allow-sys=info` | `--allow-sys=osRelease` |

---

## 6. Type Safety & Modern TypeScript (MANDATORY)

### A. Strict Typing with Deno

```typescript
// ✅ CORRECT - Native TypeScript with strict typing

/**
 * User entity representing a system user.
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
 * User role enumeration.
 */
export type UserRole = 'admin' | 'user' | 'guest';

/**
 * Result type for operations that can fail gracefully.
 */
export type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

/**
 * Type guard to check if a value is a User.
 */
export function isUser(value: unknown): value is User {
  return (
    typeof value === 'object' &&
    value !== null &&
    'id' in value &&
    typeof (value as User).id === 'string' &&
    'email' in value &&
    typeof (value as User).email === 'string'
  );
}

// ✅ CORRECT - Using Result type for error handling
export async function fetchUser(id: string): Promise<Result<User>> {
  try {
    const response = await fetch(`https://api.example.com/users/${id}`);

    if (!response.ok) {
      return {
        success: false,
        error: new Error(`HTTP ${response.status}: ${response.statusText}`),
      };
    }

    const data = await response.json();

    if (!isUser(data)) {
      return {
        success: false,
        error: new Error('Invalid user data received'),
      };
    }

    return { success: true, data };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error : new Error('Unknown error'),
    };
  }
}

// ❌ WRONG - Using any
function processData(data: any): any {
  return data.value;
}
```

### B. Validation with Zod

```typescript
// src/schemas/user_schema.ts
import { z } from 'npm:zod@^3.22.4';

/**
 * User validation schema.
 */
export const userSchema = z.object({
  id: z.string().uuid(),
  email: z.string().email(),
  name: z.string().min(1).max(100),
  role: z.enum(['admin', 'user', 'guest']).default('user'),
  createdAt: z.coerce.date(),
  updatedAt: z.coerce.date(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

/**
 * Infer User type from schema.
 */
export type User = z.infer<typeof userSchema>;

/**
 * Create user input schema (omits system fields).
 */
export const createUserSchema = userSchema.omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type CreateUserInput = z.infer<typeof createUserSchema>;

/**
 * Environment variables schema.
 */
export const envSchema = z.object({
  PORT: z.coerce.number().int().positive().default(8000),
  DATABASE_URL: z.string().url(),
  JWT_SECRET: z.string().min(32),
  LOG_LEVEL: z.enum(['debug', 'info', 'warn', 'error']).default('info'),
  NODE_ENV: z.enum(['development', 'staging', 'production']).default('development'),
});

export const env = envSchema.parse(Deno.env.toObject());

// Usage in API handler
export async function createUserHandler(req: Request): Promise<Response> {
  const body = await req.json();
  const result = createUserSchema.safeParse(body);

  if (!result.success) {
    return Response.json(
      {
        success: false,
        error: {
          code: 'VALIDATION_ERROR',
          details: result.error.format(),
        },
      },
      { status: 400 }
    );
  }

  const user = await createUser(result.data);
  return Response.json({ success: true, data: user }, { status: 201 });
}
```

---

## 7. Web Framework Patterns

### A. Fresh Framework (Recommended for Web Apps)

Fresh is a next-generation web framework for Deno with islands architecture.

```typescript
// fresh.config.ts
import { defineConfig } from '$fresh/server.ts';

export default defineConfig({
  plugins: [],
  port: 8000,
});

// routes/index.tsx
import { Handlers, PageProps } from '$fresh/server.ts';

interface Data {
  users: User[];
}

export const handler: Handlers<Data> = {
  async GET(_req, ctx) {
    const users = await fetchUsers();
    return ctx.render({ users });
  },
};

export default function Home({ data }: PageProps<Data>) {
  return (
    <div class="container mx-auto px-4">
      <h1 class="text-4xl font-bold">Users</h1>
      <ul>
        {data.users.map((user) => (
          <li key={user.id}>
            {user.name} ({user.email})
          </li>
        ))}
      </ul>
    </div>
  );
}

// routes/api/users.ts
import { Handlers } from '$fresh/server.ts';
import { z } from 'npm:zod@^3.22.4';

const createUserSchema = z.object({
  email: z.string().email(),
  name: z.string().min(1).max(100),
});

export const handler: Handlers = {
  async POST(req) {
    const body = await req.json();
    const result = createUserSchema.safeParse(body);

    if (!result.success) {
      return Response.json(
        { success: false, error: result.error.format() },
        { status: 400 }
      );
    }

    const user = await createUser(result.data);
    return Response.json({ success: true, data: user }, { status: 201 });
  },

  async GET() {
    const users = await fetchUsers();
    return Response.json({ success: true, data: users });
  },
};

// islands/Counter.tsx (interactive island)
import { signal } from '@preact/signals';

const count = signal(0);

export default function Counter() {
  return (
    <div class="flex gap-2">
      <button
        class="px-4 py-2 bg-blue-500 text-white rounded"
        onClick={() => count.value -= 1}
      >
        -
      </button>
      <span class="text-2xl">{count.value}</span>
      <button
        class="px-4 py-2 bg-blue-500 text-white rounded"
        onClick={() => count.value += 1}
      >
        +
      </button>
    </div>
  );
}
```

### B. Oak Framework (Recommended for APIs)

Oak is a middleware framework for Deno's HTTP server, similar to Express/Koa.

```typescript
// main.ts
import { Application, Router } from 'https://deno.land/x/oak@v12.6.1/mod.ts';
import { oakCors } from 'https://deno.land/x/cors@v1.2.2/mod.ts';
import { userRouter } from './routes/users.ts';
import { errorHandler } from './middleware/error_handler.ts';
import { logger } from './middleware/logger.ts';
import { authenticate } from './middleware/auth.ts';

const app = new Application();

// Global middleware
app.use(logger);
app.use(errorHandler);
app.use(oakCors());

// Routes
const router = new Router();

router.get('/health', (ctx) => {
  ctx.response.body = { status: 'ok', timestamp: new Date().toISOString() };
});

// Protected routes
app.use(router.routes());
app.use(router.allowedMethods());

// API routes
app.use(authenticate); // Authentication middleware
app.use(userRouter.routes());
app.use(userRouter.allowedMethods());

// Start server
console.log('Server running on http://localhost:8000');
await app.listen({ port: 8000 });

// routes/users.ts
import { Router } from 'https://deno.land/x/oak@v12.6.1/mod.ts';
import { createUser, getUser, listUsers } from '../controllers/users.ts';

export const userRouter = new Router({ prefix: '/api/users' });

userRouter
  .get('/', listUsers)
  .get('/:id', getUser)
  .post('/', createUser);

// controllers/users.ts
import type { Context } from 'https://deno.land/x/oak@v12.6.1/mod.ts';
import { z } from 'npm:zod@^3.22.4';
import { UserService } from '../services/user_service.ts';

const createUserSchema = z.object({
  email: z.string().email(),
  name: z.string().min(1).max(100),
});

/**
 * Creates a new user.
 */
export async function createUser(ctx: Context) {
  const body = await ctx.request.body({ type: 'json' }).value;
  const result = createUserSchema.safeParse(body);

  if (!result.success) {
    ctx.response.status = 400;
    ctx.response.body = {
      success: false,
      error: { code: 'VALIDATION_ERROR', details: result.error.format() },
    };
    return;
  }

  const userService = new UserService();
  const userResult = await userService.createUser(result.data);

  if (!userResult.success) {
    ctx.response.status = 400;
    ctx.response.body = {
      success: false,
      error: { message: userResult.error.message },
    };
    return;
  }

  ctx.response.status = 201;
  ctx.response.body = { success: true, data: userResult.data };
}

/**
 * Retrieves a user by ID.
 */
export async function getUser(ctx: Context) {
  const { id } = ctx.params;

  if (!id) {
    ctx.response.status = 400;
    ctx.response.body = { success: false, error: { message: 'User ID required' } };
    return;
  }

  const userService = new UserService();
  const result = await userService.getUserById(id);

  if (!result.success) {
    ctx.response.status = 404;
    ctx.response.body = {
      success: false,
      error: { message: result.error.message },
    };
    return;
  }

  ctx.response.body = { success: true, data: result.data };
}

/**
 * Lists all users with pagination.
 */
export async function listUsers(ctx: Context) {
  const limit = Number(ctx.request.url.searchParams.get('limit')) || 20;
  const offset = Number(ctx.request.url.searchParams.get('offset')) || 0;

  const userService = new UserService();
  const result = await userService.listUsers({ limit, offset });

  ctx.response.body = {
    success: true,
    data: result.data,
    pagination: { limit, offset, total: result.total },
  };
}

// middleware/auth.ts
import type { Context, Next } from 'https://deno.land/x/oak@v12.6.1/mod.ts';
import { verify } from 'https://deno.land/x/djwt@v3.0.1/mod.ts';

const JWT_SECRET = Deno.env.get('JWT_SECRET')!;
const key = await crypto.subtle.importKey(
  'raw',
  new TextEncoder().encode(JWT_SECRET),
  { name: 'HMAC', hash: 'SHA-256' },
  false,
  ['sign', 'verify']
);

export async function authenticate(ctx: Context, next: Next) {
  const authHeader = ctx.request.headers.get('authorization');

  if (!authHeader?.startsWith('Bearer ')) {
    ctx.response.status = 401;
    ctx.response.body = { success: false, error: { message: 'Unauthorized' } };
    return;
  }

  const token = authHeader.substring(7);

  try {
    const payload = await verify(token, key);
    ctx.state.user = payload;
    await next();
  } catch (error) {
    ctx.response.status = 401;
    ctx.response.body = {
      success: false,
      error: { message: 'Invalid or expired token' },
    };
  }
}

// middleware/error_handler.ts
import type { Context, Next } from 'https://deno.land/x/oak@v12.6.1/mod.ts';

export async function errorHandler(ctx: Context, next: Next) {
  try {
    await next();
  } catch (error) {
    console.error('Unhandled error:', error);

    ctx.response.status = 500;
    ctx.response.body = {
      success: false,
      error: {
        code: 'INTERNAL_ERROR',
        message: 'An unexpected error occurred',
      },
    };
  }
}

// middleware/logger.ts
import type { Context, Next } from 'https://deno.land/x/oak@v12.6.1/mod.ts';

export async function logger(ctx: Context, next: Next) {
  const start = Date.now();
  await next();
  const duration = Date.now() - start;

  console.log(
    `${ctx.request.method} ${ctx.request.url.pathname} - ${ctx.response.status} - ${duration}ms`
  );
}
```

### C. Native HTTP Server (Minimal)

```typescript
// main.ts - Native Deno HTTP server
import { serveHandler } from './handler.ts';

const port = 8000;

console.log(`HTTP server running on http://localhost:${port}/`);

Deno.serve({ port }, serveHandler);

// handler.ts
import { router } from './router.ts';

export function serveHandler(req: Request): Response | Promise<Response> {
  return router(req);
}

// router.ts
import { createUser, getUser, listUsers } from './controllers/users.ts';

export function router(req: Request): Response | Promise<Response> {
  const url = new URL(req.url);
  const { pathname } = url;

  // Health check
  if (pathname === '/health' && req.method === 'GET') {
    return Response.json({ status: 'ok', timestamp: new Date().toISOString() });
  }

  // User routes
  if (pathname === '/api/users') {
    if (req.method === 'GET') return listUsers(req);
    if (req.method === 'POST') return createUser(req);
  }

  if (pathname.startsWith('/api/users/')) {
    const id = pathname.split('/')[3];
    if (req.method === 'GET') return getUser(req, id);
  }

  // 404
  return Response.json(
    { success: false, error: { message: 'Not found' } },
    { status: 404 }
  );
}

// controllers/users.ts
import { z } from 'npm:zod@^3.22.4';
import { UserService } from '../services/user_service.ts';

const createUserSchema = z.object({
  email: z.string().email(),
  name: z.string().min(1).max(100),
});

export async function createUser(req: Request): Promise<Response> {
  const body = await req.json();
  const result = createUserSchema.safeParse(body);

  if (!result.success) {
    return Response.json(
      {
        success: false,
        error: { code: 'VALIDATION_ERROR', details: result.error.format() },
      },
      { status: 400 }
    );
  }

  const service = new UserService();
  const userResult = await service.createUser(result.data);

  if (!userResult.success) {
    return Response.json(
      { success: false, error: { message: userResult.error.message } },
      { status: 400 }
    );
  }

  return Response.json(
    { success: true, data: userResult.data },
    { status: 201 }
  );
}

export async function getUser(_req: Request, id: string): Promise<Response> {
  const service = new UserService();
  const result = await service.getUserById(id);

  if (!result.success) {
    return Response.json(
      { success: false, error: { message: result.error.message } },
      { status: 404 }
    );
  }

  return Response.json({ success: true, data: result.data });
}

export async function listUsers(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const limit = Number(url.searchParams.get('limit')) || 20;
  const offset = Number(url.searchParams.get('offset')) || 0;

  const service = new UserService();
  const result = await service.listUsers({ limit, offset });

  return Response.json({
    success: true,
    data: result.data,
    pagination: { limit, offset, total: result.total },
  });
}
```

---

## 8. Database Access Patterns

### A. Using Deno KV (Built-in Key-Value Database)

```typescript
// src/repositories/user_kv_repository.ts

/**
 * User repository implementation using Deno KV.
 *
 * Deno KV is a built-in key-value database with ACID transactions,
 * automatic replication, and edge deployment support.
 */
export class UserKVRepository {
  private kv: Deno.Kv;

  private constructor(kv: Deno.Kv) {
    this.kv = kv;
  }

  /**
   * Creates a new repository instance.
   */
  static async create(path?: string): Promise<UserKVRepository> {
    const kv = await Deno.openKv(path);
    return new UserKVRepository(kv);
  }

  /**
   * Finds a user by ID.
   */
  async findById(id: string): Promise<User | null> {
    const result = await this.kv.get<User>(['users', id]);
    return result.value;
  }

  /**
   * Finds a user by email.
   */
  async findByEmail(email: string): Promise<User | null> {
    const emailIndex = await this.kv.get<string>(['users_by_email', email]);
    if (!emailIndex.value) return null;
    return this.findById(emailIndex.value);
  }

  /**
   * Creates a new user with atomic transaction.
   */
  async create(input: CreateUserInput): Promise<User> {
    const id = crypto.randomUUID();
    const user: User = {
      id,
      ...input,
      role: input.role || 'user',
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    // Atomic transaction: check email uniqueness and insert
    const emailCheck = await this.kv.get(['users_by_email', user.email]);
    if (emailCheck.value) {
      throw new Error('Email already exists');
    }

    const atomic = this.kv.atomic()
      .check(emailCheck) // Ensure email still doesn't exist
      .set(['users', id], user)
      .set(['users_by_email', user.email], id);

    const result = await atomic.commit();
    if (!result.ok) {
      throw new Error('Failed to create user');
    }

    return user;
  }

  /**
   * Updates a user.
   */
  async update(id: string, data: Partial<User>): Promise<User> {
    const existing = await this.findById(id);
    if (!existing) {
      throw new Error('User not found');
    }

    const updated: User = {
      ...existing,
      ...data,
      id, // Ensure ID doesn't change
      updatedAt: new Date(),
    };

    await this.kv.set(['users', id], updated);
    return updated;
  }

  /**
   * Deletes a user.
   */
  async delete(id: string): Promise<void> {
    const user = await this.findById(id);
    if (!user) {
      throw new Error('User not found');
    }

    // Atomic delete: remove user and email index
    await this.kv.atomic()
      .delete(['users', id])
      .delete(['users_by_email', user.email])
      .commit();
  }

  /**
   * Lists users with pagination.
   */
  async list(options: { limit: number; offset: number }): Promise<User[]> {
    const users: User[] = [];
    const entries = this.kv.list<User>({ prefix: ['users'] });

    let count = 0;
    for await (const entry of entries) {
      if (count < options.offset) {
        count++;
        continue;
      }
      if (users.length >= options.limit) {
        break;
      }
      users.push(entry.value);
      count++;
    }

    return users;
  }

  /**
   * Closes the KV connection.
   */
  async close(): Promise<void> {
    this.kv.close();
  }
}

// Usage
const repository = await UserKVRepository.create();

// Create user
const user = await repository.create({
  email: 'user@example.com',
  name: 'Test User',
});

// Find user
const found = await repository.findById(user.id);
const foundByEmail = await repository.findByEmail('user@example.com');

// Update user
const updated = await repository.update(user.id, { name: 'Updated Name' });

// List users
const users = await repository.list({ limit: 10, offset: 0 });

// Delete user
await repository.delete(user.id);

await repository.close();
```

### B. PostgreSQL with Deno Postgres

```typescript
// src/config/database.ts
import { Pool } from 'https://deno.land/x/postgres@v0.17.0/mod.ts';

let pool: Pool | null = null;

export function getPool(): Pool {
  if (!pool) {
    const databaseUrl = Deno.env.get('DATABASE_URL');
    if (!databaseUrl) {
      throw new Error('DATABASE_URL environment variable is required');
    }

    pool = new Pool(databaseUrl, 10, true); // 10 connections, lazy
  }
  return pool;
}

// src/repositories/user_postgres_repository.ts
import type { Pool, PoolClient } from 'https://deno.land/x/postgres@v0.17.0/mod.ts';
import { getPool } from '../config/database.ts';

export class UserPostgresRepository {
  private pool: Pool;

  constructor(pool?: Pool) {
    this.pool = pool || getPool();
  }

  async findById(id: string): Promise<User | null> {
    const client = await this.pool.connect();
    try {
      const result = await client.queryObject<User>(
        'SELECT * FROM users WHERE id = $1',
        [id]
      );
      return result.rows[0] || null;
    } finally {
      client.release();
    }
  }

  async findByEmail(email: string): Promise<User | null> {
    const client = await this.pool.connect();
    try {
      const result = await client.queryObject<User>(
        'SELECT * FROM users WHERE email = $1',
        [email]
      );
      return result.rows[0] || null;
    } finally {
      client.release();
    }
  }

  async create(input: CreateUserInput): Promise<User> {
    const client = await this.pool.connect();
    try {
      const result = await client.queryObject<User>(
        `INSERT INTO users (email, name, role, created_at, updated_at)
         VALUES ($1, $2, $3, NOW(), NOW())
         RETURNING *`,
        [input.email, input.name, input.role || 'user']
      );
      return result.rows[0];
    } finally {
      client.release();
    }
  }

  async update(id: string, data: Partial<User>): Promise<User> {
    const client = await this.pool.connect();
    try {
      const setClauses: string[] = [];
      const values: unknown[] = [];
      let paramIndex = 1;

      if (data.name !== undefined) {
        setClauses.push(`name = $${paramIndex++}`);
        values.push(data.name);
      }
      if (data.email !== undefined) {
        setClauses.push(`email = $${paramIndex++}`);
        values.push(data.email);
      }

      setClauses.push(`updated_at = NOW()`);
      values.push(id);

      const result = await client.queryObject<User>(
        `UPDATE users SET ${setClauses.join(', ')}
         WHERE id = $${paramIndex}
         RETURNING *`,
        values
      );

      if (result.rows.length === 0) {
        throw new Error('User not found');
      }

      return result.rows[0];
    } finally {
      client.release();
    }
  }

  async delete(id: string): Promise<void> {
    const client = await this.pool.connect();
    try {
      await client.queryObject('DELETE FROM users WHERE id = $1', [id]);
    } finally {
      client.release();
    }
  }

  async list(options: { limit: number; offset: number }): Promise<User[]> {
    const client = await this.pool.connect();
    try {
      const result = await client.queryObject<User>(
        'SELECT * FROM users ORDER BY created_at DESC LIMIT $1 OFFSET $2',
        [options.limit, options.offset]
      );
      return result.rows;
    } finally {
      client.release();
    }
  }
}
```

---

## 9. Security Best Practices (MANDATORY)

### A. Environment Variables

```typescript
// src/config/env.ts
import { z } from 'npm:zod@^3.22.4';

/**
 * Environment variables schema with validation.
 */
const envSchema = z.object({
  // Server
  PORT: z.coerce.number().int().positive().default(8000),
  HOST: z.string().default('0.0.0.0'),

  // Database
  DATABASE_URL: z.string().url(),

  // Authentication
  JWT_SECRET: z.string().min(32),
  JWT_EXPIRY: z.string().default('24h'),

  // Logging
  LOG_LEVEL: z.enum(['debug', 'info', 'warn', 'error']).default('info'),

  // Environment
  NODE_ENV: z.enum(['development', 'staging', 'production']).default('development'),

  // External APIs
  STRIPE_SECRET_KEY: z.string().optional(),
  SENDGRID_API_KEY: z.string().optional(),
});

/**
 * Validated environment variables.
 *
 * @throws {Error} If environment variables are invalid
 */
export const env = envSchema.parse(Deno.env.toObject());

// ✅ CORRECT - Type-safe environment access
const port = env.PORT; // number
const dbUrl = env.DATABASE_URL; // string

// ❌ WRONG - Direct env access without validation
const unsafePort = Deno.env.get('PORT'); // string | undefined
```

### B. Secure Password Hashing

```typescript
// src/utils/crypto.ts

/**
 * Hashes a password using bcrypt.
 */
export async function hashPassword(password: string): Promise<string> {
  const bcrypt = await import('https://deno.land/x/bcrypt@v0.4.1/mod.ts');
  return await bcrypt.hash(password);
}

/**
 * Verifies a password against its hash.
 */
export async function verifyPassword(
  password: string,
  hash: string,
): Promise<boolean> {
  const bcrypt = await import('https://deno.land/x/bcrypt@v0.4.1/mod.ts');
  return await bcrypt.compare(password, hash);
}

/**
 * Generates a cryptographically secure random token.
 */
export function generateSecureToken(length: number = 32): string {
  const bytes = new Uint8Array(length);
  crypto.getRandomValues(bytes);
  return Array.from(bytes)
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Hashes data using SHA-256.
 */
export async function sha256(data: string): Promise<string> {
  const msgBuffer = new TextEncoder().encode(data);
  const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}
```

### C. JWT Authentication

```typescript
// src/services/auth_service.ts
import { create, verify, getNumericDate } from 'https://deno.land/x/djwt@v3.0.1/mod.ts';

const JWT_SECRET = Deno.env.get('JWT_SECRET')!;
const key = await crypto.subtle.importKey(
  'raw',
  new TextEncoder().encode(JWT_SECRET),
  { name: 'HMAC', hash: 'SHA-256' },
  false,
  ['sign', 'verify']
);

export interface JWTPayload {
  userId: string;
  email: string;
  role: string;
}

/**
 * Generates a JWT token for a user.
 */
export async function generateToken(payload: JWTPayload): Promise<string> {
  return await create(
    { alg: 'HS256', typ: 'JWT' },
    {
      ...payload,
      iss: 'myapp',
      aud: 'myapp-api',
      exp: getNumericDate(60 * 60 * 24), // 24 hours
    },
    key
  );
}

/**
 * Verifies and decodes a JWT token.
 */
export async function verifyToken(token: string): Promise<JWTPayload> {
  const payload = await verify(token, key);
  return payload as JWTPayload;
}
```

### D. Input Sanitization

```typescript
// src/utils/sanitize.ts

/**
 * Sanitizes HTML to prevent XSS attacks.
 */
export function sanitizeHtml(html: string): string {
  // Use a library like DOMPurify for production
  return html
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
    .replace(/\//g, '&#x2F;');
}

/**
 * Sanitizes SQL identifiers to prevent SQL injection.
 */
export function sanitizeSqlIdentifier(identifier: string): string {
  // Only allow alphanumeric and underscore
  if (!/^[a-zA-Z0-9_]+$/.test(identifier)) {
    throw new Error('Invalid SQL identifier');
  }
  return identifier;
}

/**
 * Validates and sanitizes email addresses.
 */
export function sanitizeEmail(email: string): string {
  const trimmed = email.trim().toLowerCase();
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  if (!emailRegex.test(trimmed)) {
    throw new Error('Invalid email format');
  }

  return trimmed;
}
```

### E. CORS Configuration

```typescript
// src/middleware/cors.ts

const ALLOWED_ORIGINS = [
  'https://myapp.com',
  'https://www.myapp.com',
];

if (Deno.env.get('NODE_ENV') === 'development') {
  ALLOWED_ORIGINS.push('http://localhost:3000');
}

export function corsHeaders(origin: string | null): Headers {
  const headers = new Headers();

  if (origin && ALLOWED_ORIGINS.includes(origin)) {
    headers.set('Access-Control-Allow-Origin', origin);
    headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    headers.set('Access-Control-Max-Age', '86400'); // 24 hours
  }

  return headers;
}

export function handleCors(req: Request): Response | null {
  const origin = req.headers.get('origin');

  if (req.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: corsHeaders(origin),
    });
  }

  return null;
}
```

---

## 10. Performance Optimization

### A. Caching Strategies

```typescript
// src/utils/cache.ts

/**
 * Simple in-memory cache with TTL.
 */
export class MemoryCache<T> {
  private cache = new Map<string, { value: T; expires: number }>();

  /**
   * Gets a value from cache.
   */
  get(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    if (Date.now() > entry.expires) {
      this.cache.delete(key);
      return null;
    }

    return entry.value;
  }

  /**
   * Sets a value in cache with TTL in seconds.
   */
  set(key: string, value: T, ttl: number = 60): void {
    this.cache.set(key, {
      value,
      expires: Date.now() + ttl * 1000,
    });
  }

  /**
   * Deletes a value from cache.
   */
  delete(key: string): void {
    this.cache.delete(key);
  }

  /**
   * Clears all cache entries.
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Removes expired entries.
   */
  cleanup(): void {
    const now = Date.now();
    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expires) {
        this.cache.delete(key);
      }
    }
  }
}

// Usage with memoization
const userCache = new MemoryCache<User>();

export async function getCachedUser(id: string): Promise<User> {
  const cached = userCache.get(id);
  if (cached) return cached;

  const user = await fetchUserFromDb(id);
  userCache.set(id, user, 300); // Cache for 5 minutes
  return user;
}

// Cleanup expired entries every minute
setInterval(() => userCache.cleanup(), 60_000);
```

### B. Streaming Responses

```typescript
// src/utils/streaming.ts

/**
 * Streams a large dataset as JSON array.
 */
export async function streamJsonArray<T>(
  items: AsyncIterable<T>,
  res: Response,
): Promise<Response> {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      controller.enqueue(encoder.encode('['));

      let first = true;
      for await (const item of items) {
        if (!first) {
          controller.enqueue(encoder.encode(','));
        }
        controller.enqueue(encoder.encode(JSON.stringify(item)));
        first = false;
      }

      controller.enqueue(encoder.encode(']'));
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'application/json',
      'Transfer-Encoding': 'chunked',
    },
  });
}

/**
 * Streams CSV data.
 */
export function streamCsv(
  headers: string[],
  rows: AsyncIterable<string[]>,
): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      // Write headers
      controller.enqueue(encoder.encode(headers.join(',') + '\n'));

      // Write rows
      for await (const row of rows) {
        controller.enqueue(encoder.encode(row.join(',') + '\n'));
      }

      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/csv',
      'Content-Disposition': 'attachment; filename="export.csv"',
    },
  });
}
```

### C. Parallel Processing

```typescript
// src/utils/parallel.ts

/**
 * Processes items in parallel with controlled concurrency.
 */
export async function parallelMap<T, R>(
  items: T[],
  fn: (item: T) => Promise<R>,
  concurrency: number = 10,
): Promise<R[]> {
  const results: R[] = [];
  const executing: Promise<void>[] = [];

  for (const [index, item] of items.entries()) {
    const promise = fn(item).then(result => {
      results[index] = result;
    });

    executing.push(promise);

    if (executing.length >= concurrency) {
      await Promise.race(executing);
      executing.splice(executing.findIndex(p => p === promise), 1);
    }
  }

  await Promise.all(executing);
  return results;
}

// Usage
const urls = ['https://api.example.com/1', 'https://api.example.com/2'];
const data = await parallelMap(
  urls,
  async (url) => {
    const res = await fetch(url);
    return res.json();
  },
  5 // Process 5 URLs concurrently
);
```

---

## 11. Security & Dependency Management (MANDATORY)

### A. Automated Dependency Management

**Use JSR and Deno's native workspace features for automated management:**

```json
// deno.json
{
  "imports": {
    "@std/": "jsr:@std/",
    "hono": "jsr:@hono/hono@^4.0"
  },
  "lock": true
}
```

- **Lockfiles**: Deno 2.0+ uses `deno.lock` by default. ALWAYS commit this file.
- **Frozen Builds**: Use `--frozen` in CI to ensure no lockfile changes are allowed.
- **Dependency Auditing**: Use `deno audit` to scan for known vulnerabilities.

### B. Vulnerability Scanning & Security

**Mandatory security checks for ALL Deno projects:**

1. **Vulnerability Scan**:
   ```bash
   # Scan all dependencies for CVEs
   deno audit
   ```
   - Agents MUST fix all discoverable high/critical vulnerabilities before presentation.

2. **Supply Chain Audit**:
   - Verify JSR and npm package signatures (automatic in Deno).
   - Use `deno check` to verify type safety of all dependencies.

### C. Dependency File

```json
// deno.json example
{
  "imports": {
    "@std/assert": "jsr:@std/assert@1",
    "@std/http": "jsr:@std/http@1"
  }
}
```

---

## 12. Deployment Checklist

### Agent-Generated Code Verification (MANDATORY)

#### Build & Compilation
- [ ] Code compiles: `deno check main.ts` returns exit code 0
- [ ] No type errors or warnings
- [ ] All imports/dependencies resolved
- [ ] Code formatted: `deno fmt --check` produces no changes

#### Testing
- [ ] All tests pass: `deno test --frozen` returns exit code 0
- [ ] Reasonable coverage: `deno test --coverage` shows >80%
- [ ] Integration tests pass (if applicable)

#### Security
- [ ] Dependency scan passes: `deno audit` shows 0 vulnerabilities
- [ ] Supply chain verified: `deno.lock` is in sync and verified
- [ ] Secrets check: No hardcoded secrets or sensitive data in `.env`
- [ ] Static analysis: `deno lint` passes with 0 warnings

#### Code Quality
- [ ] No unused dependencies or imports
- [ ] Small, focused modules with clear exports
- [ ] Project structure follows standard layout

#### Documentation
- [ ] All public APIs have JSDoc comments
- [ ] Documentation follows conventions
- [ ] Examples provided for complex APIs and tested with `deno test --doc`

#### Architecture
- [ ] Hexagonal architecture followed where appropriate
- [ ] Permissions are explicit and granular
- [ ] No global mutable state

#### Agent Workflow Completed
- [ ] Agent verified code compiles/builds successfully
- [ ] Agent ran all tests and verified they pass
- [ ] Agent ran formatters and linters
- [ ] Agent verified documentation
- [ ] Agent documented any fixes made during verification

---

## 13. Why This Configuration Works

**Deno 2.1+ Native Tooling**:
- Combines the best of Deno (security, ESM) with modern Node.js compatibility, allowing use of `npm:` packages without the security risks of `node_modules` by default.

**JSR (JavaScript Registry)**:
- Provides a high-performance, TypeScript-first package registry that auto-generates documentation and enforces best practices.

**Frozen Lockfiles**:
- Using `--frozen` in CI prevents "it works on my machine" issues and guards against supply chain attacks by ensuring the exact same bits are used in every environment.

---

## 14. Quick Reference

### Common Commands

```bash
# Run with explicit permissions
deno run --allow-net --allow-read main.ts

# Test with security checks
deno test --frozen --allow-all

# Security scan
deno audit

# Lint and Format
deno lint && deno fmt

# Generate Documentation
deno doc --html mod.ts

# Compile to standalone binary
deno compile --allow-all main.ts
```

### Modern Deno Patterns Cheat Sheet

```typescript
// Native .env handling (Deno 2.0+)
// Run with: deno run --env-file=.env main.ts
const apiKey = Deno.env.get("API_KEY");

// Native SQLite (Deno 2.1+)
import { DatabaseSync } from "node:sqlite";
const db = new DatabaseSync("data.db");

// Range-over-functions (Deno 2.1+)
for (const user of service.allUsers()) { ... }
```

---

**Last Updated:** 2026-02-06
**Version:** 2.1
**Maintainer:** Deno Team


**End of Deno Development Guidelines**
