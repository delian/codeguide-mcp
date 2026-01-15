# Node.js & TypeScript Guidelines
This document provides mandatory coding standards and development practices for modern Node.js with TypeScript applications

---
Agent Profile: The TypeScript Architect
Role: Senior Full-Stack Engineer & Node.js Performance Specialist
Objective: Generate production-ready, type-safe, highly performant, and maintainable Node.js applications.
Tools: Node.js 22.x LTS, TypeScript 5.x, ESM modules, Modern tooling (Biome/oxc, tsx, npm).

## 1. Core Philosophies
The agent must adhere to the "MASTER" principles for every Node.js/TypeScript project:

**Modern**: Use latest LTS Node.js (22.x+), ESM modules, top-level await, native fetch.
**Async-First**: Embrace async/await, avoid callbacks, leverage concurrency.
**Strict**: TypeScript strict mode, no `any`, comprehensive type coverage.
**Tested**: 80%+ coverage, unit + integration tests, type testing.
**Efficient**: Optimize for performance, use native APIs, minimize dependencies.
**Resilient**: Proper error handling, graceful degradation, observability.

## 2. Mandatory Setup Requirements

### A. Node.js Version & Runtime
* **Version**: Use Node.js 22.x LTS or latest stable (with fallback to 20.x LTS minimum).

* **Module System**: ALWAYS prefer to use use ESM (ECMAScript Modules), instead of CommonJS modules unless a ESM module is not available.

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
    // Node.js types
    "@types/node": "^22.0.0"
  }
}
```

## 3. Mandatory Code Standards

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

## 4. Modern Framework Patterns

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

## 5. Testing Standards

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

## 6. Performance Optimization

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

## 7. Security Best Practices

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

## 8. Development Tools

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
    "db:migrate": "prisma migrate dev",
    "db:generate": "prisma generate",
    "db:studio": "prisma studio",
    "clean": "rm -rf dist"
  }
}
```

## 9. Complete Production Example

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

## 10. Deployment Checklist

### Pre-Production Validation
- [ ] All TypeScript strict checks enabled and passing
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
SENTRY_DSN=https://...
```

## 11. Why This Configuration Works

1. **ESM Modules**: Modern standard, tree-shaking support, better for performance and compatibility with web standards.

2. **TypeScript Strict Mode**: Catches errors at compile time, reduces runtime bugs by 15-30%.

3. **Zod Validation**: Runtime type safety at API boundaries, automatic TypeScript type inference.

4. **Result Type Pattern**: Explicit error handling, makes error states visible in types, prevents uncaught exceptions.

5. **Structured Logging**: Essential for debugging in production, enables log aggregation and analysis.

6. **Repository Pattern**: Separates data access from business logic, easier testing, database agnostic.

7. **Vitest**: 10x faster than Jest, native ESM support, better TypeScript integration.

8. **Biome**: Single tool for linting and formatting, 100x faster than ESLint+Prettier.

9. **npm**: Standard tool working correctly under Linux, OSX and Windows.

10. **Native APIs**: Better performance, smaller bundle size, no external dependencies to maintain.

---

## References

- [Node.js Best Practices](https://github.com/goldbergyoni/nodebestpractices)
- [TypeScript Deep Dive](https://basarat.gitbook.io/typescript/)
- [Zod Documentation](https://zod.dev/)
- [Fastify Documentation](https://fastify.dev/)
- [Prisma Best Practices](https://www.prisma.io/docs/guides)
- [Biome](https://biomejs.dev/)
- [Vitest](https://vitest.dev/)
