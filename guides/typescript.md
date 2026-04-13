# TypeScript Development Guidelines
Mandatory coding standards and development practices for modern TypeScript applications. TypeScript 5.7+, TypeDoc, Modern testing frameworks (Vitest/Jest), ESLint, Prettier.

---

**Agent Profile**: The TypeScript Expert
**Role**: Senior TypeScript Engineer & Type Safety Specialist
**Objective**: Generate production-ready, type-safe, fully documented, minimalistic, and maintainable TypeScript code.
**Tools**: TypeScript 5.7+, TypeDoc, Modern testing frameworks (Vitest/Jest), ESLint, Biome/Oxlint (optional), Prettier.

---

## 1. Core Philosophies: TYPESCRIPT-FIRST
The agent must adhere to the "TYPESCRIPT-FIRST" principles for every TypeScript project:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Secure by Default**: Mandatory lockfile integrity checks and supply chain vulnerability scanning.
**Type Safety First**: Strict mode enabled, no `any`, comprehensive type coverage, branded types.
**You Own Your Types**: Define explicit types, avoid type inference abuse, document complex types.
**Pure Functions Preferred**: Side-effect free where possible, explicit about effects, testable, functional programming patterns.
**Explicit Over Implicit**: Clear return types, named parameters, obvious intent.

**Simple Code**: Minimalistic, readable, single responsibility, avoid over-engineering.
**Async-First**: Prefer async/await over promises, promises over callbacks.
**Functional Programming**: Immutability, higher-order functions, function composition whenever applicable.
**Automated Testing**: Unit tests for all logic, 80%+ coverage, tests must pass.
**Fully Documented**: TypeDoc comments for all exports, auto-generated API documentation.
**Every Change Verified**: Agent-generated code MUST compile and pass all tests before delivery.

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated code compiles and tests pass before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY TypeScript code, the agent MUST:**

1. **TypeScript Compilation Check**:
   ```bash
   # Verify TypeScript compiles without errors
   npx tsc --noEmit
   # Exit code MUST be 0
   ```

2. **Linting & Security Check**:
   ```bash
   # Run ESLint or Biome
   npx eslint . --ext .ts,.tsx
   # npx @biomejs/biome check .
   
   # Check for vulnerabilities (MANDATORY)
   npm audit --audit-level=high
   ```
   - Fix all errors, address warnings
   - **Audit MUST return no high/critical vulnerabilities**

3. **Lockfile & Supply Chain Verification**:
   ```bash
   # Verify lockfile matches package.json
   npm ci --dry-run
   ```
   - **MUST** be in sync

4. **Test Creation (MANDATORY)**:
   - Write unit tests for ALL new functions
   - Write unit tests for ALL new classes
   - Write unit tests for ALL new types with type guards
   - Minimum 80% code coverage
   - Tests MUST follow best practices

4. **Test Execution**:
   ```bash
   # Run all tests
   npm test
   # OR with coverage
   npm run test:coverage
   # Exit code MUST be 0, coverage MUST be ≥ 80%
   ```

5. **Documentation Generation**:
   ```bash
   # Verify TypeDoc can generate documentation
   npx typedoc
   # Exit code MUST be 0
   ```

### B. Error Correction Process

If verification fails:

1. **Read the error message** carefully
2. **Identify the root cause** (type error, missing import, test failure, etc.)
3. **Fix the issue** in the generated code
4. **Re-run verification** until all checks pass
5. **Document any non-obvious fixes** in comments

### C. Agent Workflow Example

**Complete workflow for generating a new function:**

1. **Generate function with TypeDoc**:
   ```typescript
   /**
    * Parses a user ID from a string.
    * @param input - The input string to parse
    * @returns The parsed user ID or null if invalid
    */
   export function parseUserId(input: string): string | null {
     return input.trim() || null;
   }
   ```

2. **Generate comprehensive tests**:
   ```typescript
   import { describe, it, expect } from 'vitest';
   import { parseUserId } from './user';
   
   describe('parseUserId', () => {
     it('returns trimmed string for valid input', () => {
       expect(parseUserId('  user123  ')).toBe('user123');
     });
     
     it('returns null for empty string', () => {
       expect(parseUserId('')).toBeNull();
     });
   });
   ```

3. **Verify TypeScript compilation**:
   ```bash
   npx tsc --noEmit
   # ✓ No errors
   ```

4. **Run tests**:
   ```bash
   npm test
   # ✓ All tests passed (2/2)
   ```

5. **Generate documentation**:
   ```bash
   npx typedoc
   # ✓ Documentation generated successfully
   ```

6. **Present code** to user - only after ALL checks pass

### D. Prohibited Practices

**NEVER deliver code that:**
- ❌ Has TypeScript compilation errors
- ❌ Uses `any` types to bypass type checking
- ❌ Has failing tests
- ❌ Lacks tests for new functionality
- ❌ Has test coverage < 80%
- ❌ Lacks TypeDoc comments for exported symbols
- ❌ Cannot generate documentation
- ❌ Has linter errors
- ❌ Uses `@ts-ignore` or `@ts-expect-error` without justification
- ❌ Uses nested callbacks instead of async/await
- ❌ Uses promise chains where async/await is clearer
- ❌ Mutates input parameters
- ❌ Has side effects in functions without documentation
- ❌ **Fixes bugs without adding regression tests first**
- ❌ **Writes implementation before writing tests (violates TDD)**
- ❌ **Skips Red-Green-Refactor cycle for new features**

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### TDD Cycle

```
1. 🔴 RED: Write a failing test first
   ↓
2. 🟢 GREEN: Write minimal code to make it pass
   ↓
3. 🔵 REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow

```typescript
// Step 1: RED - Write failing test first
import { describe, it, expect } from 'vitest';
import { calculateDiscount } from './pricing';

describe('calculateDiscount', () => {
  it('applies 10% discount for premium users', () => {
    expect(calculateDiscount(100, 'premium')).toBe(90);
  });
  
  it('no discount for standard users', () => {
    expect(calculateDiscount(100, 'standard')).toBe(100);
  });
});

// Run: npm test
// ❌ FAILS - calculateDiscount doesn't exist yet

// Step 2: GREEN - Write minimal implementation
type UserTier = 'standard' | 'premium';

export function calculateDiscount(price: number, tier: UserTier): number {
  if (tier === 'premium') {
    return price * 0.9;
  }
  return price;
}

// Run: npm test
// ✅ PASSES - tests pass

// Step 3: REFACTOR - Improve readability
const DISCOUNT_RATES: Record<UserTier, number> = {
  standard: 0,
  premium: 0.1,
};

export function calculateDiscount(price: number, tier: UserTier): number {
  const discountRate = DISCOUNT_RATES[tier];
  return price * (1 - discountRate);
}
// Tests still pass ✓
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

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

### Example Bug Fix

```typescript
// Bug Report #2041: formatCurrency fails with negative zero (-0)

// Step 1-2: Write test that reproduces the bug
import { describe, it, expect } from 'vitest';
import { formatCurrency } from './formatters';

describe('formatCurrency - Bug #2041', () => {
  it('formats negative zero correctly - Bug #2041', () => {
    // Bug: formatCurrency(-0) returned "-$0.00" instead of "$0.00"
    // Discovered: 2026-01-18
    // This test prevents regression
    
    expect(formatCurrency(-0)).toBe('$0.00');
  });
  
  it('formats regular negative numbers', () => {
    expect(formatCurrency(-10.5)).toBe('-$10.50');
  });
});

// Run: npm test
// ❌ FAILS - formatCurrency(-0) returns "-$0.00"

// Step 3: Fix the bug
export function formatCurrency(amount: number): string {
  // FIX: Normalize negative zero to positive zero
  const normalized = Object.is(amount, -0) ? 0 : amount;
  
  const sign = normalized < 0 ? '-' : '';
  const abs = Math.abs(normalized);
  
  return `${sign}$${abs.toFixed(2)}`;
}

// Run: npm test
// ✅ PASSES - bug fixed, regression prevented ✓
```

### Prohibited Practices for Bug Fixes

**NEVER:**
- ❌ Fix a bug without adding a regression test first
- ❌ Write implementation before writing tests (violates TDD)
- ❌ Skip the Red-Green-Refactor cycle
- ❌ Commit code with failing tests
- ❌ Remove tests to make code pass
- ❌ Use test.skip() to ignore failing tests

---

## 3. TypeScript Configuration (Mandatory)

### A. tsconfig.json

```json
{
  "compilerOptions": {
    // Language & Environment
    "target": "ES2022",
    "lib": ["ES2022"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    
    // Strict Type Checking (ALL REQUIRED)
    "strict": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "strictBindCallApply": true,
    "strictPropertyInitialization": true,
    "noImplicitAny": true,
    "noImplicitThis": true,
    "alwaysStrict": true,
    "useUnknownInCatchVariables": true,
    
    // Additional Checks (ALL REQUIRED)
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitOverride": true,
    "noPropertyAccessFromIndexSignature": true,
    "allowUnusedLabels": false,
    "allowUnreachableCode": false,
    
    // Module Resolution
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
      "@/*": ["./src/*"]
    },
    
    // Advanced
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "exactOptionalPropertyTypes": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts", "**/*.spec.ts"]
}
```

### B. Essential Dependencies

```json
{
  "devDependencies": {
    "typescript": "^5.4.0",
    // Testing
    "vitest": "^1.3.0",
    "@vitest/coverage-v8": "^1.3.0",
    // Documentation
    "typedoc": "^0.25.0",
    // Linting & Formatting
    "eslint": "^8.57.0",
    "@typescript-eslint/eslint-plugin": "^7.0.0",
    "@typescript-eslint/parser": "^7.0.0",
    "prettier": "^3.2.0"
  }
}
```

### C. Package Scripts

```json
{
  "scripts": {
    "build": "tsc",
    "dev": "tsc --watch",
    "test": "vitest",
    "test:coverage": "vitest run --coverage",
    "test:ui": "vitest --ui",
    "lint": "eslint . --ext .ts,.tsx",
    "lint:fix": "eslint . --ext .ts,.tsx --fix",
    "format": "prettier --write \"src/**/*.{ts,tsx}\"",
    "format:check": "prettier --check \"src/**/*.{ts,tsx}\"",
    "typecheck": "tsc --noEmit",
    "docs": "typedoc",
    "docs:check": "typedoc --emit none --validation.notDocumented true",
    "docs:serve": "typedoc && npx serve docs",
    "verify": "npm run typecheck && npm run lint && npm run docs:check && npm run test"
  }
}
```

## 4. Documentation Requirements (MANDATORY)

### A. TypeDoc Comments for All Exports

**ALL exported functions, classes, interfaces, types, and constants MUST have comprehensive TypeDoc documentation.**

#### Why TypeDoc?

- **Auto-Generated API Docs**: TypeDoc generates HTML documentation from code comments
- **IDE Integration**: Better IntelliSense and inline help
- **Type Safety**: Documentation stays synchronized with types
- **Maintenance**: Self-documenting code reduces onboarding time by 40%+
- **Verification**: Documentation completeness can be verified in build

### B. Function Documentation

```typescript
/**
 * Validates an email address format.
 * 
 * Checks if the provided string matches standard email format.
 * Does not verify if the email exists, only format validity.
 * 
 * @param email - The email address to validate
 * @returns `true` if email format is valid, `false` otherwise
 * 
 * @example
 * ```typescript
 * const valid = validateEmail('user@example.com');
 * console.log(valid); // true
 * 
 * const invalid = validateEmail('not-an-email');
 * console.log(invalid); // false
 * ```
 * 
 * @see {@link parseEmail} for extracting email parts
 * @public
 */
export function validateEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

/**
 * Fetches user data from the API.
 * 
 * Retrieves a user by their unique identifier. Returns `null` if the user
 * is not found. Network errors are thrown as exceptions.
 * 
 * @param userId - The unique identifier of the user
 * @param options - Optional fetch configuration
 * @param options.signal - AbortSignal for request cancellation
 * @param options.timeout - Request timeout in milliseconds (default: 5000)
 * 
 * @returns Promise resolving to User object or null if not found
 * @throws {@link NetworkError} If network request fails
 * @throws {@link TimeoutError} If request exceeds timeout
 * 
 * @example
 * ```typescript
 * try {
 *   const user = await fetchUser('user-123');
 *   if (user) {
 *     console.log('User found:', user.name);
 *   } else {
 *     console.log('User not found');
 *   }
 * } catch (error) {
 *   console.error('Failed to fetch user:', error);
 * }
 * ```
 * 
 * @example Fetch with timeout
 * ```typescript
 * const user = await fetchUser('user-123', { timeout: 3000 });
 * ```
 * 
 * @public
 */
export async function fetchUser(
  userId: string,
  options?: {
    signal?: AbortSignal;
    timeout?: number;
  }
): Promise<User | null> {
  // Implementation
  return null;
}
```

### C. Class Documentation

```typescript
/**
 * In-memory cache with TTL support.
 * 
 * Provides a simple key-value cache with automatic expiration.
 * All operations are synchronous and thread-safe in single-threaded environments.
 * 
 * @typeParam K - Type of cache keys (must be serializable)
 * @typeParam V - Type of cached values
 * 
 * @example
 * ```typescript
 * const cache = new Cache<string, User>();
 * 
 * // Set with 5-second TTL
 * cache.set('user-1', userData, 5000);
 * 
 * // Get within TTL
 * const user = cache.get('user-1'); // Returns userData
 * 
 * // After TTL expires
 * setTimeout(() => {
 *   const expired = cache.get('user-1'); // Returns undefined
 * }, 6000);
 * ```
 * 
 * @public
 */
export class Cache<K, V> {
  private store = new Map<K, CacheEntry<V>>();
  private defaultTTL: number;

  /**
   * Creates a new cache instance.
   * 
   * @param defaultTTL - Default time-to-live in milliseconds (default: 60000)
   * 
   * @example
   * ```typescript
   * // Cache with 10-second default TTL
   * const cache = new Cache<string, number>(10000);
   * ```
   */
  constructor(defaultTTL: number = 60000) {
    this.defaultTTL = defaultTTL;
  }

  /**
   * Stores a value in the cache.
   * 
   * If the key already exists, its value and TTL are updated.
   * 
   * @param key - The cache key
   * @param value - The value to cache
   * @param ttl - Time-to-live in milliseconds (uses default if not provided)
   * 
   * @returns The cache instance for method chaining
   * 
   * @example
   * ```typescript
   * cache
   *   .set('key1', 'value1', 5000)
   *   .set('key2', 'value2', 10000);
   * ```
   */
  set(key: K, value: V, ttl?: number): this {
    const expiresAt = Date.now() + (ttl ?? this.defaultTTL);
    this.store.set(key, { value, expiresAt });
    return this;
  }

  /**
   * Retrieves a value from the cache.
   * 
   * Returns `undefined` if the key doesn't exist or has expired.
   * Expired entries are automatically removed.
   * 
   * @param key - The cache key to retrieve
   * @returns The cached value or undefined if not found/expired
   * 
   * @example
   * ```typescript
   * const value = cache.get('myKey');
   * if (value !== undefined) {
   *   console.log('Cache hit:', value);
   * } else {
   *   console.log('Cache miss');
   * }
   * ```
   */
  get(key: K): V | undefined {
    const entry = this.store.get(key);
    if (!entry) return undefined;

    if (Date.now() > entry.expiresAt) {
      this.store.delete(key);
      return undefined;
    }

    return entry.value;
  }

  /**
   * Removes a key from the cache.
   * 
   * @param key - The key to remove
   * @returns `true` if the key existed and was removed, `false` otherwise
   */
  delete(key: K): boolean {
    return this.store.delete(key);
  }

  /**
   * Clears all entries from the cache.
   * 
   * @returns The cache instance for method chaining
   */
  clear(): this {
    this.store.clear();
    return this;
  }

  /**
   * Returns the number of entries in the cache.
   * 
   * Note: Includes expired entries that haven't been cleaned up yet.
   * 
   * @returns The number of cached entries
   */
  get size(): number {
    return this.store.size;
  }
}

/**
 * Internal cache entry structure.
 * @internal
 */
interface CacheEntry<V> {
  value: V;
  expiresAt: number;
}
```

### D. Interface and Type Documentation

```typescript
/**
 * Represents a user in the system.
 * 
 * Contains all user profile information and metadata.
 * Immutable after creation except for `lastLoginAt`.
 * 
 * @property id - Unique user identifier (UUID v4)
 * @property email - User's email address (unique, validated)
 * @property name - User's display name (1-100 characters)
 * @property role - User's role for authorization
 * @property createdAt - ISO 8601 timestamp of account creation
 * @property lastLoginAt - ISO 8601 timestamp of last login (optional)
 * @property metadata - Additional user-defined properties
 * 
 * @example
 * ```typescript
 * const user: User = {
 *   id: '550e8400-e29b-41d4-a716-446655440000',
 *   email: 'user@example.com',
 *   name: 'John Doe',
 *   role: 'user',
 *   createdAt: '2026-01-17T10:00:00Z',
 *   metadata: { department: 'engineering' }
 * };
 * ```
 * 
 * @public
 */
export interface User {
  readonly id: string;
  readonly email: string;
  name: string;
  role: UserRole;
  readonly createdAt: string;
  lastLoginAt?: string;
  metadata?: Record<string, unknown>;
}

/**
 * User role enumeration for authorization.
 * 
 * - `admin` - Full system access
 * - `user` - Standard user privileges
 * - `guest` - Read-only access
 * 
 * @example
 * ```typescript
 * function checkPermission(role: UserRole): boolean {
 *   return role === 'admin';
 * }
 * ```
 * 
 * @public
 */
export type UserRole = 'admin' | 'user' | 'guest';

/**
 * Result type for operations that can fail.
 * 
 * Provides type-safe error handling without exceptions.
 * Discriminated union based on `success` property.
 * 
 * @typeParam T - Type of success value
 * @typeParam E - Type of error (default: Error)
 * 
 * @example Success case
 * ```typescript
 * const result: Result<number> = {
 *   success: true,
 *   data: 42
 * };
 * ```
 * 
 * @example Error case
 * ```typescript
 * const result: Result<number> = {
 *   success: false,
 *   error: new Error('Operation failed')
 * };
 * ```
 * 
 * @example Type narrowing
 * ```typescript
 * function handle(result: Result<number>): void {
 *   if (result.success) {
 *     console.log('Value:', result.data); // TypeScript knows data exists
 *   } else {
 *     console.error('Error:', result.error.message); // TypeScript knows error exists
 *   }
 * }
 * ```
 * 
 * @public
 */
export type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

/**
 * Branded type for validated email addresses.
 * 
 * Creates a nominal type to prevent raw strings from being
 * used where validated emails are expected.
 * 
 * @example
 * ```typescript
 * function validateEmail(input: string): Email | null {
 *   if (/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(input)) {
 *     return input as Email;
 *   }
 *   return null;
 * }
 * 
 * function sendEmail(to: Email): void {
 *   // Type system ensures 'to' is a validated email
 * }
 * 
 * const raw = 'user@example.com';
 * sendEmail(raw); // ❌ Type error
 * 
 * const validated = validateEmail(raw);
 * if (validated) {
 *   sendEmail(validated); // ✓ OK
 * }
 * ```
 * 
 * @public
 */
export type Email = string & { readonly __brand: 'Email' };
```

### E. Generating Documentation with TypeDoc

#### Installation

```bash
npm install --save-dev typedoc
```

#### TypeDoc Configuration

Create `typedoc.json`:

```json
{
  "entryPoints": ["src/index.ts"],
  "entryPointStrategy": "expand",
  "out": "docs",
  "exclude": [
    "**/*.test.ts",
    "**/*.spec.ts",
    "**/test/**",
    "**/tests/**"
  ],
  "excludePrivate": true,
  "excludeProtected": false,
  "excludeInternal": true,
  "readme": "README.md",
  "plugin": [],
  "theme": "default",
  "categorizeByGroup": true,
  "categoryOrder": [
    "Classes",
    "Interfaces",
    "Type Aliases",
    "Functions",
    "Variables",
    "*"
  ],
  "sort": ["source-order"],
  "validation": {
    "notExported": true,
    "invalidLink": true,
    "notDocumented": true
  }
}
```

#### Generating Documentation

```bash
# Generate documentation
npm run docs

# Check documentation completeness
npm run docs:check

# Generate and serve
npm run docs:serve

# View documentation
open docs/index.html  # macOS
xdg-open docs/index.html  # Linux
```

### F. Documentation Best Practices

**DO:**
- ✅ Document all public exports (functions, classes, interfaces, types)
- ✅ Include `@param` for all parameters
- ✅ Include `@returns` for all return values
- ✅ Include `@throws` for functions that can throw
- ✅ Provide `@example` for complex APIs
- ✅ Use `@typeParam` for generic type parameters
- ✅ Link related items with `@see`
- ✅ Mark experimental APIs with `@beta`
- ✅ Mark deprecated APIs with `@deprecated`
- ✅ Generate docs as part of CI/CD

**DON'T:**
- ❌ Skip documentation for "obvious" functions
- ❌ Write vague descriptions
- ❌ Let documentation become outdated
- ❌ Commit generated docs to git (add `docs/` to `.gitignore`)
- ❌ Use `@internal` to hide undocumented public APIs

### G. Documentation Checklist

- [ ] All exported functions have TypeDoc comments
- [ ] All classes have TypeDoc comments
- [ ] All public interfaces and types have TypeDoc comments
- [ ] All `@param` tags match actual parameters
- [ ] All `@returns` tags describe return values
- [ ] At least one `@example` for complex APIs
- [ ] `npm run docs:check` passes without warnings
- [ ] Generated documentation is readable

---

## 5. Type Safety Standards (MANDATORY)

### A. No `any` - Ever

```typescript
// ✅ CORRECT - Explicit typing
function processData(data: unknown): string {
  if (typeof data === 'string') {
    return data.toUpperCase();
  }
  throw new Error('Invalid data type');
}

// ✅ CORRECT - Type guards
function isUser(value: unknown): value is User {
  return (
    typeof value === 'object' &&
    value !== null &&
    'id' in value &&
    typeof value.id === 'string'
  );
}

// ❌ WRONG - Using any
function processBad(data: any): any {
  return data.value;
}
```

### B. Explicit Return Types

```typescript
// ✅ CORRECT - Explicit return type
function calculateTotal(items: Item[]): number {
  return items.reduce((sum, item) => sum + item.price, 0);
}

// ✅ CORRECT - Explicit async return
async function fetchUser(id: string): Promise<User | null> {
  const response = await fetch(`/api/users/${id}`);
  if (!response.ok) return null;
  return response.json();
}

// ❌ WRONG - Inferred return type (unclear intent)
function calculateBad(items: Item[]) {
  return items.reduce((sum, item) => sum + item.price, 0);
}
```

### C. Branded Types for Safety

```typescript
// ✅ CORRECT - Branded types prevent misuse
type UserId = string & { readonly __brand: 'UserId' };
type Email = string & { readonly __brand: 'Email' };

function getUserById(id: UserId): User {
  // Type system ensures id is a validated UserId
  return database.get(id);
}

function sendEmail(to: Email): void {
  // Type system ensures to is a validated Email
  emailService.send(to);
}

// Factory functions
function createUserId(input: string): UserId {
  if (!isValidUuid(input)) {
    throw new Error('Invalid user ID');
  }
  return input as UserId;
}

// ❌ WRONG - Primitive types allow misuse
function getBadUser(id: string): User {
  return database.get(id); // Any string accepted!
}
```

### D. Discriminated Unions

```typescript
// ✅ CORRECT - Discriminated union
type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

function handleResult(result: Result<User>): void {
  if (result.success) {
    console.log('User:', result.data.name); // TypeScript knows data exists
  } else {
    console.error('Error:', result.error.message); // TypeScript knows error exists
  }
}

// ✅ CORRECT - Tagged union for state machine
type LoadingState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: User }
  | { status: 'error'; error: Error };

function renderState(state: LoadingState): string {
  switch (state.status) {
    case 'idle':
      return 'Ready';
    case 'loading':
      return 'Loading...';
    case 'success':
      return `User: ${state.data.name}`;
    case 'error':
      return `Error: ${state.error.message}`;
  }
}
```

### E. Immutability

```typescript
// ✅ CORRECT - Readonly properties
interface User {
  readonly id: string;
  readonly email: string;
  name: string; // Mutable field explicitly allowed
}

// ✅ CORRECT - Readonly arrays
function processItems(items: readonly Item[]): Item[] {
  return items.map(item => ({ ...item, processed: true }));
}

// ✅ CORRECT - Readonly parameters
function updateUser(user: Readonly<User>, updates: Partial<User>): User {
  return { ...user, ...updates };
}

// ✅ CORRECT - Deep readonly
type DeepReadonly<T> = {
  readonly [K in keyof T]: T[K] extends object
    ? DeepReadonly<T[K]>
    : T[K];
};

// ❌ WRONG - Mutable by default
function mutateBad(items: Item[]): void {
  items.push(newItem); // Modifies input!
}
```

## 6. Async/Await Hierarchy (MANDATORY)

### A. Preference Order

**ALWAYS prefer async/await > Promises > Callbacks (never use nested callbacks)**

```typescript
// ✅ CORRECT - Async/await (PREFERRED)
async function fetchUserData(userId: string): Promise<User> {
  const response = await fetch(`/api/users/${userId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch user: ${response.status}`);
  }
  const data = await response.json();
  return data as User;
}

async function processUserWorkflow(userId: string): Promise<void> {
  const user = await fetchUserData(userId);
  const profile = await fetchUserProfile(user.id);
  const settings = await fetchUserSettings(user.id);
  
  await saveProcessedData({ user, profile, settings });
}


// ✅ ACCEPTABLE - Promises when async/await isn't suitable
function fetchWithRetry(url: string, retries: number = 3): Promise<Response> {
  return fetch(url).catch((error) => {
    if (retries > 0) {
      return fetchWithRetry(url, retries - 1);
    }
    throw error;
  });
}


// ❌ WRONG - Callback hell (PROHIBITED)
function fetchUserDataBad(userId: string, callback: (error: Error | null, user?: User) => void): void {
  fetch(`/api/users/${userId}`, (error, response) => {
    if (error) {
      callback(error);
      return;
    }
    response.json((jsonError, data) => {
      if (jsonError) {
        callback(jsonError);
        return;
      }
      callback(null, data);
    });
  });
}
```

### B. Sequential Operations

```typescript
// ✅ CORRECT - Sequential with async/await (PREFERRED)
async function createUserAccount(email: string, name: string): Promise<User> {
  const validated = await validateEmail(email);
  if (!validated) {
    throw new Error('Invalid email');
  }
  
  const user = await createUser({ email, name });
  await sendWelcomeEmail(user.email);
  await createUserProfile(user.id);
  
  return user;
}


// ❌ WRONG - Promise chaining for sequential operations
function createUserAccountBad(email: string, name: string): Promise<User> {
  return validateEmail(email)
    .then((validated) => {
      if (!validated) throw new Error('Invalid email');
      return createUser({ email, name });
    })
    .then((user) => {
      return sendWelcomeEmail(user.email).then(() => user);
    })
    .then((user) => {
      return createUserProfile(user.id).then(() => user);
    });
}
```

### C. Parallel Operations

```typescript
// ✅ CORRECT - Parallel with Promise.all (PREFERRED)
async function fetchUserDetails(userId: string): Promise<UserDetails> {
  const [user, profile, settings, posts] = await Promise.all([
    fetchUser(userId),
    fetchProfile(userId),
    fetchSettings(userId),
    fetchUserPosts(userId),
  ]);
  
  return { user, profile, settings, posts };
}


// ✅ CORRECT - Parallel with error handling
async function fetchWithFallback(userId: string): Promise<UserDetails> {
  const [userResult, profileResult] = await Promise.allSettled([
    fetchUser(userId),
    fetchProfile(userId),
  ]);
  
  const user = userResult.status === 'fulfilled' 
    ? userResult.value 
    : null;
  
  const profile = profileResult.status === 'fulfilled'
    ? profileResult.value
    : null;
  
  return { user, profile };
}


// ❌ WRONG - Sequential when parallel is possible
async function fetchDetailsBad(userId: string): Promise<UserDetails> {
  const user = await fetchUser(userId);      // Waits unnecessarily
  const profile = await fetchProfile(userId); // Waits unnecessarily
  const settings = await fetchSettings(userId);
  
  return { user, profile, settings };
}
```

### D. Error Handling

```typescript
// ✅ CORRECT - Try-catch with async/await (PREFERRED)
async function safeUserFetch(userId: string): Promise<Result<User>> {
  try {
    const user = await fetchUser(userId);
    return { success: true, data: user };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error : new Error('Unknown error'),
    };
  }
}


// ✅ CORRECT - Multiple try-catch blocks for granular handling
async function processWithGranularErrors(userId: string): Promise<void> {
  let user: User;
  
  try {
    user = await fetchUser(userId);
  } catch (error) {
    logger.error('Failed to fetch user', { userId, error });
    throw new UserFetchError('User fetch failed', { cause: error });
  }
  
  try {
    await sendNotification(user.email);
  } catch (error) {
    // Log but don't fail the whole operation
    logger.warn('Failed to send notification', { userId, error });
  }
}


// ❌ WRONG - Promise catch chains
function safeFetchBad(userId: string): Promise<User | null> {
  return fetchUser(userId)
    .then((user) => user)
    .catch((error) => {
      console.error(error);
      return null;
    });
}
```

### E. Async Iteration

```typescript
// ✅ CORRECT - for-await-of for async iteration (PREFERRED)
async function processAllUsers(userIds: string[]): Promise<void> {
  for (const userId of userIds) {
    await processUser(userId);
  }
}

async function processStream(stream: AsyncIterable<Data>): Promise<void> {
  for await (const chunk of stream) {
    await processChunk(chunk);
  }
}


// ✅ CORRECT - Async generators
async function* fetchUsersBatch(
  userIds: string[],
  batchSize: number = 10
): AsyncGenerator<User[], void, void> {
  for (let i = 0; i < userIds.length; i += batchSize) {
    const batch = userIds.slice(i, i + batchSize);
    const users = await Promise.all(batch.map(fetchUser));
    yield users;
  }
}

// Usage
async function processInBatches(): Promise<void> {
  for await (const batch of fetchUsersBatch(allUserIds)) {
    await processBatch(batch);
  }
}


// ❌ WRONG - Promise.all with map for sequential processing
async function processAllBad(userIds: string[]): Promise<void> {
  await Promise.all(userIds.map(async (userId) => {
    await processUser(userId); // May overwhelm system with parallel requests
  }));
}
```

### F. Async/Await Best Practices

**DO:**
- ✅ Use async/await for all asynchronous operations
- ✅ Use Promise.all for parallel operations
- ✅ Use Promise.allSettled when some operations can fail
- ✅ Use try-catch for error handling
- ✅ Return promises from async functions (implicit)
- ✅ Use async generators for streaming data

**DON'T:**
- ❌ Use callbacks for new code
- ❌ Mix async/await with .then() chains
- ❌ Forget to await promises
- ❌ Use async without actually awaiting anything
- ❌ Use Promise constructor unless wrapping callback-based APIs

```typescript
// ✅ CORRECT - Wrapping callback API with Promise
function readFileAsync(path: string): Promise<string> {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8', (error, data) => {
      if (error) reject(error);
      else resolve(data);
    });
  });
}

// Then use with async/await
async function processFile(path: string): Promise<void> {
  const content = await readFileAsync(path);
  await processContent(content);
}
```

---

## 7. Functional Programming Patterns (MANDATORY)

### A. Functional Programming Requirements

**ALWAYS prefer functional programming patterns when applicable:**
- Write pure functions without side effects
- Use immutable data structures
- Leverage higher-order functions
- Compose functions for complex operations
- Avoid mutating state

**Benefits:**
- **Type Safety**: Immutability + types = compile-time guarantees
- **Testability**: Pure functions are trivial to test
- **Reliability**: No side effects = predictable behavior
- **Concurrency**: Immutable data is thread-safe
- **Maintainability**: Less hidden state to track

### B. Pure Functions

```typescript
// ✅ CORRECT - Pure function (PREFERRED)
function calculateTotal(items: readonly Item[], taxRate: number): number {
  const subtotal = items.reduce((sum, item) => sum + item.price, 0);
  return subtotal * (1 + taxRate);
}

function normalizeEmail(email: string): string {
  return email.toLowerCase().trim();
}


// ❌ WRONG - Function with side effects
let totalCalculations = 0;

function calculateTotalBad(items: Item[]): number {
  totalCalculations++; // Side effect!
  return items.reduce((sum, item) => sum + item.price, 0);
}
```

### C. Immutability

```typescript
// ✅ CORRECT - Immutable operations (PREFERRED)
interface User {
  readonly id: string;
  readonly email: string;
  name: string;
}

function updateUserName(user: Readonly<User>, newName: string): User {
  return { ...user, name: newName };
}

function addItem<T>(items: readonly T[], newItem: T): readonly T[] {
  return [...items, newItem];
}

function removeItem<T>(items: readonly T[], index: number): readonly T[] {
  return [...items.slice(0, index), ...items.slice(index + 1)];
}

function updateItem<T>(
  items: readonly T[],
  index: number,
  updater: (item: T) => T
): readonly T[] {
  return items.map((item, i) => (i === index ? updater(item) : item));
}


// ❌ WRONG - Mutating operations
function addItemBad<T>(items: T[], newItem: T): T[] {
  items.push(newItem); // Mutation!
  return items;
}

function updateUserBad(user: User, newName: string): User {
  user.name = newName; // Mutation!
  return user;
}
```

### D. Higher-Order Functions

```typescript
// ✅ CORRECT - Higher-order functions (PREFERRED)
type Predicate<T> = (value: T) => boolean;
type Mapper<T, U> = (value: T) => U;

function filter<T>(items: readonly T[], predicate: Predicate<T>): T[] {
  return items.filter(predicate);
}

function map<T, U>(items: readonly T[], mapper: Mapper<T, U>): U[] {
  return items.map(mapper);
}

function compose<T, U, V>(
  f: (x: U) => V,
  g: (x: T) => U
): (x: T) => V {
  return (x: T) => f(g(x));
}


// ✅ CORRECT - Partial application
function multiply(x: number): (y: number) => number {
  return (y: number) => x * y;
}

const double = multiply(2);
const triple = multiply(3);

console.log(double(5)); // 10
console.log(triple(5)); // 15


// ✅ CORRECT - Currying
function curry<T, U, V>(
  fn: (a: T, b: U) => V
): (a: T) => (b: U) => V {
  return (a: T) => (b: U) => fn(a, b);
}

const add = (a: number, b: number) => a + b;
const curriedAdd = curry(add);

const add5 = curriedAdd(5);
console.log(add5(10)); // 15
```

### E. Function Composition

```typescript
// ✅ CORRECT - Function composition (PREFERRED)
type Transform<T> = (value: T) => T;

function pipe<T>(...fns: Transform<T>[]): Transform<T> {
  return (value: T) => fns.reduce((acc, fn) => fn(acc), value);
}

// Example: Text processing pipeline
const processText = pipe(
  (s: string) => s.trim(),
  (s: string) => s.toLowerCase(),
  (s: string) => s.replace(/\s+/g, '-'),
  (s: string) => s.replace(/[^\w-]/g, '')
);

const slug = processText("  Hello, World!  ");
// "hello-world"


// ✅ CORRECT - Async function composition
type AsyncTransform<T> = (value: T) => Promise<T>;

function pipeAsync<T>(...fns: AsyncTransform<T>[]): AsyncTransform<T> {
  return async (value: T) => {
    let result = value;
    for (const fn of fns) {
      result = await fn(result);
    }
    return result;
  };
}

// Example: User processing pipeline
const processUser = pipeAsync(
  async (user: User) => validateUser(user),
  async (user: User) => enrichUserData(user),
  async (user: User) => saveUser(user)
);

await processUser(newUser);
```

### F. Functional Data Transformations

```typescript
// ✅ CORRECT - Functional array operations (PREFERRED)
const users: User[] = [
  { id: '1', name: 'Alice', age: 30, active: true },
  { id: '2', name: 'Bob', age: 25, active: false },
  { id: '3', name: 'Charlie', age: 35, active: true },
];

// Map: Transform each element
const userNames = users.map((user) => user.name);

// Filter: Select elements matching condition
const activeUsers = users.filter((user) => user.active);

// Reduce: Aggregate to single value
const totalAge = users.reduce((sum, user) => sum + user.age, 0);

// Chaining: Complex transformations
const activeUserNames = users
  .filter((user) => user.active)
  .map((user) => user.name)
  .sort();


// ✅ CORRECT - Functional object operations
function mapObject<T, U>(
  obj: Record<string, T>,
  mapper: (value: T, key: string) => U
): Record<string, U> {
  return Object.fromEntries(
    Object.entries(obj).map(([key, value]) => [key, mapper(value, key)])
  );
}

function filterObject<T>(
  obj: Record<string, T>,
  predicate: (value: T, key: string) => boolean
): Record<string, T> {
  return Object.fromEntries(
    Object.entries(obj).filter(([key, value]) => predicate(value, key))
  );
}

// Usage
const prices = { apple: 0.5, banana: 0.3, orange: 0.7 };
const discountedPrices = mapObject(prices, (price) => price * 0.9);
const expensiveItems = filterObject(prices, (price) => price > 0.5);
```

### G. Monadic Patterns (Option/Result)

```typescript
// ✅ CORRECT - Option type for nullable values (PREFERRED)
type Option<T> = { type: 'some'; value: T } | { type: 'none' };

const Some = <T>(value: T): Option<T> => ({ type: 'some', value });
const None = <T>(): Option<T> => ({ type: 'none' });

function map<T, U>(option: Option<T>, fn: (value: T) => U): Option<U> {
  return option.type === 'some' ? Some(fn(option.value)) : None();
}

function flatMap<T, U>(
  option: Option<T>,
  fn: (value: T) => Option<U>
): Option<U> {
  return option.type === 'some' ? fn(option.value) : None();
}

function getOrElse<T>(option: Option<T>, defaultValue: T): T {
  return option.type === 'some' ? option.value : defaultValue;
}

// Usage
const maybeUser = Some({ id: '1', name: 'Alice' });
const userName = map(maybeUser, (user) => user.name);
const finalName = getOrElse(userName, 'Unknown');


// ✅ CORRECT - Result type for error handling (PREFERRED)
type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

function mapResult<T, U, E>(
  result: Result<T, E>,
  fn: (value: T) => U
): Result<U, E> {
  return result.success
    ? { success: true, data: fn(result.data) }
    : result;
}

async function flatMapResult<T, U, E>(
  result: Result<T, E>,
  fn: (value: T) => Promise<Result<U, E>>
): Promise<Result<U, E>> {
  return result.success ? fn(result.data) : result;
}

// Usage
async function getUserProfile(userId: string): Promise<Result<Profile>> {
  const userResult = await fetchUser(userId);
  return flatMapResult(userResult, async (user) => {
    const profileResult = await fetchProfile(user.id);
    return profileResult;
  });
}
```

### H. Immutable Data Structures with Immer

```typescript
// ✅ CORRECT - Using Immer for complex immutable updates
import { produce } from 'immer';

interface State {
  users: User[];
  settings: Settings;
  cache: Map<string, any>;
}

// Without Immer (verbose)
function addUserVerbose(state: State, user: User): State {
  return {
    ...state,
    users: [...state.users, user],
  };
}

// With Immer (concise)
function addUser(state: State, user: User): State {
  return produce(state, (draft) => {
    draft.users.push(user);
  });
}

// Complex nested updates
function updateNestedSetting(
  state: State,
  userId: string,
  settingKey: string,
  value: any
): State {
  return produce(state, (draft) => {
    const user = draft.users.find((u) => u.id === userId);
    if (user && user.settings) {
      user.settings[settingKey] = value;
    }
  });
}
```

### I. Functional Error Handling

```typescript
// ✅ CORRECT - Railway-oriented programming (PREFERRED)
type Success<T> = { type: 'success'; value: T };
type Failure<E> = { type: 'failure'; error: E };
type Result<T, E> = Success<T> | Failure<E>;

const success = <T>(value: T): Success<T> => ({ type: 'success', value });
const failure = <E>(error: E): Failure<E> => ({ type: 'failure', error });

function bind<T, U, E>(
  result: Result<T, E>,
  fn: (value: T) => Result<U, E>
): Result<U, E> {
  return result.type === 'success' ? fn(result.value) : result;
}

// Pipeline with error handling
async function processUserPipeline(
  userId: string
): Promise<Result<void, string>> {
  let result = await fetchUserResult(userId);
  result = bind(result, validateUser);
  result = bind(result, enrichUser);
  result = bind(result, saveUser);
  return result;
}
```

### J. When to Use Functional Programming

**✅ USE functional programming for:**
- Data transformations (map, filter, reduce)
- Business logic calculations
- Validation and parsing
- State management
- Data pipelines
- Pure utility functions

**⚠️ USE CAUTION for:**
- I/O operations (inherently have side effects)
- Performance-critical code (profiling may show imperative is faster)
- Very complex logic (readability may suffer)

**❌ DON'T force functional style when:**
- Imperative code is significantly clearer
- Dealing with external state (databases, files, APIs)
- Framework/library requires imperative approach

---

## 8. Code Style (Minimalistic & Clean)

### A. Single Responsibility

```typescript
// ✅ CORRECT - Small, focused functions
function validateEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function normalizeEmail(email: string): string {
  return email.toLowerCase().trim();
}

function isEmailAvailable(email: string): Promise<boolean> {
  return database.checkEmailAvailability(email);
}

// ❌ WRONG - Function doing too much
function processEmailBad(email: string): Promise<boolean> {
  const normalized = email.toLowerCase().trim();
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(normalized)) {
    return Promise.resolve(false);
  }
  return database.checkEmailAvailability(normalized);
}
```

### B. Pure Functions Preferred

```typescript
// ✅ CORRECT - Pure function
function calculateDiscount(price: number, discountRate: number): number {
  return price * (1 - discountRate);
}

// ✅ CORRECT - Explicitly impure (side effects documented)
/**
 * Saves user to database.
 * @param user - User to save
 * @returns Promise resolving when save completes
 * @sideEffect Writes to database
 */
async function saveUser(user: User): Promise<void> {
  await database.save(user);
}

// ❌ WRONG - Hidden side effects
function calculateTotal(items: Item[]): number {
  logAnalytics('calculate_total', items); // Hidden side effect!
  return items.reduce((sum, item) => sum + item.price, 0);
}
```

### C. Early Returns

```typescript
// ✅ CORRECT - Early returns for guard clauses
function processUser(user: User | null): string {
  if (!user) return 'No user';
  if (!user.email) return 'No email';
  if (!validateEmail(user.email)) return 'Invalid email';
  
  return `User: ${user.email}`;
}

// ❌ WRONG - Nested if statements
function processBad(user: User | null): string {
  if (user) {
    if (user.email) {
      if (validateEmail(user.email)) {
        return `User: ${user.email}`;
      } else {
        return 'Invalid email';
      }
    } else {
      return 'No email';
    }
  } else {
    return 'No user';
  }
}
```

### D. Named Parameters for Clarity

```typescript
// ✅ CORRECT - Object parameter for multiple options
interface CreateUserOptions {
  email: string;
  name: string;
  role?: UserRole;
  sendWelcomeEmail?: boolean;
}

function createUser(options: CreateUserOptions): User {
  const { email, name, role = 'user', sendWelcomeEmail = true } = options;
  // Implementation
  return {} as User;
}

// Usage
const user = createUser({
  email: 'user@example.com',
  name: 'John Doe',
  sendWelcomeEmail: false
});

// ❌ WRONG - Positional boolean parameters
function createBad(email: string, name: string, role: string, send: boolean): User {
  // Implementation
  return {} as User;
}

// Usage is unclear
const badUser = createBad('user@example.com', 'John', 'user', false); // What is false?
```

## 9. Testing Requirements (MANDATORY)

### A. Test Coverage

- **Minimum 80% code coverage** for all business logic
- **100% coverage** for critical paths (auth, payments, etc.)
- Tests MUST pass before code review/merge
- Tests MUST be fast (< 100ms per test ideally)

### B. Test Structure with Vitest

```typescript
// ✅ CORRECT - Comprehensive test suite
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { UserService } from './user-service';
import type { UserRepository } from './user-repository';

describe('UserService', () => {
  let service: UserService;
  let mockRepository: UserRepository;

  beforeEach(() => {
    mockRepository = {
      findById: vi.fn(),
      save: vi.fn(),
      delete: vi.fn(),
    };
    service = new UserService(mockRepository);
  });

  describe('getUserById', () => {
    it('returns user when found', async () => {
      const mockUser = { id: '123', name: 'John', email: 'john@example.com' };
      vi.mocked(mockRepository.findById).mockResolvedValue(mockUser);

      const result = await service.getUserById('123');

      expect(result).toEqual(mockUser);
      expect(mockRepository.findById).toHaveBeenCalledWith('123');
    });

    it('returns null when user not found', async () => {
      vi.mocked(mockRepository.findById).mockResolvedValue(null);

      const result = await service.getUserById('999');

      expect(result).toBeNull();
    });

    it('throws error when repository fails', async () => {
      vi.mocked(mockRepository.findById).mockRejectedValue(
        new Error('Database error')
      );

      await expect(service.getUserById('123')).rejects.toThrow('Database error');
    });
  });

  describe('validateEmail', () => {
    it.each([
      ['user@example.com', true],
      ['user.name@example.co.uk', true],
      ['invalid', false],
      ['@example.com', false],
      ['user@', false],
      ['', false],
    ])('validates "%s" as %s', (email, expected) => {
      expect(service.validateEmail(email)).toBe(expected);
    });
  });
});
```

### C. Type Testing

```typescript
// ✅ CORRECT - Testing type guards
import { describe, it, expect } from 'vitest';
import { isUser } from './type-guards';

describe('isUser type guard', () => {
  it('returns true for valid user object', () => {
    const validUser = {
      id: '123',
      email: 'user@example.com',
      name: 'John',
      role: 'user',
      createdAt: '2026-01-17T10:00:00Z',
    };

    expect(isUser(validUser)).toBe(true);
  });

  it('returns false for invalid user object', () => {
    expect(isUser(null)).toBe(false);
    expect(isUser(undefined)).toBe(false);
    expect(isUser({})).toBe(false);
    expect(isUser({ id: 123 })).toBe(false); // Wrong type
    expect(isUser({ id: '123' })).toBe(false); // Missing fields
  });
});
```

## 10. Project Structure

```
project/
├── src/
│   ├── types/              # Type definitions
│   │   ├── user.ts
│   │   ├── result.ts
│   │   └── index.ts
│   ├── utils/              # Utility functions
│   │   ├── validation.ts
│   │   ├── formatting.ts
│   │   └── index.ts
│   ├── services/           # Business logic
│   │   ├── user-service.ts
│   │   └── index.ts
│   ├── repositories/       # Data access
│   │   ├── user-repository.ts
│   │   └── index.ts
│   └── index.ts            # Main entry point
├── tests/
│   ├── unit/
│   │   ├── utils.test.ts
│   │   └── services.test.ts
│   └── integration/
│       └── api.test.ts
├── docs/                   # Generated documentation (in .gitignore)
├── .gitignore
├── package.json
├── tsconfig.json
├── typedoc.json
├── vitest.config.ts
└── README.md
```

## 11. Complete Example

```typescript
/**
 * @file user-service.ts
 * @description User management service with validation and caching.
 */

import type { Result } from './types/result';
import type { User, UserId, Email } from './types/user';
import type { UserRepository } from './repositories/user-repository';
import { Cache } from './utils/cache';

/**
 * User service for managing user operations.
 * 
 * Provides CRUD operations with caching and validation.
 * All operations return Result types for explicit error handling.
 * 
 * @example
 * ```typescript
 * const repository = new PrismaUserRepository();
 * const service = new UserService(repository);
 * 
 * const result = await service.getUserById(userId);
 * if (result.success) {
 *   console.log('User:', result.data.name);
 * } else {
 *   console.error('Error:', result.error.message);
 * }
 * ```
 * 
 * @public
 */
export class UserService {
  private cache: Cache<UserId, User>;

  /**
   * Creates a new user service.
   * 
   * @param repository - Repository for user data access
   * @param cacheTTL - Cache TTL in milliseconds (default: 60000)
   */
  constructor(
    private readonly repository: UserRepository,
    cacheTTL: number = 60000
  ) {
    this.cache = new Cache<UserId, User>(cacheTTL);
  }

  /**
   * Retrieves a user by ID.
   * 
   * Checks cache first, then queries repository if not cached.
   * 
   * @param userId - The user's unique identifier
   * @returns Result containing the user or error
   * 
   * @example
   * ```typescript
   * const result = await service.getUserById(userId);
   * if (result.success) {
   *   console.log('User:', result.data.name);
   * }
   * ```
   * 
   * @public
   */
  async getUserById(userId: UserId): Promise<Result<User>> {
    try {
      // Check cache
      const cached = this.cache.get(userId);
      if (cached) {
        return { success: true, data: cached };
      }

      // Query repository
      const user = await this.repository.findById(userId);
      if (!user) {
        return {
          success: false,
          error: new Error('User not found'),
        };
      }

      // Cache result
      this.cache.set(userId, user);

      return { success: true, data: user };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error : new Error('Unknown error'),
      };
    }
  }

  /**
   * Validates an email address format.
   * 
   * @param email - The email to validate
   * @returns `true` if valid, `false` otherwise
   * 
   * @example
   * ```typescript
   * if (service.validateEmail('user@example.com')) {
   *   console.log('Valid email');
   * }
   * ```
   * 
   * @public
   */
  validateEmail(email: string): boolean {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  }

  /**
   * Creates a new user.
   * 
   * Validates email format before creation.
   * 
   * @param data - User creation data
   * @param data.email - User's email address
   * @param data.name - User's display name
   * @param data.role - User's role (default: 'user')
   * 
   * @returns Result containing the created user or error
   * 
   * @throws Never throws (returns Result with error instead)
   * 
   * @example
   * ```typescript
   * const result = await service.createUser({
   *   email: 'user@example.com',
   *   name: 'John Doe',
   *   role: 'user'
   * });
   * ```
   * 
   * @public
   */
  async createUser(data: {
    email: string;
    name: string;
    role?: UserRole;
  }): Promise<Result<User>> {
    // Validate email
    if (!this.validateEmail(data.email)) {
      return {
        success: false,
        error: new Error('Invalid email format'),
      };
    }

    try {
      const user = await this.repository.create({
        email: data.email as Email,
        name: data.name,
        role: data.role ?? 'user',
      });

      return { success: true, data: user };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error : new Error('Unknown error'),
      };
    }
  }
}
```

## 12. Security & Dependency Management (MANDATORY)

### A. Automated Dependency Management

**Use npm/yarn/pnpm to manage and lock dependencies:**

```bash
# Install/sync dependencies
npm install

# Add a new dependency
npm install package-name

# Update dependencies
npm update

# Verify dependency integrity
npm audit signatures
```

### B. Vulnerability Scanning & Security

**Mandatory security checks for ALL TypeScript projects:**

1. **Vulnerability Scan**:
   ```bash
   # Scan for known vulnerabilities
   npm audit
   ```
   - Agents MUST fix all HIGH/CRITICAL vulnerabilities before delivery.
   - Also audit `@types/` packages -- they can introduce transitive vulnerabilities.

2. **Supply Chain Audit**:
   - Verify `package-lock.json` integrity
   - Audit licenses for compliance
   - Use `npm audit signatures` to verify registry signatures
   - Review `@types/` packages for version alignment with their runtime counterparts

### C. Dependency File

```json
{
  "name": "my-project",
  "version": "1.0.0",
  "engines": {
    "node": ">=20.0.0"
  },
  "dependencies": {
    "express": "^4.18.0"
  },
  "devDependencies": {
    "@types/express": "^4.17.0",
    "typescript": "^5.3.0",
    "eslint": "^8.56.0",
    "prettier": "^3.2.0",
    "vitest": "^1.2.0"
  }
}
```

---

## 13. Deployment Checklist

### Pre-Production Validation

#### Agent Code Generation (MANDATORY)
- [ ] **TypeScript compilation successful**: `npx tsc --noEmit` passes
- [ ] **All public APIs documented**: TypeDoc comments on all exports
- [ ] **Documentation generation works**: `npm run docs` succeeds
- [ ] **No documentation warnings**: `npm run docs:check` passes
- [ ] **Linting passes**: `npm run lint` returns exit code 0
- [ ] **All tests passing**: `npm test` returns exit code 0
- [ ] **Test coverage ≥ 80%**: `npm run test:coverage` shows adequate coverage
- [ ] **No `any` types**: Strict mode enabled, all types explicit
- [ ] **No `@ts-ignore` without justification**: Code compiles cleanly
- [ ] **Code formatted**: `npm run format:check` passes

### Code Quality
- [ ] All functions have explicit return types
- [ ] All functions are small (< 20 lines ideally)
- [ ] All types are well-named and documented
- [ ] Branded types used for domain primitives
- [ ] Immutability enforced where appropriate
- [ ] Pure functions used where possible
- [ ] Functional programming patterns used for data transformations
- [ ] Async/await used instead of promise chains
- [ ] No nested callbacks
- [ ] Parallel operations use Promise.all/allSettled
- [ ] Early returns for guard clauses

### Security
- [ ] Input validation on all external data
- [ ] No eval or Function constructors
- [ ] No prototype pollution vulnerabilities
- [ ] Secrets not hardcoded

## 14. Why This Configuration Works

1. **Strict TypeScript**: Catches 30-40% more bugs at compile time, prevents runtime type errors.

2. **TypeDoc Documentation**: Auto-generated docs stay in sync with code, reduces onboarding time by 40%+, enables API discoverability.

3. **Async/Await First**: Reduces cognitive load by 60%, eliminates callback hell, makes async code read like synchronous code.

4. **Functional Programming**: Immutability + pure functions = predictable, testable, maintainable code. Reduces bugs by 40-50%.

5. **Result Types**: Explicit error handling, no hidden exceptions, type-safe error states.

6. **Branded Types**: Prevents primitive obsession, catches logic errors at compile time.

7. **Pure Functions**: Easier to test, reason about, and refactor. 100% testable without mocks.

8. **Minimalistic Code**: Faster to understand, fewer bugs, easier maintenance.

9. **Mandatory Testing**: Catches regressions, enables confident refactoring.

10. **Agent Verification**: Ensures all generated code compiles and works, eliminates broken examples.

---

## 15. Quick Reference

### Common Commands

```bash
# Build & Check
npx tsc --noEmit
npx tsc --build

# Test
npm test
npm run test:watch
npm run test:coverage

# Lint & Format
npm run lint
npm run lint:fix
npx prettier --check .
npx prettier --write .

# Documentation
npx typedoc --out docs src/

# Run
npx ts-node src/index.ts
npx tsx src/index.ts
```

### Type Patterns Cheat Sheet

```typescript
// Branded types
type UserId = string & { readonly brand: unique symbol };
const userId = id as UserId;

// Result type
type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

// Guard function
function isUser(obj: unknown): obj is User {
  return typeof obj === 'object' && obj !== null && 'id' in obj;
}

// Readonly deep
type DeepReadonly<T> = {
  readonly [P in keyof T]: DeepReadonly<T[P]>;
};

// Optional fields
type PartialBy<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;
```

### tsconfig.json Essentials

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "exactOptionalPropertyTypes": true
  }
}
```

### Project Structure

```
my_project/
├── src/
│   ├── index.ts          # Entry point
│   ├── types/            # Type definitions
│   ├── domain/           # Domain models
│   ├── services/         # Business logic
│   └── utils/            # Utilities
├── tests/
├── tsconfig.json
└── package.json
```

---

## References

- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [TypeDoc Documentation](https://typedoc.org/)
- [Vitest Documentation](https://vitest.dev/)
- [Type Safety Best Practices](https://www.typescriptlang.org/docs/handbook/declaration-files/do-s-and-don-ts.html)


**End of TypeScript Development Guidelines**
