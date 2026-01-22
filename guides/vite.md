# Vite Development Guidelines
This document provides mandatory coding standards and development practices for modern Vite applications with TypeScript

---
Agent Profile: The Vite Expert
Role: Senior Frontend Engineer & Vite Build Specialist
Objective: Generate production-ready, type-safe, fully documented, highly performant, and maintainable Vite applications.
Tools: Vite 5.x, TypeScript 5.x, Vitest, TypeDoc, Modern frameworks (React/Vue/Svelte/Vanilla).

## Core Philosophies

The agent must adhere to the "VITE-FIRST" principles for every Vite project:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Instant Dev Server**: Leverage Vite's instant HMR, native ESM, on-demand compilation.
**Type Safety First**: TypeScript strict mode, no `any`, comprehensive type coverage.
**Efficient Builds**: Optimized production builds with Rollup, tree-shaking, code splitting.
**Fast Feedback**: Hot module replacement (HMR) for instant updates, fast test execution.
**Isolated Dependencies**: Use Vite's dependency pre-bundling for optimal performance.
**Reactive Patterns**: Signals/reactive primitives preferred, async/await for all async operations.
**Systematic Testing**: Unit tests with Vitest, 80%+ coverage, tests must pass.
**Tested Code**: Comprehensive unit tests for all logic, component tests for UI.
**Documented APIs**: TypeDoc comments for all exports, auto-generated documentation.
**Verified Builds**: Agent-generated code MUST compile and pass all tests before delivery.

---

## 1. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated Vite code compiles and passes tests before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY Vite code, the agent MUST:**

1. **TypeScript Compilation Check**:
   ```bash
   # Verify TypeScript compiles without errors
   npm run type-check
   # OR
   npx tsc --noEmit
   # Exit code MUST be 0
   ```

2. **Build Verification**:
   ```bash
   # Verify project builds successfully
   npm run build
   # Exit code MUST be 0
   
   # Check build output
   ls dist/
   # Verify assets are generated
   ```

3. **Development Server Check**:
   ```bash
   # Verify dev server starts without errors
   npm run dev &
   sleep 3
   curl http://localhost:5173 | grep -q "html" && echo "OK" || echo "FAIL"
   kill %1
   ```

4. **Test Execution**:
   ```bash
   # Run all unit tests
   npm run test
   # Exit code MUST be 0, all tests pass
   
   # Run with coverage
   npm run test:coverage
   # Coverage should be > 80%
   ```

5. **Linting Check**:
   ```bash
   # Verify code passes linting
   npm run lint
   # Exit code MUST be 0
   ```

6. **Documentation Generation**:
   ```bash
   # Generate TypeDoc documentation
   npm run docs
   # Verify docs/ directory is created
   ```

### B. Error Correction Process

If verification fails:

1. **Read the error message** (Vite/TypeScript errors are descriptive)
2. **Identify the root cause** (type error, build config, missing dependency, etc.)
3. **Fix the issue** following Vite best practices
4. **Re-run verification** until all checks pass
5. **Document any non-obvious fixes**

### C. Prohibited Practices

**NEVER deliver Vite code that:**
- ❌ Has TypeScript compilation errors
- ❌ Has build failures
- ❌ Fails tests
- ❌ Lacks tests for new functionality
- ❌ Lacks TypeDoc comments for public APIs
- ❌ Uses `any` type without justification
- ❌ Uses CommonJS instead of ESM modules
- ❌ Has console errors in development mode
- ❌ Uses `.then()` promise chains (use `async`/`await` instead)
- ❌ Uses nested callbacks (use `async`/`await` instead)
- ❌ **Fixes bugs without adding regression tests first**
- ❌ **Writes implementation before writing tests (violates TDD)**
- ❌ **Skips Red-Green-Refactor cycle for new features**

---

## 1A. Test-Driven Development (TDD) Protocol (MANDATORY)

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

### Example TDD Workflow for Utility Function

```typescript
// Step 1: RED - Write failing test first (src/utils/validation.test.ts)
import { describe, it, expect } from 'vitest';
import { validateEmail } from './validation';

describe('validateEmail', () => {
  it('accepts valid email addresses', () => {
    expect(validateEmail('user@example.com')).toBe(true);
    expect(validateEmail('test.user@domain.co.uk')).toBe(true);
  });
  
  it('rejects invalid email addresses', () => {
    expect(validateEmail('invalid')).toBe(false);
    expect(validateEmail('user@')).toBe(false);
    expect(validateEmail('@domain.com')).toBe(false);
  });
  
  it('rejects empty strings', () => {
    expect(validateEmail('')).toBe(false);
  });
});

// Run: npm test
// ❌ FAILS - validateEmail doesn't exist yet

// Step 2: GREEN - Write minimal implementation (src/utils/validation.ts)
/**
 * Validates an email address format.
 * 
 * @param email - The email address to validate
 * @returns True if email is valid, false otherwise
 * 
 * @example
 * ```typescript
 * validateEmail('user@example.com'); // true
 * validateEmail('invalid'); // false
 * ```
 */
export function validateEmail(email: string): boolean {
  if (!email) return false;
  
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

// Run: npm test
// ✅ PASSES - tests pass

// Step 3: REFACTOR - Improve with more robust validation
const EMAIL_REGEX = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

export function validateEmail(email: string): boolean {
  if (!email || typeof email !== 'string') {
    return false;
  }
  
  // Check length constraints
  if (email.length < 3 || email.length > 254) {
    return false;
  }
  
  return EMAIL_REGEX.test(email);
}
// Tests still pass ✓
```

### Example TDD for Reactive Signal Store

```typescript
// Step 1: RED - Write failing test first
import { describe, it, expect } from 'vitest';
import { createCounter } from './counter-store';

describe('createCounter', () => {
  it('initializes with default value of 0', () => {
    const counter = createCounter();
    expect(counter.value).toBe(0);
  });
  
  it('increments the counter', () => {
    const counter = createCounter();
    counter.increment();
    expect(counter.value).toBe(1);
  });
  
  it('decrements the counter', () => {
    const counter = createCounter();
    counter.increment();
    counter.increment();
    counter.decrement();
    expect(counter.value).toBe(1);
  });
  
  it('resets to initial value', () => {
    const counter = createCounter(5);
    counter.increment();
    counter.reset();
    expect(counter.value).toBe(5);
  });
});

// Run: npm test
// ❌ FAILS - createCounter doesn't exist yet

// Step 2: GREEN - Write minimal implementation
import { signal } from '@preact/signals-core';

/**
 * Creates a reactive counter store.
 * 
 * @param initialValue - Initial counter value (default: 0)
 * @returns Counter store with value and methods
 * 
 * @example
 * ```typescript
 * const counter = createCounter(10);
 * counter.increment(); // counter.value = 11
 * counter.decrement(); // counter.value = 10
 * counter.reset();     // counter.value = 10
 * ```
 */
export function createCounter(initialValue = 0) {
  const count = signal(initialValue);
  
  return {
    get value() {
      return count.value;
    },
    increment() {
      count.value++;
    },
    decrement() {
      count.value--;
    },
    reset() {
      count.value = initialValue;
    }
  };
}

// Run: npm test
// ✅ PASSES - tests pass

// Step 3: REFACTOR - Add type safety and additional features
interface Counter {
  readonly value: number;
  increment(): void;
  decrement(): void;
  reset(): void;
  set(value: number): void;
}

export function createCounter(initialValue = 0): Counter {
  const count = signal(initialValue);
  
  return {
    get value() {
      return count.value;
    },
    increment() {
      count.value++;
    },
    decrement() {
      count.value--;
    },
    reset() {
      count.value = initialValue;
    },
    set(value: number) {
      count.value = value;
    }
  };
}
// Tests still pass ✓
```

---

## 1B. Bug Fix Protocol (MANDATORY)

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
// Bug Report #9451: parseJSON fails silently on malformed JSON

// Step 1-2: Write test that reproduces the bug
import { describe, it, expect } from 'vitest';
import { parseJSON } from './json-utils';

describe('parseJSON - Bug #9451', () => {
  it('throws error on malformed JSON - Bug #9451', () => {
    // Bug: parseJSON returned undefined instead of throwing error
    // Discovered: 2026-01-18
    // This test prevents regression
    
    expect(() => {
      parseJSON('{ invalid json }');
    }).toThrow('Invalid JSON');
  });
  
  it('handles trailing commas', () => {
    // Additional edge case discovered during bug investigation
    expect(() => {
      parseJSON('{"key": "value",}');
    }).toThrow('Invalid JSON');
  });
});

// Run: npm test
// ❌ FAILS - parseJSON returns undefined instead of throwing

// Step 3: Fix the bug
/**
 * Safely parses JSON string.
 * 
 * @param jsonString - JSON string to parse
 * @returns Parsed JSON object
 * @throws {Error} If JSON is malformed
 * 
 * @example
 * ```typescript
 * const data = parseJSON('{"key": "value"}'); // { key: "value" }
 * parseJSON('invalid'); // throws Error
 * ```
 */
export function parseJSON<T = unknown>(jsonString: string): T {
  // FIX: Don't catch errors, let them propagate
  try {
    return JSON.parse(jsonString) as T;
  } catch (error) {
    throw new Error(
      `Invalid JSON: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}

// Run: npm test
// ✅ PASSES - bug fixed, regression prevented ✓
```

### Example Bug Fix for Async Function

```typescript
// Bug Report #9452: fetchUser doesn't handle 404 errors correctly

// Step 1-2: Write test that reproduces the bug
import { describe, it, expect, vi } from 'vitest';
import { fetchUser } from './api';

describe('fetchUser - Bug #9452', () => {
  it('throws error on 404 - Bug #9452', async () => {
    // Bug: fetchUser returned null instead of throwing on 404
    // Discovered: 2026-01-18
    // This test prevents regression
    
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'Not Found'
    } as Response);
    
    await expect(fetchUser('invalid-id')).rejects.toThrow('User not found');
  });
  
  it('handles network errors', async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));
    
    await expect(fetchUser('123')).rejects.toThrow('Network error');
  });
});

// Run: npm test
// ❌ FAILS - fetchUser returns null on 404

// Step 3: Fix the bug
/**
 * Fetches user data from API.
 * 
 * @param userId - The user ID to fetch
 * @returns Promise resolving to user data
 * @throws {Error} If user not found or network error occurs
 * 
 * @example
 * ```typescript
 * const user = await fetchUser('user-123');
 * console.log(user.name);
 * ```
 */
export async function fetchUser(userId: string): Promise<User> {
  try {
    const response = await fetch(`/api/users/${userId}`);
    
    // FIX: Check response status and throw appropriate errors
    if (!response.ok) {
      if (response.status === 404) {
        throw new Error('User not found');
      }
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Unknown error occurred');
  }
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
- ❌ Use `it.skip()` to ignore failing tests

---

## 2. Mandatory Setup Requirements

### A. Project Initialization

```bash
# Create new Vite project with TypeScript
npm create vite@latest my-app -- --template vanilla-ts
# OR with React
npm create vite@latest my-app -- --template react-ts
# OR with Vue
npm create vite@latest my-app -- --template vue-ts

cd my-app
npm install
```

### B. TypeScript Configuration

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "module": "ESNext",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "skipLibCheck": true,

    /* Bundler mode */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    
    /* Linting */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "noImplicitReturns": true,
    "noUncheckedIndexedAccess": true,
    
    /* Paths */
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

```json
// tsconfig.node.json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true,
    "strict": true
  },
  "include": ["vite.config.ts"]
}
```

### C. Vite Configuration

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  // Resolve aliases
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
  
  // Development server
  server: {
    port: 5173,
    strictPort: false,
    host: true,
    open: false,
  },
  
  // Build options
  build: {
    target: 'esnext',
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: true,
    minify: 'esbuild',
    
    // Rollup options
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'], // Adjust for your framework
        },
      },
    },
    
    // Chunk size warnings
    chunkSizeWarningLimit: 1000,
  },
  
  // Optimizations
  optimizeDeps: {
    include: ['@preact/signals-core'], // Pre-bundle dependencies
  },
  
  // Test configuration
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/test/',
        '**/*.test.ts',
        '**/*.test.tsx',
        '**/*.spec.ts',
        '**/*.spec.tsx',
      ],
    },
  },
});
```

### D. Essential Dependencies

```json
{
  "name": "my-vite-app",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest",
    "test:coverage": "vitest run --coverage",
    "test:ui": "vitest --ui",
    "type-check": "tsc --noEmit",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "docs": "typedoc --out docs src",
    "docs:serve": "npx http-server docs"
  },
  "dependencies": {
    "@preact/signals-core": "^1.5.0"
  },
  "devDependencies": {
    "@types/node": "^20.11.0",
    "@typescript-eslint/eslint-plugin": "^6.19.0",
    "@typescript-eslint/parser": "^6.19.0",
    "@vitest/ui": "^1.2.0",
    "eslint": "^8.56.0",
    "jsdom": "^23.2.0",
    "typedoc": "^0.25.0",
    "typescript": "^5.3.0",
    "vite": "^5.0.0",
    "vitest": "^1.2.0"
  }
}
```

---

## 2A. TDD Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code in Vite projects.**

### TDD Cycle Diagram

```
    ┌─────────────────────────────────────────────────────────────┐
    │                    TDD CYCLE FOR VITE                       │
    └─────────────────────────────────────────────────────────────┘

                           ┌──────────┐
                           │  START   │
                           └────┬─────┘
                                │
                                ▼
           ┌────────────────────────────────────────┐
           │  🔴 RED: Write a Failing Test First    │
           │  ─────────────────────────────────────  │
           │  • Create test file (*.test.ts)        │
           │  • Define expected behavior            │
           │  • Run: npm test -- --watch           │
           │  • Verify test FAILS                   │
           └────────────────────┬───────────────────┘
                                │
                                ▼
           ┌────────────────────────────────────────┐
           │  🟢 GREEN: Make the Test Pass          │
           │  ─────────────────────────────────────  │
           │  • Write MINIMAL implementation        │
           │  • No optimization yet                 │
           │  • Just enough to pass the test        │
           │  • Run: npm test                       │
           │  • Verify test PASSES                  │
           └────────────────────┬───────────────────┘
                                │
                                ▼
           ┌────────────────────────────────────────┐
           │  🔵 REFACTOR: Improve the Code         │
           │  ─────────────────────────────────────  │
           │  • Clean up implementation             │
           │  • Add TypeDoc comments                │
           │  • Optimize if needed                  │
           │  • Run: npm test                       │
           │  • Verify tests STILL PASS             │
           └────────────────────┬───────────────────┘
                                │
                                ▼
                    ┌───────────────────┐
                    │  More features?   │
                    └─────────┬─────────┘
                              │
                   ┌──────────┴──────────┐
                   │ YES                 │ NO
                   ▼                     ▼
              [Go to RED]          ┌──────────┐
                                   │   DONE   │
                                   └──────────┘
```

### Example TDD Workflow for Vite with Vitest

**Scenario: Building a URL utility for a Vite application**

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: 🔴 RED - Write the Failing Test                             │
├─────────────────────────────────────────────────────────────────────┤
│ File: src/utils/url.test.ts                                         │
└─────────────────────────────────────────────────────────────────────┘
```

```typescript
// src/utils/url.test.ts
import { describe, it, expect } from 'vitest';
import { buildUrl, parseQueryString, joinPaths } from './url';

describe('URL Utilities', () => {
  describe('buildUrl', () => {
    it('builds URL with base and path', () => {
      expect(buildUrl('https://api.example.com', '/users')).toBe(
        'https://api.example.com/users'
      );
    });

    it('builds URL with query parameters', () => {
      expect(
        buildUrl('https://api.example.com', '/search', { q: 'vite', page: '1' })
      ).toBe('https://api.example.com/search?q=vite&page=1');
    });

    it('handles trailing slashes in base URL', () => {
      expect(buildUrl('https://api.example.com/', '/users')).toBe(
        'https://api.example.com/users'
      );
    });
  });

  describe('parseQueryString', () => {
    it('parses query string into object', () => {
      expect(parseQueryString('?name=john&age=30')).toEqual({
        name: 'john',
        age: '30'
      });
    });

    it('handles empty query string', () => {
      expect(parseQueryString('')).toEqual({});
    });
  });

  describe('joinPaths', () => {
    it('joins multiple path segments', () => {
      expect(joinPaths('api', 'v1', 'users')).toBe('api/v1/users');
    });

    it('handles leading/trailing slashes', () => {
      expect(joinPaths('/api/', '/v1/', '/users/')).toBe('/api/v1/users/');
    });
  });
});
```

```bash
# Run the test
npm test -- src/utils/url.test.ts

# Output: ❌ FAILS
# Error: Cannot find module './url'
# This is EXPECTED - we haven't written the implementation yet!
```

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: 🟢 GREEN - Write Minimal Implementation                     │
├─────────────────────────────────────────────────────────────────────┤
│ File: src/utils/url.ts                                              │
└─────────────────────────────────────────────────────────────────────┘
```

```typescript
// src/utils/url.ts - Minimal implementation to pass tests

export function buildUrl(
  base: string,
  path: string,
  params?: Record<string, string>
): string {
  // Remove trailing slash from base, ensure path starts with /
  const normalizedBase = base.replace(/\/$/, '');
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;

  let url = `${normalizedBase}${normalizedPath}`;

  if (params && Object.keys(params).length > 0) {
    const queryString = Object.entries(params)
      .map(([key, value]) => `${key}=${value}`)
      .join('&');
    url += `?${queryString}`;
  }

  return url;
}

export function parseQueryString(query: string): Record<string, string> {
  if (!query || query === '?') return {};

  const cleanQuery = query.startsWith('?') ? query.slice(1) : query;

  return cleanQuery.split('&').reduce((acc, pair) => {
    const [key, value] = pair.split('=');
    if (key) acc[key] = value || '';
    return acc;
  }, {} as Record<string, string>);
}

export function joinPaths(...paths: string[]): string {
  return paths
    .map((path, index) => {
      if (index === 0) return path.replace(/\/$/, '');
      if (index === paths.length - 1) return path.replace(/^\//, '');
      return path.replace(/^\/|\/$/g, '');
    })
    .filter(Boolean)
    .join('/');
}
```

```bash
# Run the tests again
npm test -- src/utils/url.test.ts

# Output: ✅ PASSES
# All 7 tests pass!
```

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: 🔵 REFACTOR - Improve Code Quality                          │
├─────────────────────────────────────────────────────────────────────┤
│ File: src/utils/url.ts (refactored)                                 │
└─────────────────────────────────────────────────────────────────────┘
```

```typescript
// src/utils/url.ts - Refactored with TypeDoc and improvements

/**
 * Query parameters as key-value pairs.
 */
export type QueryParams = Record<string, string>;

/**
 * Builds a complete URL from base, path, and optional query parameters.
 *
 * Handles normalization of slashes and proper URL encoding.
 *
 * @param base - Base URL (e.g., 'https://api.example.com')
 * @param path - Path to append (e.g., '/users')
 * @param params - Optional query parameters
 * @returns Complete URL string
 *
 * @example
 * ```typescript
 * buildUrl('https://api.example.com', '/users');
 * // Returns: 'https://api.example.com/users'
 *
 * buildUrl('https://api.example.com', '/search', { q: 'vite' });
 * // Returns: 'https://api.example.com/search?q=vite'
 * ```
 */
export function buildUrl(
  base: string,
  path: string,
  params?: QueryParams
): string {
  const normalizedBase = base.replace(/\/+$/, '');
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;

  const url = new URL(`${normalizedBase}${normalizedPath}`);

  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.append(key, value);
    });
  }

  return url.toString();
}

/**
 * Parses a query string into a key-value object.
 *
 * @param query - Query string (with or without leading '?')
 * @returns Object containing parsed parameters
 *
 * @example
 * ```typescript
 * parseQueryString('?name=john&age=30');
 * // Returns: { name: 'john', age: '30' }
 * ```
 */
export function parseQueryString(query: string): QueryParams {
  if (!query) return {};

  const params = new URLSearchParams(query);
  const result: QueryParams = {};

  params.forEach((value, key) => {
    result[key] = value;
  });

  return result;
}

/**
 * Joins multiple path segments into a single path.
 *
 * Preserves leading slash of first segment and trailing slash of last segment.
 *
 * @param paths - Path segments to join
 * @returns Joined path string
 *
 * @example
 * ```typescript
 * joinPaths('api', 'v1', 'users');
 * // Returns: 'api/v1/users'
 *
 * joinPaths('/api/', '/v1/', '/users/');
 * // Returns: '/api/v1/users/'
 * ```
 */
export function joinPaths(...paths: string[]): string {
  if (paths.length === 0) return '';

  const hasLeadingSlash = paths[0].startsWith('/');
  const hasTrailingSlash = paths[paths.length - 1].endsWith('/');

  const joined = paths
    .map(p => p.replace(/^\/+|\/+$/g, ''))
    .filter(Boolean)
    .join('/');

  return `${hasLeadingSlash ? '/' : ''}${joined}${hasTrailingSlash ? '/' : ''}`;
}
```

```bash
# Run tests to ensure refactoring didn't break anything
npm test -- src/utils/url.test.ts

# Output: ✅ PASSES
# All 7 tests still pass!

# Run type check
npm run type-check
# ✅ No errors

# Check coverage
npm run test:coverage -- src/utils/url.test.ts
# ✅ 100% coverage
```

### Visual Step-by-Step TDD Example

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TDD TIMELINE FOR VITE PROJECT                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TIME ──────────────────────────────────────────────────────────►   │
│                                                                     │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐          │
│  │  TEST   │    │  CODE   │    │ REFACTOR│    │  DONE   │          │
│  │  FILE   │    │  FILE   │    │  CODE   │    │   ✓     │          │
│  └────┬────┘    └────┬────┘    └────┬────┘    └─────────┘          │
│       │              │              │                               │
│       ▼              ▼              ▼                               │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                         │
│  │ npm test│    │ npm test│    │ npm test│                         │
│  │   ❌    │    │   ✅    │    │   ✅    │                         │
│  └─────────┘    └─────────┘    └─────────┘                         │
│                                                                     │
│  ◄── RED ────►  ◄── GREEN ─►  ◄─ REFACTOR ─►                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

VITEST WATCH MODE WORKFLOW:

  Terminal 1 (Vitest Watch):          Terminal 2 (Editor):
  ┌──────────────────────────┐        ┌──────────────────────────┐
  │ $ npm test -- --watch    │        │ 1. Write test            │
  │                          │        │    ↓                     │
  │ RERUN  url.test.ts       │◄───────│ 2. Save file             │
  │                          │        │                          │
  │ ❌ 3 tests failed        │        │ 3. See failure           │
  │                          │        │    ↓                     │
  │ RERUN  url.test.ts       │◄───────│ 4. Write implementation  │
  │                          │        │    ↓                     │
  │ ✅ 3 tests passed        │        │ 5. Save file             │
  │                          │        │                          │
  │ Watching for changes...  │        │ 6. See success!          │
  └──────────────────────────┘        └──────────────────────────┘
```

### TDD Commands Reference for Vite/Vitest

```bash
# Start TDD workflow with watch mode
npm test -- --watch

# Run specific test file
npm test -- src/utils/url.test.ts

# Run tests matching pattern
npm test -- --grep "buildUrl"

# Run with coverage (check after GREEN phase)
npm test -- --coverage

# Run with UI (visual test runner)
npm test -- --ui

# Type check in parallel (separate terminal)
npx tsc --noEmit --watch
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BUG FIX WORKFLOW FOR VITE                        │
└─────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │  🐛 BUG REPORTED │
    │   (Issue #1234)  │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  STEP 1: Reproduce the Bug                                   │
    │  ────────────────────────────────────────────────────────── │
    │  • Understand the bug report                                 │
    │  • Identify the failing scenario                             │
    │  • Note the expected vs actual behavior                      │
    └────────┬─────────────────────────────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  STEP 2: Write a Failing Test (BEFORE fixing!)               │
    │  ────────────────────────────────────────────────────────── │
    │  • Create test that reproduces the bug                       │
    │  • Include bug ID in test name/comment                       │
    │  • Run test to confirm it FAILS                              │
    │  ┌────────────────────────────────────────────────────────┐  │
    │  │ it('handles empty array input - Bug #1234', () => {   │  │
    │  │   // This test reproduces Bug #1234                    │  │
    │  │   expect(processItems([])).toEqual([]);                │  │
    │  │ });                                                     │  │
    │  └────────────────────────────────────────────────────────┘  │
    └────────┬─────────────────────────────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  STEP 3: Verify Test Fails for the Right Reason              │
    │  ────────────────────────────────────────────────────────── │
    │  • Run: npm test                                             │
    │  • Confirm test fails with expected error                    │
    │  • If test passes → bug is already fixed or test is wrong    │
    └────────┬─────────────────────────────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  STEP 4: Fix the Bug                                         │
    │  ────────────────────────────────────────────────────────── │
    │  • Make the minimal change to fix the bug                    │
    │  • Add defensive checks if needed                            │
    │  • Add TypeDoc comments explaining the fix                   │
    └────────┬─────────────────────────────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  STEP 5: Verify All Tests Pass                               │
    │  ────────────────────────────────────────────────────────── │
    │  • Run: npm test                                             │
    │  • New regression test PASSES                                │
    │  • All existing tests STILL PASS                             │
    │  • Run: npm run type-check                                   │
    └────────┬─────────────────────────────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  STEP 6: Document and Commit                                 │
    │  ────────────────────────────────────────────────────────── │
    │  • Commit message: "fix: handle empty array (Bug #1234)"     │
    │  • Reference bug ID in commit                                │
    │  • Bug can never regress (test protects it forever!)         │
    └────────┬─────────────────────────────────────────────────────┘
             │
             ▼
    ┌──────────────────┐
    │  ✅ BUG FIXED    │
    │  🛡️ PROTECTED    │
    └──────────────────┘
```

### Example Bug Fix with Regression Test

**Bug Report #4521**: `formatPrice` function crashes when price is undefined

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: Understand the Bug                                          │
├─────────────────────────────────────────────────────────────────────┤
│ Reporter: "When fetching product data, sometimes price is           │
│ undefined and the formatPrice function throws an error"             │
│                                                                     │
│ Expected: Should return '$0.00' or handle gracefully                │
│ Actual: TypeError: Cannot read properties of undefined              │
└─────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: Write Failing Regression Test                               │
├─────────────────────────────────────────────────────────────────────┤
│ File: src/utils/pricing.test.ts                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```typescript
// src/utils/pricing.test.ts
import { describe, it, expect } from 'vitest';
import { formatPrice } from './pricing';

describe('formatPrice', () => {
  // Existing tests...
  it('formats positive prices', () => {
    expect(formatPrice(19.99)).toBe('$19.99');
  });

  it('formats zero', () => {
    expect(formatPrice(0)).toBe('$0.00');
  });

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // REGRESSION TEST - Bug #4521
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  describe('Bug #4521 - undefined price handling', () => {
    it('handles undefined price - Bug #4521', () => {
      // Bug: formatPrice(undefined) threw TypeError
      // Expected: Should return '$0.00' for undefined input
      // Discovered: 2026-01-22
      expect(formatPrice(undefined as unknown as number)).toBe('$0.00');
    });

    it('handles null price - Bug #4521', () => {
      // Related edge case discovered during investigation
      expect(formatPrice(null as unknown as number)).toBe('$0.00');
    });

    it('handles NaN price - Bug #4521', () => {
      // Related edge case discovered during investigation
      expect(formatPrice(NaN)).toBe('$0.00');
    });
  });
});
```

```bash
# Run the test
npm test -- src/utils/pricing.test.ts

# Output:
# ❌ FAIL  src/utils/pricing.test.ts
#   formatPrice
#     Bug #4521 - undefined price handling
#       ✕ handles undefined price - Bug #4521
#         TypeError: Cannot read properties of undefined
#
# This is EXPECTED - we've reproduced the bug!
```

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: Verify Test Fails for Right Reason                          │
├─────────────────────────────────────────────────────────────────────┤
│ ✅ Test fails with: TypeError: Cannot read properties of undefined  │
│ ✅ This matches the reported bug behavior                           │
│ ✅ Proceed to fix                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: Fix the Bug                                                 │
├─────────────────────────────────────────────────────────────────────┤
│ File: src/utils/pricing.ts                                          │
└─────────────────────────────────────────────────────────────────────┘
```

```typescript
// src/utils/pricing.ts

/**
 * Formats a price as a USD currency string.
 *
 * @param price - The price to format
 * @returns Formatted price string (e.g., '$19.99')
 *
 * @remarks
 * FIX for Bug #4521: Now handles undefined, null, and NaN values
 * by returning '$0.00' instead of throwing an error.
 *
 * @example
 * ```typescript
 * formatPrice(19.99);     // '$19.99'
 * formatPrice(0);         // '$0.00'
 * formatPrice(undefined); // '$0.00' (Bug #4521 fix)
 * ```
 */
export function formatPrice(price: number): string {
  // FIX: Bug #4521 - Handle undefined, null, and NaN gracefully
  if (price === undefined || price === null || Number.isNaN(price)) {
    return '$0.00';
  }

  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(price);
}
```

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: Verify All Tests Pass                                       │
└─────────────────────────────────────────────────────────────────────┘
```

```bash
# Run all tests
npm test -- src/utils/pricing.test.ts

# Output:
# ✅ PASS  src/utils/pricing.test.ts
#   formatPrice
#     ✓ formats positive prices
#     ✓ formats zero
#     Bug #4521 - undefined price handling
#       ✓ handles undefined price - Bug #4521
#       ✓ handles null price - Bug #4521
#       ✓ handles NaN price - Bug #4521
#
# Test Files: 1 passed
# Tests:      5 passed

# Type check
npm run type-check
# ✅ No errors

# Run full test suite to ensure no regressions
npm test
# ✅ All tests pass
```

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 6: Commit with Bug Reference                                   │
└─────────────────────────────────────────────────────────────────────┘
```

```bash
git add src/utils/pricing.ts src/utils/pricing.test.ts
git commit -m "fix(pricing): handle undefined/null/NaN in formatPrice

- Add defensive checks for undefined, null, and NaN values
- Return '\$0.00' for invalid inputs instead of throwing
- Add regression tests to prevent future recurrence

Fixes #4521"
```

### Bug Fix Decision Tree

```
                        ┌─────────────────┐
                        │  Bug Reported   │
                        └────────┬────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Can you reproduce it? │
                    └────────────┬───────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │ YES              │ NO               │
              ▼                  ▼                  │
    ┌─────────────────┐  ┌─────────────────────┐   │
    │  Write failing  │  │  Request more info  │   │
    │  test FIRST     │  │  from reporter      │   │
    └────────┬────────┘  └─────────────────────┘   │
             │                                      │
             ▼                                      │
    ┌─────────────────┐                            │
    │  Test fails?    │                            │
    └────────┬────────┘                            │
             │                                      │
    ┌────────┼────────┐                            │
    │ YES    │ NO     │                            │
    ▼        ▼        │                            │
  ┌────┐  ┌────────────────────┐                   │
  │Fix │  │ Bug already fixed  │                   │
  │bug │  │ or test is wrong   │                   │
  └─┬──┘  └────────────────────┘                   │
    │                                              │
    ▼                                              │
  ┌──────────────────┐                             │
  │ All tests pass?  │                             │
  └────────┬─────────┘                             │
           │                                       │
  ┌────────┼────────┐                              │
  │ YES    │ NO     │                              │
  ▼        ▼        │                              │
┌────┐  ┌────────────────┐                         │
│Done│  │ Fix regressions│                         │
└────┘  │ you introduced │                         │
        └────────────────┘                         │
```

### Prohibited Bug Fix Practices

**NEVER:**
```
❌ Fix a bug without adding a regression test first
   → Future changes may reintroduce the bug

❌ Delete or skip tests to make the bug "go away"
   → The bug is not fixed, just hidden

❌ Merge bug fixes without all tests passing
   → You may be introducing new bugs

❌ Forget to reference the bug ID in tests and commits
   → Traceability is lost

❌ Write the fix first, then add tests
   → Violates TDD, tests might not actually test the bug
```

---

## 3. Project Structure

```
my-vite-app/
├── src/
│   ├── main.ts                 # Entry point
│   ├── vite-env.d.ts          # Vite type definitions
│   │
│   ├── core/                   # Core utilities
│   │   ├── constants.ts
│   │   ├── types.ts
│   │   └── config.ts
│   │
│   ├── features/               # Feature modules
│   │   ├── users/
│   │   │   ├── api.ts
│   │   │   ├── api.test.ts
│   │   │   ├── store.ts
│   │   │   ├── store.test.ts
│   │   │   ├── types.ts
│   │   │   └── utils.ts
│   │   └── auth/
│   │       ├── api.ts
│   │       └── store.ts
│   │
│   ├── utils/                  # Shared utilities
│   │   ├── validation.ts
│   │   ├── validation.test.ts
│   │   ├── formatting.ts
│   │   └── formatting.test.ts
│   │
│   ├── stores/                 # Global stores (signals)
│   │   ├── app-state.ts
│   │   └── user-store.ts
│   │
│   ├── test/                   # Test utilities
│   │   └── setup.ts
│   │
│   └── styles/                 # Global styles
│       └── main.css
│
├── public/                     # Static assets
├── dist/                       # Build output (gitignored)
├── docs/                       # Generated docs (gitignored)
│
├── vite.config.ts             # Vite configuration
├── tsconfig.json              # TypeScript config
├── tsconfig.node.json         # Node TypeScript config
├── package.json
└── README.md
```

---

## 4. Reactive State Management with Signals

### A. Signal Basics

```typescript
// src/stores/counter-store.ts
import { signal, computed, effect } from '@preact/signals-core';

/**
 * Counter state using signals.
 * 
 * Provides reactive counter state with computed values.
 * 
 * @example
 * ```typescript
 * import { count, double, increment } from './counter-store';
 * 
 * console.log(count.value); // 0
 * increment();
 * console.log(count.value); // 1
 * console.log(double.value); // 2
 * ```
 */

// State signal
export const count = signal(0);

// Computed signal
export const double = computed(() => count.value * 2);

// Computed with multiple dependencies
export const isEven = computed(() => count.value % 2 === 0);

// Actions
export function increment(): void {
  count.value++;
}

export function decrement(): void {
  count.value--;
}

export function reset(): void {
  count.value = 0;
}

// Effect (side effect)
effect(() => {
  console.log(`Count changed to: ${count.value}`);
  // Save to localStorage
  localStorage.setItem('count', count.value.toString());
});
```

### B. Complex Store Example

```typescript
// src/stores/user-store.ts
import { signal, computed, batch } from '@preact/signals-core';

/**
 * User data interface.
 */
export interface User {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'user';
}

/**
 * User store state using signals.
 * 
 * Manages user authentication and profile data.
 * 
 * @example
 * ```typescript
 * import { user, isAuthenticated, login, logout } from './user-store';
 * 
 * await login('user@example.com', 'password');
 * console.log(user.value?.name);
 * console.log(isAuthenticated.value); // true
 * 
 * logout();
 * console.log(isAuthenticated.value); // false
 * ```
 */

// State signals
const user = signal<User | null>(null);
const isLoading = signal(false);
const error = signal<string | null>(null);

// Computed signals
export const isAuthenticated = computed(() => user.value !== null);
export const isAdmin = computed(() => user.value?.role === 'admin');

// Getters (readonly access)
export const getUser = () => user.value;
export const getIsLoading = () => isLoading.value;
export const getError = () => error.value;

/**
 * Logs in a user.
 * 
 * @param email - User email
 * @param password - User password
 * @throws {Error} If login fails
 */
export async function login(email: string, password: string): Promise<void> {
  batch(() => {
    isLoading.value = true;
    error.value = null;
  });
  
  try {
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    
    if (!response.ok) {
      throw new Error('Login failed');
    }
    
    const userData = await response.json();
    
    batch(() => {
      user.value = userData;
      isLoading.value = false;
    });
  } catch (err) {
    batch(() => {
      error.value = err instanceof Error ? err.message : 'Unknown error';
      isLoading.value = false;
    });
    throw err;
  }
}

/**
 * Logs out the current user.
 */
export function logout(): void {
  batch(() => {
    user.value = null;
    error.value = null;
  });
}
```

---

## 5. Async/Await Patterns (MANDATORY)

### A. Async Hierarchy

**Preference order (highest to lowest):**

1. **`async`/`await`** (PREFERRED)
2. **`Promise.all()`** / **`Promise.allSettled()`** for parallel operations
3. **`.then()` chains** (LEGACY - avoid in new code)
4. **Callbacks** (NEVER use in new code)

### B. Async Best Practices

```typescript
// ✅ CORRECT - async/await with proper error handling
/**
 * Fetches user data from API.
 * 
 * @param userId - User ID to fetch
 * @returns Promise resolving to user data
 * @throws {Error} If user not found or network error
 */
export async function fetchUser(userId: string): Promise<User> {
  try {
    const response = await fetch(`/api/users/${userId}`);
    
    if (!response.ok) {
      if (response.status === 404) {
        throw new Error('User not found');
      }
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Failed to fetch user:', error);
    throw error;
  }
}

// ✅ CORRECT - Parallel async operations
/**
 * Loads all dashboard data in parallel.
 * 
 * @returns Promise resolving to dashboard data
 */
export async function loadDashboardData(): Promise<DashboardData> {
  const [users, orders, stats] = await Promise.all([
    fetchUsers(),
    fetchOrders(),
    fetchStats(),
  ]);
  
  return { users, orders, stats };
}

// ✅ CORRECT - Handling partial failures
/**
 * Loads data with fallbacks for failures.
 * 
 * @returns Promise resolving to available data
 */
export async function loadDataWithFallbacks(): Promise<Partial<Data>> {
  const results = await Promise.allSettled([
    fetchUsers(),
    fetchOrders(),
    fetchStats(),
  ]);
  
  return {
    users: results[0].status === 'fulfilled' ? results[0].value : [],
    orders: results[1].status === 'fulfilled' ? results[1].value : [],
    stats: results[2].status === 'fulfilled' ? results[2].value : null,
  };
}

// ❌ WRONG - Using .then() chains
export function fetchUser(userId: string): Promise<User> {
  return fetch(`/api/users/${userId}`)
    .then(response => {
      if (!response.ok) throw new Error('Failed');
      return response.json();
    })
    .catch(error => {
      console.error(error);
      throw error;
    });
}

// ❌ WRONG - Nested callbacks
export function fetchUser(userId: string, callback: (user: User) => void): void {
  fetch(`/api/users/${userId}`).then(response => {
    response.json().then(user => {
      callback(user);
    });
  });
}
```

### C. Async with Signals

```typescript
// src/features/users/user-loader.ts
import { signal, computed } from '@preact/signals-core';

/**
 * User data loader with reactive state.
 * 
 * @example
 * ```typescript
 * const loader = createUserLoader();
 * await loader.load('user-123');
 * console.log(loader.data.value);
 * ```
 */
export function createUserLoader() {
  const data = signal<User | null>(null);
  const isLoading = signal(false);
  const error = signal<string | null>(null);
  
  const hasData = computed(() => data.value !== null);
  const hasError = computed(() => error.value !== null);
  
  async function load(userId: string): Promise<void> {
    isLoading.value = true;
    error.value = null;
    
    try {
      const user = await fetchUser(userId);
      data.value = user;
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Unknown error';
    } finally {
      isLoading.value = false;
    }
  }
  
  function reset(): void {
    data.value = null;
    error.value = null;
    isLoading.value = false;
  }
  
  return {
    data,
    isLoading,
    error,
    hasData,
    hasError,
    load,
    reset,
  };
}
```

---

## 6. Documentation Requirements (MANDATORY)

### A. TypeDoc Comments

**ALL exported functions, classes, interfaces, and types MUST have complete TypeDoc documentation.**

```typescript
/**
 * User authentication service.
 * 
 * Provides methods for user login, logout, and session management.
 * Uses JWT tokens for authentication.
 * 
 * @example
 * ```typescript
 * const authService = new AuthService();
 * 
 * // Login user
 * await authService.login('user@example.com', 'password');
 * 
 * // Check authentication
 * if (authService.isAuthenticated()) {
 *   console.log('User is logged in');
 * }
 * 
 * // Logout
 * authService.logout();
 * ```
 */
export class AuthService {
  private token: string | null = null;
  
  /**
   * Logs in a user with email and password.
   * 
   * @param email - User email address
   * @param password - User password
   * @returns Promise resolving to authentication token
   * @throws {AuthError} If credentials are invalid
   * @throws {NetworkError} If network request fails
   * 
   * @example
   * ```typescript
   * try {
   *   const token = await authService.login('user@example.com', 'pass123');
   *   console.log('Login successful:', token);
   * } catch (error) {
   *   console.error('Login failed:', error);
   * }
   * ```
   */
  async login(email: string, password: string): Promise<string> {
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    
    if (!response.ok) {
      throw new Error('Authentication failed');
    }
    
    const { token } = await response.json();
    this.token = token;
    
    return token;
  }
  
  /**
   * Checks if user is currently authenticated.
   * 
   * @returns True if user has valid token, false otherwise
   * 
   * @example
   * ```typescript
   * if (authService.isAuthenticated()) {
   *   // Show authenticated content
   * }
   * ```
   */
  isAuthenticated(): boolean {
    return this.token !== null;
  }
  
  /**
   * Logs out the current user.
   * 
   * Clears the authentication token and invalidates the session.
   * 
   * @example
   * ```typescript
   * authService.logout();
   * console.log('User logged out');
   * ```
   */
  logout(): void {
    this.token = null;
  }
}
```

### B. Interface Documentation

```typescript
/**
 * User data interface.
 * 
 * Represents a user in the system with authentication and profile information.
 * 
 * @interface User
 * @property {string} id - Unique user identifier (UUID v4)
 * @property {string} name - User's full name (1-100 characters)
 * @property {string} email - User's email address (unique, validated)
 * @property {UserRole} role - User's role for authorization
 * @property {Date} createdAt - Account creation timestamp
 * @property {Date} [lastLoginAt] - Last login timestamp (optional)
 * 
 * @example
 * ```typescript
 * const user: User = {
 *   id: '550e8400-e29b-41d4-a716-446655440000',
 *   name: 'John Doe',
 *   email: 'john@example.com',
 *   role: 'user',
 *   createdAt: new Date('2024-01-01'),
 *   lastLoginAt: new Date()
 * };
 * ```
 */
export interface User {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  createdAt: Date;
  lastLoginAt?: Date;
}

/**
 * User role types for authorization.
 * 
 * - `admin`: Full system access, can manage all users
 * - `user`: Standard user with limited permissions
 * - `guest`: Read-only access
 * 
 * @typedef {('admin' | 'user' | 'guest')} UserRole
 */
export type UserRole = 'admin' | 'user' | 'guest';
```

### C. TypeDoc Configuration

```json
// typedoc.json
{
  "entryPoints": ["src"],
  "entryPointStrategy": "expand",
  "out": "docs",
  "exclude": [
    "**/*.test.ts",
    "**/*.spec.ts",
    "**/test/**",
    "**/node_modules/**"
  ],
  "excludePrivate": true,
  "excludeProtected": false,
  "excludeInternal": false,
  "readme": "README.md",
  "theme": "default",
  "categorizeByGroup": true,
  "categoryOrder": [
    "Core",
    "Features",
    "Utilities",
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
  }
}
```

---

## 7. Testing with Vitest (MANDATORY)

### A. Test Setup

```typescript
// src/test/setup.ts
import { expect, afterEach } from 'vitest';
import { cleanup } from '@testing-library/dom';

// Cleanup after each test
afterEach(() => {
  cleanup();
});

// Extend matchers if needed
expect.extend({
  // Custom matchers here
});
```

### B. Unit Test Examples

```typescript
// src/utils/formatting.test.ts
import { describe, it, expect } from 'vitest';
import { formatCurrency, formatDate, formatNumber } from './formatting';

describe('formatCurrency', () => {
  it('formats USD currency', () => {
    expect(formatCurrency(1234.56, 'USD')).toBe('$1,234.56');
  });
  
  it('formats EUR currency', () => {
    expect(formatCurrency(1234.56, 'EUR')).toBe('€1,234.56');
  });
  
  it('handles zero', () => {
    expect(formatCurrency(0, 'USD')).toBe('$0.00');
  });
  
  it('handles negative numbers', () => {
    expect(formatCurrency(-100, 'USD')).toBe('-$100.00');
  });
});

describe('formatDate', () => {
  it('formats date in default format', () => {
    const date = new Date('2024-01-15T10:30:00Z');
    expect(formatDate(date)).toBe('Jan 15, 2024');
  });
  
  it('formats date with custom format', () => {
    const date = new Date('2024-01-15T10:30:00Z');
    expect(formatDate(date, 'yyyy-MM-dd')).toBe('2024-01-15');
  });
});
```

### C. Async Test Examples

```typescript
// src/features/users/api.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fetchUser, fetchUsers, createUser } from './api';

describe('User API', () => {
  beforeEach(() => {
    // Reset mocks before each test
    vi.clearAllMocks();
  });
  
  describe('fetchUser', () => {
    it('fetches user successfully', async () => {
      const mockUser = { id: '1', name: 'John', email: 'john@example.com' };
      
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockUser,
      } as Response);
      
      const user = await fetchUser('1');
      
      expect(user).toEqual(mockUser);
      expect(global.fetch).toHaveBeenCalledWith('/api/users/1');
    });
    
    it('throws error on 404', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        statusText: 'Not Found',
      } as Response);
      
      await expect(fetchUser('invalid')).rejects.toThrow('User not found');
    });
    
    it('handles network errors', async () => {
      global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));
      
      await expect(fetchUser('1')).rejects.toThrow('Network error');
    });
  });
  
  describe('createUser', () => {
    it('creates user successfully', async () => {
      const userData = { name: 'Jane', email: 'jane@example.com' };
      const createdUser = { id: '2', ...userData };
      
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 201,
        json: async () => createdUser,
      } as Response);
      
      const user = await createUser(userData);
      
      expect(user).toEqual(createdUser);
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/users',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify(userData),
        })
      );
    });
  });
});
```

### D. Signal Store Tests

```typescript
// src/stores/counter-store.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { count, double, increment, decrement, reset } from './counter-store';

describe('Counter Store', () => {
  beforeEach(() => {
    reset();
  });
  
  it('initializes with zero', () => {
    expect(count.value).toBe(0);
  });
  
  it('increments counter', () => {
    increment();
    expect(count.value).toBe(1);
    
    increment();
    expect(count.value).toBe(2);
  });
  
  it('decrements counter', () => {
    increment();
    increment();
    decrement();
    
    expect(count.value).toBe(1);
  });
  
  it('computes doubled value', () => {
    increment();
    expect(double.value).toBe(2);
    
    increment();
    expect(double.value).toBe(4);
  });
  
  it('resets to zero', () => {
    increment();
    increment();
    increment();
    
    reset();
    
    expect(count.value).toBe(0);
    expect(double.value).toBe(0);
  });
});
```

---

## 8. Vite-Specific Optimizations

### A. Environment Variables

```typescript
// vite-env.d.ts
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_API_KEY: string;
  readonly VITE_ENABLE_ANALYTICS: string;
  // Add more environment variables here
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
```

```typescript
// Usage
const apiUrl = import.meta.env.VITE_API_URL;
const isDev = import.meta.env.DEV;
const isProd = import.meta.env.PROD;
```

### B. Code Splitting

```typescript
// Lazy load modules
const UserModule = () => import('./features/users');
const AdminModule = () => import('./features/admin');

// Conditional loading
async function loadModule(moduleName: string) {
  switch (moduleName) {
    case 'users':
      return await import('./features/users');
    case 'admin':
      return await import('./features/admin');
    default:
      throw new Error(`Unknown module: ${moduleName}`);
  }
}
```

### C. Asset Handling

```typescript
// Import assets
import logo from './assets/logo.svg';
import styles from './styles/main.module.css';

// Use in code
const img = new Image();
img.src = logo;

// URL imports for workers
import Worker from './worker?worker';
const worker = new Worker();
```

---

## 9. Deployment Checklist

### Test-Driven Development (TDD) Compliance
- [ ] **Tests written BEFORE implementation**: Red-Green-Refactor cycle followed for all new code
- [ ] **Each test failed first**: Verified tests fail before implementation, pass after
- [ ] **TDD cycle documented**: Commit messages or comments show test-first approach
- [ ] **Bug regression tests added**: Every bug fix has a test that reproduces the bug
- [ ] **Regression tests fail without fix**: Verified bug tests fail before fix, pass after
- [ ] **Bug IDs referenced**: Bug numbers documented in test comments

### Code Quality
- [ ] **TypeScript compilation passes**: `npm run type-check` succeeds
- [ ] **Build succeeds**: `npm run build` completes without errors
- [ ] **All tests pass**: `npm test` returns exit code 0
- [ ] **Test coverage > 80%**: `npm run test:coverage` shows adequate coverage
- [ ] **Linting passes**: `npm run lint` succeeds
- [ ] **No console errors**: Application runs without errors

### Documentation
- [ ] **All public APIs documented**: TypeDoc comments on all exported functions/classes
- [ ] **Documentation builds**: `npm run docs` succeeds
- [ ] **README.md updated**: Project documentation is current
- [ ] **API docs generated**: `docs/` directory exists with TypeDoc output

### Vite Optimizations
- [ ] **Build size acceptable**: Check `dist/` size after build
- [ ] **Code splitting configured**: Large dependencies in separate chunks
- [ ] **Assets optimized**: Images compressed, fonts subset
- [ ] **Environment variables set**: Production `.env` configured
- [ ] **Source maps generated**: For debugging production issues

### Performance
- [ ] **Bundle analyzed**: Run `npm run build -- --mode analyze` if available
- [ ] **Dependencies pre-bundled**: Critical deps in `optimizeDeps.include`
- [ ] **Lazy loading used**: Non-critical code dynamically imported
- [ ] **Tree-shaking effective**: No unused exports in bundle

### Security
- [ ] **No secrets in code**: Environment variables for sensitive data
- [ ] **Dependencies updated**: Run `npm audit` and fix issues
- [ ] **CSP headers configured**: Content Security Policy in production
- [ ] **HTTPS enabled**: Production uses HTTPS

---

## 10. Why This Configuration Works

1. **Test-Driven Development (TDD)**: Writing tests before code provides:
   - **Better design**: Tests force API thinking upfront
   - **Fewer bugs**: Catches issues before production (40-80% reduction)
   - **Living documentation**: Tests document expected behavior
   - **Fearless refactoring**: Comprehensive tests enable safe improvements
   - **Faster debugging**: Failing tests pinpoint exact issues
   - **Regression prevention**: Bug tests ensure fixed bugs stay fixed

2. **Vite's Speed**: Instant server start, lightning-fast HMR (Hot Module Replacement), on-demand compilation make development 10-100x faster than traditional bundlers.

3. **Native ESM**: Leverages browser's native module system for development, no bundling needed until production.

4. **Optimized Production Builds**: Uses Rollup for production, providing excellent tree-shaking and code splitting.

5. **Signals for Reactivity**: Fine-grained reactivity with minimal overhead, explicit dependencies, framework-agnostic.

6. **Async/Await**: Clean, readable async code with proper error handling and stack traces.

7. **TypeScript Strict Mode**: Catches 15-30% more bugs at compile time, excellent IDE support.

8. **Vitest Integration**: Same config as Vite, instant test execution, native ESM support.

9. **TypeDoc**: Generated documentation stays in sync with code, reducing documentation drift.

10. **Agent Verification**: Mandatory compilation and test checks prevent broken code from reaching users.

11. **Dependency Pre-bundling**: Vite automatically pre-bundles dependencies for optimal performance.

12. **Regression Tests for Bugs**: Every bug gets a test, creating a safety net that prevents regression.

---

## Quick Reference

### Common Commands

```bash
# ─────────────────────────────────────────────────────────────────────
# DEVELOPMENT
# ─────────────────────────────────────────────────────────────────────

# Start development server with HMR
npm run dev
# OR: npx vite

# Start dev server on specific port
npx vite --port 3000

# Start dev server and open browser
npx vite --open

# Start dev server with HTTPS
npx vite --https

# ─────────────────────────────────────────────────────────────────────
# BUILDING
# ─────────────────────────────────────────────────────────────────────

# Production build
npm run build
# OR: npx vite build

# Build with specific mode
npx vite build --mode staging

# Build and analyze bundle
npx vite build --mode analyze

# Preview production build locally
npm run preview
# OR: npx vite preview

# Preview on specific port
npx vite preview --port 4173

# ─────────────────────────────────────────────────────────────────────
# TESTING (Vitest)
# ─────────────────────────────────────────────────────────────────────

# Run all tests once
npm test
# OR: npx vitest run

# Run tests in watch mode (TDD workflow)
npm test -- --watch
# OR: npx vitest

# Run specific test file
npx vitest run src/utils/validation.test.ts

# Run tests matching pattern
npx vitest run --grep "formatPrice"

# Run tests with coverage
npm run test:coverage
# OR: npx vitest run --coverage

# Run tests with UI
npm run test:ui
# OR: npx vitest --ui

# Update snapshots
npx vitest run --update

# ─────────────────────────────────────────────────────────────────────
# TYPE CHECKING & LINTING
# ─────────────────────────────────────────────────────────────────────

# Type check without emitting
npm run type-check
# OR: npx tsc --noEmit

# Type check in watch mode
npx tsc --noEmit --watch

# Lint code
npm run lint
# OR: npx eslint . --ext ts,tsx

# Lint and fix
npx eslint . --ext ts,tsx --fix

# ─────────────────────────────────────────────────────────────────────
# DOCUMENTATION
# ─────────────────────────────────────────────────────────────────────

# Generate TypeDoc documentation
npm run docs
# OR: npx typedoc --out docs src

# Serve documentation locally
npx http-server docs -p 8080

# ─────────────────────────────────────────────────────────────────────
# DEPENDENCY MANAGEMENT
# ─────────────────────────────────────────────────────────────────────

# Install dependencies
npm install

# Add production dependency
npm install package-name

# Add dev dependency
npm install -D package-name

# Update dependencies
npm update

# Check for outdated packages
npm outdated

# Security audit
npm audit

# Fix security issues
npm audit fix
```

### Vite Configuration Patterns

```typescript
// vite.config.ts - Common Configuration Patterns

import { defineConfig, loadEnv } from 'vite';
import { resolve } from 'path';

export default defineConfig(({ command, mode }) => {
  // Load environment variables
  const env = loadEnv(mode, process.cwd(), '');

  return {
    // ───────────────────────────────────────────────────────────────
    // PATH ALIASES
    // ───────────────────────────────────────────────────────────────
    resolve: {
      alias: {
        '@': resolve(__dirname, './src'),
        '@components': resolve(__dirname, './src/components'),
        '@utils': resolve(__dirname, './src/utils'),
        '@stores': resolve(__dirname, './src/stores'),
        '@features': resolve(__dirname, './src/features'),
      },
    },

    // ───────────────────────────────────────────────────────────────
    // DEV SERVER
    // ───────────────────────────────────────────────────────────────
    server: {
      port: 5173,
      strictPort: false,
      host: true,
      open: false,
      cors: true,
      // Proxy API requests in development
      proxy: {
        '/api': {
          target: env.VITE_API_URL || 'http://localhost:3000',
          changeOrigin: true,
          secure: false,
        },
      },
    },

    // ───────────────────────────────────────────────────────────────
    // BUILD OPTIONS
    // ───────────────────────────────────────────────────────────────
    build: {
      target: 'esnext',
      outDir: 'dist',
      assetsDir: 'assets',
      sourcemap: mode !== 'production',
      minify: 'esbuild',

      // Rollup options for code splitting
      rollupOptions: {
        output: {
          // Manual chunks for better caching
          manualChunks: {
            // Vendor chunk for framework
            'vendor-framework': ['react', 'react-dom'],
            // Vendor chunk for utilities
            'vendor-utils': ['lodash-es', 'date-fns'],
          },
          // Asset naming
          assetFileNames: 'assets/[name]-[hash][extname]',
          chunkFileNames: 'chunks/[name]-[hash].js',
          entryFileNames: '[name]-[hash].js',
        },
      },

      // Chunk size warning
      chunkSizeWarningLimit: 1000,
    },

    // ───────────────────────────────────────────────────────────────
    // DEPENDENCY OPTIMIZATION
    // ───────────────────────────────────────────────────────────────
    optimizeDeps: {
      // Pre-bundle these dependencies
      include: [
        '@preact/signals-core',
        'lodash-es',
      ],
      // Exclude from pre-bundling
      exclude: ['@vite/client'],
    },

    // ───────────────────────────────────────────────────────────────
    // VITEST CONFIGURATION
    // ───────────────────────────────────────────────────────────────
    test: {
      globals: true,
      environment: 'jsdom',
      setupFiles: './src/test/setup.ts',
      include: ['src/**/*.{test,spec}.{ts,tsx}'],
      exclude: ['node_modules', 'dist'],
      coverage: {
        provider: 'v8',
        reporter: ['text', 'json', 'html'],
        exclude: [
          'node_modules/',
          'src/test/',
          '**/*.test.ts',
          '**/*.spec.ts',
          '**/*.d.ts',
        ],
        thresholds: {
          global: {
            branches: 80,
            functions: 80,
            lines: 80,
            statements: 80,
          },
        },
      },
    },

    // ───────────────────────────────────────────────────────────────
    // ENVIRONMENT VARIABLES
    // ───────────────────────────────────────────────────────────────
    define: {
      __APP_VERSION__: JSON.stringify(process.env.npm_package_version),
      __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
    },

    // ───────────────────────────────────────────────────────────────
    // CONDITIONAL CONFIG BY MODE
    // ───────────────────────────────────────────────────────────────
    ...(mode === 'development' && {
      // Development-only options
    }),
    ...(mode === 'production' && {
      // Production-only options
      build: {
        sourcemap: false,
        minify: 'terser',
      },
    }),
  };
});
```

### Project Structure

```
my-vite-app/
├── src/
│   ├── main.ts                    # Application entry point
│   ├── vite-env.d.ts             # Vite type definitions
│   │
│   ├── components/                # Reusable UI components
│   │   ├── Button/
│   │   │   ├── Button.tsx
│   │   │   ├── Button.test.tsx
│   │   │   ├── Button.module.css
│   │   │   └── index.ts
│   │   └── index.ts               # Barrel export
│   │
│   ├── features/                  # Feature-based modules
│   │   ├── auth/
│   │   │   ├── api.ts            # API calls
│   │   │   ├── api.test.ts
│   │   │   ├── store.ts          # State management
│   │   │   ├── store.test.ts
│   │   │   ├── types.ts          # TypeScript types
│   │   │   └── index.ts
│   │   ├── users/
│   │   │   ├── api.ts
│   │   │   ├── api.test.ts
│   │   │   ├── store.ts
│   │   │   ├── store.test.ts
│   │   │   ├── types.ts
│   │   │   └── index.ts
│   │   └── index.ts
│   │
│   ├── utils/                     # Shared utility functions
│   │   ├── validation.ts
│   │   ├── validation.test.ts
│   │   ├── formatting.ts
│   │   ├── formatting.test.ts
│   │   ├── url.ts
│   │   ├── url.test.ts
│   │   └── index.ts
│   │
│   ├── stores/                    # Global state (signals)
│   │   ├── app-state.ts
│   │   ├── app-state.test.ts
│   │   └── index.ts
│   │
│   ├── hooks/                     # Custom hooks (if using React/Vue)
│   │   ├── useAuth.ts
│   │   ├── useAuth.test.ts
│   │   └── index.ts
│   │
│   ├── types/                     # Shared TypeScript types
│   │   ├── api.ts
│   │   ├── models.ts
│   │   └── index.ts
│   │
│   ├── test/                      # Test utilities and setup
│   │   ├── setup.ts              # Vitest setup
│   │   ├── mocks/                # Mock data and services
│   │   │   └── handlers.ts
│   │   └── utils.ts              # Test helpers
│   │
│   ├── styles/                    # Global styles
│   │   ├── main.css
│   │   ├── variables.css
│   │   └── reset.css
│   │
│   └── assets/                    # Static assets (processed by Vite)
│       ├── images/
│       └── fonts/
│
├── public/                        # Static assets (copied as-is)
│   ├── favicon.ico
│   └── robots.txt
│
├── dist/                          # Build output (gitignored)
├── docs/                          # Generated TypeDoc (gitignored)
├── coverage/                      # Test coverage (gitignored)
│
├── .env                           # Environment variables (gitignored)
├── .env.example                   # Environment template
├── .env.development               # Dev environment
├── .env.production                # Prod environment
│
├── vite.config.ts                 # Vite configuration
├── vitest.config.ts               # Vitest configuration (if separate)
├── tsconfig.json                  # TypeScript configuration
├── tsconfig.node.json             # Node TypeScript config
├── typedoc.json                   # TypeDoc configuration
├── .eslintrc.cjs                  # ESLint configuration
├── .prettierrc                    # Prettier configuration
├── package.json                   # Dependencies and scripts
└── README.md                      # Project documentation
```

### Environment Variables Pattern

```typescript
// src/vite-env.d.ts

/// <reference types="vite/client" />

interface ImportMetaEnv {
  // API Configuration
  readonly VITE_API_URL: string;
  readonly VITE_API_KEY: string;
  readonly VITE_API_TIMEOUT: string;

  // Feature Flags
  readonly VITE_ENABLE_ANALYTICS: string;
  readonly VITE_ENABLE_DEBUG: string;

  // Application Settings
  readonly VITE_APP_NAME: string;
  readonly VITE_APP_VERSION: string;

  // Built-in Vite variables
  readonly DEV: boolean;
  readonly PROD: boolean;
  readonly MODE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
```

```bash
# .env.example

# API Configuration
VITE_API_URL=https://api.example.com
VITE_API_KEY=your-api-key-here
VITE_API_TIMEOUT=30000

# Feature Flags
VITE_ENABLE_ANALYTICS=false
VITE_ENABLE_DEBUG=true

# Application Settings
VITE_APP_NAME=My Vite App
```

```typescript
// src/config/env.ts - Type-safe environment access

/**
 * Application configuration from environment variables.
 *
 * @example
 * ```typescript
 * import { config } from '@/config/env';
 *
 * console.log(config.apiUrl);
 * console.log(config.isDev);
 * ```
 */
export const config = {
  // API
  apiUrl: import.meta.env.VITE_API_URL,
  apiKey: import.meta.env.VITE_API_KEY,
  apiTimeout: parseInt(import.meta.env.VITE_API_TIMEOUT || '30000', 10),

  // Feature Flags
  enableAnalytics: import.meta.env.VITE_ENABLE_ANALYTICS === 'true',
  enableDebug: import.meta.env.VITE_ENABLE_DEBUG === 'true',

  // Environment
  isDev: import.meta.env.DEV,
  isProd: import.meta.env.PROD,
  mode: import.meta.env.MODE,
} as const;
```

---

## References

- [Vite Documentation](https://vitejs.dev/)
- [Vitest Documentation](https://vitest.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
- [Preact Signals](https://preactjs.com/guide/v10/signals/)
- [TypeDoc Documentation](https://typedoc.org/)
- [Rollup Documentation](https://rollupjs.org/)

---

**Last Updated:** 2026-01-18
**Version:** 1.0
**Maintainer:** Development Team
