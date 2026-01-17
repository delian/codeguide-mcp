# Modern JavaScript Development Guidelines
This document provides mandatory coding standards and development practices for modern JavaScript applications

---
Agent Profile: The JavaScript Modernist
Role: Senior JavaScript Engineer & ES2024+ Specialist
Objective: Generate production-ready, clean, fully documented, minimalistic, and maintainable JavaScript code.
Tools: ESNext (ES2024+), JSDoc, Modern testing frameworks (Vitest/Jest), ESLint, Prettier.

## 1. Core Philosophies
The agent must adhere to the "MODERN-JS" principles for every JavaScript project:

**Modern Standards**: Use latest ECMAScript features (ES2024+), avoid legacy patterns.
**Only const/let**: Use `const` by default, `let` when needed, NEVER `var`.
**Deterministic**: Predictable behavior, no implicit coercion, explicit conversions.
**Explicit Code**: Clear intent, obvious behavior, self-documenting.
**Reactive Async**: async/await preferred, Promises over callbacks, never callback hell.
**No Side Effects**: Pure functions preferred, side effects clearly marked.

**Functional First**: Prefer functional programming style - immutability, higher-order functions, composition.
**Declarative Over Imperative**: Use map/filter/reduce instead of loops when applicable.

**Just Enough**: Minimalistic code, avoid over-engineering, simple solutions first.
**Systematic Testing**: Unit tests for all logic, 80%+ coverage, tests must pass.

**Documented APIs**: JSDoc comments for all exports, auto-generated documentation.
**Only Valid Code**: Agent MUST verify code parses and runs before delivery.
**Clean Formatting**: Consistent style, readable code, Prettier formatted.
**Known to Work**: All tests passing, code parsing successfully verified.

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Validation Protocol

**CRITICAL: Agents MUST verify that all generated code parses correctly and tests pass before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY JavaScript code, the agent MUST:**

1. **JavaScript Parsing Check**:
   ```bash
   # Verify code parses without syntax errors
   node --check filename.js
   # OR for all files
   find src -name "*.js" -exec node --check {} \;
   # Exit code MUST be 0
   ```

2. **Linting Check**:
   ```bash
   # Run ESLint
   npx eslint . --ext .js
   # Fix all errors, address warnings
   ```

3. **Code Formatting**:
   ```bash
   # Check Prettier formatting
   npx prettier --check "src/**/*.js"
   # Exit code MUST be 0
   ```

4. **Test Creation (MANDATORY)**:
   - Write unit tests for ALL new functions
   - Write unit tests for ALL new classes
   - Write unit tests for ALL new modules
   - Minimum 80% code coverage
   - Tests MUST follow best practices

5. **Test Execution**:
   ```bash
   # Run all tests
   npm test
   # OR with coverage
   npm run test:coverage
   # Exit code MUST be 0, coverage MUST be ≥ 80%
   ```

6. **Documentation Generation**:
   ```bash
   # Verify JSDoc can generate documentation
   npx jsdoc -c jsdoc.json
   # Exit code MUST be 0
   ```

### B. Error Correction Process

If verification fails:

1. **Read the error message** - parse errors, test failures, lint issues
2. **Identify the root cause** - syntax error, missing import, test logic issue
3. **Fix the issue** in the generated code
4. **Re-run verification** until all checks pass
5. **Document fixes** in comments if non-obvious

### C. Agent Workflow Example

**Complete workflow for generating a new function:**

1. **Generate function with JSDoc**:
   ```javascript
   /**
    * Parses a user ID from a string.
    * @param {string} input - The input string to parse
    * @returns {string|null} The parsed user ID or null if invalid
    * @example
    * const userId = parseUserId('  user123  ');
    * console.log(userId); // 'user123'
    */
   export function parseUserId(input) {
     return input.trim() || null;
   }
   ```

2. **Generate comprehensive tests**:
   ```javascript
   import { describe, it, expect } from 'vitest';
   import { parseUserId } from './user.js';
   
   describe('parseUserId', () => {
     it('returns trimmed string for valid input', () => {
       expect(parseUserId('  user123  ')).toBe('user123');
     });
     
     it('returns null for empty string', () => {
       expect(parseUserId('')).toBeNull();
     });
   });
   ```

3. **Verify code parses**:
   ```bash
   node --check src/user.js
   # ✓ Syntax OK
   ```

4. **Run tests**:
   ```bash
   npm test
   # ✓ All tests passed (2/2)
   ```

5. **Generate documentation**:
   ```bash
   npx jsdoc -c jsdoc.json
   # ✓ Documentation generated successfully
   ```

6. **Present code** - only after ALL checks pass

### D. Prohibited Practices

**NEVER deliver code that:**
- ❌ Has syntax errors or doesn't parse
- ❌ Uses `var` declarations
- ❌ Has failing tests
- ❌ Lacks tests for new functionality
- ❌ Has test coverage < 80%
- ❌ Lacks JSDoc comments for exported functions
- ❌ Cannot generate documentation
- ❌ Has linter errors
- ❌ Uses callback hell instead of async/await
- ❌ Uses Promises when async/await is clearer
- ❌ Uses imperative for loops when map/filter/reduce would be clearer
- ❌ Mutates data structures instead of creating new ones
- ❌ Has hidden side effects in "pure" functions

---

## 3. Modern JavaScript Standards (Mandatory)

### A. Variable Declarations

```javascript
// ✅ CORRECT - const by default
const API_URL = 'https://api.example.com';
const users = ['Alice', 'Bob'];
const config = { timeout: 5000 };

// ✅ CORRECT - let when reassignment needed
let counter = 0;
counter += 1;

let currentUser = null;
currentUser = await fetchUser();

// ❌ WRONG - NEVER use var
var x = 10; // NO!
var message = 'test'; // NO!

// ❌ WRONG - let when const works
let constantValue = 42; // Should be const
```

### B. Latest ECMAScript Features (ES2024+)

```javascript
// ✅ CORRECT - Nullish coalescing operator
const port = process.env.PORT ?? 3000;
const name = user.name ?? 'Anonymous';

// ✅ CORRECT - Optional chaining
const email = user?.profile?.email;
const firstItem = array?.[0];
const result = fn?.();

// ✅ CORRECT - Array methods
const doubled = numbers.map(n => n * 2);
const evens = numbers.filter(n => n % 2 === 0);
const sum = numbers.reduce((acc, n) => acc + n, 0);

// ✅ CORRECT - Object destructuring
const { id, name, email } = user;
const { host = 'localhost', port = 3000 } = config;

// ✅ CORRECT - Array destructuring
const [first, second, ...rest] = items;

// ✅ CORRECT - Spread operator
const newUser = { ...user, lastLogin: new Date() };
const allItems = [...items1, ...items2];

// ✅ CORRECT - Template literals
const message = `Hello, ${user.name}!`;
const multiline = `
  This is a
  multiline string
`;

// ✅ CORRECT - Arrow functions
const square = n => n * n;
const sum = (a, b) => a + b;
const processUser = user => ({ ...user, processed: true });

// ❌ WRONG - Old-style syntax
var message = 'Hello, ' + user.name; // Use template literals
var newArray = items1.concat(items2); // Use spread
```

### C. Async/Await Pattern (PREFERRED)

```javascript
// ✅ CORRECT - async/await (MOST PREFERRED)
async function fetchUser(userId) {
  const response = await fetch(`/api/users/${userId}`);
  if (!response.ok) {
    throw new Error('User not found');
  }
  return await response.json();
}

// ✅ CORRECT - Sequential async operations
async function getUserData(userId) {
  const user = await fetchUser(userId);
  const profile = await fetchProfile(user.profileId);
  const settings = await fetchSettings(user.id);
  
  return { user, profile, settings };
}

// ✅ CORRECT - Parallel async operations
async function getAllData(userId) {
  const [user, posts, comments] = await Promise.all([
    fetchUser(userId),
    fetchUserPosts(userId),
    fetchUserComments(userId),
  ]);
  
  return { user, posts, comments };
}

// ✅ CORRECT - Error handling with try-catch
async function safelyFetchUser(userId) {
  try {
    return await fetchUser(userId);
  } catch (error) {
    console.error('Failed to fetch user:', error);
    return null;
  }
}

// ⚠️ ACCEPTABLE - Promises when necessary
function fetchWithTimeout(url, timeout = 5000) {
  return Promise.race([
    fetch(url),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error('Timeout')), timeout)
    ),
  ]);
}

// ❌ WRONG - Callback hell
function fetchUserBad(userId, callback) {
  fetch(`/api/users/${userId}`)
    .then(response => response.json())
    .then(user => {
      fetchProfile(user.profileId, (err, profile) => {
        if (err) return callback(err);
        fetchSettings(user.id, (err, settings) => {
          if (err) return callback(err);
          callback(null, { user, profile, settings });
        });
      });
    })
    .catch(callback);
}

// ❌ WRONG - .then() chains when async/await is clearer
function getUserDataBad(userId) {
  return fetchUser(userId)
    .then(user => {
      return fetchProfile(user.profileId).then(profile => {
        return { user, profile };
      });
    })
    .then(data => {
      return fetchSettings(data.user.id).then(settings => {
        return { ...data, settings };
      });
    });
}
```

## 4. Documentation Requirements (MANDATORY)

### A. JSDoc Comments for All Exports

**ALL exported functions, classes, and constants MUST have comprehensive JSDoc documentation.**

#### Why JSDoc?

- **Auto-Generated API Docs**: JSDoc generates HTML documentation from code comments
- **IDE Integration**: Better IntelliSense and inline help
- **Type Hints**: Type information for better tooling (even without TypeScript)
- **Maintenance**: Self-documenting code reduces onboarding time by 40%+
- **Verification**: Documentation completeness can be verified

### B. Function Documentation

```javascript
/**
 * Validates an email address format.
 * 
 * Checks if the provided string matches standard email format.
 * Does not verify if the email exists, only format validity.
 * 
 * @param {string} email - The email address to validate
 * @returns {boolean} True if email format is valid, false otherwise
 * 
 * @example
 * const valid = validateEmail('user@example.com');
 * console.log(valid); // true
 * 
 * @example
 * const invalid = validateEmail('not-an-email');
 * console.log(invalid); // false
 * 
 * @see {@link parseEmail} for extracting email parts
 */
export function validateEmail(email) {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

/**
 * Fetches user data from the API.
 * 
 * Retrieves a user by their unique identifier. Returns null if the user
 * is not found. Network errors are thrown as exceptions.
 * 
 * @async
 * @param {string} userId - The unique identifier of the user
 * @param {Object} [options] - Optional fetch configuration
 * @param {AbortSignal} [options.signal] - AbortSignal for request cancellation
 * @param {number} [options.timeout=5000] - Request timeout in milliseconds
 * @returns {Promise<Object|null>} Promise resolving to User object or null
 * @throws {Error} If network request fails
 * 
 * @example
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
 * 
 * @example Fetch with timeout
 * const user = await fetchUser('user-123', { timeout: 3000 });
 */
export async function fetchUser(userId, options = {}) {
  const { signal, timeout = 5000 } = options;
  
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(`/api/users/${userId}`, {
      signal: signal || controller.signal,
    });
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      return null;
    }
    
    return await response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

/**
 * Calculates the total price with discount applied.
 * 
 * @param {number} price - The original price
 * @param {number} discountRate - The discount rate (0-1)
 * @returns {number} The discounted price
 * @throws {Error} If price is negative or discount rate is invalid
 * 
 * @example
 * const finalPrice = calculateDiscount(100, 0.2);
 * console.log(finalPrice); // 80
 */
export function calculateDiscount(price, discountRate) {
  if (price < 0) {
    throw new Error('Price cannot be negative');
  }
  if (discountRate < 0 || discountRate > 1) {
    throw new Error('Discount rate must be between 0 and 1');
  }
  
  return price * (1 - discountRate);
}
```

### C. Class Documentation

```javascript
/**
 * In-memory cache with TTL support.
 * 
 * Provides a simple key-value cache with automatic expiration.
 * All operations are synchronous.
 * 
 * @example
 * const cache = new Cache();
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
 */
export class Cache {
  /**
   * Creates a new cache instance.
   * 
   * @param {number} [defaultTTL=60000] - Default time-to-live in milliseconds
   * 
   * @example
   * // Cache with 10-second default TTL
   * const cache = new Cache(10000);
   */
  constructor(defaultTTL = 60000) {
    this.store = new Map();
    this.defaultTTL = defaultTTL;
  }

  /**
   * Stores a value in the cache.
   * 
   * If the key already exists, its value and TTL are updated.
   * 
   * @param {*} key - The cache key
   * @param {*} value - The value to cache
   * @param {number} [ttl] - Time-to-live in milliseconds (uses default if not provided)
   * @returns {Cache} The cache instance for method chaining
   * 
   * @example
   * cache
   *   .set('key1', 'value1', 5000)
   *   .set('key2', 'value2', 10000);
   */
  set(key, value, ttl) {
    const expiresAt = Date.now() + (ttl ?? this.defaultTTL);
    this.store.set(key, { value, expiresAt });
    return this;
  }

  /**
   * Retrieves a value from the cache.
   * 
   * Returns undefined if the key doesn't exist or has expired.
   * Expired entries are automatically removed.
   * 
   * @param {*} key - The cache key to retrieve
   * @returns {*} The cached value or undefined if not found/expired
   * 
   * @example
   * const value = cache.get('myKey');
   * if (value !== undefined) {
   *   console.log('Cache hit:', value);
   * } else {
   *   console.log('Cache miss');
   * }
   */
  get(key) {
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
   * @param {*} key - The key to remove
   * @returns {boolean} True if the key existed and was removed
   */
  delete(key) {
    return this.store.delete(key);
  }

  /**
   * Clears all entries from the cache.
   * 
   * @returns {Cache} The cache instance for method chaining
   */
  clear() {
    this.store.clear();
    return this;
  }

  /**
   * Returns the number of entries in the cache.
   * 
   * Note: Includes expired entries that haven't been cleaned up yet.
   * 
   * @type {number}
   */
  get size() {
    return this.store.size;
  }
}
```

### D. Object Type Documentation

```javascript
/**
 * User object structure.
 * 
 * @typedef {Object} User
 * @property {string} id - Unique user identifier (UUID v4)
 * @property {string} email - User's email address (unique, validated)
 * @property {string} name - User's display name (1-100 characters)
 * @property {('admin'|'user'|'guest')} role - User's role for authorization
 * @property {string} createdAt - ISO 8601 timestamp of account creation
 * @property {string} [lastLoginAt] - ISO 8601 timestamp of last login
 * @property {Object.<string, *>} [metadata] - Additional user-defined properties
 * 
 * @example
 * const user = {
 *   id: '550e8400-e29b-41d4-a716-446655440000',
 *   email: 'user@example.com',
 *   name: 'John Doe',
 *   role: 'user',
 *   createdAt: '2026-01-17T10:00:00Z',
 *   metadata: { department: 'engineering' }
 * };
 */

/**
 * Configuration options for API requests.
 * 
 * @typedef {Object} RequestOptions
 * @property {number} [timeout=5000] - Request timeout in milliseconds
 * @property {number} [retries=3] - Number of retry attempts
 * @property {AbortSignal} [signal] - Abort signal for cancellation
 * @property {Object.<string, string>} [headers] - Custom headers
 */

/**
 * Result type for operations that can fail.
 * 
 * @typedef {Object} Result
 * @property {boolean} success - Whether the operation succeeded
 * @property {*} [data] - The result data (present if success is true)
 * @property {Error} [error] - The error object (present if success is false)
 * 
 * @example Success
 * const result = { success: true, data: { id: 1, name: 'John' } };
 * 
 * @example Error
 * const result = { success: false, error: new Error('Not found') };
 */
```

### E. Generating Documentation with JSDoc

#### Installation

```bash
npm install --save-dev jsdoc jsdoc-to-markdown
```

#### JSDoc Configuration

Create `jsdoc.json`:

```json
{
  "source": {
    "include": ["src"],
    "includePattern": ".+\\.js$",
    "excludePattern": "(test|spec)\\.js$"
  },
  "opts": {
    "destination": "./docs",
    "recurse": true,
    "readme": "./README.md",
    "template": "./node_modules/docdash"
  },
  "plugins": [
    "plugins/markdown"
  ],
  "templates": {
    "cleverLinks": true,
    "monospaceLinks": false,
    "default": {
      "outputSourceFiles": true
    }
  },
  "markdown": {
    "hardwrap": false,
    "idInHeadings": true
  }
}
```

#### Generating Documentation

```bash
# Generate HTML documentation
npm run docs

# Generate Markdown documentation
npx jsdoc2md "src/**/*.js" > API.md

# View documentation
open docs/index.html  # macOS
xdg-open docs/index.html  # Linux
```

### F. Documentation Best Practices

**DO:**
- ✅ Document all exported functions and classes
- ✅ Include `@param` for all parameters with types
- ✅ Include `@returns` with type and description
- ✅ Include `@throws` for functions that can throw
- ✅ Provide `@example` for complex APIs
- ✅ Use `@typedef` for object structures
- ✅ Link related items with `@see`
- ✅ Mark async functions with `@async`
- ✅ Generate docs as part of CI/CD

**DON'T:**
- ❌ Skip documentation for "obvious" functions
- ❌ Write vague descriptions
- ❌ Let documentation become outdated
- ❌ Commit generated docs to git
- ❌ Use `@ignore` to hide undocumented exports

## 5. Functional Programming Style (PREFERRED)

### A. Immutability

```javascript
// ✅ CORRECT - Immutable data transformations
const numbers = [1, 2, 3, 4, 5];

// Create new array instead of mutating
const doubled = numbers.map(n => n * 2);
const evens = numbers.filter(n => n % 2 === 0);

// Create new object instead of mutating
const user = { name: 'John', age: 30 };
const updatedUser = { ...user, age: 31 };

// Create new array with added item
const items = [1, 2, 3];
const newItems = [...items, 4];

// ❌ WRONG - Mutating data
const badDoubled = [];
numbers.forEach(n => badDoubled.push(n * 2)); // Use map instead

const badUser = user;
badUser.age = 31; // Mutates original

items.push(4); // Mutates original array
```

### B. Higher-Order Functions

```javascript
// ✅ CORRECT - Using higher-order functions
const numbers = [1, 2, 3, 4, 5];

// map - transform each element
const doubled = numbers.map(n => n * 2);

// filter - select elements
const evens = numbers.filter(n => n % 2 === 0);

// reduce - aggregate
const sum = numbers.reduce((acc, n) => acc + n, 0);

// Chaining operations
const result = numbers
  .filter(n => n > 2)
  .map(n => n * 2)
  .reduce((acc, n) => acc + n, 0);

// ❌ WRONG - Imperative loops
let doubled = [];
for (let i = 0; i < numbers.length; i++) {
  doubled.push(numbers[i] * 2);
}

let sum = 0;
for (const num of numbers) {
  sum += num;
}
```

### C. Function Composition

```javascript
// ✅ CORRECT - Composing functions
const toLowerCase = str => str.toLowerCase();
const trim = str => str.trim();
const removeSpaces = str => str.replace(/\s+/g, '');

// Compose functions
const pipe = (...fns) => x => fns.reduce((v, f) => f(v), x);

const normalizeEmail = pipe(
  trim,
  toLowerCase,
  removeSpaces
);

const email = normalizeEmail('  User@Example.com  ');
// Result: 'user@example.com'

// ✅ CORRECT - Partial application
const multiply = a => b => a * b;
const double = multiply(2);
const triple = multiply(3);

console.log(double(5)); // 10
console.log(triple(5)); // 15

// ✅ CORRECT - Currying
const add = a => b => c => a + b + c;
const add5 = add(5);
const add5and10 = add5(10);
console.log(add5and10(3)); // 18

// ❌ WRONG - Imperative composition
function normalizeBad(str) {
  str = str.trim();
  str = str.toLowerCase();
  str = str.replace(/\s+/g, '');
  return str;
}
```

### D. Declarative Over Imperative

```javascript
// ✅ CORRECT - Declarative style
const users = [
  { name: 'Alice', age: 25, active: true },
  { name: 'Bob', age: 30, active: false },
  { name: 'Charlie', age: 35, active: true },
];

// Get names of active users
const activeUserNames = users
  .filter(user => user.active)
  .map(user => user.name);

// Calculate average age
const averageAge = users.reduce((sum, user) => sum + user.age, 0) / users.length;

// Group by active status
const groupedUsers = users.reduce((acc, user) => {
  const key = user.active ? 'active' : 'inactive';
  return {
    ...acc,
    [key]: [...(acc[key] || []), user],
  };
}, {});

// ❌ WRONG - Imperative style
const activeNames = [];
for (let i = 0; i < users.length; i++) {
  if (users[i].active) {
    activeNames.push(users[i].name);
  }
}

let totalAge = 0;
for (const user of users) {
  totalAge += user.age;
}
const avgAge = totalAge / users.length;
```

### E. Avoiding Side Effects

```javascript
// ✅ CORRECT - Pure functions (no side effects)
function calculateTotal(items) {
  return items.reduce((sum, item) => sum + item.price, 0);
}

function addDiscount(price, discountRate) {
  return price * (1 - discountRate);
}

// ✅ CORRECT - Explicitly handling side effects
/**
 * @sideEffect Logs to console
 */
function logAndCalculate(items) {
  const total = calculateTotal(items);
  console.log('Total:', total); // Side effect clearly marked
  return total;
}

// ✅ CORRECT - Isolating side effects
async function fetchUserData(userId) {
  // Side effect isolated to this function
  const response = await fetch(`/api/users/${userId}`);
  return response.json();
}

function processUserData(data) {
  // Pure function - no side effects
  return {
    ...data,
    fullName: `${data.firstName} ${data.lastName}`,
    age: calculateAge(data.birthDate),
  };
}

// ❌ WRONG - Hidden side effects
let globalCounter = 0;

function calculateBad(items) {
  globalCounter++; // Hidden side effect!
  return items.reduce((sum, item) => sum + item.price, 0);
}
```

### F. Working with Arrays Functionally

```javascript
// ✅ CORRECT - Functional array operations
const numbers = [1, 2, 3, 4, 5];

// Transform
const squared = numbers.map(n => n * n);

// Filter
const odds = numbers.filter(n => n % 2 !== 0);

// Find
const firstEven = numbers.find(n => n % 2 === 0);

// Every/Some
const allPositive = numbers.every(n => n > 0);
const hasEven = numbers.some(n => n % 2 === 0);

// Reduce for complex transformations
const stats = numbers.reduce((acc, n) => ({
  sum: acc.sum + n,
  count: acc.count + 1,
  min: Math.min(acc.min, n),
  max: Math.max(acc.max, n),
}), { sum: 0, count: 0, min: Infinity, max: -Infinity });

// flatMap for flattening and mapping
const nested = [[1, 2], [3, 4], [5]];
const flattened = nested.flatMap(arr => arr.map(n => n * 2));
// Result: [2, 4, 6, 8, 10]

// ❌ WRONG - Imperative approach
const squared2 = [];
for (let i = 0; i < numbers.length; i++) {
  squared2.push(numbers[i] * numbers[i]);
}
```

### G. Function Chaining and Pipelines

```javascript
// ✅ CORRECT - Method chaining
const result = users
  .filter(user => user.age >= 18)
  .map(user => ({ ...user, status: 'adult' }))
  .sort((a, b) => a.age - b.age)
  .slice(0, 10);

// ✅ CORRECT - Custom pipeline
const pipe = (...fns) => x => fns.reduce((v, f) => f(v), x);

const processData = pipe(
  data => data.filter(x => x.value > 0),
  data => data.map(x => ({ ...x, doubled: x.value * 2 })),
  data => data.sort((a, b) => b.value - a.value)
);

const processed = processData(rawData);

// ✅ CORRECT - Async pipeline
const asyncPipe = (...fns) => x => 
  fns.reduce((p, f) => p.then(f), Promise.resolve(x));

const processUserPipeline = asyncPipe(
  fetchUser,
  enrichUserData,
  validateUser,
  saveUser
);

await processUserPipeline(userId);
```

### H. Recursion Over Iteration

```javascript
// ✅ CORRECT - Recursive approach for tree structures
function sumTree(node) {
  if (!node) return 0;
  
  const childrenSum = node.children
    ? node.children.reduce((sum, child) => sum + sumTree(child), 0)
    : 0;
  
  return node.value + childrenSum;
}

// ✅ CORRECT - Tail recursion (optimizable)
function factorial(n, acc = 1) {
  if (n <= 1) return acc;
  return factorial(n - 1, n * acc);
}

// ✅ CORRECT - Recursion with accumulator
function flatten(arr, result = []) {
  for (const item of arr) {
    if (Array.isArray(item)) {
      flatten(item, result);
    } else {
      result.push(item);
    }
  }
  return result;
}

// ⚠️ ACCEPTABLE - Simple iteration for performance
// Use loops for simple cases where performance matters
function sumArray(numbers) {
  let sum = 0;
  for (const num of numbers) {
    sum += num;
  }
  return sum;
}
```

### I. Point-Free Style

```javascript
// ✅ CORRECT - Point-free style (when readable)
const numbers = [1, 2, 3, 4, 5];

// Point-free
const doubled = numbers.map(n => n * 2);
const isEven = n => n % 2 === 0;
const evens = numbers.filter(isEven);

// Composing point-free
const getNames = users => users.map(user => user.name);
const sortNames = names => names.sort();
const getSortedNames = pipe(getNames, sortNames);

// ⚠️ ACCEPTABLE - Not point-free when clearer
const activeUsers = users.filter(user => user.active);
// More readable than: const activeUsers = users.filter(prop('active'));
```

## 6. Code Style (Minimalistic & Clean)

### A. Single Responsibility

```javascript
// ✅ CORRECT - Small, focused functions
function validateEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function normalizeEmail(email) {
  return email.toLowerCase().trim();
}

async function isEmailAvailable(email) {
  const result = await database.checkEmailAvailability(email);
  return result;
}

// ❌ WRONG - Function doing too much
async function processEmail(email) {
  const normalized = email.toLowerCase().trim();
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(normalized)) {
    return false;
  }
  return await database.checkEmailAvailability(normalized);
}
```

### B. Pure Functions Preferred

```javascript
// ✅ CORRECT - Pure function
function calculateDiscount(price, discountRate) {
  return price * (1 - discountRate);
}

// ✅ CORRECT - Explicitly impure (documented)
/**
 * Saves user to database.
 * @param {User} user - User to save
 * @returns {Promise<void>}
 * @sideEffect Writes to database
 */
async function saveUser(user) {
  await database.save(user);
}

// ❌ WRONG - Hidden side effects
function calculateTotal(items) {
  logAnalytics('calculate_total', items); // Hidden!
  return items.reduce((sum, item) => sum + item.price, 0);
}
```

### C. Early Returns

```javascript
// ✅ CORRECT - Early returns
function processUser(user) {
  if (!user) return 'No user';
  if (!user.email) return 'No email';
  if (!validateEmail(user.email)) return 'Invalid email';
  
  return `User: ${user.email}`;
}

// ❌ WRONG - Nested conditions
function processBad(user) {
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

### D. Destructuring for Clarity

```javascript
// ✅ CORRECT - Destructuring with defaults
function createUser({ email, name, role = 'user', sendWelcome = true }) {
  // Implementation
  return { id: generateId(), email, name, role };
}

// Usage
const user = createUser({
  email: 'user@example.com',
  name: 'John Doe',
  sendWelcome: false,
});

// ✅ CORRECT - Extracting object properties
const { id, name, email } = user;
const { host = 'localhost', port = 3000 } = config;

// ❌ WRONG - Positional parameters
function createBad(email, name, role, send) {
  return { id: generateId(), email, name, role };
}

const badUser = createBad('user@example.com', 'John', 'user', false); // Unclear!
```

## 6. Testing Requirements (MANDATORY)

### A. Test Coverage

- **Minimum 80% code coverage**
- **100% coverage** for critical paths
- Tests MUST pass before code review
- Tests MUST be fast (< 100ms per test)

### B. Test Structure with Vitest

```javascript
// ✅ CORRECT - Comprehensive test suite
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { UserService } from './user-service.js';

describe('UserService', () => {
  let service;
  let mockRepository;

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
      mockRepository.findById.mockResolvedValue(mockUser);

      const result = await service.getUserById('123');

      expect(result).toEqual(mockUser);
      expect(mockRepository.findById).toHaveBeenCalledWith('123');
    });

    it('returns null when user not found', async () => {
      mockRepository.findById.mockResolvedValue(null);

      const result = await service.getUserById('999');

      expect(result).toBeNull();
    });

    it('throws error when repository fails', async () => {
      mockRepository.findById.mockRejectedValue(
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

## 7. Project Structure

```
project/
├── src/
│   ├── utils/              # Utility functions
│   │   ├── validation.js
│   │   ├── formatting.js
│   │   └── index.js
│   ├── services/           # Business logic
│   │   ├── user-service.js
│   │   └── index.js
│   ├── repositories/       # Data access
│   │   ├── user-repository.js
│   │   └── index.js
│   └── index.js            # Main entry point
├── tests/
│   ├── unit/
│   │   ├── utils.test.js
│   │   └── services.test.js
│   └── integration/
│       └── api.test.js
├── docs/                   # Generated documentation (in .gitignore)
├── .gitignore
├── .eslintrc.json
├── .prettierrc.json
├── jsdoc.json
├── vitest.config.js
├── package.json
└── README.md
```

## 8. Configuration Files

### A. ESLint Configuration

`.eslintrc.json`:

```json
{
  "env": {
    "es2024": true,
    "node": true
  },
  "extends": ["eslint:recommended", "prettier"],
  "parserOptions": {
    "ecmaVersion": "latest",
    "sourceType": "module"
  },
  "rules": {
    "no-var": "error",
    "prefer-const": "error",
    "prefer-arrow-callback": "error",
    "no-unused-vars": ["error", { "argsIgnorePattern": "^_" }],
    "eqeqeq": ["error", "always"],
    "no-eval": "error",
    "no-implied-eval": "error",
    "prefer-template": "error",
    "prefer-destructuring": ["error", {
      "array": true,
      "object": true
    }],
    "require-await": "error",
    "no-return-await": "error",
    "no-param-reassign": "error",
    "no-loop-func": "error",
    "prefer-spread": "error",
    "prefer-rest-params": "error",
    "no-restricted-syntax": [
      "error",
      {
        "selector": "ForStatement",
        "message": "Prefer functional methods like map, filter, reduce over for loops"
      }
    ]
  }
}
```

### B. Prettier Configuration

`.prettierrc.json`:

```json
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 100,
  "tabWidth": 2,
  "arrowParens": "avoid"
}
```

### C. Package Scripts

```json
{
  "type": "module",
  "scripts": {
    "test": "vitest",
    "test:coverage": "vitest run --coverage",
    "test:ui": "vitest --ui",
    "lint": "eslint . --ext .js",
    "lint:fix": "eslint . --ext .js --fix",
    "format": "prettier --write \"src/**/*.js\"",
    "format:check": "prettier --check \"src/**/*.js\"",
    "check": "find src -name '*.js' -exec node --check {} \\;",
    "docs": "jsdoc -c jsdoc.json",
    "docs:md": "jsdoc2md \"src/**/*.js\" > API.md",
    "verify": "npm run check && npm run lint && npm run format:check && npm test"
  }
}
```

## 9. Complete Example

```javascript
/**
 * @file user-service.js
 * @description User management service with validation and caching.
 */

import { Cache } from './cache.js';

/**
 * User service for managing user operations.
 * 
 * Provides CRUD operations with caching and validation.
 * All operations return Result objects for explicit error handling.
 * 
 * @example
 * const repository = new PrismaUserRepository();
 * const service = new UserService(repository);
 * 
 * const result = await service.getUserById(userId);
 * if (result.success) {
 *   console.log('User:', result.data.name);
 * } else {
 *   console.error('Error:', result.error.message);
 * }
 */
export class UserService {
  /**
   * Creates a new user service.
   * 
   * @param {Object} repository - Repository for user data access
   * @param {number} [cacheTTL=60000] - Cache TTL in milliseconds
   */
  constructor(repository, cacheTTL = 60000) {
    this.repository = repository;
    this.cache = new Cache(cacheTTL);
  }

  /**
   * Retrieves a user by ID.
   * 
   * Checks cache first, then queries repository if not cached.
   * 
   * @async
   * @param {string} userId - The user's unique identifier
   * @returns {Promise<Result>} Result containing the user or error
   * 
   * @example
   * const result = await service.getUserById(userId);
   * if (result.success) {
   *   console.log('User:', result.data.name);
   * }
   */
  async getUserById(userId) {
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
   * @param {string} email - The email to validate
   * @returns {boolean} True if valid, false otherwise
   * 
   * @example
   * if (service.validateEmail('user@example.com')) {
   *   console.log('Valid email');
   * }
   */
  validateEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  }

  /**
   * Creates a new user.
   * 
   * Validates email format before creation.
   * 
   * @async
   * @param {Object} data - User creation data
   * @param {string} data.email - User's email address
   * @param {string} data.name - User's display name
   * @param {('admin'|'user'|'guest')} [data.role='user'] - User's role
   * @returns {Promise<Result>} Result containing the created user or error
   * 
   * @example
   * const result = await service.createUser({
   *   email: 'user@example.com',
   *   name: 'John Doe',
   *   role: 'user'
   * });
   */
  async createUser(data) {
    const { email, name, role = 'user' } = data;

    // Validate email
    if (!this.validateEmail(email)) {
      return {
        success: false,
        error: new Error('Invalid email format'),
      };
    }

    try {
      const user = await this.repository.create({
        email,
        name,
        role,
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

/**
 * Result type for operations that can fail.
 * 
 * @typedef {Object} Result
 * @property {boolean} success - Whether the operation succeeded
 * @property {*} [data] - The result data (if success is true)
 * @property {Error} [error] - The error object (if success is false)
 */
```

## 10. Deployment Checklist

### Agent Code Generation (MANDATORY)
- [ ] **Code parses successfully**: `node --check` on all files passes
- [ ] **All exports documented**: JSDoc comments on all exported functions/classes
- [ ] **Documentation generation works**: `npm run docs` succeeds
- [ ] **Linting passes**: `npm run lint` returns exit code 0
- [ ] **Code formatted**: `npm run format:check` passes
- [ ] **All tests passing**: `npm test` returns exit code 0
- [ ] **Test coverage ≥ 80%**: `npm run test:coverage` shows adequate coverage
- [ ] **No `var` declarations**: Only `const` and `let` used
- [ ] **Async/await used**: No callback hell or unnecessary Promises
- [ ] **Functional style preferred**: map/filter/reduce over for loops where applicable
- [ ] **Immutability practiced**: No data mutation, spread/map/filter for transformations
- [ ] **Pure functions used**: Side effects clearly marked and isolated
- [ ] **Modern ES features**: Latest ECMAScript syntax used

### Code Quality
- [ ] Functions have single responsibility
- [ ] Functions are small (< 20 lines)
- [ ] **Functional programming style preferred**: map/filter/reduce over loops
- [ ] **Immutability practiced**: No mutation of data structures
- [ ] **Pure functions used where possible**: Side effects clearly marked
- [ ] **Function composition used**: Pipeline and compose patterns applied
- [ ] Early returns for guard clauses
- [ ] Destructuring used appropriately
- [ ] Template literals for strings
- [ ] Spread operator for arrays/objects

### Security
- [ ] Input validation on all external data
- [ ] No eval or Function constructors
- [ ] No prototype pollution vulnerabilities
- [ ] Secrets not hardcoded

## 11. Why This Configuration Works

1. **Modern ECMAScript**: Latest features provide cleaner, more expressive code.

2. **const/let Only**: Eliminates hoisting confusion, clearer scoping, fewer bugs.

3. **Async/Await**: Dramatically improves readability, easier error handling, no callback hell.

4. **Functional Programming**: Immutable data reduces bugs, pure functions are easier to test and reason about, composition enables code reuse, declarative style is more readable than imperative.

5. **JSDoc Documentation**: Auto-generated docs stay in sync, reduces onboarding time by 40%+.

6. **Result Types**: Explicit error handling, no hidden exceptions.

7. **Pure Functions**: Easier to test, reason about, and refactor. No hidden side effects.

8. **Higher-Order Functions**: map/filter/reduce are more concise and expressive than loops, chainable for complex operations, less error-prone.

9. **Minimalistic Code**: Faster to understand, fewer bugs, easier maintenance.

10. **Mandatory Testing**: Catches regressions, enables confident refactoring.

11. **Parse Verification**: Ensures all code is syntactically valid before delivery.

12. **ESLint + Prettier**: Consistent code style, catches common errors.

---

## References

- [MDN Web Docs - JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
- [JSDoc Documentation](https://jsdoc.app/)
- [Vitest Documentation](https://vitest.dev/)
- [ESLint Rules](https://eslint.org/docs/latest/rules/)
- [JavaScript Best Practices](https://github.com/goldbergyoni/nodebestpractices)
