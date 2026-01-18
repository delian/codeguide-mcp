# Svelte Development Guidelines
This document provides mandatory coding standards and development practices for modern Svelte applications with TypeScript

---
Agent Profile: The Svelte Expert
Role: Senior Svelte Engineer & Reactive Systems Specialist
Objective: Generate production-ready, performant, well-documented, and maintainable Svelte code with TypeScript.
Tools: Svelte 5+, SvelteKit 2+, TypeScript 5.x, Vitest, TypeDoc, Playwright.

## Core Philosophies

The agent must adhere to the "SVELTE-FIRST" principles:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Runes/Signals First**: ALWAYS use Svelte 5 runes (`$state`, `$derived`, `$effect`) — never legacy `$:` syntax for new code.
**Async-First**: Prefer `async`/`await` over `.then()` chains; use `{#await}` blocks for reactive promises.
**Reactive by Default**: Let Svelte's compiler handle reactivity; avoid manual subscriptions.
**TypeScript Strict**: Full type safety with strict mode, no `any`.
**Compiled Away**: Write declarative code; let Svelte compile to optimal JavaScript.
**Component Composition**: Small, focused components with clear interfaces.
**Tested Code**: Unit tests for all logic, component tests for UI.
**Verified Builds**: All code must compile and pass tests before delivery.
**Documented APIs**: JSDoc/TypeDoc for all public APIs.
**Hexagonal Architecture**: Clear separation of domain, application, and infrastructure.

---

## 1. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated Svelte code compiles and passes tests before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY Svelte code, the agent MUST:**

1. **TypeScript Compilation Check**:
   ```bash
   # Verify code compiles without errors
   npm run check
   # OR
   svelte-check --tsconfig ./tsconfig.json
   # Exit code MUST be 0
   ```

2. **Build Verification**:
   ```bash
   # Verify project builds successfully
   npm run build
   # Exit code MUST be 0
   
   # Verify no build warnings
   npm run build 2>&1 | grep -i "warning" && exit 1 || exit 0
   ```

3. **Linting Check**:
   ```bash
   # Verify code passes linting
   npm run lint
   # Exit code MUST be 0
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

5. **Component Tests** (for UI components):
   ```bash
   # Run component tests
   npm run test:component
   # Exit code MUST be 0
   ```

6. **Documentation Generation**:
   ```bash
   # Generate TypeDoc documentation
   npm run docs
   # Verify docs/api/ directory is created
   ```

### B. Error Correction Process

If verification fails:

1. **Read the error message** (Svelte/TypeScript errors are descriptive)
2. **Identify the root cause** (type error, reactivity issue, build config, etc.)
3. **Fix the issue** following Svelte best practices
4. **Re-run verification** until all checks pass
5. **Document any non-obvious fixes**

### C. Prohibited Practices

**NEVER deliver Svelte code that:**
- ❌ Has TypeScript compilation errors
- ❌ Has build failures
- ❌ Fails tests
- ❌ Lacks tests for new functionality
- ❌ Lacks JSDoc comments for public APIs
- ❌ Uses `any` type without justification
- ❌ Mixes Svelte 4 and Svelte 5 syntax (use runes consistently)
- ❌ Uses legacy `$:` reactive statements in Svelte 5 (use runes instead)
- ❌ Uses `.then()` promise chains (use `async`/`await` instead)
- ❌ Uses nested callbacks (use `async`/`await` instead)
- ❌ Has console errors in development mode
- ❌ Violates hexagonal architecture boundaries
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

### Example TDD Workflow for Svelte Component

```typescript
// Step 1: RED - Write failing test first
import { describe, it, expect } from 'vitest';
import { render, fireEvent } from '@testing-library/svelte';
import Counter from './Counter.svelte';

describe('Counter', () => {
  it('displays initial count', () => {
    const { getByText } = render(Counter, { props: { initial: 5 } });
    expect(getByText('Count: 5')).toBeInTheDocument();
  });
  
  it('increments count on button click', async () => {
    const { getByRole, getByText } = render(Counter);
    const button = getByRole('button', { name: /increment/i });
    
    await fireEvent.click(button);
    
    expect(getByText('Count: 1')).toBeInTheDocument();
  });
});

// Run: npm test
// ❌ FAILS - Counter component doesn't exist yet

// Step 2: GREEN - Write minimal implementation
<script lang="ts">
  interface Props {
    initial?: number;
  }
  
  let { initial = 0 }: Props = $props();
  let count = $state(initial);
  
  function increment() {
    count++;
  }
</script>

<div>
  <p>Count: {count}</p>
  <button onclick={increment}>Increment</button>
</div>

// Run: npm test
// ✅ PASSES - tests pass

// Step 3: REFACTOR - Add styling, better structure
<script lang="ts">
  interface Props {
    initial?: number;
    onUpdate?: (count: number) => void;
  }
  
  let { initial = 0, onUpdate }: Props = $props();
  let count = $state(initial);
  
  function increment() {
    count++;
    onUpdate?.(count);
  }
</script>

<div class="counter">
  <p class="count">Count: {count}</p>
  <button class="btn" onclick={increment}>Increment</button>
</div>

<style>
  .counter {
    padding: 1rem;
    border: 1px solid #ddd;
    border-radius: 8px;
  }
  
  .btn {
    background: #007bff;
    color: white;
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 4px;
    cursor: pointer;
  }
</style>
// Tests still pass ✓
```

### Example TDD for Utility Function

```typescript
// Step 1: RED - Write failing test first
import { describe, it, expect } from 'vitest';
import { formatCurrency } from './formatters';

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
});

// Run: npm test
// ❌ FAILS - formatCurrency doesn't exist yet

// Step 2: GREEN - Write minimal implementation
export function formatCurrency(amount: number, currency: 'USD' | 'EUR'): string {
  const symbol = currency === 'USD' ? '$' : '€';
  const formatted = amount.toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  });
  return `${symbol}${formatted}`;
}

// Run: npm test
// ✅ PASSES - tests pass

// Step 3: REFACTOR - Improve with better locale handling
const CURRENCY_CONFIG = {
  USD: { symbol: '$', locale: 'en-US' },
  EUR: { symbol: '€', locale: 'de-DE' }
} as const;

export function formatCurrency(
  amount: number,
  currency: keyof typeof CURRENCY_CONFIG
): string {
  const config = CURRENCY_CONFIG[currency];
  const formatted = amount.toLocaleString(config.locale, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  });
  return `${config.symbol}${formatted}`;
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

### Example Bug Fix for Component

```typescript
// Bug Report #8234: Counter allows negative values when it shouldn't

// Step 1-2: Write test that reproduces the bug
import { describe, it, expect } from 'vitest';
import { render, fireEvent } from '@testing-library/svelte';
import Counter from './Counter.svelte';

describe('Counter - Bug #8234', () => {
  it('prevents negative values - Bug #8234', async () => {
    // Bug: Counter goes negative when decrement clicked at 0
    // Discovered: 2026-01-18
    // This test prevents regression
    
    const { getByRole, getByText } = render(Counter, {
      props: { initial: 0, min: 0 }
    });
    
    const decrementBtn = getByRole('button', { name: /decrement/i });
    await fireEvent.click(decrementBtn);
    
    // Should stay at 0, not go negative
    expect(getByText('Count: 0')).toBeInTheDocument();
  });
});

// Run: npm test
// ❌ FAILS - shows "Count: -1" instead of "Count: 0"

// Step 3: Fix the bug in Counter.svelte
<script lang="ts">
  interface Props {
    initial?: number;
    min?: number;
    max?: number;
  }
  
  let { initial = 0, min, max }: Props = $props();
  let count = $state(initial);
  
  function increment() {
    if (max === undefined || count < max) {
      count++;
    }
  }
  
  function decrement() {
    // FIX: Check minimum before decrementing
    if (min === undefined || count > min) {
      count--;
    }
  }
</script>

<div class="counter">
  <p>Count: {count}</p>
  <button onclick={decrement}>Decrement</button>
  <button onclick={increment}>Increment</button>
</div>

// Run: npm test
// ✅ PASSES - bug fixed, regression prevented ✓
```

### Example Bug Fix for Store

```typescript
// Bug Report #8235: userStore doesn't persist after page reload

// Step 1-2: Write test that reproduces the bug
import { describe, it, expect, beforeEach } from 'vitest';
import { get } from 'svelte/store';
import { userStore } from './user-store';

describe('userStore - Bug #8235', () => {
  beforeEach(() => {
    localStorage.clear();
  });
  
  it('persists user data to localStorage - Bug #8235', () => {
    // Bug: userStore not persisted to localStorage
    // Discovered: 2026-01-18
    // This test prevents regression
    
    const user = { id: '1', name: 'John' };
    userStore.set(user);
    
    // Should be saved to localStorage
    const saved = localStorage.getItem('user');
    expect(saved).not.toBeNull();
    expect(JSON.parse(saved!)).toEqual(user);
  });
  
  it('restores user data from localStorage', () => {
    const user = { id: '1', name: 'John' };
    localStorage.setItem('user', JSON.stringify(user));
    
    // Should restore from localStorage on init
    const restored = get(userStore);
    expect(restored).toEqual(user);
  });
});

// Run: npm test
// ❌ FAILS - localStorage not accessed

// Step 3: Fix the bug in user-store.ts
import { writable } from 'svelte/store';
import type { User } from '$lib/domain/entities/User';

// FIX: Load initial value from localStorage
const stored = typeof window !== 'undefined'
  ? localStorage.getItem('user')
  : null;

const initial: User | null = stored ? JSON.parse(stored) : null;

function createUserStore() {
  const { subscribe, set, update } = writable<User | null>(initial);
  
  return {
    subscribe,
    set: (user: User | null) => {
      // FIX: Persist to localStorage on set
      if (typeof window !== 'undefined') {
        if (user) {
          localStorage.setItem('user', JSON.stringify(user));
        } else {
          localStorage.removeItem('user');
        }
      }
      set(user);
    },
    update
  };
}

export const userStore = createUserStore();

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

## 2. Mental Model

### A. Core Concepts

- **MUST**: Understand that Svelte components compile to imperative JavaScript — write declarative, let compiler optimize.
- **MUST**: Treat `$state` as the source of truth for reactive data (Svelte 5 runes).
- **MUST**: Use `$derived` for computed values, not manual updates.
- **MUST**: Place side effects in `$effect`, not in component body.
- **MUST**: Understand script context runs ONCE on mount (like setup in Vue 3).

### B. Svelte 5 Runes (Modern Syntax)

```typescript
<script lang="ts">
  // State (reactive)
  let count = $state(0);
  
  // Derived (computed)
  let doubled = $derived(count * 2);
  
  // Effect (side effect)
  $effect(() => {
    console.log(`Count changed to ${count}`);
    
    // Cleanup
    return () => {
      console.log('Cleanup');
    };
  });
  
  // Props (component interface)
  let { title, onClose }: { title: string; onClose: () => void } = $props();
</script>
```

### C. Svelte 4 Syntax (LEGACY - Avoid in new code)

```typescript
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  
  // Reactive variable
  let count = 0;
  
  // Reactive statement
  $: doubled = count * 2;
  
  // Reactive effect
  $: {
    console.log(`Count changed to ${count}`);
  }
  
  // Props
  export let title: string;
  export let onClose: () => void;
  
  // Lifecycle
  onMount(() => {
    console.log('Mounted');
    return () => console.log('Cleanup');
  });
</script>
```

**⚠️ CRITICAL**: 
- **MUST**: Use Svelte 5 runes for ALL new code — runes are the future of Svelte
- **MUST**: Migrate legacy `$:` syntax to runes when modifying existing components
- **SHOULD**: Use Svelte 4 syntax ONLY when maintaining legacy projects with no migration path
- **NEVER**: Mix Svelte 4 (`$:`) and Svelte 5 (runes) syntax in the same component

### D. Why Runes/Signals Over Legacy Syntax

**Runes provide:**
1. **Better TypeScript Integration**: Explicit types, no inference issues
2. **Clearer Intent**: `$state` vs `$derived` vs `$effect` are self-documenting
3. **Finer Control**: `$effect.pre`, `$effect.root` for advanced cases
4. **Better Performance**: Compiler can optimize runes more effectively
5. **Explicit Dependencies**: No hidden dependencies like `$:` reactive statements
6. **Standard Across Frameworks**: Aligns with signals in Solid, Angular, Vue, etc.

---

## 3. Async-First Patterns (MANDATORY)

### A. Async/Await Hierarchy

**Preference order (highest to lowest):**

1. **`async`/`await`** (PREFERRED)
2. **`{#await}` blocks** for reactive promises in templates
3. **Promise.all()** / **Promise.allSettled()** for parallel operations
4. **`.then()` chains** (LEGACY - avoid in new code)
5. **Callbacks** (NEVER use in new code)

### B. Async/Await Patterns

- **MUST**: Use `async`/`await` for all asynchronous operations:
  ```typescript
  // ✅ CORRECT
  async function fetchUser(id: string): Promise<User> {
    const response = await fetch(`/api/users/${id}`);
    if (!response.ok) {
      throw new Error('Failed to fetch user');
    }
    return await response.json();
  }
  
  // ❌ WRONG - using .then() chains
  function fetchUser(id: string): Promise<User> {
    return fetch(`/api/users/${id}`)
      .then(response => {
        if (!response.ok) {
          throw new Error('Failed to fetch user');
        }
        return response.json();
      });
  }
  ```

- **MUST**: Use `try`/`catch` for error handling:
  ```typescript
  async function loadUserData(): Promise<void> {
    try {
      const user = await fetchUser('123');
      userData = user;
    } catch (error) {
      errorMessage = error instanceof Error ? error.message : 'Unknown error';
    }
  }
  ```

- **MUST**: Use `Promise.all()` for parallel operations:
  ```typescript
  // ✅ CORRECT - parallel execution
  async function loadAllData(): Promise<void> {
    const [users, orders, products] = await Promise.all([
      fetchUsers(),
      fetchOrders(),
      fetchProducts()
    ]);
    
    usersData = users;
    ordersData = orders;
    productsData = products;
  }
  
  // ❌ WRONG - sequential execution (slower)
  async function loadAllData(): Promise<void> {
    usersData = await fetchUsers();
    ordersData = await fetchOrders();
    productsData = await fetchProducts();
  }
  ```

- **MUST**: Use `Promise.allSettled()` when some operations may fail:
  ```typescript
  async function loadDataWithFallbacks(): Promise<void> {
    const results = await Promise.allSettled([
      fetchUsers(),
      fetchOrders(),
      fetchProducts()
    ]);
    
    results.forEach((result, index) => {
      if (result.status === 'fulfilled') {
        // Handle success
        dataArrays[index] = result.value;
      } else {
        // Handle failure
        console.error(`Failed to load data ${index}:`, result.reason);
        dataArrays[index] = [];
      }
    });
  }
  ```

### C. Reactive Promises with {#await}

- **MUST**: Use `{#await}` blocks for reactive promises in templates:
  ```svelte
  <script lang="ts">
    let userId = $state('123');
    
    // Reactive promise - updates when userId changes
    let userPromise = $derived(
      (async () => {
        const response = await fetch(`/api/users/${userId}`);
        return await response.json();
      })()
    );
  </script>
  
  {#await userPromise}
    <p>Loading user...</p>
  {:then user}
    <div class="user-card">
      <h3>{user.name}</h3>
      <p>{user.email}</p>
    </div>
  {:catch error}
    <p class="error">Error: {error.message}</p>
  {/await}
  ```

- **SHOULD**: Extract promise creation to separate function for reusability:
  ```typescript
  <script lang="ts">
    let userId = $state('123');
    
    async function fetchUser(id: string): Promise<User> {
      const response = await fetch(`/api/users/${id}`);
      if (!response.ok) throw new Error('User not found');
      return await response.json();
    }
    
    let userPromise = $derived(fetchUser(userId));
  </script>
  
  {#await userPromise}
    <p>Loading...</p>
  {:then user}
    <p>Hello {user.name}!</p>
  {:catch error}
    <p>Error: {error.message}</p>
  {/await}
  ```

### D. Async Effects

- **MUST**: Use async functions inside `$effect` for side effects:
  ```typescript
  <script lang="ts">
    let userId = $state('123');
    let user = $state<User | null>(null);
    let loading = $state(false);
    let error = $state<string | null>(null);
    
    $effect(() => {
      // Create async function inside effect
      (async () => {
        loading = true;
        error = null;
        
        try {
          const response = await fetch(`/api/users/${userId}`);
          if (!response.ok) throw new Error('Failed to fetch');
          user = await response.json();
        } catch (err) {
          error = err instanceof Error ? err.message : 'Unknown error';
        } finally {
          loading = false;
        }
      })();
    });
  </script>
  ```

- **MUST**: Handle cleanup for abortable async operations:
  ```typescript
  $effect(() => {
    const controller = new AbortController();
    
    (async () => {
      try {
        const response = await fetch(`/api/users/${userId}`, {
          signal: controller.signal
        });
        user = await response.json();
      } catch (err) {
        if (err.name !== 'AbortError') {
          error = err instanceof Error ? err.message : 'Unknown error';
        }
      }
    })();
    
    // Cleanup: abort fetch on effect cleanup
    return () => controller.abort();
  });
  ```

### E. Async Iteration

- **SHOULD**: Use `for await...of` for async iterables:
  ```typescript
  async function processStream(stream: AsyncIterable<Data>): Promise<void> {
    for await (const chunk of stream) {
      processChunk(chunk);
    }
  }
  ```

### F. Top-Level Await in SvelteKit

- **MUST**: Use top-level await in load functions:
  ```typescript
  // +page.ts
  import type { PageLoad } from './$types';
  
  export const load: PageLoad = async ({ fetch, params }) => {
    // Top-level await in load function
    const user = await fetchUser(params.id);
    const orders = await fetchOrders(user.id);
    
    return { user, orders };
  };
  ```

### G. Anti-Patterns to Avoid

- **NEVER**: Use nested callbacks (callback hell):
  ```typescript
  // ❌ WRONG
  function loadData() {
    fetchUser((user) => {
      fetchOrders(user.id, (orders) => {
        fetchProducts(orders[0].id, (products) => {
          // Callback hell
        });
      });
    });
  }
  
  // ✅ CORRECT
  async function loadData() {
    const user = await fetchUser();
    const orders = await fetchOrders(user.id);
    const products = await fetchProducts(orders[0].id);
  }
  ```

- **NEVER**: Mix `.then()` with `async`/`await`:
  ```typescript
  // ❌ WRONG - inconsistent style
  async function loadData() {
    const user = await fetchUser();
    return fetchOrders(user.id).then(orders => orders);
  }
  
  // ✅ CORRECT - consistent async/await
  async function loadData() {
    const user = await fetchUser();
    const orders = await fetchOrders(user.id);
    return orders;
  }
  ```

- **NEVER**: Forget error handling in async functions:
  ```typescript
  // ❌ WRONG - no error handling
  async function loadData() {
    const user = await fetchUser(); // May throw
    userData = user;
  }
  
  // ✅ CORRECT - with error handling
  async function loadData() {
    try {
      const user = await fetchUser();
      userData = user;
    } catch (error) {
      console.error('Failed to load user:', error);
      errorMessage = 'Failed to load user data';
    }
  }
  ```

---

## 4. Reactivity (Svelte 5 Runes)

### A. State Management

- **MUST**: Use `$state` for reactive variables:
  ```typescript
  let count = $state(0);
  let user = $state<User | null>(null);
  let items = $state<Item[]>([]);
  ```

- **MUST**: Use `$state.frozen` for immutable data:
  ```typescript
  let config = $state.frozen({ api: 'https://api.example.com' });
  ```

- **MUST**: Update objects/arrays with reassignment for reactivity:
  ```typescript
  // ✅ CORRECT
  items = [...items, newItem];
  user = { ...user, name: 'Updated' };
  
  // ❌ WRONG - mutations don't trigger reactivity
  items.push(newItem);
  user.name = 'Updated';
  ```

- **SHOULD**: Use `$state.snapshot` to get non-reactive snapshot:
  ```typescript
  let reactive = $state({ count: 0 });
  let snapshot = $state.snapshot(reactive); // Plain object
  ```

### B. Derived Values

- **MUST**: Use `$derived` for computed values:
  ```typescript
  let count = $state(0);
  let doubled = $derived(count * 2);
  let isEven = $derived(count % 2 === 0);
  ```

- **MUST**: Use `$derived.by` for complex derivations:
  ```typescript
  let items = $state<Item[]>([]);
  let activeItems = $derived.by(() => {
    return items.filter(item => item.active);
  });
  ```

- **NEVER**: Update state inside `$derived` — it's for reading only:
  ```typescript
  // ❌ WRONG
  let bad = $derived.by(() => {
    count = count + 1; // NEVER mutate in derived
    return count * 2;
  });
  ```

### C. Effects (Side Effects)

- **MUST**: Use `$effect` for side effects (DOM, subscriptions, API calls):
  ```typescript
  let count = $state(0);
  
  $effect(() => {
    console.log(`Count is now ${count}`);
    document.title = `Count: ${count}`;
  });
  ```

- **MUST**: Return cleanup function for subscriptions:
  ```typescript
  $effect(() => {
    const interval = setInterval(() => {
      count++;
    }, 1000);
    
    return () => clearInterval(interval);
  });
  ```

- **MUST**: Use `$effect.pre` for effects that run before DOM updates:
  ```typescript
  $effect.pre(() => {
    console.log('Before DOM update');
  });
  ```

- **MUST**: Use `$effect.root` to create effect root manually (advanced):
  ```typescript
  const cleanup = $effect.root(() => {
    $effect(() => {
      console.log('Effect');
    });
    
    return () => console.log('Root cleanup');
  });
  ```

- **SHOULD**: Use `untrack` to read state without subscribing:
  ```typescript
  $effect(() => {
    const current = untrack(() => count); // Read without tracking
    console.log('Current:', current);
  });
  ```

- **NEVER**: Create infinite loops by updating tracked state:
  ```typescript
  // ❌ WRONG - infinite loop
  $effect(() => {
    count = count + 1; // NEVER do this
  });
  ```

### D. Reactive Statements (Svelte 4 Legacy)

If using Svelte 4 syntax:

- **MUST**: Use `$:` for reactive statements:
  ```typescript
  let count = 0;
  $: doubled = count * 2;
  
  $: {
    console.log(`Count: ${count}`);
  }
  
  $: if (count > 10) {
    console.log('Count is high');
  }
  ```

- **SHOULD**: Group related reactive statements:
  ```typescript
  $: {
    const isValid = count > 0;
    const message = isValid ? 'Valid' : 'Invalid';
    console.log(message);
  }
  ```

---

## 5. Props & Component Interface

### A. Svelte 5 Props

- **MUST**: Use `$props()` for component props:
  ```typescript
  <script lang="ts">
    interface Props {
      title: string;
      count?: number;
      onClose: () => void;
    }
    
    let { title, count = 0, onClose }: Props = $props();
  </script>
  ```

- **MUST**: Destructure props for easy access:
  ```typescript
  let { title, items = [] }: { title: string; items?: Item[] } = $props();
  ```

- **SHOULD**: Use rest props for pass-through:
  ```typescript
  let { title, ...rest }: Props = $props();
  
  <button {...rest}>{title}</button>
  ```

### B. Svelte 4 Props (Legacy)

- **MUST**: Use `export let` for props:
  ```typescript
  export let title: string;
  export let count = 0; // With default
  export let onClose: () => void;
  ```

### C. Children

- **MUST**: Use `$props().children` for slot content (Svelte 5):
  ```typescript
  let { children } = $props();
  
  <div class="wrapper">
    {@render children?.()}
  </div>
  ```

- **MUST**: Use `<slot>` for legacy/named slots:
  ```typescript
  <div class="card">
    <slot name="header" />
    <slot /> <!-- default slot -->
    <slot name="footer" />
  </div>
  ```

### D. Bindings

- **MUST**: Use `bind:` for two-way binding:
  ```typescript
  let value = $state('');
  
  <input bind:value />
  ```

- **MUST**: Bind to component props for two-way data flow:
  ```typescript
  // Parent
  let count = $state(0);
  <Counter bind:count />
  
  // Child
  let { count = $bindable(0) } = $props();
  ```

- **SHOULD**: Use `bind:this` for element/component references:
  ```typescript
  let input: HTMLInputElement;
  
  <input bind:this={input} />
  
  <button onclick={() => input.focus()}>Focus</button>
  ```

---

## 6. TypeScript Integration (MANDATORY)

### A. Configuration

- **MUST**: Use strict TypeScript configuration:
  ```json
  // tsconfig.json
  {
    "extends": "./.svelte-kit/tsconfig.json",
    "compilerOptions": {
      "strict": true,
      "noImplicitAny": true,
      "strictNullChecks": true,
      "strictFunctionTypes": true,
      "noUnusedLocals": true,
      "noUnusedParameters": true,
      "noImplicitReturns": true,
      "esModuleInterop": true,
      "skipLibCheck": true,
      "resolveJsonModule": true
    }
  }
  ```

### B. Component Types

- **MUST**: Define prop interfaces explicitly:
  ```typescript
  <script lang="ts">
    interface User {
      id: string;
      name: string;
      email: string;
    }
    
    interface Props {
      user: User;
      onUpdate: (user: User) => void;
      variant?: 'primary' | 'secondary';
    }
    
    let { user, onUpdate, variant = 'primary' }: Props = $props();
  </script>
  ```

- **MUST**: Type event handlers:
  ```typescript
  function handleClick(event: MouseEvent): void {
    console.log(event.currentTarget);
  }
  
  function handleSubmit(event: SubmitEvent): void {
    event.preventDefault();
    const formData = new FormData(event.currentTarget as HTMLFormElement);
  }
  ```

- **MUST**: Use generic types for reusable components:
  ```typescript
  <script lang="ts" generics="T">
    interface Props<T> {
      items: T[];
      renderItem: (item: T) => string;
    }
    
    let { items, renderItem }: Props<T> = $props();
  </script>
  
  {#each items as item}
    <div>{renderItem(item)}</div>
  {/each}
  ```

### C. Store Types

- **MUST**: Type stores explicitly:
  ```typescript
  import { writable, type Writable } from 'svelte/store';
  
  interface User {
    id: string;
    name: string;
  }
  
  export const userStore: Writable<User | null> = writable(null);
  ```

### D. SvelteKit Types

- **MUST**: Use generated types from SvelteKit:
  ```typescript
  import type { PageLoad } from './$types';
  
  export const load: PageLoad = async ({ params, fetch }) => {
    const response = await fetch(`/api/users/${params.id}`);
    const user = await response.json();
    return { user };
  };
  ```

---

## 7. Control Flow & Templating

### A. Conditionals

- **MUST**: Use `{#if}` for conditionals:
  ```svelte
  {#if count > 0}
    <p>Count is positive</p>
  {:else if count < 0}
    <p>Count is negative</p>
  {:else}
    <p>Count is zero</p>
  {/if}
  ```

- **SHOULD**: Use `{#if}` with guard for type narrowing:
  ```typescript
  <script lang="ts">
    let user = $state<User | null>(null);
  </script>
  
  {#if user}
    <p>{user.name}</p> <!-- TypeScript knows user is not null -->
  {/if}
  ```

### B. Loops

- **MUST**: Use `{#each}` with keyed items:
  ```svelte
  {#each items as item (item.id)}
    <div>{item.name}</div>
  {/each}
  ```

- **MUST**: Include index when needed:
  ```svelte
  {#each items as item, index (item.id)}
    <div>{index}: {item.name}</div>
  {/each}
  ```

- **SHOULD**: Use else block for empty state:
  ```svelte
  {#each items as item (item.id)}
    <div>{item.name}</div>
  {:else}
    <p>No items found</p>
  {/each}
  ```

- **NEVER**: Use `.map()` in templates — use `{#each}`:
  ```svelte
  <!-- ❌ WRONG -->
  {items.map(item => `<div>${item.name}</div>`)}
  
  <!-- ✅ CORRECT -->
  {#each items as item (item.id)}
    <div>{item.name}</div>
  {/each}
  ```

### C. Await Blocks

- **MUST**: Use `{#await}` for promises:
  ```svelte
  <script lang="ts">
    let promise = $state(fetchUser());
  </script>
  
  {#await promise}
    <p>Loading...</p>
  {:then user}
    <p>Hello {user.name}!</p>
  {:catch error}
    <p>Error: {error.message}</p>
  {/await}
  ```

- **SHOULD**: Use short form when you only need success:
  ```svelte
  {#await promise then user}
    <p>Hello {user.name}!</p>
  {/await}
  ```

### D. Snippets (Svelte 5)

- **MUST**: Use `{#snippet}` for reusable template fragments:
  ```svelte
  {#snippet userCard(user: User)}
    <div class="card">
      <h3>{user.name}</h3>
      <p>{user.email}</p>
    </div>
  {/snippet}
  
  {#each users as user (user.id)}
    {@render userCard(user)}
  {/each}
  ```

---

## 8. Stores (Global State)

### A. Writable Stores

- **MUST**: Use `writable` for mutable global state:
  ```typescript
  // stores/user.ts
  import { writable } from 'svelte/store';
  
  export interface User {
    id: string;
    name: string;
    email: string;
  }
  
  export const userStore = writable<User | null>(null);
  
  // Usage in component
  import { userStore } from '$lib/stores/user';
  
  let user = $state<User | null>(null);
  
  $effect(() => {
    const unsubscribe = userStore.subscribe(value => {
      user = value;
    });
    return unsubscribe;
  });
  
  // Or use $ prefix for auto-subscription (Svelte 4 style)
  // $: user = $userStore;
  ```

### B. Readable Stores

- **MUST**: Use `readable` for read-only derived state:
  ```typescript
  import { readable } from 'svelte/store';
  
  export const time = readable(new Date(), (set) => {
    const interval = setInterval(() => {
      set(new Date());
    }, 1000);
    
    return () => clearInterval(interval);
  });
  ```

### C. Derived Stores

- **MUST**: Use `derived` for computed stores:
  ```typescript
  import { derived } from 'svelte/store';
  import { userStore } from './user';
  
  export const isAuthenticated = derived(
    userStore,
    ($user) => $user !== null
  );
  
  // Multiple dependencies
  export const fullName = derived(
    [firstName, lastName],
    ([$firstName, $lastName]) => `${$firstName} ${$lastName}`
  );
  ```

### D. Custom Stores

- **MUST**: Create custom stores with subscribe method:
  ```typescript
  import { writable } from 'svelte/store';
  
  function createCounter() {
    const { subscribe, set, update } = writable(0);
    
    return {
      subscribe,
      increment: () => update(n => n + 1),
      decrement: () => update(n => n - 1),
      reset: () => set(0)
    };
  }
  
  export const counter = createCounter();
  ```

### E. Context API (Component-Scoped State)

- **MUST**: Use context for component tree state:
  ```typescript
  // Parent component
  import { setContext } from 'svelte';
  import { writable } from 'svelte/store';
  
  const theme = writable('dark');
  setContext('theme', theme);
  
  // Child component
  import { getContext } from 'svelte';
  import type { Writable } from 'svelte/store';
  
  const theme = getContext<Writable<string>>('theme');
  ```

---

## 9. Documentation Requirements (MANDATORY)

### A. JSDoc Comments

- **MUST**: Document all public components with JSDoc:
  ```typescript
  <script lang="ts">
    /**
     * User profile card component
     * 
     * Displays user information with optional actions.
     * 
     * @example
     * ```svelte
     * <UserCard
     *   user={currentUser}
     *   onEdit={handleEdit}
     *   variant="compact"
     * />
     * ```
     */
    
    /**
     * User data to display
     */
    export let user: User;
    
    /**
     * Callback when edit button is clicked
     */
    export let onEdit: (user: User) => void;
    
    /**
     * Visual variant of the card
     * @default "default"
     */
    export let variant: 'default' | 'compact' = 'default';
  </script>
  ```

### B. Function Documentation

- **MUST**: Document all exported functions:
  ```typescript
  /**
   * Fetches user data from API
   * 
   * @param userId - The unique identifier of the user
   * @returns Promise resolving to user data
   * @throws {ApiError} When user is not found or API request fails
   * 
   * @example
   * ```typescript
   * const user = await fetchUser('user-123');
   * console.log(user.name);
   * ```
   */
  export async function fetchUser(userId: string): Promise<User> {
    const response = await fetch(`/api/users/${userId}`);
    if (!response.ok) {
      throw new ApiError('Failed to fetch user');
    }
    return response.json();
  }
  ```

### C. Store Documentation

- **MUST**: Document store purpose and usage:
  ```typescript
  /**
   * Global user authentication store
   * 
   * Manages the currently authenticated user state across the application.
   * 
   * @example
   * ```typescript
   * // Subscribe to changes
   * userStore.subscribe(user => {
   *   console.log('Current user:', user);
   * });
   * 
   * // Update user
   * userStore.set(newUser);
   * ```
   */
  export const userStore = writable<User | null>(null);
  ```

### D. TypeDoc Configuration

```json
// typedoc.json
{
  "entryPoints": ["src/lib"],
  "out": "docs/api",
  "plugin": ["typedoc-plugin-markdown"],
  "excludePrivate": true,
  "excludeProtected": true,
  "excludeExternals": true,
  "readme": "none",
  "name": "My Svelte App API Documentation"
}
```

```json
// package.json scripts
{
  "scripts": {
    "docs": "typedoc",
    "docs:watch": "typedoc --watch",
    "docs:serve": "npx http-server docs/api"
  },
  "devDependencies": {
    "typedoc": "^0.25.0",
    "typedoc-plugin-markdown": "^3.17.0"
  }
}
```

---

## 10. Testing (MANDATORY)

### A. Unit Tests with Vitest

```typescript
// src/lib/utils/validation.test.ts
import { describe, it, expect } from 'vitest';
import { validateEmail, validatePassword } from './validation';

describe('validateEmail', () => {
  it('should accept valid email', () => {
    expect(validateEmail('user@example.com')).toBe(true);
  });
  
  it('should reject invalid email', () => {
    expect(validateEmail('invalid')).toBe(false);
  });
  
  it('should reject empty string', () => {
    expect(validateEmail('')).toBe(false);
  });
});

describe('validatePassword', () => {
  it('should accept strong password', () => {
    expect(validatePassword('StrongP@ss123')).toBe(true);
  });
  
  it('should reject weak password', () => {
    expect(validatePassword('weak')).toBe(false);
  });
});
```

### B. Component Tests

```typescript
// src/lib/components/Counter.test.ts
import { render, fireEvent } from '@testing-library/svelte';
import { describe, it, expect } from 'vitest';
import Counter from './Counter.svelte';

describe('Counter', () => {
  it('should render initial count', () => {
    const { getByText } = render(Counter, { props: { initial: 5 } });
    expect(getByText('Count: 5')).toBeInTheDocument();
  });
  
  it('should increment count on button click', async () => {
    const { getByText, getByRole } = render(Counter);
    const button = getByRole('button', { name: /increment/i });
    
    await fireEvent.click(button);
    
    expect(getByText('Count: 1')).toBeInTheDocument();
  });
  
  it('should call onUpdate callback', async () => {
    const onUpdate = vi.fn();
    const { getByRole } = render(Counter, {
      props: { onUpdate }
    });
    
    const button = getByRole('button', { name: /increment/i });
    await fireEvent.click(button);
    
    expect(onUpdate).toHaveBeenCalledWith(1);
  });
});
```

### C. Store Tests

```typescript
// src/lib/stores/counter.test.ts
import { describe, it, expect } from 'vitest';
import { get } from 'svelte/store';
import { counter } from './counter';

describe('counter store', () => {
  it('should start at 0', () => {
    expect(get(counter)).toBe(0);
  });
  
  it('should increment', () => {
    counter.increment();
    expect(get(counter)).toBe(1);
  });
  
  it('should decrement', () => {
    counter.decrement();
    expect(get(counter)).toBe(-1);
  });
  
  it('should reset', () => {
    counter.increment();
    counter.reset();
    expect(get(counter)).toBe(0);
  });
});
```

### D. Test Configuration

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte({ hot: !process.env.VITEST })],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./vitest-setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/setupTests.ts',
        '**/*.d.ts',
        '**/*.config.*',
        '**/mockData/**'
      ]
    }
  },
  resolve: {
    alias: {
      '$lib': '/src/lib'
    }
  }
});
```

```json
// package.json
{
  "scripts": {
    "test": "vitest run",
    "test:watch": "vitest",
    "test:coverage": "vitest run --coverage",
    "test:ui": "vitest --ui"
  },
  "devDependencies": {
    "@testing-library/svelte": "^4.0.0",
    "@testing-library/jest-dom": "^6.1.0",
    "@vitest/ui": "^1.0.0",
    "vitest": "^1.0.0",
    "jsdom": "^23.0.0"
  }
}
```

---

## 11. Hexagonal Architecture (MANDATORY)

### A. Project Structure

```
src/
├── lib/
│   ├── domain/                    # Domain layer (business logic)
│   │   ├── entities/
│   │   │   ├── User.ts
│   │   │   └── Order.ts
│   │   ├── value-objects/
│   │   │   ├── Email.ts
│   │   │   └── Money.ts
│   │   ├── services/
│   │   │   └── OrderService.ts
│   │   └── errors.ts
│   │
│   ├── application/               # Application layer (use cases)
│   │   ├── commands/              # Write operations (CQRS)
│   │   │   ├── CreateUser.ts
│   │   │   └── PlaceOrder.ts
│   │   ├── queries/               # Read operations (CQRS)
│   │   │   ├── GetUser.ts
│   │   │   └── ListOrders.ts
│   │   └── ports/                 # Interfaces
│   │       ├── UserRepository.ts
│   │       └── EmailService.ts
│   │
│   ├── infrastructure/            # Infrastructure layer (adapters)
│   │   ├── api/
│   │   │   ├── UserApiRepository.ts
│   │   │   └── http-client.ts
│   │   ├── storage/
│   │   │   └── LocalStorageRepository.ts
│   │   └── email/
│   │       └── ConsoleEmailService.ts
│   │
│   ├── ui/                        # UI layer (Svelte components)
│   │   ├── components/
│   │   │   ├── UserCard.svelte
│   │   │   ├── OrderList.svelte
│   │   │   └── Button.svelte
│   │   ├── layouts/
│   │   │   └── MainLayout.svelte
│   │   └── stores/
│   │       └── ui-state.ts
│   │
│   └── config/
│       └── settings.ts
│
└── routes/                        # SvelteKit routes
    ├── +layout.svelte
    ├── +page.svelte
    ├── users/
    │   ├── +page.svelte
    │   └── [id]/
    │       └── +page.svelte
    └── api/
        └── users/
            └── +server.ts
```

### B. Domain Layer Example

```typescript
// src/lib/domain/entities/User.ts

/**
 * User domain entity
 * 
 * Represents a user in the system with business rules.
 */
export class User {
  constructor(
    public readonly id: string,
    public readonly name: string,
    public readonly email: Email,
    public readonly createdAt: Date
  ) {}
  
  /**
   * Checks if user can perform admin actions
   */
  isAdmin(): boolean {
    return this.email.domain === 'admin.example.com';
  }
  
  /**
   * Creates a user from plain data
   */
  static fromData(data: UserData): User {
    return new User(
      data.id,
      data.name,
      Email.from(data.email),
      new Date(data.createdAt)
    );
  }
}

interface UserData {
  id: string;
  name: string;
  email: string;
  createdAt: string;
}
```

```typescript
// src/lib/domain/value-objects/Email.ts

/**
 * Email value object
 * 
 * Ensures email validity at the domain level.
 */
export class Email {
  private constructor(public readonly value: string) {}
  
  get domain(): string {
    return this.value.split('@')[1];
  }
  
  static from(value: string): Email {
    if (!this.isValid(value)) {
      throw new Error('Invalid email format');
    }
    return new Email(value);
  }
  
  static isValid(value: string): boolean {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return regex.test(value);
  }
  
  toString(): string {
    return this.value;
  }
}
```

### C. Application Layer (CQRS)

```typescript
// src/lib/application/commands/CreateUser.ts

import type { UserRepository } from '../ports/UserRepository';
import { User } from '$lib/domain/entities/User';
import { Email } from '$lib/domain/value-objects/Email';

/**
 * Command to create a new user
 */
export interface CreateUserCommand {
  name: string;
  email: string;
}

/**
 * Handler for CreateUser command
 */
export class CreateUserHandler {
  constructor(private readonly userRepository: UserRepository) {}
  
  /**
   * Executes the create user command
   * 
   * @param command - The create user command data
   * @returns The created user
   * @throws {Error} If user already exists or validation fails
   */
  async execute(command: CreateUserCommand): Promise<User> {
    // Validate
    const email = Email.from(command.email);
    
    // Check if exists
    const existing = await this.userRepository.findByEmail(email.value);
    if (existing) {
      throw new Error('User already exists');
    }
    
    // Create user
    const user = new User(
      crypto.randomUUID(),
      command.name,
      email,
      new Date()
    );
    
    // Save
    await this.userRepository.save(user);
    
    return user;
  }
}
```

```typescript
// src/lib/application/queries/GetUser.ts

import type { UserRepository } from '../ports/UserRepository';
import type { User } from '$lib/domain/entities/User';

/**
 * Query to get user by ID
 */
export interface GetUserQuery {
  userId: string;
}

/**
 * Handler for GetUser query
 */
export class GetUserHandler {
  constructor(private readonly userRepository: UserRepository) {}
  
  /**
   * Executes the get user query
   * 
   * @param query - The query parameters
   * @returns The user if found
   * @throws {Error} If user not found
   */
  async execute(query: GetUserQuery): Promise<User> {
    const user = await this.userRepository.findById(query.userId);
    
    if (!user) {
      throw new Error('User not found');
    }
    
    return user;
  }
}
```

### D. Infrastructure Layer (Adapters)

```typescript
// src/lib/infrastructure/api/UserApiRepository.ts

import type { UserRepository } from '$lib/application/ports/UserRepository';
import type { User } from '$lib/domain/entities/User';

/**
 * User repository implementation using HTTP API
 */
export class UserApiRepository implements UserRepository {
  constructor(private readonly baseUrl: string) {}
  
  async findById(id: string): Promise<User | null> {
    try {
      const response = await fetch(`${this.baseUrl}/users/${id}`);
      if (!response.ok) {
        if (response.status === 404) return null;
        throw new Error('Failed to fetch user');
      }
      const data = await response.json();
      return User.fromData(data);
    } catch (error) {
      console.error('Error fetching user:', error);
      return null;
    }
  }
  
  async findByEmail(email: string): Promise<User | null> {
    try {
      const response = await fetch(
        `${this.baseUrl}/users?email=${encodeURIComponent(email)}`
      );
      if (!response.ok) return null;
      const data = await response.json();
      return data.length > 0 ? User.fromData(data[0]) : null;
    } catch (error) {
      console.error('Error finding user by email:', error);
      return null;
    }
  }
  
  async save(user: User): Promise<void> {
    const response = await fetch(`${this.baseUrl}/users`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id: user.id,
        name: user.name,
        email: user.email.value,
        createdAt: user.createdAt.toISOString()
      })
    });
    
    if (!response.ok) {
      throw new Error('Failed to save user');
    }
  }
}
```

### E. UI Layer (Svelte Components)

```svelte
<!-- src/lib/ui/components/UserCard.svelte -->
<script lang="ts">
  import type { User } from '$lib/domain/entities/User';
  
  /**
   * User to display
   */
  interface Props {
    user: User;
    onEdit?: (user: User) => void;
  }
  
  let { user, onEdit }: Props = $props();
  
  function handleEdit() {
    onEdit?.(user);
  }
</script>

<div class="user-card">
  <h3>{user.name}</h3>
  <p>{user.email.value}</p>
  {#if user.isAdmin()}
    <span class="badge">Admin</span>
  {/if}
  {#if onEdit}
    <button onclick={handleEdit}>Edit</button>
  {/if}
</div>

<style>
  .user-card {
    border: 1px solid #ddd;
    padding: 1rem;
    border-radius: 8px;
  }
  
  .badge {
    background: #007bff;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.875rem;
  }
</style>
```

### F. Dependency Injection

```typescript
// src/lib/config/container.ts

import { CreateUserHandler } from '$lib/application/commands/CreateUser';
import { GetUserHandler } from '$lib/application/queries/GetUser';
import { UserApiRepository } from '$lib/infrastructure/api/UserApiRepository';

/**
 * Dependency injection container
 */
export class Container {
  private static instance: Container;
  
  private userRepository: UserApiRepository;
  
  private constructor() {
    // Initialize infrastructure
    this.userRepository = new UserApiRepository('https://api.example.com');
  }
  
  static getInstance(): Container {
    if (!Container.instance) {
      Container.instance = new Container();
    }
    return Container.instance;
  }
  
  getCreateUserHandler(): CreateUserHandler {
    return new CreateUserHandler(this.userRepository);
  }
  
  getGetUserHandler(): GetUserHandler {
    return new GetUserHandler(this.userRepository);
  }
}

export const container = Container.getInstance();
```

---

## 12. SvelteKit Specifics

### A. Load Functions

```typescript
// src/routes/users/[id]/+page.ts
import type { PageLoad } from './$types';
import { error } from '@sveltejs/kit';
import { container } from '$lib/config/container';

export const load: PageLoad = async ({ params, fetch }) => {
  try {
    const handler = container.getGetUserHandler();
    const user = await handler.execute({ userId: params.id });
    
    return { user };
  } catch (err) {
    throw error(404, 'User not found');
  }
};
```

### B. Form Actions

```typescript
// src/routes/users/+page.server.ts
import type { Actions } from './$types';
import { fail } from '@sveltejs/kit';
import { container } from '$lib/config/container';

export const actions = {
  createUser: async ({ request }) => {
    const formData = await request.formData();
    const name = formData.get('name') as string;
    const email = formData.get('email') as string;
    
    try {
      const handler = container.getCreateUserHandler();
      const user = await handler.execute({ name, email });
      
      return { success: true, user };
    } catch (error) {
      return fail(400, {
        error: error instanceof Error ? error.message : 'Unknown error',
        name,
        email
      });
    }
  }
} satisfies Actions;
```

### C. API Routes

```typescript
// src/routes/api/users/+server.ts
import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { container } from '$lib/config/container';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { name, email } = await request.json();
    
    const handler = container.getCreateUserHandler();
    const user = await handler.execute({ name, email });
    
    return json(user, { status: 201 });
  } catch (err) {
    throw error(400, err instanceof Error ? err.message : 'Invalid request');
  }
};
```

---

## 13. Deployment Checklist

### Test-Driven Development (TDD) Compliance
- [ ] **Tests written BEFORE implementation**: Red-Green-Refactor cycle followed for all new code
- [ ] **Each test failed first**: Verified tests fail before implementation, pass after
- [ ] **TDD cycle documented**: Commit messages or comments show test-first approach
- [ ] **Bug regression tests added**: Every bug fix has a test that reproduces the bug
- [ ] **Regression tests fail without fix**: Verified bug tests fail before fix, pass after
- [ ] **Bug IDs referenced**: Bug numbers documented in test comments

### Code Quality
- [ ] **TypeScript compilation passes**: `npm run check` succeeds
- [ ] **Build succeeds**: `npm run build` completes without errors
- [ ] **All tests pass**: `npm run test` returns exit code 0
- [ ] **Test coverage > 80%**: `npm run test:coverage` shows adequate coverage
- [ ] **Linting passes**: `npm run lint` succeeds
- [ ] **No console errors**: Application runs without console errors

### Modern Svelte Patterns (MANDATORY)
- [ ] **Svelte 5 runes used**: All components use `$state`, `$derived`, `$effect` (no legacy `$:` syntax)
- [ ] **No mixed syntax**: No mixing of Svelte 4 and Svelte 5 patterns
- [ ] **Async/await preferred**: All async operations use `async`/`await` (no `.then()` chains)
- [ ] **No callbacks**: No nested callbacks or callback hell
- [ ] **`{#await}` blocks used**: Reactive promises use `{#await}` in templates
- [ ] **Error handling**: All async operations have `try`/`catch` or error handling
- [ ] **Parallel operations**: `Promise.all()` used for independent async operations

### Documentation
- [ ] **All public APIs documented**: JSDoc comments on all exported functions/components
- [ ] **Documentation builds**: `npm run docs` succeeds
- [ ] **README.md updated**: Project documentation is current
- [ ] **API docs generated**: `docs/api/` directory exists with TypeDoc output

### Architecture
- [ ] **Hexagonal architecture followed**: Clear domain/application/infrastructure separation
- [ ] **CQRS implemented**: Commands and queries separated
- [ ] **Dependency injection used**: Container provides all handlers
- [ ] **No circular dependencies**: Import graph is acyclic
- [ ] **Components are small**: < 200 lines per component

### Performance
- [ ] **Bundle size optimized**: Check build output size
- [ ] **Code splitting used**: Dynamic imports for large dependencies
- [ ] **Images optimized**: Using appropriate formats and sizes
- [ ] **Lazy loading**: Non-critical components loaded on demand

### Security
- [ ] **No secrets in code**: Environment variables for sensitive data
- [ ] **Input validation**: All user inputs validated
- [ ] **XSS protection**: Proper escaping in templates
- [ ] **CSRF protection**: Using SvelteKit's built-in protection

---

## 14. Common Patterns

### A. Form Handling

```svelte
<script lang="ts">
  import { enhance } from '$app/forms';
  import type { ActionData } from './$types';
  
  interface Props {
    form?: ActionData;
  }
  
  let { form }: Props = $props();
  
  let isSubmitting = $state(false);
</script>

<form
  method="POST"
  action="?/createUser"
  use:enhance={() => {
    isSubmitting = true;
    
    return async ({ result, update }) => {
      await update();
      isSubmitting = false;
    };
  }}
>
  <label>
    Name:
    <input
      type="text"
      name="name"
      value={form?.name ?? ''}
      required
    />
  </label>
  
  <label>
    Email:
    <input
      type="email"
      name="email"
      value={form?.email ?? ''}
      required
    />
  </label>
  
  {#if form?.error}
    <p class="error">{form.error}</p>
  {/if}
  
  <button type="submit" disabled={isSubmitting}>
    {isSubmitting ? 'Creating...' : 'Create User'}
  </button>
</form>
```

### B. Loading States

```svelte
<script lang="ts">
  import { page } from '$app/stores';
  
  let isNavigating = $derived($page.state.loading);
</script>

{#if isNavigating}
  <div class="loading">Loading...</div>
{/if}
```

### C. Error Boundaries

```svelte
<!-- src/routes/+error.svelte -->
<script lang="ts">
  import { page } from '$app/stores';
  
  let status = $derived($page.status);
  let message = $derived($page.error?.message);
</script>

<div class="error-page">
  <h1>{status}</h1>
  <p>{message || 'An error occurred'}</p>
  <a href="/">Go home</a>
</div>
```

---

## Why This Configuration Works

1. **Test-Driven Development (TDD)**: Writing tests before code provides:
   - **Better design**: Tests force you to think about API design upfront
   - **Fewer bugs**: Catches issues before they reach production (40-80% reduction)
   - **Living documentation**: Tests document expected behavior
   - **Fearless refactoring**: Comprehensive tests enable safe code improvements
   - **Faster debugging**: Failing tests pinpoint exact issues immediately
   - **Regression prevention**: Bug tests ensure fixed bugs stay fixed

2. **Svelte's Compiler**: Svelte compiles to optimal vanilla JavaScript, resulting in smaller bundles (up to 70% smaller than React) and faster runtime performance than virtual DOM frameworks.

3. **Runes System (Svelte 5) - Signals Architecture**: Explicit reactivity with `$state`, `$derived`, and `$effect` provides:
   - **Fine-grained reactivity**: Only components using changed signals re-render
   - **Better TypeScript integration**: Explicit types, no inference issues
   - **Compile-time optimization**: Compiler can optimize signal dependencies
   - **Universal pattern**: Aligns with signals in Solid, Angular, Vue Composition API
   - **Explicit dependencies**: No hidden reactivity like `$:` statements
   - **Better debugging**: Clear data flow with explicit state transformations

4. **Async-First with async/await**: Modern async patterns provide:
   - **Readable code**: Sequential async operations are easy to understand
   - **Better error handling**: try/catch works naturally with async/await
   - **Parallel operations**: Promise.all() is clearer than nested .then()
   - **Less cognitive load**: No callback hell or promise chain mental overhead
   - **Better debugging**: Stack traces are preserved with async/await
   - **Native browser support**: All modern browsers support async/await natively

5. **TypeScript Strict Mode**: Catches errors at compile time, provides excellent IDE support, and serves as living documentation.

6. **Hexagonal Architecture**: Clean separation of concerns allows testing domain logic without UI, swapping infrastructure adapters, and independent layer development.

7. **CQRS Pattern**: Separating reads and writes optimizes each for its specific use case and provides clear code organization.

8. **Vitest**: Fast test execution with native ESM support and excellent TypeScript integration.

9. **TypeDoc**: Generated documentation stays in sync with code, reducing documentation drift.

10. **Agent Verification**: Mandatory compilation and test checks prevent broken code from reaching users.

11. **`{#await}` Blocks**: Reactive promise handling in templates eliminates manual loading state management and provides automatic cleanup.

12. **Regression Tests for Bugs**: Every bug gets a test, creating a safety net that prevents regression and documents historical issues.

---

## References

- [Svelte Documentation](https://svelte.dev/docs)
- [Svelte 5 Runes](https://svelte-5-preview.vercel.app/docs/runes)
- [SvelteKit Documentation](https://kit.svelte.dev/docs)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
- [Vitest Documentation](https://vitest.dev/)
- [Testing Library Svelte](https://testing-library.com/docs/svelte-testing-library/intro/)
- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [SolidJS Rules (Inspiration)](https://github.com/aidenybai/solid-rules/blob/main/AGENTS.md)

---

**Last Updated:** 2026-01-18
**Version:** 1.0
**Maintainer:** Development Team
