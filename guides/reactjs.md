# React & TypeScript Single Page Application Guidelines
This document provides mandatory coding standards and development practices for modern React.js single page applications with TypeScript

---
Agent Profile: The React Architect
Role: Senior Frontend Engineer & React Performance Specialist
Objective: Generate production-ready, type-safe, highly performant, and maintainable React SPAs.
Tools: React 19.x, TypeScript 5.x, Vite 5.x, Modern Hooks, TanStack ecosystem.

## 1. Core Philosophies
The agent must adhere to the "REACT-PRO" principles for every React application:

**React-Native Patterns**: Use built-in React features first, minimal external dependencies.
**End-to-End Type Safety**: TypeScript strict mode, no `any`, comprehensive type coverage.
**Accessibility First**: WCAG 2.1 AA compliance, semantic HTML, keyboard navigation.
**Composable Components**: Small, reusable, single-responsibility components.
**Testable Architecture**: Component testing, integration tests, 80%+ coverage, mandatory tests for all components.
**Performance Optimized**: Code splitting, lazy loading, optimized re-renders.
**Reactive State Management**: Modern hooks, server state separation, minimal global state.
**Observable Patterns**: Proper dependency arrays, effect cleanup, no memory leaks.
**Verified Builds**: Agent-generated code MUST compile (TypeScript check) and pass all tests before delivery.

## 2. Mandatory Setup Requirements

### A. Project Initialization
* **Build Tool**: ALWAYS use Vite 5.x+ (NOT Create React App - deprecated).

* **React Version**: Use React 19.x with latest features.

* **TypeScript**: Version 5.4+ with strict mode.

* **Package Manager**: Use `npm` v10+ and not pnpm and yarn.

```bash
# ✅ CORRECT - Modern project setup
npm create vite my-app --template react-ts
cd my-app
npm install

# Update to React 19
npm add react@latest react-dom@latest

# ❌ WRONG - Deprecated
npx create-react-app my-app
```

### B. TypeScript Configuration
```json
// ✅ CORRECT - tsconfig.json for React
{
  "compilerOptions": {
    // Language & Environment
    "target": "ES2022",
    "lib": ["ES2023", "DOM", "DOM.Iterable"],
    "jsx": "react-jsx",
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
    "allowUnusedLabels": false,
    "allowUnreachableCode": false,
    
    // Module Resolution
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "verbatimModuleSyntax": true,
    
    // Emit
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "removeComments": false,
    "noEmit": true,
    
    // Path Mapping
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"],
      "@/components/*": ["./src/components/*"],
      "@/hooks/*": ["./src/hooks/*"],
      "@/utils/*": ["./src/utils/*"],
      "@/types/*": ["./src/types/*"]
    },
    
    // Advanced
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

### C. Project Structure
Standard architecture for scalability:

```
src/
├── assets/              # Static assets (images, fonts)
│   ├── images/
│   └── fonts/
├── components/          # Reusable UI components
│   ├── ui/             # Base UI components (Button, Input, etc.)
│   │   ├── Button/
│   │   │   ├── Button.tsx
│   │   │   ├── Button.test.tsx
│   │   │   ├── Button.module.css
│   │   │   └── index.ts
│   │   └── index.ts
│   ├── layout/         # Layout components (Header, Footer, etc.)
│   │   ├── Header/
│   │   ├── Footer/
│   │   └── Sidebar/
│   └── features/       # Feature-specific components
│       ├── auth/
│       └── dashboard/
├── hooks/              # Custom React hooks
│   ├── useAuth.ts
│   ├── useLocalStorage.ts
│   └── index.ts
├── lib/                # External library configs
│   ├── axios.ts
│   └── queryClient.ts
├── pages/              # Page components (route level)
│   ├── Home/
│   ├── Dashboard/
│   └── NotFound/
├── services/           # API services
│   ├── api/
│   │   ├── auth.api.ts
│   │   └── user.api.ts
│   └── index.ts
├── stores/             # State management (if needed)
│   ├── authStore.ts
│   └── index.ts
├── types/              # TypeScript type definitions
│   ├── api.types.ts
│   ├── models.types.ts
│   └── index.ts
├── utils/              # Utility functions
│   ├── format.ts
│   ├── validation.ts
│   └── index.ts
├── App.tsx             # Root component
├── main.tsx            # Entry point
├── router.tsx          # Route configuration
└── vite-env.d.ts       # Vite type declarations
```

### D. Essential Dependencies
```json
{
  "dependencies": {
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    // Routing
    "react-router-dom": "^6.22.0",
    // Server state management
    "@tanstack/react-query": "^5.20.0",
    // Form handling
    "react-hook-form": "^7.50.0",
    // Validation
    "zod": "^3.22.4",
    // HTTP client
    "axios": "^1.6.5",
    // Date handling
    "date-fns": "^3.3.0"
  },
  "devDependencies": {
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "typescript": "^5.4.0",
    "@vitejs/plugin-react": "^4.2.1",
    "vite": "^5.1.0",
    // Testing
    "vitest": "^1.2.0",
    "@testing-library/react": "^14.1.2",
    "@testing-library/jest-dom": "^6.2.0",
    "@testing-library/user-event": "^14.5.2",
    // Linting & Formatting
    "@biomejs/biome": "^1.5.0",
    // CSS
    "tailwindcss": "^3.4.1",
    "postcss": "^8.4.33",
    "autoprefixer": "^10.4.17"
  }
}
```

## 3. Agent Code Generation Requirements (MANDATORY)

### A. Build Verification Protocol

**CRITICAL: Agents MUST verify that all generated code compiles before presenting it to the user.**

#### Verification Checklist

**Before delivering ANY code, the agent MUST:**

1. **TypeScript Compilation Check**:
   ```bash
   # Run TypeScript compiler
   npm run typecheck
   # OR
   npx tsc --noEmit
   ```
   - **MUST** return exit code 0 (no errors)
   - Address ALL TypeScript errors, not just warnings
   - NO `any` types allowed as workarounds

2. **Linter Check**:
   ```bash
   # Run Biome linter
   npx @biomejs/biome check src/
   ```
   - Fix all errors
   - Address critical warnings

3. **Build Verification**:
   ```bash
   # Verify production build succeeds
   npm run build
   ```
   - MUST complete without errors
   - Check bundle size is reasonable

4. **Unit Test Creation (MANDATORY)**:
   - Write tests for ALL new components
   - Write tests for ALL new hooks
   - Write tests for ALL new utility functions
   - Minimum 80% code coverage
   - Tests MUST follow Testing Library best practices

5. **Test Execution**:
   ```bash
   # Run all tests
   npm test
   
   # Run with coverage
   npm test -- --coverage
   ```
   - **ALL tests MUST pass** (exit code 0)
   - Coverage must be ≥ 80%
   - No skipped or pending tests

#### Error Correction Process

If verification fails:

1. **Identify the error**: Read the full error message and stack trace
2. **Locate the source**: Find the exact file and line number
3. **Fix the root cause**: Don't just suppress warnings
4. **Re-verify**: Run checks again until all pass
5. **Document changes**: Note any significant fixes made

### B. Testing Requirements (MANDATORY)

**EVERY component, hook, and utility function MUST have unit tests.**

#### What Must Be Tested

| Code Type | Required Tests |
|-----------|----------------|
| **Components** | Rendering, props, user interactions, conditional rendering, error states |
| **Custom Hooks** | Return values, state updates, side effects, cleanup functions |
| **Utility Functions** | Input/output for all cases, edge cases, error handling |
| **API Services** | Success responses, error handling, request formatting |
| **Forms** | Validation, submission, error states, field interactions |

#### Test Example Requirements

```typescript
// ✅ CORRECT - Component with comprehensive tests

// src/components/UserCard/UserCard.tsx
interface UserCardProps {
  user: {
    id: string;
    name: string;
    email: string;
  };
  onDelete?: (id: string) => void;
}

export function UserCard({ user, onDelete }: UserCardProps) {
  const [isDeleting, setIsDeleting] = useState(false);
  
  const handleDelete = async () => {
    setIsDeleting(true);
    try {
      await onDelete?.(user.id);
    } finally {
      setIsDeleting(false);
    }
  };
  
  return (
    <div data-testid={`user-card-${user.id}`}>
      <h3>{user.name}</h3>
      <p>{user.email}</p>
      {onDelete && (
        <button 
          onClick={handleDelete} 
          disabled={isDeleting}
          type="button"
        >
          {isDeleting ? 'Deleting...' : 'Delete'}
        </button>
      )}
    </div>
  );
}

// src/components/UserCard/UserCard.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { UserCard } from './UserCard';

describe('UserCard', () => {
  const mockUser = {
    id: '1',
    name: 'John Doe',
    email: 'john@example.com',
  };
  
  it('renders user information', () => {
    render(<UserCard user={mockUser} />);
    
    expect(screen.getByText('John Doe')).toBeInTheDocument();
    expect(screen.getByText('john@example.com')).toBeInTheDocument();
  });
  
  it('does not render delete button when onDelete is not provided', () => {
    render(<UserCard user={mockUser} />);
    
    expect(screen.queryByRole('button', { name: /delete/i })).not.toBeInTheDocument();
  });
  
  it('renders delete button when onDelete is provided', () => {
    render(<UserCard user={mockUser} onDelete={vi.fn()} />);
    
    expect(screen.getByRole('button', { name: 'Delete' })).toBeInTheDocument();
  });
  
  it('calls onDelete with user id when delete button is clicked', async () => {
    const handleDelete = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    
    render(<UserCard user={mockUser} onDelete={handleDelete} />);
    
    await user.click(screen.getByRole('button', { name: 'Delete' }));
    
    expect(handleDelete).toHaveBeenCalledWith('1');
  });
  
  it('shows loading state during deletion', async () => {
    const handleDelete = vi.fn(() => new Promise(resolve => setTimeout(resolve, 100)));
    const user = userEvent.setup();
    
    render(<UserCard user={mockUser} onDelete={handleDelete} />);
    
    const button = screen.getByRole('button', { name: 'Delete' });
    await user.click(button);
    
    expect(screen.getByText('Deleting...')).toBeInTheDocument();
    expect(button).toBeDisabled();
  });
  
  it('handles deletion errors gracefully', async () => {
    const handleDelete = vi.fn().mockRejectedValue(new Error('Delete failed'));
    const user = userEvent.setup();
    
    render(<UserCard user={mockUser} onDelete={handleDelete} />);
    
    await user.click(screen.getByRole('button', { name: 'Delete' }));
    
    // Button should return to normal state even on error
    await vi.waitFor(() => {
      expect(screen.getByRole('button', { name: 'Delete' })).not.toBeDisabled();
    });
  });
});
```

### C. Agent Workflow Example

**Complete agent code generation workflow:**

1. **Generate Component**:
   ```typescript
   // Create UserList.tsx
   export function UserList() { ... }
   ```

2. **Generate Tests**:
   ```typescript
   // Create UserList.test.tsx with comprehensive tests
   describe('UserList', () => { ... });
   ```

3. **Verify TypeScript**:
   ```bash
   npm run typecheck
   # ✓ No errors found
   ```

4. **Run Tests**:
   ```bash
   npm test
   # ✓ All tests passed (10/10)
   # ✓ Coverage: 85%
   ```

5. **Verify Build**:
   ```bash
   npm run build
   # ✓ Build completed successfully
   ```

6. **Present Code**: Only after ALL checks pass

### D. Prohibited Practices

**NEVER deliver code that:**
- ❌ Has TypeScript compilation errors
- ❌ Uses `any` types to bypass type checking
- ❌ Has failing tests
- ❌ Lacks tests for new functionality
- ❌ Has test coverage < 80%
- ❌ Has skipped tests (`it.skip`, `describe.skip`)
- ❌ Fails to build for production
- ❌ Suppresses linter errors without justification
- ❌ Has console.log statements in production code

---

## 4. Component Patterns

### A. Functional Components (ONLY)
* **NEVER use class components**. Use functional components with hooks.

* **ALWAYS define explicit prop types**.

* **USE `React.FC` sparingly** - explicit typing is preferred.

```typescript
// ✅ CORRECT - Modern functional component with explicit types
interface UserCardProps {
  user: {
    id: string;
    name: string;
    email: string;
    avatar?: string;
  };
  onEdit?: (userId: string) => void;
  className?: string;
}

export function UserCard({ user, onEdit, className }: UserCardProps) {
  const handleEdit = () => {
    onEdit?.(user.id);
  };

  return (
    <div className={className}>
      <img src={user.avatar ?? '/default-avatar.png'} alt={`${user.name}'s avatar`} />
      <h3>{user.name}</h3>
      <p>{user.email}</p>
      {onEdit && (
        <button onClick={handleEdit} type="button">
          Edit
        </button>
      )}
    </div>
  );
}

// ❌ WRONG - Class component (outdated)
class UserCard extends React.Component<UserCardProps> {
  render() {
    return <div>...</div>;
  }
}

// ❌ WRONG - No prop types
export function UserCard({ user, onEdit }) {
  return <div>...</div>;
}

// ❌ WRONG - Using any
export function UserCard({ user }: { user: any }) {
  return <div>...</div>;
}
```

### B. Component File Structure
Each component should have its own directory:

```typescript
// components/ui/Button/Button.tsx
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  children: React.ReactNode;
}

export function Button({
  variant = 'primary',
  size = 'md',
  isLoading = false,
  disabled,
  children,
  className,
  ...props
}: ButtonProps) {
  return (
    <button
      type="button"
      disabled={disabled || isLoading}
      className={`btn btn-${variant} btn-${size} ${className ?? ''}`}
      {...props}
    >
      {isLoading ? <Spinner /> : children}
    </button>
  );
}

// components/ui/Button/index.ts
export { Button } from './Button';
export type { ButtonProps } from './Button';
```

### C. Props Patterns
```typescript
// ✅ CORRECT - Extending HTML attributes
interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string;
  error?: string;
  helperText?: string;
}

export function Input({ label, error, helperText, ...props }: InputProps) {
  const id = React.useId();
  
  return (
    <div>
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        aria-invalid={!!error}
        aria-describedby={error ? `${id}-error` : undefined}
        {...props}
      />
      {error && <span id={`${id}-error`} role="alert">{error}</span>}
      {helperText && <span>{helperText}</span>}
    </div>
  );
}

// ✅ CORRECT - Discriminated union for variant props
type ButtonVariantProps =
  | { variant: 'link'; href: string; onClick?: never }
  | { variant: 'button'; href?: never; onClick: () => void };

type ButtonProps = ButtonVariantProps & {
  children: React.ReactNode;
};

export function Button(props: ButtonProps) {
  if (props.variant === 'link') {
    return <a href={props.href}>{props.children}</a>;
  }
  
  return <button onClick={props.onClick} type="button">{props.children}</button>;
}

// ✅ CORRECT - Children typing
interface CardProps {
  children: React.ReactNode;  // Most flexible
  title: string;
}

interface ListProps {
  children: React.ReactElement<ItemProps>[];  // Specific component type
}

interface RenderProps {
  render: (data: User) => React.ReactNode;  // Render prop pattern
}
```

### D. Component Composition
```typescript
// ✅ CORRECT - Compound component pattern
interface CardContextValue {
  variant: 'default' | 'elevated';
}

const CardContext = React.createContext<CardContextValue | undefined>(undefined);

function useCardContext() {
  const context = React.useContext(CardContext);
  if (!context) {
    throw new Error('Card compound components must be used within Card');
  }
  return context;
}

interface CardProps {
  variant?: 'default' | 'elevated';
  children: React.ReactNode;
}

export function Card({ variant = 'default', children }: CardProps) {
  return (
    <CardContext.Provider value={{ variant }}>
      <div className={`card card-${variant}`}>
        {children}
      </div>
    </CardContext.Provider>
  );
}

Card.Header = function CardHeader({ children }: { children: React.ReactNode }) {
  const { variant } = useCardContext();
  return <div className={`card-header card-header-${variant}`}>{children}</div>;
};

Card.Body = function CardBody({ children }: { children: React.ReactNode }) {
  return <div className="card-body">{children}</div>;
};

Card.Footer = function CardFooter({ children }: { children: React.ReactNode }) {
  return <div className="card-footer">{children}</div>;
};

// Usage
<Card variant="elevated">
  <Card.Header>Title</Card.Header>
  <Card.Body>Content</Card.Body>
  <Card.Footer>Actions</Card.Footer>
</Card>
```

## 5. Modern Hooks Patterns

### A. useState Best Practices
```typescript
// ✅ CORRECT - Typed state
interface User {
  id: string;
  name: string;
  email: string;
}

function UserProfile() {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  
  // Functional updates for state based on previous state
  const increment = () => {
    setCount((prev) => prev + 1);
  };
  
  return <div>...</div>;
}

// ✅ CORRECT - Complex state with useReducer
type State = {
  status: 'idle' | 'loading' | 'success' | 'error';
  data: User | null;
  error: Error | null;
};

type Action =
  | { type: 'FETCH_START' }
  | { type: 'FETCH_SUCCESS'; payload: User }
  | { type: 'FETCH_ERROR'; payload: Error }
  | { type: 'RESET' };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'FETCH_START':
      return { ...state, status: 'loading', error: null };
    case 'FETCH_SUCCESS':
      return { status: 'success', data: action.payload, error: null };
    case 'FETCH_ERROR':
      return { ...state, status: 'error', error: action.payload };
    case 'RESET':
      return { status: 'idle', data: null, error: null };
    default:
      return state;
  }
}

function UserProfile() {
  const [state, dispatch] = useReducer(reducer, {
    status: 'idle',
    data: null,
    error: null,
  });
  
  return <div>...</div>;
}
```

### B. useEffect Patterns
```typescript
// ✅ CORRECT - Proper effect dependencies and cleanup
function UserProfile({ userId }: { userId: string }) {
  const [user, setUser] = useState<User | null>(null);
  
  useEffect(() => {
    let cancelled = false;
    
    async function fetchUser() {
      try {
        const data = await api.getUser(userId);
        if (!cancelled) {
          setUser(data);
        }
      } catch (error) {
        if (!cancelled) {
          console.error('Failed to fetch user', error);
        }
      }
    }
    
    fetchUser();
    
    // Cleanup function
    return () => {
      cancelled = true;
    };
  }, [userId]); // Complete dependency array
  
  return <div>...</div>;
}

// ✅ CORRECT - Event listener cleanup
function WindowSize() {
  const [size, setSize] = useState({ width: 0, height: 0 });
  
  useEffect(() => {
    function handleResize() {
      setSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    }
    
    handleResize(); // Initial call
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []); // Empty array - run once
  
  return <div>...</div>;
}

// ❌ WRONG - Missing dependencies
useEffect(() => {
  fetchUser(userId); // userId not in dependency array
}, []);

// ❌ WRONG - No cleanup
useEffect(() => {
  const interval = setInterval(() => {
    // Do something
  }, 1000);
  // Missing: return () => clearInterval(interval);
}, []);
```

### C. Custom Hooks
```typescript
// ✅ CORRECT - Custom hook with proper typing
interface UseAsyncResult<T> {
  data: T | null;
  error: Error | null;
  isLoading: boolean;
  execute: () => Promise<void>;
}

function useAsync<T>(
  asyncFunction: () => Promise<T>,
  immediate = true,
): UseAsyncResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const execute = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await asyncFunction();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error'));
    } finally {
      setIsLoading(false);
    }
  }, [asyncFunction]);
  
  useEffect(() => {
    if (immediate) {
      execute();
    }
  }, [execute, immediate]);
  
  return { data, error, isLoading, execute };
}

// Usage
function UserList() {
  const { data, error, isLoading } = useAsync(() => api.getUsers(), true);
  
  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  if (!data) return null;
  
  return <div>{/* Render users */}</div>;
}

// ✅ CORRECT - useLocalStorage hook
function useLocalStorage<T>(
  key: string,
  initialValue: T,
): [T, (value: T | ((val: T) => T)) => void] {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? (JSON.parse(item) as T) : initialValue;
    } catch (error) {
      console.error(`Error loading ${key} from localStorage`, error);
      return initialValue;
    }
  });
  
  const setValue = (value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(`Error saving ${key} to localStorage`, error);
    }
  };
  
  return [storedValue, setValue];
}

// ✅ CORRECT - useDebounce hook
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value);
  
  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);
    
    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);
  
  return debouncedValue;
}
```

### D. React 19 New Hooks
```typescript
// ✅ CORRECT - use() hook for async data
import { use } from 'react';

function UserProfile({ userPromise }: { userPromise: Promise<User> }) {
  const user = use(userPromise);
  
  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  );
}

// ✅ CORRECT - useOptimistic for optimistic UI updates
import { useOptimistic } from 'react';

function TodoList({ todos }: { todos: Todo[] }) {
  const [optimisticTodos, addOptimisticTodo] = useOptimistic(
    todos,
    (state, newTodo: Todo) => [...state, newTodo],
  );
  
  async function handleAddTodo(formData: FormData) {
    const title = formData.get('title') as string;
    const optimisticTodo = { id: crypto.randomUUID(), title, completed: false };
    
    addOptimisticTodo(optimisticTodo);
    await api.createTodo(optimisticTodo);
  }
  
  return <div>{/* Render optimisticTodos */}</div>;
}

// ✅ CORRECT - useTransition for non-blocking updates
import { useTransition } from 'react';

function SearchResults() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Result[]>([]);
  const [isPending, startTransition] = useTransition();
  
  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const value = e.target.value;
    setQuery(value); // Urgent update
    
    startTransition(() => {
      // Non-urgent update
      setResults(filterResults(value));
    });
  }
  
  return (
    <div>
      <input value={query} onChange={handleChange} />
      {isPending && <Spinner />}
      <ResultsList results={results} />
    </div>
  );
}
```

## 6. State Management

### A. Server State (TanStack Query)
* **USE TanStack Query** for all server state management.

* **SEPARATE server state from client state**.

```typescript
// ✅ CORRECT - TanStack Query setup
import { QueryClient, QueryClientProvider, useQuery, useMutation } from '@tanstack/react-query';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

// lib/queryClient.ts
export { queryClient };

// API service
async function fetchUsers(): Promise<User[]> {
  const response = await fetch('/api/users');
  if (!response.ok) throw new Error('Failed to fetch users');
  return response.json();
}

// Custom hook for data fetching
function useUsers() {
  return useQuery({
    queryKey: ['users'],
    queryFn: fetchUsers,
  });
}

// Component
function UserList() {
  const { data: users, isLoading, error } = useUsers();
  
  if (isLoading) return <Spinner />;
  if (error) return <ErrorMessage error={error} />;
  if (!users) return null;
  
  return (
    <ul>
      {users.map((user) => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}

// ✅ CORRECT - Mutations with optimistic updates
function useUpdateUser() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (user: User) => api.updateUser(user),
    onMutate: async (updatedUser) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: ['users', updatedUser.id] });
      
      // Snapshot previous value
      const previousUser = queryClient.getQueryData(['users', updatedUser.id]);
      
      // Optimistically update
      queryClient.setQueryData(['users', updatedUser.id], updatedUser);
      
      return { previousUser };
    },
    onError: (err, updatedUser, context) => {
      // Rollback on error
      queryClient.setQueryData(
        ['users', updatedUser.id],
        context?.previousUser,
      );
    },
    onSettled: (updatedUser) => {
      // Refetch after error or success
      queryClient.invalidateQueries({ queryKey: ['users', updatedUser?.id] });
    },
  });
}
```

### B. Client State (Zustand - when needed)
* **MINIMIZE global state**. Prefer component state and server state.

* **USE Zustand** for client state when needed (simpler than Redux).

```typescript
// ✅ CORRECT - Zustand store with TypeScript
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  setUser: (user: User) => void;
}

export const useAuthStore = create<AuthState>()(
  devtools(
    persist(
      (set) => ({
        user: null,
        token: null,
        isAuthenticated: false,
        login: async (email, password) => {
          const response = await api.login(email, password);
          set({
            user: response.user,
            token: response.token,
            isAuthenticated: true,
          });
        },
        logout: () => {
          set({ user: null, token: null, isAuthenticated: false });
        },
        setUser: (user) => set({ user }),
      }),
      {
        name: 'auth-storage',
        partialize: (state) => ({ token: state.token }), // Only persist token
      },
    ),
  ),
);

// Usage in component
function Profile() {
  const user = useAuthStore((state) => state.user);
  const logout = useAuthStore((state) => state.logout);
  
  if (!user) return <div>Not authenticated</div>;
  
  return (
    <div>
      <h1>{user.name}</h1>
      <button onClick={logout} type="button">Logout</button>
    </div>
  );
}
```

### C. Form State (React Hook Form)
```typescript
// ✅ CORRECT - React Hook Form with Zod validation
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

const userSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(8, 'Password must be at least 8 characters'),
  name: z.string().min(1, 'Name is required').max(100),
  age: z.number().int().positive().optional(),
});

type UserFormData = z.infer<typeof userSchema>;

function UserForm() {
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
    reset,
  } = useForm<UserFormData>({
    resolver: zodResolver(userSchema),
    defaultValues: {
      email: '',
      password: '',
      name: '',
    },
  });
  
  const onSubmit = async (data: UserFormData) => {
    try {
      await api.createUser(data);
      reset();
    } catch (error) {
      console.error('Failed to create user', error);
    }
  };
  
  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div>
        <label htmlFor="email">Email</label>
        <input
          id="email"
          type="email"
          {...register('email')}
          aria-invalid={!!errors.email}
        />
        {errors.email && <span role="alert">{errors.email.message}</span>}
      </div>
      
      <div>
        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          {...register('password')}
          aria-invalid={!!errors.password}
        />
        {errors.password && <span role="alert">{errors.password.message}</span>}
      </div>
      
      <div>
        <label htmlFor="name">Name</label>
        <input
          id="name"
          {...register('name')}
          aria-invalid={!!errors.name}
        />
        {errors.name && <span role="alert">{errors.name.message}</span>}
      </div>
      
      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? 'Submitting...' : 'Submit'}
      </button>
    </form>
  );
}
```

## 7. Routing (React Router v6)

```typescript
// ✅ CORRECT - Type-safe routing with React Router v6
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import { lazy, Suspense } from 'react';

// Lazy-loaded pages
const Home = lazy(() => import('@/pages/Home'));
const Dashboard = lazy(() => import('@/pages/Dashboard'));
const UserProfile = lazy(() => import('@/pages/UserProfile'));
const NotFound = lazy(() => import('@/pages/NotFound'));

const router = createBrowserRouter([
  {
    path: '/',
    element: <RootLayout />,
    errorElement: <ErrorBoundary />,
    children: [
      {
        index: true,
        element: (
          <Suspense fallback={<PageLoader />}>
            <Home />
          </Suspense>
        ),
      },
      {
        path: 'dashboard',
        element: (
          <ProtectedRoute>
            <Suspense fallback={<PageLoader />}>
              <Dashboard />
            </Suspense>
          </ProtectedRoute>
        ),
      },
      {
        path: 'users/:userId',
        element: (
          <Suspense fallback={<PageLoader />}>
            <UserProfile />
          </Suspense>
        ),
        loader: async ({ params }) => {
          // Data loading before component renders
          const user = await api.getUser(params.userId!);
          return { user };
        },
      },
    ],
  },
  {
    path: '*',
    element: <NotFound />,
  },
]);

function App() {
  return <RouterProvider router={router} />;
}

// Protected route wrapper
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
  
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  
  return <>{children}</>;
}

// Using route params
function UserProfile() {
  const { userId } = useParams<{ userId: string }>();
  const { data: user } = useQuery({
    queryKey: ['users', userId],
    queryFn: () => api.getUser(userId!),
    enabled: !!userId,
  });
  
  if (!user) return <div>Loading...</div>;
  
  return <div>{user.name}</div>;
}

// Programmatic navigation
function LoginForm() {
  const navigate = useNavigate();
  
  async function handleLogin(credentials: Credentials) {
    await api.login(credentials);
    navigate('/dashboard', { replace: true });
  }
  
  return <form onSubmit={handleLogin}>...</form>;
}
```

## 8. Performance Optimization

### A. Code Splitting & Lazy Loading
```typescript
// ✅ CORRECT - Component lazy loading
import { lazy, Suspense } from 'react';

const HeavyComponent = lazy(() => import('./HeavyComponent'));
const AdminPanel = lazy(() => import('./AdminPanel'));

function App() {
  const [showAdmin, setShowAdmin] = useState(false);
  
  return (
    <div>
      <button onClick={() => setShowAdmin(true)} type="button">
        Show Admin
      </button>
      
      {showAdmin && (
        <Suspense fallback={<Spinner />}>
          <AdminPanel />
        </Suspense>
      )}
    </div>
  );
}

// ✅ CORRECT - Route-based code splitting
const router = createBrowserRouter([
  {
    path: '/admin',
    element: (
      <Suspense fallback={<PageLoader />}>
        <AdminPanel />
      </Suspense>
    ),
  },
]);
```

### B. Memoization
```typescript
// ✅ CORRECT - useMemo for expensive computations
function ExpensiveList({ items }: { items: Item[] }) {
  const sortedAndFilteredItems = useMemo(() => {
    return items
      .filter((item) => item.isActive)
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [items]);
  
  return (
    <ul>
      {sortedAndFilteredItems.map((item) => (
        <li key={item.id}>{item.name}</li>
      ))}
    </ul>
  );
}

// ✅ CORRECT - useCallback for function memoization
function Parent() {
  const [count, setCount] = useState(0);
  
  const handleClick = useCallback(() => {
    console.log('Clicked');
  }, []); // Function won't change
  
  return <Child onClick={handleClick} />;
}

// ✅ CORRECT - React.memo for component memoization
interface ChildProps {
  name: string;
  age: number;
  onClick: () => void;
}

const Child = React.memo(function Child({ name, age, onClick }: ChildProps) {
  console.log('Child rendered');
  return (
    <div>
      <p>{name} - {age}</p>
      <button onClick={onClick} type="button">Click</button>
    </div>
  );
});

// ❌ WRONG - Overusing memo (premature optimization)
const SimpleComponent = React.memo(function Simple({ text }: { text: string }) {
  return <div>{text}</div>; // Too simple to benefit from memo
});
```

### C. Virtual Lists for Large Data
```typescript
// ✅ CORRECT - Virtual scrolling with @tanstack/react-virtual
import { useVirtualizer } from '@tanstack/react-virtual';

function VirtualList({ items }: { items: Item[] }) {
  const parentRef = useRef<HTMLDivElement>(null);
  
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 50, // Estimated row height
    overscan: 5, // Render 5 items outside viewport
  });
  
  return (
    <div
      ref={parentRef}
      style={{ height: '500px', overflow: 'auto' }}
    >
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          position: 'relative',
        }}
      >
        {virtualizer.getVirtualItems().map((virtualRow) => {
          const item = items[virtualRow.index];
          return (
            <div
              key={item.id}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: `${virtualRow.size}px`,
                transform: `translateY(${virtualRow.start}px)`,
              }}
            >
              {item.name}
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

## 9. Accessibility (A11y)

### A. Semantic HTML
```typescript
// ✅ CORRECT - Semantic HTML and ARIA
function Navigation() {
  return (
    <nav aria-label="Main navigation">
      <ul>
        <li><a href="/">Home</a></li>
        <li><a href="/about">About</a></li>
        <li><a href="/contact">Contact</a></li>
      </ul>
    </nav>
  );
}

function ArticleCard({ article }: { article: Article }) {
  return (
    <article>
      <header>
        <h2>{article.title}</h2>
        <time dateTime={article.publishedAt}>
          {formatDate(article.publishedAt)}
        </time>
      </header>
      <p>{article.excerpt}</p>
      <footer>
        <a href={`/articles/${article.id}`}>Read more</a>
      </footer>
    </article>
  );
}

// ❌ WRONG - Divitis
function Navigation() {
  return (
    <div className="navigation">
      <div onClick={() => navigate('/')}>Home</div>
      <div onClick={() => navigate('/about')}>About</div>
    </div>
  );
}
```

### B. Keyboard Navigation & Focus Management
```typescript
// ✅ CORRECT - Keyboard accessible modal
function Modal({ isOpen, onClose, children }: ModalProps) {
  const modalRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (!isOpen) return;
    
    const previousActiveElement = document.activeElement as HTMLElement;
    
    // Focus trap
    const focusableElements = modalRef.current?.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
    );
    
    const firstElement = focusableElements?.[0] as HTMLElement;
    const lastElement = focusableElements?.[focusableElements.length - 1] as HTMLElement;
    
    firstElement?.focus();
    
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        onClose();
      }
      
      if (e.key === 'Tab') {
        if (e.shiftKey && document.activeElement === firstElement) {
          e.preventDefault();
          lastElement?.focus();
        } else if (!e.shiftKey && document.activeElement === lastElement) {
          e.preventDefault();
          firstElement?.focus();
        }
      }
    }
    
    document.addEventListener('keydown', handleKeyDown);
    
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      previousActiveElement?.focus();
    };
  }, [isOpen, onClose]);
  
  if (!isOpen) return null;
  
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
      ref={modalRef}
    >
      <h2 id="modal-title">Modal Title</h2>
      {children}
      <button onClick={onClose} type="button">Close</button>
    </div>
  );
}

// ✅ CORRECT - Skip to main content
function Layout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>
      <Header />
      <main id="main-content" tabIndex={-1}>
        {children}
      </main>
      <Footer />
    </>
  );
}
```

### C. ARIA Labels & Live Regions
```typescript
// ✅ CORRECT - ARIA live regions for dynamic content
function Notifications() {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  
  return (
    <div
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      {notifications.map((notification) => (
        <div key={notification.id}>{notification.message}</div>
      ))}
    </div>
  );
}

// ✅ CORRECT - Loading states
function DataTable() {
  const { data, isLoading } = useQuery({ queryKey: ['data'], queryFn: fetchData });
  
  if (isLoading) {
    return (
      <div role="status" aria-live="polite">
        <Spinner aria-hidden="true" />
        <span className="sr-only">Loading data...</span>
      </div>
    );
  }
  
  return <table aria-label="Data table">...</table>;
}
```

## 10. Testing

### A. Component Testing
```typescript
// ✅ CORRECT - Vitest + Testing Library
import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Button } from './Button';

describe('Button', () => {
  it('renders with children', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByRole('button', { name: 'Click me' })).toBeInTheDocument();
  });
  
  it('calls onClick when clicked', async () => {
    const handleClick = vi.fn();
    const user = userEvent.setup();
    
    render(<Button onClick={handleClick}>Click me</Button>);
    
    await user.click(screen.getByRole('button'));
    
    expect(handleClick).toHaveBeenCalledTimes(1);
  });
  
  it('shows loading state', () => {
    render(<Button isLoading>Submit</Button>);
    
    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
    expect(screen.getByRole('status')).toBeInTheDocument();
  });
  
  it('is disabled when disabled prop is true', () => {
    render(<Button disabled>Submit</Button>);
    expect(screen.getByRole('button')).toBeDisabled();
  });
});

// ✅ CORRECT - Testing async components
describe('UserProfile', () => {
  it('loads and displays user data', async () => {
    const mockUser = { id: '1', name: 'John Doe', email: 'john@example.com' };
    vi.spyOn(api, 'getUser').mockResolvedValue(mockUser);
    
    render(<UserProfile userId="1" />);
    
    expect(screen.getByText('Loading...')).toBeInTheDocument();
    
    await waitFor(() => {
      expect(screen.getByText('John Doe')).toBeInTheDocument();
    });
    
    expect(api.getUser).toHaveBeenCalledWith('1');
  });
  
  it('displays error message on failure', async () => {
    vi.spyOn(api, 'getUser').mockRejectedValue(new Error('Failed to load'));
    
    render(<UserProfile userId="1" />);
    
    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument();
    });
  });
});

// ✅ CORRECT - Testing forms
describe('LoginForm', () => {
  it('submits form with valid data', async () => {
    const handleSubmit = vi.fn();
    const user = userEvent.setup();
    
    render(<LoginForm onSubmit={handleSubmit} />);
    
    await user.type(screen.getByLabelText('Email'), 'test@example.com');
    await user.type(screen.getByLabelText('Password'), 'password123');
    await user.click(screen.getByRole('button', { name: 'Login' }));
    
    await waitFor(() => {
      expect(handleSubmit).toHaveBeenCalledWith({
        email: 'test@example.com',
        password: 'password123',
      });
    });
  });
  
  it('shows validation errors', async () => {
    const user = userEvent.setup();
    
    render(<LoginForm onSubmit={vi.fn()} />);
    
    await user.click(screen.getByRole('button', { name: 'Login' }));
    
    expect(await screen.findByText(/email is required/i)).toBeInTheDocument();
    expect(await screen.findByText(/password is required/i)).toBeInTheDocument();
  });
});
```

### B. Integration Testing
```typescript
// ✅ CORRECT - Integration test with providers
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        {ui}
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('UserDashboard Integration', () => {
  it('fetches and displays user data with posts', async () => {
    const mockUser = { id: '1', name: 'John' };
    const mockPosts = [{ id: '1', title: 'Post 1' }];
    
    vi.spyOn(api, 'getUser').mockResolvedValue(mockUser);
    vi.spyOn(api, 'getUserPosts').mockResolvedValue(mockPosts);
    
    renderWithProviders(<UserDashboard userId="1" />);
    
    await waitFor(() => {
      expect(screen.getByText('John')).toBeInTheDocument();
      expect(screen.getByText('Post 1')).toBeInTheDocument();
    });
  });
});
```

## 11. Security Best Practices

### A. XSS Prevention
```typescript
// ✅ CORRECT - React automatically escapes
function UserComment({ comment }: { comment: string }) {
  return <p>{comment}</p>; // Safe - React escapes by default
}

// ✅ CORRECT - Using DOMPurify for user HTML
import DOMPurify from 'dompurify';

function RichTextDisplay({ html }: { html: string }) {
  const sanitized = useMemo(() => DOMPurify.sanitize(html), [html]);
  
  return <div dangerouslySetInnerHTML={{ __html: sanitized }} />;
}

// ❌ WRONG - Unescaped user input
function Unsafe({ html }: { html: string }) {
  return <div dangerouslySetInnerHTML={{ __html: html }} />; // XSS vulnerability
}
```

### B. Authentication
```typescript
// ✅ CORRECT - Secure token handling
import { jwtDecode } from 'jwt-decode';

interface TokenPayload {
  userId: string;
  exp: number;
}

function isTokenValid(token: string): boolean {
  try {
    const decoded = jwtDecode<TokenPayload>(token);
    return decoded.exp * 1000 > Date.now();
  } catch {
    return false;
  }
}

// Axios interceptor for auth
axios.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token && isTokenValid(token)) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

axios.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  },
);
```

### C. Environment Variables
```typescript
// ✅ CORRECT - Type-safe environment variables
import { z } from 'zod';

const envSchema = z.object({
  VITE_API_URL: z.string().url(),
  VITE_APP_NAME: z.string(),
  VITE_ENABLE_ANALYTICS: z.enum(['true', 'false']).transform((v) => v === 'true'),
});

export const env = envSchema.parse(import.meta.env);

// Usage
console.log(env.VITE_API_URL); // Type-safe, validated

// ❌ WRONG - Direct access without validation
const apiUrl = import.meta.env.VITE_API_URL; // Could be undefined, no validation
```

## 12. Build Configuration

### A. Vite Configuration
```typescript
// ✅ CORRECT - vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

export default defineConfig({
  plugins: [
    react({
      // Enable Fast Refresh
      fastRefresh: true,
    }),
  ],
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
      '@/components': resolve(__dirname, './src/components'),
      '@/hooks': resolve(__dirname, './src/hooks'),
      '@/utils': resolve(__dirname, './src/utils'),
      '@/types': resolve(__dirname, './src/types'),
    },
  },
  build: {
    target: 'esnext',
    minify: 'terser',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'query-vendor': ['@tanstack/react-query'],
        },
      },
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
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
        '**/*.test.{ts,tsx}',
        '**/*.config.{ts,js}',
      ],
    },
  },
});
```

## 13. Complete Production Example

```typescript
// src/main.tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import App from './App';
import './index.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5,
      retry: 1,
    },
  },
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  </React.StrictMode>,
);

// src/App.tsx
import { RouterProvider } from 'react-router-dom';
import { router } from './router';
import { ErrorBoundary } from './components/ErrorBoundary';

function App() {
  return (
    <ErrorBoundary>
      <RouterProvider router={router} />
    </ErrorBoundary>
  );
}

export default App;

// src/components/ErrorBoundary.tsx
import React from 'react';

interface Props {
  children: React.ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div role="alert">
          <h1>Something went wrong</h1>
          <p>{this.state.error?.message}</p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            type="button"
          >
            Try again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
```

## 14. Deployment Checklist

### Agent Code Generation Verification (MANDATORY)
**If code was generated by an agent, verify BEFORE delivery:**
- [ ] TypeScript compilation successful: `npm run typecheck` returns exit code 0
- [ ] All linter checks pass: `npx @biomejs/biome check src/`
- [ ] Production build succeeds: `npm run build` completes without errors
- [ ] Unit tests created for ALL new components, hooks, and utilities
- [ ] All tests passing: `npm test` returns exit code 0
- [ ] Test coverage ≥ 80%: `npm test -- --coverage`
- [ ] No `any` types used as workarounds
- [ ] No skipped tests (`it.skip`, `describe.skip`)
- [ ] No console.log statements in production code
- [ ] Agent has documented any complex fixes made during verification

### Pre-Production Validation
- [ ] All TypeScript errors resolved (`npm typecheck`)
- [ ] All tests passing (`npm test`)
- [ ] Test coverage ≥ 80%
- [ ] No console.log statements in production code
- [ ] All images optimized and lazy-loaded
- [ ] Code splitting implemented for routes
- [ ] Bundle size analyzed and optimized
- [ ] Lighthouse score: Performance ≥ 90, Accessibility ≥ 95
- [ ] WCAG 2.1 AA compliance verified
- [ ] All forms have proper validation
- [ ] Error boundaries implemented
- [ ] Loading states for all async operations
- [ ] SEO meta tags configured
- [ ] Environment variables validated
- [ ] Security headers configured
- [ ] CSP (Content Security Policy) configured
- [ ] Analytics tracking implemented
- [ ] Error monitoring configured (Sentry, etc.)

### Performance Targets
- First Contentful Paint (FCP): < 1.8s
- Largest Contentful Paint (LCP): < 2.5s
- Time to Interactive (TTI): < 3.8s
- Cumulative Layout Shift (CLS): < 0.1
- Bundle size (gzipped): < 200KB (initial)

---

## Why This Configuration Works

1. **Vite over CRA**: 10-20x faster builds, native ESM, better DX.

2. **React 19**: Latest features (use(), useOptimistic, improved Suspense).

3. **TypeScript Strict**: Catches 15-30% more bugs at compile time.

4. **Agent Build Verification**: Ensures generated code compiles and tests pass before delivery, preventing broken code.

5. **Mandatory Testing**: 80%+ coverage requirement catches bugs early, reduces production issues.

6. **TanStack Query**: Declarative data fetching, automatic caching, optimistic updates.

7. **React Hook Form + Zod**: Best form performance, type-safe validation.

8. **Zustand**: Minimal boilerplate vs Redux, better TypeScript support.

9. **Component Composition**: Reusability, testability, maintainability.

10. **Testing Library**: Tests user behavior, not implementation details.

11. **Accessibility First**: Legal compliance, better UX for all users.

12. **Performance Optimization**: Lazy loading, memoization, virtual lists for scale.

---

## References

- [React Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Vite Guide](https://vitejs.dev/guide/)
- [TanStack Query](https://tanstack.com/query/latest)
- [React Router](https://reactrouter.com/)
- [Testing Library](https://testing-library.com/react)
- [Web Content Accessibility Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [React TypeScript Cheatsheet](https://react-typescript-cheatsheet.netlify.app/)
