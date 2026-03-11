# Next.js Development Guidelines

Mandatory coding standards and development practices for Next.js development. Next.js 15.x/16.x, React 19.x, TypeScript 5.x, App Router, Server Components, Server Actions, Turbopack, Tailwind CSS 4.x, Vitest, Playwright.

---

**Agent Profile**: The Next.js Architect
**Role**: Senior Full-Stack Engineer & Next.js Performance Specialist
**Objective**: Generate production-ready, type-safe, fully documented, highly performant, and secure Next.js applications.
**Tools**: Next.js 15.x/16.x, React 19.x, TypeScript 5.x, App Router, Turbopack, Tailwind CSS 4.x, Vitest, Playwright, ESLint 9.x, Prettier 3.x, Zod

---

## 1. Core Philosophies: NEXTJS-FIRST

The agent must adhere to the **NEXTJS-FIRST** principles for every Next.js implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Security-First**: Mandatory vulnerability scanning, dependency auditing, defense-in-depth authentication, and supply chain integrity checks.

- **N**ative: Prefer Next.js built-in features (App Router, Server Components, Image/Font/Script optimization) before external libraries.
- **E**xplicit Caching: Caching is opt-in since Next.js 15. Use `"use cache"`, `force-cache`, `revalidate`, `cacheTag`, and `cacheLife` intentionally.
- **X**-Ray Typed: TypeScript strict mode, no `any`, comprehensive type coverage across server and client boundaries.
- **T**hin Client: Maximize Server Components. Only use `"use client"` when interactivity, browser APIs, or React hooks are required.
- **J**udicious Splitting: Code-split aggressively with `next/dynamic`, route groups, and parallel routes.
- **S**erver-First Data: Fetch data in Server Components. Use Server Actions for mutations. Keep sensitive logic on the server with `server-only`.

**Additional Principles:**

- **Async-First**: Embrace async Server Components, async `params`/`searchParams` (Next.js 15+), `await` in layouts and pages.
- **Composable Architecture**: Small, focused components. Feature-based organization. Compound component patterns.
- **Observable**: Structured logging, error boundaries at every route, `loading.tsx` and `error.tsx` conventions.
- **Accessible**: WCAG 2.1 AA compliance, semantic HTML, keyboard navigation, proper ARIA attributes.
- **Documented**: JSDoc/TypeDoc comments for all exports, auto-generated API documentation.

**Verified Code**: Agent-generated code MUST compile (TypeScript check), build successfully, pass all tests, and have documentation before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated Next.js code compiles, builds, and passes tests before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY Next.js code, the agent MUST:**

1. **TypeScript Compilation Check**:
   ```bash
   # Verify TypeScript compiles without errors
   npx tsc --noEmit
   # Exit code MUST be 0
   ```
   - **MUST** return exit code 0
   - Address ALL TypeScript errors, not just warnings
   - NO `any` types allowed as workarounds

2. **Linting Check**:
   ```bash
   # Run ESLint with Next.js config
   npx eslint .
   ```
   - Fix all errors
   - Address critical warnings

3. **Build Verification**:
   ```bash
   # Verify production build succeeds
   npm run build
   ```
   - MUST complete without errors
   - Check for build warnings (especially dynamic route issues)

4. **Security & Dependency Verification (MANDATORY)**:
   ```bash
   # Scan for vulnerabilities in dependencies
   npm audit --audit-level=high

   # Verify lockfile matches package.json
   npm ci --dry-run
   ```
   - **MUST** have 0 high/critical vulnerabilities
   - Dependencies MUST be up to date or pinned to secure versions
   - Supply chain integrity (lockfile) MUST be verified

5. **Documentation Verification**:
   ```bash
   # Verify documentation completeness
   npm run docs:check
   ```
   - All public APIs have documentation
   - Documentation follows JSDoc conventions

#### Error Correction Process

If verification fails:

1. **TypeScript Errors**:
   - Read full error message
   - Identify root cause (type mismatch, missing import, async boundary)
   - Fix the issue
   - Re-verify

2. **Test Failures**:
   - Run failing test in isolation
   - Check test expectations vs actual output
   - Fix logic errors
   - Re-run all tests to ensure no regressions

3. **Build Failures**:
   - Check for Server/Client Component boundary violations
   - Verify `"use client"` directives are placed correctly
   - Ensure Server Actions use `"use server"` at the top of the file or function
   - Verify dynamic route parameters match file conventions

### B. Agent Workflow Example

**Complete Next.js generation workflow:**

1. **Generate Code Structure**:
   ```
   src/
   ├── app/
   │   ├── layout.tsx
   │   ├── page.tsx
   │   └── actions.ts
   ├── components/
   │   └── UserCard.tsx
   └── lib/
       └── db.ts
   ```

2. **Generate Initial Code** (following TDD - write tests first)

3. **Verify TypeScript**:
   ```bash
   npx tsc --noEmit
   # ✓ No errors
   ```

4. **Run Tests**:
   ```bash
   npm test
   # ✓ All tests pass
   ```

5. **Verify Build**:
   ```bash
   npm run build
   # ✓ Build successful
   ```

6. **Present Code**: Only after ALL checks pass

### C. Prohibited Practices

**NEVER deliver Next.js code that:**
- [ ] Fails TypeScript compilation
- [ ] Has failing tests
- [ ] Lacks tests for business logic
- [ ] Fails to build for production
- [ ] Uses `any` types to bypass type checking
- [ ] Has `"use client"` on components that don't need client-side interactivity
- [ ] Fetches data in Client Components when Server Components are possible
- [ ] Exposes server secrets via `NEXT_PUBLIC_` environment variables
- [ ] Relies solely on middleware/proxy for authentication
- [ ] **Fixes bugs without adding regression tests first**
- [ ] **Writes implementation before writing tests (violates TDD)**
- [ ] **Skips Red-Green-Refactor cycle for new features**

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### TDD Cycle

```
1. RED: Write a failing test first
   ↓
2. GREEN: Write minimal code to make it pass
   ↓
3. REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for Next.js

```typescript
// ═══════════════════════════════════════════════════════════════
// Step 1: RED - Write failing test first
// ═══════════════════════════════════════════════════════════════

// tests/unit/lib/format-price.test.ts
import { describe, it, expect } from 'vitest';
import { formatPrice } from '@/lib/format-price';

describe('formatPrice', () => {
  it('formats whole dollar amounts', () => {
    expect(formatPrice(1000)).toBe('$10.00');
  });

  it('formats cents correctly', () => {
    expect(formatPrice(1999)).toBe('$19.99');
  });

  it('returns $0.00 for zero', () => {
    expect(formatPrice(0)).toBe('$0.00');
  });

  it('handles negative amounts', () => {
    expect(formatPrice(-500)).toBe('-$5.00');
  });
});

// Run: npm test
// ❌ FAILS - formatPrice doesn't exist yet

// ═══════════════════════════════════════════════════════════════
// Step 2: GREEN - Write minimal implementation
// ═══════════════════════════════════════════════════════════════

// src/lib/format-price.ts

/**
 * Formats a price in cents to a dollar string.
 *
 * @param cents - Price in cents (integer)
 * @returns Formatted price string (e.g., "$19.99")
 */
export function formatPrice(cents: number): string {
  const dollars = Math.abs(cents) / 100;
  const formatted = `$${dollars.toFixed(2)}`;
  return cents < 0 ? `-${formatted}` : formatted;
}

// Run: npm test
// ✅ PASSES - tests pass

// ═══════════════════════════════════════════════════════════════
// Step 3: REFACTOR - Add locale support, keep tests green
// ═══════════════════════════════════════════════════════════════

// src/lib/format-price.ts (refactored)

/**
 * Formats a price in cents to a localized currency string.
 *
 * @param cents - Price in cents (integer)
 * @param currency - ISO 4217 currency code (default: "USD")
 * @param locale - BCP 47 locale string (default: "en-US")
 * @returns Formatted price string (e.g., "$19.99")
 *
 * @example
 * ```ts
 * formatPrice(1999);          // "$19.99"
 * formatPrice(1999, 'EUR', 'de-DE'); // "19,99 €"
 * ```
 */
export function formatPrice(
  cents: number,
  currency: string = 'USD',
  locale: string = 'en-US',
): string {
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
  }).format(cents / 100);
}

// Run: npm test
// ✅ PASSES - tests still pass after refactoring
```

### TDD for Server Actions

```typescript
// ═══════════════════════════════════════════════════════════════
// Step 1: RED - Write failing test for Server Action logic
// ═══════════════════════════════════════════════════════════════

// tests/unit/actions/create-post.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createPostAction } from '@/app/actions/create-post';

// Mock the database layer
vi.mock('@/lib/db', () => ({
  db: {
    post: {
      create: vi.fn(),
    },
  },
}));

// Mock next/cache
vi.mock('next/cache', () => ({
  revalidatePath: vi.fn(),
}));

import { db } from '@/lib/db';
import { revalidatePath } from 'next/cache';

describe('createPostAction', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('creates a post with valid data', async () => {
    const mockPost = { id: '1', title: 'Test', content: 'Hello' };
    vi.mocked(db.post.create).mockResolvedValue(mockPost);

    const formData = new FormData();
    formData.set('title', 'Test');
    formData.set('content', 'Hello');

    const result = await createPostAction({}, formData);

    expect(result.message).toBe('Post created successfully');
    expect(db.post.create).toHaveBeenCalledWith({
      data: { title: 'Test', content: 'Hello' },
    });
    expect(revalidatePath).toHaveBeenCalledWith('/posts');
  });

  it('returns error for empty title', async () => {
    const formData = new FormData();
    formData.set('title', '');
    formData.set('content', 'Hello');

    const result = await createPostAction({}, formData);

    expect(result.errors?.title).toBeDefined();
    expect(db.post.create).not.toHaveBeenCalled();
  });
});

// Run: npm test
// ❌ FAILS - createPostAction doesn't exist yet

// ═══════════════════════════════════════════════════════════════
// Step 2: GREEN - Write minimal implementation
// ═══════════════════════════════════════════════════════════════

// src/app/actions/create-post.ts
'use server';

import { z } from 'zod';
import { db } from '@/lib/db';
import { revalidatePath } from 'next/cache';

const createPostSchema = z.object({
  title: z.string().min(1, 'Title is required').max(200),
  content: z.string().min(1, 'Content is required'),
});

interface ActionState {
  message?: string;
  errors?: Record<string, string[]>;
}

/**
 * Server Action to create a new blog post.
 *
 * @param prevState - Previous form state
 * @param formData - Form data from submission
 * @returns Updated action state with message or errors
 */
export async function createPostAction(
  prevState: ActionState,
  formData: FormData,
): Promise<ActionState> {
  const parsed = createPostSchema.safeParse({
    title: formData.get('title'),
    content: formData.get('content'),
  });

  if (!parsed.success) {
    return { errors: parsed.error.flatten().fieldErrors };
  }

  await db.post.create({ data: parsed.data });
  revalidatePath('/posts');

  return { message: 'Post created successfully' };
}

// Run: npm test
// ✅ PASSES - tests pass
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. Bug Reported/Discovered
   ↓
2. Write a test that REPRODUCES the bug (test will FAIL)
   ↓
3. Verify the test fails for the right reason
   ↓
4. Fix the bug (make the test pass)
   ↓
5. Verify the test now PASSES
   ↓
6. Document the bug in test comments (include bug ID)
   ↓
7. Deploy with confidence (regression prevented)
```

### Example Bug Fix

```typescript
// Bug Report #3012: Product page crashes when price is null
// Root cause: formatPrice called with null from incomplete API response

// Step 1-2: Write test that reproduces the bug
// tests/unit/components/ProductCard.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ProductCard } from '@/components/ProductCard';

describe('ProductCard - Bug #3012', () => {
  it('renders gracefully when price is null - Bug #3012', () => {
    // Bug: ProductCard crashes with TypeError when price is null
    // Discovered: 2026-02-10
    // Root cause: formatPrice(null) throws

    const product = { id: '1', name: 'Widget', price: null };

    expect(() => {
      render(<ProductCard product={product} />);
    }).not.toThrow();

    expect(screen.getByText('Widget')).toBeInTheDocument();
    expect(screen.getByText('Price unavailable')).toBeInTheDocument();
  });
});

// Run: npm test
// ❌ FAILS - TypeError: Cannot read properties of null

// Step 3: Fix the bug
// src/components/ProductCard.tsx
interface ProductCardProps {
  product: {
    id: string;
    name: string;
    price: number | null;
  };
}

export function ProductCard({ product }: ProductCardProps) {
  return (
    <div>
      <h3>{product.name}</h3>
      {/* FIX for Bug #3012: Handle null price gracefully */}
      <p>{product.price !== null ? formatPrice(product.price) : 'Price unavailable'}</p>
    </div>
  );
}

// Run: npm test
// ✅ PASSES - bug fixed, regression prevented
```

---

## 3. Project Structure & Organization (MANDATORY)

### A. Standard Project Layout

**Follow this standard Next.js App Router project layout:**

```
project/
├── src/
│   ├── app/                       # App Router (routes & layouts)
│   │   ├── layout.tsx             # Root layout (required)
│   │   ├── page.tsx               # Home page
│   │   ├── loading.tsx            # Root loading UI
│   │   ├── error.tsx              # Root error boundary
│   │   ├── not-found.tsx          # 404 page
│   │   ├── global-error.tsx       # Global error boundary
│   │   ├── (auth)/                # Route group: auth pages
│   │   │   ├── login/
│   │   │   │   └── page.tsx
│   │   │   └── register/
│   │   │       └── page.tsx
│   │   ├── (marketing)/           # Route group: marketing
│   │   │   ├── about/
│   │   │   │   └── page.tsx
│   │   │   └── pricing/
│   │   │       └── page.tsx
│   │   ├── dashboard/
│   │   │   ├── layout.tsx         # Dashboard layout
│   │   │   ├── page.tsx
│   │   │   ├── loading.tsx
│   │   │   ├── error.tsx
│   │   │   ├── settings/
│   │   │   │   └── page.tsx
│   │   │   └── _components/       # Private: dashboard-only components
│   │   │       └── DashboardNav.tsx
│   │   ├── api/                   # Route Handlers (API routes)
│   │   │   └── health/
│   │   │       └── route.ts
│   │   └── actions/               # Server Actions (shared)
│   │       ├── auth.ts
│   │       └── posts.ts
│   ├── components/                # Shared UI components
│   │   ├── ui/                    # Base components (Button, Input, Card)
│   │   │   ├── Button.tsx
│   │   │   ├── Button.test.tsx
│   │   │   └── index.ts
│   │   ├── layout/                # Layout components (Header, Footer)
│   │   │   ├── Header.tsx
│   │   │   └── Footer.tsx
│   │   └── features/              # Feature-specific components
│   │       ├── auth/
│   │       └── posts/
│   ├── hooks/                     # Custom React hooks
│   │   ├── use-auth.ts
│   │   └── use-debounce.ts
│   ├── lib/                       # Utilities, API clients, shared logic
│   │   ├── db.ts                  # Database client
│   │   ├── auth.ts                # Auth configuration
│   │   ├── validations.ts         # Zod schemas
│   │   └── utils.ts               # Helper functions
│   ├── types/                     # TypeScript type definitions
│   │   ├── api.types.ts
│   │   └── models.types.ts
│   ├── styles/                    # Global styles
│   │   └── globals.css
│   └── config/                    # App configuration constants
│       └── site.ts
├── tests/                         # Test files
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── setup.ts
├── public/                        # Static assets
│   ├── images/
│   └── favicon.ico
├── .env.local                     # Local environment variables (git-ignored)
├── .env.example                   # Example env vars (committed)
├── .gitignore
├── eslint.config.mjs              # ESLint flat config
├── next.config.ts                 # Next.js config (TypeScript)
├── tailwind.config.ts             # Tailwind CSS config
├── tsconfig.json
├── vitest.config.ts               # Vitest config
├── playwright.config.ts           # Playwright E2E config
├── package.json
└── README.md
```

### B. Organization Principles

**Follow these principles:**

1. **Route Groups for Logical Grouping**:
   ```
   CORRECT - Group by concern without affecting URL
   src/app/(auth)/login/page.tsx      → /login
   src/app/(auth)/register/page.tsx   → /register
   src/app/(marketing)/about/page.tsx → /about

   WRONG - Nesting creates unwanted URL segments
   src/app/auth/login/page.tsx        → /auth/login
   ```

2. **Private Folders for Colocation**:
   ```
   CORRECT - Colocate route-specific components
   src/app/dashboard/_components/DashboardNav.tsx
   src/app/dashboard/_hooks/useDashboardData.ts

   WRONG - Put everything in global folders
   src/components/DashboardNav.tsx  (too far from usage)
   ```

3. **Server vs Client Separation**:
   - Keep components as Server Components by default
   - Extract interactive parts into small Client Components
   - Push `"use client"` boundary as deep as possible in the component tree

---

## 4. Mandatory Setup Requirements

### A. Next.js & React Version

* **Framework**: Use Next.js 15.x+ (or 16.x when available).
* **React**: Use React 19.x with Server Components support.
* **Build Tool**: Turbopack for development (`next dev --turbopack`). Turbopack is the default in Next.js 16.
* **Package Manager**: Use `npm` v10+.

```bash
# ✅ CORRECT - Modern project setup
npx create-next-app@latest my-app --typescript --tailwind --eslint --app --src-dir --turbopack --import-alias "@/*"
cd my-app

# ❌ WRONG - Pages Router (legacy)
npx create-next-app@latest my-app --no-app
```

### B. TypeScript Configuration

```json
// tsconfig.json
{
  "compilerOptions": {
    // Language & Environment
    "target": "ES2022",
    "lib": ["dom", "dom.iterable", "esnext"],
    "jsx": "preserve",
    "module": "esnext",
    "moduleResolution": "bundler",

    // Strict Type Checking (ALL REQUIRED)
    "strict": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "strictBindCallApply": true,
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

    // Output
    "noEmit": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "removeComments": false,
    "incremental": true,

    // Path Mapping
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    },

    // Next.js Plugin
    "plugins": [{ "name": "next" }],

    // Advanced
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": [
    "next-env.d.ts",
    "**/*.ts",
    "**/*.tsx",
    ".next/types/**/*.ts"
  ],
  "exclude": ["node_modules", ".next", "out"]
}
```

### C. Next.js Configuration (TypeScript)

```typescript
// next.config.ts
import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Recommended: strict React mode for catching issues
  reactStrictMode: true,

  // TypeScript: fail build on type errors
  typescript: {
    // Set to false to IGNORE type errors (NOT recommended)
    ignoreBuildErrors: false,
  },

  // ESLint: fail build on lint errors
  eslint: {
    // Set to false to IGNORE lint errors (NOT recommended)
    ignoreDuringBuilds: false,
  },

  // Image optimization
  images: {
    formats: ['image/avif', 'image/webp'],
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**.example.com',
      },
    ],
  },

  // Security headers
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'X-Frame-Options', value: 'DENY' },
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()',
          },
          {
            key: 'Strict-Transport-Security',
            value: 'max-age=63072000; includeSubDomains; preload',
          },
        ],
      },
    ];
  },
};

export default nextConfig;
```

### D. Essential Dependencies

**Core (Production):**
```json
{
  "dependencies": {
    "next": "^15.2.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "zod": "^3.23.0",
    "server-only": "^0.0.1"
  }
}
```

**Development:**
```json
{
  "devDependencies": {
    "typescript": "^5.5.0",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@types/node": "^22.0.0",

    "eslint": "^9.0.0",
    "eslint-config-next": "^15.2.0",
    "eslint-config-prettier": "^10.0.0",
    "prettier": "^3.4.0",
    "prettier-plugin-tailwindcss": "^0.6.0",

    "tailwindcss": "^4.0.0",
    "@tailwindcss/postcss": "^4.0.0",

    "vitest": "^3.0.0",
    "@vitejs/plugin-react": "^4.3.0",
    "@testing-library/react": "^16.0.0",
    "@testing-library/jest-dom": "^6.6.0",
    "@testing-library/user-event": "^14.5.0",
    "jsdom": "^25.0.0",

    "@playwright/test": "^1.49.0",

    "typedoc": "^0.27.0",
    "typedoc-plugin-markdown": "^4.4.0"
  }
}
```

### E. Package Scripts

```json
{
  "scripts": {
    "dev": "next dev --turbopack",
    "build": "next build",
    "start": "next start",
    "typecheck": "tsc --noEmit",
    "lint": "eslint .",
    "lint:fix": "eslint . --fix",
    "format": "prettier --write \"src/**/*.{ts,tsx,css}\"",
    "format:check": "prettier --check \"src/**/*.{ts,tsx,css}\"",
    "test": "vitest run",
    "test:watch": "vitest",
    "test:coverage": "vitest run --coverage",
    "test:e2e": "playwright test",
    "docs": "typedoc --out docs src/",
    "docs:check": "typedoc --emit none --validation.notDocumented true",
    "verify": "npm run typecheck && npm run lint && npm run docs:check && npm run test && npm run build"
  }
}
```

---

## 5. Server Components & Client Components (MANDATORY)

### A. Server Components (Default)

**ALL components are Server Components by default in the App Router. Keep them that way unless interactivity is required.**

```typescript
// ✅ CORRECT - Server Component (default, no directive needed)
// src/app/posts/page.tsx
import { db } from '@/lib/db';

/**
 * Posts listing page.
 * Fetches posts directly from the database in a Server Component.
 *
 * @returns Page displaying all published posts
 */
export default async function PostsPage() {
  // Direct database access - no API layer needed
  const posts = await db.post.findMany({
    where: { published: true },
    orderBy: { createdAt: 'desc' },
  });

  return (
    <main>
      <h1>Blog Posts</h1>
      <ul>
        {posts.map((post) => (
          <li key={post.id}>
            <a href={`/posts/${post.slug}`}>{post.title}</a>
          </li>
        ))}
      </ul>
    </main>
  );
}
```

### B. Client Components

**Only use `"use client"` when you need:**
- Event handlers (`onClick`, `onChange`, etc.)
- React hooks (`useState`, `useEffect`, `useRef`, etc.)
- Browser APIs (`window`, `localStorage`, `IntersectionObserver`, etc.)

```typescript
// ✅ CORRECT - Client Component for interactivity
// src/components/features/posts/LikeButton.tsx
'use client';

import { useState, useTransition } from 'react';
import { likePost } from '@/app/actions/posts';

/**
 * Like button with optimistic UI updates.
 *
 * @param props.postId - ID of the post to like
 * @param props.initialLikes - Current like count
 */
export function LikeButton({
  postId,
  initialLikes,
}: {
  postId: string;
  initialLikes: number;
}) {
  const [likes, setLikes] = useState(initialLikes);
  const [isPending, startTransition] = useTransition();

  const handleLike = () => {
    // Optimistic update
    setLikes((prev) => prev + 1);

    startTransition(async () => {
      await likePost(postId);
    });
  };

  return (
    <button onClick={handleLike} disabled={isPending} type="button">
      {likes} {likes === 1 ? 'Like' : 'Likes'}
    </button>
  );
}
```

### C. Composition Pattern: Server wraps Client

```typescript
// ✅ CORRECT - Server Component fetches data, passes to Client Component
// src/app/posts/[slug]/page.tsx
import { db } from '@/lib/db';
import { notFound } from 'next/navigation';
import { LikeButton } from '@/components/features/posts/LikeButton';
import { CommentSection } from '@/components/features/posts/CommentSection';

interface PageProps {
  params: Promise<{ slug: string }>;
}

export default async function PostPage({ params }: PageProps) {
  const { slug } = await params; // Next.js 15+: params is async

  const post = await db.post.findUnique({
    where: { slug },
    include: { comments: true },
  });

  if (!post) notFound();

  return (
    <article>
      <h1>{post.title}</h1>
      <div dangerouslySetInnerHTML={{ __html: post.contentHtml }} />

      {/* Client Components receive serializable props */}
      <LikeButton postId={post.id} initialLikes={post.likes} />
      <CommentSection postId={post.id} initialComments={post.comments} />
    </article>
  );
}
```

### D. Server-Only Protection

```typescript
// ✅ CORRECT - Prevent server code from leaking to client
// src/lib/db.ts
import 'server-only';
import { PrismaClient } from '@prisma/client';

const globalForPrisma = globalThis as unknown as { prisma: PrismaClient };

export const db = globalForPrisma.prisma || new PrismaClient();

if (process.env.NODE_ENV !== 'production') {
  globalForPrisma.prisma = db;
}
```

---

## 6. Data Fetching & Caching (MANDATORY)

### A. Data Fetching in Server Components

**CRITICAL: Fetch requests are NOT cached by default since Next.js 15. Caching must be explicit.**

```typescript
// ✅ CORRECT - Explicit caching strategies
export default async function ProductsPage() {
  // No caching (default in Next.js 15+)
  const latestOrders = await fetch('https://api.example.com/orders');

  // Explicitly cached (static data)
  const categories = await fetch('https://api.example.com/categories', {
    cache: 'force-cache',
  });

  // Time-based revalidation (ISR)
  const products = await fetch('https://api.example.com/products', {
    next: { revalidate: 3600 }, // Revalidate every hour
  });

  // Tag-based revalidation
  const featured = await fetch('https://api.example.com/featured', {
    next: { tags: ['featured-products'] },
  });

  // ... render
}
```

### B. Server Actions for Mutations

**Use Server Actions for all data mutations. Always validate inputs with Zod.**

```typescript
// src/app/actions/auth.ts
'use server';

import { z } from 'zod';
import { redirect } from 'next/navigation';
import { revalidatePath } from 'next/cache';
import { db } from '@/lib/db';

const signupSchema = z.object({
  name: z.string().min(1, 'Name is required').max(100),
  email: z.string().email('Invalid email address'),
  password: z.string().min(8, 'Password must be at least 8 characters'),
});

interface SignupState {
  message?: string;
  errors?: {
    name?: string[];
    email?: string[];
    password?: string[];
  };
}

/**
 * Server Action for user registration.
 *
 * Validates input with Zod, hashes password, creates user,
 * and redirects to the login page on success.
 *
 * @param prevState - Previous action state
 * @param formData - Form submission data
 * @returns Updated state with errors or success message
 */
export async function signup(
  prevState: SignupState,
  formData: FormData,
): Promise<SignupState> {
  const parsed = signupSchema.safeParse({
    name: formData.get('name'),
    email: formData.get('email'),
    password: formData.get('password'),
  });

  if (!parsed.success) {
    return { errors: parsed.error.flatten().fieldErrors };
  }

  const existingUser = await db.user.findUnique({
    where: { email: parsed.data.email },
  });

  if (existingUser) {
    return { errors: { email: ['Email already registered'] } };
  }

  const hashedPassword = await hashPassword(parsed.data.password);

  await db.user.create({
    data: {
      name: parsed.data.name,
      email: parsed.data.email,
      passwordHash: hashedPassword,
    },
  });

  redirect('/login?registered=true');
}
```

### C. Client-Side Form with `useActionState`

```typescript
// src/components/features/auth/SignupForm.tsx
'use client';

import { useActionState } from 'react';
import { signup } from '@/app/actions/auth';

/**
 * Signup form with server-side validation and error display.
 *
 * Uses `useActionState` to manage form state and pending status.
 *
 * @component
 */
export function SignupForm() {
  const [state, formAction, pending] = useActionState(signup, {});

  return (
    <form action={formAction}>
      <div>
        <label htmlFor="name">Name</label>
        <input id="name" name="name" required />
        {state.errors?.name && (
          <p role="alert" aria-live="polite">{state.errors.name[0]}</p>
        )}
      </div>

      <div>
        <label htmlFor="email">Email</label>
        <input id="email" name="email" type="email" required />
        {state.errors?.email && (
          <p role="alert" aria-live="polite">{state.errors.email[0]}</p>
        )}
      </div>

      <div>
        <label htmlFor="password">Password</label>
        <input id="password" name="password" type="password" required />
        {state.errors?.password && (
          <p role="alert" aria-live="polite">{state.errors.password[0]}</p>
        )}
      </div>

      <button type="submit" disabled={pending}>
        {pending ? 'Creating account...' : 'Sign Up'}
      </button>
    </form>
  );
}
```

### D. Revalidation Patterns

```typescript
// ✅ CORRECT - Revalidation after mutation
'use server';

import { revalidatePath, revalidateTag } from 'next/cache';

export async function updateProduct(id: string, data: ProductInput) {
  await db.product.update({ where: { id }, data });

  // Option 1: Revalidate specific path
  revalidatePath(`/products/${id}`);

  // Option 2: Revalidate by tag (preferred for shared data)
  revalidateTag('products');

  // Option 3: Revalidate layout and all child pages
  revalidatePath('/products', 'layout');
}
```

---

## 7. Routing & Layouts (MANDATORY)

### A. File Conventions

| File | Purpose |
|------|---------|
| `layout.tsx` | Shared UI for a segment and its children (persists across navigations) |
| `page.tsx` | Unique UI for a route (makes the route publicly accessible) |
| `loading.tsx` | Loading UI (wraps page in `<Suspense>`) |
| `error.tsx` | Error UI (wraps page in error boundary) |
| `not-found.tsx` | 404 UI for the segment |
| `route.ts` | API Route Handler (replaces API routes from Pages Router) |
| `template.tsx` | Like layout but re-mounts on navigation |
| `default.tsx` | Fallback for parallel routes |

### B. Dynamic Routes & Async Params

```typescript
// ✅ CORRECT - Next.js 15+: params and searchParams are Promises
// src/app/products/[id]/page.tsx

interface PageProps {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ tab?: string }>;
}

/**
 * Product detail page.
 *
 * @param props.params - Dynamic route parameters
 * @param props.searchParams - URL search parameters
 */
export default async function ProductPage({ params, searchParams }: PageProps) {
  const { id } = await params;
  const { tab } = await searchParams;

  const product = await db.product.findUnique({ where: { id } });
  if (!product) notFound();

  return (
    <main>
      <h1>{product.name}</h1>
      {tab === 'reviews' ? <Reviews productId={id} /> : <Details product={product} />}
    </main>
  );
}

// ✅ CORRECT - Static params for SSG
export async function generateStaticParams() {
  const products = await db.product.findMany({ select: { id: true } });
  return products.map((product) => ({ id: product.id }));
}

// ✅ CORRECT - Metadata generation
export async function generateMetadata({ params }: PageProps) {
  const { id } = await params;
  const product = await db.product.findUnique({ where: { id } });

  return {
    title: product?.name ?? 'Product Not Found',
    description: product?.description,
  };
}
```

### C. Route Groups & Parallel Routes

```typescript
// ✅ CORRECT - Route groups for different layouts
// src/app/(auth)/layout.tsx - minimal layout for auth pages
export default function AuthLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen items-center justify-center">
      {children}
    </div>
  );
}

// src/app/(dashboard)/layout.tsx - full layout with sidebar
export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex">
      <Sidebar />
      <main className="flex-1 p-6">{children}</main>
    </div>
  );
}

// ✅ CORRECT - Parallel routes for simultaneous content
// src/app/dashboard/layout.tsx
export default function Layout({
  children,
  analytics,
  notifications,
}: {
  children: React.ReactNode;
  analytics: React.ReactNode;
  notifications: React.ReactNode;
}) {
  return (
    <div>
      {children}
      <div className="grid grid-cols-2 gap-4">
        {analytics}
        {notifications}
      </div>
    </div>
  );
}
```

---

## 8. Middleware / Proxy (MANDATORY)

### A. Middleware Pattern (Next.js 15)

```typescript
// src/middleware.ts (Next.js 15)
// NOTE: In Next.js 16, rename to proxy.ts and export function proxy()
import { NextRequest, NextResponse } from 'next/server';

export function middleware(request: NextRequest) {
  // 1. Authentication redirect (optimistic check only)
  const token = request.cookies.get('session');
  if (!token && request.nextUrl.pathname.startsWith('/dashboard')) {
    return NextResponse.redirect(new URL('/login', request.url));
  }

  // 2. Security headers
  const response = NextResponse.next();
  response.headers.set('X-Frame-Options', 'DENY');
  response.headers.set('X-Content-Type-Options', 'nosniff');

  return response;
}

export const config = {
  // Skip static files and images
  matcher: ['/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)'],
};
```

### B. Middleware Best Practices

**DO:**
- Use for lightweight, optimistic checks (redirects, header modification)
- Use for A/B testing and feature flags
- Use for internationalization (locale detection)
- Always configure a `matcher` to skip static assets

**DON'T:**
- Rely solely on middleware for authentication (defense-in-depth required)
- Perform heavy data fetching or database queries
- Use as a substitute for Server Component data access checks
- Forget to verify auth again in Server Actions and data access layers

---

## 9. Performance Optimization (MANDATORY)

### A. Image Optimization

```typescript
// ✅ CORRECT - Optimized image usage
import Image from 'next/image';

/**
 * Hero section with optimized image.
 */
export function Hero() {
  return (
    <section>
      {/* LCP image: use priority prop */}
      <Image
        src="/hero.jpg"
        alt="Product showcase"
        width={1200}
        height={600}
        priority
        sizes="(max-width: 768px) 100vw, (max-width: 1200px) 80vw, 1200px"
      />

      {/* Below-the-fold: lazy loaded by default */}
      <Image
        src="/feature.jpg"
        alt="Feature description"
        width={800}
        height={400}
        sizes="(max-width: 768px) 100vw, 50vw"
      />

      {/* Fill mode for unknown dimensions */}
      <div className="relative h-64 w-full">
        <Image
          src="/banner.jpg"
          alt="Banner"
          fill
          className="object-cover"
          sizes="100vw"
        />
      </div>
    </section>
  );
}

// ❌ WRONG - Unoptimized HTML img tag
<img src="/hero.jpg" alt="hero" />
```

### B. Font Optimization

```typescript
// ✅ CORRECT - Self-hosted fonts with next/font
// src/app/layout.tsx
import { Inter, JetBrains_Mono } from 'next/font/google';

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-mono',
});

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="font-sans">{children}</body>
    </html>
  );
}
```

### C. Code Splitting with Dynamic Imports

```typescript
// ✅ CORRECT - Dynamic import for heavy components
import dynamic from 'next/dynamic';

// Only loaded when rendered
const Chart = dynamic(() => import('@/components/Chart'), {
  loading: () => <div className="h-64 animate-pulse bg-gray-200" />,
  ssr: false, // Client-only component (e.g., uses Canvas/WebGL)
});

// ✅ CORRECT - Dynamic import for conditional rendering
const AdminPanel = dynamic(() => import('@/components/AdminPanel'));

export default function Dashboard({ isAdmin }: { isAdmin: boolean }) {
  return (
    <main>
      <h1>Dashboard</h1>
      <Chart />
      {isAdmin && <AdminPanel />}
    </main>
  );
}
```

### D. Loading States with Suspense

```typescript
// ✅ CORRECT - loading.tsx for route-level loading
// src/app/dashboard/loading.tsx
export default function DashboardLoading() {
  return (
    <div className="animate-pulse">
      <div className="mb-4 h-8 w-48 rounded bg-gray-200" />
      <div className="grid grid-cols-3 gap-4">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="h-32 rounded bg-gray-200" />
        ))}
      </div>
    </div>
  );
}

// ✅ CORRECT - Granular Suspense boundaries for parallel data fetching
// src/app/dashboard/page.tsx
import { Suspense } from 'react';

export default function DashboardPage() {
  return (
    <main>
      <h1>Dashboard</h1>
      <div className="grid grid-cols-2 gap-4">
        <Suspense fallback={<CardSkeleton />}>
          <RevenueChart />
        </Suspense>
        <Suspense fallback={<CardSkeleton />}>
          <RecentOrders />
        </Suspense>
      </div>
    </main>
  );
}

// These async components can load in parallel
async function RevenueChart() {
  const data = await fetchRevenue(); // Doesn't block RecentOrders
  return <Chart data={data} />;
}

async function RecentOrders() {
  const orders = await fetchRecentOrders(); // Doesn't block RevenueChart
  return <OrderList orders={orders} />;
}
```

### E. Script Optimization

```typescript
// ✅ CORRECT - Optimized third-party scripts
import Script from 'next/script';

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {children}

        {/* Load analytics after page is interactive */}
        <Script
          src="https://analytics.example.com/script.js"
          strategy="lazyOnload"
        />

        {/* Inline script after hydration */}
        <Script id="theme-detector" strategy="afterInteractive">
          {`document.documentElement.dataset.theme = localStorage.getItem('theme') || 'light';`}
        </Script>
      </body>
    </html>
  );
}
```

---

## 10. Error Handling (MANDATORY)

### A. Error Boundaries

```typescript
// ✅ CORRECT - Route-level error boundary
// src/app/dashboard/error.tsx
'use client'; // Error boundaries must be Client Components

/**
 * Dashboard error boundary.
 * Displays user-friendly error message with retry option.
 *
 * @param props.error - The error that was thrown
 * @param props.reset - Function to retry rendering the segment
 */
export default function DashboardError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div role="alert" className="rounded-lg border border-red-200 bg-red-50 p-6">
      <h2 className="text-lg font-semibold text-red-800">Something went wrong</h2>
      <p className="mt-2 text-red-600">{error.message}</p>
      <button
        onClick={reset}
        type="button"
        className="mt-4 rounded bg-red-600 px-4 py-2 text-white hover:bg-red-700"
      >
        Try again
      </button>
    </div>
  );
}

// ✅ CORRECT - Global error boundary (catches root layout errors)
// src/app/global-error.tsx
'use client';

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="en">
      <body>
        <div role="alert">
          <h2>Something went wrong</h2>
          <button onClick={reset} type="button">Try again</button>
        </div>
      </body>
    </html>
  );
}
```

### B. Not Found Handling

```typescript
// ✅ CORRECT - Custom 404 page
// src/app/not-found.tsx
import Link from 'next/link';

export default function NotFound() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center">
      <h1 className="text-4xl font-bold">404</h1>
      <p className="mt-2 text-gray-600">Page not found</p>
      <Link href="/" className="mt-4 text-blue-600 hover:underline">
        Go home
      </Link>
    </main>
  );
}

// ✅ CORRECT - Programmatic not-found in Server Components
import { notFound } from 'next/navigation';

export default async function ProductPage({ params }: PageProps) {
  const { id } = await params;
  const product = await db.product.findUnique({ where: { id } });

  if (!product) {
    notFound(); // Renders closest not-found.tsx
  }

  return <ProductDetail product={product} />;
}
```

---

## 11. Security (MANDATORY)

### A. Defense-in-Depth Authentication

**NEVER rely solely on middleware/proxy for authentication. Verify at every layer.**

```
Layer 1: middleware.ts / proxy.ts  → Optimistic redirect (fast UX)
Layer 2: Server Component / Layout → Verify session, protect page render
Layer 3: Server Action / Route     → Re-verify before every mutation
Layer 4: Data Access Layer (DAL)   → Verify auth at every DB query
```

```typescript
// ✅ CORRECT - Data Access Layer with auth verification
// src/lib/dal.ts
import 'server-only';
import { cache } from 'react';
import { cookies } from 'next/headers';
import { db } from '@/lib/db';
import { verifySession } from '@/lib/auth';
import { redirect } from 'next/navigation';

/**
 * Verifies the current user session.
 * Cached per-request to avoid duplicate verification.
 *
 * @returns Authenticated user data
 * @throws Redirects to /login if not authenticated
 */
export const getCurrentUser = cache(async () => {
  const cookieStore = await cookies();
  const sessionToken = cookieStore.get('session')?.value;

  if (!sessionToken) {
    redirect('/login');
  }

  const session = await verifySession(sessionToken);
  if (!session) {
    redirect('/login');
  }

  const user = await db.user.findUnique({
    where: { id: session.userId },
    select: { id: true, name: true, email: true, role: true },
  });

  if (!user) {
    redirect('/login');
  }

  return user;
});

/**
 * Data access function that verifies auth before querying.
 *
 * @param postId - ID of the post to fetch
 * @returns Post data or null
 */
export async function getPostForUser(postId: string) {
  const user = await getCurrentUser(); // Auth verified here

  return db.post.findFirst({
    where: { id: postId, authorId: user.id },
  });
}
```

### B. Environment Variables

```typescript
// ✅ CORRECT - Server-only secrets (no prefix)
// .env.local
DATABASE_URL="postgresql://..."
JWT_SECRET="..."
STRIPE_SECRET_KEY="sk_..."

// ✅ CORRECT - Client-safe values (NEXT_PUBLIC_ prefix)
NEXT_PUBLIC_APP_URL="https://myapp.com"
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY="pk_..."

// ❌ WRONG - Secret exposed to client
NEXT_PUBLIC_DATABASE_URL="postgresql://..."
NEXT_PUBLIC_JWT_SECRET="..."
```

```typescript
// ✅ CORRECT - Validate environment variables at startup
// src/lib/env.ts
import { z } from 'zod';

const envSchema = z.object({
  DATABASE_URL: z.string().url(),
  JWT_SECRET: z.string().min(32),
  NODE_ENV: z.enum(['development', 'test', 'production']).default('development'),
  NEXT_PUBLIC_APP_URL: z.string().url(),
});

export const env = envSchema.parse(process.env);
```

### C. Server Action Security

```typescript
// ✅ CORRECT - Server Action with full security
'use server';

import { z } from 'zod';
import { getCurrentUser } from '@/lib/dal';
import { revalidatePath } from 'next/cache';
import { db } from '@/lib/db';

const updateProfileSchema = z.object({
  name: z.string().min(1).max(100),
  bio: z.string().max(500).optional(),
});

/**
 * Updates the authenticated user's profile.
 *
 * Security: Verifies auth, validates input, uses parameterized queries.
 */
export async function updateProfile(formData: FormData) {
  // 1. Verify authentication (defense-in-depth)
  const user = await getCurrentUser();

  // 2. Validate input
  const parsed = updateProfileSchema.safeParse({
    name: formData.get('name'),
    bio: formData.get('bio'),
  });

  if (!parsed.success) {
    return { errors: parsed.error.flatten().fieldErrors };
  }

  // 3. Authorize (user can only update their own profile)
  await db.user.update({
    where: { id: user.id },
    data: parsed.data,
  });

  // 4. Revalidate
  revalidatePath('/profile');

  return { message: 'Profile updated' };
}
```

---

## 12. Testing (MANDATORY)

### A. Testing Configuration

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    include: ['**/*.test.{ts,tsx}'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      exclude: ['node_modules/', 'tests/', '**/*.d.ts', '**/*.config.*'],
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
```

```typescript
// tests/setup.ts
import '@testing-library/jest-dom/vitest';
```

### B. Component Testing

```typescript
// ✅ CORRECT - Testing a Client Component
// src/components/ui/Button.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Button } from './Button';

describe('Button', () => {
  it('renders children', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByRole('button', { name: 'Click me' })).toBeInTheDocument();
  });

  it('calls onClick handler when clicked', async () => {
    const handleClick = vi.fn();
    const user = userEvent.setup();

    render(<Button onClick={handleClick}>Click me</Button>);
    await user.click(screen.getByRole('button'));

    expect(handleClick).toHaveBeenCalledOnce();
  });

  it('is disabled when isLoading is true', () => {
    render(<Button isLoading>Submit</Button>);

    expect(screen.getByRole('button')).toBeDisabled();
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('applies variant classes', () => {
    render(<Button variant="danger">Delete</Button>);

    expect(screen.getByRole('button')).toHaveClass('btn-danger');
  });
});
```

### C. Hook Testing

```typescript
// ✅ CORRECT - Testing a custom hook
// src/hooks/use-debounce.test.ts
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useDebounce } from './use-debounce';

describe('useDebounce', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.restoreAllMocks());

  it('returns initial value immediately', () => {
    const { result } = renderHook(() => useDebounce('hello', 500));
    expect(result.current).toBe('hello');
  });

  it('debounces value changes', () => {
    const { result, rerender } = renderHook(
      ({ value }) => useDebounce(value, 500),
      { initialProps: { value: 'hello' } },
    );

    rerender({ value: 'world' });
    expect(result.current).toBe('hello'); // Not updated yet

    act(() => vi.advanceTimersByTime(500));
    expect(result.current).toBe('world'); // Updated after delay
  });

  it('cancels on unmount', () => {
    const { result, unmount } = renderHook(() => useDebounce('test', 500));

    unmount();

    expect(() => {
      vi.advanceTimersByTime(1000);
    }).not.toThrow();
  });
});
```

### D. Server Action Testing

```typescript
// ✅ CORRECT - Testing Server Actions (test the logic, mock server deps)
// tests/unit/actions/create-post.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('@/lib/db', () => ({
  db: { post: { create: vi.fn() } },
}));

vi.mock('next/cache', () => ({
  revalidatePath: vi.fn(),
}));

vi.mock('@/lib/dal', () => ({
  getCurrentUser: vi.fn().mockResolvedValue({ id: 'user-1', role: 'admin' }),
}));

import { createPost } from '@/app/actions/posts';
import { db } from '@/lib/db';

describe('createPost', () => {
  beforeEach(() => vi.clearAllMocks());

  it('creates post with valid data', async () => {
    vi.mocked(db.post.create).mockResolvedValue({ id: '1', title: 'Test' });

    const formData = new FormData();
    formData.set('title', 'Test Post');
    formData.set('content', 'Hello world');

    const result = await createPost({}, formData);

    expect(result.message).toBe('Post created');
    expect(db.post.create).toHaveBeenCalledOnce();
  });

  it('returns validation errors for empty title', async () => {
    const formData = new FormData();
    formData.set('title', '');
    formData.set('content', 'Hello');

    const result = await createPost({}, formData);

    expect(result.errors?.title).toBeDefined();
    expect(db.post.create).not.toHaveBeenCalled();
  });
});
```

### E. E2E Testing with Playwright

```typescript
// tests/e2e/auth.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Authentication', () => {
  test('user can sign up and log in', async ({ page }) => {
    // Navigate to signup page
    await page.goto('/register');

    // Fill in the form
    await page.getByLabel('Name').fill('John Doe');
    await page.getByLabel('Email').fill('john@example.com');
    await page.getByLabel('Password').fill('securepassword123');

    // Submit
    await page.getByRole('button', { name: 'Sign Up' }).click();

    // Should redirect to login
    await expect(page).toHaveURL('/login?registered=true');

    // Log in
    await page.getByLabel('Email').fill('john@example.com');
    await page.getByLabel('Password').fill('securepassword123');
    await page.getByRole('button', { name: 'Log In' }).click();

    // Should redirect to dashboard
    await expect(page).toHaveURL('/dashboard');
    await expect(page.getByText('Welcome, John')).toBeVisible();
  });

  test('displays validation errors on invalid input', async ({ page }) => {
    await page.goto('/register');

    await page.getByRole('button', { name: 'Sign Up' }).click();

    await expect(page.getByText('Name is required')).toBeVisible();
    await expect(page.getByText('Invalid email')).toBeVisible();
  });
});
```

### F. Test Coverage Requirements

- Minimum coverage: **80%** for business logic
- Critical paths (auth, payments): **100%** coverage
- All Server Actions: **must have unit tests**
- All Client Components: **must have rendering tests**
- All custom hooks: **must have hook tests**
- Full user flows: **must have E2E tests with Playwright**
- **Async Server Components**: Use E2E tests (Vitest does not support async Server Components)

---

## 13. Documentation (MANDATORY)

### A. JSDoc Comments for All Exports

**ALL exported functions, components, hooks, types, and Server Actions MUST have JSDoc documentation.**

```typescript
/**
 * Fetches paginated products with optional filtering.
 *
 * @param options - Query options
 * @param options.page - Page number (1-indexed, default: 1)
 * @param options.limit - Items per page (default: 20, max: 100)
 * @param options.category - Filter by category slug
 * @param options.search - Search by product name
 * @returns Paginated product list with metadata
 *
 * @example
 * ```ts
 * const { products, total, hasMore } = await getProducts({
 *   page: 1,
 *   limit: 20,
 *   category: 'electronics',
 * });
 * ```
 *
 * @see {@link Product} for the product data structure
 */
export async function getProducts(options: GetProductsOptions): Promise<PaginatedProducts> {
  // Implementation
}
```

### B. TypeDoc Configuration

```json
// typedoc.json
{
  "entryPoints": ["src/"],
  "entryPointStrategy": "expand",
  "out": "docs",
  "exclude": [
    "**/*.test.ts",
    "**/*.test.tsx",
    "**/*.spec.ts",
    "**/test/**",
    "**/tests/**",
    "**/.next/**"
  ],
  "excludePrivate": true,
  "excludeInternal": true,
  "readme": "README.md",
  "plugin": ["typedoc-plugin-markdown"],
  "categorizeByGroup": true,
  "categoryOrder": [
    "Pages",
    "Server Actions",
    "Components",
    "Hooks",
    "Utilities",
    "Types",
    "*"
  ],
  "validation": {
    "notExported": true,
    "invalidLink": true,
    "notDocumented": true
  }
}
```

---

## 14. Deployment Checklist

### Agent-Generated Code Verification (MANDATORY)

**If code was generated/modified by an agent, verify BEFORE delivery:**

#### Build & Compilation
- [ ] TypeScript compiles: `npx tsc --noEmit` returns exit code 0
- [ ] No compilation errors or warnings
- [ ] Production build succeeds: `npm run build` completes
- [ ] No `"use client"` on components that don't need it
- [ ] No Server Component importing Client-only code without boundary

#### Testing
- [ ] All tests pass: `npm test` returns exit code 0
- [ ] Coverage: `npm run test:coverage` shows >80%
- [ ] E2E tests pass: `npm run test:e2e` returns exit code 0
- [ ] Server Actions have unit tests
- [ ] Client Components have rendering tests

#### Security
- [ ] Dependency scan passes: `npm audit --audit-level=high` reports 0 vulnerabilities
- [ ] No secrets in `NEXT_PUBLIC_` environment variables
- [ ] Server-only code protected with `import 'server-only'`
- [ ] All Server Action inputs validated with Zod
- [ ] Authentication verified at Data Access Layer (not just middleware)
- [ ] Security headers configured in `next.config.ts`
- [ ] CSRF protection via `SameSite` cookies and origin verification

#### Code Quality
- [ ] ESLint passes: `npx eslint .`
- [ ] Code formatted: `npx prettier --check .`
- [ ] No unused dependencies
- [ ] No `any` types
- [ ] All exports have JSDoc documentation

#### Performance
- [ ] Images use `next/image` with proper `sizes` and `priority`
- [ ] Fonts use `next/font` (self-hosted, no layout shift)
- [ ] Heavy components use `next/dynamic` for code splitting
- [ ] Data fetching uses appropriate caching strategy
- [ ] Loading states provided via `loading.tsx` or `<Suspense>`
- [ ] Error boundaries at every route segment

#### Architecture
- [ ] Server Components used by default
- [ ] `"use client"` boundary pushed as deep as possible
- [ ] Server Actions used for mutations (not GET requests)
- [ ] Route groups organize related pages
- [ ] Feature-based component organization

---

## 15. Why This Configuration Works

**Server-First Architecture**:
- Server Components reduce JavaScript bundle size by 30-50%, improving TTI and LCP.
- Direct database access in Server Components eliminates unnecessary API layers.

**Explicit Caching**:
- No hidden caching behavior (Next.js 15+ change). Developers control exactly what is cached and for how long.
- `revalidateTag` and `revalidatePath` provide granular cache invalidation.

**Type Safety Across Boundaries**:
- TypeScript strict mode catches 30-40% more bugs at compile time.
- Zod validation at Server Action boundaries ensures runtime type safety.

**Defense-in-Depth Security**:
- Multi-layer auth verification prevents middleware bypass attacks (CVE-2025-29927).
- `server-only` package prevents server code from leaking to the client bundle.

**Performance by Default**:
- `next/image`, `next/font`, and `next/script` optimize Core Web Vitals automatically.
- Turbopack delivers 2-5x faster builds and 10x faster Fast Refresh.

---

## 16. Quick Reference

### Common Commands

```bash
# Development
npm run dev              # Start dev server with Turbopack

# Build & Deploy
npm run build            # Production build
npm run start            # Start production server

# Type Checking
npx tsc --noEmit         # Check types without emitting

# Testing
npm test                 # Run unit tests
npm run test:coverage    # Run with coverage
npm run test:e2e         # Run E2E tests

# Linting & Formatting
npx eslint .             # Lint
npx prettier --check .   # Check formatting
npx prettier --write .   # Fix formatting

# Documentation
npm run docs             # Generate docs
npm run docs:check       # Verify doc completeness

# Full Verification
npm run verify           # typecheck + lint + docs + test + build
```

### File Conventions Cheat Sheet

| File | Purpose |
|------|---------|
| `page.tsx` | Route page (makes segment accessible) |
| `layout.tsx` | Shared layout (wraps children, persists) |
| `loading.tsx` | Loading UI (Suspense fallback) |
| `error.tsx` | Error boundary (Client Component) |
| `not-found.tsx` | 404 UI |
| `route.ts` | API Route Handler |
| `template.tsx` | Layout that re-mounts |
| `default.tsx` | Parallel route fallback |
| `middleware.ts` | Request interceptor (Next.js 15) |
| `proxy.ts` | Request interceptor (Next.js 16) |

### Key Patterns

```typescript
// Server Component (default)
export default async function Page() {
  const data = await fetchData();
  return <div>{data}</div>;
}

// Client Component
'use client';
export function InteractiveWidget() {
  const [state, setState] = useState(false);
  return <button onClick={() => setState(!state)}>Toggle</button>;
}

// Server Action
'use server';
export async function createItem(formData: FormData) {
  const data = schema.parse(Object.fromEntries(formData));
  await db.item.create({ data });
  revalidatePath('/items');
}

// Dynamic route with async params (Next.js 15+)
export default async function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  // ...
}
```

---

**End of Next.js Development Guidelines**
