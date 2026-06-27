# Next.js Development Guidelines
Mandatory standards for Next.js: App Router, Server Components, Server Actions, explicit caching, type-safe and secure. Next.js 15, React 19, TypeScript 5.6+, Turbopack, Vitest, Playwright, Zod.

---
name: nextjs
title: Next.js Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: framework
tools: [next@15, react@19, typescript@5.6, turbopack, vitest, playwright, zod, eslint@9, prettier@3]
requires:
  - reactjs
  - typescript
  - tdd
  - secure-coding
recommends:
  - rest
  - accessibility
  - e2e-testing
  - performance
  - observability
  - zod
provides:
  - nextjs-app-router
  - server-components
  - server-actions
  - nextjs-caching
  - ssr-ssg-isr
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Next.js (the App Router runtime, the server/client boundary, caching, and Next's optimization primitives). React itself, TypeScript, testing, security, and performance policy live in their owners.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Next.js code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`reactjs.md`](guides://reactjs.md) — components, hooks, JSX, state. *(This guide never restates React; it only adds the Server/Client split.)*
> - [`typescript.md`](guides://typescript.md) — strict config, no `any`, type style. *(Next binding: `tsc --noEmit`, the `next` TS plugin.)*
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Next binding: `vitest run`; async Server Components → E2E.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(Next binding: `npm audit`, `NEXT_PUBLIC_` exposure, `server-only`.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`zod.md`](guides://zod.md) — schema validation *(binding: every Server Action & route handler input)*
> - [`rest.md`](guides://rest.md) — API design for `route.ts` route handlers
> - [`accessibility.md`](guides://accessibility.md) — WCAG for forms/UI · [`e2e-testing.md`](guides://e2e-testing.md) — Playwright flows
> - [`performance.md`](guides://performance.md) — Core Web Vitals *(binding: `next/image`, `next/font`, streaming)* · [`observability.md`](guides://observability.md) — logging, tracing, error reporting

> 📎 **SEE ALSO:** [`comments.md`](guides://comments.md) · [`env-config.md`](guides://env-config.md) · [`error-handling.md`](guides://error-handling.md) · [`oauth.md`](guides://oauth.md) · [`ci-cd.md`](guides://ci-cd.md)

---

## 1. Core Philosophies: NEXTJS-FIRST

Next.js-specific principles only. TDD, security, React, TypeScript, performance, and a11y come from §0 — do **not** restate them.

- **N**ative-first: prefer Next built-ins (App Router, `next/image`/`font`/`script`, Server Actions) over external libraries that duplicate them.
- **E**xplicit caching: caching is **opt-in** since Next 15. `fetch` is uncached by default; reach for `force-cache`, `next: { revalidate }`, `next: { tags }`, or `"use cache"` deliberately (§6).
- **X**-ray boundaries: the Server/Client line is the central design decision. Default to Server Components; `"use client"` is a deliberate, pushed-as-deep-as-possible boundary (§5).
- **T**hin client: keep data fetching, secrets, and heavy logic on the server (`server-only`); ship Client Components only for interactivity.
- **J**udicious splitting: stream with `<Suspense>`/`loading.tsx`, split with `next/dynamic`, organize with route groups & parallel routes.
- **S**erver-first mutations: mutate via Server Actions (validated, authorized, revalidated) — not ad-hoc API routes.

**App Router is the default and only target of this guide.** The Pages Router (`pages/`, `getServerSideProps`, `getStaticProps`, `getInitialProps`, `_app`/`_document`) is legacy — do not generate it for new code.

**Verified Code**: agent-generated Next.js MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `NEXT-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| NEXT-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `npm test` (`vitest run`) | exit 0, 0 skips |
| NEXT-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `npm test` | failing→passing |
| NEXT-TST-03 | Async Server Components & full flows MUST be covered by E2E (see `e2e-testing.md`) | `npm run test:e2e` | exit 0 |
| NEXT-TYP-01 | Strict TypeScript, no `any` across the server/client boundary (see `typescript.md`) | `tsc --noEmit` | exit 0 |
| NEXT-FMT-01 | Code MUST be formatted | `prettier --check .` | no diff |
| NEXT-LINT-01 | Linter MUST pass clean | `next lint` / `eslint .` | exit 0 |
| NEXT-BUILD-01 | Production build MUST succeed | `next build` | exit 0 |
| NEXT-RSC-01 | `"use client"` MUST be used only for interactivity/browser APIs, boundary pushed deep | review / lint | no needless client |
| NEXT-CACHE-01 | Caching MUST be explicit (no reliance on implicit `fetch` caching) | review | cache opts stated |
| NEXT-SEC-01 | 0 high/critical CVEs in deps (see `secure-coding.md`) | `npm audit --audit-level=high` | 0 high/critical |
| NEXT-SEC-02 | No server secret MUST be under a `NEXT_PUBLIC_` var (see `secure-coding.md`) | grep / review | none exposed |
| NEXT-SEC-03 | Auth MUST be re-verified in the DAL/Server Action, not just middleware | review | enforced per query |
| NEXT-SEC-04 | Every Server Action / route handler input MUST be validated (see `zod.md`) | review | all parsed |
| NEXT-DEP-01 | Lockfile in sync & verified (see `secure-coding.md`) | `npm ci --dry-run` | in sync |
| NEXT-A11Y-01 | UI MUST meet WCAG 2.1 AA (see `accessibility.md`) | axe / Playwright a11y | 0 violations |
| NEXT-PERF-01 | Images/fonts/heavy code MUST use Next primitives (see `performance.md`) | review / Lighthouse | CWV pass |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`); fixing a bug without a regression test first; `any` to bypass the boundary; `"use client"` on a component that needs no interactivity; fetching data in a Client Component when a Server Component can; a secret under `NEXT_PUBLIC_`; auth enforced only in middleware; an unvalidated Server Action; new Pages-Router code.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
prettier --check .                 # NEXT-FMT-01
next lint                          # NEXT-LINT-01
tsc --noEmit                       # NEXT-TYP-01 (eslint does NOT type-check)
vitest run --coverage              # NEXT-TST-01/02
playwright test                    # NEXT-TST-03 (async RSC + flows)
next build                         # NEXT-BUILD-01 (catches boundary/serialization errors)
npm audit --audit-level=high       # NEXT-SEC-01
npm ci --dry-run                   # NEXT-DEP-01
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

App Router layout. Group by feature; colocate route-only code in `_private` folders; keep shared logic in `lib/`.

```
project/
├── src/
│   ├── app/                    # App Router — routes, layouts, route handlers
│   │   ├── layout.tsx          # root layout (required: <html>/<body>)
│   │   ├── page.tsx  loading.tsx  error.tsx  not-found.tsx  global-error.tsx
│   │   ├── (marketing)/        # route group — organizes, no URL segment
│   │   ├── dashboard/
│   │   │   ├── layout.tsx  page.tsx  loading.tsx  error.tsx
│   │   │   ├── @analytics/ @notifications/   # parallel route slots
│   │   │   └── _components/    # private: not routable
│   │   ├── api/health/route.ts # route handler (see rest.md)
│   │   └── actions/            # shared Server Actions ("use server")
│   ├── components/             # shared UI (Server by default; ui/ feature/)
│   ├── hooks/  lib/  types/    # client hooks; db/auth/dal/env; shared types
│   │   ├── db.ts  dal.ts  auth.ts  env.ts   # all import 'server-only'
│   └── middleware.ts           # (Next 16: rename to proxy.ts, export proxy())
├── public/                     # static assets
├── next.config.ts  tsconfig.json  vitest.config.ts  playwright.config.ts
└── package.json  package-lock.json
```

- Route groups `(name)/` organize without affecting the URL; nesting real folders adds URL segments.
- Private folders `_name/` and slots `@name/` are never routable.
- Push `"use client"` to the leaves; keep pages/layouts as Server Components.

---

## 5. Server vs Client Components — the central boundary

This split is the unique core of Next.js. React mechanics (hooks, props, JSX) are owned by [`reactjs.md`](guides://reactjs.md); below is only what the App Router adds.

**Server Components (default).** No directive. May be `async`, may `await` data and DB clients directly, may read secrets — they never ship to the browser. Cannot use hooks, event handlers, or browser APIs.

```tsx
// app/posts/page.tsx — Server Component, direct DB access, no API layer
import { db } from '@/lib/db';
export default async function PostsPage() {
  const posts = await db.post.findMany({ where: { published: true } });
  return <ul>{posts.map(p => <li key={p.id}><a href={`/posts/${p.slug}`}>{p.title}</a></li>)}</ul>;
}
```

**Client Components.** `'use client'` at the top of the file. Required only for: event handlers, React state/effect/ref hooks, browser APIs, or third-party libs that use them. Props crossing the boundary MUST be serializable (no functions, classes, Dates-as-instances).

```tsx
'use client';
import { useState, useTransition } from 'react';
import { likePost } from '@/app/actions/posts';
export function LikeButton({ postId, initial }: { postId: string; initial: number }) {
  const [likes, setLikes] = useState(initial);
  const [pending, start] = useTransition();
  return <button disabled={pending} onClick={() => { setLikes(n => n + 1); start(() => likePost(postId)); }}>{likes}</button>;
}
```

**Composition.** A Server Component fetches and passes serializable props into Client Components; it may also pass Server Components as `children`/slots into a Client Component (the children stay server-rendered). This is how you keep the client boundary thin.

**`server-only` guard.** Modules holding secrets, DB clients, or auth MUST `import 'server-only'` so an accidental client import fails the build.

```ts
// lib/db.ts
import 'server-only';
```

> Footguns: a `'use client'` file taints everything it imports as client; don't put data fetching there. Don't pass non-serializable props across the boundary. Don't add `'use client'` to a layout/page just to use one interactive child — extract the child.

---

## 6. Data Fetching, Caching & Rendering

Next.js canonically owns caching and rendering-strategy selection. **Since Next 15, `fetch` is NOT cached by default** and route handlers / page data are dynamic unless you opt into caching.

### Caching `fetch`
```ts
await fetch(url);                                  // dynamic, uncached (default)
await fetch(url, { cache: 'force-cache' });        // cached until revalidated (build-time static)
await fetch(url, { next: { revalidate: 3600 } });  // ISR — revalidate every hour
await fetch(url, { next: { tags: ['products'] } });// tag for targeted invalidation
```

### `"use cache"` (caching beyond fetch — DB calls, components, whole functions)
```ts
import { unstable_cacheLife as cacheLife, unstable_cacheTag as cacheTag } from 'next/cache';
async function getProducts() {
  'use cache';
  cacheLife('hours');          // named profile (seconds/minutes/hours/days/max)
  cacheTag('products');        // invalidate with revalidateTag('products')
  return db.product.findMany();
}
```

### Revalidation (after a mutation)
```ts
import { revalidatePath, revalidateTag } from 'next/cache';
revalidatePath('/products/[id]', 'page');  // a path (or 'layout' to include children)
revalidateTag('products');                 // everything tagged — preferred for shared data
```

### Rendering strategies (App Router)
- **Static (SSG)** — default for routes without dynamic APIs; rendered at build, served from cache. Add `generateStaticParams` to pre-render dynamic segments.
- **Dynamic (SSR)** — triggered by `cookies()`, `headers()`, `searchParams`, uncached `fetch`, or `export const dynamic = 'force-dynamic'`. Rendered per request.
- **ISR** — static output revalidated on a timer (`next: { revalidate }` or `export const revalidate = N`) or on-demand (`revalidateTag/Path`).
- **PPR (Partial Prerendering)** — a static shell streamed instantly with dynamic holes filled via `<Suspense>` (`experimental.ppr`). Prefer it over forcing a whole route dynamic.
- Route-segment config: `export const dynamic | revalidate | fetchCache | runtime = ...` controls a segment explicitly.

### Async params (Next 15+)
`params` and `searchParams` are **Promises** — always `await` them.
```tsx
export default async function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
}
```

### Server Actions (mutations)
`'use server'`; validate with Zod (see `zod.md`), authorize (§7), then revalidate. Callable from a `<form action={...}>` or via `useActionState`/`startTransition`.
```ts
'use server';
import { z } from 'zod';
import { revalidatePath } from 'next/cache';
import { getCurrentUser } from '@/lib/dal';
const schema = z.object({ title: z.string().min(1).max(200), content: z.string().min(1) });
export async function createPost(_prev: State, formData: FormData): Promise<State> {
  const user = await getCurrentUser();                       // re-verify auth (NEXT-SEC-03)
  const parsed = schema.safeParse(Object.fromEntries(formData));
  if (!parsed.success) return { errors: parsed.error.flatten().fieldErrors };
  await db.post.create({ data: { ...parsed.data, authorId: user.id } });
  revalidatePath('/posts');
  return { message: 'created' };
}
```
Client binding uses React 19 `useActionState` (form state + pending) — the hook itself is owned by [`reactjs.md`](guides://reactjs.md).

### Route Handlers (`app/api/.../route.ts`)
HTTP endpoints for webhooks, third-party callbacks, and non-RSC clients. Export `GET`/`POST`/etc.; uncached by default. API design (status codes, versioning, pagination) is owned by [`rest.md`](guides://rest.md).
```ts
export async function GET() { return Response.json({ ok: true }); }  // edge or node runtime
```
Prefer Server Actions over hand-rolled mutation endpoints for your own UI.

---

## 7. Routing, Metadata & Streaming

| File | Role |
|------|------|
| `layout.tsx` | shared UI that persists across navigation (root must render `<html>`/`<body>`) |
| `page.tsx` | makes a segment routable |
| `loading.tsx` | Suspense fallback for the segment (enables streaming) |
| `error.tsx` / `global-error.tsx` | error boundary (Client Component) / root-level boundary |
| `not-found.tsx` | 404 UI; trigger with `notFound()` |
| `template.tsx` | like layout but re-mounts each navigation |
| `route.ts` | route handler |
| `default.tsx` | fallback for an unmatched parallel-route slot |

**Metadata API** — static `export const metadata` or dynamic `generateMetadata` (async, receives awaited `params`). Drives `<title>`, OG/Twitter tags, canonical, robots. Use `generateStaticParams` to enumerate dynamic routes for SSG.

```tsx
export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params;
  const p = await db.product.findUnique({ where: { id } });
  return { title: p?.name ?? 'Not found', description: p?.description };
}
export async function generateStaticParams() {
  return (await db.product.findMany({ select: { id: true } })).map(({ id }) => ({ id }));
}
```

**Streaming & Suspense** — wrap independent async children in `<Suspense>` so they fetch in parallel and stream as they resolve; `loading.tsx` is the segment-level equivalent. This is the Next mechanism behind the Core Web Vitals goals owned by [`performance.md`](guides://performance.md).

**Route groups & parallel/intercepting routes** — `(group)/` for shared layouts without URL impact; `@slot/` parallel routes render multiple pages into one layout (with `default.tsx` fallbacks); `(.)`/`(..)` intercepting routes power modal-over-page patterns.

---

## 8. Middleware / Proxy

Runs on the Edge before a request completes. Use for **optimistic** redirects, header/cookie rewrites, locale/feature-flag routing. Always set a `matcher` to skip static assets. Keep it light — no DB queries, no heavy work.

```ts
// middleware.ts  (Next 16: rename to proxy.ts and export `function proxy(...)`)
import { NextRequest, NextResponse } from 'next/server';
export function middleware(req: NextRequest) {
  if (!req.cookies.get('session') && req.nextUrl.pathname.startsWith('/dashboard'))
    return NextResponse.redirect(new URL('/login', req.url));   // optimistic only
  return NextResponse.next();
}
export const config = { matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'] };
```

> Middleware is **not** an auth boundary — it can be bypassed (cf. CVE-2025-29927). Always re-verify in the DAL / Server Action (§9).

---

## 9. Security (Next.js bindings)

General policy (CVEs, secrets, supply chain, headers) is owned by [`secure-coding.md`](guides://secure-coding.md); auth flows by [`oauth.md`](guides://oauth.md). Next-specific bindings:

**Defense-in-depth auth — verify at every layer, never middleware alone:**
1. middleware/proxy — optimistic redirect (UX only).
2. Server Component/layout — verify session before rendering protected UI.
3. Server Action / route handler — re-verify before every mutation.
4. **Data Access Layer (DAL)** — verify auth at every query; this is the real gate (NEXT-SEC-03).

```ts
// lib/dal.ts
import 'server-only';
import { cache } from 'react';
import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';
export const getCurrentUser = cache(async () => {        // cache() dedupes per request
  const token = (await cookies()).get('session')?.value;
  const session = token && await verifySession(token);
  if (!session) redirect('/login');
  return db.user.findUnique({ where: { id: session.userId }, select: { id: true, role: true } });
});
```

**Environment variables (NEXT_PUBLIC binding).** Only `NEXT_PUBLIC_`-prefixed vars are inlined into the client bundle — everything else stays server-only. A secret under `NEXT_PUBLIC_` is a leak (NEXT-SEC-02). Validate `process.env` once at startup with Zod (see `env-config.md` for layering policy):
```ts
export const env = z.object({
  DATABASE_URL: z.string().url(),
  JWT_SECRET: z.string().min(32),                 // server-only
  NEXT_PUBLIC_APP_URL: z.string().url(),          // safe to ship
}).parse(process.env);
```

**Server Actions are public POST endpoints** — every one MUST verify auth and validate inputs (Zod), regardless of which UI calls it. Next adds CSRF protection via Origin checks + `SameSite` cookies, but authorization is your responsibility. Sanitize any `dangerouslySetInnerHTML`. Set security headers in `next.config.ts` `headers()`.

---

## 10. Performance & Optimization (Next.js primitives)

Core Web Vitals targets and budgets are owned by [`performance.md`](guides://performance.md). Next's built-in levers:

- **`next/image`** — always over `<img>`. Set `width`/`height` (or `fill`) and `sizes`; mark the LCP image `priority`; below-the-fold lazy-loads by default. Serves AVIF/WebP, prevents CLS.
- **`next/font`** — self-hosts Google/local fonts at build with `display: 'swap'`; zero layout shift, no render-blocking request.
- **`next/script`** — `strategy` of `beforeInteractive` | `afterInteractive` | `lazyOnload` for third-party scripts.
- **`next/dynamic`** — code-split heavy/conditional Client Components (`ssr: false` for browser-only libs).
- **Streaming** — `<Suspense>` + `loading.tsx` ship a fast shell and stream the rest (§7); PPR (§6) combines static shell + dynamic holes.
- **Turbopack** — default dev bundler (`next dev`); also `next build --turbopack`.

---

## 11. Error Handling & Observability

Error-strategy policy → [`error-handling.md`](guides://error-handling.md); logging/tracing/error-reporting → [`observability.md`](guides://observability.md). Next bindings:

- `error.tsx` (Client Component) catches render errors in its segment and exposes `reset()`; `global-error.tsx` catches root-layout errors and must render its own `<html>`/`<body>`.
- `not-found.tsx` + `notFound()` for the missing-resource path.
- Report server errors from Server Components/Actions/route handlers to your observability sink; the `error.digest` correlates a client boundary to the server log line.

---

## 12. Testing (Next.js bindings)

Test-first policy and coverage gates are owned by [`tdd.md`](guides://tdd.md); flow testing by [`e2e-testing.md`](guides://e2e-testing.md). Next specifics:

- **Unit/component (Vitest + Testing Library, jsdom):** Client Components, hooks, and pure `lib/` logic. Alias `@` to `src/` in `vitest.config.ts`.
- **Server Actions:** test the exported function directly; mock `@/lib/db`, `next/cache`, and `@/lib/dal`. Assert both the happy path and Zod validation failures.
- **Async Server Components:** Vitest cannot render them — cover via Playwright E2E (NEXT-TST-03). Don't fake it with shallow renders.
- **E2E (Playwright):** full auth/checkout flows and a11y assertions (axe). Run against `next build && next start`, not dev.

---

## 13. Tooling & Dependencies

Supply-chain policy → [`secure-coding.md`](guides://secure-coding.md). Next binding:

```bash
npx create-next-app@latest --typescript --tailwind --eslint --app --src-dir --turbopack --import-alias "@/*"
npm ci                  # install from lockfile (reproducible)
npm audit --audit-level=high   # NEXT-SEC-01
npm ci --dry-run        # NEXT-DEP-01: lockfile in sync
```
Pin `next`, `react`, `react-dom` to the same major; keep `eslint-config-next` matched to the Next version. Commit `package-lock.json`. Documentation of public exports follows [`comments.md`](guides://comments.md) (TSDoc/JSDoc).

---

## 14. Quick Reference

```bash
next dev               # dev server (Turbopack)
next build && next start   # production
tsc --noEmit           # types
vitest run             # unit/component tests
playwright test        # E2E + async RSC
next lint              # lint
```

```tsx
// Server Component (default)         | Client Component
export default async function Page() {   'use client';
  const data = await getData();          export function Widget() {
  return <View data={data} />; }           const [on, set] = useState(false); ... }

// Server Action                      | Async params (Next 15+)
'use server';                           async function Page({ params }:
export async function act(fd: FormData) { { params: Promise<{ id: string }> }) {
  const v = schema.parse(...);            const { id } = await params; }
  await db.x.create({ data: v });
  revalidatePath('/x'); }
```

---

## 15. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] NEXT-FMT-01 — `prettier --check` clean
- [ ] NEXT-LINT-01 — `next lint` clean
- [ ] NEXT-TYP-01 — `tsc --noEmit` clean, no `any` across boundaries
- [ ] NEXT-TST-01/02 — tests pass, bugs have regression tests
- [ ] NEXT-TST-03 — Playwright E2E (incl. async Server Components) green
- [ ] NEXT-BUILD-01 — `next build` succeeds
- [ ] NEXT-RSC-01 — `"use client"` only where needed, boundary pushed deep
- [ ] NEXT-CACHE-01 — caching explicit (no implicit `fetch` assumptions)
- [ ] NEXT-SEC-01 — `npm audit` 0 high/critical
- [ ] NEXT-SEC-02 — no secret under `NEXT_PUBLIC_`
- [ ] NEXT-SEC-03 — auth re-verified in DAL/Server Action, `server-only` on server modules
- [ ] NEXT-SEC-04 — every Server Action / route handler input validated (Zod)
- [ ] NEXT-DEP-01 — `package-lock.json` in sync, committed
- [ ] NEXT-A11Y-01 — WCAG 2.1 AA, no axe violations
- [ ] NEXT-PERF-01 — `next/image` + `next/font` + `next/dynamic`, CWV pass
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Next.js Guidelines**
