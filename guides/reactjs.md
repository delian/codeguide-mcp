# React Development Guidelines
Mandatory standards for modern React: function components, hooks, React 19 Actions, Server Components, and the React Compiler. React 19, React Compiler, Vite 6, TypeScript 5, Vitest, Testing Library.

---
name: reactjs
title: React Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: framework
tools: [react@19, react-dom@19, react-compiler, vite@6, typescript@5.6, vitest@2, "@testing-library/react@16"]
requires:
  - typescript
  - tdd
  - secure-coding
recommends:
  - javascript
  - accessibility
  - e2e-testing
  - ui
  - css
  - performance
  - zod
  - nextjs
  - observability
provides:
  - react-hooks
  - react-components
  - react19-actions
  - server-components
  - react-state
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to React.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating React code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`typescript.md`](guides://typescript.md) — strict mode, prop/hook typing, no `any`. *(React binding: type every prop interface and hook generic; never type props as `any`.)*
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(React binding: runner is Vitest + React Testing Library; query by accessible role.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy, XSS. *(React binding: `npm audit`; sanitize before `dangerouslySetInnerHTML`.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`nextjs.md`](guides://nextjs.md) — the meta-framework for routing, SSR/RSC, and bundling. *(Reach for it whenever the app needs Server Components, file-based routing, or SSR.)*
> - [`accessibility.md`](guides://accessibility.md) — WCAG, semantic HTML, ARIA, focus management.
> - [`e2e-testing.md`](guides://e2e-testing.md) — Playwright/Cypress user-flow tests.
> - [`ui.md`](guides://ui.md) — component design, design tokens, composition.
> - [`css.md`](guides://css.md) — styling strategy, CSS Modules, utility classes.
> - [`zod.md`](guides://zod.md) — runtime schema validation for forms, API responses, env.
> - [`performance.md`](guides://performance.md) · [`javascript.md`](guides://javascript.md) · [`observability.md`](guides://observability.md)

> 📎 **SEE ALSO:** [`vite.md`](guides://vite.md) · [`react-native.md`](guides://react-native.md) · [`error-handling.md`](guides://error-handling.md) · [`comments.md`](guides://comments.md)

---

## 1. Core Philosophies: REACT-FIRST

React-specific principles only. TDD, security, typing, accessibility, and performance policy come from §0.

- **R**eact-native first: use built-in primitives (Actions, `useActionState`, `useFormStatus`, `use()`, Suspense) before adding libraries; minimize dependencies.
- **E**xplicit boundaries: Server Components by default where the framework supports them; mark the client tree with `'use client'` and server functions with `'use server'` deliberately, not reflexively.
- **A**ctions over manual effects: prefer Actions and transitions for data writes and pending/optimistic UI instead of hand-rolled `useEffect` + `useState` fetch machinery.
- **C**omposition over inheritance: small, single-responsibility components composed via `children`, slots, and compound components — never class inheritance.
- **T**op-down data, lifted state: unidirectional data flow; lift state to the lowest common owner; derive, don't duplicate.
- **F**unction components only: hooks for all logic; class components and legacy lifecycle methods are forbidden in new code.
- **I**mmutable updates: never mutate state, props, or context; produce new objects/arrays so reconciliation is correct.
- **R**ules of Hooks: call hooks unconditionally at the top level; complete dependency arrays; clean up every subscription/effect.
- **S**erver state is not client state: cache remote data with a query library; keep global client state minimal.
- **T**rust the Compiler: let the React Compiler memoize; do not hand-scatter `useMemo`/`useCallback`/`memo` as premature optimization.

**Verified Code**: Agent-generated React MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `REACT-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| REACT-TST-01 | Every component/hook MUST be test-first (see `tdd.md`) | `npx vitest run` | exit 0, 0 skips |
| REACT-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `npx vitest run` | failing→passing |
| REACT-TST-03 | Tests MUST query by accessible role/label, not test IDs or class names (see `tdd.md`) | review / grep `getByTestId` | no role-able query replaced by testid |
| REACT-COMP-01 | Components MUST be function components; class components MUST NOT be added | `grep -RE "extends (React\\.)?Component"` | no matches |
| REACT-HOOK-01 | Rules of Hooks MUST hold; effect deps complete | `npx eslint . --max-warnings=0` (react-hooks plugin) | exit 0 |
| REACT-TYP-01 | Props/hooks/state MUST be fully typed, no `any` (see `typescript.md`) | `npx tsc --noEmit` | exit 0 |
| REACT-KEY-01 | List children MUST have stable, unique keys; index keys MUST NOT be used for reorderable lists | review / lint | no array-index keys on dynamic lists |
| REACT-SEC-01 | `dangerouslySetInnerHTML` MUST sanitize untrusted HTML (see `secure-coding.md`) | `grep -n dangerouslySetInnerHTML` + review | only sanitized input |
| REACT-SEC-02 | 0 high/critical CVEs in deps (see `secure-coding.md`) | `npm audit --audit-level=high` | 0 high/critical |
| REACT-SEC-03 | No secrets in client bundle; only `VITE_*`/public vars exposed | `grep -RE "(SECRET|PRIVATE|_KEY)"` src/ | no server secrets client-side |
| REACT-A11Y-01 | Interactive UI MUST be keyboard-operable & labelled (see `accessibility.md`) | axe / jest-axe in tests | 0 violations |
| REACT-FMT-01 | Code MUST be formatted | `npx biome format --check .` (or `prettier --check`) | no diff |
| REACT-LINT-01 | Linter MUST pass clean | `npx eslint . --max-warnings=0` | exit 0 |
| REACT-DEP-01 | Lockfile in sync & reproducible (see `secure-coding.md`) | `npm ci` | installs from lock, no drift |
| REACT-BUILD-01 | Production build MUST succeed | `npm run build` | exit 0 |

> **Forbidden**: class components or legacy lifecycle in new code; mutating state/props/context; conditional hook calls; `dangerouslySetInnerHTML` on unsanitized input; effects used as a substitute for derived state or event handlers; secrets shipped to the client; shipping implementation before its test (violates `tdd.md`).

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
npx biome format --check .          # REACT-FMT-01  (or: npx prettier --check .)
npx eslint . --max-warnings=0       # REACT-LINT-01 + REACT-HOOK-01 (react-hooks rules)
npx tsc --noEmit                    # REACT-TYP-01  (ESLint does NOT type-check)
npx vitest run --coverage           # REACT-TST-01/02/03
npm audit --audit-level=high        # REACT-SEC-02
npm run build                       # REACT-BUILD-01
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Feature-first layout. Cross-cutting *architecture* (dependency direction, layering) belongs to your app's architecture guide; below is the React mapping. For SSR/RSC/file-routing structure, defer to [`nextjs.md`](guides://nextjs.md).

```
src/
├── components/
│   ├── ui/                # design-system primitives (Button/, Input/, Card/)
│   └── <feature>/         # feature-scoped components
├── hooks/                 # reusable custom hooks (use*.ts)
├── lib/                   # framework-agnostic helpers, api client, query client
├── routes/ or pages/      # route components (or app/ for Next.js App Router)
├── stores/                # client state stores (only if truly global)
├── types/                 # shared TS types
└── main.tsx               # entry; mounts <App/>
```

- One component per directory with co-located test and `index.ts` barrel: `Button/{Button.tsx, Button.test.tsx, index.ts}`.
- Group by feature/domain, not by type. Keep components small and single-responsibility.
- Custom hooks own reusable stateful logic; components own rendering.

---

## 5. Component & Composition Patterns

The unique value of this guide. Type props with `interface`/`type` per [`typescript.md`](guides://typescript.md) — those rules are not restated here.

### A. Function components only
Class components and lifecycle methods (`componentDidMount`, etc.) are legacy — do not write them. Avoid `React.FC`; type props explicitly as a destructured parameter.

```tsx
interface UserCardProps {
  user: User;
  onEdit?: (id: string) => void;
  className?: string;
}

export function UserCard({ user, onEdit, className }: UserCardProps) {
  return (
    <div className={className}>
      <img src={user.avatar ?? '/default-avatar.png'} alt={`${user.name}'s avatar`} />
      <h3>{user.name}</h3>
      {onEdit && <button type="button" onClick={() => onEdit(user.id)}>Edit</button>}
    </div>
  );
}
```

### B. Composition over inheritance
Compose with `children`, render props, slots, and **compound components** sharing state via context. Extend native element props (`extends React.ButtonHTMLAttributes<HTMLButtonElement>`) instead of re-declaring `onClick`/`disabled`. Use discriminated unions to make impossible prop combinations unrepresentable.

```tsx
const CardCtx = React.createContext<{ variant: 'default' | 'elevated' } | null>(null);
const useCard = () => {
  const c = React.useContext(CardCtx);
  if (!c) throw new Error('Card.* must be used within <Card>');
  return c;
};

export function Card({ variant = 'default', children }: CardProps) {
  return <CardCtx.Provider value={{ variant }}><div className={`card-${variant}`}>{children}</div></CardCtx.Provider>;
}
Card.Header = ({ children }: PropsWithChildren) => <div className={`hdr-${useCard().variant}`}>{children}</div>;
Card.Body = ({ children }: PropsWithChildren) => <div className="card-body">{children}</div>;
// <Card variant="elevated"><Card.Header>…</Card.Header><Card.Body>…</Card.Body></Card>
```

> Component design, slot/variant APIs, and design tokens are owned by [`ui.md`](guides://ui.md); styling strategy by [`css.md`](guides://css.md).

### C. Controlled vs uncontrolled
- **Controlled**: value lives in React state (`value` + `onChange`). Use when React must read/validate/transform input each keystroke, or coordinate multiple fields.
- **Uncontrolled**: value lives in the DOM; read via a `ref` or form submission. Use for simple/large forms where per-keystroke re-renders are wasteful. Set initial value with `defaultValue`.
- Never switch a field between controlled and uncontrolled at runtime (passing `value={undefined}` then a string warns and loses state). Pick one for the field's lifetime.

### D. Keys & reconciliation
React diffs siblings by **key**. Keys MUST be stable, unique, and tied to data identity (`item.id`), never the array index for lists that can reorder/insert/delete — index keys corrupt state and inputs on reorder (REACT-KEY-01). A changing `key` is a deliberate tool to force-remount a subtree and reset its state. Do not put keys on non-list elements.

### E. ref-as-prop (React 19)
In React 19 `ref` is an ordinary prop — `forwardRef` is no longer required for new components and is deprecated. Type it directly:

```tsx
function TextInput({ ref, ...props }: React.ComponentProps<'input'> & { ref?: React.Ref<HTMLInputElement> }) {
  return <input ref={ref} {...props} />;
}
```

---

## 6. Hooks

### A. Rules of Hooks (non-negotiable)
Call hooks only at the **top level** of a component or another hook, **unconditionally** and in a stable order — never inside conditions, loops, or after an early `return`. Enforced by `eslint-plugin-react-hooks` (REACT-HOOK-01).

### B. State: `useState` and `useReducer`
Type state explicitly; use functional updates when the next value depends on the previous one (`setCount(c => c + 1)`). Reach for `useReducer` when state has multiple interdependent fields or a state machine (`idle|loading|success|error`) — model transitions with a discriminated-union action type.

### C. `useEffect` discipline — last resort, not default
Effects are for **synchronizing with external systems** (subscriptions, timers, non-React widgets, imperative DOM APIs), not for transforming data or responding to events. Before writing an effect, ask:
- Deriving a value from props/state? Compute it during render (or `useMemo`), not in an effect.
- Responding to a user action? Do it in the event handler.
- Fetching data on mount? Prefer a query library, a Server Component, or a route loader over a raw fetch effect.

Every effect MUST list complete dependencies and clean up (cancel async work, remove listeners, clear timers):

```tsx
useEffect(() => {
  const ctrl = new AbortController();
  api.getUser(userId, { signal: ctrl.signal }).then(setUser).catch(/* ignore aborts */);
  return () => ctrl.abort();         // cleanup
}, [userId]);                         // complete deps
```

### D. Refs and context
- `useRef` for mutable values that must survive re-renders without triggering them (DOM nodes, timer IDs, previous values). Mutating `ref.current` does not re-render.
- `useContext` for low-frequency, widely-read values (theme, locale, auth). Context is **not** a state manager — every consumer re-renders when the value changes; split contexts or memoize the provider value to limit churn. See §8 for when to reach past context to an external store.

### E. Custom hooks
Extract reusable stateful logic into `useX` functions returning typed tuples/objects. A custom hook composes other hooks; it does not render. Keep them pure of side effects beyond the hooks they call.

```tsx
function useDebounce<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(id);
  }, [value, delay]);
  return debounced;
}
```

---

## 7. React 19 Actions & async UI

Actions are async transitions that React tracks for pending state, errors, and optimistic updates. Prefer them over manual `isLoading`/`error` `useState` plumbing for data **writes**.

```tsx
// useActionState: form action + pending + returned state, no manual loading flags
function UpdateName() {
  const [error, submit, isPending] = useActionState(
    async (_prev: string | null, formData: FormData) => {
      const res = await updateName(formData.get('name') as string);
      return res.ok ? null : 'Update failed';
    },
    null,
  );
  return (
    <form action={submit}>
      <input name="name" />
      <SubmitButton />
      {error && <p role="alert">{error}</p>}
    </form>
  );
}

// useFormStatus: read the parent <form> action's pending state from a child
function SubmitButton() {
  const { pending } = useFormStatus();
  return <button type="submit" disabled={pending}>{pending ? 'Saving…' : 'Save'}</button>;
}
```

- **`useOptimistic`** — render an optimistic value while an Action is in flight; React reverts automatically if it rejects.
- **`use(promise)`** — unwrap a promise (or context) during render; suspends until resolved. Pair with `<Suspense>` (§9). The promise should come from a cache/Server Component, not be created inline in a Client Component each render.
- **`useTransition`** — mark non-urgent state updates so typing/clicks stay responsive; gives an `isPending` flag for the slow update.

---

## 8. Server Components & boundaries

(Requires a framework that implements RSC — see [`nextjs.md`](guides://nextjs.md).)

- **Server Components** are the default in RSC frameworks: they run only on the server, can be `async`, fetch data directly, and ship **zero JS** to the client. Keep data fetching, secrets, and heavy dependencies here.
- **`'use client'`** marks the boundary where the client bundle begins. Everything imported into a `'use client'` module becomes client code. Push the directive **as far down the tree as possible** — a leaf that needs `useState`/event handlers — so most of the tree stays server-rendered.
- **`'use server'`** marks **Server Actions**: server functions callable from client components (e.g. as a `<form action>`). Treat their arguments as untrusted input — validate (e.g. with [`zod.md`](guides://zod.md)) and authorize on the server (see `secure-coding.md`). Never assume client-side checks ran.
- Server Components cannot use hooks, state, effects, or browser APIs; Client Components cannot be `async`. Pass data from server to client via serializable props; pass Server Components into Client Components via `children` to keep them server-rendered.

---

## 9. State management & data

Choose the *narrowest* tool:

1. **Local state** (`useState`/`useReducer`) — default for anything one component or its subtree owns.
2. **Lift state up** to the lowest common parent when siblings must share it; pass down via props.
3. **Context** — for low-frequency global values (theme, auth, locale) read by many components. Not for high-churn state (causes wide re-renders).
4. **External store** (Zustand, Redux Toolkit, Jotai) — only when global client state is high-frequency, complex, or read by distant components where context would thrash. Select narrow slices to avoid over-rendering.
5. **Server state** (TanStack Query, RTK Query, or RSC/route loaders) — for all remote data. Do **not** mirror server data into a global store; let the cache own it (staleness, refetch, dedup, optimistic mutations). Keep server state separate from client state.

**Suspense & Error Boundaries** are the declarative async/error model:
- Wrap lazy/`use()`/data-fetching subtrees in `<Suspense fallback={…}>` for loading UI; combine with `React.lazy` + dynamic `import()` for code-splitting.
- Wrap subtrees in an **Error Boundary** (a class today, or `react-error-boundary`'s `<ErrorBoundary>`) to catch render-time errors and show fallback UI. Error boundaries do not catch event-handler or async errors — handle those per [`error-handling.md`](guides://error-handling.md).

Forms: prefer native form Actions (§7); for complex client validation use React Hook Form + a [`zod.md`](guides://zod.md) resolver. Validate the same schema on the server for any Server Action.

---

## 10. Performance & the React Compiler

Performance *policy* and budgets are owned by [`performance.md`](guides://performance.md); React bindings:

- **React Compiler first.** Enable the React Compiler (Babel/SWC plugin or framework flag). It auto-memoizes components and values, so **do not** pre-scatter `useMemo`/`useCallback`/`React.memo` — write idiomatic code and let the compiler optimize. Add manual memoization only when a profiler shows a real hot path the compiler missed, and keep components Rules-of-Hooks-clean so the compiler can apply.
- **Code-split** routes and heavy/rarely-used components with `React.lazy` + `<Suspense>`.
- **Virtualize** long lists (`@tanstack/react-virtual`) instead of rendering thousands of nodes.
- **Avoid render-time work**: don't create new objects/arrays/functions passed as props in ways that defeat memoization; keep keys stable (§5.D); never compute expensive values in render without memoization.
- Measure with the React DevTools Profiler before optimizing; render counts and commit duration, not guesses.

---

## 11. Security binding

XSS and supply-chain *policy* are owned by [`secure-coding.md`](guides://secure-coding.md). React bindings:

- React escapes interpolated text by default — `{userInput}` is safe. The one escape hatch, `dangerouslySetInnerHTML`, MUST receive sanitized HTML (e.g. `DOMPurify.sanitize(html)`), never raw user input (REACT-SEC-01).
- Validate untrusted data (API responses, URL params, Server Action args, env) at the boundary with [`zod.md`](guides://zod.md).
- Anything in the client bundle is public: only expose `VITE_*` (Vite) / `NEXT_PUBLIC_*` (Next.js) variables; keep secrets server-side (REACT-SEC-03). Env access goes through a validated, typed schema.
- Avoid injecting user-controlled values into `href="javascript:…"`, `<a target="_blank">` without `rel="noopener"`, or dynamic `<script>`/style injection.

---

## 12. Testing binding

Test-first policy and coverage gates are owned by [`tdd.md`](guides://tdd.md); end-to-end flows by [`e2e-testing.md`](guides://e2e-testing.md). React bindings:

- **Vitest + React Testing Library** for component/hook tests; `@testing-library/user-event` for interaction. Query by accessible role/label (`getByRole('button', { name: … })`), not test IDs or class names (REACT-TST-03) — this tests behavior and a11y simultaneously.
- Use `findBy*`/`waitFor` for async UI; assert on what the user sees, not implementation details/state internals.
- Test custom hooks via `renderHook`; wrap with required providers (QueryClient, router, context) in a shared test helper.
- Run `jest-axe`/`axe` assertions for accessibility (REACT-A11Y-01). Reserve full-browser tests for critical user journeys per [`e2e-testing.md`](guides://e2e-testing.md).

---

## 13. Tooling & Dependencies

Supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md). React binding (Vite + npm; CRA is deprecated — do not use it):

```bash
npm create vite@latest app -- --template react-ts   # scaffold (React 19 + TS)
npm ci                       # REACT-DEP-01: reproducible install from package-lock.json
npm install <pkg>            # add dep (updates lockfile) — commit package-lock.json
npm audit --audit-level=high # REACT-SEC-02: CVE scan (0 high/critical)
npm run build                # REACT-BUILD-01: production build
```

Enable the React Compiler in the build (Vite React plugin / `babel-plugin-react-compiler`). Use ESLint with `eslint-plugin-react-hooks` and Biome (or Prettier) for formatting. Commit `package-lock.json`; use `npm ci` in CI.

---

## 14. Quick Reference

```bash
npm run dev                          # Vite dev server (HMR)
npx vitest run --coverage            # test
npx eslint . --max-warnings=0        # lint (incl. rules of hooks)
npx tsc --noEmit                     # type check
npx biome format --write .           # format
npm run build && npm run preview     # production build + local preview
```

---

## 15. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] REACT-FMT-01 — formatter clean, no diff
- [ ] REACT-LINT-01 / REACT-HOOK-01 — ESLint clean, Rules of Hooks satisfied
- [ ] REACT-TYP-01 — `tsc --noEmit` clean, no `any`
- [ ] REACT-TST-01/02/03 — tests pass, bugs have regression tests, queried by role
- [ ] REACT-COMP-01 — function components only, no class components
- [ ] REACT-KEY-01 — stable unique keys, no index keys on dynamic lists
- [ ] REACT-A11Y-01 — keyboard-operable & labelled, axe clean
- [ ] REACT-SEC-01 — `dangerouslySetInnerHTML` sanitized
- [ ] REACT-SEC-02 — `npm audit` 0 high/critical
- [ ] REACT-SEC-03 — no secrets in client bundle
- [ ] REACT-DEP-01 — `package-lock.json` in sync, `npm ci` reproducible
- [ ] REACT-BUILD-01 — production build succeeds
- [ ] Agent ran every §3 command and documented any fixes

---
**End of React Guidelines**
