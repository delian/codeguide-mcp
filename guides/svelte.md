# Svelte Development Guidelines
Mandatory coding standards for Svelte: runes-first reactivity, type-safe, accessible, test-covered. Svelte 5, SvelteKit 2, Vite, TypeScript 5, Vitest, Playwright, svelte-check.

---
name: svelte
title: Svelte Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: framework
tools: [svelte@5, sveltekit@2, vite, typescript@5, vitest, playwright, svelte-check, eslint, prettier]
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
  - vite
  - performance
provides:
  - svelte5-runes
  - snippets
  - sveltekit
  - svelte-actions
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Svelte 5 and SvelteKit 2.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Svelte code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`typescript.md`](guides://typescript.md) — the language; strict config, `any` ban, generics, TSDoc. *(Svelte binding: `<script lang="ts">`, `svelte-check`, generated `./$types`.)*
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-before-fix, coverage. *(Svelte binding: `vitest` + `@testing-library/svelte`; E2E via Playwright.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(Svelte binding: `npm audit`, `{@html}`/XSS discipline, SvelteKit CSRF.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`accessibility.md`](guides://accessibility.md) — a11y policy *(binding: Svelte emits **compile-time a11y warnings** — treat them as errors)*
> - [`e2e-testing.md`](guides://e2e-testing.md) — Playwright flows *(binding: `@playwright/test`)*
> - [`vite.md`](guides://vite.md) — SvelteKit's build/dev server is Vite.
> - [`css.md`](guides://css.md) · [`ui.md`](guides://ui.md) — scoped `<style>`, design/UX.
> - [`javascript.md`](guides://javascript.md) · [`performance.md`](guides://performance.md)

> 📎 **SEE ALSO:** [`hexagonal.md`](guides://hexagonal.md) · [`cleanarch.md`](guides://cleanarch.md) *(layering for non-trivial apps — keep domain logic out of `.svelte` files)* · [`zod.md`](guides://zod.md) *(form/load validation)*

---

## 1. Core Philosophies: SVELTE-FIRST

Svelte-specific principles only. TDD, security, typing, and architecture come from §0.

- **S**ignals/runes first: `$state`, `$derived`, `$effect`, `$props`, `$bindable` are the default reactivity model. Legacy `$:`, `export let`, `on:`, and `<slot>` are Svelte 4 — use only in unmigrated code, **never mix** the two models in one component.
- **V**iew compiles away: write declarative markup; the compiler emits fine-grained DOM updates — no virtual DOM, no manual subscriptions.
- **E**vents as attributes: `onclick={fn}`, not `on:click`. Svelte 5 removed event modifiers (`|preventDefault`) — call `event.preventDefault()` in the handler.
- **L**ean components: small, prop-driven, single-responsibility; lift shared logic into `.svelte.ts` rune modules; keep business logic out of components.
- **T**yped & checked: every component is `lang="ts"`; gate with `svelte-check` (the real checker — ESLint does **not** type-check).
- **E**scape-by-default rendering: Svelte auto-escapes `{expr}`; `{@html}` is the only XSS hole (see `secure-coding.md`).

**Verified Code**: Agent-generated Svelte MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `SV-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| SV-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `npm run test` (vitest) | exit 0, 0 skips |
| SV-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `npm run test` | failing→passing |
| SV-TST-03 | Business-logic coverage MUST meet the project gate | `vitest run --coverage` | ≥ threshold |
| SV-E2E-01 | Critical user flows MUST have an E2E test (see `e2e-testing.md`) | `npx playwright test` | exit 0 |
| SV-TYP-01 | Code MUST type-check; no `any` (see `typescript.md`) | `npm run check` (`svelte-check`) | 0 errors |
| SV-RUNE-01 | New components MUST use runes; no legacy `$:`/`export let` | `svelte-check` + review | no legacy in new code |
| SV-EVT-01 | Events MUST use attribute form (`onclick`), not `on:click` | review / eslint-plugin-svelte | no `on:` directives |
| SV-A11Y-01 | 0 compile-time a11y warnings (see `accessibility.md`) | `npm run check` | 0 a11y warnings |
| SV-FMT-01 | Code MUST be formatted | `prettier --check .` (prettier-plugin-svelte) | no diff |
| SV-LINT-01 | Linter MUST pass clean | `eslint .` (eslint-plugin-svelte) | exit 0 |
| SV-SEC-01 | `{@html}` MUST NOT render untrusted input (see `secure-coding.md`) | review / grep `{@html}` | sanitized only |
| SV-SEC-02 | 0 high/critical CVEs (see `secure-coding.md`) | `npm audit --audit-level=high` | 0 high/critical |
| SV-DEP-01 | Lockfile committed & in sync | `npm ci` | installs clean |
| SV-BUILD-01 | Production build MUST succeed | `npm run build` | exit 0 |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`); fixing a bug without a regression test first; mixing runes with `$:`/`export let` in one component; mutating `$state` inside `$derived`; rendering untrusted data through `{@html}`; ignoring a compile-time a11y warning.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
npx prettier --check .              # SV-FMT-01
npx eslint .                        # SV-LINT-01
npm run check                       # SV-TYP-01 + SV-A11Y-01 (svelte-check; NOT a substitute by eslint)
npm run test -- --coverage          # SV-TST-01/02/03
npx playwright test                 # SV-E2E-01
npm audit --audit-level=high        # SV-SEC-02
npm run build                       # SV-BUILD-01
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

SvelteKit's file-based layout. Architectural *principles* (layering, dependency direction) are owned by [`hexagonal.md`](guides://hexagonal.md)/[`cleanarch.md`](guides://cleanarch.md) — below is only the Svelte mapping. Keep domain/use-case logic in plain `.ts`/`.svelte.ts`; components stay thin.

```
src/
├── lib/                      # importable as $lib
│   ├── components/           # *.svelte + co-located *.test.ts
│   ├── stores/               # global state: *.svelte.ts (runes) or svelte/store
│   ├── server/               # server-only code ($lib/server, never shipped to client)
│   └── domain/               # framework-free business logic (testable without a DOM)
├── routes/                   # file-based routing
│   ├── +layout.svelte / +layout.ts
│   ├── +page.svelte / +page.ts / +page.server.ts
│   ├── +server.ts            # API endpoints (GET/POST/…)
│   └── +error.svelte
├── app.html · app.css · app.d.ts
├── static/                   # served as-is
├── svelte.config.js · vite.config.ts · tsconfig.json
└── e2e/                      # Playwright (see e2e-testing.md)
```

- Co-locate component tests; group routes by feature.
- `$lib/server/*` is import-guarded by SvelteKit — secrets and DB code go there.

---

## 5. Svelte 5 Runes (the reactivity model)

The unique heart of the guide. All examples are modern Svelte 5.

### A. `$state` — reactive source of truth
```svelte
<script lang="ts">
  let count = $state(0);
  let user = $state<User | null>(null);
  let items = $state<Item[]>([]);     // deep-reactive proxy
</script>
```
- Mutation **works** on `$state` proxies — `items.push(x)` and `user!.name = 'x'` are reactive (a Svelte 5 change from Svelte 4, where reassignment was required).
- `$state.raw(obj)` — non-proxied; only whole-value reassignment is reactive (use for large/immutable data; replaces the old `$state.frozen`).
- `$state.snapshot(reactive)` — plain non-reactive clone (for logging, `structuredClone`, passing to non-Svelte libs).

### B. `$derived` — computed values
```svelte
<script lang="ts">
  let count = $state(0);
  let doubled = $derived(count * 2);                 // expression form
  let active = $derived.by(() => items.filter(i => i.active));  // block form
</script>
```
- Lazy, memoized, dependency-tracked automatically. **Never** mutate state inside `$derived` (read-only). Reassigning a `$derived` value is allowed as optimistic local override.

### C. `$effect` — side effects only
```svelte
<script lang="ts">
  $effect(() => {
    document.title = `Count: ${count}`;
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);          // cleanup runs before re-run / on unmount
  });
</script>
```
- For DOM, subscriptions, analytics — **not** for deriving state (use `$derived`). Mutating a tracked dependency inside its own effect is an infinite loop.
- `$effect.pre(fn)` runs before DOM update; `untrack(() => x)` reads without subscribing; `$effect.root(fn)` creates a manually-disposed scope (advanced/outside components).
- Do **not** reach for `$effect` to sync props↔state or fetch data — prefer `$derived`, load functions (§7), or event handlers.

### D. `$props` & `$bindable`
```svelte
<script lang="ts">
  interface Props {
    title: string;
    count?: number;
    value?: string;                 // two-way
    onsave?: (v: string) => void;   // callback prop replaces createEventDispatcher
    children?: import('svelte').Snippet;
    [key: string]: unknown;         // rest / spread-through
  }
  let { title, count = 0, value = $bindable(''), onsave, children, ...rest }: Props = $props();
</script>

<div {...rest}>{title}{@render children?.()}</div>
```
- Props are **read-only** unless declared `$bindable`. Parent opts in with `bind:value`.
- Callback props (`onsave`) replace Svelte 4's `createEventDispatcher` (deprecated). Component events are just function props.

### E. Reactive logic outside components — `.svelte.ts`
Runes work in any module named `*.svelte.ts`. Export reactive state/factories instead of building a global store:
```ts
// counter.svelte.ts
export function createCounter(initial = 0) {
  let count = $state(initial);
  return {
    get count() { return count; },
    increment() { count++; },
    reset() { count = 0; },
  };
}
```
This is the modern replacement for most custom stores; getters expose reactivity across module boundaries.

---

## 6. Components, Snippets, Bindings & Events

### A. Snippets replace slots
`{#snippet}` + `{@render}` are the Svelte 5 composition primitive (slots are legacy).
```svelte
{#snippet row(item: Item)}
  <li>{item.name}</li>
{/snippet}

<ul>
  {#each items as item (item.id)}
    {@render row(item)}
  {/each}
</ul>
```
- Default children: `{@render children?.()}`. Named slots → named snippet props passed as attributes (`{#snippet header()}…{/snippet}` on the child element). Snippets are typed (`Snippet<[Item]>`) and can be passed as props.
- `<slot>` / `<slot name="x">` / `$$slots` are Svelte 4 — migrate to snippets.

### B. Bindings
```svelte
<input bind:value={name} />            <!-- form element two-way -->
<input type="checkbox" bind:checked={done} />
<details bind:open><…></details>
<div bind:clientWidth={w} bind:this={el}></div>   <!-- read-only + element ref -->
<Child bind:value />                   <!-- requires $bindable in Child -->
```
Prefer one-way data flow + callbacks; reach for `bind:` for genuine two-way (form fields, `bind:this`).

### C. Events (attribute form)
```svelte
<button onclick={() => count++}>+</button>
<form onsubmit={(e) => { e.preventDefault(); save(); }}>…</form>
```
- No `on:click`, no `|preventDefault`/`|stopPropagation` modifiers (removed). Handle in the function; for capture/once use `onclickcapture` or `svelte/legacy` helpers only when migrating.

### D. Control flow & async
```svelte
{#if user}<p>{user.name}</p>{:else}<p>Guest</p>{/if}

{#each items as item, i (item.id)}<Row {item} />{:else}<p>Empty</p>{/each}

{#await userPromise}<Spinner />{:then user}<User {user} />{:catch e}<Err {e} />{/await}

{#key id}<Transitioned />{/key}
```
- Always key `{#each}` with a stable id. `{#await}` is the idiomatic reactive-promise pattern — bind it to a `$derived` promise so it re-fetches when inputs change. Use plain `async`/`await` in `<script>` and load functions (general async rules: see `javascript.md`).

### E. Transitions, animations & actions
```svelte
<script lang="ts">
  import { fade, fly } from 'svelte/transition';
  import { flip } from 'svelte/animate';
  import type { Action } from 'svelte/action';

  const tooltip: Action<HTMLElement, string> = (node, text) => {
    const show = () => {/* … */};
    node.addEventListener('mouseenter', show);
    return { destroy() { node.removeEventListener('mouseenter', show); } };
  };
</script>

{#if open}<div transition:fade={{ duration: 200 }}>…</div>{/if}
{#each rows as r (r.id)}<li animate:flip use:tooltip={r.hint}>{r.name}</li>{/each}
```
- `transition:` (in+out), `in:`/`out:` (one-way), `animate:` (keyed-list reordering). Respect reduced-motion (see `accessibility.md`).
- **Actions** (`use:fn`) are the reusable-DOM-behavior primitive (focus traps, click-outside, integrating non-Svelte libs); this guide owns `svelte-actions` — typed via `Action<El, Param>`.

---

## 7. SvelteKit (routing, loading, forms, endpoints)

### A. Load functions
```ts
// +page.ts — universal: runs on server (SSR) then client (CSR/navigation)
import type { PageLoad } from './$types';
export const load: PageLoad = async ({ params, fetch, parent }) => {
  const res = await fetch(`/api/users/${params.id}`);   // use the provided fetch (SSR-aware)
  if (!res.ok) throw error(404, 'Not found');
  return { user: await res.json() };
};
```
- `+page.ts`/`+layout.ts` = **universal** (no secrets, runs both sides). `+page.server.ts`/`+layout.server.ts` = **server-only** (DB, secrets, `cookies`, `locals`). Return data flows into the page as the `data` prop; `+layout` data is inherited.
- `import { error, redirect, fail } from '@sveltejs/kit'` for control flow. Validate inputs with [`zod.md`](guides://zod.md).

### B. Form actions + progressive enhancement
```ts
// +page.server.ts
import type { Actions } from './$types';
import { fail } from '@sveltejs/kit';
export const actions = {
  create: async ({ request }) => {
    const data = await request.formData();
    const email = String(data.get('email'));
    if (!email.includes('@')) return fail(400, { email, error: 'Invalid email' });
    /* … */ return { success: true };
  },
} satisfies Actions;
```
```svelte
<script lang="ts">
  import { enhance } from '$app/forms';
  let { form } = $props();             // ActionData
  let submitting = $state(false);
</script>
<form method="POST" action="?/create" use:enhance={() => {
  submitting = true;
  return async ({ update }) => { await update(); submitting = false; };
}}>…</form>
```
- Forms work without JS (real `method="POST"`); `use:enhance` upgrades to client-side. SvelteKit provides **CSRF protection** for actions by default (see `secure-coding.md`).

### C. Endpoints (`+server.ts`)
```ts
import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
export const GET: RequestHandler = async ({ params }) => {
  const u = await getUser(params.id);
  return u ? json(u) : error(404, 'Not found');
};
export const POST: RequestHandler = async ({ request }) => json(await create(await request.json()), { status: 201 });
```

### D. Runtime state & SSR
- Use `page`, `navigating`, `updated` from **`$app/state`** (rune-based, Svelte 5). The `$app/stores` `$page` store form is legacy/deprecated in SvelteKit 2.
- SSR is on by default; opt out per-route with `export const ssr = false` / `csr = false` / `prerender = true`. Code touching `window`/`document` must run in `$effect`/`onMount` or behind `if (browser)` (`$app/environment`) — touching them top-level breaks SSR (hydration mismatch).

---

## 8. Stores (legacy & interop)

Runes (`$state` in `.svelte.ts`, §5.E) are preferred for new shared state. The `svelte/store` contract remains for interop (e.g. SvelteKit's own stores, RxJS bridges):
```ts
import { writable, readable, derived } from 'svelte/store';
export const count = writable(0);
export const doubled = derived(count, ($c) => $c * 2);
```
- In components, `$count` auto-subscribes and auto-unsubscribes — still valid. A store is anything with a compliant `subscribe`. Use `setContext`/`getContext` for component-tree-scoped state (typed via a key symbol). Prefer migrating bespoke custom stores to rune factories.

---

## 9. Tooling & Dependencies

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); the build/dev server is Vite → [`vite.md`](guides://vite.md). Svelte binding:

```bash
npm ci                       # SV-DEP-01: reproducible install from package-lock.json
npm run check                # SV-TYP-01/A11Y: svelte-check (sync types via `svelte-kit sync`)
npm audit --audit-level=high # SV-SEC-02
npm run build && npm run preview
```
- Commit `package-lock.json`; use `npm ci` in CI. Document public component props/`Props` interfaces with TSDoc (policy: see `typescript.md`).

---

## 10. Quick Reference

```bash
npm run dev                  # Vite dev server (localhost:5173)
npm run check                # svelte-check (types + a11y)
npm run test                 # vitest
npx playwright test          # E2E
npm run build                # production build
```

```svelte
<script lang="ts">
  let count = $state(0);                 // state
  let double = $derived(count * 2);      // computed
  $effect(() => { /* side effect */ });  // effect
  let { title, value = $bindable('') } = $props();   // props + two-way
</script>
{#snippet item(x: string)}<li>{x}</li>{/snippet}
<button onclick={() => count++}>{title}: {double}</button>
```

---

## 11. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] SV-FMT-01 — `prettier --check` clean
- [ ] SV-LINT-01 — `eslint` clean
- [ ] SV-TYP-01 — `svelte-check` 0 errors, no `any`
- [ ] SV-RUNE-01 — runes only; no legacy `$:`/`export let` in new code
- [ ] SV-EVT-01 — attribute events (`onclick`), no `on:` directives
- [ ] SV-A11Y-01 — 0 compile-time a11y warnings
- [ ] SV-TST-01/02/03 — tests pass, bugs have regression tests, coverage ≥ gate
- [ ] SV-E2E-01 — critical flows covered (Playwright)
- [ ] SV-SEC-01 — no `{@html}` of untrusted input
- [ ] SV-SEC-02 — `npm audit` 0 high/critical
- [ ] SV-DEP-01 — `package-lock.json` committed, `npm ci` clean
- [ ] SV-BUILD-01 — `npm run build` succeeds
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Svelte Guidelines**
