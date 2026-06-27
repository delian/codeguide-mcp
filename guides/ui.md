# UI/UX Engineering Guidelines
Framework-agnostic standards for building consistent, usable, accessible interfaces: component design, design systems & tokens, state/props patterns, layout & responsive design, interaction feedback, and loading/empty/error states.

---
name: ui
title: UI/UX Engineering Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: []
requires: []
recommends:
  - accessibility
  - css
  - html
  - material
  - designpatterns
provides:
  - component-design
  - design-systems
  - design-tokens
  - ui-states
  - responsive-design
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns *UI/UX engineering principles* that hold across React, Vue, Angular, Svelte, Flutter, SwiftUI, and Jetpack Compose — it does **not** restate accessibility, styling mechanics, semantics, a specific design system, or generic component patterns.

---

## 0. Prerequisites & References

This guide is framework- and language-agnostic. It assumes the rules below and binds to whatever stack the task uses.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`accessibility.md`](guides://accessibility.md) — WCAG, keyboard, ARIA, focus, contrast. **A11y is mandatory and owned there; this guide never restates it.**
> - [`css.md`](guides://css.md) — styling mechanics: layout primitives (flex/grid), custom properties, cascade, units, container queries.
> - [`html.md`](guides://html.md) — semantic structure, landmarks, native form controls.
> - [`material.md`](guides://material.md) — one concrete design system (Material 3) to adopt rather than invent.
> - [`designpatterns.md`](guides://designpatterns.md) — GoF/UI patterns (composite, observer, container/presentational, render-props).

> 📎 **SEE ALSO — framework & concern owners (fetch the one your stack uses):**
> - Frameworks: [`reactjs.md`](guides://reactjs.md) · [`angular.md`](guides://angular.md) · [`svelte.md`](guides://svelte.md) · [`nextjs.md`](guides://nextjs.md) · [`flutter.md`](guides://flutter.md) · [`react-native.md`](guides://react-native.md) · [`ios.md`](guides://ios.md) · [`android.md`](guides://android.md)
> - Concerns: [`tdd.md`](guides://tdd.md) (test-first) · [`e2e-testing.md`](guides://e2e-testing.md) (visual/flow tests) · [`performance.md`](guides://performance.md) (budgets, profiling) · [`websocket.md`](guides://websocket.md) (real-time transport) · [`rest.md`](guides://rest.md)/[`graphql.md`](guides://graphql.md) (data fetching) · [`error-handling.md`](guides://error-handling.md) (failure strategy)

---

## 1. Core Philosophies: CLEAN

UI/UX-specific principles only. Test-first (`tdd.md`), accessibility (`accessibility.md`), and performance (`performance.md`) are owned elsewhere and applied, not re-explained.

- **C**onsistent: identical patterns look and behave identically everywhere; one component per concept, sourced from a design system (`material.md` or your own tokens — see §4).
- **L**egible state: every async surface has explicit loading / empty / error / success states (§6); the user is never left guessing.
- **E**ffortless: the shortest path to the user's goal — sensible defaults, minimal required input, immediate feedback (§5), forgiving forms (§7).
- **A**daptive: one component tree adapts to viewport, input modality (touch/pointer/keyboard), orientation, and user preferences (reduced-motion, color-scheme) — see §3.
- **N**ative-leaning: prefer platform/native controls and a maintained design system over bespoke widgets; build custom only when no standard control fits (§4.A).

**Verified UI**: agent-generated UI MUST satisfy every gate in §2 — including the referenced a11y audit — before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `UI-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner. `<a11y-tool>` / `<test-runner>` / `<visual-tool>` are the stack's chosen tools (e.g. axe-core, jest+RTL/vitest, Playwright/Chromatic).

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| UI-A11Y-01 | UI MUST meet WCAG 2.2 AA (owned by `accessibility.md`) | `<a11y-tool>` audit (axe/pa11y/lighthouse) | 0 violations |
| UI-TST-01 | Components MUST be test-first (see `tdd.md`) | `<test-runner>` | exit 0, 0 skips |
| UI-TST-02 | Each UI bug MUST get a regression test before fix (see `tdd.md`) | `<test-runner>` | failing→passing |
| UI-VIS-01 | Visual/interaction changes MUST pass visual regression (see `e2e-testing.md`) | `<visual-tool>` snapshot diff | no unreviewed diff |
| UI-TOK-01 | Styling MUST use design tokens, never hardcoded values (see `css.md`) | review / grep for literal colors/px | no raw literals |
| UI-STATE-01 | Every async surface MUST render loading, empty, AND error states (§6) | review / story coverage | all states exist |
| UI-RESP-01 | Layout MUST adapt down to the min target width without overflow/clipping (§3) | resize/responsive check | no h-scroll, no clip |
| UI-MOTION-01 | Animations MUST honor `prefers-reduced-motion` (see `accessibility.md`) | review / reduced-motion run | motion gated |
| UI-NAV-01 | Page-level UI state (filters/sort/scroll) MUST survive navigation (§3) | manual back/forward test | state restored |
| UI-CONTRACT-01 | Component props MUST be typed/documented with explicit defaults (§4.B) | type check / prop-types / story | typed, documented |
| UI-PERF-01 | Long lists (>~100 rows) MUST virtualize; pages lazy-load heavy chunks (see `performance.md`) | review / profile | virtualized, split |

> **Forbidden**: shipping a control with no focus-visible/keyboard path (violates `accessibility.md`); an async view with no empty or error branch; hardcoded colors/spacing instead of tokens; fetching a full dataset for a paginated/virtualized view; selecting test elements by CSS class, text, or DOM position (§4.E); blocking the UI thread on a network round-trip with no loading affordance.

---

## 3. Layout, Responsive & State Topology

Styling *mechanics* (flex, grid, container queries, units, custom properties) are owned by [`css.md`](guides://css.md); semantic structure by [`html.md`](guides://html.md). This section owns the *UX decisions* layered on top.

### A. Responsive strategy
- Design **mobile-first**: the base layout targets the smallest supported width; breakpoints add capability as space allows.
- Prefer **intrinsic/content-driven** layout (grid auto-fit, flex-wrap, `min()/max()/clamp()`) over a fixed ladder of pixel breakpoints. Reach for **container queries** so a component adapts to its slot, not the global viewport.
- Adapt to **input modality**, not just size: ≥44×44px touch targets, hover affordances only as enhancement, keyboard parity for every pointer action (a11y owner: `accessibility.md`).
- Honor user preferences: `prefers-color-scheme`, `prefers-reduced-motion`, `prefers-contrast`, text-zoom up to 200% without loss of content (UI-RESP-01, UI-MOTION-01).

### B. State topology (where state lives)
Classify every piece of state and place it at the narrowest scope that works:

| Category | Examples | Lives in |
|----------|----------|----------|
| **Ephemeral/component** | input value, toggle, hover, accordion open | local component state |
| **Page/route** | active filters, sort, pagination, scroll, selected row | the **URL** (query/path) first; route state otherwise |
| **Application/global** | session, theme, feature flags, notifications | a single global store (one per app, not many) |
| **Server/cache** | fetched entities, real-time feeds | a data-fetching/cache layer (query lib) |

- **Lift state only as far as needed.** Co-locate by default; promote when two siblings must agree.
- **Page state belongs in the URL.** Filters, sort, search, and pagination encoded as query params make views shareable, bookmarkable, and back/forward-correct (UI-NAV-01). Reserve session/local storage for non-shareable view prefs (column widths, sidebar collapsed) and persistent prefs (theme) respectively.
- **One source of truth.** Never mirror server data into global state and then mutate both. Derive; don't duplicate.
- The concrete API for each (hooks/signals/stores) is owned by the framework guide — bind the idiom, don't reinvent it.

### C. Navigation & history
Page-level UI state MUST be restored on back/forward (UI-NAV-01). Drive it from the URL (see §3.B) so the platform's history does the work; restore scroll position on return to a list; preserve in-progress form input across accidental navigation where data loss would frustrate the user.

---

## 4. Component Design & Design Systems

This is the heart of the guide. Generic *patterns* (composite, container/presentational, observer) are owned by [`designpatterns.md`](guides://designpatterns.md); a concrete system (Material 3) by [`material.md`](guides://material.md). Below is the UI engineering doctrine.

### A. Widget selection priority
1. **Native / platform control** — best accessibility, performance, familiarity (e.g. `<select>`, `<dialog>`, SwiftUI `Picker`, Compose `Button`).
2. **Maintained design-system component** — Material/Fluent/HIG via the platform binding (see `material.md`).
3. **Reputable, accessible OSS library** — active, audited, keyboard-complete.
4. **Custom** — only when 1–3 cannot meet the requirement, and then built to the same a11y/contract bar.

### B. The component contract
A component is an API. Treat its props as a typed, documented contract:
- **Typed props with explicit defaults**; required vs optional is unambiguous (UI-CONTRACT-01). Bind to the stack's type system (TS types, prop-types, Flutter `required`, Swift initializers).
- **Single responsibility**: presentational components render from props and emit events; they don't fetch, route, or own global state. Push side-effects to container/hook layers (pattern owner: `designpatterns.md`).
- **Controlled vs uncontrolled** is a deliberate choice, not an accident — pick one per prop and document it.
- **Composition over configuration**: prefer slots/children/`@ContentChild` over a `boolean`-prop explosion. A component with >~8 boolean flags is two components.
- **Events out, data in**: parents own state; children receive values and report intent via callbacks/events.

### C. Design systems & tokens — OWNED
A design system is the single source of visual/behavioral truth. Tokens are its primitives. **All styling MUST reference tokens, never raw literals** (UI-TOK-01).

Three-tier token hierarchy:
```
Primitive (raw)        Semantic (intent)            Component (binding)
color.blue.600    →    color.action.primary    →    button.primary.bg
space.4 (16px)    →    space.inset.md          →    card.padding
font.size.300     →    text.body.size          →    input.label.size
```
- **Primitive** tokens name raw values; never consumed directly by components.
- **Semantic** tokens name *intent* (`action.primary`, `surface.raised`, `text.danger`) — this is the layer components use. Theming/dark-mode swaps semantic→primitive mappings, not component code.
- **Component** tokens bind a component slot to semantic tokens.
- Cover: color, typography scale, spacing scale, radii, elevation/shadow, motion durations/easings, z-index layers, breakpoints. The *delivery mechanism* (CSS custom properties, CSS-in-JS, Tailwind config, platform theme) is owned by `css.md` / the platform — tokens are agnostic of it.

### D. Consistency
- One component per concept across the whole app — no three subtly different "primary buttons."
- Spacing and sizing come from the scale; never a one-off `13px`.
- Iconography, copy tone, empty-state illustrations, and interaction timings are part of the system, not per-screen decisions.
- Document components (props, variants, states, do/don't) in a living catalog (Storybook or platform equivalent); a component without a story for each state is incomplete (supports UI-STATE-01).

### E. Stable test selectors — OWNED
Interactive elements (buttons, inputs, links, rows) MUST carry a **stable, semantic test identifier** so automated tests don't bind to fragile attributes. Selectors MUST NOT be CSS classes, tag names, DOM position/index, visible text, or auto-generated IDs (those break on restyle, i18n, refactor).

- **Web default**: `data-testid` in kebab-case, hierarchically namespaced — e.g. `checkout-form-submit-button`, `product-list-item-42-add-to-cart-button`. Respect an existing project convention (`data-cy`, `data-qa`) if one is established.
- **Flutter**: `Key('submit-order-button')`. **SwiftUI**: `.accessibilityIdentifier("submit-order-button")`. **Jetpack Compose**: `Modifier.testTag("submit-order-button")`.
- Naming pattern: `[context]-[element]-[type]`; type suffixes `-button`, `-input`, `-link`, `-select`, `-checkbox`, `-modal`, `-list`, `-item`, `-card`, `-row`. Where the testing framework reads accessibility roles/names directly (e.g. Testing Library `getByRole`), prefer that — it doubles as an a11y check.

---

## 5. Interaction & Feedback

Every user action gets timely, proportional feedback — this is owned UX doctrine.

- **Acknowledge immediately.** A click/tap shows an effect within ~100ms (pressed state, spinner on the control). Beyond ~1s, show progress; beyond ~10s, show a determinate or cancelable indicator.
- **Optimistic UI** for high-confidence mutations: apply the change locally at once, reconcile with the server, and **roll back with a clear message** on failure. Never leave the UI showing a success that didn't happen.
- **Affordance & state**: interactive elements expose `hover`, `focus-visible`, `active`, `disabled`, `selected`, and `loading` states. Disabled controls explain *why* (tooltip/helper text) rather than silently dead-ending.
- **Confirmation vs undo**: prefer **undo** (a forgiving toast) over a blocking confirm dialog for reversible actions; reserve confirmation for destructive, irreversible ones.
- **Motion is communication**: animate to show causality and continuity (where did this come from / go to), not decoration. Keep transitions short (~150–300ms) and **gate all of it behind `prefers-reduced-motion`** (UI-MOTION-01, owner `accessibility.md`).
- **Notifications**: transient (toast) for non-blocking confirmations; inline for field-scoped messages; banner/modal only for app-level or blocking conditions. Don't stack modals.

---

## 6. Loading, Empty, Error & Success States — OWNED

**Every surface that depends on async data MUST handle all of these states explicitly** (UI-STATE-01). A view that only renders the happy path is incomplete.

- **Loading**: prefer **skeletons** that mirror final layout over spinners for content regions (less layout shift, perceived as faster). Show progress for long/determinate work. Avoid flashing a loader for sub-~200ms fetches (delay it).
- **Empty**: distinguish *no data yet* from *no results for this filter* from *first-run/never-created*. Each empty state explains what it is and offers the next action (clear filters / create first item / learn more) — never a blank pane.
- **Error**: state what failed, whether it's the user's input or the system, and the recovery path (retry / go back / contact). Keep partial good data visible where possible; don't blow away the whole screen for one failed widget. Strategy for *propagating* errors is owned by [`error-handling.md`](guides://error-handling.md); this owns how they're *surfaced* to the user.
- **Success**: confirm completion (inline check, toast, state change) so the user knows the action took.
- **Partial / paginated**: show what's loaded, indicate more is coming (load-more / infinite scroll with a sentinel), and never block the whole view on the next page.

---

## 7. Forms UX — OWNED

Native control semantics are owned by [`html.md`](guides://html.md); a11y of labels/errors by [`accessibility.md`](guides://accessibility.md). The *usability* of forms is owned here.

- **Minimize input**: ask only for what's needed; sensible defaults; smart input types (`email`, `tel`, `numeric`) to surface the right keyboard; autofill/autocomplete tokens on.
- **Label everything visibly**; placeholders are hints, never labels. Group related fields; one column beats two for scan-ability.
- **Validate at the right time**: validate a field on **blur** (or as the user fixes a known-bad field), not on every keystroke from the first character; validate the whole form on submit. Show errors **inline, next to the field**, with concrete, fixable wording ("Password needs 8+ characters"), and move focus to the first error on a failed submit.
- **Submit state**: disable the submit during in-flight requests to prevent double submission and show a loading state on the button; re-enable on error with the message preserved and the user's input intact.
- **Forgiveness**: never clear a user's input on a server error; preserve in-progress data across accidental navigation; allow undo where feasible.
- **Destructive actions**: require explicit confirmation (typed name / hold-to-confirm) only for irreversible ones; otherwise prefer undo (§5).

---

## 8. Testing & Verification

The agent runs these before presenting UI; the *policy* behind each lives in its owner.

```bash
<a11y-tool> audit            # UI-A11Y-01   (axe-core / pa11y / lighthouse — owner: accessibility.md)
<test-runner>                # UI-TST-01/02 (RTL / Vitest / Jest / widget tests — owner: tdd.md)
<visual-tool> snapshot       # UI-VIS-01    (Playwright / Chromatic / Percy — owner: e2e-testing.md)
```
- **Component tests** assert each state branch (loading/empty/error/success) and keyboard interaction; query by accessible role/name where possible (doubles as a11y signal).
- **Visual regression** guards token/layout drift; review every snapshot diff before accepting.
- **E2E** covers critical flows (the full flow policy is owned by `e2e-testing.md`).
- If a gate fails: find the root cause, fix, re-run. Do not present until every §2 gate is green.

---

## 9. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements here.

- [ ] UI-A11Y-01 — a11y audit 0 violations (owner `accessibility.md`)
- [ ] UI-TST-01/02 — components test-first; bugs have regression tests
- [ ] UI-VIS-01 — visual regression reviewed, no unintended diff
- [ ] UI-TOK-01 — design tokens only, no hardcoded colors/spacing
- [ ] UI-STATE-01 — loading, empty, error (and success) states present on every async surface
- [ ] UI-RESP-01 — adapts to min width; no overflow/clipping; 200% text zoom OK
- [ ] UI-MOTION-01 — animation honors `prefers-reduced-motion`
- [ ] UI-NAV-01 — page state survives back/forward; scroll restored
- [ ] UI-CONTRACT-01 — props typed, defaulted, documented (story per state)
- [ ] UI-PERF-01 — long lists virtualized; heavy chunks lazy-loaded
- [ ] Agent ran every §8 command and documented any fixes

---
**End of UI/UX Engineering Guidelines**
