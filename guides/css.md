# CSS Development Guidelines
Mandatory standards for modern, maintainable, performant CSS: native nesting, cascade layers, container queries, custom properties, modern color. Baseline-aware CSS, stylelint 16, PostCSS / Lightning CSS.

---
name: css
title: CSS Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [css, stylelint@16, postcss@8, lightningcss, browserslist]
requires: []
recommends:
  - html
  - accessibility
  - ui
  - performance
provides:
  - modern-css-layout
  - custom-properties
  - container-queries
  - cascade-layers
  - modern-selectors
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to CSS as a language.

---

## 0. Prerequisites & References

CSS has no hard prerequisites, but styling is meaningless without the markup it targets and the design intent it serves. Fetch the relevant guide when the task touches its concern; this guide does not repeat their rules.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`html.md`](guides://html.md) — semantic structure CSS styles; prefer styling semantic elements/states over hook classes. *(Binding: style `:is(button, [role=button])`, `[aria-current]`, `:disabled` — not `.btn-active`.)*
> - [`accessibility.md`](guides://accessibility.md) — contrast, focus, motion policy. *(Binding: `:focus-visible` outlines, `prefers-reduced-motion`, `prefers-contrast`; contrast is a WCAG rule owned there, CSS only implements it.)*
> - [`ui.md`](guides://ui.md) — design systems, design tokens, theming strategy. *(Binding: tokens surface as custom properties; this guide owns the CSS mechanism, `ui.md` owns the token taxonomy.)*
> - [`performance.md`](guides://performance.md) — critical CSS, `content-visibility`, containment, render cost. *(Binding: `contain`, `content-visibility: auto`, animate only `transform`/`opacity`.)*

> 📎 **SEE ALSO:** [`markdown.md`](guides://markdown.md) · [`semver.md`](guides://semver.md) *(for distributed design-system/theme packages)*

---

## 1. Core Philosophies: CASCADE-FIRST

CSS-specific principles only. Accessibility, performance, and token taxonomy come from §0.

- **C**ascade over override: control the cascade with **layers and low specificity**, never `!important` wars. Specificity is a debugging cost, not a feature.
- **A**daptive by default: components respond to their **container** (`@container`), the document to the **viewport**; size and space fluidly with `clamp()`/`min()`/`max()`.
- **S**ystem of tokens: every color, space, font, and duration is a **custom property**, not a literal. No magic numbers in component rules.
- **C**omposable selectors: prefer `:is()`/`:where()`/`:has()` and attribute/state selectors over deep descendant chains; shallow, flat, intention-revealing.
- **A**gnostic to direction: **logical properties** (`margin-inline`, `inset-block`) so layouts mirror for RTL and vertical writing modes for free.
- **D**egrade gracefully: target **Baseline**; gate not-yet-Baseline features behind `@supports`; native CSS before a preprocessor, a preprocessor before a framework.
- **E**xplicit motion: every transition/animation has a `prefers-reduced-motion` escape hatch and animates only compositor-friendly properties.

**Verified Code**: Agent-generated CSS MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `CSS-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| CSS-FMT-01 | Stylesheets MUST be formatted | `npx stylelint "**/*.css" && npx prettier --check "**/*.css"` | no diff |
| CSS-LINT-01 | Linter MUST pass clean | `npx stylelint "**/*.{css,scss}"` | exit 0 |
| CSS-LINT-02 | No `!important` except documented utility/override layers | `stylelint declaration-no-important` | 0 violations |
| CSS-SPEC-01 | Selector specificity MUST stay low (no IDs, ≤ 0,3,0) | `stylelint selector-max-id / selector-max-specificity` | exit 0 |
| CSS-LAYER-01 | Global styles MUST be organized in `@layer` (no implicit-order reliance) | review / grep `@layer` | layered |
| CSS-CMPAT-01 | Features MUST be Baseline or `@supports`-gated for the browserslist target | `npx stylelint` + `stylelint-no-unsupported-browser-features` | 0 unguarded |
| CSS-A11Y-01 | Interactive elements MUST have a visible `:focus-visible` style (see `accessibility.md`) | grep `:focus-visible` / axe | present |
| CSS-A11Y-02 | Motion MUST honor `prefers-reduced-motion` (see `accessibility.md`) | grep `prefers-reduced-motion` | present where animated |
| CSS-A11Y-03 | Text/UI contrast MUST meet WCAG AA (see `accessibility.md`) | axe / contrast check | ≥ 4.5:1 text, 3:1 large/UI |
| CSS-PERF-01 | Animations MUST use only `transform`/`opacity` (see `performance.md`) | review | no layout-triggering animation |
| CSS-TOK-01 | No hardcoded colors/sizes in components; tokens only (see `ui.md`) | `stylelint` custom rule / review | no raw literals |

> **Forbidden**: `!important` to win a specificity fight (fix the cascade with `@layer`); ID selectors for styling; `outline: none` without a replacement focus style (violates `accessibility.md`); animating `width`/`height`/`top`/`left`; raw hex/px literals inside component rules; not-yet-Baseline features shipped without `@supports`.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
npx stylelint "**/*.{css,scss}"                  # CSS-LINT-01/02, SPEC-01, CMPAT-01
npx prettier --check "**/*.css"                  # CSS-FMT-01
npx lightningcss --browserslist input.css -o /dev/null   # CMPAT: transpile/parse check
# accessibility (CSS-A11Y-*) via axe / pa11y on a rendered page (see accessibility.md)
```

The *why* behind contrast, motion, and render-cost gates lives in their §0 owner; do not re-derive it here.

---

## 4. Project Structure & The Cascade

A flat, layer-ordered architecture. The cascade resolves conflicts in this priority order: **origin/importance → layers → specificity → source order**. Lean on the first two; keep the last two boring.

```
styles/
├── main.css            # entry: declares @layer order, then @imports
├── tokens.css          # @layer tokens — custom properties only (see ui.md)
├── reset.css           # @layer reset — modern reset
├── base.css            # @layer base — element defaults (typography, links)
├── layouts/            # @layer layouts — grid/flex page scaffolds
├── components/         # @layer components — one file per component
└── utilities.css       # @layer utilities — single-purpose overrides
```

```css
/* main.css — declare the order ONCE; later layers always win regardless of specificity */
@layer reset, base, tokens, layouts, components, utilities;

@import url("reset.css")      layer(reset);
@import url("base.css")       layer(base);
@import url("components/card.css") layer(components);
```

- **Cascade layers (`@layer`)** are the primary specificity-management tool: a `utilities` rule beats a `components` rule even at lower specificity, so you never need `!important` or ID hacks to override.
- **Specificity / inheritance:** keep selectors at class level (0,1,0). Use `:where()` (specificity 0) for resets and defaults so authors override them trivially; use `:is()` to group without the highest-specificity-wins surprise of `:matches`-era selectors.
- Group by component/feature, one block per file. No file reaches across to style another's internals.

---

## 5. CSS Specifics

The unique value of this guide — modern, Baseline-aware CSS.

### A. Layout — Flexbox, Grid, Subgrid
One-dimensional content distribution → Flexbox; two-dimensional structure → Grid; aligning a child's tracks to its parent's → Subgrid (Baseline 2023).

```css
.toolbar { display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; }

.gallery {                                   /* intrinsic responsive grid, no media query */
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(16rem, 100%), 1fr));
  gap: 1rem;
}

.page {                                      /* named areas */
  display: grid;
  grid-template:
    "header header" auto
    "nav    main"   1fr
    "footer footer" auto / 16rem 1fr;
}

.card { display: grid; grid-template-columns: subgrid; grid-column: span 3; } /* inherit parent tracks */
```
Footgun: `minmax(16rem, 1fr)` overflows on narrow screens — use `minmax(min(16rem, 100%), 1fr)`.

### B. The box model
`box-sizing: border-box` globally (set in reset). Understand the four boxes (content/padding/border/margin), margin collapsing (vertical adjacent/parent-child margins collapse — `gap` and flow-root avoid it), and that `gap` replaces margin hacks in flex/grid.

### C. Custom properties (variables)
The runtime theming and token mechanism (taxonomy owned by [`ui.md`](guides://ui.md)). Unlike preprocessor variables they **cascade, inherit, and are live in the DOM**.

```css
@layer tokens {
  :root {
    --space: 1rem;
    --brand: oklch(62% 0.19 256);
    --radius: 0.5rem;
  }
}
.card { padding: var(--space); background: var(--surface, white); border-radius: var(--radius); }

/* theming = re-binding tokens, not rewriting rules */
:root { color-scheme: light dark; }
[data-theme="dark"] { --surface: oklch(22% 0.02 256); }
```
Use `@property` to make a custom property animatable/type-checked:
```css
@property --angle { syntax: "<angle>"; inherits: false; initial-value: 0deg; }
```

### D. Modern responsive — container queries, clamp, fluid type
Components should adapt to **their container**, not the viewport — this makes them reusable in any slot.

```css
.card-host { container: card / inline-size; }
@container card (width > 24rem) {
  .card { grid-template-columns: 12rem 1fr; }
}
.card-host { font-size: 5cqi; }              /* container query units: cqi/cqb/cqmin */

h1 { font-size: clamp(1.75rem, 1.2rem + 3vw, 3rem); }   /* fluid type, no breakpoints */
.wrap { width: min(70ch, 100% - 2rem); margin-inline: auto; }  /* fluid container */
```
Prefer `@container` + `clamp()` over a ladder of `@media` breakpoints. Reserve `@media` for true viewport/device concerns (`prefers-reduced-motion`, `prefers-color-scheme`, print).

### E. Logical properties
Author flow-relative so layouts mirror automatically for RTL/vertical writing modes.

```css
.note { margin-inline: auto; padding-block: 1rem; border-inline-start: 3px solid var(--brand); inset-block-start: 0; }
```
Use `margin-inline`/`-block`, `padding-inline-*`, `inset-*`, `border-inline-*`, `inline-size`/`block-size` instead of physical `left/right/top/bottom/width/height`.

### F. Modern color — oklch, color-mix, relative color
Author in **oklch** (perceptually uniform; predictable lightness; wide-gamut). Derive variants instead of hand-picking hexes.

```css
:root { --brand: oklch(62% 0.19 256); }
.btn        { background: var(--brand); }
.btn:hover  { background: oklch(from var(--brand) calc(l - 0.08) c h); } /* relative color */
.tint       { background: color-mix(in oklch, var(--brand) 15%, white); }
.muted      { color: oklch(62% 0.19 256 / 0.6); }                        /* alpha via slash */
```
`color-mix()` and relative color syntax are Baseline — use them to generate hover/active/disabled and tint/shade scales from one source token.

### G. Native nesting
Use **native CSS nesting** (Baseline 2023) — a preprocessor is no longer required for it. `&` is the parent reference; keep nesting ≤ 3 levels.

```css
.card {
  padding: var(--space);
  & > .title { font-weight: 600; }
  &:hover { box-shadow: var(--shadow); }
  @container (width > 24rem) { display: grid; }   /* nest at-rules too */
}
```
Footgun: a nested type selector like `& div` works, but a bare `div` (no combinator) is read as a relative selector — be explicit with `&`.

### H. Modern selectors — :has(), :is(), :where()
```css
.field:has(> input:invalid) { --surface: oklch(95% 0.04 25); }  /* parent/previous-sibling logic */
:is(h1, h2, h3):hover { text-decoration: underline; }           /* group, specificity = heaviest arg */
:where(ul, ol) { margin-block: 0; padding-inline-start: 1.5rem; } /* group, specificity 0 → easy override */
article:has(figure) { ... }   /* quantity/relationship queries without JS */
```
`:has()` is a true parent selector (Baseline 2023) — replaces most layout-driven JS class toggling.

### I. Transitions, animations & scroll-driven animation
Animate only `transform`/`opacity` (compositor-only — see [`performance.md`](guides://performance.md)). Every motion rule carries a reduced-motion escape (policy: [`accessibility.md`](guides://accessibility.md)).

```css
.panel { transition: transform 200ms ease, opacity 200ms ease; }
@media (prefers-reduced-motion: reduce) {
  *, ::before, ::after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
}
```
Scroll-driven animations (`animation-timeline`) run on the compositor — no scroll-event JS:
```css
@supports (animation-timeline: scroll()) {
  .progress { animation: grow linear; animation-timeline: scroll(root block); }
  .reveal   { animation: fade-in linear both; animation-timeline: view(); animation-range: entry 0% cover 30%; }
}
```
Also leverage Baseline transition primitives: `@starting-style` (enter animations), `transition-behavior: allow-discrete`, and `transition` on `display`/popover.

### J. Modern reset
A minimal reset belongs in `@layer reset`. Keep it small and modern — not a 200-line normalize dump.
```css
@layer reset {
  *, *::before, *::after { box-sizing: border-box; }
  * { margin: 0; }
  html { -webkit-text-size-adjust: 100%; }
  body { min-height: 100svh; line-height: 1.5; -webkit-font-smoothing: antialiased; }
  img, picture, svg, video { display: block; max-width: 100%; }
  input, button, textarea, select { font: inherit; color: inherit; }
  p, h1, h2, h3, h4 { overflow-wrap: break-word; text-wrap: balance; }
}
```

### K. Methodology — naming & scoping
Pick one and apply layer discipline:
- **BEM** (`block__element--modifier`) — predictable, low specificity, framework-free. Good default for hand-authored CSS.
- **Utility-first** — single-purpose classes in `@layer utilities`; fast iteration, but keep it a thin layer, not the whole system.
- **Scoping** — component frameworks (CSS Modules, Shadow DOM, `@scope`) localize names. `@scope (.card) to (.card__content)` bounds rules natively (Baseline 2024).

Whatever the methodology: classes carry style, attributes/state (`[aria-*]`, `:state()`, `:disabled`) carry behavior, layers carry priority.

### L. Footguns
- Reliance on source order to win conflicts → use `@layer`.
- `!important` to override a framework → put your styles in a later layer instead.
- Deep descendant selectors (`.nav ul li a span`) → flat class or `:where()`.
- `100vh` jumping on mobile → use `100svh`/`100dvh`.
- Animating `box-shadow`/`width` → animate a pseudo-element `opacity` or `transform: scale()`.
- z-index escalation → establish stacking contexts deliberately; use `isolation: isolate`.

---

## 6. Tooling & Build

Native CSS first. Reach for a preprocessor (Sass) only for build-time constructs CSS still lacks (loops, complex `@function`, `@mixin` with content blocks) — and even then, custom properties, nesting, and `color-mix()` belong in plain CSS. Reach for a utility framework only when a team needs enforced consistency at scale; never let it replace understanding the cascade.

```bash
# Lightning CSS — fast transpile + minify + browserslist-targeted polyfilling (Rust)
npx lightningcss --minify --browserslist --bundle src/main.css -o dist/main.css

# OR PostCSS pipeline
npx postcss src/main.css -o dist/main.css   # plugins: autoprefixer, postcss-preset-env

npx stylelint "**/*.{css,scss}"             # CSS-LINT-01
```

```jsonc
// .stylelintrc.json
{
  "extends": ["stylelint-config-standard"],
  "plugins": ["stylelint-no-unsupported-browser-features"],
  "rules": {
    "selector-max-id": 0,
    "selector-max-specificity": "0,3,0",
    "max-nesting-depth": 3,
    "declaration-no-important": true,
    "color-named": "never",
    "plugin/no-unsupported-browser-features": [true, { "severity": "warning" }]
  }
}
```
Browser support is declared once in `.browserslistrc` (e.g. `defaults`, `not dead`) and consumed by Lightning CSS, autoprefixer, and the stylelint compat plugin — keeping CSS-CMPAT-01 enforceable from a single source.

---

## 7. Quick Reference

```bash
npx stylelint "**/*.css"                    # lint
npx prettier --write "**/*.css"             # format
npx lightningcss --minify --browserslist src/main.css -o dist/main.css  # build
```

| Need | Reach for |
|------|-----------|
| Manage overrides | `@layer`, low specificity, `:where()` |
| Component adapts to slot | `@container` + container query units |
| Fluid size/space | `clamp()`, `min()`, `max()` |
| Derive color variants | `oklch()`, `color-mix()`, relative color |
| Parent/relationship logic | `:has()` |
| RTL-safe spacing | logical properties |
| Enter/scroll motion | `@starting-style`, `animation-timeline` |

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] CSS-FMT-01 — formatted, no diff
- [ ] CSS-LINT-01 — stylelint clean
- [ ] CSS-LINT-02 — no unjustified `!important`
- [ ] CSS-SPEC-01 — specificity low, no ID selectors
- [ ] CSS-LAYER-01 — styles organized in `@layer`
- [ ] CSS-CMPAT-01 — Baseline or `@supports`-gated for browserslist target
- [ ] CSS-A11Y-01/02/03 — focus-visible, reduced-motion, AA contrast (see `accessibility.md`)
- [ ] CSS-PERF-01 — animations limited to transform/opacity (see `performance.md`)
- [ ] CSS-TOK-01 — no hardcoded colors/sizes in components (see `ui.md`)
- [ ] Agent ran every §3 command and documented any fixes

---
**End of CSS Guidelines**
