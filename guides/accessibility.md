# Accessibility (a11y) Guidelines
Mandatory, auditable standards for accessible web and app UIs: WCAG 2.2 AA, semantic structure, ARIA, keyboard & focus, contrast, screen-reader support, automated + manual a11y testing. axe-core, Lighthouse, Pa11y, NVDA/VoiceOver/JAWS.

---
name: accessibility
title: Accessibility (a11y) Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [axe-core, "@axe-core/playwright", jest-axe, lighthouse, pa11y, eslint-plugin-jsx-a11y, NVDA, VoiceOver, JAWS]
requires: []
recommends:
  - html
  - css
  - ui
  - e2e-testing
provides:
  - wcag
  - aria
  - keyboard-nav
  - a11y-testing
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide is the canonical owner of accessibility. Semantic markup detail lives in [`html.md`](guides://html.md), visual/contrast styling in [`css.md`](guides://css.md), component patterns in [`ui.md`](guides://ui.md), and a11y test automation in [`e2e-testing.md`](guides://e2e-testing.md).

---

## 0. Prerequisites & References

This guide owns the accessibility *rules*. The guides below own the *mechanics* it builds on — fetch them when the task touches their surface; do not restate them here.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`html.md`](guides://html.md) — semantic elements, landmarks, document structure, `<track>`/media markup. *(a11y binding: semantics MUST carry meaning, not just layout.)*
> - [`css.md`](guides://css.md) — visual styling, focus-visible styling, `prefers-reduced-motion`, responsive/zoom layout. *(a11y binding: never remove focus outline without a replacement; meet contrast in §6.)*
> - [`ui.md`](guides://ui.md) — reusable component patterns (dialog, menu, combobox, tabs). *(a11y binding: components MUST implement the WAI-ARIA Authoring Practices keyboard model.)*
> - [`e2e-testing.md`](guides://e2e-testing.md) — running automated a11y scans in the E2E pipeline. *(a11y binding: gate the build on `@axe-core/playwright`, see §10.)*

> 📎 **SEE ALSO:** WAI-ARIA Authoring Practices Guide (APG) for canonical keyboard interaction patterns; WCAG 2.2 spec and Understanding documents for normative success criteria.

---

## 1. Core Philosophies: A11Y-FIRST

Accessibility-specific principles only. Markup, styling, component, and test mechanics come from §0.

- **A**ll users from the start: accessibility is a requirement, not a retrofit; bake it into design and the first commit.
- **1** equivalent experience: same content and functionality for everyone — never a degraded "accessible version".
- **1**st, native HTML: prefer a native element to ARIA every time; ARIA is a patch for what HTML cannot express (§7).
- **Y**es to real testing: automated scans catch ~30–50% of issues; the rest require keyboard-only and screen-reader passes (§10–11).

**Verified UI**: Agent-generated interfaces MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `A11Y-<TOPIC>-<NN>`. Each row has a binary gate. WCAG criteria are cited by number; "AA" is the conformance target unless stated.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| A11Y-WCAG-01 | UI MUST meet WCAG 2.2 Level AA for every shipped page/flow | `npx lighthouse <url> --only-categories=accessibility` + manual §11 | a11y score 100 & §11 checklist complete |
| A11Y-WCAG-02 | New AA criteria in WCAG 2.2 MUST hold: focus not obscured (2.4.11), target size ≥ 24×24 CSS px (2.5.8), dragging has a single-pointer alt (2.5.7), no cognitive-test auth without alt (3.3.8) | review + axe scan | all four satisfied |
| A11Y-SEM-01 | Structure MUST use semantic elements & one logical heading order; no level skips (1.3.1, 2.4.6) (markup: see `html.md`) | `axe` rule `heading-order`, `landmark-*` | 0 violations |
| A11Y-IMG-01 | Every `<img>`/icon MUST have a correct text alternative; decorative = `alt=""` (1.1.1) | `axe` rule `image-alt` | 0 violations |
| A11Y-MEDIA-01 | Video MUST have captions; audio MUST have a transcript; no autoplay > 3s without a stop control (1.2.x, 1.4.2) (markup: see `html.md`) | review of `<track>`/transcript | present for all media |
| A11Y-FORM-01 | Every control MUST have a programmatic label & correct `autocomplete`; errors MUST be programmatically associated (1.3.1, 3.3.1/2, 1.3.5) | `axe` rule `label`, manual SR pass | 0 violations |
| A11Y-KBD-01 | All functionality MUST be keyboard-operable with no traps; focus order logical (2.1.1, 2.1.2, 2.4.3) | manual keyboard pass §11 | full operation, no trap |
| A11Y-KBD-02 | Visible focus indicator MUST be present and not obscured (2.4.7, 2.4.11) (styling: see `css.md`) | manual + `:focus-visible` review | indicator visible everywhere |
| A11Y-ARIA-01 | ARIA MUST be valid, follow the 5 rules (§7), and prefer native HTML; no invalid roles/attrs/refs (4.1.2) | `axe` rules `aria-*` | 0 violations |
| A11Y-ARIA-02 | Status changes MUST be announced via live regions / `role=alert`/`status` (4.1.3) | manual SR pass | announced |
| A11Y-COLOR-01 | Contrast MUST meet AA: 4.5:1 text, 3:1 large text & UI/graphics (1.4.3, 1.4.11) (palette: see `css.md`) | `axe` rule `color-contrast` + manual graphics | 0 violations |
| A11Y-COLOR-02 | Information MUST NOT be conveyed by color alone (1.4.1) | review | text/icon backup present |
| A11Y-ZOOM-01 | Content MUST reflow & remain usable at 400% zoom / 320px width; text resizable to 200% (1.4.4, 1.4.10) (layout: see `css.md`) | resize/zoom check | no loss/clipping, no 2-axis scroll |
| A11Y-MOTION-01 | Respect `prefers-reduced-motion`; no content flashes > 3×/s (2.3.1, 2.3.3) | review + media query | honored |
| A11Y-TEST-01 | An automated a11y scan MUST run in CI and gate the build (see `e2e-testing.md`) | `@axe-core/playwright` / `jest-axe` in pipeline | 0 violations, build fails on regress |
| A11Y-TEST-02 | `eslint-plugin-jsx-a11y` (or framework equivalent) MUST pass clean for component code | `eslint` | exit 0 |

> **Forbidden**: removing focus outlines without an equivalent indicator; `<div>`/`<span>` as a button or link; positive `tabindex`; placeholder used as the only label; `aria-hidden="true"` on a focusable element; conveying required/error/status by color alone; shipping media without captions/transcript.

---

## 3. Conformance Target & Scope

- **Target: WCAG 2.2 Level AA** for all public and authenticated UI. Pursue AAA criteria (e.g. 7:1 contrast, 1.4.6) where cheap, but AA is the gate.
- WCAG 2.2 is backward-compatible with 2.1; meeting 2.2 AA meets 2.1 AA. It removed 4.1.1 (Parsing) and added six new criteria — the four AA ones are gated in A11Y-WCAG-02.
- **POUR** is the mental model, not a checklist:
  - **Perceivable** — text alternatives, captions, contrast, reflow.
  - **Operable** — keyboard, no traps, enough time, target size, no seizure risk.
  - **Understandable** — readable, predictable, input assistance.
  - **Robust** — valid, name/role/value exposed to AT, status messages.
- Legal baselines (ADA, EU EAA/EN 301 549, Section 508) all reference WCAG AA; meeting §2 satisfies them. Publish an accessibility statement and keep a VPAT/known-issues log (§11).

---

## 4. Semantic Structure & Landmarks

Markup syntax is owned by [`html.md`](guides://html.md). The a11y *rules* on top of it:

- Use one `<main>`, a single top-level `<h1>`, and a heading outline with **no skipped levels** — headings convey structure to screen-reader users who navigate by them.
- Map regions to native landmark elements (`header`/`banner`, `nav`, `main`, `aside`/`complementary`, `footer`/`contentinfo`). Redundant `role` attributes on these are unnecessary in modern HTML — omit them.
- Disambiguate repeated landmarks with `aria-label` (e.g. `<nav aria-label="Pagination">` vs `<nav aria-label="Primary">`).
- Provide a **skip link** as the first focusable element so keyboard users bypass repeated nav:

```html
<a href="#main" class="skip-link">Skip to main content</a>
...
<main id="main" tabindex="-1">…</main>
```
Style it visible on `:focus` (positioning/animation: see [`css.md`](guides://css.md)).

- Set `<html lang>` (and `lang` on inline foreign-language passages) so AT picks the right pronunciation (3.1.1/2).

---

## 5. Names, Forms & Error Handling

A control's **accessible name** is what AT announces. Compute it from a `<label for>`, `aria-labelledby`, `aria-label`, or content — in that priority. Rules:

- Every input/select/textarea MUST have a programmatic label. Placeholder text is **not** a label (it vanishes on input and often fails contrast).
- Add the correct `autocomplete` token (`name`, `email`, `current-password`, …) — required for 1.3.5 and a usability win.
- Group related controls (radios, address fieldsets) in `<fieldset>` with a `<legend>`.
- Icon-only controls MUST get a name via `aria-label`; decorative inline SVG gets `aria-hidden="true"` + `focusable="false"`.

```html
<label for="email">Email address</label>
<input id="email" type="email" name="email" autocomplete="email"
       aria-describedby="email-hint email-err" aria-invalid="true" required>
<p id="email-hint">We never share your address.</p>
<p id="email-err" role="alert">Enter a valid email address.</p>
```

Error handling (3.3.1/2/3):
- Mark invalid fields with `aria-invalid="true"` and point `aria-describedby` at the message.
- Announce the message in an `role="alert"` (assertive) or live region so it reaches AT without a focus move.
- On submit failure, render an **error summary** at the top, focus it, and link each item to its field:

```html
<div role="alert" tabindex="-1" id="errors">
  <h2>There are 2 problems</h2>
  <ul>
    <li><a href="#email">Enter a valid email address</a></li>
    <li><a href="#pwd">Password is too short</a></li>
  </ul>
</div>
```

---

## 6. Color, Contrast & Sensory

Palette and styling mechanics are owned by [`css.md`](guides://css.md); the a11y thresholds and rules:

- **Contrast (AA):** ≥ 4.5:1 normal text, ≥ 3:1 large text (≥ 24px, or ≥ 18.66px bold) and for UI component boundaries, focus indicators, and meaningful graphics (1.4.3, 1.4.11).
- **Never rely on color alone** (1.4.1): pair it with text, an icon, a pattern, or `text-decoration`. Required fields, errors, statuses, chart series, and links-in-text all need a non-color signal.
- Support both themes: verify contrast in light **and** dark mode; honor `prefers-color-scheme`.
- **Motion:** respect `prefers-reduced-motion: reduce` — disable non-essential animation/parallax. Nothing may flash more than 3 times per second (2.3.1).
- **Zoom/reflow:** content stays usable at 400% zoom and 320 CSS px wide with no loss of content or two-dimensional scrolling (1.4.10); use relative units so text scales to 200% (1.4.4).

---

## 7. ARIA — the five rules

ARIA changes how AT perceives an element; misused, it makes things *worse* than no ARIA. Apply in order:

1. **Use native HTML first.** A `<button>`, `<a href>`, `<input>`, `<select>`, `<dialog>` already carries role, state, and keyboard behavior. Reach for ARIA only when no native element fits.
2. **Don't override native semantics.** Never `<h1 role="button">` — wrap a real `<button>` instead.
3. **All interactive ARIA widgets MUST be keyboard-operable** per the WAI-ARIA APG pattern (§8).
4. **Don't put `role="presentation"`/`aria-hidden="true"` on a focusable element** — you orphan it from AT while it still takes focus.
5. **Every interactive element needs an accessible name** (§5).

State & relationship attributes you actually maintain in sync with the UI:

```text
aria-expanded   aria-selected   aria-checked   aria-pressed   aria-current
aria-disabled   aria-controls   aria-owns      aria-haspopup  aria-activedescendant
aria-labelledby / aria-label / aria-describedby
```

**Live regions** (4.1.3) announce dynamic changes without moving focus:

```html
<div aria-live="polite" aria-atomic="true" class="sr-only"></div> <!-- non-urgent status -->
<div role="alert"></div>                                          <!-- assertive, urgent -->
<div role="status"></div>                                         <!-- polite status -->
```
The region MUST exist in the DOM *before* you inject text; injecting region + text together may not announce.

---

## 8. Keyboard, Focus & Components

This guide owns the keyboard model; reusable component shells live in [`ui.md`](guides://ui.md). The rules every interactive widget MUST follow:

- **Reachable & operable by keyboard alone**, in a logical Tab order. Never positive `tabindex`; use DOM order. `tabindex="-1"` only for programmatic focus targets (skip-link target, dialog, error summary).
- Custom widgets implement the **WAI-ARIA APG** keyboard contract: Enter/Space activate; Arrow keys move within a composite (menu, listbox, tabs, radiogroup, grid); Esc closes/cancels; Home/End jump to ends; a roving `tabindex` or `aria-activedescendant` tracks the active item.
- **Focus management on view change:** when opening a dialog, move focus into it; trap focus while open; on close, restore focus to the trigger. Prefer the native `<dialog>` element (`showModal()`) which gives focus trap, `Esc`, and inertness for free — only hand-roll a trap when `<dialog>` is unavailable.
- **Never destroy the focus indicator.** If you restyle it, the replacement MUST meet 3:1 contrast and not be obscured by sticky headers (2.4.11). Use `:focus-visible` to scope rings to keyboard use (styling: see [`css.md`](guides://css.md)).
- **Target size** ≥ 24×24 CSS px (AA, 2.5.8); 44×44 is the comfortable AAA/touch target.
- Provide a **single-pointer / keyboard alternative** to any drag-and-drop interaction (2.5.7).

Pattern reference instead of code dump: build dialogs, menus, comboboxes, tabs, and disclosures from the APG patterns and the shells in [`ui.md`](guides://ui.md); wire `aria-expanded`/`aria-controls`/`aria-activedescendant` per §7. Show only the framework binding, not a generic re-implementation.

---

## 9. Screen-Reader Support

- Decide what AT announces by controlling the **accessible name and description** (§5), not by stuffing visible text.
- Use a visually-hidden `.sr-only` utility (clip, not `display:none`, which removes it from AT) for context only AT needs (e.g. "(opens in new tab)").
- Hide purely decorative/duplicate content from AT with `aria-hidden="true"`; never hide content a user needs.
- Test the real announcement, don't assume: at minimum one screen reader per platform pairing — **NVDA + Firefox/Chrome** (Windows), **VoiceOver + Safari** (macOS/iOS), **TalkBack + Chrome** (Android). JAWS for enterprise coverage.

---

## 10. Automated Testing

Test automation infrastructure is owned by [`e2e-testing.md`](guides://e2e-testing.md); the a11y bindings:

```bash
npx eslint .                       # A11Y-TEST-02: eslint-plugin-jsx-a11y (or vuejs-accessibility, etc.)
npx lighthouse <url> --only-categories=accessibility   # A11Y-WCAG-01 score
npx pa11y <url>                    # CLI scan in CI
```

```ts
// Component-level — jest-axe (A11Y-TEST-01)
import { axe, toHaveNoViolations } from 'jest-axe';
expect.extend(toHaveNoViolations);
test('signup form has no a11y violations', async () => {
  const { container } = render(<SignupForm />);
  expect(await axe(container)).toHaveNoViolations();
});

// E2E — @axe-core/playwright (A11Y-TEST-01; pipeline owned by e2e-testing.md)
import AxeBuilder from '@axe-core/playwright';
test('home page passes axe', async ({ page }) => {
  await page.goto('/');
  const { violations } = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag22aa']).analyze();
  expect(violations).toEqual([]);
});
```

Automated tools catch only structural issues (missing labels, contrast, invalid ARIA) — roughly a third to a half of WCAG. A green scan is necessary, not sufficient: §11 manual passes are mandatory.

---

## 11. Manual Verification Protocol

Run before presenting UI; automated green does not waive these.

```text
Keyboard-only (A11Y-KBD-01/02)
- [ ] Every control reachable & operable by Tab/Shift+Tab/Arrows/Enter/Space/Esc
- [ ] Focus order matches visual order; no keyboard trap
- [ ] Focus indicator visible and not obscured everywhere
- [ ] Skip link works; dialogs trap then restore focus

Screen reader (A11Y-FORM-01, ARIA-01/02, MEDIA-01)
- [ ] Page title, headings, landmarks announced & logical
- [ ] Images/controls have correct names; errors & status announced
- [ ] Dynamic updates reach AT via live regions

Visual (COLOR-01/02, ZOOM-01, MOTION-01)
- [ ] Contrast passes in light & dark; info not color-only
- [ ] Usable at 400% zoom / 320px; text scales to 200%
- [ ] reduced-motion honored; no >3Hz flashing

Governance
- [ ] VPAT / accessibility statement current; known issues logged
```

---

## 12. Quick Reference

```text
Names      label[for] > aria-labelledby > aria-label > content
States     aria-expanded / selected / checked / pressed / current / disabled / invalid
Relations  aria-controls / owns / haspopup / activedescendant / describedby
Live       aria-live=polite|assertive · role=alert · role=status
Contrast   4.5:1 text · 3:1 large/UI/focus · target 24×24 (AA)
WCAG 2.2   focus-not-obscured 2.4.11 · target-size 2.5.8 · dragging 2.5.7
Keys       Enter/Space activate · Arrows within widget · Esc close · Home/End ends
```

---

## 13. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] A11Y-WCAG-01 — WCAG 2.2 AA met; Lighthouse a11y 100
- [ ] A11Y-WCAG-02 — new 2.2 AA criteria (2.4.11 / 2.5.8 / 2.5.7 / 3.3.8) satisfied
- [ ] A11Y-SEM-01 — semantic structure & heading order, 0 axe violations
- [ ] A11Y-IMG-01 — correct text alternatives on all images/icons
- [ ] A11Y-MEDIA-01 — captions/transcripts present; no rogue autoplay
- [ ] A11Y-FORM-01 — labels, autocomplete, associated errors
- [ ] A11Y-KBD-01/02 — full keyboard operation, no trap, visible focus
- [ ] A11Y-ARIA-01/02 — valid ARIA, native-first, status announced
- [ ] A11Y-COLOR-01/02 — contrast AA; never color-alone
- [ ] A11Y-ZOOM-01 — reflow at 400% / 320px, text to 200%
- [ ] A11Y-MOTION-01 — reduced-motion honored, no seizure-risk flashing
- [ ] A11Y-TEST-01/02 — CI axe scan gates build; jsx-a11y lint clean
- [ ] Agent ran every §10 command and the §11 manual passes, documenting fixes

---
**End of Accessibility (a11y) Guidelines**
