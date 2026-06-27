# Material Design 3 Guidelines
Mandatory standards for building human-centered interfaces with Material Design 3 (Material You): token-based theming, dynamic color, the M3 type/elevation/shape/motion systems, the component library, and adaptive layouts. Material Design 3, Material Theme Builder, Material Web 2.x, Compose Material3 1.3.x, Flutter Material 3.

---
name: material
title: Material Design 3 Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: framework
tools: [material-design-3, material-you, material-theme-builder, material-web@2.3, compose-material3@1.3]
requires: []
recommends:
  - ui
  - accessibility
  - css
  - reactjs
  - angular
  - flutter
provides:
  - material-design-3
  - dynamic-color
  - material-theming
  - material-tokens
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): Material 3 is a *specific* design system that implements the general design-system, component, and token principles owned by [`ui.md`](guides://ui.md). This guide does not restate those — it spends its tokens on what is unique to Material 3.

---

## 0. Prerequisites & References

Material 3 has no hard prerequisite guide, but it specializes several cross-cutting concerns. Fetch the recommended owners when your task touches them; this guide assumes their rules and does not repeat them.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`ui.md`](guides://ui.md) — general design-system, component, and design-token *principles*. Material 3 is one concrete implementation; do not duplicate the general rules here.
> - [`accessibility.md`](guides://accessibility.md) — WCAG criteria, screen-reader/keyboard policy. *(M3 binding: HCT tonal palettes, `minimumInteractiveComponentSize`, built-in component semantics — see §9.)*
> - [`css.md`](guides://css.md) — when consuming M3 tokens as CSS custom properties / theming the web (`md-sys-*` variables).
> - [`reactjs.md`](guides://reactjs.md) · [`angular.md`](guides://angular.md) · [`flutter.md`](guides://flutter.md) — component wiring, state, and lifecycle for the platform binding you target.

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) *(visual/interaction tests are written test-first — §2)* · [`e2e-testing.md`](guides://e2e-testing.md) · [`designpatterns.md`](guides://designpatterns.md) · [`html.md`](guides://html.md)

---

## 1. Core Philosophies: TOKEN-FIRST

Material-3-specific principles only. General UI heuristics (affordance, progressive disclosure, feedback) are owned by [`ui.md`](guides://ui.md); a11y by [`accessibility.md`](guides://accessibility.md). What is unique to Material 3:

- **T**okens, not values: every color, type, shape, elevation, and motion value comes from an M3 design token. Raw hex/sp/dp/ms literals are a defect.
- **O**ne color system (HCT): color is generated from a seed through Hue-Chroma-Tone tonal palettes that *guarantee* contrast when the correct `on-*` role sits on its container. Never hand-pick contrasting hex.
- **K**inetic & expressive: motion uses M3 duration/easing tokens (and, in M3 Expressive, spring physics); it is informative and orienting, never decorative noise.
- **E**levation via tone + state layers: surfaces are differentiated by tonal `surfaceContainer*` roles and state-layer overlays — not arbitrary drop shadows.
- **N**ative dynamic color (Material You): user-generated color schemes are the primary theming strategy where the platform supports them, with a branded scheme as fallback.
- **F**orm-factor adaptive: one component tree responds to window size classes (compact/medium/expanded) and canonical layouts.
- **I**ntent-matched components: pick the lowest-effort M3 component for each interaction; never reinvent a widget M3 already ships.
- **R**oles over raw surfaces: assign the correct M3 color/typography *role* to each element; the theme — not the component — decides the concrete value.
- **S**ystematic theming: brand expression flows through the token pipeline (Material Theme Builder → tokens → theme), never ad-hoc per-component overrides.

**Verified UI**: Agent-generated Material code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `MD3-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| MD3-THEME-01 | Colors, type, shape, elevation, motion MUST use M3 tokens — zero raw hex/sp/dp/ms literals | grep for literals / review | no hardcoded values |
| MD3-THEME-02 | Both light AND dark schemes MUST render correctly for every component | snapshot both themes | both pass |
| MD3-COLOR-01 | Each surface MUST pair the correct M3 color roles (`on-X` over its `X`/container) | review | correct role pairs |
| MD3-COLOR-02 | Dynamic color MUST be supported where the platform allows, with a branded fallback scheme | review | dynamic + fallback |
| MD3-TYPE-01 | Text MUST use M3 type-scale roles, not invented sizes/weights | review | scale roles only |
| MD3-COMP-01 | Use the canonical M3 component per pattern; no custom widget when one exists | review | no reinvented components |
| MD3-STATE-01 | Interactive components MUST show all applicable state layers + ripple/indication | interaction test / review | states present |
| MD3-MOTION-01 | Transitions MUST use M3 duration + easing tokens (no arbitrary values) | review | token-based motion |
| MD3-NAV-01 | Navigation MUST adapt to window size class (bar → rail → drawer) | test compact/medium/expanded | adapts, no overflow |
| MD3-INPUT-01 | Applicable fields MUST set autofill hints + correct keyboard/`inputmode`; use a picker for date/time | review | hints + picker |
| MD3-A11Y-01 | Touch/pointer targets MUST be ≥ 48dp (see `accessibility.md`) | a11y scanner | ≥ 48dp |
| MD3-A11Y-02 | Every screen MUST pass WCAG 2.2 AA (see `accessibility.md`) | `axe` / platform scanner | 0 violations |
| MD3-TST-01 | Visual + interaction tests MUST be test-first, bug-fix gets a regression test (see `tdd.md`) | platform test runner | exit 0, 0 skips |

> **Forbidden**: hardcoded colors/fonts/dimensions instead of tokens; missing dark theme; reinventing an existing M3 component; color as the *sole* information channel (see `accessibility.md`); free-text input where a selection/picker fits; shipping UI before its visual/interaction test (violates `tdd.md`).

---

## 3. Verification Protocol

Run the platform-appropriate gates before presenting UI. Fix → re-run until green. The *why* behind each (why test-first, why WCAG) lives in its §0 owner.

```bash
# Web (Material Web)
npm run build && npm test            # MD3-TST-01
npx playwright test                  # interaction/visual (see e2e-testing.md)
npx axe-core --tags wcag2a,wcag22aa  # MD3-A11Y-01/02

# Android (Compose Material3)
./gradlew lint                       # MD3-A11Y-* accessibility checks
./gradlew testDebugUnitTest connectedDebugAndroidTest  # MD3-TST-01
./gradlew spotlessApply              # format

# Flutter (Material 3)
flutter analyze                      # lint + a11y
flutter test                         # MD3-TST-01
```

Token, color-role, type-scale, and component-choice gates (MD3-THEME/COLOR/TYPE/COMP) are review/grep gates — scan the diff for raw literals and reinvented widgets.

---

## 4. Component Library & Selection

The M3 component set, grouped by intent. Choosing the *right* component is the largest single lever on usability — pick the lowest-effort one.

| Category | Components | Use when |
|----------|-----------|----------|
| **Action** | FAB / Extended FAB, Icon Button, Filled/Tonal/Outlined/Text Button, Segmented Button | user triggers an operation |
| **Containment** | Card, Dialog, Bottom/Side Sheet, Carousel, Tooltip | group content / surface contextual info |
| **Navigation** | Navigation Bar, Navigation Rail, Navigation Drawer, Tabs, Top/Bottom App Bar | move between views (adapt per §7) |
| **Selection** | Checkbox, Radio, Switch, Chip, Slider, Date Picker, Time Picker | choose from options |
| **Text input** | Filled / Outlined Text Field, Search Bar | enter or search text |
| **Communication** | Badge, Progress Indicator, Snackbar | system reports status |

**Selection decision tree — match the input to the least-effort component:**

```
Binary state?              → Switch (settings) / Checkbox (forms)
One of 2–5 options?        → Segmented Button or Chip group
One of 5–15 options?       → Exposed Dropdown Menu (filterable)
One of 15+ options?        → Search Bar with autocomplete
Date / Time?               → Date Picker / Time Picker  (NEVER free text)
Numeric range?             → Slider (with optional text override)
Short free text?           → Text Field + autocomplete + autofill hints
```

> Never use a custom widget when an M3 component exists (MD3-COMP-01). Free text is the most error-prone input — convert to selection whenever the option set is finite.

---

## 5. Theming, Tokens & Dynamic Color

**The heart of the guide.** General token *principles* are owned by [`ui.md`](guides://ui.md); this is their Material 3 binding.

### A. Token hierarchy
M3 tokens are three-tiered: **reference** (the raw tonal palette, `md-ref-palette-primary40`) → **system** (semantic roles, `md-sys-color-primary`, `md-sys-typescale-body-large`) → **component** (per-component, `md-comp-filled-button-container-color`). Author UI against **system** roles; let the theme resolve them.

```kotlin
// Compose — CORRECT: system role tokens
Text(text = "Hello",
     style = MaterialTheme.typography.headlineMedium,
     color = MaterialTheme.colorScheme.onSurface)
Surface(color = MaterialTheme.colorScheme.surfaceContainerLow, tonalElevation = 1.dp) { /* … */ }

// WRONG: raw values bypass the theme (fails MD3-THEME-01)
Text("Hello", fontSize = 28.sp, color = Color(0xFF1C1B1F))
Surface(color = Color(0xFFF3EDF7), shadowElevation = 4.dp) { /* … */ }
```

```css
/* Web — consume M3 tokens as CSS custom properties (see css.md) */
.surface { background: var(--md-sys-color-surface-container-low);
           color:      var(--md-sys-color-on-surface); }
```

### B. Dynamic color (Material You) + branded fallback
```kotlin
@Composable
fun AppTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit,
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val ctx = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(ctx) else dynamicLightColorScheme(ctx)
        }
        darkTheme -> darkColorScheme(primary = Color(0xFFD0BCFF))   // branded fallback
        else      -> lightColorScheme(primary = Color(0xFF6750A4))
    }
    MaterialTheme(colorScheme, AppTypography, AppShapes, content)
}
```
Generate both schemes from one seed with **Material Theme Builder**; commit the exported tokens, not hand-edited hex. On the web, ship a generated light+dark token sheet and switch via `prefers-color-scheme`.

### C. Color roles (HCT guarantees contrast)
Use the role pair for each surface — the on-* role over its container is contrast-safe by construction (no manual ratio math needed for the standard pairs).

| Surface | Container role | Content role |
|---------|----------------|--------------|
| App background | `surface` | `onSurface` |
| Primary action | `primary` | `onPrimary` |
| Secondary action | `secondary` | `onSecondary` |
| Tonal container | `primaryContainer` | `onPrimaryContainer` |
| Error | `error` / `errorContainer` | `onError` / `onErrorContainer` |
| Variant / muted | `surfaceVariant` | `onSurfaceVariant` |
| Elevated surface | `surfaceContainerHigh` | `onSurface` |

> HCT (Hue, Chroma, Tone) keeps tone perceptually uniform, so a fixed tone delta yields accessible contrast across hues. Verifying *custom* (non-standard) color pairs against WCAG is still required — that policy is owned by [`accessibility.md`](guides://accessibility.md).

### D. Typography scale
Use the 15 M3 type-scale roles; do not invent sizes. Roles (display/headline/title/body/label × large/medium/small) map to weight + size automatically.

```
displayLarge 57 · displayMedium 45 · displaySmall 36
headlineLarge 32 · headlineMedium 28 · headlineSmall 24
titleLarge 22 · titleMedium 16/500 · titleSmall 14/500
bodyLarge 16 · bodyMedium 14 · bodySmall 12
labelLarge 14/500 (buttons) · labelMedium 12/500 (nav) · labelSmall 11/500 (badges)
```

### E. Shape & density
Apply the M3 **shape scale** (`extraSmall 4 · small 8 · medium 12 · large 16 · extraLarge 28`) via the theme's `shapes`, not per-component corner literals. Use M3 **density** levels (0 to -3) to compactify for data-dense desktop UIs rather than shrinking individual paddings.

---

## 6. Interaction States, Elevation & Motion

### A. State layers
Every interactive component shows applicable states via a tonal overlay of the content color. Material components apply these automatically — use them rather than rolling your own `clickable`.

```
Enabled 0% · Disabled component@38% · Hover 8% · Focus 10% (+ focus ring) · Press 10% (ripple) · Drag 16%
```
```kotlin
Button(onClick = ::submit) { Text("Submit") }   // built-in ripple + state layers — CORRECT
// WRONG: clickable(indication = null, …) — no feedback, fails MD3-STATE-01
```

### B. Elevation
M3 expresses elevation primarily through **tonal** `surfaceContainer*` roles (and shadow only where the spec calls for it). Prefer `tonalElevation`/the container role over `shadowElevation`.

### C. Motion tokens
Use M3 duration + easing tokens; never arbitrary durations.

```
Durations: short1 75 · short2 150 · medium1 200 · medium2 250 · long1 300 · long2 350 (ms)
Easing:    EmphasizedDecelerate (enter) · EmphasizedAccelerate (exit) · Standard (in-place change)
Patterns:  Container Transform · Shared Axis · Fade Through · Fade   (M3 Expressive adds spring physics)
```
```kotlin
val pad by animateDpAsState(
    targetValue = if (expanded) 16.dp else 0.dp,
    animationSpec = tween(250, easing = FastOutSlowInEasing),   // medium2 + Standard
)
```

---

## 7. Adaptive Layout

### A. Window size classes
Drive layout and navigation off the three canonical breakpoints — not raw device checks.

```
Compact  < 600dp   → Navigation Bar (bottom), single pane, stacked
Medium   600–839dp → Navigation Rail (side), optional two-pane, flexible grid
Expanded ≥ 840dp   → Navigation Drawer (side), two-pane, multi-column grid
```
```kotlin
when (windowSizeClass.widthSizeClass) {
    WindowWidthSizeClass.Compact  -> Scaffold(bottomBar = { NavigationBar { /* … */ } }) { content() }
    WindowWidthSizeClass.Medium   -> Row { NavigationRail { /* … */ }; content() }
    WindowWidthSizeClass.Expanded -> PermanentNavigationDrawer(drawerContent = { /* … */ }) { content() }
}
```

### B. Canonical layouts
Use the M3 canonical layouts (**List-Detail**, **Supporting Pane**, **Feed**) rather than bespoke responsive logic. On compact they collapse to a single navigable pane; on medium/expanded they show panes side-by-side. The M3 adaptive libraries (`androidx.compose.material3.adaptive`, Flutter adaptive widgets) ship these directly.

### C. Grid
Material uses a 4/8/12-column grid (compact/medium/expanded) with 16–24dp margins. Use an adaptive grid (`GridCells.Adaptive(minSize)` / CSS `auto-fill minmax()`) so columns reflow automatically.

---

## 8. Smart Input & Forms

Form *effort-minimization* heuristics (eliminate → automate → select → assist) are owned by [`ui.md`](guides://ui.md). The Material 3 binding:

- **Autofill hints (MD3-INPUT-01):** set them on every applicable field so the platform can fill whole forms.

  | Field | HTML `autocomplete` | Android `AutofillType` | iOS `textContentType` |
  |-------|--------------------|------------------------|-----------------------|
  | Email | `email` | `EmailAddress` | `.emailAddress` |
  | Name | `name` | `PersonName` | `.name` |
  | Street | `street-address` | `PostalAddress` | `.streetAddressLine1` |
  | Phone | `tel` | `PhoneNumber` | `.telephoneNumber` |
  | New password | `new-password` | `NewPassword` | `.newPassword` |
  | OTP | `one-time-code` | `SmsOtpCode` | `.oneTimeCode` |

- **Autocomplete:** use the M3 **Search Bar** or **Exposed Dropdown Menu**; surface recent/frequent values on focus, filter per keystroke, cap visible suggestions ~5–7, and handle the no-match state.

```kotlin
ExposedDropdownMenuBox(expanded, onExpandedChange = { expanded = it }) {
    OutlinedTextField(
        value = text, onValueChange = { text = it; expanded = true },
        label = { Text("Email") }, modifier = Modifier.menuAnchor(),
        keyboardOptions = KeyboardOptions(KeyboardType.Email, imeAction = ImeAction.Next),
    )
    ExposedDropdownMenu(expanded, onDismissRequest = { expanded = false }) {
        suggestions.filter { it.contains(text, true) }.take(7).forEach { s ->
            DropdownMenuItem(text = { Text(s) }, onClick = { text = s; expanded = false })
        }
    }
}
```
```html
<!-- Material Web -->
<md-outlined-text-field label="Email" type="email" autocomplete="email" inputmode="email" name="email"></md-outlined-text-field>
```

- **Inline validation:** validate as the user types (debounced), not on submit; pair color with an icon + text message (color is never the sole signal — see `accessibility.md`).
- **Smart defaults:** prefill currency/country/language from locale, dates to today, quantity to 1, returning users to last-used values.
- **Errors → components:** field error (inline) · Snackbar (transient / undoable) · Dialog (destructive confirm) · full-screen state (network/404/empty). Every list/grid needs a meaningful empty state.

---

## 9. Accessibility — Material bindings

WCAG criteria, screen-reader, keyboard, and contrast *policy* are owned by [`accessibility.md`](guides://accessibility.md) (target: **WCAG 2.2 AA**). Material 3 gives you a head start but does **not** make a screen accessible by itself. What Material provides vs. what you must still verify:

- **Built-in:** HCT role pairs are contrast-safe; `IconButton`/`minimumInteractiveComponentSize()` enforce the 48dp target (MD3-A11Y-01); standard components ship roles, focus order, and ripple/focus indicators.
- **You must still:** supply `contentDescription`/`aria-label` for icons and images (say the *action*, not the icon shape — e.g. "Delete", not "trash icon"); verify any **custom** color pair against WCAG; merge semantics for compound elements; never convey state by color alone; keep `sp`/relative text units so text scales to 200% without clipping; ensure full keyboard reachability of custom interactions.

```kotlin
Icon(Icons.Default.ShoppingCart, contentDescription = "Shopping cart, 3 items")  // descriptive
Row(Modifier.semantics(mergeDescendants = true) {}) { /* icon + text read as one node */ }
```

---

## 10. Quick Reference

```
Tokens     → system roles only: md-sys-color-*, MaterialTheme.colorScheme/typography/shapes
Color      → seed → HCT tonal palettes → light+dark schemes (Material Theme Builder)
Type       → 15 roles (display/headline/title/body/label × L/M/S)
Shape      → extraSmall 4 · small 8 · medium 12 · large 16 · extraLarge 28
Motion     → duration short1..long2 + Emphasized/Standard easing
Nav        → Compact bar · Medium rail · Expanded drawer
Spacing    → 4 / 8 / 12 / 16 / 24 / 32 dp
```

```
User needs to…           → component
Trigger primary action   → FAB / Extended FAB
Navigate views           → Navigation Bar / Rail / Drawer (adaptive)
Pick one of few          → Segmented Button / Radio
Enter a date/time        → Date / Time Picker (never text)
Search content           → Search Bar + autocomplete
Confirm destructive act  → Dialog with explicit action labels
Report status            → Badge / Snackbar / Progress Indicator
```

---

## 11. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] MD3-THEME-01 — zero raw hex/sp/dp/ms; system tokens only
- [ ] MD3-THEME-02 — light AND dark schemes render correctly
- [ ] MD3-COLOR-01 — correct `on-*`/container role pairs on every surface
- [ ] MD3-COLOR-02 — dynamic color supported with branded fallback
- [ ] MD3-TYPE-01 — M3 type-scale roles only
- [ ] MD3-COMP-01 — canonical M3 components; nothing reinvented
- [ ] MD3-STATE-01 — state layers + ripple/indication on all interactives
- [ ] MD3-MOTION-01 — M3 duration + easing tokens
- [ ] MD3-NAV-01 — navigation adapts across compact/medium/expanded
- [ ] MD3-INPUT-01 — autofill hints + keyboard types; pickers for date/time
- [ ] MD3-A11Y-01/02 — ≥48dp targets, WCAG 2.2 AA pass (see `accessibility.md`)
- [ ] MD3-TST-01 — visual/interaction tests written test-first; bugs have regression tests (see `tdd.md`)
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Material Design 3 Guidelines**
