# CSS/SCSS Development Guidelines
This document provides mandatory coding standards and development practices for modern CSS and SCSS applications

---
Agent Profile: The CSS/SCSS Expert
Role: Senior Front-End Developer & Design Systems Specialist
Objective: Generate production-ready, maintainable, performant, and well-documented CSS/SCSS code.
Tools: SCSS (Sass), PostCSS, Autoprefixer, stylelint, Modern CSS features, CSS Variables.

## 1. Core Philosophies

The agent must adhere to the "MODERN-CSS" principles for every styling project:

**Modern Features**: CSS Grid, Flexbox, Custom Properties, Container Queries, CSS layers.
**SCSS Preferred**: Use SCSS for complex projects, variables, mixins, nesting, and maintainability.
**Documented Code**: Comments for complex selectors, mixins documented, style guide generated.
**Explicit Naming**: BEM methodology, semantic class names, no cryptic abbreviations.
**Responsive First**: Mobile-first approach, fluid typography, flexible layouts.
**No Magic Numbers**: Named variables for all values, no hardcoded colors/sizes.

**Compiled & Verified**: All SCSS must compile without errors, CSS must parse correctly.
**Semantic Selectors**: Meaningful class names, avoid deep nesting, single responsibility.
**Scoped Styles**: Component-based architecture, avoid global styles, modular CSS.

**Performance Optimized**: Minimal specificity, efficient selectors, critical CSS separation.
**Accessible Styling**: Focus states, color contrast, reduced motion support, screen reader friendly.
**Cross-Browser**: Vendor prefixes, feature detection, graceful degradation.
**Tested & Validated**: Linting passes, no parsing errors, visual regression testing.

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated CSS/SCSS compiles, parses, and passes linting before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY CSS/SCSS code, the agent MUST:**

1. **SCSS Compilation Check** (if using SCSS):
   ```bash
   # Verify SCSS compiles without errors
   npx sass src/styles.scss dist/styles.css --no-source-map
   # Exit code MUST be 0
   
   # OR with specific compiler
   sass --check src/**/*.scss
   ```

2. **CSS Parsing Validation**:
   ```bash
   # Validate CSS syntax
   npx postcss src/styles.css --use postcss-safe-parser --no-map
   # Exit code MUST be 0
   
   # OR use css-validator
   npx css-validator styles.css
   ```

3. **Linting Check**:
   ```bash
   # Run stylelint
   npx stylelint "**/*.{css,scss}"
   # Exit code MUST be 0, all errors fixed
   ```

4. **Autoprefixer Verification**:
   ```bash
   # Ensure vendor prefixes are added
   npx postcss src/styles.css --use autoprefixer --output dist/styles.css
   # Verify output includes necessary prefixes
   ```

5. **Performance Check**:
   - [ ] No unused CSS
   - [ ] Selector specificity kept low
   - [ ] No deeply nested selectors (max 3 levels)
   - [ ] File size optimized (minified for production)

6. **Accessibility Check**:
   - [ ] Focus styles defined for interactive elements
   - [ ] Color contrast ratios meet WCAG AA (4.5:1 for text)
   - [ ] Reduced motion support with `prefers-reduced-motion`
   - [ ] No content in `::before` or `::after` that should be in HTML

### B. Error Correction Process

If verification fails:

1. **Read the error message** carefully (compilation error, parsing error, lint error)
2. **Identify the root cause** (syntax error, undefined variable, invalid property, etc.)
3. **Fix the issue** in the generated CSS/SCSS
4. **Re-run verification** until all checks pass
5. **Document any browser-specific workarounds** in comments

### C. Agent Workflow Example

**Complete workflow for generating a button component:**

1. **Generate SCSS with documentation**:
   ```scss
   /// Button Component
   /// Primary action button with multiple variants
   /// @group components
   /// @example
   ///   <button class="btn btn--primary">Click Me</button>
   ///   <button class="btn btn--secondary">Cancel</button>
   
   // Button Variables
   $btn-padding-y: 0.75rem;
   $btn-padding-x: 1.5rem;
   $btn-font-size: 1rem;
   $btn-border-radius: 0.375rem;
   $btn-transition: all 0.2s ease-in-out;
   
   // Primary button color
   $btn-primary-bg: #3b82f6;
   $btn-primary-hover-bg: #2563eb;
   $btn-primary-text: #ffffff;
   
   /// Base button styles
   /// @since 1.0.0
   .btn {
     display: inline-flex;
     align-items: center;
     justify-content: center;
     padding: $btn-padding-y $btn-padding-x;
     font-size: $btn-font-size;
     font-weight: 500;
     line-height: 1.5;
     text-decoration: none;
     border: 2px solid transparent;
     border-radius: $btn-border-radius;
     cursor: pointer;
     transition: $btn-transition;
     
     &:focus-visible {
       outline: 2px solid currentColor;
       outline-offset: 2px;
     }
     
     /// Primary button variant
     &--primary {
       background-color: $btn-primary-bg;
       color: $btn-primary-text;
       
       &:hover,
       &:focus {
         background-color: $btn-primary-hover-bg;
       }
     }
   }
   ```

2. **Verify SCSS compiles**:
   ```bash
   npx sass button.scss button.css
   # ✓ Compiled successfully
   ```

3. **Run linting**:
   ```bash
   npx stylelint button.scss
   # ✓ No issues found
   ```

4. **Check accessibility**:
   - ✓ Focus styles defined (`:focus-visible`)
   - ✓ Sufficient color contrast
   - ✓ Interactive cursor states

5. **Generate documentation**:
   ```bash
   npx sassdoc src/
   # ✓ Documentation generated
   ```

6. **Present code** to user - only after ALL checks pass

### D. Prohibited Practices

**NEVER deliver CSS/SCSS that:**
- ❌ Has SCSS compilation errors
- ❌ Has CSS parsing errors
- ❌ Fails linting checks
- ❌ Uses magic numbers (undocumented hardcoded values)
- ❌ Has deep nesting (>3 levels in SCSS)
- ❌ Uses `!important` without justification
- ❌ Lacks focus styles for interactive elements
- ❌ Has poor color contrast (<4.5:1 for text)
- ❌ Uses deprecated CSS properties
- ❌ Lacks documentation for complex selectors/mixins
- ❌ Uses inline styles (style="")
- ❌ Has overly specific selectors (high specificity)
- ❌ **Fixes bugs without adding regression tests first**
- ❌ **Writes implementation before writing tests (violates TDD)**

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new CSS/SCSS development.**

### TDD Cycle for CSS/SCSS

```
1. 🔴 RED: Write a failing visual/style test first
   ↓
2. 🟢 GREEN: Write minimal CSS to make it pass
   ↓
3. 🔵 REFACTOR: Improve styles while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for CSS/SCSS

```javascript
// Step 1: RED - Write failing test first (tests/button.test.js)
import { test, expect } from 'vitest';
import { JSDOM } from 'jsdom';
import fs from 'fs';

test('primary button has correct background color', () => {
  const html = fs.readFileSync('src/index.html', 'utf-8');
  const css = fs.readFileSync('dist/styles.css', 'utf-8');
  const dom = new JSDOM(html, { runScripts: 'dangerously' });

  const style = dom.window.document.createElement('style');
  style.textContent = css;
  dom.window.document.head.appendChild(style);

  const button = dom.window.document.querySelector('.btn--primary');
  const styles = dom.window.getComputedStyle(button);

  expect(styles.backgroundColor).toBe('rgb(59, 130, 246)');
});

test('button has focus-visible styles', () => {
  // Verify focus styles are defined in CSS
  const css = fs.readFileSync('dist/styles.css', 'utf-8');
  expect(css).toContain(':focus-visible');
  expect(css).toContain('outline');
});

// Run: npm test
// ❌ FAILS - styles don't exist yet

// Step 2: GREEN - Write minimal SCSS
// src/components/_button.scss
/*
.btn--primary {
  background-color: #3b82f6;

  &:focus-visible {
    outline: 2px solid currentColor;
    outline-offset: 2px;
  }
}
*/

// Run: npm test
// ✅ PASSES - styles match expectations

// Step 3: REFACTOR - Extract variables
/*
$btn-primary-bg: #3b82f6;

.btn--primary {
  background-color: $btn-primary-bg;
}
*/
// Tests still pass
```

### Visual Regression Testing

```javascript
// Using Playwright for visual regression
import { test, expect } from '@playwright/test';

test('button component visual regression', async ({ page }) => {
  await page.goto('/components/button');

  // Capture screenshot and compare
  await expect(page.locator('.btn--primary')).toHaveScreenshot('button-primary.png');
  await expect(page.locator('.btn--secondary')).toHaveScreenshot('button-secondary.png');
});
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every CSS bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. 🐛 Bug Reported/Discovered (e.g., focus style missing)
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

### Example Bug Fix

```javascript
// Bug Report #567: Button focus state not visible on keyboard navigation

// Step 1-2: Write test that reproduces the bug
test('button has visible focus state - Bug #567', () => {
  // Bug: Users couldn't see focus indicator when tabbing
  // Discovered: 2026-01-15
  // This test prevents regression

  const css = fs.readFileSync('dist/styles.css', 'utf-8');

  // Check that focus-visible is defined with visible outline
  expect(css).toMatch(/\.btn[^{]*:focus-visible\s*\{[^}]*outline/);
  expect(css).not.toMatch(/outline:\s*none/);
});

// Run: npm test
// ❌ FAILS - focus styles are missing or outline: none

// Step 3: Fix the SCSS
// Before (buggy):
/*
.btn {
  &:focus {
    outline: none;  // BAD: removed focus indicator
  }
}
*/

// After (fixed):
/*
.btn {
  &:focus-visible {
    outline: 2px solid currentColor;
    outline-offset: 2px;
  }
}
*/

// Run: npm test
// ✅ PASSES - bug fixed, regression prevented
```

### Color Contrast Bug Fix Example

```javascript
// Bug Report #589: Text fails WCAG contrast on light background

test('text color meets WCAG AA contrast ratio - Bug #589', () => {
  const css = fs.readFileSync('dist/styles.css', 'utf-8');

  // Verify we're not using light gray on white
  expect(css).not.toMatch(/color:\s*#[c-f]{3,6}/i); // No light colors

  // Document: Text should be at least #737373 on white for 4.5:1 ratio
});
```

---

## 3. SCSS vs CSS: When to Use Each

### A. Use SCSS (Preferred) When:

**✅ SCSS is PREFERRED for:**
- Complex projects with multiple components
- Design systems requiring variables and theming
- Need for mixins, functions, and logic
- Component-based architecture
- Projects requiring maintainability at scale
- Team collaboration on large codebases

### B. Use Plain CSS When:

**✅ Plain CSS is ACCEPTABLE for:**
- Small projects (<3 pages)
- Simple landing pages
- Prototypes and demos
- CSS-only libraries (no build step)
- Learning/educational purposes

### C. SCSS Example vs CSS Equivalent

```scss
// ✅ CORRECT - SCSS (PREFERRED)
// Variables for maintainability
$primary-color: #3b82f6;
$spacing-unit: 0.5rem;
$breakpoint-md: 768px;

// Mixin for reusable patterns
@mixin flex-center {
  display: flex;
  align-items: center;
  justify-content: center;
}

// Nested selectors for readability
.card {
  padding: $spacing-unit * 2;
  background-color: white;
  border-radius: $spacing-unit;
  
  &__header {
    @include flex-center;
    padding: $spacing-unit;
    background-color: $primary-color;
    color: white;
  }
  
  &__body {
    padding: $spacing-unit * 2;
  }
  
  @media (min-width: $breakpoint-md) {
    padding: $spacing-unit * 4;
  }
}
```

```css
/* ✅ ACCEPTABLE - Plain CSS (for simple projects) */
/* Note: Less maintainable, more repetitive */
.card {
  padding: 1rem;
  background-color: white;
  border-radius: 0.5rem;
}

.card__header {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0.5rem;
  background-color: #3b82f6;
  color: white;
}

.card__body {
  padding: 1rem;
}

@media (min-width: 768px) {
  .card {
    padding: 2rem;
  }
}
```

---

## 4. SCSS Architecture (MANDATORY)

### A. File Structure (7-1 Pattern)

```
styles/
├── abstracts/
│   ├── _variables.scss     # All variables
│   ├── _functions.scss     # SCSS functions
│   ├── _mixins.scss        # Reusable mixins
│   └── _index.scss         # Import all abstracts
├── base/
│   ├── _reset.scss         # CSS reset
│   ├── _typography.scss    # Typography rules
│   └── _index.scss         # Import all base
├── components/
│   ├── _button.scss        # Button component
│   ├── _card.scss          # Card component
│   ├── _form.scss          # Form elements
│   └── _index.scss         # Import all components
├── layout/
│   ├── _header.scss        # Header layout
│   ├── _footer.scss        # Footer layout
│   ├── _grid.scss          # Grid system
│   └── _index.scss         # Import all layout
├── pages/
│   ├── _home.scss          # Home page specific
│   ├── _about.scss         # About page specific
│   └── _index.scss         # Import all pages
├── themes/
│   ├── _dark.scss          # Dark theme
│   ├── _light.scss         # Light theme (default)
│   └── _index.scss         # Import all themes
├── vendors/
│   ├── _normalize.scss     # Normalize.css
│   └── _index.scss         # Import all vendors
└── main.scss               # Main file importing all
```

### B. Main SCSS File

```scss
// main.scss - Import order is important!

// 1. Configuration and helpers
@use 'abstracts' as *;

// 2. Vendors (third-party CSS)
@use 'vendors';

// 3. Base styles
@use 'base';

// 4. Layout
@use 'layout';

// 5. Components
@use 'components';

// 6. Pages
@use 'pages';

// 7. Themes
@use 'themes';
```

### C. Variables File Structure

```scss
// abstracts/_variables.scss

/// @group colors
/// Primary brand colors
$color-primary: #3b82f6;
$color-primary-light: #60a5fa;
$color-primary-dark: #2563eb;

/// Secondary brand colors
$color-secondary: #8b5cf6;
$color-secondary-light: #a78bfa;
$color-secondary-dark: #7c3aed;

/// Semantic colors
$color-success: #10b981;
$color-warning: #f59e0b;
$color-error: #ef4444;
$color-info: #3b82f6;

/// Neutral colors
$color-gray-50: #f9fafb;
$color-gray-100: #f3f4f6;
$color-gray-200: #e5e7eb;
$color-gray-300: #d1d5db;
$color-gray-400: #9ca3af;
$color-gray-500: #6b7280;
$color-gray-600: #4b5563;
$color-gray-700: #374151;
$color-gray-800: #1f2937;
$color-gray-900: #111827;

/// @group spacing
/// Base spacing unit (0.25rem = 4px)
$spacing-unit: 0.25rem;

/// Spacing scale
$spacing-1: $spacing-unit;        // 4px
$spacing-2: $spacing-unit * 2;    // 8px
$spacing-3: $spacing-unit * 3;    // 12px
$spacing-4: $spacing-unit * 4;    // 16px
$spacing-6: $spacing-unit * 6;    // 24px
$spacing-8: $spacing-unit * 8;    // 32px
$spacing-12: $spacing-unit * 12;  // 48px
$spacing-16: $spacing-unit * 16;  // 64px

/// @group typography
/// Font families
$font-family-sans: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
$font-family-serif: Georgia, Cambria, 'Times New Roman', serif;
$font-family-mono: 'Monaco', 'Courier New', monospace;

/// Font sizes
$font-size-xs: 0.75rem;   // 12px
$font-size-sm: 0.875rem;  // 14px
$font-size-base: 1rem;    // 16px
$font-size-lg: 1.125rem;  // 18px
$font-size-xl: 1.25rem;   // 20px
$font-size-2xl: 1.5rem;   // 24px
$font-size-3xl: 1.875rem; // 30px
$font-size-4xl: 2.25rem;  // 36px

/// Font weights
$font-weight-light: 300;
$font-weight-normal: 400;
$font-weight-medium: 500;
$font-weight-semibold: 600;
$font-weight-bold: 700;

/// Line heights
$line-height-tight: 1.25;
$line-height-normal: 1.5;
$line-height-relaxed: 1.75;

/// @group breakpoints
/// Responsive breakpoints
$breakpoint-sm: 640px;   // Small devices
$breakpoint-md: 768px;   // Medium devices
$breakpoint-lg: 1024px;  // Large devices
$breakpoint-xl: 1280px;  // Extra large devices
$breakpoint-2xl: 1536px; // 2X large devices

/// @group transitions
/// Transition durations
$transition-fast: 150ms;
$transition-base: 200ms;
$transition-slow: 300ms;

/// Transition easing
$ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);
$ease-out: cubic-bezier(0, 0, 0.2, 1);
$ease-in: cubic-bezier(0.4, 0, 1, 1);

/// @group shadows
/// Box shadows
$shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
$shadow-base: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
$shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
$shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);

/// @group z-index
/// Z-index layers
$z-index-dropdown: 1000;
$z-index-sticky: 1020;
$z-index-fixed: 1030;
$z-index-modal-backdrop: 1040;
$z-index-modal: 1050;
$z-index-popover: 1060;
$z-index-tooltip: 1070;
```

---

## 5. Mixins and Functions (MANDATORY)

### A. Essential Mixins

```scss
// abstracts/_mixins.scss

/// Responsive breakpoint mixin
/// @param {String} $breakpoint - Breakpoint name (sm, md, lg, xl, 2xl)
/// @example
///   .element {
///     @include breakpoint(md) {
///       font-size: 2rem;
///     }
///   }
@mixin breakpoint($breakpoint) {
  @if $breakpoint == sm {
    @media (min-width: $breakpoint-sm) { @content; }
  } @else if $breakpoint == md {
    @media (min-width: $breakpoint-md) { @content; }
  } @else if $breakpoint == lg {
    @media (min-width: $breakpoint-lg) { @content; }
  } @else if $breakpoint == xl {
    @media (min-width: $breakpoint-xl) { @content; }
  } @else if $breakpoint == 2xl {
    @media (min-width: $breakpoint-2xl) { @content; }
  }
}

/// Flexbox center alignment
/// @example
///   .container { @include flex-center; }
@mixin flex-center {
  display: flex;
  align-items: center;
  justify-content: center;
}

/// Flexbox space-between layout
/// @example
///   .header { @include flex-between; }
@mixin flex-between {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

/// Truncate text with ellipsis
/// @example
///   .text { @include truncate; }
@mixin truncate {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/// Multiline text truncation
/// @param {Number} $lines - Number of lines to show
/// @example
///   .description { @include line-clamp(3); }
@mixin line-clamp($lines: 2) {
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: $lines;
  overflow: hidden;
}

/// Visually hide element (accessible)
/// @example
///   .sr-only { @include visually-hidden; }
@mixin visually-hidden {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

/// Focus outline style
/// @param {Color} $color - Outline color (default: currentColor)
/// @example
///   button { @include focus-outline(#3b82f6); }
@mixin focus-outline($color: currentColor) {
  &:focus-visible {
    outline: 2px solid $color;
    outline-offset: 2px;
  }
}

/// Smooth font rendering
/// @example
///   body { @include font-smoothing; }
@mixin font-smoothing {
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/// Aspect ratio box
/// @param {Number} $width - Width ratio
/// @param {Number} $height - Height ratio
/// @example
///   .video-container { @include aspect-ratio(16, 9); }
@mixin aspect-ratio($width: 16, $height: 9) {
  position: relative;
  
  &::before {
    content: '';
    display: block;
    padding-bottom: calc(($height / $width) * 100%);
  }
  
  > * {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
  }
}

/// Clearfix for floats
/// @example
///   .container { @include clearfix; }
@mixin clearfix {
  &::after {
    content: '';
    display: table;
    clear: both;
  }
}

/// Hover effect with transition
/// @param {String} $property - CSS property to transition
/// @param {Duration} $duration - Transition duration (default: $transition-base)
/// @example
///   .button { @include hover-transition(background-color); }
@mixin hover-transition($property, $duration: $transition-base) {
  transition: $property $duration $ease-in-out;
  
  @media (prefers-reduced-motion: reduce) {
    transition: none;
  }
}

/// Generate utility classes with variants
/// @param {String} $property - CSS property
/// @param {String} $prefix - Class prefix
/// @param {Map} $values - Map of suffix: value pairs
/// @example
///   @include generate-utilities('margin-top', 'mt', (
///     '0': 0,
///     '1': $spacing-1,
///     '2': $spacing-2
///   ));
@mixin generate-utilities($property, $prefix, $values) {
  @each $key, $value in $values {
    .#{$prefix}-#{$key} {
      #{$property}: $value;
    }
  }
}
```

### B. Essential Functions

```scss
// abstracts/_functions.scss

/// Convert pixels to rem
/// @param {Number} $pixels - Pixel value
/// @param {Number} $base - Base font size (default: 16px)
/// @return {Number} - Rem value
/// @example
///   font-size: rem(24); // 1.5rem
@function rem($pixels, $base: 16) {
  @return calc($pixels / $base) * 1rem;
}

/// Convert pixels to em
/// @param {Number} $pixels - Pixel value
/// @param {Number} $base - Base font size (default: 16px)
/// @return {Number} - Em value
/// @example
///   padding: em(12); // 0.75em
@function em($pixels, $base: 16) {
  @return calc($pixels / $base) * 1em;
}

/// Lighten color
/// @param {Color} $color - Base color
/// @param {Number} $amount - Amount to lighten (0-100)
/// @return {Color} - Lightened color
/// @example
///   background: tint(#3b82f6, 20%);
@function tint($color, $amount) {
  @return mix(white, $color, $amount);
}

/// Darken color
/// @param {Color} $color - Base color
/// @param {Number} $amount - Amount to darken (0-100)
/// @return {Color} - Darkened color
/// @example
///   background: shade(#3b82f6, 20%);
@function shade($color, $amount) {
  @return mix(black, $color, $amount);
}

/// Get color with opacity
/// @param {Color} $color - Base color
/// @param {Number} $opacity - Opacity value (0-1)
/// @return {Color} - Color with opacity
/// @example
///   background: alpha(#3b82f6, 0.5);
@function alpha($color, $opacity) {
  @return rgba($color, $opacity);
}

/// Strip unit from value
/// @param {Number} $value - Value with unit
/// @return {Number} - Unitless value
/// @example
///   $num: strip-unit(16px); // 16
@function strip-unit($value) {
  @return calc($value / ($value * 0 + 1));
}

/// Get z-index from map
/// @param {String} $layer - Layer name
/// @return {Number} - Z-index value
/// @example
///   z-index: z($layer: 'modal');
@function z($layer) {
  $z-indexes: (
    'dropdown': $z-index-dropdown,
    'sticky': $z-index-sticky,
    'fixed': $z-index-fixed,
    'modal-backdrop': $z-index-modal-backdrop,
    'modal': $z-index-modal,
    'popover': $z-index-popover,
    'tooltip': $z-index-tooltip
  );
  
  @return map-get($z-indexes, $layer);
}
```

---

## 6. Modern CSS Features (MANDATORY)

### A. CSS Custom Properties (Variables)

```css
/* ✅ CORRECT - CSS Custom Properties for theming */
:root {
  /* Colors */
  --color-primary: #3b82f6;
  --color-primary-light: #60a5fa;
  --color-primary-dark: #2563eb;
  
  /* Spacing */
  --spacing-xs: 0.25rem;
  --spacing-sm: 0.5rem;
  --spacing-md: 1rem;
  --spacing-lg: 1.5rem;
  --spacing-xl: 2rem;
  
  /* Typography */
  --font-size-sm: 0.875rem;
  --font-size-base: 1rem;
  --font-size-lg: 1.125rem;
  --line-height: 1.5;
  
  /* Shadows */
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  
  /* Transitions */
  --transition-fast: 150ms;
  --transition-base: 200ms;
}

/* Dark theme */
[data-theme="dark"] {
  --color-primary: #60a5fa;
  --color-background: #1f2937;
  --color-text: #f9fafb;
}

/* Usage */
.button {
  background-color: var(--color-primary);
  padding: var(--spacing-md) var(--spacing-lg);
  font-size: var(--font-size-base);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-base) ease-in-out;
}
```

### B. CSS Grid Layout

```css
/* ✅ CORRECT - Modern CSS Grid */
.grid-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  padding: 2rem;
}

/* Named grid areas */
.layout {
  display: grid;
  grid-template-areas:
    "header header header"
    "sidebar main main"
    "footer footer footer";
  grid-template-columns: 250px 1fr 1fr;
  grid-template-rows: auto 1fr auto;
  gap: 1rem;
  min-height: 100vh;
}

.header {
  grid-area: header;
}

.sidebar {
  grid-area: sidebar;
}

.main {
  grid-area: main;
}

.footer {
  grid-area: footer;
}

/* Responsive grid */
@media (max-width: 768px) {
  .layout {
    grid-template-areas:
      "header"
      "main"
      "sidebar"
      "footer";
    grid-template-columns: 1fr;
  }
}
```

### C. Flexbox Layout

```css
/* ✅ CORRECT - Modern Flexbox patterns */
.flex-container {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem; /* Modern gap property */
  align-items: center;
  justify-content: space-between;
}

/* Flex items */
.flex-item {
  flex: 1 1 300px; /* grow shrink basis */
}

/* Center content */
.center {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
}

/* Sticky footer with flexbox */
.page {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

.main-content {
  flex: 1;
}
```

### D. Container Queries

```css
/* ✅ CORRECT - Container Queries (modern feature) */
.card-container {
  container-type: inline-size;
  container-name: card;
}

.card {
  padding: 1rem;
}

/* Style based on container size, not viewport */
@container card (min-width: 400px) {
  .card {
    display: grid;
    grid-template-columns: 150px 1fr;
    gap: 1rem;
    padding: 2rem;
  }
  
  .card__image {
    aspect-ratio: 1;
  }
}
```

### E. Logical Properties

```css
/* ✅ CORRECT - Logical properties for i18n */
.element {
  /* Instead of margin-left/right */
  margin-inline: 1rem;
  
  /* Instead of margin-top/bottom */
  margin-block: 1rem;
  
  /* Instead of padding-left */
  padding-inline-start: 1rem;
  
  /* Instead of padding-right */
  padding-inline-end: 1rem;
  
  /* Instead of border-left */
  border-inline-start: 1px solid black;
}
```

### F. Modern Selectors

```css
/* ✅ CORRECT - Modern CSS selectors */

/* :is() - Reduce repetition */
:is(h1, h2, h3, h4, h5, h6) {
  font-weight: bold;
  line-height: 1.2;
}

/* :where() - Zero specificity */
:where(ul, ol) :where(ul, ol) {
  margin-block: 0;
}

/* :has() - Parent selector */
.card:has(.card__image) {
  display: grid;
  grid-template-columns: 200px 1fr;
}

/* :not() with multiple selectors */
button:not([disabled], [aria-disabled="true"]) {
  cursor: pointer;
}

/* :focus-visible - Only on keyboard focus */
button:focus-visible {
  outline: 2px solid blue;
  outline-offset: 2px;
}
```

---

## 7. BEM Naming Convention (MANDATORY)

### A. BEM Structure

```scss
// ✅ CORRECT - BEM (Block Element Modifier)

// Block: Standalone component
.card {
  padding: 1rem;
  background: white;
  border-radius: 0.5rem;
  
  // Element: Part of the block
  &__header {
    padding: 1rem;
    border-bottom: 1px solid #e5e7eb;
  }
  
  &__title {
    font-size: 1.5rem;
    font-weight: 600;
    margin: 0;
  }
  
  &__body {
    padding: 1rem;
  }
  
  &__footer {
    padding: 1rem;
    border-top: 1px solid #e5e7eb;
  }
  
  // Modifier: Variation of the block
  &--featured {
    border: 2px solid #3b82f6;
    box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
  }
  
  &--compact {
    padding: 0.5rem;
    
    .card__header,
    .card__body,
    .card__footer {
      padding: 0.5rem;
    }
  }
}

// Usage in HTML:
// <div class="card card--featured">
//   <header class="card__header">
//     <h2 class="card__title">Title</h2>
//   </header>
//   <div class="card__body">Content</div>
//   <footer class="card__footer">Footer</footer>
// </div>
```

### B. BEM Best Practices

```scss
// ✅ CORRECT - Proper BEM usage

// Button component
.btn {
  display: inline-flex;
  padding: 0.5rem 1rem;
  
  &__icon {
    margin-inline-end: 0.5rem;
  }
  
  &__text {
    font-weight: 500;
  }
  
  &--primary {
    background: #3b82f6;
    color: white;
  }
  
  &--large {
    padding: 1rem 2rem;
    font-size: 1.125rem;
  }
  
  // Combined modifiers
  &--primary#{&}--large {
    box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
  }
}


// ❌ WRONG - Not following BEM

// Deep nesting (avoid)
.card {
  .header {
    .title {
      .icon {
        // Too deep!
      }
    }
  }
}

// Generic names (avoid)
.title {}
.content {}
.button {}

// Mixing methodologies (avoid)
.card__header-title {}  // Mix of BEM and kebab-case
```

---

## 8. Responsive Design (MANDATORY)

### A. Mobile-First Approach

```scss
// ✅ CORRECT - Mobile-first (PREFERRED)
.element {
  // Mobile styles (default)
  font-size: 1rem;
  padding: 1rem;
  
  // Tablet and up
  @include breakpoint(md) {
    font-size: 1.125rem;
    padding: 1.5rem;
  }
  
  // Desktop and up
  @include breakpoint(lg) {
    font-size: 1.25rem;
    padding: 2rem;
  }
}


// ❌ WRONG - Desktop-first (avoid)
.element {
  font-size: 1.25rem;
  padding: 2rem;
  
  @media (max-width: 1024px) {
    font-size: 1.125rem;
    padding: 1.5rem;
  }
  
  @media (max-width: 768px) {
    font-size: 1rem;
    padding: 1rem;
  }
}
```

### B. Fluid Typography

```scss
// ✅ CORRECT - Fluid typography with clamp()
.heading {
  // min: 1.5rem (24px), preferred: 4vw, max: 3rem (48px)
  font-size: clamp(1.5rem, 4vw, 3rem);
  line-height: 1.2;
}

.body-text {
  // Fluid font size between 1rem and 1.25rem
  font-size: clamp(1rem, 0.875rem + 0.5vw, 1.25rem);
}

// Fluid spacing
.section {
  padding: clamp(2rem, 5vw, 4rem) clamp(1rem, 3vw, 2rem);
}
```

### C. Container Width

```scss
// ✅ CORRECT - Fluid container with max-width
.container {
  width: 100%;
  max-width: 1280px;
  margin-inline: auto;
  padding-inline: 1rem;
  
  @include breakpoint(md) {
    padding-inline: 2rem;
  }
  
  @include breakpoint(lg) {
    padding-inline: 4rem;
  }
}
```

---

## 9. Accessibility (MANDATORY)

### A. Focus Styles

```scss
// ✅ CORRECT - Accessible focus styles (REQUIRED)
button,
a,
input,
select,
textarea {
  // Remove default outline
  &:focus {
    outline: none;
  }
  
  // Add custom focus indicator
  &:focus-visible {
    outline: 2px solid currentColor;
    outline-offset: 2px;
  }
}

// Custom focus style
.btn {
  &:focus-visible {
    outline: 2px solid #3b82f6;
    outline-offset: 2px;
    box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2);
  }
}


// ❌ WRONG - Removing focus without replacement
button {
  outline: none; /* BAD: No focus indicator */
}
```

### B. Color Contrast

```scss
// ✅ CORRECT - WCAG AA contrast ratios (REQUIRED)

// Text contrast: 4.5:1 minimum for normal text
.text {
  color: #1f2937;  // Dark gray on white: 14.5:1 ✓
  background: white;
}

// Large text contrast: 3:1 minimum for 18px+ or 14px+ bold
.heading {
  color: #4b5563;  // Medium gray on white: 7:1 ✓
  background: white;
  font-size: 1.5rem;
}

// Interactive elements
.btn--primary {
  color: white;           // White on blue: 4.5:1+ ✓
  background: #2563eb;
}


// ❌ WRONG - Poor contrast
.text-bad {
  color: #d1d5db;  // Light gray on white: 1.8:1 ✗
  background: white;
}
```

### C. Reduced Motion

```scss
// ✅ CORRECT - Respect user preferences (REQUIRED)
.animated-element {
  animation: slide-in 0.3s ease-out;
  
  @media (prefers-reduced-motion: reduce) {
    animation: none;
  }
}

// Disable transitions for reduced motion
* {
  @media (prefers-reduced-motion: reduce) {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

// Mixin for reduced motion
@mixin respect-motion-preference {
  @media (prefers-reduced-motion: reduce) {
    @content;
  }
}

// Usage
.element {
  transition: transform 0.3s;
  
  @include respect-motion-preference {
    transition: none;
  }
}
```

### D. Screen Reader Only

```scss
// ✅ CORRECT - Visually hidden but accessible to screen readers
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

// Focusable variant (for skip links)
.sr-only-focusable {
  @extend .sr-only;
  
  &:focus,
  &:active {
    position: static;
    width: auto;
    height: auto;
    overflow: visible;
    clip: auto;
    white-space: normal;
  }
}
```

---

## 10. Performance Optimization (MANDATORY)

### A. Selector Specificity

```scss
// ✅ CORRECT - Low specificity (PREFERRED)
.card {
  padding: 1rem;
}

.card--featured {
  border: 2px solid blue;
}

// ❌ WRONG - High specificity (AVOID)
div.card.featured#main-card {
  padding: 1rem;
}

body div.container section.content div.card {
  // Too specific!
}
```

### B. Avoid Expensive Properties

```scss
// ✅ CORRECT - Performant animations
.element {
  // Use transform and opacity (GPU accelerated)
  transition: transform 0.3s, opacity 0.3s;
  
  &:hover {
    transform: translateY(-2px);
  }
}

// ❌ WRONG - Expensive properties
.element-bad {
  transition: height 0.3s, width 0.3s, top 0.3s, left 0.3s;
  // These trigger layout recalculation!
}
```

### C. Critical CSS

```scss
// ✅ CORRECT - Separate critical CSS
// critical.scss - Above-the-fold styles
body {
  margin: 0;
  font-family: system-ui, sans-serif;
}

.header {
  background: white;
  padding: 1rem;
}

.hero {
  min-height: 50vh;
}

// Load critical CSS inline in <head>
// Load non-critical CSS asynchronously
```

### D. Will-Change Property

```scss
// ✅ CORRECT - Use will-change sparingly
.animated-element {
  // Only when animation is imminent
  &.is-animating {
    will-change: transform, opacity;
  }
  
  // Remove after animation
  &.animation-complete {
    will-change: auto;
  }
}

// ❌ WRONG - Overusing will-change
.element {
  will-change: transform, opacity, width, height; // Too many!
}
```

---

## 11. Documentation (MANDATORY)

### A. SassDoc Comments

```scss
/// Button Component
/// @group components
/// @author John Doe
/// @since 1.0.0
/// @example scss - Basic button
///   .btn {
///     @include button-base;
///   }

/// Button base mixin
/// Creates base button styles with consistent sizing and spacing
/// @param {Color} $bg-color [$color-primary] - Background color
/// @param {Color} $text-color [white] - Text color
/// @param {Number} $padding-y [0.75rem] - Vertical padding
/// @param {Number} $padding-x [1.5rem] - Horizontal padding
/// @output Base button styles with hover and focus states
/// @example scss - Primary button
///   .btn-primary {
///     @include button-base($color-primary, white);
///   }
/// @example scss - Custom button
///   .btn-custom {
///     @include button-base(#ff0000, white, 1rem, 2rem);
///   }
@mixin button-base(
  $bg-color: $color-primary,
  $text-color: white,
  $padding-y: 0.75rem,
  $padding-x: 1.5rem
) {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: $padding-y $padding-x;
  background-color: $bg-color;
  color: $text-color;
  border: none;
  border-radius: 0.375rem;
  cursor: pointer;
  transition: background-color 0.2s;
  
  &:hover {
    background-color: darken($bg-color, 10%);
  }
  
  &:focus-visible {
    outline: 2px solid $bg-color;
    outline-offset: 2px;
  }
}

/// Calculate fluid font size
/// @param {Number} $min-size - Minimum font size
/// @param {Number} $max-size - Maximum font size
/// @param {Number} $min-vw [320px] - Minimum viewport width
/// @param {Number} $max-vw [1200px] - Maximum viewport width
/// @return {String} - Clamp function with fluid sizing
/// @example scss
///   h1 {
///     font-size: fluid-font-size(1.5rem, 3rem);
///   }
@function fluid-font-size(
  $min-size,
  $max-size,
  $min-vw: 320px,
  $max-vw: 1200px
) {
  $slope: calc(($max-size - $min-size) / ($max-vw - $min-vw));
  $y-intercept: $min-size - $slope * $min-vw;
  
  @return clamp(
    $min-size,
    calc($y-intercept + $slope * 100vw),
    $max-size
  );
}
```

### B. File Header Comments

```scss
// components/_button.scss

/**
 * Button Component
 * 
 * Reusable button styles with multiple variants.
 * 
 * @package MyApp
 * @subpackage Components
 * @version 1.0.0
 * @author John Doe <john@example.com>
 * @since 2026-01-17
 * 
 * Usage:
 * <button class="btn btn--primary">Click Me</button>
 * <button class="btn btn--secondary">Cancel</button>
 * <button class="btn btn--outline">Learn More</button>
 */

// Button variables
$btn-font-size: 1rem;
$btn-padding-y: 0.75rem;
$btn-padding-x: 1.5rem;
// ...rest of file
```

### C. Section Comments

```scss
// ============================================================================
// Typography
// ============================================================================

// Base typography styles
body {
  font-family: $font-family-sans;
  font-size: $font-size-base;
  line-height: $line-height-normal;
}

// ----------------------------------------------------------------------------
// Headings
// ----------------------------------------------------------------------------

h1, h2, h3, h4, h5, h6 {
  font-weight: $font-weight-bold;
  line-height: $line-height-tight;
}

// ----------------------------------------------------------------------------
// Links
// ----------------------------------------------------------------------------

a {
  color: $color-primary;
  text-decoration: none;
  
  &:hover {
    text-decoration: underline;
  }
}
```

---

## 12. Configuration Files

### A. package.json

```json
{
  "name": "my-project",
  "version": "1.0.0",
  "scripts": {
    "sass:dev": "sass --watch src/scss:dist/css --style expanded",
    "sass:build": "sass src/scss:dist/css --style compressed",
    "postcss": "postcss dist/css/*.css --use autoprefixer --replace",
    "lint": "stylelint 'src/**/*.{css,scss}'",
    "lint:fix": "stylelint 'src/**/*.{css,scss}' --fix",
    "docs": "sassdoc src/scss",
    "build": "npm run sass:build && npm run postcss",
    "dev": "npm run sass:dev"
  },
  "devDependencies": {
    "sass": "^1.69.0",
    "postcss": "^8.4.32",
    "postcss-cli": "^11.0.0",
    "autoprefixer": "^10.4.16",
    "stylelint": "^16.0.0",
    "stylelint-config-standard-scss": "^12.0.0",
    "stylelint-order": "^6.0.4",
    "sassdoc": "^2.7.4"
  },
  "browserslist": [
    "last 2 versions",
    "> 1%",
    "not dead"
  ]
}
```

### B. .stylelintrc.json

```json
{
  "extends": [
    "stylelint-config-standard-scss"
  ],
  "plugins": [
    "stylelint-order"
  ],
  "rules": {
    "indentation": 2,
    "string-quotes": "single",
    "no-duplicate-selectors": true,
    "color-hex-length": "long",
    "color-named": "never",
    "selector-max-id": 0,
    "selector-max-specificity": "0,3,0",
    "selector-class-pattern": "^[a-z][a-z0-9]*(-[a-z0-9]+)*((__[a-z0-9]+(-[a-z0-9]+)*)?(--[a-z0-9]+(-[a-z0-9]+)*)?)$",
    "selector-no-qualifying-type": true,
    "number-leading-zero": "always",
    "font-weight-notation": "numeric",
    "font-family-name-quotes": "always-where-recommended",
    "comment-whitespace-inside": "always",
    "at-rule-no-vendor-prefix": true,
    "media-feature-name-no-vendor-prefix": true,
    "property-no-vendor-prefix": true,
    "selector-no-vendor-prefix": true,
    "value-no-vendor-prefix": true,
    "max-nesting-depth": 3,
    "selector-max-compound-selectors": 4,
    "declaration-no-important": true,
    "order/properties-alphabetical-order": true
  }
}
```

### C. .browserslistrc

```
# Browsers that we support
last 2 versions
> 1%
not dead
not IE 11
```

### D. postcss.config.js

```javascript
module.exports = {
  plugins: [
    require('autoprefixer'),
    require('cssnano')({
      preset: 'default',
    }),
  ],
};
```

---

## 13. Complete Example

```scss
// components/_card.scss

/**
 * Card Component
 * 
 * Flexible card component for displaying content in a contained box.
 * Supports multiple variants and responsive behavior.
 * 
 * @package MyApp
 * @subpackage Components
 * @version 1.0.0
 * @author John Doe
 * @since 2026-01-17
 */

// ============================================================================
// Variables
// ============================================================================

/// Card padding
$card-padding: $spacing-4;

/// Card border radius
$card-border-radius: $spacing-2;

/// Card shadow
$card-shadow: $shadow-md;

/// Card background color
$card-bg: white;

/// Card border color
$card-border-color: $color-gray-200;


// ============================================================================
// Mixins
// ============================================================================

/// Card hover effect
/// @access private
@mixin card-hover {
  transition: box-shadow $transition-base $ease-in-out,
              transform $transition-base $ease-in-out;
  
  &:hover {
    box-shadow: $shadow-lg;
    transform: translateY(-2px);
  }
  
  @media (prefers-reduced-motion: reduce) {
    transition: none;
    
    &:hover {
      transform: none;
    }
  }
}


// ============================================================================
// Component
// ============================================================================

/// Card component
/// @group components
/// @example html
///   <div class="card">
///     <header class="card__header">
///       <h3 class="card__title">Card Title</h3>
///     </header>
///     <div class="card__body">
///       <p>Card content goes here.</p>
///     </div>
///     <footer class="card__footer">
///       <button class="btn btn--primary">Action</button>
///     </footer>
///   </div>
.card {
  display: flex;
  flex-direction: column;
  background-color: $card-bg;
  border: 1px solid $card-border-color;
  border-radius: $card-border-radius;
  box-shadow: $card-shadow;
  overflow: hidden;
  
  // ----------------------------------------------------------------------------
  // Elements
  // ----------------------------------------------------------------------------
  
  /// Card header
  &__header {
    padding: $card-padding;
    border-bottom: 1px solid $card-border-color;
    background-color: $color-gray-50;
  }
  
  /// Card title
  &__title {
    margin: 0;
    font-size: $font-size-lg;
    font-weight: $font-weight-semibold;
    line-height: $line-height-tight;
    color: $color-gray-900;
  }
  
  /// Card subtitle
  &__subtitle {
    margin: $spacing-1 0 0;
    font-size: $font-size-sm;
    color: $color-gray-600;
  }
  
  /// Card body
  &__body {
    flex: 1;
    padding: $card-padding;
    
    > *:first-child {
      margin-top: 0;
    }
    
    > *:last-child {
      margin-bottom: 0;
    }
  }
  
  /// Card footer
  &__footer {
    padding: $card-padding;
    border-top: 1px solid $card-border-color;
    background-color: $color-gray-50;
    
    // Align buttons to the right
    display: flex;
    justify-content: flex-end;
    gap: $spacing-2;
  }
  
  /// Card image
  &__image {
    width: 100%;
    height: auto;
    object-fit: cover;
  }
  
  // ----------------------------------------------------------------------------
  // Modifiers
  // ----------------------------------------------------------------------------
  
  /// Hoverable card variant
  &--hoverable {
    @include card-hover;
    cursor: pointer;
  }
  
  /// Featured card variant
  &--featured {
    border: 2px solid $color-primary;
    box-shadow: $shadow-lg;
    
    .card__header {
      background-color: $color-primary;
      color: white;
      border-bottom-color: $color-primary-dark;
    }
    
    .card__title {
      color: white;
    }
  }
  
  /// Compact card variant
  &--compact {
    .card__header,
    .card__body,
    .card__footer {
      padding: $spacing-2;
    }
  }
  
  /// Horizontal card layout (responsive)
  &--horizontal {
    @include breakpoint(md) {
      flex-direction: row;
      
      .card__image {
        width: 250px;
        height: 100%;
        object-fit: cover;
      }
      
      .card__content {
        flex: 1;
        display: flex;
        flex-direction: column;
      }
    }
  }
  
  /// Outlined card variant (no shadow)
  &--outlined {
    box-shadow: none;
    border: 2px solid $card-border-color;
    
    &:hover {
      border-color: $color-primary;
    }
  }
  
  // ----------------------------------------------------------------------------
  // States
  // ----------------------------------------------------------------------------
  
  /// Loading state
  &.is-loading {
    opacity: 0.6;
    pointer-events: none;
    
    .card__body::after {
      content: '';
      position: absolute;
      inset: 0;
      background: linear-gradient(
        90deg,
        transparent,
        rgba(255, 255, 255, 0.5),
        transparent
      );
      animation: loading 1.5s infinite;
    }
  }
  
  /// Selected state
  &.is-selected {
    border-color: $color-primary;
    box-shadow: 0 0 0 3px rgba($color-primary, 0.2);
  }
  
  // ----------------------------------------------------------------------------
  // Responsive
  // ----------------------------------------------------------------------------
  
  @include breakpoint(md) {
    &__header {
      padding: $spacing-6;
    }
    
    &__body {
      padding: $spacing-6;
    }
    
    &__footer {
      padding: $spacing-6;
    }
  }
}

// ============================================================================
// Animations
// ============================================================================

@keyframes loading {
  0% {
    transform: translateX(-100%);
  }
  100% {
    transform: translateX(100%);
  }
}
```

---

## 14. Deployment Checklist

### Pre-Production Validation

#### Compilation & Parsing (MANDATORY)
- [ ] **SCSS compiles without errors**: `sass --check src/**/*.scss` passes
- [ ] **CSS parses correctly**: No syntax errors in output
- [ ] **Autoprefixer applied**: Vendor prefixes added for target browsers
- [ ] **Minified for production**: CSS file size optimized
- [ ] **Source maps generated**: For debugging production issues

#### Linting & Quality (MANDATORY)
- [ ] **Stylelint passes**: `stylelint "**/*.{css,scss}"` returns exit code 0
- [ ] **No `!important` without justification**: Specificity managed properly
- [ ] **Max nesting depth ≤ 3**: No deeply nested selectors
- [ ] **Selector specificity low**: No overly specific selectors
- [ ] **BEM naming followed**: Consistent class naming convention

#### Accessibility (MANDATORY - WCAG 2.1 AA)
- [ ] **Focus styles present**: All interactive elements have `:focus-visible` styles
- [ ] **Color contrast ≥ 4.5:1**: Text meets WCAG AA requirements
- [ ] **Reduced motion support**: `prefers-reduced-motion` media query implemented
- [ ] **No content in pseudo-elements**: Content belongs in HTML, not CSS

#### Performance
- [ ] **Critical CSS identified**: Above-the-fold styles separated
- [ ] **Unused CSS removed**: No dead code in production
- [ ] **File size < 100KB**: Uncompressed CSS file size optimized
- [ ] **No expensive properties in animations**: Only transform/opacity animated
- [ ] **will-change used sparingly**: Only when necessary

#### Documentation
- [ ] **All mixins documented**: SassDoc comments present
- [ ] **All functions documented**: Parameters and return values described
- [ ] **Complex selectors commented**: Explanation for non-obvious code
- [ ] **Documentation generated**: `sassdoc` output up to date

#### Browser Compatibility
- [ ] **Tested in Chrome**: Latest version
- [ ] **Tested in Firefox**: Latest version
- [ ] **Tested in Safari**: Latest version
- [ ] **Tested in Edge**: Latest version
- [ ] **Vendor prefixes present**: For properties requiring them

#### Responsive Design
- [ ] **Mobile-first approach**: Base styles for mobile, enhanced for desktop
- [ ] **Breakpoints consistent**: Using defined breakpoint variables
- [ ] **Tested on mobile devices**: iOS Safari and Chrome Android
- [ ] **Fluid typography implemented**: Using clamp() where appropriate

---

## 15. Why This Configuration Works

1. **SCSS Over CSS**: Reduces code duplication by 40-60%, improves maintainability through variables and mixins, enables component-based architecture.

2. **BEM Methodology**: Eliminates specificity wars, makes class names self-documenting, reduces naming conflicts by 90%.

3. **Mobile-First**: Results in 30-40% smaller file size, faster mobile performance, easier to enhance than strip down.

4. **Modern CSS Features**: Grid/Flexbox reduce layout code by 50%, Custom Properties enable theming without rebuilds, Container Queries improve component reusability.

5. **Strict Linting**: Catches 70% of bugs before production, enforces consistency, improves team collaboration.

6. **Accessibility First**: Reaches 20% more users, reduces legal risk, improves UX for everyone, better SEO.

7. **Documentation**: Reduces onboarding time by 50%, enables auto-generated style guides, improves team velocity.

8. **Agent Verification**: Ensures all generated CSS compiles and works, eliminates syntax errors, maintains code quality.

---

## 16. Quick Reference

### Common Commands

```bash
# Compile SCSS
npx sass src/styles.scss dist/styles.css
npx sass --watch src:dist

# Lint CSS/SCSS
npx stylelint "**/*.{css,scss}"
npx stylelint "**/*.scss" --fix

# PostCSS with Autoprefixer
npx postcss src/styles.css --use autoprefixer -o dist/styles.css

# Generate documentation
npx sassdoc src/

# Check for unused CSS
npx purgecss --css dist/styles.css --content "**/*.html" --output dist/

# Visual regression tests
npx playwright test --project=visual

# Minify CSS
npx csso dist/styles.css -o dist/styles.min.css
```

### BEM Naming Quick Guide

```scss
// Block
.card { }

// Element (double underscore)
.card__header { }
.card__body { }
.card__footer { }

// Modifier (double dash)
.card--featured { }
.card--compact { }
.card__header--large { }
```

### SCSS Variables Cheat Sheet

```scss
// Colors
$color-primary: #3b82f6;
$color-secondary: #64748b;
$color-success: #22c55e;
$color-error: #ef4444;

// Spacing
$spacing-xs: 0.25rem;  // 4px
$spacing-sm: 0.5rem;   // 8px
$spacing-md: 1rem;     // 16px
$spacing-lg: 1.5rem;   // 24px
$spacing-xl: 2rem;     // 32px

// Breakpoints
$breakpoint-sm: 640px;
$breakpoint-md: 768px;
$breakpoint-lg: 1024px;
$breakpoint-xl: 1280px;

// Typography
$font-size-base: 1rem;
$line-height-base: 1.5;
```

### Common Mixins

```scss
// Flexbox center
@mixin flex-center {
  display: flex;
  align-items: center;
  justify-content: center;
}

// Responsive breakpoint
@mixin respond-to($breakpoint) {
  @media (min-width: $breakpoint) {
    @content;
  }
}

// Visually hidden (accessible)
@mixin sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
```

### Accessibility Checklist

```
[ ] Focus styles on all interactive elements
[ ] Color contrast ≥ 4.5:1 for text
[ ] prefers-reduced-motion supported
[ ] No content in ::before/::after
[ ] Font size uses rem/em (not px)
[ ] Touch targets ≥ 44x44px
```

---

## References

- [Sass Documentation](https://sass-lang.com/documentation)
- [CSS Tricks](https://css-tricks.com/)
- [BEM Methodology](https://en.bem.info/methodology/)
- [MDN CSS Documentation](https://developer.mozilla.org/en-US/docs/Web/CSS)
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [SassDoc](http://sassdoc.com/)
- [Stylelint](https://stylelint.io/)
- [PostCSS](https://postcss.org/)

---

**Last Updated:** 2026-01-17
**Version:** 1.0
**Maintainer:** Development Team
