# HTML Development Guidelines
Mandatory coding standards and development practices for modern HTML applications. HTML5, W3C Validator, Lighthouse, axe DevTools, Semantic markup, ARIA.

---
Agent Profile: The Modern HTML Expert
Role: Senior Front-End Developer & Accessibility Specialist
Objective: Generate production-ready, semantic, accessible, performant, and maintainable HTML code.
Tools: HTML5, W3C Validator, Lighthouse, axe DevTools, Semantic markup, ARIA.

## 1. Core Philosophies

The agent must adhere to the "SEMANTIC-FIRST" principles for every HTML project:

**Semantic Markup**: Use semantic HTML5 elements, meaningful structure, proper nesting.
**Explicit Accessibility**: WCAG 2.1 AA compliance, ARIA labels, keyboard navigation.
**Minimal & Clean**: No unnecessary divs, clean structure, readable formatting.
**Accessible Forms**: Proper labels, fieldsets, error messages, focus management.
**Named Landmarks**: Clear page structure with main, nav, aside, footer.
**Text Alternatives**: Alt text for images, captions for media, descriptive links.
**Internationalization Ready**: lang attributes, dir support, character encoding.
**Content First**: Progressive enhancement, semantic structure without CSS/JS.

**Fast Loading**: Optimized assets, lazy loading, critical CSS inline.
**Independent of Styling**: Structure works without CSS, no presentation in HTML.
**Responsive by Default**: Mobile-first, viewport meta tag, fluid layouts.
**SEO Optimized**: Meta tags, structured data, semantic headings, meaningful content.
**Tested & Validated**: W3C validation, accessibility testing, cross-browser compatibility.

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated HTML is valid, accessible, and semantic before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY HTML code, the agent MUST:**

1. **HTML Validation**:
   ```bash
   # Validate HTML with W3C validator
   npx html-validate *.html
   # OR use online validator
   # https://validator.w3.org/
   # Exit code MUST be 0, no errors
   ```

2. **Accessibility Check**:
   ```bash
   # Run accessibility audit
   npx pa11y-ci *.html
   # OR lighthouse accessibility audit
   npx lighthouse --only-categories=accessibility index.html
   # Score MUST be ≥ 90
   ```

3. **Semantic Structure Verification**:
   - [ ] Uses semantic HTML5 elements (header, nav, main, article, section, aside, footer)
   - [ ] No unnecessary divs/spans
   - [ ] Proper heading hierarchy (h1 → h2 → h3, no skipping levels)
   - [ ] Meaningful link text (no "click here")
   - [ ] All images have alt text
   - [ ] Forms have proper labels and fieldsets

4. **Performance Check**:
   ```bash
   # Run performance audit
   npx lighthouse --only-categories=performance index.html
   # Score MUST be ≥ 85
   ```

5. **SEO Check**:
   ```bash
   # Run SEO audit
   npx lighthouse --only-categories=seo index.html
   # Score MUST be ≥ 90
   ```

### B. Error Correction Process

If verification fails:

1. **Read the error/warning message** carefully
2. **Identify the root cause** (invalid HTML, accessibility issue, semantic error, etc.)
3. **Fix the issue** in the generated HTML
4. **Re-run verification** until all checks pass
5. **Document any non-obvious decisions** in HTML comments

### C. Agent Workflow Example

**Complete workflow for generating an HTML page:**

1. **Generate semantic HTML structure**:
   ```html
   <!DOCTYPE html>
   <html lang="en">
   <head>
     <meta charset="UTF-8">
     <meta name="viewport" content="width=device-width, initial-scale=1.0">
     <meta name="description" content="User dashboard for managing account">
     <title>User Dashboard - MyApp</title>
   </head>
   <body>
     <header>
       <nav aria-label="Main navigation">
         <ul>
           <li><a href="/">Home</a></li>
           <li><a href="/about">About</a></li>
         </ul>
       </nav>
     </header>
     
     <main>
       <h1>Welcome to Your Dashboard</h1>
       <article>
         <h2>Recent Activity</h2>
         <p>Your recent account activity.</p>
       </article>
     </main>
     
     <footer>
       <p>&copy; 2026 MyApp. All rights reserved.</p>
     </footer>
   </body>
   </html>
   ```

2. **Verify HTML validity**:
   ```bash
   npx html-validate index.html
   # ✓ No errors
   ```

3. **Check accessibility**:
   ```bash
   npx pa11y index.html
   # ✓ No issues
   ```

4. **Verify semantic structure** (manual check):
   - ✓ Proper heading hierarchy
   - ✓ Semantic elements used
   - ✓ ARIA labels present
   - ✓ All images have alt text

5. **Run Lighthouse audit**:
   ```bash
   npx lighthouse index.html
   # ✓ Accessibility: 100
   # ✓ SEO: 100
   # ✓ Performance: 95
   ```

6. **Present code** to user - only after ALL checks pass

### D. Prohibited Practices

**NEVER deliver HTML that:**
- ❌ Has W3C validation errors
- ❌ Has accessibility violations (WCAG 2.1 AA)
- ❌ Uses non-semantic markup (div soup)
- ❌ Has missing alt text on images
- ❌ Has forms without labels
- ❌ Skips heading levels (h1 → h3)
- ❌ Uses tables for layout
- ❌ Has inline styles or presentation attributes
- ❌ Missing DOCTYPE or meta charset
- ❌ Has non-descriptive link text ("click here", "read more")
- ❌ Missing lang attribute on html element
- ❌ Uses deprecated HTML elements (font, center, marquee, etc.)

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new HTML development.**

### TDD Cycle for HTML

```
1. 🔴 RED: Write a failing test/validation first
   ↓
2. 🟢 GREEN: Write minimal HTML to make it pass
   ↓
3. 🔵 REFACTOR: Improve structure while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for HTML

```javascript
// Step 1: RED - Write failing test first (tests/html.test.js)
import { test, expect } from 'vitest';
import { JSDOM } from 'jsdom';
import fs from 'fs';

test('page has proper semantic structure', () => {
  const html = fs.readFileSync('src/index.html', 'utf-8');
  const dom = new JSDOM(html);
  const doc = dom.window.document;

  expect(doc.querySelector('header')).toBeTruthy();
  expect(doc.querySelector('main')).toBeTruthy();
  expect(doc.querySelector('footer')).toBeTruthy();
  expect(doc.querySelector('nav')).toBeTruthy();
});

test('all images have alt attributes', () => {
  const html = fs.readFileSync('src/index.html', 'utf-8');
  const dom = new JSDOM(html);
  const images = dom.window.document.querySelectorAll('img');

  images.forEach(img => {
    expect(img.hasAttribute('alt')).toBe(true);
  });
});

// Run: npm test
// ❌ FAILS - HTML structure doesn't exist yet

// Step 2: GREEN - Write minimal HTML
// src/index.html
/*
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Page Title</title>
</head>
<body>
  <header><nav>...</nav></header>
  <main>...</main>
  <footer>...</footer>
</body>
</html>
*/

// Run: npm test
// ✅ PASSES - semantic structure present
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every HTML bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. 🐛 Bug Reported/Discovered (e.g., accessibility issue)
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
// Bug Report #234: Form inputs missing labels

// Step 1-2: Write test that reproduces the bug
test('all form inputs have associated labels - Bug #234', () => {
  // Bug: Screen readers couldn't identify form fields
  // Discovered: 2026-01-18
  // This test prevents regression

  const html = fs.readFileSync('src/contact.html', 'utf-8');
  const dom = new JSDOM(html);
  const doc = dom.window.document;
  const inputs = doc.querySelectorAll('input:not([type="hidden"]), textarea, select');

  inputs.forEach(input => {
    const id = input.getAttribute('id');
    const label = doc.querySelector(`label[for="${id}"]`);
    expect(label).toBeTruthy();
  });
});

// Run: npm test
// ❌ FAILS - inputs missing labels

// Step 3: Fix the HTML
// Before (buggy):
// <input type="email" name="email">

// After (fixed):
// <label for="email">Email Address</label>
// <input type="email" id="email" name="email">

// Run: npm test
// ✅ PASSES - bug fixed, regression prevented
```

---

## 3. HTML5 Document Structure (MANDATORY)

### A. Complete HTML5 Template

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <!-- Character Encoding (MUST be first) -->
  <meta charset="UTF-8">
  
  <!-- Viewport for Responsive Design (REQUIRED) -->
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  
  <!-- IE Compatibility -->
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  
  <!-- Primary Meta Tags (REQUIRED) -->
  <title>Page Title - Site Name</title>
  <meta name="title" content="Page Title - Site Name">
  <meta name="description" content="Concise description (150-160 characters)">
  <meta name="keywords" content="keyword1, keyword2, keyword3">
  <meta name="author" content="Author Name">
  
  <!-- Open Graph / Facebook -->
  <meta property="og:type" content="website">
  <meta property="og:url" content="https://example.com/">
  <meta property="og:title" content="Page Title">
  <meta property="og:description" content="Page description">
  <meta property="og:image" content="https://example.com/image.jpg">
  
  <!-- Twitter -->
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:url" content="https://example.com/">
  <meta name="twitter:title" content="Page Title">
  <meta name="twitter:description" content="Page description">
  <meta name="twitter:image" content="https://example.com/image.jpg">
  
  <!-- Favicon -->
  <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">
  <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">
  <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">
  <link rel="manifest" href="/site.webmanifest">
  
  <!-- Stylesheets -->
  <link rel="stylesheet" href="/css/styles.css">
  
  <!-- Preload Critical Resources -->
  <link rel="preload" href="/fonts/main-font.woff2" as="font" type="font/woff2" crossorigin>
  
  <!-- Theme Color -->
  <meta name="theme-color" content="#ffffff">
</head>
<body>
  <!-- Skip to main content link (ACCESSIBILITY REQUIRED) -->
  <a href="#main-content" class="skip-link">Skip to main content</a>
  
  <!-- Header -->
  <header>
    <nav aria-label="Main navigation">
      <!-- Navigation content -->
    </nav>
  </header>
  
  <!-- Main Content (REQUIRED) -->
  <main id="main-content">
    <!-- Page content -->
  </main>
  
  <!-- Footer -->
  <footer>
    <!-- Footer content -->
  </footer>
  
  <!-- Scripts (at end of body for performance) -->
  <script src="/js/main.js" defer></script>
</body>
</html>
```

### B. Head Section Requirements

**MANDATORY meta tags:**
```html
<head>
  <!-- These 3 MUST be first in this order -->
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  
  <!-- REQUIRED -->
  <title>Descriptive Page Title (50-60 characters)</title>
  <meta name="description" content="Clear description (150-160 characters)">
</head>
```

**Page Title Best Practices:**
```html
<!-- ✅ CORRECT - Descriptive and unique -->
<title>User Profile Settings - MyApp Dashboard</title>
<title>Product Name - Category - Store Name</title>
<title>Article Title | Blog Name</title>

<!-- ❌ WRONG - Too generic or vague -->
<title>Home</title>
<title>Page</title>
<title>Welcome</title>
```

---

## 4. Semantic HTML5 Elements (MANDATORY)

### A. Structural Elements

**ALWAYS use semantic elements instead of generic divs:**

```html
<!-- ✅ CORRECT - Semantic structure -->
<header>
  <nav aria-label="Main navigation">
    <ul>
      <li><a href="/">Home</a></li>
      <li><a href="/about">About</a></li>
      <li><a href="/contact">Contact</a></li>
    </ul>
  </nav>
</header>

<main>
  <article>
    <header>
      <h1>Article Title</h1>
      <p>Published on <time datetime="2026-01-17">January 17, 2026</time></p>
    </header>
    
    <section>
      <h2>Section Heading</h2>
      <p>Content paragraph.</p>
    </section>
    
    <aside>
      <h3>Related Information</h3>
      <p>Sidebar content.</p>
    </aside>
    
    <footer>
      <p>Author: <a href="/authors/john">John Doe</a></p>
    </footer>
  </article>
</main>

<aside aria-label="Sidebar">
  <section>
    <h2>Popular Posts</h2>
    <ul>
      <li><a href="/post-1">Post Title 1</a></li>
      <li><a href="/post-2">Post Title 2</a></li>
    </ul>
  </section>
</aside>

<footer>
  <nav aria-label="Footer navigation">
    <ul>
      <li><a href="/privacy">Privacy Policy</a></li>
      <li><a href="/terms">Terms of Service</a></li>
    </ul>
  </nav>
  <p>&copy; 2026 Company Name. All rights reserved.</p>
</footer>


<!-- ❌ WRONG - Div soup -->
<div class="header">
  <div class="nav">
    <div class="nav-list">
      <div class="nav-item"><a href="/">Home</a></div>
      <div class="nav-item"><a href="/about">About</a></div>
    </div>
  </div>
</div>

<div class="content">
  <div class="post">
    <div class="post-title">Article Title</div>
    <div class="post-content">Content here</div>
  </div>
</div>
```

### B. Semantic Element Guide

| Element | Purpose | Example |
|---------|---------|---------|
| `<header>` | Page or section header | Site header, article header |
| `<nav>` | Navigation links | Main menu, breadcrumbs |
| `<main>` | Main content (one per page) | Primary page content |
| `<article>` | Self-contained content | Blog post, news article |
| `<section>` | Thematic grouping | Chapter, tab panel |
| `<aside>` | Tangentially related content | Sidebar, callout box |
| `<footer>` | Page or section footer | Site footer, article footer |
| `<figure>` | Self-contained media | Image with caption |
| `<figcaption>` | Caption for figure | Image description |
| `<time>` | Date/time | `<time datetime="2026-01-17">` |
| `<mark>` | Highlighted text | Search results highlight |
| `<details>` | Expandable content | Accordion, FAQ |
| `<summary>` | Summary for details | Accordion title |

### C. Heading Hierarchy (CRITICAL)

**MUST follow proper heading order:**

```html
<!-- ✅ CORRECT - Proper hierarchy -->
<main>
  <h1>Main Page Title</h1>
  
  <section>
    <h2>Section Title</h2>
    <p>Content</p>
    
    <h3>Subsection Title</h3>
    <p>More content</p>
    
    <h4>Sub-subsection</h4>
    <p>Detailed content</p>
  </section>
  
  <section>
    <h2>Another Section</h2>
    <p>Content</p>
  </section>
</main>


<!-- ❌ WRONG - Skipped levels -->
<main>
  <h1>Main Title</h1>
  <h3>Skipped h2!</h3>  <!-- BAD: Jumped from h1 to h3 -->
  <h2>Wrong order</h2>  <!-- BAD: h2 after h3 -->
</main>


<!-- ❌ WRONG - Multiple h1 elements -->
<main>
  <h1>First Title</h1>
  <h1>Second Title</h1>  <!-- BAD: Only one h1 per page -->
</main>
```

**Heading Checklist:**
- [ ] Only ONE `<h1>` per page
- [ ] Headings follow sequential order (h1 → h2 → h3, never skip)
- [ ] Headings describe content accurately
- [ ] Every section has a heading
- [ ] Headings are not used just for styling (use CSS)

---

## 5. Accessibility Requirements (WCAG 2.1 AA Mandatory)

### A. ARIA Labels and Landmarks

```html
<!-- ✅ CORRECT - Proper ARIA labels -->
<nav aria-label="Main navigation">
  <ul>
    <li><a href="/">Home</a></li>
    <li><a href="/products">Products</a></li>
  </ul>
</nav>

<nav aria-label="Footer navigation">
  <ul>
    <li><a href="/privacy">Privacy</a></li>
    <li><a href="/terms">Terms</a></li>
  </ul>
</nav>

<button aria-label="Close dialog" aria-describedby="close-description">
  <svg>...</svg>
</button>
<span id="close-description" class="sr-only">Closes the modal and returns to main page</span>

<form role="search" aria-label="Search site">
  <input type="search" aria-label="Search query" placeholder="Search...">
  <button type="submit">Search</button>
</form>
```

### B. Alternative Text for Images

```html
<!-- ✅ CORRECT - Descriptive alt text -->
<img src="chart.png" alt="Bar chart showing sales increase of 45% in Q4 2026">
<img src="profile.jpg" alt="Jane Doe, CEO of TechCorp">
<img src="logo.svg" alt="Company Name">

<!-- ✅ CORRECT - Decorative images (empty alt) -->
<img src="decorative-line.svg" alt="" role="presentation">
<img src="background-pattern.png" alt="" role="presentation">

<!-- ✅ CORRECT - Complex images with longer description -->
<figure>
  <img src="complex-diagram.png" alt="Network architecture diagram" aria-describedby="diagram-desc">
  <figcaption id="diagram-desc">
    Network diagram showing three-tier architecture with load balancer,
    application servers, and database cluster with master-slave replication.
  </figcaption>
</figure>

<!-- ❌ WRONG - Missing or poor alt text -->
<img src="photo.jpg">  <!-- Missing alt -->
<img src="chart.png" alt="image">  <!-- Not descriptive -->
<img src="button.png" alt="click here">  <!-- Not helpful -->
```

### C. Form Accessibility

```html
<!-- ✅ CORRECT - Accessible form -->
<form method="post" action="/submit">
  <fieldset>
    <legend>Personal Information</legend>
    
    <div class="form-group">
      <label for="fullname">Full Name <span aria-label="required">*</span></label>
      <input 
        type="text" 
        id="fullname" 
        name="fullname" 
        required 
        aria-required="true"
        aria-describedby="fullname-hint"
      >
      <small id="fullname-hint">Enter your first and last name</small>
    </div>
    
    <div class="form-group">
      <label for="email">Email Address <span aria-label="required">*</span></label>
      <input 
        type="email" 
        id="email" 
        name="email" 
        required 
        aria-required="true"
        aria-invalid="false"
        aria-describedby="email-error"
      >
      <span id="email-error" role="alert" aria-live="polite"></span>
    </div>
    
    <fieldset>
      <legend>Notification Preferences</legend>
      <div class="form-group">
        <input type="checkbox" id="email-notify" name="notifications" value="email">
        <label for="email-notify">Email notifications</label>
      </div>
      <div class="form-group">
        <input type="checkbox" id="sms-notify" name="notifications" value="sms">
        <label for="sms-notify">SMS notifications</label>
      </div>
    </fieldset>
    
    <div class="form-group">
      <label for="country">Country</label>
      <select id="country" name="country" required aria-required="true">
        <option value="">Select a country</option>
        <option value="us">United States</option>
        <option value="uk">United Kingdom</option>
        <option value="ca">Canada</option>
      </select>
    </div>
    
    <button type="submit">Submit Form</button>
    <button type="reset">Reset Form</button>
  </fieldset>
</form>


<!-- ❌ WRONG - Inaccessible form -->
<form>
  Full Name: <input type="text" name="name">  <!-- No label -->
  <input type="email" placeholder="Email">  <!-- Placeholder is not a label -->
  <input type="checkbox" value="yes"> Subscribe  <!-- Label not associated -->
  <div onclick="submitForm()">Submit</div>  <!-- Not a button -->
</form>
```

### D. Keyboard Navigation

```html
<!-- ✅ CORRECT - Keyboard accessible -->
<button type="button" onclick="openModal()">Open Modal</button>

<a href="/download.pdf" download>Download PDF</a>

<div role="button" tabindex="0" onclick="doAction()" onkeydown="handleKey(event)">
  Custom Button
</div>

<!-- Modal with focus trap -->
<div role="dialog" aria-labelledby="modal-title" aria-modal="true">
  <h2 id="modal-title">Modal Title</h2>
  <button aria-label="Close" onclick="closeModal()">×</button>
  <div>Modal content</div>
</div>


<!-- ❌ WRONG - Not keyboard accessible -->
<div onclick="doAction()">Click Me</div>  <!-- No keyboard support -->
<span onclick="submit()">Submit</span>  <!-- Should be button -->
<a href="#" onclick="action()">Action</a>  <!-- Fake link -->
```

### E. Skip Links

```html
<!-- ✅ CORRECT - Skip link at beginning of body (REQUIRED) -->
<body>
  <a href="#main-content" class="skip-link">Skip to main content</a>
  
  <header>
    <!-- Header content -->
  </header>
  
  <main id="main-content" tabindex="-1">
    <h1>Main Content</h1>
    <!-- Content -->
  </main>
</body>

<style>
  /* Skip link visible on focus */
  .skip-link {
    position: absolute;
    top: -40px;
    left: 0;
    background: #000;
    color: #fff;
    padding: 8px;
    text-decoration: none;
    z-index: 100;
  }
  
  .skip-link:focus {
    top: 0;
  }
</style>
```

---

## 6. Performance Optimization (MANDATORY)

### A. Image Optimization

```html
<!-- ✅ CORRECT - Responsive images with lazy loading -->
<img 
  src="image-800.jpg"
  srcset="
    image-400.jpg 400w,
    image-800.jpg 800w,
    image-1200.jpg 1200w
  "
  sizes="(max-width: 600px) 400px, (max-width: 900px) 800px, 1200px"
  alt="Description of image"
  loading="lazy"
  width="800"
  height="600"
>

<!-- ✅ CORRECT - Modern image formats with fallback -->
<picture>
  <source srcset="image.avif" type="image/avif">
  <source srcset="image.webp" type="image/webp">
  <img src="image.jpg" alt="Fallback image" loading="lazy">
</picture>

<!-- ✅ CORRECT - Lazy loading for off-screen images -->
<img src="hero.jpg" alt="Hero image" loading="eager">  <!-- Above fold -->
<img src="content.jpg" alt="Content image" loading="lazy">  <!-- Below fold -->


<!-- ❌ WRONG - No optimization -->
<img src="huge-image-5mb.jpg" alt="Image">  <!-- Too large -->
<img src="image.jpg">  <!-- Missing width/height causes layout shift -->
```

### B. Resource Loading

```html
<head>
  <!-- ✅ CORRECT - Preload critical resources -->
  <link rel="preload" href="/fonts/main.woff2" as="font" type="font/woff2" crossorigin>
  <link rel="preload" href="/css/critical.css" as="style">
  <link rel="preload" href="/images/hero.jpg" as="image">
  
  <!-- ✅ CORRECT - Prefetch next-page resources -->
  <link rel="prefetch" href="/next-page.html">
  <link rel="prefetch" href="/js/next-page.js">
  
  <!-- ✅ CORRECT - DNS prefetch for external resources -->
  <link rel="dns-prefetch" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.googleapis.com" crossorigin>
  
  <!-- ✅ CORRECT - Critical CSS inline, rest async -->
  <style>
    /* Critical above-the-fold CSS */
  </style>
  <link rel="preload" href="/css/styles.css" as="style" onload="this.onload=null;this.rel='stylesheet'">
  <noscript><link rel="stylesheet" href="/css/styles.css"></noscript>
</head>

<body>
  <!-- ✅ CORRECT - Scripts with defer or async -->
  <script src="/js/critical.js"></script>  <!-- Blocks parsing, use sparingly -->
  <script src="/js/analytics.js" async></script>  <!-- Doesn't block -->
  <script src="/js/main.js" defer></script>  <!-- Waits for parsing -->
  
  <!-- ❌ WRONG - Blocking scripts in head -->
  <!-- <script src="large-library.js"></script> in head -->
</body>
```

### C. Lazy Loading

```html
<!-- ✅ CORRECT - Native lazy loading -->
<img src="image.jpg" alt="Image" loading="lazy" width="800" height="600">

<iframe 
  src="https://www.youtube.com/embed/VIDEO_ID" 
  loading="lazy"
  width="560" 
  height="315"
  title="Video title"
></iframe>

<!-- ✅ CORRECT - Lazy load below-the-fold content -->
<article>
  <img src="hero.jpg" alt="Hero" loading="eager">  <!-- Load immediately -->
  <h1>Article Title</h1>
  <p>Introduction paragraph...</p>
  <img src="content-1.jpg" alt="Image 1" loading="lazy">  <!-- Lazy load -->
  <img src="content-2.jpg" alt="Image 2" loading="lazy">
</article>
```

---

## 7. SEO Best Practices (MANDATORY)

### A. Structured Data (Schema.org)

```html
<!-- ✅ CORRECT - JSON-LD structured data -->
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Article",
  "headline": "Article Headline",
  "image": "https://example.com/image.jpg",
  "author": {
    "@type": "Person",
    "name": "John Doe"
  },
  "publisher": {
    "@type": "Organization",
    "name": "Publisher Name",
    "logo": {
      "@type": "ImageObject",
      "url": "https://example.com/logo.jpg"
    }
  },
  "datePublished": "2026-01-17",
  "dateModified": "2026-01-17"
}
</script>

<!-- Product structured data -->
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Product",
  "name": "Product Name",
  "image": "https://example.com/product.jpg",
  "description": "Product description",
  "sku": "12345",
  "offers": {
    "@type": "Offer",
    "url": "https://example.com/product",
    "priceCurrency": "USD",
    "price": "29.99",
    "availability": "https://schema.org/InStock"
  }
}
</script>
```

### B. Meta Tags for SEO

```html
<head>
  <!-- ✅ CORRECT - Complete SEO meta tags -->
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Page Title - Site Name (50-60 chars)</title>
  <meta name="description" content="Clear, compelling description (150-160 chars)">
  <meta name="keywords" content="primary keyword, secondary keyword, tertiary keyword">
  <meta name="author" content="Author Name">
  <meta name="robots" content="index, follow">
  <link rel="canonical" href="https://example.com/page">
  
  <!-- Open Graph -->
  <meta property="og:type" content="website">
  <meta property="og:url" content="https://example.com/page">
  <meta property="og:title" content="Page Title">
  <meta property="og:description" content="Page description">
  <meta property="og:image" content="https://example.com/image.jpg">
  <meta property="og:site_name" content="Site Name">
  <meta property="og:locale" content="en_US">
  
  <!-- Twitter Card -->
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:site" content="@username">
  <meta name="twitter:creator" content="@username">
  <meta name="twitter:title" content="Page Title">
  <meta name="twitter:description" content="Page description">
  <meta name="twitter:image" content="https://example.com/image.jpg">
</head>
```

### C. Semantic Links

```html
<!-- ✅ CORRECT - Descriptive link text -->
<a href="/guide">Read our comprehensive getting started guide</a>
<a href="/download.pdf">Download the 2026 Annual Report (PDF, 2MB)</a>
<a href="/contact">Contact our support team</a>

<!-- External links with security -->
<a href="https://external-site.com" target="_blank" rel="noopener noreferrer">
  Visit External Site (opens in new tab)
</a>


<!-- ❌ WRONG - Non-descriptive link text -->
<a href="/guide">Click here</a>
<a href="/more">Read more</a>
<a href="/download.pdf">Download</a>
<a href="https://external-site.com" target="_blank">Link</a>  <!-- Missing rel -->
```

---

## 8. Clean Code Standards (MANDATORY)

### A. Formatting and Indentation

```html
<!-- ✅ CORRECT - Clean, readable formatting -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Page Title</title>
</head>
<body>
  <header>
    <nav aria-label="Main navigation">
      <ul>
        <li><a href="/">Home</a></li>
        <li><a href="/about">About</a></li>
      </ul>
    </nav>
  </header>
  
  <main>
    <article>
      <h1>Article Title</h1>
      <p>Paragraph text.</p>
    </article>
  </main>
  
  <footer>
    <p>&copy; 2026 Company Name</p>
  </footer>
</body>
</html>


<!-- ❌ WRONG - Poor formatting -->
<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Title</title></head>
<body><div class="header"><div class="nav"><a href="/">Home</a><a href="/about">
About</a></div></div><div class="content"><h1>Title</h1><p>Text</p></div></body>
</html>
```

### B. Comments

```html
<!-- ✅ CORRECT - Helpful comments -->
<!-- Main Navigation -->
<nav aria-label="Main navigation">
  <!-- Navigation items -->
</nav>

<!-- User Profile Section -->
<section aria-labelledby="profile-heading">
  <h2 id="profile-heading">User Profile</h2>
  <!-- Profile content -->
</section>

<!-- TODO: Add pagination controls -->
<!-- NOTE: This component requires JavaScript to function -->


<!-- ❌ WRONG - Unnecessary or outdated comments -->
<!-- div -->
<div>Content</div>
<!-- end div -->

<!-- This is a paragraph -->
<p>Text</p>
```

### C. Attribute Order

```html
<!-- ✅ CORRECT - Consistent attribute order -->
<!-- 1. Class/ID, 2. type/href/src, 3. ARIA, 4. data attributes, 5. other -->
<a 
  class="button primary" 
  href="/submit" 
  role="button"
  aria-label="Submit form"
  data-tracking="submit-button"
  target="_blank"
  rel="noopener noreferrer"
>
  Submit
</a>

<input 
  class="form-input" 
  type="email" 
  id="email" 
  name="email"
  required
  aria-required="true"
  aria-invalid="false"
  placeholder="email@example.com"
  autocomplete="email"
>
```

### D. Minimize Nesting

```html
<!-- ✅ CORRECT - Minimal nesting with semantic elements -->
<article>
  <header>
    <h1>Article Title</h1>
    <p>Published on <time datetime="2026-01-17">January 17, 2026</time></p>
  </header>
  
  <p>Article content paragraph.</p>
  
  <footer>
    <p>Author: John Doe</p>
  </footer>
</article>


<!-- ❌ WRONG - Excessive nesting -->
<div class="article-wrapper">
  <div class="article-container">
    <div class="article-inner">
      <div class="article-header">
        <div class="article-title">
          <h1>Article Title</h1>
        </div>
      </div>
    </div>
  </div>
</div>
```

---

## 9. Project Structure

```
project/
├── index.html              # Homepage
├── about.html              # About page
├── contact.html            # Contact page
├── css/
│   ├── styles.css          # Main stylesheet
│   └── critical.css        # Critical CSS
├── js/
│   ├── main.js             # Main JavaScript
│   └── analytics.js        # Analytics
├── images/
│   ├── logo.svg
│   ├── hero.jpg
│   └── favicon/
│       ├── favicon-32x32.png
│       ├── favicon-16x16.png
│       └── apple-touch-icon.png
├── fonts/
│   ├── main-font.woff2
│   └── main-font.woff
├── site.webmanifest        # PWA manifest
├── robots.txt              # Robots file
└── sitemap.xml             # XML sitemap
```

---

## 10. Deployment Checklist

### Pre-Production Validation

#### HTML Validation (MANDATORY)
- [ ] **W3C validation passes**: No errors at https://validator.w3.org/
- [ ] **Proper DOCTYPE**: `<!DOCTYPE html>` present
- [ ] **lang attribute**: `<html lang="en">` specified
- [ ] **Character encoding**: `<meta charset="UTF-8">` present and first in head
- [ ] **Viewport meta tag**: Mobile-responsive meta tag present
- [ ] **Valid nesting**: All elements properly nested
- [ ] **Closed tags**: All tags properly closed
- [ ] **No deprecated elements**: No font, center, marquee, etc.

#### Accessibility (MANDATORY - WCAG 2.1 AA)
- [ ] **Accessibility score ≥ 90**: Lighthouse accessibility audit passes
- [ ] **All images have alt text**: Including decorative images (empty alt)
- [ ] **Proper heading hierarchy**: h1 → h2 → h3, no skipping
- [ ] **Form labels**: All inputs have associated labels
- [ ] **ARIA labels**: Navigation, regions properly labeled
- [ ] **Keyboard navigation**: All interactive elements keyboard accessible
- [ ] **Skip links**: Skip to main content link present
- [ ] **Color contrast**: Text meets WCAG AA contrast ratios (4.5:1)
- [ ] **Focus indicators**: Visible focus states on all interactive elements

#### Semantic Structure
- [ ] **Semantic elements**: Using header, nav, main, article, section, aside, footer
- [ ] **One main element**: Only one `<main>` per page
- [ ] **One h1**: Only one `<h1>` per page
- [ ] **Meaningful landmarks**: Proper use of ARIA landmarks
- [ ] **No div soup**: Minimal non-semantic divs/spans

#### Performance
- [ ] **Performance score ≥ 85**: Lighthouse performance audit
- [ ] **Images optimized**: Compressed, proper formats, responsive images
- [ ] **Lazy loading**: Images below fold use loading="lazy"
- [ ] **Resource hints**: Preload critical resources
- [ ] **Scripts optimized**: defer/async attributes used
- [ ] **No render-blocking resources**: Critical CSS inline or preloaded

#### SEO
- [ ] **SEO score ≥ 90**: Lighthouse SEO audit
- [ ] **Unique page titles**: Each page has descriptive title
- [ ] **Meta descriptions**: Each page has unique description
- [ ] **Canonical URLs**: Canonical links specified
- [ ] **Structured data**: Schema.org markup present where applicable
- [ ] **Descriptive links**: No "click here" or "read more" links
- [ ] **Open Graph tags**: Social media meta tags present
- [ ] **Robots.txt**: Present and properly configured
- [ ] **Sitemap.xml**: XML sitemap available

#### Security
- [ ] **External links secure**: rel="noopener noreferrer" on target="_blank" links
- [ ] **HTTPS URLs**: All resources loaded over HTTPS
- [ ] **No inline JavaScript**: No onclick attributes (use event listeners)
- [ ] **Content Security Policy**: CSP headers configured (if applicable)

#### Cross-Browser Compatibility
- [ ] **Chrome**: Tested and working
- [ ] **Firefox**: Tested and working
- [ ] **Safari**: Tested and working
- [ ] **Edge**: Tested and working
- [ ] **Mobile browsers**: Tested on iOS Safari and Chrome Android

---

## 11. Complete Example

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <!-- Essential Meta Tags -->
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  
  <!-- Primary Meta Tags -->
  <title>Modern HTML5 Blog - Latest Web Development Articles</title>
  <meta name="title" content="Modern HTML5 Blog - Latest Web Development Articles">
  <meta name="description" content="Explore the latest articles on modern web development, HTML5, CSS3, and JavaScript best practices.">
  <meta name="keywords" content="web development, HTML5, CSS3, JavaScript, tutorials">
  <meta name="author" content="John Doe">
  
  <!-- Open Graph / Facebook -->
  <meta property="og:type" content="website">
  <meta property="og:url" content="https://example.com/">
  <meta property="og:title" content="Modern HTML5 Blog">
  <meta property="og:description" content="Latest web development articles and tutorials">
  <meta property="og:image" content="https://example.com/images/og-image.jpg">
  
  <!-- Twitter -->
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="Modern HTML5 Blog">
  <meta name="twitter:description" content="Latest web development articles">
  <meta name="twitter:image" content="https://example.com/images/twitter-image.jpg">
  
  <!-- Favicon -->
  <link rel="icon" type="image/png" sizes="32x32" href="/images/favicon/favicon-32x32.png">
  <link rel="icon" type="image/png" sizes="16x16" href="/images/favicon/favicon-16x16.png">
  <link rel="apple-touch-icon" sizes="180x180" href="/images/favicon/apple-touch-icon.png">
  
  <!-- Canonical URL -->
  <link rel="canonical" href="https://example.com/">
  
  <!-- Preload Critical Resources -->
  <link rel="preload" href="/fonts/main-font.woff2" as="font" type="font/woff2" crossorigin>
  
  <!-- Stylesheets -->
  <link rel="stylesheet" href="/css/styles.css">
  
  <!-- Theme Color -->
  <meta name="theme-color" content="#4A90E2">
  
  <!-- Structured Data -->
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "Blog",
    "name": "Modern HTML5 Blog",
    "description": "Latest web development articles",
    "url": "https://example.com"
  }
  </script>
</head>
<body>
  <!-- Skip to Main Content (Accessibility) -->
  <a href="#main-content" class="skip-link">Skip to main content</a>
  
  <!-- Site Header -->
  <header>
    <div class="logo">
      <img src="/images/logo.svg" alt="Modern HTML5 Blog" width="200" height="50">
    </div>
    
    <!-- Main Navigation -->
    <nav aria-label="Main navigation">
      <ul>
        <li><a href="/" aria-current="page">Home</a></li>
        <li><a href="/articles">Articles</a></li>
        <li><a href="/tutorials">Tutorials</a></li>
        <li><a href="/about">About</a></li>
        <li><a href="/contact">Contact</a></li>
      </ul>
    </nav>
  </header>
  
  <!-- Main Content -->
  <main id="main-content" tabindex="-1">
    <!-- Hero Section -->
    <section aria-labelledby="hero-heading">
      <h1 id="hero-heading">Latest Web Development Articles</h1>
      <p>Stay up to date with modern web development practices and techniques.</p>
    </section>
    
    <!-- Featured Article -->
    <article>
      <header>
        <h2><a href="/articles/semantic-html5">The Power of Semantic HTML5</a></h2>
        <p>
          Published on <time datetime="2026-01-17">January 17, 2026</time> by 
          <a href="/authors/john-doe">John Doe</a>
        </p>
      </header>
      
      <figure>
        <img 
          src="/images/semantic-html-800.jpg"
          srcset="
            /images/semantic-html-400.jpg 400w,
            /images/semantic-html-800.jpg 800w,
            /images/semantic-html-1200.jpg 1200w
          "
          sizes="(max-width: 600px) 400px, (max-width: 900px) 800px, 1200px"
          alt="Diagram showing HTML5 semantic elements: header, nav, main, article, aside, footer"
          loading="eager"
          width="800"
          height="450"
        >
        <figcaption>HTML5 semantic structure provides meaningful page organization</figcaption>
      </figure>
      
      <p>
        Semantic HTML5 elements provide meaning to your content structure, improving
        accessibility, SEO, and maintainability of your web pages.
      </p>
      
      <p>
        <a href="/articles/semantic-html5">Read full article on semantic HTML5 elements</a>
      </p>
      
      <footer>
        <p>Tags: <a href="/tags/html5" rel="tag">HTML5</a>, <a href="/tags/accessibility" rel="tag">Accessibility</a></p>
      </footer>
    </article>
    
    <!-- Recent Articles -->
    <section aria-labelledby="recent-heading">
      <h2 id="recent-heading">Recent Articles</h2>
      
      <ul>
        <li>
          <article>
            <h3><a href="/articles/css-grid">Modern CSS Grid Layouts</a></h3>
            <p>Learn how to create responsive layouts with CSS Grid.</p>
            <time datetime="2026-01-15">January 15, 2026</time>
          </article>
        </li>
        
        <li>
          <article>
            <h3><a href="/articles/javascript-async">Async JavaScript Patterns</a></h3>
            <p>Master async/await and promises in JavaScript.</p>
            <time datetime="2026-01-12">January 12, 2026</time>
          </article>
        </li>
      </ul>
    </section>
  </main>
  
  <!-- Sidebar -->
  <aside aria-label="Sidebar">
    <section>
      <h2>Popular Topics</h2>
      <ul>
        <li><a href="/topics/html5">HTML5</a></li>
        <li><a href="/topics/css3">CSS3</a></li>
        <li><a href="/topics/javascript">JavaScript</a></li>
        <li><a href="/topics/accessibility">Accessibility</a></li>
      </ul>
    </section>
    
    <section>
      <h2>Newsletter</h2>
      <p>Subscribe to get the latest articles delivered to your inbox.</p>
      
      <form method="post" action="/subscribe" aria-label="Newsletter subscription">
        <div class="form-group">
          <label for="newsletter-email">Email Address <span aria-label="required">*</span></label>
          <input 
            type="email" 
            id="newsletter-email" 
            name="email" 
            required 
            aria-required="true"
            placeholder="you@example.com"
            autocomplete="email"
          >
        </div>
        
        <button type="submit">Subscribe</button>
      </form>
    </section>
  </aside>
  
  <!-- Site Footer -->
  <footer>
    <!-- Footer Navigation -->
    <nav aria-label="Footer navigation">
      <ul>
        <li><a href="/privacy">Privacy Policy</a></li>
        <li><a href="/terms">Terms of Service</a></li>
        <li><a href="/accessibility">Accessibility Statement</a></li>
        <li><a href="/sitemap">Sitemap</a></li>
      </ul>
    </nav>
    
    <!-- Copyright -->
    <p>&copy; 2026 Modern HTML5 Blog. All rights reserved.</p>
    
    <!-- Social Media -->
    <ul>
      <li><a href="https://twitter.com/username" rel="noopener noreferrer" target="_blank">Twitter (opens in new tab)</a></li>
      <li><a href="https://github.com/username" rel="noopener noreferrer" target="_blank">GitHub (opens in new tab)</a></li>
    </ul>
  </footer>
  
  <!-- Scripts (at end for performance) -->
  <script src="/js/main.js" defer></script>
</body>
</html>
```

---

## 12. Why This Configuration Works

1. **Semantic HTML5**: Improves SEO by 30-40%, enhances accessibility, provides meaningful structure.

2. **WCAG 2.1 AA Compliance**: Makes content accessible to 20%+ more users, reduces legal risk, improves UX for everyone.

3. **Performance Optimization**: Lazy loading reduces initial load by 40-60%, resource hints improve perceived performance by 30%.

4. **Clean Code**: Reduces maintenance time by 50%, easier onboarding, fewer bugs.

5. **SEO Best Practices**: Structured data increases rich snippet chances by 300%, proper meta tags improve click-through rates by 20-30%.

6. **Progressive Enhancement**: Content accessible without CSS/JS, works on all devices, future-proof.

7. **Agent Verification**: Ensures all generated HTML is valid, accessible, and performant, eliminates common errors.

---

## 13. Quick Reference

### Common Commands

```bash
# Validate HTML
npx html-validate *.html

# Accessibility audit
npx pa11y-ci *.html

# Lighthouse audit (all categories)
npx lighthouse index.html --output=html --output-path=./report.html

# Lighthouse accessibility only
npx lighthouse --only-categories=accessibility index.html

# Lighthouse performance only
npx lighthouse --only-categories=performance index.html

# Lighthouse SEO only
npx lighthouse --only-categories=seo index.html

# Format HTML with Prettier
npx prettier --write "**/*.html"

# Run HTML tests
npm test
```

### Essential HTML5 Template

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Page description">
  <title>Page Title - Site Name</title>
  <link rel="stylesheet" href="/css/styles.css">
</head>
<body>
  <a href="#main" class="skip-link">Skip to main content</a>
  <header><nav aria-label="Main">...</nav></header>
  <main id="main">...</main>
  <footer>...</footer>
  <script src="/js/main.js" defer></script>
</body>
</html>
```

### Semantic Element Quick Guide

| Element | Use For |
|---------|---------|
| `<header>` | Page/section header |
| `<nav>` | Navigation links |
| `<main>` | Main content (one per page) |
| `<article>` | Self-contained content |
| `<section>` | Thematic grouping |
| `<aside>` | Related sidebar content |
| `<footer>` | Page/section footer |
| `<figure>` | Image/media with caption |
| `<time>` | Date/time values |

### Accessibility Checklist

```
[ ] Only one <h1> per page
[ ] Headings in order (h1→h2→h3)
[ ] All images have alt text
[ ] All form inputs have labels
[ ] Skip link present
[ ] ARIA labels on nav/regions
[ ] Keyboard navigation works
[ ] Color contrast ≥ 4.5:1
```

### Performance Checklist

```
[ ] Images use loading="lazy"
[ ] Critical CSS inlined
[ ] Scripts use defer/async
[ ] Images have width/height
[ ] Resources preloaded
[ ] Modern image formats (WebP/AVIF)
```

---

## References

- [HTML5 Specification](https://html.spec.whatwg.org/)
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [W3C HTML Validator](https://validator.w3.org/)
- [Schema.org](https://schema.org/)
- [MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/HTML)
- [Web Content Accessibility Guidelines](https://www.w3.org/WAI/standards-guidelines/wcag/)
- [Google Lighthouse](https://developers.google.com/web/tools/lighthouse)

---

**Last Updated:** 2026-01-17
**Version:** 1.0
**Maintainer:** Development Team


**End of HTML Development Guidelines**
