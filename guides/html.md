# HTML Development Guidelines
Mandatory standards for modern, semantic, accessible HTML: correct sectioning, robust forms, accurate metadata, and current platform elements. HTML Living Standard, html-validate, W3C Validator, axe-core, Prettier.

---
name: html
title: HTML Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [html-living-standard, html-validate@9, w3c-validator, axe-core@4, lighthouse@12, prettier@3]
requires: []
recommends:
  - accessibility
  - css
  - ui
  - secure-coding
provides:
  - semantic-html
  - html-forms
  - html-metadata
  - modern-html-elements
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to HTML — the document language itself.

---

## 0. Prerequisites & References

HTML has no hard prerequisites, but it is the substrate for several cross-cutting concerns. Fetch the relevant guide when the task touches it; this guide keeps only the HTML binding.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`accessibility.md`](guides://accessibility.md) — WCAG, ARIA, keyboard, contrast, screen-reader behavior. **Semantic HTML *is* the accessibility foundation**; this guide owns the HTML binding (landmarks, `alt`, labels, heading order), and defers all ARIA depth and audit policy to the owner.
> - [`css.md`](guides://css.md) — all styling. HTML carries structure only; presentation attributes and inline styles are forbidden (§3).
> - [`ui.md`](guides://ui.md) — form UX, validation messaging, focus/error flows. This guide owns the markup; `ui.md` owns the interaction design.
> - [`secure-coding.md`](guides://secure-coding.md) — sanitization of user-supplied HTML, `rel` hardening, Content-Security-Policy. Bindings appear in §3 and §10.

> 📎 **SEE ALSO:** [`markdown.md`](guides://markdown.md) · [`openapi.md`](guides://openapi.md) · [`e2e-testing.md`](guides://e2e-testing.md) *(DOM-level page assertions)*

---

## 1. Core Philosophies: SEMANTIC-FIRST

HTML-specific principles only. Accessibility, styling, and security policy come from §0.

- **S**emantics over `div`s: every element is chosen for its *meaning*. A `<div>`/`<span>` is the element of last resort, used only when no semantic element fits.
- **E**xpress structure, not style: HTML describes content; CSS describes appearance. No presentation attributes, no inline `style`.
- **M**eaning is the contract: landmarks, headings, and labels form the accessibility tree — they are not decoration (binding to `accessibility.md`).
- **A**ttributes do real work: native form validation, `autocomplete` tokens, `loading`, `type`, and `lang` replace JavaScript wherever the platform already solves it.
- **N**ative before custom: prefer `<dialog>`, `<details>`, the popover API, and real form controls over JS re-implementations.
- **T**est the parse tree: validity and the accessibility tree are auditable gates (§2), not opinions.
- **I**nternationalize by default: `lang`, `dir`, and `<meta charset="utf-8">` are always present and correct.
- **C**ontent works without CSS/JS: markup is meaningful and usable as plain HTML (progressive enhancement).

**Verified Code**: Agent-generated HTML MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `HTML-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| HTML-VALID-01 | Markup MUST be valid against the living standard | `npx html-validate "**/*.html"` | exit 0, 0 errors |
| HTML-STRUCT-01 | `<!DOCTYPE html>`, `<html lang>`, `<meta charset="utf-8">` (first), and `<meta name="viewport">` MUST be present | review / html-validate | all present |
| HTML-SEM-01 | One `<main>` and one `<h1>` per page; headings MUST NOT skip levels; landmarks used over generic `div`s | `npx html-validate` (heading-levels, no-redundant-role) | exit 0 |
| HTML-A11Y-01 | Every `<img>` MUST have `alt`; every control a programmatic label; 0 critical axe violations (see `accessibility.md`) | `npx @axe-core/cli <url>` | 0 critical/serious |
| HTML-FORM-01 | Inputs MUST use the correct `type`, an associated `<label>`, and an `autocomplete` token where one exists | review / html-validate (`input-missing-label`) | exit 0 |
| HTML-SEC-01 | `target="_blank"` MUST carry `rel="noopener"`; user-supplied HTML MUST be sanitized server-side (see `secure-coding.md`) | review / grep | no bare `_blank`, sanitized |
| HTML-FMT-01 | Markup MUST be formatted | `npx prettier --check "**/*.html"` | no diff |
| HTML-PERF-01 | Below-the-fold media MUST set `loading="lazy"`; raster `<img>` MUST set `width`/`height` (or `aspect-ratio`) | review / lighthouse | no CLS from media |
| HTML-SEO-01 | Each page MUST have a unique `<title>`, `<meta name="description">`, and `<link rel="canonical">` | review | present & unique |

> **Forbidden**: `div`/`span` soup where a semantic element exists; presentation attributes or inline `style` (use `css.md`); placeholder used as a label; tables for layout; deprecated elements (`<font>`, `<center>`, `<marquee>`, `<acronym>`); `<meta http-equiv="X-UA-Compatible">` and `<meta name="keywords">` (both obsolete/ignored); injecting unsanitized user HTML.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
npx prettier --check "**/*.html"     # HTML-FMT-01
npx html-validate "**/*.html"        # HTML-VALID-01/STRUCT-01/SEM-01/FORM-01
npx @axe-core/cli http://localhost:PORT   # HTML-A11Y-01 (serve first)
npx lighthouse http://localhost:PORT --only-categories=performance,seo  # HTML-PERF-01/SEO-01
```

`html-validate` is configurable via `.htmlvalidate.json` (extend `html-validate:recommended`); the W3C Nu validator (`validator.w3.org` / `vnu.jar`) is the authoritative cross-check. The *why* behind accessibility and security gates lives in their §0 owners.

---

## 4. Document Skeleton & Metadata

The canonical document. `<meta charset>` MUST be the first child of `<head>` (it must appear within the first 1024 bytes); `viewport` is required for responsive layout.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Specific, Unique Page Title — Site Name</title>
  <meta name="description" content="One clear sentence, ~150 chars, unique per page.">
  <link rel="canonical" href="https://example.com/page">

  <!-- Open Graph: the de-facto social/share contract (Twitter reads OG as fallback) -->
  <meta property="og:type" content="article">
  <meta property="og:title" content="Page Title">
  <meta property="og:description" content="Share-card description.">
  <meta property="og:image" content="https://example.com/card.jpg">
  <meta property="og:url" content="https://example.com/page">
  <meta name="twitter:card" content="summary_large_image">

  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <link rel="apple-touch-icon" href="/apple-touch-icon.png">
  <link rel="manifest" href="/site.webmanifest">
  <meta name="theme-color" content="#0b5fff">

  <link rel="stylesheet" href="/css/styles.css">
  <script src="/js/main.js" type="module"></script>  <!-- type=module defers by default -->
</head>
<body>
  <a class="skip-link" href="#main">Skip to main content</a>
  <header>…</header>
  <main id="main">…</main>
  <footer>…</footer>
</body>
</html>
```

Modern notes: drop `X-UA-Compatible` and `meta name="keywords"` (ignored). Prefer a single SVG favicon plus one PNG apple-touch fallback over the old multi-size PNG pile. Use `<script type="module">` (deferred by default) or `defer`; never block parsing in `<head>`. Resource hints — `preload` (critical font/CSS/LCP image), `preconnect` (third-party origin), `prefetch` (next navigation) — are applied sparingly; over-preloading regresses performance.

---

## 5. Semantic Structure: Landmarks, Sectioning & Headings

Choose the element that names the content. The result is the accessibility tree (see `accessibility.md`).

| Element | Implicit landmark / role | Use for |
|---|---|---|
| `<header>` | `banner` (page-level) | Site/section masthead |
| `<nav>` | `navigation` | A set of navigation links; label each: `aria-label` |
| `<main>` | `main` (one per page) | The primary unique content |
| `<article>` | `article` | Independently distributable unit (post, card, comment) |
| `<section>` | `region` (if labeled) | Thematic grouping **with a heading** |
| `<aside>` | `complementary` | Tangential content (sidebar, callout) |
| `<footer>` | `contentinfo` (page-level) | Page/section footer |
| `<search>` | `search` | Search form region (modern, replaces `role="search"`) |
| `<figure>`/`<figcaption>` | — | Self-contained media with a caption |

```html
<main id="main">
  <h1>Page Title</h1>                      <!-- exactly one h1 -->
  <article>
    <header>
      <h2>Article Title</h2>
      <p>By <a href="/u/jane" rel="author">Jane</a> ·
         <time datetime="2026-06-05">June 5, 2026</time></p>
    </header>
    <section aria-labelledby="bg">
      <h3 id="bg">Background</h3>            <!-- h1→h2→h3, no skips -->
      <p>…</p>
    </section>
  </article>
</main>
```

Rules: a `<section>` is only a landmark when it has an accessible name (`aria-labelledby` pointing at its heading) — otherwise prefer a plain heading + content or a `<div>`. Use multiple `<nav>`/`<header>`/`<footer>` freely, but distinguish same-type landmarks with labels. Headings convey document outline; **never pick a heading level for its font size** — that is CSS's job (`css.md`).

---

## 6. Forms

HTML forms own validation, input semantics, and labeling. UX (error copy, focus flow, inline messaging) belongs to [`ui.md`](guides://ui.md).

```html
<form method="post" action="/signup">
  <fieldset>
    <legend>Account</legend>

    <div class="field">
      <label for="email">Email</label>
      <input id="email" name="email" type="email"
             required autocomplete="email" inputmode="email"
             aria-describedby="email-hint">
      <small id="email-hint">We never share it.</small>
    </div>

    <div class="field">
      <label for="pw">Password</label>
      <input id="pw" name="pw" type="password"
             required minlength="12" autocomplete="new-password">
    </div>
  </fieldset>

  <fieldset>
    <legend>Contact preference</legend>
    <label><input type="radio" name="contact" value="email" checked> Email</label>
    <label><input type="radio" name="contact" value="sms"> SMS</label>
  </fieldset>

  <button type="submit">Create account</button>
</form>
```

- **Labels are mandatory.** Associate via `<label for>`/`id` (preferred) or by wrapping. `placeholder` is a hint, never a label.
- **Use the right `type`/`inputmode`**: `email`, `tel`, `url`, `number`, `date`, `search`, `color`, `range`, `file`. These give native validation, keyboards, and pickers for free.
- **Native validation attributes**: `required`, `min`/`max`/`step`, `minlength`/`maxlength`, `pattern`, `type`. Style state with `:invalid`/`:user-invalid` (CSS); the `:user-invalid` pseudo-class avoids flagging untouched fields.
- **`autocomplete` vocabulary** — use the standardized tokens so browsers/password managers fill correctly: `name`, `given-name`, `family-name`, `email`, `username`, `current-password`, `new-password`, `one-time-code`, `street-address`, `address-line1`, `postal-code`, `country`, `tel`, `cc-number`, `cc-exp`. Use `autocomplete="off"` only with cause.
- **Group** related controls with `<fieldset>`/`<legend>` (required for radio/checkbox sets). Pair `<input>`/`<textarea>`/`<select>` with `<datalist>` for suggestion lists.
- **Buttons**: always set `type` (`submit`/`button`/`reset`); a bare `<button>` defaults to `submit`. Use a real `<button>`, never a clickable `<div>`.

---

## 7. Media: Responsive Images, Video & Audio

```html
<!-- Art direction + format negotiation -->
<picture>
  <source srcset="hero.avif" type="image/avif">
  <source srcset="hero.webp" type="image/webp">
  <img src="hero.jpg" alt="Team on stage at the 2026 launch"
       width="1200" height="630" loading="eager" fetchpriority="high">
</picture>

<!-- Resolution switching: let the browser pick by viewport + DPR -->
<img src="photo-800.jpg"
     srcset="photo-400.jpg 400w, photo-800.jpg 800w, photo-1200.jpg 1200w"
     sizes="(max-width: 600px) 100vw, 800px"
     alt="…" width="800" height="600" loading="lazy" decoding="async">

<video controls width="640" height="360" poster="poster.jpg" preload="metadata">
  <source src="clip.webm" type="video/webm">
  <source src="clip.mp4" type="video/mp4">
  <track kind="captions" src="clip.en.vtt" srclang="en" label="English" default>
</video>
```

- Always set `width`/`height` (or CSS `aspect-ratio`) to reserve space and prevent layout shift (HTML-PERF-01).
- `loading="lazy"` for below-the-fold images and iframes; `loading="eager"` + `fetchpriority="high"` for the LCP image.
- `<picture>` for format/art-direction; bare `srcset`+`sizes` for resolution switching.
- Decorative images take `alt=""` (empty, not missing). Video/audio MUST ship captions (`<track kind="captions">`) — accessibility binding, see `accessibility.md`.

---

## 8. Tables, Links & Navigation

**Data tables** (never layout) carry real structure:

```html
<table>
  <caption>Q2 revenue by region</caption>
  <thead>
    <tr><th scope="col">Region</th><th scope="col">Revenue</th></tr>
  </thead>
  <tbody>
    <tr><th scope="row">EMEA</th><td>$1.2M</td></tr>
    <tr><th scope="row">APAC</th><td>$0.9M</td></tr>
  </tbody>
  <tfoot>
    <tr><th scope="row">Total</th><td>$2.1M</td></tr>
  </tfoot>
</table>
```

`<caption>`, `<th scope="col|row">`, and `<thead>`/`<tbody>`/`<tfoot>` make the table navigable by assistive tech; use `headers`/`id` for complex multi-level headers.

**Links**: text MUST be self-describing out of context (no "click here"/"read more"). Use `rel="noopener"` on any `target="_blank"` (security — `secure-coding.md`), `rel="nofollow ugc"` on user-generated links, `download` for downloads, and `aria-current="page"` on the active nav item. Mark the current page in navigation; wrap link sets in a labeled `<nav>`.

---

## 9. Modern Platform Elements

Prefer these native elements over JavaScript re-implementations.

```html
<!-- Native modal: focus trap, Esc-to-close, ::backdrop, inert background — all free -->
<dialog id="confirm">
  <form method="dialog">
    <h2>Delete file?</h2>
    <button value="cancel">Cancel</button>
    <button value="ok">Delete</button>
  </form>
</dialog>
<button onclick="document.getElementById('confirm').showModal()">Delete…</button>

<!-- Disclosure widget: no JS needed -->
<details>
  <summary>Shipping &amp; returns</summary>
  <p>Free returns within 30 days.</p>
</details>

<!-- Popover API: light-dismiss, top-layer, declarative trigger -->
<button popovertarget="menu">Menu</button>
<div id="menu" popover>
  <ul><li><a href="/profile">Profile</a></li><li><a href="/logout">Log out</a></li></ul>
</div>
```

- `<dialog>` + `showModal()` is the standard modal: it manages the top layer, focus, `Esc`, `::backdrop`, and inert content. `<form method="dialog">` closes it and reports the pressed button's `value`.
- `<details>`/`<summary>` give accordions/disclosure for free; group with a shared `name` for exclusive (one-open) accordion behavior.
- The **popover API** (`popover`, `popovertarget`) provides light-dismiss overlays (menus, tooltips, teaching UI) declaratively, with top-layer stacking and no z-index wars.
- Other workhorses: `<output>` (live calc result), `<progress>`/`<meter>`, `<time datetime>`, `<datalist>`, `<template>` (inert, cloneable fragment), `<slot>`.

**Web Components (platform basics).** Custom elements extend the platform; `<template>` + Shadow DOM + `<slot>` encapsulate markup and styles:

```html
<template id="card-tpl">
  <style>:host { display: block; border: 1px solid #ddd; }</style>
  <slot name="title"></slot>
  <slot></slot>
</template>
<script type="module">
  customElements.define('user-card', class extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({ mode: 'open' })
        .append(document.getElementById('card-tpl').content.cloneNode(true));
    }
  });
</script>
<user-card><span slot="title">Jane</span><p>Bio…</p></user-card>
```

Custom element names MUST contain a hyphen. Keep shadow-DOM components accessible (forward ARIA, label slotted controls) — accessibility policy stays in `accessibility.md`.

---

## 10. Structured Data & Security Binding

**Structured data** improves machine understanding and rich results. Prefer JSON-LD (decoupled from markup) over inline microdata:

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Article",
  "headline": "The Power of Semantic HTML",
  "author": { "@type": "Person", "name": "Jane Doe" },
  "datePublished": "2026-06-05"
}
</script>
```

Inline **microdata** (`itemscope`/`itemtype`/`itemprop`) remains valid when data must live on visible elements; validate either with Google's Rich Results Test / Schema.org validator.

**Security binding** (policy owned by [`secure-coding.md`](guides://secure-coding.md)):
- **Never** inject unsanitized user HTML; sanitize server-side or with a vetted sanitizer, and prefer `textContent` over `innerHTML`.
- `target="_blank"` → `rel="noopener"` (modern browsers imply it, but state it; add `noreferrer` to also strip the referrer). Add `rel="nofollow ugc"` to user-submitted links.
- Enforce a **Content-Security-Policy** (HTTP header preferred); a strict CSP makes inline `<script>`/`onclick`/inline `style` fail — another reason to keep behavior in external modules and presentation in CSS (`css.md`).
- Use `<iframe sandbox>` and `referrerpolicy` for embedded/third-party content; load all subresources over HTTPS.

---

## 11. Project Structure

```
project/
├── index.html
├── about.html
├── css/styles.css          # all styling (see css.md)
├── js/main.js              # behavior, ES modules
├── assets/                 # images, fonts, video, vtt captions
├── site.webmanifest
├── robots.txt
├── sitemap.xml
└── .htmlvalidate.json      # html-validate config (extends recommended)
```

Keep structure (HTML), presentation (CSS), and behavior (JS) in separate files. Validate every page; serve and audit before delivery.

---

## 12. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] HTML-VALID-01 — `html-validate` clean, 0 errors
- [ ] HTML-STRUCT-01 — doctype, `lang`, `charset` (first), viewport present
- [ ] HTML-SEM-01 — one `<main>`/`<h1>`, no skipped headings, landmarks over `div`s
- [ ] HTML-A11Y-01 — all `<img>` have `alt`, controls labeled, 0 critical axe issues (see `accessibility.md`)
- [ ] HTML-FORM-01 — correct `type`, associated labels, `autocomplete` tokens
- [ ] HTML-SEC-01 — `_blank` links carry `rel="noopener"`, user HTML sanitized (see `secure-coding.md`)
- [ ] HTML-FMT-01 — Prettier clean, no diff
- [ ] HTML-PERF-01 — below-fold media lazy, images sized (no CLS)
- [ ] HTML-SEO-01 — unique title, description, canonical per page
- [ ] Agent ran every §3 command and documented any fixes

---
**End of HTML Guidelines**
