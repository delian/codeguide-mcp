# Accessibility (a11y) Guidelines

This document provides mandatory standards for building accessible web applications following WCAG 2.1 AA guidelines.

---

**Agent Profile**: The Accessibility Expert
**Role**: Senior Accessibility Engineer & Inclusive Design Advocate
**Objective**: Generate inclusive, WCAG-compliant interfaces that work for all users regardless of ability.
**Tools**: axe-core, WAVE, Lighthouse, NVDA, VoiceOver, JAWS.

---

## 1. Core Philosophies: A11Y-FIRST

- **A**ll users: Design for everyone from the start
- **1**st class: Accessibility is a requirement, not an afterthought
- **1** experience: Same content and functionality for all users
- **Y**es to testing: Test with real assistive technologies

---

## 2. WCAG 2.1 Principles (MANDATORY)

### A. POUR Framework

```markdown
## Perceivable
Information and UI components must be presentable in ways users can perceive.

## Operable
UI components and navigation must be operable by all users.

## Understandable
Information and UI operation must be understandable.

## Robust
Content must be robust enough to work with assistive technologies.
```

---

## 3. Semantic HTML (MANDATORY)

### A. Document Structure

```html
<!-- ✅ CORRECT: Semantic HTML structure -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Page Title - Site Name</title>
</head>
<body>
  <a href="#main-content" class="skip-link">Skip to main content</a>

  <header role="banner">
    <nav aria-label="Main navigation">
      <ul>
        <li><a href="/" aria-current="page">Home</a></li>
        <li><a href="/products">Products</a></li>
        <li><a href="/about">About</a></li>
      </ul>
    </nav>
  </header>

  <main id="main-content" role="main">
    <article>
      <h1>Page Title</h1>
      <p>Main content...</p>

      <section aria-labelledby="section-heading">
        <h2 id="section-heading">Section Title</h2>
        <p>Section content...</p>
      </section>
    </article>

    <aside aria-label="Related content">
      <h2>Related Articles</h2>
      <!-- Related content -->
    </aside>
  </main>

  <footer role="contentinfo">
    <nav aria-label="Footer navigation">
      <!-- Footer links -->
    </nav>
    <p>&copy; 2024 Company Name</p>
  </footer>
</body>
</html>

<!-- ❌ WRONG: Div soup with no semantics -->
<div class="header">
  <div class="nav">
    <div class="nav-item">Home</div>
  </div>
</div>
<div class="main">
  <div class="title">Page Title</div>
</div>
```

### B. Heading Hierarchy

```html
<!-- ✅ CORRECT: Logical heading hierarchy -->
<h1>Main Page Title</h1>
  <h2>First Section</h2>
    <h3>Subsection</h3>
    <h3>Another Subsection</h3>
  <h2>Second Section</h2>
    <h3>Subsection</h3>
      <h4>Deep Subsection</h4>

<!-- ❌ WRONG: Skipping heading levels -->
<h1>Main Page Title</h1>
<h3>Section</h3>  <!-- Skipped h2 -->
<h5>Subsection</h5>  <!-- Skipped h4 -->

<!-- ❌ WRONG: Using headings for styling -->
<h3>This text just needs to be big</h3>  <!-- Use CSS instead -->
```

---

## 4. Images and Media (MANDATORY)

### A. Alt Text

```html
<!-- ✅ CORRECT: Descriptive alt text -->
<img src="chart.png" alt="Bar chart showing sales increased 25% from Q1 to Q2">

<!-- ✅ CORRECT: Decorative images -->
<img src="decorative-border.png" alt="" role="presentation">

<!-- ✅ CORRECT: Complex images with long description -->
<figure>
  <img
    src="complex-diagram.png"
    alt="System architecture diagram"
    aria-describedby="diagram-description"
  >
  <figcaption id="diagram-description">
    The system consists of three main components: a React frontend,
    a Node.js API server, and a PostgreSQL database. The frontend
    communicates with the API via REST endpoints...
  </figcaption>
</figure>

<!-- ❌ WRONG: Non-descriptive alt text -->
<img src="chart.png" alt="chart">
<img src="chart.png" alt="image">
<img src="chart.png" alt="chart.png">

<!-- ❌ WRONG: Missing alt attribute -->
<img src="important-info.png">
```

### B. Video and Audio

```html
<!-- ✅ CORRECT: Accessible video -->
<video controls>
  <source src="video.mp4" type="video/mp4">
  <track
    kind="captions"
    src="captions-en.vtt"
    srclang="en"
    label="English captions"
    default
  >
  <track
    kind="descriptions"
    src="descriptions-en.vtt"
    srclang="en"
    label="English audio descriptions"
  >
  <!-- Fallback content -->
  <p>Your browser doesn't support video.
     <a href="video.mp4">Download the video</a> or
     <a href="transcript.html">read the transcript</a>.
  </p>
</video>

<!-- ✅ CORRECT: Audio with transcript -->
<audio controls aria-describedby="audio-transcript">
  <source src="podcast.mp3" type="audio/mpeg">
</audio>
<div id="audio-transcript">
  <h3>Transcript</h3>
  <p>Full transcript text...</p>
</div>
```

---

## 5. Forms (MANDATORY)

### A. Form Labels and Structure

```html
<!-- ✅ CORRECT: Properly labeled form -->
<form>
  <fieldset>
    <legend>Personal Information</legend>

    <div class="form-group">
      <label for="full-name">Full Name <span aria-hidden="true">*</span></label>
      <input
        type="text"
        id="full-name"
        name="fullName"
        required
        aria-required="true"
        autocomplete="name"
      >
    </div>

    <div class="form-group">
      <label for="email">Email Address <span aria-hidden="true">*</span></label>
      <input
        type="email"
        id="email"
        name="email"
        required
        aria-required="true"
        aria-describedby="email-hint"
        autocomplete="email"
      >
      <p id="email-hint" class="hint">We'll never share your email.</p>
    </div>
  </fieldset>

  <fieldset>
    <legend>Notification Preferences</legend>

    <div class="checkbox-group">
      <input type="checkbox" id="email-updates" name="emailUpdates">
      <label for="email-updates">Receive email updates</label>
    </div>

    <div class="checkbox-group">
      <input type="checkbox" id="sms-updates" name="smsUpdates">
      <label for="sms-updates">Receive SMS updates</label>
    </div>
  </fieldset>

  <button type="submit">Submit</button>
</form>

<!-- ❌ WRONG: Missing labels -->
<input type="text" placeholder="Enter name">

<!-- ❌ WRONG: Label not associated -->
<label>Name</label>
<input type="text">
```

### B. Error Handling

```html
<!-- ✅ CORRECT: Accessible error messages -->
<div class="form-group" aria-live="polite">
  <label for="password">Password</label>
  <input
    type="password"
    id="password"
    name="password"
    aria-invalid="true"
    aria-describedby="password-error password-requirements"
    required
  >
  <p id="password-error" class="error" role="alert">
    Password must be at least 8 characters.
  </p>
  <ul id="password-requirements" class="requirements">
    <li>At least 8 characters</li>
    <li>At least one uppercase letter</li>
    <li>At least one number</li>
  </ul>
</div>

<!-- ✅ CORRECT: Form-level error summary -->
<div role="alert" aria-labelledby="error-summary-title" class="error-summary">
  <h2 id="error-summary-title">There are 2 errors in your form</h2>
  <ul>
    <li><a href="#email">Email address is required</a></li>
    <li><a href="#password">Password is too short</a></li>
  </ul>
</div>
```

---

## 6. Interactive Components (MANDATORY)

### A. Buttons and Links

```html
<!-- ✅ CORRECT: Button for actions -->
<button type="button" onclick="openModal()">
  Open Settings
</button>

<!-- ✅ CORRECT: Link for navigation -->
<a href="/settings">Go to Settings</a>

<!-- ✅ CORRECT: Button with icon and text -->
<button type="button">
  <svg aria-hidden="true" focusable="false">...</svg>
  <span>Delete Item</span>
</button>

<!-- ✅ CORRECT: Icon-only button with accessible name -->
<button type="button" aria-label="Close dialog">
  <svg aria-hidden="true" focusable="false">
    <use href="#icon-close"></use>
  </svg>
</button>

<!-- ❌ WRONG: Div as button -->
<div onclick="submit()" class="button">Submit</div>

<!-- ❌ WRONG: Link for action -->
<a href="#" onclick="deleteItem()">Delete</a>

<!-- ❌ WRONG: Empty link -->
<a href="/page"><img src="icon.png"></a>
```

### B. Custom Components

```tsx
// ✅ CORRECT: Accessible custom dropdown
function Dropdown({ label, options, value, onChange }) {
  const [isOpen, setIsOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  const handleKeyDown = (e: KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        if (!isOpen) {
          setIsOpen(true);
        } else {
          setActiveIndex(i => Math.min(i + 1, options.length - 1));
        }
        break;
      case 'ArrowUp':
        e.preventDefault();
        setActiveIndex(i => Math.max(i - 1, 0));
        break;
      case 'Enter':
      case ' ':
        e.preventDefault();
        if (isOpen && activeIndex >= 0) {
          onChange(options[activeIndex]);
          setIsOpen(false);
          buttonRef.current?.focus();
        } else {
          setIsOpen(true);
        }
        break;
      case 'Escape':
        setIsOpen(false);
        buttonRef.current?.focus();
        break;
      case 'Home':
        e.preventDefault();
        setActiveIndex(0);
        break;
      case 'End':
        e.preventDefault();
        setActiveIndex(options.length - 1);
        break;
    }
  };

  return (
    <div className="dropdown">
      <label id="dropdown-label">{label}</label>
      <button
        ref={buttonRef}
        type="button"
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-labelledby="dropdown-label"
        onClick={() => setIsOpen(!isOpen)}
        onKeyDown={handleKeyDown}
      >
        {value || 'Select an option'}
        <span aria-hidden="true">▼</span>
      </button>

      {isOpen && (
        <ul
          ref={listRef}
          role="listbox"
          aria-labelledby="dropdown-label"
          aria-activedescendant={activeIndex >= 0 ? `option-${activeIndex}` : undefined}
          tabIndex={-1}
        >
          {options.map((option, index) => (
            <li
              key={option.value}
              id={`option-${index}`}
              role="option"
              aria-selected={value === option.value}
              className={index === activeIndex ? 'active' : ''}
              onClick={() => {
                onChange(option);
                setIsOpen(false);
              }}
            >
              {option.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

---

## 7. Keyboard Navigation (MANDATORY)

### A. Focus Management

```css
/* ✅ CORRECT: Visible focus indicators */
:focus {
  outline: 2px solid #005fcc;
  outline-offset: 2px;
}

/* ✅ CORRECT: Enhanced focus for better visibility */
:focus-visible {
  outline: 3px solid #005fcc;
  outline-offset: 2px;
  box-shadow: 0 0 0 4px rgba(0, 95, 204, 0.3);
}

/* ❌ WRONG: Removing focus outline */
:focus {
  outline: none;
}
```

```tsx
// ✅ CORRECT: Focus trap for modals
function Modal({ isOpen, onClose, children }) {
  const modalRef = useRef<HTMLDivElement>(null);
  const previousActiveElement = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (isOpen) {
      // Store current focus
      previousActiveElement.current = document.activeElement as HTMLElement;

      // Focus the modal
      modalRef.current?.focus();

      // Trap focus
      const handleTab = (e: KeyboardEvent) => {
        if (e.key !== 'Tab') return;

        const focusableElements = modalRef.current?.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );

        if (!focusableElements?.length) return;

        const firstElement = focusableElements[0] as HTMLElement;
        const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

        if (e.shiftKey && document.activeElement === firstElement) {
          e.preventDefault();
          lastElement.focus();
        } else if (!e.shiftKey && document.activeElement === lastElement) {
          e.preventDefault();
          firstElement.focus();
        }
      };

      document.addEventListener('keydown', handleTab);
      return () => document.removeEventListener('keydown', handleTab);
    } else {
      // Restore focus
      previousActiveElement.current?.focus();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div
      ref={modalRef}
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
      tabIndex={-1}
    >
      <h2 id="modal-title">Modal Title</h2>
      {children}
      <button onClick={onClose}>Close</button>
    </div>
  );
}
```

### B. Skip Links

```html
<!-- ✅ CORRECT: Skip links for keyboard users -->
<body>
  <a href="#main-content" class="skip-link">Skip to main content</a>
  <a href="#main-nav" class="skip-link">Skip to navigation</a>

  <header>
    <nav id="main-nav">...</nav>
  </header>

  <main id="main-content">...</main>
</body>

<style>
.skip-link {
  position: absolute;
  top: -40px;
  left: 0;
  background: #000;
  color: #fff;
  padding: 8px;
  z-index: 100;
  transition: top 0.3s;
}

.skip-link:focus {
  top: 0;
}
</style>
```

---

## 8. Color and Contrast (MANDATORY)

### A. Contrast Requirements

```css
/* WCAG AA requires:
   - Normal text: 4.5:1 contrast ratio
   - Large text (18pt+ or 14pt+ bold): 3:1 contrast ratio
   - UI components: 3:1 contrast ratio
*/

/* ✅ CORRECT: Sufficient contrast */
.text-primary {
  color: #1a1a1a; /* On white: 16:1 ratio */
}

.text-secondary {
  color: #595959; /* On white: 7:1 ratio */
}

.button-primary {
  background-color: #0066cc;
  color: #ffffff; /* 4.5:1 ratio */
}

/* ❌ WRONG: Insufficient contrast */
.low-contrast {
  color: #999999; /* On white: 2.85:1 - fails AA */
}
```

### B. Don't Rely on Color Alone

```html
<!-- ✅ CORRECT: Color + icon + text -->
<span class="status status-error">
  <svg aria-hidden="true"><!-- Error icon --></svg>
  Error: Invalid email address
</span>

<span class="status status-success">
  <svg aria-hidden="true"><!-- Checkmark icon --></svg>
  Success: Form submitted
</span>

<!-- ✅ CORRECT: Links distinguished by more than color -->
<p>
  Read our <a href="/terms" class="underline">Terms of Service</a>
  for more information.
</p>

<style>
a {
  color: #0066cc;
  text-decoration: underline;
}
</style>

<!-- ❌ WRONG: Only color indicates state -->
<span class="status-error" style="color: red;">Invalid</span>
```

---

## 9. ARIA (MANDATORY)

### A. ARIA Rules

```html
<!-- Rule 1: Don't use ARIA if native HTML works -->
<!-- ❌ WRONG -->
<div role="button" tabindex="0">Click me</div>

<!-- ✅ CORRECT -->
<button>Click me</button>

<!-- Rule 2: Don't change native semantics -->
<!-- ❌ WRONG -->
<h1 role="button">Heading</h1>

<!-- ✅ CORRECT -->
<h1><button>Heading</button></h1>

<!-- Rule 3: Interactive elements must be keyboard accessible -->
<!-- ❌ WRONG: Not keyboard accessible -->
<span role="button" onclick="doSomething()">Click</span>

<!-- ✅ CORRECT: Keyboard accessible -->
<span
  role="button"
  tabindex="0"
  onclick="doSomething()"
  onkeydown="if(event.key === 'Enter' || event.key === ' ') doSomething()"
>
  Click
</span>
```

### B. Live Regions

```html
<!-- Announcements -->
<div aria-live="polite" aria-atomic="true" class="sr-only">
  <!-- Dynamic content announced when changed -->
</div>

<!-- Urgent alerts -->
<div role="alert" aria-live="assertive">
  Session expires in 2 minutes
</div>

<!-- Status messages -->
<div role="status" aria-live="polite">
  3 items added to cart
</div>

<!-- Progress updates -->
<div
  role="progressbar"
  aria-valuenow="75"
  aria-valuemin="0"
  aria-valuemax="100"
  aria-label="Upload progress"
>
  75%
</div>
```

---

## 10. Testing (MANDATORY)

### A. Automated Testing

```typescript
// Jest + axe-core testing
import { axe, toHaveNoViolations } from 'jest-axe';

expect.extend(toHaveNoViolations);

describe('Accessibility', () => {
  test('home page has no accessibility violations', async () => {
    const { container } = render(<HomePage />);
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });

  test('form has no accessibility violations', async () => {
    const { container } = render(<SignupForm />);
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });
});

// Playwright accessibility testing
import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('Accessibility', () => {
  test('should not have any automatically detectable accessibility issues', async ({ page }) => {
    await page.goto('/');

    const accessibilityScanResults = await new AxeBuilder({ page }).analyze();

    expect(accessibilityScanResults.violations).toEqual([]);
  });
});
```

### B. Manual Testing Checklist

```markdown
## Keyboard Testing
- [ ] All interactive elements reachable with Tab
- [ ] Focus order is logical
- [ ] Focus indicator is visible
- [ ] No keyboard traps
- [ ] Skip links work
- [ ] Modals trap focus correctly

## Screen Reader Testing
- [ ] Page title is descriptive
- [ ] Headings hierarchy is logical
- [ ] Images have appropriate alt text
- [ ] Forms have proper labels
- [ ] Error messages are announced
- [ ] Dynamic content updates are announced

## Visual Testing
- [ ] Text meets contrast requirements
- [ ] Page is usable at 200% zoom
- [ ] No horizontal scrolling at 320px width
- [ ] Information not conveyed by color alone
- [ ] Focus indicators visible

## Cognitive Testing
- [ ] Instructions are clear
- [ ] Error messages are helpful
- [ ] No time limits without warnings
- [ ] Consistent navigation
```

---

## 11. Deployment Checklist

### Development
- [ ] Semantic HTML used throughout
- [ ] All images have alt text
- [ ] Forms properly labeled
- [ ] Keyboard navigation works
- [ ] ARIA used correctly

### Design
- [ ] Color contrast meets AA
- [ ] Focus states designed
- [ ] Touch targets 44x44px minimum
- [ ] Responsive down to 320px

### Testing
- [ ] Automated tests pass
- [ ] Screen reader tested
- [ ] Keyboard-only testing done
- [ ] Zoom testing complete

### Documentation
- [ ] VPAT completed
- [ ] Accessibility statement published
- [ ] Known issues documented

---

## 12. Quick Reference

```html
<!-- Landmarks -->
<header role="banner">
<nav role="navigation">
<main role="main">
<aside role="complementary">
<footer role="contentinfo">

<!-- Live regions -->
aria-live="polite"    <!-- Non-urgent updates -->
aria-live="assertive" <!-- Urgent updates -->
role="alert"          <!-- Important messages -->
role="status"         <!-- Status updates -->

<!-- Forms -->
aria-required="true"
aria-invalid="true"
aria-describedby="hint-id"
aria-errormessage="error-id"

<!-- Interactive states -->
aria-expanded="true|false"
aria-selected="true|false"
aria-checked="true|false|mixed"
aria-pressed="true|false"
aria-disabled="true"

<!-- Relationships -->
aria-labelledby="id"
aria-describedby="id"
aria-controls="id"
aria-owns="id"
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** UX Team
