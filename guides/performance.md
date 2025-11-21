# Web Performance Guidelines

This document provides mandatory standards for building fast, efficient web applications.

---

**Agent Profile**: The Performance Expert
**Role**: Senior Performance Engineer & Web Vitals Specialist
**Objective**: Generate optimized, fast-loading applications that meet Core Web Vitals thresholds.
**Tools**: Lighthouse, WebPageTest, Chrome DevTools, bundlesize, webpack-bundle-analyzer.

---

## 1. Core Philosophies: PERF-FIRST

- **P**rioritize: Load critical resources first
- **E**liminate: Remove unnecessary code and requests
- **R**educe: Minimize payload sizes
- **F**ast: Optimize for perceived and actual speed

---

## 2. Core Web Vitals (MANDATORY)

### A. Metrics and Thresholds

```markdown
## Largest Contentful Paint (LCP)
Measures loading performance.
- Good: ≤ 2.5 seconds
- Needs Improvement: 2.5 - 4.0 seconds
- Poor: > 4.0 seconds

## Interaction to Next Paint (INP)
Measures interactivity/responsiveness.
- Good: ≤ 200 milliseconds
- Needs Improvement: 200 - 500 milliseconds
- Poor: > 500 milliseconds

## Cumulative Layout Shift (CLS)
Measures visual stability.
- Good: ≤ 0.1
- Needs Improvement: 0.1 - 0.25
- Poor: > 0.25
```

### B. Measuring Performance

```javascript
// Report Web Vitals
import { onCLS, onINP, onLCP, onFCP, onTTFB } from 'web-vitals';

function sendToAnalytics(metric) {
  const body = JSON.stringify({
    name: metric.name,
    value: metric.value,
    rating: metric.rating,
    delta: metric.delta,
    id: metric.id,
    navigationType: metric.navigationType
  });

  // Use sendBeacon for reliability
  if (navigator.sendBeacon) {
    navigator.sendBeacon('/analytics', body);
  } else {
    fetch('/analytics', { body, method: 'POST', keepalive: true });
  }
}

onCLS(sendToAnalytics);
onINP(sendToAnalytics);
onLCP(sendToAnalytics);
onFCP(sendToAnalytics);
onTTFB(sendToAnalytics);
```

---

## 3. Loading Performance (MANDATORY)

### A. Critical Rendering Path

```html
<!-- ✅ CORRECT: Optimized resource loading -->
<!DOCTYPE html>
<html>
<head>
  <!-- Critical CSS inline -->
  <style>
    /* Above-the-fold critical styles */
    body { margin: 0; font-family: system-ui; }
    .header { height: 60px; background: #fff; }
    .hero { min-height: 400px; }
  </style>

  <!-- Preload critical resources -->
  <link rel="preload" href="/fonts/main.woff2" as="font" type="font/woff2" crossorigin>
  <link rel="preload" href="/images/hero.webp" as="image">

  <!-- Preconnect to required origins -->
  <link rel="preconnect" href="https://api.example.com">
  <link rel="preconnect" href="https://fonts.googleapis.com">

  <!-- DNS prefetch for likely origins -->
  <link rel="dns-prefetch" href="https://analytics.example.com">

  <!-- Non-critical CSS loaded asynchronously -->
  <link rel="preload" href="/css/main.css" as="style" onload="this.onload=null;this.rel='stylesheet'">
  <noscript><link rel="stylesheet" href="/css/main.css"></noscript>
</head>
<body>
  <!-- Content... -->

  <!-- JavaScript at end or with defer -->
  <script src="/js/main.js" defer></script>
</body>
</html>
```

### B. Code Splitting

```typescript
// ✅ CORRECT: Route-based code splitting
import { lazy, Suspense } from 'react';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const Settings = lazy(() => import('./pages/Settings'));
const Reports = lazy(() =>
  import('./pages/Reports').then(module => ({
    default: module.Reports
  }))
);

function App() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <Routes>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/reports" element={<Reports />} />
      </Routes>
    </Suspense>
  );
}

// ✅ CORRECT: Component-level code splitting
const HeavyChart = lazy(() => import('./components/HeavyChart'));

function Analytics() {
  const [showChart, setShowChart] = useState(false);

  return (
    <div>
      <button onClick={() => setShowChart(true)}>Show Chart</button>
      {showChart && (
        <Suspense fallback={<ChartSkeleton />}>
          <HeavyChart />
        </Suspense>
      )}
    </div>
  );
}

// ✅ CORRECT: Prefetch on hover/focus
function NavLink({ to, children }) {
  const prefetch = () => {
    const link = document.createElement('link');
    link.rel = 'prefetch';
    link.href = to;
    document.head.appendChild(link);
  };

  return (
    <a
      href={to}
      onMouseEnter={prefetch}
      onFocus={prefetch}
    >
      {children}
    </a>
  );
}
```

### C. Resource Hints

```html
<!-- Preload: Critical resources for current page -->
<link rel="preload" href="/critical.js" as="script">
<link rel="preload" href="/hero.webp" as="image" type="image/webp">
<link rel="preload" href="/font.woff2" as="font" type="font/woff2" crossorigin>

<!-- Prefetch: Resources for likely next navigation -->
<link rel="prefetch" href="/next-page.js">
<link rel="prefetch" href="/next-page-data.json">

<!-- Prerender: Entire page for likely navigation -->
<link rel="prerender" href="/likely-next-page">

<!-- Preconnect: Establish early connection -->
<link rel="preconnect" href="https://api.example.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

<!-- DNS Prefetch: Resolve DNS early -->
<link rel="dns-prefetch" href="https://third-party.com">
```

---

## 4. Image Optimization (MANDATORY)

### A. Modern Image Formats

```html
<!-- ✅ CORRECT: Responsive images with modern formats -->
<picture>
  <!-- AVIF for browsers that support it -->
  <source
    type="image/avif"
    srcset="
      /images/hero-400.avif 400w,
      /images/hero-800.avif 800w,
      /images/hero-1200.avif 1200w
    "
    sizes="(max-width: 600px) 100vw, 50vw"
  >
  <!-- WebP fallback -->
  <source
    type="image/webp"
    srcset="
      /images/hero-400.webp 400w,
      /images/hero-800.webp 800w,
      /images/hero-1200.webp 1200w
    "
    sizes="(max-width: 600px) 100vw, 50vw"
  >
  <!-- JPEG fallback -->
  <img
    src="/images/hero-800.jpg"
    srcset="
      /images/hero-400.jpg 400w,
      /images/hero-800.jpg 800w,
      /images/hero-1200.jpg 1200w
    "
    sizes="(max-width: 600px) 100vw, 50vw"
    alt="Hero image description"
    loading="lazy"
    decoding="async"
    width="1200"
    height="600"
  >
</picture>
```

### B. Lazy Loading

```tsx
// ✅ CORRECT: Native lazy loading
<img
  src="/image.jpg"
  loading="lazy"
  decoding="async"
  alt="Description"
  width="800"
  height="600"
/>

// ✅ CORRECT: Intersection Observer for advanced lazy loading
function LazyImage({ src, alt, placeholder }) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isInView, setIsInView] = useState(false);
  const imgRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          observer.disconnect();
        }
      },
      { rootMargin: '200px' } // Start loading 200px before viewport
    );

    if (imgRef.current) {
      observer.observe(imgRef.current);
    }

    return () => observer.disconnect();
  }, []);

  return (
    <div ref={imgRef} className="image-container">
      {placeholder && !isLoaded && (
        <img src={placeholder} alt="" className="placeholder" />
      )}
      {isInView && (
        <img
          src={src}
          alt={alt}
          onLoad={() => setIsLoaded(true)}
          className={isLoaded ? 'loaded' : 'loading'}
        />
      )}
    </div>
  );
}
```

### C. Image Dimensions

```html
<!-- ✅ CORRECT: Always specify dimensions to prevent CLS -->
<img src="/image.jpg" width="800" height="600" alt="Description">

<!-- ✅ CORRECT: Aspect ratio for responsive images -->
<style>
.image-container {
  aspect-ratio: 16 / 9;
  width: 100%;
}

.image-container img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
</style>

<!-- ❌ WRONG: No dimensions causes layout shift -->
<img src="/image.jpg" alt="Description">
```

---

## 5. JavaScript Performance (MANDATORY)

### A. Bundle Optimization

```javascript
// webpack.config.js
module.exports = {
  optimization: {
    splitChunks: {
      chunks: 'all',
      cacheGroups: {
        // Separate vendor chunks
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name(module) {
            const packageName = module.context.match(
              /[\\/]node_modules[\\/](.*?)([\\/]|$)/
            )[1];
            return `vendor.${packageName.replace('@', '')}`;
          },
        },
        // Common chunks used across pages
        common: {
          minChunks: 2,
          priority: -10,
          reuseExistingChunk: true,
        },
      },
    },
    // Enable tree shaking
    usedExports: true,
    sideEffects: true,
  },
  // Minification
  mode: 'production',
};

// package.json - Mark side-effect free
{
  "sideEffects": [
    "*.css",
    "*.scss"
  ]
}
```

### B. Avoid Main Thread Blocking

```typescript
// ✅ CORRECT: Break up long tasks
async function processLargeArray(items: Item[]) {
  const CHUNK_SIZE = 100;

  for (let i = 0; i < items.length; i += CHUNK_SIZE) {
    const chunk = items.slice(i, i + CHUNK_SIZE);

    // Process chunk
    chunk.forEach(item => processItem(item));

    // Yield to main thread between chunks
    await new Promise(resolve => setTimeout(resolve, 0));

    // Or use requestIdleCallback for non-urgent work
    // await new Promise(resolve => requestIdleCallback(resolve));
  }
}

// ✅ CORRECT: Use Web Workers for heavy computation
// worker.ts
self.onmessage = (e) => {
  const result = heavyComputation(e.data);
  self.postMessage(result);
};

// main.ts
const worker = new Worker(new URL('./worker.ts', import.meta.url));

worker.onmessage = (e) => {
  console.log('Result:', e.data);
};

worker.postMessage(data);

// ✅ CORRECT: Debounce expensive operations
function debounce<T extends (...args: any[]) => void>(
  fn: T,
  delay: number
): T {
  let timeoutId: NodeJS.Timeout;

  return ((...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  }) as T;
}

const handleSearch = debounce((query: string) => {
  performSearch(query);
}, 300);
```

### C. Efficient Event Handling

```typescript
// ✅ CORRECT: Event delegation
document.getElementById('list')?.addEventListener('click', (e) => {
  const target = e.target as HTMLElement;
  const listItem = target.closest('li');

  if (listItem) {
    handleItemClick(listItem.dataset.id);
  }
});

// ✅ CORRECT: Passive event listeners for scroll/touch
window.addEventListener('scroll', handleScroll, { passive: true });
element.addEventListener('touchstart', handleTouch, { passive: true });

// ✅ CORRECT: Throttle scroll handlers
function throttle<T extends (...args: any[]) => void>(
  fn: T,
  limit: number
): T {
  let inThrottle = false;

  return ((...args: Parameters<T>) => {
    if (!inThrottle) {
      fn(...args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  }) as T;
}

const handleScroll = throttle(() => {
  // Scroll handling logic
}, 100);
```

---

## 6. CSS Performance (MANDATORY)

### A. Critical CSS

```javascript
// Extract critical CSS with critical package
const critical = require('critical');

critical.generate({
  base: 'dist/',
  src: 'index.html',
  target: {
    html: 'index-critical.html',
    css: 'critical.css',
  },
  width: 1300,
  height: 900,
  inline: true,
});
```

### B. Efficient Selectors

```css
/* ✅ CORRECT: Simple, efficient selectors */
.button { }
.nav-item { }
.card-title { }

/* ❌ WRONG: Overly specific selectors */
body div.container ul.nav li.nav-item a.nav-link { }

/* ❌ WRONG: Universal selector in key position */
.container * { }

/* ✅ CORRECT: Avoid expensive properties in animations */
.animate {
  /* Use transform and opacity for smooth animations */
  transform: translateX(100px);
  opacity: 0.5;
}

/* ❌ WRONG: Animating expensive properties */
.animate {
  left: 100px;  /* Triggers layout */
  width: 200px; /* Triggers layout */
  box-shadow: 0 0 10px rgba(0,0,0,0.5); /* Triggers paint */
}
```

### C. Contain and Content-Visibility

```css
/* ✅ CORRECT: Use CSS containment */
.card {
  contain: layout style paint;
}

/* ✅ CORRECT: Content-visibility for off-screen content */
.below-fold-section {
  content-visibility: auto;
  contain-intrinsic-size: 0 500px; /* Estimated height */
}

/* ✅ CORRECT: Will-change for known animations */
.will-animate {
  will-change: transform;
}

/* Remove will-change after animation */
.will-animate.done {
  will-change: auto;
}
```

---

## 7. Network Optimization (MANDATORY)

### A. HTTP/2 and HTTP/3

```nginx
# Enable HTTP/2
server {
    listen 443 ssl http2;

    # Enable HTTP/3 (QUIC)
    listen 443 quic reuseport;
    add_header Alt-Svc 'h3=":443"; ma=86400';

    # Enable server push (use sparingly)
    http2_push /css/critical.css;
    http2_push /js/main.js;
}
```

### B. Caching Strategy

```nginx
# nginx caching configuration
location /static/ {
    # Immutable assets with hash in filename
    location ~* \.[a-f0-9]{8,}\.(js|css|woff2)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Images
    location ~* \.(jpg|jpeg|png|webp|avif|gif|svg|ico)$ {
        expires 30d;
        add_header Cache-Control "public";
    }

    # HTML - no cache
    location ~* \.html$ {
        expires -1;
        add_header Cache-Control "no-store, must-revalidate";
    }
}
```

```javascript
// Service Worker caching
const CACHE_NAME = 'app-v1';
const STATIC_ASSETS = [
  '/',
  '/css/main.css',
  '/js/main.js',
  '/images/logo.svg'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(STATIC_ASSETS))
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
  );
});
```

### C. Compression

```nginx
# Enable Brotli (preferred) and gzip compression
brotli on;
brotli_comp_level 6;
brotli_types text/plain text/css application/json application/javascript text/xml application/xml;

gzip on;
gzip_comp_level 6;
gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
```

---

## 8. Font Performance (MANDATORY)

### A. Font Loading Strategy

```css
/* ✅ CORRECT: Font display strategy */
@font-face {
  font-family: 'CustomFont';
  src: url('/fonts/custom.woff2') format('woff2');
  font-display: swap; /* Show fallback immediately, swap when loaded */
  font-weight: 400;
  font-style: normal;
}

/* For critical text, use optional to avoid layout shift */
@font-face {
  font-family: 'HeadingFont';
  src: url('/fonts/heading.woff2') format('woff2');
  font-display: optional; /* Use only if already cached */
}
```

### B. Subset Fonts

```bash
# Subset fonts to include only needed characters
pyftsubset font.ttf \
  --output-file=font-subset.woff2 \
  --flavor=woff2 \
  --layout-features='*' \
  --unicodes="U+0000-00FF,U+2000-206F"
```

### C. System Font Stack

```css
/* ✅ CORRECT: Use system fonts for better performance */
body {
  font-family:
    -apple-system,
    BlinkMacSystemFont,
    'Segoe UI',
    Roboto,
    Oxygen,
    Ubuntu,
    Cantarell,
    'Fira Sans',
    'Droid Sans',
    'Helvetica Neue',
    sans-serif;
}

code {
  font-family:
    ui-monospace,
    SFMono-Regular,
    SF Mono,
    Menlo,
    Consolas,
    Liberation Mono,
    monospace;
}
```

---

## 9. Monitoring and Budgets (MANDATORY)

### A. Performance Budgets

```json
// bundlesize configuration
{
  "files": [
    {
      "path": "dist/js/main.*.js",
      "maxSize": "150 kB"
    },
    {
      "path": "dist/js/vendor.*.js",
      "maxSize": "250 kB"
    },
    {
      "path": "dist/css/main.*.css",
      "maxSize": "50 kB"
    }
  ]
}
```

```javascript
// Lighthouse CI configuration
module.exports = {
  ci: {
    collect: {
      numberOfRuns: 3,
      url: ['http://localhost:3000/'],
    },
    assert: {
      assertions: {
        'categories:performance': ['error', { minScore: 0.9 }],
        'first-contentful-paint': ['error', { maxNumericValue: 2000 }],
        'largest-contentful-paint': ['error', { maxNumericValue: 2500 }],
        'cumulative-layout-shift': ['error', { maxNumericValue: 0.1 }],
        'total-blocking-time': ['error', { maxNumericValue: 300 }],
      },
    },
    upload: {
      target: 'lhci',
      serverBaseUrl: 'https://lhci.example.com',
    },
  },
};
```

### B. Real User Monitoring

```typescript
// Performance observer for real user metrics
const observer = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    // Send to analytics
    sendMetric({
      name: entry.name,
      value: entry.startTime,
      type: entry.entryType
    });
  }
});

observer.observe({
  entryTypes: ['navigation', 'resource', 'paint', 'largest-contentful-paint']
});

// Long task monitoring
const longTaskObserver = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    if (entry.duration > 50) {
      console.warn('Long task detected:', entry.duration, 'ms');
      sendMetric({
        name: 'long-task',
        value: entry.duration
      });
    }
  }
});

longTaskObserver.observe({ entryTypes: ['longtask'] });
```

---

## 10. Deployment Checklist

### Build
- [ ] JavaScript minified and tree-shaken
- [ ] CSS minified and purged
- [ ] Images optimized (WebP/AVIF)
- [ ] Fonts subsetted
- [ ] Source maps generated (not deployed)

### Loading
- [ ] Critical CSS inlined
- [ ] Resources preloaded/prefetched
- [ ] Code split by route
- [ ] Third-party scripts deferred

### Caching
- [ ] Cache headers configured
- [ ] Service worker implemented
- [ ] Asset filenames hashed

### Monitoring
- [ ] Core Web Vitals tracked
- [ ] Performance budgets enforced
- [ ] RUM configured
- [ ] Alerts set up

---

## 11. Quick Reference

```html
<!-- Resource hints -->
<link rel="preload" href="..." as="...">
<link rel="prefetch" href="...">
<link rel="preconnect" href="...">
<link rel="dns-prefetch" href="...">

<!-- Image optimization -->
<img loading="lazy" decoding="async" width="..." height="...">
<picture><source type="image/avif"><source type="image/webp"><img></picture>

<!-- Script loading -->
<script defer src="..."></script>
<script async src="..."></script>
<script type="module" src="..."></script>
```

```css
/* Performance CSS */
content-visibility: auto;
contain: layout style paint;
will-change: transform;
font-display: swap;
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Performance Team
