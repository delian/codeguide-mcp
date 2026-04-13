# Web Performance Guidelines
Mandatory standards for building fast, efficient web applications. Lighthouse, WebPageTest, Chrome DevTools, bundlesize, webpack-bundle-analyzer.

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

## 9. Backend Performance Patterns (MANDATORY)

### A. Caching Strategies

```python
# Python: Multi-layer caching pattern
import functools
import hashlib
import json
import time
from typing import Optional, Any
import redis

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

class MultiLayerCache:
    """L1: In-process memory, L2: Redis, L3: Database."""

    def __init__(self):
        self._local_cache: dict[str, tuple[Any, float]] = {}  # key -> (value, expiry)
        self._local_max_size = 1000

    def get(self, key: str) -> Optional[Any]:
        # L1: Check local memory (fastest)
        if key in self._local_cache:
            value, expiry = self._local_cache[key]
            if time.time() < expiry:
                return value
            del self._local_cache[key]

        # L2: Check Redis
        cached = redis_client.get(f"cache:{key}")
        if cached is not None:
            value = json.loads(cached)
            # Promote to L1
            self._set_local(key, value, ttl=60)
            return value

        return None

    def set(self, key: str, value: Any, ttl: int = 300):
        # Set in both layers
        self._set_local(key, value, ttl=min(ttl, 60))  # L1: shorter TTL
        redis_client.setex(f"cache:{key}", ttl, json.dumps(value))

    def _set_local(self, key: str, value: Any, ttl: int):
        if len(self._local_cache) >= self._local_max_size:
            # Evict expired entries first, then oldest
            now = time.time()
            self._local_cache = {
                k: (v, e) for k, (v, e) in self._local_cache.items() if e > now
            }
        self._local_cache[key] = (value, time.time() + ttl)

    def invalidate(self, key: str):
        self._local_cache.pop(key, None)
        redis_client.delete(f"cache:{key}")

cache = MultiLayerCache()

# Cache-aside pattern decorator
def cached(ttl: int = 300, key_prefix: str = ""):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{key_prefix}{func.__name__}:{hashlib.md5(json.dumps({'a': args[1:], 'k': kwargs}, sort_keys=True, default=str).encode()).hexdigest()}"

            result = cache.get(cache_key)
            if result is not None:
                return result

            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            return result
        return wrapper
    return decorator

# Usage
class ProductService:
    @cached(ttl=600, key_prefix="products:")
    async def get_product(self, product_id: str) -> dict:
        return await self.db.fetch_one("SELECT * FROM products WHERE id = $1", product_id)

    async def update_product(self, product_id: str, data: dict):
        await self.db.execute("UPDATE products SET ... WHERE id = $1", product_id)
        cache.invalidate(f"products:get_product:{product_id}")  # Invalidate on write
```

### B. Connection Pooling

```python
# Python: Database connection pool with health checks
import asyncpg
import asyncio
from contextlib import asynccontextmanager

class DatabasePool:
    """Properly configured connection pool with monitoring."""

    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        self.pool = await asyncpg.create_pool(
            self.dsn,
            min_size=5,          # Keep 5 connections warm
            max_size=20,         # Never exceed 20 connections
            max_inactive_connection_lifetime=300,  # Close idle after 5 min
            command_timeout=30,  # Query timeout
            statement_cache_size=100,  # Cache prepared statements
        )

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection with timeout and error handling."""
        try:
            async with self.pool.acquire(timeout=5) as conn:
                yield conn
        except asyncpg.exceptions.TooManyConnectionsError:
            # Log and raise a clear error
            logger.error("connection_pool_exhausted",
                pool_size=self.pool.get_size(),
                free_size=self.pool.get_idle_size(),
            )
            raise
        except asyncio.TimeoutError:
            logger.error("connection_pool_timeout",
                pool_size=self.pool.get_size(),
                free_size=self.pool.get_idle_size(),
            )
            raise

    async def health_check(self) -> bool:
        try:
            async with self.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False
```

```go
// Go: HTTP client connection pool tuning
package main

import (
    "net"
    "net/http"
    "time"
)

func newHTTPClient() *http.Client {
    transport := &http.Transport{
        // Connection pool settings
        MaxIdleConns:        100,              // Total idle connections
        MaxIdleConnsPerHost: 10,               // Per-host idle connections
        MaxConnsPerHost:     50,               // Max connections per host
        IdleConnTimeout:     90 * time.Second, // Close idle after 90s

        // Timeouts for connection establishment
        DialContext: (&net.Dialer{
            Timeout:   5 * time.Second,  // TCP connect timeout
            KeepAlive: 30 * time.Second, // TCP keepalive interval
        }).DialContext,

        TLSHandshakeTimeout:   5 * time.Second,
        ResponseHeaderTimeout: 10 * time.Second,
        ExpectContinueTimeout: 1 * time.Second,

        // Enable HTTP/2
        ForceAttemptHTTP2: true,
    }

    return &http.Client{
        Transport: transport,
        Timeout:   30 * time.Second, // Overall request timeout
    }
}

// IMPORTANT: Reuse the client, do NOT create one per request
var httpClient = newHTTPClient()
```

### C. Query Optimization Patterns

```sql
-- Common query anti-patterns and their fixes

-- ❌ WRONG: SELECT * when you only need specific columns
SELECT * FROM orders WHERE user_id = 123;

-- ✅ CORRECT: Select only needed columns
SELECT id, status, total, created_at FROM orders WHERE user_id = 123;


-- ❌ WRONG: N+1 query pattern
-- Code: for order in orders: get_items(order.id)
SELECT * FROM orders WHERE user_id = 123;
SELECT * FROM order_items WHERE order_id = 1;
SELECT * FROM order_items WHERE order_id = 2;
-- ... repeats N times

-- ✅ CORRECT: Single JOIN or batched query
SELECT o.id, o.status, o.total, oi.product_id, oi.quantity
FROM orders o
JOIN order_items oi ON o.id = oi.order_id
WHERE o.user_id = 123;


-- ❌ WRONG: Missing index on frequently filtered/joined columns
SELECT * FROM orders WHERE status = 'pending' AND created_at > NOW() - INTERVAL '1 day';

-- ✅ CORRECT: Add composite index matching query pattern
CREATE INDEX idx_orders_status_created ON orders(status, created_at);


-- ❌ WRONG: Using OFFSET for pagination (scans all skipped rows)
SELECT * FROM products ORDER BY created_at DESC LIMIT 20 OFFSET 10000;

-- ✅ CORRECT: Keyset/cursor pagination (constant performance)
SELECT * FROM products
WHERE created_at < '2024-01-15T10:30:00Z'
ORDER BY created_at DESC
LIMIT 20;


-- ❌ WRONG: Counting all rows for "has any" check
SELECT COUNT(*) FROM notifications WHERE user_id = 123 AND read = false;

-- ✅ CORRECT: EXISTS is faster when you only need boolean
SELECT EXISTS(SELECT 1 FROM notifications WHERE user_id = 123 AND read = false);


-- ❌ WRONG: OR conditions that prevent index usage
SELECT * FROM users WHERE email = 'a@b.com' OR phone = '555-1234';

-- ✅ CORRECT: UNION ALL uses indexes on both columns
SELECT * FROM users WHERE email = 'a@b.com'
UNION ALL
SELECT * FROM users WHERE phone = '555-1234' AND email != 'a@b.com';
```

### D. Memory Profiling Techniques

```python
# Python: Memory profiling with tracemalloc
import tracemalloc
import linecache

def profile_memory(func):
    """Decorator to profile memory usage of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()

        result = func(*args, **kwargs)

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        print(f"\n--- Memory profile for {func.__name__} ---")
        print(f"Top 10 memory allocations:")
        for stat in top_stats[:10]:
            print(f"  {stat}")

        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory: {current / 1024:.1f} KB")
        print(f"Peak memory: {peak / 1024:.1f} KB")

        tracemalloc.stop()
        return result
    return wrapper

# Usage
@profile_memory
def process_large_dataset():
    # ❌ WRONG: Loading entire dataset into memory
    # data = db.fetch_all("SELECT * FROM events")  # Could be millions of rows

    # ✅ CORRECT: Process in chunks using a generator
    for chunk in db.fetch_chunks("SELECT * FROM events", chunk_size=1000):
        for record in chunk:
            process_record(record)
```

```javascript
// Node.js: Memory leak detection
// Run with: node --inspect app.js
// Then use Chrome DevTools to take heap snapshots

// Common memory leak patterns and fixes:

// ❌ WRONG: Unbounded event listener accumulation
class Leaky {
  constructor() {
    // Each instance adds a listener, never removed
    process.on('message', this.handleMessage.bind(this));
  }
}

// ✅ CORRECT: Clean up listeners
class NotLeaky {
  constructor() {
    this._handler = this.handleMessage.bind(this);
    process.on('message', this._handler);
  }
  destroy() {
    process.removeListener('message', this._handler);
  }
}

// ❌ WRONG: Unbounded cache without eviction
const cache = new Map(); // Grows forever!
function lookup(key) {
  if (!cache.has(key)) {
    cache.set(key, expensiveComputation(key));
  }
  return cache.get(key);
}

// ✅ CORRECT: LRU cache with max size
class LRUCache {
  constructor(maxSize = 1000) {
    this.maxSize = maxSize;
    this.cache = new Map();
  }

  get(key) {
    if (!this.cache.has(key)) return undefined;
    // Move to end (most recently used)
    const value = this.cache.get(key);
    this.cache.delete(key);
    this.cache.set(key, value);
    return value;
  }

  set(key, value) {
    if (this.cache.has(key)) this.cache.delete(key);
    this.cache.set(key, value);
    if (this.cache.size > this.maxSize) {
      // Delete oldest entry (first key)
      const oldestKey = this.cache.keys().next().value;
      this.cache.delete(oldestKey);
    }
  }
}
```

### E. Load Testing with k6

```javascript
// k6 load test script: load-test.js
// Run: k6 run --vus 50 --duration 5m load-test.js
import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const orderDuration = new Trend('order_processing_time');

export const options = {
  // Ramp-up pattern
  stages: [
    { duration: '1m', target: 10 },   // Warm up
    { duration: '3m', target: 50 },   // Normal load
    { duration: '2m', target: 100 },  // Peak load
    { duration: '1m', target: 0 },    // Cool down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],  // 95% under 500ms
    http_req_failed: ['rate<0.01'],                    // Error rate under 1%
    errors: ['rate<0.05'],                             // Custom error rate
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3000';

export default function () {
  group('Browse Products', () => {
    const res = http.get(`${BASE_URL}/api/products`);
    check(res, {
      'status is 200': (r) => r.status === 200,
      'response time < 200ms': (r) => r.timings.duration < 200,
      'has products': (r) => JSON.parse(r.body).length > 0,
    });
    errorRate.add(res.status !== 200);
  });

  sleep(1); // Think time between actions

  group('Place Order', () => {
    const payload = JSON.stringify({
      productId: 'PROD-001',
      quantity: 1,
    });

    const params = {
      headers: { 'Content-Type': 'application/json' },
    };

    const start = Date.now();
    const res = http.post(`${BASE_URL}/api/orders`, payload, params);
    orderDuration.add(Date.now() - start);

    check(res, {
      'order created': (r) => r.status === 201,
      'has order id': (r) => JSON.parse(r.body).orderId !== undefined,
    });
    errorRate.add(res.status !== 201);
  });

  sleep(Math.random() * 3); // Random think time 0-3s
}
```

### F. Common Performance Anti-Patterns

```yaml
anti_patterns:
  synchronous_external_calls:
    problem: "Calling external APIs sequentially when they are independent"
    impact: "Total latency = sum of all call latencies"
    fix: "Use Promise.all / asyncio.gather for independent calls"
    example_bad: |
      const user = await getUser(id);       // 100ms
      const orders = await getOrders(id);   // 150ms
      const prefs = await getPreferences(id); // 80ms
      // Total: 330ms
    example_good: |
      const [user, orders, prefs] = await Promise.all([
        getUser(id),         // 100ms
        getOrders(id),       // 150ms
        getPreferences(id),  // 80ms
      ]);
      // Total: 150ms (max of all three)

  missing_database_indexes:
    problem: "Full table scans on large tables"
    impact: "Query time grows linearly with table size"
    fix: "Add indexes for columns used in WHERE, JOIN, ORDER BY"
    detection: |
      -- PostgreSQL: Find slow queries
      SELECT query, mean_exec_time, calls
      FROM pg_stat_statements
      ORDER BY mean_exec_time DESC
      LIMIT 20;

      -- Find missing indexes
      SELECT relname, seq_scan, seq_tup_read,
             idx_scan, idx_tup_fetch
      FROM pg_stat_user_tables
      WHERE seq_scan > 1000
      ORDER BY seq_tup_read DESC;

  unbatched_operations:
    problem: "Inserting/updating records one at a time in a loop"
    impact: "1000 individual INSERTs vs 1 batch INSERT: 50x slower"
    fix: "Use batch/bulk operations"
    example_bad: |
      for item in items:
          db.execute("INSERT INTO events (data) VALUES ($1)", item)
    example_good: |
      db.executemany("INSERT INTO events (data) VALUES ($1)", items)
      # Or better: COPY for PostgreSQL bulk inserts

  no_pagination:
    problem: "Returning unbounded result sets from APIs"
    impact: "Memory exhaustion, slow responses, network timeouts"
    fix: "Always paginate with a max page size"

  serializing_too_much:
    problem: "Converting entire ORM objects to JSON including all relations"
    impact: "Excessive memory, CPU, and bandwidth usage"
    fix: "Use explicit serialization schemas / DTOs with only needed fields"
```

---

## 10. Performance Budgets (MANDATORY)

### A. Defining and Enforcing Budgets

```yaml
# performance-budget.yml
# Define budgets for different page types

budgets:
  homepage:
    lcp: 2500           # ms
    inp: 200             # ms
    cls: 0.1
    total_js: 200        # KB (compressed)
    total_css: 60        # KB (compressed)
    total_images: 500    # KB
    total_requests: 30
    total_weight: 1000   # KB
    time_to_interactive: 3500  # ms

  product_page:
    lcp: 2000
    inp: 150
    cls: 0.05
    total_js: 180
    total_css: 50
    total_images: 800
    total_weight: 1200

  checkout:
    lcp: 1500            # Fastest for conversion-critical pages
    inp: 100
    cls: 0.02
    total_js: 150
    total_css: 40
    total_images: 200
    total_weight: 600

  api_endpoints:
    p50_latency: 50      # ms
    p95_latency: 200     # ms
    p99_latency: 500     # ms
    error_rate: 0.1      # percent
    max_response_size: 500  # KB
```

```javascript
// Enforce budgets in CI with Lighthouse CI
// lighthouserc.js
module.exports = {
  ci: {
    collect: {
      numberOfRuns: 5,  // Multiple runs for stability
      url: [
        'http://localhost:3000/',
        'http://localhost:3000/products/1',
        'http://localhost:3000/checkout',
      ],
      settings: {
        preset: 'desktop',
      },
    },
    assert: {
      assertions: {
        'categories:performance': ['error', { minScore: 0.9 }],
        'largest-contentful-paint': ['error', { maxNumericValue: 2500 }],
        'cumulative-layout-shift': ['error', { maxNumericValue: 0.1 }],
        'interactive': ['error', { maxNumericValue: 3500 }],
        'total-byte-weight': ['warning', { maxNumericValue: 1000000 }],
        'mainthread-work-breakdown': ['warning', { maxNumericValue: 4000 }],
        'dom-size': ['warning', { maxNumericValue: 1500 }],
        'resource-summary:script:size': ['error', { maxNumericValue: 200000 }],
      },
    },
  },
};
```

---

## 11. Frontend Monitoring and Budgets (MANDATORY)

### A. Frontend Performance Budgets

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

## 12. Deployment Checklist

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

### Backend
- [ ] Database queries optimized (no N+1, proper indexes)
- [ ] Connection pooling configured
- [ ] Caching strategy implemented (application + HTTP)
- [ ] Pagination on all list endpoints
- [ ] Load tested with realistic traffic patterns

### Monitoring
- [ ] Core Web Vitals tracked
- [ ] Performance budgets enforced in CI
- [ ] RUM configured
- [ ] Alerts set up
- [ ] Backend latency percentiles tracked (p50, p95, p99)

---

## 13. Quick Reference

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

## 14. Why This Configuration Works

- **Core Web Vitals alignment with real user experience**: Optimizing for LCP, INP, and CLS ensures improvements target what users actually perceive rather than synthetic benchmarks. These metrics directly correlate with user satisfaction, engagement, and conversion rates.
- **Performance budgets prevent gradual degradation**: Enforcing bundle size limits and Lighthouse score thresholds in CI/CD catches performance regressions at the pull request stage, before they accumulate into a slow application that is expensive to optimize retroactively.
- **Progressive loading prioritizes perceived speed**: Techniques like critical CSS inlining, resource hints, and code splitting ensure users see meaningful content as quickly as possible, even while the full application continues loading in the background.
- **Image optimization delivers the largest payload savings**: Images are typically the heaviest assets on a page. Modern formats (AVIF, WebP), responsive srcsets, and lazy loading together can reduce image transfer sizes by 50-80%, producing the single largest performance improvement for most sites.
- **Caching strategy minimizes redundant transfers**: Content-hash-based immutable caching for static assets combined with service worker caching eliminates repeat downloads for returning visitors, reducing both load times and server costs.

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Performance Team


**End of Web Performance Guidelines**
