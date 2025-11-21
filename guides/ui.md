# User Interface Development Guidelines

This document provides mandatory standards and best practices for building clean, performant, and consistent user interfaces across web, desktop, and mobile platforms.

---

**Agent Profile**: The UI/UX Expert
**Role**: Senior User Interface Architect & Performance Specialist
**Objective**: Generate production-ready, accessible, performant, and consistent user interfaces.
**Tools**: UI frameworks (React, Vue, Angular, Flutter, SwiftUI, Jetpack Compose), testing frameworks, accessibility validators, performance profilers.

---

## 1. Core Philosophies: CLEAN-FIRST

The agent must adhere to the **CLEAN-FIRST** principles for every UI implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **C**onsistent: Uniform look, feel, and behavior across all screens and components
- **L**ightweight: Minimal memory footprint, lazy loading, virtualized rendering
- **E**fficient: Minimize data transfers, prefer server-side operations for large datasets
- **A**ccessible: WCAG 2.1+ compliance, keyboard navigation, screen reader support
- **N**ative-Preferred: Use native or standard widgets when possible for performance and familiarity

**Additional Principles:**

- **State Preservation**: Maintain UI state during navigation, page changes, and optionally reloads
- **Real-Time Communication**: Prefer bidirectional protocols (Socket.IO > WebSocket > GraphQL subscriptions > REST polling)
- **Progressive Enhancement**: Core functionality works without JavaScript; enhance progressively
- **Responsive Design**: Single codebase adapts to all screen sizes and orientations

**Verified UI**: Agent-generated UI code MUST pass visual regression tests, accessibility audits, and performance benchmarks before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated UI code meets quality standards before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY UI code, the agent MUST:**

1. **Accessibility Check**:
   ```bash
   # Run accessibility audit
   # Web: axe-core, pa11y, lighthouse
   npx lighthouse --only-categories=accessibility --output=json

   # Native: Platform-specific accessibility scanner
   # iOS: Accessibility Inspector
   # Android: Accessibility Scanner
   ```
   - **MUST** pass WCAG 2.1 Level AA
   - All interactive elements keyboard-accessible
   - Proper ARIA labels and roles

2. **Performance Check**:
   ```bash
   # Web performance audit
   npx lighthouse --only-categories=performance --output=json

   # Bundle size analysis
   # Adapt to your bundler: webpack-bundle-analyzer, vite-bundle-visualizer
   npm run build -- --analyze
   ```
   - **MUST** achieve Lighthouse performance score > 90
   - First Contentful Paint < 1.5s
   - No memory leaks in long-running sessions

3. **Visual Consistency Check**:
   ```bash
   # Run visual regression tests
   # Tools: Chromatic, Percy, BackstopJS, Playwright screenshots
   npm run test:visual
   ```
   - Components match design system
   - Consistent spacing, typography, colors

4. **Component Tests**:
   ```bash
   # Run component tests
   npm test -- --coverage

   # E2E tests for critical flows
   npx playwright test
   ```
   - All components have unit tests
   - Critical user flows have E2E tests

#### Error Correction Process

If verification fails:

1. **Accessibility Failures**:
   - Identify failing WCAG criteria
   - Add missing ARIA attributes
   - Ensure focus management
   - Re-run accessibility audit

2. **Performance Failures**:
   - Profile to identify bottlenecks
   - Implement virtualization for large lists
   - Add lazy loading for off-screen content
   - Optimize bundle size

3. **Visual Inconsistencies**:
   - Compare against design tokens
   - Verify component hierarchy
   - Check responsive breakpoints

### B. Prohibited Practices

**NEVER deliver UI code that:**
- [ ] Fails accessibility audits
- [ ] Has Lighthouse performance score < 80
- [ ] Lacks tests for interactive components
- [ ] Uses inline styles instead of design tokens
- [ ] Fetches entire datasets for paginated views
- [ ] Polls server continuously when real-time updates aren't needed
- [ ] Breaks keyboard navigation
- [ ] Loses state on back/forward navigation
- [ ] Has memory leaks (uncleaned subscriptions, event listeners)
- [ ] **Fixes bugs without adding regression tests first**
- [ ] **Writes implementation before writing tests (violates TDD)**

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new UI components.**

### TDD Cycle

```
1. RED: Write a failing test first
   ↓
2. GREEN: Write minimal code to make it pass
   ↓
3. REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for UI Components

```javascript
// Step 1: RED - Write failing test first
// tests/components/DataGrid.test.js

describe('DataGrid', () => {
  it('should render loading state while fetching data', () => {
    render(<DataGrid loading={true} data={[]} />);
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  it('should preserve filter state on page navigation', async () => {
    const { rerender } = render(<DataGrid filters={{ status: 'active' }} />);
    // Navigate away and back
    rerender(<DataGrid filters={{ status: 'active' }} />);
    expect(screen.getByDisplayValue('active')).toBeInTheDocument();
  });
});

// Run: npm test
// FAILS - DataGrid doesn't exist yet

// Step 2: GREEN - Write minimal implementation
// src/components/DataGrid.jsx

export function DataGrid({ loading, data, filters }) {
  if (loading) {
    return <div role="progressbar">Loading...</div>;
  }
  return (
    <table>
      <FilterBar filters={filters} />
      {/* Minimal implementation */}
    </table>
  );
}

// Run: npm test
// PASSES - tests pass

// Step 3: REFACTOR - Improve with proper structure
// Add virtualization, accessibility, proper styling
// Tests still pass
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every UI bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. Bug Reported/Discovered
   ↓
2. Write a test that REPRODUCES the bug (test will FAIL)
   ↓
3. Verify the test fails for the right reason
   ↓
4. Fix the bug (make the test pass)
   ↓
5. Run visual regression to ensure no side effects
   ↓
6. Document the bug in test comments
```

### Example Bug Fix

```javascript
// Bug Report #UI-042: Grid loses scroll position after data refresh

// Step 1-2: Write test that reproduces the bug
describe('DataGrid scroll position', () => {
  // BUG-UI-042: Grid loses scroll position after data refresh
  // Regression test added: 2026-01-22
  it('should preserve scroll position after data refresh', async () => {
    const { rerender } = render(<DataGrid data={initialData} />);

    // Scroll to middle of grid
    const grid = screen.getByRole('grid');
    fireEvent.scroll(grid, { target: { scrollTop: 500 } });

    // Refresh data
    rerender(<DataGrid data={updatedData} />);

    // Scroll position should be preserved
    expect(grid.scrollTop).toBe(500);
  });
});

// Run: npm test
// FAILS - scroll position resets to 0

// Step 3: Fix the bug
// Save scroll position before update, restore after
function DataGrid({ data }) {
  const scrollRef = useRef(0);
  const gridRef = useRef(null);

  // FIX(BUG-UI-042): Preserve scroll position on data refresh
  useLayoutEffect(() => {
    if (gridRef.current) {
      gridRef.current.scrollTop = scrollRef.current;
    }
  }, [data]);

  const handleScroll = (e) => {
    scrollRef.current = e.target.scrollTop;
  };

  return <div ref={gridRef} onScroll={handleScroll}>...</div>;
}

// Run: npm test
// PASSES - bug fixed, regression prevented
```

---

## 3. Platform-Specific Guidelines

### A. Web Applications (Browser-Based)

**Framework Options:**
- React, Vue, Angular, Svelte, Solid
- Widget Libraries: Material UI, Ant Design, Chakra UI, Radix, shadcn/ui

```
Browser UI Architecture
┌─────────────────────────────────────────────────────────────┐
│                     Application Shell                        │
├──────────────────┬──────────────────────────────────────────┤
│                  │                                           │
│    Navigation    │            Content Area                   │
│    (persistent)  │         (route-based)                    │
│                  │                                           │
│    ┌──────────┐  │    ┌────────────────────────────────┐    │
│    │ Menu     │  │    │  Page Component                │    │
│    │ Items    │  │    │  ┌────────────────────────┐    │    │
│    │          │  │    │  │ Data Grid / Content    │    │    │
│    │          │  │    │  │ (virtualized)          │    │    │
│    └──────────┘  │    │  └────────────────────────┘    │    │
│                  │    └────────────────────────────────┘    │
└──────────────────┴──────────────────────────────────────────┘
```

**Requirements:**
- Lazy load routes and heavy components
- Use service workers for offline capability
- Implement proper CSP headers
- Support browser back/forward navigation

### B. Desktop Applications

**Framework Options:**
- Electron, Tauri, Qt, GTK, .NET MAUI, SwiftUI (macOS)
- Native: Win32/WinUI, Cocoa/AppKit

**Requirements:**
- Follow platform HIG (Human Interface Guidelines)
- Support system themes (light/dark mode)
- Respect system accessibility settings
- Use native file dialogs and notifications
- Handle window state persistence

### C. Mobile Applications

**Framework Options:**
- Flutter, React Native, SwiftUI (iOS), Jetpack Compose (Android)
- Native: UIKit, Android Views

**Requirements:**
- Support gesture navigation
- Handle safe areas (notches, home indicators)
- Implement pull-to-refresh where appropriate
- Respect platform navigation patterns
- Optimize for battery and memory constraints

---

## 4. Widget & Component Standards (MANDATORY)

### A. Widget Selection Priority

**Follow this priority order when selecting UI components:**

```
1. Native/Platform Widgets (PREFERRED)
   ↓ Best performance, familiar UX, accessibility built-in

2. Standard Design System Widgets
   ↓ Material UI, Fluent UI, Human Interface elements

3. Well-Maintained Open Source Libraries
   ↓ Large community, regular updates, good accessibility

4. Custom Implementations
   → Only when above options don't meet requirements
```

### B. Recommended Widget Libraries (by Platform)

| Platform | Recommended | Alternative | Notes |
|----------|-------------|-------------|-------|
| Web (React) | Material UI, Radix | Chakra, Ant Design | MUI for enterprise, Radix for flexibility |
| Web (Vue) | Vuetify, Naive UI | PrimeVue | Vuetify follows Material spec |
| Web (Angular) | Angular Material | PrimeNG | Official Google library |
| Web (Svelte) | Skeleton, shadcn-svelte | Svelte Material | Growing ecosystem |
| Flutter | Material, Cupertino | flutter_hooks | Platform-adaptive by default |
| React Native | React Native Paper | NativeBase | Paper follows Material 3 |
| iOS Native | SwiftUI, UIKit | - | Always prefer native |
| Android Native | Jetpack Compose, Material 3 | - | Compose is modern standard |
| Desktop (Cross) | Tauri + Web | Electron | Tauri for smaller bundle |

### C. Component Consistency Requirements

```
Design Token Hierarchy
┌─────────────────────────────────────────────────────────────┐
│                    Global Design Tokens                      │
│  (colors, typography, spacing, shadows, breakpoints)        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Component Tokens                          │
│  (button-primary-bg, input-border-radius, card-shadow)      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Component Styles                          │
│  (Applied via CSS-in-JS, CSS Modules, or Tailwind)          │
└─────────────────────────────────────────────────────────────┘
```

**MANDATORY: All components MUST use design tokens, not hard-coded values.**

### D. Test Selectors for UI Testing (MANDATORY)

**CRITICAL: ALL UI components MUST include test-specific selectors to enable reliable automated testing.**

Every component—with particular attention to buttons, input fields, links, and interactive elements—MUST have a dedicated test selector that:

1. **Is stable**: Remains constant regardless of component position, order, or surrounding DOM structure
2. **Is readable**: Uses clear, descriptive names that convey the element's purpose
3. **Supports nesting**: Allows hierarchical organization for complex component trees
4. **Is unique**: Provides unambiguous identification within the page context

#### Web Applications (Default: `data-testid` with kebab-case)

**MANDATORY: Use `data-testid` attributes with kebab-case naming unless the user specifies a different convention.**

```javascript
// CORRECT: Clear, stable test selectors
<button data-testid="submit-order-button">Place Order</button>
<input data-testid="email-input" type="email" />
<div data-testid="user-profile-card">
  <span data-testid="user-profile-card-name">{user.name}</span>
  <button data-testid="user-profile-card-edit-button">Edit</button>
</div>

// CORRECT: Nested components with hierarchical naming
<form data-testid="checkout-form">
  <div data-testid="checkout-form-billing-section">
    <input data-testid="checkout-form-billing-address-input" />
    <input data-testid="checkout-form-billing-city-input" />
  </div>
  <button data-testid="checkout-form-submit-button">Complete Purchase</button>
</form>

// CORRECT: List items with unique identifiers
<ul data-testid="product-list">
  {products.map(product => (
    <li key={product.id} data-testid={`product-list-item-${product.id}`}>
      <span data-testid={`product-list-item-${product.id}-name`}>{product.name}</span>
      <button data-testid={`product-list-item-${product.id}-add-to-cart-button`}>
        Add to Cart
      </button>
    </li>
  ))}
</ul>

// WRONG: No test selectors
<button>Place Order</button>
<input type="email" />

// WRONG: Using CSS classes or element positions for testing
// Tests that rely on these WILL break when styling or layout changes
```

#### Naming Convention Guidelines

```
Test Selector Naming Pattern
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  Pattern: [context]-[element-description]-[element-type]    │
│                                                              │
│  Examples:                                                   │
│  ├─ login-form                     (container)              │
│  ├─ login-form-email-input         (input field)            │
│  ├─ login-form-password-input      (input field)            │
│  ├─ login-form-submit-button       (button)                 │
│  ├─ login-form-forgot-password-link (link)                  │
│  ├─ header-nav                     (navigation container)   │
│  ├─ header-nav-home-link           (navigation link)        │
│  ├─ header-nav-user-menu-button    (dropdown trigger)       │
│  └─ header-nav-user-menu-logout-item (menu item)            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Element Type Suffixes (Recommended)

| Element Type | Suffix | Example |
|--------------|--------|---------|
| Buttons | `-button` | `submit-order-button` |
| Input fields | `-input` | `email-input` |
| Links | `-link` | `forgot-password-link` |
| Containers/Sections | (none or `-section`) | `billing-section` |
| Lists | `-list` | `product-list` |
| List items | `-item` | `product-list-item-123` |
| Cards | `-card` | `user-profile-card` |
| Modals/Dialogs | `-modal`, `-dialog` | `confirm-delete-modal` |
| Dropdowns | `-dropdown`, `-select` | `country-select` |
| Checkboxes | `-checkbox` | `terms-agreement-checkbox` |
| Radio buttons | `-radio` | `shipping-method-express-radio` |
| Text areas | `-textarea` | `feedback-comment-textarea` |
| Tables | `-table` | `orders-table` |
| Table rows | `-row` | `orders-table-row-456` |

#### Mobile/Native Applications

For mobile and native applications, use the platform-appropriate testing attributes:

```dart
// Flutter: Use Key for test identification
ElevatedButton(
  key: const Key('submit-order-button'),
  onPressed: submitOrder,
  child: const Text('Place Order'),
)

TextField(
  key: const Key('email-input'),
  decoration: const InputDecoration(labelText: 'Email'),
)
```

```swift
// SwiftUI: Use accessibilityIdentifier
Button("Place Order") {
    submitOrder()
}
.accessibilityIdentifier("submit-order-button")

TextField("Email", text: $email)
    .accessibilityIdentifier("email-input")
```

```kotlin
// Jetpack Compose: Use testTag modifier
Button(
    onClick = { submitOrder() },
    modifier = Modifier.testTag("submit-order-button")
) {
    Text("Place Order")
}

TextField(
    value = email,
    onValueChange = { email = it },
    modifier = Modifier.testTag("email-input")
)
```

#### Custom Conventions

If the project specifies a different selector convention (e.g., `data-test`, `data-cy`, `data-qa`), follow that convention consistently:

```javascript
// If project uses data-cy (Cypress convention)
<button data-cy="submit-order-button">Place Order</button>

// If project uses data-test
<button data-test="submit-order-button">Place Order</button>

// If project uses camelCase instead of kebab-case
<button data-testid="submitOrderButton">Place Order</button>
```

**IMPORTANT:** Always check for existing test selector conventions in the codebase before adding new selectors. Maintain consistency with established patterns.

#### Prohibited Practices for Test Selectors

**NEVER use these for test selection—they are fragile and will break:**

- [ ] CSS class names (change with styling updates)
- [ ] Element tag names alone (not specific enough)
- [ ] DOM position/index (changes with layout)
- [ ] Text content (changes with i18n, copy updates)
- [ ] Auto-generated IDs (unpredictable)
- [ ] XPath based on structure (breaks with refactoring)

```javascript
// WRONG: All of these will cause flaky tests
document.querySelector('.btn-primary');           // CSS class
document.querySelectorAll('button')[2];           // Position
document.querySelector('div > span > button');    // DOM structure
cy.contains('Place Order');                       // Text content (fragile)
```

---

## 5. State Management (MANDATORY)

### A. State Categories

```
┌─────────────────────────────────────────────────────────────┐
│                    UI State Categories                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Component State (local)                                  │
│     - Form inputs, toggles, open/closed states               │
│     - Managed: useState, ref                                 │
│                                                              │
│  2. Page State (route-level)                                 │
│     - Filters, sorts, scroll position, selected items        │
│     - MUST persist during navigation                         │
│     - Managed: URL params, route state, context              │
│                                                              │
│  3. Application State (global)                               │
│     - User session, preferences, notifications               │
│     - Managed: Redux, Zustand, MobX, Signals                 │
│                                                              │
│  4. Server State (cached)                                    │
│     - API responses, real-time data                          │
│     - Managed: React Query, SWR, Apollo Client               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### B. Navigation State Preservation (CRITICAL)

**MANDATORY: UI state MUST be preserved during navigation.**

```javascript
// CORRECT: State preserved in URL for shareability and persistence
function DataGridPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  // Read state from URL
  const filters = {
    search: searchParams.get('search') || '',
    status: searchParams.get('status') || 'all',
    sortBy: searchParams.get('sort') || 'created',
    sortOrder: searchParams.get('order') || 'desc',
    page: parseInt(searchParams.get('page') || '1'),
  };

  // Update URL when state changes (no page reload)
  const updateFilter = (key, value) => {
    setSearchParams(prev => {
      prev.set(key, value);
      return prev;
    });
  };

  return <DataGrid filters={filters} onFilterChange={updateFilter} />;
}

// WRONG: State lost on navigation
function DataGridPage() {
  const [filters, setFilters] = useState({ search: '', status: 'all' });
  // State is lost when user navigates away and returns!
}
```

### C. Back/Forward Navigation

**Requirements for proper history management:**

```javascript
// Support browser back/forward with state restoration
function usePersistedState(key, defaultValue) {
  const [state, setState] = useState(() => {
    // 1. Try to restore from history state
    const historyState = window.history.state?.[key];
    if (historyState !== undefined) return historyState;

    // 2. Try to restore from sessionStorage
    const stored = sessionStorage.getItem(key);
    if (stored) return JSON.parse(stored);

    return defaultValue;
  });

  useEffect(() => {
    // Save to both history state and sessionStorage
    const newHistoryState = { ...window.history.state, [key]: state };
    window.history.replaceState(newHistoryState, '');
    sessionStorage.setItem(key, JSON.stringify(state));
  }, [key, state]);

  return [state, setState];
}
```

### D. Optional Reload Persistence

**For critical state that should survive page reloads:**

```javascript
// Use sessionStorage for session-lifetime persistence
// Use localStorage for cross-session persistence (user preferences)

const useReloadPersistence = (key, initialValue, storage = sessionStorage) => {
  const [value, setValue] = useState(() => {
    try {
      const item = storage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch {
      return initialValue;
    }
  });

  useEffect(() => {
    storage.setItem(key, JSON.stringify(value));
  }, [key, value, storage]);

  return [value, setValue];
};

// Usage
const [gridState, setGridState] = useReloadPersistence('dataGridState', {
  columnOrder: ['id', 'name', 'status', 'created'],
  columnWidths: {},
  hiddenColumns: [],
});
```

---

## 6. Data Grids & Tables (MANDATORY)

### A. Architecture Overview

**CRITICAL: For large datasets, ALL operations MUST happen server-side.**

```
Data Grid Architecture
┌─────────────────────────────────────────────────────────────┐
│                         UI Layer                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Virtual Viewport                    │    │
│  │  (renders only visible rows + buffer)               │    │
│  │                                                      │    │
│  │  Row 1  │ Col A │ Col B │ Col C │ Col D │ ...      │    │
│  │  Row 2  │       │       │       │       │          │    │
│  │  Row 3  │       │       │       │       │          │    │
│  │  ...    │       │       │       │       │          │    │
│  │  Row N  │       │       │       │       │          │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Request: { page, pageSize,
                            │            sort, filters,
                            │            search }
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Server/Backend                          │
│                                                              │
│  - Execute filters on database                               │
│  - Apply sorting                                             │
│  - Paginate results                                          │
│  - Return: { data: [], total: N, page: X }                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### B. Server-Side Operations (MANDATORY for Large Datasets)

```javascript
// Grid state that gets sent to server
interface GridRequest {
  // Pagination
  page: number;
  pageSize: number;

  // Sorting
  sortBy: string;
  sortOrder: 'asc' | 'desc';

  // Filtering
  filters: Record<string, FilterValue>;

  // Column-specific search
  columnSearch: Record<string, string>;

  // Global search
  globalSearch?: string;
}

// Server response
interface GridResponse<T> {
  data: T[];           // Only requested page
  total: number;       // Total records (for pagination UI)
  page: number;        // Current page
  pageSize: number;    // Records per page
  hasMore: boolean;    // For infinite scroll
}

// API call example
async function fetchGridData(request: GridRequest): Promise<GridResponse<Row>> {
  const params = new URLSearchParams({
    page: request.page.toString(),
    pageSize: request.pageSize.toString(),
    sortBy: request.sortBy,
    sortOrder: request.sortOrder,
    filters: JSON.stringify(request.filters),
    search: request.globalSearch || '',
  });

  const response = await fetch(`/api/data?${params}`);
  return response.json();
}
```

### C. Virtualization & Infinite Scroll

**MANDATORY: Use virtualization for grids with 100+ rows.**

```javascript
// Virtualized grid with infinite scroll
function VirtualizedDataGrid({ fetchData, rowHeight = 48 }) {
  const [data, setData] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const containerRef = useRef(null);

  // Calculate visible range
  const { scrollTop, clientHeight } = useVirtualScroll(containerRef);
  const startIndex = Math.floor(scrollTop / rowHeight);
  const endIndex = Math.min(
    startIndex + Math.ceil(clientHeight / rowHeight) + 5, // Buffer
    total
  );

  // Only render visible rows
  const visibleRows = data.slice(startIndex, endIndex);

  // Infinite scroll: fetch more when near bottom
  useEffect(() => {
    const container = containerRef.current;
    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      const nearBottom = scrollTop + clientHeight >= scrollHeight - 200;

      if (nearBottom && !loading && data.length < total) {
        loadMoreData();
      }
    };

    container?.addEventListener('scroll', handleScroll);
    return () => container?.removeEventListener('scroll', handleScroll);
  }, [loading, data.length, total]);

  return (
    <div
      ref={containerRef}
      style={{ height: '100%', overflow: 'auto' }}
    >
      {/* Spacer for scroll height */}
      <div style={{ height: total * rowHeight, position: 'relative' }}>
        {/* Only visible rows rendered */}
        {visibleRows.map((row, index) => (
          <div
            key={row.id}
            style={{
              position: 'absolute',
              top: (startIndex + index) * rowHeight,
              height: rowHeight,
            }}
          >
            <GridRow data={row} />
          </div>
        ))}
      </div>
    </div>
  );
}
```

### D. Column Features

**All column operations SHOULD happen server-side for large datasets:**

```javascript
interface ColumnDefinition {
  id: string;
  header: string;
  accessor: string | ((row: any) => any);

  // Sorting
  sortable: boolean;
  defaultSort?: 'asc' | 'desc';

  // Filtering
  filterable: boolean;
  filterType: 'text' | 'select' | 'date' | 'number' | 'boolean';
  filterOptions?: { value: any; label: string }[]; // For select

  // Search
  searchable: boolean;

  // Display
  width?: number;
  minWidth?: number;
  resizable: boolean;
  reorderable: boolean;
  hideable: boolean;

  // Cell
  editable: boolean;
  cellRenderer?: (value: any, row: any) => ReactNode;
  cellEditor?: (value: any, onChange: (v: any) => void) => ReactNode;
}

// Example column definitions
const columns: ColumnDefinition[] = [
  {
    id: 'id',
    header: 'ID',
    accessor: 'id',
    sortable: true,
    filterable: false,
    searchable: false,
    width: 80,
    resizable: false,
    reorderable: false,
    hideable: false,
    editable: false,
  },
  {
    id: 'name',
    header: 'Name',
    accessor: 'name',
    sortable: true,
    filterable: true,
    filterType: 'text',
    searchable: true,
    resizable: true,
    reorderable: true,
    hideable: true,
    editable: true,
  },
  {
    id: 'status',
    header: 'Status',
    accessor: 'status',
    sortable: true,
    filterable: true,
    filterType: 'select',
    filterOptions: [
      { value: 'active', label: 'Active' },
      { value: 'inactive', label: 'Inactive' },
      { value: 'pending', label: 'Pending' },
    ],
    searchable: false,
    editable: true,
  },
];
```

### E. Row Operations

**Support for selection, editing, reordering, and bulk actions:**

```javascript
interface GridRowOperations {
  // Selection
  selectionMode: 'none' | 'single' | 'multiple' | 'checkbox';
  selectedRows: Set<string>;
  onSelectionChange: (selected: Set<string>) => void;

  // Editing
  editMode: 'none' | 'cell' | 'row' | 'modal';
  editingRowId: string | null;
  onRowEdit: (rowId: string, changes: Partial<Row>) => Promise<void>;

  // Reordering (if enabled)
  reorderable: boolean;
  onReorder: (fromIndex: number, toIndex: number) => Promise<void>;

  // Deletion (if enabled)
  deletable: boolean;
  onDelete: (rowIds: string[]) => Promise<void>;

  // Bulk actions
  bulkActions?: {
    id: string;
    label: string;
    icon?: ReactNode;
    onExecute: (selectedIds: string[]) => Promise<void>;
  }[];
}

// Example implementation
function DataGridWithOperations() {
  const [selectedRows, setSelectedRows] = useState(new Set<string>());

  const handleBulkDelete = async () => {
    const ids = Array.from(selectedRows);
    await api.deleteRows(ids);
    setSelectedRows(new Set());
    refetch();
  };

  const handleRowReorder = async (fromIndex: number, toIndex: number) => {
    // Optimistic update
    const reordered = [...data];
    const [moved] = reordered.splice(fromIndex, 1);
    reordered.splice(toIndex, 0, moved);
    setData(reordered);

    // Persist to server
    try {
      await api.reorderRows(moved.id, toIndex);
    } catch (error) {
      // Rollback on error
      refetch();
    }
  };

  return (
    <DataGrid
      data={data}
      selectionMode="checkbox"
      selectedRows={selectedRows}
      onSelectionChange={setSelectedRows}
      reorderable={permissions.canReorder}
      onReorder={handleRowReorder}
      bulkActions={[
        { id: 'delete', label: 'Delete', onExecute: handleBulkDelete },
        { id: 'export', label: 'Export', onExecute: handleExport },
      ]}
    />
  );
}
```

### F. State Persistence for Grid

**MANDATORY: Grid state MUST persist across navigation.**

```javascript
interface GridState {
  // Column state
  columnOrder: string[];
  columnWidths: Record<string, number>;
  hiddenColumns: string[];

  // Sort state
  sortBy: string | null;
  sortOrder: 'asc' | 'desc';

  // Filter state
  filters: Record<string, FilterValue>;
  columnSearch: Record<string, string>;
  globalSearch: string;

  // Pagination state
  page: number;
  pageSize: number;

  // Selection state (optional - may not persist)
  selectedRows?: string[];

  // Scroll position
  scrollTop: number;
  scrollLeft: number;
}

// Save grid state to URL + storage
function useGridStatePersistence(gridId: string, defaultState: GridState) {
  const [searchParams, setSearchParams] = useSearchParams();

  // Core state in URL (shareable)
  const urlState = {
    page: parseInt(searchParams.get('page') || String(defaultState.page)),
    pageSize: parseInt(searchParams.get('size') || String(defaultState.pageSize)),
    sortBy: searchParams.get('sort') || defaultState.sortBy,
    sortOrder: (searchParams.get('order') || defaultState.sortOrder) as 'asc' | 'desc',
    filters: JSON.parse(searchParams.get('filters') || '{}'),
    search: searchParams.get('q') || '',
  };

  // UI state in sessionStorage (not shareable)
  const [uiState, setUiState] = useReloadPersistence(`grid-${gridId}`, {
    columnOrder: defaultState.columnOrder,
    columnWidths: defaultState.columnWidths,
    hiddenColumns: defaultState.hiddenColumns,
    scrollTop: 0,
    scrollLeft: 0,
  });

  const updateUrlState = (updates: Partial<typeof urlState>) => {
    setSearchParams(prev => {
      Object.entries(updates).forEach(([key, value]) => {
        if (value !== null && value !== undefined) {
          prev.set(key, typeof value === 'object' ? JSON.stringify(value) : String(value));
        }
      });
      return prev;
    });
  };

  return {
    state: { ...urlState, ...uiState },
    updateUrlState,
    updateUiState: setUiState,
  };
}
```

---

## 7. Communication Protocols (MANDATORY)

### A. Protocol Selection Priority

**CRITICAL: Choose the most efficient protocol for your use case.**

```
Protocol Priority (Best to Worst for Real-Time)
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  1. Socket.IO / Engine.IO (PREFERRED)                       │
│     ├─ Automatic fallback (WebSocket → HTTP Long Polling)   │
│     ├─ Built-in reconnection and heartbeat                  │
│     ├─ Room/namespace support for targeted updates          │
│     ├─ Works through corporate proxies and firewalls        │
│     └─ Binary data support                                  │
│                                                              │
│  2. WebSocket (Native)                                       │
│     ├─ Lower overhead than Socket.IO                        │
│     ├─ Bidirectional communication                          │
│     ├─ May be blocked by some firewalls/proxies             │
│     └─ Manual reconnection logic needed                     │
│                                                              │
│  3. Server-Sent Events (SSE)                                 │
│     ├─ Server-to-client only (unidirectional)               │
│     ├─ Works through HTTP (firewall-friendly)               │
│     ├─ Automatic reconnection                               │
│     └─ Good for feeds, notifications                        │
│                                                              │
│  4. GraphQL Subscriptions                                    │
│     ├─ Typed real-time updates                              │
│     ├─ Integrates with GraphQL ecosystem                    │
│     ├─ Uses WebSocket underneath                            │
│     └─ Higher complexity                                    │
│                                                              │
│  5. REST + Polling (AVOID for real-time)                    │
│     ├─ Simple implementation                                │
│     ├─ High server load                                     │
│     ├─ Delayed updates                                      │
│     └─ Only use when real-time not needed                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### B. Socket.IO Implementation (Recommended)

```javascript
// Client-side Socket.IO with reconnection handling
import { io, Socket } from 'socket.io-client';

class RealTimeClient {
  private socket: Socket;
  private subscriptions = new Map<string, Set<(data: any) => void>>();

  connect(url: string, options?: SocketIOClient.ConnectOpts) {
    this.socket = io(url, {
      // Transport fallback for firewall compatibility
      transports: ['websocket', 'polling'],

      // Reconnection settings
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,

      // Authentication
      auth: {
        token: getAuthToken(),
      },

      ...options,
    });

    this.socket.on('connect', () => {
      console.log('Connected to real-time server');
      // Resubscribe to channels after reconnection
      this.resubscribeAll();
    });

    this.socket.on('disconnect', (reason) => {
      console.log('Disconnected:', reason);
      if (reason === 'io server disconnect') {
        // Server initiated disconnect, need manual reconnect
        this.socket.connect();
      }
    });

    this.socket.on('error', (error) => {
      console.error('Socket error:', error);
    });
  }

  subscribe<T>(channel: string, callback: (data: T) => void) {
    // Track subscription
    if (!this.subscriptions.has(channel)) {
      this.subscriptions.set(channel, new Set());
      this.socket.emit('subscribe', { channel });
    }
    this.subscriptions.get(channel)!.add(callback);

    // Listen for updates
    this.socket.on(channel, callback);

    // Return unsubscribe function
    return () => {
      this.subscriptions.get(channel)?.delete(callback);
      this.socket.off(channel, callback);

      if (this.subscriptions.get(channel)?.size === 0) {
        this.socket.emit('unsubscribe', { channel });
        this.subscriptions.delete(channel);
      }
    };
  }

  private resubscribeAll() {
    this.subscriptions.forEach((_, channel) => {
      this.socket.emit('subscribe', { channel });
    });
  }
}

// Usage in React component
function useRealTimeData<T>(channel: string) {
  const [data, setData] = useState<T | null>(null);
  const client = useRealTimeClient();

  useEffect(() => {
    const unsubscribe = client.subscribe<T>(channel, setData);
    return unsubscribe;
  }, [channel, client]);

  return data;
}
```

### C. Avoiding Polling

**NEVER use polling when real-time alternatives exist:**

```javascript
// WRONG: Polling wastes bandwidth and server resources
function useDataWithPolling() {
  const [data, setData] = useState(null);

  useEffect(() => {
    const interval = setInterval(async () => {
      const response = await fetch('/api/data');
      setData(await response.json());
    }, 5000); // Polling every 5 seconds

    return () => clearInterval(interval);
  }, []);

  return data;
}

// CORRECT: Real-time subscription
function useDataWithRealTime() {
  const [data, setData] = useState(null);
  const socket = useSocket();

  useEffect(() => {
    // Initial fetch
    fetch('/api/data').then(r => r.json()).then(setData);

    // Real-time updates
    socket.on('data:updated', setData);
    return () => socket.off('data:updated', setData);
  }, [socket]);

  return data;
}
```

### D. Optimistic Updates

**Provide instant feedback while syncing with server:**

```javascript
function useOptimisticUpdate<T>(
  mutationFn: (data: T) => Promise<T>,
  queryKey: string
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn,

    // Optimistic update
    onMutate: async (newData: T) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: [queryKey] });

      // Snapshot previous value
      const previousData = queryClient.getQueryData([queryKey]);

      // Optimistically update cache
      queryClient.setQueryData([queryKey], (old: T[]) => {
        // Apply optimistic change
        return updateDataOptimistically(old, newData);
      });

      // Return context for rollback
      return { previousData };
    },

    // Rollback on error
    onError: (err, variables, context) => {
      queryClient.setQueryData([queryKey], context?.previousData);
      toast.error('Update failed. Changes reverted.');
    },

    // Refetch after success/error
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: [queryKey] });
    },
  });
}
```

---

## 8. Performance Optimization (MANDATORY)

### A. Data Transfer Minimization

```javascript
// MANDATORY: Only fetch visible data

// API design for minimal transfers
interface PaginatedRequest {
  offset: number;      // Starting index
  limit: number;       // Number of items (visible + buffer)
  fields?: string[];   // Only return these fields (sparse fieldsets)
}

// Example: Grid with 20 visible rows should request ~25-30 rows max
const VISIBLE_ROWS = 20;
const BUFFER_ROWS = 10;

async function fetchVisibleData(scrollTop: number, rowHeight: number) {
  const startIndex = Math.floor(scrollTop / rowHeight);

  return api.getData({
    offset: Math.max(0, startIndex - BUFFER_ROWS / 2),
    limit: VISIBLE_ROWS + BUFFER_ROWS,
    fields: ['id', 'name', 'status'], // Only needed columns
  });
}
```

### B. Memory Management

```javascript
// Prevent memory leaks

// 1. Cleanup subscriptions
useEffect(() => {
  const subscription = eventSource.subscribe(handler);
  return () => subscription.unsubscribe(); // ALWAYS cleanup
}, []);

// 2. Cancel pending requests on unmount
useEffect(() => {
  const controller = new AbortController();

  fetch('/api/data', { signal: controller.signal })
    .then(handleResponse)
    .catch(err => {
      if (err.name !== 'AbortError') throw err;
    });

  return () => controller.abort(); // Cancel on unmount
}, []);

// 3. Limit cache size for large datasets
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,      // 5 minutes
      gcTime: 30 * 60 * 1000,        // 30 minutes garbage collection
      refetchOnWindowFocus: false,
    },
  },
});

// 4. Use windowing for large lists
// Only render items in viewport
<VirtualList
  height={400}
  itemCount={10000}
  itemSize={35}
  width="100%"
>
  {({ index, style }) => (
    <div style={style}>{items[index]}</div>
  )}
</VirtualList>
```

### C. Bundle Size Optimization

```javascript
// 1. Dynamic imports for routes
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Settings = lazy(() => import('./pages/Settings'));

// 2. Tree-shakeable imports
// WRONG: imports entire library
import _ from 'lodash';

// CORRECT: imports only needed function
import debounce from 'lodash/debounce';

// 3. Code splitting for heavy components
const HeavyChart = lazy(() => import('./components/HeavyChart'));

function Dashboard() {
  return (
    <Suspense fallback={<ChartSkeleton />}>
      <HeavyChart data={data} />
    </Suspense>
  );
}
```

---

## 9. Routing & Navigation (MANDATORY)

### A. Route Structure

```javascript
// Define routes with state preservation
const routes = [
  {
    path: '/',
    element: <Layout />,
    children: [
      { path: 'dashboard', element: <Dashboard /> },
      {
        path: 'data',
        element: <DataPage />,
        // Preserve scroll position and state
        handle: { scrollRestoration: true, preserveState: true },
      },
      {
        path: 'data/:id',
        element: <DataDetail />,
        // Return to preserved parent state
        handle: { preserveParentState: true },
      },
    ],
  },
];

// Router with scroll restoration
<RouterProvider
  router={router}
  future={{ v7_startTransition: true }}
/>
```

### B. Clean Back Navigation

```javascript
// Implement clean return behavior
function DataDetail() {
  const navigate = useNavigate();
  const location = useLocation();

  const handleBack = () => {
    // Check if we came from our app
    if (location.state?.from === '/data') {
      // Use browser back to restore exact state
      navigate(-1);
    } else {
      // Direct navigation, go to list
      navigate('/data');
    }
  };

  return (
    <div>
      <button onClick={handleBack}>Back to List</button>
      {/* Detail content */}
    </div>
  );
}

// When navigating to detail, preserve state
function DataList() {
  const navigate = useNavigate();
  const location = useLocation();

  const openDetail = (id: string) => {
    navigate(`/data/${id}`, {
      state: { from: location.pathname + location.search },
    });
  };
}
```

---

## 10. Testing (MANDATORY)

### A. Component Testing

```javascript
// Test UI components thoroughly

describe('DataGrid', () => {
  // Rendering tests
  it('renders loading skeleton while fetching', () => {
    render(<DataGrid loading={true} />);
    expect(screen.getByTestId('grid-skeleton')).toBeInTheDocument();
  });

  it('renders empty state when no data', () => {
    render(<DataGrid data={[]} loading={false} />);
    expect(screen.getByText('No data available')).toBeInTheDocument();
  });

  // Interaction tests
  it('calls onSort when column header clicked', async () => {
    const onSort = jest.fn();
    render(<DataGrid data={mockData} onSort={onSort} />);

    await userEvent.click(screen.getByText('Name'));

    expect(onSort).toHaveBeenCalledWith('name', 'asc');
  });

  // State persistence tests
  it('restores filter state from URL', () => {
    renderWithRouter(<DataGridPage />, {
      route: '/data?status=active&sort=name',
    });

    expect(screen.getByDisplayValue('active')).toBeInTheDocument();
    expect(screen.getByText('Name').closest('th'))
      .toHaveAttribute('aria-sort', 'ascending');
  });

  // Accessibility tests
  it('supports keyboard navigation', async () => {
    render(<DataGrid data={mockData} />);

    const firstCell = screen.getAllByRole('gridcell')[0];
    firstCell.focus();

    await userEvent.keyboard('{ArrowRight}');
    expect(document.activeElement).toBe(screen.getAllByRole('gridcell')[1]);

    await userEvent.keyboard('{ArrowDown}');
    // Focus should move to cell below
  });
});
```

### B. Visual Regression Testing

```javascript
// Playwright visual tests
import { test, expect } from '@playwright/test';

test.describe('DataGrid visual regression', () => {
  test('default state matches snapshot', async ({ page }) => {
    await page.goto('/data');
    await page.waitForSelector('[data-testid="data-grid"]');

    await expect(page).toHaveScreenshot('data-grid-default.png');
  });

  test('with active filters matches snapshot', async ({ page }) => {
    await page.goto('/data?status=active&search=test');
    await page.waitForSelector('[data-testid="data-grid"]');

    await expect(page).toHaveScreenshot('data-grid-filtered.png');
  });

  test('selected rows matches snapshot', async ({ page }) => {
    await page.goto('/data');
    await page.click('[data-testid="select-row-1"]');
    await page.click('[data-testid="select-row-3"]');

    await expect(page).toHaveScreenshot('data-grid-selected.png');
  });
});
```

### C. Integration Tests

```javascript
// Test full user flows
describe('Data management flow', () => {
  it('filters, sorts, and exports data', async () => {
    // Setup
    render(<App />);
    await waitForDataLoad();

    // Apply filter
    await userEvent.selectOptions(
      screen.getByLabelText('Status'),
      'active'
    );
    await waitForRefetch();

    // Verify filter applied
    const rows = screen.getAllByRole('row');
    rows.forEach(row => {
      expect(within(row).getByText('Active')).toBeInTheDocument();
    });

    // Apply sort
    await userEvent.click(screen.getByText('Created Date'));
    await waitForRefetch();

    // Select rows for export
    await userEvent.click(screen.getByLabelText('Select all'));

    // Export
    await userEvent.click(screen.getByText('Export'));
    expect(mockExportFn).toHaveBeenCalledWith(
      expect.arrayContaining([
        expect.objectContaining({ status: 'active' }),
      ])
    );
  });
});
```

---

## 11. Accessibility (MANDATORY)

### A. Requirements

**MANDATORY: All UI MUST meet WCAG 2.1 Level AA.**

```javascript
// Accessibility checklist for components

// 1. Keyboard navigation
<DataGrid
  role="grid"
  aria-label="User data"
  tabIndex={0}
  onKeyDown={handleGridKeyDown}
>
  {rows.map((row, rowIndex) => (
    <div
      role="row"
      key={row.id}
      aria-rowindex={rowIndex + 1}
    >
      {columns.map((col, colIndex) => (
        <div
          role="gridcell"
          key={col.id}
          aria-colindex={colIndex + 1}
          tabIndex={-1}
        >
          {row[col.accessor]}
        </div>
      ))}
    </div>
  ))}
</DataGrid>

// 2. Screen reader announcements
function useAnnouncer() {
  const announce = useCallback((message: string, priority: 'polite' | 'assertive' = 'polite') => {
    const region = document.getElementById('announcer');
    region?.setAttribute('aria-live', priority);
    region!.textContent = message;
  }, []);

  return announce;
}

// Usage
const announce = useAnnouncer();
const handleSort = (column: string) => {
  // ... sort logic
  announce(`Table sorted by ${column} in ascending order`);
};

// 3. Focus management
function Modal({ isOpen, onClose, children }) {
  const previousFocus = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (isOpen) {
      previousFocus.current = document.activeElement as HTMLElement;
      // Focus first focusable element in modal
      modalRef.current?.querySelector<HTMLElement>('[tabindex]')?.focus();
    } else {
      // Restore focus when modal closes
      previousFocus.current?.focus();
    }
  }, [isOpen]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
    >
      {children}
    </div>
  );
}
```

### B. Color Contrast

```javascript
// Design tokens with accessible colors
const colors = {
  // Ensure 4.5:1 contrast ratio for text
  text: {
    primary: '#1a1a1a',      // On white: 16:1
    secondary: '#4a4a4a',    // On white: 7.5:1
    disabled: '#757575',     // On white: 4.6:1 (minimum)
  },

  // Ensure 3:1 for UI components
  border: {
    default: '#767676',      // On white: 4.5:1
    focus: '#0066cc',        // Clearly visible focus ring
  },

  // Error states must be distinguishable
  status: {
    error: '#d32f2f',
    errorText: '#b71c1c',    // Darker for text on light bg
  },
};
```

---

## 12. Documentation (MANDATORY)

### A. Component Documentation

```javascript
/**
 * DataGrid - A virtualized, server-side data grid component.
 *
 * @description
 * Renders large datasets efficiently using virtualization.
 * All filtering, sorting, and pagination happen server-side.
 *
 * @example
 * ```tsx
 * <DataGrid
 *   columns={columns}
 *   fetchData={fetchUsers}
 *   pageSize={50}
 *   onRowClick={handleRowClick}
 * />
 * ```
 *
 * @param {ColumnDefinition[]} columns - Column configuration
 * @param {(params: GridRequest) => Promise<GridResponse>} fetchData - Data fetcher
 * @param {number} [pageSize=20] - Rows per page
 * @param {'none'|'single'|'multiple'} [selectionMode='none'] - Row selection mode
 *
 * @accessibility
 * - Implements grid role with proper row/cell semantics
 * - Full keyboard navigation (arrow keys, Home, End, Page Up/Down)
 * - Announces sort changes to screen readers
 *
 * @see https://www.w3.org/WAI/ARIA/apg/patterns/grid/
 */
export function DataGrid({ columns, fetchData, pageSize = 20, selectionMode = 'none' }) {
  // ...
}
```

---

## 13. Deployment Checklist

### Pre-Deployment Verification (MANDATORY)

#### Accessibility
- [ ] Lighthouse accessibility score > 90
- [ ] axe-core returns 0 violations
- [ ] All interactive elements keyboard accessible
- [ ] Screen reader tested (VoiceOver/NVDA)
- [ ] Color contrast meets WCAG AA

#### Performance
- [ ] Lighthouse performance score > 90
- [ ] First Contentful Paint < 1.5s
- [ ] Largest Contentful Paint < 2.5s
- [ ] No memory leaks (checked with DevTools)
- [ ] Bundle size within budget

#### Functionality
- [ ] All component tests pass
- [ ] Visual regression tests pass
- [ ] E2E tests for critical flows pass
- [ ] State persists through navigation
- [ ] Browser back/forward works correctly
- [ ] Real-time updates working

#### Cross-Platform
- [ ] Tested on Chrome, Firefox, Safari, Edge
- [ ] Mobile responsive (iOS Safari, Chrome Android)
- [ ] Touch interactions work
- [ ] Works offline (if PWA)

#### Data Handling
- [ ] Server-side pagination working
- [ ] Virtualization for large lists
- [ ] No full dataset fetches
- [ ] Real-time protocol working through firewall

---

## 14. Quick Reference

### Common Patterns

```bash
# Run accessibility audit
npx lighthouse http://localhost:3000 --only-categories=accessibility

# Run visual regression tests
npx playwright test --update-snapshots  # Update baselines
npx playwright test                      # Compare against baselines

# Profile memory
# Chrome DevTools > Memory > Heap Snapshot

# Test keyboard navigation
# Tab, Shift+Tab, Enter, Space, Arrow keys, Escape
```

### State Persistence Checklist

```
URL (shareable):
✓ Current page/route
✓ Filter values
✓ Sort column and direction
✓ Search query
✓ Pagination state

SessionStorage (session):
✓ Column order
✓ Column widths
✓ Hidden columns
✓ Scroll position

LocalStorage (persistent):
✓ User preferences
✓ Theme selection
✓ Sidebar collapsed state
```

### Data Grid Decision Tree

```
Dataset Size?
├─ < 100 rows → Client-side OK
├─ 100-1000 rows → Virtualization recommended
└─ > 1000 rows → Server-side MANDATORY
                  └─ Use virtualization
                  └─ Fetch only visible rows
                  └─ Server-side filter/sort/page
```

### Communication Protocol Decision Tree

```
Need real-time updates?
├─ No → REST API
└─ Yes → Bidirectional needed?
         ├─ No → Server-Sent Events (SSE)
         └─ Yes → Firewall concerns?
                  ├─ Yes → Socket.IO (with fallback)
                  └─ No → WebSocket (if simpler)
```

---

## References

### Required Companion Guides
- [tdd.md](./tdd.md) - Test-driven development guide
- [todo.md](./todo.md) - State management and TODO tracking
- [agents-md.md](./agents-md.md) - AGENTS.md creation for project state

### Framework-Specific Guides
- [reactjs.md](./reactjs.md) - React-specific patterns
- [angular.md](./angular.md) - Angular-specific patterns
- [svelte.md](./svelte.md) - Svelte-specific patterns
- [flutter.md](./flutter.md) - Flutter mobile/desktop patterns

### Related Standards
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [WAI-ARIA Authoring Practices](https://www.w3.org/WAI/ARIA/apg/)
- [Material Design Guidelines](https://m3.material.io/)
- [Apple Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/)

---

**Last Updated:** 2026-01-22
**Version:** 1.0
**Maintainer:** Development Team
