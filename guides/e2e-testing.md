# End-to-End Testing Guidelines

This document provides mandatory standards for end-to-end (E2E) testing using modern testing frameworks like Playwright, Cypress, and Selenium.

---

**Agent Profile**: The E2E Testing Specialist
**Role**: Senior QA Engineer & Test Automation Architect
**Objective**: Generate reliable, maintainable, and fast end-to-end tests that validate complete user workflows.
**Tools**: Playwright, Cypress, Selenium WebDriver, TestCafe, Puppeteer.

---

## 1. Core Philosophies: E2E-FIRST

The agent must adhere to the **E2E-FIRST** principles:

- **E**ssential Paths Only: Test critical user journeys, not every possible path
- **2**-Layer Strategy: Combine E2E with unit/integration tests (test pyramid)
- **E**xplicit Waits: Never use arbitrary sleeps; wait for specific conditions
- **F**lake-Free: Eliminate test flakiness through proper synchronization
- **I**solated Tests: Each test should be independent and repeatable
- **R**ealistic Data: Use production-like test data and environments
- **S**electors Strategy: Use stable, semantic selectors (data-testid preferred)
- **T**imed Appropriately: Run E2E in CI but optimize for speed

---

## 2. Framework Selection

### A. Playwright (Recommended)

```typescript
// playwright.config.ts
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html'],
    ['junit', { outputFile: 'results/junit.xml' }]
  ],
  use: {
    baseURL: process.env.BASE_URL || 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
    { name: 'mobile', use: { ...devices['iPhone 13'] } },
  ],
  webServer: {
    command: 'npm run start',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
  },
});
```

### B. Cypress

```javascript
// cypress.config.js
const { defineConfig } = require('cypress');

module.exports = defineConfig({
  e2e: {
    baseUrl: 'http://localhost:3000',
    supportFile: 'cypress/support/e2e.js',
    specPattern: 'cypress/e2e/**/*.cy.{js,jsx,ts,tsx}',
    viewportWidth: 1280,
    viewportHeight: 720,
    video: true,
    screenshotOnRunFailure: true,
    retries: {
      runMode: 2,
      openMode: 0,
    },
    env: {
      apiUrl: 'http://localhost:3001/api',
    },
  },
});
```

### C. Selenium WebDriver

```python
# conftest.py
import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

@pytest.fixture(scope="session")
def driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(10)
    yield driver
    driver.quit()
```

---

## 3. Test Structure (MANDATORY)

### A. Page Object Model

```typescript
// pages/login.page.ts
import { Page, Locator, expect } from '@playwright/test';

export class LoginPage {
  readonly page: Page;
  readonly emailInput: Locator;
  readonly passwordInput: Locator;
  readonly submitButton: Locator;
  readonly errorMessage: Locator;

  constructor(page: Page) {
    this.page = page;
    this.emailInput = page.getByTestId('email-input');
    this.passwordInput = page.getByTestId('password-input');
    this.submitButton = page.getByTestId('login-submit');
    this.errorMessage = page.getByTestId('error-message');
  }

  async goto() {
    await this.page.goto('/login');
  }

  async login(email: string, password: string) {
    await this.emailInput.fill(email);
    await this.passwordInput.fill(password);
    await this.submitButton.click();
  }

  async expectError(message: string) {
    await expect(this.errorMessage).toContainText(message);
  }

  async expectLoggedIn() {
    await expect(this.page).toHaveURL(/.*dashboard/);
  }
}
```

### B. Test Organization

```
e2e/
├── fixtures/
│   ├── users.json
│   └── test-data.ts
├── pages/
│   ├── login.page.ts
│   ├── dashboard.page.ts
│   └── checkout.page.ts
├── support/
│   ├── commands.ts
│   └── helpers.ts
├── specs/
│   ├── auth/
│   │   ├── login.spec.ts
│   │   └── logout.spec.ts
│   ├── checkout/
│   │   └── purchase.spec.ts
│   └── user/
│       └── profile.spec.ts
└── playwright.config.ts
```

---

## 4. Selector Strategy (MANDATORY)

### A. Selector Priority

```typescript
// Priority order (most to least preferred):

// 1. data-testid (PREFERRED - most stable)
page.getByTestId('submit-button')

// 2. Accessibility roles and labels
page.getByRole('button', { name: 'Submit' })
page.getByLabel('Email address')
page.getByPlaceholder('Enter your email')

// 3. Text content (for static, unique text)
page.getByText('Welcome back')

// 4. CSS selectors (when necessary)
page.locator('.unique-class-name')

// 5. XPath (AVOID - fragile)
// Only use as last resort
```

### B. Adding Test IDs

```tsx
// React component with test IDs
function LoginForm() {
  return (
    <form data-testid="login-form">
      <input
        data-testid="email-input"
        type="email"
        aria-label="Email address"
      />
      <input
        data-testid="password-input"
        type="password"
        aria-label="Password"
      />
      <button data-testid="login-submit" type="submit">
        Log In
      </button>
    </form>
  );
}
```

---

## 5. Waiting Strategies (MANDATORY)

### A. Explicit Waits

```typescript
// ✅ CORRECT - Wait for specific conditions
await page.waitForSelector('[data-testid="results"]');
await page.waitForURL('**/dashboard');
await page.waitForLoadState('networkidle');
await page.waitForResponse(resp => resp.url().includes('/api/users'));

// Wait for element state
await expect(element).toBeVisible();
await expect(element).toBeEnabled();
await expect(element).toHaveText('Expected text');

// ❌ WRONG - Arbitrary sleep
await page.waitForTimeout(5000); // NEVER do this!
```

### B. Custom Wait Helpers

```typescript
// support/helpers.ts
export async function waitForApi(page: Page, urlPattern: string) {
  return page.waitForResponse(
    response => response.url().includes(urlPattern) && response.status() === 200
  );
}

export async function waitForTableLoad(page: Page) {
  await page.waitForSelector('[data-testid="table-body"]');
  await page.waitForFunction(() => {
    const rows = document.querySelectorAll('[data-testid="table-row"]');
    return rows.length > 0;
  });
}

export async function retryUntil(
  action: () => Promise<boolean>,
  maxAttempts = 5,
  delay = 1000
): Promise<boolean> {
  for (let i = 0; i < maxAttempts; i++) {
    if (await action()) return true;
    await new Promise(r => setTimeout(r, delay));
  }
  return false;
}
```

---

## 6. Test Patterns (MANDATORY)

### A. Authentication Flow

```typescript
// e2e/specs/auth/login.spec.ts
import { test, expect } from '@playwright/test';
import { LoginPage } from '../pages/login.page';

test.describe('Authentication', () => {
  test.beforeEach(async ({ page }) => {
    // Clear any existing session
    await page.context().clearCookies();
  });

  test('successful login redirects to dashboard', async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.login('user@example.com', 'validPassword123');
    await loginPage.expectLoggedIn();
  });

  test('invalid credentials show error message', async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.login('user@example.com', 'wrongPassword');
    await loginPage.expectError('Invalid email or password');
  });

  test('empty form shows validation errors', async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.submitButton.click();
    await expect(page.getByText('Email is required')).toBeVisible();
    await expect(page.getByText('Password is required')).toBeVisible();
  });
});
```

### B. CRUD Operations

```typescript
// e2e/specs/user/profile.spec.ts
import { test, expect } from '@playwright/test';

test.describe('User Profile', () => {
  test.use({ storageState: 'auth.json' }); // Pre-authenticated

  test('can update profile information', async ({ page }) => {
    await page.goto('/profile');

    // Update name
    await page.getByTestId('name-input').fill('New Name');
    await page.getByTestId('save-button').click();

    // Verify success
    await expect(page.getByTestId('success-toast')).toBeVisible();

    // Verify persistence (reload and check)
    await page.reload();
    await expect(page.getByTestId('name-input')).toHaveValue('New Name');
  });

  test('can delete account with confirmation', async ({ page }) => {
    await page.goto('/profile/settings');

    await page.getByTestId('delete-account-button').click();

    // Confirmation dialog
    await expect(page.getByRole('dialog')).toBeVisible();
    await page.getByTestId('confirm-delete').click();

    // Should redirect to home
    await expect(page).toHaveURL('/');
  });
});
```

### C. E-Commerce Checkout Flow

```typescript
// e2e/specs/checkout/purchase.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Checkout Flow', () => {
  test('complete purchase flow', async ({ page }) => {
    // Step 1: Browse and add to cart
    await page.goto('/products');
    await page.getByTestId('product-card').first().click();
    await page.getByTestId('add-to-cart').click();
    await expect(page.getByTestId('cart-count')).toHaveText('1');

    // Step 2: Go to cart
    await page.getByTestId('cart-icon').click();
    await expect(page.getByTestId('cart-item')).toHaveCount(1);

    // Step 3: Proceed to checkout
    await page.getByTestId('checkout-button').click();

    // Step 4: Fill shipping information
    await page.getByTestId('shipping-name').fill('John Doe');
    await page.getByTestId('shipping-address').fill('123 Main St');
    await page.getByTestId('shipping-city').fill('New York');
    await page.getByTestId('shipping-zip').fill('10001');
    await page.getByTestId('continue-to-payment').click();

    // Step 5: Fill payment information
    await page.getByTestId('card-number').fill('4111111111111111');
    await page.getByTestId('card-expiry').fill('12/28');
    await page.getByTestId('card-cvc').fill('123');

    // Step 6: Place order
    await page.getByTestId('place-order').click();

    // Step 7: Verify confirmation
    await expect(page.getByTestId('order-confirmation')).toBeVisible();
    await expect(page.getByTestId('order-number')).toBeVisible();
  });
});
```

---

## 7. Test Data Management

### A. Fixtures

```typescript
// fixtures/users.ts
export const testUsers = {
  admin: {
    email: 'admin@test.com',
    password: 'AdminPass123!',
    role: 'admin',
  },
  regular: {
    email: 'user@test.com',
    password: 'UserPass123!',
    role: 'user',
  },
  premium: {
    email: 'premium@test.com',
    password: 'PremiumPass123!',
    role: 'premium',
  },
};

// fixtures/products.ts
export const testProducts = {
  basic: {
    name: 'Test Product',
    price: 29.99,
    sku: 'TEST-001',
  },
};
```

### B. Database Seeding

```typescript
// support/database.ts
import { prisma } from './prisma';

export async function seedTestData() {
  // Clean up
  await prisma.order.deleteMany();
  await prisma.user.deleteMany();

  // Seed users
  await prisma.user.createMany({
    data: [
      { email: 'admin@test.com', role: 'ADMIN' },
      { email: 'user@test.com', role: 'USER' },
    ],
  });
}

export async function cleanupTestData() {
  await prisma.order.deleteMany();
  await prisma.user.deleteMany({ where: { email: { contains: '@test.com' } } });
}
```

### C. API Mocking

```typescript
// Playwright route mocking
test('handles API error gracefully', async ({ page }) => {
  // Mock API to return error
  await page.route('**/api/users', route => {
    route.fulfill({
      status: 500,
      body: JSON.stringify({ error: 'Internal Server Error' }),
    });
  });

  await page.goto('/users');
  await expect(page.getByTestId('error-message')).toContainText('Unable to load users');
});

// Mock successful response
test('displays user list', async ({ page }) => {
  await page.route('**/api/users', route => {
    route.fulfill({
      status: 200,
      body: JSON.stringify([
        { id: 1, name: 'John Doe', email: 'john@example.com' },
        { id: 2, name: 'Jane Doe', email: 'jane@example.com' },
      ]),
    });
  });

  await page.goto('/users');
  await expect(page.getByTestId('user-row')).toHaveCount(2);
});
```

---

## 8. Authentication Handling

### A. Storage State (Playwright)

```typescript
// Setup authentication state
// auth.setup.ts
import { test as setup, expect } from '@playwright/test';

setup('authenticate', async ({ page }) => {
  await page.goto('/login');
  await page.getByTestId('email-input').fill('user@example.com');
  await page.getByTestId('password-input').fill('password123');
  await page.getByTestId('login-submit').click();

  await expect(page).toHaveURL(/.*dashboard/);

  // Save authentication state
  await page.context().storageState({ path: 'auth.json' });
});

// Use in tests
test.describe('Authenticated tests', () => {
  test.use({ storageState: 'auth.json' });

  test('can access dashboard', async ({ page }) => {
    await page.goto('/dashboard');
    await expect(page.getByTestId('welcome-message')).toBeVisible();
  });
});
```

### B. Custom Commands (Cypress)

```javascript
// cypress/support/commands.js
Cypress.Commands.add('login', (email, password) => {
  cy.session([email, password], () => {
    cy.visit('/login');
    cy.get('[data-testid="email-input"]').type(email);
    cy.get('[data-testid="password-input"]').type(password);
    cy.get('[data-testid="login-submit"]').click();
    cy.url().should('include', '/dashboard');
  });
});

// Usage in tests
describe('Dashboard', () => {
  beforeEach(() => {
    cy.login('user@example.com', 'password123');
  });

  it('shows user data', () => {
    cy.visit('/dashboard');
    cy.get('[data-testid="user-name"]').should('be.visible');
  });
});
```

---

## 9. CI/CD Integration

### A. GitHub Actions

```yaml
# .github/workflows/e2e.yml
name: E2E Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Install Playwright browsers
        run: npx playwright install --with-deps

      - name: Build application
        run: npm run build

      - name: Run E2E tests
        run: npx playwright test
        env:
          BASE_URL: http://localhost:3000
          CI: true

      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: playwright-report
          path: playwright-report/
          retention-days: 30

      - name: Upload test videos
        uses: actions/upload-artifact@v4
        if: failure()
        with:
          name: test-videos
          path: test-results/
          retention-days: 7
```

### B. Docker Support

```dockerfile
# Dockerfile.e2e
FROM mcr.microsoft.com/playwright:v1.40.0-jammy

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

CMD ["npx", "playwright", "test"]
```

```yaml
# docker-compose.e2e.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgres://test:test@db:5432/test

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
      POSTGRES_DB: test

  e2e:
    build:
      dockerfile: Dockerfile.e2e
    depends_on:
      - app
    environment:
      - BASE_URL=http://app:3000
    volumes:
      - ./playwright-report:/app/playwright-report
```

---

## 10. Flakiness Prevention

### A. Common Causes and Solutions

```typescript
// Problem 1: Race conditions with animations
// Solution: Disable animations in tests
await page.addStyleTag({
  content: `
    *, *::before, *::after {
      animation-duration: 0s !important;
      transition-duration: 0s !important;
    }
  `,
});

// Problem 2: Element not yet interactive
// Solution: Wait for element to be ready
await page.getByTestId('button').waitFor({ state: 'visible' });
await page.getByTestId('button').click();

// Problem 3: Network timing issues
// Solution: Wait for network idle
await page.waitForLoadState('networkidle');

// Problem 4: Dynamic content loading
// Solution: Wait for specific content
await page.waitForFunction(() => {
  const items = document.querySelectorAll('[data-testid="list-item"]');
  return items.length >= 5;
});

// Problem 5: Date/time dependent tests
// Solution: Mock the clock
await page.clock.setFixedTime(new Date('2024-01-15T10:00:00'));
```

### B. Retry Strategies

```typescript
// playwright.config.ts
export default defineConfig({
  retries: process.env.CI ? 2 : 0, // Retry failed tests in CI

  expect: {
    timeout: 10000, // Increase assertion timeout
  },

  use: {
    actionTimeout: 15000, // Increase action timeout
  },
});

// Test-specific retry
test('flaky external service', async ({ page }) => {
  test.slow(); // Mark as slow, triple the timeout

  // Or set specific timeout
  test.setTimeout(60000);

  // Implementation
});
```

---

## 11. Visual Regression Testing

### A. Playwright Screenshots

```typescript
test('visual regression - homepage', async ({ page }) => {
  await page.goto('/');
  await page.waitForLoadState('networkidle');

  // Full page screenshot comparison
  await expect(page).toHaveScreenshot('homepage.png', {
    fullPage: true,
    maxDiffPixelRatio: 0.01,
  });
});

test('visual regression - component', async ({ page }) => {
  await page.goto('/components');

  // Element screenshot comparison
  const card = page.getByTestId('product-card').first();
  await expect(card).toHaveScreenshot('product-card.png');
});
```

### B. Percy Integration

```typescript
// With Percy for cross-browser visual testing
import percySnapshot from '@percy/playwright';

test('visual test with Percy', async ({ page }) => {
  await page.goto('/dashboard');
  await page.waitForLoadState('networkidle');

  await percySnapshot(page, 'Dashboard');
});
```

---

## 12. Accessibility Testing

```typescript
import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('Accessibility', () => {
  test('homepage has no accessibility violations', async ({ page }) => {
    await page.goto('/');

    const accessibilityScanResults = await new AxeBuilder({ page }).analyze();

    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test('login form is accessible', async ({ page }) => {
    await page.goto('/login');

    const results = await new AxeBuilder({ page })
      .include('[data-testid="login-form"]')
      .analyze();

    expect(results.violations).toEqual([]);
  });
});
```

---

## 13. Performance Testing

```typescript
test('page load performance', async ({ page }) => {
  await page.goto('/');

  const performanceMetrics = await page.evaluate(() => {
    const timing = performance.timing;
    return {
      loadTime: timing.loadEventEnd - timing.navigationStart,
      domContentLoaded: timing.domContentLoadedEventEnd - timing.navigationStart,
      firstPaint: performance.getEntriesByName('first-paint')[0]?.startTime,
    };
  });

  expect(performanceMetrics.loadTime).toBeLessThan(3000);
  expect(performanceMetrics.domContentLoaded).toBeLessThan(2000);
});

test('no memory leaks during navigation', async ({ page }) => {
  const initialMemory = await page.evaluate(() =>
    (performance as any).memory?.usedJSHeapSize
  );

  // Navigate through multiple pages
  for (const path of ['/page1', '/page2', '/page3', '/page1']) {
    await page.goto(path);
    await page.waitForLoadState('networkidle');
  }

  const finalMemory = await page.evaluate(() =>
    (performance as any).memory?.usedJSHeapSize
  );

  // Memory shouldn't grow more than 50%
  expect(finalMemory).toBeLessThan(initialMemory * 1.5);
});
```

---

## 14. Deployment Checklist

### Test Coverage
- [ ] Critical user journeys covered
- [ ] Authentication flows tested
- [ ] Error states handled
- [ ] Cross-browser testing configured
- [ ] Mobile viewport testing included

### Test Quality
- [ ] Page Object Model implemented
- [ ] Stable selectors (data-testid) used
- [ ] No arbitrary waits/sleeps
- [ ] Tests are independent
- [ ] Proper test data management

### CI/CD Integration
- [ ] Tests run on every PR
- [ ] Parallel execution configured
- [ ] Artifacts uploaded on failure
- [ ] Test reports generated
- [ ] Flaky test monitoring in place

### Accessibility
- [ ] Axe accessibility checks included
- [ ] Keyboard navigation tested
- [ ] Screen reader compatibility verified

---

## 15. Quick Reference

### Playwright Commands

```bash
# Run all tests
npx playwright test

# Run specific file
npx playwright test login.spec.ts

# Run in headed mode
npx playwright test --headed

# Run with UI mode
npx playwright test --ui

# Debug mode
npx playwright test --debug

# Generate tests
npx playwright codegen localhost:3000

# Show report
npx playwright show-report

# Update snapshots
npx playwright test --update-snapshots
```

### Cypress Commands

```bash
# Open Cypress UI
npx cypress open

# Run headless
npx cypress run

# Run specific spec
npx cypress run --spec "cypress/e2e/login.cy.js"

# Run in specific browser
npx cypress run --browser chrome
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** QA Team
