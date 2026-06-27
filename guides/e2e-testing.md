# End-to-End Testing Guidelines
Mandatory standards for the end-to-end (UI) test tier: Playwright/Cypress patterns, page objects, stable locators, fixtures, flake elimination, visual regression, and running e2e in CI. Playwright 1.50+, Cypress 14+, @axe-core/playwright.

---
name: e2e-testing
title: End-to-End Testing Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [playwright@1.50, cypress@14, "@axe-core/playwright", percy]
requires: []
recommends:
  - tdd
  - ci-cd
  - accessibility
  - observability
provides:
  - e2e-patterns
  - page-objects
  - flake-elimination
  - visual-regression
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns the **e2e / UI tier only** — the unit and integration tiers below it belong to [`tdd.md`](guides://tdd.md).

---

## 0. Prerequisites & References

This guide has **no hard prerequisites**, but it sits on top of several owners. Fetch them when the task touches their concern; this guide assumes their rules and does not repeat them.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`tdd.md`](guides://tdd.md) — the test pyramid and the unit/integration tiers **below** e2e, Red-Green-Refactor, regression-test-before-fix, coverage. *(This guide owns only the top tier.)*
> - [`ci-cd.md`](guides://ci-cd.md) — pipeline stages, gating, artifact retention, parallelism policy. *(e2e binding: where the e2e job runs and what gates the merge.)*
> - [`accessibility.md`](guides://accessibility.md) — WCAG scope, a11y rules and severity. *(e2e binding: assert with `@axe-core/playwright` in the e2e suite.)*
> - [`observability.md`](guides://observability.md) — metrics/tracing policy. *(e2e binding: emit flake-rate and duration metrics from the runner.)*

> 📎 **SEE ALSO:** [`code-review.md`](guides://code-review.md) · [`env-config.md`](guides://env-config.md) — base URLs / credentials come from config, never hardcoded. UI-level behaviour and component conventions live in the relevant framework guide (e.g. [`reactjs.md`](guides://reactjs.md), [`nextjs.md`](guides://nextjs.md), [`ui.md`](guides://ui.md)).

---

## 1. Core Philosophies: E2E-FIRST

E2E-specific principles only. The pyramid itself, coverage, and regression-before-fix are owned by [`tdd.md`](guides://tdd.md) — not restated here.

- **E**ssential paths only: e2e is the **smallest, slowest** tier — test critical user journeys end-to-end; push edge cases down to unit/integration (see `tdd.md`).
- **2**-way state setup: arrange state through the **fastest reliable door** (API/DB seed), assert through the UI. Never drive 6 screens to reach the screen under test.
- **E**xplicit waits: assert on conditions (web-first auto-waiting assertions), never fixed `sleep`/`waitForTimeout`.
- **F**lake-free: a test that needs a retry to pass is a **defect**, not noise — quarantine and fix it (see §7).
- **I**solated & idempotent: each test creates and tears down its own data; tests pass in any order, in parallel, run twice.
- **R**esilient locators: user-facing role/label/text first; `data-testid` for ambiguous nodes; CSS/XPath as last resort (see §4).
- **S**ealed inputs: freeze clock, control randomness, stub third-party calls — the SUT is deterministic for a given input.
- **T**iered in CI: e2e runs on every PR, sharded and headless, gating merge (see `ci-cd.md`).

**Verified Code**: agent-generated e2e suites MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `E2E-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| E2E-TST-01 | E2E suite MUST be green with **0 retries needed** to pass | `npx playwright test` (CI: `retries=0` audit run) | exit 0, no test passed-on-retry |
| E2E-TST-02 | Each user-facing bug MUST get a failing e2e/regression test before the fix (see `tdd.md`) | run new spec on old build | failing→passing |
| E2E-SCOPE-01 | E2E MUST cover only critical journeys; non-journey logic MUST live in lower tiers (see `tdd.md`) | review test inventory | no edge cases at e2e tier |
| E2E-WAIT-01 | Tests MUST NOT use arbitrary sleeps (`waitForTimeout`, `cy.wait(<ms>)`) | `grep -rE "waitForTimeout|cy\.wait\([0-9]" e2e/` | 0 matches |
| E2E-LOC-01 | Locators MUST be role/label/text or `data-testid`; raw XPath MUST NOT be used | `grep -rE "xpath=|//" e2e/ pages/` | 0 matches |
| E2E-POM-01 | Page/screen interactions MUST be encapsulated in page objects/fixtures, not inlined per spec | review | no raw selectors in `*.spec.*` |
| E2E-ISO-01 | Each test MUST set up & tear down its own data; suite MUST pass in parallel and when re-run | `npx playwright test --workers=4 --repeat-each=2` | exit 0 |
| E2E-CFG-01 | Base URL & credentials MUST come from config/env, never hardcoded (see `env-config.md`) | `grep -rE "https?://(localhost|[0-9])" e2e/` | only via `baseURL`/env |
| E2E-VIS-01 | Visual baselines (if used) MUST be deterministic: clock frozen, animations disabled, dynamic content masked | review config + run twice | byte-stable diff |
| E2E-A11Y-01 | Critical pages SHOULD assert 0 a11y violations in e2e (see `accessibility.md`) | `npx playwright test a11y` | `violations == []` |
| E2E-CI-01 | E2E suite MUST run on every PR, sharded, headless, gating merge (see `ci-cd.md`) | CI config review | required check present |
| E2E-CI-02 | Failure artifacts (trace, video, screenshot) MUST be retained on failure | inspect pipeline artifacts | uploaded on failure |

> **Forbidden**: shipping a flaky test masked by retries (violates E2E-TST-01), arbitrary sleeps, XPath locators, tests that depend on execution order or leftover data, hardcoded URLs/credentials, or driving the full UI to set up state that an API/DB seed can establish.

---

## 3. Verification Protocol

Run, in order, before presenting an e2e change. Fix → re-run until every gate is green.

```bash
grep -rE "waitForTimeout|cy\.wait\([0-9]" e2e/ pages/   # E2E-WAIT-01
grep -rE "xpath=|//[a-z]" e2e/ pages/                    # E2E-LOC-01
npx playwright test --workers=4 --repeat-each=2          # E2E-ISO-01 (order/parallel/idempotency)
npx playwright test --retries=0                          # E2E-TST-01 (no hidden flake)
npx playwright test a11y                                 # E2E-A11Y-01 (see accessibility.md)
```

The *why* behind the pyramid, coverage, CI gating, and a11y severity lives in their §0 owners; do not re-derive it here.

---

## 4. Locator Strategy

The single biggest lever on suite stability. Prefer locators a user (and a screen reader) understands; they survive refactors and double as accessibility coverage.

```typescript
// Priority — most to least preferred:
page.getByRole('button', { name: 'Submit' });   // 1. role + accessible name (also a11y signal)
page.getByLabel('Email address');                // 1. form controls by label
page.getByText('Welcome back');                  // 2. stable, unique visible text
page.getByTestId('order-row');                   // 3. data-testid for ambiguous/dynamic nodes
page.locator('.order-table tbody tr');           // 4. CSS — only when nothing semantic fits
// XPath — FORBIDDEN (E2E-LOC-01): brittle, opaque, breaks on any DOM reshuffle
```

Rules:
- Scope locators to a container (`row.getByRole(...)`) instead of global, brittle CSS chains.
- `data-testid` is a contract: add it in the component, keep it stable, never reuse it for styling. Configure `testIdAttribute` once in config.
- Never select on auto-generated/hashed classes (`.css-1a2b3c`) or DOM position (`nth-child`).
- Role-first locators give you a11y coverage for free — see `accessibility.md` for the assertion policy.

---

## 5. Page-Object Model & Fixtures

Encapsulate *where things are* and *how to act* so specs read as user intent. Specs assert; page objects locate and interact (E2E-POM-01).

```typescript
// pages/base.page.ts — shared chrome
import { Page, Locator, expect } from '@playwright/test';

export abstract class BasePage {
  protected readonly toast: Locator;
  constructor(protected readonly page: Page) {
    this.toast = page.getByTestId('toast');
  }
  abstract readonly path: string;
  async goto() { await this.page.goto(this.path); }
  async expectToast(msg: string) { await expect(this.toast).toContainText(msg); }
}

// pages/login.page.ts
import { Page, expect } from '@playwright/test';
import { BasePage } from './base.page';

export class LoginPage extends BasePage {
  readonly path = '/login';
  private email = this.page.getByLabel('Email address');
  private password = this.page.getByLabel('Password');
  private submit = this.page.getByRole('button', { name: 'Log in' });

  async login(email: string, password: string) {
    await this.email.fill(email);
    await this.password.fill(password);
    await this.submit.click();
  }
  async expectError(msg: string) {
    await expect(this.page.getByRole('alert')).toContainText(msg);
  }
}
```

Promote page objects to **custom fixtures** so specs declare what they need and get pre-wired, auto-torn-down objects:

```typescript
// fixtures.ts
import { test as base } from '@playwright/test';
import { LoginPage } from './pages/login.page';

export const test = base.extend<{ loginPage: LoginPage }>({
  loginPage: async ({ page }, use) => { await use(new LoginPage(page)); },
});
export { expect } from '@playwright/test';
```

Conventions: one page object per screen/significant component; expose intent methods (`login`, `addToCart`), not raw clicks; return the next page object on navigation; keep assertions thin (`expectError`) but let specs own the *what*. Suggested layout:

```
e2e/
├── fixtures.ts            # custom test fixtures (page objects, seeded data, auth)
├── pages/                 # page objects (one per screen/component)
├── support/               # API/DB seed + teardown helpers
└── specs/<journey>/*.spec.ts
```

---

## 6. Test Data & State Setup

Arrange state through the fastest reliable door; assert through the UI (E2E-ISO-01).

- **Seed via API/DB, assert via UI.** Building state by clicking through screens is slow and flaky.
- **Own your data.** Each test creates uniquely-keyed data (e.g. `user+${uuid}@test.com`) and tears it down — no shared mutable fixtures, no reliance on a previous test.
- **Static fixtures** (`fixtures/*.json|.ts`) are for read-only reference data only; anything a test mutates must be created per-test.
- **Authenticate once, reuse session.** Save storage state in setup and reuse it instead of logging in through the form every test.

```typescript
// auth.setup.ts — runs once, persists session
import { test as setup } from '@playwright/test';
setup('authenticate', async ({ page }) => {
  await page.goto('/login');
  await page.getByLabel('Email address').fill(process.env.E2E_USER!);     // from env (E2E-CFG-01)
  await page.getByLabel('Password').fill(process.env.E2E_PASS!);
  await page.getByRole('button', { name: 'Log in' }).click();
  await page.context().storageState({ path: '.auth/user.json' });
});

// stub third-party / unstable upstreams at the network boundary
test('handles upstream 500 gracefully', async ({ page }) => {
  await page.route('**/api/users', r => r.fulfill({ status: 500, body: '{}' }));
  await page.goto('/users');
  await expect(page.getByRole('alert')).toContainText('Unable to load users');
});
```

Cypress equivalent: `cy.session()` to cache login; `cy.intercept()` to stub; seed via `cy.task()` hitting the DB.

---

## 7. Flake Elimination

A test that passes only on retry is a defect (E2E-TST-01). Retries are a *detector and safety net for CI*, never a fix. Root-cause every flake; quarantine (skip + ticket) only to keep the suite green while you fix it.

| Root cause | Fix |
|---|---|
| Asserting before data lands | Wait on the populating response/state, then assert: `await page.waitForResponse(r => r.url().includes('/api/dashboard'))`. Lean on web-first auto-retrying assertions (`await expect(loc).toHaveText(...)`). |
| Arbitrary `sleep` | Delete it. Wait for a condition (`toBeVisible`, `toHaveURL`), never a duration (E2E-WAIT-01). |
| Overlay/cookie banner intercepts click | Dismiss the overlay first; never paper over with `{ force: true }`. |
| Animations/transitions mid-action | Disable globally in tests: inject CSS zeroing `animation/transition-duration`. |
| Clock/`Date`/timezone drift | Freeze time: `await page.clock.setFixedTime(new Date('2026-01-15T10:00:00Z'))`. |
| Shared/order-dependent state | Per-test setup + teardown; run with `--repeat-each` and shuffled order to prove isolation. |
| Element below the fold / virtualized | `scrollIntoViewIfNeeded()` before interacting. |
| Animation/network race on navigation | Wait for the specific network/DOM signal, not `networkidle` as a blanket crutch. |

Track flake rate as a metric (see `observability.md`): emit per-test pass/fail/retry counts from a custom reporter and alert when a test both passes and fails across runs. Quarantined tests MUST carry a ticket reference and a deadline.

---

## 8. Visual Regression

Pixel comparison catches unintended UI changes that functional assertions miss. It is only viable if baselines are **deterministic** (E2E-VIS-01) — otherwise every run is noise.

```typescript
test.beforeEach(async ({ page }) => {
  await page.clock.setFixedTime(new Date('2026-06-15T10:00:00Z'));   // freeze time
  await page.addStyleTag({ content: `*,*::before,*::after{
    animation-duration:0s!important;transition-duration:0s!important;}` });
});

test('dashboard matches baseline', async ({ page }) => {
  await page.route('**/api/dashboard/**', r => r.fulfill({           // pin dynamic data
    status: 200, body: JSON.stringify({ users: 1234, orders: 567 }) }));
  await page.goto('/dashboard');
  await page.getByTestId('live-timestamp').evaluate(el => el.remove()); // mask volatile nodes
  await expect(page).toHaveScreenshot('dashboard.png', {
    fullPage: true, maxDiffPixelRatio: 0.01, animations: 'disabled',
  });
});
```

Rules: generate baselines in the **same environment as CI** (font rendering differs across OSes — run on the Playwright container image or upload from CI). Review baseline diffs in PRs like code. For cross-browser/cross-width coverage at scale, offload to a hosted differ (Percy: `percySnapshot(page, 'Dashboard', { widths: [375, 768, 1280] })`) instead of committing thousands of PNGs. Update intentionally with `--update-snapshots`, never blindly.

---

## 9. Running E2E in CI

The *pipeline policy* (stages, required checks, retention) is owned by [`ci-cd.md`](guides://ci-cd.md). E2E binding (E2E-CI-01/02):

- **Headless, on every PR**, as a required check gating merge.
- **Shard** across parallel jobs; merge blob reports into one HTML report.
- **CI retries = a flake detector, not a pass.** Keep `retries: process.env.CI ? 2 : 0`, but also run an audit lane at `--retries=0` to enforce E2E-TST-01; surface any passed-on-retry as a failure signal.
- **Always upload trace + video + screenshot on failure** — these are how you debug a CI-only failure.
- Use the official `mcr.microsoft.com/playwright:v1.50.0-noble` image to pin browser + OS for stable visual baselines.

```yaml
# .github/workflows/e2e.yml — minimal binding; full pipeline policy in ci-cd.md
jobs:
  e2e:
    runs-on: ubuntu-latest
    strategy: { fail-fast: false, matrix: { shard: [1, 2, 3, 4] } }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '22', cache: 'npm' }
      - run: npm ci
      - run: npx playwright install --with-deps chromium
      - run: npx playwright test --shard=${{ matrix.shard }}/4
        env: { CI: 'true', BASE_URL: 'http://localhost:3000' }   # config, not hardcoded
      - if: always()
        uses: actions/upload-artifact@v4
        with: { name: blob-${{ matrix.shard }}, path: blob-report/, retention-days: 7 }
```

```typescript
// playwright.config.ts — the load-bearing CI knobs
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,          // no .only sneaks into CI
  retries: process.env.CI ? 2 : 0,       // detector + audit lane enforces TST-01
  reporter: process.env.CI ? [['blob'], ['github']] : [['html']],
  use: {
    baseURL: process.env.BASE_URL,       // E2E-CFG-01: from env
    testIdAttribute: 'data-testid',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
});
```

---

## 10. Quick Reference

```bash
npx playwright test                       # run all (headless)
npx playwright test --ui                  # interactive UI mode (local debugging)
npx playwright test --debug               # step debugger
npx playwright codegen localhost:3000     # record locators/actions
npx playwright show-report                # open HTML report
npx playwright test --update-snapshots    # refresh visual baselines (intentional only)
npx playwright merge-reports --reporter html ./all-blob-reports  # merge shards
# Cypress: npx cypress open | npx cypress run --browser chrome
```

---

## 11. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements here.

- [ ] E2E-TST-01 — suite green with 0 retries needed (audit lane at `--retries=0`)
- [ ] E2E-TST-02 — each user-facing bug has a failing-first e2e regression test
- [ ] E2E-SCOPE-01 — only critical journeys at e2e; edge cases pushed to lower tiers
- [ ] E2E-WAIT-01 — no arbitrary sleeps/`waitForTimeout`
- [ ] E2E-LOC-01 — role/label/text or `data-testid` only; no XPath
- [ ] E2E-POM-01 — interactions encapsulated in page objects/fixtures
- [ ] E2E-ISO-01 — self-contained data; passes in parallel and on re-run
- [ ] E2E-CFG-01 — base URL & credentials from config/env
- [ ] E2E-VIS-01 — visual baselines deterministic (clock/animations/dynamic data)
- [ ] E2E-A11Y-01 — critical pages assert 0 a11y violations
- [ ] E2E-CI-01 — e2e runs on every PR, sharded, headless, gates merge
- [ ] E2E-CI-02 — trace/video/screenshot retained on failure
- [ ] Agent ran every §3 command and documented any fixes

---
**End of End-to-End Testing Guidelines**
