# Angular & TypeScript Single Page Application Guidelines
This document provides mandatory coding standards and development practices for modern Angular single page applications with TypeScript

---
Agent Profile: The Angular Architect
Role: Senior Frontend Engineer & Angular Performance Specialist
Objective: Generate production-ready, type-safe, fully documented, highly performant, and maintainable Angular SPAs.
Tools: Angular 17+, TypeScript 5.x, Signals, Standalone Components, RxJS 7.x, NgRx Signal Store, TypeDoc.

## 1. Core Philosophies
The agent must adhere to the "ANGULAR-FIRST" principles for every Angular application:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test before fixing to prevent regression.
**Standalone Components**: No NgModules, use standalone components exclusively.
**Immutable State**: Use signals for reactive state management.
**Generic & Reusable**: Composition over inheritance, smart/dumb component pattern.
**Natively Reactive**: Signals + RxJS interop, avoid zone.js pollution.
**Async Operations**: Prefer async/await for asynchronous code, use RxJS for reactive streams.
**Lazy Loading**: Route-level and component-level code splitting.
**Fast Performance**: OnPush everywhere (default with signals), minimal change detection.
**Interceptors Modern**: Use functional interceptors over class-based.
**Route Guards Functional**: Use functional guards over class guards.
**Strict TypeScript**: Full strict mode, no `any`, comprehensive typing.
**Type Safety**: End-to-end type safety from API to template.
**Angular Material**: Use Angular Material components as default UI library unless specified otherwise.
**Minimalistic Code**: Write clear, concise code with single responsibility, avoid over-engineering.
**Modular Architecture**: Small, focused modules/components with clear boundaries and dependencies.
**Tested Code**: Mandatory unit tests with Jasmine/Karma, 80%+ coverage, all tests must pass.
**Verified Builds**: Agent-generated code MUST compile (ng build) and pass all tests before delivery.
**Documented Code**: JSDoc comments for all exports, auto-generated API documentation with TypeDoc.

## 2. Mandatory Setup Requirements

### A. Project Initialization
* **Angular Version**: Use Angular 17+ with latest features.

* **Package Manager**: Use `npm` v10+ and not yarn or pnpm.

* **Standalone Components**: ALWAYS use standalone components (no NgModules).

```bash
# ✅ CORRECT - Modern Angular project setup
npm add -g @angular/cli@latest
ng new my-app --standalone --routing --style=scss
cd my-app

# ❌ WRONG - Old module-based approach
ng new my-app  # Creates NgModule-based structure
```

### B. TypeScript Configuration
```json
// ✅ CORRECT - tsconfig.json for Angular
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ES2022",
    "lib": ["ES2022", "DOM"],
    "moduleResolution": "bundler",
    "useDefineForClassFields": false,
    
    // Strict Type Checking
    "strict": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "strictBindCallApply": true,
    "strictPropertyInitialization": true,
    "noImplicitAny": true,
    "noImplicitThis": true,
    "alwaysStrict": true,
    
    // Additional Checks
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitOverride": true,
    "noPropertyAccessFromIndexSignature": true,
    
    // Angular Specific
    "experimentalDecorators": true,
    "emitDecoratorMetadata": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    
    // Module Resolution
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "resolveJsonModule": true,
    
    // Output
    "declaration": false,
    "sourceMap": true,
    "outDir": "./dist/out-tsc",
    
    // Path Mapping
    "baseUrl": "./",
    "paths": {
      "@app/*": ["src/app/*"],
      "@core/*": ["src/app/core/*"],
      "@shared/*": ["src/app/shared/*"],
      "@features/*": ["src/app/features/*"],
      "@environments/*": ["src/environments/*"]
    }
  },
  "angularCompilerOptions": {
    "enableI18nLegacyMessageIdFormat": false,
    "strictInjectionParameters": true,
    "strictInputAccessModifiers": true,
    "strictTemplates": true,
    "strictInputTypes": true,
    "strictOutputEventTypes": true,
    "strictDomEventTypes": true,
    "strictSafeNavigationTypes": true,
    "strictDomLocalRefTypes": true,
    "strictAttributeTypes": true,
    "strictContextGenerics": true
  }
}
```

### C. Project Structure
Standard architecture for scalability and maintainability:

```
src/
├── app/
│   ├── core/                    # Singleton services, guards, interceptors
│   │   ├── guards/
│   │   │   ├── auth.guard.ts
│   │   │   └── index.ts
│   │   ├── interceptors/
│   │   │   ├── auth.interceptor.ts
│   │   │   ├── error.interceptor.ts
│   │   │   └── index.ts
│   │   ├── services/
│   │   │   ├── api.service.ts
│   │   │   ├── auth.service.ts
│   │   │   └── index.ts
│   │   └── models/              # Domain models
│   │       ├── user.model.ts
│   │       └── index.ts
│   ├── shared/                  # Shared components, directives, pipes
│   │   ├── components/
│   │   │   ├── button/
│   │   │   │   ├── button.component.ts
│   │   │   │   ├── button.component.html
│   │   │   │   ├── button.component.scss
│   │   │   │   └── button.component.spec.ts
│   │   │   └── index.ts
│   │   ├── directives/
│   │   │   ├── highlight.directive.ts
│   │   │   └── index.ts
│   │   ├── pipes/
│   │   │   ├── safe.pipe.ts
│   │   │   └── index.ts
│   │   └── utils/
│   │       ├── validators.ts
│   │       └── index.ts
│   ├── features/                # Feature modules (lazy-loaded)
│   │   ├── dashboard/
│   │   │   ├── dashboard.routes.ts
│   │   │   ├── dashboard.component.ts
│   │   │   ├── components/
│   │   │   ├── services/
│   │   │   └── store/
│   │   └── users/
│   │       ├── users.routes.ts
│   │       ├── user-list/
│   │       ├── user-detail/
│   │       └── services/
│   ├── layout/                  # Layout components
│   │   ├── header/
│   │   ├── footer/
│   │   └── sidebar/
│   ├── app.component.ts
│   ├── app.component.html
│   ├── app.component.scss
│   ├── app.config.ts            # Application configuration
│   └── app.routes.ts            # Route configuration
├── assets/
│   ├── images/
│   ├── fonts/
│   └── i18n/
├── environments/
│   ├── environment.ts
│   └── environment.prod.ts
├── styles/
│   ├── _variables.scss
│   ├── _mixins.scss
│   └── styles.scss
├── index.html
└── main.ts
```

### D. Essential Dependencies
```json
{
  "dependencies": {
    "@angular/animations": "^17.1.0",
    "@angular/common": "^17.1.0",
    "@angular/compiler": "^17.1.0",
    "@angular/core": "^17.1.0",
    "@angular/forms": "^17.1.0",
    "@angular/platform-browser": "^17.1.0",
    "@angular/platform-browser-dynamic": "^17.1.0",
    "@angular/router": "^17.1.0",
    // Angular Material (DEFAULT UI Library)
    "@angular/material": "^17.1.0",
    "@angular/cdk": "^17.1.0",
    // State Management
    "@ngrx/signals": "^17.0.0",
    "@ngrx/store": "^17.0.0",
    "@ngrx/effects": "^17.0.0",
    // RxJS
    "rxjs": "^7.8.0",
    // Utilities
    "date-fns": "^3.3.0"
  },
  "devDependencies": {
    "@angular-devkit/build-angular": "^17.1.0",
    "@angular/cli": "^17.1.0",
    "@angular/compiler-cli": "^17.1.0",
    "typescript": "~5.3.0",
    // Testing
    "@angular/core": "^17.1.0",
    "jasmine-core": "~5.1.0",
    "karma": "~6.4.0",
    "karma-chrome-launcher": "~3.2.0",
    "karma-coverage": "~2.2.0",
    "karma-jasmine": "~5.1.0",
    "karma-jasmine-html-reporter": "~2.1.0",
    // Linting
    "@angular-eslint/builder": "^17.2.0",
    "@angular-eslint/eslint-plugin": "^17.2.0",
    "@angular-eslint/eslint-plugin-template": "^17.2.0",
    "@angular-eslint/schematics": "^17.2.0",
    "@angular-eslint/template-parser": "^17.2.0",
    "@typescript-eslint/eslint-plugin": "^6.19.0",
    "@typescript-eslint/parser": "^6.19.0",
    "eslint": "^8.56.0",
    // Documentation generation
    "typedoc": "^0.25.0",
    "typedoc-plugin-markdown": "^3.17.0"
  }
}
```

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### TDD Cycle

```
    +------------------+
    |                  |
    v                  |
+-------+    +-------+ |    +----------+
|  RED  |--->| GREEN |-+--->| REFACTOR |
+-------+    +-------+      +----------+
    ^                            |
    |                            |
    +----------------------------+

1. RED: Write a failing test first
   - Test describes the expected behavior
   - Test MUST fail initially (confirms test is valid)

2. GREEN: Write minimal code to make it pass
   - Only write enough code to pass the test
   - Do not optimize or refactor yet

3. REFACTOR: Improve code while keeping tests green
   - Clean up code structure
   - Remove duplication
   - Improve naming
   - All tests must still pass
```

### Example TDD Workflow for Angular (Jasmine/Karma)

```typescript
// ============================================
// Step 1: RED - Write failing test first
// ============================================

// user-filter.service.spec.ts
import { TestBed } from '@angular/core/testing';
import { UserFilterService } from './user-filter.service';

describe('UserFilterService', () => {
  let service: UserFilterService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [UserFilterService]
    });
    service = TestBed.inject(UserFilterService);
  });

  it('should filter users by active status', () => {
    const users = [
      { id: '1', name: 'Alice', email: 'alice@example.com', isActive: true },
      { id: '2', name: 'Bob', email: 'bob@example.com', isActive: false },
      { id: '3', name: 'Charlie', email: 'charlie@example.com', isActive: true }
    ];

    const result = service.filterActiveUsers(users);

    expect(result.length).toBe(2);
    expect(result.every(u => u.isActive)).toBe(true);
  });

  it('should return empty array when no active users exist', () => {
    const users = [
      { id: '1', name: 'Bob', email: 'bob@example.com', isActive: false }
    ];

    const result = service.filterActiveUsers(users);

    expect(result).toEqual([]);
  });

  it('should handle empty array input', () => {
    const result = service.filterActiveUsers([]);

    expect(result).toEqual([]);
  });
});

// Run: ng test --include=**/user-filter.service.spec.ts
// FAILS - UserFilterService doesn't exist yet

// ============================================
// Step 2: GREEN - Write minimal implementation
// ============================================

// user-filter.service.ts
import { Injectable } from '@angular/core';

interface User {
  id: string;
  name: string;
  email: string;
  isActive: boolean;
}

@Injectable({ providedIn: 'root' })
export class UserFilterService {
  filterActiveUsers(users: User[]): User[] {
    return users.filter(user => user.isActive);
  }
}

// Run: ng test --include=**/user-filter.service.spec.ts
// PASSES - All 3 tests pass

// ============================================
// Step 3: REFACTOR - Improve while tests stay green
// ============================================

// user-filter.service.ts (refactored with additional utilities)
import { Injectable, signal, computed } from '@angular/core';

interface User {
  id: string;
  name: string;
  email: string;
  isActive: boolean;
}

type UserPredicate = (user: User) => boolean;

@Injectable({ providedIn: 'root' })
export class UserFilterService {
  /**
   * Filters users to return only active users.
   * @param users - Array of users to filter
   * @returns Array of active users
   */
  filterActiveUsers(users: User[]): User[] {
    return this.filterUsers(users, user => user.isActive);
  }

  /**
   * Generic filter method for users.
   * @param users - Array of users to filter
   * @param predicate - Filter condition
   * @returns Filtered array of users
   */
  filterUsers(users: User[], predicate: UserPredicate): User[] {
    return users.filter(predicate);
  }
}

// Run: ng test --include=**/user-filter.service.spec.ts
// PASSES - All tests still pass after refactoring
```

### Example TDD Workflow for Component Testing

```typescript
// ============================================
// Step 1: RED - Write failing test for component
// ============================================

// counter.component.spec.ts
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { CounterComponent } from './counter.component';

describe('CounterComponent', () => {
  let component: CounterComponent;
  let fixture: ComponentFixture<CounterComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CounterComponent]
    }).compileComponents();

    fixture = TestBed.createComponent(CounterComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should display initial count of 0', () => {
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('[data-testid="count"]')?.textContent).toContain('0');
  });

  it('should increment count when increment button is clicked', () => {
    const button = fixture.nativeElement.querySelector('[data-testid="increment"]');
    button.click();
    fixture.detectChanges();

    expect(component.count()).toBe(1);
  });

  it('should decrement count when decrement button is clicked', () => {
    component.count.set(5);
    fixture.detectChanges();

    const button = fixture.nativeElement.querySelector('[data-testid="decrement"]');
    button.click();
    fixture.detectChanges();

    expect(component.count()).toBe(4);
  });

  it('should not decrement below 0', () => {
    const button = fixture.nativeElement.querySelector('[data-testid="decrement"]');
    button.click();
    fixture.detectChanges();

    expect(component.count()).toBe(0);
  });

  it('should emit countChange when count changes', () => {
    let emittedValue: number | undefined;
    component.countChange.subscribe((value: number) => {
      emittedValue = value;
    });

    component.increment();

    expect(emittedValue).toBe(1);
  });
});

// Run: ng test --include=**/counter.component.spec.ts
// FAILS - CounterComponent doesn't exist yet

// ============================================
// Step 2: GREEN - Write minimal implementation
// ============================================

// counter.component.ts
import { Component, signal, output, ChangeDetectionStrategy } from '@angular/core';

@Component({
  selector: 'app-counter',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="counter">
      <button data-testid="decrement" (click)="decrement()" type="button">-</button>
      <span data-testid="count">{{ count() }}</span>
      <button data-testid="increment" (click)="increment()" type="button">+</button>
    </div>
  `
})
export class CounterComponent {
  count = signal(0);
  countChange = output<number>();

  increment(): void {
    this.count.update(c => c + 1);
    this.countChange.emit(this.count());
  }

  decrement(): void {
    if (this.count() > 0) {
      this.count.update(c => c - 1);
      this.countChange.emit(this.count());
    }
  }
}

// Run: ng test --include=**/counter.component.spec.ts
// PASSES - All 6 tests pass
```

### Example TDD Workflow with Jest (Alternative)

```typescript
// If using Jest instead of Jasmine/Karma:

// jest.config.js
module.exports = {
  preset: 'jest-preset-angular',
  setupFilesAfterEnv: ['<rootDir>/setup-jest.ts'],
  testPathIgnorePatterns: ['<rootDir>/node_modules/', '<rootDir>/dist/'],
  coverageDirectory: 'coverage',
  collectCoverageFrom: ['src/app/**/*.ts', '!src/app/**/*.module.ts']
};

// user.service.spec.ts (Jest style)
import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { UserService } from './user.service';

describe('UserService', () => {
  let service: UserService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [UserService]
    });

    service = TestBed.inject(UserService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should fetch users from API', async () => {
    const mockUsers = [
      { id: '1', name: 'Alice', email: 'alice@example.com' }
    ];

    const usersPromise = service.getUsers();

    const req = httpMock.expectOne('/api/users');
    expect(req.request.method).toBe('GET');
    req.flush(mockUsers);

    const users = await usersPromise;
    expect(users).toEqual(mockUsers);
  });
});
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
+------------------------+
| 1. Bug Reported        |
+------------------------+
           |
           v
+------------------------+
| 2. Write Failing Test  |<---- Test MUST reproduce the bug
+------------------------+
           |
           v
+------------------------+
| 3. Verify Test Fails   |<---- Confirms bug exists
+------------------------+
           |
           v
+------------------------+
| 4. Implement Fix       |
+------------------------+
           |
           v
+------------------------+
| 5. Verify Test Passes  |<---- Confirms bug is fixed
+------------------------+
           |
           v
+------------------------+
| 6. Run All Tests       |<---- No regressions introduced
+------------------------+
           |
           v
+------------------------+
| 7. Document Bug in     |<---- Include bug ID, date, description
|    Test Comments       |
+------------------------+
           |
           v
+------------------------+
| 8. Deploy with         |<---- Regression prevented forever
|    Confidence          |
+------------------------+
```

### Example Bug Fix with Regression Test

```typescript
// =====================================================
// Bug Report #1234: Email validation allows invalid emails
// Reported: 2026-01-20
// Severity: High
// Description: Email validation accepts emails without TLD
//              (e.g., "user@domain" passes validation)
// =====================================================

// =====================================================
// Step 1-2: Write test that REPRODUCES the bug
// =====================================================

// email-validator.service.spec.ts
import { TestBed } from '@angular/core/testing';
import { EmailValidatorService } from './email-validator.service';

describe('EmailValidatorService', () => {
  let service: EmailValidatorService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [EmailValidatorService]
    });
    service = TestBed.inject(EmailValidatorService);
  });

  // Existing tests...
  it('should validate correct email', () => {
    expect(service.isValidEmail('user@example.com')).toBe(true);
  });

  it('should reject email without @', () => {
    expect(service.isValidEmail('userexample.com')).toBe(false);
  });

  // =====================================================
  // REGRESSION TEST for Bug #1234
  // Added: 2026-01-20
  // Bug: Email validation accepts emails without TLD
  // =====================================================
  describe('Bug #1234 - Email must have valid TLD', () => {
    it('should reject email without TLD - Bug #1234', () => {
      // Bug: "user@domain" was incorrectly accepted
      // Expected: Should be rejected (no TLD like .com, .org, etc.)
      expect(service.isValidEmail('user@domain')).toBe(false);
    });

    it('should reject email with single-character TLD - Bug #1234', () => {
      // Edge case discovered during bug investigation
      expect(service.isValidEmail('user@domain.a')).toBe(false);
    });

    it('should accept email with valid two-letter TLD - Bug #1234', () => {
      expect(service.isValidEmail('user@domain.co')).toBe(true);
    });

    it('should accept email with valid three-letter TLD - Bug #1234', () => {
      expect(service.isValidEmail('user@domain.com')).toBe(true);
    });
  });
});

// Run: ng test --include=**/email-validator.service.spec.ts
// FAILS - Bug #1234 tests fail, confirming the bug exists

// =====================================================
// Step 3: Verify test fails for the RIGHT reason
// =====================================================

// Output:
// FAILED: should reject email without TLD - Bug #1234
//   Expected: false
//   Actual: true
//
// This confirms the bug: emails without TLD are incorrectly accepted

// =====================================================
// Step 4: Implement the fix
// =====================================================

// email-validator.service.ts (BEFORE - buggy version)
@Injectable({ providedIn: 'root' })
export class EmailValidatorService {
  isValidEmail(email: string): boolean {
    // BUG: This regex doesn't require a TLD
    const emailRegex = /^[^\s@]+@[^\s@]+$/;
    return emailRegex.test(email);
  }
}

// email-validator.service.ts (AFTER - fixed version)
import { Injectable } from '@angular/core';

/**
 * Service for validating email addresses.
 *
 * @service
 * @providedIn 'root'
 */
@Injectable({ providedIn: 'root' })
export class EmailValidatorService {
  /**
   * Validates an email address format.
   *
   * @param email - The email address to validate
   * @returns true if the email format is valid, false otherwise
   *
   * @example
   * ```typescript
   * service.isValidEmail('user@example.com'); // true
   * service.isValidEmail('user@domain'); // false (no TLD)
   * ```
   *
   * @remarks
   * Fix for Bug #1234: Now requires a valid TLD (minimum 2 characters)
   */
  isValidEmail(email: string): boolean {
    // FIX for Bug #1234: Require TLD with minimum 2 characters
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;
    return emailRegex.test(email);
  }
}

// =====================================================
// Step 5-6: Verify test passes and no regressions
// =====================================================

// Run: ng test --include=**/email-validator.service.spec.ts
// PASSES - All tests pass including Bug #1234 regression tests

// Run: ng test --no-watch --browsers=ChromeHeadless
// PASSES - All project tests pass, no regressions
```

### Example Bug Fix for Component Interaction

```typescript
// =====================================================
// Bug Report #5678: Search results not cleared on empty query
// Reported: 2026-01-21
// Severity: Medium
// Description: When user clears search input, old results remain visible
// =====================================================

// search.component.spec.ts
describe('SearchComponent - Bug #5678', () => {
  let component: SearchComponent;
  let fixture: ComponentFixture<SearchComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SearchComponent],
      providers: [
        { provide: SearchService, useValue: mockSearchService }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(SearchComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  // =====================================================
  // REGRESSION TEST for Bug #5678
  // Added: 2026-01-21
  // Bug: Search results persist when query is cleared
  // =====================================================
  describe('Bug #5678 - Clear results on empty query', () => {
    it('should clear results when search query becomes empty - Bug #5678', async () => {
      // Setup: Perform a search first
      component.searchQuery.set('test');
      await component.performSearch();
      fixture.detectChanges();

      expect(component.results().length).toBeGreaterThan(0);

      // Bug reproduction: Clear the search query
      component.searchQuery.set('');
      await component.performSearch();
      fixture.detectChanges();

      // Expected: Results should be empty
      // Bug behavior: Results remained from previous search
      expect(component.results()).toEqual([]);
    });

    it('should clear results when search query is only whitespace - Bug #5678', async () => {
      component.searchQuery.set('test');
      await component.performSearch();
      fixture.detectChanges();

      component.searchQuery.set('   ');
      await component.performSearch();
      fixture.detectChanges();

      expect(component.results()).toEqual([]);
    });
  });
});

// search.component.ts (FIXED)
@Component({
  selector: 'app-search',
  standalone: true,
  imports: [CommonModule, FormsModule, MatInputModule, MatIconModule],
  template: `
    <mat-form-field>
      <mat-label>Search</mat-label>
      <input matInput [(ngModel)]="searchQuery" (ngModelChange)="onSearchChange($event)" />
      <mat-icon matSuffix>search</mat-icon>
    </mat-form-field>

    @if (isLoading()) {
      <mat-spinner diameter="24" />
    }

    @for (result of results(); track result.id) {
      <div class="search-result">{{ result.title }}</div>
    } @empty {
      @if (hasSearched() && !isLoading()) {
        <p>No results found</p>
      }
    }
  `,
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class SearchComponent {
  private searchService = inject(SearchService);

  searchQuery = signal('');
  results = signal<SearchResult[]>([]);
  isLoading = signal(false);
  hasSearched = signal(false);

  async onSearchChange(query: string): Promise<void> {
    this.searchQuery.set(query);
    await this.performSearch();
  }

  async performSearch(): Promise<void> {
    const query = this.searchQuery().trim();

    // FIX for Bug #5678: Clear results if query is empty
    if (!query) {
      this.results.set([]);
      this.hasSearched.set(false);
      return;
    }

    this.isLoading.set(true);
    this.hasSearched.set(true);

    try {
      const results = await this.searchService.search(query);
      this.results.set(results);
    } finally {
      this.isLoading.set(false);
    }
  }
}
```

### Bug Fix Checklist

Before marking a bug fix as complete, verify:

- [ ] Regression test written BEFORE implementing fix
- [ ] Regression test fails initially (reproduces the bug)
- [ ] Bug ID referenced in test description and comments
- [ ] Fix implemented with minimal changes
- [ ] Regression test passes after fix
- [ ] All existing tests still pass
- [ ] Code documented with JSDoc including bug reference
- [ ] No new warnings or errors introduced

---

## 3. Agent Code Generation Requirements (MANDATORY)

### A. Build Verification Protocol

**CRITICAL: Agents MUST verify that all generated code compiles and tests pass before presenting it to the user.**

#### Verification Checklist

**Before delivering ANY code, the agent MUST:**

1. **TypeScript Compilation Check**:
   ```bash
   # Run TypeScript compiler
   ng build
   # OR for development build
   ng build --configuration development
   ```
   - **MUST** return exit code 0 (no errors)
   - Address ALL TypeScript errors, not just warnings
   - NO `any` types allowed as workarounds
   - Verify strict template type checking passes

2. **Linter Check**:
   ```bash
   # Run ESLint
   ng lint
   ```
   - Fix all errors
   - Address critical warnings

3. **Unit Test Creation (MANDATORY)**:
   - Write tests for ALL new components
   - Write tests for ALL new services
   - Write tests for ALL new pipes and directives
   - Write tests for ALL new guards and interceptors
   - Minimum 80% code coverage
   - Tests MUST follow Angular testing best practices

4. **Test Execution**:
   ```bash
   # Run all tests
   ng test --watch=false --browsers=ChromeHeadless
   
   # Run with coverage
   ng test --no-watch --code-coverage --browsers=ChromeHeadless
   ```
   - **ALL tests MUST pass** (exit code 0)
   - Coverage must be ≥ 80%
   - No skipped or pending tests (`xit`, `xdescribe`)

5. **Production Build Verification**:
   ```bash
   # Verify production build succeeds
   ng build --configuration production
   ```
   - MUST complete without errors
   - Check bundle sizes are reasonable
   - Verify AOT compilation succeeds

#### Error Correction Process

If verification fails:

1. **Identify the error**: Read the full error message and stack trace
2. **Locate the source**: Find the exact file and line number
3. **Fix the root cause**: Don't just suppress warnings or errors
4. **Re-verify**: Run checks again until all pass
5. **Document changes**: Note any significant fixes made

### B. Testing Requirements (MANDATORY)

**EVERY component, service, pipe, directive, guard, and interceptor MUST have unit tests.**

#### What Must Be Tested

| Code Type | Required Tests |
|-----------|----------------|
| **Components** | Rendering, signals, inputs/outputs, user interactions, conditional rendering, error states |
| **Services** | Methods, HTTP calls, signal updates, error handling |
| **Pipes** | Transform logic, edge cases, null/undefined handling |
| **Directives** | DOM manipulation, input changes, event handlers |
| **Guards** | Route access logic, redirect behavior |
| **Interceptors** | Request/response modification, error handling |

#### Test Example Requirements

```typescript
// ✅ CORRECT - Component with comprehensive tests

// user-card.component.ts
import { Component, signal, input, output, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';

interface User {
  id: string;
  name: string;
  email: string;
}

@Component({
  selector: 'app-user-card',
  standalone: true,
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="user-card" [attr.data-testid]="'user-card-' + user().id">
      <h3>{{ user().name }}</h3>
      <p>{{ user().email }}</p>
      @if (showActions()) {
        <button 
          (click)="handleDelete()" 
          [disabled]="isDeleting()"
          type="button"
        >
          {{ isDeleting() ? 'Deleting...' : 'Delete' }}
        </button>
      }
    </div>
  `,
  styleUrls: ['./user-card.component.scss']
})
export class UserCardComponent {
  user = input.required<User>();
  showActions = input(false);
  
  delete = output<string>();
  
  isDeleting = signal(false);
  
  handleDelete(): void {
    this.isDeleting.set(true);
    this.delete.emit(this.user().id);
    // Simulate async operation
    setTimeout(() => this.isDeleting.set(false), 1000);
  }
}

// user-card.component.spec.ts
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { UserCardComponent } from './user-card.component';

describe('UserCardComponent', () => {
  let component: UserCardComponent;
  let fixture: ComponentFixture<UserCardComponent>;
  
  const mockUser = {
    id: '1',
    name: 'John Doe',
    email: 'john@example.com'
  };
  
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [UserCardComponent]
    }).compileComponents();
    
    fixture = TestBed.createComponent(UserCardComponent);
    component = fixture.componentInstance;
  });
  
  it('should create', () => {
    expect(component).toBeTruthy();
  });
  
  it('should display user information', () => {
    fixture.componentRef.setInput('user', mockUser);
    fixture.detectChanges();
    
    const compiled = fixture.nativeElement;
    expect(compiled.querySelector('h3')?.textContent).toContain('John Doe');
    expect(compiled.querySelector('p')?.textContent).toContain('john@example.com');
  });
  
  it('should not show delete button when showActions is false', () => {
    fixture.componentRef.setInput('user', mockUser);
    fixture.componentRef.setInput('showActions', false);
    fixture.detectChanges();
    
    const button = fixture.nativeElement.querySelector('button');
    expect(button).toBeNull();
  });
  
  it('should show delete button when showActions is true', () => {
    fixture.componentRef.setInput('user', mockUser);
    fixture.componentRef.setInput('showActions', true);
    fixture.detectChanges();
    
    const button = fixture.nativeElement.querySelector('button');
    expect(button).toBeTruthy();
    expect(button?.textContent?.trim()).toBe('Delete');
  });
  
  it('should emit delete event with user id', () => {
    fixture.componentRef.setInput('user', mockUser);
    fixture.componentRef.setInput('showActions', true);
    fixture.detectChanges();
    
    let emittedId: string | undefined;
    component.delete.subscribe((id: string) => {
      emittedId = id;
    });
    
    component.handleDelete();
    
    expect(emittedId).toBe('1');
  });
  
  it('should show loading state during deletion', () => {
    fixture.componentRef.setInput('user', mockUser);
    fixture.componentRef.setInput('showActions', true);
    fixture.detectChanges();
    
    component.handleDelete();
    fixture.detectChanges();
    
    const button = fixture.nativeElement.querySelector('button');
    expect(button?.textContent?.trim()).toBe('Deleting...');
    expect(button?.disabled).toBe(true);
  });
  
  it('should handle multiple rapid delete clicks', () => {
    fixture.componentRef.setInput('user', mockUser);
    fixture.componentRef.setInput('showActions', true);
    fixture.detectChanges();
    
    let emitCount = 0;
    component.delete.subscribe(() => {
      emitCount++;
    });
    
    component.handleDelete();
    component.handleDelete();
    component.handleDelete();
    
    expect(emitCount).toBe(3);
  });
});

// ✅ CORRECT - Service with comprehensive tests

// user.service.ts
import { Injectable, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, catchError, map, of } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class UserService {
  private http = inject(HttpClient);
  
  isLoading = signal(false);
  error = signal<string | null>(null);
  
  getUsers(): Observable<User[]> {
    this.isLoading.set(true);
    this.error.set(null);
    
    return this.http.get<User[]>('/api/users').pipe(
      map(users => {
        this.isLoading.set(false);
        return users;
      }),
      catchError(err => {
        this.isLoading.set(false);
        this.error.set(err.message);
        return of([]);
      })
    );
  }
  
  getUser(id: string): Observable<User | null> {
    return this.http.get<User>(`/api/users/${id}`).pipe(
      catchError(() => of(null))
    );
  }
}

// user.service.spec.ts
import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { UserService } from './user.service';

describe('UserService', () => {
  let service: UserService;
  let httpMock: HttpTestingController;
  
  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [UserService]
    });
    
    service = TestBed.inject(UserService);
    httpMock = TestBed.inject(HttpTestingController);
  });
  
  afterEach(() => {
    httpMock.verify();
  });
  
  it('should be created', () => {
    expect(service).toBeTruthy();
  });
  
  it('should fetch users successfully', (done) => {
    const mockUsers = [
      { id: '1', name: 'User 1', email: 'user1@example.com' },
      { id: '2', name: 'User 2', email: 'user2@example.com' }
    ];
    
    expect(service.isLoading()).toBe(false);
    
    service.getUsers().subscribe(users => {
      expect(users).toEqual(mockUsers);
      expect(service.isLoading()).toBe(false);
      expect(service.error()).toBeNull();
      done();
    });
    
    expect(service.isLoading()).toBe(true);
    
    const req = httpMock.expectOne('/api/users');
    expect(req.request.method).toBe('GET');
    req.flush(mockUsers);
  });
  
  it('should handle error when fetching users', (done) => {
    service.getUsers().subscribe(users => {
      expect(users).toEqual([]);
      expect(service.isLoading()).toBe(false);
      expect(service.error()).toBeTruthy();
      done();
    });
    
    const req = httpMock.expectOne('/api/users');
    req.error(new ProgressEvent('Network error'));
  });
  
  it('should fetch single user by id', (done) => {
    const mockUser = { id: '1', name: 'John', email: 'john@example.com' };
    
    service.getUser('1').subscribe(user => {
      expect(user).toEqual(mockUser);
      done();
    });
    
    const req = httpMock.expectOne('/api/users/1');
    expect(req.request.method).toBe('GET');
    req.flush(mockUser);
  });
  
  it('should return null when user not found', (done) => {
    service.getUser('999').subscribe(user => {
      expect(user).toBeNull();
      done();
    });
    
    const req = httpMock.expectOne('/api/users/999');
    req.error(new ProgressEvent('Not found'));
  });
});
```

### C. Agent Workflow Example

**Complete agent code generation workflow:**

1. **Generate Component**:
   ```typescript
   // Create user-list.component.ts with signals and standalone
   @Component({ standalone: true, ... })
   export class UserListComponent { ... }
   ```

2. **Generate Tests**:
   ```typescript
   // Create user-list.component.spec.ts with comprehensive tests
   describe('UserListComponent', () => { ... });
   ```

3. **Verify TypeScript Compilation**:
   ```bash
   ng build
   # ✓ Build completed successfully
   ```

4. **Run Tests**:
   ```bash
   ng test --no-watch --code-coverage --browsers=ChromeHeadless
   # ✓ All tests passed (15/15)
   # ✓ Coverage: 85%
   ```

5. **Verify Production Build**:
   ```bash
   ng build --configuration production
   # ✓ Production build completed successfully
   # ✓ Bundle sizes within limits
   ```

6. **Present Code**: Only after ALL checks pass

### D. Test-Driven Development (TDD) Workflow (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

#### TDD Cycle

```
1. 🔴 RED: Write a failing test first
   ↓
2. 🟢 GREEN: Write minimal code to make it pass  
   ↓
3. 🔵 REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

#### Example TDD Workflow

```typescript
// Step 1: RED - Write failing test first
describe('UserService', () => {
  it('should filter active users', () => {
    const service = TestBed.inject(UserService);
    const users = [
      { id: '1', name: 'John', isActive: true },
      { id: '2', name: 'Jane', isActive: false }
    ];
    
    const result = service.filterActiveUsers(users);
    
    expect(result).toEqual([{ id: '1', name: 'John', isActive: true }]);
  });
});
// Test fails - filterActiveUsers doesn't exist yet

// Step 2: GREEN - Write minimal implementation
@Injectable({ providedIn: 'root' })
export class UserService {
  filterActiveUsers(users: User[]): User[] {
    return users.filter(u => u.isActive);
  }
}
// Test passes ✓

// Step 3: REFACTOR - Improve if needed
// (In this case, code is already clean)
```

### E. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

#### Bug Fix Workflow

```
1. 🐛 Bug Reported/Discovered
   ↓
2. ✍️ Write a test that REPRODUCES the bug (test will FAIL)
   ↓
3. ✅ Verify the test fails for the right reason
   ↓
4. 🔧 Fix the bug (make the test pass)
   ↓
5. 🟢 Verify the test now PASSES
   ↓
6. 📝 Document the bug in test comments
   ↓
7. 🚀 Deploy with confidence (regression prevented)
```

#### Example Bug Fix

```typescript
// Bug Report: User login fails when email contains uppercase letters

// Step 1-2: Write test that reproduces the bug
describe('AuthService - Bug #1234', () => {
  it('should handle login with uppercase email - Bug #1234', async () => {
    // Bug: Login failed with "User@Example.com" but worked with "user@example.com"
    // Discovered: 2026-01-18
    // This test prevents regression
    
    const service = TestBed.inject(AuthService);
    const result = await service.login('User@Example.com', 'password123');
    
    expect(result.success).toBe(true);
    expect(result.user.email).toBe('user@example.com');
  });
});
// Test FAILS - reproduces the bug ✓

// Step 3: Fix the bug
@Injectable({ providedIn: 'root' })
export class AuthService {
  async login(email: string, password: string): Promise<LoginResult> {
    // FIX: Normalize email to lowercase before lookup
    const normalizedEmail = email.toLowerCase();
    
    const user = await this.userRepository.findByEmail(normalizedEmail);
    if (!user || !this.verifyPassword(password, user.passwordHash)) {
      return { success: false };
    }
    
    return { success: true, user };
  }
}
// Test PASSES - bug fixed ✓
```

### F. Prohibited Practices

**NEVER deliver code that:**
- ❌ Has TypeScript compilation errors
- ❌ Has template type checking errors
- ❌ Uses `any` types to bypass type checking
- ❌ Has failing tests
- ❌ Lacks tests for new functionality
- ❌ Has test coverage < 80%
- ❌ Has skipped tests (`xit`, `xdescribe`, `fdescribe`, `fit`)
- ❌ Fails to build for production
- ❌ Suppresses linter errors without justification
- ❌ Has console.log statements in production code
- ❌ Uses NgModules instead of standalone components
- ❌ Uses old *ngIf/*ngFor instead of @if/@for
- ❌ **Fixes bugs without adding regression tests first**
- ❌ **Writes implementation before writing tests (violates TDD)**
- ❌ **Skips Red-Green-Refactor cycle for new features**

---

## 4. Documentation Requirements (MANDATORY)

### A. JSDoc Comments for All Code

**ALL exported components, services, pipes, directives, guards, interceptors, and interfaces MUST have comprehensive JSDoc documentation.**

#### Why JSDoc Documentation?

- **Auto-Generated API Docs**: TypeDoc generates complete API documentation from JSDoc comments
- **IDE IntelliSense**: Better autocomplete and inline documentation for developers
- **Type Safety**: JSDoc + TypeScript provides comprehensive type information
- **Maintenance**: Self-documenting code reduces onboarding time by 40%+
- **Verification**: Documentation is verified during build process

### B. Component Documentation

```typescript
/**
 * User card component for displaying user information.
 * 
 * Displays user details with optional action buttons. Supports editing and
 * deletion through event outputs. Uses Angular Material for consistent UI.
 * 
 * @component
 * @example
 * ```typescript
 * <app-user-card
 *   [user]="currentUser"
 *   [showActions]="true"
 *   (edit)="handleEdit($event)"
 *   (delete)="handleDelete($event)"
 * />
 * ```
 * 
 * @see {@link User} for the user data structure
 */
@Component({
  selector: 'app-user-card',
  standalone: true,
  imports: [CommonModule, MatCardModule, MatButtonModule],
  templateUrl: './user-card.component.html',
  styleUrls: ['./user-card.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class UserCardComponent {
  /**
   * User data to display (required).
   * @type {InputSignal<User>}
   */
  user = input.required<User>();
  
  /**
   * Whether to show action buttons (default: false).
   * @type {InputSignal<boolean>}
   */
  showActions = input(false);
  
  /**
   * Emitted when the edit button is clicked.
   * Passes the user ID to the parent component.
   * @type {OutputEmitterRef<string>}
   */
  edit = output<string>();
  
  /**
   * Emitted when the delete button is clicked.
   * Passes the user ID to the parent component.
   * @type {OutputEmitterRef<string>}
   */
  delete = output<string>();
  
  /**
   * Indicates whether a delete operation is in progress.
   * @type {WritableSignal<boolean>}
   */
  isDeleting = signal(false);
  
  /**
   * Computed avatar URL with fallback to default.
   * @type {Signal<string>}
   */
  avatarUrl = computed(() => this.user().avatar ?? '/assets/default-avatar.png');
  
  /**
   * Handles the edit action.
   * Emits the edit event with the user's ID.
   * 
   * @returns {void}
   */
  handleEdit(): void {
    this.edit.emit(this.user().id);
  }
  
  /**
   * Handles the delete action.
   * Sets the deleting state and emits the delete event with the user's ID.
   * 
   * @returns {void}
   */
  handleDelete(): void {
    this.isDeleting.set(true);
    this.delete.emit(this.user().id);
  }
}
```

### C. Service Documentation

```typescript
/**
 * Service for managing user data operations.
 * 
 * Provides methods for CRUD operations on users, including loading,
 * creating, updating, and deleting. Manages loading and error states
 * using signals. All HTTP operations use async/await pattern.
 * 
 * @service
 * @providedIn 'root'
 * 
 * @example
 * ```typescript
 * @Component({...})
 * export class UserListComponent {
 *   private userService = inject(UserService);
 *   
 *   async ngOnInit(): Promise<void> {
 *     await this.userService.loadUsers();
 *   }
 * }
 * ```
 */
@Injectable({ providedIn: 'root' })
export class UserService {
  private http = inject(HttpClient);
  private config = inject(APP_CONFIG);
  
  /**
   * List of all users.
   * @type {WritableSignal<User[]>}
   */
  users = signal<User[]>([]);
  
  /**
   * Indicates whether a data operation is in progress.
   * @type {WritableSignal<boolean>}
   */
  isLoading = signal(false);
  
  /**
   * Error message from the last failed operation, or null.
   * @type {WritableSignal<string | null>}
   */
  error = signal<string | null>(null);
  
  /**
   * Loads all users from the API.
   * 
   * Sets loading state, fetches users, and updates the users signal.
   * Handles errors by setting the error signal and logging.
   * 
   * @async
   * @returns {Promise<void>} Promise that resolves when users are loaded
   * @throws {Error} If the HTTP request fails
   * 
   * @example
   * ```typescript
   * await userService.loadUsers();
   * console.log('Users:', userService.users());
   * ```
   */
  async loadUsers(): Promise<void> {
    this.isLoading.set(true);
    this.error.set(null);
    
    try {
      const users = await firstValueFrom(
        this.http.get<User[]>(`${this.config.apiUrl}/users`)
      );
      this.users.set(users);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      this.error.set(message);
      console.error('Failed to load users:', err);
    } finally {
      this.isLoading.set(false);
    }
  }
  
  /**
   * Retrieves a single user by ID.
   * 
   * @async
   * @param {string} id - The unique identifier of the user
   * @returns {Promise<User | null>} Promise resolving to the user or null if not found
   * 
   * @example
   * ```typescript
   * const user = await userService.getUser('123');
   * if (user) {
   *   console.log('Found user:', user.name);
   * }
   * ```
   */
  async getUser(id: string): Promise<User | null> {
    try {
      return await firstValueFrom(
        this.http.get<User>(`${this.config.apiUrl}/users/${id}`)
      );
    } catch {
      return null;
    }
  }
  
  /**
   * Creates a new user.
   * 
   * @async
   * @param {Partial<User>} userData - The user data to create
   * @returns {Promise<User>} Promise resolving to the created user
   * @throws {Error} If the creation fails or validation errors occur
   * 
   * @example
   * ```typescript
   * const newUser = await userService.createUser({
   *   name: 'John Doe',
   *   email: 'john@example.com'
   * });
   * ```
   */
  async createUser(userData: Partial<User>): Promise<User> {
    return await firstValueFrom(
      this.http.post<User>(`${this.config.apiUrl}/users`, userData)
    );
  }
  
  /**
   * Updates an existing user.
   * 
   * @async
   * @param {string} id - The unique identifier of the user to update
   * @param {Partial<User>} updates - The fields to update
   * @returns {Promise<User>} Promise resolving to the updated user
   * @throws {Error} If the user doesn't exist or update fails
   * 
   * @example
   * ```typescript
   * const updated = await userService.updateUser('123', {
   *   name: 'Jane Doe'
   * });
   * ```
   */
  async updateUser(id: string, updates: Partial<User>): Promise<User> {
    return await firstValueFrom(
      this.http.put<User>(`${this.config.apiUrl}/users/${id}`, updates)
    );
  }
  
  /**
   * Deletes a user by ID.
   * 
   * @async
   * @param {string} id - The unique identifier of the user to delete
   * @returns {Promise<void>} Promise that resolves when deletion is complete
   * @throws {Error} If the user doesn't exist or deletion fails
   * 
   * @example
   * ```typescript
   * await userService.deleteUser('123');
   * console.log('User deleted successfully');
   * ```
   */
  async deleteUser(id: string): Promise<void> {
    await firstValueFrom(
      this.http.delete<void>(`${this.config.apiUrl}/users/${id}`)
    );
  }
}
```

### D. Interface and Type Documentation

```typescript
/**
 * Represents a user in the application.
 * 
 * Contains all user information including identification, profile data,
 * and system metadata. Used throughout the application for user operations.
 * 
 * @interface
 * @property {string} id - Unique identifier (UUID v4)
 * @property {string} name - User's full name (1-100 characters)
 * @property {string} email - User's email address (unique, validated)
 * @property {UserRole} role - User's role for authorization
 * @property {string} [avatar] - Optional URL to user's avatar image
 * @property {boolean} isActive - Whether the user account is active
 * @property {Date} createdAt - Account creation timestamp
 * @property {Date} updatedAt - Last modification timestamp
 * 
 * @example
 * ```typescript
 * const user: User = {
 *   id: '550e8400-e29b-41d4-a716-446655440000',
 *   name: 'John Doe',
 *   email: 'john@example.com',
 *   role: 'user',
 *   isActive: true,
 *   createdAt: new Date(),
 *   updatedAt: new Date()
 * };
 * ```
 */
export interface User {
  readonly id: string;
  name: string;
  email: string;
  role: UserRole;
  avatar?: string;
  isActive: boolean;
  createdAt: Date;
  updatedAt: Date;
}

/**
 * User role types for authorization.
 * 
 * - `admin`: Full system access, can manage all users and settings
 * - `user`: Standard user with limited permissions
 * - `guest`: Read-only access, no modification rights
 * 
 * @typedef {('admin' | 'user' | 'guest')} UserRole
 */
export type UserRole = 'admin' | 'user' | 'guest';

/**
 * Application configuration interface.
 * 
 * Defines the structure for application-wide configuration including
 * API endpoints, feature flags, and environment settings.
 * 
 * @interface
 * @property {string} apiUrl - Base URL for API endpoints
 * @property {boolean} production - Whether running in production mode
 * @property {object} features - Feature flag configuration
 * @property {boolean} features.enableAnalytics - Analytics tracking enabled
 * @property {boolean} features.enableDebug - Debug mode enabled
 */
export interface AppConfig {
  apiUrl: string;
  production: boolean;
  features: {
    enableAnalytics: boolean;
    enableDebug: boolean;
  };
}
```

### E. Guard and Interceptor Documentation

```typescript
/**
 * Authentication guard for protecting routes.
 * 
 * Checks if the user is authenticated before allowing route activation.
 * Redirects to login page with return URL if not authenticated.
 * 
 * @function
 * @param {ActivatedRouteSnapshot} route - The route being activated
 * @param {RouterStateSnapshot} state - Current router state
 * @returns {boolean} True if user can activate route, false otherwise
 * 
 * @example
 * ```typescript
 * const routes: Routes = [
 *   {
 *     path: 'dashboard',
 *     canActivate: [authGuard],
 *     loadComponent: () => import('./dashboard.component')
 *   }
 * ];
 * ```
 */
export const authGuard: CanActivateFn = (route, state) => {
  const authService = inject(AuthService);
  const router = inject(Router);
  
  if (!authService.isAuthenticated()) {
    router.navigate(['/login'], {
      queryParams: { returnUrl: state.url }
    });
    return false;
  }
  
  return true;
};

/**
 * HTTP interceptor for adding authentication tokens to requests.
 * 
 * Automatically adds the Bearer token to all outgoing HTTP requests
 * if the user is authenticated. Handles token refresh if needed.
 * 
 * @function
 * @param {HttpRequest<unknown>} req - The outgoing HTTP request
 * @param {HttpHandlerFn} next - The next handler in the chain
 * @returns {Observable<HttpEvent<unknown>>} Observable of HTTP events
 * 
 * @example
 * ```typescript
 * export const appConfig: ApplicationConfig = {
 *   providers: [
 *     provideHttpClient(
 *       withInterceptors([authInterceptor])
 *     )
 *   ]
 * };
 * ```
 */
export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const authService = inject(AuthService);
  const token = authService.getToken();
  
  if (token) {
    const authReq = req.clone({
      setHeaders: {
        Authorization: `Bearer ${token}`
      }
    });
    return next(authReq);
  }
  
  return next(req);
};
```

### F. Pipe and Directive Documentation

```typescript
/**
 * Custom date formatting pipe using date-fns.
 * 
 * Formats dates according to the specified format string.
 * Handles various input types (Date, string, number) and provides
 * error handling for invalid dates.
 * 
 * @pipe
 * @standalone
 * 
 * @example
 * ```html
 * <p>{{ createdAt() | customDate:'PPpp' }}</p>
 * <!-- Output: Apr 29, 2021, 12:30:00 PM -->
 * 
 * <p>{{ createdAt() | customDate:'yyyy-MM-dd' }}</p>
 * <!-- Output: 2021-04-29 -->
 * ```
 */
@Pipe({
  name: 'customDate',
  standalone: true
})
export class CustomDatePipe implements PipeTransform {
  /**
   * Transforms a date value into a formatted string.
   * 
   * @param {Date | string | number} value - The date to format
   * @param {string} [formatString='PP'] - The format pattern (date-fns format)
   * @returns {string} Formatted date string or empty string if invalid
   * 
   * @see {@link https://date-fns.org/docs/format} for format options
   */
  transform(value: Date | string | number, formatString: string = 'PP'): string {
    if (!value) return '';
    
    try {
      const date = typeof value === 'string' || typeof value === 'number'
        ? new Date(value)
        : value;
      
      return format(date, formatString);
    } catch (error) {
      console.error('Date formatting error:', error);
      return '';
    }
  }
}

/**
 * Directive for highlighting elements on hover.
 * 
 * Changes the background color of the host element when the user
 * hovers over it. Configurable highlight color via input.
 * 
 * @directive
 * @standalone
 * @selector [appHighlight]
 * 
 * @example
 * ```html
 * <p appHighlight [highlightColor]="'lightblue'">
 *   Hover over me!
 * </p>
 * ```
 */
@Directive({
  selector: '[appHighlight]',
  standalone: true
})
export class HighlightDirective {
  /**
   * The color to use for highlighting (default: 'yellow').
   * @type {InputSignal<string>}
   */
  highlightColor = input<string>('yellow');
  
  private el = inject(ElementRef);
  
  /**
   * Handles mouse enter event.
   * Applies the highlight color to the element.
   */
  @HostListener('mouseenter')
  onMouseEnter(): void {
    this.highlight(this.highlightColor());
  }
  
  /**
   * Handles mouse leave event.
   * Removes the highlight color from the element.
   */
  @HostListener('mouseleave')
  onMouseLeave(): void {
    this.highlight('');
  }
  
  /**
   * Applies or removes highlight color.
   * 
   * @private
   * @param {string} color - The color to apply or empty string to remove
   */
  private highlight(color: string): void {
    this.el.nativeElement.style.backgroundColor = color;
  }
}
```

### G. Generating Documentation with TypeDoc

#### Installation and Setup

```bash
# Install TypeDoc and plugins
npm add --save-dev typedoc typedoc-plugin-markdown

# Add scripts to package.json
npm pkg set scripts.docs="typedoc --out docs src/app"
npm pkg set scripts.docs:check="typedoc --emit none --validation.notDocumented true"
npm pkg set scripts.docs:serve="typedoc --out docs src/app && npx serve docs"
npm pkg set scripts.docs:json="typedoc --json docs/api.json src/app"
```

#### TypeDoc Configuration

Create `typedoc.json` in project root:

```json
{
  "entryPoints": ["src/app"],
  "entryPointStrategy": "expand",
  "out": "docs",
  "exclude": [
    "**/*.spec.ts",
    "**/test/**",
    "**/tests/**",
    "**/__tests__/**",
    "**/environments/**"
  ],
  "excludePrivate": true,
  "excludeProtected": false,
  "excludeInternal": false,
  "readme": "README.md",
  "plugin": ["typedoc-plugin-markdown"],
  "theme": "default",
  "categorizeByGroup": true,
  "categoryOrder": [
    "Components",
    "Services",
    "Guards",
    "Interceptors",
    "Pipes",
    "Directives",
    "Models",
    "*"
  ],
  "navigation": {
    "includeCategories": true,
    "includeGroups": true
  },
  "sort": ["source-order"],
  "validation": {
    "notExported": true,
    "invalidLink": true,
    "notDocumented": true
  },
  "compilerOptions": {
    "moduleResolution": "bundler"
  }
}
```

#### Generating Documentation

```bash
# Generate HTML documentation
npm run docs

# Check documentation completeness
npm run docs:check

# Generate and serve documentation
npm run docs:serve

# Generate JSON documentation (for tooling)
npm run docs:json

# Open generated docs
open docs/index.html  # macOS
xdg-open docs/index.html  # Linux
```

#### Documentation Categories

Organize your code with JSDoc tags:

```typescript
/**
 * User list component.
 * @category Components
 */
export class UserListComponent {}

/**
 * User service.
 * @category Services
 */
export class UserService {}

/**
 * Authentication guard.
 * @category Guards
 */
export const authGuard: CanActivateFn = () => true;

/**
 * Custom date pipe.
 * @category Pipes
 */
export class CustomDatePipe {}

/**
 * User model interface.
 * @category Models
 */
export interface User {}
```

### H. Documentation Verification

**Add documentation checks to package.json:**

```json
{
  "scripts": {
    "ng": "ng",
    "start": "ng serve",
    "build": "ng build",
    "test": "ng test",
    "test:ci": "ng test --no-watch --code-coverage --browsers=ChromeHeadless",
    "lint": "ng lint",
    "docs": "typedoc --out docs src/app",
    "docs:check": "typedoc --emit none --validation.notDocumented true",
    "docs:serve": "typedoc --out docs src/app && npx serve docs",
    "docs:json": "typedoc --json docs/api.json src/app",
    "verify": "ng lint && npm run docs:check && ng test --no-watch --browsers=ChromeHeadless"
  }
}
```

### I. CI/CD Integration

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Lint code
        run: ng lint
      
      - name: Verify documentation
        run: npm run docs:check
      
      - name: Run tests
        run: ng test --no-watch --code-coverage --browsers=ChromeHeadless
      
      - name: Build production
        run: ng build --configuration production
      
      - name: Generate documentation
        run: npm run docs
      
      - name: Upload documentation artifacts
        uses: actions/upload-artifact@v3
        with:
          name: api-documentation
          path: docs/
      
      - name: Deploy documentation to GitHub Pages
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
```

### J. Documentation Best Practices

**DO:**
- ✅ Document all public exports (components, services, pipes, directives, guards, interceptors)
- ✅ Include `@param` for all method parameters
- ✅ Include `@returns` for all return values
- ✅ Include `@throws` for methods that can throw
- ✅ Provide at least one `@example` for complex APIs
- ✅ Use `@type` for signals and class properties
- ✅ Include `@see` tags for related components/services
- ✅ Keep examples up-to-date with implementation
- ✅ Generate docs as part of CI/CD pipeline
- ✅ Use `@category` to organize documentation

**DON'T:**
- ❌ Skip documentation for "obvious" components
- ❌ Write vague descriptions ("Does stuff", "Helper component")
- ❌ Let examples become outdated
- ❌ Commit generated docs to git (add `docs/` to `.gitignore`)
- ❌ Use `@ts-ignore` to suppress documentation warnings
- ❌ Document private implementation details excessively

### K. Documentation Checklist

**Before committing code, verify:**
- [ ] All exported components have JSDoc comments
- [ ] All services have JSDoc comments
- [ ] All pipes and directives have JSDoc comments
- [ ] All guards and interceptors have JSDoc comments
- [ ] All public interfaces and types have JSDoc comments
- [ ] All `@param` tags document parameter types and purpose
- [ ] All `@returns` tags document return types and values
- [ ] At least one `@example` provided for complex APIs
- [ ] TypeDoc can generate docs: `npm run docs:check` passes
- [ ] Generated documentation is readable and complete
- [ ] No "not documented" warnings from TypeDoc
- [ ] Examples compile and run correctly

### L. .gitignore Configuration

```gitignore
# Dependencies
/node_modules

# Build output
/dist
/out-tsc
/tmp

# Generated documentation (regenerate during CI/CD)
/docs
/api-docs

# IDEs and editors
.idea/
.project
.classpath
.c9/
*.launch
.settings/
*.sublime-workspace
.vscode/*

# Testing
/coverage
/.nyc_output

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db

# Angular
/.angular/cache
```

---

## 5. Code Style Principles (MANDATORY)

### A. Async/Await Pattern (Preferred)

**ALWAYS prefer async/await over raw Promises or callbacks for better readability.**

```typescript
// ✅ CORRECT - Async/await pattern (PREFERRED)
import { Injectable, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class UserService {
  private http = inject(HttpClient);
  
  users = signal<User[]>([]);
  isLoading = signal(false);
  error = signal<string | null>(null);
  
  // Clean async/await pattern
  async loadUsers(): Promise<void> {
    this.isLoading.set(true);
    this.error.set(null);
    
    try {
      const users = await firstValueFrom(this.http.get<User[]>('/api/users'));
      this.users.set(users);
    } catch (err) {
      this.error.set(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      this.isLoading.set(false);
    }
  }
  
  async getUser(id: string): Promise<User | null> {
    try {
      return await firstValueFrom(this.http.get<User>(`/api/users/${id}`));
    } catch {
      return null;
    }
  }
  
  async createUser(user: Partial<User>): Promise<User> {
    return await firstValueFrom(
      this.http.post<User>('/api/users', user)
    );
  }
  
  async updateUser(id: string, updates: Partial<User>): Promise<User> {
    return await firstValueFrom(
      this.http.put<User>(`/api/users/${id}`, updates)
    );
  }
  
  async deleteUser(id: string): Promise<void> {
    await firstValueFrom(
      this.http.delete<void>(`/api/users/${id}`)
    );
  }
}

// Component usage with async/await
@Component({
  selector: 'app-user-list',
  standalone: true,
  template: `
    @if (userService.isLoading()) {
      <mat-spinner />
    } @else if (userService.error()) {
      <mat-error>{{ userService.error() }}</mat-error>
    } @else {
      @for (user of userService.users(); track user.id) {
        <app-user-card [user]="user" />
      }
    }
  `
})
export class UserListComponent {
  userService = inject(UserService);
  
  async ngOnInit(): Promise<void> {
    await this.userService.loadUsers();
  }
  
  async handleCreateUser(userData: Partial<User>): Promise<void> {
    try {
      const newUser = await this.userService.createUser(userData);
      console.log('User created:', newUser);
      await this.userService.loadUsers(); // Refresh list
    } catch (error) {
      console.error('Failed to create user:', error);
    }
  }
}

// ❌ WRONG - Nested callbacks (callback hell)
loadUsers(): void {
  this.http.get<User[]>('/api/users').subscribe(
    users => {
      this.users.set(users);
      this.http.get<Settings>('/api/settings').subscribe(
        settings => {
          this.processUsers(users, settings);
        },
        error => console.error(error)
      );
    },
    error => console.error(error)
  );
}

// ❌ WRONG - Raw promises with .then() chains
loadUsers(): void {
  firstValueFrom(this.http.get<User[]>('/api/users'))
    .then(users => {
      this.users.set(users);
      return firstValueFrom(this.http.get<Settings>('/api/settings'));
    })
    .then(settings => {
      this.processUsers(settings);
    })
    .catch(error => console.error(error));
}

// ✅ CORRECT - Clean async/await with sequential operations
async loadUsers(): Promise<void> {
  try {
    const users = await firstValueFrom(this.http.get<User[]>('/api/users'));
    this.users.set(users);
    
    const settings = await firstValueFrom(this.http.get<Settings>('/api/settings'));
    this.processUsers(users, settings);
  } catch (error) {
    console.error('Failed to load data:', error);
  }
}

// ✅ CORRECT - Parallel async operations
async loadAllData(): Promise<void> {
  try {
    const [users, settings, preferences] = await Promise.all([
      firstValueFrom(this.http.get<User[]>('/api/users')),
      firstValueFrom(this.http.get<Settings>('/api/settings')),
      firstValueFrom(this.http.get<Preferences>('/api/preferences'))
    ]);
    
    this.users.set(users);
    this.settings.set(settings);
    this.preferences.set(preferences);
  } catch (error) {
    console.error('Failed to load data:', error);
  }
}
```

**When to use RxJS vs Async/Await:**

| Use Case | Preferred Approach | Reason |
|----------|-------------------|---------|
| Single HTTP call | `async/await` | Cleaner, easier to read |
| Sequential operations | `async/await` | Avoids nested subscriptions |
| Multiple parallel calls | `async/await` with `Promise.all()` | Simpler than `forkJoin` |
| Form value changes | RxJS with `toSignal()` | Reactive updates |
| WebSocket streams | RxJS | Continuous data stream |
| Complex operators (debounce, retry) | RxJS | Built-in operators |
| Event streams | RxJS | Multiple emissions |

### B. Minimalistic Code Style

**Write clear, concise code with single responsibility. Avoid over-engineering.**

```typescript
// ✅ CORRECT - Minimalistic, clear component
@Component({
  selector: 'app-user-profile',
  standalone: true,
  imports: [MatCardModule, MatButtonModule],
  template: `
    <mat-card>
      <mat-card-header>
        <mat-card-title>{{ user().name }}</mat-card-title>
      </mat-card-header>
      <mat-card-content>
        <p>{{ user().email }}</p>
      </mat-card-content>
      <mat-card-actions>
        <button mat-button (click)="edit.emit(user().id)">Edit</button>
      </mat-card-actions>
    </mat-card>
  `
})
export class UserProfileComponent {
  user = input.required<User>();
  edit = output<string>();
}

// ❌ WRONG - Over-engineered, unnecessary complexity
@Component({
  selector: 'app-user-profile',
  standalone: true,
  template: `...`
})
export class UserProfileComponent implements OnInit, OnDestroy, AfterViewInit {
  private readonly COMPONENT_NAME = 'UserProfileComponent';
  private readonly DEFAULT_TIMEOUT = 5000;
  
  user = input.required<User>();
  edit = output<string>();
  
  private _internalState = signal<Record<string, any>>({});
  private _subscriptions: Subscription[] = [];
  private _config: UserProfileConfig = this.createDefaultConfig();
  
  ngOnInit(): void {
    this.logLifecycle('init');
    this.setupComponent();
  }
  
  ngAfterViewInit(): void {
    this.logLifecycle('afterViewInit');
  }
  
  ngOnDestroy(): void {
    this.logLifecycle('destroy');
    this.cleanup();
  }
  
  private createDefaultConfig(): UserProfileConfig {
    return { /* ... */ };
  }
  
  private setupComponent(): void { /* ... */ }
  private cleanup(): void { /* ... */ }
  private logLifecycle(stage: string): void { /* ... */ }
}

// ✅ CORRECT - Small, focused service
@Injectable({ providedIn: 'root' })
export class UserService {
  private http = inject(HttpClient);
  
  async getUsers(): Promise<User[]> {
    return await firstValueFrom(this.http.get<User[]>('/api/users'));
  }
}

// ❌ WRONG - Overly complex service
@Injectable({ providedIn: 'root' })
export class UserService {
  private http = inject(HttpClient);
  private cache = new Map<string, CachedItem<User>>();
  private readonly CACHE_DURATION = 5 * 60 * 1000;
  
  async getUsers(options?: GetUsersOptions): Promise<User[]> {
    const cacheKey = this.generateCacheKey(options);
    const cached = this.getCachedItem(cacheKey);
    
    if (cached && !this.isCacheExpired(cached)) {
      return this.transformUsers(cached.data, options);
    }
    
    const users = await this.fetchUsersWithRetry(options);
    this.setCacheItem(cacheKey, users);
    return this.transformUsers(users, options);
  }
  
  private generateCacheKey(options?: GetUsersOptions): string { /* ... */ }
  private getCachedItem(key: string): CachedItem<User> | undefined { /* ... */ }
  private isCacheExpired(item: CachedItem<User>): boolean { /* ... */ }
  private setCacheItem(key: string, data: User[]): void { /* ... */ }
  private async fetchUsersWithRetry(options?: GetUsersOptions): Promise<User[]> { /* ... */ }
  private transformUsers(users: User[], options?: GetUsersOptions): User[] { /* ... */ }
}
```

### C. Modular Architecture

**Create small, focused modules/components with clear boundaries.**

```typescript
// ✅ CORRECT - Modular structure with clear separation
// users/services/user.service.ts
@Injectable({ providedIn: 'root' })
export class UserService {
  private http = inject(HttpClient);
  
  async getUsers(): Promise<User[]> {
    return await firstValueFrom(this.http.get<User[]>('/api/users'));
  }
}

// users/services/user-cache.service.ts
@Injectable({ providedIn: 'root' })
export class UserCacheService {
  private cache = signal<Map<string, User>>(new Map());
  
  set(id: string, user: User): void {
    this.cache.update(map => new Map(map).set(id, user));
  }
  
  get(id: string): User | undefined {
    return this.cache().get(id);
  }
}

// users/components/user-card/user-card.component.ts
@Component({
  selector: 'app-user-card',
  standalone: true,
  imports: [MatCardModule, MatButtonModule],
  template: `...`
})
export class UserCardComponent {
  user = input.required<User>();
  edit = output<string>();
}

// users/components/user-list/user-list.component.ts
@Component({
  selector: 'app-user-list',
  standalone: true,
  imports: [UserCardComponent, MatProgressSpinnerModule],
  template: `...`
})
export class UserListComponent {
  private userService = inject(UserService);
  users = signal<User[]>([]);
  
  async ngOnInit(): Promise<void> {
    this.users.set(await this.userService.getUsers());
  }
}

// ❌ WRONG - Monolithic component doing everything
@Component({
  selector: 'app-users',
  standalone: true,
  template: `
    <!-- 500+ lines of template -->
  `
})
export class UsersComponent {
  // User management
  async loadUsers(): Promise<void> { /* ... */ }
  async createUser(user: User): Promise<void> { /* ... */ }
  async updateUser(id: string, user: User): Promise<void> { /* ... */ }
  async deleteUser(id: string): Promise<void> { /* ... */ }
  
  // Caching logic
  private cacheUsers(users: User[]): void { /* ... */ }
  private getCachedUsers(): User[] | null { /* ... */ }
  
  // Filtering logic
  filterUsersByRole(role: string): User[] { /* ... */ }
  filterUsersByStatus(status: string): User[] { /* ... */ }
  
  // Sorting logic
  sortUsersByName(): void { /* ... */ }
  sortUsersByDate(): void { /* ... */ }
  
  // Export logic
  exportToCsv(): void { /* ... */ }
  exportToJson(): void { /* ... */ }
  
  // Validation logic
  validateUser(user: User): boolean { /* ... */ }
  validateEmail(email: string): boolean { /* ... */ }
}
```

### D. Angular Material (Default UI Library)

**ALWAYS use Angular Material components unless explicitly specified otherwise.**

```typescript
// ✅ CORRECT - Using Angular Material components
import { Component, signal } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTableModule } from '@angular/material/table';
import { MatDialogModule } from '@angular/material/dialog';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatSnackBarModule } from '@angular/material/snack-bar';

@Component({
  selector: 'app-user-list',
  standalone: true,
  imports: [
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatTableModule
  ],
  template: `
    <mat-card>
      <mat-card-header>
        <mat-card-title>Users</mat-card-title>
        <mat-card-subtitle>Manage your users</mat-card-subtitle>
      </mat-card-header>
      
      <mat-card-content>
        @if (isLoading()) {
          <div class="loading">
            <mat-spinner />
          </div>
        } @else {
          <table mat-table [dataSource]="users()">
            <ng-container matColumnDef="name">
              <th mat-header-cell *matHeaderCellDef>Name</th>
              <td mat-cell *matCellDef="let user">{{ user.name }}</td>
            </ng-container>
            
            <ng-container matColumnDef="email">
              <th mat-header-cell *matHeaderCellDef>Email</th>
              <td mat-cell *matCellDef="let user">{{ user.email }}</td>
            </ng-container>
            
            <ng-container matColumnDef="actions">
              <th mat-header-cell *matHeaderCellDef>Actions</th>
              <td mat-cell *matCellDef="let user">
                <button mat-icon-button (click)="editUser(user)">
                  <mat-icon>edit</mat-icon>
                </button>
                <button mat-icon-button color="warn" (click)="deleteUser(user.id)">
                  <mat-icon>delete</mat-icon>
                </button>
              </td>
            </ng-container>
            
            <tr mat-header-row *matHeaderRowDef="displayedColumns"></tr>
            <tr mat-row *matRowDef="let row; columns: displayedColumns;"></tr>
          </table>
        }
      </mat-card-content>
      
      <mat-card-actions>
        <button mat-raised-button color="primary" (click)="addUser()">
          <mat-icon>add</mat-icon>
          Add User
        </button>
      </mat-card-actions>
    </mat-card>
  `,
  styles: [`
    .loading {
      display: flex;
      justify-content: center;
      padding: 2rem;
    }
  `]
})
export class UserListComponent {
  private userService = inject(UserService);
  private dialog = inject(MatDialog);
  private snackBar = inject(MatSnackBar);
  
  users = signal<User[]>([]);
  isLoading = signal(false);
  displayedColumns = ['name', 'email', 'actions'];
  
  async ngOnInit(): Promise<void> {
    await this.loadUsers();
  }
  
  async loadUsers(): Promise<void> {
    this.isLoading.set(true);
    try {
      const users = await this.userService.getUsers();
      this.users.set(users);
    } catch (error) {
      this.snackBar.open('Failed to load users', 'Close', { duration: 3000 });
    } finally {
      this.isLoading.set(false);
    }
  }
  
  addUser(): void {
    const dialogRef = this.dialog.open(UserDialogComponent, {
      width: '500px',
      data: { mode: 'create' }
    });
    
    dialogRef.afterClosed().subscribe(async (result) => {
      if (result) {
        await this.loadUsers();
        this.snackBar.open('User created successfully', 'Close', { duration: 3000 });
      }
    });
  }
  
  editUser(user: User): void {
    const dialogRef = this.dialog.open(UserDialogComponent, {
      width: '500px',
      data: { mode: 'edit', user }
    });
    
    dialogRef.afterClosed().subscribe(async (result) => {
      if (result) {
        await this.loadUsers();
        this.snackBar.open('User updated successfully', 'Close', { duration: 3000 });
      }
    });
  }
  
  async deleteUser(id: string): Promise<void> {
    try {
      await this.userService.deleteUser(id);
      await this.loadUsers();
      this.snackBar.open('User deleted successfully', 'Close', { duration: 3000 });
    } catch (error) {
      this.snackBar.open('Failed to delete user', 'Close', { duration: 3000 });
    }
  }
}

// ❌ WRONG - Using custom HTML elements instead of Material
@Component({
  selector: 'app-user-list',
  standalone: true,
  template: `
    <div class="card">
      <div class="card-header">
        <h2>Users</h2>
      </div>
      <div class="card-body">
        <div *ngIf="isLoading()" class="spinner"></div>
        <table class="custom-table">
          <!-- Custom table implementation -->
        </table>
      </div>
      <div class="card-actions">
        <button class="btn btn-primary">Add User</button>
      </div>
    </div>
  `
})
export class UserListComponent {
  // Same implementation but without Material components
}
```

### E. Material Theme Configuration

```typescript
// ✅ CORRECT - Configure Material theme in styles.scss
@use '@angular/material' as mat;

@include mat.core();

// Define custom theme
$my-primary: mat.define-palette(mat.$indigo-palette);
$my-accent: mat.define-palette(mat.$pink-palette, A200, A100, A400);
$my-warn: mat.define-palette(mat.$red-palette);

$my-theme: mat.define-light-theme((
  color: (
    primary: $my-primary,
    accent: $my-accent,
    warn: $my-warn,
  ),
  typography: mat.define-typography-config(),
  density: 0,
));

@include mat.all-component-themes($my-theme);

// Dark theme (optional)
.dark-theme {
  $dark-theme: mat.define-dark-theme((
    color: (
      primary: $my-primary,
      accent: $my-accent,
      warn: $my-warn,
    )
  ));
  
  @include mat.all-component-colors($dark-theme);
}
```

### F. Code Style Summary

**Follow these principles:**

1. **Async/Await**: Use `async/await` for all asynchronous operations
2. **Single Responsibility**: Each component/service does one thing well
3. **Small Functions**: Keep functions under 20 lines
4. **Clear Naming**: Use descriptive, self-documenting names
5. **No Nested Logic**: Avoid deeply nested conditionals (max 2 levels)
6. **Early Returns**: Use guard clauses for validation
7. **Material First**: Always use Angular Material components
8. **Signals Over Properties**: Use signals for reactive state
9. **Standalone**: Never use NgModules
10. **Typed Everything**: No `any` types

---

## 5. Standalone Components (Mandatory)

### A. Basic Standalone Component
* **ALWAYS use standalone: true**.

* **Import dependencies directly** in component metadata.

* **NO NgModules** - ever.

```typescript
// ✅ CORRECT - Standalone component
import { Component, signal, computed, input, output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';

interface User {
  id: string;
  name: string;
  email: string;
  avatar?: string;
}

@Component({
  selector: 'app-user-card',
  standalone: true,
  imports: [CommonModule, RouterLink],
  template: `
    <div class="user-card">
      <img [src]="avatar()" [alt]="user().name" />
      <h3>{{ user().name }}</h3>
      <p>{{ user().email }}</p>
      @if (showActions()) {
        <button (click)="handleEdit()" type="button">Edit</button>
      }
    </div>
  `,
  styleUrls: ['./user-card.component.scss']
})
export class UserCardComponent {
  // Modern signal-based inputs
  user = input.required<User>();
  showActions = input(false);
  
  // Modern signal-based outputs
  edit = output<string>();
  
  // Computed signal
  avatar = computed(() => this.user().avatar ?? '/assets/default-avatar.png');
  
  handleEdit(): void {
    this.edit.emit(this.user().id);
  }
}

// ❌ WRONG - Using NgModule
@NgModule({
  declarations: [UserCardComponent],
  imports: [CommonModule],
  exports: [UserCardComponent]
})
export class UserCardModule { }

// ❌ WRONG - Old @Input/@Output decorators (still works but signals preferred)
@Component({
  selector: 'app-user-card',
  standalone: true,
  imports: [CommonModule],
  template: `...`
})
export class UserCardComponent {
  @Input({ required: true }) user!: User;  // Old style
  @Output() edit = new EventEmitter<string>();  // Old style
}
```

### B. Component File Organization
```typescript
// user-card.component.ts
import { Component, signal, computed, input, output } from '@angular/core';
import { CommonModule } from '@angular/common';

// 1. Interfaces at the top
interface UserCardProps {
  user: User;
  showActions?: boolean;
}

// 2. Component decorator
@Component({
  selector: 'app-user-card',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './user-card.component.html',
  styleUrls: ['./user-card.component.scss'],
  // Use OnPush by default (signals handle change detection)
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class UserCardComponent implements OnInit, OnDestroy {
  // 3. Inputs (signal-based)
  user = input.required<User>();
  showActions = input(false);
  
  // 4. Outputs (signal-based)
  edit = output<string>();
  delete = output<string>();
  
  // 5. Public signals
  isLoading = signal(false);
  error = signal<string | null>(null);
  
  // 6. Computed signals
  displayName = computed(() => {
    const user = this.user();
    return `${user.name} (${user.email})`;
  });
  
  // 7. Private fields
  private destroyRef = inject(DestroyRef);
  
  // 8. Lifecycle hooks
  ngOnInit(): void {
    // Initialization logic
  }
  
  ngOnDestroy(): void {
    // Cleanup logic
  }
  
  // 9. Public methods
  handleEdit(): void {
    this.edit.emit(this.user().id);
  }
  
  // 10. Private methods
  private loadData(): void {
    // Implementation
  }
}
```

### C. Template Syntax (Modern @-syntax)
```html
<!-- ✅ CORRECT - Modern @if, @for, @switch syntax (Angular 17+) -->
<div class="container">
  <!-- Conditional rendering with @if -->
  @if (isLoading()) {
    <app-spinner />
  } @else if (error()) {
    <app-error-message [error]="error()" />
  } @else {
    <div class="content">
      <!-- Loop with @for -->
      @for (user of users(); track user.id) {
        <app-user-card
          [user]="user"
          (edit)="handleEdit($event)"
        />
      } @empty {
        <p>No users found</p>
      }
    </div>
  }
  
  <!-- Switch statement with @switch -->
  @switch (status()) {
    @case ('loading') {
      <app-spinner />
    }
    @case ('success') {
      <app-success />
    }
    @case ('error') {
      <app-error />
    }
    @default {
      <p>Unknown status</p>
    }
  }
</div>

<!-- ❌ WRONG - Old *ngIf, *ngFor syntax (deprecated) -->
<div *ngIf="isLoading(); else content">
  <app-spinner />
</div>
<ng-template #content>
  <div *ngFor="let user of users(); trackBy: trackById">
    <app-user-card [user]="user" />
  </div>
</ng-template>
```

## 6. Signals (Primary State Management)

### A. Basic Signal Usage
```typescript
// ✅ CORRECT - Signals for component state
import { Component, signal, computed, effect } from '@angular/core';

@Component({
  selector: 'app-counter',
  standalone: true,
  template: `
    <div>
      <p>Count: {{ count() }}</p>
      <p>Double: {{ double() }}</p>
      <button (click)="increment()" type="button">+</button>
      <button (click)="decrement()" type="button">-</button>
      <button (click)="reset()" type="button">Reset</button>
    </div>
  `
})
export class CounterComponent {
  // Writable signal
  count = signal(0);
  
  // Computed signal (derived state)
  double = computed(() => this.count() * 2);
  
  // Effect (side effects based on signal changes)
  constructor() {
    effect(() => {
      console.log(`Count changed to: ${this.count()}`);
      // Save to localStorage
      localStorage.setItem('count', this.count().toString());
    });
  }
  
  increment(): void {
    this.count.update(value => value + 1);
  }
  
  decrement(): void {
    this.count.update(value => value - 1);
  }
  
  reset(): void {
    this.count.set(0);
  }
}

// ❌ WRONG - Using class properties without signals
export class CounterComponent {
  count = 0;  // Won't trigger change detection properly
  
  increment(): void {
    this.count++;  // Manual change detection needed
  }
}
```

### B. Complex Signal Patterns
```typescript
// ✅ CORRECT - Signal-based state management
import { Component, signal, computed } from '@angular/core';

interface Todo {
  id: string;
  title: string;
  completed: boolean;
}

type Filter = 'all' | 'active' | 'completed';

@Component({
  selector: 'app-todo-list',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div>
      <input
        [value]="newTodoTitle()"
        (input)="newTodoTitle.set($any($event.target).value)"
        (keyup.enter)="addTodo()"
      />
      
      <div class="filters">
        @for (f of filters; track f) {
          <button
            (click)="currentFilter.set(f)"
            [class.active]="currentFilter() === f"
            type="button"
          >
            {{ f }}
          </button>
        }
      </div>
      
      @for (todo of filteredTodos(); track todo.id) {
        <div class="todo-item">
          <input
            type="checkbox"
            [checked]="todo.completed"
            (change)="toggleTodo(todo.id)"
          />
          <span [class.completed]="todo.completed">{{ todo.title }}</span>
          <button (click)="deleteTodo(todo.id)" type="button">Delete</button>
        </div>
      }
      
      <p>{{ stats().active }} active, {{ stats().completed }} completed</p>
    </div>
  `
})
export class TodoListComponent {
  // State signals
  todos = signal<Todo[]>([]);
  currentFilter = signal<Filter>('all');
  newTodoTitle = signal('');
  
  filters: Filter[] = ['all', 'active', 'completed'];
  
  // Computed signals
  filteredTodos = computed(() => {
    const filter = this.currentFilter();
    const todos = this.todos();
    
    switch (filter) {
      case 'active':
        return todos.filter(t => !t.completed);
      case 'completed':
        return todos.filter(t => t.completed);
      default:
        return todos;
    }
  });
  
  stats = computed(() => {
    const todos = this.todos();
    return {
      total: todos.length,
      active: todos.filter(t => !t.completed).length,
      completed: todos.filter(t => t.completed).length
    };
  });
  
  // Actions
  addTodo(): void {
    const title = this.newTodoTitle().trim();
    if (!title) return;
    
    const newTodo: Todo = {
      id: crypto.randomUUID(),
      title,
      completed: false
    };
    
    this.todos.update(todos => [...todos, newTodo]);
    this.newTodoTitle.set('');
  }
  
  toggleTodo(id: string): void {
    this.todos.update(todos =>
      todos.map(todo =>
        todo.id === id ? { ...todo, completed: !todo.completed } : todo
      )
    );
  }
  
  deleteTodo(id: string): void {
    this.todos.update(todos => todos.filter(todo => todo.id !== id));
  }
}
```

### C. Signal Store (NgRx Signals)
```typescript
// ✅ CORRECT - Signal Store for complex state
import { signalStore, withState, withComputed, withMethods } from '@ngrx/signals';
import { computed } from '@angular/core';

// Define state shape
interface UserState {
  users: User[];
  selectedUserId: string | null;
  isLoading: boolean;
  error: string | null;
}

// Initial state
const initialState: UserState = {
  users: [],
  selectedUserId: null,
  isLoading: false,
  error: null
};

// Create signal store
export const UserStore = signalStore(
  { providedIn: 'root' },
  withState(initialState),
  withComputed((store) => ({
    selectedUser: computed(() => {
      const users = store.users();
      const id = store.selectedUserId();
      return users.find(u => u.id === id) ?? null;
    }),
    activeUsers: computed(() =>
      store.users().filter(u => u.isActive)
    )
  })),
  withMethods((store) => ({
    setUsers(users: User[]): void {
      patchState(store, { users, isLoading: false });
    },
    selectUser(userId: string): void {
      patchState(store, { selectedUserId: userId });
    },
    setLoading(isLoading: boolean): void {
      patchState(store, { isLoading });
    },
    setError(error: string): void {
      patchState(store, { error, isLoading: false });
    }
  }))
);

// Usage in component
@Component({
  selector: 'app-user-list',
  standalone: true,
  template: `
    @if (store.isLoading()) {
      <app-spinner />
    } @else {
      @for (user of store.users(); track user.id) {
        <app-user-card
          [user]="user"
          [selected]="user.id === store.selectedUserId()"
          (click)="store.selectUser(user.id)"
        />
      }
    }
  `
})
export class UserListComponent {
  store = inject(UserStore);
  
  ngOnInit(): void {
    this.loadUsers();
  }
  
  private async loadUsers(): Promise<void> {
    this.store.setLoading(true);
    try {
      const users = await this.userService.getUsers();
      this.store.setUsers(users);
    } catch (error) {
      this.store.setError('Failed to load users');
    }
  }
}
```

## 7. Dependency Injection (Modern inject())

### A. Functional Injection
```typescript
// ✅ CORRECT - Modern inject() function
import { Component, inject } from '@angular/core';
import { Router } from '@angular/router';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-user-profile',
  standalone: true,
  template: `...`
})
export class UserProfileComponent {
  // Inject dependencies functionally
  private http = inject(HttpClient);
  private router = inject(Router);
  private userService = inject(UserService);
  
  // Optional injection
  private analytics = inject(AnalyticsService, { optional: true });
  
  // Self injection
  private elementRef = inject(ElementRef);
  
  loadUser(): void {
    this.userService.getUser().subscribe();
  }
}

// ❌ WRONG - Constructor injection (old style, still works but verbose)
export class UserProfileComponent {
  constructor(
    private http: HttpClient,
    private router: Router,
    private userService: UserService
  ) {}
}
```

### B. Injection Tokens
```typescript
// ✅ CORRECT - Typed injection tokens
import { InjectionToken } from '@angular/core';

export interface AppConfig {
  apiUrl: string;
  production: boolean;
  features: {
    enableAnalytics: boolean;
    enableDebug: boolean;
  };
}

export const APP_CONFIG = new InjectionToken<AppConfig>('app.config', {
  providedIn: 'root',
  factory: () => ({
    apiUrl: 'https://api.example.com',
    production: false,
    features: {
      enableAnalytics: false,
      enableDebug: true
    }
  })
});

// Usage
@Component({
  selector: 'app-root',
  standalone: true,
  template: `...`
})
export class AppComponent {
  private config = inject(APP_CONFIG);
  
  ngOnInit(): void {
    console.log('API URL:', this.config.apiUrl);
  }
}

// Provide in app.config.ts
export const appConfig: ApplicationConfig = {
  providers: [
    { provide: APP_CONFIG, useValue: environment },
    // ...
  ]
};
```

## 8. Routing with Lazy Loading

### A. Route Configuration
```typescript
// ✅ CORRECT - Functional route configuration with lazy loading
// app.routes.ts
import { Routes } from '@angular/router';
import { authGuard } from '@core/guards/auth.guard';

export const routes: Routes = [
  {
    path: '',
    redirectTo: 'home',
    pathMatch: 'full'
  },
  {
    path: 'home',
    loadComponent: () => import('./features/home/home.component')
      .then(m => m.HomeComponent)
  },
  {
    path: 'dashboard',
    loadComponent: () => import('./features/dashboard/dashboard.component')
      .then(m => m.DashboardComponent),
    canActivate: [authGuard],
    children: [
      {
        path: 'overview',
        loadComponent: () => import('./features/dashboard/overview/overview.component')
          .then(m => m.OverviewComponent)
      },
      {
        path: 'analytics',
        loadComponent: () => import('./features/dashboard/analytics/analytics.component')
          .then(m => m.AnalyticsComponent)
      }
    ]
  },
  {
    path: 'users',
    loadChildren: () => import('./features/users/users.routes')
      .then(m => m.USERS_ROUTES)
  },
  {
    path: '**',
    loadComponent: () => import('./shared/components/not-found/not-found.component')
      .then(m => m.NotFoundComponent)
  }
];

// features/users/users.routes.ts
export const USERS_ROUTES: Routes = [
  {
    path: '',
    loadComponent: () => import('./user-list/user-list.component')
      .then(m => m.UserListComponent)
  },
  {
    path: ':id',
    loadComponent: () => import('./user-detail/user-detail.component')
      .then(m => m.UserDetailComponent),
    resolve: {
      user: userResolver
    }
  }
];
```

### B. Functional Guards
```typescript
// ✅ CORRECT - Functional guard
import { inject } from '@angular/core';
import { Router, CanActivateFn } from '@angular/router';
import { AuthService } from '@core/services/auth.service';

export const authGuard: CanActivateFn = (route, state) => {
  const authService = inject(AuthService);
  const router = inject(Router);
  
  const isAuthenticated = authService.isAuthenticated();
  
  if (!isAuthenticated) {
    router.navigate(['/login'], {
      queryParams: { returnUrl: state.url }
    });
    return false;
  }
  
  return true;
};

// Role-based guard
export function hasRoleGuard(roles: string[]): CanActivateFn {
  return (route, state) => {
    const authService = inject(AuthService);
    const router = inject(Router);
    
    const userRoles = authService.getUserRoles();
    const hasRole = roles.some(role => userRoles.includes(role));
    
    if (!hasRole) {
      router.navigate(['/unauthorized']);
      return false;
    }
    
    return true;
  };
}

// Usage
const routes: Routes = [
  {
    path: 'admin',
    canActivate: [authGuard, hasRoleGuard(['admin'])],
    loadComponent: () => import('./admin/admin.component')
      .then(m => m.AdminComponent)
  }
];

// ❌ WRONG - Class-based guard (old style)
@Injectable({ providedIn: 'root' })
export class AuthGuard implements CanActivate {
  constructor(
    private authService: AuthService,
    private router: Router
  ) {}
  
  canActivate(): boolean {
    // Implementation
    return true;
  }
}
```

### C. Functional Resolvers
```typescript
// ✅ CORRECT - Functional resolver
import { inject } from '@angular/core';
import { ResolveFn } from '@angular/router';
import { UserService } from '@core/services/user.service';

export const userResolver: ResolveFn<User> = (route, state) => {
  const userService = inject(UserService);
  const userId = route.paramMap.get('id')!;
  
  return userService.getUser(userId);
};

// Usage in routes
const routes: Routes = [
  {
    path: 'users/:id',
    loadComponent: () => import('./user-detail/user-detail.component')
      .then(m => m.UserDetailComponent),
    resolve: {
      user: userResolver
    }
  }
];

// Accessing resolved data in component
@Component({
  selector: 'app-user-detail',
  standalone: true,
  template: `
    <div>
      <h1>{{ user().name }}</h1>
      <p>{{ user().email }}</p>
    </div>
  `
})
export class UserDetailComponent {
  private route = inject(ActivatedRoute);
  
  // Signal from resolved data
  user = signal<User>(this.route.snapshot.data['user']);
  
  // Or reactive approach
  user$ = this.route.data.pipe(
    map(data => data['user'] as User)
  );
}
```

## 9. RxJS Integration

### A. Observable to Signal Conversion
```typescript
// ✅ CORRECT - Converting observables to signals
import { Component, signal } from '@angular/core';
import { toSignal, toObservable } from '@angular/core/rxjs-interop';
import { interval } from 'rxjs';
import { map } from 'rxjs/operators';

@Component({
  selector: 'app-timer',
  standalone: true,
  template: `
    <div>
      <p>Timer: {{ time() }}</p>
      <p>Double: {{ doubleTime() }}</p>
    </div>
  `
})
export class TimerComponent {
  // Observable from timer
  private timer$ = interval(1000);
  
  // Convert observable to signal
  time = toSignal(this.timer$, { initialValue: 0 });
  
  // Computed from signal
  doubleTime = computed(() => this.time() * 2);
  
  // Convert signal back to observable (for RxJS operators)
  time$ = toObservable(this.time);
  
  constructor() {
    // Use observable operators
    this.time$.pipe(
      map(t => t * 3)
    ).subscribe(tripleTime => {
      console.log('Triple time:', tripleTime);
    });
  }
}
```

### B. HTTP with Signals
```typescript
// ✅ CORRECT - HTTP service with signals
import { Injectable, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { toSignal } from '@angular/core/rxjs-interop';
import { catchError, map, of } from 'rxjs';

interface User {
  id: string;
  name: string;
  email: string;
}

@Injectable({ providedIn: 'root' })
export class UserService {
  private http = inject(HttpClient);
  private apiUrl = inject(APP_CONFIG).apiUrl;
  
  // State signals
  isLoading = signal(false);
  error = signal<string | null>(null);
  
  getUsers() {
    this.isLoading.set(true);
    this.error.set(null);
    
    return this.http.get<User[]>(`${this.apiUrl}/users`).pipe(
      map(users => {
        this.isLoading.set(false);
        return users;
      }),
      catchError(err => {
        this.isLoading.set(false);
        this.error.set(err.message);
        return of([]);
      })
    );
  }
  
  // Or use toSignal for reactive data
  users$ = this.http.get<User[]>(`${this.apiUrl}/users`);
  users = toSignal(this.users$, { initialValue: [] });
}

// Component usage
@Component({
  selector: 'app-user-list',
  standalone: true,
  template: `
    @if (userService.isLoading()) {
      <app-spinner />
    } @else if (userService.error()) {
      <app-error [message]="userService.error()" />
    } @else {
      @for (user of users(); track user.id) {
        <app-user-card [user]="user" />
      }
    }
  `
})
export class UserListComponent {
  userService = inject(UserService);
  users = signal<User[]>([]);
  
  ngOnInit(): void {
    this.userService.getUsers().subscribe(users => {
      this.users.set(users);
    });
  }
}
```

### C. Complex RxJS Patterns
```typescript
// ✅ CORRECT - Search with debounce and switchMap
import { Component, signal, DestroyRef, inject } from '@angular/core';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { debounceTime, distinctUntilChanged, switchMap, catchError } from 'rxjs/operators';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { of } from 'rxjs';

@Component({
  selector: 'app-search',
  standalone: true,
  imports: [ReactiveFormsModule, CommonModule],
  template: `
    <div>
      <input
        [formControl]="searchControl"
        placeholder="Search users..."
      />
      
      @if (isLoading()) {
        <app-spinner />
      }
      
      @if (error()) {
        <p class="error">{{ error() }}</p>
      }
      
      @for (result of results(); track result.id) {
        <app-user-card [user]="result" />
      } @empty {
        @if (searchControl.value && !isLoading()) {
          <p>No results found</p>
        }
      }
    </div>
  `
})
export class SearchComponent {
  private searchService = inject(SearchService);
  private destroyRef = inject(DestroyRef);
  
  searchControl = new FormControl('');
  
  results = signal<User[]>([]);
  isLoading = signal(false);
  error = signal<string | null>(null);
  
  ngOnInit(): void {
    this.searchControl.valueChanges.pipe(
      debounceTime(300),
      distinctUntilChanged(),
      switchMap(query => {
        if (!query || query.trim().length === 0) {
          this.results.set([]);
          return of([]);
        }
        
        this.isLoading.set(true);
        this.error.set(null);
        
        return this.searchService.search(query).pipe(
          catchError(err => {
            this.error.set('Search failed');
            return of([]);
          })
        );
      }),
      takeUntilDestroyed(this.destroyRef)
    ).subscribe(results => {
      this.results.set(results);
      this.isLoading.set(false);
    });
  }
}
```

## 10. Functional Interceptors

```typescript
// ✅ CORRECT - Functional HTTP interceptor
import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { AuthService } from '@core/services/auth.service';

// Auth interceptor
export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const authService = inject(AuthService);
  const token = authService.getToken();
  
  if (token) {
    const authReq = req.clone({
      setHeaders: {
        Authorization: `Bearer ${token}`
      }
    });
    return next(authReq);
  }
  
  return next(req);
};

// Error interceptor
export const errorInterceptor: HttpInterceptorFn = (req, next) => {
  return next(req).pipe(
    catchError((error: HttpErrorResponse) => {
      let errorMessage = 'An error occurred';
      
      if (error.error instanceof ErrorEvent) {
        // Client-side error
        errorMessage = `Error: ${error.error.message}`;
      } else {
        // Server-side error
        errorMessage = `Error Code: ${error.status}\nMessage: ${error.message}`;
      }
      
      console.error(errorMessage);
      return throwError(() => new Error(errorMessage));
    })
  );
};

// Logging interceptor
export const loggingInterceptor: HttpInterceptorFn = (req, next) => {
  const startTime = Date.now();
  
  console.log(`Request: ${req.method} ${req.url}`);
  
  return next(req).pipe(
    tap({
      next: (event) => {
        if (event.type === HttpEventType.Response) {
          const elapsedTime = Date.now() - startTime;
          console.log(`Response: ${req.url} - ${elapsedTime}ms`);
        }
      },
      error: (error) => {
        const elapsedTime = Date.now() - startTime;
        console.error(`Error: ${req.url} - ${elapsedTime}ms`, error);
      }
    })
  );
};

// Register in app.config.ts
export const appConfig: ApplicationConfig = {
  providers: [
    provideHttpClient(
      withInterceptors([
        loggingInterceptor,
        authInterceptor,
        errorInterceptor
      ])
    )
  ]
};

// ❌ WRONG - Class-based interceptor (old style)
@Injectable()
export class AuthInterceptor implements HttpInterceptor {
  constructor(private authService: AuthService) {}
  
  intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    // Implementation
    return next.handle(req);
  }
}
```

## 11. Forms (Reactive & Typed)

### A. Typed Forms
```typescript
// ✅ CORRECT - Strongly typed reactive forms
import { Component, signal } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

interface UserForm {
  email: FormControl<string>;
  password: FormControl<string>;
  profile: FormGroup<{
    firstName: FormControl<string>;
    lastName: FormControl<string>;
    age: FormControl<number | null>;
  }>;
}

@Component({
  selector: 'app-user-form',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  template: `
    <form [formGroup]="form" (ngSubmit)="onSubmit()">
      <div>
        <label for="email">Email</label>
        <input
          id="email"
          type="email"
          formControlName="email"
          [class.error]="isFieldInvalid('email')"
        />
        @if (isFieldInvalid('email')) {
          <span class="error-message">
            @if (form.controls.email.errors?.['required']) {
              Email is required
            }
            @if (form.controls.email.errors?.['email']) {
              Invalid email format
            }
          </span>
        }
      </div>
      
      <div>
        <label for="password">Password</label>
        <input
          id="password"
          type="password"
          formControlName="password"
          [class.error]="isFieldInvalid('password')"
        />
        @if (isFieldInvalid('password')) {
          <span class="error-message">
            Password must be at least 8 characters
          </span>
        }
      </div>
      
      <div formGroupName="profile">
        <div>
          <label for="firstName">First Name</label>
          <input id="firstName" formControlName="firstName" />
        </div>
        
        <div>
          <label for="lastName">Last Name</label>
          <input id="lastName" formControlName="lastName" />
        </div>
        
        <div>
          <label for="age">Age</label>
          <input id="age" type="number" formControlName="age" />
        </div>
      </div>
      
      <button
        type="submit"
        [disabled]="form.invalid || isSubmitting()"
      >
        {{ isSubmitting() ? 'Submitting...' : 'Submit' }}
      </button>
    </form>
  `
})
export class UserFormComponent {
  private fb = inject(FormBuilder);
  private userService = inject(UserService);
  
  isSubmitting = signal(false);
  
  form = this.fb.group({
    email: this.fb.control('', {
      validators: [Validators.required, Validators.email],
      nonNullable: true
    }),
    password: this.fb.control('', {
      validators: [Validators.required, Validators.minLength(8)],
      nonNullable: true
    }),
    profile: this.fb.group({
      firstName: this.fb.control('', {
        validators: [Validators.required],
        nonNullable: true
      }),
      lastName: this.fb.control('', {
        validators: [Validators.required],
        nonNullable: true
      }),
      age: this.fb.control<number | null>(null)
    })
  });
  
  isFieldInvalid(fieldName: keyof UserForm): boolean {
    const field = this.form.get(fieldName);
    return !!(field && field.invalid && (field.dirty || field.touched));
  }
  
  async onSubmit(): Promise<void> {
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }
    
    this.isSubmitting.set(true);
    
    try {
      const formValue = this.form.getRawValue();
      await this.userService.createUser(formValue);
      this.form.reset();
    } catch (error) {
      console.error('Form submission failed', error);
    } finally {
      this.isSubmitting.set(false);
    }
  }
}
```

### B. Custom Validators
```typescript
// ✅ CORRECT - Custom validators
import { AbstractControl, ValidationErrors, ValidatorFn } from '@angular/forms';

// Password strength validator
export function passwordStrengthValidator(): ValidatorFn {
  return (control: AbstractControl): ValidationErrors | null => {
    const value = control.value as string;
    
    if (!value) {
      return null;
    }
    
    const hasUpperCase = /[A-Z]/.test(value);
    const hasLowerCase = /[a-z]/.test(value);
    const hasNumeric = /[0-9]/.test(value);
    const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(value);
    
    const passwordValid = hasUpperCase && hasLowerCase && hasNumeric && hasSpecialChar;
    
    return !passwordValid ? { passwordStrength: true } : null;
  };
}

// Match validator (for password confirmation)
export function matchValidator(matchTo: string): ValidatorFn {
  return (control: AbstractControl): ValidationErrors | null => {
    const parent = control.parent;
    
    if (!parent) {
      return null;
    }
    
    const matchToControl = parent.get(matchTo);
    
    if (!matchToControl) {
      return null;
    }
    
    return control.value === matchToControl.value ? null : { match: true };
  };
}

// Usage
form = this.fb.group({
  password: ['', [
    Validators.required,
    Validators.minLength(8),
    passwordStrengthValidator()
  ]],
  confirmPassword: ['', [
    Validators.required,
    matchValidator('password')
  ]]
});
```

## 12. Directives & Pipes

### A. Custom Directive
```typescript
// ✅ CORRECT - Standalone directive
import { Directive, ElementRef, HostListener, input } from '@angular/core';

@Directive({
  selector: '[appHighlight]',
  standalone: true
})
export class HighlightDirective {
  highlightColor = input<string>('yellow');
  
  private el = inject(ElementRef);
  
  @HostListener('mouseenter')
  onMouseEnter(): void {
    this.highlight(this.highlightColor());
  }
  
  @HostListener('mouseleave')
  onMouseLeave(): void {
    this.highlight('');
  }
  
  private highlight(color: string): void {
    this.el.nativeElement.style.backgroundColor = color;
  }
}

// Usage
@Component({
  selector: 'app-example',
  standalone: true,
  imports: [HighlightDirective],
  template: `
    <p appHighlight [highlightColor]="'lightblue'">
      Hover over me!
    </p>
  `
})
export class ExampleComponent {}
```

### B. Custom Pipe
```typescript
// ✅ CORRECT - Standalone pipe
import { Pipe, PipeTransform } from '@angular/core';
import { format } from 'date-fns';

@Pipe({
  name: 'customDate',
  standalone: true
})
export class CustomDatePipe implements PipeTransform {
  transform(value: Date | string | number, formatString: string = 'PP'): string {
    if (!value) return '';
    
    try {
      const date = typeof value === 'string' || typeof value === 'number'
        ? new Date(value)
        : value;
      
      return format(date, formatString);
    } catch (error) {
      console.error('Date formatting error:', error);
      return '';
    }
  }
}

// Usage
@Component({
  selector: 'app-example',
  standalone: true,
  imports: [CustomDatePipe],
  template: `
    <p>{{ createdAt() | customDate:'PPpp' }}</p>
  `
})
export class ExampleComponent {
  createdAt = signal(new Date());
}
```

## 13. Testing

### A. Component Testing
```typescript
// ✅ CORRECT - Testing standalone component with signals
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { UserCardComponent } from './user-card.component';

describe('UserCardComponent', () => {
  let component: UserCardComponent;
  let fixture: ComponentFixture<UserCardComponent>;
  
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [UserCardComponent]  // Import standalone component
    }).compileComponents();
    
    fixture = TestBed.createComponent(UserCardComponent);
    component = fixture.componentInstance;
  });
  
  it('should create', () => {
    expect(component).toBeTruthy();
  });
  
  it('should display user information', () => {
    const mockUser = {
      id: '1',
      name: 'John Doe',
      email: 'john@example.com'
    };
    
    fixture.componentRef.setInput('user', mockUser);
    fixture.detectChanges();
    
    const compiled = fixture.nativeElement;
    expect(compiled.querySelector('h3')?.textContent).toContain('John Doe');
    expect(compiled.querySelector('p')?.textContent).toContain('john@example.com');
  });
  
  it('should emit edit event', () => {
    const mockUser = {
      id: '1',
      name: 'John Doe',
      email: 'john@example.com'
    };
    
    let emittedId: string | undefined;
    fixture.componentRef.setInput('user', mockUser);
    
    component.edit.subscribe((id: string) => {
      emittedId = id;
    });
    
    component.handleEdit();
    
    expect(emittedId).toBe('1');
  });
});
```

### B. Service Testing
```typescript
// ✅ CORRECT - Testing service with signals
import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { UserService } from './user.service';

describe('UserService', () => {
  let service: UserService;
  let httpMock: HttpTestingController;
  
  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [UserService]
    });
    
    service = TestBed.inject(UserService);
    httpMock = TestBed.inject(HttpTestingController);
  });
  
  afterEach(() => {
    httpMock.verify();
  });
  
  it('should fetch users', () => {
    const mockUsers = [
      { id: '1', name: 'User 1', email: 'user1@example.com' },
      { id: '2', name: 'User 2', email: 'user2@example.com' }
    ];
    
    service.getUsers().subscribe();
    
    const req = httpMock.expectOne('/api/users');
    expect(req.request.method).toBe('GET');
    req.flush(mockUsers);
    
    expect(service.isLoading()).toBe(false);
    expect(service.error()).toBeNull();
  });
  
  it('should handle error', () => {
    service.getUsers().subscribe();
    
    const req = httpMock.expectOne('/api/users');
    req.error(new ProgressEvent('Network error'));
    
    expect(service.isLoading()).toBe(false);
    expect(service.error()).toBeTruthy();
  });
});
```

## 14. Performance Optimization

### A. OnPush Change Detection
```typescript
// ✅ CORRECT - OnPush with signals (default behavior with signals)
import { Component, ChangeDetectionStrategy, signal, input } from '@angular/core';

@Component({
  selector: 'app-user-list',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,  // Recommended with signals
  template: `
    @for (user of users(); track user.id) {
      <app-user-card [user]="user" />
    }
  `
})
export class UserListComponent {
  users = input.required<User[]>();
}
```

### B. TrackBy Functions
```typescript
// ✅ CORRECT - Track by with @for
@Component({
  selector: 'app-list',
  standalone: true,
  template: `
    <!-- Modern @for with track -->
    @for (item of items(); track item.id) {
      <div>{{ item.name }}</div>
    }
    
    <!-- Track by index -->
    @for (item of items(); track $index) {
      <div>{{ item }}</div>
    }
  `
})
export class ListComponent {
  items = signal<Item[]>([]);
}
```

### C. Lazy Loading Images
```typescript
// ✅ CORRECT - Lazy loading directive
import { Directive, ElementRef, OnInit, input } from '@angular/core';

@Directive({
  selector: 'img[appLazyLoad]',
  standalone: true
})
export class LazyLoadDirective implements OnInit {
  src = input.required<string>();
  
  private el = inject(ElementRef);
  
  ngOnInit(): void {
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            this.loadImage();
            observer.unobserve(this.el.nativeElement);
          }
        });
      });
      
      observer.observe(this.el.nativeElement);
    } else {
      this.loadImage();
    }
  }
  
  private loadImage(): void {
    const img = this.el.nativeElement as HTMLImageElement;
    img.src = this.src();
  }
}

// Usage
<img appLazyLoad [src]="imageUrl" alt="Lazy loaded image" />
```

## 15. Application Configuration

### A. Main Entry Point
```typescript
// ✅ CORRECT - main.ts with standalone bootstrap
import { bootstrapApplication } from '@angular/platform-browser';
import { AppComponent } from './app/app.component';
import { appConfig } from './app/app.config';

bootstrapApplication(AppComponent, appConfig)
  .catch(err => console.error(err));
```

### B. App Configuration
```typescript
// ✅ CORRECT - app.config.ts
import { ApplicationConfig } from '@angular/core';
import { provideRouter, withComponentInputBinding, withViewTransitions } from '@angular/router';
import { provideHttpClient, withInterceptors, withFetch } from '@angular/common/http';
import { provideAnimations } from '@angular/platform-browser/animations';
import { routes } from './app.routes';
import { authInterceptor, errorInterceptor } from '@core/interceptors';

export const appConfig: ApplicationConfig = {
  providers: [
    provideRouter(
      routes,
      withComponentInputBinding(),  // Bind route params to component inputs
      withViewTransitions()  // Enable view transitions API
    ),
    provideHttpClient(
      withInterceptors([authInterceptor, errorInterceptor]),
      withFetch()  // Use fetch API instead of XMLHttpRequest
    ),
    provideAnimations(),
    // Global services
    { provide: APP_CONFIG, useValue: environment }
  ]
};
```

## 16. Deployment Checklist

### Agent Code Generation Verification (MANDATORY)
**If code was generated by an agent, verify BEFORE delivery:**

#### Test-Driven Development (TDD) Compliance
- [ ] **Tests written BEFORE implementation** (Red-Green-Refactor cycle followed)
- [ ] **Each test failed first**, then passed after implementation
- [ ] **TDD cycle documented** in commit messages or comments
- [ ] **If fixing a bug**: Regression test added BEFORE fix
- [ ] **Regression test reproduces bug** (fails without fix, passes with fix)
- [ ] **Bug ID referenced** in regression test comments

#### Code Quality & Compilation
- [ ] TypeScript compilation successful: `ng build` returns exit code 0
- [ ] All linter checks pass: `ng lint`
- [ ] **All exported components/services documented with JSDoc** (complete with `@param`, `@returns`, `@example`)
- [ ] **Documentation check passes**: `npm run docs:check` returns exit code 0
- [ ] **Documentation can be generated**: `npm run docs` succeeds without errors
- [ ] Strict template type checking passes
- [ ] Unit tests created for ALL new components, services, pipes, directives, guards, and interceptors
- [ ] All tests passing: `ng test --no-watch --browsers=ChromeHeadless` returns exit code 0
- [ ] Test coverage ≥ 80%: `ng test --no-watch --code-coverage --browsers=ChromeHeadless`
- [ ] Production build succeeds: `ng build --configuration production` completes without errors
- [ ] No `any` types used as workarounds
- [ ] No skipped tests (`xit`, `xdescribe`, `fdescribe`, `fit`)
- [ ] No console.log statements in production code
- [ ] All components use standalone: true
- [ ] All templates use modern @if/@for syntax (not *ngIf/*ngFor)
- [ ] All async operations use async/await (not raw Promises or callbacks)
- [ ] All UI components use Angular Material (unless explicitly specified otherwise)
- [ ] Code follows minimalistic principles (no over-engineering)
- [ ] Components and services are small and focused (single responsibility)
- [ ] Agent has documented any complex fixes made during verification

### Pre-Production Validation
- [ ] All TypeScript errors resolved
- [ ] **All components/services documented with JSDoc** (`npm run docs:check`)
- [ ] **API documentation generated successfully** (`npm run docs`)
- [ ] All tests passing (`ng test`)
- [ ] E2E tests passing
- [ ] All components use standalone: true
- [ ] All routes use lazy loading
- [ ] OnPush change detection enabled
- [ ] Proper track-by in @for loops
- [ ] No memory leaks (unsubscribe from observables)
- [ ] All images optimized and lazy-loaded
- [ ] Bundle size analyzed (`ng build --stats-json`)
- [ ] Lighthouse score: Performance ≥ 90, Accessibility ≥ 95
- [ ] WCAG 2.1 AA compliance verified
- [ ] AOT compilation enabled (default in production)
- [ ] Service workers configured (if needed)
- [ ] Environment variables configured
- [ ] Security headers configured
- [ ] Error monitoring configured
- [ ] Analytics tracking implemented

### Performance Targets
- First Contentful Paint (FCP): < 1.8s
- Largest Contentful Paint (LCP): < 2.5s
- Time to Interactive (TTI): < 3.8s
- Total Bundle Size (gzipped): < 200KB (initial)

---

## Why This Configuration Works

1. **Standalone Components**: Simplified architecture, better tree-shaking, faster builds.

2. **Signals**: Fine-grained reactivity, better performance than zone.js, simpler mental model.

3. **JSDoc + TypeDoc**: Auto-generated documentation from code, always in sync, reduces onboarding time by 40%+, better IDE IntelliSense, verified during build.

4. **Async/Await**: Cleaner, more readable async code, easier error handling, no callback hell.

5. **Angular Material**: Consistent UI, accessibility built-in, battle-tested components, responsive design.

6. **Minimalistic Code**: Easier to maintain, faster to understand, fewer bugs, better performance.

7. **Modular Architecture**: Clear separation of concerns, reusable code, easier testing, scalable codebase.

8. **Functional APIs**: Less boilerplate, better type inference, easier testing.

9. **Agent Build Verification**: Ensures generated code compiles, documented, and tests pass before delivery, preventing broken code.

10. **Mandatory Testing**: 80%+ coverage requirement catches bugs early, reduces production issues.

11. **Lazy Loading**: Reduced initial bundle size, faster first paint.

12. **OnPush Detection**: Minimal change detection runs, better performance.

13. **TypeScript Strict**: Catches 15-30% more bugs at compile time.

14. **Modern @-syntax**: Cleaner templates, better performance than structural directives.

15. **RxJS Interop**: Best of both worlds - signals for state, RxJS for reactive streams.

16. **Typed Forms**: Type safety from form to submission, better DX.

17. **Functional Guards/Interceptors**: Less boilerplate, better composition.

---

## Quick Reference

### Common Commands

```bash
# Project Setup
ng new my-app --standalone --routing --style=scss    # Create new standalone Angular app
ng add @angular/material                              # Add Angular Material
ng add @angular/pwa                                   # Add PWA support

# Development
ng serve                                              # Start dev server (http://localhost:4200)
ng serve --port 3000                                  # Start on custom port
ng serve --open                                       # Start and open browser
ng serve --configuration=production                   # Run with production config

# Code Generation
ng generate component features/users/user-list        # Generate component
ng generate component shared/components/button --inline-template --inline-style
ng generate service core/services/auth               # Generate service
ng generate pipe shared/pipes/truncate               # Generate pipe
ng generate directive shared/directives/highlight    # Generate directive
ng generate guard core/guards/auth                   # Generate guard
ng generate interceptor core/interceptors/auth       # Generate interceptor
ng generate interface core/models/user               # Generate interface

# Aliases for generate (shorter commands)
ng g c features/dashboard                            # Generate component
ng g s core/services/api                             # Generate service
ng g p shared/pipes/date-format                      # Generate pipe
ng g d shared/directives/auto-focus                  # Generate directive

# Testing
ng test                                              # Run tests in watch mode
ng test --no-watch                                   # Run tests once
ng test --no-watch --browsers=ChromeHeadless         # Run headless (CI)
ng test --code-coverage                              # Generate coverage report
ng test --include=**/user.service.spec.ts            # Run specific test file

# Building
ng build                                             # Development build
ng build --configuration=production                  # Production build
ng build --configuration=production --stats-json    # Build with bundle stats
ng build --source-map                                # Include source maps

# Linting & Formatting
ng lint                                              # Run ESLint
ng lint --fix                                        # Auto-fix linting issues

# Analysis
ng build --stats-json && npx webpack-bundle-analyzer dist/my-app/stats.json
                                                     # Analyze bundle size

# Documentation (with TypeDoc)
npm run docs                                         # Generate API documentation
npm run docs:check                                   # Verify JSDoc coverage

# Update
ng update                                            # Check for updates
ng update @angular/core @angular/cli                 # Update Angular
ng update @angular/material                          # Update Material

# Other
ng version                                           # Show Angular CLI version
ng cache clean                                       # Clear Angular cache
ng analytics disable                                 # Disable analytics
```

### Angular Patterns Cheat Sheet

```typescript
// ============================================
// COMPONENT PATTERNS
// ============================================

// Signal-based inputs (required)
user = input.required<User>();

// Signal-based inputs (optional with default)
showActions = input(false);

// Signal-based outputs
edit = output<string>();
delete = output<{ id: string; reason: string }>();

// Internal signals
isLoading = signal(false);
error = signal<string | null>(null);

// Computed signals
fullName = computed(() => `${this.user().firstName} ${this.user().lastName}`);

// Signal effects (use sparingly)
constructor() {
  effect(() => {
    console.log('User changed:', this.user());
  });
}

// ============================================
// SERVICE PATTERNS
// ============================================

// Modern service with inject()
@Injectable({ providedIn: 'root' })
export class UserService {
  private http = inject(HttpClient);
  private config = inject(APP_CONFIG);

  users = signal<User[]>([]);
  isLoading = signal(false);
}

// Async/await HTTP calls
async getUsers(): Promise<User[]> {
  return await firstValueFrom(this.http.get<User[]>('/api/users'));
}

// ============================================
// TEMPLATE PATTERNS
// ============================================

// Modern @if syntax
@if (isLoading()) {
  <mat-spinner />
} @else if (error()) {
  <p>Error: {{ error() }}</p>
} @else {
  <div>Content here</div>
}

// Modern @for syntax with track
@for (user of users(); track user.id) {
  <app-user-card [user]="user" />
} @empty {
  <p>No users found</p>
}

// Modern @switch syntax
@switch (status()) {
  @case ('loading') { <mat-spinner /> }
  @case ('error') { <p>Error</p> }
  @default { <div>Content</div> }
}

// ============================================
// ROUTING PATTERNS
// ============================================

// Lazy loaded routes
export const routes: Routes = [
  {
    path: 'users',
    loadComponent: () => import('./features/users/user-list.component')
      .then(m => m.UserListComponent),
    canActivate: [authGuard]
  },
  {
    path: 'dashboard',
    loadChildren: () => import('./features/dashboard/dashboard.routes')
      .then(m => m.DASHBOARD_ROUTES)
  }
];

// Functional guard
export const authGuard: CanActivateFn = (route, state) => {
  const authService = inject(AuthService);
  const router = inject(Router);

  if (authService.isAuthenticated()) {
    return true;
  }

  return router.createUrlTree(['/login'], {
    queryParams: { returnUrl: state.url }
  });
};

// ============================================
// FORM PATTERNS
// ============================================

// Typed reactive forms
interface UserForm {
  name: FormControl<string>;
  email: FormControl<string>;
  age: FormControl<number | null>;
}

form = new FormGroup<UserForm>({
  name: new FormControl('', { nonNullable: true, validators: [Validators.required] }),
  email: new FormControl('', { nonNullable: true, validators: [Validators.required, Validators.email] }),
  age: new FormControl(null, { validators: [Validators.min(0)] })
});

// ============================================
// INTERCEPTOR PATTERNS
// ============================================

// Functional interceptor
export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const authService = inject(AuthService);
  const token = authService.getToken();

  if (token) {
    req = req.clone({
      headers: req.headers.set('Authorization', `Bearer ${token}`)
    });
  }

  return next(req);
};

// Error handling interceptor
export const errorInterceptor: HttpInterceptorFn = (req, next) => {
  return next(req).pipe(
    catchError((error: HttpErrorResponse) => {
      if (error.status === 401) {
        inject(Router).navigate(['/login']);
      }
      return throwError(() => error);
    })
  );
};

// ============================================
// TESTING PATTERNS
// ============================================

// Component test setup
beforeEach(async () => {
  await TestBed.configureTestingModule({
    imports: [ComponentUnderTest],
    providers: [
      { provide: SomeService, useValue: mockService }
    ]
  }).compileComponents();

  fixture = TestBed.createComponent(ComponentUnderTest);
  component = fixture.componentInstance;
  fixture.detectChanges();
});

// Setting signal inputs in tests
fixture.componentRef.setInput('user', mockUser);
fixture.detectChanges();

// Service test with HTTP mock
let httpMock: HttpTestingController;

beforeEach(() => {
  TestBed.configureTestingModule({
    imports: [HttpClientTestingModule],
    providers: [ServiceUnderTest]
  });

  service = TestBed.inject(ServiceUnderTest);
  httpMock = TestBed.inject(HttpTestingController);
});

afterEach(() => {
  httpMock.verify();  // Verify no outstanding requests
});
```

### Project Structure

```
src/
├── app/
│   ├── core/                           # Singleton services, guards, interceptors
│   │   ├── guards/
│   │   │   ├── auth.guard.ts          # Authentication guard
│   │   │   ├── role.guard.ts          # Role-based access guard
│   │   │   └── index.ts               # Public exports
│   │   ├── interceptors/
│   │   │   ├── auth.interceptor.ts    # Add auth token to requests
│   │   │   ├── error.interceptor.ts   # Global error handling
│   │   │   ├── loading.interceptor.ts # Loading state management
│   │   │   └── index.ts
│   │   ├── services/
│   │   │   ├── api.service.ts         # Base API service
│   │   │   ├── auth.service.ts        # Authentication logic
│   │   │   ├── storage.service.ts     # LocalStorage/SessionStorage
│   │   │   └── index.ts
│   │   └── models/
│   │       ├── user.model.ts          # User interface/type
│   │       ├── api-response.model.ts  # API response types
│   │       └── index.ts
│   │
│   ├── shared/                         # Shared components, pipes, directives
│   │   ├── components/
│   │   │   ├── button/
│   │   │   │   ├── button.component.ts
│   │   │   │   ├── button.component.html
│   │   │   │   ├── button.component.scss
│   │   │   │   └── button.component.spec.ts
│   │   │   ├── loading-spinner/
│   │   │   ├── confirm-dialog/
│   │   │   └── index.ts
│   │   ├── directives/
│   │   │   ├── highlight.directive.ts
│   │   │   ├── click-outside.directive.ts
│   │   │   └── index.ts
│   │   ├── pipes/
│   │   │   ├── truncate.pipe.ts
│   │   │   ├── time-ago.pipe.ts
│   │   │   └── index.ts
│   │   └── utils/
│   │       ├── validators.ts          # Custom form validators
│   │       ├── helpers.ts             # Utility functions
│   │       └── index.ts
│   │
│   ├── features/                       # Feature modules (lazy-loaded)
│   │   ├── dashboard/
│   │   │   ├── dashboard.routes.ts    # Feature routes
│   │   │   ├── dashboard.component.ts # Main feature component
│   │   │   ├── components/            # Feature-specific components
│   │   │   │   ├── stats-card/
│   │   │   │   └── activity-feed/
│   │   │   ├── services/              # Feature-specific services
│   │   │   │   └── dashboard.service.ts
│   │   │   └── store/                 # Feature state (NgRx signals)
│   │   │       └── dashboard.store.ts
│   │   │
│   │   ├── users/
│   │   │   ├── users.routes.ts
│   │   │   ├── user-list/
│   │   │   │   ├── user-list.component.ts
│   │   │   │   └── user-list.component.spec.ts
│   │   │   ├── user-detail/
│   │   │   ├── user-form/
│   │   │   └── services/
│   │   │       └── user.service.ts
│   │   │
│   │   └── settings/
│   │       ├── settings.routes.ts
│   │       └── ...
│   │
│   ├── layout/                         # Layout components
│   │   ├── header/
│   │   │   ├── header.component.ts
│   │   │   └── header.component.spec.ts
│   │   ├── footer/
│   │   ├── sidebar/
│   │   └── main-layout/
│   │
│   ├── app.component.ts               # Root component
│   ├── app.component.html
│   ├── app.component.scss
│   ├── app.component.spec.ts
│   ├── app.config.ts                  # Application configuration
│   └── app.routes.ts                  # Root routes
│
├── assets/
│   ├── images/
│   ├── fonts/
│   ├── icons/
│   └── i18n/                          # Translation files
│       ├── en.json
│       └── es.json
│
├── environments/
│   ├── environment.ts                 # Development config
│   └── environment.prod.ts            # Production config
│
├── styles/
│   ├── _variables.scss                # SCSS variables
│   ├── _mixins.scss                   # SCSS mixins
│   ├── _typography.scss               # Typography styles
│   └── styles.scss                    # Global styles
│
├── index.html
├── main.ts                            # Application entry point
└── test.ts                            # Test configuration

# Configuration Files (root)
├── angular.json                       # Angular CLI configuration
├── package.json                       # Dependencies
├── tsconfig.json                      # TypeScript config
├── tsconfig.app.json                  # App-specific TS config
├── tsconfig.spec.json                 # Test-specific TS config
├── karma.conf.js                      # Karma test runner config
├── .eslintrc.json                     # ESLint configuration
└── .prettierrc                        # Prettier configuration
```

### File Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Component | `name.component.ts` | `user-card.component.ts` |
| Service | `name.service.ts` | `auth.service.ts` |
| Pipe | `name.pipe.ts` | `truncate.pipe.ts` |
| Directive | `name.directive.ts` | `highlight.directive.ts` |
| Guard | `name.guard.ts` | `auth.guard.ts` |
| Interceptor | `name.interceptor.ts` | `error.interceptor.ts` |
| Model/Interface | `name.model.ts` | `user.model.ts` |
| Test | `name.*.spec.ts` | `user.service.spec.ts` |
| Routes | `name.routes.ts` | `dashboard.routes.ts` |
| Store | `name.store.ts` | `user.store.ts` |

---

## References

- [Angular Documentation](https://angular.dev/)
- [Angular Signals Guide](https://angular.dev/guide/signals)
- [Standalone Components](https://angular.dev/guide/components/importing)
- [Angular Router](https://angular.dev/guide/routing)
- [RxJS Documentation](https://rxjs.dev/)
- [NgRx Signals](https://ngrx.io/guide/signals)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Web Content Accessibility Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
