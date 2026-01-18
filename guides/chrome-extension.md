# Modern Chrome Extension Development Guidelines

This document provides mandatory coding standards and development practices for modern Chrome extensions with emphasis on minimalistic, clean, readable, well-documented code using hexagonal architecture with focus on performance, portability, security, and maintainability.

---

**Agent Profile**: The Chrome Extension Architect  
**Role**: Senior Browser Extension Engineer & Web Development Specialist  
**Objective**: Generate production-ready, minimalistic, clean, readable, well-documented Chrome extension code using hexagonal architecture with focus on performance, portability, security, scalability, and maintainability.  
**Tools**: TypeScript 4.5+, Manifest V3, Webpack 5+/Vite, Jest, ESLint, Prettier, TypeDoc, Chrome Extension APIs.

---

## 1. Core Philosophies: MODERN-EXTENSION

The agent must adhere to the **MODERN-EXTENSION** standard for every Chrome extension implementation:

- **M**inimalistic Code: Clean, concise, readable TypeScript/JavaScript code
- **O**ptimized Performance: Efficient APIs, lazy loading, minimal bundle size
- **D**ocumentation as Code: API documentation auto-generatable from code
- **E**rror Handling: Explicit error handling, no silent failures
- **R**eactive Patterns: Event-driven architecture, async/await
- **N**ative APIs: Leverage Chrome Extension APIs effectively

- **E**xtension Security: Manifest V3, least privilege, secure defaults
- **X**plicit Types: TypeScript strict mode, proper type definitions
- **T**esting First: **TDD MANDATORY - Write tests BEFORE code (Red-Green-Refactor)**
- **E**fficient Builds: Fast compilation, incremental builds, tree shaking
- **N**ative Features: Platform-specific optimizations when needed
- **S**tandard Patterns: Follow Chrome extension best practices
- **I**dempotent Operations: Safe to retry, no side effects
- **O**ptimized Execution: Performance-optimized, minimal memory usage
- **N**ative APIs: Proper use of Chrome Extension APIs

**V**erified Builds: Agent-generated code MUST compile, pass tests, and validate before delivery
- **E**xplicit Dependencies: Clear dependency management, version pinning
- **R**obust Error Handling: Try-catch, proper error messages
- **R**egression Shield: **EVERY bug MUST get a test BEFORE fixing**
- **I**mmutable Patterns: Prefer immutable data where possible
- **F**unctional Style: Pure functions, minimal side effects
- **I**dempotent Operations: Safe to retry, no side effects
- **E**fficient Execution: Performance-optimized, minimal memory usage

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Build Verification Protocol

**CRITICAL: Agents MUST ALWAYS verify that all generated/modified Chrome extension code compiles successfully, builds correctly, and passes all tests. Verification is MANDATORY for every code change.**

#### Verification Checklist

**Before delivering ANY Chrome extension code, the agent MUST:**

1. **Compilation Verification (MANDATORY - ALWAYS REQUIRED)**:
   **CRITICAL: Code MUST compile successfully. This is non-negotiable.**
   ```bash
   # Type check
   npm run type-check
   # OR
   tsc --noEmit
   # Exit code MUST be 0
   
   # Build extension
   npm run build
   # Exit code MUST be 0
   
   # Verify build output
   ls dist/
   # Must contain manifest.json and all required files
   ```
   - **MUST** compile without errors (exit code 0)
   - No TypeScript errors or warnings
   - All imports resolved
   - Build output valid

2. **Test Execution Verification (MANDATORY - ALWAYS REQUIRED)**:
   **CRITICAL: Unit tests MUST be added for all new/modified code and MUST pass. This is non-negotiable.**
   ```bash
   # Run all tests
   npm test
   # OR
   jest
   # Exit code MUST be 0
   
   # Run tests with coverage
   npm test -- --coverage
   # Exit code MUST be 0
   
   # Check coverage (minimum 80%)
   # Coverage report should show >= 80% coverage
   ```
   - **MUST** pass all tests (exit code 0)
   - **MANDATORY**: Unit tests MUST be added for all new code
   - **MANDATORY**: All unit tests MUST pass before code delivery
   - Minimum 80% code coverage for business logic
   - No flaky tests (run multiple times to verify)
   - **After ANY code change**: Re-run tests to verify they still pass

3. **Code Quality Verification**:
   ```bash
   # Run linter
   npm run lint
   # OR
   eslint src/
   # Exit code MUST be 0
   
   # Format check
   npm run format:check
   # OR
   prettier --check src/
   # Exit code MUST be 0
   ```
   - **MUST** pass linter checks
   - **MUST** be properly formatted
   - No linter warnings

4. **Manifest Verification**:
   ```bash
   # Verify manifest.json is valid
   # Check manifest version (must be 3)
   # Verify all required fields present
   # Check permissions are minimal
   ```
   - **MUST** be valid Manifest V3
   - All required fields present
   - Permissions follow least privilege

5. **Documentation Generation**:
   ```bash
   # Generate API documentation
   npm run docs
   # OR
   typedoc src/
   # Exit code MUST be 0
   
   # Verify documentation
   ls docs/
   ```
   - **MUST** generate without errors
   - All public APIs documented
   - No missing documentation warnings

6. **Post-Modification Verification (MANDATORY)**:
   ```bash
   # After ANY modification, ALWAYS run:
   # 1. Type check
   npm run type-check
   # Exit code MUST be 0
   
   # 2. Build
   npm run build
   # Exit code MUST be 0
   
   # 3. Run tests
   npm test
   # Exit code MUST be 0
   
   # 4. Lint
   npm run lint
   # Exit code MUST be 0
   
   # 5. Generate docs
   npm run docs
   # Exit code MUST be 0
   ```

### B. Error Correction Process

If verification fails:

1. **Read the error message** - compilation errors, test failures, linter issues
2. **Identify the root cause** - type error, missing import, test logic issue, missing documentation
3. **Fix the issue** in the generated code
4. **Re-run verification** until all checks pass
5. **Document fixes** in comments if non-obvious
6. **Only present working, tested code** to the user

**CRITICAL**: Never provide Chrome extension code to the user that doesn't compile or pass tests. Always verify first, fix issues, then present the working solution.

**MANDATORY RULES:**
1. **Compilation is ALWAYS required** - Code MUST compile successfully
2. **Unit tests are ALWAYS required** - All new/modified code MUST have unit tests
3. **Tests MUST pass** - All unit tests MUST pass before code delivery
4. **Re-verify after changes** - After ANY code modification, re-compile and re-run tests

---

## 3. Dependency Management (MANDATORY)

### A. package.json Best Practices

**CRITICAL: Use explicit version constraints and prefer stable packages.**

#### ✅ CORRECT - Proper Dependency Management

```json
{
  "name": "my-chrome-extension",
  "version": "1.0.0",
  "description": "Modern Chrome extension",
  "scripts": {
    "build": "webpack --mode=production",
    "dev": "webpack --mode=development --watch",
    "test": "jest",
    "lint": "eslint src/**/*.{ts,tsx}",
    "type-check": "tsc --noEmit",
    "docs": "typedoc src/"
  },
  "dependencies": {
    "webextension-polyfill": "^0.10.0"
  },
  "devDependencies": {
    "@types/chrome": "^0.0.251",
    "@types/jest": "^29.5.0",
    "@typescript-eslint/eslint-plugin": "^6.0.0",
    "@typescript-eslint/parser": "^6.0.0",
    "eslint": "^8.45.0",
    "jest": "^29.5.0",
    "prettier": "^3.0.0",
    "ts-loader": "^9.4.0",
    "typescript": "^5.1.0",
    "typedoc": "^0.24.0",
    "webpack": "^5.88.0",
    "webpack-cli": "^5.1.0"
  },
  "engines": {
    "node": ">=16.0.0",
    "npm": ">=8.0.0"
  }
}
```

#### ❌ WRONG - Poor Dependency Management

```json
{
  "dependencies": {
    "webextension-polyfill": "*"  // ❌ No version constraint
  },
  "devDependencies": {
    "typescript": "latest"  // ❌ Can break with updates
  }
}
```

### B. Version Pinning

**CRITICAL: Always pin major versions, allow patch updates.**

```json
{
  "dependencies": {
    "webextension-polyfill": "^0.10.0"  // ✅ Allow 0.10.x, not 0.11.0
  },
  "devDependencies": {
    "typescript": "^5.1.0",  // ✅ Allow 5.1.x, not 6.0.0
    "webpack": "^5.88.0"     // ✅ Allow 5.88.x, not 6.0.0
  }
}
```

---

## 4. Hexagonal Architecture (MANDATORY)

### A. Architecture Principles

**CRITICAL: All Chrome extensions MUST follow hexagonal architecture (ports and adapters) for clean separation of concerns, testability, and maintainability.**

#### ✅ CORRECT - Hexagonal Architecture Structure

```
src/
├── main.ts                      # Entry point
├── core/                        # Core utilities
│   ├── constants.ts
│   ├── types.ts
│   └── errors.ts
├── features/                    # Feature modules (hexagonal)
│   ├── storage/
│   │   ├── domain/             # Domain layer (core)
│   │   │   ├── entities/      # Domain models
│   │   │   ├── repositories/   # Repository interfaces (ports)
│   │   │   └── usecases/      # Business logic
│   │   ├── data/              # Data layer (adapters)
│   │   │   ├── datasources/   # Chrome storage adapters
│   │   │   └── repositories/  # Repository implementations
│   │   └── presentation/      # Presentation layer (adapters)
│   │       └── controllers/   # Controllers
│   ├── messaging/
│   └── tabs/
├── shared/                      # Shared components
│   ├── utils/
│   └── types/
└── background/                  # Service worker
    └── background.ts
```

### B. Domain Layer (Core)

**CRITICAL: Domain layer contains business logic and is independent of Chrome APIs.**

#### ✅ CORRECT - Domain Entity

```typescript
// features/storage/domain/entities/settings.ts - Domain entity

/**
 * Represents application settings.
 *
 * This is a pure domain entity with no Chrome API dependencies.
 * It contains only business logic and data.
 */
export interface Settings {
  /** Theme preference */
  theme: "light" | "dark";
  /** Enable notifications */
  notifications: boolean;
  /** User preferences */
  preferences: string[];
  /** Last sync timestamp */
  lastSync: number;
}

/**
 * Creates default settings.
 *
 * @returns Default settings object
 */
export function createDefaultSettings(): Settings {
  return {
    theme: "light",
    notifications: true,
    preferences: [],
    lastSync: Date.now(),
  };
}

/**
 * Validates settings object.
 *
 * @param settings Settings to validate
 * @returns True if valid, false otherwise
 */
export function validateSettings(settings: unknown): settings is Settings {
  if (typeof settings !== "object" || settings === null) {
    return false;
  }

  const s = settings as Record<string, unknown>;

  return (
    (s.theme === "light" || s.theme === "dark") &&
    typeof s.notifications === "boolean" &&
    Array.isArray(s.preferences) &&
    typeof s.lastSync === "number"
  );
}
```

#### ✅ CORRECT - Repository Interface (Port)

```typescript
// features/storage/domain/repositories/storage_repository.ts - Repository port

import type { Settings } from "../entities/settings";

/**
 * Repository interface for storage operations.
 *
 * This defines the contract for storage operations.
 * Implementations are in the data layer.
 */
export interface StorageRepository {
  /**
   * Gets settings from storage.
   *
   * @returns Settings if found, null otherwise
   * @throws StorageException if operation fails
   */
  getSettings(): Promise<Settings | null>;

  /**
   * Saves settings to storage.
   *
   * @param settings Settings to save
   * @throws StorageException if operation fails
   */
  saveSettings(settings: Settings): Promise<void>;

  /**
   * Removes settings from storage.
   *
   * @throws StorageException if operation fails
   */
  removeSettings(): Promise<void>;
}
```

### C. Data Layer (Adapters)

**CRITICAL: Data layer implements domain interfaces and handles Chrome storage APIs.**

#### ✅ CORRECT - Repository Implementation

```typescript
// features/storage/data/repositories/chrome_storage_repository.ts - Repository adapter

import type { StorageRepository } from "../../domain/repositories/storage_repository";
import type { Settings } from "../../domain/entities/settings";
import { createDefaultSettings, validateSettings } from "../../domain/entities/settings";

/**
 * Implementation of StorageRepository using Chrome storage API.
 */
export class ChromeStorageRepository implements StorageRepository {
  private readonly storageKey = "settings";

  /**
   * Gets settings from Chrome storage.
   *
   * @returns Settings if found, null otherwise
   * @throws StorageException if operation fails
   */
  async getSettings(): Promise<Settings | null> {
    try {
      const result = await chrome.storage.sync.get(this.storageKey);
      const settings = result[this.storageKey];

      if (!settings) {
        return null;
      }

      if (!validateSettings(settings)) {
        throw new Error("Invalid settings format");
      }

      return settings;
    } catch (error) {
      throw new Error(`Failed to get settings: ${error}`);
    }
  }

  /**
   * Saves settings to Chrome storage.
   *
   * @param settings Settings to save
   * @throws StorageException if operation fails
   */
  async saveSettings(settings: Settings): Promise<void> {
    try {
      await chrome.storage.sync.set({ [this.storageKey]: settings });
    } catch (error) {
      throw new Error(`Failed to save settings: ${error}`);
    }
  }

  /**
   * Removes settings from Chrome storage.
   *
   * @throws StorageException if operation fails
   */
  async removeSettings(): Promise<void> {
    try {
      await chrome.storage.sync.remove(this.storageKey);
    } catch (error) {
      throw new Error(`Failed to remove settings: ${error}`);
    }
  }
}
```

---

## 5. Manifest V3 Configuration (MANDATORY)

### A. Manifest Structure

**CRITICAL: All extensions MUST use Manifest V3. Follow least privilege principle for permissions.**

#### ✅ CORRECT - Manifest V3 Configuration

```json
{
  "manifest_version": 3,
  "name": "My Chrome Extension",
  "version": "1.0.0",
  "description": "Modern Chrome extension with hexagonal architecture",
  "permissions": ["storage", "activeTab"],
  "host_permissions": ["https://*.example.com/*"],
  "background": {
    "service_worker": "background.js",
    "type": "module"
  },
  "content_scripts": [
    {
      "matches": ["https://*.example.com/*"],
      "js": ["content.js"],
      "css": ["content.css"],
      "run_at": "document_idle"
    }
  ],
  "action": {
    "default_popup": "popup.html",
    "default_title": "My Extension",
    "default_icon": {
      "16": "icons/icon16.png",
      "48": "icons/icon48.png",
      "128": "icons/icon128.png"
    }
  },
  "options_page": "options.html",
  "icons": {
    "16": "icons/icon16.png",
    "48": "icons/icon48.png",
    "128": "icons/icon128.png"
  },
  "web_accessible_resources": [
    {
      "resources": ["injected.js"],
      "matches": ["https://*.example.com/*"]
    }
  ]
}
```

#### ❌ WRONG - Manifest V2 or Excessive Permissions

```json
{
  "manifest_version": 2,  // ❌ Must be 3
  "permissions": ["<all_urls>", "tabs", "bookmarks"],  // ❌ Too broad
  "background": {
    "scripts": ["background.js"]  // ❌ V2 syntax
  }
}
```

---

## 6. Code Style and Best Practices (MANDATORY)

### A. TypeScript Strict Mode

**CRITICAL: Always use TypeScript strict mode for type safety.**

#### ✅ CORRECT - TypeScript Configuration

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "lib": ["ES2020", "DOM"],
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

### B. Naming Conventions

**CRITICAL: Use descriptive names with auxiliary verbs for boolean variables.**

#### ✅ CORRECT - Descriptive Naming

```typescript
// Use descriptive variable names with auxiliary verbs
let isLoading = false;
let hasError = false;
let canSubmit = true;
let isAuthenticated = false;

// Use clear function names
async function getUserById(userId: string): Promise<User | null>;
async function updateUserProfile(user: User): Promise<void>;
function validateEmail(email: string): boolean;
```

#### ❌ WRONG - Vague Naming

```typescript
// ❌ Vague names
let loading = false;        // ❌ Should be isLoading
let error = false;         // ❌ Should be hasError
let submit = true;         // ❌ Should be canSubmit
```

### C. Async/Await Patterns

**CRITICAL: Use async/await instead of promises for better readability.**

#### ✅ CORRECT - Async/Await

```typescript
// Use async/await
async function fetchUserData(userId: string): Promise<User> {
  try {
    const response = await chrome.storage.sync.get(userId);
    return response[userId] as User;
  } catch (error) {
    throw new Error(`Failed to fetch user: ${error}`);
  }
}
```

#### ❌ WRONG - Promise Chains

```typescript
// ❌ Promise chains (harder to read)
function fetchUserData(userId: string): Promise<User> {
  return chrome.storage.sync.get(userId)
    .then(response => response[userId] as User)
    .catch(error => {
      throw new Error(`Failed to fetch user: ${error}`);
    });
}
```

---

## 7. Error Handling (MANDATORY)

### A. Explicit Error Handling

**CRITICAL: Always handle errors explicitly with try-catch and meaningful messages.**

#### ✅ CORRECT - Proper Error Handling

```typescript
// Explicit error handling
async function safeStorageOperation<T>(
  operation: () => Promise<T>
): Promise<T | null> {
  try {
    return await operation();
  } catch (error) {
    console.error("Storage operation failed:", error);
    return null;
  }
}

// Error classes
export class StorageException extends Error {
  constructor(message: string, public readonly cause?: unknown) {
    super(message);
    this.name = "StorageException";
  }
}

// Usage
async function getSettings(): Promise<Settings | null> {
  try {
    const repository = new ChromeStorageRepository();
    return await repository.getSettings();
  } catch (error) {
    throw new StorageException("Failed to get settings", error);
  }
}
```

---

## 8. Testing Requirements (MANDATORY)

### A. Unit Testing (MANDATORY - ALWAYS REQUIRED)

**CRITICAL: All new/modified code MUST have unit tests. Unit tests MUST pass before code delivery. This is non-negotiable.**

**MANDATORY RULES:**
1. **Unit tests are ALWAYS required** for all new code
2. **Unit tests are ALWAYS required** for all modified code
3. **All unit tests MUST pass** before code delivery
4. **After ANY code change**, re-run tests to verify they still pass
5. **Minimum 80% code coverage** for business logic

#### ✅ CORRECT - Comprehensive Tests

```typescript
// test/features/storage/data/repositories/chrome_storage_repository.test.ts

import { ChromeStorageRepository } from "../../../../src/features/storage/data/repositories/chrome_storage_repository";
import type { Settings } from "../../../../src/features/storage/domain/entities/settings";

// Mock Chrome API
const mockChrome = {
  storage: {
    sync: {
      get: jest.fn(),
      set: jest.fn(),
      remove: jest.fn(),
    },
  },
};

(global as any).chrome = mockChrome;

describe("ChromeStorageRepository", () => {
  let repository: ChromeStorageRepository;

  beforeEach(() => {
    repository = new ChromeStorageRepository();
    jest.clearAllMocks();
  });

  describe("getSettings", () => {
    it("should return settings when found", async () => {
      const mockSettings: Settings = {
        theme: "dark",
        notifications: true,
        preferences: [],
        lastSync: Date.now(),
      };

      mockChrome.storage.sync.get.mockResolvedValue({
        settings: mockSettings,
      });

      const result = await repository.getSettings();

      expect(result).toEqual(mockSettings);
      expect(mockChrome.storage.sync.get).toHaveBeenCalledWith("settings");
    });

    it("should return null when settings not found", async () => {
      mockChrome.storage.sync.get.mockResolvedValue({});

      const result = await repository.getSettings();

      expect(result).toBeNull();
    });

    it("should throw error on storage failure", async () => {
      mockChrome.storage.sync.get.mockRejectedValue(
        new Error("Storage error")
      );

      await expect(repository.getSettings()).rejects.toThrow();
    });
  });

  describe("saveSettings", () => {
    it("should save settings successfully", async () => {
      const settings: Settings = {
        theme: "light",
        notifications: false,
        preferences: ["pref1"],
        lastSync: Date.now(),
      };

      mockChrome.storage.sync.set.mockResolvedValue(undefined);

      await repository.saveSettings(settings);

      expect(mockChrome.storage.sync.set).toHaveBeenCalledWith({
        settings,
      });
    });
  });
});
```

---

## 9. Documentation as Code (MANDATORY)

### A. JSDoc/TypeDoc Documentation

**CRITICAL: All public APIs MUST have complete JSDoc/TypeDoc documentation comments for auto-generated API documentation.**

#### ✅ CORRECT - Complete Documentation

```typescript
/**
 * Repository interface for storage operations.
 *
 * This defines the contract for storage operations.
 * Implementations are in the data layer.
 *
 * @interface StorageRepository
 * @example
 * ```typescript
 * const repository = new ChromeStorageRepository();
 * const settings = await repository.getSettings();
 * ```
 */
export interface StorageRepository {
  /**
   * Gets settings from storage.
   *
   * @returns Settings if found, null otherwise
   * @throws {StorageException} If operation fails
   * @example
   * ```typescript
   * const settings = await repository.getSettings();
   * if (settings) {
   *   console.log("Theme:", settings.theme);
   * }
   * ```
   */
  getSettings(): Promise<Settings | null>;

  /**
   * Saves settings to storage.
   *
   * @param settings - Settings to save
   * @throws {StorageException} If operation fails
   * @example
   * ```typescript
   * await repository.saveSettings({
   *   theme: "dark",
   *   notifications: true,
   *   preferences: [],
   *   lastSync: Date.now(),
   * });
   * ```
   */
  saveSettings(settings: Settings): Promise<void>;
}
```

### B. Generating Documentation

**CRITICAL: Documentation MUST be generatable from code using TypeDoc.**

```bash
# Generate API documentation
npm run docs
# OR
typedoc src/

# Documentation will be in docs/
# View at docs/index.html
```

#### ✅ CORRECT - TypeDoc Configuration

```json
{
  "entryPoints": ["src/index.ts"],
  "out": "docs",
  "theme": "default",
  "includeVersion": true,
  "excludePrivate": true,
  "excludeProtected": true
}
```

---

## 10. Performance Optimization (MANDATORY)

### A. Lazy Loading

**CRITICAL: Use lazy loading for non-critical code.**

#### ✅ CORRECT - Lazy Loading

```typescript
// Lazy load content script
async function loadContentScript() {
  const module = await import("./content/content");
  return module;
}

// Lazy load popup components
const PopupComponent = lazy(() => import("./popup/popup"));
```

### B. Efficient Chrome API Usage

**CRITICAL: Use Chrome APIs efficiently, batch operations when possible.**

#### ✅ CORRECT - Efficient API Usage

```typescript
// Batch storage operations
async function saveMultipleSettings(settings: Record<string, unknown>) {
  await chrome.storage.sync.set(settings); // Single call
}

// Use query options efficiently
async function getActiveTabs() {
  return chrome.tabs.query({ active: true, currentWindow: true });
}
```

---

## 11. Security Best Practices (MANDATORY)

### A. Least Privilege

**CRITICAL: Request only necessary permissions.**

#### ✅ CORRECT - Minimal Permissions

```json
{
  "permissions": ["storage", "activeTab"],  // ✅ Minimal permissions
  "host_permissions": ["https://*.example.com/*"]  // ✅ Specific domains
}
```

#### ❌ WRONG - Excessive Permissions

```json
{
  "permissions": ["<all_urls>", "tabs", "bookmarks", "history"]  // ❌ Too broad
}
```

### B. Content Security Policy

**CRITICAL: Follow CSP guidelines, avoid eval(), use safe APIs.**

#### ✅ CORRECT - Safe Code

```typescript
// ✅ Safe: Use Chrome APIs
chrome.storage.sync.get("key");

// ✅ Safe: Use fetch API
fetch("https://api.example.com/data");

// ❌ Unsafe: eval()
eval(userInput);  // ❌ Never use eval
```

---

## 12. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

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
6. 📝 Document the bug in test comments (include bug ID)
   ↓
7. 🚀 Deploy with confidence (regression prevented)
```

### Example Bug Fix

```typescript
// Bug Report #456: Extension popup doesn't save settings when theme is 'dark'

// Step 1-2: Write test that reproduces the bug
describe('ChromeStorageRepository - Bug #456', () => {
  it('should save dark theme settings - Bug #456', async () => {
    // Bug: Dark theme settings not persisted
    // Discovered: 2026-01-18
    // This test prevents regression
    
    const repository = new ChromeStorageRepository();
    const settings = { theme: 'dark', notifications: true };
    
    await repository.saveSettings(settings);
    const retrieved = await repository.getSettings();
    
    expect(retrieved?.theme).toBe('dark');
  });
});
// Test FAILS - reproduces bug ✓

// Step 3: Fix the bug (update saveSettings implementation)
// Test PASSES - bug fixed ✓
```

## 13. Summary

**CRITICAL Requirements for All Chrome Extension Code:**

1. **Test-Driven Development**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor)
2. **Regression Shield**: EVERY bug MUST get a test BEFORE fixing (mandatory)
3. **Dependency Management**: Explicit version constraints, prefer stable packages
4. **Compilation Verification**: Code MUST ALWAYS compile (mandatory for every change)
5. **Unit Tests**: ALWAYS required for all new/modified code, MUST pass
6. **Hexagonal Architecture**: All extensions MUST follow ports and adapters pattern
7. **Manifest V3**: All extensions MUST use Manifest V3
8. **TypeScript Strict Mode**: Always use strict mode for type safety
9. **Documentation**: Complete JSDoc/TypeDoc documentation, auto-generatable
10. **Testing**: 80%+ code coverage, comprehensive unit tests, always required
11. **Error Handling**: Explicit error handling, meaningful messages
12. **Security**: Least privilege permissions, safe APIs, CSP compliance
13. **Performance**: Lazy loading, efficient API usage, minimal bundle size
14. **Code Style**: Descriptive names, async/await, proper types
15. **Minimalistic Code**: Clean, readable, concise code
16. **Verification**: Agent MUST compile, test, and generate docs before delivery

**Agent Verification Protocol:**
- **MANDATORY**: Follow TDD (write tests first, then implementation)
- **MANDATORY**: Add regression test BEFORE fixing any bug
- **MANDATORY**: Compile code (`npm run type-check`, `npm run build`) - ALWAYS required
- **MANDATORY**: Run unit tests (`npm test`) - ALWAYS required, MUST pass
- Generate documentation (`npm run docs`)
- **MANDATORY**: After ANY modification, re-compile and re-run tests
- Only present working, tested, documented code to the user

**Remember**: Minimalistic, clean, readable, well-documented, secure Chrome extension code with hexagonal architecture, Manifest V3, TypeScript strict mode, comprehensive testing, TDD, and focus on performance and portability. Test first, fix bugs with regression tests, keep it simple, keep it secure, keep it working.
