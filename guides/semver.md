# Semantic Versioning Guidelines
Mandatory standards for versioning software following Semantic Versioning (SemVer) 2.0.0. semantic-release, standard-version, commitizen, lerna.

---

**Agent Profile**: The Versioning Expert
**Role**: Senior Release Engineer & API Compatibility Specialist
**Objective**: Generate consistent, predictable version numbers that communicate change impact clearly.
**Tools**: semantic-release, standard-version, commitizen, lerna.

---

## 1. Core Philosophies: SEMVER-FIRST

- **S**tandard: Follow SemVer 2.0.0 specification
- **E**xplicit: Version numbers communicate meaning
- **M**aintainable: Clear upgrade paths for users
- **V**erifiable: Automated version determination
- **E**volutionary: Allow growth while maintaining compatibility
- **R**eliable: Predictable behavior across versions

---

## 2. Version Format (MANDATORY)

### A. Basic Structure

```
MAJOR.MINOR.PATCH

Examples:
1.0.0
2.4.1
10.20.300
```

### B. Components

```markdown
## MAJOR version (X.y.z)
Increment when you make incompatible API changes.

Examples:
- Removing a public API method
- Changing method signature
- Changing behavior in incompatible way
- Dropping support for older platform version

## MINOR version (x.Y.z)
Increment when you add functionality in a backwards compatible manner.

Examples:
- Adding new public methods
- Adding optional parameters
- Adding new features
- Deprecating functionality (without removing)

## PATCH version (x.y.Z)
Increment when you make backwards compatible bug fixes.

Examples:
- Fixing bugs without changing API
- Performance improvements
- Documentation fixes
- Security patches
```

### C. Pre-release and Build Metadata

```
VERSION = MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

Pre-release versions:
1.0.0-alpha
1.0.0-alpha.1
1.0.0-beta.2
1.0.0-rc.1

Build metadata:
1.0.0+20240115
1.0.0+build.123
1.0.0-beta.1+build.456

Precedence (lowest to highest):
1.0.0-alpha < 1.0.0-alpha.1 < 1.0.0-beta < 1.0.0-rc.1 < 1.0.0
```

---

## 3. Version Increment Rules (MANDATORY)

### A. Decision Tree

```markdown
## When to increment MAJOR (Breaking Change)

Ask: "Will this change break existing users?"

Breaking changes include:
- Removing public API
- Renaming public API without alias
- Changing return types
- Changing parameter types
- Changing default behavior
- Removing configuration options
- Requiring new dependencies
- Dropping platform support

## When to increment MINOR (New Feature)

Ask: "Does this add new capability without breaking existing code?"

New features include:
- New public methods/classes
- New optional parameters
- New configuration options
- New supported platforms
- New functionality
- Deprecation notices

## When to increment PATCH (Bug Fix)

Ask: "Does this fix a bug without adding features or breaking compatibility?"

Bug fixes include:
- Correcting incorrect behavior
- Fixing security vulnerabilities
- Fixing memory leaks
- Fixing race conditions
- Documentation corrections
- Performance improvements (same behavior)
```

### B. Examples

```javascript
// Version 1.0.0 - Initial release
function getUser(id) {
  return database.findUser(id);
}

// Version 1.0.1 - PATCH: Bug fix
function getUser(id) {
  if (!id) throw new Error('ID required'); // Fix: was silently failing
  return database.findUser(id);
}

// Version 1.1.0 - MINOR: New feature
function getUser(id, options = {}) {
  // NEW: Added options parameter (optional, backwards compatible)
  if (!id) throw new Error('ID required');
  return database.findUser(id, options);
}

// NEW in 1.1.0: Added new method
function getUserByEmail(email) {
  return database.findUserByEmail(email);
}

// Version 2.0.0 - MAJOR: Breaking change
async function getUser(id, options = {}) {
  // BREAKING: Changed from sync to async
  if (!id) throw new Error('ID required');
  return await database.findUser(id, options);
}
```

---

## 4. Pre-release Versions (MANDATORY)

### A. Pre-release Naming

```markdown
## Alpha (a.b.c-alpha.N)
- Early development stage
- APIs may change significantly
- Not feature complete
- For internal testing only

## Beta (a.b.c-beta.N)
- Feature complete (mostly)
- APIs relatively stable
- May have known bugs
- For external testing

## Release Candidate (a.b.c-rc.N)
- Production ready candidate
- All features complete
- All known bugs fixed
- Final testing before release
```

### B. Pre-release Workflow

```bash
# Development workflow
1.0.0-alpha.1   # First alpha
1.0.0-alpha.2   # Second alpha (bug fixes)
1.0.0-alpha.3   # Third alpha (more changes)
1.0.0-beta.1    # Feature freeze, start beta
1.0.0-beta.2    # Beta bug fixes
1.0.0-rc.1      # Release candidate
1.0.0-rc.2      # RC bug fixes
1.0.0           # Stable release!

# Continued development
1.1.0-alpha.1   # Start next minor version
```

---

## 5. Conventional Commits Integration (MANDATORY)

### A. Commit Types and Version Impact

```markdown
## PATCH increment triggers:
- fix: Bug fixes
- perf: Performance improvements
- revert: Reverting changes

## MINOR increment triggers:
- feat: New features

## MAJOR increment triggers:
- Any commit with "BREAKING CHANGE:" in body/footer
- Any commit with "!" after type (e.g., "feat!:", "fix!:")

## No version change:
- docs: Documentation only
- style: Formatting
- refactor: Code restructuring
- test: Adding tests
- chore: Maintenance tasks
- ci: CI configuration
- build: Build system changes
```

### B. Commit Examples

```bash
# PATCH increment
git commit -m "fix: correct null pointer exception in user lookup"
git commit -m "perf: optimize database query for user list"

# MINOR increment
git commit -m "feat: add email notification support"
git commit -m "feat(api): add pagination to user endpoint"

# MAJOR increment (breaking change)
git commit -m "feat!: change authentication to OAuth 2.0"

git commit -m "refactor: rename User to Account

BREAKING CHANGE: User class renamed to Account. Update all imports."

git commit -m "feat(api): change response format

BREAKING CHANGE: API now returns data in { data: ..., meta: ... } format
instead of raw arrays."
```

---

## 6. Automation (MANDATORY)

### A. semantic-release Configuration

```json
// package.json
{
  "name": "my-package",
  "version": "0.0.0-development",
  "release": {
    "branches": ["main", "next"],
    "plugins": [
      "@semantic-release/commit-analyzer",
      "@semantic-release/release-notes-generator",
      "@semantic-release/changelog",
      "@semantic-release/npm",
      "@semantic-release/github",
      "@semantic-release/git"
    ]
  }
}
```

```javascript
// release.config.js
module.exports = {
  branches: [
    'main',
    { name: 'beta', prerelease: true },
    { name: 'alpha', prerelease: true }
  ],
  plugins: [
    ['@semantic-release/commit-analyzer', {
      preset: 'conventionalcommits',
      releaseRules: [
        { type: 'docs', scope: 'README', release: 'patch' },
        { type: 'refactor', release: 'patch' },
        { type: 'style', release: 'patch' },
        { type: 'perf', release: 'patch' },
        { breaking: true, release: 'major' }
      ]
    }],
    ['@semantic-release/release-notes-generator', {
      preset: 'conventionalcommits',
      presetConfig: {
        types: [
          { type: 'feat', section: 'Features' },
          { type: 'fix', section: 'Bug Fixes' },
          { type: 'perf', section: 'Performance' },
          { type: 'revert', section: 'Reverts' },
          { type: 'docs', section: 'Documentation', hidden: false },
          { type: 'chore', section: 'Miscellaneous', hidden: true }
        ]
      }
    }],
    '@semantic-release/changelog',
    '@semantic-release/npm',
    ['@semantic-release/git', {
      assets: ['CHANGELOG.md', 'package.json'],
      message: 'chore(release): ${nextRelease.version} [skip ci]\n\n${nextRelease.notes}'
    }],
    '@semantic-release/github'
  ]
};
```

### B. GitHub Actions Workflow

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    branches: [main, beta, alpha]

permissions:
  contents: write
  issues: write
  pull-requests: write
  packages: write

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-node@v4
        with:
          node-version: 20

      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test

      - name: Release
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          NPM_TOKEN: ${{ secrets.NPM_TOKEN }}
        run: npx semantic-release
```

---

## 7. Changelog Generation (MANDATORY)

### A. Changelog Format

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New feature in development

## [2.0.0] - 2024-02-01

### Added
- OAuth 2.0 authentication support
- New `/api/v2/users` endpoint

### Changed
- **BREAKING:** API response format changed to `{ data, meta }` structure
- **BREAKING:** Minimum Node.js version is now 18

### Deprecated
- `/api/v1/users` endpoint (will be removed in 3.0.0)

### Removed
- **BREAKING:** Removed support for Node.js 14 and 16

### Fixed
- Memory leak in connection pool
- Race condition in batch processing

### Security
- Updated dependencies to patch CVE-2024-XXXX

## [1.5.0] - 2024-01-15

### Added
- Pagination support for list endpoints
- Email notification feature

### Fixed
- Incorrect timezone handling in date parsing

## [1.4.1] - 2024-01-10

### Fixed
- Critical bug in payment processing

### Security
- Patched SQL injection vulnerability
```

---

## 8. Version Ranges (MANDATORY)

### A. Dependency Ranges

```json
{
  "dependencies": {
    // Exact version
    "lodash": "4.17.21",

    // Patch updates allowed (4.17.x)
    "axios": "~1.6.0",

    // Minor updates allowed (1.x.x)
    "react": "^18.2.0",

    // Any version >= 2.0.0 and < 3.0.0
    "express": ">=2.0.0 <3.0.0",

    // Latest version (not recommended for production)
    "dev-tool": "*"
  }
}
```

### B. Range Recommendations

```markdown
## For Libraries (published packages)
Use caret (^) for dependencies to allow minor updates:
- "react": "^18.0.0" allows 18.0.0 to 18.x.x

## For Applications
Use exact versions or tilde (~) for more control:
- "express": "4.18.2" - exact version
- "lodash": "~4.17.0" - allows 4.17.x

## For Development Dependencies
Caret (^) is usually fine:
- "jest": "^29.0.0"
- "typescript": "^5.0.0"

## Lock Files
Always commit lock files:
- package-lock.json (npm)
- yarn.lock (yarn)
- pnpm-lock.yaml (pnpm)
```

---

## 9. Breaking Change Guidelines (MANDATORY)

### A. Communicating Breaking Changes

```markdown
## In Commit Messages
feat!: remove deprecated authentication method

BREAKING CHANGE: The `basicAuth` method has been removed.
Use `oauth2Auth` instead.

Migration:
- Before: client.basicAuth(user, pass)
- After: client.oauth2Auth({ clientId, clientSecret })

## In Changelog
### [3.0.0] - 2024-03-01

### Removed
- **BREAKING:** `basicAuth()` method removed. Use `oauth2Auth()` instead.
  See [Migration Guide](./docs/migration-v3.md).

## In Release Notes
## Breaking Changes

### Authentication Method Changed

The `basicAuth` method has been removed in favor of OAuth 2.0.

**Before (v2.x):**
```javascript
const client = new Client();
client.basicAuth('user', 'password');
```

**After (v3.x):**
```javascript
const client = new Client();
client.oauth2Auth({
  clientId: 'your-client-id',
  clientSecret: 'your-client-secret'
});
```

See the [full migration guide](./docs/migration-v3.md) for details.
```

### B. Deprecation Process

```javascript
// Step 1: Deprecate in MINOR version (e.g., 2.5.0)
/**
 * @deprecated Use newMethod() instead. Will be removed in v3.0.0.
 */
function oldMethod() {
  console.warn('Warning: oldMethod() is deprecated. Use newMethod() instead.');
  return newMethod();
}

// Step 2: Keep deprecated method working through 2.x
// Step 3: Remove in MAJOR version (3.0.0)
```

---

## 10. Special Cases (MANDATORY)

### A. Version 0.x.x (Initial Development)

```markdown
## 0.y.z - Initial Development Phase

During initial development (0.x.x):
- API may change at any time
- MINOR version increments may include breaking changes
- PATCH version for bug fixes

Example progression:
0.1.0 - Initial alpha
0.2.0 - Major API redesign (breaking changes OK)
0.2.1 - Bug fix
0.3.0 - More breaking changes
1.0.0 - First stable release (API stability commitment)
```

### B. Monorepo Versioning

```json
// lerna.json - Independent versioning
{
  "version": "independent",
  "packages": ["packages/*"],
  "command": {
    "version": {
      "conventionalCommits": true,
      "message": "chore(release): publish"
    }
  }
}

// lerna.json - Fixed versioning
{
  "version": "2.0.0",
  "packages": ["packages/*"]
}
```

---

## 11. Deployment Checklist

### Before Release
- [ ] All tests passing
- [ ] Changelog updated
- [ ] Breaking changes documented
- [ ] Migration guide written (if needed)
- [ ] Version number correct

### During Release
- [ ] Tag created
- [ ] Package published
- [ ] Release notes published
- [ ] Announcements made

### After Release
- [ ] Verify package installable
- [ ] Update documentation site
- [ ] Notify users of breaking changes
- [ ] Monitor for issues

---

## 12. Quick Reference

```markdown
## Version Format
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

## When to Increment
MAJOR: Breaking changes
MINOR: New features (backwards compatible)
PATCH: Bug fixes (backwards compatible)

## Commit Prefixes
fix:  → PATCH
feat: → MINOR
BREAKING CHANGE: → MAJOR

## Version Ranges
^1.2.3 → 1.x.x (>=1.2.3 <2.0.0)
~1.2.3 → 1.2.x (>=1.2.3 <1.3.0)
1.2.3  → exact version

## Pre-release Order
alpha < beta < rc < stable
1.0.0-alpha.1 < 1.0.0-beta.1 < 1.0.0-rc.1 < 1.0.0
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Release Team


**End of Semantic Versioning Guidelines**
