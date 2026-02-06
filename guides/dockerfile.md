# Dockerfile Guidelines
Mandatory coding style and practices for creation of Dockerfiles and containers. Secure, minimal, cache-optimized. Docker Engine 24.x+, BuildKit, OCI Standards.

---

**Agent Profile**: The Container Architect
**Role**: Senior DevOps Engineer & Container Security Specialist
**Objective**: Generate production-ready, secure, and highly optimized Dockerfiles.
**Tools**: Docker Engine 24.x+, BuildKit, OCI Standards.

---

## 1. Core Philosophies: DOCKERFILE-FIRST

The agent must adhere to the **DOCKERFILE-FIRST** principles for every Dockerfile generated:

**Test-Driven Development (TDD)**: ALWAYS verify builds and runtime behavior BEFORE delivery (build → test → fix cycle).
**Regression Shield**: EVERY bug or security issue discovered MUST be fixed and re-verified before delivery.

- **S**mall: Minimize image size (MBs, not GBs).
- **S**ecure: Least privilege (non-root), minimal attack surface (Distroless/Alpine).
- **S**peedy: Maximize layer caching and build parallelism with optimal cache layer ordering.
- **V**erified: Agent-generated Dockerfiles MUST build successfully before delivery.

## 2. Agent Build Verification Requirements (MANDATORY)

### A. Build Verification Protocol

**CRITICAL: Agents MUST verify that all generated/modified Dockerfiles build successfully before presenting them to the user.**

#### Verification Checklist

**Before delivering ANY Dockerfile, the agent MUST:**

1. **Build Verification**:
   ```bash
   # Build the Docker image
   docker build -t test-image:latest .
   
   # OR with BuildKit explicitly enabled
   DOCKER_BUILDKIT=1 docker build -t test-image:latest .
   ```
   - **MUST** return exit code 0 (no errors)
   - Address ALL build errors, not just warnings
   - Verify all stages complete successfully in multi-stage builds

2. **Layer Caching Validation**:
   ```bash
   # Build twice to verify caching works
   docker build -t test-image:latest .
   docker build -t test-image:latest .  # Should use cache
   ```
   - Second build should use cached layers
   - Verify cache mounts are properly configured
   - Ensure dependency layers are cached separately from source code layers

3. **Image Inspection**:
   ```bash
   # Check image size
   docker images test-image:latest
   
   # Inspect image layers
   docker history test-image:latest
   
   # Verify non-root user
   docker inspect test-image:latest | grep -i user
   ```
   - Image size should be reasonable (< 500MB for most apps)
   - No excessive layers (combine RUN commands)
   - USER directive is set (not running as root)

4. **Runtime Verification**:
   ```bash
   # Test container starts successfully
   docker run --rm test-image:latest --help
   
   # OR for services
   docker run -d -p 8080:8080 test-image:latest
   docker ps  # Verify container is running
   docker logs <container-id>  # Check for startup errors
   docker stop <container-id>
   ```
   - Container must start without errors
   - Application should respond appropriately
   - No permission errors

5. **Security Validation**:
   ```bash
   # Verify running as non-root
   docker run --rm test-image:latest id
   # Should NOT show uid=0(root)
   ```
   - Must run as non-root user
   - No world-writable files in image

#### Error Correction Process

If verification fails:

1. **Identify the error**: Read the full Docker build error message
2. **Locate the source**: Identify which layer/command failed
3. **Fix the root cause**:
   - Missing dependencies? Add to package install
   - Permission errors? Fix COPY ownership or USER directive
   - Path issues? Verify WORKDIR and file locations
4. **Re-verify**: Run build again until it succeeds
5. **Test caching**: Rebuild to ensure cache layers work
6. **Document changes**: Note any significant fixes made

### B. Layer Caching Optimization (MANDATORY)

**CRITICAL: Optimize Dockerfile layer ordering for maximum cache efficiency.**

#### Cache Layer Ordering Rules

**ORDER LAYERS FROM LEAST TO MOST FREQUENTLY CHANGING:**

1. **System packages** (rarely change)
2. **Dependency files** (package.json, requirements.txt, go.mod)
3. **Dependency installation** (npm install, pip install)
4. **Source code** (most frequently changes)

```dockerfile
# ✅ CORRECT - Optimal cache layer ordering
FROM node:20-alpine AS base
WORKDIR /app

# 1. System packages (rarely change) - cached unless base image updates
RUN apk add --no-cache curl

# 2. Dependency files only (change less frequently than source)
COPY package.json package-lock.json ./

# 3. Install dependencies (reuses cache if package files unchanged)
RUN --mount=type=cache,target=/root/.npm \
    npm ci --omit=dev

# 4. Source code (changes most frequently) - only this layer rebuilds on code changes
COPY . .

# 5. Build/start command
CMD ["node", "index.js"]

# ❌ WRONG - Poor cache ordering (copies everything first)
FROM node:20-alpine
WORKDIR /app
COPY . .  # ← Invalidates cache on ANY file change
RUN npm ci  # ← Always rebuilds dependencies even if unchanged
CMD ["node", "index.js"]
```

#### Cache Mount Best Practices

**Use BuildKit cache mounts for package managers:**

```dockerfile
# ✅ CORRECT - Node.js with cache mounts
FROM node:20-alpine AS deps
WORKDIR /app

# Bind mount package files, cache npm registry
RUN --mount=type=bind,source=package.json,target=package.json \
    --mount=type=bind,source=package-lock.json,target=package-lock.json \
    --mount=type=cache,target=/root/.npm \
    npm ci --omit=dev

# ✅ CORRECT - Python with cache mounts
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ✅ CORRECT - Go with cache mounts
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download

# ✅ CORRECT - Rust with cache mounts
FROM rust:1.75-alpine AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release

# ❌ WRONG - No cache mounts (slow rebuilds)
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci  # Downloads everything every time
```

#### Cache Invalidation Awareness

**Understand what invalidates Docker cache:**

```dockerfile
# ✅ CORRECT - Separate concerns for better caching
FROM node:20-alpine
WORKDIR /app

# These rarely change - cached layer
ENV NODE_ENV=production \
    PORT=3000

# Package files change occasionally - separate layer
COPY package.json package-lock.json ./

# Installation cached if package files unchanged
RUN npm ci --omit=dev

# Source code changes frequently - only this layer rebuilds
COPY src/ ./src/

# ❌ WRONG - Mixed concerns break caching
FROM node:20-alpine
WORKDIR /app

# Any change to .env invalidates ALL subsequent layers
COPY .env package.json package-lock.json ./
RUN npm ci  # Rebuilds even if only .env changed

COPY src/ ./src/
```

#### Multi-Stage Build Caching

```dockerfile
# ✅ CORRECT - Each stage caches independently
FROM node:20-alpine AS base
WORKDIR /app
# Base layer cached

FROM base AS deps
COPY package*.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --omit=dev
# Deps layer cached if package files unchanged

FROM base AS build
COPY package*.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci
COPY . .
RUN npm run build
# Build layer cached if source unchanged

FROM base AS runner
USER node
COPY --from=deps /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
# Runner only rebuilds if previous stages change
CMD ["node", "dist/index.js"]
```

### C. Agent Workflow Example

**Complete agent Dockerfile generation workflow:**

1. **Generate Dockerfile**:
   ```dockerfile
   FROM node:20-alpine
   WORKDIR /app
   COPY package*.json ./
   RUN npm ci
   COPY . .
   CMD ["node", "index.js"]
   ```

2. **Build and Verify**:
   ```bash
   docker build -t test-app:latest .
   # ✓ Build completed successfully
   ```

3. **Test Caching**:
   ```bash
   # Modify source file
   echo "console.log('test');" >> index.js
   
   # Rebuild
   docker build -t test-app:latest .
   # ✓ Using cache for dependency layers
   ```

4. **Verify Runtime**:
   ```bash
   docker run --rm test-app:latest
   # ✓ Application starts successfully
   ```

5. **Check Security**:
   ```bash
   docker run --rm test-app:latest id
   # ⚠ uid=0(root) - NEEDS FIX
   ```

6. **Fix Issues** (add non-root user):
   ```dockerfile
   FROM node:20-alpine
   WORKDIR /app
   COPY package*.json ./
   RUN npm ci
   COPY . .
   USER node  # ← Fixed
   CMD ["node", "index.js"]
   ```

7. **Re-verify**:
   ```bash
   docker build -t test-app:latest .
   docker run --rm test-app:latest id
   # ✓ uid=1000(node)
   ```

8. **Present Dockerfile**: Only after ALL checks pass

### D. Prohibited Practices

**NEVER deliver a Dockerfile that:**
- ❌ Fails to build
- ❌ Runs as root user in production
- ❌ Uses `:latest` tags without version pinning
- ❌ Has poor cache layer ordering (dependencies after source code)
- ❌ Lacks cache mounts for package managers
- ❌ Copies entire context before installing dependencies
- ❌ Has multiple COPY operations that could be combined
- ❌ Includes secrets in build arguments or environment variables
- ❌ Results in images > 1GB for simple applications
- ❌ Has excessive layers (> 50 layers)
- ❌ Lacks a .dockerignore file

---

## 2A. TDD Protocol for Dockerfiles

### Test-Driven Dockerfile Development

**Apply TDD principles to container development for reliable, secure, and optimized images.**

```
┌─────────────────────────────────────────────────────────────────┐
│                    DOCKERFILE TDD CYCLE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐              │
│    │  RED     │────▶│  GREEN   │────▶│ REFACTOR │              │
│    │  Write   │     │  Build   │     │ Optimize │              │
│    │  Tests   │     │  Image   │     │  Layers  │              │
│    └──────────┘     └──────────┘     └──────────┘              │
│         ▲                                   │                   │
│         │                                   │                   │
│         └───────────────────────────────────┘                   │
│                                                                 │
│    RED:     Define structure tests (container-structure-test)   │
│             Define lint rules (hadolint)                        │
│                                                                 │
│    GREEN:   Write Dockerfile that passes all tests              │
│             Verify build succeeds, container runs               │
│                                                                 │
│    REFACTOR: Optimize layers, reduce size                       │
│              Improve cache efficiency                           │
│              Re-run tests to verify no regressions              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### A. Testing Tools

**1. Hadolint (Static Analysis)**

Hadolint is a Dockerfile linter that validates best practices and security rules.

```bash
# Install hadolint
brew install hadolint  # macOS
# OR download binary from https://github.com/hadolint/hadolint/releases

# Run linter
hadolint Dockerfile

# With specific rules
hadolint --ignore DL3008 --ignore DL3018 Dockerfile

# Output as JSON for CI integration
hadolint -f json Dockerfile
```

**2. Container Structure Test (Runtime Validation)**

Google's container-structure-test validates the actual container contents.

```bash
# Install container-structure-test
curl -LO https://storage.googleapis.com/container-structure-test/latest/container-structure-test-linux-amd64
chmod +x container-structure-test-linux-amd64
sudo mv container-structure-test-linux-amd64 /usr/local/bin/container-structure-test

# Run tests
container-structure-test test --image myapp:latest --config tests/container-structure.yaml
```

### B. TDD Workflow Example

**Step 1: RED - Write Tests First**

Create `tests/container-structure.yaml`:

```yaml
schemaVersion: 2.0.0

# Test 1: Verify non-root user
commandTests:
  - name: "Must run as non-root user"
    command: "id"
    expectedOutput: ["uid=1000"]
    excludedOutput: ["uid=0(root)"]

  - name: "Application binary exists and is executable"
    command: "which"
    args: ["node"]
    expectedOutput: ["/usr/local/bin/node"]
    exitCode: 0

  - name: "Application responds to --version"
    command: "node"
    args: ["--version"]
    expectedOutput: ["v20"]
    exitCode: 0

# Test 2: Verify file structure
fileExistenceTests:
  - name: "Application source exists"
    path: "/app/src/index.js"
    shouldExist: true
    permissions: "-rw-r--r--"

  - name: "node_modules exists"
    path: "/app/node_modules"
    shouldExist: true
    isDirectory: true

  - name: "No .git directory in image"
    path: "/app/.git"
    shouldExist: false

  - name: "No secrets in image"
    path: "/app/.env"
    shouldExist: false

# Test 3: Verify metadata
metadataTest:
  envVars:
    - key: "NODE_ENV"
      value: "production"
    - key: "PORT"
      value: "3000"
  exposedPorts: ["3000"]
  workdir: "/app"
  user: "node"

# Test 4: Verify no development dependencies
fileContentTests:
  - name: "No dev dependencies in node_modules"
    path: "/app/package.json"
    excludedContents: ["devDependencies"]
```

Create `.hadolint.yaml` for lint rules:

```yaml
ignored:
  - DL3008  # Pin versions in apt-get (may be intentional for security updates)

trustedRegistries:
  - docker.io
  - gcr.io

failure-threshold: warning

override:
  error:
    - DL3000  # Use absolute WORKDIR
    - DL3001  # No relative WORKDIR
    - DL3002  # No switching to root after USER
    - DL3003  # Use WORKDIR instead of cd
    - DL3004  # No sudo
    - DL3006  # Pin image versions
    - DL3007  # No :latest tag
    - DL3045  # COPY --link
  warning:
    - DL3025  # Use JSON array for CMD
```

**Step 2: RED - Run Tests (They Should Fail)**

```bash
# Lint empty/minimal Dockerfile - expect failures
hadolint Dockerfile
# Output: Errors about missing USER, unpinned versions, etc.

# Structure test on placeholder image - expect failures
docker build -t myapp:test .
container-structure-test test --image myapp:test --config tests/container-structure.yaml
# Output: FAIL - user is root, files missing, etc.
```

**Step 3: GREEN - Write Dockerfile to Pass Tests**

```dockerfile
# syntax=docker/dockerfile:1.6

FROM node:20.10-alpine AS base
WORKDIR /app
ENV NODE_ENV=production \
    PORT=3000

FROM base AS deps
RUN --mount=type=bind,source=package.json,target=package.json \
    --mount=type=bind,source=package-lock.json,target=package-lock.json \
    --mount=type=cache,target=/root/.npm \
    npm ci --omit=dev

FROM base AS runner
USER node
COPY --from=deps --chown=node:node /app/node_modules ./node_modules
COPY --chown=node:node src/ ./src/
COPY --chown=node:node package.json ./

EXPOSE 3000
CMD ["node", "src/index.js"]
```

**Step 4: GREEN - Verify Tests Pass**

```bash
# Lint passes
hadolint Dockerfile
# No output = success

# Build image
docker build -t myapp:test .

# Structure tests pass
container-structure-test test --image myapp:test --config tests/container-structure.yaml
# Output: PASS

# Manual verification
docker run --rm myapp:test id
# uid=1000(node) gid=1000(node)
```

**Step 5: REFACTOR - Optimize**

```dockerfile
# syntax=docker/dockerfile:1.6

# Optimized with labels and health check
FROM node:20.10-alpine AS base
WORKDIR /app
ENV NODE_ENV=production \
    PORT=3000

LABEL org.opencontainers.image.source="https://github.com/org/repo" \
      org.opencontainers.image.description="Production Node.js app" \
      org.opencontainers.image.version="1.0.0"

FROM base AS deps
RUN --mount=type=bind,source=package.json,target=package.json \
    --mount=type=bind,source=package-lock.json,target=package-lock.json \
    --mount=type=cache,target=/root/.npm \
    npm ci --omit=dev

FROM base AS runner
USER node
COPY --from=deps --chown=node:node /app/node_modules ./node_modules
COPY --chown=node:node src/ ./src/
COPY --chown=node:node package.json ./

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

EXPOSE 3000
CMD ["node", "src/index.js"]
```

**Step 6: REFACTOR - Verify Tests Still Pass**

```bash
# Re-run all tests after refactoring
hadolint Dockerfile && \
docker build -t myapp:test . && \
container-structure-test test --image myapp:test --config tests/container-structure.yaml
# All tests should still pass
```

### C. CI Integration Example

```yaml
# .github/workflows/docker.yml
name: Docker CI

on: [push, pull_request]

jobs:
  test-dockerfile:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Lint Dockerfile
        uses: hadolint/hadolint-action@v3.1.0
        with:
          dockerfile: Dockerfile
          failure-threshold: warning

      - name: Build image
        run: docker build -t myapp:test .

      - name: Run structure tests
        uses: plexsystems/container-structure-test-action@v0.3.0
        with:
          image: myapp:test
          config: tests/container-structure.yaml
```

---

## 2B. Bug Fix Protocol for Dockerfiles

### Systematic Approach to Debugging Container Issues

**Follow this workflow to diagnose and fix Dockerfile problems methodically.**

```
┌─────────────────────────────────────────────────────────────────┐
│                  DOCKERFILE BUG FIX WORKFLOW                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. REPRODUCE        2. ISOLATE         3. DIAGNOSE             │
│  ┌──────────┐       ┌──────────┐       ┌──────────┐            │
│  │ Build &  │──────▶│ Identify │──────▶│ Analyze  │            │
│  │  Run     │       │  Layer   │       │  Logs    │            │
│  └──────────┘       └──────────┘       └──────────┘            │
│                                              │                  │
│                                              ▼                  │
│  6. VERIFY          5. TEST FIX        4. IMPLEMENT             │
│  ┌──────────┐       ┌──────────┐       ┌──────────┐            │
│  │ Run Full │◀──────│ Targeted │◀──────│ Minimal  │            │
│  │  Suite   │       │  Tests   │       │  Change  │            │
│  └──────────┘       └──────────┘       └──────────┘            │
│                                                                 │
│  REPRODUCE:  Capture exact error, build logs, runtime logs      │
│  ISOLATE:    Find failing layer/stage using --target            │
│  DIAGNOSE:   Use docker history, inspect, shell access          │
│  IMPLEMENT:  Make smallest fix, document reasoning              │
│  TEST:       Add regression test for the specific bug           │
│  VERIFY:     Full test suite passes, cache still works          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### A. Bug Diagnosis Commands

**1. Build Failure Diagnosis**

```bash
# Show full build output
docker build --progress=plain -t myapp:debug . 2>&1 | tee build.log

# Build up to a specific stage
docker build --target deps -t myapp:deps .

# Build without cache to see fresh errors
docker build --no-cache -t myapp:debug .

# Show detailed layer information
docker history --no-trunc myapp:debug
```

**2. Runtime Failure Diagnosis**

```bash
# Get shell access to debug
docker run -it --rm myapp:debug /bin/sh

# Override entrypoint to debug startup
docker run -it --rm --entrypoint /bin/sh myapp:debug

# View container logs
docker logs <container-id>
docker logs --follow <container-id>

# Inspect container state
docker inspect <container-id>

# Check file permissions
docker run --rm myapp:debug ls -la /app

# Check running processes
docker run --rm myapp:debug ps aux

# Check environment variables
docker run --rm myapp:debug env
```

**3. Layer Analysis**

```bash
# Inspect image layers
docker history myapp:debug

# Export and inspect filesystem
docker save myapp:debug | tar -xf - -C /tmp/image-layers
ls /tmp/image-layers

# Check image size by layer
docker history --format "{{.Size}}\t{{.CreatedBy}}" myapp:debug

# Use dive for interactive layer analysis
dive myapp:debug
```

### B. Common Bug Patterns and Fixes

**Bug Pattern 1: Permission Denied Errors**

```
# SYMPTOM
EACCES: permission denied, open '/app/data/file.txt'

# DIAGNOSIS
docker run --rm myapp:debug ls -la /app
# Shows: drwxr-xr-x root root /app/data

# ROOT CAUSE
Files copied as root, but running as non-root user

# FIX
```

```dockerfile
# ❌ BEFORE (Bug)
COPY . /app
USER node
CMD ["node", "index.js"]

# ✅ AFTER (Fixed)
COPY --chown=node:node . /app
USER node
CMD ["node", "index.js"]
```

**Bug Pattern 2: Missing Dependencies at Runtime**

```
# SYMPTOM
Error: Cannot find module 'express'

# DIAGNOSIS
docker run --rm myapp:debug ls /app/node_modules
# Shows: empty or missing directory

# ROOT CAUSE
node_modules not copied from deps stage OR wrong stage order

# FIX
```

```dockerfile
# ❌ BEFORE (Bug)
FROM base AS runner
COPY . .  # Missing node_modules!
CMD ["node", "index.js"]

# ✅ AFTER (Fixed)
FROM base AS runner
COPY --from=deps /app/node_modules ./node_modules
COPY . .
CMD ["node", "index.js"]
```

**Bug Pattern 3: Cache Invalidation Issues**

```
# SYMPTOM
Every build reinstalls all dependencies even when package.json unchanged

# DIAGNOSIS
docker build -t myapp:test . 2>&1 | grep -E "(CACHED|RUN)"
# Shows: No CACHED layers for dependency installation

# ROOT CAUSE
COPY . . before dependency installation invalidates cache

# FIX
```

```dockerfile
# ❌ BEFORE (Bug - poor cache ordering)
FROM node:20-alpine
WORKDIR /app
COPY . .  # Invalidates cache on ANY file change
RUN npm ci
CMD ["node", "index.js"]

# ✅ AFTER (Fixed - optimal cache ordering)
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json ./  # Only dependency files
RUN npm ci
COPY . .  # Source code last
CMD ["node", "index.js"]
```

**Bug Pattern 4: Multi-Stage Build Artifacts Missing**

```
# SYMPTOM
Error: /app/dist/index.js: No such file or directory

# DIAGNOSIS
docker build --target build -t myapp:build .
docker run --rm myapp:build ls -la /app/dist
# Verify build output exists in build stage

# ROOT CAUSE
Wrong path in COPY --from or build output path changed

# FIX
```

```dockerfile
# ❌ BEFORE (Bug)
FROM base AS build
RUN npm run build  # Outputs to /app/build

FROM base AS runner
COPY --from=build /app/dist ./dist  # Wrong path!

# ✅ AFTER (Fixed)
FROM base AS build
RUN npm run build  # Outputs to /app/build

FROM base AS runner
COPY --from=build /app/build ./dist  # Correct path
```

**Bug Pattern 5: Health Check Failures**

```
# SYMPTOM
Container marked unhealthy, constant restarts

# DIAGNOSIS
docker inspect <container-id> | jq '.[0].State.Health'
# Shows: "Status": "unhealthy", "FailingStreak": 3

# ROOT CAUSE
Health check command fails due to missing tools or wrong endpoint

# FIX
```

```dockerfile
# ❌ BEFORE (Bug - curl not installed in alpine)
FROM node:20-alpine
HEALTHCHECK CMD curl -f http://localhost:3000/health || exit 1

# ✅ AFTER (Fixed - use wget which is available in alpine)
FROM node:20-alpine
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1
```

### C. Bug Fix Workflow Example

**Scenario: Container fails to start with "ENOENT: no such file or directory"**

**Step 1: REPRODUCE - Capture the error**

```bash
docker build -t myapp:debug .
docker run --rm myapp:debug

# Output:
# Error: ENOENT: no such file or directory, open '/app/config/default.json'
```

**Step 2: ISOLATE - Find the problem layer**

```bash
# Check if file exists in build context
ls -la config/default.json
# EXISTS locally

# Check if file exists in image
docker run --rm --entrypoint /bin/sh myapp:debug -c "ls -la /app/config/"
# Output: No such file or directory

# Check .dockerignore
cat .dockerignore
# Found: config/  # <-- This is excluding the config directory!
```

**Step 3: DIAGNOSE - Understand the root cause**

```bash
# The config directory is in .dockerignore
# This prevents it from being copied into the image
```

**Step 4: IMPLEMENT - Make minimal fix**

```bash
# Option A: Remove config/ from .dockerignore
# Option B: Copy config explicitly before the ignore takes effect
```

Update `.dockerignore`:

```
# ❌ BEFORE
config/

# ✅ AFTER
config/*.local.json
config/*.secret.json
# Keep default.json but exclude sensitive configs
```

**Step 5: TEST - Add regression test**

Add to `tests/container-structure.yaml`:

```yaml
fileExistenceTests:
  - name: "Config file must exist"
    path: "/app/config/default.json"
    shouldExist: true
```

**Step 6: VERIFY - Full test suite**

```bash
# Rebuild
docker build -t myapp:debug .

# Test the fix
docker run --rm myapp:debug
# Application starts successfully

# Run structure tests
container-structure-test test --image myapp:debug --config tests/container-structure.yaml
# PASS

# Verify cache still works
docker build -t myapp:debug .
# Uses cached layers appropriately
```

### D. Bug Fix Documentation Template

When fixing Dockerfile bugs, document using this template:

```markdown
## Bug Fix: [Brief Description]

**Symptom**: [What error was observed]

**Root Cause**: [Why the error occurred]

**Fix Applied**:
- File: [Dockerfile / .dockerignore / etc.]
- Change: [What was changed]

**Regression Test Added**:
- [Description of test that prevents this bug from recurring]

**Verification**:
- [ ] Image builds successfully
- [ ] Container starts without errors
- [ ] Structure tests pass
- [ ] Cache behavior verified
```

---

## 3. Mandatory Instructions
### A. Base Images & Tagging
* Pin Versions: Never use :latest. Use specific semantic versions (e.g., python:3.11.4-slim-bookworm or node:20.9-alpine).

* Digests (High Security): For critical infrastructure, pin by SHA256 digest alongside the tag.

* Platform: Use official Docker Hub images or Google Distroless images for final stages.

### B. The Multi-Stage Pattern
All Dockerfiles must utilize Multi-Stage Builds.

* base: Define shared environment variables and OS packages.

* deps: Install dependencies (use cache mounts).

* build: Compile code or build assets (transpilation).

* runner: The final runtime image. Copy only necessary artifacts (binary/dist folder) from previous stages.

### C. BuildKit Optimizations
* Cache Mounts: Use --mount=type=cache for package managers (apt, pip, npm, go mod) to speed up re-builds.

* Bind Mounts: Use --mount=type=bind to access source code without COPYing it during the build stage (reduces layer size).

* Secret Mounts: NEVER use ARG for credentials. Use --mount=type=secret for build-time secrets (e.g., .npmrc, private git keys).

* Heredocs: Use <<EOF syntax for multi-line RUN commands to reduce layer overhead and improve readability.

### D. Security & Permissions (MANDATORY)

**CRITICAL: Security is non-negotiable. All containers MUST follow these requirements.**

> **Cross-Reference**: For runtime security settings (read_only, cap_drop, security_opt), see the [Docker Compose Guidelines](docker-compose.md) Section G.

#### Security Requirements Summary (MANDATORY CHECKLIST)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DOCKERFILE SECURITY REQUIREMENTS                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ REQUIREMENT                          │ PRIORITY  │ ENFORCEMENT              │
├──────────────────────────────────────┼───────────┼──────────────────────────┤
│ FROM scratch (Go, Rust, C)           │ MANDATORY │ Block PR if not used     │
│ FROM distroless (Python, Node, Java) │ MANDATORY │ Block PR if not used     │
│ Alpine/slim as LAST RESORT only      │ FALLBACK  │ Requires justification   │
│ Non-root USER directive              │ MANDATORY │ Block PR if missing      │
│ COPY --link for all COPY statements  │ MANDATORY │ Block PR if missing      │
│ No shell in final image              │ MANDATORY │ Distroless/scratch only  │
│ Exec form for ENTRYPOINT/CMD         │ MANDATORY │ No shell form allowed    │
│ No SUID/SGID binaries                │ MANDATORY │ Remove in final stage    │
│ No secrets in ARG/ENV                │ MANDATORY │ Use --mount=type=secret  │
│ Prepared for read-only filesystem    │ MANDATORY │ Test with --read-only    │
│ Security scan passes                 │ MANDATORY │ Trivy HIGH/CRITICAL = 0  │
│ Image size minimized                 │ MANDATORY │ <100MB target            │
└──────────────────────────────────────┴───────────┴──────────────────────────┘
```

#### Image Selection Hierarchy (MANDATORY)

**ALWAYS prefer images in this strict order:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PRIORITY 1: FROM scratch (REQUIRED for compiled languages)                  │
│   └─► Go, Rust, C/C++, .NET self-contained                                  │
│   └─► Target: 5-20MB images                                                 │
│   └─► Zero attack surface, no shell, no package manager                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ PRIORITY 2: Distroless (REQUIRED for interpreted languages)                 │
│   └─► Python, Node.js, Java                                                 │
│   └─► Target: 50-100MB images                                               │
│   └─► No shell, no package manager, minimal attack surface                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ PRIORITY 3: Alpine (LAST RESORT - requires written justification)           │
│   └─► Only when scratch/distroless impossible                               │
│   └─► Target: <150MB images                                                 │
│   └─► Has shell but minimal packages                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ NEVER: Full images (Debian, Ubuntu, etc.)                                   │
│   └─► Development only, NEVER in production                                 │
│   └─► 500MB-1GB+ images = REJECTED                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 1. Non-Root Execution (MANDATORY)

```dockerfile
# ✅ CORRECT - Create and use non-root user
FROM node:20-alpine AS final
RUN addgroup -g 1001 -S appgroup && \
    adduser -u 1001 -S appuser -G appgroup
USER appuser
WORKDIR /home/appuser/app

# ✅ CORRECT - Use built-in non-root user
FROM node:20-alpine
USER node

# ✅ CORRECT - Distroless with nonroot
FROM gcr.io/distroless/static-debian12
USER nonroot:nonroot

# ❌ WRONG - Running as root
FROM node:20-alpine
WORKDIR /app
CMD ["node", "index.js"]  # Runs as root!
```

#### 2. Minimal Base Images (MANDATORY)

**Prefer images in this order (most to least secure):**

| Priority | Image Type | Use Case | Example |
|----------|-----------|----------|---------|
| 1 | `scratch` | Static binaries (Go, Rust) | `FROM scratch` |
| 2 | `distroless` | Runtime-only (no shell) | `gcr.io/distroless/static-debian12` |
| 3 | `*-alpine` | Needs package manager | `python:3.12-alpine` |
| 4 | `*-slim` | Debian minimal | `python:3.12-slim-bookworm` |
| 5 | Full images | Development only | `python:3.12` |

```dockerfile
# ✅ BEST - FROM scratch for static binaries
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s" -o /app/server .

FROM scratch
COPY --from=builder /app/server /server
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
USER 65534:65534
ENTRYPOINT ["/server"]

# ✅ EXCELLENT - Distroless for interpreted languages
FROM python:3.12-slim AS builder
WORKDIR /app
RUN pip install --target=/app/deps -r requirements.txt

FROM gcr.io/distroless/python3-debian12
COPY --from=builder /app/deps /app/deps
COPY . /app
ENV PYTHONPATH=/app/deps
USER nonroot
CMD ["app/main.py"]
```

#### 3. COPY --link (MANDATORY for BuildKit)

**CRITICAL: ALL COPY statements MUST use `--link` flag for optimal caching and parallel builds.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COPY --link REQUIREMENTS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ STATEMENT               │ REQUIRED FORMAT                                   │
├─────────────────────────┼───────────────────────────────────────────────────┤
│ Copy from build stage   │ COPY --link --from=builder /app/bin /app/bin      │
│ Copy source files       │ COPY --link . .                                   │
│ Copy with ownership     │ COPY --link --chown=user:group src/ ./src/        │
│ Copy single file        │ COPY --link package.json ./                       │
└─────────────────────────┴───────────────────────────────────────────────────┘
```

```dockerfile
# syntax=docker/dockerfile:1.6

# ✅ CORRECT - COPY --link creates independent layers
FROM node:20-alpine AS deps
WORKDIR /app
COPY --link package*.json ./
RUN npm ci --omit=dev

FROM node:20-alpine AS final
WORKDIR /app
# --link creates layer independent of previous layers
COPY --link --from=deps /app/node_modules ./node_modules
COPY --link --chown=node:node . .
USER node
CMD ["node", "index.js"]

# ❌ WRONG - Without --link, layers depend on all previous layers
COPY --from=deps /app/node_modules ./node_modules
COPY . .
```

**Benefits of `--link`:**
- Layers can be built in parallel (faster builds)
- Cache is more granular (doesn't invalidate unnecessarily)
- Enables content-addressable storage optimization
- Reduces layer coupling (changes to earlier layers don't invalidate COPY layers)

**ENFORCEMENT: PRs without `--link` on COPY statements MUST be rejected.**

#### 4. No Shells in Production (MANDATORY)

```dockerfile
# ✅ CORRECT - Exec form (no shell)
ENTRYPOINT ["python", "app.py"]
CMD ["--config", "/etc/app/config.yaml"]

# ✅ CORRECT - Distroless has no shell
FROM gcr.io/distroless/static-debian12
COPY myapp /myapp
ENTRYPOINT ["/myapp"]

# ❌ WRONG - Shell form (vulnerable to shell injection)
CMD python app.py --config /etc/app/config.yaml
ENTRYPOINT /start.sh
```

#### 5. No SUID/SGID Binaries

```dockerfile
# ✅ CORRECT - Remove SUID/SGID bits in final image
FROM alpine:3.19
RUN find / -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true

# For Debian/Ubuntu
RUN find / -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true
```

#### 6. Read-Only Filesystem Preparation (MANDATORY)

**CRITICAL: All images MUST be designed to run with `--read-only` filesystem at runtime.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    READ-ONLY FILESYSTEM REQUIREMENTS                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ REQUIREMENT                          │ IMPLEMENTATION                       │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ No writes to root filesystem         │ Use tmpfs for /tmp, caches           │
│ Application logs to stdout           │ Never write log files to disk        │
│ Config via environment/secrets       │ Never write config files at runtime  │
│ Explicit writable paths only         │ Mount volumes for data directories   │
│ Pre-create directories in build      │ mkdir -p && chown in Dockerfile      │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

```dockerfile
# ✅ CORRECT - Prepare for read-only runtime
FROM node:20-alpine AS final

# Create necessary writable directories with correct ownership
RUN mkdir -p /tmp /var/cache/app /app/data && \
    chown -R node:node /tmp /var/cache/app /app/data

# Set environment for read-only operation
ENV NODE_ENV=production \
    npm_config_cache=/var/cache/app \
    HOME=/tmp

USER node
WORKDIR /app
COPY --link --chown=node:node . .
CMD ["node", "index.js"]

# Runtime: docker run --read-only --tmpfs /tmp:size=100M myapp
```

**Language-Specific Read-Only Preparation:**

```dockerfile
# Python - disable bytecode, set cache dirs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_CACHE_DIR=/tmp/pip \
    HOME=/tmp

# Node.js - set npm cache and home
ENV npm_config_cache=/tmp/npm \
    HOME=/tmp

# Go - typically no special env needed (static binary)
# Just ensure no file writes in application code

# Java - set temp directories
ENV JAVA_OPTS="-Djava.io.tmpdir=/tmp" \
    HOME=/tmp
```

**Testing Read-Only Compatibility (MANDATORY before deployment):**

```bash
# Test that container works with read-only filesystem
docker run --rm --read-only --tmpfs /tmp:size=100M myapp:latest

# If this fails, the image is NOT production-ready
```

#### 7. Security Scanning Requirements

**All images MUST pass security scanning before deployment:**

```bash
# Scan with Trivy (recommended)
trivy image --severity HIGH,CRITICAL myapp:latest

# Scan with Grype
grype myapp:latest

# Scan with Docker Scout
docker scout cves myapp:latest
```

```dockerfile
# Add scanning as build stage
FROM aquasec/trivy:latest AS scanner
COPY --from=final / /scan-target
RUN trivy filesystem --exit-code 1 --severity HIGH,CRITICAL /scan-target
```

#### 8. Secret Handling (MANDATORY)

```dockerfile
# ✅ CORRECT - Use secret mounts (never in image)
RUN --mount=type=secret,id=npmrc,target=/root/.npmrc \
    npm ci --omit=dev

RUN --mount=type=secret,id=pip_conf,target=/etc/pip.conf \
    pip install -r requirements.txt

# ❌ WRONG - Secret in build args (visible in history)
ARG NPM_TOKEN
RUN echo "//registry.npmjs.org/:_authToken=${NPM_TOKEN}" > .npmrc

# ❌ WRONG - Secret copied into image
COPY .npmrc /root/.npmrc
```

#### 9. Layer Hygiene

```dockerfile
# ✅ CORRECT - Clean up in same layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ✅ CORRECT - Alpine cleanup
RUN apk add --no-cache curl ca-certificates && \
    rm -rf /var/cache/apk/*

# ❌ WRONG - Cleanup in separate layer (doesn't reduce size)
RUN apt-get update && apt-get install -y curl
RUN apt-get clean  # This doesn't help!
```

### E. Housekeeping
* .dockerignore: The agent must always suggest or generate a .dockerignore file excluding .git, node_modules, venv, secrets, and test files.

* Labels: Add OCI-compliant labels (org.opencontainers.image.source, authors, etc.).

## 4. "Gold Standard" Examples
The agent should use the following patterns as the baseline for generation.

Example: Node.js (Production)
```Dockerfile
# syntax=docker/dockerfile:1.6

# 1. Base Stage
FROM node:20-alpine AS base
WORKDIR /app
ENV NODE_ENV=production

# 2. Dependencies Stage
FROM base AS deps
# Use cache mount to speed up install
RUN --mount=type=bind,source=package.json,target=package.json \
    --mount=type=bind,source=package-lock.json,target=package-lock.json \
    --mount=type=cache,target=/root/.npm \
    npm ci --omit=dev

# 3. Final Runtime Stage
FROM base AS final

# Run as non-root user
USER node

# Copy with --link for better caching and parallel builds
COPY --link --from=deps --chown=node:node /app/node_modules ./node_modules
COPY --link --chown=node:node . .

# Use array syntax for signal handling
EXPOSE 3000
CMD ["node", "src/index.js"]
```

Example: Go (FROM scratch - PREFERRED)
```Dockerfile
# syntax=docker/dockerfile:1.6

# 1. Build Stage
FROM golang:1.22-alpine AS builder
WORKDIR /app

# Add CA certs and create non-root user
RUN apk add --no-cache ca-certificates && \
    adduser -D -g '' -u 10001 appuser

# Cache Go modules
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download

# Build static binary
COPY . .
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s" -o /bin/server ./cmd/server

# 2. Final Stage (FROM scratch - smallest possible)
FROM scratch

# Copy CA certs for HTTPS
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /etc/passwd /etc/passwd

# Copy binary with --link
COPY --link --from=builder /bin/server /server

USER 10001:10001
ENTRYPOINT ["/server"]
```

Example: Python (Optimization)
```Dockerfile
# syntax=docker/dockerfile:1.6
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

# Create virtual environment
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install deps with cache
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Final Stage
FROM python:3.12-slim-bookworm

# Create non-root user
RUN groupadd -g 999 python && \
    useradd -r -u 999 -g python python && \
    # Remove SUID/SGID binaries for security
    find / -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true

USER 999

# Copy with --link for better caching
COPY --link --from=builder --chown=python:python /app/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

COPY --link --chown=python:python . /app
WORKDIR /app

CMD ["python", "main.py"]
```

## 4A. Ultra-Minimal Images (FROM scratch) - PREFERRED

**When possible, use FROM scratch for the smallest, most secure images.**

### When to Use FROM scratch

| Language | FROM scratch | Why |
|----------|-------------|-----|
| Go | ✅ Yes | Static binary with CGO_ENABLED=0 |
| Rust | ✅ Yes | Static binary with musl |
| C/C++ | ✅ Yes | Static linking possible |
| Java | ⚠️ Distroless | Needs JVM runtime |
| Python | ⚠️ Distroless | Needs interpreter |
| Node.js | ⚠️ Distroless | Needs runtime |
| .NET | ⚠️ Distroless | Needs runtime (or self-contained) |

### Example: Go FROM scratch (PREFERRED)

```dockerfile
# syntax=docker/dockerfile:1.6

# Build stage
FROM golang:1.22-alpine AS builder
WORKDIR /app

# Install CA certificates and create user
RUN apk add --no-cache ca-certificates tzdata && \
    adduser -D -g '' -u 10001 appuser

# Cache dependencies
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download

# Build static binary
COPY . .
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s -extldflags '-static'" \
    -o /app/server ./cmd/server

# Final stage - FROM SCRATCH (smallest possible)
FROM scratch

# Import CA certificates for HTTPS
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Import timezone data
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# Import passwd for user
COPY --from=builder /etc/passwd /etc/passwd

# Copy binary with --link for better caching
COPY --link --from=builder /app/server /server

# Run as non-root (user created in builder)
USER 10001:10001

# Use exec form - no shell available!
ENTRYPOINT ["/server"]
```

**Result: ~10-15MB image vs ~300MB with golang:alpine**

### Example: Rust FROM scratch

```dockerfile
# syntax=docker/dockerfile:1.6

FROM rust:1.75-alpine AS builder
WORKDIR /app

# Install musl for static linking
RUN apk add --no-cache musl-dev ca-certificates

# Create non-root user
RUN adduser -D -g '' -u 10001 appuser

# Cache dependencies with cargo-chef
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release --target x86_64-unknown-linux-musl && \
    rm -rf src

# Build actual binary
COPY src ./src
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release --target x86_64-unknown-linux-musl && \
    cp target/x86_64-unknown-linux-musl/release/myapp /app/myapp

# FROM scratch final
FROM scratch

COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /etc/passwd /etc/passwd
COPY --link --from=builder /app/myapp /myapp

USER 10001:10001
ENTRYPOINT ["/myapp"]
```

### Example: .NET Self-Contained FROM scratch

```dockerfile
# syntax=docker/dockerfile:1.6

FROM mcr.microsoft.com/dotnet/sdk:8.0-alpine AS builder
WORKDIR /app

# Restore dependencies
COPY *.csproj ./
RUN dotnet restore

# Build self-contained binary
COPY . .
RUN dotnet publish -c Release -r linux-musl-x64 \
    --self-contained true \
    -p:PublishSingleFile=true \
    -p:PublishTrimmed=true \
    -p:TrimMode=full \
    -o /app/publish

# FROM scratch
FROM scratch

# Copy only the self-contained binary
COPY --link --from=builder /app/publish/myapp /myapp

# Copy CA certs if needed for HTTPS
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

USER 65534:65534
ENTRYPOINT ["/myapp"]
```

### Distroless for Interpreted Languages

When FROM scratch isn't possible, use Google Distroless:

```dockerfile
# Python with Distroless
# syntax=docker/dockerfile:1.6

FROM python:3.12-slim AS builder
WORKDIR /app

RUN pip install --target=/app/deps --no-cache-dir -r requirements.txt

FROM gcr.io/distroless/python3-debian12

COPY --link --from=builder /app/deps /app/deps
COPY --link . /app

WORKDIR /app
ENV PYTHONPATH=/app/deps

USER nonroot:nonroot
CMD ["main.py"]
```

```dockerfile
# Node.js with Distroless
# syntax=docker/dockerfile:1.6

FROM node:20-alpine AS builder
WORKDIR /app

COPY package*.json ./
RUN npm ci --omit=dev

FROM gcr.io/distroless/nodejs20-debian12

COPY --link --from=builder /app/node_modules /app/node_modules
COPY --link . /app

WORKDIR /app
USER nonroot:nonroot
CMD ["app/index.js"]
```

```dockerfile
# Java with Distroless
# syntax=docker/dockerfile:1.6

FROM eclipse-temurin:21-jdk-alpine AS builder
WORKDIR /app

COPY . .
RUN ./gradlew build -x test

FROM gcr.io/distroless/java21-debian12

COPY --link --from=builder /app/build/libs/app.jar /app.jar

USER nonroot:nonroot
ENTRYPOINT ["java", "-jar", "/app.jar"]
```

### Image Size Comparison

| Language | Base Image | Distroless | FROM scratch |
|----------|-----------|------------|--------------|
| Go | 300MB (golang:alpine) | 20MB | **5-15MB** |
| Rust | 400MB (rust:alpine) | 25MB | **5-15MB** |
| Node.js | 180MB (node:alpine) | **70MB** | N/A |
| Python | 150MB (python:slim) | **50MB** | N/A |
| Java | 350MB (eclipse-temurin) | **200MB** | N/A |
| .NET | 200MB (dotnet:alpine) | 100MB | **30-50MB** |

### FROM scratch Checklist

- [ ] Binary is statically linked (CGO_ENABLED=0 for Go, musl for Rust)
- [ ] CA certificates copied if HTTPS needed
- [ ] Timezone data copied if time parsing needed
- [ ] /etc/passwd copied for non-root user
- [ ] No shell commands in ENTRYPOINT (exec form only)
- [ ] Health check uses binary or HTTP probe (no shell)

## 5. Interaction Protocol
User Input: "Create a Dockerfile for a React app using Vite."

Agent Response Strategy:

* Analyze Context: React = Static assets. Needs build node, serve Nginx.

* Select Pattern: Multi-stage (Node Build -> Nginx Alpine Run).

* Draft Code: Apply syntax=docker/dockerfile:1.6, .dockerignore, and Nginx security hardening.

* Optimize Caching: Order layers from least to most frequently changing (base image → dependencies → source code).

* Verify Build: Run `docker build` to ensure it completes successfully.

* Test Caching: Rebuild to verify cache layers work correctly.

* Check Security: Verify non-root user with `docker run --rm <image> id`.

* Review: Check against "Triple-S" (Small, Secure, Speedy) + Verified.

* Output: Return code block + brief explanation of specific flags used.

## 6. Deployment Checklist

### Agent-Generated Dockerfile Verification (MANDATORY)
**If Dockerfile was generated/modified by an agent, verify BEFORE delivery:**

#### Build Verification
- [ ] Docker build succeeds: `docker build -t test:latest .` returns exit code 0
- [ ] All multi-stage build stages complete successfully
- [ ] Second build uses cached layers appropriately
- [ ] Container starts successfully: `docker run --rm test:latest` works

#### Caching Verification
- [ ] Cache layers optimized: Dependencies cached separately from source code
- [ ] Cache mounts configured for all package managers (npm, pip, go, cargo)
- [ ] COPY --link used for independent layer caching
- [ ] Layer ordering: system packages → dependencies → source code

#### Size Verification
- [ ] Image size is reasonable (< 500MB for most applications)
- [ ] Layer count is minimal (combined RUN commands where appropriate)
- [ ] Consider FROM scratch for Go/Rust (< 20MB target)
- [ ] Consider Distroless for interpreted languages

#### Security Verification (MANDATORY)
- [ ] Non-root user configured: `docker run --rm test:latest id` shows uid != 0
- [ ] No SUID/SGID binaries: `docker run --rm test:latest find / -perm /6000 2>/dev/null`
- [ ] No world-writable files
- [ ] No secrets in ARG or ENV variables
- [ ] No sensitive files in image: `docker history test:latest`
- [ ] Secret mounts used for build-time credentials
- [ ] ENTRYPOINT uses exec form (no shell injection)
- [ ] No shell in final image (for scratch/distroless)
- [ ] Prepared for read-only filesystem runtime

#### Documentation
- [ ] .dockerignore file provided and comprehensive
- [ ] Base images use specific version tags (not :latest)
- [ ] BuildKit syntax enabled: `# syntax=docker/dockerfile:1.6`
- [ ] OCI-compliant labels added
- [ ] Agent has documented any build fixes made during verification

### Security Scan (MANDATORY before production)
```bash
# Scan with Trivy
trivy image --severity HIGH,CRITICAL test:latest

# Verify non-root
docker run --rm test:latest id
# Expected: uid=1001(appuser) or similar, NOT uid=0(root)

# Test read-only filesystem compatibility
docker run --rm --read-only --tmpfs /tmp test:latest
```

### General Best Practices
- [ ] Multi-stage build used appropriately
- [ ] Health checks configured for services
- [ ] Signal handling works correctly (no shell wrapping)
- [ ] Minimal base image selected (scratch > distroless > alpine > slim)

## 7. Why This Configuration Works

* **syntax=docker/dockerfile:1.6**: This directive tells the Docker builder to use the latest frontend, unlocking features like Heredocs and improved cache mounting even if the user has an older Docker daemon (as long as BuildKit is enabled).

* **Agent Build Verification**: Ensures generated Dockerfiles actually build before delivery, preventing broken builds and reducing debugging time for users.

* **Optimal Cache Layer Ordering**: By ordering layers from least to most frequently changing (system packages → dependencies → source code), builds leverage Docker's layer cache maximally. This can reduce rebuild times from minutes to seconds.

* **BuildKit Cache Mounts**: This is the single biggest "modern" improvement. It prevents re-downloading dependencies if only the source code changed, but package files didn't. Cache mounts persist across builds, providing 5-10x speedup for dependency installation.

* **Multi-Stage Builds**: Separates build-time dependencies from runtime, resulting in images that are 50-90% smaller. A Node.js app with build tools (1.2GB) becomes a production image with just the runtime (150MB).

* **Security First**: By baking USER nonroot into the agent's instructions, you prevent the most common security vulnerability (running as root) by default. Non-root execution is a requirement for most enterprise Kubernetes environments.

* **Verified Caching**: Testing cache behavior during verification ensures that small code changes don't trigger full dependency reinstalls, maintaining developer productivity.

---

## 8. Quick Reference

### Build Commands
```bash
# Standard build with BuildKit
DOCKER_BUILDKIT=1 docker build -t myapp:latest .

# Build with build arguments
docker build --build-arg NODE_VERSION=20 -t myapp:latest .

# Build specific stage
docker build --target builder -t myapp:builder .

# Build with cache from registry
docker build --cache-from myapp:latest -t myapp:latest .

# Show build output
docker build --progress=plain -t myapp:latest .
```

### Verification Commands
```bash
# Check image size
docker images myapp:latest

# Inspect layers
docker history myapp:latest

# Verify user
docker run --rm myapp:latest id

# Test application
docker run --rm -p 8080:8080 myapp:latest

# Check for vulnerabilities (if using Docker Scout)
docker scout cve myapp:latest
```

### Cache Management
```bash
# Clear build cache
docker builder prune

# Clear specific cache mount
docker builder prune --filter type=exec.cachemount

# Show cache usage
docker system df
```

---

## 9. Common Patterns

### Node.js with pnpm
```dockerfile
FROM node:20-alpine AS base
RUN npm install -g pnpm
WORKDIR /app

FROM base AS deps
COPY package.json pnpm-lock.yaml ./
RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile --prod

FROM base AS runner
USER node
COPY --from=deps --chown=node:node /app/node_modules ./node_modules
COPY --chown=node:node . .
CMD ["node", "index.js"]
```

### Python with Poetry
```dockerfile
FROM python:3.11-slim AS builder
RUN pip install poetry
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --no-dev --no-root

FROM python:3.11-slim
RUN useradd -m -u 1000 python
USER 1000
WORKDIR /app
COPY --from=builder --chown=python:python /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY --chown=python:python . .
CMD ["python", "main.py"]
```

### Rust
```dockerfile
FROM rust:1.75-alpine AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release

FROM alpine:3.19
RUN adduser -D -u 1000 app
USER 1000
COPY --from=builder /app/target/release/myapp /app/myapp
ENTRYPOINT ["/app/myapp"]
```


**End of Dockerfile Guidelines**
