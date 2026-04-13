# Modern GitLab Workflow Guidelines
Mandatory standards and best practices for GitLab usage, including CI/CD pipelines, security, automation, issue tracking, and container management. GitLab CI/CD, GitLab Security, GitLab Container Registry, GitLab Pages, GitLab Runners.

---

**Agent Profile**: The GitLab DevOps Expert  
**Role**: Senior DevOps Engineer & GitLab Specialist  
**Objective**: Generate efficient, secure, automated GitLab pipelines with comprehensive CI/CD and best practices.  
**Tools**: GitLab CI/CD, GitLab Security, GitLab Container Registry, GitLab Pages, GitLab Runners.

## Core Philosophies

The agent must adhere to the "GITLAB-FIRST" principles for every GitLab workflow:

**Test-Driven Development (TDD)**: ALL pipelines MUST verify tests pass before merge (Red-Green-Refactor mandatory).
**Regression Shield**: EVERY bug fix MUST reference issue ID and include regression test verification in CI.
**Security First**: SAST, DAST, dependency scanning, container scanning, secrets detection enabled.
**Everything as Code**: Pipelines, configurations, infrastructure defined in .gitlab-ci.yml.
**Automated CI/CD**: Every push triggers tests, every merge to main triggers deployment.
**Branch Protection**: Main/develop protected, require approvals, status checks, merge trains.
**Issue Tracking**: Every commit/MR links to issue, clear templates, labels, milestones.

**Container Ready**: GitLab Container Registry for images, automated builds, vulnerability scanning.
**Documentation**: README, CONTRIBUTING, documentation in GitLab Pages, comprehensive wikis.
**Reproducible Builds**: Dependency locking, deterministic pipelines, versioned images.
**Observability**: Pipeline metrics, deployment tracking, security dashboards.
**Efficiency**: Pipeline caching, parallelization, DAG pipelines, selective job execution.
**Automation**: Auto DevOps, scheduled pipelines, release automation, merge request automation.

---

## 1. GitLab CI/CD Pipelines (MANDATORY)

### A. Comprehensive CI/CD Pipeline Structure

```yaml
# .gitlab-ci.yml - Comprehensive CI/CD pipeline

# Define stages
stages:
  - validate
  - test
  - build
  - security
  - deploy
  - verify

# Global variables
variables:
  NODE_VERSION: "20"
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"
  FF_USE_FASTZIP: "true"
  ARTIFACT_COMPRESSION_LEVEL: "fast"
  CACHE_COMPRESSION_LEVEL: "fast"
  # Disable shallow clone for proper analysis
  GIT_DEPTH: "0"

# Default settings
default:
  image: node:${NODE_VERSION}-alpine
  cache:
    key:
      files:
        - package-lock.json
    paths:
      - node_modules/
      - .npm/
  retry:
    max: 2
    when:
      - runner_system_failure
      - stuck_or_timeout_failure

# Workflow rules
workflow:
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
    - if: $CI_COMMIT_TAG
    - if: $CI_PIPELINE_SOURCE == "web"

# ============================================
# VALIDATE STAGE
# ============================================

# Lint code
lint:code:
  stage: validate
  script:
    - npm ci --cache .npm --prefer-offline
    - npm run lint
    - npm run format:check
  artifacts:
    reports:
      codequality: gl-code-quality-report.json
    when: always

# Lint commit messages
lint:commits:
  stage: validate
  image: node:${NODE_VERSION}-alpine
  script:
    - npm install -g @commitlint/cli @commitlint/config-conventional
    - echo "module.exports = {extends: ['@commitlint/config-conventional']}" > commitlint.config.js
    - commitlint --from ${CI_MERGE_REQUEST_DIFF_BASE_SHA} --to HEAD --verbose
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"

# Verify issue link in MR
verify:issue-link:
  stage: validate
  image: alpine:latest
  before_script:
    - apk add --no-cache curl jq
  script:
    - |
      # Check MR title and description for issue references
      if ! echo "$CI_MERGE_REQUEST_TITLE $CI_MERGE_REQUEST_DESCRIPTION" | grep -qE '#[0-9]+|Closes #[0-9]+|Fixes #[0-9]+'; then
        echo "ERROR: Merge request must reference an issue (#123, Closes #123, or Fixes #123)"
        exit 1
      fi
      echo "✓ Issue reference found"
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"

# ============================================
# TEST STAGE (TDD VERIFICATION)
# ============================================

# Unit tests with TDD verification
test:unit:
  stage: test
  coverage: '/All files[^|]*\|[^|]*\s+([\d\.]+)/'
  script:
    - npm ci --cache .npm --prefer-offline
    # Verify tests exist
    - |
      TEST_COUNT=$(find tests/ -name "*.test.ts" -o -name "*.spec.ts" | wc -l)
      if [ "$TEST_COUNT" -eq 0 ]; then
        echo "ERROR: No test files found - TDD violation"
        exit 1
      fi
      echo "✓ Found $TEST_COUNT test files"
    # Run tests
    - npm test -- --coverage --ci --maxWorkers=2
    # Check coverage threshold
    - |
      COVERAGE=$(cat coverage/coverage-summary.json | jq '.total.lines.pct')
      echo "Coverage: $COVERAGE%"
      if [ $(echo "$COVERAGE < 80" | bc -l) -eq 1 ]; then
        echo "ERROR: Coverage $COVERAGE% is below 80% threshold"
        exit 1
      fi
      echo "✓ Coverage threshold met"
  artifacts:
    when: always
    reports:
      junit: coverage/junit.xml
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml
    paths:
      - coverage/
    expire_in: 30 days
  parallel:
    matrix:
      - NODE_VERSION: ["18", "20"]

# Integration tests
test:integration:
  stage: test
  services:
    - postgres:15-alpine
    - redis:7-alpine
  variables:
    POSTGRES_DB: test_db
    POSTGRES_USER: test_user
    POSTGRES_PASSWORD: test_pass
    DATABASE_URL: "postgresql://test_user:test_pass@postgres:5432/test_db"
    REDIS_URL: "redis://redis:6379"
  script:
    - npm ci --cache .npm --prefer-offline
    - npm run test:integration
  artifacts:
    when: always
    reports:
      junit: test-results/integration-junit.xml
    expire_in: 7 days

# Verify regression tests for bug fixes
verify:bug-fix-tests:
  stage: test
  image: alpine:latest
  before_script:
    - apk add --no-cache git grep
  script:
    - |
      # Check if MR title indicates bug fix
      if echo "$CI_MERGE_REQUEST_TITLE" | grep -qiE "fix|bug"; then
        echo "Bug fix detected - verifying regression test..."
        
        # Extract issue number from title
        ISSUE_NUM=$(echo "$CI_MERGE_REQUEST_TITLE" | grep -oE '#[0-9]+' | head -1 | sed 's/#//')
        
        if [ -z "$ISSUE_NUM" ]; then
          echo "ERROR: Bug fix MR must reference issue number (#123)"
          exit 1
        fi
        
        # Check if tests reference the issue
        if ! grep -r "issue.*#$ISSUE_NUM\|bug.*#$ISSUE_NUM\|Bug #$ISSUE_NUM" tests/; then
          echo "ERROR: Bug fix for issue #$ISSUE_NUM missing regression test"
          echo "ERROR: Add a test with comment: // Bug #$ISSUE_NUM"
          exit 1
        fi
        
        echo "✓ Regression test found for issue #$ISSUE_NUM"
      else
        echo "Not a bug fix - skipping regression test verification"
      fi
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"

# ============================================
# BUILD STAGE
# ============================================

build:application:
  stage: build
  script:
    - npm ci --cache .npm --prefer-offline
    - npm run build
    - |
      BUILD_SIZE=$(du -sh dist/ | cut -f1)
      echo "Build size: $BUILD_SIZE"
      echo "build_size=$BUILD_SIZE" >> build.env
  artifacts:
    paths:
      - dist/
    reports:
      dotenv: build.env
    expire_in: 1 week

# Build Docker image
build:docker:
  stage: build
  image: docker:24-cli
  services:
    - docker:24-dind
  variables:
    DOCKER_BUILDKIT: 1
  before_script:
    - echo "$CI_REGISTRY_PASSWORD" | docker login -u "$CI_REGISTRY_USER" --password-stdin "$CI_REGISTRY"
  script:
    # Build image with cache
    - |
      docker build \
        --cache-from $CI_REGISTRY_IMAGE:latest \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
        --build-arg VCS_REF=$CI_COMMIT_SHORT_SHA \
        --build-arg VERSION=$CI_COMMIT_TAG \
        --tag $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA \
        --tag $CI_REGISTRY_IMAGE:$CI_COMMIT_REF_SLUG \
        .
    # Push images
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_REF_SLUG
    # Tag and push latest for default branch
    - |
      if [ "$CI_COMMIT_BRANCH" == "$CI_DEFAULT_BRANCH" ]; then
        docker tag $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA $CI_REGISTRY_IMAGE:latest
        docker push $CI_REGISTRY_IMAGE:latest
      fi
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
    - if: $CI_COMMIT_TAG
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"

# ============================================
# SECURITY STAGE
# ============================================

# SAST (Static Application Security Testing)
sast:
  stage: security
  needs: []
  # GitLab SAST template
include:
  - template: Security/SAST.gitlab-ci.yml

# Dependency scanning
dependency_scanning:
  stage: security
  needs: []
include:
  - template: Security/Dependency-Scanning.gitlab-ci.yml

# Secret detection
secret_detection:
  stage: security
  needs: []
include:
  - template: Security/Secret-Detection.gitlab-ci.yml

# Container scanning
container_scanning:
  stage: security
  needs: ["build:docker"]
  variables:
    CS_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA
include:
  - template: Security/Container-Scanning.gitlab-ci.yml

# License scanning
license_scanning:
  stage: security
  needs: []
include:
  - template: Security/License-Scanning.gitlab-ci.yml

# Custom security checks
security:custom:
  stage: security
  image: alpine:latest
  before_script:
    - apk add --no-cache git grep
  script:
    # Check for .env files in repository
    - |
      if find . -name ".env*" -not -name ".env.example" -not -path "*/node_modules/*" | grep -q .; then
        echo "ERROR: Found .env files in repository"
        find . -name ".env*" -not -name ".env.example" -not -path "*/node_modules/*"
        exit 1
      fi
      echo "✓ No .env files found"
    # Verify .gitignore has required patterns
    - |
      REQUIRED_IGNORES=".env .env.local *.pem *.key secrets.yml"
      for pattern in $REQUIRED_IGNORES; do
        if ! grep -q "^$pattern$" .gitignore; then
          echo "WARNING: .gitignore missing pattern: $pattern"
        fi
      done
      echo "✓ .gitignore patterns verified"
  allow_failure: true

# ============================================
# DEPLOY STAGE
# ============================================

.deploy_template: &deploy_template
  image: alpine:latest
  before_script:
    - apk add --no-cache curl
  script:
    - |
      echo "Deploying to $ENVIRONMENT"
      echo "Image: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA"
      # Add deployment commands here (kubectl, helm, etc.)
      
      # Example: kubectl deployment
      # kubectl set image deployment/myapp myapp=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA
      
      # Wait for deployment
      sleep 10
      
      # Health check
      if ! curl -f "$HEALTH_CHECK_URL"; then
        echo "ERROR: Health check failed"
        exit 1
      fi
      echo "✓ Deployment successful"

deploy:dev:
  <<: *deploy_template
  stage: deploy
  needs: ["build:docker", "test:unit", "test:integration"]
  variables:
    ENVIRONMENT: "development"
    HEALTH_CHECK_URL: "https://dev.example.com/health"
  environment:
    name: development
    url: https://dev.example.com
    auto_stop_in: 1 day
  rules:
    - if: $CI_COMMIT_BRANCH == "develop"

deploy:staging:
  <<: *deploy_template
  stage: deploy
  needs: ["build:docker", "test:unit", "test:integration", "sast", "dependency_scanning"]
  variables:
    ENVIRONMENT: "staging"
    HEALTH_CHECK_URL: "https://staging.example.com/health"
  environment:
    name: staging
    url: https://staging.example.com
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

deploy:production:
  <<: *deploy_template
  stage: deploy
  needs: 
    - job: build:docker
    - job: test:unit
    - job: test:integration
    - job: sast
    - job: dependency_scanning
    - job: container_scanning
  variables:
    ENVIRONMENT: "production"
    HEALTH_CHECK_URL: "https://example.com/health"
  environment:
    name: production
    url: https://example.com
    action: start
  rules:
    - if: $CI_COMMIT_TAG =~ /^v\d+\.\d+\.\d+$/
      when: manual
  # Deployment protection
  resource_group: production

# ============================================
# VERIFY STAGE
# ============================================

verify:production:
  stage: verify
  image: alpine:latest
  needs: ["deploy:production"]
  before_script:
    - apk add --no-cache curl
  script:
    - |
      echo "Running production smoke tests..."
      
      # Health check
      if ! curl -f "https://example.com/health"; then
        echo "ERROR: Production health check failed"
        exit 1
      fi
      
      # API endpoint check
      if ! curl -f "https://example.com/api/v1/status"; then
        echo "ERROR: API status check failed"
        exit 1
      fi
      
      echo "✓ Production verification passed"
  rules:
    - if: $CI_COMMIT_TAG =~ /^v\d+\.\d+\.\d+$/
      when: on_success

# ============================================
# SCHEDULED JOBS
# ============================================

# Nightly security scan
security:nightly:
  stage: security
  needs: []
  script:
    - echo "Running comprehensive security scan..."
  rules:
    - if: $CI_PIPELINE_SOURCE == "schedule"
```

### B. Advanced Pipeline with DAG (Directed Acyclic Graph)

```yaml
# .gitlab-ci.yml - DAG pipeline for parallel execution

stages:
  - validate
  - test
  - build
  - deploy

# Use needs to create DAG and parallelize jobs

lint:
  stage: validate
  script:
    - npm ci
    - npm run lint

test:unit:
  stage: test
  needs: []  # Can run without waiting for lint
  script:
    - npm ci
    - npm test

test:integration:
  stage: test
  needs: []  # Can run in parallel with unit tests
  script:
    - npm ci
    - npm run test:integration

test:e2e:
  stage: test
  needs: ["build:app"]  # Depends on build
  script:
    - npm ci
    - npm run test:e2e

build:app:
  stage: build
  needs: []  # Can start immediately
  script:
    - npm ci
    - npm run build

build:docker:
  stage: build
  needs: ["build:app"]  # Depends on app build
  script:
    - docker build -t myapp .

deploy:staging:
  stage: deploy
  needs:
    - job: build:docker
    - job: test:unit
    - job: test:integration
    # Doesn't wait for e2e tests
  script:
    - kubectl apply -f k8s/
```

### C. Reusable Pipeline Components

```yaml
# .gitlab/ci/templates/test.yml - Reusable test template

.test_template:
  image: node:20-alpine
  cache:
    key:
      files:
        - package-lock.json
    paths:
      - node_modules/
  before_script:
    - npm ci --cache .npm --prefer-offline
  retry:
    max: 2
    when:
      - runner_system_failure

# Include in main .gitlab-ci.yml:
# include:
#   - local: '.gitlab/ci/templates/test.yml'
```

```yaml
# .gitlab/ci/templates/docker.yml - Reusable Docker template

.docker_build:
  image: docker:24-cli
  services:
    - docker:24-dind
  variables:
    DOCKER_BUILDKIT: 1
  before_script:
    - echo "$CI_REGISTRY_PASSWORD" | docker login -u "$CI_REGISTRY_USER" --password-stdin "$CI_REGISTRY"
  script:
    - |
      docker build \
        --cache-from $CI_REGISTRY_IMAGE:latest \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --tag $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA \
        .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA
```

### D. Multi-Project Pipeline (Parent-Child)

```yaml
# .gitlab-ci.yml - Parent pipeline

stages:
  - trigger

trigger:frontend:
  stage: trigger
  trigger:
    project: my-group/frontend
    branch: main
    strategy: depend

trigger:backend:
  stage: trigger
  trigger:
    project: my-group/backend
    branch: main
    strategy: depend

trigger:infra:
  stage: trigger
  trigger:
    include: .gitlab/ci/infrastructure.yml
    strategy: depend
```

---

## 2. GitLab Security (MANDATORY)

### A. Complete Security Scanning Configuration

```yaml
# .gitlab-ci.yml - Comprehensive security

include:
  # SAST (Static Application Security Testing)
  - template: Security/SAST.gitlab-ci.yml
  # Dependency Scanning
  - template: Security/Dependency-Scanning.gitlab-ci.yml
  # Secret Detection
  - template: Security/Secret-Detection.gitlab-ci.yml
  # Container Scanning
  - template: Security/Container-Scanning.gitlab-ci.yml
  # License Scanning
  - template: Security/License-Scanning.gitlab-ci.yml
  # DAST (Dynamic Application Security Testing)
  - template: Security/DAST.gitlab-ci.yml
  # API Fuzzing
  - template: Security/API-Fuzzing.gitlab-ci.yml
  # Coverage-Guided Fuzz Testing
  - template: Security/Coverage-Fuzzing.gitlab-ci.yml

# Configure SAST
sast:
  variables:
    SAST_EXCLUDED_PATHS: "spec, test, tests, tmp, node_modules"
    SAST_EXCLUDED_ANALYZERS: "eslint"  # Use dedicated linter

# Configure DAST
dast:
  variables:
    DAST_WEBSITE: https://staging.example.com
    DAST_FULL_SCAN_ENABLED: "true"
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

# Configure Container Scanning
container_scanning:
  variables:
    CS_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA
    CS_SEVERITY_THRESHOLD: "medium"
    CS_DISABLE_LANGUAGE_VULNERABILITY_SCAN: "false"

# Configure Dependency Scanning
dependency_scanning:
  variables:
    DS_EXCLUDED_PATHS: "spec, test, tests, tmp"
    DS_DEFAULT_ANALYZERS: "gemnasium, retire.js"

# Custom security policy
security:policy:
  stage: security
  script:
    - |
      # Fail if critical vulnerabilities found
      if [ -f gl-dependency-scanning-report.json ]; then
        CRITICAL=$(cat gl-dependency-scanning-report.json | jq '[.vulnerabilities[] | select(.severity == "Critical")] | length')
        if [ "$CRITICAL" -gt 0 ]; then
          echo "ERROR: Found $CRITICAL critical vulnerabilities"
          exit 1
        fi
      fi
      echo "✓ No critical vulnerabilities"
  artifacts:
    when: always
    expire_in: 1 week
```

### B. Security Policy File

```yaml
# .gitlab/security-policies/policy.yml

---
type: scan_execution_policy
name: Comprehensive Security Scanning
description: Enforce security scanning on all merge requests and scheduled
enabled: true
rules:
  - type: pipeline
    branches:
      - main
      - develop
      - release/*
actions:
  - scan: sast
  - scan: secret_detection
  - scan: dependency_scanning
  - scan: container_scanning
  - scan: dast
    scanner_profile: Full Scan
    site_profile: staging

approval_settings:
  block_branch_modification: true
  prevent_pushing_and_force_pushing: true
  prevent_approval_by_author: true
  prevent_approval_by_commit_author: true
  require_password_to_approve: true
```

### C. Security Policy as Code

```markdown
# SECURITY.md

# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x     | :white_check_mark: |
| 1.x     | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**DO NOT** open a public issue for security vulnerabilities.

### GitLab Confidential Issues

1. Navigate to [Issues](../../issues)
2. Click "New Issue"
3. Check "This issue is confidential"
4. Use the "Security Vulnerability" template
5. Add label: `security`

### Email

Alternatively, email: security@example.com

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if available)
- CVE ID (if applicable)

## Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: 
  - Critical: 24-48 hours
  - High: 7 days
  - Medium: 30 days
  - Low: 90 days

## Security Scanning

Our CI/CD pipeline includes:

- ✅ SAST (Static Application Security Testing)
- ✅ DAST (Dynamic Application Security Testing)
- ✅ Dependency Scanning
- ✅ Container Scanning
- ✅ Secret Detection
- ✅ License Compliance
- ✅ API Fuzzing
- ✅ Coverage-Guided Fuzzing

## Security Best Practices

### For Contributors

1. Never commit secrets, API keys, or credentials
2. Use environment variables for sensitive data
3. Run security scans locally before pushing
4. Keep dependencies up to date
5. Follow secure coding practices
6. Enable 2FA on your GitLab account
7. Sign commits with GPG

### For Maintainers

1. Review security dashboard daily
2. Enable all security scanning templates
3. Configure security policies
4. Require security approvals for MRs
5. Keep pipeline runners updated
6. Regularly audit dependencies
7. Monitor security advisories

## Vulnerability Disclosure

After a vulnerability is fixed:

1. Security advisory published
2. CVE assigned (if applicable)
3. Credit given to reporter (unless anonymous)
4. Fix documented in changelog
5. Patch version released
6. Customers notified (if applicable)

## Compliance

This project complies with:
- OWASP Top 10
- CWE Top 25
- ISO 27001
- SOC 2

## Contact

- Security Team: security@example.com
- Security Issues: Use confidential issues
- Bug Bounty: https://example.com/security/bounty
```

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new pipeline configurations and CI/CD code.**

### TDD Cycle

```
┌─────────────────────────────────────────────────────────────┐
│                    TDD CYCLE FOR GITLAB CI/CD               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    ┌─────────┐                                              │
│    │  RED    │  1. Write a failing test/job first           │
│    │  (FAIL) │     - Define expected pipeline behavior      │
│    └────┬────┘     - Create test that validates behavior    │
│         │          - Run pipeline: MUST FAIL                │
│         ▼                                                   │
│    ┌─────────┐                                              │
│    │  GREEN  │  2. Write minimal code to pass               │
│    │  (PASS) │     - Implement pipeline job/stage           │
│    └────┬────┘     - Run pipeline: MUST PASS                │
│         │                                                   │
│         ▼                                                   │
│    ┌─────────┐                                              │
│    │REFACTOR │  3. Improve while keeping tests green        │
│    │(IMPROVE)│     - Optimize pipeline performance          │
│    └────┬────┘     - Add caching, parallelization           │
│         │          - Run pipeline: STILL PASSES             │
│         │                                                   │
│         └──────────────► Repeat for next feature            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Example TDD Workflow for GitLab CI/CD

**Scenario**: Adding a new deployment validation job to ensure deployments are healthy.

```yaml
# Step 1: RED - Write failing test job first
# .gitlab-ci.yml

# Define the test that validates deployment behavior
test:deployment-validation:
  stage: test
  image: alpine:latest
  script:
    - |
      echo "Testing deployment validation logic..."

      # Test 1: Health check function should return success for healthy endpoint
      # This test will FAIL because deploy:validate job doesn't exist yet
      if ! grep -q "deploy:validate" .gitlab-ci.yml; then
        echo "FAIL: deploy:validate job not found"
        exit 1
      fi

      # Test 2: Verify health check timeout is configured
      if ! grep -q "HEALTH_CHECK_TIMEOUT" .gitlab-ci.yml; then
        echo "FAIL: HEALTH_CHECK_TIMEOUT not configured"
        exit 1
      fi

      # Test 3: Verify rollback mechanism exists
      if ! grep -q "rollback" .gitlab-ci.yml; then
        echo "FAIL: Rollback mechanism not defined"
        exit 1
      fi

      echo "All deployment validation tests passed"
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
      changes:
        - .gitlab-ci.yml

# Run: git push → Pipeline runs → test:deployment-validation FAILS
# ✗ FAIL: deploy:validate job not found
```

```yaml
# Step 2: GREEN - Write minimal implementation to pass

variables:
  HEALTH_CHECK_TIMEOUT: "30"
  HEALTH_CHECK_RETRIES: "5"

deploy:validate:
  stage: verify
  image: alpine:latest
  before_script:
    - apk add --no-cache curl
  script:
    - |
      echo "Validating deployment health..."

      # Health check with timeout and retries
      RETRY=0
      while [ $RETRY -lt $HEALTH_CHECK_RETRIES ]; do
        if curl -sf --max-time $HEALTH_CHECK_TIMEOUT "$DEPLOYMENT_URL/health"; then
          echo "✓ Health check passed"
          exit 0
        fi
        RETRY=$((RETRY + 1))
        echo "Health check attempt $RETRY failed, retrying..."
        sleep 5
      done

      echo "✗ Health check failed after $HEALTH_CHECK_RETRIES attempts"
      exit 1
  environment:
    name: $ENVIRONMENT
    action: verify

# Rollback job for failed deployments
rollback:deployment:
  stage: verify
  image: alpine:latest
  script:
    - |
      echo "Rolling back deployment..."
      # Rollback to previous version
      # kubectl rollout undo deployment/myapp
      echo "✓ Rollback completed"
  when: on_failure
  needs: ["deploy:validate"]

# Run: git push → Pipeline runs → test:deployment-validation PASSES
# ✓ All deployment validation tests passed
```

```yaml
# Step 3: REFACTOR - Improve while keeping tests green

variables:
  HEALTH_CHECK_TIMEOUT: "30"
  HEALTH_CHECK_RETRIES: "5"
  HEALTH_CHECK_INTERVAL: "5"
  DEPLOYMENT_VALIDATION_ENABLED: "true"

# Reusable deployment validation template
.deployment_validation_template: &deployment_validation
  image: alpine:latest
  before_script:
    - apk add --no-cache curl jq
  script:
    - |
      if [ "$DEPLOYMENT_VALIDATION_ENABLED" != "true" ]; then
        echo "Deployment validation disabled, skipping..."
        exit 0
      fi

      echo "Validating deployment to $ENVIRONMENT..."

      # Enhanced health check with JSON response validation
      RETRY=0
      while [ $RETRY -lt $HEALTH_CHECK_RETRIES ]; do
        RESPONSE=$(curl -sf --max-time $HEALTH_CHECK_TIMEOUT "$DEPLOYMENT_URL/health" 2>/dev/null)

        if [ $? -eq 0 ]; then
          STATUS=$(echo "$RESPONSE" | jq -r '.status // "unknown"')
          if [ "$STATUS" = "healthy" ] || [ "$STATUS" = "ok" ]; then
            echo "✓ Health check passed (status: $STATUS)"

            # Additional validation: Check version matches
            DEPLOYED_VERSION=$(echo "$RESPONSE" | jq -r '.version // "unknown"')
            echo "✓ Deployed version: $DEPLOYED_VERSION"

            exit 0
          fi
        fi

        RETRY=$((RETRY + 1))
        echo "Health check attempt $RETRY/$HEALTH_CHECK_RETRIES failed"
        sleep $HEALTH_CHECK_INTERVAL
      done

      echo "✗ Deployment validation failed"
      exit 1

deploy:validate:staging:
  <<: *deployment_validation
  stage: verify
  variables:
    ENVIRONMENT: "staging"
    DEPLOYMENT_URL: "https://staging.example.com"
  environment:
    name: staging
    action: verify
  needs: ["deploy:staging"]
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

deploy:validate:production:
  <<: *deployment_validation
  stage: verify
  variables:
    ENVIRONMENT: "production"
    DEPLOYMENT_URL: "https://example.com"
  environment:
    name: production
    action: verify
  needs: ["deploy:production"]
  rules:
    - if: $CI_COMMIT_TAG =~ /^v\d+\.\d+\.\d+$/

# Enhanced rollback with notification
rollback:deployment:
  stage: verify
  image: alpine:latest
  before_script:
    - apk add --no-cache curl
  script:
    - |
      echo "🔄 Initiating rollback for $ENVIRONMENT..."

      # Notify team of rollback
      curl -X POST "$SLACK_WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d "{\"text\":\"⚠️ Rollback initiated for $ENVIRONMENT (Pipeline: $CI_PIPELINE_URL)\"}" || true

      # Perform rollback
      # kubectl rollout undo deployment/myapp -n $ENVIRONMENT

      echo "✓ Rollback completed"
  when: on_failure
  needs:
    - job: deploy:validate:staging
      optional: true
    - job: deploy:validate:production
      optional: true

# Run: git push → Pipeline runs → All tests STILL PASS
# ✓ Refactored with templates, parallel validation, enhanced checks
```

### Visual Step-by-Step TDD Example

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     TDD WORKFLOW FOR NEW CI JOB                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  REQUIREMENT: Add container vulnerability blocking to pipeline          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ STEP 1: RED - Write Test First                                  │    │
│  │                                                                 │    │
│  │ test:container-security-policy:                                 │    │
│  │   script:                                                       │    │
│  │     # Test that critical vulnerabilities block deployment       │    │
│  │     - if ! grep -q "CS_SEVERITY_THRESHOLD" .gitlab-ci.yml;     │    │
│  │       then exit 1; fi                                           │    │
│  │     # Test that container scanning is required for deploy       │    │
│  │     - if ! grep -q "needs:.*container_scanning" .gitlab-ci.yml;│    │
│  │       then exit 1; fi                                           │    │
│  │                                                                 │    │
│  │ Pipeline Result: ❌ FAILED                                      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ STEP 2: GREEN - Implement Minimum to Pass                       │    │
│  │                                                                 │    │
│  │ container_scanning:                                             │    │
│  │   variables:                                                    │    │
│  │     CS_SEVERITY_THRESHOLD: "high"                               │    │
│  │                                                                 │    │
│  │ deploy:production:                                              │    │
│  │   needs:                                                        │    │
│  │     - container_scanning                                        │    │
│  │     - test:unit                                                 │    │
│  │                                                                 │    │
│  │ Pipeline Result: ✅ PASSED                                      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ STEP 3: REFACTOR - Improve Without Breaking Tests               │    │
│  │                                                                 │    │
│  │ container_scanning:                                             │    │
│  │   variables:                                                    │    │
│  │     CS_SEVERITY_THRESHOLD: "high"                               │    │
│  │     CS_DISABLE_LANGUAGE_VULNERABILITY_SCAN: "false"             │    │
│  │   rules:                                                        │    │
│  │     - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH               │    │
│  │     - if: $CI_COMMIT_TAG                                        │    │
│  │                                                                 │    │
│  │ # Added: Security gate job                                      │    │
│  │ security:gate:                                                  │    │
│  │   script:                                                       │    │
│  │     - check_vulnerabilities.sh                                  │    │
│  │   allow_failure: false                                          │    │
│  │                                                                 │    │
│  │ Pipeline Result: ✅ STILL PASSING                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every pipeline bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BUG FIX WORKFLOW FOR GITLAB CI/CD                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐                                                   │
│  │  1. BUG REPORTED │  Issue #789: Pipeline fails intermittently        │
│  │     (DISCOVER)   │  on concurrent deployments                        │
│  └────────┬─────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌──────────────────┐                                                   │
│  │  2. WRITE TEST   │  Create test that reproduces the bug              │
│  │     (REPRODUCE)  │  Test MUST FAIL to prove bug exists               │
│  └────────┬─────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌──────────────────┐                                                   │
│  │  3. VERIFY FAIL  │  Confirm test fails for the RIGHT reason          │
│  │     (CONFIRM)    │  Not a flaky test or unrelated failure            │
│  └────────┬─────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌──────────────────┐                                                   │
│  │  4. FIX BUG      │  Implement the fix                                │
│  │     (IMPLEMENT)  │  Test should now PASS                             │
│  └────────┬─────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌──────────────────┐                                                   │
│  │  5. VERIFY PASS  │  Run full test suite                              │
│  │     (VALIDATE)   │  Regression test passes, no new failures          │
│  └────────┬─────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌──────────────────┐                                                   │
│  │  6. DOCUMENT     │  Add comment with bug ID to test                  │
│  │     (RECORD)     │  Update CHANGELOG and issue                       │
│  └────────┬─────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌──────────────────┐                                                   │
│  │  7. DEPLOY       │  Merge with confidence                            │
│  │     (SHIP)       │  Regression permanently prevented                 │
│  └──────────────────┘                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Example Bug Fix with Regression Test

**Bug Report #456**: Pipeline cache not invalidating when package-lock.json changes, causing stale dependencies.

```yaml
# ============================================================
# STEP 1-2: Write test that reproduces Bug #456
# ============================================================

# .gitlab-ci.yml - Add regression test job

# Bug #456: Cache invalidation test
# This test verifies that cache is properly invalidated when
# package-lock.json changes
test:cache-invalidation:
  stage: validate
  image: alpine:latest
  script:
    - |
      # Bug #456: Regression test for cache invalidation
      # This test ensures the cache key includes package-lock.json hash

      echo "Testing cache configuration for Bug #456..."

      # Test 1: Verify cache key uses file hash
      if ! grep -A5 "cache:" .gitlab-ci.yml | grep -q "files:"; then
        echo "FAIL: Cache key does not use file-based hashing"
        echo "Bug #456: Cache not invalidating on dependency changes"
        exit 1
      fi

      # Test 2: Verify package-lock.json is in cache key files
      if ! grep -A10 "cache:" .gitlab-ci.yml | grep -q "package-lock.json"; then
        echo "FAIL: package-lock.json not in cache key"
        echo "Bug #456: Cache not invalidating on dependency changes"
        exit 1
      fi

      # Test 3: Verify cache policy is set correctly
      if grep -q 'policy: "pull"' .gitlab-ci.yml | head -1; then
        # Check if there's a corresponding push job
        if ! grep -q 'policy: "push"' .gitlab-ci.yml; then
          echo "FAIL: Cache pull policy without push policy"
          echo "Bug #456: Cache may become stale"
          exit 1
        fi
      fi

      echo "✓ Bug #456: Cache invalidation tests passed"
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
      changes:
        - .gitlab-ci.yml
        - package-lock.json

# Run: git push → Pipeline runs → test:cache-invalidation FAILS
# ✗ FAIL: Cache key does not use file-based hashing
```

```yaml
# ============================================================
# STEP 3-4: Fix Bug #456 - Proper cache invalidation
# ============================================================

# BEFORE (Buggy configuration):
# default:
#   cache:
#     key: "$CI_COMMIT_REF_SLUG"  # Bug #456: Only uses branch name!
#     paths:
#       - node_modules/

# AFTER (Fixed configuration):
default:
  cache:
    # Bug #456 Fix: Use file hash for cache key invalidation
    key:
      files:
        - package-lock.json      # Invalidate when dependencies change
        - yarn.lock              # Support yarn projects
      prefix: "$CI_COMMIT_REF_SLUG"  # Still include branch for isolation
    paths:
      - node_modules/
      - .npm/
    policy: pull-push  # Bug #456 Fix: Ensure cache is updated

# For jobs that only need to read cache (optimization)
.cache_pull_only:
  cache:
    key:
      files:
        - package-lock.json
      prefix: "$CI_COMMIT_REF_SLUG"
    paths:
      - node_modules/
      - .npm/
    policy: pull  # Only pull, don't update cache

# Job that updates the cache (runs first)
install:dependencies:
  stage: .pre
  script:
    - npm ci --cache .npm --prefer-offline
  cache:
    key:
      files:
        - package-lock.json
      prefix: "$CI_COMMIT_REF_SLUG"
    paths:
      - node_modules/
      - .npm/
    policy: push  # Bug #456 Fix: Explicitly push new cache

# Subsequent jobs use pull-only for speed
lint:
  stage: validate
  extends: .cache_pull_only
  script:
    - npm run lint

test:unit:
  stage: test
  extends: .cache_pull_only
  script:
    - npm test

# Run: git push → Pipeline runs → test:cache-invalidation PASSES
# ✓ Bug #456: Cache invalidation tests passed
```

```yaml
# ============================================================
# STEP 5-6: Verify and Document Bug #456 Fix
# ============================================================

# Additional validation job to prevent regression
verify:bug-fixes:
  stage: validate
  image: alpine:latest
  script:
    - |
      echo "Verifying bug fix regressions..."

      # Bug #456: Cache invalidation
      # Added: 2024-01-15
      # Fixed by: Properly using file-based cache keys
      if ! grep -A5 "key:" .gitlab-ci.yml | grep -q "files:"; then
        echo "REGRESSION: Bug #456 - Cache invalidation broken"
        exit 1
      fi
      echo "✓ Bug #456: Cache invalidation - PROTECTED"

      # Bug #123: Missing retry on runner failures
      # Added: 2024-01-10
      if ! grep -q "runner_system_failure" .gitlab-ci.yml; then
        echo "REGRESSION: Bug #123 - Missing retry configuration"
        exit 1
      fi
      echo "✓ Bug #123: Retry configuration - PROTECTED"

      # Bug #234: Security scan timeout
      # Added: 2024-01-12
      if ! grep -A3 "sast:" .gitlab-ci.yml | grep -q "timeout:"; then
        echo "WARNING: Bug #234 - Consider adding SAST timeout"
      fi

      echo ""
      echo "═══════════════════════════════════════"
      echo "All regression tests passed!"
      echo "═══════════════════════════════════════"
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
```

### Bug Fix Checklist

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      BUG FIX VERIFICATION CHECKLIST                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Bug ID: #___________  Date: ____________  Fixed by: _______________    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ BEFORE FIXING                                                   │    │
│  │ □ Bug is documented in GitLab issue                             │    │
│  │ □ Root cause is identified                                      │    │
│  │ □ Regression test written that reproduces bug                   │    │
│  │ □ Regression test FAILS (proves bug exists)                     │    │
│  │ □ Test includes comment: // Bug #XXX: Description               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ AFTER FIXING                                                    │    │
│  │ □ Regression test now PASSES                                    │    │
│  │ □ All other tests still pass                                    │    │
│  │ □ No new warnings or errors                                     │    │
│  │ □ Pipeline completes successfully                               │    │
│  │ □ MR title references bug: "Fix #XXX: Description"              │    │
│  │ □ MR description explains root cause and fix                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ DOCUMENTATION                                                   │    │
│  │ □ Issue updated with fix details                                │    │
│  │ □ CHANGELOG updated (if applicable)                             │    │
│  │ □ Test file documents bug prevention                            │    │
│  │ □ Related documentation updated                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Issue and Merge Request Management (MANDATORY)

### A. Issue Templates

```markdown
# .gitlab/issue_templates/Bug.md

<!-- 
IMPORTANT: Every bug fix MUST include a regression test.
This ensures the bug doesn't reoccur.
-->

## Bug Description
<!-- Clear and concise description of the bug -->

## Steps to Reproduce
1. 
2. 
3. 

## Expected Behavior
<!-- What should happen -->

## Actual Behavior
<!-- What actually happens -->

## Environment
- OS: 
- Browser/Version: 
- Node.js Version: 
- Application Version: 

## Impact
<!-- Select one -->
- [ ] Critical (System down, data loss, security breach)
- [ ] High (Major feature broken, significant user impact)
- [ ] Medium (Feature degraded, workaround exists)
- [ ] Low (Minor issue, cosmetic)

## Screenshots/Logs
<!-- Paste relevant logs or screenshots -->

## Root Cause Analysis
<!-- To be filled after investigation -->

## Fix Plan
<!-- To be filled by assignee -->

## Regression Test
<!-- MANDATORY: Link to regression test after fix -->
- [ ] Regression test added
- Test location: `tests/path/to/test.ts:LINE`
- Test reproduces bug before fix
- Test passes after fix

## Checklist
- [ ] I have searched existing issues to ensure this is not a duplicate
- [ ] I have provided clear reproduction steps
- [ ] I have included environment information
- [ ] I understand a regression test will be required

/label ~bug ~needs-triage
```

```markdown
# .gitlab/issue_templates/Feature.md

<!--
IMPORTANT: All new features MUST include tests (TDD approach).
Write tests before implementation.
-->

## Feature Description
<!-- What feature do you want to add? -->

## Problem Statement
<!-- What problem does this feature solve? -->

## Proposed Solution
<!-- How should this work? -->

## User Story
As a [type of user]
I want [goal]
So that [benefit]

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Architecture Layer
<!-- Select one or more -->
- [ ] Domain (core business logic)
- [ ] Application (use cases)
- [ ] Infrastructure (external dependencies)
- [ ] Adapter (API/UI)

## Technical Design
<!-- High-level technical approach -->

## API Changes
<!-- If applicable, describe API changes -->

## Database Changes
<!-- If applicable, describe schema changes -->

## Testing Requirements (MANDATORY)
- [ ] Unit tests (TDD - write tests first)
- [ ] Integration tests
- [ ] E2E tests (if UI changes)
- [ ] Performance tests (if applicable)
- [ ] Security considerations

## Dependencies
<!-- Related issues, external dependencies -->

## Documentation Requirements
- [ ] README update
- [ ] API documentation
- [ ] Wiki page
- [ ] Architecture decision record (ADR)

## Definition of Done
- [ ] Tests written first (TDD)
- [ ] Code implemented
- [ ] All tests pass
- [ ] Code reviewed (2+ approvals)
- [ ] Documentation updated
- [ ] Merged to main
- [ ] Deployed to development
- [ ] QA verified

## Alternatives Considered
<!-- What other solutions were considered? -->

/label ~feature ~needs-triage
```

### B. Merge Request Template

```markdown
# .gitlab/merge_request_templates/Default.md

## Description
<!-- Provide a clear description of your changes -->

**Related Issue:** Closes #

## Type of Change
- [ ] 🐛 Bug fix (fixes issue #)
- [ ] ✨ New feature (implements #)
- [ ] 💥 Breaking change
- [ ] 📝 Documentation update
- [ ] ♻️ Refactoring
- [ ] ✅ Test update
- [ ] 🔧 Configuration change
- [ ] 🚀 Performance improvement

## Architecture Layer
- [ ] Domain (core business logic)
- [ ] Application (use cases)
- [ ] Infrastructure (external dependencies)
- [ ] Adapter (API/UI)

## Changes Made
<!-- List specific changes -->
- 
- 
- 

## TDD Compliance (MANDATORY)
- [ ] Tests written BEFORE implementation (Red-Green-Refactor)
- [ ] All tests pass locally
- [ ] Code coverage maintained/increased (≥80%)
- [ ] No skipped or disabled tests
- [ ] Test names are descriptive

**Test Coverage:**
- Before: __%
- After: __%
- New lines covered: __%

## Regression Testing (MANDATORY for bug fixes)
**For bug fixes ONLY:**
- [ ] Regression test added that reproduces the bug
- [ ] Test fails before fix
- [ ] Test passes after fix
- [ ] Test includes comment referencing issue (e.g., `// Bug #123`)
- [ ] Test is not skipped or disabled

**Test Location:** `tests/path/to/test.ts:LINE`

## Testing Performed
- [ ] Unit tests
- [ ] Integration tests
- [ ] E2E tests
- [ ] Manual testing
- [ ] Performance testing
- [ ] Security testing

**Test Commands:**
```bash
npm test
npm run test:coverage
npm run test:integration
npm run test:e2e
```

## Security Considerations
- [ ] No secrets or credentials in code
- [ ] Input validation added
- [ ] Output sanitization added
- [ ] Authorization checks added
- [ ] Security scanning passed

## Performance Impact
<!-- Describe any performance implications -->
- [ ] No negative performance impact
- [ ] Performance improved
- [ ] Performance benchmarks added

## Database Changes
- [ ] No database changes
- [ ] Migration scripts added
- [ ] Rollback scripts added
- [ ] Database documented

## API Changes
- [ ] No API changes
- [ ] Backward compatible
- [ ] Breaking changes documented
- [ ] API version bumped

## Documentation
- [ ] README updated
- [ ] API documentation updated
- [ ] Wiki updated
- [ ] CHANGELOG updated
- [ ] Architecture decision record (ADR) added

## Screenshots
<!-- Add screenshots for UI changes -->

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] No new warnings or errors
- [ ] All pipeline jobs pass
- [ ] Conventional commit format used
- [ ] Issue/bug reference included
- [ ] Squash commits before merge

## Breaking Changes
<!-- Describe any breaking changes -->

## Migration Guide
<!-- If breaking changes, provide migration steps for users -->

## Rollback Plan
<!-- How to rollback if issues arise -->

## Additional Notes
<!-- Any additional context or notes for reviewers -->

---

**Reviewers:** Please verify:
- [ ] Tests follow TDD principles (tests before code)
- [ ] Bug fixes include regression tests
- [ ] Code coverage threshold met (≥80%)
- [ ] Architecture principles followed
- [ ] Security best practices followed
- [ ] Documentation is complete
- [ ] No security vulnerabilities introduced

/assign @reviewer1 @reviewer2
/label ~needs-review
```

### C. Automated MR Management

```yaml
# .gitlab-ci.yml - Automated MR workflows

# Danger.js for automated code review
danger:review:
  stage: validate
  image: node:20-alpine
  script:
    - npm install -g danger
    - danger ci
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"

# Check MR size
mr:check-size:
  stage: validate
  image: alpine:latest
  before_script:
    - apk add --no-cache curl jq
  script:
    - |
      # Get MR changes
      CHANGES=$(curl --header "PRIVATE-TOKEN: $CI_JOB_TOKEN" \
        "$CI_API_V4_URL/projects/$CI_PROJECT_ID/merge_requests/$CI_MERGE_REQUEST_IID/changes" | \
        jq '.changes | length')
      
      echo "MR has $CHANGES file changes"
      
      # Add labels based on size
      if [ "$CHANGES" -lt 10 ]; then
        SIZE_LABEL="size::XS"
      elif [ "$CHANGES" -lt 30 ]; then
        SIZE_LABEL="size::S"
      elif [ "$CHANGES" -lt 100 ]; then
        SIZE_LABEL="size::M"
      elif [ "$CHANGES" -lt 300 ]; then
        SIZE_LABEL="size::L"
        echo "::warning::Large MR - consider splitting"
      else
        SIZE_LABEL="size::XL"
        echo "::error::Very large MR - please split into smaller MRs"
      fi
      
      # Add label via API
      curl --request POST --header "PRIVATE-TOKEN: $CI_JOB_TOKEN" \
        "$CI_API_V4_URL/projects/$CI_PROJECT_ID/merge_requests/$CI_MERGE_REQUEST_IID/labels?labels=$SIZE_LABEL"
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"

# Auto-label based on files changed
mr:auto-label:
  stage: validate
  image: alpine:latest
  before_script:
    - apk add --no-cache curl jq
  script:
    - |
      # Get changed files
      FILES=$(curl --header "PRIVATE-TOKEN: $CI_JOB_TOKEN" \
        "$CI_API_V4_URL/projects/$CI_PROJECT_ID/merge_requests/$CI_MERGE_REQUEST_IID/changes" | \
        jq -r '.changes[].new_path')
      
      LABELS=""
      
      # Add labels based on changed files
      if echo "$FILES" | grep -q "^src/domain/"; then
        LABELS="$LABELS,layer::domain"
      fi
      
      if echo "$FILES" | grep -q "^src/infrastructure/"; then
        LABELS="$LABELS,layer::infrastructure"
      fi
      
      if echo "$FILES" | grep -q "^tests/"; then
        LABELS="$LABELS,type::test"
      fi
      
      if echo "$FILES" | grep -q "\.md$"; then
        LABELS="$LABELS,type::documentation"
      fi
      
      # Add labels
      if [ -n "$LABELS" ]; then
        curl --request POST --header "PRIVATE-TOKEN: $CI_JOB_TOKEN" \
          "$CI_API_V4_URL/projects/$CI_PROJECT_ID/merge_requests/$CI_MERGE_REQUEST_IID/labels?labels=${LABELS:1}"
      fi
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
```

---

## 4. GitLab Container Registry (MANDATORY)

### A. Optimized Dockerfile for GitLab

```dockerfile
# Dockerfile - Optimized for GitLab Container Registry

# syntax=docker/dockerfile:1.6

ARG NODE_VERSION=20
ARG ALPINE_VERSION=3.19

# ============================================
# Stage 1: Base
# ============================================
FROM node:${NODE_VERSION}-alpine${ALPINE_VERSION} AS base

# Set labels
LABEL maintainer="team@example.com"
LABEL org.opencontainers.image.source="https://gitlab.com/group/project"
LABEL org.opencontainers.image.description="Application description"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Install dumb-init for signal handling
RUN apk add --no-cache dumb-init

# ============================================
# Stage 2: Dependencies
# ============================================
FROM base AS dependencies

# Copy package files
COPY package.json package-lock.json ./

# Install all dependencies with cache mount
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit

# ============================================
# Stage 3: Build
# ============================================
FROM dependencies AS build

# Copy source code
COPY . .

# Build application
RUN npm run build

# Run tests in build stage (fail fast)
RUN npm test

# ============================================
# Stage 4: Production Dependencies
# ============================================
FROM base AS prod-dependencies

COPY package.json package-lock.json ./

# Install only production dependencies
RUN --mount=type=cache,target=/root/.npm \
    npm ci --only=production --prefer-offline --no-audit --ignore-scripts

# ============================================
# Stage 5: Production
# ============================================
FROM base AS production

# Build arguments for metadata
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Set labels with build info
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.version="${VERSION}"

# Set NODE_ENV
ENV NODE_ENV=production

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

# Copy production dependencies
COPY --from=prod-dependencies --chown=nodejs:nodejs /app/node_modules ./node_modules

# Copy built application
COPY --from=build --chown=nodejs:nodejs /app/dist ./dist
COPY --from=build --chown=nodejs:nodejs /app/package.json ./

# Switch to non-root user
USER nodejs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000/health', (r) => process.exit(r.statusCode === 200 ? 0 : 1))"

# Use dumb-init
ENTRYPOINT ["dumb-init", "--"]

# Start application
CMD ["node", "dist/main.js"]
```

### B. .dockerignore

```
# .dockerignore

# Git
.git
.gitignore
.gitlab-ci.yml
.gitlab/

# Dependencies
node_modules
npm-debug.log

# Testing
coverage
.nyc_output
*.test.ts
*.spec.ts
tests/

# Documentation
docs/
*.md
!README.md

# IDE
.vscode
.idea
*.swp
*.swo

# CI/CD
.gitlab-ci.yml

# Environment
.env
.env.*
!.env.example

# Build artifacts
dist
build
out

# Logs
logs
*.log

# OS
.DS_Store
Thumbs.db
```

### C. Container Registry Management

```yaml
# .gitlab-ci.yml - Advanced container management

variables:
  # Use GitLab's built-in variables
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"
  # Enable BuildKit
  DOCKER_BUILDKIT: 1
  # Registry configuration
  CI_REGISTRY_IMAGE: $CI_REGISTRY/$CI_PROJECT_PATH

# Build multi-platform images
build:multiplatform:
  stage: build
  image: docker:24-cli
  services:
    - docker:24-dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    # Set up buildx for multi-platform
    - docker buildx create --use --name multiplatform
  script:
    - |
      docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
        --build-arg VCS_REF=$CI_COMMIT_SHORT_SHA \
        --build-arg VERSION=${CI_COMMIT_TAG:-$CI_COMMIT_SHORT_SHA} \
        --tag $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA \
        --tag $CI_REGISTRY_IMAGE:$CI_COMMIT_REF_SLUG \
        --cache-from type=registry,ref=$CI_REGISTRY_IMAGE:buildcache \
        --cache-to type=registry,ref=$CI_REGISTRY_IMAGE:buildcache,mode=max \
        --push \
        .
    # Tag latest for default branch
    - |
      if [ "$CI_COMMIT_BRANCH" == "$CI_DEFAULT_BRANCH" ]; then
        docker buildx imagetools create \
          --tag $CI_REGISTRY_IMAGE:latest \
          $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA
      fi
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
    - if: $CI_COMMIT_TAG

# Cleanup old images
cleanup:registry:
  stage: cleanup
  image: alpine:latest
  before_script:
    - apk add --no-cache curl jq
  script:
    - |
      # Get list of tags
      TAGS=$(curl --header "PRIVATE-TOKEN: $CI_JOB_TOKEN" \
        "$CI_API_V4_URL/projects/$CI_PROJECT_ID/registry/repositories" | \
        jq -r '.[].location')
      
      echo "Registry: $TAGS"
      
      # Delete tags older than 30 days (except latest, main, develop)
      # Add your cleanup logic here
  rules:
    - if: $CI_PIPELINE_SOURCE == "schedule"
  allow_failure: true
```

---

## 5. Branch Protection & Code Review (MANDATORY)

### A. Protected Branches Configuration

```ruby
# Configure via GitLab UI or API

# Settings > Repository > Protected branches

# Main branch protection:
{
  "name": "main",
  "push_access_levels": [
    {"access_level": 0}  # No one can push
  ],
  "merge_access_levels": [
    {"access_level": 40}  # Maintainers can merge
  ],
  "allow_force_push": false,
  "code_owner_approval_required": true,
  "unprotect_access_levels": [
    {"access_level": 40}  # Only maintainers can unprotect
  ]
}

# Develop branch protection:
{
  "name": "develop",
  "push_access_levels": [
    {"access_level": 30}  # Developers can push
  ],
  "merge_access_levels": [
    {"access_level": 30}  # Developers can merge
  ],
  "allow_force_push": false,
  "code_owner_approval_required": true
}
```

### B. CODEOWNERS File

```
# CODEOWNERS

# Global owners
* @group/core-team

# Domain layer (critical code)
/src/domain/ @group/domain-experts @group/architects

# Application layer
/src/application/ @group/backend-team

# Infrastructure layer
/src/infrastructure/ @group/devops-team @group/backend-team

# API adapters
/src/adapters/api/ @group/api-team

# Tests (require thorough review)
/tests/ @group/qa-team @group/core-team

# CI/CD configuration
/.gitlab-ci.yml @group/devops-team
/.gitlab/ @group/devops-team

# Security-sensitive files
/Dockerfile @group/security-team @group/devops-team
/SECURITY.md @group/security-team
/.gitlab/security-policies/ @group/security-team

# Documentation
/docs/ @group/docs-team
/*.md @group/docs-team

# Configuration files
*.config.js @group/devops-team
*.config.ts @group/devops-team
```

### C. Approval Rules

```yaml
# .gitlab/approval_rules.yml

---
# Require 2 approvals for all MRs
default:
  approvals_required: 2
  reset_approvals_on_push: true
  disable_overriding_approvers_per_merge_request: false
  merge_requests_author_approval: false
  merge_requests_disable_committers_approval: true

# Security-sensitive changes require security team approval
security:
  name: "Security Team Approval"
  approvals_required: 1
  groups:
    - security-team
  protected_branches:
    - main
    - develop
  rule_type: regular
  contains_hidden_groups: false

# Infrastructure changes require DevOps approval
infrastructure:
  name: "DevOps Approval"
  approvals_required: 1
  groups:
    - devops-team
  files:
    - "*.gitlab-ci.yml"
    - ".gitlab/"
    - "Dockerfile"
    - "k8s/"
  rule_type: regular

# Domain changes require architect approval
domain:
  name: "Architecture Approval"
  approvals_required: 1
  groups:
    - architects
  files:
    - "src/domain/"
  rule_type: regular
```

---

## 6. GitLab Pages & Documentation (MANDATORY)

### A. GitLab Pages Deployment

```yaml
# .gitlab-ci.yml - Deploy documentation to GitLab Pages

pages:
  stage: deploy
  image: node:20-alpine
  script:
    - npm ci
    # Generate API documentation
    - npm run docs:generate
    # Generate coverage report
    - npm run test:coverage
    # Create public directory
    - mkdir -p public
    # Copy documentation
    - cp -r docs/generated/* public/
    # Copy coverage report
    - cp -r coverage/lcov-report public/coverage
    # Create index page
    - |
      cat > public/index.html <<EOF
      <!DOCTYPE html>
      <html>
      <head>
        <title>Project Documentation</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 40px; }
          h1 { color: #333; }
          ul { list-style: none; padding: 0; }
          li { margin: 10px 0; }
          a { color: #1890ff; text-decoration: none; }
          a:hover { text-decoration: underline; }
        </style>
      </head>
      <body>
        <h1>Project Documentation</h1>
        <ul>
          <li><a href="api/">API Documentation</a></li>
          <li><a href="coverage/">Test Coverage Report</a></li>
          <li><a href="architecture/">Architecture Documentation</a></li>
        </ul>
      </body>
      </html>
      EOF
  artifacts:
    paths:
      - public
    expire_in: 30 days
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
```

### B. Documentation Generation

```json
// package.json - Documentation scripts

{
  "scripts": {
    "docs:generate": "typedoc --out docs/generated/api src/",
    "docs:serve": "npm run docs:generate && cd docs/generated && python3 -m http.server 8080",
    "docs:coverage": "npm run test:coverage && open-cli coverage/lcov-report/index.html"
  },
  "devDependencies": {
    "typedoc": "^0.25.0",
    "typedoc-plugin-markdown": "^3.17.0"
  }
}
```

### C. Wiki Automation

```yaml
# .gitlab-ci.yml - Auto-update wiki

wiki:update:
  stage: deploy
  image: alpine:latest
  before_script:
    - apk add --no-cache git
    - git config --global user.email "ci@example.com"
    - git config --global user.name "GitLab CI"
  script:
    - |
      # Clone wiki repository
      git clone https://gitlab-ci-token:${CI_JOB_TOKEN}@gitlab.com/${CI_PROJECT_PATH}.wiki.git wiki
      cd wiki
      
      # Copy documentation to wiki
      cp ../docs/architecture/*.md ./
      
      # Commit and push
      git add .
      git commit -m "Auto-update from CI pipeline $CI_PIPELINE_ID" || true
      git push origin master
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
      changes:
        - docs/**/*
  allow_failure: true
```

---

## 7. Security & Dependency Management (MANDATORY)

### A. Infrastructure Security Scanning

```yaml
# GitLab SAST - built into CI/CD pipeline
# Add to .gitlab-ci.yml
include:
  - template: Security/SAST.gitlab-ci.yml
  - template: Security/DAST.gitlab-ci.yml
  - template: Security/Dependency-Scanning.gitlab-ci.yml
  - template: Security/Container-Scanning.gitlab-ci.yml
  - template: Security/Secret-Detection.gitlab-ci.yml
  - template: Security/License-Scanning.gitlab-ci.yml

# SAST configuration overrides
sast:
  stage: test
  variables:
    SAST_EXCLUDED_PATHS: "spec,test,tests,tmp"
    SEARCH_MAX_DEPTH: 10
    SAST_BANDIT_EXCLUDED_PATHS: "*/test/**"

# DAST configuration
dast:
  stage: dast
  variables:
    DAST_WEBSITE: "https://staging.example.com"
    DAST_FULL_SCAN_ENABLED: "true"
```

### B. Vulnerability Scanning

```yaml
# Dependency scanning with gemnasium
dependency_scanning:
  stage: test
  variables:
    DS_DEFAULT_ANALYZERS: "gemnasium,gemnasium-python,gemnasium-maven"
    DS_EXCLUDED_PATHS: "spec,test,tests"

# Container scanning
container_scanning:
  stage: test
  variables:
    CS_IMAGE: "$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA"
    CS_SEVERITY_THRESHOLD: "HIGH"

# Secret detection - prevent credentials in source code
secret_detection:
  stage: test
  variables:
    SECRET_DETECTION_HISTORIC_SCAN: "true"
    SECRET_DETECTION_EXCLUDED_PATHS: "tests/"
```

```bash
# Query vulnerability reports via GitLab API
curl --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
  "https://gitlab.com/api/v4/projects/$PROJECT_ID/vulnerability_findings?severity=critical&state=detected"

# Export vulnerability report
curl --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
  "https://gitlab.com/api/v4/security/projects/$PROJECT_ID/vulnerability_exports" -X POST
```

### C. Policy & Compliance

```yaml
# License compliance scanning
license_scanning:
  stage: test
  variables:
    LICENSE_MANAGEMENT_SETTINGS_FILE: ".license-compliance.yml"

# Merge request approval rules for security
# Configure via Settings > Merge Requests > Approval Rules
# Require security team approval when vulnerabilities detected

# Compliance frameworks (GitLab Ultimate)
# Settings > General > Compliance framework
# Enforces specific pipeline configurations across projects
```

```yaml
# .license-compliance.yml - allowed and denied licenses
allowlist:
  - MIT
  - Apache-2.0
  - BSD-2-Clause
  - BSD-3-Clause
  - ISC
denylist:
  - GPL-3.0
  - AGPL-3.0
  - SSPL-1.0
```

---

## 8. Deployment Checklist

### Repository Setup
- [ ] **README.md**: Comprehensive with badges
- [ ] **LICENSE**: Appropriate license file
- [ ] **CONTRIBUTING.md**: Contribution guidelines
- [ ] **CODE_OF_CONDUCT.md**: Community standards
- [ ] **SECURITY.md**: Security policy
- [ ] **.gitignore**: Comprehensive exclusions
- [ ] **Issue templates**: Bug, feature templates in `.gitlab/issue_templates/`
- [ ] **MR template**: In `.gitlab/merge_request_templates/Default.md`
- [ ] **CODEOWNERS**: Code ownership defined

### GitLab CI/CD
- [ ] **.gitlab-ci.yml**: Comprehensive pipeline
- [ ] **Stages**: validate, test, build, security, deploy, verify
- [ ] **TDD verification**: Automated test checks
- [ ] **Bug fix verification**: Regression test checks
- [ ] **DAG pipeline**: Optimized with `needs:`
- [ ] **Caching**: Dependency caching configured
- [ ] **Artifacts**: Build artifacts uploaded
- [ ] **Environments**: Dev, staging, production configured
- [ ] **Manual approvals**: Production deployment requires approval

### Security
- [ ] **SAST**: Static analysis enabled
- [ ] **DAST**: Dynamic analysis enabled
- [ ] **Dependency scanning**: Enabled
- [ ] **Container scanning**: Enabled
- [ ] **Secret detection**: Enabled
- [ ] **License scanning**: Enabled
- [ ] **Security policies**: Configured in `.gitlab/security-policies/`
- [ ] **Vulnerability management**: Dashboard monitored
- [ ] **Dependency updates**: Auto-update configured

### Branch Protection
- [ ] **Main branch**: Protected with no direct push
- [ ] **Develop branch**: Protected with approvals
- [ ] **Approval rules**: Configured (minimum 2 approvals)
- [ ] **Code owners**: Required for sensitive paths
- [ ] **Merge checks**: All pipelines must pass
- [ ] **Reset approvals**: On new push enabled
- [ ] **Author approval**: Disabled
- [ ] **Force push**: Disabled

### Container Registry
- [ ] **Dockerfile**: Multi-stage, optimized
- [ ] **.dockerignore**: Configured
- [ ] **Multi-platform**: amd64 and arm64 builds
- [ ] **Image tagging**: Semantic versioning
- [ ] **Container scanning**: Automated
- [ ] **Registry cleanup**: Old images removed
- [ ] **Health checks**: Defined in Dockerfile

### Documentation
- [ ] **API docs**: Auto-generated from code
- [ ] **Architecture docs**: ADRs and diagrams
- [ ] **GitLab Pages**: Deployed
- [ ] **Wiki**: Active and updated
- [ ] **Badges**: Build, coverage, security in README
- [ ] **CHANGELOG**: Maintained

### Automation
- [ ] **Auto-labeling**: MR labels automated
- [ ] **MR size checking**: Automated
- [ ] **Issue linking**: Required
- [ ] **Commit linting**: Conventional Commits enforced
- [ ] **Release automation**: Configured
- [ ] **Scheduled pipelines**: Security scans, cleanup

### TDD & Testing
- [ ] **TDD workflow**: Verification automated
- [ ] **Coverage tracking**: Configured and visible
- [ ] **Coverage threshold**: ≥80% enforced
- [ ] **Regression tests**: Required for bug fixes
- [ ] **Test reports**: Published in pipeline
- [ ] **Parallel testing**: Matrix configured

---

## 9. Why This Configuration Works

1. **TDD Enforcement**: Pipelines verify tests exist and pass before merge, preventing untested code.
2. **Regression Shield**: Bug fixes require regression tests, building safety net over time.
3. **Security First**: Multiple scanning layers (SAST, DAST, dependencies, containers, secrets).
4. **Everything as Code**: All configuration in `.gitlab-ci.yml`, versioned and reviewed.
5. **Automated CI/CD**: Push → Test → Build → Deploy pipeline fully automated.
6. **DAG Pipelines**: Parallel execution with `needs:` reduces pipeline time significantly.
7. **Branch Protection**: Quality gates prevent bad code from reaching production.
8. **Issue Tracking**: Full traceability from issue → code → deployment.
9. **Container Ready**: GitLab Container Registry with multi-platform builds and scanning.
10. **GitLab Pages**: Automated documentation deployment, always up-to-date.
11. **Efficiency**: Caching, parallelization, selective execution optimize pipeline performance.
12. **Observability**: Security dashboard, coverage reports, pipeline metrics provide visibility.

---

## Quick Reference

### Common glab CLI Commands

```bash
# ═══════════════════════════════════════════════════════════════════════
# GLAB CLI - QUICK REFERENCE
# ═══════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────
# AUTHENTICATION & CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
glab auth login                          # Authenticate with GitLab
glab auth status                         # Check authentication status
glab config set editor vim               # Set default editor
glab config set git_protocol ssh         # Set git protocol (ssh/https)

# ──────────────────────────────────────────────────────────────────────
# MERGE REQUESTS
# ──────────────────────────────────────────────────────────────────────
glab mr create                           # Create MR interactively
glab mr create --fill                    # Create MR with commit info
glab mr create --draft                   # Create draft MR
glab mr create -t "Title" -d "Desc"      # Create MR with title and description
glab mr create --target-branch develop   # Create MR targeting specific branch

glab mr list                             # List open MRs
glab mr list --state merged              # List merged MRs
glab mr list --author @me                # List your MRs
glab mr list --reviewer @me              # List MRs for your review

glab mr view 123                         # View MR #123
glab mr view --web                       # Open MR in browser
glab mr checkout 123                     # Checkout MR #123 locally
glab mr diff 123                         # Show MR diff

glab mr approve 123                      # Approve MR
glab mr merge 123                        # Merge MR
glab mr merge 123 --squash               # Squash merge
glab mr merge 123 --remove-source-branch # Delete branch after merge
glab mr close 123                        # Close MR without merging

glab mr note 123 -m "Comment"            # Add comment to MR
glab mr update 123 --title "New title"   # Update MR title
glab mr update 123 --draft               # Convert to draft
glab mr update 123 --ready               # Mark as ready for review

# ──────────────────────────────────────────────────────────────────────
# ISSUES
# ──────────────────────────────────────────────────────────────────────
glab issue create                        # Create issue interactively
glab issue create -t "Bug: X" -l bug     # Create issue with title and label
glab issue list                          # List open issues
glab issue list --label bug              # List issues with label
glab issue list --assignee @me           # List issues assigned to you
glab issue view 456                      # View issue #456
glab issue view 456 --web                # Open issue in browser
glab issue close 456                     # Close issue
glab issue reopen 456                    # Reopen issue
glab issue note 456 -m "Comment"         # Add comment to issue

# ──────────────────────────────────────────────────────────────────────
# PIPELINES & CI/CD
# ──────────────────────────────────────────────────────────────────────
glab ci status                           # Show pipeline status
glab ci view                             # View current pipeline
glab ci list                             # List recent pipelines
glab ci run                              # Trigger new pipeline
glab ci run -b feature-branch            # Trigger pipeline on branch

glab ci trace                            # Show running job logs
glab ci trace JOB_ID                     # Show specific job logs
glab ci retry                            # Retry failed pipeline
glab ci cancel                           # Cancel running pipeline

glab ci lint                             # Lint .gitlab-ci.yml
glab ci lint .gitlab-ci.yml              # Lint specific file

# View job artifacts
glab ci artifact JOB_ID                  # Download job artifacts
glab ci artifact list                    # List available artifacts

# ──────────────────────────────────────────────────────────────────────
# REPOSITORY & PROJECT
# ──────────────────────────────────────────────────────────────────────
glab repo clone group/project            # Clone repository
glab repo fork                           # Fork current repository
glab repo view                           # View repository info
glab repo view --web                     # Open repository in browser

glab release create v1.0.0               # Create new release
glab release list                        # List releases
glab release view v1.0.0                 # View release details

glab label list                          # List project labels
glab label create "priority::high" -c red # Create new label

# ──────────────────────────────────────────────────────────────────────
# SEARCH
# ──────────────────────────────────────────────────────────────────────
glab search issues "bug"                 # Search issues
glab search mrs "feature"                # Search merge requests
glab search projects "api"               # Search projects

# ──────────────────────────────────────────────────────────────────────
# ALIASES & SHORTCUTS
# ──────────────────────────────────────────────────────────────────────
glab alias set mrc 'mr create --fill'    # Create alias for MR creation
glab alias set cis 'ci status'           # Create alias for CI status
glab alias list                          # List all aliases
```

### GitLab CI/CD Patterns Cheat Sheet

```yaml
# ═══════════════════════════════════════════════════════════════════════
# GITLAB CI/CD PATTERNS CHEAT SHEET
# ═══════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────
# WORKFLOW RULES (Control when pipelines run)
# ──────────────────────────────────────────────────────────────────────
workflow:
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"  # MR pipelines
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH       # Main branch
    - if: $CI_COMMIT_TAG                                 # Tags
    - if: $CI_PIPELINE_SOURCE == "web"                   # Manual trigger
    - if: $CI_PIPELINE_SOURCE == "schedule"              # Scheduled

# ──────────────────────────────────────────────────────────────────────
# JOB RULES (Control when jobs run)
# ──────────────────────────────────────────────────────────────────────
job:
  rules:
    # Run only on main branch
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

    # Run on MRs only
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"

    # Run on tags matching pattern
    - if: $CI_COMMIT_TAG =~ /^v\d+\.\d+\.\d+$/

    # Run when specific files change
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
      changes:
        - src/**/*
        - package.json

    # Manual trigger with conditions
    - if: $CI_COMMIT_BRANCH == "production"
      when: manual

    # Never run on specific conditions
    - if: $CI_COMMIT_MESSAGE =~ /skip-ci/
      when: never

# ──────────────────────────────────────────────────────────────────────
# CACHING STRATEGIES
# ──────────────────────────────────────────────────────────────────────
# File-based cache key (recommended)
cache:
  key:
    files:
      - package-lock.json
    prefix: $CI_COMMIT_REF_SLUG
  paths:
    - node_modules/
    - .npm/

# Branch-based cache (simpler but may use stale deps)
cache:
  key: $CI_COMMIT_REF_SLUG
  paths:
    - node_modules/

# Pull-only cache (faster for read-heavy jobs)
cache:
  key: deps-$CI_COMMIT_REF_SLUG
  paths:
    - node_modules/
  policy: pull

# Push-only cache (for dependency installation jobs)
cache:
  key: deps-$CI_COMMIT_REF_SLUG
  paths:
    - node_modules/
  policy: push

# ──────────────────────────────────────────────────────────────────────
# ARTIFACTS PATTERNS
# ──────────────────────────────────────────────────────────────────────
# Build artifacts
artifacts:
  paths:
    - dist/
    - build/
  expire_in: 1 week

# Test reports
artifacts:
  when: always
  reports:
    junit: coverage/junit.xml
    coverage_report:
      coverage_format: cobertura
      path: coverage/cobertura-coverage.xml
    codequality: gl-code-quality-report.json

# Environment variables from job
artifacts:
  reports:
    dotenv: build.env

# Conditional artifacts
artifacts:
  paths:
    - logs/
  when: on_failure
  expire_in: 1 day

# ──────────────────────────────────────────────────────────────────────
# DEPENDENCIES WITH NEEDS (DAG)
# ──────────────────────────────────────────────────────────────────────
# Basic needs (wait for specific jobs)
deploy:
  needs: ["build", "test"]

# Needs with artifacts
deploy:
  needs:
    - job: build
      artifacts: true
    - job: test
      artifacts: false

# Optional needs (don't fail if job doesn't exist)
deploy:
  needs:
    - job: e2e-test
      optional: true

# Cross-pipeline needs
deploy:
  needs:
    - project: group/other-project
      job: build
      ref: main
      artifacts: true

# ──────────────────────────────────────────────────────────────────────
# REUSABLE TEMPLATES
# ──────────────────────────────────────────────────────────────────────
# Hidden job template (starts with .)
.base_template:
  image: node:20-alpine
  before_script:
    - npm ci
  cache:
    key:
      files:
        - package-lock.json
    paths:
      - node_modules/

# Extend template
test:
  extends: .base_template
  script:
    - npm test

# YAML anchors
.defaults: &defaults
  image: node:20-alpine
  tags:
    - docker

job1:
  <<: *defaults
  script: echo "job1"

job2:
  <<: *defaults
  script: echo "job2"

# ──────────────────────────────────────────────────────────────────────
# PARALLEL & MATRIX JOBS
# ──────────────────────────────────────────────────────────────────────
# Parallel matrix
test:
  parallel:
    matrix:
      - NODE_VERSION: ["18", "20", "22"]
        OS: ["alpine", "slim"]
  image: node:${NODE_VERSION}-${OS}
  script:
    - npm test

# Simple parallel
test:
  parallel: 5
  script:
    - npm test -- --shard=$CI_NODE_INDEX/$CI_NODE_TOTAL

# ──────────────────────────────────────────────────────────────────────
# ENVIRONMENTS
# ──────────────────────────────────────────────────────────────────────
deploy:staging:
  environment:
    name: staging
    url: https://staging.example.com
    on_stop: stop:staging
    auto_stop_in: 1 week

deploy:production:
  environment:
    name: production
    url: https://example.com
    action: start
  when: manual
  resource_group: production

stop:staging:
  environment:
    name: staging
    action: stop
  when: manual

# ──────────────────────────────────────────────────────────────────────
# SERVICES (SIDECARS)
# ──────────────────────────────────────────────────────────────────────
test:integration:
  services:
    - name: postgres:15-alpine
      alias: db
    - name: redis:7-alpine
      alias: cache
  variables:
    POSTGRES_DB: test_db
    POSTGRES_USER: test
    POSTGRES_PASSWORD: test
    DATABASE_URL: postgresql://test:test@db:5432/test_db
    REDIS_URL: redis://cache:6379

# ──────────────────────────────────────────────────────────────────────
# INCLUDE PATTERNS
# ──────────────────────────────────────────────────────────────────────
include:
  # Local file
  - local: '.gitlab/ci/templates/test.yml'

  # Remote file
  - remote: 'https://example.com/ci/template.yml'

  # Project file
  - project: 'group/shared-ci'
    ref: main
    file: '/templates/deploy.yml'

  # GitLab templates
  - template: Security/SAST.gitlab-ci.yml
  - template: Security/Dependency-Scanning.gitlab-ci.yml

# ──────────────────────────────────────────────────────────────────────
# RETRY & TIMEOUT
# ──────────────────────────────────────────────────────────────────────
job:
  retry:
    max: 2
    when:
      - runner_system_failure
      - stuck_or_timeout_failure
      - scheduler_failure
  timeout: 30 minutes
  interruptible: true

# ──────────────────────────────────────────────────────────────────────
# SECURITY SCANNING (Quick Setup)
# ──────────────────────────────────────────────────────────────────────
include:
  - template: Security/SAST.gitlab-ci.yml
  - template: Security/Dependency-Scanning.gitlab-ci.yml
  - template: Security/Secret-Detection.gitlab-ci.yml
  - template: Security/Container-Scanning.gitlab-ci.yml

sast:
  variables:
    SAST_EXCLUDED_PATHS: "spec,test,tests,node_modules"

container_scanning:
  variables:
    CS_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA
    CS_SEVERITY_THRESHOLD: "medium"
```

### Pipeline Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     GITLAB CI/CD PIPELINE STRUCTURE                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STAGES (Sequential)                                                    │
│  ═══════════════════                                                    │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ .pre (built-in)     │ Runs before all stages                     │   │
│  │                     │ Good for: dependency installation          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ validate             │ lint:code    lint:commits   verify:issue   │   │
│  │                      │     │            │              │          │   │
│  │                      │     └────────────┴──────────────┘          │   │
│  │                      │            (parallel)                      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ test                 │ test:unit [18] test:unit [20] test:integ   │   │
│  │                      │      │              │             │        │   │
│  │                      │      └──────────────┴─────────────┘        │   │
│  │                      │            (parallel matrix)               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ build                │ build:app ───────► build:docker            │   │
│  │                      │               (needs build:app)            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ security             │ sast   dast   dependency   container       │   │
│  │                      │   │      │        │           │            │   │
│  │                      │   └──────┴────────┴───────────┘            │   │
│  │                      │          (parallel, can start early        │   │
│  │                      │           with needs: [])                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ deploy               │ deploy:dev ──► deploy:staging ──► deploy:prod│  │
│  │                      │  (auto)         (auto)          (manual)   │   │
│  │                      │                                            │   │
│  │                      │ resource_group: production (prevents       │   │
│  │                      │ concurrent production deployments)         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ verify                │ verify:production   │ rollback (on_failure)│  │
│  │                       │ (smoke tests)       │ (auto rollback)      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ .post (built-in)     │ Runs after all stages                     │   │
│  │                      │ Good for: cleanup, notifications          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  DAG VISUALIZATION (with needs:)                                        │
│  ═══════════════════════════════                                        │
│                                                                         │
│     lint ─────────────────────────────────────┐                         │
│       │                                       │                         │
│       ▼                                       ▼                         │
│  test:unit ──────────────────┬────────► deploy:staging                  │
│       │                      │               │                          │
│       │                      │               ▼                          │
│  test:integration ───────────┤         deploy:prod (manual)             │
│       │                      │               │                          │
│       ▼                      │               ▼                          │
│  build:app ──► build:docker ─┘         verify:prod                      │
│                     │                                                   │
│                     ▼                                                   │
│              container_scan                                             │
│                                                                         │
│  Legend:                                                                │
│  ───────► needs (explicit dependency)                                   │
│  │        stage ordering (implicit)                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

```yaml
# ═══════════════════════════════════════════════════════════════════════
# MINIMAL STARTER PIPELINE
# ═══════════════════════════════════════════════════════════════════════

stages:
  - validate
  - test
  - build
  - deploy

default:
  image: node:20-alpine
  cache:
    key:
      files:
        - package-lock.json
    paths:
      - node_modules/

workflow:
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
    - if: $CI_COMMIT_TAG

lint:
  stage: validate
  script:
    - npm ci
    - npm run lint

test:
  stage: test
  script:
    - npm ci
    - npm test
  coverage: '/All files[^|]*\|[^|]*\s+([\d\.]+)/'
  artifacts:
    reports:
      junit: coverage/junit.xml

build:
  stage: build
  script:
    - npm ci
    - npm run build
  artifacts:
    paths:
      - dist/
    expire_in: 1 week

deploy:
  stage: deploy
  script:
    - echo "Deploying..."
  environment:
    name: production
    url: https://example.com
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
```

---

## References

- [GitLab CI/CD Documentation](https://docs.gitlab.com/ee/ci/)
- [GitLab Container Registry](https://docs.gitlab.com/ee/user/packages/container_registry/)
- [GitLab Security](https://docs.gitlab.com/ee/user/application_security/)
- [GitLab Pages](https://docs.gitlab.com/ee/user/project/pages/)
- [Protected Branches](https://docs.gitlab.com/ee/user/project/protected_branches.html)
- [Merge Request Approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [GitLab Auto DevOps](https://docs.gitlab.com/ee/topics/autodevops/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)

---

**Last Updated:** 2026-01-18  
**Version:** 1.0  
**Maintainer:** DevOps Team


**End of Modern GitLab Workflow Guidelines**
