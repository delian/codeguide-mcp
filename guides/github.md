# Modern GitHub Workflow Guidelines
Mandatory standards and best practices for GitHub usage, including CI/CD, security, automation, issue tracking, and container management. GitHub Actions, GitHub Security, Dependabot, GitHub Container Registry, GitHub Pages.

---

**Agent Profile**: The GitHub DevOps Expert  
**Role**: Senior DevOps Engineer & GitHub Specialist  
**Objective**: Generate efficient, secure, automated GitHub workflows with comprehensive CI/CD pipelines and best practices.  
**Tools**: GitHub Actions, GitHub Security, Dependabot, GitHub Container Registry, GitHub Pages.

## Core Philosophies

The agent must adhere to the "GITHUB-FIRST" principles for every GitHub workflow:

**Test-Driven Development (TDD)**: ALL workflows MUST verify tests pass before merge (Red-Green-Refactor mandatory).
**Regression Shield**: EVERY bug fix MUST reference issue ID and include regression test verification in CI.
**Security First**: Secrets in GitHub Secrets/Environments, CodeQL scanning, Dependabot enabled.
**Everything as Code**: Workflows, configurations, infrastructure defined in repository.
**Automated CI/CD**: Every push triggers tests, every merge to main triggers deployment.
**Branch Protection**: Main/develop protected, require reviews, status checks, linear history.
**Issue Tracking**: Every commit/PR links to issue, clear templates, labels, milestones.

**Container Ready**: GitHub Container Registry for images, automated builds, vulnerability scanning.
**Documentation**: README, CONTRIBUTING, CODE_OF_CONDUCT, comprehensive docs.
**Reproducible Builds**: Dependency locking, deterministic workflows, versioned actions.
**Observability**: Workflow metrics, deployment tracking, security alerts.
**Community Standards**: License, security policy, issue/PR templates, code of conduct.
**Automation**: Automated labeling, stale issue management, release notes generation.

---

## 1. GitHub Actions CI/CD (MANDATORY)

### A. Basic CI Workflow Structure

**Every repository MUST have a comprehensive CI workflow:**

```yaml
# .github/workflows/ci.yml - Comprehensive CI pipeline

name: CI

on:
  push:
    branches:
      - main
      - develop
    paths-ignore:
      - '**.md'
      - 'docs/**'
  pull_request:
    branches:
      - main
      - develop
  workflow_dispatch:

# Cancel in-progress runs for same workflow
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

env:
  NODE_VERSION: '20.x'
  PYTHON_VERSION: '3.12'
  GO_VERSION: '1.22'

jobs:
  # Job 1: Lint and format check
  lint:
    name: Lint Code
    runs-on: ubuntu-latest
    timeout-minutes: 10
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for proper analysis
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run linter
        run: npm run lint
      
      - name: Check formatting
        run: npm run format:check
      
      - name: Check for TODOs
        run: |
          if grep -r "TODO" src/; then
            echo "::warning::Found TODO comments in code"
          fi

  # Job 2: TDD Verification (MANDATORY)
  test:
    name: Test (TDD Verification)
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    strategy:
      matrix:
        node-version: [18.x, 20.x]
        os: [ubuntu-latest, windows-latest, macos-latest]
      fail-fast: false
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Setup Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Verify tests exist
        run: |
          # Verify that tests exist for all source files
          if [ $(find tests/ -name "*.test.ts" | wc -l) -eq 0 ]; then
            echo "::error::No test files found"
            exit 1
          fi
          echo "✓ Test files found"
      
      - name: Run tests (TDD)
        run: npm test
        env:
          CI: true
      
      - name: Check test coverage
        run: npm run test:coverage
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: ./coverage/coverage-final.json
          flags: unittests
          name: codecov-${{ matrix.os }}-node-${{ matrix.node-version }}
          fail_ci_if_error: true
      
      - name: Verify coverage threshold
        run: |
          # Extract coverage percentage
          COVERAGE=$(cat coverage/coverage-summary.json | jq '.total.lines.pct')
          echo "Coverage: $COVERAGE%"
          
          if (( $(echo "$COVERAGE < 80" | bc -l) )); then
            echo "::error::Coverage $COVERAGE% is below 80% threshold"
            exit 1
          fi
          echo "✓ Coverage threshold met"

  # Job 3: Build verification
  build:
    name: Build Application
    needs: [lint, test]
    runs-on: ubuntu-latest
    timeout-minutes: 10
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Build application
        run: npm run build
      
      - name: Check build size
        run: |
          SIZE=$(du -sh dist/ | cut -f1)
          echo "Build size: $SIZE"
          echo "::notice::Build completed successfully (size: $SIZE)"
      
      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: build-artifacts
          path: dist/
          retention-days: 7

  # Job 4: Security scanning
  security:
    name: Security Scan
    runs-on: ubuntu-latest
    timeout-minutes: 15
    permissions:
      security-events: write
      contents: read
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'
      
      - name: Run npm audit
        run: npm audit --audit-level=moderate
        continue-on-error: false

  # Job 5: Bug fix verification (MANDATORY)
  verify-bug-fixes:
    name: Verify Bug Fixes Have Tests
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    timeout-minutes: 5
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Check if PR fixes a bug
        id: check-bug
        run: |
          # Check PR title and body for bug references
          PR_TITLE="${{ github.event.pull_request.title }}"
          PR_BODY="${{ github.event.pull_request.body }}"
          
          if echo "$PR_TITLE" | grep -iE "fix|bug|issue #[0-9]+"; then
            echo "is_bug_fix=true" >> $GITHUB_OUTPUT
            echo "Bug fix detected in PR"
          else
            echo "is_bug_fix=false" >> $GITHUB_OUTPUT
          fi
      
      - name: Verify regression test exists
        if: steps.check-bug.outputs.is_bug_fix == 'true'
        run: |
          # Extract issue number
          ISSUE_NUM=$(echo "${{ github.event.pull_request.title }}" | grep -oE "#[0-9]+" | head -1 | sed 's/#//')
          
          if [ -z "$ISSUE_NUM" ]; then
            echo "::error::Bug fix PR must reference issue number (#123)"
            exit 1
          fi
          
          # Check if tests reference the issue
          if ! grep -r "issue.*#$ISSUE_NUM\|bug.*#$ISSUE_NUM" tests/; then
            echo "::error::Bug fix for issue #$ISSUE_NUM missing regression test"
            echo "::error::Add a test with comment: // Bug #$ISSUE_NUM"
            exit 1
          fi
          
          echo "✓ Regression test found for issue #$ISSUE_NUM"

  # Job 6: Integration status
  ci-status:
    name: CI Status Check
    needs: [lint, test, build, security, verify-bug-fixes]
    if: always()
    runs-on: ubuntu-latest
    
    steps:
      - name: Check all jobs passed
        run: |
          if [[ "${{ needs.lint.result }}" != "success" ]] || \
             [[ "${{ needs.test.result }}" != "success" ]] || \
             [[ "${{ needs.build.result }}" != "success" ]] || \
             [[ "${{ needs.security.result }}" != "success" ]]; then
            echo "::error::One or more CI jobs failed"
            exit 1
          fi
          echo "✓ All CI checks passed"
```

### B. CD Workflow for Deployment

```yaml
# .github/workflows/cd.yml - Continuous deployment

name: CD

on:
  push:
    branches:
      - main
    tags:
      - 'v*'
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy to'
        required: true
        type: choice
        options:
          - development
          - staging
          - production

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # Job 1: Build and push Docker image
  build-image:
    name: Build Docker Image
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
      image-digest: ${{ steps.build.outputs.digest }}
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix={{branch}}-
            type=raw,value=latest,enable={{is_default_branch}}
      
      - name: Build and push Docker image
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          build-args: |
            BUILD_DATE=${{ github.event.head_commit.timestamp }}
            VCS_REF=${{ github.sha }}
            VERSION=${{ github.ref_name }}
      
      - name: Run Trivy vulnerability scanner on image
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-image-results.sarif'
      
      - name: Upload Trivy results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-image-results.sarif'

  # Job 2: Deploy to development
  deploy-dev:
    name: Deploy to Development
    needs: build-image
    if: github.ref == 'refs/heads/develop'
    runs-on: ubuntu-latest
    environment:
      name: development
      url: https://dev.example.com
    
    steps:
      - name: Deploy to development
        run: |
          echo "Deploying ${{ needs.build-image.outputs.image-tag }} to development"
          # Add deployment commands here
      
      - name: Run smoke tests
        run: |
          # Wait for deployment
          sleep 30
          
          # Health check
          RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" https://dev.example.com/health)
          if [ "$RESPONSE" != "200" ]; then
            echo "::error::Health check failed with status $RESPONSE"
            exit 1
          fi
          echo "✓ Smoke tests passed"
      
      - name: Update deployment status
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.repos.createDeploymentStatus({
              owner: context.repo.owner,
              repo: context.repo.repo,
              deployment_id: context.payload.deployment.id,
              state: 'success',
              environment_url: 'https://dev.example.com',
              description: 'Deployment successful'
            });

  # Job 3: Deploy to staging (requires approval)
  deploy-staging:
    name: Deploy to Staging
    needs: build-image
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment:
      name: staging
      url: https://staging.example.com
    
    steps:
      - name: Deploy to staging
        run: |
          echo "Deploying ${{ needs.build-image.outputs.image-tag }} to staging"
          # Add deployment commands here
      
      - name: Run integration tests
        run: |
          echo "Running integration tests..."
          # Add test commands here

  # Job 4: Deploy to production (requires approval)
  deploy-prod:
    name: Deploy to Production
    needs: [build-image, deploy-staging]
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://example.com
    
    steps:
      - name: Deploy to production
        run: |
          echo "Deploying ${{ needs.build-image.outputs.image-tag }} to production"
          # Add deployment commands here
      
      - name: Create GitHub Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref_name }}
          release_name: Release ${{ github.ref_name }}
          draft: false
          prerelease: false
      
      - name: Notify deployment
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: '🚀 Deployed to production: ${{ github.ref_name }}'
            });
```

### C. Reusable Workflow for TDD Verification

```yaml
# .github/workflows/tdd-verification.yml - Reusable TDD workflow

name: TDD Verification

on:
  workflow_call:
    inputs:
      language:
        required: true
        type: string
      test-command:
        required: true
        type: string
      coverage-threshold:
        required: false
        type: number
        default: 80

jobs:
  verify-tdd:
    name: Verify TDD Compliance
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Count source files
        id: count-source
        run: |
          case "${{ inputs.language }}" in
            typescript|javascript)
              SRC_COUNT=$(find src/ -name "*.ts" -o -name "*.js" | wc -l)
              TEST_COUNT=$(find tests/ -name "*.test.ts" -o -name "*.test.js" | wc -l)
              ;;
            python)
              SRC_COUNT=$(find src/ -name "*.py" | wc -l)
              TEST_COUNT=$(find tests/ -name "test_*.py" | wc -l)
              ;;
            go)
              SRC_COUNT=$(find . -name "*.go" -not -path "*/vendor/*" -not -name "*_test.go" | wc -l)
              TEST_COUNT=$(find . -name "*_test.go" | wc -l)
              ;;
            *)
              echo "Unsupported language: ${{ inputs.language }}"
              exit 1
              ;;
          esac
          
          echo "source-count=$SRC_COUNT" >> $GITHUB_OUTPUT
          echo "test-count=$TEST_COUNT" >> $GITHUB_OUTPUT
          
          echo "Source files: $SRC_COUNT"
          echo "Test files: $TEST_COUNT"
      
      - name: Verify test coverage ratio
        run: |
          SRC_COUNT=${{ steps.count-source.outputs.source-count }}
          TEST_COUNT=${{ steps.count-source.outputs.test-count }}
          
          if [ $TEST_COUNT -eq 0 ]; then
            echo "::error::No test files found - TDD violation"
            exit 1
          fi
          
          RATIO=$(echo "scale=2; $TEST_COUNT / $SRC_COUNT" | bc)
          echo "Test to source ratio: $RATIO"
          
          if (( $(echo "$RATIO < 0.5" | bc -l) )); then
            echo "::warning::Low test coverage ratio: $RATIO (recommended: > 0.5)"
          fi
      
      - name: Run tests
        run: ${{ inputs.test-command }}
      
      - name: Verify no skipped tests
        run: |
          # Check for skipped/ignored tests
          if grep -r "@skip\|@ignore\|@disabled" tests/; then
            echo "::error::Found skipped tests - all tests must run"
            exit 1
          fi
          echo "✓ No skipped tests"
      
      - name: Check coverage threshold
        run: |
          # This step should parse coverage output and verify threshold
          echo "Verifying coverage >= ${{ inputs.coverage-threshold }}%"
          # Add language-specific coverage check here
```

---

## 2. GitHub Security (MANDATORY)

### A. CodeQL Security Scanning

```yaml
# .github/workflows/codeql.yml - Code security analysis

name: "CodeQL Security Scan"

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday at 6 AM UTC

jobs:
  analyze:
    name: Analyze Code
    runs-on: ubuntu-latest
    timeout-minutes: 30
    permissions:
      actions: read
      contents: read
      security-events: write
    
    strategy:
      fail-fast: false
      matrix:
        language: [ 'javascript', 'python', 'go' ]
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: ${{ matrix.language }}
          queries: +security-extended,security-and-quality
      
      - name: Autobuild
        uses: github/codeql-action/autobuild@v3
      
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3
        with:
          category: "/language:${{ matrix.language }}"
```

### B. Dependabot Configuration

```yaml
# .github/dependabot.yml - Automated dependency updates

version: 2
updates:
  # Enable version updates for npm
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "06:00"
    open-pull-requests-limit: 10
    reviewers:
      - "team-backend"
    assignees:
      - "devops-lead"
    commit-message:
      prefix: "chore(deps)"
      include: "scope"
    labels:
      - "dependencies"
      - "automated"
    ignore:
      # Ignore major version updates for stable packages
      - dependency-name: "react"
        update-types: ["version-update:semver-major"]
    groups:
      # Group development dependencies
      dev-dependencies:
        dependency-type: "development"
        update-types:
          - "minor"
          - "patch"

  # Enable version updates for Docker
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    commit-message:
      prefix: "chore(docker)"
    labels:
      - "docker"
      - "dependencies"

  # Enable version updates for GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    commit-message:
      prefix: "chore(actions)"
    labels:
      - "github-actions"
      - "dependencies"

  # Enable version updates for Terraform
  - package-ecosystem: "terraform"
    directory: "/infrastructure"
    schedule:
      interval: "weekly"
    commit-message:
      prefix: "chore(terraform)"
    labels:
      - "terraform"
      - "infrastructure"
```

### C. Secret Scanning Configuration

```yaml
# .github/workflows/secret-scan.yml - Secret scanning

name: Secret Scanning

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  gitleaks:
    name: Scan for Secrets
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Run Gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}
      
      - name: Check for .env files
        run: |
          if find . -name ".env*" -not -name ".env.example" | grep -q .; then
            echo "::error::Found .env files in repository"
            find . -name ".env*" -not -name ".env.example"
            exit 1
          fi
          echo "✓ No .env files found"
      
      - name: Verify .gitignore
        run: |
          REQUIRED_IGNORES=(".env" ".env.local" "*.pem" "*.key" "secrets.yml")
          
          for pattern in "${REQUIRED_IGNORES[@]}"; do
            if ! grep -q "^$pattern$" .gitignore; then
              echo "::warning::.gitignore missing pattern: $pattern"
            fi
          done
```

### D. Security Policy

```markdown
# .github/SECURITY.md

# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x     | :white_check_mark: |
| 1.x     | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**DO NOT** open a public issue for security vulnerabilities.

Instead, please report security vulnerabilities to:
- Email: security@example.com
- GitHub Security Advisory: https://github.com/org/repo/security/advisories/new

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if available)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: 24-48 hours
  - High: 7 days
  - Medium: 30 days
  - Low: 90 days

## Security Best Practices

### For Contributors

1. Never commit secrets, API keys, or credentials
2. Use environment variables for sensitive data
3. Run `npm audit` before submitting PRs
4. Keep dependencies up to date
5. Follow secure coding practices
6. Enable 2FA on your GitHub account

### For Maintainers

1. Enable Dependabot security updates
2. Enable CodeQL scanning
3. Review Dependabot PRs promptly
4. Use branch protection rules
5. Require signed commits for releases
6. Regularly audit dependencies
7. Keep GitHub Actions up to date

## Disclosure Policy

After a vulnerability is fixed:
1. We will publish a security advisory
2. We will credit the reporter (unless they prefer anonymity)
3. We will document the fix in the changelog
4. We will release a patch version
```

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new GitHub Actions workflows and automation code.**

### TDD Cycle Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    TDD CYCLE FOR GITHUB ACTIONS                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌───────────┐                                                │
│    │   RED     │  Write a failing workflow test first           │
│    │  (FAIL)   │  - Define expected behavior                    │
│    └─────┬─────┘  - Create test workflow that validates         │
│          │                                                       │
│          ▼                                                       │
│    ┌───────────┐                                                │
│    │  GREEN    │  Write minimal workflow to pass                │
│    │  (PASS)   │  - Implement just enough to satisfy tests      │
│    └─────┬─────┘  - Avoid over-engineering                      │
│          │                                                       │
│          ▼                                                       │
│    ┌───────────┐                                                │
│    │ REFACTOR  │  Improve while keeping tests green             │
│    │ (IMPROVE) │  - Optimize workflow steps                     │
│    └─────┬─────┘  - Remove duplication                          │
│          │        - Improve caching                              │
│          │                                                       │
│          └──────────────── Repeat ──────────────────────────────│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Example TDD Workflow for GitHub Actions

**Scenario**: Creating a workflow to validate Terraform configurations

```yaml
# Step 1: RED - Write failing test workflow first
# .github/workflows/test-terraform-validation.yml

name: Test Terraform Validation Workflow

on:
  push:
    paths:
      - '.github/workflows/terraform-*.yml'
      - 'test-fixtures/terraform/**'

jobs:
  test-workflow:
    name: Test Terraform Workflow
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      # Test 1: Verify workflow validates valid Terraform
      - name: Test valid Terraform passes
        run: |
          cd test-fixtures/terraform/valid
          # This will FAIL until we create the workflow
          act -j validate-terraform --dryrun
          if [ $? -ne 0 ]; then
            echo "::error::Valid Terraform should pass validation"
            exit 1
          fi
          echo "✓ Valid Terraform test passed"

      # Test 2: Verify workflow catches invalid Terraform
      - name: Test invalid Terraform fails
        run: |
          cd test-fixtures/terraform/invalid
          # Should return non-zero exit code
          act -j validate-terraform --dryrun 2>&1 || EXIT_CODE=$?
          if [ ${EXIT_CODE:-0} -eq 0 ]; then
            echo "::error::Invalid Terraform should fail validation"
            exit 1
          fi
          echo "✓ Invalid Terraform correctly fails"

# Run: Test workflow FAILS - terraform validation workflow doesn't exist yet
```

```yaml
# Step 2: GREEN - Write minimal workflow to make tests pass
# .github/workflows/terraform-validate.yml

name: Terraform Validation

on:
  push:
    paths:
      - '**/*.tf'
      - '**/*.tfvars'
  pull_request:
    paths:
      - '**/*.tf'
      - '**/*.tfvars'

jobs:
  validate-terraform:
    name: Validate Terraform
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: 1.6.0

      - name: Terraform Init
        run: terraform init -backend=false

      - name: Terraform Validate
        run: terraform validate

      - name: Terraform Format Check
        run: terraform fmt -check -recursive

# Run: Tests now PASS - workflow validates Terraform correctly
```

```yaml
# Step 3: REFACTOR - Improve workflow while keeping tests green
# .github/workflows/terraform-validate.yml (improved)

name: Terraform Validation

on:
  push:
    paths:
      - '**/*.tf'
      - '**/*.tfvars'
  pull_request:
    paths:
      - '**/*.tf'
      - '**/*.tfvars'

concurrency:
  group: terraform-${{ github.ref }}
  cancel-in-progress: true

jobs:
  validate-terraform:
    name: Validate Terraform
    runs-on: ubuntu-latest
    timeout-minutes: 10

    strategy:
      matrix:
        directory: [infrastructure, modules/vpc, modules/eks]
      fail-fast: false

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: 1.6.0

      # Improved: Cache Terraform plugins
      - name: Cache Terraform plugins
        uses: actions/cache@v4
        with:
          path: ~/.terraform.d/plugin-cache
          key: terraform-${{ runner.os }}-${{ hashFiles('**/.terraform.lock.hcl') }}
          restore-keys: terraform-${{ runner.os }}-

      - name: Terraform Init
        working-directory: ${{ matrix.directory }}
        run: terraform init -backend=false
        env:
          TF_PLUGIN_CACHE_DIR: ~/.terraform.d/plugin-cache

      - name: Terraform Validate
        working-directory: ${{ matrix.directory }}
        run: terraform validate

      - name: Terraform Format Check
        working-directory: ${{ matrix.directory }}
        run: terraform fmt -check -recursive

      # Improved: Add tflint for additional validation
      - name: Setup TFLint
        uses: terraform-linters/setup-tflint@v4

      - name: Run TFLint
        working-directory: ${{ matrix.directory }}
        run: tflint --format=compact

# Tests still PASS - workflow is now more robust with caching and matrix builds
```

### Visual Step-by-Step TDD Example

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: RED - Write Failing Test                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  📝 Create test-deploy-workflow.yml:                                    │
│                                                                          │
│  ┌────────────────────────────────────────┐                             │
│  │ jobs:                                   │                             │
│  │   test-deployment:                      │                             │
│  │     steps:                              │                             │
│  │       - name: Verify deploy runs        │                             │
│  │         run: |                          │                             │
│  │           # Assert deployment workflow  │                             │
│  │           # exists and functions        │   ❌ FAILS                  │
│  │           gh workflow run deploy.yml    │   (workflow missing)        │
│  └────────────────────────────────────────┘                             │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ STEP 2: GREEN - Write Minimal Implementation                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  📝 Create deploy.yml:                                                  │
│                                                                          │
│  ┌────────────────────────────────────────┐                             │
│  │ jobs:                                   │                             │
│  │   deploy:                               │                             │
│  │     steps:                              │                             │
│  │       - uses: actions/checkout@v4       │                             │
│  │       - run: echo "Deploying..."        │   ✅ PASSES                 │
│  └────────────────────────────────────────┘   (test satisfied)          │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ STEP 3: REFACTOR - Improve While Tests Pass                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  📝 Enhance deploy.yml:                                                 │
│                                                                          │
│  ┌────────────────────────────────────────┐                             │
│  │ jobs:                                   │                             │
│  │   deploy:                               │                             │
│  │     environment: production             │                             │
│  │     concurrency: deploy-${{ ref }}      │                             │
│  │     steps:                              │                             │
│  │       - uses: actions/checkout@v4       │                             │
│  │       - name: Build                     │                             │
│  │         run: npm run build              │                             │
│  │       - name: Deploy                    │   ✅ STILL PASSES           │
│  │         run: ./deploy.sh                │   (tests green)             │
│  │       - name: Notify                    │                             │
│  │         run: ./notify.sh                │                             │
│  └────────────────────────────────────────┘                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### TDD Testing Strategies for GitHub Actions

```yaml
# .github/workflows/workflow-tests.yml - Meta-testing for workflows

name: Workflow Tests

on:
  push:
    paths:
      - '.github/workflows/**'
  pull_request:
    paths:
      - '.github/workflows/**'

jobs:
  # Test 1: Syntax validation
  syntax-validation:
    name: Validate Workflow Syntax
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Validate all workflows
        run: |
          for workflow in .github/workflows/*.yml; do
            echo "Validating: $workflow"
            # Use actionlint to validate syntax
            actionlint "$workflow"
            if [ $? -ne 0 ]; then
              echo "::error::Invalid workflow syntax in $workflow"
              exit 1
            fi
          done
          echo "✓ All workflows have valid syntax"

  # Test 2: Security checks
  security-checks:
    name: Workflow Security Checks
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Check for pinned action versions
        run: |
          # Ensure all actions use SHA or version tags
          if grep -rE "uses: [^@]+@(main|master)" .github/workflows/; then
            echo "::error::Found actions using main/master branch"
            echo "All actions must be pinned to a version or SHA"
            exit 1
          fi
          echo "✓ All actions are properly pinned"

      - name: Check for secret exposure risks
        run: |
          # Ensure secrets aren't echoed
          if grep -rE "echo.*\\\$\{\{ secrets\." .github/workflows/; then
            echo "::error::Found potential secret exposure via echo"
            exit 1
          fi
          echo "✓ No secret exposure risks found"

  # Test 3: Best practices
  best-practices:
    name: Workflow Best Practices
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Check for timeout settings
        run: |
          for workflow in .github/workflows/*.yml; do
            if ! grep -q "timeout-minutes:" "$workflow"; then
              echo "::warning::$workflow missing timeout-minutes"
            fi
          done
          echo "✓ Timeout check complete"

      - name: Check for concurrency settings
        run: |
          for workflow in .github/workflows/*.yml; do
            if ! grep -q "concurrency:" "$workflow"; then
              echo "::warning::$workflow missing concurrency settings"
            fi
          done
          echo "✓ Concurrency check complete"
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug in GitHub Actions workflows MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  BUG FIX WORKFLOW FOR GITHUB                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  1. BUG REPORTED                                            │ │
│  │     - Issue #123: "Deploy workflow fails on ARM runners"    │ │
│  └─────────────────────────────┬──────────────────────────────┘ │
│                                │                                 │
│                                ▼                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  2. WRITE REGRESSION TEST                                   │ │
│  │     - Create test that reproduces the bug                   │ │
│  │     - Test MUST fail before fix                             │ │
│  │     - Reference issue ID in test comments                   │ │
│  └─────────────────────────────┬──────────────────────────────┘ │
│                                │                                 │
│                                ▼                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  3. VERIFY TEST FAILS                                       │ │
│  │     ❌ Test fails for the expected reason                   │ │
│  │     - Confirms we're testing the right thing                │ │
│  └─────────────────────────────┬──────────────────────────────┘ │
│                                │                                 │
│                                ▼                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  4. FIX THE BUG                                             │ │
│  │     - Apply minimal fix to resolve the issue                │ │
│  │     - Don't add unrelated changes                           │ │
│  └─────────────────────────────┬──────────────────────────────┘ │
│                                │                                 │
│                                ▼                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  5. VERIFY TEST PASSES                                      │ │
│  │     ✅ Test now passes                                      │ │
│  │     - Bug is fixed                                          │ │
│  │     - Regression prevented forever                          │ │
│  └─────────────────────────────┬──────────────────────────────┘ │
│                                │                                 │
│                                ▼                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  6. DOCUMENT & DEPLOY                                       │ │
│  │     - PR references issue: "Fixes #123"                     │ │
│  │     - Commit message follows format                         │ │
│  │     - Deploy with confidence                                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Example Bug Fix: Workflow Fails on Matrix Builds

```yaml
# Bug Report #456: Matrix build fails when node version contains 'x'
# Example: node-version: '20.x' causes setup-node to fail

# ─────────────────────────────────────────────────────────────────
# Step 1-2: Write test that reproduces the bug
# ─────────────────────────────────────────────────────────────────

# .github/workflows/test-regression-456.yml
name: Regression Test - Bug #456

on:
  push:
    paths:
      - '.github/workflows/ci.yml'

jobs:
  # Bug #456: Verify matrix builds work with 'x' notation
  test-node-version-notation:
    name: "Test Node Version Notation (Bug #456)"
    runs-on: ubuntu-latest

    strategy:
      matrix:
        # These notations caused failures before fix
        node-version: ['18.x', '20.x', '22.x']

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      # This step would FAIL before the fix
      - name: Setup Node.js ${{ matrix.node-version }}
        id: setup-node
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}

      - name: Verify Node.js version
        run: |
          # Bug #456: Version check would fail on 'x' notation
          NODE_VERSION=$(node --version)
          echo "Installed Node.js version: $NODE_VERSION"

          # Extract major version from matrix input (e.g., '20.x' -> '20')
          EXPECTED_MAJOR=$(echo "${{ matrix.node-version }}" | cut -d'.' -f1)
          ACTUAL_MAJOR=$(echo "$NODE_VERSION" | sed 's/v//' | cut -d'.' -f1)

          if [ "$EXPECTED_MAJOR" != "$ACTUAL_MAJOR" ]; then
            echo "::error::Expected Node.js $EXPECTED_MAJOR.x but got $NODE_VERSION"
            exit 1
          fi

          echo "✓ Node.js version matches expected major version $EXPECTED_MAJOR"

# Run: gh workflow run test-regression-456.yml
# Result: ❌ FAILS on matrix builds (confirms bug exists)
```

```yaml
# ─────────────────────────────────────────────────────────────────
# Step 3-4: Fix the bug in the main CI workflow
# ─────────────────────────────────────────────────────────────────

# .github/workflows/ci.yml (FIXED)
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    name: Test
    runs-on: ubuntu-latest

    strategy:
      matrix:
        # Bug #456 Fix: Use explicit version strings instead of 'x' notation
        # when actions/setup-node has issues with wildcard versions
        node-version: ['18', '20', '22']  # Changed from '18.x', '20.x', '22.x'
      fail-fast: false

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          # Bug #456 Fix: Explicitly check for version file first
          node-version-file: '.nvmrc'
          check-latest: true  # Ensure we get the latest patch version

      - name: Verify Node.js setup
        run: |
          echo "Node.js version: $(node --version)"
          echo "npm version: $(npm --version)"

      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test

# Run: gh workflow run ci.yml
# Result: ✅ PASSES - Bug #456 is fixed
```

### Example Bug Fix: Secret Exposure in Logs

```yaml
# Bug Report #789: Secrets accidentally exposed in workflow logs
# Environment variables containing secrets were being echoed

# ─────────────────────────────────────────────────────────────────
# Step 1-2: Write regression test
# ─────────────────────────────────────────────────────────────────

# .github/workflows/test-regression-789.yml
name: Regression Test - Bug #789 (Secret Exposure)

on:
  push:
    paths:
      - '.github/workflows/deploy.yml'

jobs:
  test-secret-safety:
    name: "Verify Secrets Not Exposed (Bug #789)"
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      # Bug #789: Check that deploy workflow doesn't echo secrets
      - name: Scan for secret exposure patterns
        run: |
          DEPLOY_WORKFLOW=".github/workflows/deploy.yml"

          # Pattern 1: Direct echo of secrets
          if grep -E "echo.*\\\$\{\{ secrets\." "$DEPLOY_WORKFLOW"; then
            echo "::error::Bug #789: Found direct echo of secrets"
            exit 1
          fi

          # Pattern 2: Secrets in run commands that might log
          if grep -E "printenv|env \|" "$DEPLOY_WORKFLOW" | grep -v "::add-mask::"; then
            echo "::error::Bug #789: Found env dump without masking"
            exit 1
          fi

          # Pattern 3: Secrets passed to commands that log verbosely
          if grep -E "curl.*-v.*\\\$\{\{ secrets\." "$DEPLOY_WORKFLOW"; then
            echo "::error::Bug #789: Found verbose curl with secrets"
            exit 1
          fi

          echo "✓ No secret exposure patterns found (Bug #789 prevented)"

      - name: Verify secret masking is used
        run: |
          DEPLOY_WORKFLOW=".github/workflows/deploy.yml"

          # Check that add-mask is used for dynamic secrets
          if grep -E "\\\$\{\{ steps\.[^}]+\.outputs\." "$DEPLOY_WORKFLOW" | grep -v "::add-mask::"; then
            echo "::warning::Consider masking sensitive step outputs"
          fi

          echo "✓ Secret masking check complete"

# Run: This would FAIL on the old workflow that had secret exposure
```

```yaml
# ─────────────────────────────────────────────────────────────────
# Step 3-4: Fix the deployment workflow
# ─────────────────────────────────────────────────────────────────

# .github/workflows/deploy.yml (FIXED - Bug #789)
name: Deploy

on:
  push:
    tags:
      - 'v*'

jobs:
  deploy:
    name: Deploy to Production
    runs-on: ubuntu-latest
    environment: production

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      # Bug #789 Fix: Mask all sensitive values
      - name: Configure deployment
        run: |
          # FIXED: Use add-mask to prevent logging
          echo "::add-mask::${{ secrets.DEPLOY_TOKEN }}"
          echo "::add-mask::${{ secrets.AWS_ACCESS_KEY_ID }}"
          echo "::add-mask::${{ secrets.AWS_SECRET_ACCESS_KEY }}"

      # Bug #789 Fix: Don't echo environment for debugging
      - name: Deploy application
        env:
          DEPLOY_TOKEN: ${{ secrets.DEPLOY_TOKEN }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          # FIXED: Removed "echo $DEPLOY_TOKEN" debugging line
          # FIXED: Using --silent instead of --verbose for curl

          curl --silent --fail \
            -H "Authorization: Bearer $DEPLOY_TOKEN" \
            -X POST \
            https://api.example.com/deploy

          echo "✓ Deployment triggered successfully"

      # Bug #789 Fix: Safe status check without exposing secrets
      - name: Verify deployment
        run: |
          # FIXED: Don't log the full response which might contain secrets
          RESPONSE=$(curl --silent --fail \
            -H "Authorization: Bearer ${{ secrets.DEPLOY_TOKEN }}" \
            https://api.example.com/status)

          # Only log safe portions
          STATUS=$(echo "$RESPONSE" | jq -r '.status')
          echo "Deployment status: $STATUS"

          if [ "$STATUS" != "success" ]; then
            echo "::error::Deployment failed"
            exit 1
          fi

# Result: ✅ Regression test passes - secrets are properly protected
```

### Bug Fix Verification Checklist

```yaml
# .github/workflows/verify-bug-fixes.yml
name: Verify Bug Fixes

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  verify-regression-tests:
    name: Verify Bug Fix Has Regression Test
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Check if PR is a bug fix
        id: check-bug-fix
        run: |
          PR_TITLE="${{ github.event.pull_request.title }}"
          PR_BODY="${{ github.event.pull_request.body }}"

          # Check for bug fix indicators
          if echo "$PR_TITLE" | grep -iE "(fix|bug|issue|closes|fixes|resolves) #[0-9]+"; then
            echo "is_bug_fix=true" >> $GITHUB_OUTPUT

            # Extract issue number
            ISSUE_NUM=$(echo "$PR_TITLE" | grep -oE "#[0-9]+" | head -1 | tr -d '#')
            echo "issue_number=$ISSUE_NUM" >> $GITHUB_OUTPUT
            echo "Found bug fix for issue #$ISSUE_NUM"
          else
            echo "is_bug_fix=false" >> $GITHUB_OUTPUT
          fi

      - name: Verify regression test exists
        if: steps.check-bug-fix.outputs.is_bug_fix == 'true'
        run: |
          ISSUE_NUM="${{ steps.check-bug-fix.outputs.issue_number }}"

          # Check for test file referencing the bug
          if grep -rE "(Bug|Issue|Regression).*#?$ISSUE_NUM" tests/ .github/workflows/test-*.yml; then
            echo "✓ Found regression test for bug #$ISSUE_NUM"
          else
            echo "::error::Bug fix for #$ISSUE_NUM is missing a regression test"
            echo ""
            echo "REQUIRED: Add a test that:"
            echo "  1. Reproduces the bug (would fail before fix)"
            echo "  2. References the issue number in comments"
            echo "  3. Passes after the fix is applied"
            echo ""
            echo "Example test comment:"
            echo "  // Bug #$ISSUE_NUM: Description of the bug"
            echo "  // This test prevents regression"
            exit 1
          fi

      - name: Verify test was added in this PR
        if: steps.check-bug-fix.outputs.is_bug_fix == 'true'
        run: |
          ISSUE_NUM="${{ steps.check-bug-fix.outputs.issue_number }}"

          # Check that the regression test was added in this PR
          git diff origin/main...HEAD --name-only | grep -E "test.*\.yml$|test.*\.(ts|js|py)$" || {
            echo "::warning::No test files were modified in this PR"
            echo "Ensure the regression test is included in this PR, not a separate one"
          }
```

---

## 3. Issue and Bug Tracking (MANDATORY)

### A. Issue Templates

```yaml
# .github/ISSUE_TEMPLATE/config.yml

blank_issues_enabled: false
contact_links:
  - name: Question or Discussion
    url: https://github.com/org/repo/discussions
    about: Ask questions or start discussions here
  - name: Security Vulnerability
    url: https://github.com/org/repo/security/advisories/new
    about: Report security vulnerabilities privately
```

```yaml
# .github/ISSUE_TEMPLATE/bug_report.yml

name: 🐛 Bug Report
description: Report a bug to help us improve
title: "[Bug]: "
labels: ["bug", "needs-triage"]
assignees: []

body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to report this bug!
        
        **IMPORTANT**: Every bug fix MUST include a regression test.

  - type: input
    id: version
    attributes:
      label: Version
      description: What version are you using?
      placeholder: e.g., 2.1.0
    validations:
      required: true

  - type: textarea
    id: description
    attributes:
      label: Bug Description
      description: A clear and concise description of the bug
      placeholder: When I do X, Y happens instead of Z
    validations:
      required: true

  - type: textarea
    id: reproduction
    attributes:
      label: Steps to Reproduce
      description: Detailed steps to reproduce the behavior
      placeholder: |
        1. Go to '...'
        2. Click on '...'
        3. Scroll down to '...'
        4. See error
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: Expected Behavior
      description: What should happen?
      placeholder: I expected Z to happen
    validations:
      required: true

  - type: textarea
    id: actual
    attributes:
      label: Actual Behavior
      description: What actually happened?
      placeholder: Y happened instead
    validations:
      required: true

  - type: textarea
    id: environment
    attributes:
      label: Environment
      description: System information
      placeholder: |
        - OS: [e.g., Ubuntu 22.04]
        - Browser: [e.g., Chrome 120]
        - Node.js: [e.g., 20.10.0]
    validations:
      required: true

  - type: textarea
    id: logs
    attributes:
      label: Relevant Log Output
      description: Copy and paste any relevant log output
      render: shell

  - type: checkboxes
    id: checklist
    attributes:
      label: Pre-submission Checklist
      description: Please verify the following
      options:
        - label: I have searched existing issues to ensure this is not a duplicate
          required: true
        - label: I have provided a clear description and reproduction steps
          required: true
        - label: I have included version and environment information
          required: true
        - label: I understand a regression test will be required for the fix
          required: true
```

```yaml
# .github/ISSUE_TEMPLATE/feature_request.yml

name: ✨ Feature Request
description: Suggest a new feature or enhancement
title: "[Feature]: "
labels: ["enhancement", "needs-triage"]
assignees: []

body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting a new feature!
        
        **IMPORTANT**: All new features MUST include tests (TDD).

  - type: textarea
    id: problem
    attributes:
      label: Problem Description
      description: What problem does this feature solve?
      placeholder: I'm frustrated when..
    validations:
      required: true

  - type: textarea
    id: solution
    attributes:
      label: Proposed Solution
      description: How would you like this to work?
      placeholder: I would like to be able to..
    validations:
      required: true

  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives Considered
      description: What other solutions have you considered?
      placeholder: I also thought about..

  - type: dropdown
    id: architecture-layer
    attributes:
      label: Architecture Layer
      description: Which layer does this affect?
      options:
        - Domain (core business logic)
        - Application (use cases)
        - Infrastructure (external dependencies)
        - Adapter (API/UI)
        - Multiple layers
        - Not sure
    validations:
      required: true

  - type: checkboxes
    id: requirements
    attributes:
      label: Requirements
      description: What will be needed for this feature?
      options:
        - label: Unit tests (TDD)
        - label: Integration tests
        - label: Documentation update
        - label: API changes
        - label: Database migration
        - label: Breaking changes

  - type: checkboxes
    id: checklist
    attributes:
      label: Pre-submission Checklist
      options:
        - label: I have searched existing issues/PRs
          required: true
        - label: I have provided a clear use case
          required: true
        - label: I understand tests will be required (TDD)
          required: true
```

### B. Pull Request Template

```markdown
# .github/pull_request_template.md

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

## Architecture Layer
- [ ] Domain (core business logic)
- [ ] Application (use cases)
- [ ] Infrastructure (external dependencies)
- [ ] Adapter (API/UI)

## Changes Made
<!-- List the specific changes you made -->
- 
- 
- 

## TDD Compliance (MANDATORY)
- [ ] Tests written BEFORE implementation (Red-Green-Refactor)
- [ ] All tests pass locally
- [ ] Code coverage maintained/increased (>80%)
- [ ] No skipped or disabled tests

**Test Coverage:**
- Before: __%
- After: __%

## Regression Testing (MANDATORY for bug fixes)
**For bug fixes ONLY:**
- [ ] Regression test added that reproduces the bug
- [ ] Test fails before fix, passes after fix
- [ ] Test includes comment referencing issue (e.g., `// Bug #123`)

**Test Location:** `tests/path/to/test.ts:LINE`

## Testing Performed
<!-- Describe the testing you performed -->
- [ ] Unit tests
- [ ] Integration tests
- [ ] E2E tests
- [ ] Manual testing

**Test Commands:**
```bash
npm test
npm run test:coverage
npm run test:e2e
```

## Screenshots
<!-- Add screenshots for UI changes -->

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated (README, API docs)
- [ ] No new warnings or errors
- [ ] All CI checks pass
- [ ] Conventional commit format used
- [ ] Issue/bug reference included

## Breaking Changes
<!-- Describe any breaking changes -->

## Migration Guide
<!-- If breaking changes, provide migration steps -->

## Additional Notes
<!-- Any additional context or notes -->

---

**Reviewers:** Please verify:
- [ ] Tests follow TDD principles
- [ ] Bug fixes include regression tests
- [ ] Code coverage threshold met
- [ ] Architecture principles followed
- [ ] Documentation is complete
- [ ] No security issues introduced
```

### C. Automated Issue Management

```yaml
# .github/workflows/issue-management.yml

name: Issue Management

on:
  issues:
    types: [opened, labeled, assigned]
  issue_comment:
    types: [created]
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight

jobs:
  # Auto-label issues based on content
  auto-label:
    name: Auto Label Issues
    if: github.event_name == 'issues' && github.event.action == 'opened'
    runs-on: ubuntu-latest
    permissions:
      issues: write
    
    steps:
      - name: Label bug reports
        if: contains(github.event.issue.title, '[Bug]')
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.addLabels({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              labels: ['bug', 'needs-triage']
            });
      
      - name: Label feature requests
        if: contains(github.event.issue.title, '[Feature]')
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.addLabels({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              labels: ['enhancement', 'needs-triage']
            });
      
      - name: Detect security issues
        uses: actions/github-script@v7
        with:
          script: |
            const body = context.payload.issue.body.toLowerCase();
            const securityKeywords = ['security', 'vulnerability', 'exploit', 'xss', 'sql injection'];
            
            if (securityKeywords.some(keyword => body.includes(keyword))) {
              await github.rest.issues.addLabels({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: context.issue.number,
                labels: ['security']
              });
              
              await github.rest.issues.createComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: context.issue.number,
                body: '⚠️ This issue may contain security-related content. Please do not disclose sensitive details publicly. Use our [security advisory process](../../security/advisories/new) instead.'
              });
            }

  # Close stale issues
  stale-issues:
    name: Close Stale Issues
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    permissions:
      issues: write
      pull-requests: write
    
    steps:
      - name: Close stale issues and PRs
        uses: actions/stale@v9
        with:
          repo-token: ${{ secrets.GITHUB_TOKEN }}
          stale-issue-message: 'This issue has been automatically marked as stale because it has not had recent activity. It will be closed if no further activity occurs. Thank you for your contributions.'
          stale-pr-message: 'This PR has been automatically marked as stale because it has not had recent activity. It will be closed if no further activity occurs.'
          close-issue-message: 'This issue was automatically closed due to inactivity.'
          close-pr-message: 'This PR was automatically closed due to inactivity.'
          days-before-stale: 30
          days-before-close: 7
          stale-issue-label: 'stale'
          stale-pr-label: 'stale'
          exempt-issue-labels: 'pinned,security,bug'
          exempt-pr-labels: 'pinned,security'

  # Require issue link in PRs
  require-issue-link:
    name: Require Issue Link
    if: github.event_name == 'pull_request' && github.event.action == 'opened'
    runs-on: ubuntu-latest
    
    steps:
      - name: Check for issue reference
        uses: actions/github-script@v7
        with:
          script: |
            const pr = context.payload.pull_request;
            const body = pr.body || '';
            const title = pr.title || '';
            
            // Check for issue reference (#123, fixes #123, closes #123)
            const issuePattern = /#\d+|closes #\d+|fixes #\d+|resolves #\d+/i;
            
            if (!issuePattern.test(title) && !issuePattern.test(body)) {
              await github.rest.issues.createComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: context.issue.number,
                body: '⚠️ This PR does not reference an issue. Please link to the related issue using `Closes #123` or include `#123` in the title.\n\n**All PRs must reference an issue for traceability.**'
              });
              
              await github.rest.issues.addLabels({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: context.issue.number,
                labels: ['needs-issue-link']
              });
            }
```

---

## 4. GitHub Container Registry (MANDATORY)

### A. Multi-Stage Dockerfile Optimized for GHCR

```dockerfile
# Dockerfile - Optimized for GitHub Container Registry

# syntax=docker/dockerfile:1.6

# Stage 1: Base image with common dependencies
FROM node:20-alpine AS base
LABEL org.opencontainers.image.source=https://github.com/org/repo
LABEL org.opencontainers.image.description="Application description"
LABEL org.opencontainers.image.licenses=MIT

WORKDIR /app

# Install dumb-init for proper signal handling
RUN apk add --no-cache dumb-init

# Stage 2: Dependencies
FROM base AS dependencies

# Copy package files
COPY package.json package-lock.json ./

# Install production dependencies
RUN --mount=type=cache,target=/root/.npm \
    npm ci --only=production --ignore-scripts

# Install all dependencies for build
RUN --mount=type=cache,target=/root/.npm \
    npm ci --ignore-scripts

# Stage 3: Build
FROM dependencies AS build

# Copy source code
COPY . .

# Build application
RUN npm run build

# Run tests
RUN npm test

# Stage 4: Production image
FROM base AS production

# Set NODE_ENV
ENV NODE_ENV=production

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

# Copy production dependencies
COPY --from=dependencies --chown=nodejs:nodejs /app/node_modules ./node_modules

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

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start application
CMD ["node", "dist/main.js"]
```

### B. Container Build and Push Workflow

```yaml
# .github/workflows/container.yml - Container management

name: Container Build & Push

on:
  push:
    branches:
      - main
      - develop
    tags:
      - 'v*'
  pull_request:
    branches:
      - main

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    name: Build and Push Container
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      security-events: write
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        with:
          driver-opts: |
            image=moby/buildkit:latest
            network=host
      
      - name: Log in to GitHub Container Registry
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=semver,pattern={{major}}
            type=sha,prefix={{branch}}-
            type=raw,value=latest,enable={{is_default_branch}}
          labels: |
            org.opencontainers.image.title=My Application
            org.opencontainers.image.description=Application description
            org.opencontainers.image.vendor=Organization Name
      
      - name: Build and push Docker image
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          build-args: |
            BUILD_DATE=${{ fromJSON(steps.meta.outputs.json).labels['org.opencontainers.image.created'] }}
            VCS_REF=${{ github.sha }}
            VERSION=${{ fromJSON(steps.meta.outputs.json).labels['org.opencontainers.image.version'] }}
          provenance: true
          sbom: true
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ fromJSON(steps.meta.outputs.json).tags[0] }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'
      
      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'
      
      - name: Run Hadolint
        uses: hadolint/hadolint-action@v3.1.0
        with:
          dockerfile: Dockerfile
          format: sarif
          output-file: hadolint-results.sarif
          no-fail: true
      
      - name: Upload Hadolint results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: hadolint-results.sarif
      
      - name: Generate SBOM
        uses: anchore/sbom-action@v0
        with:
          image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ fromJSON(steps.meta.outputs.json).tags[0] }}
          format: spdx-json
          output-file: sbom.spdx.json
      
      - name: Upload SBOM
        uses: actions/upload-artifact@v4
        with:
          name: sbom
          path: sbom.spdx.json
```

---

## 5. Branch Protection & Code Review (MANDATORY)

### A. Branch Protection Rules

**Configure these rules for `main` and `develop` branches:**

```json
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "CI / Lint Code",
      "CI / Test (TDD Verification)",
      "CI / Build Application",
      "CI / Security Scan",
      "CodeQL Security Scan / Analyze Code"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismissal_restrictions": {
      "users": [],
      "teams": ["maintainers"]
    },
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "required_approving_review_count": 2,
    "require_last_push_approval": true
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true,
  "lock_branch": false,
  "allow_fork_syncing": true
}
```

### B. CODEOWNERS File

```
# .github/CODEOWNERS

# Global owners
* @org/core-team

# Domain layer (critical code)
/src/domain/ @org/domain-experts @org/architects

# Application layer
/src/application/ @org/backend-team

# Infrastructure layer
/src/infrastructure/ @org/devops-team @org/backend-team

# API adapters
/src/adapters/api/ @org/api-team

# Tests (require thorough review)
/tests/ @org/qa-team @org/core-team

# CI/CD workflows
/.github/workflows/ @org/devops-team

# Security-sensitive files
/Dockerfile @org/security-team @org/devops-team
/.github/workflows/security.yml @org/security-team
/SECURITY.md @org/security-team

# Documentation
/docs/ @org/docs-team
*.md @org/docs-team

# Configuration
*.config.js @org/devops-team
*.config.ts @org/devops-team
```

### C. Automated Code Review

```yaml
# .github/workflows/code-review.yml

name: Automated Code Review

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  review-size:
    name: Check PR Size
    runs-on: ubuntu-latest
    
    steps:
      - name: Check PR size
        uses: actions/github-script@v7
        with:
          script: |
            const pr = context.payload.pull_request;
            const additions = pr.additions;
            const deletions = pr.deletions;
            const totalChanges = additions + deletions;
            
            let label = null;
            let warning = null;
            
            if (totalChanges < 100) {
              label = 'size/XS';
            } else if (totalChanges < 300) {
              label = 'size/S';
            } else if (totalChanges < 500) {
              label = 'size/M';
            } else if (totalChanges < 1000) {
              label = 'size/L';
              warning = 'This PR is large. Consider splitting it into smaller PRs.';
            } else {
              label = 'size/XL';
              warning = 'This PR is very large. Please split it into multiple smaller PRs for easier review.';
            }
            
            // Add size label
            await github.rest.issues.addLabels({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              labels: [label]
            });
            
            // Add warning comment if needed
            if (warning) {
              await github.rest.issues.createComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: context.issue.number,
                body: `⚠️ **${warning}**\n\nTotal changes: ${totalChanges} lines (+${additions}/-${deletions})`
              });
            }

  check-commit-messages:
    name: Check Commit Messages
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Validate commit messages
        uses: wagoid/commitlint-github-action@v5
        with:
          configFile: .commitlintrc.json

  comment-coverage:
    name: Comment Coverage Report
    runs-on: ubuntu-latest
    needs: [test]
    if: always()
    
    steps:
      - name: Download coverage artifact
        uses: actions/download-artifact@v4
        with:
          name: coverage-report
      
      - name: Comment PR with coverage
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const coverage = JSON.parse(fs.readFileSync('coverage-summary.json', 'utf8'));
            
            const totalCoverage = coverage.total.lines.pct;
            const emoji = totalCoverage >= 80 ? '✅' : '⚠️';
            
            const body = `## ${emoji} Test Coverage Report
            
            | Metric | Coverage |
            |--------|----------|
            | Lines | ${coverage.total.lines.pct}% |
            | Statements | ${coverage.total.statements.pct}% |
            | Functions | ${coverage.total.functions.pct}% |
            | Branches | ${coverage.total.branches.pct}% |
            
            ${totalCoverage >= 80 ? '✅ Coverage threshold met!' : '⚠️ Coverage below 80% threshold'}
            `;
            
            await github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: body
            });
```

---

## 6. Documentation & Community (MANDATORY)

### A. README Template

```markdown
# Project Name

[![CI](https://github.com/org/repo/workflows/CI/badge.svg)](https://github.com/org/repo/actions)
[![Coverage](https://codecov.io/gh/org/repo/branch/main/graph/badge.svg)](https://codecov.io/gh/org/repo)
[![License](https://img.shields.io/github/license/org/repo)](LICENSE)
[![Version](https://img.shields.io/github/v/release/org/repo)](https://github.com/org/repo/releases)

> One-line description of your project

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Architecture](#architecture)
- [License](#license)

## ✨ Features

- **Feature 1**: Description
- **Feature 2**: Description
- **Feature 3**: Description

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/org/repo.git
cd repo

# Install dependencies
npm install

# Copy environment file
cp .env.example .env
```

## 🚀 Usage

```typescript
// Basic usage example
import { MyClass } from './my-class';

const instance = new MyClass();
instance.doSomething();
```

## 🛠 Development

### Prerequisites

- Node.js 20.x or later
- npm 10.x or later
- Docker (for containerized development)

### Development Setup

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Run tests in watch mode
npm run test:watch
```

### Project Structure

```
project-root/
├── src/
│   ├── domain/          # Core business logic
│   ├── application/     # Use cases
│   ├── infrastructure/  # External dependencies
│   └── adapters/        # API/UI adapters
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── docs/
```

## 🧪 Testing

We follow **Test-Driven Development (TDD)**. All code must have tests.

```bash
# Run all tests
npm test

# Run tests with coverage
npm run test:coverage

# Run specific test file
npm test -- path/to/test.ts
```

### Coverage Requirements

- **Minimum coverage**: 80%
- **All new features**: Must include tests
- **All bug fixes**: Must include regression tests

## 🚢 Deployment

### Docker

```bash
# Build image
docker build -t my-app .

# Run container
docker run -p 3000:3000 my-app
```

### GitHub Container Registry

Images are automatically published to `ghcr.io/org/repo` on every release.

```bash
# Pull latest image
docker pull ghcr.io/org/repo:latest
```

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests first (TDD)
4. Implement your feature
5. Ensure all tests pass
6. Commit your changes (follow Conventional Commits)
7. Push to the branch
8. Open a Pull Request

### Commit Message Format

```
type(scope): subject

body (optional)

footer (optional)
```

Examples:
- `feat(api): add user authentication endpoint`
- `fix(database): resolve connection pool leak #123`
- `docs(readme): update installation instructions`

## 🏗 Architecture

This project follows **Hexagonal Architecture** (Ports and Adapters):

- **Domain**: Core business logic, no external dependencies
- **Application**: Use cases and orchestration
- **Infrastructure**: Database, external APIs, messaging
- **Adapters**: REST API, GraphQL, CLI

See [Architecture Documentation](docs/architecture.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📮 Contact

- **Issues**: https://github.com/org/repo/issues
- **Discussions**: https://github.com/org/repo/discussions
- **Email**: contact@example.com

## 🙏 Acknowledgments

- List contributors
- List inspirations
- List dependencies
```

### B. Contributing Guide

```markdown
# .github/CONTRIBUTING.md

# Contributing to Project Name

Thank you for your interest in contributing!

## Code of Conduct

This project adheres to the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

1. **Check existing issues** to avoid duplicates
2. **Use the bug report template** when creating an issue
3. **Include reproduction steps** and environment details
4. **IMPORTANT**: Every bug fix requires a regression test

### Suggesting Features

1. **Check existing issues/discussions** first
2. **Use the feature request template**
3. **Explain the problem** you're trying to solve
4. **IMPORTANT**: All features require tests (TDD)

### Pull Requests

#### Before Starting

1. **Create or comment on an issue** first
2. **Discuss your approach** with maintainers
3. **Ensure you understand TDD** requirements

#### Development Process

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/repo.git
   cd repo
   git remote add upstream https://github.com/org/repo.git
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-123
   ```

3. **Follow TDD (MANDATORY)**
   ```bash
   # 1. Write failing test
   npm test -- path/to/new-test.ts
   
   # 2. Write minimal code to pass
   # 3. Refactor
   # 4. Repeat
   ```

4. **Ensure tests pass**
   ```bash
   npm test
   npm run test:coverage
   npm run lint
   npm run format:check
   ```

5. **Commit your changes**
   ```bash
   # Follow Conventional Commits
   git commit -m "feat(scope): add amazing feature

   Implements #123
   
   - Detail 1
   - Detail 2"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

#### PR Requirements

- [ ] Tests written BEFORE implementation (TDD)
- [ ] All tests pass
- [ ] Coverage >= 80%
- [ ] Linting passes
- [ ] Documentation updated
- [ ] Conventional Commits format
- [ ] Issue reference included
- [ ] For bug fixes: Regression test included

### TDD Requirements (MANDATORY)

**Every contribution MUST follow Test-Driven Development:**

1. **RED**: Write a failing test first
2. **GREEN**: Write minimal code to pass the test
3. **REFACTOR**: Clean up code while keeping tests green

**Example:**

```typescript
// 1. RED - Write failing test
describe('UserService', () => {
  it('should create a user with valid email', () => {
    const service = new UserService();
    const user = service.createUser('test@example.com', 'John');
    expect(user.email).toBe('test@example.com');
  });
});

// Run: npm test -- will FAIL

// 2. GREEN - Implement minimal code
class UserService {
  createUser(email: string, name: string): User {
    return { email, name };
  }
}

// Run: npm test -- will PASS

// 3. REFACTOR - Improve code
class UserService {
  createUser(email: string, name: string): User {
    if (!this.isValidEmail(email)) {
      throw new Error('Invalid email');
    }
    return new User(email, name);
  }
  
  private isValidEmail(email: string): boolean {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  }
}

// Tests still pass ✓
```

### Bug Fix Requirements (MANDATORY)

**Every bug fix MUST include a regression test:**

```typescript
// 1. Write test that reproduces the bug
describe('UserService', () => {
  /**
   * Bug #123: UserService crashes with null email
   * This test prevents regression
   */
  it('should handle null email gracefully (Bug #123)', () => {
    const service = new UserService();
    expect(() => service.createUser(null, 'John'))
      .toThrow('Email is required');
  });
});

// Test will FAIL (reproduces bug)

// 2. Fix the bug
class UserService {
  createUser(email: string | null, name: string): User {
    if (!email) {
      throw new Error('Email is required');
    }
    // ... rest of implementation
  }
}

// Test will PASS (bug fixed)
```

## Style Guide

### Code Style

- Follow the project's ESLint/Prettier configuration
- Use TypeScript strict mode
- Prefer functional programming patterns
- Use meaningful variable names
- Add comments for complex logic

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): subject

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
```
feat(auth): add JWT authentication

Implements #45

- Added JWT token generation
- Added middleware for token validation
- Added refresh token support

fix(database): resolve connection pool leak

Fixes #123

Bug occurred when connections weren't properly released.
Added regression test to prevent future occurrences.
```

### Architecture Guidelines

Follow **Hexagonal Architecture**:

- **Domain**: No external dependencies
- **Application**: Orchestration only
- **Infrastructure**: External integrations
- **Adapters**: Input/output interfaces

## Testing Guidelines

### Unit Tests
- Test individual functions/methods
- Mock external dependencies
- Fast execution (< 1s per test)

### Integration Tests
- Test component interactions
- Use test database/services
- Moderate execution time

### E2E Tests
- Test complete user flows
- Use test environment
- Slower execution acceptable

## Getting Help

- **Questions**: Use GitHub Discussions
- **Bugs**: Create an issue with bug template
- **Features**: Create an issue with feature template
- **Security**: Email security@example.com

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- GitHub contributors page

Thank you for contributing! 🎉
```

---

## 7. Deployment Checklist

### Repository Setup
- [ ] **README.md**: Comprehensive with badges
- [ ] **LICENSE**: Appropriate open source license
- [ ] **CONTRIBUTING.md**: Clear contribution guidelines
- [ ] **CODE_OF_CONDUCT.md**: Community standards
- [ ] **SECURITY.md**: Security policy defined
- [ ] **.gitignore**: Comprehensive exclusions
- [ ] **Issue templates**: Bug, feature, question templates
- [ ] **PR template**: Comprehensive with TDD checklist

### GitHub Actions
- [ ] **CI workflow**: Lint, test, build configured
- [ ] **CD workflow**: Deployment automation
- [ ] **CodeQL**: Security scanning enabled
- [ ] **Dependabot**: Automated dependency updates
- [ ] **Secret scanning**: Gitleaks or equivalent
- [ ] **Container workflow**: GHCR publishing
- [ ] **TDD verification**: Automated test checks
- [ ] **Bug fix verification**: Regression test checks

### Branch Protection
- [ ] **Main branch**: Protected with required checks
- [ ] **Develop branch**: Protected with required checks
- [ ] **Required reviews**: Minimum 2 reviewers
- [ ] **Status checks**: All CI checks required
- [ ] **Linear history**: Enforced
- [ ] **Code owners**: CODEOWNERS file configured
- [ ] **Dismiss stale reviews**: Enabled
- [ ] **Conversation resolution**: Required

### Security
- [ ] **Dependabot**: Configured for all ecosystems
- [ ] **CodeQL**: Scheduled scans configured
- [ ] **Secret scanning**: Enabled
- [ ] **Vulnerability alerts**: Enabled
- [ ] **Security advisories**: Private reporting enabled
- [ ] **Secrets**: Using GitHub Secrets/Environments
- [ ] **Container scanning**: Trivy or equivalent
- [ ] **SBOM generation**: Enabled for containers

### Documentation
- [ ] **API documentation**: Auto-generated from code
- [ ] **Architecture docs**: ADRs and diagrams
- [ ] **Setup guide**: Clear installation steps
- [ ] **Testing guide**: How to run tests
- [ ] **Deployment guide**: How to deploy
- [ ] **GitHub Pages**: Documentation site (if applicable)
- [ ] **Wiki**: Additional documentation (if needed)

### Container Registry
- [ ] **GHCR configured**: Package permissions set
- [ ] **Multi-platform builds**: amd64 and arm64
- [ ] **Image tagging**: Semantic versioning
- [ ] **Vulnerability scanning**: Automated on push
- [ ] **Image signing**: Provenance and SBOM
- [ ] **Cleanup policy**: Old images removed

### Automation
- [ ] **Issue labeling**: Automated based on content
- [ ] **Stale issue management**: Configured
- [ ] **PR size labeling**: Automated
- [ ] **Commit linting**: Conventional Commits enforced
- [ ] **Release notes**: Auto-generated
- [ ] **Deployment notifications**: Configured

### TDD & Testing
- [ ] **TDD workflow**: Verification automated
- [ ] **Coverage tracking**: Codecov or equivalent
- [ ] **Coverage threshold**: >= 80% enforced
- [ ] **Regression tests**: Required for bug fixes
- [ ] **Test documentation**: Clear testing guide

---

## 8. Why This Configuration Works

1. **TDD Enforcement**: Automated workflows verify tests exist and pass, preventing bugs.
2. **Regression Shield**: Bug fixes require tests, building safety net over time.
3. **Security First**: Multiple layers (CodeQL, Dependabot, secret scanning, container scanning).
4. **Everything as Code**: Workflows, configs, infrastructure versioned and reviewed.
5. **Automated CI/CD**: Every push tested, every merge deployable.
6. **Branch Protection**: Quality gates prevent bad code from reaching production.
7. **Issue Tracking**: Full traceability from issue → code → deployment.
8. **Container Ready**: GHCR integration with multi-platform builds and security scanning.
9. **Documentation**: Auto-generated, version-controlled, always up-to-date.
10. **Community Standards**: Clear templates, guidelines, and processes.
11. **Observability**: Metrics, coverage, security alerts provide visibility.
12. **Automation**: Reduces manual work, ensures consistency.

---

## Quick Reference

### Common gh CLI Commands

```bash
# ─────────────────────────────────────────────────────────────────
# REPOSITORY MANAGEMENT
# ─────────────────────────────────────────────────────────────────

# Clone repository
gh repo clone owner/repo

# Create new repository
gh repo create my-repo --public --clone

# Fork repository
gh repo fork owner/repo --clone

# View repository
gh repo view owner/repo

# List repositories
gh repo list owner --limit 20

# ─────────────────────────────────────────────────────────────────
# PULL REQUESTS
# ─────────────────────────────────────────────────────────────────

# Create PR (interactive)
gh pr create

# Create PR with title and body
gh pr create --title "feat: add new feature" --body "Description here"

# Create PR from current branch
gh pr create --fill  # Auto-fill from commits

# Create draft PR
gh pr create --draft

# List PRs
gh pr list
gh pr list --state open --author @me

# View PR
gh pr view 123
gh pr view 123 --web  # Open in browser

# Checkout PR locally
gh pr checkout 123

# Review PR
gh pr review 123 --approve
gh pr review 123 --request-changes --body "Please fix X"
gh pr review 123 --comment --body "LGTM"

# Merge PR
gh pr merge 123
gh pr merge 123 --squash --delete-branch
gh pr merge 123 --rebase

# Close PR
gh pr close 123

# ─────────────────────────────────────────────────────────────────
# ISSUES
# ─────────────────────────────────────────────────────────────────

# Create issue
gh issue create --title "Bug: something broken" --body "Details"

# Create issue with labels
gh issue create --title "Feature request" --label "enhancement,needs-triage"

# List issues
gh issue list
gh issue list --state open --label "bug"
gh issue list --assignee @me

# View issue
gh issue view 123
gh issue view 123 --comments

# Close issue
gh issue close 123

# Reopen issue
gh issue reopen 123

# Add comment
gh issue comment 123 --body "Working on this"

# Edit issue
gh issue edit 123 --add-label "in-progress"
gh issue edit 123 --add-assignee @me

# ─────────────────────────────────────────────────────────────────
# WORKFLOWS & ACTIONS
# ─────────────────────────────────────────────────────────────────

# List workflows
gh workflow list

# View workflow
gh workflow view ci.yml

# Run workflow
gh workflow run ci.yml
gh workflow run deploy.yml --ref main

# Run workflow with inputs
gh workflow run deploy.yml -f environment=production -f version=1.2.3

# List workflow runs
gh run list
gh run list --workflow=ci.yml --status=failure

# View run details
gh run view 12345
gh run view 12345 --log

# Watch run in progress
gh run watch 12345

# Rerun failed jobs
gh run rerun 12345 --failed

# Cancel run
gh run cancel 12345

# Download artifacts
gh run download 12345

# ─────────────────────────────────────────────────────────────────
# RELEASES
# ─────────────────────────────────────────────────────────────────

# Create release
gh release create v1.0.0

# Create release with notes
gh release create v1.0.0 --title "Release 1.0.0" --notes "Release notes here"

# Create release from tag
gh release create v1.0.0 --generate-notes

# Upload assets
gh release upload v1.0.0 ./dist/*.zip

# List releases
gh release list

# View release
gh release view v1.0.0

# Download release assets
gh release download v1.0.0

# Delete release
gh release delete v1.0.0

# ─────────────────────────────────────────────────────────────────
# SECRETS & VARIABLES
# ─────────────────────────────────────────────────────────────────

# Set secret
gh secret set MY_SECRET

# Set secret from file
gh secret set MY_SECRET < secret.txt

# Set secret for environment
gh secret set MY_SECRET --env production

# List secrets
gh secret list

# Delete secret
gh secret delete MY_SECRET

# Set variable
gh variable set MY_VAR --body "value"

# List variables
gh variable list

# ─────────────────────────────────────────────────────────────────
# GISTS
# ─────────────────────────────────────────────────────────────────

# Create gist
gh gist create file.txt

# Create public gist with description
gh gist create file.txt --public --desc "My gist"

# Create gist from multiple files
gh gist create file1.txt file2.txt

# List gists
gh gist list

# View gist
gh gist view abc123

# Edit gist
gh gist edit abc123

# ─────────────────────────────────────────────────────────────────
# API CALLS
# ─────────────────────────────────────────────────────────────────

# GET request
gh api repos/owner/repo

# POST request
gh api repos/owner/repo/issues --method POST -f title="New issue" -f body="Body"

# GraphQL query
gh api graphql -f query='{ viewer { login } }'

# Paginated results
gh api repos/owner/repo/issues --paginate

# With jq processing
gh api repos/owner/repo/pulls --jq '.[].title'
```

### GitHub Actions Patterns Cheat Sheet

```yaml
# ─────────────────────────────────────────────────────────────────
# TRIGGER PATTERNS
# ─────────────────────────────────────────────────────────────────

# Multiple triggers
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'  # Monday 6 AM UTC
  workflow_dispatch:
    inputs:
      environment:
        type: choice
        options: [dev, staging, prod]

# Path filtering
on:
  push:
    paths:
      - 'src/**'
      - '!src/**/*.md'  # Exclude markdown
    paths-ignore:
      - 'docs/**'
      - '**.md'

# Tag filtering
on:
  push:
    tags:
      - 'v*'
      - '!v*-beta*'  # Exclude beta tags

# ─────────────────────────────────────────────────────────────────
# JOB PATTERNS
# ─────────────────────────────────────────────────────────────────

# Matrix builds
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        node: [18, 20, 22]
        exclude:
          - os: windows-latest
            node: 18
        include:
          - os: ubuntu-latest
            node: 22
            experimental: true
      fail-fast: false
      max-parallel: 4

# Job dependencies
jobs:
  build:
    runs-on: ubuntu-latest
  test:
    needs: build
  deploy:
    needs: [build, test]
    if: success()

# Conditional execution
jobs:
  deploy:
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'

  notify:
    if: always()  # Run even if previous jobs fail
    needs: [deploy]

  production:
    if: startsWith(github.ref, 'refs/tags/v')

# ─────────────────────────────────────────────────────────────────
# STEP PATTERNS
# ─────────────────────────────────────────────────────────────────

# Conditional steps
steps:
  - name: Deploy to production
    if: github.ref == 'refs/heads/main'
    run: ./deploy.sh

  - name: Run on failure
    if: failure()
    run: ./notify-failure.sh

  - name: Always run
    if: always()
    run: ./cleanup.sh

# Continue on error
steps:
  - name: Optional step
    continue-on-error: true
    run: ./optional.sh

# Timeout
steps:
  - name: Long running task
    timeout-minutes: 30
    run: ./long-task.sh

# Working directory
steps:
  - name: Build frontend
    working-directory: ./frontend
    run: npm run build

# ─────────────────────────────────────────────────────────────────
# OUTPUT & ARTIFACTS
# ─────────────────────────────────────────────────────────────────

# Set output
steps:
  - name: Set output
    id: vars
    run: echo "sha_short=$(git rev-parse --short HEAD)" >> $GITHUB_OUTPUT

  - name: Use output
    run: echo "Short SHA: ${{ steps.vars.outputs.sha_short }}"

# Upload artifact
steps:
  - name: Upload build
    uses: actions/upload-artifact@v4
    with:
      name: build-${{ github.sha }}
      path: dist/
      retention-days: 7

# Download artifact
steps:
  - name: Download build
    uses: actions/download-artifact@v4
    with:
      name: build-${{ github.sha }}
      path: dist/

# Pass data between jobs
jobs:
  build:
    outputs:
      version: ${{ steps.version.outputs.value }}
    steps:
      - id: version
        run: echo "value=1.2.3" >> $GITHUB_OUTPUT

  deploy:
    needs: build
    steps:
      - run: echo "Deploying version ${{ needs.build.outputs.version }}"

# ─────────────────────────────────────────────────────────────────
# CACHING PATTERNS
# ─────────────────────────────────────────────────────────────────

# npm cache
- uses: actions/cache@v4
  with:
    path: ~/.npm
    key: npm-${{ runner.os }}-${{ hashFiles('**/package-lock.json') }}
    restore-keys: npm-${{ runner.os }}-

# pip cache
- uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: pip-${{ runner.os }}-${{ hashFiles('**/requirements.txt') }}

# Gradle cache
- uses: actions/cache@v4
  with:
    path: |
      ~/.gradle/caches
      ~/.gradle/wrapper
    key: gradle-${{ runner.os }}-${{ hashFiles('**/*.gradle*') }}

# Docker layer cache
- uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max

# ─────────────────────────────────────────────────────────────────
# SECURITY PATTERNS
# ─────────────────────────────────────────────────────────────────

# Minimal permissions
permissions:
  contents: read
  pull-requests: write

# Job-level permissions
jobs:
  deploy:
    permissions:
      contents: read
      packages: write
      id-token: write  # For OIDC

# Secret masking
steps:
  - run: |
      echo "::add-mask::${{ secrets.API_KEY }}"
      ./deploy.sh

# Environment protection
jobs:
  deploy:
    environment:
      name: production
      url: https://example.com
    steps:
      - run: ./deploy.sh
        env:
          API_KEY: ${{ secrets.PROD_API_KEY }}

# ─────────────────────────────────────────────────────────────────
# REUSABLE WORKFLOWS
# ─────────────────────────────────────────────────────────────────

# Define reusable workflow
# .github/workflows/deploy-template.yml
name: Deploy Template
on:
  workflow_call:
    inputs:
      environment:
        required: true
        type: string
    secrets:
      deploy_token:
        required: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - run: echo "Deploying to ${{ inputs.environment }}"

# Use reusable workflow
# .github/workflows/deploy-prod.yml
jobs:
  deploy:
    uses: ./.github/workflows/deploy-template.yml
    with:
      environment: production
    secrets:
      deploy_token: ${{ secrets.DEPLOY_TOKEN }}

# ─────────────────────────────────────────────────────────────────
# COMPOSITE ACTIONS
# ─────────────────────────────────────────────────────────────────

# .github/actions/setup-project/action.yml
name: Setup Project
description: Setup project dependencies
inputs:
  node-version:
    description: Node.js version
    default: '20'
runs:
  using: composite
  steps:
    - uses: actions/setup-node@v4
      with:
        node-version: ${{ inputs.node-version }}
    - run: npm ci
      shell: bash

# Usage
steps:
  - uses: ./.github/actions/setup-project
    with:
      node-version: '20'
```

### Workflow Structure Template

```yaml
# .github/workflows/comprehensive-ci.yml
# Template for a comprehensive CI/CD workflow

name: CI/CD Pipeline

# ─────────────────────────────────────────────────────────────────
# TRIGGERS
# ─────────────────────────────────────────────────────────────────
on:
  push:
    branches: [main, develop]
    paths-ignore:
      - '**.md'
      - 'docs/**'
  pull_request:
    branches: [main]
  workflow_dispatch:
    inputs:
      skip_tests:
        description: 'Skip tests'
        type: boolean
        default: false

# ─────────────────────────────────────────────────────────────────
# CONCURRENCY CONTROL
# ─────────────────────────────────────────────────────────────────
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}

# ─────────────────────────────────────────────────────────────────
# GLOBAL ENVIRONMENT
# ─────────────────────────────────────────────────────────────────
env:
  NODE_VERSION: '20'
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

# ─────────────────────────────────────────────────────────────────
# PERMISSIONS (principle of least privilege)
# ─────────────────────────────────────────────────────────────────
permissions:
  contents: read

# ─────────────────────────────────────────────────────────────────
# JOBS
# ─────────────────────────────────────────────────────────────────
jobs:
  # ─────────────────────────────────────────────────────────────
  # STAGE 1: Quality Gates
  # ─────────────────────────────────────────────────────────────
  lint:
    name: Lint & Format
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run linter
        run: npm run lint

      - name: Check formatting
        run: npm run format:check

  # ─────────────────────────────────────────────────────────────
  # STAGE 2: Testing (TDD Verification)
  # ─────────────────────────────────────────────────────────────
  test:
    name: Test
    if: ${{ !inputs.skip_tests }}
    runs-on: ubuntu-latest
    timeout-minutes: 15

    strategy:
      matrix:
        node: [18, 20, 22]
      fail-fast: false

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js ${{ matrix.node }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node }}
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test

      - name: Upload coverage
        if: matrix.node == env.NODE_VERSION
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}

  # ─────────────────────────────────────────────────────────────
  # STAGE 3: Security Scanning
  # ─────────────────────────────────────────────────────────────
  security:
    name: Security Scan
    runs-on: ubuntu-latest
    timeout-minutes: 15
    permissions:
      security-events: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: javascript

      - name: Perform analysis
        uses: github/codeql-action/analyze@v3

      - name: Run npm audit
        run: npm audit --audit-level=moderate

  # ─────────────────────────────────────────────────────────────
  # STAGE 4: Build
  # ─────────────────────────────────────────────────────────────
  build:
    name: Build
    needs: [lint, test]
    runs-on: ubuntu-latest
    timeout-minutes: 10

    outputs:
      version: ${{ steps.version.outputs.value }}

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Build
        run: npm run build

      - name: Get version
        id: version
        run: echo "value=$(node -p "require('./package.json').version")" >> $GITHUB_OUTPUT

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: build-${{ github.sha }}
          path: dist/
          retention-days: 7

  # ─────────────────────────────────────────────────────────────
  # STAGE 5: Container Build
  # ─────────────────────────────────────────────────────────────
  container:
    name: Build Container
    needs: build
    runs-on: ubuntu-latest
    timeout-minutes: 20
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Download build artifacts
        uses: actions/download-artifact@v4
        with:
          name: build-${{ github.sha }}
          path: dist/

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=sha,prefix=

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ─────────────────────────────────────────────────────────────
  # STAGE 6: Deploy
  # ─────────────────────────────────────────────────────────────
  deploy-staging:
    name: Deploy to Staging
    needs: [container, security]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment:
      name: staging
      url: https://staging.example.com

    steps:
      - name: Deploy to staging
        run: echo "Deploying version ${{ needs.build.outputs.version }} to staging"

  deploy-production:
    name: Deploy to Production
    needs: deploy-staging
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://example.com

    steps:
      - name: Deploy to production
        run: echo "Deploying to production"

  # ─────────────────────────────────────────────────────────────
  # FINAL: Status Check
  # ─────────────────────────────────────────────────────────────
  status:
    name: Pipeline Status
    needs: [lint, test, security, build, container]
    if: always()
    runs-on: ubuntu-latest

    steps:
      - name: Check status
        run: |
          if [[ "${{ needs.lint.result }}" != "success" ]] || \
             [[ "${{ needs.test.result }}" != "success" && "${{ needs.test.result }}" != "skipped" ]] || \
             [[ "${{ needs.build.result }}" != "success" ]]; then
            echo "::error::Pipeline failed"
            exit 1
          fi
          echo "✓ All checks passed"
```

---

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [CodeQL](https://codeql.github.com/)
- [Dependabot](https://docs.github.com/en/code-security/dependabot)
- [Branch Protection Rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)

---

**Last Updated:** 2026-01-18  
**Version:** 1.0  
**Maintainer:** DevOps Team


**End of Modern GitHub Workflow Guidelines**
