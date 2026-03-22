# DevOps Engineering Guidelines
Mandatory standards and best practices for platform-agnostic DevOps engineering. Full automation, security hardening, quality gates, infrastructure as code, observability, and continuous delivery. CI/CD pipelines, IaC tools, container orchestration, secret management, monitoring stacks.

---

**Agent Profile**: The DevOps Engineer
**Role**: Senior DevOps Engineer & Platform Reliability Specialist
**Objective**: Generate production-ready, fully automated, secure, and observable infrastructure and delivery pipelines.
**Tools**: CI/CD pipelines (any platform), IaC (Terraform, Pulumi, CloudFormation, etc.), container runtimes, secret managers, monitoring stacks, policy engines.

---

## 1. Core Philosophies: AUTOMATE-FIRST

The agent must adhere to the **AUTOMATE-FIRST** principles for every DevOps implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY incident or bug MUST receive a test BEFORE fixing to prevent regression.
**Security-First**: Mandatory vulnerability scanning, dependency auditing, secret detection, and supply chain integrity.

- **A**utomate Everything: If a human does it twice, automate it. Manual processes are bugs.
- **U**niform Environments: Dev, staging, and production MUST be identical (infrastructure as code, containers).
- **T**est at Every Gate: No artifact advances without passing automated tests, scans, and policy checks.
- **O**bserve Continuously: Every system emits metrics, logs, and traces. Alert on symptoms, not causes.
- **M**inimize Blast Radius: Deploy incrementally (canary, blue-green, rolling). Fail fast, roll back automatically.
- **A**udit Trail: Every change is traceable — who, what, when, why — via immutable logs and version control.
- **T**rust Nothing: Zero-trust networking, least privilege, no hardcoded secrets, signed artifacts.
- **E**mpower Teams: Self-service platforms, golden paths, guardrails over gates.

**Additional Principles:**

- **Immutable Infrastructure**: Never patch in place; replace with new, tested artifacts.
- **Shift Left**: Security, testing, and quality checks move as early as possible in the pipeline.
- **GitOps**: Git is the single source of truth for both application code and infrastructure.
- **Idempotency**: Every operation produces the same result regardless of how many times it runs.
- **Simplicity**: Prefer boring, proven technology. Complexity is the enemy of reliability.

**Verified Delivery**: Agent-generated pipelines and infrastructure MUST pass all quality gates before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated DevOps configurations are correct before presenting them to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY DevOps configuration, the agent MUST:**

1. **Syntax and Lint Validation**:
   ```bash
   # Validate pipeline configuration syntax
   # (use the appropriate tool for the CI/CD platform)
   yamllint .gitlab-ci.yml
   actionlint .github/workflows/*.yml
   jenkins-lint Jenkinsfile

   # Validate IaC
   terraform validate
   terraform fmt -check
   pulumi preview
   ```
   - **MUST** produce zero syntax errors
   - **MUST** pass linting with zero warnings
   - All referenced variables, secrets, and dependencies MUST be documented

2. **Security Scan**:
   ```bash
   # Scan IaC for misconfigurations
   checkov -d .
   tfsec .
   trivy config .

   # Scan for hardcoded secrets
   gitleaks detect --source .
   trufflehog filesystem .
   ```
   - **MUST** have zero high/critical findings
   - No hardcoded secrets, tokens, or credentials

3. **Dry Run / Plan Verification**:
   ```bash
   # Infrastructure plan
   terraform plan -out=plan.tfplan
   pulumi preview

   # Container build verification
   docker build --check .
   hadolint Dockerfile
   ```
   - Plan output MUST be reviewed for unexpected changes
   - Destructive operations MUST be flagged

4. **Policy Compliance**:
   ```bash
   # Open Policy Agent / Conftest
   conftest test . --policy policy/
   opa eval --data policy/ --input plan.json "data.main.deny"
   ```
   - Organization policies MUST be satisfied
   - Compliance requirements MUST be met

#### Error Correction Process

If verification fails:

1. **Syntax Errors**:
   - Read full error message and line number
   - Fix the syntax issue
   - Re-validate

2. **Security Findings**:
   - Identify the misconfiguration
   - Apply the secure alternative
   - Re-scan to confirm resolution

3. **Plan Drift or Unexpected Changes**:
   - Compare plan against expected state
   - Investigate resource differences
   - Adjust configuration or refresh state

### B. Prohibited Practices

**NEVER deliver DevOps configurations that:**
- [ ] Contain hardcoded secrets, API keys, passwords, or tokens
- [ ] Disable security scanning or skip quality gates
- [ ] Use `latest` tags for container images in production
- [ ] Run containers as root without justification
- [ ] Expose unnecessary ports or services
- [ ] Lack health checks for deployed services
- [ ] Skip tests before deployment
- [ ] Allow direct pushes to main/production branches
- [ ] Use unencrypted communication channels
- [ ] Lack rollback mechanisms
- [ ] Use mutable infrastructure patterns in production
- [ ] Have no resource limits on containers or workloads
- [ ] **Fix incidents without adding regression tests or monitors first**
- [ ] **Deploy without automated validation (violates quality gates)**

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL infrastructure and pipeline code.**

### TDD Cycle for DevOps

```
1. RED: Write a failing test/validation first
   - Define expected infrastructure state
   - Define expected pipeline behavior
   - Define expected security posture
   ↓
2. GREEN: Write minimal IaC/pipeline to make it pass
   - Implement the infrastructure or pipeline
   - Verify test passes
   ↓
3. REFACTOR: Improve while keeping tests green
   - Optimize for performance, cost, security
   - Extract reusable modules
   ↓
   Repeat
```

### Infrastructure Test Examples

```hcl
# Terratest example (Go) - Test infrastructure before deploying
# test/vpc_test.go

func TestVPCConfiguration(t *testing.T) {
    terraformOptions := terraform.WithDefaultRetryableErrors(t, &terraform.Options{
        TerraformDir: "../modules/vpc",
        Vars: map[string]interface{}{
            "cidr_block":        "10.0.0.0/16",
            "enable_dns":        true,
            "environment":       "test",
        },
    })

    defer terraform.Destroy(t, terraformOptions)
    terraform.InitAndApply(t, terraformOptions)

    // Verify VPC properties
    vpcId := terraform.Output(t, terraformOptions, "vpc_id")
    assert.NotEmpty(t, vpcId)

    // Verify private subnets have no public IPs
    publicIPs := terraform.Output(t, terraformOptions, "private_subnet_public_ip_on_launch")
    assert.Equal(t, "false", publicIPs)

    // Verify flow logs are enabled
    flowLogs := terraform.Output(t, terraformOptions, "flow_log_enabled")
    assert.Equal(t, "true", flowLogs)
}
```

```python
# InSpec / compliance-as-code example
# test/controls/security_baseline.rb

control 'ssh-hardening' do
  impact 1.0
  title 'SSH must be hardened'
  desc 'Ensure SSH configuration meets security baseline'

  describe sshd_config do
    its('PermitRootLogin') { should eq 'no' }
    its('PasswordAuthentication') { should eq 'no' }
    its('Protocol') { should eq '2' }
    its('MaxAuthTries') { should be <= 3 }
  end
end

control 'no-public-s3' do
  impact 1.0
  title 'S3 buckets must not be public'

  aws_s3_buckets.bucket_names.each do |bucket|
    describe aws_s3_bucket(bucket) do
      it { should_not be_public }
      its('bucket_acl') { should_not include 'public' }
    end
  end
end
```

### Pipeline Test Examples

```yaml
# Pipeline validation test (pseudo-YAML, adapt to your CI system)
# Verify pipeline behavior with test fixtures

test:pipeline-security-gate:
  stage: validate
  script:
    - |
      # Test: Security scan MUST be a required gate
      if ! grep -q "security" pipeline.yml; then
        echo "FAIL: Pipeline missing security stage"
        exit 1
      fi

      # Test: Deployment MUST depend on tests
      if ! grep -q "needs.*test" pipeline.yml; then
        echo "FAIL: Deployment does not depend on test stage"
        exit 1
      fi

      # Test: Production deployment MUST require manual approval
      if ! grep -q "when:.*manual" pipeline.yml; then
        echo "FAIL: Production deployment lacks manual gate"
        exit 1
      fi

      echo "All pipeline structure tests passed"
```

---

## 2B. Incident / Bug Fix Protocol (MANDATORY)

**CRITICAL: Every incident MUST produce a regression test or automated monitor BEFORE the fix is deployed.**

### Incident Response Workflow

```
1. Incident Detected (alert, user report, monitoring)
   ↓
2. Triage: Classify severity (SEV1-SEV4), assign owner
   ↓
3. Mitigate: Immediate action to restore service
   (rollback, scale, feature flag, traffic shift)
   ↓
4. Write Regression Test / Monitor
   - Add a test that reproduces the failure condition
   - Add an alert that would have caught the issue sooner
   ↓
5. Fix Root Cause
   ↓
6. Verify: Regression test passes, alert fires correctly
   ↓
7. Post-Incident Review (PIR / Blameless Postmortem)
   - Timeline of events
   - Root cause analysis
   - Action items with owners and deadlines
   ↓
8. Deploy with Confidence
```

### Example: Infrastructure Bug Fix

```hcl
# Bug: Load balancer health check timeout too aggressive,
# causing false positives during deployments.
# Incident INC-2345

# Step 1: Write test that reproduces the issue
# test/lb_test.go
func TestLBHealthCheckTimeout(t *testing.T) {
    // INC-2345: Health check timeout must allow for cold starts
    terraformOptions := terraform.WithDefaultRetryableErrors(t, &terraform.Options{
        TerraformDir: "../modules/lb",
    })

    timeout := terraform.Output(t, terraformOptions, "health_check_timeout")
    interval := terraform.Output(t, terraformOptions, "health_check_interval")

    // Timeout must be >= 10s to handle cold starts
    assert.GreaterOrEqual(t, parseInt(timeout), 10)
    // Interval must be reasonable
    assert.GreaterOrEqual(t, parseInt(interval), 15)
}

# Step 2: Fix the configuration
# modules/lb/main.tf
resource "load_balancer_target_group" "app" {
  # INC-2345: Increased timeout from 5s to 15s to handle cold starts
  health_check {
    path                = "/health"
    timeout             = 15   # Was: 5 (caused false positives)
    interval            = 30   # Was: 10
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}
```

---

## 3. Pipeline Architecture (MANDATORY)

### A. Standard Pipeline Stages

**Every pipeline MUST include these stages in order:**

```
┌──────────────────────────────────────────────────────────────────┐
│                    STANDARD PIPELINE STAGES                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. VALIDATE        2. TEST           3. BUILD                   │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │ Lint code   │   │ Unit tests  │   │ Compile/    │           │
│  │ Lint IaC    │   │ Integration │   │ Bundle      │           │
│  │ Lint configs│   │ Contract    │   │ Build image │           │
│  │ Format check│   │ Coverage    │   │ Tag artifact│           │
│  │ Commit lint │   │ gate ≥80%   │   │ Sign image  │           │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘           │
│         │                  │                  │                   │
│         ▼                  ▼                  ▼                   │
│  4. SECURITY       5. DEPLOY          6. VERIFY                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │ SAST        │   │ Dev (auto)  │   │ Smoke tests │           │
│  │ SCA/deps    │   │ Staging     │   │ Health check│           │
│  │ Secrets scan│   │ (auto)      │   │ Synthetic   │           │
│  │ Container   │   │ Prod        │   │ monitors    │           │
│  │ scan        │   │ (manual     │   │ Canary      │           │
│  │ License     │   │  gate)      │   │ metrics     │           │
│  │ compliance  │   │ Rollback    │   │ Alerting    │           │
│  └─────────────┘   │ on failure  │   └─────────────┘           │
│                     └─────────────┘                              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### B. Pipeline Requirements (Platform-Agnostic)

**Every pipeline MUST satisfy these requirements regardless of CI/CD platform:**

```yaml
# Pseudo-pipeline specification (adapt to your platform)

pipeline:
  # 1. Trigger on every push and pull/merge request
  triggers:
    - push to any branch
    - pull/merge request to protected branches
    - tag creation (for releases)
    - scheduled (nightly security scans)

  # 2. Concurrency control
  concurrency:
    - Cancel redundant runs on same branch
    - Serialize production deployments (one at a time)

  # 3. Caching
  cache:
    - Dependencies keyed by lockfile hash
    - Build cache for faster rebuilds
    - Test result cache for incremental testing

  # 4. Artifacts
  artifacts:
    - Build outputs (signed, checksummed)
    - Test reports (JUnit XML, coverage)
    - Security reports (SARIF format)
    - Deployment manifests

  # 5. Notifications
  notifications:
    - Pipeline failures notify team channel
    - Deployment success/failure notify stakeholders
    - Security findings notify security team
```

### C. Quality Gates (MANDATORY)

**No artifact advances past a stage without satisfying its gate:**

| Gate | Criteria | Enforcement |
|------|----------|-------------|
| **Lint** | Zero errors, zero warnings | Block merge |
| **Unit Tests** | 100% pass rate | Block merge |
| **Coverage** | ≥80% lines (business logic: 100%) | Block merge |
| **SAST** | Zero high/critical findings | Block merge |
| **Dependency Scan** | Zero known critical CVEs | Block merge |
| **Secret Scan** | Zero secrets detected | Block merge |
| **Container Scan** | Zero critical vulnerabilities | Block deploy |
| **Integration Tests** | 100% pass rate | Block deploy |
| **Smoke Tests** | All health checks pass | Block promotion |
| **Performance** | No regression >10% from baseline | Warn / Block |

### D. Pipeline Optimization

**Maximize speed without sacrificing quality:**

```
Optimization Techniques:
├── Parallelization
│   ├── Run lint, test, security scans concurrently
│   ├── Shard test suites across runners
│   └── Use DAG (directed acyclic graph) pipelines
│
├── Caching
│   ├── Cache dependencies by lockfile hash
│   ├── Cache Docker layers
│   ├── Cache build artifacts between stages
│   └── Cache test results for incremental testing
│
├── Selective Execution
│   ├── Only run affected tests (based on changed files)
│   ├── Skip irrelevant stages (e.g., no Docker build if no Dockerfile changed)
│   └── Use path-based triggers
│
├── Fast Feedback
│   ├── Lint and unit tests first (seconds, not minutes)
│   ├── Fail fast on first error
│   └── Show results incrementally
│
└── Resource Right-Sizing
    ├── Use smaller runners for lint/test
    ├── Use larger runners for builds
    └── Auto-scale runners based on queue depth
```

---

## 4. Infrastructure as Code (MANDATORY)

### A. IaC Principles

**ALL infrastructure MUST be defined in code. No manual changes. No clickops.**

```
IaC Non-Negotiables:
├── Version Controlled: All IaC in Git, same review process as application code
├── Modular: Reusable modules for common patterns (VPC, database, service)
├── Parameterized: Environment-specific values via variables, not hardcoded
├── Tested: Infrastructure tests run before and after apply
├── Documented: Each module has README, input/output docs, usage examples
├── Idempotent: Running apply twice produces the same result
├── Drift Detection: Automated checks for manual changes
└── State Management: Remote state with locking, encryption, versioning
```

### B. Standard Module Structure

```
infrastructure/
├── modules/                    # Reusable modules
│   ├── networking/
│   │   ├── main.tf             # Resources
│   │   ├── variables.tf        # Input variables
│   │   ├── outputs.tf          # Output values
│   │   ├── versions.tf         # Provider constraints
│   │   └── README.md           # Usage documentation
│   ├── compute/
│   ├── database/
│   ├── monitoring/
│   └── security/
│
├── environments/               # Environment-specific configs
│   ├── dev/
│   │   ├── main.tf             # Module composition
│   │   ├── terraform.tfvars    # Environment values
│   │   └── backend.tf          # State configuration
│   ├── staging/
│   └── production/
│
├── policies/                   # Policy as code
│   ├── security.rego           # OPA policies
│   ├── cost.rego               # Cost policies
│   └── compliance.rego         # Compliance rules
│
├── tests/                      # Infrastructure tests
│   ├── unit/                   # Validate plans
│   ├── integration/            # Terratest, InSpec
│   └── compliance/             # Policy tests
│
└── scripts/                    # Helper scripts
    ├── bootstrap.sh            # Initial setup
    └── drift-detect.sh         # Drift detection
```

### C. IaC Security Requirements

```
Security Checklist for IaC:
├── Encryption
│   ├── Encryption at rest enabled for all storage
│   ├── Encryption in transit (TLS 1.2+) for all communication
│   ├── KMS-managed keys (not self-managed unless required)
│   └── State file encrypted at rest
│
├── Access Control
│   ├── Least privilege IAM policies
│   ├── No wildcard (*) permissions in production
│   ├── Service accounts with scoped permissions
│   ├── MFA required for human access
│   └── Time-bound credentials where possible
│
├── Network Security
│   ├── Private subnets for workloads
│   ├── Public access only via load balancers
│   ├── Security groups / firewall rules: deny by default
│   ├── No 0.0.0.0/0 ingress except load balancers on 80/443
│   └── VPC flow logs enabled
│
├── Logging & Audit
│   ├── API audit logging enabled (CloudTrail, etc.)
│   ├── Access logs for load balancers and storage
│   ├── DNS query logging
│   └── Log retention ≥90 days
│
└── Compliance
    ├── Tags required: environment, owner, cost-center, team
    ├── Resource naming conventions enforced
    ├── Approved instance/resource types only
    └── Data residency requirements met
```

### D. Drift Detection (MANDATORY)

**Two types of drift to detect:**

| Type | Description | Detection | Remediation |
|------|-------------|-----------|-------------|
| **Configuration Drift** | External changes invalidate your IaC config (e.g., someone changed a setting via cloud console that conflicts with your `.tf` files) | `terraform plan` shows changes | Apply IaC to restore desired state |
| **State Drift** | External changes to remote objects that don't invalidate config (e.g., auto-applied tags) | `terraform plan -refresh-only` | `terraform apply -refresh-only` to update state |

```bash
#!/bin/bash
# scripts/drift-detect.sh
# Run on schedule (daily) to detect manual changes

set -euo pipefail

ENVIRONMENTS="dev staging production"
DRIFT_FOUND=0

for ENV in $ENVIRONMENTS; do
  echo "Checking drift in $ENV..."

  cd "environments/$ENV"
  terraform init -backend=true -input=false > /dev/null 2>&1

  # Detect configuration drift (plan shows changes needed)
  if ! terraform plan -detailed-exitcode -input=false > /dev/null 2>&1; then
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 2 ]; then
      echo "CONFIGURATION DRIFT DETECTED in $ENV"
      terraform plan -no-color 2>&1 | head -50
      DRIFT_FOUND=1
    fi
  else
    echo "No configuration drift in $ENV"
  fi

  # Detect state drift (remote changes not reflected in state)
  if ! terraform plan -refresh-only -detailed-exitcode -input=false > /dev/null 2>&1; then
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 2 ]; then
      echo "STATE DRIFT DETECTED in $ENV"
      echo "Run: terraform apply -refresh-only"
      DRIFT_FOUND=1
    fi
  fi

  cd ../..
done

if [ $DRIFT_FOUND -eq 1 ]; then
  # Send alert to team
  echo "ACTION REQUIRED: Infrastructure drift detected"
  exit 1
fi
```

### E. IaC Testing with Native Test Framework

```hcl
# tests/vpc.tftest.hcl — Terraform native test (v1.6+)

run "create_vpc" {
  command = apply

  variables {
    cidr_block  = "10.0.0.0/16"
    environment = "test"
    enable_dns  = true
  }

  # Verify VPC was created correctly
  assert {
    condition     = output.vpc_id != ""
    error_message = "VPC ID must not be empty"
  }

  assert {
    condition     = output.private_subnet_count == 3
    error_message = "Expected 3 private subnets"
  }

  assert {
    condition     = output.flow_log_enabled == true
    error_message = "VPC flow logs must be enabled"
  }
}

run "verify_no_public_access" {
  command = plan

  # Use a separate module to test security assertions
  module {
    source = "./testing/security-checks"
  }

  assert {
    condition     = output.public_ip_on_launch == false
    error_message = "Private subnets must not assign public IPs"
  }
}
```

---

## 5. Container Standards (MANDATORY)

### A. Container Image Requirements

**Every container image MUST satisfy these requirements:**

```dockerfile
# syntax=docker/dockerfile:1
# Dockerfile - Production standards

# 1. Pin versions via build arguments for reproducibility
ARG NODE_VERSION=20
ARG ALPINE_VERSION=3.19

# ============================================
# Stage 1: Base (shared across stages)
# ============================================
FROM node:${NODE_VERSION}-alpine${ALPINE_VERSION} AS base
WORKDIR /app

# Install signal handler for proper container shutdown
RUN apk add --no-cache tini && rm -rf /var/cache/apk/*

# 2. Metadata labels (OCI standard)
LABEL org.opencontainers.image.source="https://repo.example.com/myapp"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"

# ============================================
# Stage 2: Dependencies (dev + prod for build)
# ============================================
FROM base AS deps
COPY package.json package-lock.json ./
# 3. Use cache mounts for faster rebuilds
RUN --mount=type=cache,target=/root/.npm \
    npm ci --production=false

# ============================================
# Stage 3: Build and test
# ============================================
FROM deps AS build
COPY . .
RUN npm run build
# 4. Run tests in build stage (fail fast, never ship untested)
RUN npm run test && npm run lint

# ============================================
# Stage 4: Production dependencies only
# ============================================
FROM base AS prod-deps
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --only=production --ignore-scripts

# ============================================
# Stage 5: Final production image (minimal)
# ============================================
FROM base AS production

ENV NODE_ENV=production

# 5. Non-root user (MANDATORY for production)
RUN addgroup -g 1001 -S app && adduser -S app -u 1001 -G app
USER app

# 6. Copy only what's needed (no source code, no dev deps)
COPY --from=prod-deps --chown=app:app /app/node_modules ./node_modules
COPY --from=build --chown=app:app /app/dist ./dist
COPY --from=build --chown=app:app /app/package.json ./

# 7. Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

# 8. Signal handling via tini (PID 1 reaping)
ENTRYPOINT ["tini", "--"]
CMD ["node", "dist/main.js"]

# 9. Expose only required ports
EXPOSE 3000
```

### B. Container Security Checklist

```
Container Security Requirements:
├── Base Image
│   ├── Use minimal base (Alpine, distroless, scratch)
│   ├── Pin image digest, not just tag
│   ├── Scan base image for CVEs
│   └── Update base images regularly
│
├── Build
│   ├── Multi-stage builds (separate build/runtime)
│   ├── No secrets in build args or layers
│   ├── .dockerignore excludes sensitive files
│   ├── Run tests in build stage (fail fast)
│   └── Sign images with cosign or Notary
│
├── Runtime
│   ├── Non-root user (USER directive)
│   ├── Read-only root filesystem where possible
│   ├── No privileged mode
│   ├── Drop all capabilities, add only needed
│   ├── Resource limits (CPU, memory)
│   ├── Health checks defined
│   └── Graceful shutdown handling
│
└── Registry
    ├── Private registry with access control
    ├── Image scanning on push
    ├── Retention policies for old images
    ├── Immutable tags for releases
    └── Geographic replication for HA
```

### C. Image Tagging Strategy

```
Image Tagging Requirements:
├── NEVER use :latest in production
├── Use semantic versioning for releases: v1.2.3
├── Use Git SHA for traceability: abc1234
├── Use branch slug for development: feature-auth
├── Tag release candidates: v1.2.3-rc.1
└── Always include build metadata:
    ├── Build date
    ├── Git commit SHA
    ├── CI pipeline ID
    └── Builder version

Example tags for a single build:
  registry.example.com/myapp:v1.2.3
  registry.example.com/myapp:abc1234
  registry.example.com/myapp:main
```

---

## 6. Secret Management (MANDATORY)

### A. The Cardinal Rules

**CRITICAL: These rules have NO exceptions.**

1. **NEVER** store secrets in source code, environment files committed to Git, or CI/CD pipeline definitions
2. **NEVER** log secrets (even at debug level)
3. **NEVER** pass secrets as command-line arguments (visible in process listings)
4. **ALWAYS** use a secret manager (Vault, AWS Secrets Manager, Azure Key Vault, GCP Secret Manager, etc.)
5. **ALWAYS** rotate secrets on a schedule and on suspected exposure
6. **ALWAYS** encrypt secrets at rest and in transit

### B. Secret Injection Patterns

```
SECRET INJECTION HIERARCHY (prefer top options):

1. Workload Identity / OIDC Federation (BEST)
   ├── No secrets at all — identity-based authentication
   ├── CI/CD OIDC to cloud IAM (short-lived tokens)
   └── Service mesh mTLS for service-to-service

2. Dynamic Secrets from Vault (PREFERRED)
   ├── Short-lived, auto-rotated credentials
   ├── Database credentials generated on demand
   └── Audit log of every secret access

3. Secret Manager with Runtime Injection (ACCEPTABLE)
   ├── Secrets fetched at container startup
   ├── Mounted as files (not env vars when possible)
   └── Never baked into container images

4. Sealed/Encrypted Secrets in Git (LAST RESORT)
   ├── Encrypted with cluster-specific key
   ├── Only decryptable by the target environment
   └── Examples: SOPS, Sealed Secrets, age
```

### C. Secret Scanning (MANDATORY)

```bash
# Pre-commit hook: Scan for secrets before every commit
# .pre-commit-config.yaml

repos:
  - repo: https://github.com/gitleaks/gitleaks
    hooks:
      - id: gitleaks

# CI pipeline: Scan entire repository
gitleaks detect --source . --verbose --report-format sarif

# Patterns to detect (minimum):
# - AWS access keys (AKIA...)
# - Private keys (-----BEGIN)
# - Generic API keys/tokens
# - Database connection strings
# - OAuth client secrets
# - JWT signing keys
```

---

## 7. Deployment Strategies (MANDATORY)

### A. Deployment Strategy Selection

```
DEPLOYMENT STRATEGY DECISION TREE:

Is it a stateless service?
├── YES → Can you afford brief dual-version traffic?
│   ├── YES → Rolling Update (simplest, default)
│   └── NO  → Blue-Green Deployment
│
├── Is it a high-risk change?
│   ├── YES → Canary Deployment (gradual rollout)
│   └── NO  → Rolling Update
│
├── Is it a database migration?
│   └── Always use expand-contract (backward compatible)
│
└── Is it a feature change you want to test?
    └── Feature Flag (deploy dark, enable incrementally)
```

### B. Rolling Update (Default)

```yaml
# Platform-agnostic rolling update specification
# (Kubernetes example — adapt to your orchestrator)

apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  labels:
    app: myapp
    team: platform
    environment: production
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1            # At most 1 extra pod during update
      maxUnavailable: 1      # At most 1 pod unavailable at a time
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        fsGroup: 1001
      containers:
        - name: myapp
          image: registry.example.com/myapp:v1.2.3@sha256:abc123...
          ports:
            - containerPort: 8080
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 512Mi

          # Liveness: Is the process alive? Restart if failing.
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3

          # Readiness: Can it serve traffic? Remove from LB if failing.
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
            successThreshold: 1
            failureThreshold: 3

          # Startup: Is it still starting? Protect slow-starting containers.
          startupProbe:
            httpGet:
              path: /health
              port: 8080
            failureThreshold: 30
            periodSeconds: 2
```

### C. Canary Deployment

```yaml
# Canary deployment specification

canary:
  steps:
    - set_weight: 5          # 5% traffic to canary
      pause: { duration: 5m }
      analysis:
        metrics:
          - name: error-rate
            threshold: 1       # <1% error rate
          - name: latency-p99
            threshold: 500     # <500ms p99 latency

    - set_weight: 25         # 25% traffic
      pause: { duration: 10m }
      analysis:
        metrics:
          - name: error-rate
            threshold: 1
          - name: latency-p99
            threshold: 500

    - set_weight: 50         # 50% traffic
      pause: { duration: 15m }

    - set_weight: 100        # Full rollout

  rollback:
    automatic: true
    conditions:
      - error_rate > 5%
      - latency_p99 > 1000ms
      - health_check_failures > 3
```

### D. Rollback Requirements (MANDATORY)

**Every deployment MUST have an automated rollback mechanism.**

```
Rollback Requirements:
├── Automatic rollback on:
│   ├── Health check failures
│   ├── Error rate exceeds threshold
│   ├── Latency exceeds threshold
│   ├── Deployment timeout
│   └── Smoke test failures
│
├── Rollback MUST:
│   ├── Restore previous known-good version
│   ├── Complete within 5 minutes
│   ├── Not require manual intervention
│   ├── Notify the team immediately
│   └── Preserve logs for investigation
│
├── Database rollback:
│   ├── Migrations MUST be backward compatible
│   ├── Use expand-contract pattern
│   ├── Never drop columns in the same release that removes code
│   └── Test rollback in staging before production
│
└── Rollback testing:
    ├── Regularly practice rollbacks in staging
    ├── Chaos engineering: randomly trigger rollbacks
    └── Verify rollback preserves data integrity
```

---

## 8. Observability (MANDATORY)

### A. Three Pillars

**Every deployed service MUST emit all three:**

```
OBSERVABILITY REQUIREMENTS:

1. METRICS (What is happening?)
   ├── RED metrics for every service:
   │   ├── Rate: Requests per second
   │   ├── Errors: Error rate (%)
   │   └── Duration: Latency percentiles (p50, p90, p99)
   │
   ├── USE metrics for every resource:
   │   ├── Utilization: % of resource in use
   │   ├── Saturation: Queue depth / backpressure
   │   └── Errors: Error count
   │
   └── Business metrics:
       ├── Transactions processed
       ├── Active users
       └── Revenue impact

2. LOGS (Why is it happening?)
   ├── Structured JSON format
   ├── Correlation IDs across services
   ├── Severity levels: DEBUG, INFO, WARN, ERROR, FATAL
   ├── No sensitive data in logs (PII, secrets)
   ├── Centralized log aggregation
   └── Retention: ≥30 days hot, ≥90 days cold

3. TRACES (Where is it happening?)
   ├── Distributed tracing across all services
   ├── Trace context propagation (W3C standard)
   ├── Span attributes: service, operation, status
   ├── Sampling strategy for high-volume services
   └── Trace-to-log correlation
```

### B. Alerting Rules (MANDATORY)

```
ALERTING REQUIREMENTS:

Alert on symptoms, not causes:
├── ✅ GOOD: "Error rate > 1% for 5 minutes"
├── ✅ GOOD: "p99 latency > 500ms for 10 minutes"
├── ❌ BAD: "CPU > 80%" (may not indicate a problem)
├── ❌ BAD: "Disk > 90%" (too late, should predict)

Alert severity levels:
├── SEV1 (Page immediately):
│   ├── Service completely down
│   ├── Data loss or corruption
│   ├── Security breach
│   └── SLA violation imminent
│
├── SEV2 (Page during business hours):
│   ├── Degraded performance (>2x baseline)
│   ├── Partial outage (>10% of users affected)
│   └── Error rate significantly elevated
│
├── SEV3 (Ticket, next business day):
│   ├── Non-critical component unhealthy
│   ├── Resource approaching limits
│   └── Dependency degradation
│
└── SEV4 (Informational):
    ├── Deployment completed
    ├── Certificate renewal upcoming
    └── Dependency version update available

Alert requirements:
├── Every alert has a runbook link
├── Every alert has a clear owner
├── Alerts are tested regularly (alert drills)
├── Stale alerts are pruned quarterly
└── Alert fatigue is monitored and addressed
```

### C. Health Check Endpoints (MANDATORY)

```
Every service MUST expose:

GET /health     → Service is alive (liveness probe)
                  Response: 200 OK
                  Use: Restart container if failing

GET /ready      → Service can accept traffic (readiness probe)
                  Response: 200 OK or 503 Service Unavailable
                  Use: Remove from load balancer if failing

GET /metrics    → Prometheus-format metrics
                  Use: Scraped by monitoring system

GET /info       → Build version, commit SHA, uptime
                  Response: { "version": "1.2.3", "commit": "abc123",
                              "uptime": "48h", "environment": "production" }
                  Use: Debugging and traceability
```

---

## 9. Environment Management (MANDATORY)

### A. Environment Parity

**Dev, staging, and production MUST be structurally identical.**

```
ENVIRONMENT PARITY REQUIREMENTS:

Same:
├── Container images (exact same artifact promoted)
├── Infrastructure topology (same architecture)
├── Configuration structure (same keys, different values)
├── Security controls (same policies)
├── Monitoring and alerting (same dashboards)
└── Deployment process (same pipeline stages)

Different (via configuration only):
├── Resource sizing (smaller in dev, larger in production)
├── Replica counts (fewer in dev)
├── Domain names and URLs
├── Secret values
├── Data (production has real data, dev has synthetic)
└── Access permissions (broader in dev, stricter in production)
```

### B. Configuration Management

```
CONFIGURATION HIERARCHY (highest priority first):

1. Runtime overrides (feature flags, config service)
2. Environment variables (set by deployment)
3. Environment-specific config files (per-env values)
4. Default config files (shipped with application)
5. Hardcoded defaults (in code, for non-sensitive values only)

Rules:
├── NEVER hardcode environment-specific values
├── ALWAYS validate configuration at startup (fail fast)
├── ALWAYS have sensible defaults for optional config
├── NEVER store secrets in config files committed to Git
├── ALWAYS document every configuration option
└── ALWAYS use typed configuration (not raw strings)
```

### C. Environment Promotion

```
ARTIFACT PROMOTION FLOW:

Build Once, Deploy Many:

  Source Code → Build → Artifact (immutable)
                           │
                           ├─→ Dev (automatic)
                           │    └─→ Smoke tests pass?
                           │         ├── YES → promote
                           │         └── NO  → alert, fix
                           │
                           ├─→ Staging (automatic)
                           │    └─→ Integration + E2E tests pass?
                           │         ├── YES → promote
                           │         └── NO  → alert, fix
                           │
                           └─→ Production (manual gate + auto-canary)
                                └─→ Canary metrics healthy?
                                     ├── YES → full rollout
                                     └── NO  → auto rollback

CRITICAL:
- The SAME artifact is deployed to all environments
- NEVER rebuild for a different environment
- Configuration is injected at deploy time, not build time
```

---

## 10. Security Automation (MANDATORY)

### A. Supply Chain Security

```
SOFTWARE SUPPLY CHAIN REQUIREMENTS:

1. Dependency Management
   ├── Lock files committed and verified
   ├── Automated dependency updates (Dependabot, Renovate)
   ├── SCA scan on every build
   ├── License compliance check
   └── No direct downloads from untrusted sources

2. Build Integrity
   ├── Reproducible builds (same input = same output)
   ├── Build in isolated, ephemeral environment
   ├── SBOM (Software Bill of Materials) generated
   ├── Build provenance attestation (SLSA Level 2+)
   └── Signed artifacts (container images, binaries)

3. Artifact Security
   ├── Private registries with access control
   ├── Image scanning before deployment
   ├── Admission control (only signed/scanned images deploy)
   ├── Immutable tags for production releases
   └── Vulnerability alerts for deployed images

4. Runtime Protection
   ├── Network policies (deny by default)
   ├── Pod security standards (restricted)
   ├── Seccomp and AppArmor profiles
   ├── Runtime threat detection
   └── Automated patching for critical CVEs
```

### B. Security Scanning Pipeline

```yaml
# Security scanning stages (platform-agnostic specification)

security_pipeline:
  # Run in parallel with build for speed
  parallel: true

  stages:
    # Static Application Security Testing
    sast:
      tools: [semgrep, codeql, sonarqube]
      fail_on: critical, high
      report_format: sarif

    # Software Composition Analysis
    sca:
      tools: [snyk, trivy, npm-audit]
      fail_on: critical
      report_format: sarif

    # Secret Detection
    secrets:
      tools: [gitleaks, trufflehog]
      fail_on: any_detection
      scan_scope: full_history  # Not just current commit

    # Container Scanning
    container:
      tools: [trivy, grype]
      fail_on: critical, high
      scan_base_image: true
      scan_application_deps: true

    # Infrastructure as Code Scanning
    iac:
      tools: [checkov, tfsec, kics]
      fail_on: critical, high
      policy_as_code: true

    # License Compliance
    license:
      allowed: [MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC]
      denied: [GPL-3.0, AGPL-3.0]  # Adjust per project
      fail_on: denied_license
```

### C. Security Policy as Code

```rego
# policy/deployment.rego - OPA policy for deployments

package deployment

# Deny containers running as root
deny[msg] {
    input.spec.containers[_].securityContext.runAsRoot == true
    msg := "Containers must not run as root"
}

# Deny containers without resource limits
deny[msg] {
    container := input.spec.containers[_]
    not container.resources.limits.memory
    msg := sprintf("Container '%s' missing memory limit", [container.name])
}

deny[msg] {
    container := input.spec.containers[_]
    not container.resources.limits.cpu
    msg := sprintf("Container '%s' missing CPU limit", [container.name])
}

# Deny images without digest
deny[msg] {
    container := input.spec.containers[_]
    not contains(container.image, "@sha256:")
    not startswith(container.image, "registry.internal/")
    msg := sprintf("Container '%s' image must use digest pinning", [container.name])
}

# Deny containers without health checks
deny[msg] {
    container := input.spec.containers[_]
    not container.livenessProbe
    msg := sprintf("Container '%s' missing liveness probe", [container.name])
}

# Deny privileged containers
deny[msg] {
    container := input.spec.containers[_]
    container.securityContext.privileged == true
    msg := sprintf("Container '%s' must not be privileged", [container.name])
}

# Require specific labels
deny[msg] {
    not input.metadata.labels.team
    msg := "Deployment must have 'team' label"
}

deny[msg] {
    not input.metadata.labels.environment
    msg := "Deployment must have 'environment' label"
}
```

---

## 11. Disaster Recovery & Business Continuity (MANDATORY)

### A. Backup Strategy

```
BACKUP REQUIREMENTS:

1. Data Backups
   ├── Automated, scheduled backups (at least daily)
   ├── Encrypted at rest (KMS-managed keys)
   ├── Stored in different region/zone than source
   ├── Retention: 30 days daily, 12 months monthly
   ├── Integrity verification (checksums)
   └── Tested restores: monthly minimum

2. Infrastructure State
   ├── IaC state files backed up and versioned
   ├── Secrets vault backed up separately
   ├── Configuration backed up with infrastructure
   └── Documentation kept in version control

3. Recovery Objectives
   ├── RTO (Recovery Time Objective): Define per service
   ├── RPO (Recovery Point Objective): Define per data store
   ├── Recovery procedures documented and tested
   └── Runbooks for every failure scenario
```

### B. Disaster Recovery Testing

```bash
# Quarterly DR drill checklist

DR_DRILL_CHECKLIST:
  - [ ] Restore database from backup in isolated environment
  - [ ] Verify data integrity after restore
  - [ ] Deploy application to DR region
  - [ ] Verify all services are functional
  - [ ] Test failover and failback
  - [ ] Measure actual RTO and RPO
  - [ ] Document any issues found
  - [ ] Update runbooks with lessons learned
  - [ ] Review and update DR plan
```

---

## 12. Documentation (MANDATORY)

### A. Operational Documentation

**Every service MUST have these documents:**

```
Documentation Requirements:
├── README.md
│   ├── What the service does
│   ├── Architecture overview
│   ├── Quick start guide
│   ├── Configuration reference
│   └── Links to other docs
│
├── RUNBOOK.md
│   ├── Common operational procedures
│   ├── Troubleshooting guide
│   ├── Alert response procedures
│   ├── Scaling procedures
│   └── Disaster recovery steps
│
├── ARCHITECTURE.md (or ADRs)
│   ├── System architecture diagram
│   ├── Data flow diagrams
│   ├── Key design decisions
│   └── Dependency map
│
├── DEPLOYMENT.md
│   ├── Deployment process
│   ├── Environment configuration
│   ├── Rollback procedures
│   └── Feature flag management
│
└── SECURITY.md
    ├── Security contacts
    ├── Vulnerability reporting process
    ├── Security scanning results
    └── Compliance requirements
```

### B. Runbook Template

```markdown
# Runbook: [Service Name]

## Service Overview
- **Owner**: [Team]
- **On-call**: [Rotation link]
- **Dashboard**: [URL]
- **Logs**: [URL]

## Common Alerts

### Alert: High Error Rate
**Severity**: SEV2
**Description**: Error rate exceeds 1% for 5 minutes
**Steps**:
1. Check dashboard: [URL]
2. Check recent deployments: `git log --since="1 hour ago" --oneline`
3. Check dependency health: [URL]
4. If recent deployment: Rollback with `[rollback command]`
5. If dependency issue: Check status page, enable circuit breaker
6. Escalate to [team] if unresolved in 15 minutes

### Alert: High Latency
...

## Scaling Procedures

### Scale Up
```
[specific commands for your platform]
```

### Scale Down
```
[specific commands for your platform]
```

## Disaster Recovery
1. [Step-by-step recovery procedure]
2. [Verification steps]
```

---

## 13. Deployment Checklist

### Pre-Deployment Verification (MANDATORY)

#### Pipeline & Automation
- [ ] All pipeline stages pass (lint, test, build, security, deploy)
- [ ] No manual steps required for deployment
- [ ] Rollback mechanism tested and functional
- [ ] Deployment is idempotent (safe to re-run)

#### Testing
- [ ] Unit tests pass: ≥80% coverage
- [ ] Integration tests pass
- [ ] Security scans pass: zero critical/high findings
- [ ] Smoke tests defined for post-deployment verification
- [ ] Regression tests for any bug fixes included

#### Security
- [ ] No hardcoded secrets in code or configuration
- [ ] Container images scanned and signed
- [ ] Dependencies scanned, no known critical CVEs
- [ ] IaC scanned, no misconfigurations
- [ ] Access controls verified (least privilege)
- [ ] Secrets rotated if compromised

#### Infrastructure
- [ ] IaC validated and planned
- [ ] No unexpected infrastructure changes
- [ ] Resource limits set on all containers
- [ ] Health checks configured
- [ ] Auto-scaling policies defined
- [ ] Backup verified

#### Observability
- [ ] Metrics emitted (RED and USE)
- [ ] Structured logging configured
- [ ] Distributed tracing enabled
- [ ] Alerts configured with runbook links
- [ ] Dashboards updated

#### Documentation
- [ ] README current
- [ ] Runbook updated
- [ ] Architecture diagrams current
- [ ] CHANGELOG updated
- [ ] Breaking changes communicated

#### Release
- [ ] Semantic version bumped appropriately
- [ ] Git tag created and signed
- [ ] Release notes generated
- [ ] Stakeholders notified

---

## 14. Why This Configuration Works

**Full Automation**:
- Eliminates human error, the #1 cause of outages
- Enables rapid, confident deployments (multiple per day)
- Frees engineers to solve problems instead of running scripts

**Security at Every Layer**:
- Shift-left catches vulnerabilities before they reach production
- Supply chain security prevents compromised dependencies
- Zero-trust and least-privilege minimize blast radius of breaches

**Quality Gates**:
- Automated gates prevent bad code from advancing
- Coverage thresholds ensure adequate testing
- Policy as code enforces organizational standards

**Observability**:
- Three pillars (metrics, logs, traces) enable rapid incident response
- Alerting on symptoms catches real problems, not noise
- Runbooks enable anyone to respond to incidents

**Environment Parity**:
- Same artifact in all environments eliminates "works on my machine"
- Configuration injection separates code from environment
- Promotion flow ensures only tested artifacts reach production

**Immutable Infrastructure**:
- Replace, don't patch — eliminates configuration drift
- Rollback is just deploying the previous artifact
- Reproducible environments from version-controlled IaC

**Simplicity**:
- Boring, proven technology over cutting-edge complexity
- Golden paths over infinite flexibility
- Guardrails over gates — empower teams safely

---

## 15. Quick Reference

### Common Commands

```bash
# ═══════════════════════════════════════════════════════════════
# INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════

# Validate IaC
terraform fmt -check -recursive
terraform validate
terraform plan -out=plan.tfplan

# Apply IaC
terraform apply plan.tfplan

# Detect drift
terraform plan -detailed-exitcode

# ═══════════════════════════════════════════════════════════════
# CONTAINERS
# ═══════════════════════════════════════════════════════════════

# Build and scan
docker build -t myapp:$(git rev-parse --short HEAD) .
trivy image myapp:$(git rev-parse --short HEAD)
hadolint Dockerfile

# Sign image
cosign sign --key cosign.key registry.example.com/myapp:v1.2.3

# ═══════════════════════════════════════════════════════════════
# SECURITY
# ═══════════════════════════════════════════════════════════════

# Scan for secrets
gitleaks detect --source . --verbose

# Scan dependencies
npm audit --audit-level=high
trivy fs --scanners vuln .

# Scan IaC
checkov -d .
tfsec .

# Policy check
conftest test . --policy policy/

# ═══════════════════════════════════════════════════════════════
# DEPLOYMENT
# ═══════════════════════════════════════════════════════════════

# Deploy (platform-specific, examples)
kubectl rollout status deployment/myapp
kubectl rollout undo deployment/myapp    # Rollback

# Health check
curl -sf https://myapp.example.com/health
curl -sf https://myapp.example.com/ready
```

### Pipeline Stage Summary

| Stage | Purpose | Gate Criteria | Runs On |
|-------|---------|---------------|---------|
| Validate | Lint, format, syntax check | Zero errors | Every push |
| Test | Unit, integration, coverage | ≥80% coverage, 100% pass | Every push |
| Build | Compile, package, tag | Successful build | Every push |
| Security | SAST, SCA, secrets, container scan | Zero critical/high | Every push |
| Deploy:Dev | Deploy to development | Smoke tests pass | Push to main/develop |
| Deploy:Staging | Deploy to staging | Integration + E2E pass | Push to main |
| Deploy:Prod | Deploy to production | Manual approval + canary | Tag / manual |
| Verify | Smoke tests, health checks | All checks pass | After every deploy |

### Deployment Strategy Summary

| Strategy | Risk | Speed | Complexity | Use When |
|----------|------|-------|------------|----------|
| Rolling | Low | Fast | Low | Default for most services |
| Blue-Green | Very Low | Fast | Medium | Zero-downtime required |
| Canary | Very Low | Slow | High | High-risk or high-traffic |
| Feature Flag | Very Low | Instant | Medium | Gradual feature rollout |
| Recreate | High | Fast | Low | Dev/test environments only |

### Observability Checklist

```
For every service:
[ ] RED metrics: Rate, Errors, Duration
[ ] USE metrics: Utilization, Saturation, Errors
[ ] Structured JSON logs with correlation ID
[ ] Distributed tracing with W3C context
[ ] /health endpoint (liveness)
[ ] /ready endpoint (readiness)
[ ] /metrics endpoint (Prometheus)
[ ] Alerts with severity levels and runbook links
[ ] Dashboard with key business and technical metrics
```

---

## References

- [DORA Metrics](https://dora.dev/) - DevOps Research and Assessment
- [The Twelve-Factor App](https://12factor.net/) - Methodology for building SaaS apps
- [SLSA Framework](https://slsa.dev/) - Supply chain Levels for Software Artifacts
- [OpenTelemetry](https://opentelemetry.io/) - Observability framework
- [OWASP DevSecOps Guideline](https://owasp.org/www-project-devsecops-guideline/)
- [GitOps Principles](https://opengitops.dev/) - OpenGitOps standards
- [CIS Benchmarks](https://www.cisecurity.org/cis-benchmarks) - Security configuration guides
- [SRE Book](https://sre.google/sre-book/table-of-contents/) - Google Site Reliability Engineering

---

**Last Updated:** 2026-03-15
**Version:** 1.0
**Maintainer:** DevOps Team


**End of DevOps Engineering Guidelines**
