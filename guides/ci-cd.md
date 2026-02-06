# CI/CD Pipeline Guidelines
Mandatory standards for implementing continuous integration and continuous deployment pipelines. GitHub Actions, GitLab CI, Jenkins, CircleCI, ArgoCD, Flux.

---

**Agent Profile**: The CI/CD Expert
**Role**: Senior DevOps Engineer & Release Manager
**Objective**: Generate reliable, fast, and secure CI/CD pipelines that enable rapid and safe deployments.
**Tools**: GitHub Actions, GitLab CI, Jenkins, CircleCI, ArgoCD, Flux.

---

## 1. Core Philosophies: CICD-FIRST

- **C**ontinuous: Every commit triggers the pipeline
- **I**ncremental: Small, frequent deployments
- **C**onsistent: Same process for all environments
- **D**ependable: Automated testing and rollback

### Mandatory Security & Secret Handling

- **No secrets** stored in pipelines, code, or transported to agents.
- Prefer **secretless authentication** (managed identities, IAM roles, service accounts, OIDC/workload identity).
- If secretless auth is not possible, secrets **must** be stored in a vault/secret store and retrieved at runtime based on the environment.
- Never hardcode or echo secrets; avoid passing secrets via CLI arguments.

---

## 2. Pipeline Structure (MANDATORY)

- Requirements are **platform-agnostic** (apply to any CI/CD system) and **language-agnostic** (apply to any stack).

### A. Standard Stages

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  # Stage 1: Build and Lint
  build:
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

      - name: Lint
        run: npm run lint

      - name: Type check
        run: npm run typecheck

      - name: Build
        run: npm run build

      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: build
          path: dist/
          retention-days: 1

  # Stage 2: Test
  test:
    needs: build
    runs-on: ubuntu-latest
    strategy:
      matrix:
        shard: [1, 2, 3, 4]
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test -- --shard=${{ matrix.shard }}/4

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}

  # Stage 3: Security Scan
  security:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Snyk security scan
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}

      - name: Run SAST scan
        uses: github/codeql-action/analyze@v3

  # Stage 4: Deploy to Staging
  deploy-staging:
    needs: [test, security]
    if: github.ref == 'refs/heads/develop'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4

      - name: Download build artifacts
        uses: actions/download-artifact@v4
        with:
          name: build
          path: dist/

      - name: Deploy to staging
        run: ./scripts/deploy.sh staging
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

  # Stage 5: Integration Tests
  integration-tests:
    needs: deploy-staging
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run integration tests
        run: npm run test:integration
        env:
          API_URL: https://staging-api.example.com

  # Stage 6: Deploy to Production
  deploy-production:
    needs: [test, security]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4

      - name: Download build artifacts
        uses: actions/download-artifact@v4
        with:
          name: build
          path: dist/

      - name: Deploy to production
        run: ./scripts/deploy.sh production
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Notify deployment
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "Deployed ${{ github.sha }} to production"
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### B. GitLab CI Example

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - security
  - deploy-staging
  - integration
  - deploy-production

variables:
  DOCKER_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

# Build Stage
build:
  stage: build
  image: node:20-alpine
  script:
    - npm ci
    - npm run build
  artifacts:
    paths:
      - dist/
    expire_in: 1 day
  cache:
    key: ${CI_COMMIT_REF_SLUG}
    paths:
      - node_modules/

# Test Stage
test:
  stage: test
  image: node:20-alpine
  needs: [build]
  parallel: 4
  script:
    - npm ci
    - npm test -- --shard=$CI_NODE_INDEX/$CI_NODE_TOTAL
  coverage: '/Lines\s*:\s*(\d+\.?\d*)%/'
  artifacts:
    reports:
      junit: junit.xml
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml

# Security Stage
security:
  stage: security
  needs: [build]
  image:
    name: snyk/snyk:node
    entrypoint: [""]
  script:
    - snyk test
    - snyk monitor
  allow_failure: true

sast:
  stage: security
  needs: [build]

# Deploy to Staging
deploy-staging:
  stage: deploy-staging
  needs: [test, security]
  image: alpine:latest
  script:
    - ./scripts/deploy.sh staging
  environment:
    name: staging
    url: https://staging.example.com
  only:
    - develop

# Integration Tests
integration:
  stage: integration
  needs: [deploy-staging]
  script:
    - npm run test:integration
  only:
    - develop

# Deploy to Production
deploy-production:
  stage: deploy-production
  needs: [test, security]
  script:
    - ./scripts/deploy.sh production
  environment:
    name: production
    url: https://example.com
  when: manual
  only:
    - main
```

---

## 3. Testing in CI (MANDATORY)

### A. Test Matrix

```yaml
jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        node: [18, 20, 22]
        exclude:
          - os: macos-latest
            node: 18
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js ${{ matrix.node }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node }}

      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test
```

### B. Database Testing

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - name: Run migrations
        run: npm run db:migrate
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test

      - name: Run tests
        run: npm test
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test
          REDIS_URL: redis://localhost:6379
```

### C. E2E Testing

```yaml
jobs:
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies
        run: npm ci

      - name: Install Playwright
        run: npx playwright install --with-deps

      - name: Build application
        run: npm run build

      - name: Start application
        run: npm run start &
        env:
          NODE_ENV: test

      - name: Wait for application
        run: npx wait-on http://localhost:3000

      - name: Run E2E tests
        run: npm run test:e2e

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: playwright-report
          path: playwright-report/
          retention-days: 30
```

---

## 4. Docker Builds (MANDATORY)

### A. Multi-Stage Build

```dockerfile
# Dockerfile
# Stage 1: Dependencies
FROM node:20-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

# Stage 2: Build
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 3: Production
FROM node:20-alpine AS production
WORKDIR /app

# Security: Don't run as root
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nodejs -u 1001

# Copy only necessary files
COPY --from=deps --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=build --chown=nodejs:nodejs /app/dist ./dist
COPY --from=build --chown=nodejs:nodejs /app/package.json ./

USER nodejs
EXPOSE 3000
ENV NODE_ENV=production

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

CMD ["node", "dist/index.js"]
```

### B. Docker Build Pipeline

```yaml
jobs:
  build-image:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=sha,prefix=
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=raw,value=latest,enable=${{ github.ref == 'refs/heads/main' }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64

      - name: Scan image
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ghcr.io/${{ github.repository }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload scan results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'
```

---

## 5. Deployment Strategies (MANDATORY)

### A. Blue-Green Deployment

```yaml
# deploy-blue-green.yml
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to green environment
        run: |
          aws ecs update-service \
            --cluster production \
            --service app-green \
            --task-definition app:${{ github.sha }}

      - name: Wait for deployment
        run: |
          aws ecs wait services-stable \
            --cluster production \
            --services app-green

      - name: Run smoke tests
        run: ./scripts/smoke-test.sh https://green.example.com

      - name: Switch traffic to green
        run: |
          aws elbv2 modify-listener \
            --listener-arn ${{ secrets.ALB_LISTENER_ARN }} \
            --default-actions Type=forward,TargetGroupArn=${{ secrets.GREEN_TG_ARN }}

      - name: Verify production
        run: ./scripts/smoke-test.sh https://example.com

      - name: Rollback on failure
        if: failure()
        run: |
          aws elbv2 modify-listener \
            --listener-arn ${{ secrets.ALB_LISTENER_ARN }} \
            --default-actions Type=forward,TargetGroupArn=${{ secrets.BLUE_TG_ARN }}
```

### B. Canary Deployment

```yaml
# deploy-canary.yml
jobs:
  deploy-canary:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy canary (10% traffic)
        run: |
          kubectl apply -f k8s/canary.yaml
          kubectl set image deployment/app-canary app=myimage:${{ github.sha }}

      - name: Wait for canary
        run: kubectl rollout status deployment/app-canary

      - name: Monitor canary metrics
        run: |
          # Check error rate for 10 minutes
          ./scripts/monitor-canary.sh --duration 600 --threshold 0.01

      - name: Promote to production
        if: success()
        run: |
          kubectl set image deployment/app app=myimage:${{ github.sha }}
          kubectl rollout status deployment/app

      - name: Rollback canary on failure
        if: failure()
        run: kubectl rollout undo deployment/app-canary
```

### C. Rolling Deployment

```yaml
# Kubernetes rolling update
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  replicas: 4
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
        - name: app
          image: myimage:latest
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 15
            periodSeconds: 10
```

---

## 6. Environment Management (MANDATORY)

### A. GitHub Environments

```yaml
jobs:
  deploy-staging:
    environment:
      name: staging
      url: https://staging.example.com
    runs-on: ubuntu-latest
    steps:
      - name: Deploy
        run: ./deploy.sh
        env:
          API_KEY: ${{ secrets.STAGING_API_KEY }}

  deploy-production:
    needs: deploy-staging
    environment:
      name: production
      url: https://example.com
    runs-on: ubuntu-latest
    steps:
      - name: Deploy
        run: ./deploy.sh
        env:
          API_KEY: ${{ secrets.PRODUCTION_API_KEY }}
```

### B. Environment Variables

```yaml
# Use repository secrets for sensitive data
# Use environment variables for configuration

jobs:
  deploy:
    runs-on: ubuntu-latest
    env:
      # Non-sensitive configuration
      NODE_ENV: production
      LOG_LEVEL: info
    steps:
      - name: Deploy
        run: ./deploy.sh
        env:
          # Sensitive data from secrets
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          API_KEY: ${{ secrets.API_KEY }}
```

---

## 7. Security Scanning (MANDATORY)

### A. Dependency Scanning

```yaml
jobs:
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # NPM audit
      - name: Run npm audit
        run: npm audit --audit-level=high

      # Snyk scan
      - name: Run Snyk
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high

      # Dependabot alerts
      - name: Check Dependabot alerts
        uses: actions/github-script@v7
        with:
          script: |
            const alerts = await github.rest.dependabot.listAlertsForRepo({
              owner: context.repo.owner,
              repo: context.repo.repo,
              state: 'open',
              severity: 'critical,high'
            });
            if (alerts.data.length > 0) {
              core.setFailed(`Found ${alerts.data.length} critical/high Dependabot alerts`);
            }
```

### B. SAST/DAST

```yaml
jobs:
  sast:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    steps:
      - uses: actions/checkout@v4

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: javascript

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3

  dast:
    needs: deploy-staging
    runs-on: ubuntu-latest
    steps:
      - name: OWASP ZAP Scan
        uses: zaproxy/action-full-scan@v0.10.0
        with:
          target: 'https://staging.example.com'
          rules_file_name: '.zap/rules.tsv'
```

---

## 8. Monitoring and Notifications (MANDATORY)

### A. Deployment Notifications

```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy
        id: deploy
        run: ./deploy.sh

      - name: Notify success
        if: success()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "✅ *Deployment Successful*\n*Repo:* ${{ github.repository }}\n*Branch:* ${{ github.ref_name }}\n*Commit:* ${{ github.sha }}"
                  }
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}

      - name: Notify failure
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "❌ *Deployment Failed*\n*Repo:* ${{ github.repository }}\n*Branch:* ${{ github.ref_name }}\n*Commit:* ${{ github.sha }}\n*Workflow:* ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"
                  }
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### B. Metrics Collection

```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Record deployment start
        run: |
          curl -X POST "${{ secrets.METRICS_ENDPOINT }}/deployments" \
            -H "Content-Type: application/json" \
            -d '{
              "repo": "${{ github.repository }}",
              "sha": "${{ github.sha }}",
              "environment": "production",
              "status": "started"
            }'

      - name: Deploy
        run: ./deploy.sh

      - name: Record deployment result
        if: always()
        run: |
          curl -X POST "${{ secrets.METRICS_ENDPOINT }}/deployments" \
            -H "Content-Type: application/json" \
            -d '{
              "repo": "${{ github.repository }}",
              "sha": "${{ github.sha }}",
              "environment": "production",
              "status": "${{ job.status }}"
            }'
```

---

## 9. Deployment Checklist

### Pipeline Quality
- [ ] All tests pass before deploy
- [ ] Security scans integrated
- [ ] Build artifacts cached
- [ ] Parallel jobs where possible

### Deployment Safety
- [ ] Environment approvals configured
- [ ] Rollback mechanism tested
- [ ] Health checks implemented
- [ ] Canary/blue-green strategy

### Security
- [ ] Secrets properly managed
- [ ] Least privilege permissions
- [ ] Image scanning enabled
- [ ] Dependency updates automated

### Monitoring
- [ ] Deployment notifications
- [ ] Metrics collection
- [ ] Error alerting
- [ ] Audit logging

---

## 10. Quick Reference

```yaml
# Common GitHub Actions triggers
on:
  push:
    branches: [main]
    paths: ['src/**']
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'
  workflow_dispatch:
  release:
    types: [published]

# Useful actions
actions/checkout@v4
actions/setup-node@v4
actions/upload-artifact@v4
actions/download-artifact@v4
docker/build-push-action@v5
aws-actions/configure-aws-credentials@v4

# Job dependencies
needs: [build, test]

# Conditions
if: github.ref == 'refs/heads/main'
if: success()
if: failure()
if: always()
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** DevOps Team


**End of CI/CD Pipeline Guidelines**
