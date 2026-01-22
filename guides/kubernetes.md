# Kubernetes Best Practices Guidelines

This document provides mandatory standards and development practices for secure, production-ready Kubernetes deployments with emphasis on security, observability, resilience, and maintainability. This guide is cloud-agnostic and applies to any Kubernetes distribution (on-premises, managed, or self-hosted).

---

**Agent Profile**: The Kubernetes Platform Engineer
**Role**: Senior Platform Engineer & Cloud-Native Security Specialist
**Objective**: Generate production-ready, secure, observable, and maintainable Kubernetes configurations with Istio service mesh, proper resource management, and GitOps deployment practices.
**Tools**: Kubernetes 1.28+, Istio 1.20+, Helm 3.x, Kustomize, kubectl, Message Brokers (Kafka/RabbitMQ/Redis/NATS).

---

## 1. Core Philosophies: KUBERNETES

The agent must adhere to the **KUBERNETES** standard for every deployment:

- **K**ept Secure: Security by default, least privilege, zero trust
- **U**nified Mesh: Istio service mesh for traffic management, security, observability
- **B**ounded Resources: Resource requests/limits, quotas, LimitRanges
- **E**vent-Driven: Message brokers for async communication between services
- **R**esilient Design: Health probes, PDBs, anti-affinity, graceful shutdown
- **N**amespace Isolation: Logical separation, RBAC, NetworkPolicies
- **E**xternalized Config: ConfigMaps, Secrets, environment-specific overlays
- **T**ested Manifests: Validated YAML, policy checks, dry-run before apply
- **E**phemeral Mindset: Stateless services, external state, immutable deployments
- **S**calable Architecture: HPA, VPA, cluster autoscaling, right-sized pods

**Additional Principles:**

- **GitOps**: All manifests version-controlled, declarative, automated sync
- **Infrastructure as Code**: No manual kubectl apply in production
- **Observability Built-in**: Metrics, logs, traces via Istio and standard tooling
- **Defense in Depth**: Multiple security layers (network, pod, container, runtime)

**Verified Deployments**: Agent-generated manifests MUST pass validation, security scanning, and dry-run before delivery.

---

## 2. Namespace Organization (MANDATORY)

### A. Namespace Strategy

**CRITICAL: Organize namespaces by team, environment, or bounded context.**

```yaml
# Namespace Structure
namespaces/
├── system/                    # Platform components
│   ├── istio-system          # Istio control plane
│   ├── monitoring            # Prometheus, Grafana, Alertmanager
│   ├── logging               # Fluent Bit, Loki, Elasticsearch
│   ├── cert-manager          # Certificate management
│   └── secrets-management    # External Secrets, Vault
├── infrastructure/            # Shared infrastructure
│   ├── message-broker        # Kafka/RabbitMQ/Redis
│   └── databases             # Shared database operators
└── applications/              # Application workloads
    ├── team-orders           # Order team services
    ├── team-payments         # Payment team services
    └── team-users            # User team services
```

### B. Namespace Configuration

```yaml
# ✅ CORRECT - Well-configured namespace
apiVersion: v1
kind: Namespace
metadata:
  name: team-orders
  labels:
    # Istio injection
    istio-injection: enabled
    # Pod security standard
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
    # Organization labels
    team: orders
    environment: production
    cost-center: engineering
  annotations:
    # Contact information
    owner: orders-team@company.com
    slack-channel: "#team-orders"
---
# Resource Quota
apiVersion: v1
kind: ResourceQuota
metadata:
  name: team-orders-quota
  namespace: team-orders
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 40Gi
    limits.cpu: "40"
    limits.memory: 80Gi
    pods: "100"
    services: "20"
    secrets: "50"
    configmaps: "50"
    persistentvolumeclaims: "20"
---
# LimitRange for defaults
apiVersion: v1
kind: LimitRange
metadata:
  name: team-orders-limits
  namespace: team-orders
spec:
  limits:
    - type: Container
      default:
        cpu: 500m
        memory: 512Mi
      defaultRequest:
        cpu: 100m
        memory: 128Mi
      min:
        cpu: 50m
        memory: 64Mi
      max:
        cpu: 2000m
        memory: 4Gi
    - type: PersistentVolumeClaim
      min:
        storage: 1Gi
      max:
        storage: 100Gi
```

### C. Namespace Rules

```
NAMESPACE REQUIREMENTS:

□ Naming Convention
  □ Lowercase, alphanumeric, hyphens only
  □ Format: {team}-{context} or {environment}-{app}
  □ Examples: team-orders, prod-payments, staging-users

□ Labels (MANDATORY)
  □ istio-injection: enabled/disabled
  □ pod-security.kubernetes.io/enforce: restricted
  □ team: <team-name>
  □ environment: <env>

□ Resource Management
  □ ResourceQuota defined
  □ LimitRange with defaults
  □ Prevents resource exhaustion

□ Access Control
  □ RBAC roles scoped to namespace
  □ NetworkPolicies restricting traffic
  □ ServiceAccount per application
```

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new Kubernetes manifests and configurations.**

### TDD Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    TDD CYCLE FOR KUBERNETES                      │
│                                                                  │
│    ┌──────────┐                                                 │
│    │   RED    │◀────────────────────────────────────┐           │
│    │          │                                      │           │
│    │ Write a  │                                      │           │
│    │ failing  │                                      │           │
│    │  test    │                                      │           │
│    └────┬─────┘                                      │           │
│         │                                            │           │
│         ▼                                            │           │
│    ┌──────────┐                                      │           │
│    │  GREEN   │                                      │           │
│    │          │                                      │           │
│    │  Write   │                                      │           │
│    │ minimal  │                                      │           │
│    │ manifest │                                      │           │
│    └────┬─────┘                                      │           │
│         │                                            │           │
│         ▼                                            │           │
│    ┌──────────┐                                      │           │
│    │ REFACTOR │──────────────────────────────────────┘           │
│    │          │                                                  │
│    │ Improve  │                                                  │
│    │ keeping  │                                                  │
│    │  tests   │                                                  │
│    │  green   │                                                  │
│    └──────────┘                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Example TDD Workflow for Kubernetes Manifests

```yaml
# Step 1: RED - Write failing validation test first
# File: tests/deployment_test.yaml (using kubeconform/conftest)

# conftest policy: policy/deployment.rego
package main

deny[msg] {
  input.kind == "Deployment"
  not input.spec.template.spec.securityContext.runAsNonRoot
  msg := "Deployment must have runAsNonRoot: true"
}

deny[msg] {
  input.kind == "Deployment"
  container := input.spec.template.spec.containers[_]
  not container.resources.limits.memory
  msg := sprintf("Container %s must have memory limits", [container.name])
}

deny[msg] {
  input.kind == "Deployment"
  container := input.spec.template.spec.containers[_]
  container.securityContext.allowPrivilegeEscalation == true
  msg := sprintf("Container %s must not allow privilege escalation", [container.name])
}

# Run: conftest test deployment.yaml --policy policy/
# FAILS - deployment.yaml doesn't exist yet or fails policies
```

```yaml
# Step 2: GREEN - Write minimal manifest to pass tests
# File: deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
  namespace: team-orders
spec:
  replicas: 1
  selector:
    matchLabels:
      app: order-service
  template:
    metadata:
      labels:
        app: order-service
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
      containers:
        - name: order-service
          image: registry.company.com/orders/order-service:v1.0.0
          securityContext:
            allowPrivilegeEscalation: false
          resources:
            requests:
              cpu: 100m
              memory: 256Mi
            limits:
              cpu: 500m
              memory: 512Mi

# Run: conftest test deployment.yaml --policy policy/
# PASSES - all policies satisfied
```

```yaml
# Step 3: REFACTOR - Add production-ready configurations
# File: deployment.yaml (enhanced)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
  namespace: team-orders
  labels:
    app: order-service
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: order-service
  template:
    metadata:
      labels:
        app: order-service
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
    spec:
      serviceAccountName: order-service
      automountServiceAccountToken: false
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        runAsGroup: 10001
        fsGroup: 10001
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: order-service
          image: registry.company.com/orders/order-service:v1.0.0
          imagePullPolicy: Always
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop:
                - ALL
          resources:
            requests:
              cpu: 100m
              memory: 256Mi
            limits:
              cpu: 500m
              memory: 512Mi
          livenessProbe:
            httpGet:
              path: /health/live
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
          ports:
            - name: http
              containerPort: 8080
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app: order-service
                topologyKey: kubernetes.io/hostname
      terminationGracePeriodSeconds: 30

# Run: conftest test deployment.yaml --policy policy/
# PASSES - tests still green after refactoring
```

### Visual Step-by-Step TDD Example

```
TDD WORKFLOW FOR HELM CHARTS:

Step 1: RED - Define expected behavior
┌─────────────────────────────────────────────────────────────────┐
│  # Write helm unittest tests first                              │
│  # tests/deployment_test.yaml                                   │
│  suite: deployment test                                         │
│  templates:                                                     │
│    - deployment.yaml                                            │
│  tests:                                                         │
│    - it: should set security context                            │
│      asserts:                                                   │
│        - equal:                                                 │
│            path: spec.template.spec.securityContext.runAsNonRoot│
│            value: true                                          │
│    - it: should have resource limits                            │
│      asserts:                                                   │
│        - isNotNull:                                             │
│            path: spec.template.spec.containers[0].resources.limits│
│                                                                 │
│  $ helm unittest ./mychart                                      │
│  FAIL - templates/deployment.yaml not found                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 2: GREEN - Create minimal chart
┌─────────────────────────────────────────────────────────────────┐
│  # templates/deployment.yaml                                    │
│  apiVersion: apps/v1                                            │
│  kind: Deployment                                               │
│  metadata:                                                      │
│    name: {{ .Release.Name }}                                    │
│  spec:                                                          │
│    template:                                                    │
│      spec:                                                      │
│        securityContext:                                         │
│          runAsNonRoot: true                                     │
│        containers:                                              │
│          - name: app                                            │
│            resources:                                           │
│              limits:                                            │
│                cpu: 500m                                        │
│                memory: 512Mi                                    │
│                                                                 │
│  $ helm unittest ./mychart                                      │
│  PASS - all tests green                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 3: REFACTOR - Add production features
┌─────────────────────────────────────────────────────────────────┐
│  # Enhance with values.yaml templating                          │
│  # Add more security controls                                   │
│  # Add health probes                                            │
│  # Add pod disruption budget                                    │
│                                                                 │
│  $ helm unittest ./mychart                                      │
│  PASS - tests still green                                       │
│                                                                 │
│  $ helm lint ./mychart                                          │
│  PASS - no linting errors                                       │
│                                                                 │
│  $ helm template ./mychart | kubeconform -strict                │
│  PASS - valid Kubernetes manifests                              │
└─────────────────────────────────────────────────────────────────┘
```

### TDD Testing Tools for Kubernetes

```
KUBERNETES TDD TOOLKIT:

Tool                 │ Purpose                    │ Command
─────────────────────┼────────────────────────────┼──────────────────────────
kubeconform          │ Schema validation          │ kubeconform -strict *.yaml
conftest             │ Policy testing (OPA)       │ conftest test *.yaml
helm unittest        │ Helm chart testing         │ helm unittest ./chart
kuttl                │ E2E K8s testing            │ kubectl kuttl test
kubeval              │ Legacy schema validation   │ kubeval *.yaml
pluto                │ Deprecated API detection   │ pluto detect-files -d .
polaris              │ Best practices audit       │ polaris audit --audit-path .
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every Kubernetes configuration bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    BUG FIX WORKFLOW                              │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 1. Bug Reported/Discovered                                  │ │
│  │    "Pods failing to start due to missing security context" │ │
│  └─────────────────────────┬──────────────────────────────────┘ │
│                            ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 2. Write test that REPRODUCES the bug (test will FAIL)     │ │
│  │    conftest policy to detect missing securityContext       │ │
│  └─────────────────────────┬──────────────────────────────────┘ │
│                            ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 3. Verify the test fails for the right reason              │ │
│  │    $ conftest test deployment.yaml                         │ │
│  │    FAIL - "securityContext.runAsNonRoot required"          │ │
│  └─────────────────────────┬──────────────────────────────────┘ │
│                            ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 4. Fix the bug (make the test pass)                        │ │
│  │    Add securityContext to deployment spec                  │ │
│  └─────────────────────────┬──────────────────────────────────┘ │
│                            ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 5. Verify the test now PASSES                              │ │
│  │    $ conftest test deployment.yaml                         │ │
│  │    PASS - all policies satisfied                           │ │
│  └─────────────────────────┬──────────────────────────────────┘ │
│                            ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 6. Document the bug in test comments (include bug ID)      │ │
│  │    # BUG-1234: Ensure pods have security context           │ │
│  └─────────────────────────┬──────────────────────────────────┘ │
│                            ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 7. Deploy with confidence (regression prevented)           │ │
│  │    Policy now part of CI/CD pipeline                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Example Bug Fix with Regression Test

```yaml
# Bug Report #K8S-1234: Pods OOMKilled due to missing memory limits
# Impact: Production services experiencing random restarts
# Root Cause: Deployment missing resource limits configuration

# Step 1-2: Write test that reproduces the bug
# File: policy/resource-limits.rego
package kubernetes.resources

# BUG-K8S-1234: Ensure all containers have memory limits
deny[msg] {
  input.kind == "Deployment"
  container := input.spec.template.spec.containers[_]
  not container.resources.limits.memory
  msg := sprintf(
    "BUG-K8S-1234: Container '%s' in Deployment '%s' must have memory limits to prevent OOMKilled",
    [container.name, input.metadata.name]
  )
}

# Also check for CPU limits
deny[msg] {
  input.kind == "Deployment"
  container := input.spec.template.spec.containers[_]
  not container.resources.limits.cpu
  msg := sprintf(
    "Container '%s' in Deployment '%s' should have CPU limits",
    [container.name, input.metadata.name]
  )
}

# Ensure requests are also set
deny[msg] {
  input.kind == "Deployment"
  container := input.spec.template.spec.containers[_]
  not container.resources.requests.memory
  msg := sprintf(
    "Container '%s' in Deployment '%s' must have memory requests",
    [container.name, input.metadata.name]
  )
}
```

```bash
# Run: conftest test deployment.yaml --policy policy/
# FAILS - "BUG-K8S-1234: Container 'order-service' must have memory limits"
```

```yaml
# Step 3: Fix the bug by adding resource limits
# File: deployment.yaml (BEFORE - buggy)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
spec:
  template:
    spec:
      containers:
        - name: order-service
          image: registry.company.com/orders/order-service:v1.0.0
          # BUG: Missing resources section caused OOMKilled
```

```yaml
# File: deployment.yaml (AFTER - fixed)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
spec:
  template:
    spec:
      containers:
        - name: order-service
          image: registry.company.com/orders/order-service:v1.0.0
          # FIX for BUG-K8S-1234: Added resource limits
          resources:
            requests:
              cpu: 100m
              memory: 256Mi
            limits:
              cpu: 500m
              memory: 512Mi  # Prevents OOMKilled
```

```bash
# Run: conftest test deployment.yaml --policy policy/
# PASSES - bug fixed, regression prevented

# Add to CI/CD pipeline to prevent future occurrences
# .gitlab-ci.yml or similar
# validate:
#   script:
#     - conftest test manifests/ --policy policy/
```

### Common K8s Configuration Bugs and Regression Tests

```
BUG CATEGORY           │ REGRESSION TEST                         │ POLICY EXAMPLE
───────────────────────┼─────────────────────────────────────────┼─────────────────────────
Missing resource limits│ Check .resources.limits exists          │ deny if no limits.memory
No security context    │ Check runAsNonRoot: true                │ deny if privileged
Latest image tag       │ Check image tag != "latest"             │ deny if tag == "latest"
Missing health probes  │ Check livenessProbe exists              │ deny if no probes
No PDB defined         │ Check PDB exists for Deployment         │ warn if replicas > 1, no PDB
Privileged containers  │ Check privileged: false                 │ deny if privileged == true
Host network access    │ Check hostNetwork: false                │ deny if hostNetwork
Missing namespace      │ Check namespace is specified            │ deny if namespace == "default"
No anti-affinity       │ Check podAntiAffinity for HA            │ warn if replicas > 1, no affinity
Missing labels         │ Check required labels exist             │ deny if missing app label
```

---

## 3. Pod Security (MANDATORY)

### A. Pod Security Standards

**CRITICAL: All pods MUST run with restricted security context.**

```yaml
# ✅ CORRECT - Secure Pod specification
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
  namespace: team-orders
spec:
  replicas: 3
  selector:
    matchLabels:
      app: order-service
  template:
    metadata:
      labels:
        app: order-service
        version: v1.2.3
      annotations:
        # Istio sidecar injection
        sidecar.istio.io/inject: "true"
        # Prometheus scraping
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      # Service account (not default)
      serviceAccountName: order-service
      automountServiceAccountToken: false

      # Security context at pod level
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        runAsGroup: 10001
        fsGroup: 10001
        seccompProfile:
          type: RuntimeDefault

      containers:
        - name: order-service
          image: registry.company.com/orders/order-service:v1.2.3
          imagePullPolicy: Always

          # Container security context
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            runAsNonRoot: true
            runAsUser: 10001
            capabilities:
              drop:
                - ALL

          # Resource management
          resources:
            requests:
              cpu: 100m
              memory: 256Mi
            limits:
              cpu: 500m
              memory: 512Mi

          # Health probes
          livenessProbe:
            httpGet:
              path: /health/live
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3

          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3

          startupProbe:
            httpGet:
              path: /health/startup
              port: 8080
            initialDelaySeconds: 0
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 30

          # Ports
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
            - name: metrics
              containerPort: 9090
              protocol: TCP

          # Environment from ConfigMap and Secrets
          envFrom:
            - configMapRef:
                name: order-service-config
            - secretRef:
                name: order-service-secrets

          # Volume mounts
          volumeMounts:
            - name: tmp
              mountPath: /tmp
            - name: cache
              mountPath: /app/cache

      # Volumes
      volumes:
        - name: tmp
          emptyDir:
            sizeLimit: 100Mi
        - name: cache
          emptyDir:
            sizeLimit: 500Mi

      # Pod scheduling
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app: order-service
                topologyKey: kubernetes.io/hostname

      # Tolerate node issues briefly
      tolerations:
        - key: "node.kubernetes.io/not-ready"
          operator: "Exists"
          effect: "NoExecute"
          tolerationSeconds: 30

      # Graceful termination
      terminationGracePeriodSeconds: 30
```

### B. Security Context Rules

```
POD SECURITY CHECKLIST:

□ Pod Level
  □ runAsNonRoot: true
  □ runAsUser: non-zero UID (e.g., 10001)
  □ runAsGroup: non-zero GID
  □ fsGroup: set for volume permissions
  □ seccompProfile: RuntimeDefault

□ Container Level
  □ allowPrivilegeEscalation: false
  □ readOnlyRootFilesystem: true
  □ runAsNonRoot: true
  □ capabilities.drop: ALL

□ Image Security
  □ Specific tag (never :latest)
  □ Pulled from trusted registry
  □ Scanned for vulnerabilities
  □ Signed and verified

□ Service Account
  □ Dedicated ServiceAccount (not default)
  □ automountServiceAccountToken: false (unless needed)
  □ Minimal RBAC permissions
```

### C. Prohibited Configurations

```yaml
# ❌ WRONG - Insecure configurations

# Never use privileged mode
securityContext:
  privileged: true  # PROHIBITED

# Never run as root
securityContext:
  runAsUser: 0  # PROHIBITED

# Never allow privilege escalation
securityContext:
  allowPrivilegeEscalation: true  # PROHIBITED

# Never use host namespaces
hostNetwork: true    # PROHIBITED
hostPID: true        # PROHIBITED
hostIPC: true        # PROHIBITED

# Never mount host paths (except for specific system pods)
volumes:
  - name: host-root
    hostPath:
      path: /  # PROHIBITED

# Never use latest tag
image: myapp:latest  # PROHIBITED - use specific version
```

---

## 4. Istio Service Mesh (MANDATORY)

### A. Istio Architecture

**CRITICAL: Use Istio as the preferred service mesh for ingress, traffic management, security, and observability.**

```
ISTIO ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────┐
│                         CLUSTER                                  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    istio-system                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │
│  │  │   istiod    │  │ Istio       │  │ Kiali/Jaeger   │  │   │
│  │  │ (Control    │  │ Ingress     │  │ (Observability)│  │   │
│  │  │  Plane)     │  │ Gateway     │  │                │  │   │
│  │  └─────────────┘  └──────┬──────┘  └─────────────────┘  │   │
│  └──────────────────────────┼───────────────────────────────┘   │
│                             │                                    │
│  External Traffic ──────────┘                                    │
│                             │                                    │
│  ┌──────────────────────────┼───────────────────────────────┐   │
│  │                    application namespace                  │   │
│  │                          │                                │   │
│  │    ┌─────────────────────▼─────────────────────────┐     │   │
│  │    │              VirtualService                    │     │   │
│  │    │         (routing rules)                        │     │   │
│  │    └─────────────────────┬─────────────────────────┘     │   │
│  │                          │                                │   │
│  │    ┌─────────────────────▼─────────────────────────┐     │   │
│  │    │           DestinationRule                      │     │   │
│  │    │    (load balancing, circuit breaker)          │     │   │
│  │    └─────────────────────┬─────────────────────────┘     │   │
│  │                          │                                │   │
│  │    ┌──────────┐    ┌─────▼────┐    ┌──────────┐         │   │
│  │    │ Pod      │    │ Pod      │    │ Pod      │         │   │
│  │    │┌────────┐│    │┌────────┐│    │┌────────┐│         │   │
│  │    ││  App   ││    ││  App   ││    ││  App   ││         │   │
│  │    │└────────┘│    │└────────┘│    │└────────┘│         │   │
│  │    │┌────────┐│    │┌────────┐│    │┌────────┐│         │   │
│  │    ││Envoy   ││◀──▶││Envoy   ││◀──▶││Envoy   ││         │   │
│  │    ││Sidecar ││mTLS││Sidecar ││mTLS││Sidecar ││         │   │
│  │    │└────────┘│    │└────────┘│    │└────────┘│         │   │
│  │    └──────────┘    └──────────┘    └──────────┘         │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### B. Istio Gateway Configuration

```yaml
# ✅ CORRECT - Istio Ingress Gateway
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: main-gateway
  namespace: istio-system
spec:
  selector:
    istio: ingressgateway
  servers:
    # HTTPS (primary)
    - port:
        number: 443
        name: https
        protocol: HTTPS
      tls:
        mode: SIMPLE
        credentialName: main-gateway-tls  # Kubernetes secret
      hosts:
        - "api.company.com"
        - "*.api.company.com"
    # HTTP redirect to HTTPS
    - port:
        number: 80
        name: http
        protocol: HTTP
      hosts:
        - "api.company.com"
        - "*.api.company.com"
      tls:
        httpsRedirect: true
---
# VirtualService for routing
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: order-service-routes
  namespace: team-orders
spec:
  hosts:
    - "api.company.com"
  gateways:
    - istio-system/main-gateway
  http:
    # API versioning via path
    - match:
        - uri:
            prefix: /api/v1/orders
      rewrite:
        uri: /orders
      route:
        - destination:
            host: order-service
            port:
              number: 8080
      # Timeouts and retries
      timeout: 30s
      retries:
        attempts: 3
        perTryTimeout: 10s
        retryOn: 5xx,reset,connect-failure,retriable-4xx
      # CORS
      corsPolicy:
        allowOrigins:
          - exact: "https://app.company.com"
        allowMethods:
          - GET
          - POST
          - PUT
          - DELETE
        allowHeaders:
          - authorization
          - content-type
        maxAge: "24h"
    # Canary routing (header-based)
    - match:
        - headers:
            x-canary:
              exact: "true"
          uri:
            prefix: /api/v1/orders
      route:
        - destination:
            host: order-service
            subset: canary
            port:
              number: 8080
```

### C. Destination Rules and Traffic Policies

```yaml
# ✅ CORRECT - DestinationRule with traffic policies
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: order-service
  namespace: team-orders
spec:
  host: order-service
  trafficPolicy:
    # Connection pool settings
    connectionPool:
      tcp:
        maxConnections: 100
        connectTimeout: 5s
      http:
        h2UpgradePolicy: UPGRADE
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
        maxRequestsPerConnection: 100
        maxRetries: 3

    # Load balancing
    loadBalancer:
      simple: LEAST_REQUEST
      localityLbSetting:
        enabled: true
        failoverPriority:
          - "topology.kubernetes.io/zone"

    # Circuit breaker (outlier detection)
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 10s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
      minHealthPercent: 30

    # mTLS
    tls:
      mode: ISTIO_MUTUAL

  # Subsets for canary/blue-green
  subsets:
    - name: stable
      labels:
        version: stable
    - name: canary
      labels:
        version: canary
```

### D. Istio Security Policies

```yaml
# ✅ CORRECT - PeerAuthentication (mTLS)
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: team-orders
spec:
  mtls:
    mode: STRICT  # Enforce mTLS for all traffic
---
# AuthorizationPolicy - Deny all by default
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: deny-all
  namespace: team-orders
spec:
  {}  # Empty spec denies all traffic
---
# AuthorizationPolicy - Allow specific traffic
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: order-service-policy
  namespace: team-orders
spec:
  selector:
    matchLabels:
      app: order-service
  action: ALLOW
  rules:
    # Allow from ingress gateway
    - from:
        - source:
            principals:
              - "cluster.local/ns/istio-system/sa/istio-ingressgateway-service-account"
      to:
        - operation:
            methods: ["GET", "POST", "PUT", "DELETE"]
            paths: ["/orders/*", "/health/*"]

    # Allow from payment service
    - from:
        - source:
            principals:
              - "cluster.local/ns/team-payments/sa/payment-service"
      to:
        - operation:
            methods: ["GET"]
            paths: ["/orders/*"]

    # Allow from message broker namespace
    - from:
        - source:
            namespaces: ["message-broker"]
---
# RequestAuthentication (JWT)
apiVersion: security.istio.io/v1beta1
kind: RequestAuthentication
metadata:
  name: jwt-auth
  namespace: team-orders
spec:
  selector:
    matchLabels:
      app: order-service
  jwtRules:
    - issuer: "https://auth.company.com"
      jwksUri: "https://auth.company.com/.well-known/jwks.json"
      audiences:
        - "order-service"
      forwardOriginalToken: true
```

### E. Service-to-Service Communication

```yaml
# ✅ CORRECT - Internal service communication via Istio

# Service definition
apiVersion: v1
kind: Service
metadata:
  name: order-service
  namespace: team-orders
  labels:
    app: order-service
spec:
  ports:
    - name: http  # Named port required for Istio
      port: 8080
      targetPort: 8080
      protocol: TCP
    - name: grpc
      port: 9090
      targetPort: 9090
      protocol: TCP
  selector:
    app: order-service
---
# VirtualService for internal routing
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: order-service-internal
  namespace: team-orders
spec:
  hosts:
    - order-service  # Short name for same namespace
    - order-service.team-orders.svc.cluster.local
  http:
    - match:
        - port: 8080
      route:
        - destination:
            host: order-service
            port:
              number: 8080
      timeout: 10s
      retries:
        attempts: 3
        perTryTimeout: 3s
        retryOn: 5xx,reset,connect-failure
    - match:
        - port: 9090
      route:
        - destination:
            host: order-service
            port:
              number: 9090
```

---

## 5. Message Broker Integration (MANDATORY)

### A. Databus Architecture

**CRITICAL: Use a message broker for asynchronous communication between microservices. The specific broker (Kafka, RabbitMQ, Redis, NATS, etc.) is a deployment choice - applications should use abstraction layers.**

```
MESSAGE BROKER ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────┐
│                         CLUSTER                                  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  message-broker namespace                  │  │
│  │                                                            │  │
│  │  ┌──────────────────────────────────────────────────────┐ │  │
│  │  │              Message Broker                           │ │  │
│  │  │   (Kafka / RabbitMQ / Redis / NATS / Pulsar)         │ │  │
│  │  │                                                       │ │  │
│  │  │  Topics/Queues:                                       │ │  │
│  │  │  ├── orders.created                                   │ │  │
│  │  │  ├── orders.updated                                   │ │  │
│  │  │  ├── payments.processed                               │ │  │
│  │  │  ├── inventory.reserved                               │ │  │
│  │  │  └── notifications.send                               │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│       ┌──────────────────────┼──────────────────────┐           │
│       │                      │                      │           │
│       ▼                      ▼                      ▼           │
│  ┌─────────┐           ┌─────────┐           ┌─────────┐       │
│  │ Order   │  publish  │ Payment │  publish  │Inventory│       │
│  │ Service │ ────────▶ │ Service │ ────────▶ │ Service │       │
│  │         │ ◀──────── │         │ ◀──────── │         │       │
│  │         │  consume  │         │  consume  │         │       │
│  └─────────┘           └─────────┘           └─────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

BENEFITS:
• Loose coupling between services
• Resilience (async processing, retries)
• Scalability (consumers scale independently)
• Event sourcing and audit trail
• Peak load buffering
```

### B. Broker-Agnostic Configuration

```yaml
# ✅ CORRECT - ConfigMap for broker configuration (agnostic)
apiVersion: v1
kind: ConfigMap
metadata:
  name: message-broker-config
  namespace: team-orders
data:
  # Connection settings (broker-agnostic naming)
  BROKER_TYPE: "kafka"  # or "rabbitmq", "redis", "nats", "pulsar"
  BROKER_HOSTS: "broker.message-broker.svc.cluster.local:9092"

  # Topic/Queue configuration
  EVENTS_TOPIC_ORDERS_CREATED: "orders.created"
  EVENTS_TOPIC_ORDERS_UPDATED: "orders.updated"
  EVENTS_TOPIC_PAYMENTS_PROCESSED: "payments.processed"

  # Consumer configuration
  CONSUMER_GROUP_ID: "order-service"
  CONSUMER_AUTO_OFFSET_RESET: "earliest"
  CONSUMER_MAX_POLL_RECORDS: "100"

  # Producer configuration
  PRODUCER_ACKS: "all"
  PRODUCER_RETRIES: "3"
  PRODUCER_BATCH_SIZE: "16384"

  # Resilience settings
  CONNECTION_TIMEOUT_MS: "30000"
  REQUEST_TIMEOUT_MS: "30000"
  RETRY_BACKOFF_MS: "1000"
---
# Secret for broker credentials
apiVersion: v1
kind: Secret
metadata:
  name: message-broker-credentials
  namespace: team-orders
type: Opaque
stringData:
  BROKER_USERNAME: "${BROKER_USERNAME}"
  BROKER_PASSWORD: "${BROKER_PASSWORD}"
  # For Kafka with SASL
  BROKER_SASL_MECHANISM: "SCRAM-SHA-512"
  # For TLS
  BROKER_TLS_ENABLED: "true"
```

### C. Broker Deployment Patterns

```yaml
# ✅ CORRECT - External Secrets for broker credentials
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: message-broker-credentials
  namespace: team-orders
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: ClusterSecretStore
  target:
    name: message-broker-credentials
    creationPolicy: Owner
  data:
    - secretKey: BROKER_USERNAME
      remoteRef:
        key: secret/data/message-broker
        property: username
    - secretKey: BROKER_PASSWORD
      remoteRef:
        key: secret/data/message-broker
        property: password
---
# NetworkPolicy allowing broker access
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-message-broker
  namespace: team-orders
spec:
  podSelector:
    matchLabels:
      app: order-service
  policyTypes:
    - Egress
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: message-broker
          podSelector:
            matchLabels:
              app: kafka  # or rabbitmq, redis, nats
      ports:
        - protocol: TCP
          port: 9092  # Kafka
        # - port: 5672  # RabbitMQ
        # - port: 6379  # Redis
        # - port: 4222  # NATS
```

### D. Event Schema Standards

```yaml
# ✅ CORRECT - ConfigMap for event schemas
apiVersion: v1
kind: ConfigMap
metadata:
  name: event-schemas
  namespace: team-orders
data:
  order-created-v1.json: |
    {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "type": "object",
      "required": ["eventId", "eventType", "timestamp", "data"],
      "properties": {
        "eventId": { "type": "string", "format": "uuid" },
        "eventType": { "const": "OrderCreated" },
        "version": { "const": "v1" },
        "timestamp": { "type": "string", "format": "date-time" },
        "correlationId": { "type": "string" },
        "source": { "type": "string" },
        "data": {
          "type": "object",
          "required": ["orderId", "customerId", "items", "totalAmount"],
          "properties": {
            "orderId": { "type": "string" },
            "customerId": { "type": "string" },
            "items": { "type": "array" },
            "totalAmount": { "type": "number" },
            "currency": { "type": "string" }
          }
        }
      }
    }
```

---

## 6. Network Policies (MANDATORY)

### A. Default Deny Policy

**CRITICAL: Implement default deny and explicitly allow required traffic.**

```yaml
# ✅ CORRECT - Default deny all traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: team-orders
spec:
  podSelector: {}  # Applies to all pods
  policyTypes:
    - Ingress
    - Egress
---
# Allow DNS resolution (required)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns
  namespace: team-orders
spec:
  podSelector: {}
  policyTypes:
    - Egress
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: kube-system
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
        - protocol: TCP
          port: 53
```

### B. Service-Specific Policies

```yaml
# ✅ CORRECT - Allow Istio sidecar communication
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-istio
  namespace: team-orders
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow from Istio control plane
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: istio-system
      ports:
        - protocol: TCP
          port: 15012  # istiod
        - protocol: TCP
          port: 15017  # istiod webhook
    # Allow from same namespace (sidecar to app)
    - from:
        - podSelector: {}
      ports:
        - protocol: TCP
  egress:
    # Allow to Istio control plane
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: istio-system
      ports:
        - protocol: TCP
          port: 15012
        - protocol: TCP
          port: 443
---
# Allow ingress from Istio gateway
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-istio-ingress
  namespace: team-orders
spec:
  podSelector:
    matchLabels:
      app: order-service
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: istio-system
          podSelector:
            matchLabels:
              istio: ingressgateway
      ports:
        - protocol: TCP
          port: 8080
---
# Allow communication between services
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-from-payment-service
  namespace: team-orders
spec:
  podSelector:
    matchLabels:
      app: order-service
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: team-payments
          podSelector:
            matchLabels:
              app: payment-service
      ports:
        - protocol: TCP
          port: 8080
```

---

## 7. ConfigMaps and Secrets (MANDATORY)

### A. ConfigMap Best Practices

```yaml
# ✅ CORRECT - Well-structured ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: order-service-config
  namespace: team-orders
  labels:
    app: order-service
    config-version: v1.2.0
data:
  # Application configuration
  APP_NAME: "order-service"
  APP_ENV: "production"
  LOG_LEVEL: "info"
  LOG_FORMAT: "json"

  # Server configuration
  SERVER_PORT: "8080"
  SERVER_READ_TIMEOUT: "30s"
  SERVER_WRITE_TIMEOUT: "30s"
  SERVER_IDLE_TIMEOUT: "120s"

  # Feature flags (non-sensitive)
  FEATURE_NEW_CHECKOUT: "true"
  FEATURE_ASYNC_PROCESSING: "true"

  # External service URLs (non-sensitive)
  PAYMENT_SERVICE_URL: "http://payment-service.team-payments.svc.cluster.local:8080"
  INVENTORY_SERVICE_URL: "http://inventory-service.team-inventory.svc.cluster.local:8080"

  # Mounted configuration file
  application.yaml: |
    server:
      port: 8080
      gracefulShutdown: 30s
    logging:
      level: info
      format: json
    features:
      newCheckout: true
      asyncProcessing: true
```

### B. Secrets Management

**CRITICAL: Never store secrets in plain ConfigMaps or source code. Use External Secrets Operator or similar.**

```yaml
# ✅ CORRECT - External Secrets (preferred)
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: order-service-secrets
  namespace: team-orders
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: ClusterSecretStore
  target:
    name: order-service-secrets
    creationPolicy: Owner
    template:
      type: Opaque
      data:
        DATABASE_URL: "{{ .database_url }}"
        API_KEY: "{{ .api_key }}"
  data:
    - secretKey: database_url
      remoteRef:
        key: secret/data/order-service/production
        property: database_url
    - secretKey: api_key
      remoteRef:
        key: secret/data/order-service/production
        property: api_key
---
# ClusterSecretStore for Vault
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: vault-backend
spec:
  provider:
    vault:
      server: "https://vault.company.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "external-secrets"
          serviceAccountRef:
            name: external-secrets
            namespace: secrets-management
```

### C. Sealed Secrets (Alternative)

```yaml
# ✅ CORRECT - Sealed Secrets (for GitOps)
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: order-service-secrets
  namespace: team-orders
spec:
  encryptedData:
    DATABASE_URL: AgBy8hCi...encrypted...==
    API_KEY: AgDK9xLm...encrypted...==
  template:
    metadata:
      name: order-service-secrets
      namespace: team-orders
    type: Opaque
```

---

## 8. Resource Management (MANDATORY)

### A. Resource Requests and Limits

```yaml
# ✅ CORRECT - Properly sized resources
resources:
  requests:
    cpu: 100m        # 0.1 CPU cores (guaranteed)
    memory: 256Mi    # 256 MiB (guaranteed)
  limits:
    cpu: 500m        # 0.5 CPU cores (maximum)
    memory: 512Mi    # 512 MiB (maximum, OOM if exceeded)

# Resource sizing guidelines:
#
# Development/Staging:
#   requests: cpu: 50m, memory: 128Mi
#   limits:   cpu: 200m, memory: 256Mi
#
# Production (typical microservice):
#   requests: cpu: 100m, memory: 256Mi
#   limits:   cpu: 500m, memory: 512Mi
#
# Production (heavy workload):
#   requests: cpu: 500m, memory: 1Gi
#   limits:   cpu: 2000m, memory: 2Gi
```

### B. Horizontal Pod Autoscaler

```yaml
# ✅ CORRECT - HPA with multiple metrics
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: order-service-hpa
  namespace: team-orders
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: order-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
    # CPU-based scaling
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    # Memory-based scaling
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    # Custom metric (requests per second via Istio)
    - type: Pods
      pods:
        metric:
          name: istio_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
        - type: Pods
          value: 4
          periodSeconds: 15
      selectPolicy: Max
```

### C. Pod Disruption Budget

```yaml
# ✅ CORRECT - PDB for high availability
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: order-service-pdb
  namespace: team-orders
spec:
  minAvailable: 2  # or use maxUnavailable: 1
  selector:
    matchLabels:
      app: order-service
```

### D. Vertical Pod Autoscaler (Optional)

```yaml
# ✅ CORRECT - VPA for resource recommendations
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: order-service-vpa
  namespace: team-orders
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: order-service
  updatePolicy:
    updateMode: "Off"  # "Off" for recommendations only, "Auto" to apply
  resourcePolicy:
    containerPolicies:
      - containerName: order-service
        minAllowed:
          cpu: 50m
          memory: 128Mi
        maxAllowed:
          cpu: 2000m
          memory: 4Gi
        controlledResources: ["cpu", "memory"]
```

---

## 9. RBAC (MANDATORY)

### A. ServiceAccount Per Application

```yaml
# ✅ CORRECT - Dedicated ServiceAccount
apiVersion: v1
kind: ServiceAccount
metadata:
  name: order-service
  namespace: team-orders
  labels:
    app: order-service
automountServiceAccountToken: false
---
# Role with minimal permissions
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: order-service-role
  namespace: team-orders
spec:
  rules:
    # Read own ConfigMaps
    - apiGroups: [""]
      resources: ["configmaps"]
      resourceNames: ["order-service-config"]
      verbs: ["get", "watch"]
    # Read own Secrets
    - apiGroups: [""]
      resources: ["secrets"]
      resourceNames: ["order-service-secrets"]
      verbs: ["get", "watch"]
---
# RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: order-service-rolebinding
  namespace: team-orders
subjects:
  - kind: ServiceAccount
    name: order-service
    namespace: team-orders
roleRef:
  kind: Role
  name: order-service-role
  apiGroup: rbac.authorization.k8s.io
```

### B. Team Access RBAC

```yaml
# ✅ CORRECT - Team role with namespace access
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: team-orders-developer
  namespace: team-orders
rules:
  # Read most resources
  - apiGroups: ["", "apps", "networking.k8s.io"]
    resources: ["pods", "services", "deployments", "configmaps", "ingresses"]
    verbs: ["get", "list", "watch"]
  # Logs access
  - apiGroups: [""]
    resources: ["pods/log"]
    verbs: ["get", "list"]
  # Port-forward for debugging
  - apiGroups: [""]
    resources: ["pods/portforward"]
    verbs: ["create"]
  # Exec into pods (staging only)
  - apiGroups: [""]
    resources: ["pods/exec"]
    verbs: ["create"]
---
# ClusterRole for read-only cluster resources
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: namespace-viewer
rules:
  - apiGroups: [""]
    resources: ["namespaces"]
    verbs: ["get", "list"]
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["get", "list"]
```

---

## 10. Observability (MANDATORY)

### A. Prometheus Metrics

```yaml
# ✅ CORRECT - ServiceMonitor for Prometheus
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: order-service
  namespace: team-orders
  labels:
    app: order-service
spec:
  selector:
    matchLabels:
      app: order-service
  endpoints:
    - port: http
      path: /metrics
      interval: 30s
      scrapeTimeout: 10s
  namespaceSelector:
    matchNames:
      - team-orders
---
# PrometheusRule for alerts
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: order-service-alerts
  namespace: team-orders
spec:
  groups:
    - name: order-service
      rules:
        - alert: HighErrorRate
          expr: |
            sum(rate(http_requests_total{app="order-service",status=~"5.."}[5m]))
            /
            sum(rate(http_requests_total{app="order-service"}[5m]))
            > 0.05
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "High error rate for order-service"
            description: "Error rate is {{ $value | humanizePercentage }}"

        - alert: HighLatency
          expr: |
            histogram_quantile(0.99,
              sum(rate(http_request_duration_seconds_bucket{app="order-service"}[5m])) by (le)
            ) > 1
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "High latency for order-service"
            description: "P99 latency is {{ $value }}s"

        - alert: PodNotReady
          expr: |
            kube_pod_status_ready{namespace="team-orders",pod=~"order-service.*",condition="true"} == 0
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "Order service pod not ready"
```

### B. Istio Telemetry

```yaml
# ✅ CORRECT - Istio telemetry configuration
apiVersion: telemetry.istio.io/v1alpha1
kind: Telemetry
metadata:
  name: order-service-telemetry
  namespace: team-orders
spec:
  selector:
    matchLabels:
      app: order-service
  # Tracing configuration
  tracing:
    - providers:
        - name: jaeger
      randomSamplingPercentage: 10.0
      customTags:
        app:
          literal:
            value: order-service
        environment:
          literal:
            value: production
  # Access logging
  accessLogging:
    - providers:
        - name: envoy
      filter:
        expression: "response.code >= 400"
```

---

## 11. GitOps Deployment (MANDATORY)

### A. Directory Structure

```
kubernetes/
├── base/                           # Base configurations
│   ├── namespace.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── hpa.yaml
│   ├── pdb.yaml
│   ├── networkpolicy.yaml
│   ├── serviceaccount.yaml
│   ├── istio/
│   │   ├── virtualservice.yaml
│   │   ├── destinationrule.yaml
│   │   └── authorizationpolicy.yaml
│   └── kustomization.yaml
├── overlays/                       # Environment-specific
│   ├── development/
│   │   ├── kustomization.yaml
│   │   ├── replicas-patch.yaml
│   │   └── resources-patch.yaml
│   ├── staging/
│   │   ├── kustomization.yaml
│   │   ├── replicas-patch.yaml
│   │   └── configmap-patch.yaml
│   └── production/
│       ├── kustomization.yaml
│       ├── replicas-patch.yaml
│       ├── hpa-patch.yaml
│       └── external-secret.yaml
└── argocd/                         # ArgoCD applications
    ├── order-service-dev.yaml
    ├── order-service-staging.yaml
    └── order-service-prod.yaml
```

### B. Kustomize Base

```yaml
# base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: team-orders

resources:
  - namespace.yaml
  - serviceaccount.yaml
  - configmap.yaml
  - deployment.yaml
  - service.yaml
  - hpa.yaml
  - pdb.yaml
  - networkpolicy.yaml
  - istio/virtualservice.yaml
  - istio/destinationrule.yaml
  - istio/authorizationpolicy.yaml

commonLabels:
  app: order-service
  team: orders

images:
  - name: order-service
    newName: registry.company.com/orders/order-service
    newTag: latest  # Overridden in overlays
```

### C. Production Overlay

```yaml
# overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: team-orders

resources:
  - ../../base
  - external-secret.yaml

patches:
  - path: replicas-patch.yaml
  - path: hpa-patch.yaml
  - path: resources-patch.yaml

images:
  - name: order-service
    newName: registry.company.com/orders/order-service
    newTag: v1.2.3  # Specific version for production

configMapGenerator:
  - name: order-service-config
    behavior: merge
    literals:
      - APP_ENV=production
      - LOG_LEVEL=info
---
# overlays/production/replicas-patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
spec:
  replicas: 5
---
# overlays/production/resources-patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
spec:
  template:
    spec:
      containers:
        - name: order-service
          resources:
            requests:
              cpu: 200m
              memory: 512Mi
            limits:
              cpu: 1000m
              memory: 1Gi
```

### D. ArgoCD Application

```yaml
# ✅ CORRECT - ArgoCD Application
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: order-service-prod
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: team-orders

  source:
    repoURL: https://github.com/company/kubernetes-manifests.git
    targetRevision: main
    path: services/order-service/overlays/production

  destination:
    server: https://kubernetes.default.svc
    namespace: team-orders

  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
      - PruneLast=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m

  # Health checks
  ignoreDifferences:
    - group: apps
      kind: Deployment
      jsonPointers:
        - /spec/replicas  # Ignore if HPA manages replicas
```

---

## 12. Health Probes (MANDATORY)

### A. Probe Configuration

```yaml
# ✅ CORRECT - Well-configured probes
containers:
  - name: order-service
    # Startup probe - for slow-starting containers
    startupProbe:
      httpGet:
        path: /health/startup
        port: 8080
      initialDelaySeconds: 0
      periodSeconds: 5
      timeoutSeconds: 3
      successThreshold: 1
      failureThreshold: 30  # 30 * 5s = 150s max startup time

    # Liveness probe - restart if unhealthy
    livenessProbe:
      httpGet:
        path: /health/live
        port: 8080
      initialDelaySeconds: 0  # startupProbe handles initial delay
      periodSeconds: 10
      timeoutSeconds: 5
      successThreshold: 1
      failureThreshold: 3

    # Readiness probe - remove from service if not ready
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8080
      initialDelaySeconds: 0
      periodSeconds: 5
      timeoutSeconds: 3
      successThreshold: 1
      failureThreshold: 3
```

### B. Health Check Endpoints

```
HEALTH CHECK DESIGN:

/health/startup
├── Purpose: Has the application finished initializing?
├── Checks:
│   ├── Configuration loaded
│   ├── Database migrations complete
│   └── Caches warmed (if required)
└── Failure: Container restart delayed until ready

/health/live
├── Purpose: Is the application process running correctly?
├── Checks:
│   ├── Process is responsive
│   └── No deadlocks detected
├── Failure: Container restarted
└── IMPORTANT: Do NOT check external dependencies

/health/ready
├── Purpose: Can the application handle traffic?
├── Checks:
│   ├── Database connection healthy
│   ├── Cache connection healthy
│   └── Critical dependencies available
├── Failure: Pod removed from service endpoints
└── Returns: Pod receives no traffic until ready
```

---

## 13. Common Anti-Patterns (PROHIBITED)

### A. Security Anti-Patterns

```yaml
# ❌ PROHIBITED - Running as root
securityContext:
  runAsUser: 0
  runAsGroup: 0

# ❌ PROHIBITED - Privileged container
securityContext:
  privileged: true

# ❌ PROHIBITED - Using latest tag
image: myapp:latest

# ❌ PROHIBITED - No resource limits
resources: {}  # Missing limits

# ❌ PROHIBITED - Secrets in ConfigMap
kind: ConfigMap
data:
  DATABASE_PASSWORD: "supersecret123"

# ❌ PROHIBITED - Host network
hostNetwork: true

# ❌ PROHIBITED - Default service account with auto-mount
serviceAccountName: default
automountServiceAccountToken: true
```

### B. Reliability Anti-Patterns

```yaml
# ❌ PROHIBITED - Single replica in production
spec:
  replicas: 1

# ❌ PROHIBITED - No health probes
containers:
  - name: app
    # Missing livenessProbe and readinessProbe

# ❌ PROHIBITED - No PDB
# Missing PodDisruptionBudget for critical services

# ❌ PROHIBITED - No anti-affinity (all pods on same node)
affinity: {}  # Missing podAntiAffinity

# ❌ PROHIBITED - No graceful shutdown
terminationGracePeriodSeconds: 0
```

### C. Configuration Anti-Patterns

```yaml
# ❌ PROHIBITED - Hardcoded configuration
containers:
  - name: app
    env:
      - name: DATABASE_URL
        value: "postgres://prod-db:5432/orders"  # Hardcoded

# ❌ PROHIBITED - No namespace isolation
metadata:
  namespace: default  # Using default namespace

# ❌ PROHIBITED - Over-privileged RBAC
rules:
  - apiGroups: ["*"]
    resources: ["*"]
    verbs: ["*"]  # Too permissive
```

---

## 14. Verification Checklist (MANDATORY)

### A. Pre-Deployment Validation

```bash
# ✅ CORRECT - Validation commands

# 1. Validate YAML syntax
kubectl apply --dry-run=client -f manifests/

# 2. Validate against cluster (server-side)
kubectl apply --dry-run=server -f manifests/

# 3. Check Kustomize build
kustomize build overlays/production | kubectl apply --dry-run=server -f -

# 4. Validate with kubeval/kubeconform
kubeconform -strict -kubernetes-version 1.28.0 manifests/

# 5. Security scan with kubesec
kubesec scan deployment.yaml

# 6. Policy validation with OPA/Gatekeeper
gator test manifests/ --policies=policies/

# 7. Check Istio configuration
istioctl analyze -n team-orders

# 8. Validate resource requests/limits
kubectl get pods -n team-orders -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[*].resources}{"\n"}{end}'
```

### B. Architecture Verification Checklist

```
VERIFICATION CHECKLIST:

□ Security
  □ Pod security context configured (non-root, read-only fs)
  □ No privileged containers
  □ Specific image tags (not :latest)
  □ Secrets in External Secrets / Sealed Secrets
  □ NetworkPolicies defined (default deny)
  □ RBAC with least privilege
  □ ServiceAccount per application

□ Istio Service Mesh
  □ Namespace labeled for injection
  □ Gateway configured with TLS
  □ VirtualService for routing
  □ DestinationRule with traffic policies
  □ AuthorizationPolicy for zero-trust
  □ PeerAuthentication for mTLS

□ Message Broker
  □ Broker-agnostic configuration
  □ Credentials in secrets
  □ NetworkPolicy allowing broker access
  □ Event schemas documented

□ Resource Management
  □ Resource requests and limits set
  □ HPA configured
  □ PDB defined
  □ ResourceQuota per namespace
  □ LimitRange with defaults

□ Observability
  □ Health probes (startup, liveness, readiness)
  □ Prometheus metrics exposed
  □ ServiceMonitor configured
  □ Alerts defined
  □ Structured logging

□ Reliability
  □ Multiple replicas
  □ Pod anti-affinity
  □ Graceful shutdown handling
  □ Retry policies in VirtualService

□ GitOps
  □ All manifests in Git
  □ Kustomize overlays per environment
  □ ArgoCD Application defined
  □ No manual kubectl apply
```

### C. Post-Deployment Verification

```bash
# ✅ CORRECT - Post-deployment checks

# 1. Check pod status
kubectl get pods -n team-orders -l app=order-service

# 2. Check events for errors
kubectl get events -n team-orders --sort-by='.lastTimestamp'

# 3. Verify Istio sidecar injection
kubectl get pods -n team-orders -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[*].name}{"\n"}{end}'

# 4. Check service connectivity
kubectl exec -n team-orders deploy/order-service -c istio-proxy -- pilot-agent request GET /clusters

# 5. Verify mTLS
istioctl x describe pod order-service-xxx -n team-orders

# 6. Test health endpoints
kubectl port-forward -n team-orders svc/order-service 8080:8080 &
curl localhost:8080/health/ready

# 7. Check HPA status
kubectl get hpa -n team-orders

# 8. Verify network policies
kubectl get networkpolicies -n team-orders
```

---

## 15. Summary

### Core Principles

1. **Security by default**: Non-root, read-only filesystem, no privilege escalation
2. **Istio for everything**: Ingress, service mesh, mTLS, observability
3. **Message broker for async**: Decouple services with event-driven communication
4. **GitOps only**: No manual changes, everything in version control
5. **Observability built-in**: Metrics, logs, traces, health checks

### Key Components

| Component | Purpose | Tool |
|-----------|---------|------|
| Ingress | External traffic | Istio Gateway |
| Service Mesh | Internal traffic, mTLS | Istio |
| Message Broker | Async communication | Kafka/RabbitMQ/Redis/NATS |
| Secrets | Credential management | External Secrets + Vault |
| Monitoring | Metrics and alerts | Prometheus + Grafana |
| Tracing | Distributed tracing | Jaeger (via Istio) |
| Deployment | GitOps | ArgoCD + Kustomize |

### Remember

> "Security is not a feature, it's a requirement. Every pod runs non-root, every secret is encrypted, every connection is mTLS."

> "If it's not in Git, it doesn't exist. No kubectl apply in production - ever."

---

## 16. Quick Reference

### Common kubectl Commands

```bash
# ═══════════════════════════════════════════════════════════════════
# CLUSTER INFORMATION
# ═══════════════════════════════════════════════════════════════════
kubectl cluster-info                              # Display cluster info
kubectl get nodes -o wide                         # List nodes with details
kubectl top nodes                                 # Node resource usage
kubectl api-resources                             # List all resource types

# ═══════════════════════════════════════════════════════════════════
# NAMESPACE OPERATIONS
# ═══════════════════════════════════════════════════════════════════
kubectl get namespaces                            # List all namespaces
kubectl create namespace team-orders              # Create namespace
kubectl config set-context --current --namespace=team-orders  # Set default ns

# ═══════════════════════════════════════════════════════════════════
# POD OPERATIONS
# ═══════════════════════════════════════════════════════════════════
kubectl get pods -n team-orders                   # List pods in namespace
kubectl get pods -o wide                          # Pods with node info
kubectl get pods -l app=order-service             # Filter by label
kubectl describe pod <pod-name> -n team-orders    # Pod details
kubectl logs <pod-name> -n team-orders            # View logs
kubectl logs <pod-name> -c <container> -f         # Follow container logs
kubectl logs <pod-name> --previous                # Previous container logs
kubectl exec -it <pod-name> -- /bin/sh            # Shell into pod
kubectl port-forward <pod-name> 8080:8080         # Port forward
kubectl top pods -n team-orders                   # Pod resource usage
kubectl delete pod <pod-name> -n team-orders      # Delete pod (restart)

# ═══════════════════════════════════════════════════════════════════
# DEPLOYMENT OPERATIONS
# ═══════════════════════════════════════════════════════════════════
kubectl get deployments -n team-orders            # List deployments
kubectl describe deployment <name>                # Deployment details
kubectl rollout status deployment/<name>          # Rollout status
kubectl rollout history deployment/<name>         # Rollout history
kubectl rollout undo deployment/<name>            # Rollback to previous
kubectl rollout undo deployment/<name> --to-revision=2  # Rollback to rev
kubectl scale deployment/<name> --replicas=5      # Scale deployment
kubectl set image deployment/<name> app=image:v2  # Update image

# ═══════════════════════════════════════════════════════════════════
# SERVICE & NETWORKING
# ═══════════════════════════════════════════════════════════════════
kubectl get services -n team-orders               # List services
kubectl get endpoints -n team-orders              # List endpoints
kubectl get ingress -n team-orders                # List ingresses
kubectl get networkpolicies -n team-orders        # List network policies

# ═══════════════════════════════════════════════════════════════════
# CONFIG & SECRETS
# ═══════════════════════════════════════════════════════════════════
kubectl get configmaps -n team-orders             # List ConfigMaps
kubectl get secrets -n team-orders                # List Secrets
kubectl create secret generic my-secret \
  --from-literal=key=value                        # Create secret
kubectl get secret my-secret -o jsonpath='{.data.key}' | base64 -d  # Decode

# ═══════════════════════════════════════════════════════════════════
# DEBUGGING & TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════════════
kubectl get events -n team-orders --sort-by='.lastTimestamp'  # Recent events
kubectl get events --field-selector type=Warning  # Warning events only
kubectl run debug --image=busybox -it --rm -- sh  # Temporary debug pod
kubectl auth can-i create pods -n team-orders     # Check permissions
kubectl explain deployment.spec.template          # API documentation

# ═══════════════════════════════════════════════════════════════════
# VALIDATION & DRY-RUN
# ═══════════════════════════════════════════════════════════════════
kubectl apply --dry-run=client -f manifest.yaml   # Client-side validation
kubectl apply --dry-run=server -f manifest.yaml   # Server-side validation
kubectl diff -f manifest.yaml                     # Show diff before apply
kubeconform -strict manifest.yaml                 # Schema validation
conftest test manifest.yaml --policy policy/      # Policy validation

# ═══════════════════════════════════════════════════════════════════
# ISTIO-SPECIFIC
# ═══════════════════════════════════════════════════════════════════
istioctl analyze -n team-orders                   # Analyze Istio config
istioctl proxy-status                             # Envoy sync status
istioctl x describe pod <pod-name>                # Describe pod mesh config
kubectl get virtualservices -n team-orders        # List VirtualServices
kubectl get destinationrules -n team-orders       # List DestinationRules
kubectl get authorizationpolicies -n team-orders  # List AuthorizationPolicies
```

### Kubernetes Patterns Cheat Sheet

```
┌─────────────────────────────────────────────────────────────────┐
│                 KUBERNETES PATTERNS CHEAT SHEET                  │
└─────────────────────────────────────────────────────────────────┘

HIGH AVAILABILITY PATTERN
─────────────────────────
□ replicas: 3+ (never 1 in production)
□ podAntiAffinity across nodes/zones
□ PodDisruptionBudget (minAvailable: 2)
□ Multiple availability zones
□ Resource requests/limits set

SECURITY PATTERN
────────────────
□ runAsNonRoot: true
□ readOnlyRootFilesystem: true
□ allowPrivilegeEscalation: false
□ capabilities.drop: ALL
□ seccompProfile: RuntimeDefault
□ Dedicated ServiceAccount
□ automountServiceAccountToken: false
□ NetworkPolicies (default deny)
□ No :latest image tags

RESILIENCE PATTERN
──────────────────
□ startupProbe (slow-starting apps)
□ livenessProbe (restart unhealthy)
□ readinessProbe (traffic control)
□ terminationGracePeriodSeconds: 30+
□ preStop hook for graceful shutdown
□ Resource limits prevent cascading failures

OBSERVABILITY PATTERN
─────────────────────
□ Prometheus annotations for scraping
□ ServiceMonitor for metrics collection
□ Structured JSON logging
□ Distributed tracing headers
□ PrometheusRule for alerts
□ Istio telemetry configuration

CONFIGURATION PATTERN
─────────────────────
□ ConfigMap for non-sensitive config
□ External Secrets for sensitive data
□ envFrom for bulk injection
□ Mounted config files for complex config
□ Environment-specific overlays

GITOPS PATTERN
──────────────
□ All manifests in Git
□ Kustomize base + overlays
□ ArgoCD Application per environment
□ Automated sync with self-heal
□ No manual kubectl apply

TRAFFIC MANAGEMENT PATTERN (Istio)
──────────────────────────────────
□ Gateway for ingress (TLS termination)
□ VirtualService for routing rules
□ DestinationRule for traffic policies
□ Circuit breaker (outlierDetection)
□ Retry policies with backoff
□ Timeout configuration
□ Canary/Blue-Green via subsets

ZERO-TRUST PATTERN (Istio)
──────────────────────────
□ PeerAuthentication: STRICT mTLS
□ AuthorizationPolicy: deny-all default
□ Explicit allow rules per service
□ RequestAuthentication for JWT
□ Service identity via SPIFFE
```

### Manifest Structure Quick Reference

```yaml
# ═══════════════════════════════════════════════════════════════════
# DEPLOYMENT - Complete Production Template
# ═══════════════════════════════════════════════════════════════════
apiVersion: apps/v1
kind: Deployment
metadata:
  name: <service-name>
  namespace: <namespace>
  labels:
    app: <service-name>
    version: <version>
spec:
  replicas: 3                              # HA: minimum 3
  selector:
    matchLabels:
      app: <service-name>
  template:
    metadata:
      labels:
        app: <service-name>
        version: <version>
      annotations:
        prometheus.io/scrape: "true"       # Observability
        prometheus.io/port: "8080"
    spec:
      serviceAccountName: <service-name>   # Dedicated SA
      automountServiceAccountToken: false
      securityContext:                     # Pod security
        runAsNonRoot: true
        runAsUser: 10001
        runAsGroup: 10001
        fsGroup: 10001
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: <service-name>
          image: <registry>/<image>:<tag>  # Never :latest
          imagePullPolicy: Always
          securityContext:                 # Container security
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
          resources:                       # Resource management
            requests:
              cpu: 100m
              memory: 256Mi
            limits:
              cpu: 500m
              memory: 512Mi
          ports:
            - name: http
              containerPort: 8080
          startupProbe:                    # Health probes
            httpGet:
              path: /health/startup
              port: 8080
            failureThreshold: 30
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /health/live
              port: 8080
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8080
            periodSeconds: 5
          envFrom:                         # Configuration
            - configMapRef:
                name: <service-name>-config
            - secretRef:
                name: <service-name>-secrets
          volumeMounts:
            - name: tmp
              mountPath: /tmp
      volumes:
        - name: tmp
          emptyDir:
            sizeLimit: 100Mi
      affinity:                            # Pod anti-affinity
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app: <service-name>
                topologyKey: kubernetes.io/hostname
      terminationGracePeriodSeconds: 30    # Graceful shutdown
---
# ═══════════════════════════════════════════════════════════════════
# SERVICE
# ═══════════════════════════════════════════════════════════════════
apiVersion: v1
kind: Service
metadata:
  name: <service-name>
  namespace: <namespace>
spec:
  ports:
    - name: http                           # Named ports for Istio
      port: 8080
      targetPort: 8080
  selector:
    app: <service-name>
---
# ═══════════════════════════════════════════════════════════════════
# HPA
# ═══════════════════════════════════════════════════════════════════
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: <service-name>-hpa
  namespace: <namespace>
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: <service-name>
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
---
# ═══════════════════════════════════════════════════════════════════
# PDB
# ═══════════════════════════════════════════════════════════════════
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: <service-name>-pdb
  namespace: <namespace>
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: <service-name>
---
# ═══════════════════════════════════════════════════════════════════
# NETWORKPOLICY - Default Deny
# ═══════════════════════════════════════════════════════════════════
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: <namespace>
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
---
# ═══════════════════════════════════════════════════════════════════
# ISTIO VIRTUALSERVICE
# ═══════════════════════════════════════════════════════════════════
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: <service-name>
  namespace: <namespace>
spec:
  hosts:
    - <service-name>
  http:
    - route:
        - destination:
            host: <service-name>
            port:
              number: 8080
      timeout: 30s
      retries:
        attempts: 3
        perTryTimeout: 10s
        retryOn: 5xx,reset,connect-failure
---
# ═══════════════════════════════════════════════════════════════════
# ISTIO DESTINATIONRULE
# ═══════════════════════════════════════════════════════════════════
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: <service-name>
  namespace: <namespace>
spec:
  host: <service-name>
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http2MaxRequests: 1000
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 10s
      baseEjectionTime: 30s
    tls:
      mode: ISTIO_MUTUAL
```

---

## References

- **[istio.md](istio.md)**: Detailed Istio configuration including static IP setup for different platforms
- **[microservices.md](microservices.md)**: Microservices architecture patterns deployed on Kubernetes
- **[kafka.md](kafka.md)**: Apache Kafka as a message broker for Kubernetes workloads
- **[dockerfile_style.md](dockerfile_style.md)**: Container image best practices for Kubernetes deployments

> "Istio is your service mesh. Use it for ingress, traffic management, security policies, and observability. Don't reinvent the wheel."

> "Message brokers decouple your services. The specific broker doesn't matter - Kafka, RabbitMQ, Redis, NATS - what matters is that your services communicate asynchronously."
