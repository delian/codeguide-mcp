# Istio Service Mesh Guidelines
Mandatory standards and best practices for secure, production-ready Istio service mesh deployments. Covers generic Kubernetes installations, Azure AKS Istio add-on, and AWS EKS configurations with emphasis on security, static IP addressing, traffic management, and observability. Istio 1.20+, istioctl, Helm 3.x, Kiali, Jaeger, Prometheus, Grafana.

---

**Agent Profile**: The Istio Service Mesh Architect
**Role**: Senior Platform Engineer & Service Mesh Security Specialist
**Objective**: Generate production-ready, secure, observable Istio configurations with proper mTLS, authorization policies, traffic management, and platform-specific optimizations for generic Kubernetes, Azure AKS, and AWS EKS.
**Tools**: Istio 1.20+, istioctl, Helm 3.x, Kiali, Jaeger, Prometheus, Grafana.

---

## 1. Core Philosophies: ISTIO-MESH

The agent must adhere to the **ISTIO-MESH** standard for every Istio deployment:

- **I**solated Security: mTLS everywhere, zero-trust by default
- **S**tatic Ingress: Predictable gateway IP addresses for DNS and firewall rules
- **T**raffic Control: Fine-grained routing, retries, timeouts, circuit breaking
- **I**ntelligent Observability: Metrics, traces, logs automatically collected
- **O**utlier Detection: Automatic removal of unhealthy endpoints

- **M**ulti-Platform: Consistent configuration across generic K8s, AKS, EKS
- **E**xternalized Configuration: Gateway, VirtualService, DestinationRule as code
- **S**trict Authorization: Deny by default, explicit allow policies
- **H**ardened Control Plane: Secured istiod, minimal permissions

**Additional Principles:**

- **Defense in Depth**: Network policies + Istio authorization + application auth
- **GitOps Deployment**: All Istio resources version-controlled
- **Gradual Rollout**: Canary deployments, traffic shifting
- **Platform Optimization**: Use managed Istio where available (AKS add-on)

**Verified Configuration**: Agent-generated Istio configs MUST pass `istioctl analyze` before delivery.

---

## 2. Installation Options

### A. Installation Decision Matrix

```
ISTIO INSTALLATION OPTIONS:

┌─────────────────────┬──────────────┬──────────────┬──────────────┐
│ Feature             │ Generic K8s  │ AKS Add-on   │ EKS (Manual) │
├─────────────────────┼──────────────┼──────────────┼──────────────┤
│ Management          │ Self-managed │ Microsoft    │ Self-managed │
│ Upgrades            │ Manual       │ Automatic    │ Manual       │
│ Support             │ Community    │ Azure Support│ AWS/Community│
│ Customization       │ Full         │ Limited      │ Full         │
│ Control Plane       │ In-cluster   │ Managed      │ In-cluster   │
│ Static IP           │ Manual       │ Annotation   │ Manual/NLB   │
│ Integration         │ Generic      │ Azure native │ AWS native   │
└─────────────────────┴──────────────┴──────────────┴──────────────┘

RECOMMENDATION:
- Azure AKS: Use Istio add-on (managed, supported)
- AWS EKS: Use istioctl or Helm (full control)
- Generic K8s: Use istioctl or Helm (full control)
```

### B. Generic Kubernetes Installation

```bash
# ✅ CORRECT - Production installation with istioctl

# 1. Download istioctl
curl -L https://istio.io/downloadIstio | ISTIO_VERSION=1.20.0 sh -
export PATH=$PWD/istio-1.20.0/bin:$PATH

# 2. Install with production profile
istioctl install --set profile=default \
  --set meshConfig.accessLogFile=/dev/stdout \
  --set meshConfig.enableTracing=true \
  --set meshConfig.defaultConfig.tracing.sampling=10 \
  --set values.global.proxy.resources.requests.cpu=100m \
  --set values.global.proxy.resources.requests.memory=128Mi \
  --set values.global.proxy.resources.limits.cpu=500m \
  --set values.global.proxy.resources.limits.memory=256Mi \
  -y

# 3. Verify installation
istioctl verify-install
istioctl analyze --all-namespaces
```

```yaml
# ✅ CORRECT - IstioOperator for GitOps deployment
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: istio-control-plane
  namespace: istio-system
spec:
  profile: default

  meshConfig:
    # Access logging
    accessLogFile: /dev/stdout
    accessLogFormat: |
      {
        "timestamp": "%START_TIME%",
        "method": "%REQ(:METHOD)%",
        "path": "%REQ(X-ENVOY-ORIGINAL-PATH?:PATH)%",
        "protocol": "%PROTOCOL%",
        "response_code": "%RESPONSE_CODE%",
        "response_flags": "%RESPONSE_FLAGS%",
        "bytes_received": "%BYTES_RECEIVED%",
        "bytes_sent": "%BYTES_SENT%",
        "duration": "%DURATION%",
        "upstream_service_time": "%RESP(X-ENVOY-UPSTREAM-SERVICE-TIME)%",
        "x_forwarded_for": "%REQ(X-FORWARDED-FOR)%",
        "user_agent": "%REQ(USER-AGENT)%",
        "request_id": "%REQ(X-REQUEST-ID)%",
        "authority": "%REQ(:AUTHORITY)%",
        "upstream_host": "%UPSTREAM_HOST%",
        "upstream_cluster": "%UPSTREAM_CLUSTER%",
        "trace_id": "%REQ(X-B3-TRACEID)%"
      }

    # Tracing
    enableTracing: true
    defaultConfig:
      tracing:
        sampling: 10.0  # 10% sampling in production
        zipkin:
          address: jaeger-collector.observability.svc.cluster.local:9411

    # Security defaults
    enableAutoMtls: true
    trustDomain: cluster.local

    # Outbound traffic policy
    outboundTrafficPolicy:
      mode: REGISTRY_ONLY  # Strict: only allow registered services

    # Default destination rules
    defaultDestinationRuleExportTo:
      - "."
    defaultServiceExportTo:
      - "."
    defaultVirtualServiceExportTo:
      - "."

  components:
    # Ingress gateway
    ingressGateways:
      - name: istio-ingressgateway
        enabled: true
        k8s:
          service:
            type: LoadBalancer
            ports:
              - name: http2
                port: 80
                targetPort: 8080
              - name: https
                port: 443
                targetPort: 8443
          hpaSpec:
            minReplicas: 2
            maxReplicas: 10
            metrics:
              - type: Resource
                resource:
                  name: cpu
                  target:
                    type: Utilization
                    averageUtilization: 80
          resources:
            requests:
              cpu: 200m
              memory: 256Mi
            limits:
              cpu: 1000m
              memory: 512Mi
          affinity:
            podAntiAffinity:
              preferredDuringSchedulingIgnoredDuringExecution:
                - weight: 100
                  podAffinityTerm:
                    labelSelector:
                      matchLabels:
                        app: istio-ingressgateway
                    topologyKey: kubernetes.io/hostname

    # Egress gateway (optional, for controlled egress)
    egressGateways:
      - name: istio-egressgateway
        enabled: true
        k8s:
          hpaSpec:
            minReplicas: 2
            maxReplicas: 5
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 256Mi

    # Control plane
    pilot:
      k8s:
        hpaSpec:
          minReplicas: 2
          maxReplicas: 5
        resources:
          requests:
            cpu: 200m
            memory: 256Mi
          limits:
            cpu: 1000m
            memory: 1Gi

  values:
    global:
      # Proxy resources
      proxy:
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 256Mi
        # Lifecycle hooks for graceful shutdown
        lifecycle:
          preStop:
            exec:
              command:
                - /bin/sh
                - -c
                - sleep 5

      # Logging
      logging:
        level: "default:info"

    # Sidecar injector
    sidecarInjectorWebhook:
      rewriteAppHTTPProbe: true

    # Pilot settings
    pilot:
      autoscaleEnabled: true
      traceSampling: 10.0
```

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new Istio configurations.**

### TDD Cycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ISTIO TDD WORKFLOW                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   1. RED: Write failing validation/test first                       │
│      │                                                              │
│      │   • Define expected behavior in test                         │
│      │   • Run istioctl analyze (should report issues)              │
│      │   • Verify connectivity test fails                           │
│      ▼                                                              │
│   2. GREEN: Write minimal config to make it pass                    │
│      │                                                              │
│      │   • Create VirtualService/DestinationRule                    │
│      │   • Apply configuration                                      │
│      │   • Run istioctl analyze (no issues)                         │
│      ▼                                                              │
│   3. REFACTOR: Improve config while keeping tests green             │
│      │                                                              │
│      │   • Add traffic policies                                     │
│      │   • Optimize connection pools                                │
│      │   • Add observability                                        │
│      ▼                                                              │
│   ┌──┴──┐                                                           │
│   │     │ Repeat for each configuration requirement                 │
│   └─────┘                                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Example TDD Workflow for VirtualService Configuration

```yaml
# ═══════════════════════════════════════════════════════════════════════
# Step 1: RED - Define the test/validation first
# ═══════════════════════════════════════════════════════════════════════

# test-connectivity.sh - Write this FIRST before any VirtualService
#!/bin/bash
set -e

echo "Testing order-service routing..."

# Test 1: Verify VirtualService exists
kubectl get virtualservice order-service -n production || {
    echo "FAIL: VirtualService order-service not found"
    exit 1
}

# Test 2: Verify traffic reaches the service
RESPONSE=$(kubectl exec -n production deploy/test-client -- \
    curl -s -o /dev/null -w "%{http_code}" \
    http://order-service:8080/health)

if [ "$RESPONSE" != "200" ]; then
    echo "FAIL: Expected 200, got $RESPONSE"
    exit 1
fi

# Test 3: Verify canary header routing works
CANARY_RESPONSE=$(kubectl exec -n production deploy/test-client -- \
    curl -s -H "x-canary: true" \
    http://order-service:8080/version)

if [[ "$CANARY_RESPONSE" != *"canary"* ]]; then
    echo "FAIL: Canary routing not working"
    exit 1
fi

echo "PASS: All routing tests passed"

# Run: ./test-connectivity.sh
# FAILS - VirtualService not found (expected in RED phase)
```

```yaml
# ═══════════════════════════════════════════════════════════════════════
# Step 2: GREEN - Write minimal VirtualService to pass tests
# ═══════════════════════════════════════════════════════════════════════

# order-service-virtualservice.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: order-service
  namespace: production
spec:
  hosts:
    - order-service
  http:
    # Canary routing (required by test)
    - match:
        - headers:
            x-canary:
              exact: "true"
      route:
        - destination:
            host: order-service
            subset: canary
            port:
              number: 8080
    # Default routing
    - route:
        - destination:
            host: order-service
            subset: stable
            port:
              number: 8080
---
# Minimal DestinationRule for subsets
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: order-service
  namespace: production
spec:
  host: order-service
  subsets:
    - name: stable
      labels:
        version: stable
    - name: canary
      labels:
        version: canary

# Apply: kubectl apply -f order-service-virtualservice.yaml
# Run: ./test-connectivity.sh
# PASSES - minimal configuration satisfies tests
```

```yaml
# ═══════════════════════════════════════════════════════════════════════
# Step 3: REFACTOR - Enhance configuration while keeping tests green
# ═══════════════════════════════════════════════════════════════════════

apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: order-service
  namespace: production
spec:
  hosts:
    - order-service
    - order-service.production.svc.cluster.local
  http:
    # Canary routing with full traffic control
    - match:
        - headers:
            x-canary:
              exact: "true"
      route:
        - destination:
            host: order-service
            subset: canary
            port:
              number: 8080
      timeout: 30s
      retries:
        attempts: 3
        perTryTimeout: 10s
        retryOn: 5xx,reset,connect-failure

    # Production traffic with weighted routing
    - route:
        - destination:
            host: order-service
            subset: stable
            port:
              number: 8080
          weight: 95
        - destination:
            host: order-service
            subset: canary
            port:
              number: 8080
          weight: 5
      timeout: 30s
      retries:
        attempts: 3
        perTryTimeout: 10s
        retryOn: 5xx,reset,connect-failure
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: order-service
  namespace: production
spec:
  host: order-service
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
        connectTimeout: 5s
      http:
        h2UpgradePolicy: UPGRADE
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
    loadBalancer:
      simple: LEAST_REQUEST
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 10s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
    tls:
      mode: ISTIO_MUTUAL
  subsets:
    - name: stable
      labels:
        version: stable
    - name: canary
      labels:
        version: canary

# Apply: kubectl apply -f order-service-virtualservice.yaml
# Run: ./test-connectivity.sh
# PASSES - enhanced configuration still satisfies all tests

# Run: istioctl analyze -n production
# No validation issues found - configuration is valid
```

### Visual TDD Step-by-Step Example

```
┌─────────────────────────────────────────────────────────────────────────┐
│ TDD FOR ISTIO AUTHORIZATION POLICY                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ REQUIREMENT: Only payment-service can call order-service /api/orders/*  │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ STEP 1: RED - Write authorization test                              │ │
│ ├─────────────────────────────────────────────────────────────────────┤ │
│ │                                                                     │ │
│ │  # test-authz.sh                                                    │ │
│ │  # Test: payment-service CAN access order-service                   │ │
│ │  kubectl exec -n production deploy/payment-service -- \             │ │
│ │      curl -s -o /dev/null -w "%{http_code}" \                       │ │
│ │      http://order-service:8080/api/orders/123                       │ │
│ │  # Expected: 200                                                    │ │
│ │                                                                     │ │
│ │  # Test: other-service CANNOT access order-service                  │ │
│ │  kubectl exec -n production deploy/other-service -- \               │ │
│ │      curl -s -o /dev/null -w "%{http_code}" \                       │ │
│ │      http://order-service:8080/api/orders/123                       │ │
│ │  # Expected: 403 (RBAC denied)                                      │ │
│ │                                                                     │ │
│ │  Result: FAILS (no policy = all traffic allowed)                    │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                            │                                            │
│                            ▼                                            │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ STEP 2: GREEN - Create AuthorizationPolicy                          │ │
│ ├─────────────────────────────────────────────────────────────────────┤ │
│ │                                                                     │ │
│ │  apiVersion: security.istio.io/v1beta1                              │ │
│ │  kind: AuthorizationPolicy                                          │ │
│ │  metadata:                                                          │ │
│ │    name: order-service-authz                                        │ │
│ │    namespace: production                                            │ │
│ │  spec:                                                              │ │
│ │    selector:                                                        │ │
│ │      matchLabels:                                                   │ │
│ │        app: order-service                                           │ │
│ │    action: ALLOW                                                    │ │
│ │    rules:                                                           │ │
│ │      - from:                                                        │ │
│ │          - source:                                                  │ │
│ │              principals:                                            │ │
│ │                - "cluster.local/ns/production/sa/payment-service"   │ │
│ │        to:                                                          │ │
│ │          - operation:                                               │ │
│ │              paths: ["/api/orders/*"]                               │ │
│ │                                                                     │ │
│ │  Result: PASSES (policy enforces access control)                    │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                            │                                            │
│                            ▼                                            │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ STEP 3: REFACTOR - Add health check and metrics exceptions          │ │
│ ├─────────────────────────────────────────────────────────────────────┤ │
│ │                                                                     │ │
│ │  rules:                                                             │ │
│ │    # Health checks from anywhere                                    │ │
│ │    - to:                                                            │ │
│ │        - operation:                                                 │ │
│ │            paths: ["/health/*", "/metrics"]                         │ │
│ │    # Payment service access                                         │ │
│ │    - from:                                                          │ │
│ │        - source:                                                    │ │
│ │            principals: ["cluster.local/ns/production/sa/payment"]   │ │
│ │      to:                                                            │ │
│ │        - operation:                                                 │ │
│ │            paths: ["/api/orders/*"]                                 │ │
│ │            methods: ["GET", "POST"]                                 │ │
│ │                                                                     │ │
│ │  Result: PASSES (all original tests still pass)                     │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every Istio configuration bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ISTIO BUG FIX WORKFLOW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   1. Bug Reported/Discovered                                            │
│      │   (e.g., "503 errors when calling payment-service")              │
│      │                                                                  │
│      ▼                                                                  │
│   2. Write test that REPRODUCES the bug (test will FAIL)                │
│      │   • Create connectivity test script                              │
│      │   • Test should demonstrate the failure                          │
│      │   • Document expected vs actual behavior                         │
│      │                                                                  │
│      ▼                                                                  │
│   3. Verify the test fails for the RIGHT reason                         │
│      │   • Run istioctl analyze                                         │
│      │   • Check proxy logs: kubectl logs <pod> -c istio-proxy          │
│      │   • Verify mTLS status: istioctl x authz check <pod>             │
│      │                                                                  │
│      ▼                                                                  │
│   4. Fix the Istio configuration (make test pass)                       │
│      │   • Apply DestinationRule/PeerAuthentication fix                 │
│      │   • Update VirtualService if routing issue                       │
│      │   • Verify with istioctl analyze                                 │
│      │                                                                  │
│      ▼                                                                  │
│   5. Verify the test now PASSES                                         │
│      │   • Run regression test                                          │
│      │   • Verify no other tests broken                                 │
│      │                                                                  │
│      ▼                                                                  │
│   6. Document the bug in test comments (include ticket ID)              │
│      │                                                                  │
│      ▼                                                                  │
│   7. Add to permanent test suite (regression prevented forever)         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Example Bug Fix: mTLS Mismatch Causing 503 Errors

```yaml
# ═══════════════════════════════════════════════════════════════════════
# Bug Report #IST-1234: 503 Service Unavailable when calling payment-service
#
# Symptoms:
# - Intermittent 503 errors from order-service to payment-service
# - Error in logs: "upstream connect error or disconnect/reset before headers"
# - Started after enabling STRICT mTLS on payment-service namespace
# ═══════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────
# Step 1-2: Write test that REPRODUCES the bug
# ───────────────────────────────────────────────────────────────────────

# test-ist-1234-mtls-regression.sh
#!/bin/bash
# Bug #IST-1234: Verify mTLS connectivity between order and payment services

set -e

echo "=== Bug #IST-1234 Regression Test ==="
echo "Testing mTLS connectivity: order-service -> payment-service"

# Test: order-service can reach payment-service
RESPONSE=$(kubectl exec -n production deploy/order-service -c order-service -- \
    curl -s -o /dev/null -w "%{http_code}" \
    --max-time 10 \
    http://payment-service.payment.svc.cluster.local:8080/health)

if [ "$RESPONSE" != "200" ]; then
    echo "FAIL: Bug #IST-1234 - Expected 200, got $RESPONSE"
    echo "mTLS mismatch likely causing 503 errors"

    # Diagnostic info
    echo ""
    echo "=== Diagnostic Information ==="
    echo "Checking mTLS status..."
    istioctl x authz check deploy/payment-service -n payment 2>/dev/null || true

    echo ""
    echo "Checking DestinationRule..."
    kubectl get destinationrule -n production -o yaml 2>/dev/null | grep -A5 "payment-service" || echo "No DestinationRule found"

    exit 1
fi

echo "PASS: Bug #IST-1234 regression test passed"
echo "mTLS connectivity working correctly"

# Run: ./test-ist-1234-mtls-regression.sh
# FAILS - demonstrates the bug exists
```

```yaml
# ───────────────────────────────────────────────────────────────────────
# Step 3: Root cause analysis
# ───────────────────────────────────────────────────────────────────────

# Diagnosis commands:
#
# 1. Check PeerAuthentication in payment namespace
#    kubectl get peerauthentication -n payment
#    > Shows: STRICT mTLS enabled
#
# 2. Check if DestinationRule exists for cross-namespace call
#    kubectl get destinationrule -n production | grep payment
#    > Shows: No DestinationRule (MISSING!)
#
# 3. Check proxy config
#    istioctl proxy-config cluster deploy/order-service -n production | grep payment
#    > Shows: outbound|8080||payment-service.payment.svc.cluster.local
#    > TLS mode: DISABLED (should be ISTIO_MUTUAL)
#
# ROOT CAUSE: Missing DestinationRule in production namespace for
#             cross-namespace mTLS to payment-service

# ───────────────────────────────────────────────────────────────────────
# Step 4: Fix the bug - Add missing DestinationRule
# ───────────────────────────────────────────────────────────────────────

# fix-ist-1234-mtls.yaml
# Bug Fix #IST-1234: Add DestinationRule for cross-namespace mTLS
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: payment-service-mtls
  namespace: production  # In the CALLER's namespace
  labels:
    bug-fix: "IST-1234"
    description: "mTLS for cross-namespace call to payment-service"
spec:
  host: payment-service.payment.svc.cluster.local  # Full FQDN required
  trafficPolicy:
    tls:
      mode: ISTIO_MUTUAL  # Enable mTLS to match STRICT PeerAuthentication
    connectionPool:
      tcp:
        maxConnections: 100
        connectTimeout: 5s
      http:
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 10s
      baseEjectionTime: 30s

# Apply: kubectl apply -f fix-ist-1234-mtls.yaml
```

```bash
# ───────────────────────────────────────────────────────────────────────
# Step 5: Verify the fix
# ───────────────────────────────────────────────────────────────────────

# Run regression test
./test-ist-1234-mtls-regression.sh
# PASSES - bug is fixed

# Verify with istioctl
istioctl analyze -n production
# No validation issues found

# Verify mTLS is now active
istioctl proxy-config cluster deploy/order-service -n production | grep payment
# outbound|8080||payment-service.payment.svc.cluster.local - TLS: ISTIO_MUTUAL

# ───────────────────────────────────────────────────────────────────────
# Step 6-7: Document and add to permanent test suite
# ───────────────────────────────────────────────────────────────────────

# Move test to permanent regression suite
mv test-ist-1234-mtls-regression.sh tests/regression/

# Add to CI/CD pipeline
# tests/regression/run-all.sh:
#   ./test-ist-1234-mtls-regression.sh  # Bug #IST-1234: mTLS cross-namespace
```

### Visual Bug Fix Example: VirtualService Routing Issue

```
┌─────────────────────────────────────────────────────────────────────────┐
│ BUG FIX: Canary traffic not being routed correctly                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ Bug #IST-5678: Canary header "x-canary: true" routes to stable instead  │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ STEP 1: Write regression test                                       │ │
│ ├─────────────────────────────────────────────────────────────────────┤ │
│ │                                                                     │ │
│ │  # test-ist-5678-canary-routing.sh                                  │ │
│ │  VERSION=$(kubectl exec deploy/test-client -n production -- \       │ │
│ │      curl -s -H "x-canary: true" http://api-service:8080/version)   │ │
│ │                                                                     │ │
│ │  if [[ "$VERSION" != *"canary"* ]]; then                            │ │
│ │      echo "FAIL: Bug #IST-5678 - Got '$VERSION', expected 'canary'" │ │
│ │      exit 1                                                         │ │
│ │  fi                                                                 │ │
│ │                                                                     │ │
│ │  Result: FAILS - returns "stable" instead of "canary"               │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                            │                                            │
│                            ▼                                            │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ STEP 2: Identify root cause                                         │ │
│ ├─────────────────────────────────────────────────────────────────────┤ │
│ │                                                                     │ │
│ │  # Check VirtualService                                             │ │
│ │  kubectl get vs api-service -n production -o yaml                   │ │
│ │                                                                     │ │
│ │  # PROBLEM FOUND: Match order is wrong!                             │ │
│ │  http:                                                              │ │
│ │    - route:              # <-- Default route FIRST (catches all)    │ │
│ │        - destination:                                               │ │
│ │            subset: stable                                           │ │
│ │    - match:              # <-- Canary match SECOND (never reached)  │ │
│ │        - headers:                                                   │ │
│ │            x-canary:                                                │ │
│ │              exact: "true"                                          │ │
│ │      route:                                                         │ │
│ │        - destination:                                               │ │
│ │            subset: canary                                           │ │
│ │                                                                     │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                            │                                            │
│                            ▼                                            │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ STEP 3: Fix - Correct match order                                   │ │
│ ├─────────────────────────────────────────────────────────────────────┤ │
│ │                                                                     │ │
│ │  # fix-ist-5678-virtualservice.yaml                                 │ │
│ │  apiVersion: networking.istio.io/v1beta1                            │ │
│ │  kind: VirtualService                                               │ │
│ │  metadata:                                                          │ │
│ │    name: api-service                                                │ │
│ │    labels:                                                          │ │
│ │      bug-fix: "IST-5678"                                            │ │
│ │  spec:                                                              │ │
│ │    http:                                                            │ │
│ │      - match:            # <-- Specific match FIRST                 │ │
│ │          - headers:                                                 │ │
│ │              x-canary:                                              │ │
│ │                exact: "true"                                        │ │
│ │        route:                                                       │ │
│ │          - destination:                                             │ │
│ │              subset: canary                                         │ │
│ │      - route:            # <-- Default route LAST                   │ │
│ │          - destination:                                             │ │
│ │              subset: stable                                         │ │
│ │                                                                     │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                            │                                            │
│                            ▼                                            │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ STEP 4: Verify fix and add to regression suite                      │ │
│ ├─────────────────────────────────────────────────────────────────────┤ │
│ │                                                                     │ │
│ │  kubectl apply -f fix-ist-5678-virtualservice.yaml                  │ │
│ │  ./test-ist-5678-canary-routing.sh                                  │ │
│ │                                                                     │ │
│ │  Result: PASSES - canary routing now works correctly                │ │
│ │                                                                     │ │
│ │  # Add to permanent test suite                                      │ │
│ │  mv test-ist-5678-canary-routing.sh tests/regression/               │ │
│ │                                                                     │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ KEY LEARNING: In VirtualService, specific matches MUST come before      │
│               default routes. Istio evaluates matches in order.         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Bug Fix Checklist

```
ISTIO BUG FIX VERIFICATION:

□ Regression Test Written
  □ Test reproduces the exact bug behavior
  □ Test fails BEFORE the fix is applied
  □ Test is documented with bug ID and description
  □ Test includes diagnostic output on failure

□ Root Cause Identified
  □ Ran istioctl analyze
  □ Checked proxy logs (kubectl logs -c istio-proxy)
  □ Verified mTLS status (istioctl x authz check)
  □ Inspected relevant resources (VS, DR, PA, AP)

□ Fix Applied
  □ Configuration change is minimal and targeted
  □ Fix includes bug ID in labels/annotations
  □ istioctl analyze passes after fix
  □ No unintended side effects on other services

□ Verification Complete
  □ Regression test now PASSES
  □ All existing tests still pass
  □ Manual verification performed
  □ Monitoring confirms fix (no more errors)

□ Documentation Updated
  □ Test added to permanent regression suite
  □ Bug documented in runbook/wiki
  □ Team notified of root cause and fix
```

---

## 3. Static Gateway IP Addresses (MANDATORY)

### A. Generic Kubernetes (MetalLB / On-Premises)

```yaml
# ✅ CORRECT - Static IP with MetalLB or bare-metal
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: istio-control-plane
  namespace: istio-system
spec:
  components:
    ingressGateways:
      - name: istio-ingressgateway
        enabled: true
        k8s:
          service:
            type: LoadBalancer
            loadBalancerIP: "192.168.1.100"  # Static IP
            externalTrafficPolicy: Local     # Preserve source IP
          serviceAnnotations:
            # MetalLB specific
            metallb.universe.tf/loadBalancerIPs: "192.168.1.100"
---
# Alternative: Patch existing gateway service
apiVersion: v1
kind: Service
metadata:
  name: istio-ingressgateway
  namespace: istio-system
  annotations:
    metallb.universe.tf/loadBalancerIPs: "192.168.1.100"
spec:
  type: LoadBalancer
  loadBalancerIP: "192.168.1.100"
  externalTrafficPolicy: Local
```

### B. Azure AKS - Static IP Configuration

```bash
# ✅ CORRECT - Create static public IP in Azure

# 1. Get the AKS node resource group
AKS_RESOURCE_GROUP="myAKSResourceGroup"
AKS_CLUSTER_NAME="myAKSCluster"
NODE_RESOURCE_GROUP=$(az aks show \
  --resource-group $AKS_RESOURCE_GROUP \
  --name $AKS_CLUSTER_NAME \
  --query nodeResourceGroup -o tsv)

# 2. Create static public IP
STATIC_IP_NAME="istio-ingress-ip"
az network public-ip create \
  --resource-group $NODE_RESOURCE_GROUP \
  --name $STATIC_IP_NAME \
  --sku Standard \
  --allocation-method Static \
  --zone 1 2 3  # Zone-redundant

# 3. Get the IP address
STATIC_IP=$(az network public-ip show \
  --resource-group $NODE_RESOURCE_GROUP \
  --name $STATIC_IP_NAME \
  --query ipAddress -o tsv)

echo "Static IP: $STATIC_IP"
```

```yaml
# ✅ CORRECT - AKS with static IP (Manual Istio installation)
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: istio-control-plane
  namespace: istio-system
spec:
  components:
    ingressGateways:
      - name: istio-ingressgateway
        enabled: true
        k8s:
          service:
            type: LoadBalancer
            loadBalancerIP: "${STATIC_IP}"  # Your Azure static IP
            externalTrafficPolicy: Local
          serviceAnnotations:
            # Azure Load Balancer annotations
            service.beta.kubernetes.io/azure-load-balancer-resource-group: "${NODE_RESOURCE_GROUP}"
            service.beta.kubernetes.io/azure-dns-label-name: "myapp-ingress"
            # Health probe
            service.beta.kubernetes.io/azure-load-balancer-health-probe-request-path: "/healthz/ready"
            service.beta.kubernetes.io/port_443_health-probe_protocol: "https"
            service.beta.kubernetes.io/port_443_health-probe_port: "15021"
---
# For existing installation, patch the service
apiVersion: v1
kind: Service
metadata:
  name: istio-ingressgateway
  namespace: istio-system
  annotations:
    service.beta.kubernetes.io/azure-load-balancer-resource-group: "MC_myAKSResourceGroup_myAKSCluster_eastus"
    service.beta.kubernetes.io/azure-dns-label-name: "myapp-ingress"
spec:
  type: LoadBalancer
  loadBalancerIP: "20.120.xxx.xxx"  # Your Azure static IP
  externalTrafficPolicy: Local
```

### C. Azure AKS - Istio Add-on (Managed)

```bash
# ✅ CORRECT - Enable Istio add-on on AKS

# 1. Enable Istio add-on
az aks mesh enable \
  --resource-group myAKSResourceGroup \
  --name myAKSCluster

# 2. Enable external ingress gateway
az aks mesh enable-ingress-gateway \
  --resource-group myAKSResourceGroup \
  --name myAKSCluster \
  --ingress-gateway-type external

# 3. Verify
az aks show \
  --resource-group myAKSResourceGroup \
  --name myAKSCluster \
  --query 'serviceMeshProfile'

kubectl get pods -n aks-istio-system
kubectl get svc -n aks-istio-ingress
```

```yaml
# ✅ CORRECT - AKS Istio add-on with static IP
# Patch the managed ingress gateway service
apiVersion: v1
kind: Service
metadata:
  name: aks-istio-ingressgateway-external
  namespace: aks-istio-ingress
  annotations:
    service.beta.kubernetes.io/azure-load-balancer-resource-group: "MC_myAKSResourceGroup_myAKSCluster_eastus"
spec:
  type: LoadBalancer
  loadBalancerIP: "20.120.xxx.xxx"  # Your pre-created static IP
  externalTrafficPolicy: Local
---
# Gateway using AKS Istio add-on
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: main-gateway
  namespace: aks-istio-ingress  # AKS Istio add-on namespace
spec:
  selector:
    istio: aks-istio-ingressgateway-external  # AKS add-on selector
  servers:
    - port:
        number: 443
        name: https
        protocol: HTTPS
      tls:
        mode: SIMPLE
        credentialName: gateway-tls-secret
      hosts:
        - "*.myapp.com"
    - port:
        number: 80
        name: http
        protocol: HTTP
      hosts:
        - "*.myapp.com"
      tls:
        httpsRedirect: true
```

```yaml
# ✅ CORRECT - AKS Istio add-on internal gateway (Private)
# Enable internal ingress gateway
# az aks mesh enable-ingress-gateway \
#   --resource-group myAKSResourceGroup \
#   --name myAKSCluster \
#   --ingress-gateway-type internal

apiVersion: v1
kind: Service
metadata:
  name: aks-istio-ingressgateway-internal
  namespace: aks-istio-ingress
  annotations:
    service.beta.kubernetes.io/azure-load-balancer-internal: "true"
    service.beta.kubernetes.io/azure-load-balancer-internal-subnet: "ingress-subnet"
spec:
  type: LoadBalancer
  loadBalancerIP: "10.0.1.100"  # Private static IP
```

### D. AWS EKS - Static IP Configuration

```yaml
# ✅ CORRECT - EKS with Network Load Balancer and Elastic IP
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: istio-control-plane
  namespace: istio-system
spec:
  components:
    ingressGateways:
      - name: istio-ingressgateway
        enabled: true
        k8s:
          service:
            type: LoadBalancer
            externalTrafficPolicy: Local
          serviceAnnotations:
            # Use NLB instead of CLB
            service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
            # Use external NLB (internet-facing)
            service.beta.kubernetes.io/aws-load-balancer-scheme: "internet-facing"
            # Cross-zone load balancing
            service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
            # Target type (instance for NodePort)
            service.beta.kubernetes.io/aws-load-balancer-nlb-target-type: "instance"
            # Health check
            service.beta.kubernetes.io/aws-load-balancer-healthcheck-protocol: "TCP"
            service.beta.kubernetes.io/aws-load-balancer-healthcheck-port: "15021"
            service.beta.kubernetes.io/aws-load-balancer-healthcheck-path: "/healthz/ready"
---
# For static Elastic IPs with AWS Load Balancer Controller
apiVersion: v1
kind: Service
metadata:
  name: istio-ingressgateway
  namespace: istio-system
  annotations:
    # AWS Load Balancer Controller annotations
    service.beta.kubernetes.io/aws-load-balancer-type: "external"
    service.beta.kubernetes.io/aws-load-balancer-nlb-target-type: "ip"
    service.beta.kubernetes.io/aws-load-balancer-scheme: "internet-facing"
    # Static Elastic IPs (must be pre-allocated)
    service.beta.kubernetes.io/aws-load-balancer-eip-allocations: "eipalloc-xxxxxxxxx,eipalloc-yyyyyyyyy"
    # Subnets with the Elastic IPs
    service.beta.kubernetes.io/aws-load-balancer-subnets: "subnet-aaaa,subnet-bbbb"
    # Enable cross-zone
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
    # Proxy protocol v2 (optional, for source IP)
    service.beta.kubernetes.io/aws-load-balancer-proxy-protocol: "*"
spec:
  type: LoadBalancer
  externalTrafficPolicy: Local
```

```bash
# ✅ CORRECT - Pre-allocate Elastic IPs in AWS

# 1. Allocate Elastic IPs
EIP_1=$(aws ec2 allocate-address --domain vpc --query 'AllocationId' --output text)
EIP_2=$(aws ec2 allocate-address --domain vpc --query 'AllocationId' --output text)

echo "Elastic IP Allocations: $EIP_1, $EIP_2"

# 2. Tag for identification
aws ec2 create-tags --resources $EIP_1 $EIP_2 \
  --tags Key=Name,Value=istio-ingress Key=Environment,Value=production

# 3. Get the actual IP addresses
aws ec2 describe-addresses --allocation-ids $EIP_1 $EIP_2 \
  --query 'Addresses[*].PublicIp' --output table

# 4. Update DNS records with these static IPs
```

```yaml
# ✅ CORRECT - EKS with internal NLB (Private)
apiVersion: v1
kind: Service
metadata:
  name: istio-ingressgateway-internal
  namespace: istio-system
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "external"
    service.beta.kubernetes.io/aws-load-balancer-nlb-target-type: "ip"
    service.beta.kubernetes.io/aws-load-balancer-scheme: "internal"
    service.beta.kubernetes.io/aws-load-balancer-subnets: "subnet-private-1,subnet-private-2"
    # Private static IP (not Elastic IP)
    service.beta.kubernetes.io/aws-load-balancer-private-ipv4-addresses: "10.0.1.100,10.0.2.100"
spec:
  type: LoadBalancer
  selector:
    app: istio-ingressgateway
    istio: ingressgateway
```

---

## 4. mTLS Configuration (MANDATORY)

### A. Strict mTLS Enforcement

**CRITICAL: Enable STRICT mTLS for all production workloads.**

```yaml
# ✅ CORRECT - Mesh-wide STRICT mTLS
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: istio-system  # Mesh-wide when in istio-system
spec:
  mtls:
    mode: STRICT
---
# Per-namespace STRICT mTLS (recommended)
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: production
spec:
  mtls:
    mode: STRICT
---
# Per-workload mTLS (for exceptions)
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: legacy-service-permissive
  namespace: production
spec:
  selector:
    matchLabels:
      app: legacy-service
  mtls:
    mode: PERMISSIVE  # Only for legacy services during migration
  portLevelMtls:
    8080:
      mode: STRICT  # But enforce on specific ports
```

### B. Destination Rules for mTLS

```yaml
# ✅ CORRECT - DestinationRule enforcing mTLS
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: default-mtls
  namespace: istio-system
spec:
  host: "*.local"  # All services in mesh
  trafficPolicy:
    tls:
      mode: ISTIO_MUTUAL
---
# Service-specific DestinationRule
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: order-service
  namespace: production
spec:
  host: order-service.production.svc.cluster.local
  trafficPolicy:
    tls:
      mode: ISTIO_MUTUAL
    connectionPool:
      tcp:
        maxConnections: 100
        connectTimeout: 5s
      http:
        h2UpgradePolicy: UPGRADE
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 10s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
```

### C. Certificate Management

```yaml
# ✅ CORRECT - External CA integration (cert-manager)
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: istio-control-plane
  namespace: istio-system
spec:
  values:
    global:
      pilotCertProvider: istiod  # Default: istiod manages certs
  meshConfig:
    # Custom CA settings
    caCertificates:
      - pem: |
          -----BEGIN CERTIFICATE-----
          # Your CA certificate
          -----END CERTIFICATE-----
    defaultConfig:
      proxyMetadata:
        # Custom root CA
        ISTIO_META_TLS_CLIENT_ROOT_CERT: /etc/certs/root-cert.pem
---
# For cert-manager integration (istio-csr)
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: istio-ca
  namespace: istio-system
spec:
  isCA: true
  duration: 87600h  # 10 years
  secretName: istio-ca-secret
  commonName: istio-ca
  issuerRef:
    name: selfsigned-issuer
    kind: ClusterIssuer
    group: cert-manager.io
```

---

## 5. Authorization Policies (MANDATORY)

### A. Default Deny Policy

**CRITICAL: Implement deny-by-default, then explicitly allow traffic.**

```yaml
# ✅ CORRECT - Deny all traffic by default (per namespace)
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: deny-all
  namespace: production
spec:
  {}  # Empty spec = deny all
---
# ✅ CORRECT - Mesh-wide deny all (in istio-system)
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: deny-all
  namespace: istio-system
spec:
  {}
```

### B. Service-to-Service Authorization

```yaml
# ✅ CORRECT - Allow specific service-to-service communication
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: order-service-policy
  namespace: production
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
            # For AKS Istio add-on:
            # - "cluster.local/ns/aks-istio-ingress/sa/aks-istio-ingressgateway-external"
      to:
        - operation:
            methods: ["GET", "POST", "PUT", "DELETE"]
            paths: ["/api/*", "/health/*"]

    # Allow from payment service
    - from:
        - source:
            principals:
              - "cluster.local/ns/production/sa/payment-service"
      to:
        - operation:
            methods: ["GET"]
            paths: ["/api/orders/*"]

    # Allow from specific namespace
    - from:
        - source:
            namespaces: ["monitoring"]
      to:
        - operation:
            methods: ["GET"]
            paths: ["/metrics", "/health/*"]

    # Deny specific paths (even if other rules allow)
---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: deny-admin-external
  namespace: production
spec:
  selector:
    matchLabels:
      app: order-service
  action: DENY
  rules:
    - from:
        - source:
            notPrincipals:
              - "cluster.local/ns/production/sa/admin-service"
      to:
        - operation:
            paths: ["/admin/*", "/internal/*"]
```

### C. JWT Authentication

```yaml
# ✅ CORRECT - JWT authentication
apiVersion: security.istio.io/v1beta1
kind: RequestAuthentication
metadata:
  name: jwt-auth
  namespace: production
spec:
  selector:
    matchLabels:
      app: order-service
  jwtRules:
    # Auth0
    - issuer: "https://myapp.auth0.com/"
      jwksUri: "https://myapp.auth0.com/.well-known/jwks.json"
      audiences:
        - "https://api.myapp.com"
      forwardOriginalToken: true
      outputClaimToHeaders:
        - header: "x-user-id"
          claim: "sub"
        - header: "x-user-email"
          claim: "email"

    # Azure AD
    - issuer: "https://login.microsoftonline.com/{tenant-id}/v2.0"
      jwksUri: "https://login.microsoftonline.com/{tenant-id}/discovery/v2.0/keys"
      audiences:
        - "api://myapp-api"

    # AWS Cognito
    - issuer: "https://cognito-idp.{region}.amazonaws.com/{user-pool-id}"
      jwksUri: "https://cognito-idp.{region}.amazonaws.com/{user-pool-id}/.well-known/jwks.json"
---
# Require valid JWT for API access
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: require-jwt
  namespace: production
spec:
  selector:
    matchLabels:
      app: order-service
  action: ALLOW
  rules:
    # Public endpoints (no JWT required)
    - to:
        - operation:
            paths: ["/health/*", "/public/*"]

    # Authenticated endpoints
    - from:
        - source:
            requestPrincipals: ["*"]  # Any valid JWT
      to:
        - operation:
            paths: ["/api/*"]

    # Admin endpoints (specific claim required)
    - from:
        - source:
            requestPrincipals: ["*"]
      when:
        - key: request.auth.claims[roles]
          values: ["admin"]
      to:
        - operation:
            paths: ["/admin/*"]
```

---

## 6. Traffic Management (MANDATORY)

### A. Gateway Configuration

```yaml
# ✅ CORRECT - Production Gateway with TLS
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: main-gateway
  namespace: istio-system  # or aks-istio-ingress for AKS add-on
spec:
  selector:
    istio: ingressgateway  # or aks-istio-ingressgateway-external for AKS
  servers:
    # HTTPS with TLS termination
    - port:
        number: 443
        name: https
        protocol: HTTPS
      tls:
        mode: SIMPLE
        credentialName: gateway-tls-secret  # Kubernetes TLS secret
        minProtocolVersion: TLSV1_2
        cipherSuites:
          - ECDHE-ECDSA-AES128-GCM-SHA256
          - ECDHE-RSA-AES128-GCM-SHA256
          - ECDHE-ECDSA-AES256-GCM-SHA384
          - ECDHE-RSA-AES256-GCM-SHA384
      hosts:
        - "api.myapp.com"
        - "app.myapp.com"

    # HTTP to HTTPS redirect
    - port:
        number: 80
        name: http
        protocol: HTTP
      hosts:
        - "api.myapp.com"
        - "app.myapp.com"
      tls:
        httpsRedirect: true

    # Wildcard subdomain
    - port:
        number: 443
        name: https-wildcard
        protocol: HTTPS
      tls:
        mode: SIMPLE
        credentialName: wildcard-tls-secret
      hosts:
        - "*.myapp.com"
---
# TLS Secret (created by cert-manager or manually)
apiVersion: v1
kind: Secret
metadata:
  name: gateway-tls-secret
  namespace: istio-system  # Must be in same namespace as gateway
type: kubernetes.io/tls
data:
  tls.crt: <base64-encoded-cert>
  tls.key: <base64-encoded-key>
```

### B. VirtualService with Traffic Control

```yaml
# ✅ CORRECT - VirtualService with full traffic control
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: order-service
  namespace: production
spec:
  hosts:
    - "api.myapp.com"
    - order-service.production.svc.cluster.local
  gateways:
    - istio-system/main-gateway  # External traffic
    - mesh                        # Internal traffic
  http:
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
      timeout: 30s
      retries:
        attempts: 3
        perTryTimeout: 10s
        retryOn: 5xx,reset,connect-failure,retriable-4xx

    # A/B testing (cookie-based)
    - match:
        - headers:
            cookie:
              regex: ".*version=v2.*"
          uri:
            prefix: /api/v1/orders
      route:
        - destination:
            host: order-service
            subset: v2

    # Weighted routing (gradual rollout)
    - match:
        - uri:
            prefix: /api/v1/orders
      route:
        - destination:
            host: order-service
            subset: stable
            port:
              number: 8080
          weight: 90
        - destination:
            host: order-service
            subset: canary
            port:
              number: 8080
          weight: 10

      # Timeouts
      timeout: 30s

      # Retries
      retries:
        attempts: 3
        perTryTimeout: 10s
        retryOn: 5xx,reset,connect-failure,retriable-4xx
        retryRemoteLocalities: true

      # Fault injection (testing only)
      # fault:
      #   delay:
      #     percentage:
      #       value: 10
      #     fixedDelay: 5s
      #   abort:
      #     percentage:
      #       value: 5
      #     httpStatus: 503

      # CORS policy
      corsPolicy:
        allowOrigins:
          - exact: "https://app.myapp.com"
          - regex: "https://.*\\.myapp\\.com"
        allowMethods:
          - GET
          - POST
          - PUT
          - DELETE
          - OPTIONS
        allowHeaders:
          - authorization
          - content-type
          - x-requested-with
        exposeHeaders:
          - x-request-id
        maxAge: "24h"
        allowCredentials: true

    # Default route
    - route:
        - destination:
            host: order-service
            port:
              number: 8080
---
# DestinationRule with subsets
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: order-service
  namespace: production
spec:
  host: order-service
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
        connectTimeout: 5s
      http:
        h2UpgradePolicy: UPGRADE
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
        maxRequestsPerConnection: 100
    loadBalancer:
      simple: LEAST_REQUEST
      localityLbSetting:
        enabled: true
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 10s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
    tls:
      mode: ISTIO_MUTUAL
  subsets:
    - name: stable
      labels:
        version: stable
    - name: canary
      labels:
        version: canary
    - name: v2
      labels:
        version: v2
```

### C. Rate Limiting

```yaml
# ✅ CORRECT - Rate limiting with EnvoyFilter
apiVersion: networking.istio.io/v1alpha3
kind: EnvoyFilter
metadata:
  name: rate-limit
  namespace: istio-system
spec:
  workloadSelector:
    labels:
      istio: ingressgateway
  configPatches:
    - applyTo: HTTP_FILTER
      match:
        context: GATEWAY
        listener:
          filterChain:
            filter:
              name: envoy.filters.network.http_connection_manager
              subFilter:
                name: envoy.filters.http.router
      patch:
        operation: INSERT_BEFORE
        value:
          name: envoy.filters.http.local_ratelimit
          typed_config:
            "@type": type.googleapis.com/udpa.type.v1.TypedStruct
            type_url: type.googleapis.com/envoy.extensions.filters.http.local_ratelimit.v3.LocalRateLimit
            value:
              stat_prefix: http_local_rate_limiter
              token_bucket:
                max_tokens: 1000
                tokens_per_fill: 100
                fill_interval: 1s
              filter_enabled:
                runtime_key: local_rate_limit_enabled
                default_value:
                  numerator: 100
                  denominator: HUNDRED
              filter_enforced:
                runtime_key: local_rate_limit_enforced
                default_value:
                  numerator: 100
                  denominator: HUNDRED
              response_headers_to_add:
                - append: false
                  header:
                    key: x-rate-limit
                    value: "1000"
```

---

## 7. Observability (MANDATORY)

### A. Distributed Tracing

```yaml
# ✅ CORRECT - Tracing configuration
apiVersion: telemetry.istio.io/v1alpha1
kind: Telemetry
metadata:
  name: mesh-default
  namespace: istio-system
spec:
  tracing:
    - providers:
        - name: jaeger
      randomSamplingPercentage: 10.0
      customTags:
        environment:
          literal:
            value: production
        cluster:
          literal:
            value: primary
---
# Per-service tracing (higher sampling for debugging)
apiVersion: telemetry.istio.io/v1alpha1
kind: Telemetry
metadata:
  name: order-service-tracing
  namespace: production
spec:
  selector:
    matchLabels:
      app: order-service
  tracing:
    - providers:
        - name: jaeger
      randomSamplingPercentage: 50.0  # Higher sampling for debugging
      customTags:
        service:
          literal:
            value: order-service
```

### B. Metrics Configuration

```yaml
# ✅ CORRECT - Custom metrics
apiVersion: telemetry.istio.io/v1alpha1
kind: Telemetry
metadata:
  name: custom-metrics
  namespace: production
spec:
  metrics:
    - providers:
        - name: prometheus
      overrides:
        - match:
            metric: REQUEST_COUNT
            mode: CLIENT_AND_SERVER
          tagOverrides:
            request_path:
              operation: UPSERT
              value: request.url_path
            user_agent:
              operation: UPSERT
              value: request.headers["user-agent"]
---
# Prometheus ServiceMonitor for Istio
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: istio-component-monitor
  namespace: monitoring
  labels:
    monitoring: istio-components
spec:
  jobLabel: istio
  targetLabels: [app]
  selector:
    matchExpressions:
      - key: istio
        operator: In
        values: [pilot, ingressgateway, egressgateway]
  namespaceSelector:
    matchNames:
      - istio-system
  endpoints:
    - port: http-monitoring
      interval: 15s
---
# ServiceMonitor for Envoy sidecars
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: envoy-stats-monitor
  namespace: monitoring
spec:
  selector:
    matchExpressions:
      - key: security.istio.io/tlsMode
        operator: Exists
  namespaceSelector:
    any: true
  podMetricsEndpoints:
    - port: http-envoy-prom
      path: /stats/prometheus
      interval: 15s
```

### C. Access Logging

```yaml
# ✅ CORRECT - Access logging configuration
apiVersion: telemetry.istio.io/v1alpha1
kind: Telemetry
metadata:
  name: access-logging
  namespace: istio-system
spec:
  accessLogging:
    - providers:
        - name: envoy
      filter:
        expression: "response.code >= 400 || connection.mtls == false"
---
# Per-namespace logging (all requests)
apiVersion: telemetry.istio.io/v1alpha1
kind: Telemetry
metadata:
  name: full-access-logging
  namespace: production
spec:
  accessLogging:
    - providers:
        - name: envoy
```

### D. Kiali and Jaeger Deployment

```yaml
# ✅ CORRECT - Kiali for visualization
apiVersion: kiali.io/v1alpha1
kind: Kiali
metadata:
  name: kiali
  namespace: istio-system
spec:
  auth:
    strategy: openid  # or token, anonymous for dev
    openid:
      client_id: kiali
      issuer_uri: https://auth.myapp.com
  deployment:
    accessible_namespaces:
      - "**"
    view_only_mode: false
  external_services:
    prometheus:
      url: http://prometheus.monitoring.svc.cluster.local:9090
    tracing:
      enabled: true
      in_cluster_url: http://jaeger-query.observability.svc.cluster.local:16685/jaeger
      url: https://jaeger.myapp.com
    grafana:
      enabled: true
      in_cluster_url: http://grafana.monitoring.svc.cluster.local:3000
      url: https://grafana.myapp.com
  server:
    web_root: /kiali
```

---

## 8. Security Hardening (MANDATORY)

### A. Control Plane Security

```yaml
# ✅ CORRECT - Hardened istiod configuration
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: istio-control-plane
  namespace: istio-system
spec:
  meshConfig:
    # Strict mTLS
    enableAutoMtls: true

    # Outbound traffic policy - only allow registered services
    outboundTrafficPolicy:
      mode: REGISTRY_ONLY

    # Trust domain
    trustDomain: cluster.local

    # Access log for security events
    accessLogFile: /dev/stdout

  values:
    global:
      # Disable privileged init containers where possible
      proxy:
        privileged: false

      # Image pull policy
      imagePullPolicy: Always

    pilot:
      # Enable network policy
      enableNetworkPolicy: true

    # Disable unnecessary features
    telemetry:
      enabled: true
      v2:
        enabled: true
        prometheus:
          enabled: true
---
# Network policy for istiod
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: istiod-network-policy
  namespace: istio-system
spec:
  podSelector:
    matchLabels:
      app: istiod
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow webhook traffic from API server
    - from: []
      ports:
        - protocol: TCP
          port: 15017
    # Allow xDS from sidecars
    - from:
        - namespaceSelector: {}
      ports:
        - protocol: TCP
          port: 15012
    # Allow metrics scraping
    - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
      ports:
        - protocol: TCP
          port: 15014
  egress:
    # Allow DNS
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: kube-system
      ports:
        - protocol: UDP
          port: 53
    # Allow API server
    - to: []
      ports:
        - protocol: TCP
          port: 443
        - protocol: TCP
          port: 6443
```

### B. Sidecar Resource Limits

```yaml
# ✅ CORRECT - Sidecar injection with resource limits
apiVersion: v1
kind: ConfigMap
metadata:
  name: istio-sidecar-injector
  namespace: istio-system
data:
  values: |
    {
      "global": {
        "proxy": {
          "resources": {
            "requests": {
              "cpu": "100m",
              "memory": "128Mi"
            },
            "limits": {
              "cpu": "500m",
              "memory": "256Mi"
            }
          },
          "privileged": false,
          "enableCoreDump": false
        }
      }
    }
---
# Per-pod resource override via annotation
apiVersion: apps/v1
kind: Deployment
metadata:
  name: high-traffic-service
spec:
  template:
    metadata:
      annotations:
        sidecar.istio.io/proxyCPU: "200m"
        sidecar.istio.io/proxyMemory: "256Mi"
        sidecar.istio.io/proxyCPULimit: "1000m"
        sidecar.istio.io/proxyMemoryLimit: "512Mi"
```

### C. Egress Control

```yaml
# ✅ CORRECT - Controlled egress via egress gateway
apiVersion: networking.istio.io/v1beta1
kind: ServiceEntry
metadata:
  name: external-api
  namespace: production
spec:
  hosts:
    - api.external-service.com
  ports:
    - number: 443
      name: https
      protocol: HTTPS
  location: MESH_EXTERNAL
  resolution: DNS
---
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: egress-gateway
  namespace: istio-system
spec:
  selector:
    istio: egressgateway
  servers:
    - port:
        number: 443
        name: https
        protocol: HTTPS
      hosts:
        - api.external-service.com
      tls:
        mode: PASSTHROUGH
---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: external-api-egress
  namespace: production
spec:
  hosts:
    - api.external-service.com
  gateways:
    - mesh
    - istio-system/egress-gateway
  tls:
    - match:
        - gateways:
            - mesh
          port: 443
          sniHosts:
            - api.external-service.com
      route:
        - destination:
            host: istio-egressgateway.istio-system.svc.cluster.local
            port:
              number: 443
    - match:
        - gateways:
            - istio-system/egress-gateway
          port: 443
          sniHosts:
            - api.external-service.com
      route:
        - destination:
            host: api.external-service.com
            port:
              number: 443
---
# Authorization policy for egress
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-external-api
  namespace: istio-system
spec:
  selector:
    matchLabels:
      istio: egressgateway
  action: ALLOW
  rules:
    - from:
        - source:
            namespaces: ["production"]
      to:
        - operation:
            hosts: ["api.external-service.com"]
```

---

## 9. Platform-Specific Configurations

### A. Azure AKS Istio Add-on Reference

```yaml
# AKS Istio Add-on specific configurations

# Namespace: aks-istio-system (control plane)
# Namespace: aks-istio-ingress (gateways)

# Gateway selector for AKS add-on
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: main-gateway
  namespace: aks-istio-ingress
spec:
  selector:
    istio: aks-istio-ingressgateway-external  # AKS add-on selector
  servers:
    - port:
        number: 443
        name: https
        protocol: HTTPS
      tls:
        mode: SIMPLE
        credentialName: gateway-tls  # Secret in aks-istio-ingress namespace
      hosts:
        - "*.myapp.com"
---
# TLS secret must be in aks-istio-ingress namespace
apiVersion: v1
kind: Secret
metadata:
  name: gateway-tls
  namespace: aks-istio-ingress  # AKS add-on gateway namespace
type: kubernetes.io/tls
data:
  tls.crt: <base64>
  tls.key: <base64>
---
# VirtualService referencing AKS gateway
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: my-service
  namespace: production
spec:
  hosts:
    - "api.myapp.com"
  gateways:
    - aks-istio-ingress/main-gateway  # Reference AKS gateway namespace
  http:
    - route:
        - destination:
            host: my-service
            port:
              number: 8080
```

```bash
# AKS Istio add-on management commands

# Upgrade Istio revision (managed by Azure)
az aks mesh get-upgrades \
  --resource-group myAKSResourceGroup \
  --name myAKSCluster

az aks mesh upgrade start \
  --resource-group myAKSResourceGroup \
  --name myAKSCluster \
  --revision asm-1-20

# Enable/disable namespaces for sidecar injection
kubectl label namespace production istio.io/rev=asm-1-20

# Check Istio add-on status
kubectl get pods -n aks-istio-system
kubectl get pods -n aks-istio-ingress
```

### B. AWS EKS Specific Configuration

```yaml
# EKS with AWS Load Balancer Controller

# Service annotations for NLB
apiVersion: v1
kind: Service
metadata:
  name: istio-ingressgateway
  namespace: istio-system
  annotations:
    # Use AWS Load Balancer Controller
    service.beta.kubernetes.io/aws-load-balancer-type: "external"
    service.beta.kubernetes.io/aws-load-balancer-nlb-target-type: "ip"
    service.beta.kubernetes.io/aws-load-balancer-scheme: "internet-facing"

    # SSL/TLS termination at NLB (optional)
    service.beta.kubernetes.io/aws-load-balancer-ssl-cert: "arn:aws:acm:region:account:certificate/xxx"
    service.beta.kubernetes.io/aws-load-balancer-ssl-ports: "443"
    service.beta.kubernetes.io/aws-load-balancer-ssl-negotiation-policy: "ELBSecurityPolicy-TLS13-1-2-2021-06"

    # Access logs to S3
    service.beta.kubernetes.io/aws-load-balancer-access-log-enabled: "true"
    service.beta.kubernetes.io/aws-load-balancer-access-log-s3-bucket-name: "my-nlb-logs"
    service.beta.kubernetes.io/aws-load-balancer-access-log-s3-bucket-prefix: "istio-ingress"

    # Health check
    service.beta.kubernetes.io/aws-load-balancer-healthcheck-protocol: "HTTP"
    service.beta.kubernetes.io/aws-load-balancer-healthcheck-port: "15021"
    service.beta.kubernetes.io/aws-load-balancer-healthcheck-path: "/healthz/ready"

    # Cross-zone load balancing
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"

    # Target group attributes
    service.beta.kubernetes.io/aws-load-balancer-target-group-attributes: "deregistration_delay.timeout_seconds=30"
spec:
  type: LoadBalancer
  externalTrafficPolicy: Local
---
# IAM policy for AWS Load Balancer Controller (IRSA)
# Ensure the controller has permissions to manage NLBs and Target Groups
```

```bash
# EKS Istio installation with eksctl

# 1. Install AWS Load Balancer Controller first
eksctl create iamserviceaccount \
  --cluster=my-cluster \
  --namespace=kube-system \
  --name=aws-load-balancer-controller \
  --attach-policy-arn=arn:aws:iam::aws:policy/AWSLoadBalancerControllerIAMPolicy \
  --approve

helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=my-cluster \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller

# 2. Install Istio
istioctl install -f istio-operator.yaml -y

# 3. Verify NLB creation
kubectl get svc istio-ingressgateway -n istio-system
aws elbv2 describe-load-balancers --names "k8s-istiosys-istioing-xxx"
```

---

## 10. Troubleshooting

### A. Common Issues and Solutions

```bash
# ✅ CORRECT - Diagnostic commands

# 1. Analyze configuration
istioctl analyze --all-namespaces

# 2. Check proxy status
istioctl proxy-status

# 3. Describe pod Istio config
istioctl x describe pod <pod-name> -n <namespace>

# 4. Check proxy config
istioctl proxy-config all <pod-name> -n <namespace>
istioctl proxy-config cluster <pod-name> -n <namespace>
istioctl proxy-config listener <pod-name> -n <namespace>
istioctl proxy-config route <pod-name> -n <namespace>
istioctl proxy-config endpoint <pod-name> -n <namespace>

# 5. Debug mTLS
istioctl x authz check <pod-name> -n <namespace>

# 6. View Envoy logs
kubectl logs <pod-name> -c istio-proxy -n <namespace>

# 7. Enable debug logging
istioctl proxy-config log <pod-name> --level debug

# 8. Check certificate info
istioctl proxy-config secret <pod-name> -n <namespace>
```

### B. Common Problems

```yaml
# Problem: 503 Service Unavailable
# Cause: Usually mTLS mismatch or missing DestinationRule

# Solution: Check PeerAuthentication and DestinationRule
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: fix-mtls
  namespace: production
spec:
  host: problematic-service
  trafficPolicy:
    tls:
      mode: ISTIO_MUTUAL  # Must match PeerAuthentication
---
# Problem: Connection refused to external service
# Cause: REGISTRY_ONLY mode blocking external traffic

# Solution: Add ServiceEntry
apiVersion: networking.istio.io/v1beta1
kind: ServiceEntry
metadata:
  name: allow-external
  namespace: production
spec:
  hosts:
    - "api.external.com"
  ports:
    - number: 443
      name: https
      protocol: HTTPS
  location: MESH_EXTERNAL
  resolution: DNS
---
# Problem: JWT validation failing
# Cause: Missing or incorrect RequestAuthentication

# Debug: Check JWT with
# istioctl x authz check <pod> -n <namespace>
```

---

## 11. Verification Checklist (MANDATORY)

### A. Pre-Deployment Validation

```bash
# ✅ CORRECT - Validation commands

# 1. Analyze all Istio configuration
istioctl analyze --all-namespaces
# Must return: No validation issues found

# 2. Validate YAML syntax
kubectl apply --dry-run=server -f istio-configs/

# 3. Check Gateway TLS secrets exist
kubectl get secret gateway-tls -n istio-system

# 4. Verify mTLS status
istioctl x authz check <test-pod> -n production

# 5. Test connectivity
kubectl exec -it <client-pod> -c istio-proxy -- \
  curl -v http://target-service:8080/health
```

### B. Architecture Verification Checklist

```
ISTIO VERIFICATION CHECKLIST:

□ Installation
  □ Istio control plane healthy (istiod running)
  □ Ingress gateway deployed and accessible
  □ Sidecar injection enabled for application namespaces
  □ istioctl analyze returns no issues

□ Static IP (if required)
  □ Static IP allocated (Azure/AWS/on-prem)
  □ Gateway service using static IP
  □ DNS records point to static IP
  □ Firewall rules configured

□ Security
  □ mTLS STRICT enabled (PeerAuthentication)
  □ Default deny AuthorizationPolicy in place
  □ Explicit allow policies for each service
  □ JWT authentication configured (if required)
  □ Egress controlled via ServiceEntry

□ Traffic Management
  □ Gateway configured with TLS
  □ VirtualService routing defined
  □ DestinationRule with traffic policies
  □ Timeouts and retries configured
  □ Circuit breaker (outlier detection) enabled

□ Observability
  □ Tracing configured (Jaeger/Zipkin)
  □ Metrics exported (Prometheus)
  □ Access logging enabled
  □ Kiali deployed for visualization

□ Platform-Specific
  □ AKS: Istio add-on enabled and configured
  □ AKS: Gateway in aks-istio-ingress namespace
  □ EKS: AWS Load Balancer Controller installed
  □ EKS: NLB annotations configured
  □ Generic: MetalLB or equivalent configured
```

---

## 12. Summary

### Core Principles

1. **mTLS everywhere**: STRICT mode in production, no exceptions
2. **Deny by default**: Empty AuthorizationPolicy, explicit allows
3. **Static IPs**: Predictable addresses for DNS and firewall rules
4. **Platform-native**: Use managed Istio (AKS add-on) where available
5. **Observability**: Tracing, metrics, and logging from day one

### Platform Quick Reference

| Platform | Istio Install | Gateway Namespace | Static IP Method |
|----------|---------------|-------------------|------------------|
| Generic K8s | istioctl/Helm | istio-system | loadBalancerIP + MetalLB |
| Azure AKS (add-on) | az aks mesh enable | aks-istio-ingress | Azure annotation |
| Azure AKS (manual) | istioctl/Helm | istio-system | Azure annotation |
| AWS EKS | istioctl/Helm | istio-system | NLB + Elastic IP |

### Key Configurations

| Component | Purpose | Resource |
|-----------|---------|----------|
| Gateway | Ingress traffic | Gateway |
| Routing | Traffic control | VirtualService |
| Traffic Policy | Load balancing, circuit breaker | DestinationRule |
| mTLS | Encryption | PeerAuthentication |
| Authorization | Access control | AuthorizationPolicy |
| JWT Auth | Token validation | RequestAuthentication |
| External Access | Egress control | ServiceEntry |

### Remember

> "mTLS STRICT is not optional. Every service-to-service call must be encrypted and authenticated."

> "Deny by default. If there's no AuthorizationPolicy allowing traffic, it should be blocked."

> "Static IPs enable reliable DNS, firewall rules, and external integrations. Configure them from day one."

> "Use the managed Istio add-on on AKS. It's supported, automatically upgraded, and reduces operational burden."


---

## 13. Deployment Checklist

### Mesh Installation
- [ ] Istio version is current and supported (N-1 policy)
- [ ] Control plane high availability configured (2+ istiod replicas)
- [ ] Istio CNI plugin installed (eliminates init container privileges)
- [ ] Resource limits set on istiod and sidecar proxies

### Security
- [ ] PeerAuthentication set to STRICT mTLS cluster-wide
- [ ] Default-deny AuthorizationPolicy applied to all namespaces
- [ ] Explicit allow policies for each service-to-service path
- [ ] RequestAuthentication configured for external JWT validation
- [ ] Egress traffic controlled via ServiceEntry (no open egress)

### Traffic Management
- [ ] Gateway configured with TLS certificates (auto-renewed)
- [ ] Static IP assigned to ingress gateway
- [ ] VirtualService routing rules defined for all services
- [ ] DestinationRule circuit breakers configured
- [ ] Retry policies with exponential backoff set
- [ ] Timeout values defined for all routes

### Observability
- [ ] Distributed tracing enabled (Jaeger/Zipkin backend)
- [ ] Prometheus scraping Istio metrics
- [ ] Grafana dashboards for mesh traffic, latency, and errors
- [ ] Access logging enabled on ingress gateway
- [ ] Kiali installed for service mesh visualization

### Platform-Specific
- [ ] AKS: Managed Istio add-on enabled (or manual install documented)
- [ ] EKS: NLB with Elastic IP configured for ingress
- [ ] On-prem: MetalLB or equivalent load balancer configured

---

## 14. Why This Configuration Works

1. **STRICT mTLS Everywhere**: Encrypting and authenticating all service-to-service traffic eliminates eavesdropping and spoofing without application code changes.

2. **Deny-by-Default Authorization**: Starting with an empty AuthorizationPolicy that denies all traffic forces explicit allow rules, ensuring no unintended service access.

3. **Static IP for Ingress**: Assigning a fixed IP to the ingress gateway enables stable DNS records, predictable firewall rules, and reliable external integrations.

4. **Circuit Breakers with Outlier Detection**: Automatically ejecting unhealthy endpoints prevents cascading failures from propagating through the service mesh.

5. **Sidecar Proxy Pattern**: Handling security, observability, and traffic management in the Envoy sidecar keeps application code free of cross-cutting infrastructure concerns.

6. **Managed Istio on AKS**: Using the platform-managed add-on reduces operational burden with automatic upgrades, integrated monitoring, and Microsoft support.

7. **Egress Control via ServiceEntry**: Restricting outbound traffic to declared external services prevents data exfiltration and catches unintended external dependencies.

8. **Retry Policies with Budget**: Configuring retries with `perTryTimeout` and backoff improves reliability while retry budgets prevent retry storms from amplifying failures.

9. **Canary Deployments via Traffic Splitting**: VirtualService weighted routing enables gradual rollouts with instant rollback by adjusting traffic percentages.

10. **Observability without Code Changes**: Automatic metrics, traces, and access logs from Envoy proxies provide full visibility into mesh traffic with zero application instrumentation.

**End of Istio Service Mesh Guidelines**
