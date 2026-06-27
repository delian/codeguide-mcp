# Istio Service Mesh Guidelines
Mandatory standards for secure, observable, production-ready Istio service meshes: zero-trust mTLS, deny-by-default authorization, declarative traffic management, and mesh-level resilience. Istio 1.24+, istioctl, ambient mode, Kubernetes Gateway API.

---
name: istio
title: Istio Service Mesh Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: infra
tools: [istio@1.24, istioctl, ambient-mode, gateway-api, helm@3]
requires:
  - secure-coding
  - observability
recommends:
  - kubernetes
  - microservices
provides:
  - service-mesh
  - istio-traffic-management
  - mtls
  - istio-authz
  - ambient-mode
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Istio — the mesh binding of security, observability, and resilience. The platform (Kubernetes) and the architecture (microservices) it serves are owned elsewhere.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Istio configuration. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — zero-trust, secrets, supply chain, TLS policy. *(Istio binding: mTLS is the data-plane enforcement of zero-trust; STRICT `PeerAuthentication` + deny-by-default `AuthorizationPolicy`.)*
> - [`observability.md`](guides://observability.md) — metrics, tracing, logging strategy. *(Istio binding: the Telemetry API emits RED metrics, spans, and access logs from Envoy with zero app code.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`kubernetes.md`](guides://kubernetes.md) — the platform Istio runs on (namespaces, Services, LoadBalancer, NetworkPolicy, RBAC). This guide does **not** restate Kubernetes mechanics.
> - [`microservices.md`](guides://microservices.md) — the architecture Istio serves. Resilience *patterns* (circuit breaker, retry budget, bulkhead, timeout strategy) are owned there; Istio only **enforces** them at the mesh layer.

> 📎 **SEE ALSO:** [`oauth.md`](guides://oauth.md) *(JWT/OIDC token policy behind `RequestAuthentication`)* · [`tdd.md`](guides://tdd.md) *(test-first config: write the connectivity/authz assertion before the resource)* · [`ci-cd.md`](guides://ci-cd.md) *(GitOps delivery of mesh resources)* · [`grpc.md`](guides://grpc.md) · [`rest.md`](guides://rest.md)

---

## 1. Core Philosophies

Istio-specific principles only. Zero-trust, observability strategy, and resilience patterns come from §0 — do **not** restate them here.

- **Zero-trust data plane**: every hop is encrypted and authenticated by the proxy, not the app. STRICT mTLS and deny-by-default are the defaults, not opt-ins.
- **Declarative, not imperative**: traffic, security, and telemetry are CRDs in git. The desired state is reconciled; never `kubectl edit` live mesh config.
- **Right proxy for the job**: sidecar for full per-pod L7 control; **ambient** (ztunnel L4 + waypoint L7) when you want mesh security/telemetry without a sidecar per pod and lower overhead.
- **Mesh owns the network, app owns the logic**: retries, timeouts, circuit breaking, routing, and identity live in the mesh; business logic stays free of cross-cutting infrastructure.
- **Verified before applied**: every change passes `istioctl analyze` and a connectivity/authz assertion before it reaches a cluster.

**Verified Config**: Agent-generated Istio resources MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `ISTIO-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| ISTIO-TST-01 | Config MUST be validated before apply; new behavior gets a connectivity/authz assertion first (see `tdd.md`) | `istioctl analyze --all-namespaces` | "No validation issues" |
| ISTIO-MTLS-01 | mTLS MUST be STRICT mesh-wide; PERMISSIVE only during a tracked migration (see `secure-coding.md`) | `kubectl get peerauthentication -A` / `istioctl proxy-config secret` | mode STRICT |
| ISTIO-AUTHZ-01 | Every workload namespace MUST have a default-deny `AuthorizationPolicy` (see `secure-coding.md`) | `kubectl get authorizationpolicy -A` | deny-all present |
| ISTIO-AUTHZ-02 | Access MUST be least-privilege: explicit ALLOW by SPIFFE principal/path, no blanket `*` source | review / `istioctl analyze` | no over-broad ALLOW |
| ISTIO-AUTHN-01 | External-facing services MUST validate JWTs via `RequestAuthentication` (token policy: `oauth.md`) | `kubectl get requestauthentication -A` | present on edge |
| ISTIO-EGRESS-01 | Outbound MUST be `REGISTRY_ONLY`; external hosts declared via `ServiceEntry` (see `secure-coding.md`) | check `meshConfig.outboundTrafficPolicy` | REGISTRY_ONLY |
| ISTIO-GW-01 | Ingress Gateways MUST terminate TLS ≥1.2; port 80 redirects to 443 | inspect `Gateway`/TLS secret | no plaintext serving |
| ISTIO-TM-01 | Every production route MUST set an explicit `timeout` and `retries` (pattern: `microservices.md`) | review VirtualService/HTTPRoute | timeout + retries set |
| ISTIO-TM-02 | Production `DestinationRule`s MUST enable `outlierDetection` (circuit breaking — `microservices.md`) | `kubectl get destinationrule -o yaml` | outlierDetection set |
| ISTIO-OBS-01 | Mesh MUST emit metrics, traces, and access logs via the Telemetry API (see `observability.md`) | `kubectl get telemetry -A` | configured |
| ISTIO-STRUCT-01 | All mesh resources MUST be version-controlled and applied via GitOps (see `ci-cd.md`) | repo review | no out-of-band edits |
| ISTIO-VER-01 | Control plane MUST run a supported release (N-1 of latest minor); upgrades canary-tested | `istioctl version` | supported revision |

> **Forbidden**: shipping mesh config that fails `istioctl analyze`; PERMISSIVE/DISABLE mTLS in production without a migration ticket; a namespace with no default-deny policy; an open egress (`ALLOW_ANY`); a production route with no timeout; deprecated `v1alpha3`/`v1beta1` API versions when `v1` exists.

---

## 3. Verification Protocol

Run, in order, before presenting config. Fix → re-run until every gate is green.

```bash
istioctl analyze --all-namespaces          # ISTIO-TST-01: static validation
kubectl apply --dry-run=server -f mesh/     # schema + admission validation
istioctl proxy-status                       # every proxy SYNCED to istiod
istioctl proxy-config secret <pod> -n <ns>  # ISTIO-MTLS-01: workload cert present
istioctl proxy-config route <pod> -n <ns>   # routes resolve as intended
# Behavioral assertion (write FIRST per tdd.md): expected vs actual HTTP code
kubectl exec deploy/<client> -n <ns> -- curl -s -o /dev/null -w '%{http_code}' http://<svc>:<port>/health
```

The *why* behind each gate (zero-trust, RED metrics, retry budgets) lives in its §0 owner; do not re-derive it here.

---

## 4. Mesh Architecture: Sidecar vs Ambient

Istio offers two data planes. Choose per workload; they interoperate in one mesh.

| | **Sidecar** (per-pod Envoy) | **Ambient** (GA in 1.24) |
|---|---|---|
| L4 (mTLS, TCP authz, telemetry) | Envoy sidecar | **ztunnel** — one per node, shared |
| L7 (HTTP routing, L7 authz, retries) | Envoy sidecar | **waypoint** proxy — opt-in per namespace/service |
| Injection | `istio-injection=enabled` / pod restart | `istio.io/dataplane-mode=ambient` label — **no restart** |
| Cost | ~1 proxy per pod | shared node proxy; L7 only where needed |
| Use when | fine-grained per-pod L7, EnvoyFilter, legacy parity | broad zero-trust + telemetry at low overhead |

```bash
# Sidecar: opt a namespace in (pods must be (re)created to get the sidecar)
kubectl label namespace prod istio-injection=enabled

# Ambient: enroll a namespace into the shared L4 mesh — instant, no pod restart
kubectl label namespace prod istio.io/dataplane-mode=ambient

# Add L7 (routing, HTTP authz, retries) for a namespace by deploying a waypoint
istioctl waypoint apply -n prod --enroll-namespace
```

Identity is **SPIFFE** in both: `spiffe://<trust-domain>/ns/<namespace>/sa/<serviceaccount>`. Policies (`PeerAuthentication`, `AuthorizationPolicy`) and the Telemetry API apply uniformly; ambient enforces L7 rules at the waypoint, so an `AuthorizationPolicy` with HTTP `paths`/`methods` requires a waypoint in ambient.

---

## 5. Traffic Management

Istio's core owned surface. Prefer the **Kubernetes Gateway API** (GA in Istio) for ingress and routing; use Istio's `VirtualService`/`DestinationRule` for advanced policy (subsets, outlier detection, mTLS origination) not yet covered by Gateway API.

### A. Ingress — Kubernetes Gateway API (preferred)
```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: main-gateway
  namespace: istio-ingress
spec:
  gatewayClassName: istio                 # Istio implements the Gateway API
  listeners:
    - name: https
      port: 443
      protocol: HTTPS
      hostname: "*.myapp.com"
      tls:
        mode: Terminate
        certificateRefs:
          - name: gateway-tls             # Kubernetes TLS Secret (cert-manager)
      allowedRoutes:
        namespaces: { from: All }
---
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: order-service
  namespace: production
spec:
  parentRefs: [{ name: main-gateway, namespace: istio-ingress }]
  hostnames: ["api.myapp.com"]
  rules:
    - matches: [{ path: { type: PathPrefix, value: /api/orders } }]
      timeouts: { request: 30s }          # ISTIO-TM-01
      backendRefs:
        - { name: order-service, port: 8080, weight: 90 }   # canary split
        - { name: order-service-canary, port: 8080, weight: 10 }
```

### B. Istio Gateway + VirtualService (advanced policy)
```yaml
apiVersion: networking.istio.io/v1                  # v1 GA — not v1beta1/v1alpha3
kind: Gateway
metadata: { name: main-gateway, namespace: istio-ingress }
spec:
  selector: { istio: ingressgateway }
  servers:
    - port: { number: 443, name: https, protocol: HTTPS }
      tls:
        mode: SIMPLE
        credentialName: gateway-tls
        minProtocolVersion: TLSV1_2                  # ISTIO-GW-01
      hosts: ["api.myapp.com"]
    - port: { number: 80, name: http, protocol: HTTP }
      hosts: ["api.myapp.com"]
      tls: { httpsRedirect: true }                   # 80 → 443
---
apiVersion: networking.istio.io/v1
kind: VirtualService
metadata: { name: order-service, namespace: production }
spec:
  hosts: ["api.myapp.com"]
  gateways: ["istio-ingress/main-gateway", "mesh"]   # ingress + internal
  http:
    - match: [{ headers: { x-canary: { exact: "true" } } }]  # specific match FIRST
      route: [{ destination: { host: order-service, subset: canary } }]
      timeout: 30s
      retries: { attempts: 3, perTryTimeout: 10s, retryOn: "5xx,reset,connect-failure" }
    - route:                                          # weighted rollout
        - { destination: { host: order-service, subset: stable }, weight: 95 }
        - { destination: { host: order-service, subset: canary }, weight: 5 }
      timeout: 30s
      retries: { attempts: 3, perTryTimeout: 10s, retryOn: "5xx,reset,connect-failure" }
```
> **Footgun:** Istio evaluates `http` rules top-down — a default catch-all route placed before a specific `match` makes the match unreachable. Order specific matches first.

### C. DestinationRule — subsets, load balancing, circuit breaking
```yaml
apiVersion: networking.istio.io/v1
kind: DestinationRule
metadata: { name: order-service, namespace: production }
spec:
  host: order-service
  trafficPolicy:
    tls: { mode: ISTIO_MUTUAL }                       # mesh mTLS (ISTIO-MTLS-01)
    loadBalancer: { simple: LEAST_REQUEST }
    connectionPool:
      tcp: { maxConnections: 100, connectTimeout: 5s }
      http: { http2MaxRequests: 1000, http1MaxPendingRequests: 100, maxRequestsPerConnection: 100 }
    outlierDetection:                                 # circuit breaking (ISTIO-TM-02)
      consecutive5xxErrors: 5
      interval: 10s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
  subsets:
    - { name: stable, labels: { version: stable } }
    - { name: canary, labels: { version: canary } }
```
Canary/blue-green is just weight shifting on the `VirtualService`/`HTTPRoute` over these subsets; roll back by reverting weights. Pair with a progressive-delivery controller (e.g. Argo Rollouts/Flagger) for automated, metric-gated promotion.

### D. Egress control
`meshConfig.outboundTrafficPolicy.mode: REGISTRY_ONLY` (ISTIO-EGRESS-01) blocks undeclared egress. Declare each external dependency:
```yaml
apiVersion: networking.istio.io/v1
kind: ServiceEntry
metadata: { name: external-api, namespace: production }
spec:
  hosts: ["api.partner.com"]
  ports: [{ number: 443, name: https, protocol: TLS }]
  location: MESH_EXTERNAL
  resolution: DNS
```
Route high-sensitivity egress through an egress Gateway to apply policy/logging at a chokepoint.

---

## 6. Security: mTLS & Authorization

Istio is the data-plane enforcement of the zero-trust model owned by [`secure-coding.md`](guides://secure-coding.md). It encrypts and authenticates every hop using auto-rotated SPIFFE workload certs issued by istiod — no app code, no secret handling.

### A. STRICT mTLS (ISTIO-MTLS-01)
```yaml
apiVersion: security.istio.io/v1                      # v1 GA
kind: PeerAuthentication
metadata: { name: default, namespace: istio-system }  # mesh-wide when in the root ns
spec:
  mtls: { mode: STRICT }
---
# Port-level exception for a legacy port during migration (still STRICT elsewhere)
apiVersion: security.istio.io/v1
kind: PeerAuthentication
metadata: { name: legacy, namespace: production }
spec:
  selector: { matchLabels: { app: legacy-service } }
  mtls: { mode: PERMISSIVE }
  portLevelMtls: { 8080: { mode: STRICT } }
```
PERMISSIVE accepts both plaintext and mTLS — use it **only** while migrating, then flip to STRICT and delete the exception.

### B. Deny-by-default, then explicit allow (ISTIO-AUTHZ-01/02)
```yaml
apiVersion: security.istio.io/v1
kind: AuthorizationPolicy
metadata: { name: deny-all, namespace: production }
spec: {}                                              # empty spec = deny all
---
apiVersion: security.istio.io/v1
kind: AuthorizationPolicy
metadata: { name: order-service, namespace: production }
spec:
  selector: { matchLabels: { app: order-service } }
  action: ALLOW
  rules:
    - from: [{ source: { principals: ["cluster.local/ns/production/sa/payment-service"] } }]
      to:   [{ operation: { methods: ["GET","POST"], paths: ["/api/orders/*"] } }]
    - to:   [{ operation: { paths: ["/healthz/*","/metrics"] } }]   # probes/scrape
```
Allow by **identity (SPIFFE principal)**, not IP. A separate `action: DENY` policy always wins over ALLOW — use it to hard-block `/admin/*` from anything but the admin SA. In ambient mode, L7 conditions (`paths`/`methods`) require a waypoint; L4 conditions (principals/ports) are enforced by ztunnel.

### C. JWT validation at the edge (ISTIO-AUTHN-01)
Token issuance/rotation/scopes policy is owned by [`oauth.md`](guides://oauth.md). Istio binding — validate and pin issuers:
```yaml
apiVersion: security.istio.io/v1
kind: RequestAuthentication
metadata: { name: jwt, namespace: production }
spec:
  selector: { matchLabels: { app: order-service } }
  jwtRules:
    - issuer: "https://auth.myapp.com/"
      jwksUri: "https://auth.myapp.com/.well-known/jwks.json"
      audiences: ["https://api.myapp.com"]
      outputClaimToHeaders: [{ header: x-user-id, claim: sub }]
---
# RequestAuthentication only validates a PRESENT token; require one with authz:
apiVersion: security.istio.io/v1
kind: AuthorizationPolicy
metadata: { name: require-jwt, namespace: production }
spec:
  selector: { matchLabels: { app: order-service } }
  action: ALLOW
  rules:
    - to: [{ operation: { paths: ["/healthz/*","/public/*"] } }]    # open
    - from: [{ source: { requestPrincipals: ["*"] } }]              # any valid JWT
      to:   [{ operation: { paths: ["/api/*"] } }]
```
> **Footgun:** `RequestAuthentication` does **not** reject missing tokens — without a companion `AuthorizationPolicy` requiring `requestPrincipals`, unauthenticated requests pass.

### D. Control-plane & sidecar hardening
Run Istio CNI to drop the privileged `istio-init` container; set CPU/memory requests+limits on istiod and proxies; restrict istiod with a Kubernetes `NetworkPolicy` (owned by [`kubernetes.md`](guides://kubernetes.md)). Keep `proxy.privileged: false` and `enableCoreDump: false`.

---

## 7. Resilience (mesh binding)

The *patterns* — timeout budgets, retry budgets, circuit breaker, bulkhead, outlier ejection — are owned by [`microservices.md`](guides://microservices.md). Istio is where you **enforce** them without touching application code:

| Pattern (owner: `microservices.md`) | Istio enforcement |
|---|---|
| Timeout | `http.timeout` on VirtualService / `timeouts.request` on HTTPRoute (ISTIO-TM-01) |
| Retry + budget | `retries.{attempts,perTryTimeout,retryOn}`; cap with `retryRemoteLocalities` and per-try timeouts to avoid storms |
| Circuit breaker | `DestinationRule.trafficPolicy.outlierDetection` (ISTIO-TM-02) |
| Bulkhead | `connectionPool` TCP/HTTP limits isolate a slow dependency |
| Fault injection (test) | `http.fault.{delay,abort}` — staging only, never production |

Set retries to **idempotent conditions** (`5xx,reset,connect-failure`); never blindly retry non-idempotent `POST`. Keep `perTryTimeout × attempts ≤` the route `timeout`, or the outer timeout cancels retries mid-flight.

---

## 8. Observability (mesh binding)

Strategy (RED/USE metrics, trace sampling, log levels, dashboards/alerts) is owned by [`observability.md`](guides://observability.md). Istio's Telemetry API emits all three signals from Envoy with zero application instrumentation (ISTIO-OBS-01):

```yaml
apiVersion: telemetry.istio.io/v1                     # v1 GA
kind: Telemetry
metadata: { name: mesh-default, namespace: istio-system }
spec:
  metrics:
    - providers: [{ name: prometheus }]
  tracing:
    - providers: [{ name: otel }]                     # OpenTelemetry / Tempo / Jaeger
      randomSamplingPercentage: 10.0                  # production sample rate
  accessLogging:
    - providers: [{ name: otel }]
      filter: { expression: "response.code >= 400 || connection.mtls == false" }
```
Mesh telemetry gives request rate/error/duration and mTLS coverage per workload edge; the app still propagates trace context (B3/W3C `traceparent`) for end-to-end spans. Use **Kiali** for the topology/mTLS view; ship metrics to Prometheus and traces to an OTLP backend per `observability.md`.

---

## 9. Installation & Platform Notes

- **Install with Helm** (recommended) or `istioctl install`. The in-cluster IstioOperator controller is **removed** in current Istio; the `IstioOperator` CR remains only as an `istioctl install -f` config format. Pick the `ambient` profile for ambient mesh, `default` for sidecar.
  ```bash
  istioctl install --set profile=ambient -y      # or: helm install istiod istio/istiod ...
  istioctl verify-install && istioctl analyze --all-namespaces
  ```
- **HA control plane**: 2+ istiod replicas with a `PodDisruptionBudget` and anti-affinity (mechanics owned by [`kubernetes.md`](guides://kubernetes.md)).
- **Static ingress IP / LoadBalancer**: the Gateway is a normal Kubernetes `Service type=LoadBalancer`; static-IP, NLB, and annotation mechanics (MetalLB, AKS `azure-load-balancer-*`, AWS `aws-load-balancer-*` / EIP) belong to the cloud/k8s layer — see [`kubernetes.md`](guides://kubernetes.md). Istio only selects the Gateway workload.
- **Managed offerings** (e.g. AKS Istio add-on, GKE managed Istio/ASM) move upgrades/patching to the provider but constrain custom `meshConfig`; gateways live in the provider namespace (e.g. `aks-istio-ingress`) and TLS secrets must co-locate there.
- **Upgrades**: use revision-based canary upgrades (`istio.io/rev` labels), shift a test namespace first, verify, then roll the fleet (ISTIO-VER-01).

---

## 10. Troubleshooting

```bash
istioctl analyze --all-namespaces            # config errors before they bite
istioctl proxy-status                         # SYNCED? STALE = istiod push problem
istioctl x describe pod <pod> -n <ns>         # what policies/routes apply to a pod
istioctl proxy-config {cluster|listener|route|endpoint|secret} <pod> -n <ns>
kubectl logs <pod> -c istio-proxy -n <ns>     # sidecar; ambient: logs in ztunnel/waypoint
istioctl proxy-config log <pod> --level debug # temporary verbose Envoy logging
```

| Symptom | Usual cause | Fix |
|---|---|---|
| `503 UC`/`upstream reset` | mTLS mismatch (STRICT server, plaintext/no `DestinationRule` client) | add `DestinationRule` `tls: ISTIO_MUTUAL` for the FQDN, or align `PeerAuthentication` |
| `RBAC: access denied` (403) | deny-by-default with no matching ALLOW | add `AuthorizationPolicy` rule for the caller's SPIFFE principal |
| Connect refused to external host | `REGISTRY_ONLY` blocks it | declare a `ServiceEntry` |
| Canary header ignored | catch-all route precedes the `match` | reorder: specific matches first |
| 401 with valid token | `RequestAuthentication` present but no authz requiring `requestPrincipals` | add the companion ALLOW rule |

---

## 11. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] ISTIO-TST-01 — `istioctl analyze` clean; connectivity/authz assertion written first and passing
- [ ] ISTIO-MTLS-01 — STRICT `PeerAuthentication` mesh-wide; no stray PERMISSIVE/DISABLE
- [ ] ISTIO-AUTHZ-01 — default-deny `AuthorizationPolicy` in every workload namespace
- [ ] ISTIO-AUTHZ-02 — explicit, least-privilege ALLOW rules by SPIFFE principal/path
- [ ] ISTIO-AUTHN-01 — `RequestAuthentication` + require-JWT authz on edge services
- [ ] ISTIO-EGRESS-01 — `REGISTRY_ONLY`; all external hosts declared via `ServiceEntry`
- [ ] ISTIO-GW-01 — Gateways terminate TLS ≥1.2; port 80 redirects to 443
- [ ] ISTIO-TM-01 — every production route has a timeout and retries
- [ ] ISTIO-TM-02 — production `DestinationRule`s enable `outlierDetection`
- [ ] ISTIO-OBS-01 — Telemetry API emits metrics, traces, access logs (see `observability.md`)
- [ ] ISTIO-STRUCT-01 — all mesh resources in git, applied via GitOps
- [ ] ISTIO-VER-01 — control plane on a supported revision; upgrade canary-tested
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Istio Service Mesh Guidelines**
