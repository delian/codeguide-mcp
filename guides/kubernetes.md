# Kubernetes Deployment Guidelines
Mandatory standards for secure, resilient, production-ready Kubernetes workloads: least-privilege pods, right-sized resources, health probes, autoscaling, default-deny networking, GitOps. Kubernetes 1.31+, kubectl, Helm 3, Kustomize, Gateway API.

---
name: kubernetes
title: Kubernetes Deployment Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: infra
tools: [kubernetes@1.31, kubectl, helm@3, kustomize, gateway-api, kubeconform, conftest, trivy]
requires:
  - secure-coding
  - observability
recommends:
  - dockerfile
  - istio
  - terraform
  - ci-cd
  - env-config
  - microservices
provides:
  - k8s-workloads
  - k8s-networking
  - k8s-security
  - k8s-autoscaling
  - helm-kustomize
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Kubernetes.

---

## 0. Prerequisites & References

Fetch and apply these **before** authoring Kubernetes manifests. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, CVE policy, secrets, least privilege. *(K8s binding: Pod Security Standards, RBAC, NetworkPolicies, `trivy k8s`, image scanning, admission policy.)*
> - [`observability.md`](guides://observability.md) — metrics, tracing, SLOs, health semantics. *(K8s binding: Prometheus `ServiceMonitor`, the three-probe pattern, stdout JSON logs.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`dockerfile.md`](guides://dockerfile.md) — the container image that runs in the pod *(binding: non-root, pinned digest, minimal base)*
> - [`istio.md`](guides://istio.md) — service mesh: mTLS, traffic routing, mesh authz *(binding: sidecar injection, `VirtualService`/`DestinationRule`)*
> - [`env-config.md`](guides://env-config.md) — config layering & secrets policy *(binding: ConfigMap/Secret, External Secrets)*
> - [`ci-cd.md`](guides://ci-cd.md) — GitOps pipeline & promotion *(binding: ArgoCD/Flux sync)*
> - [`microservices.md`](guides://microservices.md) — Kubernetes is the deployment target *(binding: one Deployment + Service per service, async via a broker)*
> - [`terraform.md`](guides://terraform.md) — cluster, node pools, and IAM provisioning live in IaC, not `kubectl`

> 📎 **SEE ALSO:** [`kafka.md`](guides://kafka.md) · [`grpc.md`](guides://grpc.md) · [`rest.md`](guides://rest.md) · [`tdd.md`](guides://tdd.md) *(manifest tests: kubeconform/conftest/helm-unittest)*

---

## 1. Core Philosophies: KUBERNETES

Kubernetes-specific principles only. Security policy, observability, and the image build come from §0.

- **K**ept declarative: every object lives in Git; the cluster is reconciled to it, never mutated by hand (see `ci-cd.md`).
- **U**niform least privilege: restricted Pod Security Standard, dedicated ServiceAccount, default-deny network (policy: `secure-coding.md`).
- **B**ounded resources: every container has CPU/memory requests + memory limits; namespaces have `ResourceQuota`/`LimitRange`.
- **E**phemeral pods: stateless containers, read-only rootfs, external state in PVCs or managed services; any pod may die anytime.
- **R**esilient by config: 3+ replicas, anti-affinity across nodes/zones, PDB, three-probe health, graceful termination.
- **N**amespace isolation: one bounded context per namespace with scoped RBAC and NetworkPolicies.
- **E**xternalized config: ConfigMaps/Secrets, environment overlays — never baked into the image (policy: `env-config.md`).
- **S**calable: HPA on real signals, cluster autoscaler bounded, right-sized pods over oversized ones.

**Verified Manifests**: agent-generated manifests MUST pass every gate in §2 (schema validation, policy, security scan, server dry-run) before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `K8S-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| K8S-STRUCT-01 | Manifests MUST validate against the cluster API schema | `kubeconform -strict` + `kubectl apply --dry-run=server` | exit 0 |
| K8S-IMG-01 | Images MUST pin an immutable tag or digest, never `:latest` (see `dockerfile.md`) | `grep -rE 'image:.*(:latest|@sha256)?$' manifests/` / conftest | no `:latest` |
| K8S-SEC-01 | Pods MUST run restricted: non-root, `allowPrivilegeEscalation:false`, `readOnlyRootFilesystem:true`, drop `ALL` caps, `seccompProfile:RuntimeDefault` (see `secure-coding.md`) | `kubesec scan` / conftest | pass |
| K8S-SEC-02 | Namespaces MUST enforce Pod Security Standard `restricted` (see `secure-coding.md`) | `kubectl get ns -L pod-security.kubernetes.io/enforce` | `restricted` |
| K8S-SEC-03 | No host namespaces, hostPath, or privileged containers | conftest / `trivy config` | 0 violations |
| K8S-NET-01 | Each namespace MUST have a default-deny `NetworkPolicy` with explicit allows (incl. DNS) | `kubectl get netpol -n <ns>` | default-deny present |
| K8S-RBAC-01 | Each workload MUST use a dedicated ServiceAccount, least-privilege RBAC, `automountServiceAccountToken:false` | review / `kubectl auth can-i --list` | no wildcard `*` rules |
| K8S-RES-01 | Every container MUST set CPU+memory requests and a memory limit | conftest / `kubectl get pods -o ...resources` | all set |
| K8S-PROBE-01 | Workloads MUST define readiness + liveness (and startup for slow boot) probes | conftest | probes present |
| K8S-AUTO-01 | Production Deployments MUST have an HPA and a PodDisruptionBudget | `kubectl get hpa,pdb -n <ns>` | both present |
| K8S-CFG-01 | No plaintext secrets in ConfigMaps/manifests/Git; use External Secrets or SealedSecrets (see `env-config.md`, `secure-coding.md`) | `trivy config` / grep | 0 plaintext secrets |
| K8S-NET-02 | External traffic MUST terminate TLS at a Gateway/Ingress (Gateway API or Istio) | review manifest | TLS configured |
| K8S-OBS-01 | Workloads MUST expose Prometheus metrics + structured stdout logs (see `observability.md`) | `ServiceMonitor` present, `/metrics` scraped | scraped |
| K8S-DEP-01 | Manifests MUST be GitOps-managed; no manual `kubectl apply` in prod (see `ci-cd.md`) | ArgoCD/Flux Application exists & synced | synced, no drift |
| K8S-SEC-04 | Cluster workloads + images MUST have 0 high/critical CVEs (see `secure-coding.md`) | `trivy k8s --severity CRITICAL,HIGH cluster` / `trivy image` | 0 high/critical |

> **Forbidden**: shipping a manifest that fails any gate above, running as root or privileged, mounting host paths, using `:latest`, putting secrets in ConfigMaps, deploying to `default` namespace, granting `*/*/*` RBAC, or `kubectl apply` straight to production outside GitOps.

---

## 3. Verification Protocol

Run, in order, before presenting manifests. Fix → re-run until every gate is green.

```bash
kubeconform -strict -summary -kubernetes-version 1.31.0 manifests/   # K8S-STRUCT-01
kustomize build overlays/production | kubectl apply --dry-run=server -f -   # K8S-STRUCT-01
conftest test manifests/ --policy policy/      # K8S-IMG/SEC/RES/PROBE/AUTO (OPA Rego policies)
kubesec scan manifests/deployment.yaml         # K8S-SEC-01 (score must be > 0)
trivy config ./manifests/                      # K8S-SEC-03/CFG-01 misconfig + secret scan
trivy k8s --severity CRITICAL,HIGH cluster     # K8S-SEC-04 (or trivy image <ref> in CI)
istioctl analyze -n <ns>                        # if mesh in use (see istio.md)
```

Manifest tests are written first (kubeconform/conftest/`helm unittest` — see `tdd.md`); the *why* behind each security/observability gate lives in its §0 owner.

---

## 4. Manifest Structure (Kustomize & Helm)

Lay out a service as a Kustomize **base** plus per-environment **overlays**; package reusable apps as a **Helm** chart. Pick one packaging tool per repo and stay consistent.

```
service/
├── base/                         # environment-neutral manifests
│   ├── deployment.yaml  service.yaml  serviceaccount.yaml
│   ├── configmap.yaml   hpa.yaml  pdb.yaml  networkpolicy.yaml
│   ├── gateway.yaml     httproute.yaml          # Gateway API (or istio/*.yaml)
│   └── kustomization.yaml
├── overlays/
│   ├── staging/  kustomization.yaml + patches
│   └── production/ kustomization.yaml + replicas/resources patches + external-secret.yaml
└── argocd/ app-prod.yaml         # GitOps entrypoint (see ci-cd.md)
```

```yaml
# base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: team-orders
resources: [namespace.yaml, serviceaccount.yaml, configmap.yaml, deployment.yaml,
            service.yaml, hpa.yaml, pdb.yaml, networkpolicy.yaml, httproute.yaml]
labels: [{ pairs: { app: order-service, team: orders } }]
images: [{ name: order-service, newName: registry.example.com/orders/order-service }]
# overlays/production sets newTag to an immutable digest and patches replicas/resources.
```

- **Kustomize**: overlays patch a shared base — no duplicated YAML; `configMapGenerator`/`secretGenerator` add a content hash so pods roll on config change.
- **Helm 3**: parameterize via `values.yaml`; gate with `helm lint`, test with `helm unittest`, validate rendered output `helm template . | kubeconform -strict`. Keep templates minimal — push policy to admission control, not Go templating.
- The namespace carries the security & routing contract:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: team-orders
  labels:
    pod-security.kubernetes.io/enforce: restricted   # K8S-SEC-02
    pod-security.kubernetes.io/warn: restricted
    istio-injection: enabled                          # if mesh (see istio.md)
---
apiVersion: v1
kind: ResourceQuota          # caps team consumption; pair with a LimitRange for defaults
metadata: { name: quota, namespace: team-orders }
spec:
  hard: { requests.cpu: "20", requests.memory: 40Gi, limits.cpu: "40", limits.memory: 80Gi, pods: "100" }
```

---

## 5. Workloads

Choose the controller by lifecycle; never create bare Pods.

| Controller | Use for | Key traits |
|---|---|---|
| **Deployment** | stateless services | rolling update, `RollingUpdate` `maxUnavailable/maxSurge`, replicas managed by HPA |
| **StatefulSet** | ordered, sticky identity + per-pod storage (databases, brokers) | stable network IDs, `volumeClaimTemplates`, ordered/parallel `podManagementPolicy` |
| **DaemonSet** | one pod per node (log/metrics agents, CNI) | tolerations for control-plane taints |
| **Job / CronJob** | run-to-completion / scheduled | `backoffLimit`, `activeDeadlineSeconds`, `ttlSecondsAfterFinished`, CronJob `concurrencyPolicy` |

This Deployment is the canonical secure template; the same `spec.template` applies to StatefulSets/DaemonSets.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
  namespace: team-orders
  labels: { app: order-service, version: v1.2.3 }
spec:
  replicas: 3                                    # K8S-AUTO-01: never 1 in prod
  strategy:
    rollingUpdate: { maxUnavailable: 0, maxSurge: 1 }   # zero-downtime rollout
  selector: { matchLabels: { app: order-service } }
  template:
    metadata:
      labels: { app: order-service, version: v1.2.3 }
      annotations: { prometheus.io/scrape: "true", prometheus.io/port: "9090" }   # K8S-OBS-01
    spec:
      serviceAccountName: order-service          # K8S-RBAC-01
      automountServiceAccountToken: false
      securityContext:                           # K8S-SEC-01 (pod level)
        runAsNonRoot: true
        runAsUser: 10001
        fsGroup: 10001
        seccompProfile: { type: RuntimeDefault }
      containers:
        - name: app
          image: registry.example.com/orders/order-service@sha256:<digest>   # K8S-IMG-01
          securityContext:                       # K8S-SEC-01 (container level)
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: { drop: ["ALL"] }
          resources:                             # K8S-RES-01
            requests: { cpu: 100m, memory: 256Mi }
            limits:   { cpu: 500m, memory: 512Mi }   # CPU limit optional; memory limit prevents OOM noisy-neighbor
          ports: [{ name: http, containerPort: 8080 }, { name: metrics, containerPort: 9090 }]
          envFrom:
            - configMapRef: { name: order-service-config }
            - secretRef:    { name: order-service-secrets }
          startupProbe:   { httpGet: { path: /health/startup, port: 8080 }, periodSeconds: 5, failureThreshold: 30 }
          readinessProbe: { httpGet: { path: /health/ready,   port: 8080 }, periodSeconds: 5,  failureThreshold: 3 }
          livenessProbe:  { httpGet: { path: /health/live,    port: 8080 }, periodSeconds: 10, failureThreshold: 3 }
          lifecycle: { preStop: { exec: { command: ["sh", "-c", "sleep 5"] } } }   # drain before SIGTERM
          volumeMounts: [{ name: tmp, mountPath: /tmp }]
      volumes: [{ name: tmp, emptyDir: { sizeLimit: 100Mi } }]   # writable scratch for read-only rootfs
      topologySpreadConstraints:                 # spread replicas across zones
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: ScheduleAnyway
          labelSelector: { matchLabels: { app: order-service } }
      terminationGracePeriodSeconds: 30
```

**Probes (K8S-PROBE-01)** — three distinct roles; do not conflate them:
- **startup**: gates the others during slow boot (migrations, cache warm); `failureThreshold × periodSeconds` = max boot time.
- **liveness**: restarts a wedged process. Check *only* in-process health — never external dependencies, or one DB blip restarts every pod.
- **readiness**: pulls a pod out of Service endpoints when it can't serve (dependency down, overloaded); restores traffic when healthy.

---

## 6. Security: Pod, RBAC & Network

Cluster security *policy* is owned by [`secure-coding.md`](guides://secure-coding.md); below is the Kubernetes binding.

- **Pod Security Standards (K8S-SEC-01/02/03)**: enforce `restricted` at the namespace; the securityContext in §5 satisfies it. Prohibited everywhere: `privileged`, `runAsUser:0`, `allowPrivilegeEscalation:true`, `hostNetwork/hostPID/hostIPC`, `hostPath` mounts, `:latest`.
- **RBAC (K8S-RBAC-01)**: one ServiceAccount per workload; `Role`/`RoleBinding` scoped to the namespace; only the verbs/`resourceNames` actually needed; never `apiGroups/resources/verbs: ["*"]`. Reserve `ClusterRole` for genuinely cluster-scoped reads (nodes, namespaces).

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata: { name: order-service, namespace: team-orders }
rules:
  - apiGroups: [""]
    resources: ["configmaps", "secrets"]
    resourceNames: ["order-service-config", "order-service-secrets"]
    verbs: ["get", "watch"]
```

- **NetworkPolicies (K8S-NET-01)**: start default-deny ingress+egress, then allow only what's needed. **Always allow DNS** or every lookup fails.

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: default-deny-all, namespace: team-orders }
spec: { podSelector: {}, policyTypes: [Ingress, Egress] }
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy                # required companion: allow kube-dns egress
metadata: { name: allow-dns, namespace: team-orders }
spec:
  podSelector: {}
  policyTypes: [Egress]
  egress:
    - to: [{ namespaceSelector: { matchLabels: { kubernetes.io/metadata.name: kube-system } } }]
      ports: [{ protocol: UDP, port: 53 }, { protocol: TCP, port: 53 }]
```

- **Admission policy (K8S-SEC-03)**: enforce the above in-cluster with built-in **ValidatingAdmissionPolicy** (CEL) or **Kyverno**/**OPA Gatekeeper** — e.g. deny `:latest`, deny missing resources, require `runAsNonRoot`. This is the runtime backstop for the §3 conftest gate.
- **mTLS / east-west authz**: delegate to the mesh — see [`istio.md`](guides://istio.md) for `PeerAuthentication: STRICT` and `AuthorizationPolicy` deny-by-default. Do not hand-roll TLS in app pods when a mesh is present.

---

## 7. Resources & Autoscaling

- **Requests/limits (K8S-RES-01)**: requests drive scheduling and the Guaranteed/Burstable QoS class; the memory limit caps blast radius. Omitting a CPU limit avoids throttling latency-sensitive services while keeping the request for fair scheduling. Right-size from observed usage (VPA in `Off`/recommend mode) rather than guessing.
- **HPA (K8S-AUTO-01)**: scale on the signal that reflects load — CPU/memory for compute-bound, a custom/external metric (RPS, queue depth) for I/O-bound. Tune `behavior` so scale-up is fast and scale-down is damped.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: order-service, namespace: team-orders }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: order-service }
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - { type: Resource, resource: { name: cpu, target: { type: Utilization, averageUtilization: 70 } } }
  behavior:
    scaleDown: { stabilizationWindowSeconds: 300 }   # damp flapping
    scaleUp:   { stabilizationWindowSeconds: 0 }
```

- **PodDisruptionBudget (K8S-AUTO-01)**: keep a quorum available during voluntary disruptions (drains, upgrades).

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata: { name: order-service, namespace: team-orders }
spec: { minAvailable: 2, selector: { matchLabels: { app: order-service } } }
```

- **Cluster autoscaler / Karpenter** provisions nodes for unschedulable pods; bound min/max. Node/pool provisioning is IaC — see [`terraform.md`](guides://terraform.md).

---

## 8. Services & Networking

- **Service**: stable virtual IP + DNS for a pod set; use **named ports** (meshes and `targetPort` references rely on them). `ClusterIP` for internal, `Headless` (`clusterIP: None`) for StatefulSet pod-addressing, `LoadBalancer`/`NodePort` only at the true edge.
- **Gateway API (K8S-NET-02)** is the current standard for ingress/L7 routing, superseding the legacy `Ingress`. Roles split cleanly: platform owns `GatewayClass`/`Gateway` (listeners, TLS); app teams own `HTTPRoute`/`GRPCRoute`.

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata: { name: edge, namespace: gateway-system }
spec:
  gatewayClassName: istio                       # or any conformant controller
  listeners:
    - name: https
      protocol: HTTPS
      port: 443
      tls: { mode: Terminate, certificateRefs: [{ name: edge-tls }] }   # K8S-NET-02
      allowedRoutes: { namespaces: { from: Selector, selector: { matchLabels: { gateway-access: "true" } } } }
---
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata: { name: order-service, namespace: team-orders }
spec:
  parentRefs: [{ name: edge, namespace: gateway-system }]
  hostnames: ["api.example.com"]
  rules:
    - matches: [{ path: { type: PathPrefix, value: /v1/orders } }]
      backendRefs: [{ name: order-service, port: 8080 }]
```

- **Mesh routing** (canary, retries, circuit breaking, weighted splits) belongs to the service mesh — see [`istio.md`](guides://istio.md) for `VirtualService`/`DestinationRule`. Async service-to-service traffic goes through a broker, not synchronous calls — see [`microservices.md`](guides://microservices.md) / [`kafka.md`](guides://kafka.md).

---

## 9. Config & Secrets

Config layering and secrets *policy* are owned by [`env-config.md`](guides://env-config.md) and [`secure-coding.md`](guides://secure-coding.md). Kubernetes binding:

- **ConfigMap** for non-sensitive config (env vars via `envFrom`, or mounted files). Roll pods on change with a hashed `configMapGenerator` (Kustomize) or a checksum annotation.
- **Secret (K8S-CFG-01)**: base64 is **not** encryption. Never commit raw Secrets. Source them externally:
  - **External Secrets Operator** — syncs from Vault/cloud secret managers into a K8s Secret (preferred for live rotation).
  - **Sealed Secrets** — encrypt at rest in Git for pure-GitOps flows.
  - Enable etcd encryption-at-rest and restrict Secret RBAC regardless.

```yaml
apiVersion: external-secrets.io/v1
kind: ExternalSecret
metadata: { name: order-service-secrets, namespace: team-orders }
spec:
  refreshInterval: 1h
  secretStoreRef: { name: vault-backend, kind: ClusterSecretStore }
  target: { name: order-service-secrets }
  data:
    - secretKey: DATABASE_URL
      remoteRef: { key: secret/data/order-service/production, property: database_url }
```

---

## 10. Storage

Stateless pods are the default; when state is unavoidable, bind it explicitly.

- **PersistentVolumeClaim** requests storage; a **StorageClass** dynamically provisions the **PersistentVolume**. Set a default StorageClass and pick `accessModes` honestly (most block volumes are `ReadWriteOnce`).
- **StatefulSet** owns per-replica state via `volumeClaimTemplates` — each pod gets its own PVC with stable identity.
- Set `persistentVolumeReclaimPolicy: Retain` for data you cannot lose; `Delete` only for reproducible caches.

```yaml
spec:                                 # inside a StatefulSet
  volumeClaimTemplates:
    - metadata: { name: data }
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd
        resources: { requests: { storage: 20Gi } }
```

---

## 11. Observability & GitOps Delivery

- **Observability (K8S-OBS-01)** — strategy owned by [`observability.md`](guides://observability.md). K8s binding: expose `/metrics`, scrape via a Prometheus `ServiceMonitor`/`PodMonitor`, define `PrometheusRule` alerts on SLO burn, ship structured JSON logs to stdout (a DaemonSet collector forwards them), propagate trace headers (the mesh emits spans). The readiness/liveness split in §5 is the availability signal.

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata: { name: order-service, namespace: team-orders, labels: { app: order-service } }
spec:
  selector: { matchLabels: { app: order-service } }
  endpoints: [{ port: metrics, path: /metrics, interval: 30s }]
```

- **GitOps (K8S-DEP-01)** — pipeline policy owned by [`ci-cd.md`](guides://ci-cd.md). K8s binding: the repo is the single source of truth; **ArgoCD**/**Flux** continuously reconciles the cluster to it with `prune` + `selfHeal`. No human runs `kubectl apply` against prod. CI gates the §3 checks before merge; CD promotes by editing the overlay's image digest.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata: { name: order-service-prod, namespace: argocd }
spec:
  project: team-orders
  source: { repoURL: https://github.com/example/manifests.git, targetRevision: main,
            path: services/order-service/overlays/production }
  destination: { server: https://kubernetes.default.svc, namespace: team-orders }
  syncPolicy:
    automated: { prune: true, selfHeal: true }
    syncOptions: [CreateNamespace=true]
  ignoreDifferences: [{ group: apps, kind: Deployment, jsonPointers: ["/spec/replicas"] }]  # HPA owns replicas
```

---

## 12. Quick Reference

```bash
# inspect
kubectl get pods,svc,hpa,pdb -n <ns> -o wide
kubectl describe pod <pod> -n <ns>; kubectl logs <pod> -c <ctr> -f --previous
kubectl top pods -n <ns>; kubectl get events -n <ns> --sort-by=.lastTimestamp
# rollouts
kubectl rollout status deploy/<name> -n <ns>
kubectl rollout undo deploy/<name> -n <ns> [--to-revision=N]
# validate (run before commit — see §3)
kubeconform -strict manifests/ ; kustomize build overlays/prod | kubectl apply --dry-run=server -f -
conftest test manifests/ --policy policy/ ; trivy config ./manifests/
# rbac & mesh
kubectl auth can-i --list --as=system:serviceaccount:<ns>:<sa>
istioctl analyze -n <ns> ; istioctl proxy-status
```

---

## 13. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements here.

- [ ] K8S-STRUCT-01 — `kubeconform -strict` + server dry-run pass
- [ ] K8S-IMG-01 — images pin a digest/tag, no `:latest`
- [ ] K8S-SEC-01 — restricted securityContext on every pod/container
- [ ] K8S-SEC-02 — namespaces enforce PSS `restricted`
- [ ] K8S-SEC-03 — no privileged / host namespaces / hostPath
- [ ] K8S-NET-01 — default-deny NetworkPolicy + DNS allow per namespace
- [ ] K8S-RBAC-01 — dedicated SA, least-privilege RBAC, automount off
- [ ] K8S-RES-01 — CPU/memory requests + memory limit on all containers
- [ ] K8S-PROBE-01 — readiness + liveness (+ startup) probes set
- [ ] K8S-AUTO-01 — HPA and PDB defined for production Deployments
- [ ] K8S-CFG-01 — no plaintext secrets; External/Sealed Secrets
- [ ] K8S-NET-02 — TLS terminated at Gateway/Ingress
- [ ] K8S-OBS-01 — Prometheus ServiceMonitor + structured stdout logs
- [ ] K8S-DEP-01 — GitOps-managed (ArgoCD/Flux), no manual apply
- [ ] K8S-SEC-04 — `trivy` reports 0 high/critical CVEs
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Kubernetes Deployment Guidelines**
