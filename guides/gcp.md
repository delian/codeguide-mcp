# Google Cloud Platform (GCP) Development Guidelines
Mandatory standards for designing on Google Cloud: service selection, IAM least-privilege, Workload Identity, resource hierarchy, labels, and cost control. gcloud CLI, Cloud Deploy, IAM, Cloud Asset Inventory.

---
name: gcp
title: Google Cloud Platform (GCP) Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: infra
tools: [gcloud, cloud-deploy, gcloud-iam, cloud-asset-inventory]
requires:
  - secure-coding
  - observability
recommends:
  - ci-cd
  - terraform
  - kubernetes
  - dockerfile
  - env-config
provides:
  - gcp-services
  - gcp-iam
  - workload-identity
  - gcp-cost
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Google Cloud — the platform's service model, IAM, resource hierarchy, and cost/governance levers.

---

## 0. Prerequisites & References

Fetch and apply these **before** designing or provisioning GCP resources. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — secrets, supply chain, CVE policy. *(GCP binding: IAM least-privilege, Secret Manager, Artifact Analysis, Security Command Center, Org Policy.)*
> - [`observability.md`](guides://observability.md) — metrics, tracing, SLOs, alerting. *(GCP binding: Cloud Monitoring, Cloud Logging, Cloud Trace; OTel exports natively.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`terraform.md`](guides://terraform.md) — IaC policy & workflow *(binding: `hashicorp/google` provider, GCS remote state)*
> - [`kubernetes.md`](guides://kubernetes.md) — workload/cluster policy *(binding: GKE, Workload Identity)*
> - [`dockerfile.md`](guides://dockerfile.md) — image authoring *(binding: Artifact Registry, Artifact Analysis scanning)*
> - [`env-config.md`](guides://env-config.md) — config/secret separation *(binding: Secret Manager, runtime env)*
> - [`ci-cd.md`](guides://ci-cd.md) — pipeline policy *(binding: Cloud Build, Cloud Deploy)*

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) *(test IaC with Terratest/native plan checks)* · [`microservices.md`](guides://microservices.md) · [`grpc.md`](guides://grpc.md) · [`rest.md`](guides://rest.md)

---

## 1. Core Philosophies: GCP-FIRST

GCP-specific principles only. Security, secrets, observability, and IaC policy come from §0.

- **G**overned hierarchy: every resource lives under an Org → Folder → Project tree; the Project is the unit of IAM, billing, and quota isolation. One environment = one project.
- **C**onsumption-aware: prefer managed/serverless (Cloud Run, Cloud Functions, BigQuery, Firestore) that scale to zero; choose the **smallest** service that meets the workload (see §4 selection matrix).
- **P**rincipled identity: workloads authenticate as dedicated service accounts with least privilege; **no long-lived service-account keys** — use Workload Identity / Workload Identity Federation.
- **F**ramework-aligned: design against the [Google Cloud Architecture Framework](https://cloud.google.com/architecture/framework) pillars (operational excellence, security, reliability, cost, performance).
- **I**nfrastructure-as-code: all resources are declared (Terraform — see §0), never click-ops; state is versioned and access-controlled.
- **R**egional intent: choose region/zone explicitly for latency, data residency, and cost; design multi-zone (and multi-region where the SLO demands) rather than defaulting.
- **S**ignal-rich: labels on every resource for cost, ownership, and lifecycle attribution; budgets and alerts on every billing account.

**Verified Code**: Agent-generated GCP designs MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `GCP-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| GCP-IAM-01 | Workloads MUST use a dedicated, least-privilege service account; primitive roles (`roles/owner`, `roles/editor`) MUST NOT be granted to workload SAs | `gcloud projects get-iam-policy PROJECT_ID --flatten="bindings[].members" --filter="bindings.role:(roles/owner OR roles/editor)"` | no workload SA bound |
| GCP-IAM-02 | User-managed service-account keys MUST NOT be created; CI/CD & external workloads MUST use Workload Identity Federation (see `secure-coding.md`) | `gcloud iam service-accounts keys list --iam-account=SA --managed-by=user` | only `system-managed` |
| GCP-IAM-03 | IAM Recommender findings (unused/excess permissions) MUST be triaged | `gcloud recommender recommendations list --recommender=google.iam.policy.Recommender --location=global` | 0 unactioned high |
| GCP-SEC-01 | Secrets MUST live in Secret Manager, never in env literals or source (see `secure-coding.md`, `env-config.md`) | `gcloud secrets list` + repo grep | 0 plaintext secrets |
| GCP-SEC-02 | Container & IaC scans MUST be clean of high/critical (see `secure-coding.md`) | `gcloud artifacts docker images scan IMG --remote`; `trivy config .` | 0 high/critical |
| GCP-SEC-03 | Org Policy guardrails MUST be enforced (key-creation off, uniform bucket access, shielded VM) | `gcloud org-policies list --organization=ORG_ID` | required constraints present |
| GCP-NET-01 | Managed data services (Cloud SQL, Memorystore, etc.) MUST use private IP; no public ingress unless justified | `gcloud sql instances describe INST --format='value(settings.ipConfiguration.ipv4Enabled)'` | `False` |
| GCP-STRUCT-01 | Each environment MUST be an isolated project under the Org/Folder hierarchy | `gcloud projects list`; `gcloud resource-manager folders list` | per-env projects |
| GCP-STRUCT-02 | Every resource MUST carry mandatory labels (`environment`, `team`, `managed-by`) | `gcloud asset search-all-resources --scope=projects/PROJECT_ID --query="NOT labels:environment"` | 0 unlabeled |
| GCP-IAC-01 | All resources MUST be provisioned via IaC with remote, versioned state (see `terraform.md`) | `terraform plan` | no drift |
| GCP-OBS-01 | Services MUST export metrics/logs/traces to Cloud Operations with alerting + uptime checks (see `observability.md`) | `gcloud monitoring policies list`; `gcloud logging sinks list` | alerts + sinks exist |
| GCP-COST-01 | Every billing account MUST have a budget with threshold alerts | `gcloud billing budgets list --billing-account=ACCT` | ≥1 budget with alerts |

> **Forbidden**: creating user-managed SA keys; granting Owner/Editor to a workload; public IP on a database without an approved exception; hardcoded secrets; click-ops resources outside IaC; unlabeled or unbudgeted resources.

---

## 3. Verification Protocol

Run before presenting a design or change. Fix → re-run until every gate is green.

```bash
terraform plan                                              # GCP-IAC-01 (no drift)
gcloud projects get-iam-policy PROJECT_ID --flatten="bindings[].members" \
  --format="table(bindings.role,bindings.members)"          # GCP-IAM-01
gcloud iam service-accounts keys list --iam-account=SA --managed-by=user   # GCP-IAM-02
gcloud recommender recommendations list \
  --recommender=google.iam.policy.Recommender --location=global            # GCP-IAM-03
trivy config .                                              # GCP-SEC-02 (IaC)
gcloud artifacts docker images scan "$IMG" --remote         # GCP-SEC-02 (image)
gcloud org-policies list --organization=ORG_ID              # GCP-SEC-03
gcloud asset search-all-resources --scope=projects/PROJECT_ID \
  --query="NOT labels:environment"                          # GCP-STRUCT-02
gcloud billing budgets list --billing-account=ACCT          # GCP-COST-01
```

The *why* behind security/observability gates lives in their §0 owners; do not re-derive it here.

---

## 4. Service Selection — choose the smallest fit

GCP's core value is picking the right managed service. Default to serverless; move down only when a constraint forces it.

### Compute
| Need | Service | Use when |
|------|---------|----------|
| Stateless HTTP/gRPC, scale-to-zero | **Cloud Run** | APIs, web backends, event consumers; containerized, bursty/variable traffic. **Default choice.** |
| Event-driven function | **Cloud Functions (2nd gen)** | Single-purpose glue triggered by HTTP/Eventarc/Pub/Sub; minimal ops. |
| Orchestrated containers, mesh, custom networking | **GKE** (Autopilot first) | Multi-service platforms, stateful workloads, sidecars/operators. Policy → [`kubernetes.md`](guides://kubernetes.md). |
| Full OS / legacy / GPU / specific kernel | **Compute Engine (GCE)** | Lift-and-shift, licensed software, niche hardware. Last resort. |
| Batch / scheduled jobs | **Cloud Run Jobs** or **Batch** | Finite tasks, fan-out, nightly ETL. |

> Prefer **GKE Autopilot** over Standard unless you need node-level control. Prefer **Cloud Run Direct VPC egress** over the legacy Serverless VPC Access connector.

### Storage & data
| Need | Service |
|------|---------|
| Object/blob storage, static assets, data lake | **Cloud Storage (GCS)** — uniform bucket-level access, lifecycle classes (Standard→Nearline→Coldline→Archive) |
| Relational OLTP, regional | **Cloud SQL** (Postgres/MySQL/SQL Server) — private IP, HA, PITR |
| Relational, global, horizontal scale | **Spanner** — strong consistency at planet scale; use when Cloud SQL can't scale |
| Document/serverless NoSQL | **Firestore** (Native mode) — mobile/web, real-time listeners, offline |
| Wide-column, high-throughput | **Bigtable** — time-series, IoT, > 10k QPS sustained |
| Analytics / OLAP / warehouse | **BigQuery** — serverless SQL; partition + cluster; separate from OLTP |
| In-memory cache | **Memorystore** (Redis/Memcached) |

### Networking & messaging
- **VPC** (custom-mode, not auto) with explicit subnets; **Shared VPC** for centrally-managed networking across projects; **VPC Service Controls** to build data-exfiltration perimeters.
- **Cloud Load Balancing** (Global external HTTPS LB) fronts Cloud Run/GKE/GCE via NEGs; pair with **Cloud CDN** for cacheable content and **Cloud Armor** for WAF/DDoS on public endpoints.
- **Pub/Sub** for async/event-driven decoupling: at-least-once by default, opt into ordering and exactly-once; always configure a **dead-letter topic** and retry policy. **Eventarc** routes events (GCS, Audit Logs, Pub/Sub) to Cloud Run/Functions.

---

## 5. Resource Hierarchy, Projects & Labels

The hierarchy is the backbone of governance — IAM, Org Policy, and budgets all inherit down it.

```
Organization (1 per company; ties to Cloud Identity/Workspace)
└── Folders (by department / environment / team)
    ├── prod/     → project: acme-prod-api,  acme-prod-data
    ├── nonprod/  → project: acme-stg-api,   acme-dev-api
    └── shared/   → project: acme-shared-vpc, acme-logging
```

- **Project = isolation unit** for IAM, billing, quota, and APIs. One environment per project (GCP-STRUCT-01); never share prod and dev in one project.
- **Folders** group projects so an IAM/Org-Policy binding applies to many projects at once (least privilege at the right altitude).
- **Naming**: `{org}-{env}-{app}` for project IDs (globally unique, immutable); resource names `{project}-{resource}-{purpose}`.
- **Labels (GCP-STRUCT-02)** — mandatory on every resource for cost attribution and automation: `environment`, `team`, `cost-center`, `managed-by` (=`terraform`). Labels flow into billing export and Cloud Asset Inventory queries.
- Enable only the APIs a project needs (`gcloud services enable …`) — reduces attack surface and quota noise.

---

## 6. IAM & Identity

GCP's authorization model is `member × role × resource`, evaluated with inheritance down the hierarchy. This is the area to get right.

- **Least privilege (GCP-IAM-01)**: grant predefined roles scoped as tightly as possible; build a **custom role** when predefined roles are too broad. Never grant primitive `roles/owner`/`roles/editor` to a workload.
- **Service accounts** are the identity of workloads, not people. One SA per workload, named for it. Grant the SA only the roles its workload calls.
- **No keys (GCP-IAM-02)**: exported SA key files are the leading GCP credential-leak vector.
  - **On GKE**: bind the Kubernetes SA to a Google SA via **Workload Identity** (`iam.workloadIdentityUser`).
  - **On Cloud Run/Functions/GCE**: attach the SA directly — the metadata server issues short-lived tokens.
  - **From CI/CD or other clouds**: use **Workload Identity Federation** — an OIDC/SAML provider (e.g. GitHub Actions) impersonates the SA with no stored key.
- **IAM Recommender (GCP-IAM-03)** continuously flags unused permissions — triage and tighten.
- **Org Policy (GCP-SEC-03)** sets guardrails that even project owners can't override: `iam.disableServiceAccountKeyCreation`, `compute.requireShieldedVm`, `storage.uniformBucketLevelAccess`, `iam.allowedPolicyMemberDomains`.

```bash
# Bind a workload SA to a minimal predefined role (least privilege)
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:app@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" --condition=None

# Workload Identity Federation: let GitHub Actions impersonate an SA, no key
gcloud iam service-accounts add-iam-policy-binding deploy@PROJECT_ID.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/NUM/locations/global/workloadIdentityPools/github/attribute.repository/acme/repo"
```

Secrets (GCP-SEC-01) live in **Secret Manager** with IAM-scoped access, versioning, and optional Pub/Sub-triggered rotation. Mount them as runtime references (Cloud Run `--set-secrets`, GKE CSI driver) — never bake into images or env literals. Policy → [`secure-coding.md`](guides://secure-coding.md), [`env-config.md`](guides://env-config.md).

---

## 7. Security, Observability & Reliability bindings

The *policies* are owned by §0 references; below is only the GCP binding.

- **Security** (see [`secure-coding.md`](guides://secure-coding.md)): **Security Command Center** for org-wide findings; **Artifact Analysis** for automatic image CVE scanning on push; **Binary Authorization** to enforce attested images at deploy; **Cloud Armor** WAF on public LBs; **VPC Service Controls** perimeters for sensitive data. Scan IaC with `trivy config` / `checkov` in CI.
- **Observability** (see [`observability.md`](guides://observability.md)): instrument with **OpenTelemetry**, exporting to **Cloud Monitoring** (metrics, SLOs, alert policies, uptime checks), **Cloud Logging** (emit structured JSON to stdout — auto-ingested; correlate with `logging.googleapis.com/trace`), and **Cloud Trace**. Route logs to BigQuery via a **log sink** for long-term analysis.
- **Reliability**: configure startup/liveness/readiness probes; min-instances to tame Cold Starts on latency-critical Cloud Run; regional HA + PITR for Cloud SQL; multi-region for the data tier where the SLO requires it; dead-letter topics for Pub/Sub.

---

## 8. Provisioning & Delivery

- **IaC (GCP-IAC-01)**: declare everything in Terraform with the `hashicorp/google` (and `google-beta`) provider; keep state in a **versioned GCS backend** with object versioning and locking; structure as reusable modules per service and a thin per-environment composition. Full policy → [`terraform.md`](guides://terraform.md). Enable APIs declaratively via `google_project_service`.
- **Images**: build per [`dockerfile.md`](guides://dockerfile.md); push to **Artifact Registry** (regional, immutable tags, cleanup policies) — `gcr.io` is deprecated. Authenticate with `gcloud auth configure-docker REGION-docker.pkg.dev`.
- **CI/CD** (see [`ci-cd.md`](guides://ci-cd.md)): build/test in **Cloud Build** (or any CI authenticating via Workload Identity Federation), then progressive delivery with **Cloud Deploy** (canary/rollout pipelines to Cloud Run/GKE). Promote with `--no-traffic` + traffic-split, gate on integration tests and image scans.

---

## 9. Cost Control

Cost is a first-class design constraint (Architecture Framework pillar).

- **Budgets & alerts (GCP-COST-01)** on every billing account, with thresholds (50/90/100%) wired to Pub/Sub/email for programmatic response.
- **Billing export → BigQuery**; slice spend by the §5 labels to attribute cost to teams/environments.
- **Right-size**: scale-to-zero serverless for spiky traffic; **Committed Use Discounts** and **Spot/Preemptible VMs** for steady or fault-tolerant compute; **GCS lifecycle rules** to tier cold data down to Coldline/Archive; partition + cluster BigQuery and always filter on the partition column to avoid full scans; set quotas to cap runaway spend.
- Use the **Recommender** suite (cost, IAM, sizing) and the Pricing Calculator before committing to an architecture.

---

## 10. Quick Reference

```bash
gcloud auth login                                           # authenticate (humans)
gcloud config set project PROJECT_ID                        # target project
gcloud services enable run.googleapis.com                   # enable an API
gcloud run deploy SVC --image IMG --region REGION \
  --no-allow-unauthenticated --service-account SA           # deploy Cloud Run
gcloud secrets versions access latest --secret=SECRET_ID    # read a secret
gcloud projects get-iam-policy PROJECT_ID                   # audit IAM
gcloud asset search-all-resources --scope=projects/PROJECT_ID  # inventory + labels
gcloud recommender recommendations list \
  --recommender=google.iam.policy.Recommender --location=global  # excess perms
gcloud billing budgets list --billing-account=ACCT          # budgets
bq query --use_legacy_sql=false 'SELECT ...'                # BigQuery
```

---

## 11. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] GCP-IAM-01 — workloads on least-privilege SAs; no Owner/Editor on workloads
- [ ] GCP-IAM-02 — no user-managed SA keys; Workload Identity (Federation) for CI/external
- [ ] GCP-IAM-03 — IAM Recommender findings triaged
- [ ] GCP-SEC-01 — secrets in Secret Manager, none in source/env
- [ ] GCP-SEC-02 — image & IaC scans clean of high/critical
- [ ] GCP-SEC-03 — Org Policy guardrails enforced
- [ ] GCP-NET-01 — managed data services on private IP, no unjustified public ingress
- [ ] GCP-STRUCT-01 — one project per environment under the Org/Folder hierarchy
- [ ] GCP-STRUCT-02 — mandatory labels on every resource
- [ ] GCP-IAC-01 — provisioned via Terraform, remote versioned state, no drift
- [ ] GCP-OBS-01 — metrics/logs/traces to Cloud Operations, alerts + uptime checks
- [ ] GCP-COST-01 — budgets with threshold alerts on every billing account
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Google Cloud Platform (GCP) Guidelines**
