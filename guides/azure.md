# Microsoft Azure Development Guidelines
Mandatory standards for building secure, well-architected, cost-aware workloads on Microsoft Azure. Azure CLI, Bicep/ARM, Entra ID, managed identities, Azure Monitor.

---
name: azure
title: Microsoft Azure Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: infra
tools: [azure-cli@2.71, bicep@0.33, arm, entra-id]
requires:
  - secure-coding
  - observability
recommends:
  - ci-cd
  - terraform
  - kubernetes
  - dockerfile
  - env-config
  - azuredevops
provides:
  - azure-services
  - entra-rbac
  - azure-well-architected
  - azure-cost
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Azure — service selection, Entra/RBAC, the Well-Architected Framework, governance, and cost.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Azure infrastructure or code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE/IaC scanning. *(Azure binding: Entra RBAC, Key Vault, Defender for Cloud, `checkov`/`trivy` on Bicep.)*
> - [`observability.md`](guides://observability.md) — metrics/tracing/logging strategy. *(Azure binding: Azure Monitor, Log Analytics, Application Insights, KQL.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`terraform.md`](guides://terraform.md) — when provisioning Azure with Terraform instead of Bicep (`azurerm`/`azapi` providers).
> - [`kubernetes.md`](guides://kubernetes.md) — when the workload runs on AKS.
> - [`dockerfile.md`](guides://dockerfile.md) — container image build for ACR / App Service / Container Apps.
> - [`env-config.md`](guides://env-config.md) — config policy *(binding: Key Vault references, App Configuration)*.
> - [`azuredevops.md`](guides://azuredevops.md) · [`ci-cd.md`](guides://ci-cd.md) — pipelines that build, scan, and deploy to Azure.

> 📎 **SEE ALSO:** [`microservices.md`](guides://microservices.md) · [`performance.md`](guides://performance.md) · [`csharp.md`](guides://csharp.md) *(SDK client code: `DefaultAzureCredential`, `Azure.*` libraries)* · [`tdd.md`](guides://tdd.md) *(test-first IaC with Terratest / `az deployment group what-if`)*.

---

## 1. Core Philosophies: AZURE-FIRST

Azure-specific principles only. TDD, security, observability, and error handling come from §0.

- **A**utomated: every resource is Infrastructure as Code (Bicep, ARM, or Terraform). No portal click-ops; the portal is read-only in shared/prod subscriptions.
- **Z**ero-trust identity: managed identities + Entra RBAC everywhere; local auth, account keys, and connection-string secrets are disabled.
- **U**nified governance: management groups, Azure Policy, and mandatory tags enforce standards *before* resources are created.
- **R**esilient by design: availability zones, paired-region DR, and health-probed autoscaling per the Well-Architected reliability pillar (§6).
- **E**fficient: right-sized SKUs, autoscale-to-zero where possible, budgets and cost alerts on every subscription (§9).

**Verified Code**: Agent-generated Azure artifacts MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `AZ-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| AZ-IAM-01 | Workloads MUST authenticate via managed identity, never account keys or static secrets (see `secure-coding.md`) | `az webapp identity show` / grep IaC for `listKeys`/`AccountKey` | identity present, no keys |
| AZ-IAM-02 | Key Vault and data services MUST use Entra RBAC with local auth disabled | check `enableRbacAuthorization` / `disableLocalAuth` / `allowSharedKeyAccess:false` | all true |
| AZ-IAM-03 | RBAC role assignments MUST be least-privilege & scoped (no `Owner`/subscription-wide for apps) | `az role assignment list --all` review | scoped, no broad Owner |
| AZ-SEC-01 | Secrets MUST live in Key Vault, never in app settings/code (see `secure-coding.md`, `env-config.md`) | scan IaC + `az keyvault secret list` | no plaintext secrets |
| AZ-SEC-02 | IaC MUST pass security scan, 0 high/critical (see `secure-coding.md`) | `checkov -d . --framework bicep` / `trivy config .` | 0 high/critical |
| AZ-SEC-03 | TLS 1.2+ MUST be enforced; HTTPS-only; public network access disabled for PaaS data | `az policy state summarize` | compliant |
| AZ-NET-01 | PaaS data services (SQL, Storage, Cosmos, Key Vault) MUST use private endpoints + private DNS | `az network private-endpoint list -g <rg>` | all data PaaS private |
| AZ-OBS-01 | Every resource MUST send diagnostics to Log Analytics; apps MUST emit to App Insights (see `observability.md`) | `az monitor diagnostic-settings list` | settings present |
| AZ-GOV-01 | Every resource MUST carry required tags (`Environment`,`Application`,`Owner`,`CostCenter`) | `az policy state summarize` (require-tag initiative) | compliant |
| AZ-GOV-02 | Resources MUST follow the naming convention and RG-per-lifecycle layout (§4) | review / `az resource list` | conformant |
| AZ-REL-01 | Production workloads MUST be zone-redundant with a defined DR/failover target | review SKU `zoneRedundant` + paired region | zones + DR defined |
| AZ-IAC-01 | All infra MUST be code; deployments MUST pass what-if/plan before apply | `az deployment group what-if` / `terraform plan` | reviewed, no drift |
| AZ-IAC-02 | Bicep MUST lint clean and use recent API versions (≤ 730 days old) | `az bicep lint --file main.bicep` | 0 errors |
| AZ-COST-01 | Each subscription MUST have a budget and cost alert configured (§9) | `az consumption budget list` | budget present |

> **Forbidden**: portal-created prod resources, account keys / connection-string secrets in app settings, public endpoints on data PaaS, `Owner` granted to a workload identity, deploying without `what-if`/`plan`, or shipping IaC that fails `checkov`/`trivy`.

---

## 3. Azure Service Selection (azure-services)

Pick the *most managed* service that meets the requirement; drop down a layer only when a hard constraint forces it. This decision map is the canonical Azure-owned content other guides reference.

### A. Compute — when to use what
| Need | Service | Use when |
|------|---------|----------|
| HTTP app / API, fully managed | **App Service** | Web apps/APIs, deployment slots, easy autoscale, no container required |
| Event/serverless functions | **Azure Functions** | Event-driven, bursty, pay-per-execution (Consumption) or no-cold-start (Premium EP) |
| Containers without cluster ops | **Container Apps** | Microservices, KEDA event scaling, scale-to-zero, Dapr, revisions/canary |
| Full Kubernetes control | **AKS** | Need raw k8s API, custom operators, node pools, service mesh — see [`kubernetes.md`](guides://kubernetes.md) |
| OS-level / legacy / GPU | **Virtual Machines / VMSS** | Lift-and-shift, specialized OS, GPU; last resort — you own patching |

> Default to App Service or Container Apps; reach for AKS only when its control surface is genuinely required.

### B. Data & storage
| Need | Service |
|------|---------|
| Relational, SQL Server compatible | **Azure SQL Database** (serverless/GP for variable load; Hyperscale for large) |
| Global, low-latency NoSQL | **Cosmos DB** (Session consistency default; partition key = high cardinality, query-aligned) |
| Object/blob storage | **Storage — Blob** (hot/cool/archive tiers; lifecycle policies) |
| Managed file shares | **Storage — Files** (SMB/NFS) |
| Cache / session / queues | **Azure Cache for Redis** |
| Analytics / lakehouse | **Microsoft Fabric / Synapse / ADLS Gen2** |

### C. Messaging & integration
| Need | Service |
|------|---------|
| Ordered, transactional, enterprise queues/topics | **Service Bus** (sessions, dead-letter, duplicate detection) |
| Reactive event routing (pub/sub over resources) | **Event Grid** |
| High-throughput streaming/telemetry | **Event Hubs** (Kafka-compatible) |
| Workflow orchestration | **Logic Apps** / **Durable Functions** |

### D. Networking & edge
| Need | Service |
|------|---------|
| Private network boundary | **VNet** + subnets + NSGs |
| Regional L7 load balancing + WAF | **Application Gateway** (+ WAF) |
| Global L7 CDN + WAF + failover | **Front Door** (Premium: private-link origins) |
| Private connectivity to PaaS | **Private Endpoint** + **Private DNS** |
| Hybrid/on-prem connectivity | **VPN Gateway** / **ExpressRoute** |

---

## 4. Resource Organization & Governance

### A. Management hierarchy
Group by **lifecycle and blast radius**, not by team org chart:
```
Management Groups
├── Platform        → shared-services sub (identity, monitoring, connectivity hub)
├── Landing Zones
│   ├── Production   → prod-workloads sub  (rg per app + lifecycle)
│   └── Non-Prod     → staging / dev subs
└── Sandbox         → time-boxed, budget-capped experimentation
```
One **resource group per app + lifecycle stage** (e.g. `rg-myapp-prod-eastus`) so deployments and teardowns are atomic and don't touch unrelated resources.

### B. Naming & tags (AZ-GOV-01/02)
Pattern: `{type}-{workload}-{env}-{region}-{instance}` (storage/ACR drop hyphens, must be globally unique).
```
rg-myapp-prod-eastus     app-myapp-prod-eastus-001   func-myapp-prod-eastus-001
sql-myapp-prod-eastus    stmyappprodeastus           kv-myapp-prod-eastus
aks-myapp-prod-eastus    acrmyappprod                ca-myapp-prod-eastus
```
Mandatory tags enforced by Azure Policy (deny/append): `Environment`, `Application`, `Owner`, `CostCenter`, `ManagedBy`. Inherit at RG level where possible.

### C. Policy as governance
Enforce standards *before* creation, not after:
```bash
# Assign a built-in initiative (e.g. CIS Azure Benchmark) at subscription scope
az policy assignment create --name cis-benchmark \
  --policy-set-definition 06f19060-9e68-4070-92ca-f15cc126059e \
  --scope "/subscriptions/${SUBSCRIPTION_ID}"
# Require tags, allowed locations, allowed SKUs, deny public IPs — built-in defs
az policy state summarize --subscription "${SUBSCRIPTION_ID}" --output table   # AZ-GOV-01
```

---

## 5. Identity & Access — Entra ID & RBAC (entra-rbac)

The canonical Azure rule: **managed identity + Entra RBAC, zero secrets** (AZ-IAM-01/02/03). The cross-cutting *why* of secret-zero and least privilege lives in [`secure-coding.md`](guides://secure-coding.md); the Azure binding is here.

- **User-assigned managed identity** (preferred): independent lifecycle, shareable across resources, survives resource recreation. System-assigned only for single-resource, tied-lifecycle cases.
- **Never**: store keys in app settings, use account/connection-string keys, or hardcode credentials. Disable `allowSharedKeyAccess`, `disableLocalAuth`, and SQL SQL-auth where the platform supports Entra auth.
- **Always**: `DefaultAzureCredential` in app code (resolves managed identity in Azure, dev credential locally), least-privilege built-in roles scoped to the resource.

```bicep
// User-assigned identity + scoped Key Vault role (RBAC, not access policies)
resource id 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-07-31-preview' = {
  name: 'id-${appName}-${env}'
  location: location
  tags: tags
}
resource kvRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, id.id, 'Key Vault Secrets User')
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')
    principalId: id.properties.principalId
    principalType: 'ServicePrincipal'
  }
}
```

Common role definition IDs (assign by GUID, least privilege): Key Vault Secrets User `4633458b-17de-408a-b874-0445c86b69e6`, Storage Blob Data Contributor `ba92f5b4-2d11-453d-a403-e96b0029c9fe`, Service Bus Data Sender `69a216fc-b8fb-44d8-bc22-1f3c2cd27a39`, AcrPull `7f951dda-4ed3-4680-a7ca-43fe172d538d`.

In app code use the binding only (full SDK examples belong to [`csharp.md`](guides://csharp.md) and language guides):
```csharp
var cred = new DefaultAzureCredential(new DefaultAzureCredentialOptions {
    ManagedIdentityClientId = Environment.GetEnvironmentVariable("AZURE_CLIENT_ID") });
// Pass to SecretClient / BlobServiceClient / ServiceBusClient / CosmosClient — no keys.
```

---

## 6. Well-Architected Framework (azure-well-architected)

Design and review every workload against the five WAF pillars. This is the Azure-owned design rubric other guides reference.

| Pillar | Azure binding / what to enforce |
|--------|---------------------------------|
| **Reliability** | Availability zones (`zoneRedundant: true`), paired-region DR, health probes (liveness+readiness), autoscale, retries with backoff, Service Bus dead-letter monitoring (AZ-REL-01). |
| **Security** | Managed identity + Entra RBAC, private endpoints, Key Vault, Defender for Cloud, Azure Policy, TLS 1.2+ (AZ-IAM/SEC/NET — see `secure-coding.md`). |
| **Cost Optimization** | Right-size SKUs, autoscale (scale-to-zero on Container Apps/Functions Consumption), reservations/savings plans, budgets + alerts (§9, AZ-COST-01). |
| **Operational Excellence** | IaC for everything, what-if before apply, CI/CD with approvals, diagnostics to Log Analytics, runbooks (AZ-IAC/OBS — see `azuredevops.md`, `ci-cd.md`). |
| **Performance Efficiency** | Match SKU to load profile, Cosmos partition-key design, caching (Redis/Front Door), Premium Functions to kill cold start, CDN at the edge (see `performance.md`). |

Run `az advisor recommendation list --category <Cost|Security|HighAvailability|Performance>` and Defender for Cloud secure score as the ongoing WAF scorecard.

---

## 7. Infrastructure as Code — Bicep

Bicep is the Azure-native IaC language (transpiles to ARM) and is owned by this guide. For multi-cloud or existing Terraform estates, provision Azure via [`terraform.md`](guides://terraform.md) (`azurerm`/`azapi`) — the same requirements (§2) apply.

### A. Module layout
```
infra/
├── main.bicep                       # entry point (targetScope = 'resourceGroup'|'subscription')
├── parameters/{dev,staging,prod}.bicepparam
├── modules/
│   ├── networking/   vnet.bicep, nsg.bicep, private-endpoints.bicep
│   ├── compute/      app-service.bicep, function-app.bicep, container-app.bicep
│   ├── data/         sql.bicep, cosmos.bicep, storage.bicep
│   ├── security/     key-vault.bicep, managed-identity.bicep
│   └── monitoring/   app-insights.bicep, log-analytics.bicep
└── bicepconfig.json                 # linter rules
```

### B. Patterns
```bicep
targetScope = 'resourceGroup'
param appName string
@allowed(['dev','staging','prod']) param env string
param location string = resourceGroup().location
param tags object = { Application: appName, Environment: env, ManagedBy: 'Bicep' }

module net 'modules/networking/vnet.bicep' = {
  name: 'net-${uniqueString(deployment().name)}'
  params: { appName: appName, env: env, location: location, tags: tags }
}
// Conditional (prod-only) + loop over a list
resource pe 'Microsoft.Network/privateEndpoints@2024-05-01' = if (env == 'prod') { /* ... */ }
resource queues 'Microsoft.ServiceBus/namespaces/queues@2024-01-01' = [for q in ['orders','audit']: {
  parent: sbNamespace
  name: q
  properties: { maxDeliveryCount: 5, deadLetteringOnMessageExpiration: true }
}]
```

### C. Security-critical defaults to bake into every module
Set these on the relevant resources — they map directly to §2 gates:
- **App Service / Functions / Container Apps**: `httpsOnly: true`, `minTlsVersion: '1.2'`, `ftpsState: 'Disabled'`, managed identity, `healthCheckPath`, App Insights connection string, VNet integration.
- **Storage**: `allowBlobPublicAccess: false`, `allowSharedKeyAccess: false`, `minimumTlsVersion: 'TLS1_2'`, `networkAcls.defaultAction: 'Deny'`.
- **SQL**: `publicNetworkAccess: 'Disabled'`, `minimalTlsVersion: '1.2'`, Entra-only admin, private endpoint.
- **Key Vault**: `enableRbacAuthorization: true`, `enableSoftDelete: true`, `enablePurgeProtection: true`, `networkAcls.defaultAction: 'Deny'`.
- **Cosmos**: `disableLocalAuth: true`, `publicNetworkAccess: 'Disabled'`, multi-region + `enableAutomaticFailover`.

### D. Linting & validation (AZ-IAC-01/02)
```json
// bicepconfig.json — key rules
{ "analyzers": { "core": { "rules": {
  "secure-parameter-default": { "level": "error" },
  "no-hardcoded-env-urls":    { "level": "error" },
  "no-unused-params":         { "level": "error" },
  "use-recent-api-versions":  { "level": "warning", "configuration": { "maxAgeInDays": 730 } }
}}}}
```
```bash
az bicep lint --file main.bicep                                            # AZ-IAC-02
az deployment group what-if -g rg-myapp --template-file main.bicep \
  --parameters @parameters/prod.bicepparam                                 # AZ-IAC-01: preview before apply
az bicep decompile --file azuredeploy.json                                 # migrate ARM → Bicep
```

> Containers: build images per [`dockerfile.md`](guides://dockerfile.md), push to **ACR**, and pull via managed identity (`AcrPull`) — never registry admin user/password.

---

## 8. Networking & Edge

- **VNet + subnets + NSGs** as the boundary; delegate subnets for App Service / Container Apps / AKS as needed.
- **Private endpoints + Private DNS** for all data PaaS (AZ-NET-01). DNS zones: Key Vault `privatelink.vaultcore.azure.net`, SQL `privatelink.database.windows.net`, Blob `privatelink.blob.core.windows.net`, Cosmos `privatelink.documents.azure.com`.
- **Front Door Premium** for global L7 + WAF (`Microsoft_DefaultRuleSet 2.1` + `Microsoft_BotManagerRuleSet` + rate-limit rules) with private-link origins to keep backends off the public internet; **Application Gateway + WAF** for regional.
- NSG baseline: `AllowHTTPS (100)`, `AllowAzureLB (110)`, `DenyAllInbound (4096)`; private-endpoint subnets allow only intra-VNet inbound.

```bash
az afd profile create --profile-name afd-myapp -g rg-myapp --sku Premium_AzureFrontDoor
az network private-endpoint list -g rg-myapp --output table                 # AZ-NET-01
```

---

## 9. Cost Management (azure-cost)

Cost is a first-class design constraint (WAF Cost pillar). The Azure-owned controls:

- **Budgets + alerts per subscription** (AZ-COST-01) with action groups for 50/80/100% thresholds.
- **Right-sizing & autoscale**: scale-to-zero on Container Apps and Functions Consumption; autoscale App Service / VMSS on CPU/queue metrics; stop non-prod out of hours.
- **Reservations & savings plans** for steady-state compute; **spot** VMs for interruptible work.
- **Tag-driven showback/chargeback**: `CostCenter`/`Application` tags (AZ-GOV-01) drive Cost Management views.
- **Storage lifecycle**: tier hot→cool→archive, delete stale blobs via lifecycle policy.

```bash
az consumption budget list --output table                                   # AZ-COST-01
az advisor recommendation list --category Cost --output table
az costmanagement query --type ActualCost --timeframe MonthToDate \
  --scope "/subscriptions/${SUBSCRIPTION_ID}"
```

---

## 10. Observability binding (Azure Monitor)

Strategy (what to measure, SLOs, trace propagation, alert design) is owned by [`observability.md`](guides://observability.md). The Azure binding (AZ-OBS-01):

- **Log Analytics workspace** is the sink; wire `Microsoft.Insights/diagnosticSettings` on every resource.
- **Application Insights** (workspace-based) for distributed tracing and app telemetry; inject `APPLICATIONINSIGHTS_CONNECTION_STRING`, enable for App Service/Functions/Container Apps.
- **Metric alerts** (`Microsoft.Insights/metricAlerts`) on Http5xx, latency, CPU, dead-letter depth → **action groups** (`evaluationFrequency: PT5M`, `windowSize: PT15M` for prod).

KQL is Azure-specific — keep these in the operator's toolbox:
```kql
requests | where timestamp > ago(1h) and success == false
        | summarize count() by resultCode, name, bin(timestamp, 5m)
requests | where timestamp > ago(24h)
        | summarize availability = countif(success)*100.0/count(), p95 = percentile(duration,95) by name
union requests, dependencies, exceptions | where operation_Id == "<id>" | order by timestamp asc   // end-to-end trace
```

---

## 11. Security binding (Defender, Policy, Key Vault)

Policy (CVE management, supply chain, secret handling) is owned by [`secure-coding.md`](guides://secure-coding.md); config policy by [`env-config.md`](guides://env-config.md). The Azure binding:

- **Key Vault** is the only secret/cert store. Consume via `@Microsoft.KeyVault(...)` references in App Service settings so secrets surface as env vars with no code change (set `keyVaultReferenceIdentity`). For non-secret config use **App Configuration**. Cache secrets with short TTL for rotation.
- **Defender for Cloud** for posture management and secure score; **Azure Policy** for compliance enforcement.
- **IaC + image scanning** in CI (see `secure-coding.md` for the gate): `checkov`/`trivy` on Bicep/ARM/Terraform, Defender for Containers on ACR images.

```bash
az security secure-score list --output table          # Defender for Cloud posture
az policy state list --filter "complianceState eq 'NonCompliant'" --output table
checkov -d . --framework bicep                         # AZ-SEC-02
trivy config --severity HIGH,CRITICAL .                # AZ-SEC-02
az keyvault update --name kv-myapp --enable-purge-protection true
```

---

## 12. Quick Reference

```bash
az login && az account set --subscription "Name"
az group create --name rg-myapp --location eastus
az deployment group what-if -g rg-myapp --template-file main.bicep --parameters @parameters/prod.bicepparam
az deployment group create  -g rg-myapp --template-file main.bicep --parameters @parameters/prod.bicepparam
az bicep lint --file main.bicep
az webapp identity show -g rg-myapp -n app-myapp                 # confirm managed identity
az keyvault secret show --vault-name kv-myapp --name secret-name
az network private-endpoint list -g rg-myapp --output table
az policy state summarize --subscription "${SUBSCRIPTION_ID}"
az consumption budget list --output table
```

---

## 13. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] AZ-IAM-01 — managed identity used, no account keys/static secrets
- [ ] AZ-IAM-02 — Entra RBAC on Key Vault & data; local auth/shared-key disabled
- [ ] AZ-IAM-03 — role assignments least-privilege & scoped (no broad Owner)
- [ ] AZ-SEC-01 — all secrets in Key Vault, none in app settings/code
- [ ] AZ-SEC-02 — `checkov`/`trivy` on IaC, 0 high/critical
- [ ] AZ-SEC-03 — TLS 1.2+, HTTPS-only, PaaS public access disabled
- [ ] AZ-NET-01 — private endpoints + private DNS for all data PaaS
- [ ] AZ-OBS-01 — diagnostics → Log Analytics, apps → App Insights
- [ ] AZ-GOV-01 — required tags present (policy compliant)
- [ ] AZ-GOV-02 — naming convention & RG-per-lifecycle followed
- [ ] AZ-REL-01 — zone-redundant prod + DR/failover defined
- [ ] AZ-IAC-01 — infra as code, what-if/plan reviewed, no drift
- [ ] AZ-IAC-02 — Bicep lints clean, recent API versions
- [ ] AZ-COST-01 — budget + cost alert configured per subscription
- [ ] Agent ran every §12 verification and documented any fixes

---
**End of Microsoft Azure Development Guidelines**
