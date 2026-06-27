# AWS Development Guidelines
Mandatory standards for architecting on Amazon Web Services: right service for the job, least-privilege IAM, Well-Architected, cost-optimized, fully tagged, multi-account. AWS CLI v2, CloudFormation/CDK, SAM, IAM.

---
name: aws
title: AWS Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: infra
tools: [aws-cli@2, cloudformation, aws-cdk@2, sam-cli, iam]
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
  - aws-services
  - iam-least-privilege
  - well-architected
  - aws-cost-optimization
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns **AWS service selection, IAM least privilege, the Well-Architected pillars, cost optimization, tagging, and multi-account strategy**. It does not restate generic security, observability, IaC, container, or config rules — those are bound to their canonical owners below.

---

## 0. Prerequisites & References

Fetch and apply these **before** designing or provisioning AWS infrastructure. Their rules are assumed here and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — vulnerability scanning, supply chain, secrets, crypto policy. *(AWS binding: IAM least privilege, KMS for encryption, Secrets Manager, GuardDuty/Inspector/Security Hub/Access Analyzer.)*
> - [`observability.md`](guides://observability.md) — metrics, tracing, SLO/SLI, alerting. *(AWS binding: CloudWatch metrics/alarms/Logs, X-Ray / OpenTelemetry via ADOT, CloudTrail audit.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`terraform.md`](guides://terraform.md) — IaC workflow & policy *(AWS binding: `aws` provider, or native CloudFormation/CDK/SAM)*
> - [`env-config.md`](guides://env-config.md) — configuration policy *(AWS binding: SSM Parameter Store + Secrets Manager)*
> - [`dockerfile.md`](guides://dockerfile.md) — container build/security *(AWS binding: ECR, image scanning)*
> - [`kubernetes.md`](guides://kubernetes.md) — K8s workloads *(AWS binding: EKS, IRSA / Pod Identity)*
> - [`ci-cd.md`](guides://ci-cd.md) — pipeline stages, deployment strategies, rollback.

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) *(infra is test-first; CDK assertions / cfn-lint bind the cycle)* · [`gcp.md`](guides://gcp.md) · [`azure.md`](guides://azure.md) · [`microservices.md`](guides://microservices.md)

---

## 1. Core Philosophies: AWS-FIRST

AWS-specific principles only. Security, observability, IaC, and config policy come from §0 — do not restate them here.

- **A**utomate everything as code: no resource is created in the console; every account, role, and resource is defined in IaC (see `terraform.md`) and deployed via a pipeline (see `ci-cd.md`).
- **W**ell-Architected: every design is justified against the six pillars (§3); trade-offs are recorded.
- **S**coped identity: least privilege by default — scope every policy to specific actions, resources, and conditions; prefer roles with temporary STS credentials over long-lived keys.
- **F**it the service to the workload: pick the managed service that removes the most undifferentiated heavy lifting (§4); do not self-host what AWS operates.
- **I**solate blast radius: multi-account by environment/workload (§7); private subnets, VPC endpoints, and security groups locked down.
- **R**ight-sized & tagged: every resource carries the mandatory tag set (§6) and is sized/purchased for cost (§5).
- **T**raceable: CloudTrail in every account/region, CloudWatch + X-Ray on every workload (see `observability.md`).

**Verified Architecture**: Agent-generated AWS infrastructure MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `AWS-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| AWS-IAM-01 | Identity/resource policies MUST NOT use `"Action":"*"` with `"Resource":"*"`; scope to specific actions + ARNs | `checkov -d . `; IAM Access Analyzer policy validation | 0 wildcard-on-wildcard |
| AWS-IAM-02 | Workloads MUST use IAM roles (STS), never embedded long-lived access keys (see `secure-coding.md`) | `aws iam list-access-keys`; grep IaC for `AKIA` | 0 static keys in workloads |
| AWS-IAM-03 | Delegated/CI roles MUST carry a permission boundary | review / `checkov` | boundary attached |
| AWS-SEC-01 | Data MUST be encrypted at rest (KMS) and in transit (TLS); S3 deny non-TLS (see `secure-coding.md`) | `checkov` / `trivy config .` | 0 high/critical |
| AWS-SEC-02 | S3 buckets MUST block all public access unless explicitly justified | `aws s3control get-public-access-block`; `checkov` | all 4 flags true |
| AWS-SEC-03 | Secrets MUST live in Secrets Manager / SSM SecureString, never plaintext (see `env-config.md`, `secure-coding.md`) | `git-secrets --scan`; grep | 0 plaintext secrets |
| AWS-SEC-04 | IaC MUST pass a security scan with 0 high/critical (see `secure-coding.md`) | `checkov -d .` / `trivy config .` | 0 high/critical |
| AWS-OBS-01 | CloudTrail MUST be enabled in all regions; logs immutable (see `observability.md`) | `aws cloudtrail describe-trails` | multi-region trail on |
| AWS-OBS-02 | Workloads MUST emit metrics + traces and have alarms on error/latency/saturation (see `observability.md`) | review dashboards/alarms | alarms wired to SNS |
| AWS-ARCH-01 | All infrastructure MUST be defined as code; no console-created resources (see `terraform.md`) | drift detection (`cdk diff` / `terraform plan` / CFN drift) | no drift |
| AWS-ARCH-02 | Production workloads MUST be multi-AZ; async paths MUST have a DLQ | review / `checkov` | ≥2 AZ, DLQ present |
| AWS-COST-01 | Every resource MUST carry the mandatory tag set (§6) | `aws resourcegroupstaggingapi get-resources`; AWS Config rule `required-tags` | 0 untagged |
| AWS-COST-02 | Accounts MUST have a Budget + Cost Anomaly Detection alert | `aws budgets describe-budgets` | budget + anomaly monitor exist |

> **Forbidden**: `iam:*`/`*:*` policies, root-account access keys, public S3 unless reviewed, secrets in env vars or templates, console-created ("ClickOps") resources, single-AZ production, or untagged resources.

---

## 3. Well-Architected Framework

Every design is evaluated against the six pillars. This is the AWS lens on principles owned elsewhere — bind, don't restate.

| Pillar | What AWS-specific decisions it drives | Bound owner |
|--------|---------------------------------------|-------------|
| **Operational Excellence** | IaC for all changes; small reversible deploys; runbooks; game days | `ci-cd.md`, `terraform.md` |
| **Security** | IAM least privilege, KMS, GuardDuty/Security Hub/Inspector, SCPs, encryption everywhere | `secure-coding.md` |
| **Reliability** | Multi-AZ, auto-scaling, health checks, DLQs, backups (RDS snapshots, DynamoDB PITR), tested recovery | `observability.md` |
| **Performance Efficiency** | Right service & instance family (Graviton/ARM64), caching (CloudFront/ElastiCache), serverless-first | `performance.md` |
| **Cost Optimization** | Right-sizing, Savings Plans/Spot, storage tiering, tagging-driven attribution (§5) | this guide |
| **Sustainability** | Graviton, managed services, right-sizing, region choice | this guide |

Use the AWS Well-Architected Tool to run reviews; record material trade-offs as ADRs (see `adr.md`).

---

## 4. Core Services — When to Use Which

The heart of this guide: pick the service that removes the most undifferentiated work for the workload. Default **serverless/managed first**; move down the stack only when you need the control.

### A. Compute

| Service | Use when | Avoid when |
|---------|----------|-----------|
| **Lambda** | Event-driven, spiky, short (<15 min) tasks; glue; APIs with variable load | Long-running, heavy CPU/GPU, sustained high throughput where cost flips |
| **Fargate (ECS/EKS)** | Containers without node management; steady or bursty services | Need GPU/special kernels or per-second bin-packing economics |
| **ECS on EC2** | Container orchestration, AWS-native control plane, cost control via reserved capacity | You need K8s API/ecosystem portability |
| **EKS** | You require the Kubernetes API/ecosystem or multi-cloud portability (see `kubernetes.md`) | A simpler ECS/Fargate setup suffices — EKS adds operational overhead |
| **EC2** | Full OS control, licensing, legacy lift-and-shift, GPU/HPC | A managed/serverless option fits — prefer it |

Prefer **Graviton (ARM64)** across Lambda/Fargate/EC2/RDS for ~20% better price-performance. Use **Spot/Fargate Spot** for fault-tolerant, interruptible work.

### B. Storage

| Service | Use for |
|---------|---------|
| **S3** | Object storage, data lakes, static assets, backups; lifecycle to IA/Glacier; default to SSE-KMS, versioning, Block Public Access |
| **EBS** | Block storage for a single EC2 instance (gp3 default; provisioned IOPS for databases) |
| **EFS** | Shared POSIX file system across many instances/containers; lifecycle to IA |
| **FSx** | Windows/Lustre/NetApp/OpenZFS workloads needing specific file semantics |

### C. Databases

| Service | Use for |
|---------|---------|
| **DynamoDB** | Serverless key-value/document, single-digit-ms at any scale, on-demand billing; single-table design; PITR + streams |
| **Aurora (Serverless v2)** | Relational at scale, MySQL/Postgres-compatible, auto-scaling, multi-AZ |
| **RDS** | Standard managed relational (Postgres/MySQL/etc.) where Aurora isn't needed |
| **ElastiCache (Redis/Valkey/Memcached)** | Caching, sessions, rate limiting |
| Purpose-built | Neptune (graph), OpenSearch (search/logs), Timestream (time-series), Keyspaces (Cassandra) — match the access pattern, don't force RDS |

### D. Networking & Edge

| Service | Use for |
|---------|---------|
| **VPC** | Network isolation; workloads in **private subnets**, NAT for egress, **VPC endpoints** (Gateway for S3/DynamoDB, Interface/PrivateLink for others) to keep traffic off the internet |
| **ALB** | HTTP/HTTPS L7 routing, host/path rules, OIDC auth, target groups |
| **NLB** | L4, ultra-low latency, static IPs, TCP/UDP, PrivateLink front |
| **API Gateway** | Managed REST/HTTP/WebSocket APIs; HTTP API is cheaper/faster — use REST only for API keys, usage plans, request validation, WAF, or caching |
| **CloudFront** | CDN, edge TLS, WAF attachment, S3/ALB origins, edge functions |
| **Route 53** | DNS, health checks, latency/geo/weighted routing, failover |

Front internet-facing apps with **WAF + Shield**; security groups are stateful allow-lists, NACLs are stateless subnet guards.

### E. Messaging & Orchestration

| Service | Use for | Not for |
|---------|---------|---------|
| **SQS** | Decoupling, work queues, buffering; always long-poll; pair every queue with a **DLQ** (FIFO for strict ordering/dedup) | Pub/sub fan-out, ordered streaming replay |
| **SNS** | Pub/sub fan-out to many subscribers (SQS/Lambda/HTTP/email); filter policies | Durable retention/replay |
| **EventBridge** | Event bus with rich pattern matching, schema registry, scheduler, SaaS/cross-account routing | Highest-throughput streaming (use Kinesis) |
| **Kinesis / MSK** | High-volume ordered streaming, multiple consumers replaying the same data | Simple decoupling (SQS is simpler/cheaper) |
| **Step Functions** | Orchestrating multi-step workflows with retries/catch/parallel; Standard for long/exactly-once, Express for high-volume short | Single-step glue (just call the service) |

> Implementation code for these services (handlers, repositories, SDK calls) belongs in the **language guide** (e.g. `python.md`, `typescript.md`, `go.md`) using its boto3/SDK idioms — this guide names the service and its fit, not the code.

---

## 5. Cost Optimization

This guide owns AWS cost discipline. Optimize structurally, then continuously.

- **Attribute then optimize**: enforce tagging (§6) so Cost Explorer / CUR can split spend by team, env, and product. You cannot optimize what you cannot attribute.
- **Right-size & right-purchase**: use Compute Optimizer; buy **Savings Plans** (Compute SP for flexibility) for steady baseline, **Spot** for interruptible, on-demand only for spiky remainder. Graviton/ARM64 for ~20% savings.
- **Storage tiering**: S3 Lifecycle → Standard-IA → Glacier/Deep Archive, or **S3 Intelligent-Tiering** for unknown access patterns; delete incomplete multipart uploads and old object versions; gp3 over gp2 for EBS.
- **Serverless economics**: DynamoDB on-demand vs provisioned (with auto-scaling) by traffic shape; Lambda memory right-sizing (more memory = more CPU, often cheaper per request); kill idle NAT gateways/EIPs/unattached EBS.
- **Guardrails (AWS-COST-02)**: AWS Budgets with alerts + **Cost Anomaly Detection**; review the Cost Optimization Hub.

---

## 6. Tagging Strategy

Tags drive cost attribution (§5), access control (ABAC), automation, and inventory. Define them once in IaC and enforce with AWS Config `required-tags` + Organizations **Tag Policies**.

Mandatory tag set (AWS-COST-01):

| Tag | Example | Purpose |
|-----|---------|---------|
| `Environment` | `prod` / `staging` / `dev` | env separation, cost split |
| `Owner` | `team-payments` | accountability |
| `CostCenter` | `CC-1042` | chargeback |
| `Application` | `checkout-api` | grouping |
| `ManagedBy` | `cdk` / `terraform` | drift / ClickOps detection |
| `DataClassification` | `pii` / `internal` / `public` | security & compliance |

Use a consistent case convention; apply tags at the stack/module level so every child resource inherits them.

---

## 7. Multi-Account & Organizations

Isolate blast radius and simplify guardrails by separating workloads into accounts, not just VPCs.

- **AWS Organizations + Control Tower**: a landing zone with OUs (e.g. `Security`, `Infrastructure`, `Workloads/Prod`, `Workloads/NonProd`, `Sandbox`). One account per workload × environment is the strong default.
- **SCPs (Service Control Policies)**: org-level guardrails that cap maximum permissions — deny disabling CloudTrail/GuardDuty, deny non-approved regions, deny root access-key creation. SCPs bound IAM; they never grant.
- **Identity**: centralize human access in **IAM Identity Center (SSO)** with permission sets federated to an IdP — no per-account IAM users. Cross-account access uses assumed roles.
- **Centralized security & logging**: dedicated `Security` and `Log Archive` accounts aggregate CloudTrail, Config, GuardDuty, and Security Hub findings org-wide.
- **Networking**: share connectivity via **Transit Gateway** / VPC sharing from a network account; resource sharing via RAM.

---

## 8. IAM Least Privilege

This guide owns the AWS IAM least-privilege idiom; KMS/secrets/crypto *policy* is owned by [`secure-coding.md`](guides://secure-coding.md).

- **Roles over users**: workloads assume roles for short-lived STS credentials (Lambda/ECS task roles, EC2 instance profiles, **EKS IRSA / Pod Identity**). Reserve IAM users only for break-glass; never for applications.
- **Scope tightly (AWS-IAM-01)**: name explicit actions and resource ARNs; add `Condition` keys (e.g. `aws:SourceVpce`, `aws:RequestedRegion`, `aws:PrincipalTag`). Prefer customer-managed policies over `*FullAccess`.
- **Permission boundaries (AWS-IAM-03)**: cap what delegated/CI-created roles can ever grant.
- **Validate**: IAM Access Analyzer (policy validation + external-access findings) in CI; generate least-privilege policies from CloudTrail access activity.

```json
// ✅ Least privilege: specific actions, specific ARN, conditioned
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "ReadAppObjects",
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:PutObject"],
    "Resource": "arn:aws:s3:::my-app-bucket/${aws:PrincipalTag/team}/*",
    "Condition": { "Bool": { "aws:SecureTransport": "true" } }
  }]
}
// ❌ Forbidden (AWS-IAM-01): {"Effect":"Allow","Action":"*","Resource":"*"}
```

When using CDK/SAM, prefer **grant helpers** (`table.grantReadData(fn)`) which synthesize a scoped policy automatically instead of hand-writing `*`.

---

## 9. Infrastructure as Code

All infrastructure is code (AWS-ARCH-01). The IaC *workflow, state, review, and policy-scanning* rules are owned by [`terraform.md`](guides://terraform.md) — do not restate them. AWS-native options:

| Tool | Use when |
|------|----------|
| **Terraform / OpenTofu** | Multi-cloud, large module ecosystem, existing TF estate (see `terraform.md`) |
| **AWS CDK (v2)** | Type-safe constructs in TS/Python; grant helpers; rich testing (`Template.fromStack` assertions) |
| **CloudFormation** | Declarative, no extra toolchain; the substrate CDK/SAM compile to |
| **SAM** | Serverless-focused (Lambda/API/Step Functions) with local emulation (`sam local`) |

Containers: build/scan per [`dockerfile.md`](guides://dockerfile.md), push to **ECR** with image scanning enabled. Config & secrets: SSM Parameter Store + Secrets Manager per [`env-config.md`](guides://env-config.md).

Infra is **test-first** (see `tdd.md`): assert resources/properties with CDK `Template`/`Match` or `cfn-lint`/`cfn-guard`, and snapshot-test to catch unintended drift. Scan every template with `checkov`/`trivy config` (AWS-SEC-04) in CI.

---

## 10. Security & Observability Bindings

Generic policy lives in the owners; these are the AWS service bindings.

- **Security (see `secure-coding.md`)**: GuardDuty (threat detection), Security Hub (aggregated findings/standards), Inspector (EC2/Lambda/ECR CVE scanning), Config (compliance rules), Access Analyzer (unintended access), Macie (PII in S3). KMS for encryption keys; Secrets Manager with rotation. Run `prowler aws --severity critical high` for posture audits.
- **Observability (see `observability.md`)**: CloudWatch metrics/alarms/Logs (+ Logs Insights), **X-Ray** or OpenTelemetry via ADOT for tracing, CloudTrail for the audit trail (AWS-OBS-01). Alarm on errors, p99 latency, saturation, and DLQ depth; route to SNS/PagerDuty. Embed correlation IDs to trace requests across services.

---

## 11. Quick Reference

```bash
# Identity & inventory
aws sts get-caller-identity
aws resourcegroupstaggingapi get-resources --tag-filters Key=Environment,Values=prod

# IaC
cdk diff && cdk deploy --context env=prod      # CDK
sam build && sam deploy --guided               # SAM
aws cloudformation detect-stack-drift --stack-name my-stack

# Security & cost
checkov -d .                                   # AWS-SEC-04
prowler aws --severity critical high           # posture audit
aws budgets describe-budgets --account-id <id>  # AWS-COST-02

# Operate
aws logs tail /aws/lambda/my-func --follow
aws ssm get-parameter --name /app/prod/db/host --with-decryption
```

---

## 12. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] AWS-IAM-01 — no wildcard action+resource; policies scoped to ARNs + conditions
- [ ] AWS-IAM-02 — workloads use roles/STS, no static access keys
- [ ] AWS-IAM-03 — permission boundaries on delegated/CI roles
- [ ] AWS-SEC-01 — encryption at rest (KMS) and in transit (TLS); S3 deny non-TLS
- [ ] AWS-SEC-02 — S3 Block Public Access on (all 4 flags)
- [ ] AWS-SEC-03 — secrets in Secrets Manager / SSM SecureString, none plaintext
- [ ] AWS-SEC-04 — IaC scan (checkov/trivy) 0 high/critical
- [ ] AWS-OBS-01 — multi-region CloudTrail enabled
- [ ] AWS-OBS-02 — metrics/traces emitted; alarms on error/latency/saturation/DLQ
- [ ] AWS-ARCH-01 — all infra is code; no drift, no ClickOps
- [ ] AWS-ARCH-02 — production multi-AZ; async paths have DLQs
- [ ] AWS-COST-01 — mandatory tag set on every resource
- [ ] AWS-COST-02 — Budgets + Cost Anomaly Detection configured
- [ ] Agent validated against the six Well-Architected pillars (§3) and recorded trade-offs

---
**End of AWS Guidelines**
