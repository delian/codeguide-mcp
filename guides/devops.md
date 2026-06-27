# DevOps Engineering Guidelines
The umbrella guide for DevOps culture and practice: CALMS, IaC principles, automation philosophy, environment promotion, SRE/DORA, and incident management. Technology-agnostic; binds to the deep guides for pipelines, infra, and observability.

---
name: devops
title: DevOps Engineering Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: []
requires: []
recommends:
  - ci-cd
  - kubernetes
  - terraform
  - observability
  - secure-coding
  - git
provides:
  - devops-culture
  - dora-metrics
  - iac-principles
  - incident-management
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns DevOps *culture and practice*; the *mechanics* of pipelines, infra, and monitoring live in their own guides.

---

## 0. Prerequisites & References

This is the overview/umbrella guide. It defines the *practices and culture*; the deep mechanics are owned elsewhere. Fetch the deep guide whenever the task touches its concern.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`ci-cd.md`](guides://ci-cd.md) — pipeline stages, quality gates, caching, artifacts, build provenance. *(This guide owns the promotion/gating policy; ci-cd owns the pipeline implementation.)*
> - [`kubernetes.md`](guides://kubernetes.md) · [`docker-compose.md`](guides://docker-compose.md) · [`dockerfile.md`](guides://dockerfile.md) — container orchestration, local stacks, image build standards.
> - [`terraform.md`](guides://terraform.md) — IaC tool mechanics, state, modules, drift, native tests. *(This guide owns IaC *principles*; terraform owns the tool.)*
> - [`observability.md`](guides://observability.md) — metrics/logs/traces, RED/USE, alerting, SLOs. *(Also: [`logging.md`](guides://logging.md).)*
> - [`secure-coding.md`](guides://secure-coding.md) — DevSecOps, supply chain (SLSA/SBOM), secrets, CVE/scan policy.
> - [`git.md`](guides://git.md) — branching, trunk, signed commits/tags; the basis for GitOps.

> 📎 **SEE ALSO:** [`env-config.md`](guides://env-config.md) · [`feature-flags.md`](guides://feature-flags.md) · [`semver.md`](guides://semver.md) · [`tdd.md`](guides://tdd.md) · [`microservices.md`](guides://microservices.md) · [`mlops.md`](guides://mlops.md) · [`adr.md`](guides://adr.md)

---

## 1. Core Philosophies: CALMS

DevOps is a culture, not a toolchain. The **CALMS** model (Culture, Automation, Lean, Measurement, Sharing) is the lens; everything mechanical is delegated to the guides in §0.

- **C**ulture: shared ownership of delivery and operations ("you build it, you run it"). No throw-over-the-wall handoffs; dev and ops share one backlog, one set of goals, and one definition of done. Blameless by default (see §6).
- **A**utomation: if a human does it twice, automate it — manual processes are latent bugs. Toil is tracked and budgeted down. The *implementation* of that automation lives in [`ci-cd.md`](guides://ci-cd.md), [`terraform.md`](guides://terraform.md), etc.
- **L**ean: optimize the whole flow, not local stages. Small batches, short lead time, work-in-progress limits, fast feedback. Reduce handoffs and queue time, not just execution time.
- **M**easurement: decisions are data-driven. The canonical delivery metrics are **DORA** (§5); operational health is measured via SLIs/SLOs/error budgets (owned by [`observability.md`](guides://observability.md)).
- **S**haring: golden paths, internal platforms, runbooks, and postmortems are shared assets. Knowledge is documented and discoverable, not tribal.

**Cross-cutting principles this culture rests on** (each owned elsewhere — do not re-derive):

- **Everything as code & GitOps**: Git is the single source of truth for app *and* infra; the running state converges to the declared state (basis: [`git.md`](guides://git.md); applied in [`terraform.md`](guides://terraform.md) / [`kubernetes.md`](guides://kubernetes.md)).
- **Immutable infrastructure**: never patch in place — replace with a new, tested artifact. Rollback is "deploy the previous artifact" (see §4).
- **Idempotency**: every operation yields the same result regardless of how often it runs.
- **Shift left**: testing, security, and policy checks run as early as possible (test-first: [`tdd.md`](guides://tdd.md); security: [`secure-coding.md`](guides://secure-coding.md)).
- **Build once, promote many**: one immutable artifact flows through every environment; config is injected at deploy time (see §3).
- **Simplicity**: prefer boring, proven technology; guardrails over gates; complexity is the enemy of reliability.

**Verified Delivery**: Agent-generated infrastructure and delivery automation MUST satisfy every gate in §2 — verified with the tool owned by the relevant deep guide — before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `DO-<TOPIC>-<NN>`. Rows that bind a rule owned by another guide cite that owner and defer the *mechanism* to it; this table asserts the *practice* must exist and be verifiable.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| DO-AUTO-01 | No production change MAY be applied by hand; all change flows through a pipeline (see `ci-cd.md`) | Audit deploy logs for out-of-band changes | 0 manual prod mutations |
| DO-AUTO-02 | Every deployment MUST be idempotent and re-runnable | Re-run deploy; diff state | no-op on second run |
| DO-IAC-01 | All infrastructure MUST be declared as version-controlled code; no clickops (see `terraform.md`) | Drift check (`terraform plan -detailed-exitcode`) | exit 0 (no drift) |
| DO-IAC-02 | IaC MUST be tested before apply (see `tdd.md`, `terraform.md`) | `terraform test` / Terratest in CI | exit 0 |
| DO-PROMOTE-01 | The SAME artifact MUST be promoted across envs; never rebuild per env | Compare artifact digest dev→prod | identical digest |
| DO-PROMOTE-02 | Dev/staging/prod MUST be structurally identical, differing only by config (see `env-config.md`) | Diff env topology / config keys | same keys, no struct diff |
| DO-GATE-01 | No artifact advances without passing its stage's gates (see `ci-cd.md`) | CI required-checks status | all required green |
| DO-SEC-01 | No hardcoded secrets; a secret manager / OIDC MUST be used (see `secure-coding.md`) | `gitleaks detect --source .` | 0 findings |
| DO-SEC-02 | Supply-chain controls (signed artifacts, SBOM, SCA) MUST gate deploy (see `secure-coding.md`) | CI security stage status | 0 critical/high |
| DO-DEPLOY-01 | Every deployment MUST have an automated rollback path | Trigger rollback in staging | restores last-good <5 min |
| DO-DEPLOY-02 | Prod deploy MUST be progressive (rolling/blue-green/canary), not big-bang | Inspect rollout strategy | non-recreate strategy |
| DO-OBS-01 | Every service MUST emit metrics/logs/traces and expose health endpoints (see `observability.md`) | Probe `/health`, `/ready`, `/metrics` | 200 / scrapeable |
| DO-OBS-02 | Alerts MUST fire on symptoms (SLO burn), each with an owner + runbook link (see `observability.md`) | Review alert rules | 100% have runbook + owner |
| DO-INC-01 | Every incident MUST get a regression test or monitor BEFORE the fix ships (see `tdd.md`) | Link incident → test/alert PR | test/alert exists |
| DO-INC-02 | Every SEV1/SEV2 MUST produce a blameless postmortem with owned action items | Postmortem doc exists | published, actions tracked |
| DO-DORA-01 | The four DORA metrics MUST be tracked and visible | Delivery dashboard exists | all four reported |
| DO-DOC-01 | Every service MUST have a runbook with on-call + alert response (see §6) | `RUNBOOK.md` present & current | exists, links resolve |

> **Forbidden**: manual production changes outside the pipeline; rebuilding an artifact per environment; deploying without a rollback path; using `:latest` for prod images (see `dockerfile.md`); fixing an incident without a regression test/monitor first (violates `tdd.md`); disabling a security gate to ship (violates `secure-coding.md`).

---

## 3. Environments, Promotion & Configuration

This is owned by this guide: how artifacts move toward production. (Config *layering & secret handling* policy is owned by [`env-config.md`](guides://env-config.md); the *pipeline* that executes promotion is owned by [`ci-cd.md`](guides://ci-cd.md).)

### A. Build once, promote many

```
Source → Build → IMMUTABLE ARTIFACT (digest-pinned, signed)
                      │
                      ├─→ Dev        (auto)   → smoke tests
                      ├─→ Staging    (auto)   → integration + E2E
                      └─→ Production (gated)   → progressive rollout + auto-canary analysis
```

- The **same** artifact (same digest) is deployed everywhere. Never rebuild for a different environment — a rebuild is a different, untested artifact.
- Configuration is **injected at deploy time**, never baked at build time.
- Promotion is automatic up to staging; production is a deliberate gate (manual approval and/or automated canary analysis).

### B. Environment parity

Dev, staging, and prod are **structurally identical** — same images, same topology, same config *keys*, same security controls, same dashboards, same pipeline. They differ only via configuration: resource sizing, replica counts, URLs, secret *values*, data realism, and access scope. This is what kills "works on my machine."

### C. Configuration

Resolution order, highest priority first: runtime overrides ([`feature-flags.md`](guides://feature-flags.md)) → env vars set by the deployer → per-env files → shipped defaults → in-code defaults (non-sensitive only). Validate config at startup and **fail fast**. Full policy (typed config, secret separation, never-commit-secrets) is owned by [`env-config.md`](guides://env-config.md) and [`secure-coding.md`](guides://secure-coding.md).

---

## 4. Infrastructure-as-Code Principles & Deployment

IaC *principles* are owned here; the *tool* (HCL, state, providers, modules, native tests, drift mechanics) is owned by [`terraform.md`](guides://terraform.md) (and per-cloud guides [`aws.md`](guides://aws.md) · [`gcp.md`](guides://gcp.md) · [`azure.md`](guides://azure.md)).

### A. The IaC non-negotiables

| Principle | Why it matters |
|---|---|
| **Version-controlled** | Same review/PR process as app code; full history and blame. |
| **Declarative & idempotent** | Describe desired state; apply converges and re-applies safely. |
| **Modular & parameterized** | Reusable modules; env values via variables, never hardcoded. |
| **Tested** | Plans validated and asserted before apply (see [`tdd.md`](guides://tdd.md)). |
| **Drift-detected** | Scheduled checks surface out-of-band changes; reconcile to declared state. |
| **State secured** | Remote state with locking, encryption, and versioning. |

> No manual changes. No clickops. The console is read-only; changes go through code review and CI. The *commands* for plan/apply/drift live in [`terraform.md`](guides://terraform.md).

### B. Deployment strategies (owned here)

Choosing *how* to release is a practice decision; the orchestrator-specific manifests live in [`kubernetes.md`](guides://kubernetes.md).

| Strategy | Risk | Use when |
|---|---|---|
| **Rolling** | Low | Default for stateless services that tolerate brief dual-version traffic. |
| **Blue-Green** | Very low | Zero-downtime cutover needed; instant rollback by switching traffic. |
| **Canary** | Very low | High-risk or high-traffic change; shift 5→25→50→100% with automated metric analysis and auto-rollback on SLO breach. |
| **Feature flag** | Very low | Deploy dark, enable incrementally; decouples deploy from release ([`feature-flags.md`](guides://feature-flags.md)). |
| **Recreate** | High | Dev/test only — never production. |

**Database changes** use the **expand-contract** pattern: add the new shape (expand), migrate and dual-write, switch reads, then remove the old shape (contract) in a *later* release. Never drop a column in the same release that removes its code — this keeps every step backward-compatible and rollback-safe.

### C. Rollback (mandatory)

Every deployment has an **automated** rollback path. Auto-rollback triggers on: health-check failures, error-rate or latency SLO breach, deployment timeout, or smoke-test failure. Rollback restores the last known-good artifact, completes within minutes, needs no human, notifies the team, and preserves logs for investigation. Because infrastructure is immutable and artifacts are promoted (not rebuilt), rollback is simply re-deploying the previous artifact digest.

---

## 5. Measurement: SRE & DORA

What gets measured gets improved. Operational health (SLIs/SLOs/error budgets, alerting) is owned by [`observability.md`](guides://observability.md); the *delivery* metrics below are owned here.

### A. DORA — the four keys

| Metric | What it measures | Elite band (reference) |
|---|---|---|
| **Deployment Frequency** | How often you deploy to prod | On-demand (multiple/day) |
| **Lead Time for Changes** | Commit → running in prod | < 1 day |
| **Change Failure Rate** | % of deploys causing degraded service | 0–15% |
| **Failed Deployment Recovery Time** | Time to restore after a failed deploy | < 1 hour |

The first two measure **throughput**; the last two measure **stability**. High performers improve both simultaneously — they are not a trade-off. Track all four on a delivery dashboard (DO-DORA-01). DORA's "reliability" addition is captured by SLOs in [`observability.md`](guides://observability.md).

### B. SRE concepts (the reliability discipline)

- **SLI / SLO / Error budget**: define what "reliable enough" means and spend the remaining unreliability deliberately. An exhausted error budget freezes risky changes; a healthy budget licenses faster shipping. *(SLI/SLO definition mechanics: [`observability.md`](guides://observability.md).)*
- **Toil budget**: cap manual operational work (target ≤ ~50% of an SRE's time); the surplus funds automation. Toil is measured, not endured.
- **Embrace risk**: 100% reliability is the wrong target — it costs more than it returns and slows delivery. The SLO sets the right, deliberate level.
- **Capacity & demand**: plan against measured growth; autoscale on saturation signals, not vanity CPU.

---

## 6. Incident Management & On-Call

Owned by this guide. (The *detection* side — alerts, dashboards — is owned by [`observability.md`](guides://observability.md).)

### A. Severity & response

| Sev | Meaning | Response |
|---|---|---|
| **SEV1** | Total outage, data loss/corruption, security breach, imminent SLA breach | Page immediately, all-hands |
| **SEV2** | Major degradation (>2× baseline) or partial outage (>10% users) | Page during/near business hours |
| **SEV3** | Non-critical component unhealthy, resource nearing limits | Ticket, next business day |
| **SEV4** | Informational (deploy done, cert renewal upcoming) | No action |

### B. Incident workflow

1. **Detect** (alert / report / monitor).
2. **Triage** — assign severity and an incident commander.
3. **Mitigate first** — restore service before root-causing (rollback, scale, traffic shift, feature-flag kill).
4. **Write the regression test / monitor BEFORE the fix** — a test that reproduces the failure and/or an alert that would have caught it sooner (DO-INC-01; test-first is owned by [`tdd.md`](guides://tdd.md)).
5. **Fix the root cause.**
6. **Verify** — regression test passes, alert fires correctly.
7. **Blameless postmortem** (SEV1/SEV2 always) — timeline, contributing factors, action items with owners and due dates (DO-INC-02).

### C. On-call & runbooks

- Humane rotations: bounded shift length, sane paging volume, follow-the-sun where possible, comp/time-off for night pages. Alert fatigue is a tracked metric — every page must be actionable.
- **Every service ships a `RUNBOOK.md`** (DO-DOC-01): owner + on-call rotation link, dashboard/log links, and per-alert response steps (diagnose → mitigate → escalate). Every alert links to its runbook entry (DO-OBS-02).
- **Blameless culture**: postmortems target systems and process, never individuals. The goal is to make the same failure impossible (or auto-detected), not to assign fault.

---

## 7. Quick Reference

Practice → owner guide for the mechanics:

| Need | Owner guide |
|---|---|
| Pipeline stages, gates, caching, artifacts | [`ci-cd.md`](guides://ci-cd.md) |
| IaC tool, state, modules, drift, tests | [`terraform.md`](guides://terraform.md) |
| Container orchestration / rollout manifests | [`kubernetes.md`](guides://kubernetes.md) |
| Local multi-service stacks | [`docker-compose.md`](guides://docker-compose.md) |
| Image build, multi-stage, non-root, scanning | [`dockerfile.md`](guides://dockerfile.md) |
| Metrics/logs/traces, RED/USE, SLOs, alerting | [`observability.md`](guides://observability.md) · [`logging.md`](guides://logging.md) |
| DevSecOps, secrets, SBOM/SLSA, CVE policy | [`secure-coding.md`](guides://secure-coding.md) |
| Branching, signed commits/tags, GitOps base | [`git.md`](guides://git.md) |
| Config layering & secret separation | [`env-config.md`](guides://env-config.md) |
| Dark launches / progressive rollout flags | [`feature-flags.md`](guides://feature-flags.md) |
| Release versioning | [`semver.md`](guides://semver.md) |

External standards: [DORA](https://dora.dev/) · [Google SRE Book](https://sre.google/books/) · [The Twelve-Factor App](https://12factor.net/) · [OpenGitOps](https://opengitops.dev/) · [SLSA](https://slsa.dev/).

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements here.

- [ ] DO-AUTO-01 — no manual production changes; all change via pipeline
- [ ] DO-AUTO-02 — deployments idempotent / re-runnable
- [ ] DO-IAC-01 — all infra is version-controlled code, no clickops; no drift
- [ ] DO-IAC-02 — IaC tested before apply
- [ ] DO-PROMOTE-01 — same artifact digest promoted across envs
- [ ] DO-PROMOTE-02 — envs structurally identical, differ only by config
- [ ] DO-GATE-01 — all required CI gates green before advance
- [ ] DO-SEC-01 — no hardcoded secrets (gitleaks clean)
- [ ] DO-SEC-02 — supply-chain controls gate deploy (0 critical/high)
- [ ] DO-DEPLOY-01 — automated rollback path verified
- [ ] DO-DEPLOY-02 — progressive (non-recreate) prod rollout
- [ ] DO-OBS-01 — metrics/logs/traces + health endpoints present
- [ ] DO-OBS-02 — alerts on symptoms, each with owner + runbook link
- [ ] DO-INC-01 — regression test/monitor before every incident fix
- [ ] DO-INC-02 — blameless postmortem for every SEV1/SEV2
- [ ] DO-DORA-01 — four DORA metrics tracked and visible
- [ ] DO-DOC-01 — runbook present, current, links resolve

---
**End of DevOps Engineering Guidelines**
