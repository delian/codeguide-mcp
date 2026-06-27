# CI/CD Pipeline Guidelines
Provider-agnostic standards for continuous integration and delivery: pipeline stages, quality gates, artifact management, deployment strategies (blue-green/canary/rolling), and rollback.

---
name: ci-cd
title: CI/CD Pipeline Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: []
requires: []
recommends:
  - git
  - tdd
  - secure-coding
  - semver
  - observability
  - pre-commit
provides:
  - pipeline-stages
  - quality-gates
  - deployment-strategies
  - rollback
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns the **pipeline-agnostic** principles of CI/CD. Platform syntax (GitHub Actions, GitLab CI, Jenkins, Azure DevOps) lives in the provider guides; this guide states the principle once and points there for the YAML.

---

## 0. Prerequisites & References

This guide defines *what* a pipeline must do and *how* changes flow to production. The concerns below define rules that pipeline stages enforce — fetch them when the task touches them.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`git.md`](guides://git.md) — branching/trigger model the pipeline reacts to (push, PR, tag).
> - [`tdd.md`](guides://tdd.md) — test-first, coverage; the pipeline only *gates* on these, it does not define them.
> - [`secure-coding.md`](guides://secure-coding.md) — SAST/DAST/SCA, secrets, supply chain; the pipeline runs these scans as gates.
> - [`semver.md`](guides://semver.md) — release/tag versioning that drives artifact tags and release stages.
> - [`observability.md`](guides://observability.md) — deploy/DORA metrics, canary signal sources, alerting.
> - [`pre-commit.md`](guides://pre-commit.md) — the same gates run locally before push (shift-left).

> 📎 **SEE ALSO (platform syntax — pick the one you use):**
> - [`github.md`](guides://github.md) · [`gitlab.md`](guides://gitlab.md) · [`jenkins.md`](guides://jenkins.md) · [`azuredevops.md`](guides://azuredevops.md)
> - [`dockerfile.md`](guides://dockerfile.md) · [`kubernetes.md`](guides://kubernetes.md) · [`terraform.md`](guides://terraform.md) — artifact/build and deploy targets.
> - [`feature-flags.md`](guides://feature-flags.md) — decouple deploy from release. [`mlops.md`](guides://mlops.md) — CI/CD for models. [`e2e-testing.md`](guides://e2e-testing.md) — post-deploy verification.

---

## 1. Core Philosophies: CICD-FIRST

CI/CD-specific principles only. Test policy, security policy, and versioning come from §0.

- **C**ontinuous: every push to a tracked branch runs the full pipeline; `main`/trunk is always releasable. No long-lived broken builds.
- **I**dentical artifact: **build once, promote the same artifact** through every environment. Never rebuild per environment — that breaks reproducibility. Environment differences are injected as config (see `env-config.md`), not baked into the binary.
- **C**onsistent: the same pipeline definition and the same gates apply to every environment; staging differs from prod only in scale and config.
- **D**ecoupled deploy from release: deploying code and exposing a feature are separate acts — ship dark, flip a flag (see `feature-flags.md`).
- **F**ast feedback: fail fast, run cheap gates first, parallelize the rest; keep PR feedback under ~10 minutes so developers stay in flow.
- **A**utomated & reversible: every deploy is automated, observable, and has a tested, automated rollback. Manual production steps are forbidden.
- **P**ipeline-as-code: pipeline definitions are version-controlled, reviewed (see `code-review.md`), and reproducible — never click-configured in a UI.

**Verified Code**: a pipeline change MUST pass every gate in §2 and its definition MUST lint clean before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `CICD-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner. "Verify" methods are provider-agnostic — bind them to your platform's CLI/lint via the §0 SEE ALSO guides.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| CICD-STRUCT-01 | Pipeline MUST be defined as version-controlled code, reviewed before merge | file in repo + review (see `code-review.md`) | no UI-only config |
| CICD-STRUCT-02 | Pipeline definition MUST lint clean | platform linter (`actionlint`/`gitlab-ci-lint`/`jenkins-lint`) | exit 0 |
| CICD-STRUCT-03 | Every job MUST set an explicit timeout | grep for timeout on each job | no job without timeout |
| CICD-GATE-01 | Deploy jobs MUST depend on (`needs`) all test & security jobs | inspect job dependency graph | no deploy without test+security upstream |
| CICD-GATE-02 | A failing gate MUST block the pipeline (no `allow_failure` on required gates) | inspect job config | required gates non-bypassable |
| CICD-GATE-03 | Tests MUST pass before deploy (gate only; policy in `tdd.md`) | pipeline run | test stage green, 0 skips |
| CICD-GATE-04 | Coverage gate MUST be enforced in CI (threshold owned by `tdd.md`) | coverage step exit code | meets project threshold |
| CICD-SEC-01 | SAST + dependency/SCA scan MUST run and block on high/critical (see `secure-coding.md`) | scan step exit code | 0 high/critical |
| CICD-SEC-02 | Built container/artifact images MUST be vulnerability-scanned (see `secure-coding.md`) | image scan (e.g. Trivy/Grype) | 0 high/critical |
| CICD-SEC-03 | Pipelines MUST use secretless auth (OIDC/workload identity) or a vault; no plaintext secrets in config/logs (see `secure-coding.md`) | review + secret scan | no embedded/echoed secrets |
| CICD-SEC-04 | Jobs MUST run with least-privilege, scoped tokens | review job permissions | minimal scopes only |
| CICD-ART-01 | The SAME artifact MUST be promoted across environments (build once) | artifact digest equal staging→prod | identical digest |
| CICD-ART-02 | Artifacts MUST be immutable and uniquely versioned (see `semver.md`) | tag = semver/commit SHA | no mutable/overwritten tags |
| CICD-DEP-01 | Production deploy MUST use a controlled strategy (blue-green/canary/rolling), not in-place restart | inspect deploy job | named strategy present |
| CICD-DEP-02 | Production environment MUST require approval and/or wait timer | environment protection rules | gated promotion |
| CICD-ROLL-01 | An automated rollback MUST exist and be exercised | rollback job + drill record | rollback returns to last-good |
| CICD-ROLL-02 | Deploys MUST gate on a health/smoke check; failure auto-rolls-back | post-deploy check + `on failure` rollback | unhealthy → reverted |
| CICD-OBS-01 | Deploy events & DORA metrics MUST be emitted (see `observability.md`) | metrics/notification step | event recorded per deploy |

> **Forbidden**: deploying an artifact different from the one tested (CICD-ART-01); a deploy job with no test/security upstream; `allow_failure: true` on a required security/test gate; long-lived cloud credentials in CI variables; manual, undocumented production steps; mutable image tags reused across releases.

---

## 3. Pipeline Stages (owned)

A canonical pipeline is a directed acyclic graph of stages, ordered cheapest-and-fastest first so failures surface early. The *policy* enforced by each gate lives in its §0 owner; this guide owns the **shape and ordering**.

| # | Stage | Purpose | Gate (see §2) | Owner of the rule |
|---|-------|---------|---------------|-------------------|
| 1 | **Validate** | Lint, format, type-check, pipeline-lint | CICD-STRUCT-02 | language guide / `pre-commit.md` |
| 2 | **Build** | Compile/package **once**; emit the artifact | CICD-ART-01/02 | `dockerfile.md`, language guide |
| 3 | **Test** | Unit + integration (parallel/sharded) | CICD-GATE-03/04 | `tdd.md` |
| 4 | **Security scan** | SAST, SCA/dependency audit, image scan, secret scan | CICD-SEC-01/02/03 | `secure-coding.md` |
| 5 | **Publish** | Push immutable artifact to registry | CICD-ART-02 | `semver.md` |
| 6 | **Deploy staging** | Promote the published artifact | CICD-ART-01 | `env-config.md` |
| 7 | **Verify** | Smoke/E2E/DAST against staging | CICD-ROLL-02 | `e2e-testing.md` |
| 8 | **Deploy production** | Approved, strategy-based rollout | CICD-DEP-01/02 | this guide §5 |
| 9 | **Observe** | Emit deploy event + DORA metrics; watch alerts | CICD-OBS-01 | `observability.md` |

Ordering principles:
- **Cheap before expensive**: lint (seconds) before unit tests (minutes) before E2E (tens of minutes). Don't pay for an E2E run a lint would have caught.
- **Fail fast on PRs, full gauntlet on merge**: PR pipelines run validate→test→security for tight feedback; merge/tag pipelines add publish→deploy→verify.
- **Parallelize independent work**: test sharding and independent scans run concurrently; `needs`/`depends_on` expresses only real data dependencies.
- **Build once, promote**: stages 6–8 download the stage-5 artifact by digest; they never rebuild.
- **Trigger model** (push / PR / tag / scheduled / manual dispatch) follows the branching strategy in [`git.md`](guides://git.md).

> Platform YAML for these stages: [`github.md`](guides://github.md), [`gitlab.md`](guides://gitlab.md), [`jenkins.md`](guides://jenkins.md), [`azuredevops.md`](guides://azuredevops.md). Don't hand-write provider syntax here — bind these stages to the provider's job/stage primitives.

---

## 4. Quality Gates (owned)

A **gate** is a binary pass/fail check that halts promotion. A gate is only meaningful if it is **non-bypassable** (CICD-GATE-02) and **upstream of deploy** (CICD-GATE-01).

Design rules:
- **Binary, not advisory.** A gate that warns is not a gate. `allow_failure`/`continue-on-error` is permitted only for genuinely informational steps, never for test/security gates.
- **Cite the owner, don't redefine.** The coverage threshold belongs to `tdd.md`; the CVE severity bar belongs to `secure-coding.md`. The pipeline imports those thresholds — it does not invent new ones here.
- **Shift left.** The same gates run locally via [`pre-commit.md`](guides://pre-commit.md) so failures are caught before push; CI is the authoritative re-run, not the first run.
- **One source of truth.** A gate command (e.g. `pytest --cov`) is defined once and called identically locally and in CI; divergence causes "passes locally, fails in CI".
- **Required vs. optional checks.** Mark gate jobs as *required status checks* on the protected branch so merges are mechanically blocked; non-gate jobs stay optional.

Standard gate set (each maps to a §2 ID): pipeline-lint → format/lint/type → unit+integration tests → coverage → SAST → dependency/SCA → image scan → secret scan → post-deploy smoke. Add domain gates (a11y, performance budgets, license compliance) by extending §2, never by loosening an existing gate.

---

## 5. Deployment Strategies (owned)

Choose a strategy by risk and infrastructure. **Every** strategy MUST pair with a health gate (CICD-ROLL-02) and an automated rollback (CICD-ROLL-01). Concrete orchestrator manifests live in [`kubernetes.md`](guides://kubernetes.md) / [`terraform.md`](guides://terraform.md); the strategy *semantics* are owned here.

### A. Rolling
Replace instances in batches, keeping the service available. Configure surge/unavailable so capacity never drops below SLA, and gate each batch on a readiness probe before proceeding.
- **Use when**: stateless services, backward-compatible changes.
- **Pros**: no extra capacity; simple. **Cons**: mixed versions serve traffic mid-rollout (requires N/N-1 compatibility); slower to fully revert.
- **Rollback**: roll the deployment back to the previous revision (`undo`), batch by batch.

### B. Blue-Green
Stand up a full **green** environment alongside live **blue**, run smoke tests against green, then switch the router/load-balancer atomically. Keep blue warm for instant fallback.
- **Use when**: you need instant, atomic cutover and instant rollback; can afford 2× capacity briefly.
- **Pros**: zero mixed-version window; rollback is a single traffic switch back to blue. **Cons**: double capacity; DB/schema must be compatible across both.
- **Rollback (CICD-ROLL-01)**: re-point the router to blue; no redeploy needed.

### C. Canary
Route a small percentage (e.g. 1–10%) of traffic to the new version, **watch real metrics** (error rate, latency, saturation — sourced per `observability.md`) for a bake period, then progressively promote or auto-abort.
- **Use when**: high-traffic, high-risk changes where real-user signal is the best test.
- **Pros**: smallest blast radius; data-driven promotion. **Cons**: needs traffic-splitting (service mesh/ingress) and solid metrics; slowest rollout.
- **Rollback**: shift the canary's traffic weight to 0 and remove the canary; the analysis gate triggers this automatically on threshold breach.

### Cross-cutting deployment rules
- **Decouple deploy from release** with [`feature-flags.md`](guides://feature-flags.md): ship the artifact dark, then ramp exposure independently of the rollout.
- **Backward-compatible schema migrations** (expand/contract): deploy code that tolerates both old and new schema before destructive migration, so any strategy can roll back without data loss.
- **Production promotion is gated** by approval/wait timers (CICD-DEP-02) configured as environment protection rules.

---

## 6. Artifact Management (owned)

The artifact (container image, package, binary, bundle) is the unit promoted through the pipeline.

- **Build once (CICD-ART-01).** Stage 2 produces the artifact; every later environment pulls *that* artifact by content digest. Rebuilding per environment is forbidden — it invalidates the tested→deployed guarantee.
- **Immutable & uniquely versioned (CICD-ART-02).** Tag by commit SHA and/or semver (see `semver.md`); never overwrite or reuse a tag. `latest` is a convenience pointer, never a deployment target.
- **Provenance & integrity.** Sign artifacts and generate an SBOM; verify signature before deploy (policy: `secure-coding.md`). Record which commit produced which artifact for auditability.
- **Retention.** Keep released artifacts per the retention/compliance policy; expire ephemeral PR-build artifacts quickly to control storage.
- **Cache, don't conflate.** Dependency/build caches speed builds but are NOT artifacts — a cache miss must never change the produced artifact.

---

## 7. Rollback (owned)

Rollback is a **first-class, automated, tested** capability — not an emergency improvisation.

- **Automated (CICD-ROLL-01).** A single action (job/command) returns production to the last-known-good artifact. No manual surgery.
- **Health-gated (CICD-ROLL-02).** Post-deploy smoke/health checks run automatically; on failure the pipeline triggers rollback (`on: failure`) without human latency.
- **Tested.** Exercise rollback in staging and in periodic game-days; an untested rollback is a liability. Record the drill (CICD-ROLL-01 gate).
- **Strategy-native** (§5): blue-green → re-point router to blue; canary → set new-version weight to 0; rolling → revert to the previous revision.
- **Forward-fix vs. rollback.** Prefer rollback to restore service fast; fix forward only when rollback is impossible (e.g. an already-applied irreversible migration — which is why migrations must be expand/contract, §5).
- **Data safety.** Because the same artifact and backward-compatible schema are used (§5/§6), rollback does not lose or corrupt data.

---

## 8. Quick Reference

```text
PR pipeline:     validate → test → security            (fast feedback, <~10 min)
Merge/tag:       …→ build(once) → publish → deploy-staging → verify → deploy-prod → observe
Promote rule:    pull artifact by DIGEST; never rebuild
Gate rule:       binary, non-bypassable, upstream of deploy
Deploy rule:     blue-green | canary | rolling  + health gate + auto-rollback
Secrets:         OIDC/workload identity or vault; never plaintext in config/logs
Release:         deploy ≠ release — flip a feature flag
```

Provider syntax: `github.md` · `gitlab.md` · `jenkins.md` · `azuredevops.md`.

---

## 9. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements here.

- [ ] CICD-STRUCT-01/02/03 — pipeline is reviewed code, lints clean, every job has a timeout
- [ ] CICD-GATE-01/02 — deploy depends on test+security; required gates are non-bypassable
- [ ] CICD-GATE-03/04 — tests pass with 0 skips; coverage gate enforced (see `tdd.md`)
- [ ] CICD-SEC-01/02 — SAST/SCA and image scan block on high/critical (see `secure-coding.md`)
- [ ] CICD-SEC-03/04 — secretless auth/vault, no leaked secrets, least-privilege tokens
- [ ] CICD-ART-01/02 — identical artifact promoted; immutable, uniquely versioned (see `semver.md`)
- [ ] CICD-DEP-01/02 — controlled deploy strategy; production approval/wait timer
- [ ] CICD-ROLL-01/02 — automated, tested rollback; health-gated with auto-revert
- [ ] CICD-OBS-01 — deploy events & DORA metrics emitted (see `observability.md`)
- [ ] Bound every stage/gate to the chosen platform via the §0 SEE ALSO guide

---
**End of CI/CD Pipeline Guidelines**
