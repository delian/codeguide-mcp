# GitLab Platform Guidelines
Mandatory standards for GitLab CI/CD: pipelines, runners, components, environments, merge trains, registries, and built-in security scanning. GitLab 17.x, .gitlab-ci.yml, GitLab Runner, CI/CD Components, Merge Trains, Auto DevOps.

---
name: gitlab
title: GitLab Platform Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: infra
tools: [gitlab@17, gitlab-runner@17, glab, gitlab-ci-components, merge-trains, auto-devops]
requires:
  - secure-coding
recommends:
  - ci-cd
  - git
  - dockerfile
  - kubernetes
provides:
  - gitlab-ci
  - gitlab-runners
  - merge-trains
  - gitlab-components
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to the GitLab platform — its CI/CD engine, runners, registries, and built-in scanners. Generic CI/CD *concepts* and git/MR *workflow* live in the references below.

---

## 0. Prerequisites & References

Fetch and apply these **before** authoring GitLab pipelines. Their rules are assumed here and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(GitLab binding: the built-in SAST/DAST/Dependency/Secret/Container scanners in §6 are the enforcement surface; severity gates and triage policy are owned there.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`ci-cd.md`](guides://ci-cd.md) — pipeline *philosophy* (stages, gates, fail-fast, caching strategy, deployment models). This guide only maps those concepts onto GitLab syntax.
> - [`git.md`](guides://git.md) — branch/MR workflow, protected branches, review, conventional commits. GitLab is the host; the workflow rules are owned there.
> - [`dockerfile.md`](guides://dockerfile.md) — image authoring (multi-stage, non-root, `.dockerignore`). This guide only covers pushing to the GitLab registry.
> - [`kubernetes.md`](guides://kubernetes.md) — cluster/deploy mechanics. GitLab binding is the **agent** (`agentk`) and environments.

> 📎 **SEE ALSO:** [`github.md`](guides://github.md) · [`semver.md`](guides://semver.md) · [`pre-commit.md`](guides://pre-commit.md) · [`observability.md`](guides://observability.md)

---

## 1. Core Philosophies: GITLAB-FIRST

GitLab-platform principles only. TDD, security policy, git workflow, and image authoring come from §0.

- **G**raph the pipeline: model job order with `needs:` (DAG), not just `stages`. Idle jobs that wait on an unrelated stage are a defect.
- **I**nclude, don't copy: shared CI lives in **CI/CD Components** (`include: component:`) or `include:` files, versioned and reused — never pasted between projects.
- **T**rusted variables: secrets are **masked + protected** CI/CD variables or external secrets (`secrets:` via Vault/OIDC), never inline in `.gitlab-ci.yml`.
- **L**east-privilege runners: pin jobs to runners by **tag**; use ephemeral executors; never run untrusted MR code on a privileged shared runner.
- **A**utomated gates: `rules:` decide *when* a job runs; merge is blocked by required pipeline + **merge trains** serialize integration.
- **B**uilt-in scanning: enable GitLab's native SAST/DAST/Dependency/Secret/Container scanners via templates/components — they emit the security reports GitLab gates on.

**Verified Code**: Agent-authored GitLab config MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `GL-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| GL-PIPE-01 | `.gitlab-ci.yml` MUST be syntactically valid | `glab ci lint` (or CI Lint API) | valid: true |
| GL-PIPE-02 | Pipelines MUST run on MRs, default branch, and tags via `workflow:rules` (avoid duplicate detached+branch pipelines) | inspect `workflow.rules` | no double pipelines |
| GL-PIPE-03 | Inter-job order SHOULD use `needs:` (DAG); long jobs MUST set explicit `timeout` and bounded `retry` | review YAML | DAG + timeouts set |
| GL-VAR-01 | No secrets in `.gitlab-ci.yml` or repo; sensitive CI/CD vars MUST be **masked + protected** (see `secure-coding.md`) | `gitlab-leaks`/Secret Detection + var settings | 0 inline secrets |
| GL-VAR-02 | Deploy/prod credentials MUST come from protected vars or `secrets:` (Vault/OIDC), scoped to protected branches/envs | review var protection | protected-only |
| GL-SEC-01 | SAST, Secret Detection & Dependency Scanning MUST be enabled (see `secure-coding.md`) | presence of templates/components + reports | reports produced |
| GL-SEC-02 | Container images MUST be scanned before deploy; build MUST gate on severity threshold (see `secure-coding.md`) | `container_scanning` report | 0 over threshold |
| GL-RUN-01 | Jobs MUST target runners by `tags:`; privileged/DinD runners MUST NOT run untrusted MR pipelines | review tags + runner config | tagged, isolated |
| GL-ENV-01 | Deployments MUST declare an `environment:` (name + url); prod MUST be `when: manual` or merge-train gated with `resource_group` | review deploy jobs | env + protection |
| GL-REG-01 | Images/packages MUST publish to the project's GitLab registry with immutable tags (SHA/semver), `latest` only on default branch | inspect build job | immutable tags |
| GL-CMP-01 | Reused CI MUST be a versioned Component or `include:` (pinned ref), not copy-paste | review `include:` | pinned refs |
| GL-MR-01 | Merge MUST be blocked until the required pipeline passes; integration uses merge trains where enabled (workflow: see `git.md`) | project merge settings | "pipelines must succeed" on |

> **Forbidden**: inline secrets or unmasked tokens; `latest`-only image tags; privileged DinD on shared untrusted runners; deploying without an `environment:`; copy-pasting pipeline blocks instead of a component; merging with a failing or skipped required pipeline.

---

## 3. Verification Protocol

Run before presenting GitLab config. Fix → re-run until green.

```bash
glab ci lint                          # GL-PIPE-01: validate .gitlab-ci.yml (server-side)
glab ci config compile                # expand includes/components; inspect the merged YAML
glab ci view / glab ci status         # GL-PIPE-02/03: observe pipeline graph + job timing
glab ci trace <job>                   # read a failing job's log
# Secret/var hygiene: confirm vars are Masked + Protected in Settings > CI/CD > Variables
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Repository & Pipeline Layout

GitLab discovers config by convention. Architectural/workflow *principles* are owned by `ci-cd.md` and `git.md`; below is only the GitLab mapping.

```
project/
├── .gitlab-ci.yml                    # root pipeline (or set a custom path in Settings)
├── .gitlab/
│   ├── ci/                           # included pipeline fragments (include: local:)
│   │   ├── test.yml  build.yml  deploy.yml
│   │   └── components/               # CI/CD Component definitions (templates/<name>/template.yml)
│   ├── agents/<name>/config.yaml     # GitLab Kubernetes agent (see kubernetes.md)
│   ├── issue_templates/  merge_request_templates/   # workflow templates (content owned by git.md)
│   └── CODEOWNERS                    # review routing (policy owned by git.md)
├── Dockerfile  .dockerignore         # image authoring owned by dockerfile.md
└── README.md
```

- Split large pipelines into `include: local:` files by stage/domain; keep the root file to globals + includes.
- A repo that *publishes* reusable CI is a **Component project**: `templates/<component>/template.yml` + a release tag.

---

## 5. GitLab CI/CD Mechanics

The unique value of this guide — GitLab's pipeline engine.

### A. Pipeline skeleton (stages, workflow, default)
```yaml
stages: [validate, test, build, security, deploy]

workflow:                              # GL-PIPE-02: run once per change, no duplicate pipelines
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH && $CI_PIPELINE_SOURCE != "merge_request_event"
    - if: $CI_COMMIT_TAG

default:
  image: node:22-alpine
  tags: [docker]                       # GL-RUN-01: pick the runner explicitly
  retry: { max: 2, when: [runner_system_failure, stuck_or_timeout_failure] }
  cache:                               # file-hash key invalidates on dep change
    key: { files: [package-lock.json] }
    paths: [node_modules/, .npm/]
```

### B. `rules:` — when a job runs
Modern control flow. Prefer `rules:` over deprecated `only/except`.
```yaml
deploy:prod:
  rules:
    - if: $CI_COMMIT_TAG =~ /^v\d+\.\d+\.\d+$/    # semver tag (see semver.md)
      when: manual                                 # GL-ENV-01: manual prod gate
    - if: $CI_PIPELINE_SOURCE == "schedule"        # scheduled pipelines
    - when: never
lint:
  rules:
    - changes: [src/**/*, .gitlab-ci.yml]          # run only when relevant files change
```
Useful predefined vars: `CI_PIPELINE_SOURCE`, `CI_COMMIT_BRANCH`, `CI_DEFAULT_BRANCH`, `CI_COMMIT_TAG`, `CI_MERGE_REQUEST_IID`, `CI_COMMIT_SHORT_SHA`, `CI_REGISTRY_IMAGE`, `CI_ENVIRONMENT_SLUG`.

### C. `needs:` — DAG & artifacts
```yaml
test:unit:    { stage: test,  needs: [] }                 # start immediately, ignore stage order
test:e2e:     { stage: test,  needs: [build:app] }        # cross-stage dependency
deploy:stg:
  stage: deploy
  needs:
    - build:image
    - job: test:unit
    - job: container_scanning
      optional: true                                       # don't fail if absent
```
`needs:` builds the DAG; jobs run as soon as their dependencies finish. `parallel:` / `parallel:matrix:` fan a job across variable sets (e.g. `NODE_VERSION: ["20","22"]`).

### D. `include:` & CI/CD Components — reuse (GL-CMP-01)
Components are GitLab's modern, versioned, input-typed replacement for copy-pasted templates and `extends`.
```yaml
include:
  # CI/CD Component from the catalog — PIN to a release tag, never a branch
  - component: $CI_SERVER_FQDN/my-group/ci-components/build@1.4.0
    inputs: { stage: build, image: node:22 }
  - local: .gitlab/ci/test.yml
  - project: my-group/shared-ci
    ref: v3.2.0
    file: /templates/deploy.yml
  - template: Jobs/SAST.gitlab-ci.yml          # GitLab-maintained template
```
Authoring a component (`templates/build/template.yml`):
```yaml
spec:
  inputs:
    stage: { default: build }
    image: { default: node:22-alpine }
---
"build-$[[ inputs.stage ]]":
  stage: $[[ inputs.stage ]]
  image: $[[ inputs.image ]]
  script: [npm ci, npm run build]
```
Within a project, hidden jobs (`.name`) + `extends:` still work for local reuse; prefer components for cross-project sharing.

### E. Multi-project & parent-child pipelines
```yaml
trigger:downstream:
  trigger:
    project: my-group/service-b
    branch: main
    strategy: depend              # mirror downstream status into this pipeline
child:
  trigger:
    include: .gitlab/ci/child.yml
    strategy: depend
```

### F. Merge request pipelines & merge trains (provides: merge-trains)
- **MR pipelines** run against the source branch; **merged-results pipelines** run against the *result* of merging into target — enable them so CI tests the post-merge state.
- **Merge trains** serialize merges: each MR's pipeline runs against the cumulative result of MRs ahead of it, preventing "green-on-its-own, broken-when-combined" regressions. Enable in *Settings → Merge requests → Merge trains*; require "Pipelines must succeed".
- Pair with `resource_group:` to serialize a deploy job so only one environment deployment runs at a time.

```yaml
deploy:prod:
  resource_group: production          # serialize; no concurrent prod deploys
  environment: { name: production, url: https://example.com }
```

---

## 6. Runners, Variables & Secrets

### A. Runners (provides: gitlab-runners) — GL-RUN-01
| Aspect | Choices |
|---|---|
| Scope | **Instance** (shared) · **Group** · **Project** runners |
| Executor | `docker` (most common) · `kubernetes` (autoscale pods) · `docker-autoscaler`/`instance` (cloud VMs) · `shell` (avoid for untrusted) |
| Selection | Jobs match runners by **`tags:`**; untagged jobs only run on runners that accept untagged jobs |
| Isolation | Use ephemeral executors; **never** run untrusted forked-MR code on a privileged/DinD shared runner |

Docker-in-Docker for image builds (or prefer rootless **Kaniko**/**Buildah** to avoid `--privileged`):
```yaml
build:image:
  image: docker:27-cli
  services: [docker:27-dind]
  tags: [dind]
  variables: { DOCKER_TLS_CERTDIR: "/certs", DOCKER_BUILDKIT: "1" }
```

### B. CI/CD variables & secrets — GL-VAR-01/02
- Define in *Settings → CI/CD → Variables*; mark **Masked** (hidden in logs) and **Protected** (only protected branches/tags/environments). Use **environment scopes** to bind a value to one environment.
- Precedence (high→low): job `variables:` → pipeline trigger/schedule → project → group → instance.
- File-type variables for certs/kubeconfig; **never** echo secret vars.
- External secrets via OIDC — no long-lived tokens in GitLab:
```yaml
deploy:
  id_tokens: { VAULT_ID_TOKEN: { aud: https://vault.example.com } }
  secrets:
    DB_PASSWORD:
      vault: prod/db/password@secret
      token: $VAULT_ID_TOKEN
```
Built-in tokens: `CI_JOB_TOKEN` (scoped, short-lived — prefer for API/registry), `$CI_REGISTRY_*` (registry login). Use **CI/CD job token allowlists**; avoid personal access tokens in pipelines.

---

## 7. Built-in Security Scanning

**Policy is owned by [`secure-coding.md`](guides://secure-coding.md)** — severity thresholds, triage, and supply-chain rules live there. GitLab's binding is its native scanners, which emit reports the platform gates on and renders in the MR security widget (GL-SEC-01/02).

```yaml
include:
  - template: Jobs/SAST.gitlab-ci.yml                 # static analysis (auto-detects language)
  - template: Jobs/Secret-Detection.gitlab-ci.yml     # scans history for credentials
  - template: Jobs/Dependency-Scanning.gitlab-ci.yml  # SCA / CVEs in deps
  - template: Jobs/Container-Scanning.gitlab-ci.yml   # image CVEs (needs a built image)
  - template: DAST.gitlab-ci.yml                       # dynamic scan of a running app

container_scanning:
  variables: { CS_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA, CS_SEVERITY_THRESHOLD: high }
dast:
  variables: { DAST_WEBSITE: https://staging.example.com }
  rules: [{ if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH }]
```
- **Scan Execution Policies** & **Merge Request Approval Policies** (`.gitlab/security-policies/`, Ultimate) enforce scans and required approvals org-wide — defense in depth beyond per-project YAML.
- Reports are GitLab JSON artifacts (`gl-sast-report.json`, etc.); GitLab dedups/compares against the target branch and shows only newly introduced findings.

---

## 8. Registries, Environments & Deployment

### A. Container & package registry — GL-REG-01
Build with a `dockerfile.md`-compliant image; here is only the GitLab push:
```yaml
build:image:
  stage: build
  image: docker:27-cli
  services: [docker:27-dind]
  before_script:
    - echo "$CI_REGISTRY_PASSWORD" | docker login -u "$CI_REGISTRY_USER" --password-stdin "$CI_REGISTRY"
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA .   # immutable SHA tag
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA
    - |
      if [ "$CI_COMMIT_BRANCH" = "$CI_DEFAULT_BRANCH" ]; then
        docker tag  $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA $CI_REGISTRY_IMAGE:latest
        docker push $CI_REGISTRY_IMAGE:latest                     # latest only on default branch
      fi
```
Configure a **cleanup policy** (Settings → Packages & registries) to expire untagged/old images. The same registry hosts npm/Maven/PyPI/Generic **package** registries — authenticate with `CI_JOB_TOKEN`.

### B. Environments & deployments — GL-ENV-01
```yaml
deploy:staging:
  stage: deploy
  needs: [build:image, container_scanning]
  environment: { name: staging, url: https://staging.example.com }
  rules: [{ if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH }]

deploy:production:
  stage: deploy
  environment: { name: production, url: https://example.com }
  resource_group: production
  rules: [{ if: $CI_COMMIT_TAG =~ /^v\d+\.\d+\.\d+$/, when: manual }]
```
- Dynamic review apps: `environment: { name: review/$CI_COMMIT_REF_SLUG, on_stop: stop_review, auto_stop_in: 1 week }`.
- Kubernetes deploys use the **GitLab agent** (`.gitlab/agents/<name>/config.yaml`) — cluster mechanics owned by [`kubernetes.md`](guides://kubernetes.md).
- **Auto DevOps** provides a zero-config build→test→scan→deploy pipeline (great to bootstrap); for production, prefer an explicit `.gitlab-ci.yml` so gates and environments are reviewable.

---

## 9. Quick Reference (`glab`)

```bash
glab auth login                       # authenticate
glab ci lint                          # validate .gitlab-ci.yml
glab ci config compile                # expand includes/components
glab ci view / status / trace <job>   # inspect pipeline graph / logs
glab mr create -f / glab mr merge     # MR workflow (policy: git.md)
glab variable set KEY val --masked --protected -g GROUP   # secret var
glab release create v1.2.3            # tag-driven release
```

---

## 10. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] GL-PIPE-01 — `glab ci lint` valid
- [ ] GL-PIPE-02 — `workflow:rules` run once per change, no duplicate pipelines
- [ ] GL-PIPE-03 — DAG via `needs:`, explicit timeouts + bounded retry
- [ ] GL-VAR-01/02 — no inline secrets; sensitive vars masked + protected; prod creds via protected vars / OIDC `secrets:`
- [ ] GL-SEC-01 — SAST, Secret Detection, Dependency Scanning enabled, reports produced
- [ ] GL-SEC-02 — container scanning before deploy, gated on severity threshold
- [ ] GL-RUN-01 — jobs tagged; no privileged DinD on untrusted shared runners
- [ ] GL-ENV-01 — deploys declare `environment:`; prod manual/merge-train gated with `resource_group`
- [ ] GL-REG-01 — immutable image tags; `latest` only on default branch; cleanup policy set
- [ ] GL-CMP-01 — reuse via versioned Component / pinned `include:`, no copy-paste
- [ ] GL-MR-01 — merge blocked until required pipeline passes; merge trains where enabled
- [ ] Agent ran every §3 command and documented any fixes

---
**End of GitLab Platform Guidelines**
