# Azure DevOps Guidelines
Mandatory standards for Azure DevOps platform mechanics: YAML pipelines, templates, variable groups, service connections, environments, Repos branch policies, Boards, and Artifacts. Azure Pipelines (YAML), Azure Repos, Azure Boards, Azure Artifacts.

---
name: azuredevops
title: Azure DevOps Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: infra
tools: [azure-pipelines-yaml, azure-repos, azure-boards, azure-artifacts, az-cli@2, az-devops-ext]
requires:
  - secure-coding
recommends:
  - ci-cd
  - git
  - azure
  - dockerfile
provides:
  - azure-pipelines
  - pipeline-templates
  - variable-groups
  - azure-boards
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only Azure DevOps **platform mechanics**. The general CI/CD lifecycle (stages, gating, test-first delivery) is owned by [`ci-cd.md`](guides://ci-cd.md); Git workflow and PR review by [`git.md`](guides://git.md).

---

## 0. Prerequisites & References

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — secrets, supply chain, CVE policy. *(ADO binding: secrets live in Key Vault-backed variable groups; service connections use workload identity federation; CredScan/dependency scan run as pipeline gates.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`ci-cd.md`](guides://ci-cd.md) — CI/CD lifecycle, gating, deployment strategy *concepts*. This guide only maps them onto Azure Pipelines.
> - [`git.md`](guides://git.md) — branching, PR review, commit hygiene. ADO Repos branch policies enforce these.
> - [`azure.md`](guides://azure.md) — the deploy target (App Service, AKS, Functions, ARM/Bicep).
> - [`dockerfile.md`](guides://dockerfile.md) — container image builds run as pipeline jobs.

> 📎 **SEE ALSO:** [`semver.md`](guides://semver.md) (artifact versioning) · [`observability.md`](guides://observability.md) (pipeline analytics) · [`tdd.md`](guides://tdd.md) (the test gate a pipeline enforces).

---

## 1. Core Philosophies

Azure DevOps-specific principles only. Lifecycle/gating principles come from `ci-cd.md`.

- **Pipelines-as-code**: every pipeline is YAML in the repo, versioned and reviewed with the code. No Classic (UI-defined) build or release pipelines — they are unreviewable and non-portable.
- **Template, don't copy**: shared steps/jobs/stages live in `templates/` and are referenced with parameters. Duplication is a defect.
- **Secretless by default**: service connections use workload identity federation; secrets resolve at runtime from Key Vault-backed variable groups. No secret ever appears in YAML or logs (see `secure-coding.md`).
- **Environments are the deployment boundary**: deployments to protected targets run as `deployment:` jobs against an Azure DevOps **Environment** carrying approvals and checks — not as plain `job:` steps.
- **Traceability**: every change links a Boards work item (`AB#`); every artifact is semantically versioned and traceable to a commit and run.

**Verified Config**: agent-generated Azure DevOps configuration MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `ADO-<TOPIC>-<NN>`. Rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| ADO-PIPE-01 | Pipelines MUST be YAML committed to the repo; no Classic pipelines | `az pipelines list` shows only YAML; repo has `azure-pipelines.yml` | 0 classic build/release defs |
| ADO-PIPE-02 | Reusable logic MUST use templates, not copy-paste | review `templates/`; grep for duplicated step blocks | no duplicated job/step blocks |
| ADO-PIPE-03 | Task `@major` versions and `pool.vmImage` MUST be pinned (no floating) | grep tasks/`vmImage` in YAML | all pinned |
| ADO-VAR-01 | Secrets MUST come from Key Vault-backed variable groups or `secret` vars; never plaintext YAML (see `secure-coding.md`) | grep YAML for inline credentials | 0 inline secrets |
| ADO-SVC-01 | Service connections MUST use workload identity federation (no long-lived SP secret/PAT) (see `secure-coding.md`) | `az devops service-endpoint list` → auth scheme | federation only |
| ADO-SVC-02 | Service connections MUST be least-privilege and not "grant access to all pipelines" | review connection security | per-pipeline authorization |
| ADO-POL-01 | Protected branches MUST have branch policy: min reviewers, build validation, comment resolution, linked work items (see `git.md`) | `az repos policy list -b refs/heads/main` | all required policies on |
| ADO-POL-02 | PR completion MUST require a passing build-validation pipeline | branch policy build validation = required | required |
| ADO-DEPLOY-01 | Deploys to protected envs MUST use `deployment:` jobs targeting an Environment with approvals/checks | inspect stage; Environment has approvals | approvals present |
| ADO-BOARD-01 | Every PR/commit MUST link a work item (`AB#`) | branch policy "Check for linked work items" | enforced |
| ADO-ART-01 | Internal packages MUST publish to an Azure Artifacts feed with pinned versions; public deps via upstream sources | `az artifacts feed list`; feed has upstream | feed + upstream configured |
| ADO-SEC-01 | Pipelines MUST run secret-detection and dependency-CVE gates (see `secure-coding.md`) | pipeline contains scan tasks that fail on findings | gates fail the build |

> **Forbidden**: Classic pipelines; secrets in YAML or `echo`'d to logs; PATs in service connections; `vmImage` or tasks without pinned versions; deploying to prod from a plain `job:` (bypasses Environment checks); merging without a linked work item.

---

## 3. Pipeline Anatomy

A YAML pipeline is `stages → jobs → steps`. Keep `azure-pipelines.yml` thin: triggers, variables, and template references.

```yaml
name: $(Build.DefinitionName)_$(Date:yyyyMMdd)$(Rev:.r)   # build number format

trigger:                          # CI trigger
  batch: true
  branches: { include: [main, release/*] }
  paths: { include: [src/*, tests/*], exclude: [docs/*] }

pr:                               # PR (validation) trigger — runs on PRs into these
  branches: { include: [main] }

variables:
  - group: build-variables        # variable GROUP (shared, can be KV-backed)
  - name: buildConfiguration       # inline variable
    value: Release

stages:
  - stage: Build
    jobs:
      - job: BuildJob
        pool: { vmImage: ubuntu-24.04 }   # pin the image (ADO-PIPE-03)
        steps:
          - template: templates/build-steps.yml      # reuse (ADO-PIPE-02)
            parameters: { buildConfiguration: $(buildConfiguration) }

  - stage: Deploy_Prod
    dependsOn: Build
    condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
    jobs:
      - deployment: DeployProd     # deployment job → targets an Environment
        environment: production
        strategy:
          runOnce:
            deploy:
              steps:
                - template: templates/deploy-steps.yml
```

- **Stage** = a phase (build/test/deploy) with its own gate; stages run sequentially unless `dependsOn` defines a DAG.
- **Job** = a unit on one agent; jobs in a stage run in parallel by default. A `deployment:` job is a special job with strategies and Environment binding.
- **Step** = a `task:` (`Task@major`), `script:`/`bash:`/`pwsh:`, `checkout:`, `download:`/`publish:`.
- `condition:` + status functions (`succeeded()`, `eq()`, `and()`) gate execution.

### Caching & artifacts

```yaml
- task: Cache@2                                   # incremental builds
  inputs:
    key: 'nuget | "$(Agent.OS)" | **/packages.lock.json'
    restoreKeys: 'nuget | "$(Agent.OS)"'
    path: $(NUGET_PACKAGES)

- publish: $(Build.ArtifactStagingDirectory)/app  # shortcut for PublishPipelineArtifact@1
  artifact: drop
# consume in a later stage/job:
- download: current
  artifact: drop
```

> Prefer the `publish:`/`download:` shortcuts (Pipeline Artifacts) over the deprecated `PublishBuildArtifacts@1`. Use `PublishTestResults`/`PublishCodeCoverageResults@2` to surface gates in the run UI.

---

## 4. Templates, Parameters & Expressions

Templates are the reuse mechanism — for steps, jobs, or whole stages.

```yaml
# templates/build-steps.yml
parameters:
  - name: buildConfiguration
    type: string
    default: Release
  - name: runTests
    type: boolean
    default: true
steps:
  - checkout: self
    fetchDepth: 0
  - task: UseDotNet@2
    inputs: { version: 8.x }
  - ${{ if eq(parameters.runTests, true) }}:        # compile-time conditional
    - task: DotNetCoreCLI@2
      inputs: { command: test, arguments: '--collect:"XPlat Code Coverage"' }
```

**Two expression syntaxes — know the difference:**

| Syntax | When evaluated | Use for |
|---|---|---|
| `${{ }}` | compile time (template expansion) | `parameters`, `if`/`each`, structural choices |
| `$( )` | runtime (agent) | `variables`, task inputs, secrets |
| `$[ ]` | runtime, before job start | variable expressions, `dependencies.*` outputs |

- Use **typed `parameters`** (compile-time, can shape YAML) vs **`variables`** (runtime strings). Secrets are always `variables`, never `parameters`.
- `extends:` a template to enforce an org-wide pipeline shell (required-template governance).
- Pass step/job outputs across jobs with `isOutput=true` and `$[ dependencies.JobA.outputs['step.var'] ]`.

---

## 5. Variables, Variable Groups & Service Connections

### Variable groups (the `provides: variable-groups` surface)

- Inline `variables:` for pipeline-local values; **variable groups** for values shared across pipelines.
- Link a group to **Azure Key Vault** so secrets resolve at runtime and are never stored in ADO. Mark non-KV secrets with the lock icon (`secret: true`) — they are masked in logs but still SHOULD prefer Key Vault.

```yaml
variables:
  - group: keyvault-secrets        # variable group linked to Key Vault
steps:
  - task: AzureKeyVault@2          # or pull directly at runtime
    inputs:
      azureSubscription: Azure-WIF-Connection
      KeyVaultName: myapp-kv
      SecretsFilter: 'DbConnString,ApiKey'
      RunAsPreJob: true
  - script: deploy.sh
    env: { DB_CONN: $(DbConnString) }   # reference as runtime var, never echo
```

```bash
az pipelines variable-group create --name keyvault-secrets --authorize true \
  --type AzureKeyVault --azure-key-vault-name myapp-kv
```

### Service connections (secrets binding → `secure-coding.md`)

Service connections authenticate pipelines to Azure/registries/feeds. **Use workload identity federation (OIDC)** — no stored secret to rotate or leak.

```bash
az devops service-endpoint azurerm create --name Azure-WIF-Connection \
  --azure-rm-subscription-id <sub> --azure-rm-subscription-name "Prod" \
  --azure-rm-tenant-id <tenant> --azure-rm-service-principal-id <app-id>
```

Scope each connection to specific pipelines (disable "grant access permission to all pipelines"). The Azure deploy target itself is owned by [`azure.md`](guides://azure.md).

---

## 6. Agent Pools

| | Microsoft-hosted | Self-hosted |
|---|---|---|
| Provisioning | fresh VM per job (e.g. `ubuntu-24.04`, `windows-2022`) | you manage the agent |
| Use when | standard builds, no special hardware | private network access, GPUs, large caches, custom tooling |
| Cleanliness | guaranteed ephemeral | MUST be kept ephemeral/patched yourself |

- Always **pin `vmImage`** (`ubuntu-24.04`, not `ubuntu-latest`) so builds are reproducible (ADO-PIPE-03).
- Self-hosted agents SHOULD run in containers/ephemeral VMs reset per job to match hosted hygiene; keep them patched and isolated from prod credentials.
- Use **container jobs** (`container:` resource) for a controlled toolchain on hosted pools — image build/run policy from [`dockerfile.md`](guides://dockerfile.md).

---

## 7. Deployment Jobs, Strategies & Environments

Deploy to protected targets via a `deployment:` job bound to an **Environment**. Environments carry approvals, checks (gates), and deployment history per resource.

```yaml
- deployment: DeployProd
  environment: production           # approvals/checks attach here, not in YAML
  strategy:
    runOnce:                        # also: rolling, canary
      preDeploy:  { steps: [ ... ] }
      deploy:     { steps: [ ... ] }
      routeTraffic: { steps: [ ... ] }
      postRouteTraffic: { steps: [ ... ] }
      on: { failure: { steps: [ ... ] }, success: { steps: [ ... ] } }
```

**Strategies:** `runOnce` (single pass), `rolling` (batched over VM resources), `canary` (incremental traffic with `increments`). Choose per `ci-cd.md`'s deployment-strategy guidance; this section is only the YAML binding. Blue-green is done with App Service slot swap (`AzureWebApp@1` to a slot, then `AzureAppServiceManage@0` Swap Slots).

**Approvals & checks** are configured on the Environment (UI/API), not in the pipeline:
- Manual approvals (named approvers, min count, timeout).
- Gates: Azure Function, REST, query work items (e.g. block if open Critical bugs), Invoke business hours.
- A `pool: server` job with `ManualValidation@0` adds an inline mid-pipeline approval.

> Reference build pipelines as a `resources.pipelines` input and `download:` their artifacts for release stages — keeps build and deploy decoupled.

---

## 8. Azure Repos: Branch Policies (Git binding)

Workflow rules (trunk-based, PR review, small commits) are owned by [`git.md`](guides://git.md). Azure Repos **enforces** them via branch policies on protected branches:

- **Minimum reviewers** (`creatorVoteCounts: false`, `resetOnSourcePush: true`, block last-pusher self-approval).
- **Build validation** — a CI pipeline must pass before completion (ADO-POL-02).
- **Status checks** — external gates (security, coverage) post status; mark required.
- **Comment resolution required** + **linked work items required** (ADO-BOARD-01).
- **Merge strategy** — enforce squash/rebase for linear history.

```bash
az repos policy approver-count create --branch main --repository-id <id> \
  --minimum-approver-count 2 --reset-on-source-push true --creator-vote-counts false
az repos policy build create --branch main --repository-id <id> \
  --build-definition-id <pipelineId> --queue-on-source-update-only true
az repos policy work-item-linking create --branch main --repository-id <id> --blocking true
```

**Commit/PR linking** — Azure DevOps auto-links work items from `AB#1234` in commit/PR text; keywords `Fixes/Closes/Resolves AB#1234` transition the item on merge. Put a PR template at `.azuredevops/pull_request_template.md`.

---

## 9. Azure Boards: Work Items

Boards is the traceability backbone (`provides: azure-boards`). Hierarchy: **Epic → Feature → Product Backlog Item/User Story → Task**, with **Bug** linked to the work it affects.

- Every branch/PR/deploy traces to a work item; CI/CD links them automatically (`--work-items` on PR create; `Build.SourceVersionMessage` parsing).
- Use **area paths** (ownership) and **iteration paths** (sprints); configure board columns to mirror your states.
- Query with WIQL for gates and reporting (e.g. deployment gate "0 active Critical bugs").

```bash
az boards work-item create --type Bug --title "Login fails on SSO" --area "Proj\\Backend"
az boards work-item update --id 12345 --state Resolved --discussion "Fixed in !456"
az boards query --wiql "SELECT [Id],[Title] FROM WorkItems WHERE [State]='Active'"
```

---

## 10. Azure Artifacts

Private package feeds for NuGet, npm, Maven, PyPI, Cargo, and Universal Packages.

- Publish internal packages to a **feed**; consume public packages through **upstream sources** on the same feed (caches + provides provenance/auditability).
- Version with SemVer (see `semver.md`); derive build metadata from `$(Build.BuildId)`. Authenticate in-pipeline with `NuGetAuthenticate@1` / `npmAuthenticate@0` — never embed feed tokens.

```yaml
- task: NuGetAuthenticate@1
- task: DotNetCoreCLI@2
  inputs:
    command: push
    packagesToPush: '$(Build.ArtifactStagingDirectory)/**/*.nupkg'
    nuGetFeedType: internal
    publishVstsFeed: 'my-org/my-feed'
```

Configure upstreams in feed settings; pin direct dependency versions and commit lockfiles (`packages.lock.json`, `package-lock.json`).

---

## 11. Security & Compliance Gates

Policy is owned by [`secure-coding.md`](guides://secure-coding.md); Azure DevOps bindings:

- **Secrets**: Key Vault-backed variable groups + `AzureKeyVault@2`; never in YAML/logs (ADO-VAR-01).
- **Secret detection**: run a credential scanner (e.g. CredScan / `gitleaks`) as a failing gate.
- **Dependency CVEs**: OWASP Dependency-Check / `dependency scanning` task that fails on CVSS ≥ 7 (ADO-SEC-01).
- **SAST**: SonarCloud/CodeQL tasks as required status checks on PRs.
- **Governance**: required `extends:` template, restricted variable/secret authorization, environment approvals form the audit trail.

---

## 12. Quick Reference (az + az devops)

```bash
az devops configure --defaults organization=https://dev.azure.com/org project=Proj
az pipelines run --name CI-Pipeline --branch main
az pipelines list --output table
az pipelines variable-group create --name build-vars --variables k=v
az repos pr create --source-branch feature/AB12345 --target-branch main --work-items 12345
az repos pr update --id 456 --auto-complete true --delete-source-branch true
az repos policy list --branch main --repository-id <id>
az boards work-item create --type Bug --title "..."
az artifacts feed list --output table
az devops service-endpoint list --output table
```

---

## 13. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] ADO-PIPE-01 — pipelines are YAML in-repo; no Classic pipelines
- [ ] ADO-PIPE-02 — shared logic uses templates, no duplication
- [ ] ADO-PIPE-03 — task `@major` and `vmImage` pinned
- [ ] ADO-VAR-01 — no secrets in YAML; Key Vault / secret vars only
- [ ] ADO-SVC-01/02 — service connections use federation, least-privilege, per-pipeline scope
- [ ] ADO-POL-01/02 — branch policies: reviewers, build validation, comments, work items
- [ ] ADO-DEPLOY-01 — protected deploys use Environment-bound `deployment:` jobs with approvals
- [ ] ADO-BOARD-01 — every change links a work item (`AB#`)
- [ ] ADO-ART-01 — internal feed with upstream sources; pinned versions
- [ ] ADO-SEC-01 — secret-detection and dependency-CVE gates fail the build
- [ ] Agent verified each gate and documented any fixes

---
**End of Azure DevOps Guidelines**
