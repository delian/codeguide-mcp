# GitHub Platform Guidelines
Mandatory standards for GitHub Actions, Actions supply-chain security, branch protection, environments, Dependabot, GHCR and releases. GitHub Actions, OIDC, Dependabot v2, GHCR.

---
name: github
title: GitHub Platform Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: infra
tools: [github-actions, actions/checkout@v4, github-oidc, dependabot@v2, ghcr.io, gh-cli]
requires:
  - secure-coding
recommends:
  - ci-cd
  - git
  - semver
  - dockerfile
provides:
  - github-actions
  - reusable-workflows
  - actions-security
  - branch-protection
  - dependabot
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only GitHub-platform mechanics. General CI/CD *concepts* live in [`ci-cd.md`](guides://ci-cd.md); the general git/PR *workflow* lives in [`git.md`](guides://git.md).

---

## 0. Prerequisites & References

Fetch and apply these **before** authoring GitHub configuration. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — supply-chain, secrets, CVE policy. *(GitHub binding: pin actions to a full commit SHA, least-privilege `GITHUB_TOKEN`, OIDC instead of long-lived cloud keys.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`ci-cd.md`](guides://ci-cd.md) — pipeline stages, gates, deployment strategy (Actions is the *executor* of these concepts).
> - [`git.md`](guides://git.md) — branch model, PR/review flow, commit conventions (branch protection *enforces* this).
> - [`semver.md`](guides://semver.md) — version/tag scheme that drives Releases.
> - [`dockerfile.md`](guides://dockerfile.md) — image build rules (GHCR is the *registry*, not the Dockerfile).

> 📎 **SEE ALSO:** [`pre-commit.md`](guides://pre-commit.md) · [`code-review.md`](guides://code-review.md) · [`observability.md`](guides://observability.md)

---

## 1. Core Philosophies: GitHub-platform

Platform mechanics only. CI/CD strategy, git workflow, and security policy come from §0 — do **not** restate them here.

- **Pin everything immutable.** Third-party actions are remote code; pin to a full commit SHA, not a moving tag (policy: `secure-coding.md`). First-party `actions/*` MAY use a major tag only when Dependabot keeps them current.
- **Least privilege by default.** Set `permissions: {}` at workflow level and grant the minimum per job. The default `GITHUB_TOKEN` should never be write unless a step needs it.
- **Keyless to the cloud.** Use OIDC (`id-token: write`) to mint short-lived cloud credentials; never store long-lived provider keys as secrets.
- **DRY workflows.** Factor shared logic into reusable workflows (`workflow_call`) and composite actions; never copy-paste jobs across repos.
- **The platform is the gate.** A green check is meaningless unless branch protection / rulesets *require* it. Configuration-as-policy lives in the repo (`.github/`), reviewed like code.
- **Automate dependency hygiene.** Dependabot covers every ecosystem in the repo, including `github-actions` itself.

**Verified Config**: Agent-authored GitHub config MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `GH-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| GH-SEC-01 | Third-party actions MUST be pinned to a full 40-char commit SHA (see `secure-coding.md`) | `grep -rEn 'uses: [^/]+/[^@]+@v?[0-9.]+$' .github/` | no non-SHA third-party pins |
| GH-SEC-02 | Every workflow MUST declare explicit `permissions` (default-deny) | `grep -L 'permissions:' .github/workflows/*.yml` | every workflow lists it |
| GH-SEC-03 | `GITHUB_TOKEN` MUST be least-privilege; no blanket `write-all` | review / `actionlint` | no `permissions: write-all` |
| GH-SEC-04 | Cloud auth MUST use OIDC, not stored long-lived keys (see `secure-coding.md`) | grep for `id-token: write`; grep secrets for `AWS_SECRET`/`AZURE_*`/`GCP_*` | no static cloud creds |
| GH-SEC-05 | Secrets MUST never be echoed/interpolated into shell args (see `secure-coding.md`) | `actionlint` / review | no `run:` echoing secrets |
| GH-WF-01 | Workflows MUST lint clean | `actionlint .github/workflows/*.yml` | exit 0 |
| GH-WF-02 | Long-running workflows MUST set `concurrency` + a job `timeout-minutes` | review / grep | both present |
| GH-WF-03 | Shared CI logic MUST be a reusable workflow or composite action, not copy-paste | review | no duplicated jobs |
| GH-DEP-01 | `.github/dependabot.yml` MUST cover every ecosystem incl. `github-actions` | inspect file | all ecosystems present |
| GH-BRN-01 | Default branch MUST be protected: required checks + ≥1 review + linear history (see `git.md`) | `gh api repos/:owner/:repo/branches/main/protection` | rules present |
| GH-BRN-02 | Critical paths MUST have CODEOWNERS + required owner review | inspect `.github/CODEOWNERS`; branch rule | owner review required |
| GH-ENV-01 | Production deploys MUST target a protected `environment` with required reviewers | `gh api repos/:owner/:repo/environments` | protection rules set |
| GH-REL-01 | Releases MUST be tagged per SemVer and built from a pinned ref (see `semver.md`) | `gh release list` / tag check | tags valid SemVer |
| GH-PKG-01 | Images MUST publish to GHCR with provenance + SBOM; never embed secrets in layers | inspect container workflow | `provenance:`+`sbom:` true |

> **Forbidden**: floating-tag third-party actions, `pull_request_target` running untrusted checkout code, `permissions: write-all`, long-lived cloud keys in secrets, secrets interpolated into `run:` strings, or an unprotected default branch.

---

## 3. Verification Protocol

Run before presenting GitHub config. Fix → re-run until green.

```bash
actionlint .github/workflows/*.yml                          # GH-WF-01/02/03, GH-SEC-03/05
grep -rEn 'uses: [^./][^/]+/[^@]+@v?[0-9.]+$' .github/       # GH-SEC-01: unpinned third-party
grep -rL 'permissions:' .github/workflows/                  # GH-SEC-02: missing permissions
zizmor .github/workflows/                                   # GH-SEC-*: Actions-aware static audit
gh api repos/:owner/:repo/branches/main/protection          # GH-BRN-01
```

`actionlint` catches syntax, shell, and expression injection; `zizmor` audits Actions-specific supply-chain risks. The *why* behind each gate lives in its §0 owner.

---

## 4. Repository Layout (`.github/`)

GitHub reads configuration from conventional paths. This is the platform contract, not project structure.

```
.github/
├── workflows/
│   ├── ci.yml                  # PR gate: lint/test/build (see ci-cd.md for stages)
│   ├── release.yml             # tag-triggered build + publish (see semver.md)
│   ├── container.yml           # build & push to GHCR
│   └── reusable-deploy.yml     # workflow_call template
├── actions/<name>/action.yml   # local composite actions
├── dependabot.yml              # version-update config (all ecosystems)
├── CODEOWNERS                  # review routing → branch-protection owner reviews
├── ISSUE_TEMPLATE/             # form-based issue templates (*.yml)
├── pull_request_template.md
└── SECURITY.md                 # vuln-reporting policy (see secure-coding.md)
```

---

## 5. GitHub Actions Mechanics

The unique value of this guide. The *concepts* (pipeline stages, gating, deployment strategy) are owned by [`ci-cd.md`](guides://ci-cd.md); below is the GitHub binding.

### A. Workflow skeleton — triggers, concurrency, default-deny permissions
```yaml
# .github/workflows/ci.yml
name: CI
on:
  push: { branches: [main] }
  pull_request:
    branches: [main]
    paths-ignore: ['**.md', 'docs/**']   # skip doc-only churn
  workflow_dispatch:                       # manual run

permissions: {}                            # GH-SEC-02: default-deny at top level
concurrency:                               # GH-WF-02: cancel superseded PR runs
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 15                     # GH-WF-02
    permissions:
      contents: read                        # GH-SEC-03: grant only what this job needs
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@<full-sha>   # GH-SEC-01: third-party → SHA-pinned
        with: { version: "0.5.x" }
      - run: uv run pytest                    # tool gate lives in the language guide
```
Key fields: `needs:` for job dependencies (DAG), `if:` for conditional jobs/steps, `outputs:` to pass data downstream, `continue-on-error` only for non-gating steps.

### B. Matrix builds
```yaml
strategy:
  fail-fast: false                          # see every cell's failure, not just the first
  matrix:
    os: [ubuntu-latest, macos-latest]
    version: ["3.12", "3.13"]
    include: [{ os: ubuntu-latest, version: "3.13", coverage: true }]
    exclude: [{ os: macos-latest, version: "3.12" }]
runs-on: ${{ matrix.os }}
```
Quote numeric versions (`"3.10"`) — YAML strips the trailing zero otherwise. A `matrix` value of `'20.x'` is fine for setup actions but never feed it to literal version math.

### C. Reusable workflows (`workflow_call`) — cross-job/repo DRY
```yaml
# .github/workflows/reusable-deploy.yml  (the callee)
on:
  workflow_call:
    inputs:   { environment: { required: true, type: string } }
    secrets:  { deploy_token: { required: true } }
    outputs:  { url: { value: ${{ jobs.deploy.outputs.url }} } }
```
```yaml
# caller
jobs:
  prod:
    uses: ./.github/workflows/reusable-deploy.yml          # local
    # uses: org/shared/.github/workflows/deploy.yml@<sha>  # cross-repo → pin to SHA
    with: { environment: production }
    secrets: inherit                          # or pass explicitly; prefer explicit
```
Reusable workflows nest jobs and have their own `permissions`. Composite actions (below) factor *step* sequences inside one job.

### D. Composite actions — reusable step sequences
```yaml
# .github/actions/setup-project/action.yml
name: Setup project
inputs:
  uv-version: { default: "0.5.x" }
runs:
  using: composite
  steps:
    - uses: astral-sh/setup-uv@<full-sha>
      with: { version: ${{ inputs.uv-version }} }
    - run: uv sync
      shell: bash                             # `shell:` is MANDATORY in composite run steps
```
Usage: `- uses: ./.github/actions/setup-project`.

---

## 6. Actions Security (supply chain)

Policy is owned by [`secure-coding.md`](guides://secure-coding.md). GitHub binding:

- **SHA-pin third-party actions** (GH-SEC-01). A tag is mutable — a compromised maintainer can repoint `@v3` to malicious code (cf. the `tj-actions/changed-files` incident). Pin to the full 40-char commit and let Dependabot bump it with a comment:
  ```yaml
  - uses: docker/build-push-action@4f58ea79222b3b9dc2c8bbdd6debcd730fa81323  # v6.9.0
  ```
- **Least-privilege `GITHUB_TOKEN`** (GH-SEC-02/03). `permissions: {}` at top, grant per job (`contents: read`, add `packages: write`, `id-token: write`, `security-events: write` only where used). Never `write-all`.
- **OIDC, not stored keys** (GH-SEC-04). Request `id-token: write` and exchange the token for short-lived cloud credentials via the provider's official login action — no `AWS_SECRET_ACCESS_KEY`/service-account JSON in secrets.
  ```yaml
  permissions: { id-token: write, contents: read }
  steps:
    - uses: aws-actions/configure-aws-credentials@<sha>
      with: { role-to-assume: arn:aws:iam::123:role/ci, aws-region: us-east-1 }
  ```
- **Untrusted PRs** (GH-SEC-05). Prefer `pull_request` (token is read-only, no secrets on forks). Use `pull_request_target` only for trusted automation and **never** check out + execute the PR head with it. Never interpolate `${{ github.event.* }}` into a `run:` string — it is shell injection; pass via `env:` instead.
- **Secrets**: repo/org/environment-scoped, masked in logs automatically (don't defeat masking by base64/echo). Environment secrets gate behind environment protection rules.

---

## 7. Branch Protection & Rulesets

Enforce the [`git.md`](guides://git.md) branch model at the platform. Rulesets are the modern, layerable successor to classic branch protection (support tag rules, bypass lists, org-level application).

Required on the default branch (GH-BRN-01): required status checks (`strict: true` = up-to-date before merge), ≥1 PR review, dismiss stale reviews, require CODEOWNER review, linear history, no force-push, no deletion, conversation resolution, and `enforce_admins`.

```bash
gh api -X PUT repos/:owner/:repo/branches/main/protection --input protection.json
# protection.json: required_status_checks{strict,contexts[]}, enforce_admins,
#   required_pull_request_reviews{require_code_owner_reviews,required_approving_review_count},
#   required_linear_history, allow_force_pushes:false, allow_deletions:false,
#   required_conversation_resolution:true
```

**CODEOWNERS** (GH-BRN-02) routes mandatory review by path:
```
# .github/CODEOWNERS — last matching pattern wins
*                       @org/core-team
/src/domain/            @org/architects
/.github/workflows/     @org/devops
/Dockerfile             @org/security @org/devops
```

---

## 8. Environments & Deployments

Environments add deployment-time gates and scoped secrets — the GitHub mechanism behind a CD promotion (strategy owned by [`ci-cd.md`](guides://ci-cd.md)).

```yaml
deploy-prod:
  environment:
    name: production                          # GH-ENV-01: required reviewers, wait timer,
    url: ${{ steps.deploy.outputs.url }}      #   and branch policy enforced here
  permissions: { id-token: write, contents: read }
  steps: [...]
```
Configure on the environment (not in YAML): required reviewers (manual approval gate), wait timer, allowed deployment branches, and environment secrets/variables. Use `concurrency: { group: deploy-prod, cancel-in-progress: false }` so deploys queue rather than abort.

---

## 9. Dependabot

`.github/dependabot.yml` (v2) — one entry per ecosystem present, including `github-actions` so pinned SHAs stay current (GH-DEP-01):
```yaml
version: 2
updates:
  - package-ecosystem: github-actions          # bumps SHA pins + leaves the version comment
    directory: "/"
    schedule: { interval: weekly }
    groups: { actions: { patterns: ["*"] } }   # one grouped PR, less noise
  - package-ecosystem: docker
    directory: "/"
    schedule: { interval: weekly }
  - package-ecosystem: npm                      # repeat per language manifest
    directory: "/"
    schedule: { interval: weekly }
    open-pull-requests-limit: 10
    ignore:
      - dependency-name: "react"
        update-types: ["version-update:semver-major"]
```
Pair with Dependabot **security updates** (auto-enabled via repo settings / org policy) for CVE-triggered PRs. CVE *policy* and triage are owned by [`secure-coding.md`](guides://secure-coding.md).

---

## 10. GitHub Packages / Container Registry (GHCR)

Push images to `ghcr.io` with the workflow `GITHUB_TOKEN` (no PAT needed). Dockerfile rules are owned by [`dockerfile.md`](guides://dockerfile.md); this is the publish binding (GH-PKG-01):

```yaml
permissions: { contents: read, packages: write, id-token: write }
steps:
  - uses: actions/checkout@v4
  - uses: docker/login-action@<sha>
    with: { registry: ghcr.io, username: ${{ github.actor }}, password: ${{ secrets.GITHUB_TOKEN }} }
  - id: meta
    uses: docker/metadata-action@<sha>
    with:
      images: ghcr.io/${{ github.repository }}
      tags: |
        type=semver,pattern={{version}}
        type=sha
        type=raw,value=latest,enable={{is_default_branch}}
  - uses: docker/build-push-action@<sha>
    with:
      push: ${{ github.event_name != 'pull_request' }}
      tags: ${{ steps.meta.outputs.tags }}
      labels: ${{ steps.meta.outputs.labels }}
      cache-from: type=gha
      cache-to: type=gha,mode=max
      provenance: true        # GH-PKG-01: build provenance attestation
      sbom: true              # GH-PKG-01: SBOM attestation
```
Scan the pushed image (e.g. Trivy) and upload SARIF to code scanning. Link the package to the repo for inherited visibility/permissions; default new packages to private.

---

## 11. Releases

Releases turn a SemVer tag (owned by [`semver.md`](guides://semver.md)) into a published artifact set (GH-REL-01). Trigger on the tag, build from that immutable ref, attach artifacts:
```yaml
on: { push: { tags: ['v*.*.*'] } }
permissions: { contents: write }              # only the release job
jobs:
  release:
    steps:
      - uses: actions/checkout@v4
      - run: gh release create "$GITHUB_REF_NAME" --generate-notes ./dist/*
        env: { GH_TOKEN: ${{ secrets.GITHUB_TOKEN }} }
```
Prefer auto-generated notes (configurable via `.github/release.yml`) over hand-maintained changelogs. Mark pre-1.0 / RC tags as prereleases. Never re-tag a published version — cut a new one.

---

## 12. Quick Reference (gh CLI)

```bash
gh workflow run ci.yml -f key=val      # dispatch a workflow
gh run watch                           # follow latest run
gh run rerun <id> --failed             # re-run failed jobs only
gh release create v1.2.0 --generate-notes ./dist/*
gh api repos/:owner/:repo/branches/main/protection   # inspect protection
gh secret set NAME --env production    # environment-scoped secret
actionlint && zizmor .github/workflows/              # lint + security audit
```

---

## 13. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] GH-SEC-01 — third-party actions pinned to full commit SHA
- [ ] GH-SEC-02 — every workflow declares explicit `permissions`
- [ ] GH-SEC-03 — `GITHUB_TOKEN` least-privilege, no `write-all`
- [ ] GH-SEC-04 — cloud auth via OIDC, no stored long-lived keys
- [ ] GH-SEC-05 — no secrets interpolated into `run:` / echoed to logs
- [ ] GH-WF-01 — `actionlint` clean
- [ ] GH-WF-02 — `concurrency` + job `timeout-minutes` set
- [ ] GH-WF-03 — shared logic is a reusable workflow / composite action
- [ ] GH-DEP-01 — Dependabot covers all ecosystems incl. `github-actions`
- [ ] GH-BRN-01 — default branch protected (checks + review + linear history)
- [ ] GH-BRN-02 — CODEOWNERS + required owner review on critical paths
- [ ] GH-ENV-01 — production environment protected with required reviewers
- [ ] GH-REL-01 — releases tagged per SemVer from a pinned ref
- [ ] GH-PKG-01 — GHCR images publish with provenance + SBOM, no embedded secrets
- [ ] Agent ran every §3 command and documented any fixes

---
**End of GitHub Platform Guidelines**
