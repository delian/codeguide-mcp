# Terraform Development Guidelines
Mandatory standards for Terraform/OpenTofu IaC: modular, tested, secure, remote-state, version-pinned. Terraform 1.9+ / OpenTofu, tflint, tfsec/checkov, terraform-docs.

---
name: terraform
title: Terraform Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: infra
tools: [terraform@1.9, opentofu@1.8, tflint, tfsec, checkov, terraform-docs, terratest, infracost]
requires:
  - secure-coding
recommends:
  - ci-cd
  - aws
  - azure
  - gcp
  - git
  - env-config
  - kubernetes
provides:
  - hcl
  - terraform-modules
  - state-management
  - terraform-testing
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Terraform/OpenTofu. Everything here applies equally to **OpenTofu** (the MPL-2.0 open-source fork of Terraform, governed by the Linux Foundation) unless a feature is called out as tool-specific — substitute the `tofu` binary for `terraform`.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Terraform code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — vulnerability scanning, supply chain, secrets policy, least privilege. *(Terraform binding: IaC scanning via `tfsec`/`checkov`; passwordless auth; state-as-secret.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`ci-cd.md`](guides://ci-cd.md) — pipeline policy for the plan→apply / GitOps workflow *(binding: §8)*
> - [`env-config.md`](guides://env-config.md) — variable & secret injection across environments *(binding: `*.tfvars`, `TF_VAR_*`, vault data sources)*
> - [`git.md`](guides://git.md) — branch/PR workflow that gates `apply` on merge.
> - [`aws.md`](guides://aws.md) · [`azure.md`](guides://azure.md) · [`gcp.md`](guides://gcp.md) — the cloud providers; this guide never restates their services.
> - [`kubernetes.md`](guides://kubernetes.md) — when Terraform provisions clusters or the `kubernetes`/`helm` providers.

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) (test-first, regression-before-fix — the policy behind §7) · [`semver.md`](guides://semver.md) (module versioning) · [`comments.md`](guides://comments.md) (doc policy behind `terraform-docs`) · [`pre-commit.md`](guides://pre-commit.md).

---

## 1. Core Philosophies: TERRAFORM-FIRST

Terraform-specific principles only. Security, secrets, and least privilege come from [`secure-coding.md`](guides://secure-coding.md); CI policy from [`ci-cd.md`](guides://ci-cd.md).

- **T**ested modules: every module ships plan-level and lifecycle tests (§7); no module merges untested.
- **E**nvironment parity: one codebase across dev/test/prod; differences live only in `*.tfvars` / workspace, never in forked HCL.
- **R**eusable modules: small, composable, single-responsibility modules with explicit input/output contracts (§5).
- **R**emote state: state is always remote, encrypted, and locked (§6); never local, never in VCS.
- **A**pply-after-plan: `apply` only ever consumes a reviewed saved plan; never blind-apply.
- **F**ail-safe refactors: address changes go through `moved`/`removed`/`import` blocks, not destroy-and-recreate (§9).
- **O**utputs documented: every resource a module creates or references is exported via `outputs.tf` and documented with `terraform-docs`.
- **R**eproducible versions: pin `required_version`, every provider, and every module source; commit `.terraform.lock.hcl`.
- **M**inimal privilege & passwordless: managed identities / OIDC federation only — no static credentials (see `secure-coding.md`).

**Verified Code**: Agent-generated Terraform MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `TF-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| TF-FMT-01 | HCL MUST be canonically formatted | `terraform fmt -check -recursive` | no diff |
| TF-VAL-01 | Configuration MUST validate | `terraform validate` | exit 0 |
| TF-LINT-01 | Linter MUST pass clean | `tflint --recursive` | exit 0 |
| TF-SEC-01 | 0 high/critical IaC findings (see `secure-coding.md`) | `tfsec .` **and** `checkov -d .` | 0 high/critical |
| TF-SEC-02 | No static credentials/secrets in code, tfvars, or state (see `secure-coding.md`) | review / secret scan; passwordless auth | none committed |
| TF-TST-01 | Every module MUST have tests, written test-first (see `tdd.md`) | `terraform test` / `go test ./tests/...` | exit 0 |
| TF-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | rerun test suite | failing→passing |
| TF-DEP-01 | `required_version`, providers, and module sources MUST be pinned; lockfile committed | grep constraints; `.terraform.lock.hcl` in VCS | all pinned & locked |
| TF-STATE-01 | Remote state MUST be encrypted, locked, and out of VCS | inspect backend; `git check-ignore *.tfstate` | remote + encrypted + locked |
| TF-PLAN-01 | `apply` MUST consume a reviewed saved plan (see `ci-cd.md`) | CI plan artifact / PR comment | reviewed; no surprise destroys |
| TF-DOC-01 | Module inputs/outputs MUST be documented | `terraform-docs markdown --output-check` | in sync |
| TF-STRUCT-01 | Root modules own provider+backend; child modules own neither | review | compliant |

> **Forbidden**: `apply` without a reviewed plan; `-lock=false` in shared/prod state; static credentials anywhere; `ref=main` / unpinned module sources; committing `*.tfstate`; deleting tests to make a pipeline green; fixing a bug without a regression test first (violates `tdd.md`).

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
terraform fmt -check -recursive        # TF-FMT-01
terraform init -backend=false          # providers/modules resolve
terraform validate                     # TF-VAL-01
tflint --recursive                     # TF-LINT-01
tfsec . && checkov -d .                # TF-SEC-01  (IaC scan — policy: secure-coding.md)
terraform test                         # TF-TST-01  (native test framework)
terraform-docs markdown --output-check ./modules/...   # TF-DOC-01
terraform plan -var-file=env/<ws>.tfvars -out=tfplan   # TF-PLAN-01 (review before apply)
```

The *why* behind each gate (why scan, why test-first, why plan) lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

One file = one function; name files `NNN-function.tf` (e.g. `010-network.tf`, `020-compute.tf`) for deterministic read order and fewer merge conflicts. Co-locate only the variables specific to a file.

```
infrastructure/
├── modules/                 # reusable, provider-scoped child modules
│   └── network/
│       ├── main.tf          # resources
│       ├── variables.tf     # typed, described inputs
│       ├── outputs.tf       # every created/referenced resource exported
│       ├── versions.tf      # required_version + required_providers (NO provider/backend block)
│       ├── README.md        # generated by terraform-docs
│       └── examples/complete/   # runnable example, also the test fixture
├── root/                    # root module: orchestrates child modules
│   ├── main.tf  providers.tf  backend.tf  variables.tf  outputs.tf  versions.tf
├── env/                     # per-environment inputs (see env-config.md)
│   ├── dev.tfvars  test.tfvars  prod.tfvars
├── tests/                   # *.tftest.hcl and/or Terratest (Go)
├── .tflint.hcl  .terraform-docs.yml  .terraform.lock.hcl
└── .gitignore               # excludes *.tfstate, .terraform/
```

**Root vs child (TF-STRUCT-01):**
- Root module: holds **all** `provider` and `backend` configuration, wires child modules together, contains no raw `resource` blocks of its own where avoidable.
- Child module: declares `required_providers` in `versions.tf`, **never** a `provider` or `backend` block; receives providers from the caller; exports every resource via `outputs.tf`.
- Inputs: required inputs have **no `default`** (Terraform then errors when one is missing — fail fast); optional inputs carry a typed `default` and a `description`. Use `validation` blocks to reject bad values early.
- `locals` are for values **derived** from variables/data sources only — never to hold raw input (inputs come from variables). No hidden magic constants.

---

## 5. Module Design (owned: `terraform-modules`)

### A. Composition patterns
- **Flat** — root invokes each child directly; default for small stacks.
- **Nested** — a parent module composes children that always deploy together (e.g. `application` = `compute` + `load_balancer`), exposing one aggregated output surface.
- **Facade** — a thin wrapper that hard-codes organizational defaults over a generic module (e.g. `standard_database` enforces encryption, backups, multi-AZ in prod), passing only a minimal interface to callers.

Wire modules through outputs, not implicit ordering; reserve `depends_on` for genuinely hidden dependencies.

### B. Versioning & sources (see `semver.md`)
Always pin the source; never `ref=main` or an unconstrained registry source.

```hcl
# Git tag (semver: MAJOR=breaking I/O, MINOR=new optional input/output, PATCH=fix)
module "network" {
  source = "git::https://github.com/org/tf-modules.git//modules/network?ref=v2.1.0"
}

# Private/public registry — pessimistic constraint
module "vpc" {
  source  = "app.terraform.io/my-org/network/aws"
  version = "~> 2.1"          # >= 2.1.0, < 3.0.0
}
```

Constraint syntax: `= 2.1.0` exact · `~> 2.1` pessimistic · `>= 2.0, < 3.0` range. Tag releases in CI only after tests pass; keep a `CHANGELOG.md` per module. Published modules follow the `terraform-<PROVIDER>-<NAME>` repo convention with at least one `examples/` and `terraform-docs`-generated docs.

### C. HCL idioms

**`for_each` over `count`** — `for_each` keys resources by map key/set element, so adding or removing one entry never reindexes (and silently recreates) the others. Reserve `count` for binary enable/disable.

```hcl
resource "cloud_subnet" "this" {
  for_each          = var.subnets                 # map(object({...}))
  name              = "${var.project}-${var.environment}-${each.key}"
  cidr_block        = each.value.cidr_block
  availability_zone = each.value.availability_zone
}
# Addresses: cloud_subnet.this["public-a"] — removing one entry destroys only that subnet.

resource "cloud_monitoring" "main" {
  count = var.enable_monitoring ? 1 : 0           # count: enable/disable only
}
```

**Dynamic blocks** — generate repeated nested blocks from a collection; keep to one nesting level (refactor into a module beyond that). Do not use them for static, known-at-authoring blocks.

**`for` expressions & functions** — filter/transform collections inline:
```hcl
for_each = { for k, v in var.instances : k => v if v.enabled }   # filter map
```
Reach for `lookup`/`coalesce` (safe defaults), `merge` (compose tag maps), `flatten`/`distinct`, `try` (tolerate missing attrs), `templatefile` (render user-data). Extract complex expressions into a well-named `local`.

**Complex types** — model inputs with `map(object({...}))` / `optional(...)` so the module interface is self-documenting and validated by Terraform, not by convention.

---

## 6. State Management (owned: `state-management`)

### A. Remote backends
State is remote, encrypted at rest and in transit, locked, and versioned. Pick the backend for your primary cloud; the bucket/account itself follows least privilege (see `secure-coding.md`) — only the Terraform execution identity may read/write it.

```hcl
# AWS S3 + DynamoDB lock (bucket: versioned + public-access-blocked; KMS-encrypted)
backend "s3" {
  bucket = "myorg-tf-state"
  key    = "infra/network/terraform.tfstate"
  region = "us-east-1"
  encrypt        = true
  kms_key_id     = "alias/terraform-state"
  dynamodb_table = "terraform-state-lock"   # partition key LockID (String)
}
# GCS: backend "gcs" { bucket = ...  prefix = ... }   # built-in locking, enable object versioning + CMEK
# Azure: backend "azurerm" { ... use_oidc = true }    # blob versioning + soft-delete, lease-based lock
```
OpenTofu also supports **native client-side state encryption** (1.7+) — prefer it when the backend lacks server-side encryption.

### B. Locking & state-as-secret
- Locking is always on; never `-lock=false` outside a throwaway local sandbox. CI must bound lock-acquisition (e.g. fail after 5 min and alert).
- `terraform force-unlock <ID>` only after confirming no run is in flight.
- State contains IDs, IPs, and any unmarked secret. Mark sensitive outputs `sensitive = true`; treat the backend as confidential. Never commit state — `.gitignore` `*.tfstate*`, `.terraform/`, `crash.log`, `*override.tf`.

### C. Cross-state references
Use `terraform_remote_state` only for **cross-state/cross-team** reads (networking team publishes a VPC; app team consumes it). Within one state, use module outputs or data sources. Reference specific outputs, never the whole state; document the dependency in the README.

### D. Environments: workspaces vs separate state
- **Workspaces** — single backend, per-workspace `*.tfvars`; best when environments are identical bar size and share one cloud account. Blast radius: a backend misconfig hits every workspace.
- **Separate state files** (`environments/{dev,test,prod}/` each with its own `backend.tf`) — full isolation, different accounts/providers, independent access control; preferred for production isolation.

Choose workspaces for simple same-account setups; separate state when environments need different accounts, providers, or access controls. Migrating backends: `terraform state pull > backup.tfstate` → update `backend` → `terraform init -migrate-state` → `terraform plan` shows **no changes** → keep the backup ≥30 days.

---

## 7. Testing (owned: `terraform-testing`)

Tests are written **first** and bugs get a regression test **before** the fix — policy in [`tdd.md`](guides://tdd.md). Two complementary layers:

**Native `terraform test` (1.6+ / OpenTofu)** — HCL test files, no Go toolchain; `run` blocks default to plan, opt into `apply` for real resources.
```hcl
# tests/network.tftest.hcl
run "plan_is_valid" {
  command = plan
  variables { name = "test-net", network_cidr = "10.0.0.0/16" }
  assert {
    condition     = can(cidrhost(var.network_cidr, 0))
    error_message = "network_cidr must be a valid CIDR"
  }
}

run "apply_creates_subnets" {
  command = apply
  variables {
    name = "test-net", network_cidr = "10.0.0.0/16"
    public_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  }
  assert {
    condition     = length(output.public_subnet_ids) == 2
    error_message = "expected 2 public subnets"
  }
}
```

**Terratest (Go)** — full lifecycle against ephemeral infrastructure when you need to assert real provider behavior:
```go
func TestNetworkModule(t *testing.T) {
    t.Parallel()
    opts := &terraform.Options{
        TerraformDir: "../modules/network/examples/complete",
        Vars: map[string]any{"name": "test-" + random.UniqueId(), "network_cidr": "10.0.0.0/16"},
    }
    defer terraform.Destroy(t, opts)        // always clean up
    terraform.InitAndApply(t, opts)
    require.NotEmpty(t, terraform.Output(t, opts, "network_id"))
}
```

Rules: tests live in `tests/`, named per module; use `t.Parallel()` + `random.UniqueId()` to avoid collisions; ship both fast plan-only tests (no cost) and full apply→destroy tests; assert on outputs, resource counts, and naming. Regression tests name the bug ID in the assertion message.

---

## 8. Providers, Plan/Apply & Pipelines

### A. Provider & version pinning (TF-DEP-01)
Every module declares providers in `versions.tf` with pessimistic constraints; the root commits `.terraform.lock.hcl`.
```hcl
terraform {
  required_version = ">= 1.9.0"
  required_providers {
    aws    = { source = "hashicorp/aws",    version = "~> 5.0" }   # 5.x.x
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
}
```
Exact-pin (`= 5.40.0`) for critical prod stacks. `terraform init -upgrade` to bump within constraints; read the provider changelog before a major bump. Provider config (incl. `default_tags`) and `alias` (multi-region/multi-account) live in the root module and are passed to children via `providers = { aws = aws.secondary }`.

### B. Passwordless authentication
All provider auth is passwordless — OIDC federation in CI, instance/workload identity at runtime. Never put credentials in provider blocks, tfvars, or CI secrets. The policy is owned by [`secure-coding.md`](guides://secure-coding.md); cloud specifics live in [`aws.md`](guides://aws.md)/[`azure.md`](guides://azure.md)/[`gcp.md`](guides://gcp.md). Inject non-secret per-env values via `*.tfvars` or `TF_VAR_*`, secrets via vault **data sources** only (see `env-config.md`).

### C. Plan→apply in CI/CD (TF-PLAN-01)
Pipeline policy (stages, approvals, GitOps, OIDC setup) is owned by [`ci-cd.md`](guides://ci-cd.md); the `apply`-gated-on-merge branch/PR flow by [`git.md`](guides://git.md). Terraform-specific bindings:
- PRs run `fmt`/`validate`/`tflint`/`tfsec`+`checkov`/`plan`; the plan is posted as a **PR comment** and is non-destructive (no `-auto-approve`).
- `apply` runs only on `main` after merge, consuming the **saved plan artifact**; gate with a `concurrency` group so two applies never touch one state. `terraform plan -detailed-exitcode` (exit 2 = changes) decides whether apply runs.
- **Drift detection**: scheduled `plan -detailed-exitcode` (≥ once per business day for prod); alert on exit 2 and investigate within 24h; record intentional drift in an allowlist.
- **Policy as code**: enforce org rules pre-apply with OPA/Conftest (`terraform show -json tfplan | conftest test -`) or Sentinel; failures block the pipeline.
- **Cost**: `infracost` posts an estimate per PR; fail above a budget threshold.

Use `opentofu/setup-opentofu` + `tofu` commands in CI if OpenTofu is authoritative; pin the tool version identically across every stage.

---

## 9. Refactoring State Safely

Prefer declarative, reviewable blocks (1.5+) over imperative `terraform state` commands — they are version-controlled and show up in `plan`.

```hcl
import  { to = module.network.cloud_network.main  id = "existing-id-123" }   # adopt existing infra
moved   { from = cloud_network.legacy_vpc  to = cloud_network.main }          # rename / move w/o recreate
removed { from = cloud_network.legacy  lifecycle { destroy = false } }        # drop from state, keep infra (1.7+)
```

- **import**: write the matching `resource` first; `plan` until it shows *no changes*; apply; then delete the import block (one-time). `terraform plan -generate-config-out=gen.tf` scaffolds HCL — review and parameterize it before committing.
- **moved**: `plan` must show zero create/destroy; keep the block ≥2 release cycles so every environment applies it.
- **removed**: default `destroy = false`; keep ≥1 release cycle.
- Imperative `terraform state mv/rm/pull` only for emergencies — always `terraform state pull > backup.tfstate` first, then `plan` to confirm.

---

## 10. Tooling & Documentation

| Tool | Purpose | Gate |
|------|---------|------|
| `terraform fmt` / `validate` | format + static validation | TF-FMT-01 / TF-VAL-01 |
| `tflint` (`.tflint.hcl`) | lint + provider rulesets | TF-LINT-01 |
| `tfsec` + `checkov` | IaC security/compliance scan (policy: `secure-coding.md`) | TF-SEC-01 |
| `terraform-docs` (`.terraform-docs.yml`) | generate input/output tables into README | TF-DOC-01 |
| `infracost` | cost estimate in PRs | — |

Wire these into [`pre-commit`](guides://pre-commit.md) for local enforcement and re-run them in CI as the safety net:
```yaml
# .pre-commit-config.yaml
- repo: https://github.com/antonbabenko/pre-commit-terraform
  rev: v1.96.0
  hooks: [{id: terraform_fmt}, {id: terraform_validate}, {id: terraform_tflint},
          {id: terraform_tfsec}, {id: terraform_docs}, {id: terraform_checkov}]
```

Tag every resource for cost allocation and search (`Project`, `Environment`, `Owner`, `CostCenter`, `ManagedBy`) via a provider `default_tags` block or a merged `tags` map.

---

## 11. Quick Reference

```bash
terraform init                       # init (downloads providers, writes lockfile)
terraform fmt -recursive             # format
terraform validate                   # validate
terraform plan -var-file=env/dev.tfvars -out=tfplan   # preview (save the plan)
terraform apply tfplan               # apply the reviewed plan
terraform test                       # native tests
terraform workspace select prod      # switch environment
terraform state pull > backup.tfstate                 # back up state before any state op
TF_LOG=DEBUG terraform plan          # debug
```

---

## 12. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] TF-FMT-01 — `terraform fmt -check -recursive` clean
- [ ] TF-VAL-01 — `terraform validate` passes
- [ ] TF-LINT-01 — `tflint --recursive` clean
- [ ] TF-SEC-01 — `tfsec` + `checkov` 0 high/critical
- [ ] TF-SEC-02 — no static secrets; passwordless auth; secrets via data sources
- [ ] TF-TST-01/02 — tests pass (written first), bugs have regression tests
- [ ] TF-DEP-01 — version + provider + module sources pinned; `.terraform.lock.hcl` committed
- [ ] TF-STATE-01 — remote state encrypted, locked, out of VCS
- [ ] TF-PLAN-01 — reviewed plan precedes apply; no surprise destroys
- [ ] TF-DOC-01 — `terraform-docs` output in sync
- [ ] TF-STRUCT-01 — root owns provider+backend; child modules own neither
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Terraform Guidelines**
