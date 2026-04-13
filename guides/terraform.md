# Terraform Infrastructure as Code Guidelines (Provider-Agnostic)
Mandatory standards for Terraform development, infrastructure management, and IaC best practices. Cloud-agnostic (Azure, AWS, GCP). Terraform, OpenTofu, Terragrunt, tflint, tfsec, Checkov, Infracost.

---

**Agent Profile**: The Infrastructure Architect
**Role**: Senior DevOps Engineer & Cloud Infrastructure Specialist
**Objective**: Generate maintainable, secure, and scalable infrastructure code using Terraform.
**Tools**: Terraform, OpenTofu, Terragrunt, tflint, tfsec, Checkov, Infracost.

---

## 1. Core Philosophies: TERRAFORM-FIRST

The agent must adhere to the **TERRAFORM-FIRST** principles:

- **T**estable Infrastructure: All infrastructure must be testable and validated
- **E**nvironment Parity: Dev, Test, and Prod should be identical
- **R**eusable Modules: Build modular, reusable components
- **R**emote State: Always use remote state with locking
- **A**udit Trail: All changes tracked through version control
- **F**ail-Safe: Plan before apply, never skip confirmation
- **O**utputs Documented: Export all resources used/created by a module
- **R**eview Required: All infrastructure changes require review
- **M**inimal Permissions: Follow principle of least privilege

### Mandatory Global Rules

- **Hexagonal architecture** (ports & adapters) is required for all Terraform designs.
- **No locals** in Terraform files. Every value must be driven by variables, data sources, or direct expressions.
- **Every parameter** passed to external modules or commands must be bound to a **variable** with a **default value**.
- **Every resource used or created** by a module must be exported via **outputs**.
- **Passwordless authentication** is mandatory (managed identities, IAM roles, service accounts). No static secrets in code or pipelines.
- **Secrets** must be stored in a vault/secret store and referenced only via data sources.
- **Tags/labels** and **resource groups/projects/folders** must be used to group resources by function for easy search and selection.
- **Workspaces** must be used for Dev/Test/Prod. Code is shared; differences are only in workspace-specific `*.tfvars`.
- **Documentation & examples** must always be generated, kept up to date, and produced with **terraform-docs**.
- **Dependency management** must be explicit and clear. Use module outputs and `depends_on` only when required.
- **File naming convention**: `001-function-file-name.tf` (incremental prefix, short function description).
- **File scope**: One file = one function. Keep each file minimal; include only variables that are specific to that file (with defaults, types, and descriptions).
- **Variable standards**: Every variable must have `description`, `type`, and `default`.
- **Pre-commit** usage is **recommended** for fmt/validate/docs checks.

---

## 1A. OpenTofu Compatibility

### Overview

OpenTofu is an open-source fork of Terraform, maintained by the Linux Foundation. It was created in response to HashiCorp's license change from Mozilla Public License (MPL 2.0) to Business Source License (BSL 1.1) in August 2023. All guidelines in this document apply equally to both Terraform and OpenTofu unless noted otherwise.

### License Differences

| Aspect | Terraform | OpenTofu |
|--------|-----------|----------|
| License | BSL 1.1 (post 1.5.6) | MPL 2.0 (open source) |
| Governance | HashiCorp / IBM | Linux Foundation |
| Commercial use | Restricted for competing products | Unrestricted |
| CLI binary | `terraform` | `tofu` |
| Registry | registry.terraform.io | registry.opentofu.org (mirrors Terraform registry) |

### Compatibility Considerations

- OpenTofu maintains **drop-in compatibility** with Terraform for versions up through 1.5.x. Post-fork features may diverge.
- State files are **interchangeable** between Terraform and OpenTofu at compatible versions. However, once a state file is written by a newer version of either tool, downgrading may not be supported.
- Provider binaries are **identical** -- both tools use the same provider plugin protocol.
- Module syntax is **fully compatible**. Modules written for Terraform work with OpenTofu and vice versa.
- OpenTofu introduces some features ahead of Terraform (e.g., early support for state encryption, `removed` block). Verify feature parity before relying on tool-specific syntax.

### Migration Path

When migrating from Terraform to OpenTofu:

```bash
# 1. Install OpenTofu (via package manager or tfenv-compatible tooling)
# macOS
brew install opentofu

# Linux
curl --proto '=https' --tlsv1.2 -fsSL https://get.opentofu.org/install-opentofu.sh \
  -o install-opentofu.sh && chmod +x install-opentofu.sh && ./install-opentofu.sh

# 2. Replace terraform binary references in scripts and CI
# Replace: terraform init / plan / apply
# With:    tofu init / plan / apply

# 3. Verify state compatibility
tofu init
tofu plan  # Should produce identical plan to terraform plan

# 4. Update CI/CD pipelines
# Replace hashicorp/setup-terraform with opentofu/setup-opentofu
# Replace terraform commands with tofu commands

# 5. Update .terraform-version to .opentofu-version if using tofuenv
```

### Mandatory Rules for Dual Compatibility

- Do **not** use features exclusive to one tool unless the team has committed to a single tool.
- Pin tool versions in `.terraform-version` or `.opentofu-version` files.
- Document which tool is authoritative for state operations in the project README.
- When using CI/CD, configure the pipeline to use the same tool version across all stages.

---

## 2. Project Structure (MANDATORY)

### A. Standard Layout

```
infrastructure/
├── modules/                    # Reusable, domain-focused modules
│   ├── network/
│   ├── compute/
│   ├── database/
│   └── storage/
│
├── adapters/                   # Provider-specific adapters (optional)
│   ├── azure/
│   ├── aws/
│   └── gcp/
│
├── root/                       # Single root module
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── versions.tf
│   ├── providers.tf
│   ├── backend.tf
│   └── README.md
│
├── env/                        # Workspace-specific variables
│   ├── dev.tfvars
│   ├── test.tfvars
│   └── prod.tfvars
│
├── scripts/                    # Helper scripts
│   ├── validate.sh
│   └── plan.sh
│
├── tests/                      # Infrastructure tests
│   └── module_test.go
│
├── .terraform-version          # tfenv version
├── .tflint.hcl                 # Linter config
└── README.md
```

### B. Module Structure

```
modules/network/
├── main.tf           # Primary resources
├── variables.tf      # Input variables
├── outputs.tf        # Output values
├── versions.tf       # Provider requirements
├── data.tf           # Data sources (optional)
├── README.md         # Documentation
└── examples/         # Usage examples
    └── complete/
        └── main.tf
```

### C. Root vs Child Modules

**Root module** is the top-level working directory where `terraform init` and `terraform apply` are executed. It orchestrates child modules, configures providers, and manages state.

**Child modules** are reusable components invoked via `module` blocks. They should be self-contained, provider-agnostic where possible, and tested independently.

```hcl
# root/main.tf -- Root module orchestrates child modules
module "network" {
  source = "../modules/network"
  # ...
}

module "compute" {
  source     = "../modules/compute"
  network_id = module.network.network_id
  subnet_ids = module.network.private_subnet_ids
  # ...
}

module "database" {
  source     = "../modules/database"
  network_id = module.network.network_id
  subnet_ids = module.network.private_subnet_ids
  # ...
}
```

Rules for root modules:
- The root module contains **no resource blocks** directly. All resources live in child modules.
- Provider configuration belongs **only** in the root module.
- Backend configuration belongs **only** in the root module.
- The root module wires child modules together via outputs and variables.

Rules for child modules:
- Child modules must **never** contain `provider` blocks (use `required_providers` in `versions.tf` instead).
- Child modules must **never** contain `backend` blocks.
- Child modules must declare all required providers in `versions.tf`.
- Every input must have a `description`, `type`, and `default`.
- Every created or referenced resource must be exported via `outputs.tf`.

### D. Module Composition Patterns

#### Flat Composition

All child modules are invoked directly from the root module. Simple and easy to reason about.

```hcl
# root/main.tf
module "network"  { source = "../modules/network"  }
module "compute"  { source = "../modules/compute"  }
module "database" { source = "../modules/database" }
module "storage"  { source = "../modules/storage"  }
```

#### Nested Composition

A parent module composes multiple child modules. Use when a logical unit requires several resources that always deploy together.

```hcl
# modules/application/main.tf -- Parent module
module "compute" {
  source     = "../compute"
  subnet_ids = var.subnet_ids
  tags       = var.tags
}

module "load_balancer" {
  source     = "../load_balancer"
  target_ids = module.compute.instance_ids
  tags       = var.tags
}

# modules/application/outputs.tf
output "endpoint" {
  description = "Application load balancer endpoint"
  value       = module.load_balancer.dns_name
}
```

#### Facade Pattern

A thin wrapper module that presents a simplified interface over a complex child module. Useful for enforcing organizational defaults.

```hcl
# modules/standard_database/main.tf -- Facade over a generic database module
module "database" {
  source = "../database"

  # Enforce organizational standards
  engine_version    = "14.0"
  instance_class    = var.environment == "prod" ? "large" : "small"
  multi_az          = var.environment == "prod" ? true : false
  backup_retention  = var.environment == "prod" ? 30 : 7
  encryption        = true
  deletion_protection = var.environment == "prod" ? true : false

  # Pass through from caller
  name              = var.name
  allocated_storage = var.allocated_storage
  tags              = var.tags
}
```

### E. Module Versioning with Git Tags

All shared modules must be versioned using **semantic versioning** via git tags.

```hcl
# Pinning a module to a specific version tag
module "network" {
  source = "git::https://github.com/org/terraform-modules.git//modules/network?ref=v2.1.0"

  name         = var.name
  network_cidr = var.network_cidr
  tags         = var.tags
}
```

Version pinning rules:
- **Always** pin to a specific tag or commit hash. Never use `ref=main` or omit `ref` entirely.
- Use **semantic versioning**: `vMAJOR.MINOR.PATCH`.
  - MAJOR: Breaking changes to inputs, outputs, or behavior.
  - MINOR: New features, new optional variables, new outputs.
  - PATCH: Bug fixes, documentation updates.
- Tag modules in CI after all tests pass.
- Document breaking changes in a CHANGELOG.md at the module root.

```bash
# Tagging a module release
git tag -a v2.1.0 -m "network module: add IPv6 support"
git push origin v2.1.0
```

### F. Module Registry Publishing

For organizations using a private module registry (Terraform Cloud, Spacelift, or a self-hosted registry):

- Follow the **standard module structure**: `terraform-<PROVIDER>-<NAME>` repository naming.
- Include a root `main.tf`, `variables.tf`, `outputs.tf`, and `versions.tf`.
- Include at least one example under `examples/`.
- Generate documentation with `terraform-docs`.
- Publish only after all tests pass.

```hcl
# Consuming a module from a private registry
module "network" {
  source  = "app.terraform.io/my-org/network/cloud"
  version = "~> 2.1"

  name         = var.name
  network_cidr = var.network_cidr
  tags         = var.tags
}
```

Version constraint syntax:
- `= 2.1.0` -- Exact version.
- `~> 2.1` -- Any 2.x version >= 2.1.0 (pessimistic constraint).
- `>= 2.0, < 3.0` -- Range constraint.

### G. Module Testing with Terratest

Every module must have integration tests. Use Terratest (Go) for full lifecycle testing against real infrastructure.

```go
// tests/network_test.go
package test

import (
    "testing"
    "fmt"
    "github.com/gruntwork-io/terratest/modules/terraform"
    "github.com/gruntwork-io/terratest/modules/random"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestNetworkModuleComplete(t *testing.T) {
    t.Parallel()

    uniqueId := random.UniqueId()
    name := fmt.Sprintf("test-net-%s", uniqueId)

    terraformOptions := &terraform.Options{
        TerraformDir: "../modules/network/examples/complete",
        Vars: map[string]interface{}{
            "name":               name,
            "network_cidr":       "10.0.0.0/16",
            "availability_zones": []string{"zone-a", "zone-b"},
            "public_subnets":     []string{"10.0.1.0/24", "10.0.2.0/24"},
            "private_subnets":    []string{"10.0.11.0/24", "10.0.12.0/24"},
        },
    }

    // Always clean up
    defer terraform.Destroy(t, terraformOptions)

    // Validate before applying
    terraform.InitAndPlan(t, terraformOptions)

    // Apply and verify
    terraform.InitAndApply(t, terraformOptions)

    // Assert all outputs are populated
    networkId := terraform.Output(t, terraformOptions, "network_id")
    require.NotEmpty(t, networkId)

    publicSubnetIds := terraform.OutputList(t, terraformOptions, "public_subnet_ids")
    assert.Equal(t, 2, len(publicSubnetIds))

    privateSubnetIds := terraform.OutputList(t, terraformOptions, "private_subnet_ids")
    assert.Equal(t, 2, len(privateSubnetIds))

    sgId := terraform.Output(t, terraformOptions, "security_group_id")
    assert.NotEmpty(t, sgId)
}

// Plan-only test for fast feedback (no real infrastructure)
func TestNetworkModulePlanOnly(t *testing.T) {
    t.Parallel()

    terraformOptions := &terraform.Options{
        TerraformDir: "../modules/network/examples/complete",
        Vars: map[string]interface{}{
            "name":         "plan-test",
            "network_cidr": "10.0.0.0/16",
        },
        // PlanOnly ensures no resources are created
        PlanOnly: true,
    }

    // Validate the plan succeeds
    terraform.InitAndPlan(t, terraformOptions)
}
```

Test organization rules:
- Place tests in the `tests/` directory at the project root.
- Name test files after the module: `network_test.go`, `compute_test.go`.
- Use `t.Parallel()` for all tests to enable concurrent execution.
- Use `random.UniqueId()` to avoid name collisions in shared environments.
- Always use `defer terraform.Destroy()` to clean up resources.
- Include both **plan-only** tests (fast, no cost) and **full lifecycle** tests (apply + destroy).

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### TDD Cycle

```
1. RED: Write a failing test first
   ↓
2. GREEN: Write minimal code to make it pass
   ↓
3. REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for Terraform

```go
// Step 1: RED - Write failing Terratest
package test

import (
    "testing"
    "github.com/gruntwork-io/terratest/modules/terraform"
    "github.com/stretchr/testify/assert"
)

func TestVpcModule(t *testing.T) {
    t.Parallel()

    terraformOptions := &terraform.Options{
        TerraformDir: "../modules/network/examples/complete",
        Vars: map[string]interface{}{
            "name":           "test-vpc",
            "network_cidr":   "10.0.0.0/16",
            "public_subnets": []string{"10.0.1.0/24", "10.0.2.0/24"},
        },
    }

    defer terraform.Destroy(t, terraformOptions)

    // Validate plan before applying
    terraform.InitAndPlan(t, terraformOptions)

    terraform.InitAndApply(t, terraformOptions)

    // Assert outputs
    networkId := terraform.Output(t, terraformOptions, "network_id")
    assert.NotEmpty(t, networkId)

    vpcCidr := terraform.Output(t, terraformOptions, "network_cidr")
    assert.Equal(t, "10.0.0.0/16", vpcCidr)

    publicSubnetIds := terraform.OutputList(t, terraformOptions, "public_subnet_ids")
    assert.Equal(t, 2, len(publicSubnetIds))
}
// FAILS - module does not yet exist or outputs are missing

// Step 2: GREEN - Implement the module
// modules/network/main.tf
// resource "cloud_network" "main" {
//   name       = var.name
//   cidr_block = var.network_cidr
//   tags       = var.tags
// }
//
// modules/network/outputs.tf
// output "network_id"   { value = cloud_network.main.id }
// output "network_cidr" { value = cloud_network.main.cidr_block }
// output "public_subnet_ids" { value = cloud_subnet.public[*].id }
// PASSES

// Step 3: REFACTOR - Add validation, improve variable descriptions,
// run terraform fmt and terraform validate to ensure quality.
// All tests still PASS.
```

### Terraform-Specific TDD Practices

- Always run `terraform validate` and `terraform plan` as part of your test pipeline before `terraform apply`.
- Use Terratest (Go) for integration tests against real or ephemeral infrastructure.
- Use `pytest` with `python-terraform` for lightweight plan-level assertions.
- Test module outputs, resource counts, and naming conventions in every test.

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. Bug Reported/Discovered
   ↓
2. Write a test that REPRODUCES the bug (test will FAIL)
   ↓
3. Verify the test fails for the right reason
   ↓
4. Fix the bug (make the test pass)
   ↓
5. Verify the test now PASSES
   ↓
6. Document the bug in test comments (include bug ID)
   ↓
7. Deploy with confidence (regression prevented)
```

### Example Bug Fix

```go
// Bug Report: BUG-4521 - VPC module outputs empty subnet list
// when only one availability zone is provided.

func TestVpcModule_SingleAZ_Bug4521(t *testing.T) {
    t.Parallel()

    terraformOptions := &terraform.Options{
        TerraformDir: "../modules/network/examples/complete",
        Vars: map[string]interface{}{
            "name":               "test-vpc-single-az",
            "network_cidr":       "10.0.0.0/16",
            "availability_zones": []string{"zone-a"},
            "public_subnets":     []string{"10.0.1.0/24"},
            "private_subnets":    []string{"10.0.11.0/24"},
        },
    }

    defer terraform.Destroy(t, terraformOptions)
    terraform.InitAndApply(t, terraformOptions)

    // Regression: previously returned empty list for single AZ
    publicSubnetIds := terraform.OutputList(t, terraformOptions, "public_subnet_ids")
    assert.Equal(t, 1, len(publicSubnetIds), "BUG-4521: single AZ must still produce subnet output")

    privateSubnetIds := terraform.OutputList(t, terraformOptions, "private_subnet_ids")
    assert.Equal(t, 1, len(privateSubnetIds), "BUG-4521: single AZ must still produce subnet output")
}

// Fix: Updated the validation rule in variables.tf to allow >= 1
// availability zones instead of requiring >= 2.
```

### Prohibited Practices for Bug Fixes

**NEVER:**
- Fix a bug without adding a regression test first
- Write implementation before writing tests (violates TDD)
- Skip the Red-Green-Refactor cycle
- Commit code with failing tests
- Remove tests to make code pass
- Skip `terraform validate` and `terraform plan` before applying changes

---

## 3. Naming Conventions (MANDATORY)

### A. Resource Naming

```hcl
# Use consistent naming pattern: {project}-{environment}-{domain}-{purpose}

# ✅ CORRECT (placeholder resource types)
resource "cloud_network" "main" {
  name = "${var.project}-${var.environment}-network-main"
  tags = var.tags
}

resource "cloud_subnet" "private" {
  count = length(var.availability_zones)
  name  = "${var.project}-${var.environment}-subnet-private-${var.availability_zones[count.index]}"
  tags  = var.tags
}

resource "cloud_security_group" "web" {
  name = "${var.project}-${var.environment}-sg-web"
  tags = var.tags
}

# ❌ WRONG - Inconsistent, unclear naming
resource "cloud_network" "net1" { }
resource "cloud_subnet" "subnet" { }
```

### B. Variable Naming

```hcl
# variables.tf

# Use descriptive, lowercase names with underscores
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "app"
}

variable "environment" {
  description = "Environment (dev, test, prod)"
  type        = string
  default     = "dev"
  validation {
    condition     = contains(["dev", "test", "prod"], var.environment)
    error_message = "Environment must be dev, test, or prod."
  }
}

variable "network_cidr" {
  description = "CIDR block for the network"
  type        = string
  default     = "10.0.0.0/16"
}

variable "enable_public_endpoints" {
  description = "Whether public endpoints are enabled"
  type        = bool
  default     = false
}

variable "database_config" {
  description = "Database configuration"
  type = object({
    instance_class    = string
    allocated_storage = number
    engine_version    = string
    multi_az          = bool
  })
  default = {
    instance_class    = "standard"
    allocated_storage = 20
    engine_version    = "1.0"
    multi_az          = false
  }
}
```

---

## 4. Resource Configuration (MANDATORY)

### A. Network Module Example (Provider-Agnostic)

```hcl
# modules/network/main.tf

resource "cloud_network" "main" {
  name                = var.name
  cidr_block          = var.network_cidr
  resource_group_name = var.resource_group_name
  tags                = var.tags
}

resource "cloud_subnet" "public" {
  count               = length(var.public_subnets)
  network_id          = cloud_network.main.id
  cidr_block          = var.public_subnets[count.index]
  availability_zone   = var.availability_zones[count.index]
  resource_group_name = var.resource_group_name
  tags                = var.tags
}

resource "cloud_subnet" "private" {
  count               = length(var.private_subnets)
  network_id          = cloud_network.main.id
  cidr_block          = var.private_subnets[count.index]
  availability_zone   = var.availability_zones[count.index]
  resource_group_name = var.resource_group_name
  tags                = var.tags
}

resource "cloud_security_group" "app" {
  name                = "${var.name}-sg-app"
  network_id          = cloud_network.main.id
  resource_group_name = var.resource_group_name
  tags                = var.tags
}
```

### B. Variables with Validation (All Inputs Have Defaults)

```hcl
# modules/network/variables.tf

variable "name" {
  description = "Name prefix for all resources"
  type        = string
  default     = "app-dev"
  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.name))
    error_message = "Name must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "network_cidr" {
  description = "CIDR block for the network"
  type        = string
  default     = "10.0.0.0/16"
  validation {
    condition     = can(cidrhost(var.network_cidr, 0))
    error_message = "network_cidr must be a valid IPv4 CIDR block."
  }
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["zone-a", "zone-b"]
  validation {
    condition     = length(var.availability_zones) >= 2
    error_message = "At least 2 availability zones are required."
  }
}

variable "public_subnets" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnets" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.11.0/24", "10.0.12.0/24"]
}

variable "resource_group_name" {
  description = "Provider-specific grouping container name"
  type        = string
  default     = "rg-app-dev"
}

variable "tags" {
  description = "Tags/labels applied to all resources"
  type        = map(string)
  default     = {
    ManagedBy = "terraform"
    Project   = "app"
    Env       = "dev"
  }
}
```

### C. Outputs (Every Resource Exported)

```hcl
# modules/network/outputs.tf

output "network_id" {
  description = "ID of the network"
  value       = cloud_network.main.id
}

output "public_subnet_ids" {
  description = "IDs of public subnets"
  value       = cloud_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of private subnets"
  value       = cloud_subnet.private[*].id
}

output "security_group_id" {
  description = "ID of the security group"
  value       = cloud_security_group.app.id
}
```

---

## 5. State Management (MANDATORY)

### A. Remote Backend Configuration

- Remote state **must** be enabled.
- State **must** be locked and encrypted.
- Use a backend suitable for your provider (remote, s3, azurerm, gcs).

```hcl
# backend.tf (example: remote backend)
terraform {
  backend "remote" {
    organization = "example-org"
    workspaces {
      prefix = "app-"
    }
  }
}
```

### B. Remote State Backends by Provider

Choose the backend that matches your primary cloud provider. Each backend must enable locking, encryption, and versioning.

#### AWS S3 Backend

```hcl
# backend.tf
terraform {
  backend "s3" {
    bucket         = "myorg-terraform-state"
    key            = "infrastructure/network/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
    # Use a KMS key for server-side encryption
    kms_key_id     = "alias/terraform-state"
  }
}
```

Prerequisites for the S3 backend:
- S3 bucket with **versioning enabled** and **public access blocked**.
- DynamoDB table with partition key `LockID` (type: String) for state locking.
- KMS key for encryption. Do **not** rely on default S3 encryption alone.
- Bucket policy restricting access to the Terraform execution role only.

#### GCS Backend

```hcl
# backend.tf
terraform {
  backend "gcs" {
    bucket = "myorg-terraform-state"
    prefix = "infrastructure/network"
    # GCS provides built-in locking; no separate lock resource needed
  }
}
```

Prerequisites for the GCS backend:
- GCS bucket with **versioning enabled** and **uniform bucket-level access**.
- IAM policy restricting `roles/storage.objectAdmin` to the Terraform service account.
- Customer-managed encryption key (CMEK) recommended for production.

#### Azure Blob Backend

```hcl
# backend.tf
terraform {
  backend "azurerm" {
    resource_group_name  = "rg-terraform-state"
    storage_account_name = "myorgterraformstate"
    container_name       = "tfstate"
    key                  = "infrastructure/network/terraform.tfstate"
    use_oidc             = true
  }
}
```

Prerequisites for the Azure Blob backend:
- Storage account with **blob versioning** and **soft delete** enabled.
- Container with **private** access level.
- Azure Blob provides built-in lease-based locking.
- Use `use_oidc = true` for passwordless authentication from CI/CD.

### C. State Locking Strategies

State locking prevents concurrent operations from corrupting the state file.

**Mandatory rules:**
- State locking must **always** be enabled. Never use `-lock=false` in production.
- If a lock is stuck due to a failed run, use `terraform force-unlock <LOCK_ID>` only after verifying no other operation is in progress.
- CI/CD pipelines must include a timeout for lock acquisition. If the lock cannot be acquired within 5 minutes, fail the pipeline and alert the team.

```bash
# Force-unlock (use only after verifying no concurrent operations)
terraform force-unlock 799e13c4-823a-7e1c-cb15-f3b4218a9b65

# Check who holds the lock (backend-specific)
# S3/DynamoDB: Query the DynamoDB lock table
# GCS: Check the .tflock file in the bucket
# Azure: Check the blob lease status
```

### D. State File Security

The state file contains **sensitive information** including resource IDs, IP addresses, and potentially secrets (if outputs are not marked sensitive). Treat state files as confidential.

**Mandatory rules:**
- State files must **never** be committed to version control. Add `*.tfstate` and `*.tfstate.backup` to `.gitignore`.
- State backends must enforce **encryption at rest** and **encryption in transit** (TLS).
- Access to the state backend must follow **least privilege**. Only the Terraform execution identity (CI/CD service account or automation role) should have read/write access.
- Mark outputs containing sensitive values with `sensitive = true`.

```hcl
# outputs.tf -- Mark sensitive outputs
output "database_connection_string" {
  description = "Database connection string"
  value       = module.database.connection_string
  sensitive   = true
}

output "api_key" {
  description = "Generated API key"
  value       = module.api.key
  sensitive   = true
}
```

```gitignore
# .gitignore -- Always exclude state files
*.tfstate
*.tfstate.backup
*.tfstate.*.backup
.terraform/
.terraform.lock.hcl
crash.log
override.tf
override.tf.json
*_override.tf
*_override.tf.json
```

### E. State Migration Between Backends

When migrating state from one backend to another (e.g., local to S3, or S3 to Terraform Cloud):

```bash
# Step 1: Back up current state
terraform state pull > terraform.tfstate.backup

# Step 2: Update backend.tf with the new backend configuration

# Step 3: Reinitialize with migration flag
terraform init -migrate-state

# Step 4: Verify the state was migrated correctly
terraform plan  # Should show "No changes"

# Step 5: Remove the old backend state (after verification)
```

**Mandatory rules:**
- Always create a **local backup** of the state before migration.
- After migration, run `terraform plan` to confirm no changes are detected.
- Keep the backup for at least 30 days after successful migration.
- Document the migration in the project's change log.

### F. terraform_remote_state Data Source

Use `terraform_remote_state` to read outputs from another Terraform state. This enables cross-project or cross-team data sharing without tight coupling.

```hcl
# Read network outputs from a separate state file
data "terraform_remote_state" "network" {
  backend = "s3"
  config = {
    bucket = "myorg-terraform-state"
    key    = "infrastructure/network/terraform.tfstate"
    region = "us-east-1"
  }
}

# Use the outputs
module "compute" {
  source = "../modules/compute"

  network_id = data.terraform_remote_state.network.outputs.network_id
  subnet_ids = data.terraform_remote_state.network.outputs.private_subnet_ids
  tags       = var.tags
}
```

**Guidelines for `terraform_remote_state`:**
- Prefer **data sources** or **module outputs** over `terraform_remote_state` when resources are managed in the same state.
- Use `terraform_remote_state` only for **cross-state** references (e.g., networking team provides VPC, application team consumes it).
- Always reference **specific outputs** rather than the entire state.
- Document all cross-state dependencies in the project README.

### G. Workspaces vs Separate State Files

Two strategies exist for managing multiple environments:

#### Workspaces (Recommended for Simple Setups)

```bash
terraform workspace new dev
terraform workspace new test
terraform workspace new prod
terraform workspace select prod
terraform plan -var-file=env/prod.tfvars
```

Advantages:
- Single backend configuration.
- Easy to switch between environments.
- Shared codebase with per-workspace variable files.

Limitations:
- All workspaces share the same backend credentials.
- Blast radius: a misconfigured backend affects all environments.
- Not suitable when environments require different provider configurations or accounts.

#### Separate State Files (Recommended for Production Isolation)

```
infrastructure/
├── environments/
│   ├── dev/
│   │   ├── main.tf        # Calls shared modules
│   │   ├── backend.tf     # Points to dev state
│   │   └── terraform.tfvars
│   ├── test/
│   │   ├── main.tf
│   │   ├── backend.tf     # Points to test state
│   │   └── terraform.tfvars
│   └── prod/
│       ├── main.tf
│       ├── backend.tf     # Points to prod state (different account)
│       └── terraform.tfvars
└── modules/               # Shared modules
```

Advantages:
- Complete isolation between environments.
- Different cloud accounts/subscriptions per environment.
- Independent state locking and access control.
- Reduced blast radius.

**Choose workspaces** when environments are identical except for size/scale and live in the same cloud account. **Choose separate state files** when environments require different accounts, providers, or access controls.

### H. Backend Configuration per Workspace

Use `-backend-config` files (one per workspace) and keep them outside versioned secrets.

```bash
terraform init -backend-config=env/${TF_WORKSPACE}.backend.hcl
```

---

## 6. Security Best Practices (MANDATORY)

### A. Passwordless Authentication

- Azure: System Managed Identity (preferred), User Managed Identity
- AWS: IAM Roles with Instance Profiles / OIDC
- GCP: Service Accounts / Workload Identity

Static credentials are **not allowed** in code or CI/CD.

### B. Secrets Management

- Secrets must live in a **vault/secret store**.
- Retrieve secrets only via **data sources**.

```hcl
# Example placeholder (replace with provider-specific secret store)
# data "secret_store" "db_password" { key = "prod/db/password" }
```

### C. Least Privilege

- Assign only the permissions required for Terraform and workloads.
- Avoid wildcards and broad admin roles.

---

## 7. Environment Management

### A. Root Module Configuration (Provider-Agnostic)

```hcl
# root/main.tf

terraform {
  required_version = ">= 1.5.0"
}

# Provider configuration lives here and is specific to your cloud.
# All values must be variables with defaults.

module "network" {
  source = var.network_module_source

  name                = var.name
  network_cidr        = var.network_cidr
  availability_zones  = var.availability_zones
  public_subnets      = var.public_subnets
  private_subnets     = var.private_subnets
  resource_group_name = var.resource_group_name
  tags                = var.tags
}
```

### B. Workspace Variables

```hcl
# env/prod.tfvars

project             = "app"
environment         = "prod"
name                = "app-prod"
network_cidr        = "10.0.0.0/16"
availability_zones  = ["zone-a", "zone-b"]
public_subnets      = ["10.0.1.0/24", "10.0.2.0/24"]
private_subnets     = ["10.0.11.0/24", "10.0.12.0/24"]
resource_group_name = "rg-app-prod"
```

### C. No Locals

Locals are not permitted. Use variables or direct expressions only.

---

## 7A. Import and Moved Blocks (MANDATORY)

Modern Terraform (1.5+) provides declarative blocks for importing existing infrastructure, refactoring resource addresses, and safely removing resources from state. These blocks are **preferred** over imperative `terraform import` and `terraform state mv` commands because they are reviewable, auditable, and version-controlled.

### A. Import Block (Terraform 1.5+)

Use the `import` block to bring existing infrastructure under Terraform management without destroying and recreating it.

```hcl
# import.tf -- Import an existing network into Terraform state
import {
  to = module.network.cloud_network.main
  id = "existing-network-id-12345"
}

# The corresponding resource block must exist in the module
# resource "cloud_network" "main" {
#   name       = var.name
#   cidr_block = var.network_cidr
#   tags       = var.tags
# }
```

Workflow for importing resources:

```bash
# Step 1: Write the resource block that matches the existing infrastructure
# Step 2: Add the import block with the resource address and provider-specific ID
# Step 3: Run plan to verify the import will not cause changes
terraform plan

# The plan output will show:
# cloud_network.main: Importing... [id=existing-network-id-12345]
# cloud_network.main: Import complete
# No changes. Infrastructure is up-to-date.

# Step 4: Apply to execute the import
terraform apply

# Step 5: Remove the import block after successful import
# Import blocks are one-time operations; remove them after apply
```

**Mandatory rules for imports:**
- Always run `terraform plan` before `terraform apply` to verify the import will not trigger unintended changes.
- Write the resource configuration to **match the existing infrastructure exactly**. Adjust the HCL until the plan shows no changes after import.
- Remove import blocks from code after the import is applied. They are one-time operations.
- Document all imports in commit messages with the resource ID and reason.

#### Generating Configuration for Imports

Terraform 1.5+ can generate HCL configuration from imported resources:

```bash
# Generate configuration for an import (writes to generated_imports.tf)
terraform plan -generate-config-out=generated_imports.tf
```

Review and refine the generated configuration before committing. Generated code often needs cleanup: removing read-only attributes, replacing hardcoded values with variables, and aligning with naming conventions.

### B. Moved Block

Use the `moved` block to **refactor resource addresses** without destroying and recreating infrastructure. This is essential for renaming resources, moving resources into or out of modules, and reorganizing code.

```hcl
# moved.tf -- Rename a resource
moved {
  from = cloud_network.legacy_vpc
  to   = cloud_network.main
}

# Move a resource into a module
moved {
  from = cloud_network.main
  to   = module.network.cloud_network.main
}

# Move a resource from one module to another
moved {
  from = module.old_network.cloud_subnet.private
  to   = module.network.cloud_subnet.private
}

# Rename a module
moved {
  from = module.vpc
  to   = module.network
}

# Move an indexed resource to a for_each key
moved {
  from = cloud_subnet.public[0]
  to   = cloud_subnet.public["us-east-1a"]
}
```

**Mandatory rules for moved blocks:**
- Always run `terraform plan` after adding a `moved` block to verify the operation is correct. The plan should show **zero** creates or destroys -- only moves.
- Keep `moved` blocks in the codebase for **at least two release cycles** so all environments pick up the change.
- After all environments have applied the move, the `moved` blocks can be removed.
- Document each move with a comment explaining the reason for refactoring.

### C. Removed Block (Terraform 1.7+ / OpenTofu 1.7+)

Use the `removed` block to **stop managing a resource** without destroying it. This is the declarative replacement for `terraform state rm`.

```hcl
# removed.tf -- Stop managing a resource without destroying it
removed {
  from = cloud_network.legacy

  lifecycle {
    destroy = false
  }
}

# Stop managing an entire module
removed {
  from = module.deprecated_service

  lifecycle {
    destroy = false
  }
}
```

When `destroy = false`, Terraform removes the resource from state but does **not** destroy the actual infrastructure. Set `destroy = true` only when you intend to both remove from state and destroy the real resource.

**Mandatory rules for removed blocks:**
- Default to `destroy = false` unless intentional destruction is required and reviewed.
- Document the reason for removal in a comment and the commit message.
- Keep `removed` blocks for at least **one release cycle** so all environments process them.

### D. Imperative State Manipulation (Use Sparingly)

The imperative `terraform state` commands are still available but should be used **only** when declarative blocks are insufficient (e.g., emergency operations, complex cross-state moves).

```bash
# Move a resource address within state
terraform state mv 'cloud_network.old' 'cloud_network.new'

# Move a resource into a module
terraform state mv 'cloud_network.main' 'module.network.cloud_network.main'

# Remove a resource from state (does NOT destroy infrastructure)
terraform state rm 'cloud_network.legacy'

# Move a resource between state files
terraform state mv -state=old.tfstate -state-out=new.tfstate \
  'cloud_network.main' 'cloud_network.main'
```

**Mandatory rules for imperative state commands:**
- Always create a state backup before any imperative state operation: `terraform state pull > backup.tfstate`
- Imperative state commands are **not tracked in version control**. Document the operation in a commit message or runbook.
- Prefer declarative `moved` and `removed` blocks for all routine refactoring.
- Run `terraform plan` after any state manipulation to verify correctness.

---

## 7B. Advanced HCL Patterns

### A. Dynamic Blocks

Use `dynamic` blocks to generate repeated nested blocks from a collection. Avoid excessive nesting; prefer flat structures where possible.

```hcl
# Generate security group rules dynamically
variable "ingress_rules" {
  description = "List of ingress rules for the security group"
  type = list(object({
    port        = number
    protocol    = string
    cidr_blocks = list(string)
    description = string
  }))
  default = [
    {
      port        = 443
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
      description = "HTTPS from anywhere"
    },
    {
      port        = 80
      protocol    = "tcp"
      cidr_blocks = ["10.0.0.0/8"]
      description = "HTTP from internal only"
    }
  ]
}

resource "cloud_security_group" "app" {
  name       = "${var.project}-${var.environment}-sg-app"
  network_id = var.network_id
  tags       = var.tags

  dynamic "ingress" {
    for_each = var.ingress_rules
    content {
      from_port   = ingress.value.port
      to_port     = ingress.value.port
      protocol    = ingress.value.protocol
      cidr_blocks = ingress.value.cidr_blocks
      description = ingress.value.description
    }
  }
}
```

**Rules for dynamic blocks:**
- Use only when the number of nested blocks is **variable** and driven by input data.
- Do **not** use dynamic blocks when the nested blocks are static and known at authoring time.
- Limit dynamic block nesting to **one level**. If you need nested dynamic blocks, refactor into a separate module.

### B. for_each vs count

**Prefer `for_each`** over `count` for most use cases. `for_each` creates resources keyed by map keys or set elements, which prevents reordering issues when elements are added or removed.

```hcl
# PREFERRED: for_each with a map -- stable resource addresses
variable "subnets" {
  description = "Map of subnet configurations"
  type = map(object({
    cidr_block        = string
    availability_zone = string
    public            = bool
  }))
  default = {
    "public-a" = {
      cidr_block        = "10.0.1.0/24"
      availability_zone = "zone-a"
      public            = true
    }
    "public-b" = {
      cidr_block        = "10.0.2.0/24"
      availability_zone = "zone-b"
      public            = true
    }
    "private-a" = {
      cidr_block        = "10.0.11.0/24"
      availability_zone = "zone-a"
      public            = false
    }
  }
}

resource "cloud_subnet" "this" {
  for_each = var.subnets

  name              = "${var.project}-${var.environment}-subnet-${each.key}"
  cidr_block        = each.value.cidr_block
  availability_zone = each.value.availability_zone
  network_id        = var.network_id
  tags              = var.tags
}

# Addresses: cloud_subnet.this["public-a"], cloud_subnet.this["public-b"], etc.
# Removing "public-b" only destroys that subnet, not others.
```

```hcl
# ACCEPTABLE: count for simple enable/disable patterns
variable "enable_monitoring" {
  description = "Whether to create monitoring resources"
  type        = bool
  default     = true
}

resource "cloud_monitoring_dashboard" "main" {
  count = var.enable_monitoring ? 1 : 0
  name  = "${var.project}-${var.environment}-dashboard"
  tags  = var.tags
}
```

**When to use `count`:**
- Binary enable/disable: `count = var.enable_feature ? 1 : 0`
- Creating a fixed number of identical resources where order does not matter.

**When to use `for_each`:**
- Creating resources from a map or set of distinct items.
- Any time adding or removing elements should not affect other resources.

### C. Complex Variable Types

Use structured variable types to keep module interfaces clean and self-documenting.

```hcl
# Map of objects for multi-environment database configuration
variable "databases" {
  description = "Map of database configurations"
  type = map(object({
    engine            = string
    engine_version    = string
    instance_class    = string
    allocated_storage = number
    multi_az          = bool
    backup_retention  = number
    tags              = map(string)
  }))
  default = {
    "orders" = {
      engine            = "postgres"
      engine_version    = "14.0"
      instance_class    = "standard"
      allocated_storage = 50
      multi_az          = true
      backup_retention  = 30
      tags              = { Service = "orders" }
    }
    "users" = {
      engine            = "postgres"
      engine_version    = "14.0"
      instance_class    = "small"
      allocated_storage = 20
      multi_az          = false
      backup_retention  = 7
      tags              = { Service = "users" }
    }
  }
}

# Create a database for each entry
resource "cloud_database" "this" {
  for_each = var.databases

  name              = "${var.project}-${var.environment}-db-${each.key}"
  engine            = each.value.engine
  engine_version    = each.value.engine_version
  instance_class    = each.value.instance_class
  allocated_storage = each.value.allocated_storage
  multi_az          = each.value.multi_az
  backup_retention  = each.value.backup_retention
  tags              = merge(var.tags, each.value.tags)
}
```

```hcl
# Optional nested objects with defaults
variable "monitoring_config" {
  description = "Monitoring configuration"
  type = object({
    enabled         = bool
    retention_days  = number
    alert_endpoints = list(string)
  })
  default = {
    enabled         = true
    retention_days  = 90
    alert_endpoints = []
  }
}
```

### D. Terraform Functions

Use built-in functions to transform data. Keep expressions readable; extract complex transformations into well-named variables (when locals are permitted) or outputs.

```hcl
# lookup -- Safe map access with default
variable "instance_sizes" {
  description = "Map of environment to instance size"
  type        = map(string)
  default = {
    dev  = "small"
    test = "medium"
    prod = "large"
  }
}

resource "cloud_compute" "app" {
  instance_class = lookup(var.instance_sizes, var.environment, "small")
  tags           = var.tags
}

# merge -- Combine maps (later maps override earlier keys)
resource "cloud_compute" "web" {
  tags = merge(
    var.tags,
    {
      Role = "web"
      Tier = "frontend"
    }
  )
}

# flatten -- Flatten nested lists
variable "team_permissions" {
  description = "Map of teams to their permitted actions"
  type        = map(list(string))
  default = {
    dev    = ["read", "write"]
    ops    = ["read", "write", "admin"]
    viewer = ["read"]
  }
}

output "all_permissions" {
  description = "Flattened list of all unique permissions"
  value       = distinct(flatten(values(var.team_permissions)))
}

# coalesce -- Return first non-null/non-empty value
resource "cloud_compute" "app" {
  name = coalesce(var.custom_name, "${var.project}-${var.environment}-app")
  tags = var.tags
}

# try -- Gracefully handle missing attributes
output "endpoint" {
  description = "Application endpoint"
  value       = try(cloud_load_balancer.main[0].dns_name, "none")
}

# templatefile -- Render a template with variables
resource "cloud_compute" "app" {
  user_data = templatefile("${path.module}/templates/init.sh.tpl", {
    environment = var.environment
    region      = var.region
    app_version = var.app_version
  })
  tags = var.tags
}
```

### E. Conditional Resources with count

Use `count` with a ternary expression to conditionally create resources.

```hcl
# Create a public IP only in production
resource "cloud_public_ip" "app" {
  count = var.environment == "prod" ? 1 : 0
  name  = "${var.project}-${var.environment}-pip-app"
  tags  = var.tags
}

# Reference conditional resources safely
resource "cloud_load_balancer" "app" {
  name      = "${var.project}-${var.environment}-lb-app"
  public_ip = var.environment == "prod" ? cloud_public_ip.app[0].id : null
  tags      = var.tags
}
```

```hcl
# Create resources only when a list is non-empty
variable "additional_cidrs" {
  description = "Additional CIDR blocks to allow"
  type        = list(string)
  default     = []
}

resource "cloud_security_rule" "additional" {
  count       = length(var.additional_cidrs) > 0 ? length(var.additional_cidrs) : 0
  cidr_block  = var.additional_cidrs[count.index]
  description = "Additional CIDR allowance"
}
```

### F. For Expressions

Use `for` expressions to transform collections inline.

```hcl
# Transform a list into a map
variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["zone-a", "zone-b", "zone-c"]
}

# Create a map of zone name to index for use with for_each
output "zone_map" {
  description = "Map of zone names to indices"
  value       = { for idx, zone in var.availability_zones : zone => idx }
}

# Filter a map
variable "all_instances" {
  description = "Map of all instance configurations"
  type = map(object({
    size    = string
    enabled = bool
  }))
  default = {
    web  = { size = "small", enabled = true }
    api  = { size = "medium", enabled = true }
    cron = { size = "small", enabled = false }
  }
}

# Only create enabled instances
resource "cloud_compute" "this" {
  for_each = { for k, v in var.all_instances : k => v if v.enabled }

  name           = "${var.project}-${var.environment}-${each.key}"
  instance_class = each.value.size
  tags           = var.tags
}

# Transform list of objects to map keyed by name
output "instance_ids" {
  description = "Map of instance name to ID"
  value       = { for k, v in cloud_compute.this : k => v.id }
}
```

---

## 7C. Provider Patterns (MANDATORY)

### A. Required Providers Block

Every module must declare its required providers in `versions.tf`. This block specifies which providers the module depends on, their source, and version constraints.

```hcl
# versions.tf (in every module)
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    cloud = {
      source  = "hashicorp/cloud"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}
```

**Version constraint rules:**
- Always use **pessimistic constraints** (`~>`) to allow patch updates while preventing breaking changes.
- Pin to a **major.minor** range: `~> 4.0` allows 4.0.0 through 4.x.x.
- For critical production infrastructure, pin to an **exact version**: `= 4.12.0`.
- Run `terraform init -upgrade` periodically to pick up new versions within constraints.
- Review the provider changelog before upgrading major versions.

### B. Provider Configuration in Root Modules

Provider configuration belongs **only** in root modules. Child modules inherit providers from their callers.

```hcl
# root/providers.tf
provider "cloud" {
  region = var.region

  default_tags {
    tags = {
      Project     = var.project
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}
```

### C. Provider Aliases for Multi-Region and Multi-Account

Use provider aliases when you need to manage resources in multiple regions or accounts within the same configuration.

```hcl
# root/providers.tf -- Multi-region configuration
provider "cloud" {
  alias  = "primary"
  region = var.primary_region
}

provider "cloud" {
  alias  = "secondary"
  region = var.secondary_region
}

# Pass aliased providers to modules
module "network_primary" {
  source = "../modules/network"
  providers = {
    cloud = cloud.primary
  }
  name         = "${var.project}-primary"
  network_cidr = var.primary_cidr
  tags         = var.tags
}

module "network_secondary" {
  source = "../modules/network"
  providers = {
    cloud = cloud.secondary
  }
  name         = "${var.project}-secondary"
  network_cidr = var.secondary_cidr
  tags         = var.tags
}
```

```hcl
# Multi-account configuration (e.g., shared services account + workload account)
provider "cloud" {
  alias   = "shared"
  region  = var.region
  profile = "shared-services"
}

provider "cloud" {
  alias   = "workload"
  region  = var.region
  profile = "workload"
}

module "dns" {
  source = "../modules/dns"
  providers = {
    cloud = cloud.shared
  }
  domain_name = var.domain_name
}

module "application" {
  source = "../modules/application"
  providers = {
    cloud = cloud.workload
  }
  dns_zone_id = module.dns.zone_id
}
```

### D. Provider Authentication Patterns

All provider authentication must be **passwordless**. Static credentials are never permitted in code, environment variables, or CI/CD secrets.

```hcl
# AWS -- Use OIDC federation for CI/CD
provider "aws" {
  region = var.region
  # No credentials in provider block
  # Authentication via:
  #   - OIDC federation (GitHub Actions, GitLab CI)
  #   - Instance profile (EC2)
  #   - ECS task role (ECS/Fargate)
}

# Azure -- Use OIDC or Managed Identity
provider "azurerm" {
  features {}
  use_oidc        = true        # For CI/CD with federated credentials
  # use_msi       = true        # For Managed Identity on Azure VMs
  subscription_id = var.subscription_id
  tenant_id       = var.tenant_id
}

# GCP -- Use Workload Identity Federation
provider "google" {
  project = var.project_id
  region  = var.region
  # Authentication via:
  #   - Workload Identity Federation (CI/CD)
  #   - Attached service account (GCE/GKE)
}
```

**CI/CD authentication setup:**

```yaml
# GitHub Actions -- AWS OIDC
- name: Configure AWS Credentials
  uses: aws-actions/configure-aws-credentials@v4
  with:
    role-to-assume: arn:aws:iam::123456789012:role/terraform-deploy
    aws-region: us-east-1

# GitHub Actions -- Azure OIDC
- name: Azure Login
  uses: azure/login@v2
  with:
    client-id: ${{ secrets.AZURE_CLIENT_ID }}
    tenant-id: ${{ secrets.AZURE_TENANT_ID }}
    subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}

# GitHub Actions -- GCP Workload Identity
- name: Authenticate to Google Cloud
  uses: google-github-actions/auth@v2
  with:
    workload_identity_provider: projects/123456789/locations/global/workloadIdentityPools/github/providers/github
    service_account: terraform@my-project.iam.gserviceaccount.com
```

---

## 8. Testing Infrastructure

### A. Terraform Validate and Format

```bash
#!/bin/bash
# scripts/validate.sh

set -e

echo "Formatting Terraform files..."
terraform fmt -recursive -check

echo "Validating Terraform configuration..."
for ws in dev test prod; do
  terraform workspace select "$ws" || terraform workspace new "$ws"
  terraform init -backend=false
  terraform validate
  terraform plan -var-file="env/${ws}.tfvars" -out="tfplan-${ws}"
done

echo "Running tflint..."
tflint --recursive

echo "Running tfsec..."
tfsec .

echo "Running checkov..."
checkov -d .

echo "All validations passed!"
```

### B. Terratest

```go
// tests/module_test.go
package test

import (
    "testing"
    "github.com/gruntwork-io/terratest/modules/terraform"
    "github.com/stretchr/testify/assert"
)

func TestNetworkModule(t *testing.T) {
    t.Parallel()

    terraformOptions := &terraform.Options{
        TerraformDir: "../modules/network/examples/complete",
        Vars: map[string]interface{}{
          "name":               "test-network",
          "network_cidr":       "10.0.0.0/16",
          "availability_zones": []string{"zone-a", "zone-b"},
          "public_subnets":     []string{"10.0.1.0/24", "10.0.2.0/24"},
          "private_subnets":    []string{"10.0.11.0/24", "10.0.12.0/24"},
        },
    }

    defer terraform.Destroy(t, terraformOptions)
    terraform.InitAndApply(t, terraformOptions)

    networkId := terraform.Output(t, terraformOptions, "network_id")
    assert.NotEmpty(t, networkId)

    publicSubnetIds := terraform.OutputList(t, terraformOptions, "public_subnet_ids")
    assert.Equal(t, 2, len(publicSubnetIds))

    privateSubnetIds := terraform.OutputList(t, terraformOptions, "private_subnet_ids")
    assert.Equal(t, 2, len(privateSubnetIds))
}
```

---

## 9. CI/CD Integration

### A. GitHub Actions (Provider-Agnostic)

```yaml
# .github/workflows/terraform.yml
name: Terraform

on:
  pull_request:
    paths: ['infrastructure/**']
  push:
    branches: [main]
    paths: ['infrastructure/**']

env:
  TF_VERSION: "1.6.0"

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}
      - name: Terraform Format
        run: terraform fmt -check -recursive
      - name: Terraform Validate
        run: terraform validate

  plan:
    needs: validate
    runs-on: ubuntu-latest
    strategy:
      matrix:
        workspace: [dev, test, prod]
    steps:
      - uses: actions/checkout@v4
      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}

      # Configure cloud credentials via OIDC/Workload Identity (provider-specific)

      - name: Terraform Init
        run: terraform init

      - name: Terraform Workspace
        run: terraform workspace select ${{ matrix.workspace }} || terraform workspace new ${{ matrix.workspace }}

      - name: Terraform Plan
        run: terraform plan -var-file=env/${{ matrix.workspace }}.tfvars -out=tfplan-${{ matrix.workspace }}
```

### B. Plan/Apply Workflow in CI

The CI/CD pipeline must enforce a strict **plan-then-apply** workflow. Plans are generated on pull requests; applies are executed only after merge to the main branch with manual or automated approval.

```yaml
# .github/workflows/terraform-apply.yml
name: Terraform Apply

on:
  push:
    branches: [main]
    paths: ['infrastructure/**']

concurrency:
  group: terraform-apply-${{ github.ref }}
  cancel-in-progress: false

env:
  TF_VERSION: "1.6.0"

jobs:
  apply:
    runs-on: ubuntu-latest
    environment: production  # Requires manual approval in GitHub
    strategy:
      max-parallel: 1       # Apply environments sequentially
      matrix:
        workspace: [dev, test, prod]
    steps:
      - uses: actions/checkout@v4
      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}

      # Configure cloud credentials via OIDC (provider-specific step)

      - name: Terraform Init
        run: terraform init
        working-directory: infrastructure/root

      - name: Select Workspace
        run: terraform workspace select ${{ matrix.workspace }}
        working-directory: infrastructure/root

      - name: Terraform Plan
        run: |
          terraform plan \
            -var-file=../env/${{ matrix.workspace }}.tfvars \
            -out=tfplan-${{ matrix.workspace }} \
            -detailed-exitcode
        working-directory: infrastructure/root
        id: plan

      - name: Terraform Apply
        if: steps.plan.outputs.exitcode == '2'  # Changes detected
        run: terraform apply -auto-approve tfplan-${{ matrix.workspace }}
        working-directory: infrastructure/root
```

**Mandatory CI/CD rules:**
- Plans on pull requests must be **non-destructive** (no `-auto-approve`).
- Plans must be posted as **PR comments** for reviewer visibility.
- Applies must **only** execute on the main branch after merge.
- Use `concurrency` groups to prevent parallel applies to the same state.
- Use `-detailed-exitcode` to detect whether changes exist (exit code 2 = changes pending).
- Store plan files as artifacts when using separate plan and apply jobs.

### C. Drift Detection

Schedule periodic drift detection to identify manual changes made outside Terraform.

```yaml
# .github/workflows/drift-detection.yml
name: Drift Detection

on:
  schedule:
    - cron: '0 6 * * 1-5'  # Weekdays at 6 AM UTC
  workflow_dispatch: {}

env:
  TF_VERSION: "1.6.0"

jobs:
  drift:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        workspace: [dev, test, prod]
    steps:
      - uses: actions/checkout@v4
      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}

      # Configure cloud credentials

      - name: Terraform Init
        run: terraform init
        working-directory: infrastructure/root

      - name: Select Workspace
        run: terraform workspace select ${{ matrix.workspace }}
        working-directory: infrastructure/root

      - name: Detect Drift
        id: drift
        run: |
          terraform plan \
            -var-file=../env/${{ matrix.workspace }}.tfvars \
            -detailed-exitcode \
            -out=drift-${{ matrix.workspace }}.plan 2>&1 | tee plan-output.txt
          echo "exitcode=$?" >> "$GITHUB_OUTPUT"
        working-directory: infrastructure/root
        continue-on-error: true

      - name: Alert on Drift
        if: steps.drift.outputs.exitcode == '2'
        run: |
          echo "::warning::Drift detected in ${{ matrix.workspace }} environment"
          # Send alert to Slack, PagerDuty, or create a GitHub issue
```

**Drift detection rules:**
- Run drift detection at least **once per business day** for production.
- Alert the infrastructure team when drift is detected.
- Investigate drift within **24 hours** of detection.
- Document intentional drift exceptions (e.g., auto-scaling changes) in a drift allowlist.

### D. Policy as Code (Sentinel / OPA)

Use policy-as-code frameworks to enforce organizational standards before infrastructure changes are applied.

#### Open Policy Agent (OPA) with Conftest

```bash
# Install conftest
brew install conftest

# Run policy checks against a Terraform plan
terraform plan -out=tfplan.binary
terraform show -json tfplan.binary > tfplan.json
conftest test tfplan.json -p policy/
```

```rego
# policy/tags.rego -- Enforce mandatory tags
package main

mandatory_tags := {"Project", "Environment", "Owner", "ManagedBy"}

deny[msg] {
    resource := input.resource_changes[_]
    resource.change.after.tags != null
    existing_tags := {key | resource.change.after.tags[key]}
    missing := mandatory_tags - existing_tags
    count(missing) > 0
    msg := sprintf(
        "Resource %s is missing required tags: %v",
        [resource.address, missing]
    )
}

# policy/security.rego -- Block public access
deny[msg] {
    resource := input.resource_changes[_]
    resource.type == "cloud_storage_bucket"
    resource.change.after.public_access == true
    msg := sprintf(
        "Resource %s must not have public access enabled",
        [resource.address]
    )
}

# policy/cost.rego -- Block oversized instances in non-prod
deny[msg] {
    resource := input.resource_changes[_]
    resource.type == "cloud_compute"
    input.variables.environment.value != "prod"
    resource.change.after.instance_class == "xlarge"
    msg := sprintf(
        "Resource %s uses xlarge instance in non-prod environment",
        [resource.address]
    )
}
```

#### Sentinel (Terraform Cloud / Enterprise)

```python
# sentinel/enforce-tags.sentinel
import "tfplan/v2" as tfplan

mandatory_tags = ["Project", "Environment", "Owner", "ManagedBy"]

all_resources = filter tfplan.resource_changes as _, rc {
    rc.mode is "managed" and
    (rc.change.actions contains "create" or rc.change.actions contains "update")
}

tags_present = rule {
    all all_resources as _, resource {
        all mandatory_tags as tag {
            resource.change.after.tags contains tag
        }
    }
}

main = rule {
    tags_present
}
```

**Policy rules:**
- Every Terraform plan must pass policy checks **before** apply.
- Policies must be version-controlled alongside infrastructure code.
- Policy failures must **block** the pipeline (not just warn).
- Review and update policies quarterly or when organizational standards change.

### E. Automated Formatting Checks

Formatting must be enforced in CI. Use pre-commit hooks locally and CI checks as a safety net.

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/antonbabenko/pre-commit-terraform
    rev: v1.86.0
    hooks:
      - id: terraform_fmt
      - id: terraform_validate
      - id: terraform_tflint
        args: ['--args=--recursive']
      - id: terraform_tfsec
      - id: terraform_docs
        args: ['--args=--config=.terraform-docs.yml']
      - id: terraform_checkov
```

```bash
# Install pre-commit and run hooks
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### F. Cost Estimation in CI

Integrate cost estimation into the pull request workflow so reviewers can evaluate the financial impact of changes before approval.

```yaml
# Add to the PR workflow (after plan)
      - name: Run Infracost
        uses: infracost/actions/setup@v3
        with:
          api-key: ${{ secrets.INFRACOST_API_KEY }}

      - name: Generate Cost Estimate
        run: |
          infracost breakdown \
            --path=infrastructure/root \
            --terraform-var-file=env/${{ matrix.workspace }}.tfvars \
            --format=json \
            --out-file=/tmp/infracost-${{ matrix.workspace }}.json

      - name: Post Cost Comment
        if: github.event_name == 'pull_request'
        uses: infracost/actions/comment@v1
        with:
          path: /tmp/infracost-${{ matrix.workspace }}.json
          behavior: update
```

**Cost estimation rules:**
- Cost estimates must be posted as **PR comments** for every infrastructure change.
- Set **budget thresholds** and fail the pipeline if estimated monthly cost exceeds the threshold.
- Review cost estimates as part of the PR approval process.

---

## 10. Cost Management

### A. Infracost Integration

```yaml
# .github/workflows/infracost.yml
name: Infracost

on:
  pull_request:
    paths: ['infrastructure/**']

jobs:
  infracost:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Infracost
        uses: infracost/actions/setup@v3
        with:
          api-key: ${{ secrets.INFRACOST_API_KEY }}

      - name: Generate Infracost JSON
        run: |
          infracost breakdown --path=infrastructure/root \
            --terraform-var-file=env/prod.tfvars \
            --format=json --out-file=/tmp/infracost.json

      - name: Post Infracost comment
        uses: infracost/actions/comment@v1
        with:
          path: /tmp/infracost.json
          behavior: update
```

### B. Resource Tagging for Cost Allocation

```hcl
variable "required_tags" {
  description = "Required tags/labels for cost tracking"
  type        = map(string)
  default     = {
    Project     = "app"
    Environment = "dev"
    Owner       = "platform"
    CostCenter  = "0000"
    ManagedBy   = "terraform"
  }
}

resource "cloud_compute" "example" {
  name                = "${var.project}-${var.environment}-compute-web"
  resource_group_name = var.resource_group_name
  tags                = var.required_tags
}
```

---

## 11. Deployment Checklist

### Pre-Deployment
- [ ] `terraform fmt` - Code is formatted
- [ ] `terraform validate` - Configuration is valid
- [ ] `tflint` - Linting passes
- [ ] `tfsec` - Security scan passes
- [ ] `checkov` - Compliance checks pass
- [ ] `terraform plan` - Plan reviewed

### State Management
- [ ] Remote backend configured
- [ ] State locking enabled
- [ ] State encryption enabled
- [ ] State backup strategy in place

### Security
- [ ] No secrets in code or tfvars
- [ ] Sensitive variables marked
- [ ] Identity/roles follow least privilege
- [ ] Encryption enabled for data at rest
- [ ] Security groups are restrictive

### Documentation
- [ ] README.md updated
- [ ] Variables documented
- [ ] Outputs documented
- [ ] Module examples provided
- [ ] Documentation and examples generated with terraform-docs

---

## 12. Quick Reference

```bash
# Common commands
terraform init                    # Initialize
terraform plan                    # Preview changes
terraform apply                   # Apply changes
terraform destroy                 # Destroy resources
terraform fmt -recursive          # Format code
terraform validate                # Validate syntax
terraform output                  # Show outputs
terraform state list              # List resources in state
terraform import                  # Import existing resource

# Workspace management
terraform workspace list
terraform workspace new dev
terraform workspace new test
terraform workspace new prod
terraform workspace select test

# State management
terraform state mv               # Move resource
terraform state rm               # Remove from state
terraform state pull             # Download state
terraform state push             # Upload state

# Debugging
TF_LOG=DEBUG terraform plan      # Enable debug logging
terraform graph | dot -Tpng > graph.png  # Visualize
```

---

## 13. Why This Configuration Works

1. **Remote State with Locking**: Storing state in S3/GCS/Azure Blob with DynamoDB/native locking prevents concurrent `apply` runs from corrupting infrastructure state.

2. **Workspace-per-Environment**: Using workspaces with environment-specific `.tfvars` files keeps code identical across dev/staging/prod while varying only configuration values.

3. **Module Composition**: Building infrastructure from small, tested modules with clear input/output contracts enables reuse across teams and projects.

4. **Plan Before Apply**: Mandatory `terraform plan` review before every `apply` prevents accidental resource destruction and surfaces unexpected changes.

5. **No Locals Rule**: Driving all values through variables with defaults, types, and descriptions makes modules self-documenting and prevents hidden logic.

6. **Passwordless Authentication**: Using managed identities, IAM roles, or workload identity federation eliminates static credentials in pipelines and state files.

7. **File Naming Convention**: Prefixing files with incremental numbers (`001-network.tf`, `002-compute.tf`) provides clear reading order and prevents merge conflicts.

8. **Pre-commit with tflint and tfsec**: Automated formatting, linting, and security scanning before commit catches misconfigurations before they enter version control.

9. **Explicit Dependency Management**: Using module outputs instead of implicit dependencies makes the resource graph predictable and parallelizable.

10. **Infracost Integration**: Estimating cost changes in pull requests prevents surprise cloud bills and enables cost-aware infrastructure decisions during code review.

---

**Last Updated:** 2026-02-27
**Version:** 3.0
**Maintainer:** Infrastructure Team


**End of Terraform Infrastructure as Code Guidelines (Provider-Agnostic)**
