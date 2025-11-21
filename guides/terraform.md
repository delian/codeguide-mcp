# Terraform Infrastructure as Code Guidelines (Provider-Agnostic)

This document provides mandatory standards for Terraform development, infrastructure management, and IaC best practices. It is **cloud-agnostic** and must work across Azure, AWS, GCP, and other providers.

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

### B. Backend Configuration per Workspace

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

**Last Updated:** 2026-01-31
**Version:** 2.0
**Maintainer:** Infrastructure Team
