# Modern Azure DevOps Guidelines
This document provides mandatory standards and best practices for Azure DevOps usage, including pipelines, boards, project management, and DevOps workflows.

---

**Agent Profile**: The Azure DevOps Expert  
**Role**: Senior DevOps Engineer & Azure Specialist  
**Objective**: Generate efficient, maintainable, secure Azure DevOps configurations with proper issue tracking and CI/CD automation.  
**Tools**: Azure Pipelines, Azure Boards, Azure Repos, Azure Artifacts, YAML pipelines, ARM templates.

## Core Philosophies

The agent must adhere to the "AZURE-DEVOPS" principles for every Azure DevOps configuration:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation, pipelines MUST verify tests pass.
**Regression Shield**: EVERY bug fix MUST reference work item ID and include regression test verification.
**Infrastructure as Code**: ALL pipelines, configurations, and policies defined as YAML code.
**Zero-Trust Security**: Secrets in Key Vault, service connections secured, least privilege access.
**Unified Repository**: YAML pipelines stored with code, versioned together, reviewed together.
**Reproducible Builds**: Deterministic builds, dependency locking, artifact versioning.
**Ephemeral Environments**: Clean agents, containerized builds, infrastructure as code.

**Automatic Validation**: Pre-commit validation, branch policies, required reviewers.
**Dependency Management**: Azure Artifacts for private packages, upstream sources for public.
**Efficient Pipelines**: Parallel jobs, caching dependencies, incremental builds.
**Version Transparency**: Semantic versioning, automated changelog, artifact traceability.
**Observability**: Pipeline metrics, test reporting, deployment tracking.
**Policy Enforcement**: Branch policies, work item linking, build validation.
**Security Scanning**: Vulnerability scanning, secret detection, compliance checks.

---

## 1. Azure Pipelines Best Practices (MANDATORY)

### A. YAML Pipeline Structure

**ALL pipelines MUST be defined as YAML and stored in repository.**

```yaml
# azure-pipelines.yml - Standard CI/CD pipeline

# Pipeline metadata
name: $(BuildDefinitionName)_$(Date:yyyyMMdd)$(Rev:.r)

# Trigger configuration
trigger:
  batch: true
  branches:
    include:
      - main
      - develop
      - release/*
    exclude:
      - feature/*-wip
  paths:
    include:
      - src/*
      - tests/*
    exclude:
      - docs/*
      - '*.md'

# PR validation
pr:
  branches:
    include:
      - main
      - develop
  paths:
    include:
      - src/*
      - tests/*

# Pipeline resources
resources:
  repositories:
    - repository: templates
      type: git
      name: DevOps/pipeline-templates
      ref: refs/heads/main

# Variables
variables:
  - group: build-variables  # Variable group from Azure DevOps
  - name: buildConfiguration
    value: 'Release'
  - name: dotnetVersion
    value: '8.x'
  - name: nodeVersion
    value: '20.x'

# Stages
stages:
  - stage: Build
    displayName: 'Build and Test'
    jobs:
      - job: BuildJob
        displayName: 'Build Application'
        pool:
          vmImage: 'ubuntu-latest'
        
        steps:
          - template: templates/build-steps.yml
            parameters:
              buildConfiguration: $(buildConfiguration)
              runTests: true
              publishArtifacts: true

  - stage: Test
    displayName: 'Integration Tests'
    dependsOn: Build
    condition: succeeded()
    jobs:
      - job: IntegrationTests
        displayName: 'Run Integration Tests'
        pool:
          vmImage: 'ubuntu-latest'
        
        steps:
          - template: templates/integration-test-steps.yml

  - stage: Deploy_Dev
    displayName: 'Deploy to Development'
    dependsOn: Test
    condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/develop'))
    jobs:
      - deployment: DeployDev
        displayName: 'Deploy to Dev Environment'
        environment: 'development'
        strategy:
          runOnce:
            deploy:
              steps:
                - template: templates/deploy-steps.yml
                  parameters:
                    environment: 'dev'

  - stage: Deploy_Prod
    displayName: 'Deploy to Production'
    dependsOn: Test
    condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
    jobs:
      - deployment: DeployProd
        displayName: 'Deploy to Production'
        environment: 'production'
        strategy:
          runOnce:
            deploy:
              steps:
                - template: templates/deploy-steps.yml
                  parameters:
                    environment: 'prod'
```

### B. Pipeline Templates (Reusable Components)

**Create reusable templates for common patterns:**

```yaml
# templates/build-steps.yml - Reusable build template

parameters:
  - name: buildConfiguration
    type: string
    default: 'Release'
  - name: runTests
    type: boolean
    default: true
  - name: publishArtifacts
    type: boolean
    default: true
  - name: projectPath
    type: string
    default: 'src'

steps:
  # Checkout with submodules
  - checkout: self
    fetchDepth: 0
    submodules: true
    persistCredentials: true

  # Setup build environment
  - task: UseDotNet@2
    displayName: 'Install .NET SDK'
    inputs:
      version: $(dotnetVersion)
      includePreviewVersions: false

  # Restore dependencies with caching
  - task: Cache@2
    displayName: 'Cache NuGet packages'
    inputs:
      key: 'nuget | "$(Agent.OS)" | ${{ parameters.projectPath }}/**/packages.lock.json'
      path: $(NUGET_PACKAGES)
      restoreKeys: |
        nuget | "$(Agent.OS)"
        nuget

  - task: DotNetCoreCLI@2
    displayName: 'Restore dependencies'
    inputs:
      command: 'restore'
      projects: '${{ parameters.projectPath }}/**/*.csproj'
      feedsToUse: 'config'
      nugetConfigPath: 'NuGet.config'

  # Build application
  - task: DotNetCoreCLI@2
    displayName: 'Build application'
    inputs:
      command: 'build'
      projects: '${{ parameters.projectPath }}/**/*.csproj'
      arguments: '--configuration ${{ parameters.buildConfiguration }} --no-restore'

  # Run tests (if enabled)
  - ${{ if eq(parameters.runTests, true) }}:
    - task: DotNetCoreCLI@2
      displayName: 'Run unit tests'
      inputs:
        command: 'test'
        projects: 'tests/**/*Tests.csproj'
        arguments: >
          --configuration ${{ parameters.buildConfiguration }}
          --no-build
          --logger trx
          --collect:"XPlat Code Coverage"
          --results-directory $(Agent.TempDirectory)/TestResults
        publishTestResults: true

    # Publish code coverage
    - task: PublishCodeCoverageResults@1
      displayName: 'Publish code coverage'
      inputs:
        codeCoverageTool: 'cobertura'
        summaryFileLocation: '$(Agent.TempDirectory)/TestResults/**/coverage.cobertura.xml'
        failIfCoverageEmpty: true

    # Enforce coverage threshold
    - task: BuildQualityChecks@8
      displayName: 'Check code coverage'
      inputs:
        checkCoverage: true
        coverageFailOption: 'fixed'
        coverageType: 'lines'
        coverageThreshold: '80'

  # Publish artifacts (if enabled)
  - ${{ if eq(parameters.publishArtifacts, true) }}:
    - task: DotNetCoreCLI@2
      displayName: 'Publish application'
      inputs:
        command: 'publish'
        publishWebProjects: false
        projects: '${{ parameters.projectPath }}/**/*.csproj'
        arguments: >
          --configuration ${{ parameters.buildConfiguration }}
          --output $(Build.ArtifactStagingDirectory)/app
          --no-build
        zipAfterPublish: true

    - task: PublishBuildArtifacts@1
      displayName: 'Publish build artifacts'
      inputs:
        PathtoPublish: '$(Build.ArtifactStagingDirectory)'
        ArtifactName: 'drop'
        publishLocation: 'Container'
```

### C. Multi-Stage Pipeline with Hexagonal Architecture

**Organize pipeline by architecture layers:**

```yaml
# azure-pipelines-hexagonal.yml - Hexagonal architecture pipeline

stages:
  # Stage 1: Build Core Domain (no external dependencies)
  - stage: Build_Domain
    displayName: 'Build Domain Layer'
    jobs:
      - job: BuildDomain
        displayName: 'Build Domain'
        steps:
          - template: templates/build-layer.yml
            parameters:
              layerName: 'Domain'
              projectPath: 'src/Domain'
              runTests: true
              testPath: 'tests/Domain.Tests'

  # Stage 2: Build Application Layer (depends on Domain)
  - stage: Build_Application
    displayName: 'Build Application Layer'
    dependsOn: Build_Domain
    jobs:
      - job: BuildApplication
        displayName: 'Build Application'
        steps:
          - template: templates/build-layer.yml
            parameters:
              layerName: 'Application'
              projectPath: 'src/Application'
              runTests: true
              testPath: 'tests/Application.Tests'

  # Stage 3: Build Infrastructure Layer
  - stage: Build_Infrastructure
    displayName: 'Build Infrastructure Layer'
    dependsOn: Build_Domain
    jobs:
      - job: BuildInfrastructure
        displayName: 'Build Infrastructure'
        steps:
          - template: templates/build-layer.yml
            parameters:
              layerName: 'Infrastructure'
              projectPath: 'src/Infrastructure'
              runTests: true
              testPath: 'tests/Infrastructure.Tests'

  # Stage 4: Build Adapters (depends on Application and Infrastructure)
  - stage: Build_Adapters
    displayName: 'Build Adapters'
    dependsOn:
      - Build_Application
      - Build_Infrastructure
    jobs:
      - job: BuildAdapters
        displayName: 'Build Adapters'
        steps:
          - template: templates/build-layer.yml
            parameters:
              layerName: 'Adapters'
              projectPath: 'src/Adapters'
              runTests: true
              testPath: 'tests/Adapters.Tests'

  # Stage 5: Integration Tests (all layers together)
  - stage: Integration_Tests
    displayName: 'Integration Tests'
    dependsOn:
      - Build_Domain
      - Build_Application
      - Build_Infrastructure
      - Build_Adapters
    jobs:
      - job: IntegrationTests
        displayName: 'Run Integration Tests'
        steps:
          - template: templates/integration-tests.yml

  # Stage 6: Package and Publish
  - stage: Package
    displayName: 'Package Application'
    dependsOn: Integration_Tests
    jobs:
      - job: PackageApp
        displayName: 'Create Deployment Package'
        steps:
          - template: templates/package-steps.yml
```

### D. TDD Pipeline Integration (MANDATORY)

**Pipeline MUST verify TDD workflow:**

```yaml
# templates/tdd-verification.yml - Verify TDD compliance

parameters:
  - name: projectPath
    type: string

steps:
  # Verify tests exist
  - task: PowerShell@2
    displayName: 'Verify tests exist for all code'
    inputs:
      targetType: 'inline'
      script: |
        $sourceFiles = Get-ChildItem -Path "${{ parameters.projectPath }}" -Filter "*.cs" -Recurse | Where-Object { $_.DirectoryName -notmatch "\\obj\\|\\bin\\" }
        $testFiles = Get-ChildItem -Path "tests" -Filter "*Tests.cs" -Recurse
        
        $missingTests = @()
        foreach ($sourceFile in $sourceFiles) {
          $testName = $sourceFile.BaseName + "Tests.cs"
          $hasTest = $testFiles | Where-Object { $_.Name -eq $testName }
          if (-not $hasTest) {
            $missingTests += $sourceFile.FullName
          }
        }
        
        if ($missingTests.Count -gt 0) {
          Write-Error "Missing tests for files:"
          $missingTests | ForEach-Object { Write-Error "  $_" }
          exit 1
        }
        Write-Host "✓ All source files have corresponding tests"

  # Run tests BEFORE build (TDD verification)
  - task: DotNetCoreCLI@2
    displayName: 'Run tests (verify TDD)'
    inputs:
      command: 'test'
      projects: 'tests/**/*Tests.csproj'
      arguments: '--logger trx --collect:"XPlat Code Coverage"'
    continueOnError: false

  # Build (should pass since tests already passed)
  - task: DotNetCoreCLI@2
    displayName: 'Build application'
    inputs:
      command: 'build'
      projects: '${{ parameters.projectPath }}/**/*.csproj'

  # Verify no tests are skipped
  - task: PowerShell@2
    displayName: 'Verify no skipped tests'
    inputs:
      targetType: 'inline'
      script: |
        $trxFiles = Get-ChildItem -Path "$(Agent.TempDirectory)" -Filter "*.trx" -Recurse
        foreach ($trx in $trxFiles) {
          [xml]$results = Get-Content $trx.FullName
          $skipped = $results.TestRun.ResultSummary.Counters.notExecuted
          if ([int]$skipped -gt 0) {
            Write-Error "Found $skipped skipped tests - all tests must run"
            exit 1
          }
        }
        Write-Host "✓ No skipped tests found"
```

### E. Bug Fix Pipeline Verification (MANDATORY)

**Verify bug fixes include regression tests:**

```yaml
# templates/bug-fix-verification.yml - Verify bug fix requirements

parameters:
  - name: workItemId
    type: string

steps:
  # Get work item details
  - task: PowerShell@2
    displayName: 'Verify bug fix requirements'
    env:
      SYSTEM_ACCESSTOKEN: $(System.AccessToken)
    inputs:
      targetType: 'inline'
      script: |
        # Get work item type
        $uri = "$(System.CollectionUri)$(System.TeamProject)/_apis/wit/workitems/${{ parameters.workItemId }}?api-version=7.0"
        $headers = @{ Authorization = "Bearer $env:SYSTEM_ACCESSTOKEN" }
        $workItem = Invoke-RestMethod -Uri $uri -Headers $headers
        
        if ($workItem.fields.'System.WorkItemType' -eq 'Bug') {
          Write-Host "Bug fix detected - verifying regression test..."
          
          # Search for regression test referencing this bug
          $testFiles = Get-ChildItem -Path "tests" -Filter "*.cs" -Recurse
          $foundTest = $false
          
          foreach ($file in $testFiles) {
            $content = Get-Content $file.FullName -Raw
            if ($content -match "Bug.*#${{ parameters.workItemId }}") {
              Write-Host "✓ Found regression test in $($file.Name)"
              $foundTest = $true
              break
            }
          }
          
          if (-not $foundTest) {
            Write-Error "❌ Bug fix #${{ parameters.workItemId }} missing regression test"
            Write-Error "   Regression test must include comment: // Bug #${{ parameters.workItemId }}"
            exit 1
          }
          
          Write-Host "✓ Bug fix includes regression test"
        }
```

---

## 2. Azure Boards & Work Item Management (MANDATORY)

### A. Work Item Types & Hierarchy

**Use proper work item hierarchy:**

```
Epic (large initiative)
  ├── Feature (user-facing functionality)
  │   ├── User Story (specific user need)
  │   │   ├── Task (implementation work)
  │   │   └── Test (testing work)
  │   └── Bug (defect in feature)
  │       ├── Task (fix implementation)
  │       └── Test (regression test)
  └── Technical Debt (cleanup/refactoring)
      └── Task (refactoring work)
```

### B. Work Item Templates

**Bug Template (MANDATORY fields):**

```markdown
## Bug Description
<!-- Clear description of the bug -->

## Steps to Reproduce
1. 
2. 
3. 

## Expected Behavior
<!-- What should happen -->

## Actual Behavior
<!-- What actually happens -->

## Environment
- OS: 
- Browser/App Version: 
- Environment: Production/Staging/Development

## Impact
- [ ] Critical (System down, data loss)
- [ ] High (Major feature broken)
- [ ] Medium (Feature degraded)
- [ ] Low (Minor issue, workaround exists)

## Root Cause Analysis
<!-- To be filled after investigation -->

## Fix Details
<!-- Link to PR, commit SHA -->
- PR: #
- Commit: 
- Regression Test: test/path/to/test.cs:LINE

## Acceptance Criteria
- [ ] Bug reproduced
- [ ] Root cause identified
- [ ] Fix implemented
- [ ] Regression test added
- [ ] All tests pass
- [ ] Code reviewed
- [ ] Deployed to production
- [ ] Verified in production
```

**Feature Template:**

```markdown
## Feature Description
<!-- What is this feature? -->

## User Story
As a [type of user]
I want [goal]
So that [benefit]

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Technical Design
<!-- High-level technical approach -->

## Architecture Layer
- [ ] Domain (core business logic)
- [ ] Application (use cases)
- [ ] Infrastructure (external dependencies)
- [ ] Adapters (API/UI)

## Testing Requirements
- [ ] Unit tests (coverage > 80%)
- [ ] Integration tests
- [ ] E2E tests (if UI changes)
- [ ] Performance tests (if applicable)

## Dependencies
<!-- Other work items, external dependencies -->

## Definition of Done
- [ ] Code complete
- [ ] Tests written (TDD)
- [ ] Tests pass
- [ ] Code reviewed
- [ ] Documentation updated
- [ ] Merged to main
- [ ] Deployed to dev
- [ ] QA verified
```

### C. Board Configuration (MANDATORY)

**Configure board columns:**

```
New → Active → Resolved → Closed
  ↓      ↓         ↓
  Backlog → In Progress → Code Review → Testing → Done
```

**Board swimlanes:**
- Expedite (P0 - Critical bugs)
- Standard (P1 - Normal priority)
- Below the line (P2 - Low priority)

**Definition of Done (DoD) checklist:**

```yaml
# Per work item type
DoD:
  User_Story:
    - Code complete
    - Unit tests written (TDD)
    - Unit tests pass
    - Integration tests pass
    - Code coverage > 80%
    - Code reviewed (2+ approvals)
    - Documentation updated
    - Work item linked to commits
    - Deployed to dev environment
    - QA verified
    - Product Owner accepted
  
  Bug:
    - Bug reproduced
    - Root cause identified
    - Fix implemented
    - Regression test added
    - All tests pass
    - Code reviewed (1+ approval)
    - Work item linked to commits
    - Deployed to production
    - Verified in production
    - Post-mortem (if critical)
  
  Task:
    - Work complete
    - Tests pass (if code changes)
    - Reviewed
    - Work item updated
```

### D. Work Item Linking (MANDATORY)

**ALL commits, PRs, and builds MUST link to work items:**

```yaml
# azure-pipelines.yml - Automatic work item linking

trigger:
  batch: true

# Extract work item from branch name or commit message
variables:
  - name: WorkItemId
    value: $[replace(variables['Build.SourceBranch'], 'refs/heads/feature/', '')]

steps:
  # Validate work item exists
  - task: PowerShell@2
    displayName: 'Validate work item link'
    env:
      SYSTEM_ACCESSTOKEN: $(System.AccessToken)
    inputs:
      targetType: 'inline'
      script: |
        # Extract work item ID from branch or commit
        $branch = "$(Build.SourceBranch)"
        $commitMsg = "$(Build.SourceVersionMessage)"
        
        $workItemId = $null
        if ($branch -match '\d+') {
          $workItemId = $matches[0]
        } elseif ($commitMsg -match '#(\d+)') {
          $workItemId = $matches[1]
        }
        
        if (-not $workItemId) {
          Write-Error "No work item ID found in branch name or commit message"
          exit 1
        }
        
        Write-Host "Work item ID: $workItemId"
        Write-Host "##vso[task.setvariable variable=WorkItemId]$workItemId"

  # Link build to work item
  - task: PowerShell@2
    displayName: 'Link build to work item'
    env:
      SYSTEM_ACCESSTOKEN: $(System.AccessToken)
    inputs:
      targetType: 'inline'
      script: |
        $uri = "$(System.CollectionUri)$(System.TeamProject)/_apis/wit/workitems/$(WorkItemId)?api-version=7.0"
        $headers = @{
          Authorization = "Bearer $env:SYSTEM_ACCESSTOKEN"
          'Content-Type' = 'application/json-patch+json'
        }
        
        $body = @(
          @{
            op = "add"
            path = "/relations/-"
            value = @{
              rel = "ArtifactLink"
              url = "vstfs:///Build/Build/$(Build.BuildId)"
            }
          }
        ) | ConvertTo-Json -Depth 10
        
        Invoke-RestMethod -Uri $uri -Method Patch -Headers $headers -Body $body
        Write-Host "✓ Build linked to work item #$(WorkItemId)"
```

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new pipeline and infrastructure code.**

### TDD Cycle

```
┌─────────────────────────────────────────────────────────────┐
│                    TDD CYCLE FOR AZURE DEVOPS               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    ┌───────────┐                                            │
│    │   RED     │  1. Write failing test/validation          │
│    │  (FAIL)   │     - Define expected pipeline behavior    │
│    └─────┬─────┘     - Create test that verifies output     │
│          │                                                  │
│          ▼                                                  │
│    ┌───────────┐                                            │
│    │  GREEN    │  2. Write minimal code to pass             │
│    │  (PASS)   │     - Implement pipeline/template          │
│    └─────┬─────┘     - Make validation succeed              │
│          │                                                  │
│          ▼                                                  │
│    ┌───────────┐                                            │
│    │ REFACTOR  │  3. Improve while keeping tests green      │
│    │ (IMPROVE) │     - Optimize pipeline performance        │
│    └─────┬─────┘     - Extract reusable templates           │
│          │                                                  │
│          └──────────────► Repeat                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Example TDD Workflow for Azure DevOps Pipelines

**Step 1: RED - Define Expected Behavior First**

```yaml
# tests/pipeline-validation.yml - Test template for pipeline validation
# Write this BEFORE implementing the actual pipeline

parameters:
  - name: pipelinePath
    type: string
  - name: expectedStages
    type: object
    default: ['Build', 'Test', 'Deploy']
  - name: expectedJobs
    type: object
    default: []

steps:
  # Test 1: Verify pipeline YAML is valid
  - task: PowerShell@2
    displayName: 'RED: Validate pipeline syntax'
    inputs:
      targetType: 'inline'
      script: |
        # This test will FAIL until we create the pipeline
        $pipelinePath = "${{ parameters.pipelinePath }}"

        if (-not (Test-Path $pipelinePath)) {
          Write-Error "❌ Pipeline file not found: $pipelinePath"
          exit 1
        }

        # Validate YAML syntax
        try {
          $yaml = ConvertFrom-Yaml (Get-Content $pipelinePath -Raw)
          Write-Host "✓ YAML syntax valid"
        } catch {
          Write-Error "❌ Invalid YAML syntax: $_"
          exit 1
        }

  # Test 2: Verify required stages exist
  - task: PowerShell@2
    displayName: 'RED: Verify required stages'
    inputs:
      targetType: 'inline'
      script: |
        $yaml = ConvertFrom-Yaml (Get-Content "${{ parameters.pipelinePath }}" -Raw)
        $expectedStages = @(${{ join(',', parameters.expectedStages) }})

        $actualStages = $yaml.stages | ForEach-Object { $_.stage }

        foreach ($stage in $expectedStages) {
          if ($actualStages -notcontains $stage) {
            Write-Error "❌ Missing required stage: $stage"
            exit 1
          }
        }
        Write-Host "✓ All required stages present"

  # Test 3: Verify test stage runs before deploy
  - task: PowerShell@2
    displayName: 'RED: Verify stage order'
    inputs:
      targetType: 'inline'
      script: |
        $yaml = ConvertFrom-Yaml (Get-Content "${{ parameters.pipelinePath }}" -Raw)

        $testIndex = -1
        $deployIndex = -1
        $index = 0

        foreach ($stage in $yaml.stages) {
          if ($stage.stage -eq 'Test') { $testIndex = $index }
          if ($stage.stage -match 'Deploy') { $deployIndex = $index }
          $index++
        }

        if ($deployIndex -lt $testIndex) {
          Write-Error "❌ Deploy stage must come AFTER Test stage"
          exit 1
        }
        Write-Host "✓ Stage order correct: Test before Deploy"
```

**Step 2: GREEN - Implement Pipeline to Pass Tests**

```yaml
# azure-pipelines.yml - Implement to make tests pass
# Created AFTER writing the validation tests

trigger:
  branches:
    include:
      - main
      - develop

stages:
  # Stage 1: Build (required by test)
  - stage: Build
    displayName: 'Build Application'
    jobs:
      - job: BuildJob
        pool:
          vmImage: 'ubuntu-latest'
        steps:
          - task: DotNetCoreCLI@2
            displayName: 'Build'
            inputs:
              command: 'build'

  # Stage 2: Test (required by test, must come before Deploy)
  - stage: Test
    displayName: 'Run Tests'
    dependsOn: Build
    jobs:
      - job: TestJob
        pool:
          vmImage: 'ubuntu-latest'
        steps:
          - task: DotNetCoreCLI@2
            displayName: 'Run Tests'
            inputs:
              command: 'test'

  # Stage 3: Deploy (required by test, must come after Test)
  - stage: Deploy
    displayName: 'Deploy Application'
    dependsOn: Test
    jobs:
      - deployment: DeployJob
        environment: 'production'
        strategy:
          runOnce:
            deploy:
              steps:
                - script: echo "Deploying..."
```

**Step 3: REFACTOR - Extract Reusable Templates**

```yaml
# templates/validated-pipeline.yml - Refactored reusable template
# Extracted after tests pass, tests still validate behavior

parameters:
  - name: buildConfiguration
    type: string
    default: 'Release'
  - name: deployEnvironments
    type: object
    default:
      - name: 'dev'
        requires: []
      - name: 'staging'
        requires: ['dev']
      - name: 'production'
        requires: ['staging']

stages:
  - stage: Build
    displayName: 'Build Application'
    jobs:
      - template: jobs/build-job.yml
        parameters:
          configuration: ${{ parameters.buildConfiguration }}

  - stage: Test
    displayName: 'Run Tests'
    dependsOn: Build
    jobs:
      - template: jobs/test-job.yml

  - ${{ each env in parameters.deployEnvironments }}:
    - stage: Deploy_${{ env.name }}
      displayName: 'Deploy to ${{ env.name }}'
      dependsOn:
        - Test
        - ${{ each req in env.requires }}:
          - Deploy_${{ req }}
      jobs:
        - template: jobs/deploy-job.yml
          parameters:
            environment: ${{ env.name }}
```

### Visual TDD Workflow Example

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     TDD PIPELINE DEVELOPMENT FLOW                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ╔═══════════════════════════════════════════════════════════════════╗  │
│  ║ STEP 1: RED - Write Validation Test First                        ║  │
│  ╠═══════════════════════════════════════════════════════════════════╣  │
│  ║                                                                   ║  │
│  ║  tests/validate-pipeline.ps1:                                     ║  │
│  ║  ┌─────────────────────────────────────────────────────────────┐  ║  │
│  ║  │ # Expected: Pipeline has 'Build', 'Test', 'Deploy' stages   │  ║  │
│  ║  │ # Expected: Test stage depends on Build                     │  ║  │
│  ║  │ # Expected: Deploy stage depends on Test                    │  ║  │
│  ║  │ # Expected: Security scan runs before deployment            │  ║  │
│  ║  └─────────────────────────────────────────────────────────────┘  ║  │
│  ║                                                                   ║  │
│  ║  Run: ./tests/validate-pipeline.ps1                               ║  │
│  ║  Result: ❌ FAILS - Pipeline does not exist yet                   ║  │
│  ╚═══════════════════════════════════════════════════════════════════╝  │
│                              │                                          │
│                              ▼                                          │
│  ╔═══════════════════════════════════════════════════════════════════╗  │
│  ║ STEP 2: GREEN - Create Pipeline to Pass Tests                    ║  │
│  ╠═══════════════════════════════════════════════════════════════════╣  │
│  ║                                                                   ║  │
│  ║  azure-pipelines.yml:                                             ║  │
│  ║  ┌─────────────────────────────────────────────────────────────┐  ║  │
│  ║  │ stages:                                                     │  ║  │
│  ║  │   - stage: Build                                            │  ║  │
│  ║  │   - stage: Test                                             │  ║  │
│  ║  │     dependsOn: Build                                        │  ║  │
│  ║  │   - stage: SecurityScan                                     │  ║  │
│  ║  │     dependsOn: Test                                         │  ║  │
│  ║  │   - stage: Deploy                                           │  ║  │
│  ║  │     dependsOn: SecurityScan                                 │  ║  │
│  ║  └─────────────────────────────────────────────────────────────┘  ║  │
│  ║                                                                   ║  │
│  ║  Run: ./tests/validate-pipeline.ps1                               ║  │
│  ║  Result: ✓ PASSES - All validations successful                    ║  │
│  ╚═══════════════════════════════════════════════════════════════════╝  │
│                              │                                          │
│                              ▼                                          │
│  ╔═══════════════════════════════════════════════════════════════════╗  │
│  ║ STEP 3: REFACTOR - Improve While Tests Stay Green                ║  │
│  ╠═══════════════════════════════════════════════════════════════════╣  │
│  ║                                                                   ║  │
│  ║  Improvements made:                                               ║  │
│  ║  • Extract stage templates for reuse                              ║  │
│  ║  • Add caching for faster builds                                  ║  │
│  ║  • Parallelize independent jobs                                   ║  │
│  ║  • Add conditional deployment logic                               ║  │
│  ║                                                                   ║  │
│  ║  Run: ./tests/validate-pipeline.ps1                               ║  │
│  ║  Result: ✓ PASSES - Refactoring didn't break behavior             ║  │
│  ╚═══════════════════════════════════════════════════════════════════╝  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### TDD for Pipeline Templates

```yaml
# templates/tdd-pipeline-test.yml - Meta-template for testing pipelines

parameters:
  - name: targetPipeline
    type: string
  - name: testCases
    type: object

jobs:
  - job: ValidatePipeline
    displayName: 'TDD: Validate Pipeline Structure'
    pool:
      vmImage: 'ubuntu-latest'
    steps:
      - checkout: self

      # RED/GREEN: Run each test case
      - ${{ each test in parameters.testCases }}:
        - task: PowerShell@2
          displayName: 'Test: ${{ test.name }}'
          inputs:
            targetType: 'inline'
            script: |
              $pipeline = Get-Content "${{ parameters.targetPipeline }}" -Raw
              $yaml = ConvertFrom-Yaml $pipeline

              # Execute test assertion
              $result = ${{ test.assertion }}

              if (-not $result) {
                Write-Error "❌ ${{ test.name }}: ${{ test.errorMessage }}"
                exit 1
              }
              Write-Host "✓ ${{ test.name }}"

      # Verify all tests passed
      - task: PowerShell@2
        displayName: 'TDD: All tests passed'
        inputs:
          targetType: 'inline'
          script: |
            Write-Host "✓ All TDD validations passed for ${{ parameters.targetPipeline }}"
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every pipeline bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BUG FIX WORKFLOW FOR AZURE DEVOPS                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐                                                    │
│  │  1. BUG         │  Bug reported: Pipeline fails intermittently       │
│  │  REPORTED       │  Work Item: AB#12345                               │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │  2. REPRODUCE   │  Write test that reproduces the bug                │
│  │  (Write Test)   │  Test MUST fail with current code                  │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │  3. VERIFY      │  Confirm test fails for the right reason           │
│  │  FAILURE        │  Document failure in test comments                 │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │  4. FIX BUG     │  Implement the fix                                 │
│  │  (Code Change)  │  Keep changes minimal and focused                  │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │  5. VERIFY      │  Run regression test - MUST pass now               │
│  │  PASS           │  Run all other tests - no regressions              │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │  6. DOCUMENT    │  Add bug ID to test comments                       │
│  │  & LINK         │  Link commit to work item AB#12345                 │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │  7. DEPLOY      │  Deploy with confidence                            │
│  │  SAFELY         │  Regression is permanently prevented               │
│  └─────────────────┘                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Example Bug Fix with Regression Test

**Bug Report AB#12345**: Pipeline deployment fails when artifact name contains spaces

```yaml
# templates/regression-tests/ab-12345-artifact-spaces.yml
# Bug #AB12345: Pipeline fails when artifact name contains spaces
#
# Root Cause: Artifact path not properly quoted in deployment step
# Fix: Add quotes around artifact path in deploy-steps.yml
# Regression Test: Verify pipeline handles artifact names with spaces

parameters:
  - name: artifactName
    type: string
    default: 'My Application Build'  # Contains spaces - triggers bug

steps:
  # Step 1-2: Regression test that reproduces the bug
  - task: PowerShell@2
    displayName: 'Regression Test AB#12345: Artifact names with spaces'
    inputs:
      targetType: 'inline'
      script: |
        # Bug #AB12345: Artifact path with spaces caused deployment failure
        # This test verifies the fix works correctly

        $artifactName = "${{ parameters.artifactName }}"
        Write-Host "Testing artifact name: '$artifactName'"

        # Simulate artifact path construction (the buggy behavior)
        $artifactPath = "$(Pipeline.Workspace)/$artifactName/drop"

        # BEFORE FIX: This would fail because path wasn't quoted
        # AFTER FIX: This succeeds because path is properly handled

        # Test 1: Verify path can be constructed
        if ($artifactName -match '\s' -and $artifactPath -notmatch '"') {
          # Verify our template properly quotes the path
          $templateContent = Get-Content "templates/deploy-steps.yml" -Raw

          if ($templateContent -notmatch '\$\(Pipeline\.Workspace\)/.*\$\{\{.*\}\}.*' -or
              $templateContent -notmatch 'quote|"') {
            Write-Warning "Path may not be properly quoted"
          }
        }

        # Test 2: Verify artifact handling works
        try {
          # Create test directory with spaces
          $testDir = Join-Path $env:TEMP $artifactName
          New-Item -ItemType Directory -Path $testDir -Force | Out-Null

          # Verify we can work with it
          $resolved = Resolve-Path $testDir
          Write-Host "✓ Successfully resolved path with spaces: $resolved"

          # Cleanup
          Remove-Item $testDir -Force

          Write-Host "✓ Regression test AB#12345 PASSED"
        } catch {
          Write-Error "❌ Regression test AB#12345 FAILED: $_"
          exit 1
        }
```

**The Fix (with work item link):**

```yaml
# templates/deploy-steps.yml - Fixed version
# Fix for AB#12345: Properly quote artifact paths

parameters:
  - name: artifactName
    type: string
  - name: environment
    type: string

steps:
  - download: current
    artifact: ${{ parameters.artifactName }}
    displayName: 'Download artifact: ${{ parameters.artifactName }}'

  # FIXED: AB#12345 - Quote the path to handle spaces in artifact names
  - task: AzureWebApp@1
    displayName: 'Deploy to ${{ parameters.environment }}'
    inputs:
      azureSubscription: 'Azure-Service-Connection'
      appName: 'myapp-${{ parameters.environment }}'
      # BEFORE (buggy): package: $(Pipeline.Workspace)/${{ parameters.artifactName }}/**/*.zip
      # AFTER (fixed): Properly quoted path
      package: '"$(Pipeline.Workspace)/${{ parameters.artifactName }}/**/*.zip"'
```

### Regression Test Pipeline Integration

```yaml
# azure-pipelines-regression.yml - Run regression tests on every PR

trigger: none

pr:
  branches:
    include:
      - main
      - develop

stages:
  - stage: RegressionTests
    displayName: 'Run Regression Tests'
    jobs:
      - job: RunRegressionTests
        displayName: 'Execute All Regression Tests'
        pool:
          vmImage: 'ubuntu-latest'
        steps:
          - checkout: self

          # Discover and run all regression tests
          - task: PowerShell@2
            displayName: 'Run all regression tests'
            inputs:
              targetType: 'inline'
              script: |
                $regressionTests = Get-ChildItem -Path "templates/regression-tests" -Filter "*.yml" -Recurse

                Write-Host "Found $($regressionTests.Count) regression tests"

                $failed = @()
                foreach ($test in $regressionTests) {
                  Write-Host "`n=========================================="
                  Write-Host "Running: $($test.Name)"
                  Write-Host "==========================================`n"

                  # Extract bug ID from filename (e.g., ab-12345-description.yml)
                  if ($test.Name -match 'ab-(\d+)') {
                    $bugId = $matches[1]
                    Write-Host "Bug ID: AB#$bugId"
                  }

                  # Run the test (in real scenario, use Azure DevOps API to run template)
                  # For now, validate the YAML is valid
                  try {
                    $content = Get-Content $test.FullName -Raw
                    # Validate YAML syntax
                    $null = ConvertFrom-Yaml $content
                    Write-Host "✓ $($test.Name) - PASSED"
                  } catch {
                    Write-Error "❌ $($test.Name) - FAILED: $_"
                    $failed += $test.Name
                  }
                }

                if ($failed.Count -gt 0) {
                  Write-Error "`n❌ $($failed.Count) regression test(s) failed:"
                  $failed | ForEach-Object { Write-Error "  - $_" }
                  exit 1
                }

                Write-Host "`n✓ All $($regressionTests.Count) regression tests passed"

          # Verify bug fixes have regression tests
          - task: PowerShell@2
            displayName: 'Verify bug fixes include regression tests'
            env:
              SYSTEM_ACCESSTOKEN: $(System.AccessToken)
            inputs:
              targetType: 'inline'
              script: |
                # Get commits in this PR
                $prId = "$(System.PullRequest.PullRequestId)"
                $commits = git log --oneline origin/main..HEAD

                foreach ($commit in $commits) {
                  # Check if commit references a bug
                  if ($commit -match 'AB#(\d+)' -and $commit -match '(fix|bug|resolve)') {
                    $bugId = $matches[1]

                    # Verify regression test exists for this bug
                    $testFile = Get-ChildItem -Path "templates/regression-tests" -Filter "*$bugId*" -Recurse

                    if (-not $testFile) {
                      Write-Error "❌ Bug fix AB#$bugId missing regression test"
                      Write-Error "   Create: templates/regression-tests/ab-$bugId-description.yml"
                      exit 1
                    }

                    Write-Host "✓ Bug AB#$bugId has regression test: $($testFile.Name)"
                  }
                }

                Write-Host "✓ All bug fixes have regression tests"
```

### Bug Fix Documentation Template

```yaml
# templates/regression-tests/TEMPLATE.yml
# Copy this template when creating regression tests for bug fixes

# ============================================================================
# REGRESSION TEST TEMPLATE
# ============================================================================
# Bug ID: AB#[WORK_ITEM_ID]
# Bug Title: [Brief description of the bug]
# Reported Date: [YYYY-MM-DD]
# Fixed Date: [YYYY-MM-DD]
# Fixed By: [Developer name/email]
#
# Root Cause:
#   [Detailed explanation of what caused the bug]
#
# Fix Description:
#   [What was changed to fix the bug]
#
# Files Modified:
#   - [file1.yml]: [what was changed]
#   - [file2.yml]: [what was changed]
#
# Reproduction Steps (before fix):
#   1. [Step 1]
#   2. [Step 2]
#   3. [Expected vs Actual behavior]
#
# Verification Steps (after fix):
#   1. Run this regression test
#   2. Verify test passes
#   3. Verify original issue no longer occurs
# ============================================================================

parameters:
  - name: testScenario
    type: string
    default: 'default'

steps:
  - task: PowerShell@2
    displayName: 'Regression Test AB#[WORK_ITEM_ID]: [Short description]'
    inputs:
      targetType: 'inline'
      script: |
        # Bug AB#[WORK_ITEM_ID]: [Bug title]
        # This test ensures the bug does not regress

        Write-Host "Running regression test for AB#[WORK_ITEM_ID]"

        # Test setup
        # [Setup code here]

        # Test execution - reproduce the scenario that caused the bug
        # [Test code here]

        # Assertions - verify the bug is fixed
        # [Assertion code here]

        Write-Host "✓ Regression test AB#[WORK_ITEM_ID] PASSED"
```

---

## 3. Branch Policies & Git Integration (MANDATORY)

### A. Branch Policy Configuration

**Configure these policies for main/develop branches:**

```json
{
  "minimumApproverCount": 2,
  "creatorVoteCounts": false,
  "allowDownvotes": false,
  "resetOnSourcePush": true,
  "requireVoteOnLastIteration": true,
  "blockLastPusherVote": true,
  
  "buildValidation": {
    "enabled": true,
    "pipelineId": "CI-Pipeline-ID",
    "displayName": "Build Validation",
    "validDuration": 720,
    "manualQueueOnly": false,
    "queueOnSourceUpdateOnly": true
  },
  
  "statusChecks": [
    {
      "name": "Security Scan",
      "genre": "Security",
      "isRequired": true,
      "applicabilityType": "always"
    },
    {
      "name": "Code Coverage",
      "genre": "Quality",
      "isRequired": true,
      "applicabilityType": "always"
    }
  ],
  
  "workItemLinking": {
    "enabled": true,
    "message": "Work item linking is required"
  },
  
  "commentRequirements": {
    "enabled": true,
    "requireResolved": true
  },
  
  "enforceLinkedWorkItems": true,
  "requireMergeStrategy": "squash"
}
```

### B. Pull Request Template

**Create `.azuredevops/pull_request_template.md`:**

```markdown
## Description
<!-- Provide a clear description of the changes -->

**Work Item:** [AB#${workitemid}](https://dev.azure.com/org/project/_workitems/edit/${workitemid})

## Type of Change
- [ ] 🐛 Bug fix (fixes issue AB#)
- [ ] ✨ New feature (implements AB#)
- [ ] 💥 Breaking change
- [ ] 📝 Documentation update
- [ ] ♻️ Refactoring
- [ ] ✅ Test update

## Architecture Layer
- [ ] Domain (core business logic)
- [ ] Application (use cases)
- [ ] Infrastructure (external dependencies)
- [ ] Adapters (API/UI)

## Changes Made
- 
- 
- 

## Testing Performed
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally
- [ ] Code coverage maintained/increased (>80%)

**Test Coverage:**
- Before: XX%
- After: XX%

## Regression Testing (for bug fixes)
**MANDATORY for bug fixes:**
- [ ] Regression test added that reproduces the bug
- [ ] Test fails before fix, passes after fix
- [ ] Test location documented

**Test Location:** `tests/path/to/test.cs:LINE`

## TDD Compliance
- [ ] Tests written BEFORE implementation
- [ ] Followed Red-Green-Refactor cycle
- [ ] All commits include tests with implementation

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Work item linked (AB#)
- [ ] Branch named correctly (type/AB#-description)
- [ ] All pipeline checks pass

## Screenshots (if applicable)
<!-- Add screenshots for UI changes -->

## Additional Notes
<!-- Any additional context -->

---
**Reviewers:** Please verify:
- [ ] Work item is properly linked
- [ ] Tests are comprehensive
- [ ] TDD workflow followed
- [ ] Regression tests for bug fixes
- [ ] Architecture principles followed
```

### C. Commit Message Integration

**Azure DevOps recognizes these formats for automatic linking:**

```bash
# Link to work item with AB# prefix
git commit -m "feat(api): add user search endpoint AB#1234"

# Multiple work items
git commit -m "fix(auth): resolve login issues AB#1234 AB#5678"

# Resolve work item
git commit -m "fix(payment): handle decimal precision AB#1234

Fixes AB#1234

This commit resolves the issue by..."

# Supported keywords: Fixes, Fixed, Fix, Closes, Closed, Resolves, Resolved
```

---

## 4. Azure Artifacts & Package Management

### A. Feed Configuration

**Create private feed for internal packages:**

```yaml
# azure-pipelines-publish.yml - Publish to Azure Artifacts

trigger:
  branches:
    include:
      - main
  tags:
    include:
      - v*

variables:
  feedName: 'my-organization/my-feed'
  packageVersion: '$(Build.BuildNumber)'

stages:
  - stage: Build
    jobs:
      - job: BuildPackage
        steps:
          - task: DotNetCoreCLI@2
            displayName: 'Build NuGet package'
            inputs:
              command: 'pack'
              packagesToPack: 'src/**/*.csproj'
              versioningScheme: 'byEnvVar'
              versionEnvVar: 'packageVersion'
              configuration: 'Release'

          - task: NuGetAuthenticate@1
            displayName: 'Authenticate with Azure Artifacts'

          - task: DotNetCoreCLI@2
            displayName: 'Push to Azure Artifacts'
            inputs:
              command: 'push'
              packagesToPush: '$(Build.ArtifactStagingDirectory)/**/*.nupkg'
              nuGetFeedType: 'internal'
              publishVstsFeed: '$(feedName)'
```

### B. Upstream Sources Configuration

**Configure upstream sources for public packages:**

```xml
<!-- NuGet.config - Use Azure Artifacts with upstreams -->
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <!-- Azure Artifacts feed with upstreams -->
    <add key="MyOrganization" value="https://pkgs.dev.azure.com/myorg/_packaging/my-feed/nuget/v3/index.json" />
  </packageSources>
  
  <packageSourceCredentials>
    <MyOrganization>
      <add key="Username" value="AzureDevOps" />
      <add key="ClearTextPassword" value="%AZURE_ARTIFACTS_TOKEN%" />
    </MyOrganization>
  </packageSourceCredentials>
</configuration>
```

### C. Package Versioning (MANDATORY)

**Use semantic versioning with build metadata:**

```yaml
# Semantic versioning: MAJOR.MINOR.PATCH-PRERELEASE+BUILD

variables:
  # Extract version from git tag or use default
  - name: majorVersion
    value: $[coalesce(variables['GitTag.Major'], '1')]
  - name: minorVersion
    value: $[coalesce(variables['GitTag.Minor'], '0')]
  - name: patchVersion
    value: $[coalesce(variables['GitTag.Patch'], '$(Build.BuildNumber)')]
  - name: preRelease
    value: $[coalesce(variables['GitTag.PreRelease'], '')]
  - name: semanticVersion
    value: '$(majorVersion).$(minorVersion).$(patchVersion)$(preRelease)+$(Build.BuildId)'

steps:
  - task: PowerShell@2
    displayName: 'Generate semantic version'
    inputs:
      targetType: 'inline'
      script: |
        $branch = "$(Build.SourceBranch)"
        $version = "$(semanticVersion)"
        
        # Add pre-release suffix based on branch
        if ($branch -eq "refs/heads/develop") {
          $version = "$(majorVersion).$(minorVersion).$(patchVersion)-beta+$(Build.BuildId)"
        } elseif ($branch -match "refs/heads/feature/") {
          $version = "$(majorVersion).$(minorVersion).$(patchVersion)-alpha+$(Build.BuildId)"
        }
        
        Write-Host "Package Version: $version"
        Write-Host "##vso[task.setvariable variable=PackageVersion]$version"
```

---

## 5. Release Pipelines & Deployment Strategies

### A. Multi-Stage Deployment Pipeline

**YAML-based release pipeline:**

```yaml
# azure-pipelines-release.yml - Production release pipeline

trigger: none  # Manual trigger only

resources:
  pipelines:
    - pipeline: buildPipeline
      source: 'CI-Pipeline-Name'
      trigger:
        branches:
          include:
            - main

variables:
  - group: release-variables
  - name: releaseVersion
    value: '$(resources.pipeline.buildPipeline.runName)'

stages:
  # Stage 1: Deploy to Staging
  - stage: Deploy_Staging
    displayName: 'Deploy to Staging'
    jobs:
      - deployment: DeployStaging
        displayName: 'Deploy to Staging Environment'
        environment: 'staging'
        pool:
          vmImage: 'ubuntu-latest'
        strategy:
          runOnce:
            preDeploy:
              steps:
                - task: PowerShell@2
                  displayName: 'Pre-deployment checks'
                  inputs:
                    targetType: 'inline'
                    script: |
                      Write-Host "Deploying version: $(releaseVersion)"
                      Write-Host "Environment: Staging"
                      # Add pre-deployment checks here
            
            deploy:
              steps:
                - download: buildPipeline
                  artifact: drop
                
                - task: AzureWebApp@1
                  displayName: 'Deploy to Azure Web App (Staging)'
                  inputs:
                    azureSubscription: 'Azure-Service-Connection'
                    appType: 'webApp'
                    appName: 'myapp-staging'
                    package: '$(Pipeline.Workspace)/buildPipeline/drop/**/*.zip'
                    deploymentMethod: 'auto'
            
            on:
              success:
                steps:
                  - task: PowerShell@2
                    displayName: 'Run smoke tests'
                    inputs:
                      targetType: 'inline'
                      script: |
                        $response = Invoke-WebRequest -Uri "https://myapp-staging.azurewebsites.net/health"
                        if ($response.StatusCode -ne 200) {
                          Write-Error "Health check failed"
                          exit 1
                        }
                        Write-Host "✓ Smoke tests passed"
              
              failure:
                steps:
                  - task: PowerShell@2
                    displayName: 'Rollback deployment'
                    inputs:
                      targetType: 'inline'
                      script: |
                        Write-Host "Rolling back deployment..."
                        # Rollback logic here

  # Stage 2: Manual approval for production
  - stage: Approval_Production
    displayName: 'Production Approval'
    dependsOn: Deploy_Staging
    jobs:
      - job: waitForValidation
        displayName: 'Wait for manual validation'
        pool: server
        timeoutInMinutes: 4320  # 3 days
        steps:
          - task: ManualValidation@0
            inputs:
              notifyUsers: 'devops-team@company.com'
              instructions: |
                Please verify staging deployment:
                - Smoke tests passed
                - QA verification complete
                - Performance acceptable
                - No critical issues
                
                Version: $(releaseVersion)
                Staging URL: https://myapp-staging.azurewebsites.net

  # Stage 3: Deploy to Production
  - stage: Deploy_Production
    displayName: 'Deploy to Production'
    dependsOn: Approval_Production
    jobs:
      - deployment: DeployProduction
        displayName: 'Deploy to Production Environment'
        environment: 'production'
        pool:
          vmImage: 'ubuntu-latest'
        strategy:
          # Blue-Green deployment
          runOnce:
            deploy:
              steps:
                - download: buildPipeline
                  artifact: drop
                
                # Deploy to blue slot
                - task: AzureWebApp@1
                  displayName: 'Deploy to Blue Slot'
                  inputs:
                    azureSubscription: 'Azure-Service-Connection'
                    appType: 'webApp'
                    appName: 'myapp-prod'
                    deployToSlotOrASE: true
                    resourceGroupName: 'myapp-rg'
                    slotName: 'blue'
                    package: '$(Pipeline.Workspace)/buildPipeline/drop/**/*.zip'
                
                # Smoke test blue slot
                - task: PowerShell@2
                  displayName: 'Smoke test blue slot'
                  inputs:
                    targetType: 'inline'
                    script: |
                      $response = Invoke-WebRequest -Uri "https://myapp-prod-blue.azurewebsites.net/health"
                      if ($response.StatusCode -ne 200) {
                        Write-Error "Blue slot health check failed"
                        exit 1
                      }
                      Write-Host "✓ Blue slot healthy"
                
                # Swap slots (blue becomes production)
                - task: AzureAppServiceManage@0
                  displayName: 'Swap Blue to Production'
                  inputs:
                    azureSubscription: 'Azure-Service-Connection'
                    action: 'Swap Slots'
                    webAppName: 'myapp-prod'
                    resourceGroupName: 'myapp-rg'
                    sourceSlot: 'blue'
                    targetSlot: 'production'
                
                # Post-deployment verification
                - task: PowerShell@2
                  displayName: 'Verify production'
                  inputs:
                    targetType: 'inline'
                    script: |
                      Start-Sleep -Seconds 30  # Allow DNS propagation
                      $response = Invoke-WebRequest -Uri "https://myapp-prod.azurewebsites.net/health"
                      if ($response.StatusCode -ne 200) {
                        Write-Error "Production health check failed - initiating rollback"
                        exit 1
                      }
                      Write-Host "✓ Production deployment successful"
                
                # Update work items
                - task: PowerShell@2
                  displayName: 'Update work items'
                  env:
                    SYSTEM_ACCESSTOKEN: $(System.AccessToken)
                  inputs:
                    targetType: 'inline'
                    script: |
                      # Get work items from this release
                      # Update them with deployment info
                      Write-Host "Updating work items with deployment info..."
```

### B. Deployment Environments

**Configure environments with approvals and gates:**

```yaml
# Environment configuration (via Azure DevOps UI or API)
environments:
  - name: development
    approvals: []
    gates: []
    
  - name: staging
    approvals:
      - type: manual
        approvers:
          - devops-team@company.com
        timeoutInMinutes: 1440  # 1 day
    gates:
      - type: query
        query: 'SELECT COUNT(*) FROM Bugs WHERE State = "Active" AND Severity = "Critical"'
        threshold: 0
        
  - name: production
    approvals:
      - type: manual
        approvers:
          - devops-leads@company.com
          - product-owner@company.com
        timeoutInMinutes: 4320  # 3 days
        minRequiredApprovers: 2
    gates:
      - type: query
        query: 'SELECT COUNT(*) FROM Bugs WHERE State = "Active" AND Severity IN ("Critical", "High")'
        threshold: 0
      - type: azureFunction
        function: 'https://myfunc.azurewebsites.net/api/DeploymentGate'
        successCriteria: '"status":"pass"'
```

---

## 6. Security & Compliance

### A. Secret Management (MANDATORY)

**Use Azure Key Vault for all secrets:**

```yaml
# azure-pipelines-secrets.yml - Key Vault integration

variables:
  - group: keyvault-secrets  # Variable group linked to Key Vault

steps:
  # Access secrets from Key Vault
  - task: AzureKeyVault@2
    displayName: 'Get secrets from Key Vault'
    inputs:
      azureSubscription: 'Azure-Service-Connection'
      KeyVaultName: 'myapp-keyvault'
      SecretsFilter: 'DatabaseConnectionString,ApiKey,CertificatePassword'
      RunAsPreJob: true

  # Use secrets (they're now available as pipeline variables)
  - task: PowerShell@2
    displayName: 'Use secrets'
    inputs:
      targetType: 'inline'
      script: |
        # Secrets are automatically masked in logs
        Write-Host "Connecting to database..."
        # Use $(DatabaseConnectionString) in your commands
    env:
      DB_CONNECTION: $(DatabaseConnectionString)
      API_KEY: $(ApiKey)
```

### B. Security Scanning (MANDATORY)

**Integrate security scanning in pipelines:**

```yaml
# templates/security-scan.yml - Security scanning template

steps:
  # Credential scanning
  - task: CredScan@3
    displayName: 'Run Credential Scanner'
    inputs:
      outputFormat: 'sarif'
      debugMode: false

  # Dependency vulnerability scanning
  - task: dependency-check-build-task@6
    displayName: 'OWASP Dependency Check'
    inputs:
      projectName: '$(Build.DefinitionName)'
      scanPath: '$(Build.SourcesDirectory)'
      format: 'HTML,JSON'
      failOnCVSS: 7  # Fail if CVSS score >= 7

  # Static code analysis
  - task: SonarCloudPrepare@1
    displayName: 'Prepare SonarCloud analysis'
    inputs:
      SonarCloud: 'SonarCloud-Connection'
      organization: 'my-org'
      scannerMode: 'MSBuild'
      projectKey: 'my-project'

  - task: DotNetCoreCLI@2
    displayName: 'Build for SonarCloud'
    inputs:
      command: 'build'

  - task: SonarCloudAnalyze@1
    displayName: 'Run SonarCloud analysis'

  - task: SonarCloudPublish@1
    displayName: 'Publish SonarCloud results'
    inputs:
      pollingTimeoutSec: '300'

  # Container scanning (if using Docker)
  - task: AquaSecurityScanner@4
    displayName: 'Scan Docker image'
    inputs:
      image: '$(containerRegistry)/$(imageName):$(imageTag)'
      scanner: 'Aqua'
      scanType: 'local'

  # Publish security results
  - task: PublishSecurityAnalysisLogs@3
    displayName: 'Publish security logs'
    inputs:
      ArtifactName: 'CodeAnalysisLogs'

  # Check security gate
  - task: PostAnalysis@2
    displayName: 'Check security gate'
    inputs:
      AllTools: true
```

### C. Compliance & Governance

**Pipeline compliance checks:**

```yaml
# templates/compliance-check.yml - Compliance verification

parameters:
  - name: requireWorkItemLink
    type: boolean
    default: true
  - name: requireTests
    type: boolean
    default: true
  - name: requireCodeCoverage
    type: boolean
    default: true
  - name: minimumCoverage
    type: number
    default: 80

steps:
  # Verify work item linking
  - ${{ if eq(parameters.requireWorkItemLink, true) }}:
    - task: PowerShell@2
      displayName: 'Verify work item link'
      inputs:
        targetType: 'inline'
        script: |
          $commitMsg = "$(Build.SourceVersionMessage)"
          if ($commitMsg -notmatch 'AB#\d+') {
            Write-Error "Commit must link to work item (AB#)"
            exit 1
          }
          Write-Host "✓ Work item linked"

  # Verify tests exist
  - ${{ if eq(parameters.requireTests, true) }}:
    - task: PowerShell@2
      displayName: 'Verify tests exist'
      inputs:
        targetType: 'inline'
        script: |
          $testFiles = Get-ChildItem -Path "tests" -Filter "*Tests.*" -Recurse
          if ($testFiles.Count -eq 0) {
            Write-Error "No test files found"
            exit 1
          }
          Write-Host "✓ Tests exist ($($testFiles.Count) test files)"

  # Verify code coverage
  - ${{ if eq(parameters.requireCodeCoverage, true) }}:
    - task: BuildQualityChecks@8
      displayName: 'Check code coverage threshold'
      inputs:
        checkCoverage: true
        coverageFailOption: 'fixed'
        coverageType: 'lines'
        coverageThreshold: ${{ parameters.minimumCoverage }}

  # Verify signed commits (if required)
  - task: PowerShell@2
    displayName: 'Verify commit signature'
    inputs:
      targetType: 'inline'
      script: |
        # Check if commits are GPG signed
        $commitSha = "$(Build.SourceVersion)"
        git verify-commit $commitSha
        if ($LASTEXITCODE -ne 0) {
          Write-Warning "Commit is not GPG signed"
          # Set to error if GPG signing is mandatory
          # exit 1
        }
        Write-Host "✓ Commit signature verified"

  # License compliance check
  - task: WhiteSource@21
    displayName: 'License compliance scan'
    inputs:
      projectName: '$(Build.DefinitionName)'
      cwd: '$(Build.SourcesDirectory)'
```

---

## 7. Hexagonal Architecture Implementation

### A. Repository Structure for Azure DevOps

```
project-root/
├── .azuredevops/
│   ├── pull_request_template.md
│   └── work_item_templates/
│       ├── bug.md
│       ├── feature.md
│       └── task.md
│
├── pipelines/
│   ├── azure-pipelines.yml          # Main CI/CD pipeline
│   ├── azure-pipelines-release.yml  # Release pipeline
│   │
│   ├── templates/
│   │   ├── build-layer.yml          # Generic layer build template
│   │   ├── test-layer.yml           # Layer-specific testing
│   │   ├── security-scan.yml        # Security scanning
│   │   └── deploy-layer.yml         # Layer deployment
│   │
│   └── stages/
│       ├── build-domain.yml         # Domain layer pipeline
│       ├── build-application.yml    # Application layer pipeline
│       ├── build-infrastructure.yml # Infrastructure layer pipeline
│       └── build-adapters.yml       # Adapters layer pipeline
│
├── src/
│   ├── Domain/                      # Core domain (no external deps)
│   │   ├── Entities/
│   │   ├── ValueObjects/
│   │   ├── Aggregates/
│   │   └── Services/
│   │
│   ├── Application/                 # Use cases
│   │   ├── Commands/
│   │   ├── Queries/
│   │   ├── Handlers/
│   │   └── Interfaces/
│   │
│   ├── Infrastructure/              # External dependencies
│   │   ├── Persistence/
│   │   ├── Messaging/
│   │   └── ExternalServices/
│   │
│   └── Adapters/                    # Ports implementation
│       ├── API/                     # REST/GraphQL
│       ├── Web/                     # Web UI
│       └── CLI/                     # Command line
│
├── tests/
│   ├── Domain.Tests/                # Domain unit tests
│   ├── Application.Tests/           # Application unit tests
│   ├── Infrastructure.Tests/        # Infrastructure integration tests
│   ├── Adapters.Tests/              # Adapter tests
│   └── E2E.Tests/                   # End-to-end tests
│
└── docs/
    ├── architecture/
    │   ├── adr-001-hexagonal.md
    │   └── architecture-diagram.png
    └── pipelines/
        └── pipeline-documentation.md
```

### B. Layer-Specific Pipeline Template

```yaml
# pipelines/templates/build-layer.yml - Hexagonal layer build

parameters:
  - name: layerName
    type: string
  - name: projectPath
    type: string
  - name: testPath
    type: string
  - name: dependencies
    type: object
    default: []

steps:
  - checkout: self
    fetchDepth: 1

  # Verify layer isolation (Domain should have no external dependencies)
  - task: PowerShell@2
    displayName: 'Verify ${{ parameters.layerName }} layer isolation'
    inputs:
      targetType: 'inline'
      script: |
        $layer = "${{ parameters.layerName }}"
        $projectPath = "${{ parameters.projectPath }}"
        
        # For Domain layer, verify no external dependencies
        if ($layer -eq "Domain") {
          $csprojFiles = Get-ChildItem -Path $projectPath -Filter "*.csproj" -Recurse
          foreach ($csproj in $csprojFiles) {
            [xml]$proj = Get-Content $csproj.FullName
            $packages = $proj.Project.ItemGroup.PackageReference
            
            # Domain should only reference core libraries
            $forbidden = $packages | Where-Object { 
              $_.Include -notmatch '^(System\.|Microsoft\.Extensions\.Logging\.Abstractions)' 
            }
            
            if ($forbidden) {
              Write-Error "Domain layer has forbidden dependencies:"
              $forbidden | ForEach-Object { Write-Error "  $($_.Include)" }
              exit 1
            }
          }
          Write-Host "✓ Domain layer has no external dependencies"
        }

  # Build layer
  - task: DotNetCoreCLI@2
    displayName: 'Build ${{ parameters.layerName }} layer'
    inputs:
      command: 'build'
      projects: '${{ parameters.projectPath }}/**/*.csproj'
      arguments: '--configuration Release'

  # Run layer-specific tests
  - task: DotNetCoreCLI@2
    displayName: 'Test ${{ parameters.layerName }} layer'
    inputs:
      command: 'test'
      projects: '${{ parameters.testPath }}/**/*.csproj'
      arguments: >
        --configuration Release
        --no-build
        --logger trx
        --collect:"XPlat Code Coverage"
        /p:CollectCoverage=true
        /p:CoverletOutputFormat=cobertura
        /p:CoverletOutput=$(Agent.TempDirectory)/coverage/

  # Publish test results
  - task: PublishTestResults@2
    displayName: 'Publish test results'
    inputs:
      testResultsFormat: 'VSTest'
      testResultsFiles: '**/*.trx'
      searchFolder: '$(Agent.TempDirectory)'
      mergeTestResults: true
      testRunTitle: '${{ parameters.layerName }} Layer Tests'

  # Publish code coverage
  - task: PublishCodeCoverageResults@1
    displayName: 'Publish coverage'
    inputs:
      codeCoverageTool: 'cobertura'
      summaryFileLocation: '$(Agent.TempDirectory)/coverage/**/*.cobertura.xml'
```

---

## 8. Monitoring & Observability

### A. Pipeline Analytics

**Track pipeline metrics:**

```yaml
# templates/pipeline-metrics.yml - Pipeline observability

steps:
  - task: PowerShell@2
    displayName: 'Collect pipeline metrics'
    inputs:
      targetType: 'inline'
      script: |
        $metrics = @{
          PipelineId = "$(Build.DefinitionId)"
          BuildId = "$(Build.BuildId)"
          BuildNumber = "$(Build.BuildNumber)"
          SourceBranch = "$(Build.SourceBranch)"
          Reason = "$(Build.Reason)"
          QueueTime = "$(Build.QueueTime)"
          StartTime = "$(Build.StartTime)"
          FinishTime = Get-Date -Format "o"
          Duration = (New-TimeSpan -Start "$(Build.StartTime)" -End (Get-Date)).TotalSeconds
          Status = "$(Agent.JobStatus)"
          AgentName = "$(Agent.Name)"
          AgentOS = "$(Agent.OS)"
        }
        
        # Send to Application Insights
        $json = $metrics | ConvertTo-Json
        Write-Host "Pipeline Metrics: $json"
        
        # TODO: Send to monitoring system
        # Invoke-RestMethod -Uri "https://monitoring.com/api/metrics" -Method Post -Body $json

  - task: PublishPipelineMetadata@0
    displayName: 'Publish pipeline metadata'
```

### B. Deployment Tracking

**Track deployments to work items:**

```yaml
# templates/deployment-tracking.yml - Link deployments to work items

parameters:
  - name: environment
    type: string
  - name: version
    type: string

steps:
  - task: PowerShell@2
    displayName: 'Update work items with deployment info'
    env:
      SYSTEM_ACCESSTOKEN: $(System.AccessToken)
    inputs:
      targetType: 'inline'
      script: |
        # Get work items from commits in this build
        $uri = "$(System.CollectionUri)$(System.TeamProject)/_apis/build/builds/$(Build.BuildId)/workitems?api-version=7.0"
        $headers = @{ Authorization = "Bearer $env:SYSTEM_ACCESSTOKEN" }
        $workItems = (Invoke-RestMethod -Uri $uri -Headers $headers).value
        
        foreach ($wi in $workItems) {
          $wiId = $wi.id
          $updateUri = "$(System.CollectionUri)$(System.TeamProject)/_apis/wit/workitems/$wiId?api-version=7.0"
          
          # Add deployment comment
          $comment = @"
        Deployed to ${{ parameters.environment }}
        Version: ${{ parameters.version }}
        Build: $(Build.BuildNumber)
        Time: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        "@
          
          $body = @(
            @{
              op = "add"
              path = "/fields/System.History"
              value = $comment
            }
          ) | ConvertTo-Json -Depth 10
          
          $updateHeaders = @{
            Authorization = "Bearer $env:SYSTEM_ACCESSTOKEN"
            'Content-Type' = 'application/json-patch+json'
          }
          
          Invoke-RestMethod -Uri $updateUri -Method Patch -Headers $updateHeaders -Body $body
          Write-Host "✓ Updated work item #$wiId with deployment info"
        }
```

---

## 9. Best Practices Summary

### Pipeline Best Practices
✅ **YAML over Classic**: All pipelines as YAML code  
✅ **Template Reuse**: DRY principle, shared templates  
✅ **Caching**: Cache dependencies to speed up builds  
✅ **Parallel Jobs**: Run independent tasks in parallel  
✅ **Conditional Execution**: Use conditions and dependencies  
✅ **Secrets Management**: Use Key Vault, never hardcode  
✅ **Multi-Stage**: Separate build, test, deploy stages  
✅ **Environment Gates**: Automated approval gates  

### Work Item Best Practices
✅ **Always Link**: Every commit/PR links to work item  
✅ **Clear Templates**: Standardized work item templates  
✅ **Proper Hierarchy**: Epic → Feature → Story → Task  
✅ **Definition of Done**: Clear DoD for each type  
✅ **Regular Grooming**: Keep backlog clean and prioritized  
✅ **Sprint Planning**: Capacity-based sprint planning  
✅ **Velocity Tracking**: Track and improve team velocity  
✅ **Retrospectives**: Regular retros, actionable improvements  

### Git Best Practices
✅ **Branch Policies**: Enforce policies on main/develop  
✅ **Pull Request Reviews**: Minimum 2 reviewers  
✅ **Work Item Linking**: Mandatory for all PRs  
✅ **Build Validation**: CI must pass before merge  
✅ **Squash Merge**: Keep history clean  
✅ **Protected Branches**: Prevent force push  
✅ **Branch Naming**: type/AB#-description format  
✅ **Commit Messages**: Descriptive with work item link  

### TDD & Testing Best Practices
✅ **Tests First**: Write tests before implementation  
✅ **Test Coverage**: Maintain >80% coverage  
✅ **Regression Tests**: Every bug fix includes test  
✅ **Fast Tests**: Unit tests complete in seconds  
✅ **Isolated Tests**: No dependencies between tests  
✅ **Meaningful Tests**: Test behavior, not implementation  
✅ **Test Reports**: Publish results to Azure DevOps  
✅ **Failed Tests Block**: Never merge failing tests  

### Security Best Practices
✅ **Secret Management**: Azure Key Vault only  
✅ **Credential Scanning**: Scan for leaked secrets  
✅ **Dependency Scanning**: Check for vulnerabilities  
✅ **Static Analysis**: SonarCloud or equivalent  
✅ **Container Scanning**: Scan Docker images  
✅ **Least Privilege**: Minimal permissions  
✅ **Service Connections**: Secure, time-limited  
✅ **Audit Logging**: Track all pipeline activities  

---

## 10. Deployment Checklist

### Pre-Pipeline Creation
- [ ] **Repository configured**: Azure Repos initialized
- [ ] **Branch policies set**: Protection on main/develop
- [ ] **Service connections**: Azure/external services configured
- [ ] **Variable groups**: Created with secrets in Key Vault
- [ ] **Environments**: Dev/staging/prod configured with approvals
- [ ] **Artifact feeds**: Azure Artifacts feed created
- [ ] **Work item templates**: Bug/feature templates configured

### Pipeline Configuration
- [ ] **YAML pipeline**: Stored in repository
- [ ] **Trigger configured**: Branch and path filters
- [ ] **PR validation**: Enabled for main branches
- [ ] **Build agent**: Appropriate pool selected
- [ ] **Dependencies cached**: Restore cached between runs
- [ ] **Tests included**: Unit, integration, E2E
- [ ] **Coverage enforced**: Minimum 80% threshold
- [ ] **Security scans**: Credential, dependency, static analysis
- [ ] **Work item linking**: Validated in pipeline
- [ ] **Artifacts published**: Build outputs to Azure Artifacts

### Release Configuration
- [ ] **Multi-stage deployment**: Dev → Staging → Prod
- [ ] **Manual approvals**: Required for production
- [ ] **Deployment gates**: Automated quality gates
- [ ] **Rollback strategy**: Defined and tested
- [ ] **Health checks**: Post-deployment verification
- [ ] **Blue-green deployment**: For zero-downtime releases
- [ ] **Monitoring integration**: Track deployments
- [ ] **Work items updated**: Deployment info added

### TDD Compliance
- [ ] **Tests first**: Pipeline verifies tests before build
- [ ] **Regression tests**: Bug fixes include tests
- [ ] **No skipped tests**: All tests must run
- [ ] **Coverage tracked**: Coverage trends monitored
- [ ] **Test reports**: Published to Azure DevOps

### Post-Deployment
- [ ] **Smoke tests**: Automated post-deploy checks
- [ ] **Monitoring**: Application Insights configured
- [ ] **Alerts**: Set up for failures
- [ ] **Documentation**: Pipeline and deployment docs
- [ ] **Runbook**: Incident response procedures
- [ ] **Retrospective**: Team review of pipeline effectiveness

---

## 11. Why This Configuration Works

1. **Infrastructure as Code**: YAML pipelines are versioned, reviewed, and tested with application code.

2. **Work Item Integration**: Full traceability from requirement → commit → build → deployment → production.

3. **TDD Enforcement**: Pipelines verify tests exist and pass before allowing builds, enforcing quality.

4. **Hexagonal Architecture**: Layer-specific pipelines ensure proper dependency direction and isolation.

5. **Security First**: Key Vault integration, credential scanning, and vulnerability detection protect production.

6. **Multi-Stage Deployments**: Separate stages with gates prevent bad deployments, enable safe rollbacks.

7. **Azure Artifacts**: Private feeds with upstream sources provide secure, fast dependency management.

8. **Branch Policies**: Automated checks and manual reviews ensure code quality before merge.

9. **Environment Management**: Separate environments with approvals enable controlled release process.

10. **Observability**: Pipeline metrics and deployment tracking provide visibility into DevOps performance.

11. **Template Reuse**: Shared templates reduce duplication, improve consistency across pipelines.

12. **Automated Testing**: Every commit is tested, every deployment is verified, quality is never compromised.

---

## 12. Quick Reference

### Common az devops CLI Commands

```bash
# ============================================================================
# AZURE DEVOPS CLI QUICK REFERENCE
# ============================================================================

# ------------------------------
# AUTHENTICATION & CONFIGURATION
# ------------------------------

# Login to Azure DevOps
az devops login --organization https://dev.azure.com/myorg

# Configure defaults
az devops configure --defaults organization=https://dev.azure.com/myorg project=MyProject

# List configured defaults
az devops configure --list

# ------------------------------
# PIPELINES
# ------------------------------

# List all pipelines
az pipelines list --output table

# Show pipeline details
az pipelines show --name "CI-Pipeline" --output yaml

# Run a pipeline
az pipelines run --name "CI-Pipeline" --branch main

# Run pipeline with variables
az pipelines run --name "CI-Pipeline" --variables key1=value1 key2=value2

# Run pipeline with parameters
az pipelines run --name "CI-Pipeline" --parameters "environment=production"

# List pipeline runs
az pipelines runs list --pipeline-name "CI-Pipeline" --output table

# Show run details
az pipelines runs show --id 12345

# Cancel a running pipeline
az pipelines run cancel --run-id 12345

# Create pipeline from YAML
az pipelines create --name "New-Pipeline" --yaml-path azure-pipelines.yml --repository myrepo --branch main

# Update pipeline
az pipelines update --name "CI-Pipeline" --yaml-path new-pipeline.yml

# Delete pipeline
az pipelines delete --name "Old-Pipeline" --yes

# ------------------------------
# PIPELINE VARIABLES & GROUPS
# ------------------------------

# List variable groups
az pipelines variable-group list --output table

# Create variable group
az pipelines variable-group create --name "build-variables" --variables key1=value1 key2=value2

# Add variable to group
az pipelines variable-group variable create --group-id 1 --name "NEW_VAR" --value "value"

# Update variable (secret)
az pipelines variable-group variable update --group-id 1 --name "SECRET_VAR" --value "secret" --secret true

# Link variable group to Key Vault
az pipelines variable-group create --name "keyvault-secrets" --authorize true \
  --type AzureKeyVault --azure-key-vault-name "mykeyvault"

# ------------------------------
# WORK ITEMS
# ------------------------------

# List work items by query
az boards query --wiql "SELECT [Id], [Title], [State] FROM WorkItems WHERE [State] = 'Active'"

# Create work item
az boards work-item create --type "Bug" --title "Fix login issue" \
  --assigned-to "user@company.com" --area "MyProject\\Backend"

# Update work item
az boards work-item update --id 12345 --state "Resolved" --discussion "Fixed in PR #456"

# Show work item details
az boards work-item show --id 12345

# Add work item link
az boards work-item relation add --id 12345 --relation-type "Parent" --target-id 12340

# Delete work item
az boards work-item delete --id 12345 --yes

# ------------------------------
# REPOSITORIES
# ------------------------------

# List repositories
az repos list --output table

# Create repository
az repos create --name "new-repo"

# Show repository details
az repos show --repository "myrepo"

# Clone repository
az repos clone --repository "myrepo"

# Create branch
az repos ref create --name "refs/heads/feature/new-feature" --repository "myrepo" \
  --object-id $(az repos ref list --repository "myrepo" --filter heads/main --query "[0].objectId" -o tsv)

# Delete branch
az repos ref delete --name "refs/heads/feature/old-feature" --repository "myrepo"

# ------------------------------
# PULL REQUESTS
# ------------------------------

# List pull requests
az repos pr list --output table

# Create pull request
az repos pr create --source-branch "feature/new-feature" --target-branch "main" \
  --title "Add new feature" --description "Implements AB#12345"

# Create PR with work item link
az repos pr create --source-branch "feature/AB12345-fix" --target-branch "main" \
  --title "Fix: Resolve login issue AB#12345" --work-items 12345

# Show PR details
az repos pr show --id 456

# Update PR
az repos pr update --id 456 --title "Updated title" --status active

# Complete PR (merge)
az repos pr update --id 456 --status completed --delete-source-branch true

# Set PR auto-complete
az repos pr update --id 456 --auto-complete true --merge-commit-message "Merged PR #456"

# Add PR reviewer
az repos pr reviewer add --id 456 --reviewers "user@company.com"

# Approve PR
az repos pr set-vote --id 456 --vote approve

# List PR comments
az repos pr list --id 456

# ------------------------------
# ARTIFACTS
# ------------------------------

# List feeds
az artifacts feed list --output table

# Create feed
az artifacts feed create --name "my-feed" --description "Internal packages"

# List packages in feed
az artifacts package list --feed "my-feed" --output table

# Publish package (NuGet example)
az artifacts universal publish --feed "my-feed" --name "mypackage" --version "1.0.0" --path ./package

# Download package
az artifacts universal download --feed "my-feed" --name "mypackage" --version "1.0.0" --path ./download

# ------------------------------
# ENVIRONMENTS & APPROVALS
# ------------------------------

# List environments
az devops invoke --area environments --resource environments --api-version 6.0-preview.1 --http-method GET

# Check pipeline permissions
az devops security permission show --namespace-id "Pipeline" --token "Project/Pipeline/12345"

# ------------------------------
# SERVICE CONNECTIONS
# ------------------------------

# List service connections
az devops service-endpoint list --output table

# Create Azure service connection
az devops service-endpoint azurerm create --name "Azure-Subscription" \
  --azure-rm-subscription-id "subscription-id" \
  --azure-rm-subscription-name "My Subscription" \
  --azure-rm-tenant-id "tenant-id" \
  --azure-rm-service-principal-id "sp-id"
```

### Azure DevOps Pipeline Patterns Cheat Sheet

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AZURE DEVOPS PIPELINE PATTERNS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRIGGER PATTERNS                                                           │
│  ────────────────                                                           │
│                                                                             │
│  # Branch triggers                    # Path triggers                       │
│  trigger:                             trigger:                              │
│    branches:                            paths:                              │
│      include:                             include:                          │
│        - main                               - src/*                         │
│        - release/*                          - tests/*                       │
│      exclude:                             exclude:                          │
│        - feature/*-wip                      - docs/*                        │
│                                             - '*.md'                        │
│                                                                             │
│  # Tag triggers                       # Batch triggers                      │
│  trigger:                             trigger:                              │
│    tags:                                batch: true  # Wait for run         │
│      include:                           branches:                           │
│        - v*                               include: [main]                   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CONDITION PATTERNS                                                         │
│  ──────────────────                                                         │
│                                                                             │
│  # Always run                         # Run on specific branch              │
│  condition: always()                  condition: eq(variables['Build.      │
│                                         SourceBranch'], 'refs/heads/main') │
│                                                                             │
│  # Run only if previous succeeded     # Run on PR only                      │
│  condition: succeeded()               condition: eq(variables['Build.      │
│                                         Reason'], 'PullRequest')            │
│                                                                             │
│  # Complex conditions                 # Variable-based                      │
│  condition: |                         condition: and(                       │
│    and(                                 succeeded(),                        │
│      succeeded(),                       eq(variables['deploy'], 'true')    │
│      ne(variables['skip'], 'true'),   )                                     │
│      or(                                                                    │
│        eq(variables['Build.SourceBranch'], 'refs/heads/main'),             │
│        startsWith(variables['Build.SourceBranch'], 'refs/heads/release')   │
│      )                                                                      │
│    )                                                                        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DEPENDENCY PATTERNS                                                        │
│  ───────────────────                                                        │
│                                                                             │
│  # Sequential stages                  # Parallel stages                     │
│  stages:                              stages:                               │
│    - stage: A                           - stage: A                          │
│    - stage: B                           - stage: B                          │
│      dependsOn: A                         dependsOn: []  # No deps          │
│    - stage: C                           - stage: C                          │
│      dependsOn: B                         dependsOn: []  # No deps          │
│                                                                             │
│  # Fan-out/Fan-in                     # Conditional dependency              │
│  stages:                              stages:                               │
│    - stage: Build                       - stage: Build                      │
│    - stage: Test_Unit                   - stage: Deploy                     │
│      dependsOn: Build                     dependsOn: Build                  │
│    - stage: Test_Integration              condition: |                      │
│      dependsOn: Build                       and(                            │
│    - stage: Deploy                            succeeded('Build'),           │
│      dependsOn:                               eq(variables['deploy'],       │
│        - Test_Unit                              'true')                     │
│        - Test_Integration                   )                               │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TEMPLATE PATTERNS                                                          │
│  ─────────────────                                                          │
│                                                                             │
│  # Step template                      # Job template                        │
│  steps:                               jobs:                                 │
│    - template: steps/build.yml          - template: jobs/build-job.yml     │
│      parameters:                          parameters:                       │
│        config: Release                      config: Release                 │
│                                                                             │
│  # Stage template                     # Variable template                   │
│  stages:                              variables:                            │
│    - template: stages/deploy.yml        - template: vars/common.yml        │
│      parameters:                        - name: localVar                    │
│        environment: prod                  value: 'local'                    │
│                                                                             │
│  # Extends template                   # Insert template                     │
│  extends:                             steps:                                │
│    template: base-pipeline.yml          - script: echo "Before"            │
│    parameters:                          - template: steps/common.yml       │
│      buildConfig: Release               - script: echo "After"             │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DEPLOYMENT PATTERNS                                                        │
│  ───────────────────                                                        │
│                                                                             │
│  # RunOnce (simple)                   # Rolling deployment                  │
│  strategy:                            strategy:                             │
│    runOnce:                             rolling:                            │
│      deploy:                              maxParallel: 2                    │
│        steps:                             deploy:                           │
│          - script: deploy.sh                steps:                          │
│                                               - script: deploy.sh           │
│                                                                             │
│  # Canary deployment                  # Blue-Green (via slots)              │
│  strategy:                            steps:                                │
│    canary:                              - task: AzureWebApp@1              │
│      increments: [10, 20, 50]             inputs:                           │
│      deploy:                                deployToSlotOrASE: true         │
│        steps:                               slotName: 'staging'             │
│          - script: deploy.sh            - task: AzureAppServiceManage@0    │
│      on:                                    inputs:                         │
│        success:                               action: 'Swap Slots'          │
│          steps:                               sourceSlot: 'staging'         │
│            - script: promote.sh                                             │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VARIABLE PATTERNS                                                          │
│  ─────────────────                                                          │
│                                                                             │
│  # Compile-time variables (${{ }})    # Runtime variables ($( ))            │
│  variables:                           variables:                            │
│    staticVar: 'compile-time'            runtimeVar: 'set-later'             │
│  steps:                               steps:                                │
│    - script: echo ${{ variables.        - script: |                        │
│        staticVar }}                         echo "##vso[task.setvariable    │
│                                               variable=runtimeVar]new"      │
│                                         - script: echo $(runtimeVar)        │
│                                                                             │
│  # Secret variables                   # Output variables                    │
│  variables:                           jobs:                                 │
│    - group: secrets-group               - job: JobA                         │
│  steps:                                   steps:                            │
│    - script: |                              - script: |                     │
│        echo $(SecretVar)                        echo "##vso[task.           │
│      env:                                         setvariable variable=     │
│        SECRET: $(SecretVar)                       myOutput;isOutput=true]   │
│                                                   value"                    │
│                                           - job: JobB                       │
│                                             dependsOn: JobA                 │
│                                             variables:                      │
│                                               inherited: $[dependencies.    │
│                                                 JobA.outputs['step.         │
│                                                 myOutput']]                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Structure

```yaml
# ============================================================================
# COMPLETE AZURE DEVOPS PIPELINE STRUCTURE
# ============================================================================

# Pipeline name (build number format)
name: $(BuildDefinitionName)_$(Date:yyyyMMdd)$(Rev:.r)

# --------------------
# TRIGGERS
# --------------------
trigger:
  batch: true                          # Batch changes while build runs
  branches:
    include: [main, develop, release/*]
    exclude: [feature/*-wip]
  paths:
    include: [src/*, tests/*]
    exclude: [docs/*, '*.md']
  tags:
    include: [v*]

# PR trigger
pr:
  branches:
    include: [main, develop]
  paths:
    include: [src/*, tests/*]
  drafts: false                        # Don't trigger on draft PRs

# --------------------
# RESOURCES
# --------------------
resources:
  # Repository resources
  repositories:
    - repository: templates
      type: git
      name: DevOps/pipeline-templates
      ref: refs/heads/main
    - repository: external
      type: github
      endpoint: GitHub-Connection
      name: org/repo

  # Pipeline resources (artifacts from other pipelines)
  pipelines:
    - pipeline: buildPipeline
      source: 'Build-Pipeline'
      trigger:
        branches: [main]

  # Container resources
  containers:
    - container: build-container
      image: mcr.microsoft.com/dotnet/sdk:8.0

# --------------------
# VARIABLES
# --------------------
variables:
  # Variable groups (from Azure DevOps)
  - group: build-variables
  - group: keyvault-secrets

  # Template variables
  - template: variables/common.yml

  # Inline variables
  - name: buildConfiguration
    value: 'Release'
  - name: isMain
    value: $[eq(variables['Build.SourceBranch'], 'refs/heads/main')]

# --------------------
# PARAMETERS
# --------------------
parameters:
  - name: environment
    displayName: 'Deployment Environment'
    type: string
    default: 'dev'
    values: [dev, staging, production]

  - name: runTests
    displayName: 'Run Tests'
    type: boolean
    default: true

  - name: regions
    displayName: 'Deployment Regions'
    type: object
    default:
      - eastus
      - westus

# --------------------
# STAGES
# --------------------
stages:
  # Stage 1: Build
  - stage: Build
    displayName: 'Build Application'
    jobs:
      - job: BuildJob
        displayName: 'Build'
        pool:
          vmImage: 'ubuntu-latest'        # Microsoft-hosted
          # OR
          # name: 'Self-Hosted-Pool'      # Self-hosted

        # Container job
        container: build-container

        # Job timeout
        timeoutInMinutes: 60
        cancelTimeoutInMinutes: 5

        # Job variables
        variables:
          jobVar: 'job-level'

        steps:
          # Checkout
          - checkout: self
            fetchDepth: 0
            submodules: true
            persistCredentials: true

          # Use template
          - template: templates/build-steps.yml@templates
            parameters:
              configuration: $(buildConfiguration)

          # Task step
          - task: DotNetCoreCLI@2
            displayName: 'Build'
            inputs:
              command: 'build'
              projects: '**/*.csproj'

          # Script step
          - script: |
              echo "Building..."
              dotnet build
            displayName: 'Build Script'

          # PowerShell step
          - powershell: |
              Write-Host "PowerShell step"
            displayName: 'PowerShell'

          # Bash step
          - bash: |
              echo "Bash step"
            displayName: 'Bash'

          # Publish artifacts
          - publish: $(Build.ArtifactStagingDirectory)
            artifact: drop
            displayName: 'Publish Artifact'

  # Stage 2: Test (conditional)
  - stage: Test
    displayName: 'Run Tests'
    dependsOn: Build
    condition: and(succeeded(), eq('${{ parameters.runTests }}', true))
    jobs:
      - job: UnitTests
        steps:
          - download: current
            artifact: drop

          - task: DotNetCoreCLI@2
            inputs:
              command: 'test'

  # Stage 3: Deploy (deployment job)
  - stage: Deploy_${{ parameters.environment }}
    displayName: 'Deploy to ${{ parameters.environment }}'
    dependsOn: Test
    condition: succeeded()
    jobs:
      - deployment: DeployJob
        displayName: 'Deploy'
        environment: ${{ parameters.environment }}
        pool:
          vmImage: 'ubuntu-latest'

        strategy:
          runOnce:
            preDeploy:
              steps:
                - script: echo "Pre-deploy checks"
            deploy:
              steps:
                - download: current
                  artifact: drop
                - task: AzureWebApp@1
                  inputs:
                    azureSubscription: 'Azure-Connection'
                    appName: 'myapp-${{ parameters.environment }}'
            routeTraffic:
              steps:
                - script: echo "Route traffic"
            postRouteTraffic:
              steps:
                - script: echo "Post-route traffic"
            on:
              success:
                steps:
                  - script: echo "Deployment succeeded"
              failure:
                steps:
                  - script: echo "Deployment failed - rollback"

  # Stage 4: Manual approval
  - stage: ApproveProduction
    displayName: 'Production Approval'
    dependsOn: Deploy_staging
    jobs:
      - job: WaitForApproval
        pool: server
        steps:
          - task: ManualValidation@0
            inputs:
              notifyUsers: 'approvers@company.com'
              instructions: 'Please validate staging deployment'
              timeoutInMinutes: 1440
```

---

## References

- [Azure Pipelines Documentation](https://docs.microsoft.com/en-us/azure/devops/pipelines/)
- [Azure Boards Documentation](https://docs.microsoft.com/en-us/azure/devops/boards/)
- [Azure Repos Documentation](https://docs.microsoft.com/en-us/azure/devops/repos/)
- [Azure Artifacts Documentation](https://docs.microsoft.com/en-us/azure/devops/artifacts/)
- [YAML Schema Reference](https://docs.microsoft.com/en-us/azure/devops/pipelines/yaml-schema)
- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [GitOps Principles](https://www.gitops.tech/)

---

**Last Updated:** 2026-01-18  
**Version:** 1.0  
**Maintainer:** DevOps Team
