# [TECHNOLOGY_NAME] Development Guidelines

This document provides mandatory coding standards and development practices for [TECHNOLOGY_NAME] development.

---

**Agent Profile**: The [TECHNOLOGY_NAME] Expert
**Role**: Senior [ROLE_DESCRIPTION] & [SECONDARY_SPECIALTY]
**Objective**: Generate production-ready, [QUALITY_ATTRIBUTES] code.
**Tools**: [TOOL_LIST_WITH_VERSIONS]

---

## 1. Core Philosophies: [ACRONYM]-FIRST

The agent must adhere to the **[ACRONYM]-FIRST** principles for every [TECHNOLOGY_NAME] implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **[LETTER_1]**[rest_of_word]: [Description of principle]
- **[LETTER_2]**[rest_of_word]: [Description of principle]
- **[LETTER_3]**[rest_of_word]: [Description of principle]
- **[LETTER_4]**[rest_of_word]: [Description of principle]
- **[LETTER_5]**[rest_of_word]: [Description of principle]

**Additional Principles:**

- [Additional principle 1]
- [Additional principle 2]
- [Additional principle 3]

**Verified Code**: Agent-generated code MUST [verification_requirements] before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated [TECHNOLOGY_NAME] code [VERIFICATION_CRITERIA] before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY [TECHNOLOGY_NAME] code, the agent MUST:**

1. **[Check_Type_1]**:
   ```bash
   # [Description of check]
   [command_1]
   # Exit code MUST be 0

   # [Additional check description]
   [command_2]
   ```
   - **MUST** [requirement_1]
   - [requirement_2]
   - [requirement_3]

2. **[Check_Type_2]**:
   ```bash
   # [Description of check]
   [command_1]

   # [Additional check description]
   [command_2]
   ```
   - **MUST** [requirement_1]
   - [requirement_2]

3. **[Check_Type_3]**:
   ```bash
   # [Description of check]
   [command_1]
   ```
   - [requirement_1]
   - [requirement_2]

4. **Documentation Verification**:
   ```bash
   # Generate/check documentation
   [documentation_command]
   ```
   - All public APIs have documentation
   - Documentation follows conventions
   - Examples compile and run successfully

#### Error Correction Process

If verification fails:

1. **[Error_Type_1]**:
   - Read full error message
   - Identify root cause
   - Fix the issue
   - Re-verify

2. **[Error_Type_2]**:
   - Run failing test in isolation
   - Check test expectations vs actual output
   - Fix logic errors
   - Re-run all tests to ensure no regressions

3. **[Error_Type_3]**:
   - [Specific remediation steps]
   - [Additional steps]

### B. Agent Workflow Example

**Complete [TECHNOLOGY_NAME] generation workflow:**

1. **Generate Code Structure**:
   ```
   project/
   ├── [directory_1]/
   │   └── [file_1]
   ├── [directory_2]/
   │   └── [file_2]
   └── [config_file]
   ```

2. **Generate Initial Code**:
   ```[language]
   // Example code
   [sample_code]
   ```

3. **Verify**:
   ```bash
   [verification_command]
   # ✓ Verification successful
   ```

4. **Add Tests**:
   ```[language]
   // Example test
   [sample_test]
   ```

5. **Run Tests**:
   ```bash
   [test_command]
   # ✓ All tests pass
   ```

6. **Final Verification**:
   ```bash
   [final_verification_commands]
   # ✓ All checks passed
   ```

7. **Present Code**: Only after ALL checks pass

### C. Prohibited Practices

**NEVER deliver [TECHNOLOGY_NAME] code that:**
- [ ] Fails [primary_check]
- [ ] Has failing tests
- [ ] Lacks tests for business logic
- [ ] Is not properly formatted
- [ ] Has [common_issue_1]
- [ ] Has [common_issue_2]
- [ ] Uses [anti_pattern_1]
- [ ] Uses [anti_pattern_2]
- [ ] **Fixes bugs without adding regression tests first**
- [ ] **Writes implementation before writing tests (violates TDD)**
- [ ] **Skips Red-Green-Refactor cycle for new features**

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

### Example TDD Workflow for [TECHNOLOGY_NAME]

```[language]
// Step 1: RED - Write failing test first
[failing_test_example]

// Run: [test_command]
// FAILS - [reason]

// Step 2: GREEN - Write minimal implementation
[minimal_implementation]

// Run: [test_command]
// PASSES - tests pass

// Step 3: REFACTOR - Improve code
[refactored_implementation]
// Tests still pass
```

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

```[language]
// Bug Report #[ID]: [Description]

// Step 1-2: Write test that reproduces the bug
[regression_test_example]

// Run: [test_command]
// FAILS - [failure_reason]

// Step 3: Fix the bug
[bug_fix_implementation]

// Run: [test_command]
// PASSES - bug fixed, regression prevented
```

---

## 3. Project Structure & Organization (MANDATORY)

### A. Standard Project Layout

**Follow the standard [TECHNOLOGY_NAME] project layout:**

```
project/
├── [src_directory]/           # Source code
│   ├── [main_module]/         # Main application
│   │   └── [entry_point]
│   ├── [core_module]/         # Business logic
│   │   ├── [domain]/          # Domain models
│   │   └── [services]/        # Business services
│   └── [adapters]/            # External integrations
│       ├── [database]/        # Database adapters
│       └── [http]/            # HTTP handlers
├── [test_directory]/          # Tests
│   ├── [unit]/                # Unit tests
│   └── [integration]/         # Integration tests
├── [config_directory]/        # Configuration
├── [docs_directory]/          # Documentation
├── [build_config]             # Build configuration
├── [dependency_file]          # Dependency management
└── README.md
```

### B. Package/Module Organization Principles

**Follow these principles for organization:**

1. **Group by Feature, Not by Type**:
   ```
   CORRECT - Group by domain
   [correct_structure_example]

   WRONG - Group by type
   [wrong_structure_example]
   ```

2. **Keep Modules Small and Focused**:
   - Each module should have a clear, single responsibility
   - [Additional guidance]

3. **Avoid Circular Dependencies**:
   - Dependency graph should be acyclic
   - Use interfaces/abstractions to break cycles

---

## 4. [ARCHITECTURE_PATTERN] Architecture (MANDATORY)

### A. Architecture Overview

**MANDATORY: Use [ARCHITECTURE_PATTERN] for clean separation:**

```
[ASCII_DIAGRAM_OF_ARCHITECTURE]
```

### B. Implementation Example

```[language]
// [Layer 1]: Domain/Core
[domain_example]

// [Layer 2]: Ports/Interfaces
[ports_example]

// [Layer 3]: Services/Use Cases
[services_example]

// [Layer 4]: Adapters/Infrastructure
[adapters_example]
```

**Benefits:**
- [Benefit 1]
- [Benefit 2]
- [Benefit 3]

---

## 5. Design Patterns (MANDATORY)

### A. [Pattern_1_Name]

**Use [Pattern_1_Name] for [use_case]:**

```[language]
[pattern_1_example]
```

**Benefits:**
- [Benefit 1]
- [Benefit 2]

### B. [Pattern_2_Name]

**Use [Pattern_2_Name] for [use_case]:**

```[language]
[pattern_2_example]
```

---

## 6. Configuration & Environment (MANDATORY)

### A. Configuration Management

**Use [CONFIGURATION_APPROACH] for configuration:**

```[language]
[configuration_example]
```

### B. Environment Variables

**Required environment variables:**

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| [VAR_1]  | [Description] | [default] | [Yes/No] |
| [VAR_2]  | [Description] | [default] | [Yes/No] |

---

## 7. Logging & Observability (MANDATORY)

### A. Structured Logging

**Use structured logging:**

```[language]
[logging_example]
```

### B. Metrics & Monitoring

**Implement observability:**

```[language]
[metrics_example]
```

---

## 8. Testing (MANDATORY)

### A. Unit Tests

**Use [TEST_PATTERN] for comprehensive coverage:**

```[language]
[unit_test_example]
```

### B. Integration Tests

```[language]
[integration_test_example]
```

### C. Test Coverage Requirements

- Minimum coverage: [COVERAGE_PERCENTAGE]% for business logic
- Critical paths: 100% coverage
- All public APIs must have tests

---

## 9. Error Handling (MANDATORY)

### A. Error Handling Strategy

**Follow [ERROR_HANDLING_APPROACH]:**

```[language]
[error_handling_example]
```

### B. Common Errors

| Error Type | Description | Handling |
|------------|-------------|----------|
| [Error_1]  | [Description] | [How to handle] |
| [Error_2]  | [Description] | [How to handle] |

---

## 10. Documentation (MANDATORY)

### A. Code Documentation

**Follow documentation conventions:**

```[language]
[documentation_example]
```

### B. Generate Documentation

```bash
# Generate documentation
[documentation_command]

# View documentation
[view_command]
```

---

## 11. Dependencies & Package Management (MANDATORY)

### A. Dependency Management

**Use [PACKAGE_MANAGER] for dependencies:**

```bash
# Add dependency
[add_dependency_command]

# Update dependencies
[update_command]

# Clean/verify dependencies
[clean_command]
```

### B. Dependency File

```[config_format]
[dependency_file_example]
```

---

## 12. Deployment Checklist

### Agent-Generated Code Verification (MANDATORY)

**If code was generated/modified by an agent, verify BEFORE delivery:**

#### Build & Compilation
- [ ] Code compiles: [compile_command] returns exit code 0
- [ ] No compilation errors or warnings
- [ ] All imports/dependencies resolved
- [ ] Code formatted: [format_command] produces no changes

#### Testing
- [ ] All tests pass: [test_command] returns exit code 0
- [ ] Reasonable coverage: [coverage_command] shows >[PERCENTAGE]%
- [ ] Integration tests pass (if applicable)

#### Code Quality
- [ ] Linter passes: [lint_command]
- [ ] No unused dependencies
- [ ] No circular dependencies
- [ ] Project structure follows standard layout

#### Documentation
- [ ] All public APIs have documentation
- [ ] Documentation follows conventions
- [ ] Examples provided for complex APIs

#### Architecture
- [ ] [Architecture pattern] followed
- [ ] Dependency injection used
- [ ] No global mutable state

#### Agent Workflow Completed
- [ ] Agent verified code compiles/builds successfully
- [ ] Agent ran all tests and verified they pass
- [ ] Agent ran formatters and linters
- [ ] Agent verified documentation
- [ ] Agent documented any fixes made during verification

---

## 13. Why This Configuration Works

**[Key_Benefit_1]**:
- [Explanation of why this approach is valuable]

**[Key_Benefit_2]**:
- [Explanation of why this approach is valuable]

**[Key_Benefit_3]**:
- [Explanation of why this approach is valuable]

**[Key_Benefit_4]**:
- [Explanation of why this approach is valuable]

---

## 14. Quick Reference

### Common Commands

```bash
# Build
[build_command]

# Test
[test_command]

# Lint
[lint_command]

# Format
[format_command]

# Run
[run_command]

# Documentation
[doc_command]
```

### Build Automation Template

```[build_tool_format]
[makefile_or_build_script_template]
```

---

**End of [TECHNOLOGY_NAME] Guidelines**
