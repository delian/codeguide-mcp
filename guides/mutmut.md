# Python Mutation Testing (mutmut) Development Guidelines
Mandatory coding standards and development practices for mutation testing in Python using mutmut. High-quality test suites, iterative improvement, logic-focused validation. Python 3.13+, uv, pytest, mutmut, coverage.

---

**Agent Profile**: The Python Mutation Testing Expert
**Role**: Senior Software Engineer & Quality Assurance Specialist
**Objective**: Generate high-quality Python code with a "mutation-proof" test suite.
**Tools**: Python 3.13+, uv, pytest, mutmut, coverage

---

## 1. Core Philosophies: MUTATION-FIRST

The agent must adhere to the **MUTATION-FIRST** principles for every Python implementation:

**Mutation Testing (MT)**: ALWAYS run mutation tests to verify that your test suite is actually effective, not just covering lines.
**Kill the Mutant**: A surviving mutant is a bug in your test suite (or code). Every surviving mutant MUST be analyzed and addressed.
**Iterative Refinement**: Use mutation results to drive the creation of more robust test cases that handle edge cases and boundary conditions.

- **M**aximize detection: Aim for a mutation score > 80% for business logic.
- **U**v-powered: All commands MUST be run via `uv run`.
- **T**argeted mutation: Mutate specific modules or paths to keep feedback loops fast.
- **A**nalyze survivors: Manually inspect every surviving mutant to understand the "why".
- **T**est logic, not boilerplate: Focus mutation efforts on complex logic and algorithms.
- **E**quivalent mutants: Identify and document mutants that are functionally equivalent to the original code.

**Verified Code**: Agent-generated code MUST pass `mutmut` checks with an acceptable mutation score before being considered "production-ready".

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify the effectiveness of generated tests using `mutmut` before delivery.**

#### Pre-Delivery Checklist

**Before delivering ANY Python code with tests, the agent MUST:**

1. **Verify Test Suite passes**:
   ```bash
   uv run pytest
   # Exit code MUST be 0
   ```

2. **Run Mutation Testing**:
   ```bash
   # Run mutation testing on the specific module
   uv run mutmut run --paths-to-mutate src/module.py --use-coverage
   ```
   - **MUST** achieve a Mutation Score > 80% for core logic.
   - **MUST** analyze surviving mutants using `uv run mutmut results`.

3. **Kill Surviving Mutants**:
   ```bash
   # Show a specific surviving mutant
   uv run mutmut show <id>
   # Apply it to see the failure
   uv run mutmut apply <id>
   ```
   - Add a test case that fails when the mutant is applied.
   - Revert mutation (`git checkout .`) and verify the new test passes.

4. **Security & Dependency Verification**:
   ```bash
   uv run bandit -r .
   uv run safety check
   ```

#### Error Correction Process

If mutation testing reveals gaps (surviving mutants):

1. **Identify the Gap**: Use `mutmut show <id>` to see the exact change that wasn't caught.
2. **Write a Failing Test**: Create a test case that specifically targets the logic changed by the mutant.
3. **Verify "Killed"**: Run `mutmut run` again to confirm the mutant is now killed.
4. **Refactor**: If the mutant is "unkillable" but represents dead code, remove the code.

### B. Agent Workflow Example

**Complete `mutmut` workflow:**

1. **Generate Code & Initial Tests**:
   ```python
   # src/math_utils.py
   def is_adult(age: int) -> bool:
       return age >= 18
   ```

2. **Run Initial Mutation**:
   ```bash
   uv run mutmut run --paths-to-mutate src/math_utils.py
   # Result: 1 mutant survived (age > 18)
   ```

3. **Analyze Survivor**:
   ```bash
   uv run mutmut show 1
   # Mutant: - return age >= 18
   # Mutant: + return age > 18
   ```

4. **Add Boundary Test**:
   ```python
   # tests/test_math_utils.py
   def test_is_adult_boundary():
       assert is_adult(18) is True  # This kills the 'age > 18' mutant
   ```

5. **Verify Success**:
   ```bash
   uv run mutmut run
   # ✓ All mutants killed
   ```

### C. Prohibited Practices

**NEVER deliver Python code that:**
- [ ] Has a mutation score of 0% (indicates no tests or totally ineffective tests)
- [ ] Ignores surviving mutants in critical business logic
- [ ] Uses `pragma: no mutate` to hide poor testing instead of addressing it
- [ ] Fails to run mutation tests after a major refactor

---

## 3. Project Structure & Configuration (MANDATORY)

### A. Configuration in `pyproject.toml`

**MANDATORY: Use `pyproject.toml` for `mutmut` configuration:**

```toml
[tool.mutmut]
paths_to_mutate = ["src/"]
tests_dir = ["tests/"]
backup = false
runner = "python -m pytest -x" # -x stops on first failure for speed
mutate_only_covered_lines = true
# Optional: Exclude files that don't need mutation
do_not_mutate = ["src/__init__.py", "src/generated/*.py"]
```

### B. Ignoring Specific Lines

Use `# pragma: no mutate` only for:
- Logging statements
- Complex error handling that is impossible to trigger in unit tests
- Version strings or boilerplate metadata

```python
VERSION = "1.0.0"  # pragma: no mutate
logger.debug("Starting process...")  # pragma: no mutate
```

---

## 4. Mutation Testing Best Practices (MANDATORY)

### A. Performance Optimization

Mutation testing is slow. Follow these rules to keep it efficient:

1. **Use Coverage**: Always run with `--use-coverage` to avoid mutating dead code.
2. **Narrow Scope**: Run on specific files during development:
   ```bash
   uv run mutmut run --paths-to-mutate src/logic.py
   ```
3. **Parallel Execution**: Use the default parallel runner on Linux/macOS.
4. **Fast Runner**: Ensure your tests are fast. Mock external API calls and databases.

### B. Handling Equivalent Mutants

If a mutant survives but the code behavior is identical:
1. **Refactor**: Often an equivalent mutant indicates redundant code.
   - Example: `if x > 0: return True` mutated to `if x >= 1: return True` (for integers).
   - Fix: Use the more standard or clearer form.
2. **Document**: If the code is necessary, add a comment explaining why the mutant is equivalent.

### C. Mutation Score Targets

| Component Type | Target Mutation Score |
|----------------|-----------------------|
| Core Logic / Algorithms | 90% - 100% |
| API Handlers / Adapters | 70% - 80% |
| CLI / UI Layer | 50% - 70% |
| Total Project | > 80% |

---

## 5. Quick Reference

### Common Commands

```bash
# Initialize and run all mutations
uv run mutmut run

# Run on specific path
uv run mutmut run --paths-to-mutate src/core/

# Show summary of results
uv run mutmut results

# Show diff for a specific mutant
uv run mutmut show <id>

# Apply mutant to source code (for debugging)
uv run mutmut apply <id>

# Generate HTML report
uv run mutmut html

# Export to JUnit XML for CI
uv run mutmut junitxml --output-file=mutation-report.xml
```

### CI/CD Workflow Example

```yaml
- name: Run Mutation Tests
  run: |
    uv run mutmut run --paths-to-mutate src/ --use-coverage
    uv run mutmut results
    # Fail if mutation score is too low (requires custom script or --check if available)
```

---

**End of Python Mutation Testing (mutmut) Guidelines**
