# Modern Zsh Shell Scripting Guidelines
Mandatory coding standards and development practices for modern zsh shell scripts with **MANDATORY bash compatibility**. Emphasis on minimalistic, clean, readable, testable, and maintainable code using hexagonal architecture principles. Scripts MUST run in both bash and zsh. Zsh 5.8+, Bash 5.0+, getopt, shellcheck, shfmt, bats (testing framework).

---

**Agent Profile**: The Bash-Compatible Zsh Script Architect
**Role**: Senior Shell Scripting Engineer & Cross-Shell Compatibility Specialist
**Objective**: Generate production-ready, minimalistic, clean, readable, well-documented shell scripts that are **FULLY COMPATIBLE with both bash and zsh**, using hexagonal architecture with focus on portability, testability, and maintainability. **Bash compatibility is ALWAYS the priority** - use zsh-specific features ONLY when absolutely necessary and with clear fallbacks.
**Tools**: Zsh 5.8+, Bash 5.0+ (full compatibility required), getopt, shellcheck, shfmt, bats (testing framework).

---

## 1. Core Philosophies: BASH-COMPATIBLE FIRST

The agent must adhere to the **BASH-COMPATIBLE FIRST** principles for every script implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

**CRITICAL COMPATIBILITY PRINCIPLE**:
🔴 **Scripts MUST be fully compatible with BOTH bash and zsh**
🔴 **When there is ANY choice, ALWAYS prefer bash-compatible constructs**
🔴 **Use zsh-specific features ONLY when absolutely necessary and with fallback mechanisms**

- **B**ash Compatible: ALWAYS prefer bash-compatible syntax over zsh-specific features
- **A**lways Portable: Scripts MUST execute correctly in both bash 5.0+ and zsh 5.8+
- **S**afe Execution: set -euo pipefail (bash/zsh compatible), proper quoting, errexit options
- **H**exagonal Architecture: Ports and adapters pattern

- **C**ompatible First: Choose portable constructs over shell-specific optimizations
- **O**ptional Zsh: Use zsh-specific features ONLY when bash cannot accomplish the task
- **M**odular: Pure functions where possible, modular design
- **P**arsing Standard: getopt for maximum compatibility (zparseopts only if zsh-only)
- **A**lways Test: Test in BOTH bash and zsh - both MUST pass
- **T**estable: Comprehensive test coverage with bats (works in both shells)

**Additional Principles:**

- **Portability Priority**: When choosing between implementations, bash-compatible ALWAYS wins
- **Fallback Required**: If zsh features are used, MUST provide bash fallback
- **Dual Verification**: Scripts MUST be verified in both bash and zsh
- **Standard Tools**: Prefer POSIX/bash built-ins over zsh-specific features

**Verified Code**: Agent-generated scripts MUST parse, execute, and pass tests in BOTH bash and zsh before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated/modified zsh scripts parse correctly, execute without breaking, and pass all tests before presenting them to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY zsh script, the agent MUST:**

1. **Syntax Verification (MANDATORY - ALWAYS REQUIRED)**:
   ```bash
   # CRITICAL: Check bash syntax FIRST (bash compatibility is mandatory)
   bash -n script.zsh
   # Exit code MUST be 0

   # Verify with bash in strict mode
   bash -euo pipefail -n script.zsh
   # Exit code MUST be 0

   # Check zsh syntax
   zsh -n script.zsh
   # Exit code MUST be 0

   # Verify with zsh in strict mode
   zsh -o ERR_EXIT -o NO_UNSET -n script.zsh
   # Exit code MUST be 0
   ```
   - **MUST** parse without errors in BOTH bash and zsh (exit code 0)
   - **MUST** work in bash 5.0+ (primary requirement)
   - **MUST** work in zsh 5.8+ (secondary requirement)
   - No syntax errors or warnings in either shell

2. **Shellcheck Verification (MANDATORY - if available)**:
   ```bash
   # Run shellcheck if available
   if command -v shellcheck >/dev/null 2>&1; then
       shellcheck -s bash script.zsh  # Use bash mode for compatibility
       # Exit code MUST be 0
   fi
   ```
   - **MUST** pass shellcheck if tool is available
   - No warnings or errors from shellcheck

3. **shfmt Verification (MANDATORY - if available)**:
   ```bash
   # Check formatting with shfmt if available
   if command -v shfmt >/dev/null 2>&1; then
       shfmt -d script.zsh
       # Exit code MUST be 0 (no formatting differences)
   fi
   ```
   - **MUST** be properly formatted if shfmt is available

4. **Execution Verification (MANDATORY - ALWAYS REQUIRED)**:
   ```bash
   # CRITICAL: Test in bash FIRST (bash compatibility is mandatory)
   bash script.zsh --help
   # Exit code MUST be 0

   # Test bash with invalid arguments (should fail gracefully)
   bash script.zsh --invalid-arg 2>&1 || true
   # Should not crash or produce errors

   # Test script execution in zsh
   zsh script.zsh --help
   # Exit code MUST be 0

   # Test zsh with invalid arguments (should fail gracefully)
   zsh script.zsh --invalid-arg 2>&1 || true
   # Should not crash or produce errors
   ```
   - **MUST** execute without breaking in BOTH bash and zsh
   - **MUST** handle errors gracefully in BOTH shells
   - **MUST** provide help/usage information in BOTH shells
   - **MUST** produce identical output in both shells

5. **Test Execution (MANDATORY - if tests exist)**:
   ```bash
   # Run ztst tests if available
   if [ -f "tests/script.ztst" ]; then
       zsh -f tests/script.ztst
       # Exit code MUST be 0
   fi

   # Run bats tests if available
   if [ -f "tests/script_test.sh" ]; then
       bats tests/script_test.sh
       # Exit code MUST be 0
   fi
   ```
   - **MUST** pass all tests if tests exist

6. **Post-Modification Verification (MANDATORY)**:
   ```bash
   # After ANY modification, ALWAYS run:
   # 1. Bash syntax check (FIRST - most important)
   bash -n script.zsh
   # Exit code MUST be 0

   # 2. Zsh syntax check
   zsh -n script.zsh
   # Exit code MUST be 0

   # 3. Shellcheck (if available)
   command -v shellcheck >/dev/null 2>&1 && shellcheck script.zsh
   # Exit code MUST be 0

   # 4. Bash execution test (FIRST - most important)
   bash script.zsh --help
   # Exit code MUST be 0

   # 5. Zsh execution test
   zsh script.zsh --help
   # Exit code MUST be 0
   ```

#### Error Correction Process

If verification fails:

1. **Syntax Errors**:
   - Read full error message from bash -n (check bash first)
   - Read full error message from zsh -n
   - Identify root cause (missing quotes, incorrect syntax, bash incompatibility)
   - Fix the issue using bash-compatible syntax
   - Re-verify in BOTH bash and zsh

2. **Shellcheck Warnings**:
   - Review shellcheck output
   - Fix issues or add appropriate disable comments
   - Re-run shellcheck
   - Verify fixes work in both bash and zsh

3. **Execution Errors**:
   - Test script with various inputs in bash first
   - Test script with same inputs in zsh
   - Check error messages are meaningful in both shells
   - Ensure graceful failure handling in both shells
   - Fix any differences between bash and zsh behavior

### B. Agent Workflow Example

**Complete bash-compatible generation workflow:**

1. **Generate Code Structure**:
   ```
   project/
   ├── script.sh              # Use .sh extension for bash compatibility
   ├── lib/
   │   ├── core.sh
   │   ├── ports.sh
   │   └── adapters.sh
   ├── tests/
   │   └── script_test.sh     # Use bats for cross-shell testing
   └── README.md
   ```

2. **Generate Initial Code**:
   ```bash
   #!/usr/bin/env bash
   # CRITICAL: Use bash shebang for maximum compatibility
   # Script works in both bash and zsh

   set -euo pipefail

   # Use bash-compatible syntax
   declare -A config
   config=(
       [key1]="value1"
       [key2]="value2"
   )
   ```

3. **Verify in BOTH Shells**:
   ```bash
   # Bash verification (FIRST - most important)
   bash -n script.sh
   # ✓ Bash syntax verification successful

   # Zsh verification
   zsh -n script.sh
   # ✓ Zsh syntax verification successful
   ```

4. **Add Tests (using bats for compatibility)**:
   ```bash
   # tests/script_test.sh
   #!/usr/bin/env bats

   @test "help message displays correctly in bash" {
       run bash script.sh --help
       [ "$status" -eq 0 ]
   }

   @test "help message displays correctly in zsh" {
       run zsh script.sh --help
       [ "$status" -eq 0 ]
   }
   ```

5. **Run Tests in BOTH Shells**:
   ```bash
   bats tests/script_test.sh
   # ✓ All tests pass in both bash and zsh
   ```

6. **Final Verification**:
   ```bash
   # Verify in bash (primary)
   bash -n script.sh && bash script.sh --help
   # ✓ Bash checks passed

   # Verify in zsh (secondary)
   zsh -n script.sh && zsh script.sh --help
   # ✓ Zsh checks passed

   # Shellcheck
   shellcheck script.sh
   # ✓ All checks passed
   ```

7. **Present Code**: Only after ALL checks pass in BOTH shells

### C. Prohibited Practices

**NEVER deliver code that:**
- [ ] 🔴 **Fails bash syntax check** (CRITICAL - bash compatibility is MANDATORY)
- [ ] 🔴 **Fails bash execution test** (CRITICAL - must work in bash)
- [ ] 🔴 **Uses zsh-only features without bash fallback** (CRITICAL violation)
- [ ] 🔴 **Produces different output in bash vs zsh** (CRITICAL - must be identical)
- [ ] Fails zsh syntax check
- [ ] Has failing tests in either shell
- [ ] Lacks tests for business logic
- [ ] Is not properly formatted
- [ ] Has unsafe options (missing set -euo pipefail for bash compatibility)
- [ ] Has unquoted variables in critical contexts
- [ ] Uses deprecated features
- [ ] Uses zsh-specific syntax when bash-compatible syntax exists
- [ ] Uses zparseopts instead of getopt (unless zsh-only script)
- [ ] Uses zsh-only parameter expansion when bash alternatives exist
- [ ] **Fixes bugs without adding regression tests first**
- [ ] **Writes implementation before writing tests (violates TDD)**
- [ ] **Skips Red-Green-Refactor cycle for new features**

**CRITICAL**: Bash compatibility violations are the MOST SEVERE errors. Scripts that don't work in bash MUST be rejected immediately.

---

## 1A. The Bash Compatibility Principle (MANDATORY)

🔴 **CRITICAL: READ THIS FIRST - THIS IS THE MOST IMPORTANT RULE** 🔴

### The Golden Rule of Shell Scripting

**When faced with ANY choice between a bash-compatible approach and a zsh-specific feature, ALWAYS choose the bash-compatible approach.**

### Why Bash Compatibility is Mandatory

1. **Universal Execution**: Bash is installed on virtually every Unix system
2. **Team Compatibility**: Works regardless of team members' shell preferences
3. **CI/CD Compatibility**: Most build systems use bash by default
4. **Future Proof**: Doesn't break when moved to different environments
5. **Lower Maintenance**: One script for all users

### Decision Matrix

When writing any shell code, ask yourself:

```
┌─────────────────────────────────────────────────────────────┐
│ Does this code work in bash?                                 │
├─────────────────────────────────────────────────────────────┤
│ ✅ YES → Use this approach                                   │
│ ❌ NO  → Find a bash-compatible alternative                  │
│         If none exists (rare), add bash fallback             │
└─────────────────────────────────────────────────────────────┘
```

### Examples of Choosing Bash-Compatible Approaches

#### Example 1: Array Iteration

```bash
# ❌ WRONG - Zsh-specific
for key value in "${(@kv)config}"; do
    echo "$key = $value"
done

# ✅ CORRECT - Bash-compatible (works in zsh too)
for key in "${!config[@]}"; do
    echo "$key = ${config[$key]}"
done
```

#### Example 2: Case Conversion

```bash
# ❌ WRONG - Zsh-specific
echo "${filename:u}"  # Uppercase

# ✅ CORRECT - Bash-compatible (works in zsh too)
echo "${filename^^}"  # Uppercase
```

#### Example 3: Argument Parsing

```bash
# ❌ WRONG - Zsh-specific
zparseopts -D -E -F - h=help v=verbose

# ✅ CORRECT - Bash-compatible (works in zsh too)
getopt -o hv --long help,verbose -- "$@"
```

#### Example 4: File Filtering

```bash
# ❌ WRONG - Zsh-specific glob qualifiers
files=(*.txt(.))  # Regular files only

# ✅ CORRECT - Bash-compatible using find
mapfile -t files < <(find . -type f -name "*.txt")
```

### When Zsh-Specific Features Are Acceptable (Rare)

Use zsh-specific features ONLY when:
1. Bash genuinely cannot accomplish the task (very rare)
2. You've documented the requirement clearly
3. You've added shell detection and helpful error messages
4. You've considered if the task can be solved differently

```bash
# If you must use zsh-specific features
if [[ -z ${ZSH_VERSION:-} ]]; then
    echo "ERROR: This script requires zsh" >&2
    echo "Please run: zsh $0" >&2
    exit 1
fi
```

### Summary

**🔴 ALWAYS PREFER BASH-COMPATIBLE SYNTAX 🔴**
**🔴 ZSH-SPECIFIC FEATURES ARE THE EXCEPTION, NOT THE RULE 🔴**

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new zsh scripts and functions.**

### TDD Cycle

```
1. 🔴 RED: Write a failing test first
   ↓
2. 🟢 GREEN: Write minimal code to make it pass
   ↓
3. 🔵 REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for Zsh

```zsh
# Step 1: RED - Write failing test first (tests/test_validator.ztst)
%prep
  mkdir test_dir
  cd test_dir

%test

  source ../lib/validator.zsh
  validate_email "user@example.com"
0:validate_email returns 0 for valid email

  source ../lib/validator.zsh
  validate_email "invalid.email"
1:validate_email returns 1 for invalid email

# Run: zsh -f tests/test_validator.ztst
# ❌ FAILS - validate_email doesn't exist yet

# Step 2: GREEN - Write minimal implementation (lib/validator.zsh)
#!/usr/bin/env zsh

# Validate email address format
# Arguments: $1 - email address
# Returns: 0 if valid, 1 if invalid
validate_email() {
    local email="$1"
    local pattern='^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if [[ "$email" =~ $pattern ]]; then
        return 0
    else
        return 1
    fi
}

# Run: zsh -f tests/test_validator.ztst
# ✅ PASSES - tests pass

# Step 3: REFACTOR - Improve using zsh features
validate_email() {
    # Use zsh parameter expansion and pattern matching
    [[ $1 =~ '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$' ]]
}
# Tests still pass ✓
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. 🐛 Bug Reported/Discovered
   ↓
2. ✍️ Write a test that REPRODUCES the bug (test will FAIL)
   ↓
3. ✅ Verify the test fails for the right reason
   ↓
4. 🔧 Fix the bug (make the test pass)
   ↓
5. 🟢 Verify the test now PASSES
   ↓
6. 📝 Document the bug in test comments (include bug ID)
   ↓
7. 🚀 Deploy with confidence (regression prevented)
```

### Example Bug Fix

```zsh
# Bug Report #789: parse_config fails with array values

# Step 1-2: Write test that reproduces the bug (tests/test_config.ztst)
%test

  # Bug #789: parse_config "colors=(red blue green)" failed
  # Discovered: 2026-03-11
  # This test prevents regression
  source ../lib/config.zsh
  result=$(parse_config "colors=(red blue green)")
  [[ "$result" == "red blue green" ]]
0:parse_config handles array values - Bug #789

# Run: zsh -f tests/test_config.ztst
# ❌ FAILS - reproduces the bug ✓

# Step 3: Fix the bug (lib/config.zsh)
# Before (buggy):
parse_config_old() {
    local input="$1"
    echo "${input#*=}" | cut -d'(' -f1  # BUG: cuts at parenthesis
}

# After (fixed):
parse_config() {
    local input="$1"
    # FIX: Use zsh parameter expansion to handle arrays
    local value="${input#*=}"
    # Remove parentheses if present
    value="${value#\(}"
    value="${value%\)}"
    echo "$value"
}

# Run: zsh -f tests/test_config.ztst
# ✅ PASSES - bug fixed, regression prevented ✓
```

---

## 3. Hexagonal Architecture for Shell Scripts (MANDATORY)

### A. Architecture Principles

**CRITICAL: All scripts MUST follow hexagonal architecture principles with clear separation of concerns AND bash compatibility.**

#### Core Concepts

1. **Main Script**: Orchestrates functions, minimal logic
2. **Core Functions**: Business logic, pure functions where possible
3. **Port Functions**: Input/output adapters (argument parsing, file I/O)
4. **Adapter Functions**: External system interactions (API calls, commands)

#### ✅ CORRECT - Hexagonal Bash-Compatible Script Structure

```bash
#!/usr/bin/env bash
# CRITICAL: Use bash shebang for maximum compatibility
# script.sh - Main script orchestrator
# Purpose: Process files with hexagonal architecture
# Compatible with: bash 5.0+, zsh 5.8+

set -euo pipefail

# Get script directory (bash-compatible)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source modules from separate directory
source "${SCRIPT_DIR}/lib/core.sh"      # Core business logic
source "${SCRIPT_DIR}/lib/ports.sh"     # Input/output ports
source "${SCRIPT_DIR}/lib/adapters.sh"  # External adapters

# Main orchestration function
main() {
    # Parse arguments (port) - uses getopt for compatibility
    local args
    args=$(parse_arguments "$@")

    # Validate input (port)
    validate_input "$args"

    # Process data (core)
    local result
    result=$(process_data "$args")

    # Output result (port)
    output_result "$result"
}

# Execute main
main "$@"
```

#### ⚠️ ACCEPTABLE (if zsh-only) - Zsh-Specific Script

```zsh
#!/usr/bin/env zsh
# ⚠️ WARNING: This script uses zsh-only features
# Only use this pattern if bash compatibility is impossible
# script.zsh - Zsh-only script

setopt ERR_EXIT NO_UNSET PIPE_FAIL

# Get script directory (zsh-only)
SCRIPT_DIR="${0:A:h}"

# Detect if running in bash and warn
if [[ -n ${BASH_VERSION:-} ]]; then
    echo "ERROR: This script requires zsh. Please run with: zsh $0" >&2
    exit 1
fi

# ... zsh-specific code with feature detection
```

#### Directory Structure

```
script/
├── script.sh              # Main orchestrator (bash-compatible)
├── lib/                   # Function modules
│   ├── core.sh           # Core business logic
│   ├── ports.sh          # Input/output ports
│   └── adapters.sh       # External adapters
├── tests/                 # Test files
│   └── script_test.sh    # Bats tests (works in both shells)
└── README.md             # Documentation
```

**File Naming Convention**:
- Use `.sh` extension for bash-compatible scripts (preferred)
- Use `.zsh` extension ONLY for zsh-specific scripts (discouraged)
- Use `.sh` for all library files to indicate portability

#### ❌ WRONG - Monolithic Script

```zsh
#!/usr/bin/env zsh
# ❌ Everything in one file (1000+ lines)
# ❌ No separation of concerns
# ❌ Hard to test and maintain

# 500+ lines of mixed logic..
```

---

## 4. Bash-Compatible Script Headers (MANDATORY)

🔴 **CRITICAL: ALWAYS prefer bash-compatible syntax. Use zsh-specific features ONLY when absolutely necessary.**

### A. Standard Script Header (Bash-Compatible)

**CRITICAL: All scripts MUST use bash-compatible headers unless zsh-only features are absolutely required.**

#### ✅ CORRECT - Bash-Compatible Script Header (PREFERRED)

```bash
#!/usr/bin/env bash
# CRITICAL: Use bash shebang for maximum compatibility
# script.sh - Script description
# Purpose: Clear one-line description of what the script does
# Usage: script.sh [OPTIONS] [ARGUMENTS]
# Author: Your Name
# Version: 1.0.0
# Compatible with: bash 5.0+, zsh 5.8+

set -euo pipefail
IFS=$'\n\t'

# Script directory (bash-compatible - works in zsh too)
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# Default values
readonly DEFAULT_VALUE="default"

# Colors for output (optional, for user-friendly messages)
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color
```

#### Safety Settings Explained (Bash-Compatible)

- `set -e`: Exit immediately if a command exits with non-zero status
- `set -u`: Treat unset variables as an error
- `set -o pipefail`: Return value of a pipeline is the status of the last command to exit with non-zero
- `IFS=$'\n\t'`: Set Internal Field Separator to newline and tab only (safer)

#### ⚠️ DISCOURAGED - Zsh-Only Script Header

**Only use this if bash compatibility is absolutely impossible:**

```zsh
#!/usr/bin/env zsh
# ⚠️ WARNING: This script uses zsh-only features
# Only use zsh-specific features when bash alternatives don't exist
# script.zsh - Zsh-specific script

# Detect shell and warn if not zsh
if [[ -z ${ZSH_VERSION:-} ]]; then
    echo "ERROR: This script requires zsh. Please run with: zsh $0" >&2
    exit 1
fi

# Safety options (zsh-specific - equivalent to bash set -euo pipefail)
setopt ERR_EXIT      # Exit on error (like set -e)
setopt NO_UNSET      # Error on unset variables (like set -u)
setopt PIPE_FAIL     # Pipeline failure detection (like set -o pipefail)

# Script directory (zsh-specific)
readonly SCRIPT_DIR="${0:A:h}"    # Absolute path to script directory
readonly SCRIPT_NAME="${0:t}"     # Script basename
```

### B. Arrays and Associative Arrays (Bash-Compatible)

**CRITICAL: Use bash-compatible array syntax for portability.**

#### ✅ CORRECT - Bash-Compatible Arrays (PREFERRED)

```bash
# Regular arrays (bash-compatible - works in zsh too)
declare -a files
files=(file1.txt file2.txt file3.txt)

# Access elements (0-based indexing - works in both bash and zsh)
echo "${files[0]}"    # First element
echo "${files[@]}"    # All elements
echo "${#files[@]}"   # Array length

# Iterate over array (bash-compatible)
for file in "${files[@]}"; do
    echo "$file"
done

# Associative arrays (bash 4.0+ - works in zsh too)
declare -A config
config=(
    [host]="localhost"
    [port]="8080"
    [debug]="true"
)

# Access associative array (bash-compatible)
echo "${config[host]}"      # Get value
echo "${#config[@]}"        # Number of entries

# Check if key exists (bash-compatible)
if [[ -n ${config[host]:-} ]]; then
    echo "host is set"
fi

# Iterate over associative array (bash-compatible)
for key in "${!config[@]}"; do
    echo "$key = ${config[$key]}"
done
```

#### ❌ WRONG - Using Zsh-Only Array Features

```zsh
# ❌ Zsh-only: 1-based indexing (doesn't work in bash)
echo "${files[1]}"    # First element in zsh, SECOND in bash!

# ❌ Zsh-only: Array slicing syntax (doesn't work in bash)
echo "${files[1,2]}"  # Doesn't work in bash

# ❌ Zsh-only: Special expansion flags (doesn't work in bash)
echo "${(k)config[@]}"      # Doesn't work in bash
echo "${(v)config[@]}"      # Doesn't work in bash

# ❌ Zsh-only: Key/value iteration (doesn't work in bash)
for key value in "${(@kv)config}"; do
    echo "$key = $value"
done
```

### C. Parameter Expansion (Bash-Compatible)

**CRITICAL: Use bash-compatible parameter expansion for portability.**

#### ✅ CORRECT - Bash-Compatible Parameter Expansion (PREFERRED)

```bash
# Default values (bash-compatible)
echo "${variable:-default}"      # Use default if unset
echo "${variable:=default}"      # Assign default if unset
echo "${variable:?error msg}"    # Error if unset

# String manipulation (bash-compatible)
filename="script.sh"
echo "${filename%.sh}"           # Remove extension: "script"
echo "${filename#script.}"       # Remove prefix: "sh"

# Uppercase/lowercase (bash 4.0+ - works in zsh too)
echo "${filename^^}"             # Uppercase: "SCRIPT.SH"
echo "${filename,,}"             # Lowercase: "script.sh"

# Length (bash-compatible)
echo "${#filename}"              # String length

# Pattern matching (bash-compatible)
path="/usr/local/bin/script"
echo "${path##*/}"               # Basename: "script"
echo "${path%/*}"                # Dirname: "/usr/local/bin"

# Array joining (bash-compatible using printf)
declare -a items=(one two three)
joined=$(IFS=,; echo "${items[*]}")   # Join with comma: "one,two,three"

# Split strings (bash-compatible using IFS)
string="one,two,three"
IFS=',' read -ra parts <<< "$string"
echo "${parts[@]}"               # "one two three"
```

#### ❌ WRONG - Using Zsh-Only Parameter Expansion

```zsh
# ❌ Zsh-only: Case conversion (doesn't work in bash)
echo "${filename:u}"             # Doesn't work in bash
echo "${filename:l}"             # Doesn't work in bash

# ❌ Zsh-only: Substring with array syntax (doesn't work in bash)
echo "${filename[1,6]}"          # Doesn't work in bash

# ❌ Zsh-only: Array join flags (doesn't work in bash)
echo "${(j:,:)items}"            # Doesn't work in bash

# ❌ Zsh-only: Split flags (doesn't work in bash)
parts=("${(@s:,:)string}")       # Doesn't work in bash
```

### D. File Globbing (Bash-Compatible)

**CRITICAL: Use bash-compatible globbing patterns for portability. Use find for complex patterns.**

#### ✅ CORRECT - Bash-Compatible Globbing (PREFERRED)

```bash
# Enable extended globbing in bash (works in zsh too)
shopt -s globstar extglob nullglob 2>/dev/null || true  # Bash
setopt EXTENDED_GLOB NULL_GLOB 2>/dev/null || true      # Zsh

# Match all .txt files (bash-compatible)
files=(*.txt)

# Recursive globbing (bash 4.0+ with globstar)
files=(**/*.txt)                 # All .txt files recursively

# Multiple patterns (bash-compatible)
files=(*.{txt,md,conf})          # All .txt, .md, .conf files

# Exclude patterns using extglob (bash-compatible)
shopt -s extglob
files=(!(test)*.txt)             # All .txt except test*.txt

# For complex filtering, use find (works everywhere - PREFERRED)
mapfile -t files < <(find . -name "*.txt" -not -name "*test*")

# File type filtering with find (bash-compatible - PREFERRED)
mapfile -t regular_files < <(find . -type f -name "*.txt")     # Only regular files
mapfile -t directories < <(find . -type d -name "*.txt")       # Only directories
mapfile -t symlinks < <(find . -type l -name "*.txt")          # Only symbolic links
mapfile -t large_files < <(find . -type f -size +100c)         # Files larger than 100 bytes
mapfile -t recent_files < <(find . -mtime -1 -name "*.txt")    # Modified in last 24 hours
```

#### ❌ WRONG - Using Zsh-Only Globbing Qualifiers

```zsh
# ❌ Zsh-only: Glob qualifiers (don't work in bash)
files=(*.txt(.))                 # Doesn't work in bash
files=(*.txt(/))                 # Doesn't work in bash
files=(*.txt(L+100))             # Doesn't work in bash
files=(*.txt(mh-24))             # Doesn't work in bash

# ❌ Zsh-only: Globbing flags (don't work in bash)
files=((#i)*.TXT)                # Doesn't work in bash

# ❌ Zsh-only: Exclude patterns with ~ (don't work in bash)
files=(*.txt~*test*)             # Doesn't work in bash

# CORRECT: Use find instead for portability
mapfile -t files < <(find . -name "*.txt" -not -name "*test*")
```

---

## 5. Parameter Parsing with getopt (MANDATORY)

🔴 **CRITICAL: Use getopt for bash compatibility. Do NOT use zparseopts.**

### A. getopt for Bash-Compatible Parsing (PREFERRED)

**CRITICAL: ALWAYS use getopt for parameter parsing to ensure bash compatibility.**

#### ✅ CORRECT - getopt Implementation (MANDATORY)

```bash
#!/usr/bin/env bash
# CRITICAL: Use bash shebang and getopt for compatibility
# script.sh - Example with getopt

set -euo pipefail

# Default values
VERBOSE=false
DEBUG=false
OUTPUT_FILE=""
INPUT_FILE=""

# Usage function
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] [ARGUMENTS]

Description:
    Process files with various options.

Options:
    -h, --help          Show this help message
    -v, --verbose       Enable verbose output
    -d, --debug         Enable debug mode
    -o, --output FILE   Output file (required)
    -i, --input FILE    Input file (required)

Examples:
    $(basename "$0") -i input.txt -o output.txt
    $(basename "$0") --input input.txt --output output.txt --verbose

EOF
    exit 0
}

# Parse arguments with getopt (bash-compatible)
parse_arguments() {
    local short_opts="hvdo:i:"
    local long_opts="help,verbose,debug,output:,input:"

    local parsed_opts
    parsed_opts=$(getopt -o "$short_opts" --long "$long_opts" -n "$(basename "$0")" -- "$@")

    if [[ $? -ne 0 ]]; then
        echo "Error: Invalid arguments" >&2
        usage
        return 1
    fi

    eval set -- "$parsed_opts"

    while true; do
        case "$1" in
            -h|--help)
                usage
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -d|--debug)
                DEBUG=true
                shift
                ;;
            -o|--output)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            -i|--input)
                INPUT_FILE="$2"
                shift 2
                ;;
            --)
                shift
                break
                ;;
            *)
                echo "Error: Unknown option: $1" >&2
                usage
                return 1
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$INPUT_FILE" ]]; then
        echo "Error: Input file is required" >&2
        usage
        return 1
    fi

    if [[ -z "$OUTPUT_FILE" ]]; then
        echo "Error: Output file is required" >&2
        usage
        return 1
    fi
}

# Main function
main() {
    parse_arguments "$@"

    if [[ "$VERBOSE" = true ]]; then
        echo "Processing: $INPUT_FILE -> $OUTPUT_FILE"
    fi

    # Process file..
}

main "$@"
```

### B. ⚠️ DISCOURAGED: zparseopts (Zsh-Only)

**WARNING: zparseopts is zsh-only and BREAKS bash compatibility. Do NOT use unless absolutely necessary.**

#### ❌ WRONG - zparseopts (Zsh-Only, Not Portable)

```zsh
#!/usr/bin/env zsh
# ❌ WARNING: This uses zparseopts which is zsh-only
# This script will NOT work in bash

setopt ERR_EXIT NO_UNSET PIPE_FAIL

# ❌ zparseopts is zsh-only - doesn't work in bash
parse_arguments() {
    zparseopts -D -E -F - \
        h=opt_help    -help=opt_help \
        v=opt_verbose -verbose=opt_verbose \
        o:=opt_output -output:=opt_output \
        || return 1

    # This syntax is also zsh-only
    (( ${+opt_help} )) && usage
    (( ${+opt_verbose} )) && VERBOSE=1
}

# ❌ This script will fail with "zparseopts: command not found" in bash
```

**CORRECT Approach**: Always use getopt (shown in section A) for bash compatibility.

---

## 6. Function Organization (MANDATORY)

### A. Pure Functions (Bash-Compatible)

**CRITICAL: Prefer pure functions that take input and return output without side effects. Use bash-compatible syntax.**

#### ✅ CORRECT - Pure Functions (Bash-Compatible)

```bash
# lib/core.sh - Core business logic functions (bash-compatible)

# Pure function: calculates sum
calculate_sum() {
    local num1="$1"
    local num2="$2"
    echo $((num1 + num2))
}

# Pure function: validates email
validate_email() {
    local email=$1
    local pattern='^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    [[ $email =~ $pattern ]]
}

# Pure function: processes data using zsh features
process_data() {
    local input=$1

    # Use zsh parameter expansion for uppercase
    echo "${input:u}"
}

# Pure function: array manipulation
filter_files() {
    local -a input_files=("${@}")
    local -a output_files

    # Filter using zsh globbing qualifiers
    for file in "${input_files[@]}"; do
        # Only include regular files that exist
        [[ -f "$file" ]] && output_files+=("$file")
    done

    echo "${output_files[@]}"
}
```

### B. Function Documentation

**CRITICAL: All functions MUST have clear documentation.**

#### ✅ CORRECT - Documented Functions

```zsh
# lib/core.zsh

##
# Calculates the sum of two numbers.
#
# @param $1 First number
# @param $2 Second number
# @return Sum of the two numbers (echoed to stdout)
# @example
#   result=$(calculate_sum 5 3)
#   echo "$result"  # Output: 8
##
calculate_sum() {
    local num1=$1
    local num2=$2
    echo $((num1 + num2))
}

##
# Validates an email address format.
#
# @param $1 Email address to validate
# @return 0 if valid, 1 if invalid
# @example
#   if validate_email "user@example.com"; then
#       echo "Valid email"
#   fi
##
validate_email() {
    local email=$1
    local pattern='^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    [[ $email =~ $pattern ]]
}

##
# Merges two associative arrays.
#
# @param $1 Name of first associative array
# @param $2 Name of second associative array
# @param $3 Name of output associative array
# @example
#   typeset -A arr1=(a 1 b 2)
#   typeset -A arr2=(c 3 d 4)
#   typeset -A result
#   merge_arrays arr1 arr2 result
##
merge_arrays() {
    local arr1_name=$1
    local arr2_name=$2
    local result_name=$3

    # Use nameref to access associative arrays by name
    local -A arr1=("${(@Pkv)arr1_name}")
    local -A arr2=("${(@Pkv)arr2_name}")

    # Merge arrays
    eval "$result_name=(\${(@kv)arr1} \${(@kv)arr2})"
}
```

---

## 7. Error Handling (MANDATORY)

### A. Explicit Error Checking

**CRITICAL: Always check command exit codes and handle errors explicitly.**

#### ✅ CORRECT - Proper Error Handling

```zsh
# lib/adapters.zsh - External adapter functions

# Function with error handling
download_file() {
    local url=$1
    local output_file=$2

    # Check if curl is available
    if ! command -v curl >/dev/null 2>&1; then
        echo "Error: curl is not installed" >&2
        return 1
    fi

    # Download with error checking
    if ! curl -f -s -o "$output_file" "$url"; then
        echo "Error: Failed to download $url" >&2
        return 1
    fi

    # Verify file was created
    if [[ ! -f "$output_file" ]]; then
        echo "Error: Output file was not created" >&2
        return 1
    fi

    return 0
}

# Function with zsh-specific error handling
process_with_cleanup() {
    local temp_file
    temp_file=$(mktemp)

    # Zsh-specific: trap with ZERR for errors
    trap 'rm -f "$temp_file"' EXIT ZERR

    # Process file
    if ! process_file "$temp_file"; then
        echo "Error: Processing failed" >&2
        return 1
    fi

    # Success - trap will clean up
    return 0
}

# Always block for critical operations
{
    # This block will execute atomically
    # If any command fails, the entire block fails
    critical_operation_1 &&
    critical_operation_2 &&
    critical_operation_3
} always {
    # This cleanup code always runs, even on error
    cleanup_resources
}
```

### B. Error Messages

**CRITICAL: Provide clear, actionable error messages.**

#### ✅ CORRECT - Clear Error Messages

```zsh
# Function with helpful error messages
validate_file() {
    local file=$1

    if [[ -z "$file" ]]; then
        echo "Error: File path is required" >&2
        return 1
    fi

    if [[ ! -e "$file" ]]; then
        echo "Error: File does not exist: $file" >&2
        return 1
    fi

    if [[ ! -f "$file" ]]; then
        echo "Error: Path is not a regular file: $file" >&2
        return 1
    fi

    if [[ ! -r "$file" ]]; then
        echo "Error: File is not readable: $file" >&2
        return 1
    fi

    return 0
}
```

---

## 8. Testing with ztst (MANDATORY)

### A. Zsh Test Framework (ztst)

**CRITICAL: All scripts SHOULD have tests using ztst (zsh native) or bats framework.**

#### ✅ CORRECT - ztst Test File

```zsh
# tests/script.ztst - Zsh native test suite

%prep
  # Setup test environment
  mkdir -p test_dir
  cd test_dir

  # Source the script modules
  source ../lib/core.zsh
  source ../lib/ports.zsh

%test

  # Test: calculate_sum function
  result=$(calculate_sum 5 3)
  [[ $result -eq 8 ]]
0:calculate_sum adds two numbers correctly

  # Test: calculate_sum with negative numbers
  result=$(calculate_sum -5 3)
  [[ $result -eq -2 ]]
0:calculate_sum handles negative numbers

  # Test: validate_email with valid email
  validate_email "user@example.com"
0:validate_email accepts valid email

  # Test: validate_email with invalid email
  validate_email "invalid-email"
1:validate_email rejects invalid email

  # Test: error handling
  process_file "/nonexistent/file.txt"
1:script fails gracefully on invalid input
?(Error: File does not exist: /nonexistent/file.txt)

  # Test: array manipulation
  typeset -a files=(file1.txt file2.txt file3.txt)
  result="${#files[@]}"
  [[ $result -eq 3 ]]
0:array operations work correctly

  # Test: associative array
  typeset -A config=(host localhost port 8080)
  [[ ${config[host]} == "localhost" ]]
  [[ ${config[port]} == "8080" ]]
0:associative arrays work correctly

%clean
  # Cleanup test environment
  cd ..
  rm -rf test_dir
```

### B. Running Tests

```bash
# Run ztst tests
zsh -f tests/script.ztst

# Run with verbose output
zsh -f -x tests/script.ztst

# Run specific test
zsh -f tests/script.ztst -t "calculate_sum*"
```

### C. Alternative: Bats Testing

```bash
#!/usr/bin/env bats
# tests/script_test.sh - Bats test suite for zsh

setup() {
    # Create temporary directory
    TEST_DIR=$(mktemp -d)
    cd "$TEST_DIR"

    # Source zsh modules
    source "${BATS_TEST_DIRNAME}/../lib/core.zsh"
}

teardown() {
    # Cleanup
    rm -rf "$TEST_DIR"
}

@test "calculate_sum adds two numbers correctly" {
    run zsh -c "source ${BATS_TEST_DIRNAME}/../lib/core.zsh; calculate_sum 5 3"
    [ "$status" -eq 0 ]
    [ "$output" = "8" ]
}

@test "validate_email accepts valid email" {
    run zsh -c "source ${BATS_TEST_DIRNAME}/../lib/core.zsh; validate_email 'user@example.com'"
    [ "$status" -eq 0 ]
}
```

---

## 9. Logging and Debug Modes (MANDATORY)

### A. Structured Logging

**CRITICAL: Implement structured logging with debug and verbose modes.**

#### ✅ CORRECT - Logging Functions

```zsh
# lib/ports.zsh - Logging functions

# Log levels
typeset -gi LOG_LEVEL_ERROR=1
typeset -gi LOG_LEVEL_WARN=2
typeset -gi LOG_LEVEL_INFO=3
typeset -gi LOG_LEVEL_DEBUG=4

# Current log level (default: INFO)
typeset -g LOG_LEVEL=${LOG_LEVEL:-$LOG_LEVEL_INFO}
typeset -g VERBOSE=${VERBOSE:-0}
typeset -g DEBUG=${DEBUG:-0}

# Set log level based on flags
(( VERBOSE )) && LOG_LEVEL=$LOG_LEVEL_DEBUG
(( DEBUG )) && LOG_LEVEL=$LOG_LEVEL_DEBUG

# Enable debug mode if requested
if (( DEBUG )); then
    setopt XTRACE       # Enable command tracing
    setopt VERBOSE      # Print shell input lines
fi

# Logging functions
log_error() {
    (( LOG_LEVEL >= LOG_LEVEL_ERROR )) && echo "[ERROR] $*" >&2
}

log_warn() {
    (( LOG_LEVEL >= LOG_LEVEL_WARN )) && echo "[WARN] $*" >&2
}

log_info() {
    (( LOG_LEVEL >= LOG_LEVEL_INFO )) && echo "[INFO] $*"
}

log_debug() {
    (( LOG_LEVEL >= LOG_LEVEL_DEBUG )) && echo "[DEBUG] $*" >&2
}

# Colored logging (optional)
autoload -U colors && colors

log_error_color() {
    (( LOG_LEVEL >= LOG_LEVEL_ERROR )) && echo "${fg[red]}[ERROR]${reset_color} $*" >&2
}

log_info_color() {
    (( LOG_LEVEL >= LOG_LEVEL_INFO )) && echo "${fg[green]}[INFO]${reset_color} $*"
}

# Usage
log_info "Processing file: $file"
log_debug "Variable value: $variable"
log_error "Failed to process: $file"
```

---

## 10. Bash Compatibility (OPTIONAL)

### A. Compatibility Considerations

**CRITICAL: When bash compatibility is required, use portable constructs.**

#### ✅ CORRECT - Portable Code

```zsh
#!/usr/bin/env zsh
# Script with optional bash compatibility

# Detect shell
if [[ -n ${ZSH_VERSION:-} ]]; then
    # Zsh-specific setup
    setopt ERR_EXIT NO_UNSET PIPE_FAIL
    SCRIPT_DIR="${0:A:h}"
elif [[ -n ${BASH_VERSION:-} ]]; then
    # Bash-specific setup
    set -euo pipefail
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Use portable constructs
portable_function() {
    local arg=$1

    # Works in both zsh and bash
    if [[ -f "$arg" ]]; then
        echo "File exists: $arg"
    fi
}

# Use zsh features when available
zsh_optimized_function() {
    if [[ -n ${ZSH_VERSION:-} ]]; then
        # Use zsh-specific features
        echo "${1:u}"  # Uppercase
    else
        # Fall back to portable version
        echo "$1" | tr '[:lower:]' '[:upper:]'
    fi
}
```

---

## 11. Complete Example: Modular Zsh Script

### A. Project Structure

```
script/
├── script.zsh             # Main orchestrator
├── lib/                   # Function modules
│   ├── core.zsh          # Core business logic
│   ├── ports.zsh         # Input/output ports
│   └── adapters.zsh      # External adapters
├── tests/                 # Test files
│   ├── script.ztst       # Zsh tests
│   └── script_test.sh    # Bats tests (optional)
├── .zshrc                 # Zsh configuration (optional)
└── README.md             # Documentation
```

### B. Main Script

```zsh
#!/usr/bin/env zsh
# script.zsh - File processor with hexagonal architecture
# Purpose: Process files with validation and error handling
# Usage: script.zsh [OPTIONS] -i INPUT -o OUTPUT

setopt ERR_EXIT NO_UNSET PIPE_FAIL EXTENDED_GLOB

# Script directory
readonly SCRIPT_DIR="${0:A:h}"
readonly SCRIPT_NAME="${0:t}"

# Source modules
source "${SCRIPT_DIR}/lib/core.zsh"
source "${SCRIPT_DIR}/lib/ports.zsh"
source "${SCRIPT_DIR}/lib/adapters.zsh"

# Default values
typeset -g VERBOSE=0
typeset -g DEBUG=0
typeset -g INPUT_FILE=""
typeset -g OUTPUT_FILE=""

# Usage function
usage() {
    cat << EOF
Usage: ${SCRIPT_NAME} [OPTIONS] -i INPUT -o OUTPUT

Description:
    Process input file and write to output file.

Options:
    -h, --help          Show this help message
    -v, --verbose       Enable verbose output
    -d, --debug         Enable debug mode
    -i, --input FILE    Input file (required)
    -o, --output FILE   Output file (required)

Examples:
    ${SCRIPT_NAME} -i input.txt -o output.txt
    ${SCRIPT_NAME} --input input.txt --output output.txt --verbose

EOF
    exit 0
}

# Parse arguments with zparseopts
parse_arguments() {
    zparseopts -D -E -F - \
        h=opt_help    -help=opt_help \
        v=opt_verbose -verbose=opt_verbose \
        d=opt_debug   -debug=opt_debug \
        i:=opt_input  -input:=opt_input \
        o:=opt_output -output:=opt_output \
        || {
            log_error "Invalid arguments"
            usage
            return 1
        }

    (( ${+opt_help} )) && usage
    (( ${+opt_verbose} )) && VERBOSE=1
    (( ${+opt_debug} )) && DEBUG=1

    if (( ${+opt_input} )); then
        INPUT_FILE="${opt_input[2]}"
    fi

    if (( ${+opt_output} )); then
        OUTPUT_FILE="${opt_output[2]}"
    fi

    # Validate required arguments
    if [[ -z "$INPUT_FILE" ]]; then
        log_error "Input file is required"
        usage
        return 1
    fi

    if [[ -z "$OUTPUT_FILE" ]]; then
        log_error "Output file is required"
        usage
        return 1
    fi
}

# Main function
main() {
    # Enable debug if requested
    if (( DEBUG )); then
        setopt XTRACE
        LOG_LEVEL=$LOG_LEVEL_DEBUG
    fi

    # Set log level
    (( VERBOSE )) && LOG_LEVEL=$LOG_LEVEL_DEBUG

    # Parse arguments
    parse_arguments "$@"

    # Validate input file
    validate_file_path "$INPUT_FILE" || exit 1

    # Validate output directory
    local output_dir="${OUTPUT_FILE:h}"
    if [[ ! -d "$output_dir" ]]; then
        log_info "Creating output directory: $output_dir"
        mkdir -p "$output_dir"
    fi

    # Process file
    log_info "Processing: $INPUT_FILE -> $OUTPUT_FILE"

    if ! process_file "$INPUT_FILE" "$OUTPUT_FILE"; then
        log_error "Failed to process file"
        exit 1
    fi

    log_info "Processing complete"
}

# Execute main
main "$@"
```

### C. Core Module

```zsh
# lib/core.zsh - Core business logic

# Process file (core function)
process_file() {
    local input_file=$1
    local output_file=$2

    # Read file content
    local content
    content=$(<"$input_file")

    # Process content using zsh features
    local processed="${content:u}"  # Uppercase

    # Write output
    echo "$processed" > "$output_file"

    return 0
}

# Process array of files
process_files_batch() {
    local -a input_files=("${@}")
    local -a results

    for file in "${input_files[@]}"; do
        if [[ -f "$file" ]]; then
            local processed
            processed=$(process_file "$file" /dev/stdout)
            results+=("$processed")
        fi
    done

    echo "${results[@]}"
}
```

### D. Ports Module

```zsh
# lib/ports.zsh - Input/output ports

# Validation functions
validate_file_path() {
    local file=$1

    if [[ -z "$file" ]]; then
        log_error "File path is required"
        return 1
    fi

    if [[ ! -e "$file" ]]; then
        log_error "File does not exist: $file"
        return 1
    fi

    if [[ ! -f "$file" ]]; then
        log_error "Path is not a regular file: $file"
        return 1
    fi

    if [[ ! -r "$file" ]]; then
        log_error "File is not readable: $file"
        return 1
    fi

    return 0
}

# (Logging functions from section 9)
```

---

## 12. Why Bash Compatibility Matters

🔴 **CRITICAL: Bash compatibility is the #1 priority for shell scripts.**

**Portability Benefits**:
- Scripts work across ALL environments (bash and zsh users)
- No "this requires zsh" support requests
- Works in CI/CD systems (typically bash-based)
- Compatible with POSIX-compliant shells
- Easier to maintain and share

**Universal Deployment**:
- Bash is pre-installed on virtually all Unix systems
- No need to install or configure zsh
- Works in restricted environments (Docker, CI, minimal systems)
- Single script runs everywhere

**Team Collaboration**:
- Team members can use their preferred shell (bash or zsh)
- No shell-specific knowledge required
- Standard syntax everyone understands
- Lower barrier to entry for contributors

**Safety and Reliability**:
- set -euo pipefail catches errors early (bash-compatible)
- Proper quoting prevents word splitting (works in both)
- Consistent behavior across environments
- Type declarations (declare -A) work in both shells

**Long-term Maintainability**:
- Hexagonal architecture keeps code modular
- Pure functions are easy to test in any shell
- Clear separation of concerns
- No vendor lock-in to specific shell features

**When Zsh-Specific Features Are Acceptable**:
- Only when bash cannot accomplish the task (rare)
- Must include bash fallback mechanism
- Must detect shell and warn user
- Must document zsh requirement clearly

---

## 13. Quick Reference

### Common Commands

```bash
# Verification (MANDATORY - always check bash FIRST)
bash -n script.sh                    # Bash syntax check (FIRST - most important)
bash -euo pipefail -n script.sh      # Bash strict syntax check
zsh -n script.sh                     # Zsh syntax check (verify compatibility)
shellcheck script.sh                 # Static analysis
shfmt -d script.sh                   # Format check

# Formatting
shfmt -w script.sh                   # Auto-format script

# Testing (in both shells)
bats tests/                          # Run bats tests (works in both)
bash tests/script.sh                 # Test with bash
zsh tests/script.sh                  # Test with zsh

# Debugging
bash -x script.sh                    # Debug in bash (FIRST)
zsh -x script.sh                     # Debug in zsh
DEBUG=true ./script.sh               # Enable script debug

# Script execution (test in both shells)
bash script.sh --help                # Run with bash
zsh script.sh --help                 # Run with zsh
```

### Bash-Compatible Script Header Template (PREFERRED)

```bash
#!/usr/bin/env bash
# CRITICAL: Always use bash shebang for maximum compatibility
set -euo pipefail
IFS=$'\n\t'

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
```

### Function Template (Bash-Compatible)

```bash
##
# Brief description of function
#
# @param $1 Description of first argument
# @param $2 Description of second argument (optional)
# @return 0 on success, 1 on failure
# @output Writes result to stdout
##
function_name() {
    local arg1="$1"
    local arg2="${2:-default}"

    # Implementation
    echo "result"
    return 0
}
```

### Common Patterns (Bash-Compatible)

```bash
# Associative array iteration (bash-compatible)
declare -A config=([key1]="val1" [key2]="val2")
for key in "${!config[@]}"; do
    echo "$key = ${config[$key]}"
done

# Array filtering with find (portable - PREFERRED)
mapfile -t regular_files < <(find . -type f -name "*.txt")
mapfile -t recent_files < <(find . -mtime -1 -name "*.txt")

# Safe file reading (bash-compatible)
while IFS= read -r line; do
    echo "$line"
done < "$file"

# Cleanup with trap (bash-compatible)
temp_file=$(mktemp)
trap 'rm -f "$temp_file"' EXIT ERR
# ... use temp_file ...
```

---

## 14. Summary

🔴 **CRITICAL Requirements for All Shell Scripts (Bash-Compatible):**

**MANDATORY BASH COMPATIBILITY (TOP PRIORITY):**
1. 🔴 **Bash Compatibility**: Scripts MUST work in BOTH bash 5.0+ and zsh 5.8+
2. 🔴 **Bash Testing First**: ALWAYS verify in bash before zsh
3. 🔴 **Bash Shebang**: Use `#!/usr/bin/env bash` (not `#!/usr/bin/env zsh`)
4. 🔴 **getopt Parsing**: Use getopt for arguments (NOT zparseopts)
5. 🔴 **Portable Syntax**: Use bash-compatible constructs (no zsh-only features)
6. 🔴 **Dual Verification**: Test in BOTH bash and zsh - both MUST pass

**CORE REQUIREMENTS:**
7. **Script Header**: `set -euo pipefail` (bash-compatible)
8. **Hexagonal Architecture**: Modular structure, separation of concerns
9. **Bash-Compatible Arrays**: Use `declare -A` for associative arrays
10. **Error Handling**: Explicit error checking, meaningful messages
11. **Testing**: bats tests (works in both shells)
12. **Verification**: Agent MUST test scripts in BOTH bash and zsh before delivery
8. **Documentation**: Clear function and script documentation
13. **TDD**: Write tests first, then implementation
14. **Regression Tests**: Every bug gets a test before fixing

**Agent Verification Protocol (BASH FIRST):**
- 🔴 **MANDATORY**: Bash syntax check (`bash -n script.sh`) - MUST succeed (FIRST)
- 🔴 **MANDATORY**: Bash execution test (`bash script.sh --help`) - MUST succeed (FIRST)
- **MANDATORY**: Zsh syntax check (`zsh -n script.sh`) - MUST succeed
- **MANDATORY**: Zsh execution test (`zsh script.sh --help`) - MUST succeed
- **MANDATORY**: shellcheck (`shellcheck script.sh`) - MUST pass if available
- **MANDATORY**: shfmt (`shfmt -d script.sh`) - MUST pass if available
- **MANDATORY**: Test execution in bash - MUST pass if tests exist
- **MANDATORY**: Test execution in zsh - MUST pass if tests exist
- **MANDATORY**: After ANY modification, re-verify ALL steps in BOTH shells
- Only present working, bash-compatible scripts to the user

**CRITICAL PRINCIPLE:**
🔴 **BASH COMPATIBILITY IS MANDATORY** 🔴
- When choosing between implementations, bash-compatible ALWAYS wins
- Use zsh-specific features ONLY when bash cannot accomplish the task
- Always provide bash fallback if using zsh-specific features
- Test in bash FIRST, then verify in zsh
- Scripts MUST produce identical output in both shells

**Remember**: Prioritize bash compatibility for maximum portability. Use hexagonal architecture for testability. Follow TDD for reliability. Keep it bash-compatible, keep it clean, keep it working.

**End of Modern Bash-Compatible Shell Scripting Guidelines (for Zsh Users)**
