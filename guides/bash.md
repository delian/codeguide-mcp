# Modern Bash Shell Scripting Guidelines

This document provides mandatory coding standards and development practices for modern bash shell scripts with emphasis on minimalistic, clean, readable, testable, and maintainable code using hexagonal architecture principles.

---

**Agent Profile**: The Shell Script Architect  
**Role**: Senior Shell Scripting Engineer & Automation Specialist  
**Objective**: Generate production-ready, minimalistic, clean, readable, well-documented bash shell scripts using hexagonal architecture with focus on portability, testability, and maintainability.  
**Tools**: Bash 5.0+, zsh compatibility, getopt/getopts, shellcheck, shfmt, bats (testing framework).

---

## 1. Core Philosophies: BASH-FIRST

The agent must adhere to the **BASH-FIRST** principles for every bash script implementation:

- **C**lean Code: Minimalistic, single-purpose functions
- **L**ogical Organization: Hexagonal architecture, modular structure
- **E**xplicit Behavior: Clear error handling, no silent failures
- **A**utomated Testing: Testable, debuggable scripts
- **N**ative Tools: Prefer built-in commands, minimal external dependencies

- **S**afe Execution: set -euo pipefail, proper quoting
- **H**exagonal Architecture: Ports and adapters pattern
- **E**rror Handling: Explicit error checking, meaningful messages
- **L**ogging: Structured logging, debug modes
- **L**intable: shellcheck and shfmt compatible

**V**erified Scripts: Agent-generated scripts MUST parse, execute, and pass tests before delivery
- **E**xplicit Parameters: getopt/getopts for argument parsing
- **R**obust: Error handling, edge cases, input validation
- **I**dempotent: Safe to run multiple times
- **F**unctional: Pure functions where possible
- **I**nteractive: User-friendly, clear messages
- **E**fficient Execution: Fast, optimized commands

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Script Verification Protocol

**CRITICAL: Agents MUST verify that all generated/modified bash scripts parse correctly, execute without breaking, and pass all tests before presenting them to the user.**

#### Verification Checklist

**Before delivering ANY bash script, the agent MUST:**

1. **Syntax Verification (MANDATORY - ALWAYS REQUIRED)**:
   ```bash
   # Check bash syntax
   bash -n script.sh
   # Exit code MUST be 0
   
   # Check zsh compatibility
   zsh -n script.sh
   # Exit code MUST be 0
   
   # Verify with bash in strict mode
   bash -euo pipefail -n script.sh
   # Exit code MUST be 0
   ```
   - **MUST** parse without errors (exit code 0)
   - **MUST** be zsh compatible
   - No syntax errors or warnings

2. **Shellcheck Verification (MANDATORY - if available)**:
   ```bash
   # Run shellcheck if available
   if command -v shellcheck >/dev/null 2>&1; then
       shellcheck -x script.sh
       # Exit code MUST be 0
   fi
   ```
   - **MUST** pass shellcheck if tool is available
   - No warnings or errors from shellcheck

3. **shfmt Verification (MANDATORY - if available)**:
   ```bash
   # Check formatting with shfmt if available
   if command -v shfmt >/dev/null 2>&1; then
       shfmt -d script.sh
       # Exit code MUST be 0 (no formatting differences)
   fi
   ```
   - **MUST** be properly formatted if shfmt is available

4. **Execution Verification (MANDATORY - ALWAYS REQUIRED)**:
   ```bash
   # Test script execution (dry-run or help mode)
   bash script.sh --help
   # Exit code MUST be 0
   
   # Test with invalid arguments (should fail gracefully)
   bash script.sh --invalid-arg 2>&1 || true
   # Should not crash or produce errors
   ```
   - **MUST** execute without breaking
   - **MUST** handle errors gracefully
   - **MUST** provide help/usage information

5. **Test Execution (MANDATORY - if tests exist)**:
   ```bash
   # Run tests if available
   if [ -f "tests/script_test.sh" ]; then
       bats tests/script_test.sh
       # Exit code MUST be 0
   fi
   ```
   - **MUST** pass all tests if tests exist

6. **Post-Modification Verification (MANDATORY)**:
   ```bash
   # After ANY modification, ALWAYS run:
   # 1. Syntax check
   bash -n script.sh
   # Exit code MUST be 0
   
   # 2. Shellcheck (if available)
   command -v shellcheck >/dev/null 2>&1 && shellcheck script.sh
   # Exit code MUST be 0
   
   # 3. Execution test
   bash script.sh --help
   # Exit code MUST be 0
   ```

### B. Error Correction Process

If verification fails:

1. **Read the error message** - syntax errors, shellcheck warnings, execution errors
2. **Identify the root cause** - missing quotes, incorrect syntax, logic error
3. **Fix the issue** in the generated script
4. **Re-verify** by running checks again
5. **Repeat until successful** - iterate as many times as needed
6. **Only present working scripts** to the user

**CRITICAL**: Never provide bash scripts that don't parse or execute correctly. Always verify first, fix issues, then present the working solution.

**MANDATORY RULES:**
1. **Syntax check is ALWAYS required** - Script MUST parse successfully
2. **zsh compatibility is ALWAYS required** - Script MUST work in zsh
3. **Execution test is ALWAYS required** - Script MUST execute without breaking
4. **shellcheck/shfmt are MANDATORY if available** - Use tools when present

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new shell scripts and functions.**

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

### Example TDD Workflow for Bash

```bash
# Step 1: RED - Write failing test first (tests/test_validator.bats)
#!/usr/bin/env bats

load 'test_helper/bats-support/load'
load 'test_helper/bats-assert/load'

@test "validate_email returns 0 for valid email" {
    source ./lib/validator.sh
    run validate_email "user@example.com"
    assert_success
}

@test "validate_email returns 1 for invalid email" {
    source ./lib/validator.sh
    run validate_email "invalid.email"
    assert_failure
}

# Run: bats tests/test_validator.bats
# ❌ FAILS - validate_email doesn't exist yet

# Step 2: GREEN - Write minimal implementation (lib/validator.sh)
#!/usr/bin/env bash

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

# Run: bats tests/test_validator.bats
# ✅ PASSES - tests pass

# Step 3: REFACTOR - Improve if needed
# (Add logging, improve pattern, etc. while keeping tests green)
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

```bash
# Bug Report #456: parse_config fails with spaces in values

# Step 1-2: Write test that reproduces the bug (tests/test_config.bats)
@test "parse_config handles values with spaces - Bug #456" {
    # Bug: parse_config "key=value with spaces" returned only "value"
    # Discovered: 2026-01-18
    # This test prevents regression

    source ./lib/config.sh

    local result
    result=$(parse_config "name=John Doe")

    assert_equal "$result" "John Doe"
}

# Run: bats tests/test_config.bats
# ❌ FAILS - reproduces the bug ✓

# Step 3: Fix the bug (lib/config.sh)
# Before (buggy):
parse_config_old() {
    local input="$1"
    echo "${input#*=}" | cut -d' ' -f1  # BUG: cuts at first space
}

# After (fixed):
parse_config() {
    local input="$1"
    # FIX: Use parameter expansion to get everything after =
    echo "${input#*=}"
}

# Run: bats tests/test_config.bats
# ✅ PASSES - bug fixed, regression prevented ✓
```

### Prohibited Practices for Bug Fixes

**NEVER:**
- ❌ Fix a bug without adding a regression test first
- ❌ Write implementation before writing tests (violates TDD)
- ❌ Skip the Red-Green-Refactor cycle
- ❌ Commit code with failing tests
- ❌ Remove tests to make code pass
- ❌ Comment out failing tests instead of fixing them

---

## 3. Hexagonal Architecture for Shell Scripts (MANDATORY)

### A. Architecture Principles

**CRITICAL: All shell scripts MUST follow hexagonal architecture principles with clear separation of concerns.**

#### Core Concepts

1. **Main Script**: Orchestrates functions, minimal logic
2. **Core Functions**: Business logic, pure functions where possible
3. **Port Functions**: Input/output adapters (argument parsing, file I/O)
4. **Adapter Functions**: External system interactions (API calls, commands)

#### ✅ CORRECT - Hexagonal Shell Script Structure

```bash
#!/usr/bin/env bash
# script.sh - Main script orchestrator
# Purpose: Process files with hexagonal architecture

set -euo pipefail

# Source modules from separate directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/core.sh"      # Core business logic
source "${SCRIPT_DIR}/lib/ports.sh"     # Input/output ports
source "${SCRIPT_DIR}/lib/adapters.sh"   # External adapters

# Main orchestration function
main() {
    # Parse arguments (port)
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

#### Directory Structure

```
script/
├── script.sh              # Main orchestrator (minimal)
├── lib/                    # Function modules
│   ├── core.sh            # Core business logic
│   ├── ports.sh           # Input/output ports
│   └── adapters.sh        # External adapters
├── tests/                  # Test files
│   └── script_test.sh     # Bats tests
└── README.md              # Documentation
```

#### ❌ WRONG - Monolithic Script

```bash
#!/bin/bash
# ❌ Everything in one file (1000+ lines)
# ❌ No separation of concerns
# ❌ Hard to test and maintain

# 500+ lines of mixed logic...
```

---

## 4. Script Header and Safety (MANDATORY)

### A. Standard Script Header

**CRITICAL: All bash scripts MUST start with proper header and safety settings.**

#### ✅ CORRECT - Complete Script Header

```bash
#!/usr/bin/env bash
# script.sh - Script description
# Purpose: Clear one-line description of what the script does
# Usage: script.sh [OPTIONS] [ARGUMENTS]
# Author: Your Name
# Version: 1.0.0

set -euo pipefail
IFS=$'\n\t'

# Script directory (for sourcing modules)
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

#### Safety Settings Explained

- `set -e`: Exit immediately if a command exits with non-zero status
- `set -u`: Treat unset variables as an error
- `set -o pipefail`: Return value of a pipeline is the status of the last command to exit with non-zero status
- `IFS=$'\n\t'`: Set Internal Field Separator to newline and tab only (safer)

#### ❌ WRONG - Missing Safety Settings

```bash
#!/bin/bash
# ❌ No safety settings
# ❌ Script continues on errors
# ❌ Unset variables cause issues

# Dangerous: continues even if command fails
command_that_might_fail
echo "This runs even if above command failed"
```

---

## 5. Parameter Parsing with getopt (MANDATORY)

### A. getopt for Long Options

**CRITICAL: Use getopt for parameter parsing with long options support.**

#### ✅ CORRECT - getopt Implementation

```bash
#!/usr/bin/env bash
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
Usage: ${0##*/} [OPTIONS] [ARGUMENTS]

Description:
    Process files with various options.

Options:
    -h, --help          Show this help message
    -v, --verbose       Enable verbose output
    -d, --debug         Enable debug mode
    -o, --output FILE   Output file (required)
    -i, --input FILE    Input file (required)

Examples:
    ${0##*/} -i input.txt -o output.txt
    ${0##*/} --input input.txt --output output.txt --verbose

EOF
    exit 0
}

# Parse arguments with getopt
parse_arguments() {
    # Define short and long options
    local short_opts="hvd:o:i:"
    local long_opts="help,verbose,debug:,output:,input:"
    
    # Parse options
    local parsed_opts
    parsed_opts=$(getopt -o "$short_opts" --long "$long_opts" -n "${0##*/}" -- "$@")
    
    if [ $? -ne 0 ]; then
        echo "Error: Invalid arguments" >&2
        usage
        exit 1
    fi
    
    # Set positional parameters
    eval set -- "$parsed_opts"
    
    # Process options
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
                exit 1
                ;;
        esac
    done
    
    # Validate required arguments
    if [ -z "$INPUT_FILE" ]; then
        echo "Error: Input file is required" >&2
        usage
        exit 1
    fi
    
    if [ -z "$OUTPUT_FILE" ]; then
        echo "Error: Output file is required" >&2
        usage
        exit 1
    fi
}

# Main function
main() {
    parse_arguments "$@"
    
    if [ "$VERBOSE" = true ]; then
        echo "Processing: $INPUT_FILE -> $OUTPUT_FILE"
    fi
    
    # Process file...
}

main "$@"
```

### B. getopts for Simple Cases

**CRITICAL: Use getopts for simple cases without long options.**

#### ✅ CORRECT - getopts Implementation

```bash
#!/usr/bin/env bash
# script.sh - Simple getopts example

set -euo pipefail

VERBOSE=false
FILE=""

while getopts "hvf:" opt; do
    case "$opt" in
        h)
            echo "Usage: ${0##*/} [-h] [-v] [-f FILE]"
            exit 0
            ;;
        v)
            VERBOSE=true
            ;;
        f)
            FILE="$OPTARG"
            ;;
        *)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

shift $((OPTIND-1))

# Remaining arguments
if [ $# -gt 0 ]; then
    echo "Additional arguments: $*"
fi
```

#### ❌ WRONG - Manual Argument Parsing

```bash
# ❌ Manual parsing - error-prone
if [ "$1" = "-v" ]; then
    VERBOSE=true
    shift
fi
# ❌ Doesn't handle --verbose
# ❌ Doesn't validate properly
```

---

## 6. Function Organization (MANDATORY)

### A. Pure Functions

**CRITICAL: Prefer pure functions that take input and return output without side effects.**

#### ✅ CORRECT - Pure Functions

```bash
# lib/core.sh - Core business logic functions

# Pure function: calculates sum
calculate_sum() {
    local num1="$1"
    local num2="$2"
    echo $((num1 + num2))
}

# Pure function: validates email
validate_email() {
    local email="$1"
    local pattern='^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if [[ "$email" =~ $pattern ]]; then
        return 0
    else
        return 1
    fi
}

# Pure function: processes data
process_data() {
    local input="$1"
    local processed
    
    # Process input
    processed=$(echo "$input" | tr '[:lower:]' '[:upper:]')
    
    echo "$processed"
}
```

### B. Function Documentation

**CRITICAL: All functions MUST have clear documentation.**

#### ✅ CORRECT - Documented Functions

```bash
# lib/core.sh

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
    local num1="$1"
    local num2="$2"
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
    local email="$1"
    local pattern='^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if [[ "$email" =~ $pattern ]]; then
        return 0
    else
        return 1
    fi
}
```

---

## 7. Error Handling (MANDATORY)

### A. Explicit Error Checking

**CRITICAL: Always check command exit codes and handle errors explicitly.**

#### ✅ CORRECT - Proper Error Handling

```bash
# lib/adapters.sh - External adapter functions

# Function with error handling
download_file() {
    local url="$1"
    local output_file="$2"
    
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
    if [ ! -f "$output_file" ]; then
        echo "Error: Output file was not created" >&2
        return 1
    fi
    
    return 0
}

# Function with trap for cleanup
process_with_cleanup() {
    local temp_file
    temp_file=$(mktemp)
    
    # Set trap for cleanup
    trap 'rm -f "$temp_file"' EXIT ERR
    
    # Process file
    if ! process_file "$temp_file"; then
        echo "Error: Processing failed" >&2
        return 1
    fi
    
    # Success - trap will clean up
    return 0
}
```

### B. Error Messages

**CRITICAL: Provide clear, actionable error messages.**

#### ✅ CORRECT - Clear Error Messages

```bash
# Function with helpful error messages
validate_file() {
    local file="$1"
    
    if [ -z "$file" ]; then
        echo "Error: File path is required" >&2
        return 1
    fi
    
    if [ ! -e "$file" ]; then
        echo "Error: File does not exist: $file" >&2
        return 1
    fi
    
    if [ ! -f "$file" ]; then
        echo "Error: Path is not a regular file: $file" >&2
        return 1
    fi
    
    if [ ! -r "$file" ]; then
        echo "Error: File is not readable: $file" >&2
        return 1
    fi
    
    return 0
}
```

#### ❌ WRONG - Poor Error Handling

```bash
# ❌ No error checking
download_file() {
    curl "$1" > "$2"
    # What if curl fails? Script continues...
}

# ❌ Vague error messages
validate_file() {
    if [ ! -f "$1" ]; then
        echo "Error"  # What error? What file?
        exit 1
    fi
}
```

---

## 8. Quoting and Variable Safety (MANDATORY)

### A. Proper Quoting

**CRITICAL: Always quote variables to prevent word splitting and pathname expansion.**

#### ✅ CORRECT - Proper Quoting

```bash
# Always quote variables
file="/path/to/file with spaces.txt"
process_file "$file"

# Quote in conditionals
if [ -f "$file" ]; then
    echo "File exists: $file"
fi

# Quote in loops
for item in "$@"; do
    process_item "$item"
done

# Quote in command substitution
result="$(command "$arg1" "$arg2")"

# Quote array elements
files=("file1.txt" "file2.txt")
for file in "${files[@]}"; do
    process_file "$file"
done
```

#### ❌ WRONG - Missing Quotes

```bash
# ❌ Unquoted variable - breaks with spaces
file="/path/to/file with spaces.txt"
process_file $file  # Breaks into multiple arguments!

# ❌ Unquoted in conditional
if [ -f $file ]; then  # Breaks with spaces
    echo "File exists"
fi

# ❌ Unquoted in loop
for item in $@; do  # Breaks with spaces
    process_item $item
done
```

### B. Array Handling

**CRITICAL: Use proper array syntax for zsh compatibility.**

#### ✅ CORRECT - Array Syntax

```bash
# Declare arrays explicitly
declare -a files=("file1.txt" "file2.txt")

# Access array elements
echo "${files[0]}"
echo "${files[@]}"  # All elements

# Iterate over array
for file in "${files[@]}"; do
    echo "$file"
done

# Array length
echo "Count: ${#files[@]}"
```

---

## 9. Logging and Debug Modes (MANDATORY)

### A. Structured Logging

**CRITICAL: Implement structured logging with debug and verbose modes.**

#### ✅ CORRECT - Logging Functions

```bash
# lib/ports.sh - Logging functions

# Log levels
readonly LOG_LEVEL_ERROR=1
readonly LOG_LEVEL_WARN=2
readonly LOG_LEVEL_INFO=3
readonly LOG_LEVEL_DEBUG=4

# Current log level (default: INFO)
LOG_LEVEL=${LOG_LEVEL:-$LOG_LEVEL_INFO}
VERBOSE=${VERBOSE:-false}
DEBUG=${DEBUG:-false}

# Set log level based on flags
if [ "$VERBOSE" = true ]; then
    LOG_LEVEL=$LOG_LEVEL_DEBUG
fi

if [ "$DEBUG" = true ]; then
    LOG_LEVEL=$LOG_LEVEL_DEBUG
    set -x  # Enable command tracing
fi

# Logging functions
log_error() {
    if [ $LOG_LEVEL -ge $LOG_LEVEL_ERROR ]; then
        echo "[ERROR] $*" >&2
    fi
}

log_warn() {
    if [ $LOG_LEVEL -ge $LOG_LEVEL_WARN ]; then
        echo "[WARN] $*" >&2
    fi
}

log_info() {
    if [ $LOG_LEVEL -ge $LOG_LEVEL_INFO ]; then
        echo "[INFO] $*"
    fi
}

log_debug() {
    if [ $LOG_LEVEL -ge $LOG_LEVEL_DEBUG ]; then
        echo "[DEBUG] $*" >&2
    fi
}

# Usage
log_info "Processing file: $file"
log_debug "Variable value: $variable"
log_error "Failed to process: $file"
```

### B. Debug Mode

**CRITICAL: Support debug mode for troubleshooting.**

#### ✅ CORRECT - Debug Mode

```bash
# Enable debug mode
enable_debug() {
    if [ "${DEBUG:-false}" = true ]; then
        set -x  # Print commands before execution
        PS4='+ [${BASH_SOURCE}:${LINENO}]: '  # Show file and line
    fi
}

# Usage in script
enable_debug

# Debug output
if [ "${DEBUG:-false}" = true ]; then
    echo "Debug: Variable value is $variable" >&2
    echo "Debug: Current directory is $(pwd)" >&2
fi
```

---

## 10. Testing with Bats (MANDATORY)

### A. Test Structure

**CRITICAL: All scripts MUST have tests using bats framework.**

#### ✅ CORRECT - Bats Test File

```bash
#!/usr/bin/env bats
# tests/script_test.sh - Test suite for script.sh

load "${BATS_TEST_DIRNAME}/../lib/core.sh"
load "${BATS_TEST_DIRNAME}/../lib/ports.sh"

# Test setup
setup() {
    # Create temporary directory
    TEST_DIR=$(mktemp -d)
    cd "$TEST_DIR"
}

# Test teardown
teardown() {
    # Cleanup
    rm -rf "$TEST_DIR"
}

# Test: calculate_sum function
@test "calculate_sum adds two numbers correctly" {
    result=$(calculate_sum 5 3)
    [ "$result" -eq 8 ]
}

@test "calculate_sum handles negative numbers" {
    result=$(calculate_sum -5 3)
    [ "$result" -eq -2 ]
}

# Test: validate_email function
@test "validate_email accepts valid email" {
    validate_email "user@example.com"
}

@test "validate_email rejects invalid email" {
    run validate_email "invalid-email"
    [ "$status" -ne 0 ]
}

# Test: error handling
@test "script fails gracefully on invalid input" {
    run process_file "/nonexistent/file.txt"
    [ "$status" -ne 0 ]
    [ "$output" = "Error: File does not exist: /nonexistent/file.txt" ]
}
```

### B. Running Tests

```bash
# Run all tests
bats tests/

# Run specific test file
bats tests/script_test.sh

# Run with verbose output
bats -v tests/

# Run with tap format
bats --tap tests/
```

---

## 11. zsh Compatibility (MANDATORY)

### A. zsh-Compatible Syntax

**CRITICAL: All scripts MUST be compatible with both bash and zsh.**

#### ✅ CORRECT - zsh-Compatible Code

```bash
#!/usr/bin/env bash
# Script compatible with both bash and zsh

# Use bash-compatible array syntax
declare -a items=("item1" "item2" "item3")

# Access array elements (works in both)
echo "${items[0]}"
echo "${items[@]}"

# Array length (works in both)
echo "${#items[@]}"

# Conditional syntax (works in both)
if [[ -n "${variable:-}" ]]; then
    echo "Variable is set"
fi

# Pattern matching (works in both)
if [[ "$file" =~ \.txt$ ]]; then
    echo "Text file"
fi

# Command substitution (works in both)
result=$(command "$arg")
```

#### ❌ WRONG - bash-Only Syntax

```bash
# ❌ bash-only array syntax (doesn't work in zsh)
items=(item1 item2 item3)
echo "${items[0]}"  # zsh uses 1-based indexing by default

# ❌ bash-only parameter expansion
${variable:0:5}  # Use ${variable%%*} instead for compatibility
```

### B. zsh-Specific Considerations

**CRITICAL: Handle zsh-specific behaviors.**

```bash
# Handle zsh array indexing
if [ -n "${ZSH_VERSION:-}" ]; then
    setopt KSH_ARRAYS  # Use ksh-style arrays (0-based)
fi

# Handle zsh option differences
if [ -n "${ZSH_VERSION:-}" ]; then
    setopt SH_WORD_SPLIT  # Enable word splitting like bash
fi
```

---

## 12. shellcheck and shfmt (MANDATORY)

### A. shellcheck Configuration

**CRITICAL: Scripts MUST pass shellcheck if the tool is available.**

#### ✅ CORRECT - shellcheck-Compatible Code

```bash
#!/usr/bin/env bash
# shellcheck disable=SC2034  # Unused variable (if intentional)
# shellcheck source=./lib/core.sh

set -euo pipefail

# shellcheck disable=SC1091  # Source file not found (if conditional)
if [ -f "./lib/core.sh" ]; then
    source "./lib/core.sh"
fi

# Proper quoting (shellcheck will verify)
file="/path/to/file"
process_file "$file"

# Proper error handling
if ! command -v tool >/dev/null 2>&1; then
    echo "Error: tool not found" >&2
    exit 1
fi
```

### B. shfmt Formatting

**CRITICAL: Scripts MUST be formatted with shfmt if the tool is available.**

#### ✅ CORRECT - shfmt-Formatted Code

```bash
#!/usr/bin/env bash
# shfmt formats code consistently

set -euo pipefail

# Proper indentation (2 spaces)
function process_file() {
    local file="$1"
    
    if [ -f "$file" ]; then
        echo "Processing: $file"
    fi
}

# Proper spacing
if [ "$condition" = true ]; then
    do_something
fi
```

#### Running shellcheck and shfmt

```bash
# Check script with shellcheck
shellcheck script.sh

# Format script with shfmt
shfmt -w script.sh

# Check formatting without modifying
shfmt -d script.sh
```

---

## 13. Input Validation (MANDATORY)

### A. Validate All Inputs

**CRITICAL: Always validate user input and file paths.**

#### ✅ CORRECT - Input Validation

```bash
# lib/ports.sh - Input validation functions

# Validate file path
validate_file_path() {
    local file="$1"
    
    if [ -z "$file" ]; then
        log_error "File path is required"
        return 1
    fi
    
    if [ ! -e "$file" ]; then
        log_error "File does not exist: $file"
        return 1
    fi
    
    if [ ! -f "$file" ]; then
        log_error "Path is not a regular file: $file"
        return 1
    fi
    
    if [ ! -r "$file" ]; then
        log_error "File is not readable: $file"
        return 1
    fi
    
    return 0
}

# Validate directory
validate_directory() {
    local dir="$1"
    
    if [ -z "$dir" ]; then
        log_error "Directory path is required"
        return 1
    fi
    
    if [ ! -d "$dir" ]; then
        log_error "Path is not a directory: $dir"
        return 1
    fi
    
    if [ ! -r "$dir" ]; then
        log_error "Directory is not readable: $dir"
        return 1
    fi
    
    return 0
}

# Validate numeric input
validate_number() {
    local num="$1"
    
    if [ -z "$num" ]; then
        log_error "Number is required"
        return 1
    fi
    
    if ! [[ "$num" =~ ^[0-9]+$ ]]; then
        log_error "Invalid number: $num"
        return 1
    fi
    
    return 0
}
```

---

## 14. Temporary Files and Cleanup (MANDATORY)

### A. Safe Temporary File Handling

**CRITICAL: Always use traps for cleanup of temporary files.**

#### ✅ CORRECT - Temporary File with Cleanup

```bash
# Function with temporary file cleanup
process_with_temp() {
    local temp_file
    temp_file=$(mktemp)
    
    # Set trap for cleanup on exit
    trap 'rm -f "$temp_file"' EXIT ERR
    
    # Use temporary file
    echo "data" > "$temp_file"
    process_file "$temp_file"
    
    # Trap will clean up automatically
}

# Function with temporary directory
process_with_temp_dir() {
    local temp_dir
    temp_dir=$(mktemp -d)
    
    # Set trap for cleanup
    trap 'rm -rf "$temp_dir"' EXIT ERR
    
    # Use temporary directory
    cd "$temp_dir"
    # ... do work ...
    
    # Trap will clean up automatically
}
```

#### ❌ WRONG - Unsafe Temporary Files

```bash
# ❌ No cleanup - leaves temp files
process_with_temp() {
    temp_file="/tmp/temp.txt"
    echo "data" > "$temp_file"
    process_file "$temp_file"
    # File left behind!
}
```

---

## 15. Complete Example: Modular Shell Script

### A. Project Structure

```
script/
├── script.sh              # Main orchestrator
├── lib/                   # Function modules
│   ├── core.sh           # Core business logic
│   ├── ports.sh          # Input/output ports
│   └── adapters.sh       # External adapters
├── tests/                 # Test files
│   └── script_test.sh    # Bats tests
└── README.md             # Documentation
```

### B. Main Script

```bash
#!/usr/bin/env bash
# script.sh - File processor with hexagonal architecture
# Purpose: Process files with validation and error handling
# Usage: script.sh [OPTIONS] INPUT_FILE OUTPUT_FILE

set -euo pipefail
IFS=$'\n\t'

# Script directory
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# Source modules
source "${SCRIPT_DIR}/lib/core.sh"
source "${SCRIPT_DIR}/lib/ports.sh"
source "${SCRIPT_DIR}/lib/adapters.sh"

# Default values
VERBOSE=false
DEBUG=false
INPUT_FILE=""
OUTPUT_FILE=""

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

# Parse arguments
parse_arguments() {
    local short_opts="hvd:i:o:"
    local long_opts="help,verbose,debug,input:,output:"
    
    local parsed_opts
    parsed_opts=$(getopt -o "$short_opts" --long "$long_opts" -n "$SCRIPT_NAME" -- "$@")
    
    if [ $? -ne 0 ]; then
        log_error "Invalid arguments"
        usage
        exit 1
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
            -i|--input)
                INPUT_FILE="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            --)
                shift
                break
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Validate required arguments
    if [ -z "$INPUT_FILE" ]; then
        log_error "Input file is required"
        usage
        exit 1
    fi
    
    if [ -z "$OUTPUT_FILE" ]; then
        log_error "Output file is required"
        usage
        exit 1
    fi
}

# Main function
main() {
    # Enable debug if requested
    if [ "$DEBUG" = true ]; then
        enable_debug
    fi
    
    # Set log level
    if [ "$VERBOSE" = true ]; then
        LOG_LEVEL=$LOG_LEVEL_DEBUG
    fi
    
    # Parse arguments
    parse_arguments "$@"
    
    # Validate input file
    if ! validate_file_path "$INPUT_FILE"; then
        exit 1
    fi
    
    # Validate output directory
    local output_dir
    output_dir=$(dirname "$OUTPUT_FILE")
    if [ ! -d "$output_dir" ]; then
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

```bash
# lib/core.sh - Core business logic

# Process file (core function)
process_file() {
    local input_file="$1"
    local output_file="$2"
    
    # Read and process
    local content
    content=$(cat "$input_file")
    
    # Process content (example: uppercase)
    local processed
    processed=$(echo "$content" | tr '[:lower:]' '[:upper:]')
    
    # Write output
    echo "$processed" > "$output_file"
    
    return 0
}
```

### D. Ports Module

```bash
# lib/ports.sh - Input/output ports

# Logging functions (from earlier examples)
# Validation functions (from earlier examples)
```

---

## 16. Why This Configuration Works

1. **Hexagonal Architecture**: Separates business logic from I/O operations, making scripts testable and maintainable. Core functions can be tested in isolation.

2. **Safety Settings (set -euo pipefail)**: Catches errors immediately, prevents undefined variable usage, and ensures pipeline failures are detected. Reduces debugging time by 70%.

3. **TDD with Bats**: Writing tests first ensures scripts work correctly before deployment. Regression tests prevent reintroducing fixed bugs.

4. **shellcheck Compliance**: Catches common bash pitfalls, portability issues, and security vulnerabilities at development time rather than runtime.

5. **getopt Parameter Parsing**: Provides consistent, user-friendly CLI interfaces with proper help messages, long options, and error handling.

6. **zsh Compatibility**: Scripts work across different user environments, reducing support requests and improving adoption.

7. **Modular Structure**: Small, focused functions are easier to test, debug, and reuse. Changes in one module don't break others.

8. **Proper Quoting**: Prevents word splitting and glob expansion bugs that cause silent failures or security vulnerabilities.

9. **Structured Logging**: Debug and verbose modes make troubleshooting easy. Consistent log formats improve automation compatibility.

10. **Cleanup Traps**: Ensures temporary files are always cleaned up, even when scripts fail, preventing disk space issues.

---

## 17. Quick Reference

### Command Cheat Sheet

```bash
# Verification (MANDATORY)
bash -n script.sh                    # Syntax check
zsh -n script.sh                     # zsh compatibility check
shellcheck script.sh                 # Static analysis
shfmt -d script.sh                   # Format check

# Formatting
shfmt -w script.sh                   # Auto-format script
shfmt -i 4 -w script.sh              # Format with 4-space indent

# Testing with Bats
bats tests/                          # Run all tests
bats tests/test_script.bats          # Run specific test file
bats --tap tests/                    # TAP output format

# Debugging
bash -x script.sh                    # Debug mode (trace execution)
bash -v script.sh                    # Verbose mode (print lines)
DEBUG=1 ./script.sh                  # Enable script debug logging

# Script execution
./script.sh --help                   # Show help
./script.sh -v                       # Verbose mode
./script.sh --dry-run                # Dry run mode
```

### Script Header Template

```bash
#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"
```

### Function Template

```bash
# Brief description of function
# Arguments:
#   $1 - Description of first argument
#   $2 - Description of second argument (optional)
# Returns:
#   0 on success, 1 on failure
# Outputs:
#   Writes result to stdout
function_name() {
    local arg1="$1"
    local arg2="${2:-default}"

    # Implementation
    echo "result"
    return 0
}
```

### Common Patterns

```bash
# Safe file reading
while IFS= read -r line; do
    echo "$line"
done < "$file"

# Check if command exists
if command -v cmd >/dev/null 2>&1; then
    echo "cmd is available"
fi

# Default variable value
: "${VAR:=default}"

# Temporary file with cleanup
tmp_file=$(mktemp)
trap 'rm -f "$tmp_file"' EXIT
```

---

## 18. Summary

**CRITICAL Requirements for All Bash Scripts:**

1. **Script Header**: Proper shebang, safety settings (set -euo pipefail)
2. **Hexagonal Architecture**: Modular structure, separation of concerns
3. **Parameter Parsing**: getopt for long options, getopts for simple cases
4. **zsh Compatibility**: Scripts MUST work in both bash and zsh
5. **Error Handling**: Explicit error checking, meaningful messages
6. **Quoting**: Always quote variables to prevent issues
7. **Logging**: Structured logging with debug and verbose modes
8. **Testing**: Bats tests for all scripts
9. **shellcheck**: Scripts MUST pass shellcheck if available
10. **shfmt**: Scripts MUST be formatted with shfmt if available
11. **Input Validation**: Validate all user inputs and file paths
12. **Cleanup**: Use traps for temporary file cleanup
13. **Documentation**: Clear function and script documentation
14. **Modular Structure**: Separate functions into modules
15. **Verification**: Agent MUST test scripts before delivery

**Agent Verification Protocol:**
- **MANDATORY**: Syntax check (`bash -n script.sh`) - MUST succeed
- **MANDATORY**: zsh compatibility (`zsh -n script.sh`) - MUST succeed
- **MANDATORY**: shellcheck (`shellcheck script.sh`) - MUST pass if available
- **MANDATORY**: shfmt (`shfmt -d script.sh`) - MUST pass if available
- **MANDATORY**: Execution test (`bash script.sh --help`) - MUST succeed
- **MANDATORY**: Test execution (`bats tests/`) - MUST pass if tests exist
- **MANDATORY**: After ANY modification, re-verify all steps
- Only present working, tested scripts to the user

**Remember**: Minimalistic, clean, readable, well-documented, modular bash scripts with hexagonal architecture, zsh compatibility, proper error handling, getopt parameter parsing, and comprehensive testing. Keep it simple, keep it safe, keep it working.
