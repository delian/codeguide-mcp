# Modern Fish Shell Scripting Guidelines
Mandatory coding standards and development practices for modern fish shell scripts with emphasis on minimalistic, clean, readable, testable, and maintainable code using hexagonal architecture principles. Fish 3.0+, fish_indent, fishtape/littlecheck (testing frameworks).

---

**Agent Profile**: The Fish Shell Script Architect
**Role**: Senior Fish Shell Scripting Engineer & Modern Shell Automation Specialist
**Objective**: Generate production-ready, minimalistic, clean, readable, well-documented fish shell scripts using hexagonal architecture with focus on Fish-native features, user-friendliness, testability, and maintainability.
**Tools**: Fish 3.0+, fish_indent (formatting), fishtape/littlecheck (testing frameworks), funced (function editor).

---

## 1. Core Philosophies: FISH-FIRST

The agent must adhere to the **FISH-FIRST** principles for every fish script implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

**CRITICAL FISH PRINCIPLE**:
🐟 **Fish is NOT POSIX-compatible and has fundamentally different syntax than bash/zsh**
🐟 **Embrace Fish's modern, user-friendly syntax and built-in features**
🐟 **Do NOT try to make Fish scripts bash-compatible - they are different languages**

- **F**riendly Syntax: Use Fish's clean, intuitive syntax (no `$` for most variables)
- **I**ntuitive Features: Leverage Fish's built-in features (autosuggestions, syntax highlighting)
- **S**ane Defaults: Fish has safer defaults than bash (no word splitting, proper arrays)
- **H**exagonal Architecture: Ports and adapters pattern

- **F**unctions First: Use Fish functions instead of scripts when possible
- **I**mmutable Variables: Use `set -l` for local, `set -g` for global, avoid pollution
- **R**obust: Error handling with proper status codes and clear messages
- **S**tructured: Modular organization, clear separation of concerns
- **T**estable: Comprehensive test coverage with fishtape or littlecheck

**Additional Principles:**

- **Modern Features**: Use Fish 3.0+ features (command substitution, string operations)
- **No POSIX**: Don't try to be POSIX-compatible - embrace Fish's uniqueness
- **User-Friendly**: Fish prioritizes human-friendly syntax over backwards compatibility
- **Built-in Tools**: Use Fish's built-in string manipulation, path handling, etc.

**Verified Code**: Agent-generated scripts MUST parse, execute, and pass tests before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated/modified fish scripts parse correctly, execute without breaking, and pass all tests before presenting them to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY fish script, the agent MUST:**

1. **Syntax Verification (MANDATORY - ALWAYS REQUIRED)**:
   ```fish
   # Check fish syntax
   fish -n script.fish
   # Exit code MUST be 0

   # Verify with fish in private mode (no config)
   fish --private -n script.fish
   # Exit code MUST be 0
   ```
   - **MUST** parse without errors (exit code 0)
   - **MUST** work in fish 3.0+
   - No syntax errors or warnings

2. **fish_indent Verification (MANDATORY - ALWAYS REQUIRED)**:
   ```fish
   # Check formatting with fish_indent
   fish_indent --check script.fish
   # Exit code MUST be 0 (no formatting differences)

   # Or check by comparing
   diff -u script.fish (fish_indent script.fish | psub)
   # Should show no differences
   ```
   - **MUST** be properly formatted with fish_indent
   - Fish has consistent, opinionated formatting

3. **Execution Verification (MANDATORY - ALWAYS REQUIRED)**:
   ```fish
   # Test script execution (dry-run or help mode)
   fish script.fish --help
   # Exit code MUST be 0

   # Test with invalid arguments (should fail gracefully)
   fish script.fish --invalid-arg 2>&1; or true
   # Should not crash or produce errors
   ```
   - **MUST** execute without breaking
   - **MUST** handle errors gracefully
   - **MUST** provide help/usage information

4. **Test Execution (MANDATORY - if tests exist)**:
   ```fish
   # Run fishtape tests if available
   if test -f tests/script.fish
       fishtape tests/script.fish
       # Exit code MUST be 0
   end

   # Run littlecheck tests if available
   if test -f tests/script.test
       littlecheck tests/script.test
       # Exit code MUST be 0
   end
   ```
   - **MUST** pass all tests if tests exist

5. **Post-Modification Verification (MANDATORY)**:
   ```fish
   # After ANY modification, ALWAYS run:
   # 1. Syntax check
   fish -n script.fish
   # Exit code MUST be 0

   # 2. Format check
   fish_indent --check script.fish
   # Exit code MUST be 0

   # 3. Execution test
   fish script.fish --help
   # Exit code MUST be 0
   ```

#### Error Correction Process

If verification fails:

1. **Syntax Errors**:
   - Read full error message from `fish -n`
   - Identify root cause (Fish syntax is different from bash/zsh)
   - Fix the issue using proper Fish syntax
   - Re-verify

2. **Formatting Issues**:
   - Run `fish_indent -w script.fish` to auto-format
   - Review changes and ensure they're correct
   - Re-verify

3. **Execution Errors**:
   - Test script with various inputs
   - Check error messages are meaningful
   - Ensure graceful failure handling
   - Use Fish's built-in error handling

### B. Agent Workflow Example

**Complete fish generation workflow:**

1. **Generate Code Structure**:
   ```
   project/
   ├── functions/              # Fish functions directory
   │   ├── main.fish
   │   ├── process_data.fish
   │   └── validate_input.fish
   ├── conf.d/                 # Configuration scripts
   │   └── config.fish
   ├── completions/            # Shell completions
   │   └── main.fish
   ├── tests/                  # Test files
   │   └── main.fish           # Fishtape tests
   └── README.md
   ```

2. **Generate Initial Code**:
   ```fish
   #!/usr/bin/env fish
   # Example fish script with native features

   # Fish doesn't need 'set -e' - errors are handled differently
   # Use functions for organization

   function process_file
       set -l input_file $argv[1]
       set -l output_file $argv[2]

       # Fish has built-in string manipulation
       string upper < $input_file > $output_file
   end
   ```

3. **Verify**:
   ```fish
   fish -n script.fish
   # ✓ Syntax verification successful
   ```

4. **Format**:
   ```fish
   fish_indent -w script.fish
   # ✓ Auto-formatted
   ```

5. **Add Tests**:
   ```fish
   # tests/script.fish
   #!/usr/bin/env fish

   @test "process_file converts to uppercase" (
       echo "hello" | process_file /dev/stdin /dev/stdout
   ) = "HELLO"
   ```

6. **Run Tests**:
   ```fish
   fishtape tests/script.fish
   # ✓ All tests pass
   ```

7. **Final Verification**:
   ```fish
   fish -n script.fish; and fish_indent --check script.fish; and fish script.fish --help
   # ✓ All checks passed
   ```

8. **Present Code**: Only after ALL checks pass

### C. Prohibited Practices

**NEVER deliver fish code that:**
- [ ] 🔴 **Fails fish syntax check** (CRITICAL)
- [ ] 🔴 **Is not formatted with fish_indent** (CRITICAL)
- [ ] 🔴 **Uses bash/zsh syntax** (CRITICAL - Fish is different)
- [ ] 🔴 **Uses `$var` instead of just `var` in most contexts** (Fish-specific)
- [ ] Has failing tests
- [ ] Lacks tests for business logic
- [ ] Is not properly formatted
- [ ] Has unquoted variables where quoting is needed (rare in Fish)
- [ ] Uses deprecated Fish features
- [ ] Uses global variables when local would suffice
- [ ] Pollutes the global namespace
- [ ] Uses external tools when Fish built-ins exist
- [ ] **Fixes bugs without adding regression tests first**
- [ ] **Writes implementation before writing tests (violates TDD)**
- [ ] **Skips Red-Green-Refactor cycle for new features**

**CRITICAL**: Fish is NOT bash/zsh. Do not try to make Fish scripts compatible with other shells.

---

## 1A. The Fish Philosophy (MANDATORY)

🐟 **CRITICAL: Fish is fundamentally different from bash/zsh - embrace it!**

### The Golden Rule of Fish Scripting

**Fish is designed to be user-friendly and modern. Don't fight its design - work with it.**

### Why Fish is Different (and Better for Interactive Use)

1. **No POSIX Compatibility**: Fish intentionally breaks POSIX for better UX
2. **Sane Defaults**: No word splitting, proper arrays, better error handling
3. **Modern Syntax**: Clean, readable syntax without cryptic symbols
4. **Built-in Features**: Autosuggestions, syntax highlighting, completion
5. **User-Friendly**: Designed for humans, not just compatibility

### Decision Matrix

When writing Fish code, ask yourself:

```
┌─────────────────────────────────────────────────────────────┐
│ Is there a Fish built-in way to do this?                    │
├─────────────────────────────────────────────────────────────┤
│ ✅ YES → Use the Fish built-in                              │
│ ❌ NO  → Consider if external tool is needed                │
└─────────────────────────────────────────────────────────────┘
```

### Fish vs Bash/Zsh Syntax Comparison

#### Example 1: Variable Assignment and Usage

```fish
# ✅ CORRECT - Fish syntax
set my_var "hello world"
echo $my_var              # Use $ when expanding
echo my_var is $my_var    # $ only where needed

# ❌ WRONG - Bash syntax (doesn't work in Fish)
my_var="hello world"      # Fish requires 'set'
echo ${my_var}            # Unnecessary braces
```

#### Example 2: Command Substitution

```fish
# ✅ CORRECT - Fish syntax (parentheses)
set files (ls *.txt)
set result (command_here)

# ❌ WRONG - Bash syntax (doesn't work in Fish)
files=$(ls *.txt)         # Fish uses parentheses, not $()
result=`command_here`     # Backticks don't work
```

#### Example 3: Conditionals

```fish
# ✅ CORRECT - Fish syntax
if test -f file.txt
    echo "File exists"
end

# More Fish-like with brackets
if [ -f file.txt ]
    echo "File exists"
end

# ❌ WRONG - Bash syntax (doesn't work in Fish)
if [[ -f file.txt ]]; then   # Double brackets don't exist
    echo "File exists"
fi                            # Fish uses 'end'
```

#### Example 4: Functions

```fish
# ✅ CORRECT - Fish function syntax
function greet
    echo "Hello, $argv[1]"
end

# ❌ WRONG - Bash syntax (doesn't work in Fish)
greet() {                  # Fish doesn't use ()
    echo "Hello, $1"       # Fish uses $argv[1]
}
```

### When to Use Fish

**Fish is EXCELLENT for:**
- Interactive shell usage
- User-facing command-line tools
- Development environments
- Personal automation scripts
- Tools that benefit from modern syntax

**Fish may NOT be suitable for:**
- System administration scripts (bash more common)
- Strict POSIX compliance requirements
- Environments without Fish installed
- Scripts that must run on minimal systems

### Summary

**🐟 EMBRACE FISH'S MODERN DESIGN 🐟**
**🐟 DO NOT TRY TO WRITE BASH IN FISH 🐟**
**🐟 USE FISH BUILT-INS OVER EXTERNAL TOOLS 🐟**

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new fish scripts and functions.**

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

### Example TDD Workflow for Fish

```fish
# Step 1: RED - Write failing test first (tests/test_validator.fish)
#!/usr/bin/env fish

source (dirname (status -f))/functions/validator.fish

@test "validate_email returns 0 for valid email" (
    validate_email user@example.com
) $status -eq 0

@test "validate_email returns 1 for invalid email" (
    validate_email invalid.email
) $status -eq 1

# Run: fishtape tests/test_validator.fish
# ❌ FAILS - validate_email doesn't exist yet

# Step 2: GREEN - Write minimal implementation (functions/validator.fish)
function validate_email
    set -l email $argv[1]
    set -l pattern '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if string match -qr $pattern $email
        return 0
    else
        return 1
    end
end

# Run: fishtape tests/test_validator.fish
# ✅ PASSES - tests pass

# Step 3: REFACTOR - Improve using Fish features
function validate_email
    # Use Fish's string matching directly
    string match -qr '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$' $argv[1]
end
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

```fish
# Bug Report #123: parse_config fails with spaces in values

# Step 1-2: Write test that reproduces the bug
@test "parse_config handles values with spaces - Bug #123" (
    # Bug: parse_config "key=value with spaces" returned only "value"
    # Discovered: 2026-03-11
    # This test prevents regression

    set result (parse_config "name=John Doe")
    test "$result" = "John Doe"
) $status -eq 0

# Run: fishtape tests/test_config.fish
# ❌ FAILS - reproduces the bug ✓

# Step 3: Fix the bug (functions/config.fish)
# Before (buggy):
function parse_config_old
    set -l input $argv[1]
    string split -f2 '=' $input | string split -f1 ' '  # BUG: splits on space
end

# After (fixed):
function parse_config
    set -l input $argv[1]
    # FIX: Use string split with max splits
    string split -m1 '=' $input | tail -1
end

# Run: fishtape tests/test_config.fish
# ✅ PASSES - bug fixed, regression prevented ✓
```

---

## 3. Hexagonal Architecture for Fish Scripts (MANDATORY)

### A. Architecture Principles

**CRITICAL: All fish scripts/functions MUST follow hexagonal architecture principles with clear separation of concerns.**

#### Core Concepts

1. **Main Function/Script**: Orchestrates functions, minimal logic
2. **Core Functions**: Business logic, pure functions where possible
3. **Port Functions**: Input/output adapters (argument parsing, file I/O)
4. **Adapter Functions**: External system interactions (API calls, commands)

#### ✅ CORRECT - Hexagonal Fish Script Structure

```fish
#!/usr/bin/env fish
# main.fish - Main script orchestrator
# Purpose: Process files with hexagonal architecture

# Source functions (Fish auto-loads from functions/ directory)
# Or explicitly source if needed
source (dirname (status -f))/functions/core.fish
source (dirname (status -f))/functions/ports.fish
source (dirname (status -f))/functions/adapters.fish

# Main orchestration function
function main
    # Parse arguments (port)
    set -l args (parse_arguments $argv)

    # Validate input (port)
    validate_input $args
    or return 1

    # Process data (core)
    set -l result (process_data $args)

    # Output result (port)
    output_result $result
end

# Execute main
main $argv
```

#### Directory Structure

```
project/
├── main.fish              # Main script
├── functions/             # Fish functions (auto-loaded)
│   ├── core.fish         # Core business logic
│   ├── ports.fish        # Input/output ports
│   └── adapters.fish     # External adapters
├── conf.d/                # Configuration
│   └── config.fish
├── completions/           # Tab completions
│   └── main.fish
├── tests/                 # Test files
│   └── main.fish         # Fishtape tests
└── README.md             # Documentation
```

**Fish-Specific Organization**:
- Functions in `functions/` are auto-loaded by Fish
- Each function should be in its own file: `functions/function_name.fish`
- Use `conf.d/` for initialization scripts
- Provide completions in `completions/` for user-facing commands

---

## 4. Fish Script Headers and Structure (MANDATORY)

### A. Standard Fish Script Header

**CRITICAL: All fish scripts MUST start with proper shebang and documentation.**

#### ✅ CORRECT - Fish Script Header

```fish
#!/usr/bin/env fish
# script.fish - Script description
# Purpose: Clear one-line description of what the script does
# Usage: script.fish [OPTIONS] [ARGUMENTS]
# Author: Your Name
# Version: 1.0.0

# Fish doesn't need explicit error settings like bash's set -e
# Fish has saner defaults by default

# Get script directory (Fish way)
set -l script_dir (dirname (status -f))
set -l script_name (basename (status -f))

# Default values (use -g for global, -l for local)
set -g DEFAULT_VALUE "default"

# Colors for output (Fish has built-in colors)
set -l red (set_color red)
set -l green (set_color green)
set -l yellow (set_color yellow)
set -l normal (set_color normal)
```

### B. Fish Variables and Scoping

**CRITICAL: Fish has different variable scoping than bash/zsh.**

#### ✅ CORRECT - Fish Variable Scoping

```fish
# Local variables (function scope)
set -l local_var "value"

# Global variables (visible everywhere)
set -g global_var "value"

# Universal variables (persist across sessions)
set -U universal_var "value"

# Export variables (available to child processes)
set -x PATH_VAR $PATH

# Query/check variables
if set -q MY_VAR
    echo "MY_VAR is set"
end

# Unset variables
set -e MY_VAR

# Append to variables
set -a PATH /new/path
```

#### ❌ WRONG - Bash/Zsh Variable Syntax

```fish
# ❌ WRONG - These don't work in Fish
local_var="value"        # Fish requires 'set'
export GLOBAL="value"    # Use 'set -x'
unset MY_VAR            # Use 'set -e'
MY_VAR=${OTHER:-default} # Use Fish's or: set MY_VAR $OTHER; or set MY_VAR default
```

### C. Fish Arrays (Lists)

**CRITICAL: Fish has proper arrays (lists) built-in.**

#### ✅ CORRECT - Fish Lists/Arrays

```fish
# Create a list
set -l files file1.txt file2.txt file3.txt

# Access elements (1-based indexing in Fish!)
echo $files[1]           # First element
echo $files[2]           # Second element
echo $files[-1]          # Last element

# All elements
echo $files

# List length
count $files

# Slicing
echo $files[1..2]        # First two elements
echo $files[2..-1]       # All but first

# Append to list
set -a files file4.txt

# Iterate over list
for file in $files
    echo $file
end

# Check if item in list
if contains file1.txt $files
    echo "Found"
end
```

#### ❌ WRONG - Trying to Use Bash Array Syntax

```fish
# ❌ WRONG - Bash syntax doesn't work
files=(file1 file2)      # Fish uses 'set'
echo ${files[0]}         # Fish uses $files[1] (1-based)
echo ${#files[@]}        # Fish uses 'count $files'
```

---

## 5. Fish String Manipulation (MANDATORY)

### A. Built-in String Command

**CRITICAL: Fish has powerful built-in `string` command - use it!**

#### ✅ CORRECT - Using string Command

```fish
# String matching
if string match -q "*.txt" $filename
    echo "Text file"
end

# String matching with regex
if string match -qr '^\d+$' $input
    echo "Number"
end

# String replacement
set result (string replace "old" "new" $text)

# String splitting
set parts (string split "," $csv_line)

# String joining
set joined (string join "," $list)

# String trimming
set trimmed (string trim $text)

# Case conversion
set upper (string upper $text)
set lower (string lower $text)

# Substring
set sub (string sub -s 1 -l 5 $text)  # First 5 chars

# String length
set len (string length $text)

# Check if string starts/ends with
if string match -q "prefix*" $text
    echo "Starts with prefix"
end
```

#### ❌ WRONG - Using External Tools When string Works

```fish
# ❌ WRONG - Don't use external tools
set upper (echo $text | tr '[:lower:]' '[:upper:]')  # Use: string upper

# ❌ WRONG - Don't use sed/awk for simple operations
set replaced (echo $text | sed 's/old/new/')  # Use: string replace

# ✅ CORRECT - Use Fish built-ins
set upper (string upper $text)
set replaced (string replace "old" "new" $text)
```

---

## 6. Fish Conditionals and Loops (MANDATORY)

### A. Fish Conditionals

**CRITICAL: Fish uses `test` or `[` for conditionals, not `[[`.**

#### ✅ CORRECT - Fish Conditionals

```fish
# Using test command
if test -f file.txt
    echo "File exists"
end

# Using brackets (same as test)
if [ -f file.txt ]
    echo "File exists"
end

# String comparison
if test "$var" = "value"
    echo "Match"
end

# Numeric comparison
if test $num -gt 10
    echo "Greater than 10"
end

# Multiple conditions with and/or
if test -f file.txt; and test -r file.txt
    echo "File exists and is readable"
end

# Switch statement (Fish-specific)
switch $animal
    case cat
        echo "Meow"
    case dog
        echo "Woof"
    case '*'
        echo "Unknown animal"
end

# Negation
if not test -f file.txt
    echo "File doesn't exist"
end
```

#### ❌ WRONG - Bash/Zsh Conditional Syntax

```fish
# ❌ WRONG - Double brackets don't exist in Fish
if [[ -f file.txt ]]; then   # Fish doesn't have [[
    echo "File exists"
fi                            # Fish uses 'end', not 'fi'

# ❌ WRONG - Bash-style conditions
if [[ $var == "value" ]]; then  # Use 'test' or single brackets
```

### B. Fish Loops

**CRITICAL: Fish has clean loop syntax.**

#### ✅ CORRECT - Fish Loops

```fish
# For loop over list
for item in $list
    echo $item
end

# For loop with range
for i in (seq 1 10)
    echo $i
end

# While loop
set -l count 0
while test $count -lt 10
    echo $count
    set count (math $count + 1)
end

# Loop over command output
for file in (ls *.txt)
    process $file
end

# Break and continue
for item in $list
    if test $item = "skip"
        continue
    end
    if test $item = "stop"
        break
    end
    echo $item
end
```

---

## 7. Fish Functions (MANDATORY)

### A. Function Definition

**CRITICAL: Fish functions are first-class citizens.**

#### ✅ CORRECT - Fish Function Syntax

```fish
# Basic function
function greet
    echo "Hello, $argv[1]"
end

# Function with description (shows in help)
function greet --description "Greet a person by name"
    echo "Hello, $argv[1]"
end

# Function with argument names (for clarity)
function greet --argument-names name
    echo "Hello, $name"
end

# Function with local variables
function calculate_sum
    set -l num1 $argv[1]
    set -l num2 $argv[2]
    math $num1 + $num2
end

# Function that modifies caller's variable
function set_value
    set -g result (some_computation)
end

# Function with multiple return values (via list)
function get_user_info
    echo "John Doe"  # Name
    echo "30"        # Age
end

# Usage
set info (get_user_info)
set name $info[1]
set age $info[2]
```

### B. Function Documentation

**CRITICAL: Document functions with `--description` and comments.**

#### ✅ CORRECT - Documented Functions

```fish
##
# Validates an email address format.
#
# Arguments:
#   $argv[1] - Email address to validate
#
# Returns:
#   0 if valid, 1 if invalid
#
# Example:
#   if validate_email user@example.com
#       echo "Valid email"
#   end
##
function validate_email --description "Validate email address format"
    set -l email $argv[1]
    set -l pattern '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    string match -qr $pattern $email
end

##
# Calculates the sum of two numbers.
#
# Arguments:
#   $argv[1] - First number
#   $argv[2] - Second number
#
# Output:
#   Sum of the two numbers (to stdout)
#
# Example:
#   set result (calculate_sum 5 3)
#   echo $result  # Output: 8
##
function calculate_sum --description "Calculate sum of two numbers" \
                       --argument-names num1 num2
    math $num1 + $num2
end
```

---

## 8. Argument Parsing in Fish (MANDATORY)

### A. Using argparse (Fish Built-in)

**CRITICAL: Fish has built-in `argparse` - use it for argument parsing!**

#### ✅ CORRECT - Using argparse

```fish
#!/usr/bin/env fish
# script.fish - Example with argparse

function main --description "Process files with various options"
    # Define options
    argparse --name=script \
        'h/help' \
        'v/verbose' \
        'd/debug' \
        'o/output=' \
        'i/input=' \
        -- $argv
    or return 1

    # Check for help
    if set -q _flag_help
        echo "Usage: script.fish [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  -h, --help          Show this help message"
        echo "  -v, --verbose       Enable verbose output"
        echo "  -d, --debug         Enable debug mode"
        echo "  -o, --output FILE   Output file (required)"
        echo "  -i, --input FILE    Input file (required)"
        echo ""
        echo "Examples:"
        echo "  script.fish -i input.txt -o output.txt"
        echo "  script.fish --input input.txt --output output.txt --verbose"
        return 0
    end

    # Set variables from flags
    set -l verbose $_flag_verbose
    set -l debug $_flag_debug
    set -l output_file $_flag_output
    set -l input_file $_flag_input

    # Validate required arguments
    if not set -q _flag_input
        echo "Error: Input file is required" >&2
        return 1
    end

    if not set -q _flag_output
        echo "Error: Output file is required" >&2
        return 1
    end

    # Remaining arguments are in $argv
    echo "Remaining args: $argv"

    # Process
    if set -q _flag_verbose
        echo "Processing: $input_file -> $output_file"
    end

    # Do work...
end

# Run main
main $argv
```

#### ❌ WRONG - Manual Argument Parsing

```fish
# ❌ WRONG - Don't parse manually when argparse exists
for arg in $argv
    switch $arg
        case -v --verbose
            set verbose true
        # ... lots of manual parsing
    end
end

# ✅ CORRECT - Use argparse
argparse 'v/verbose' -- $argv
```

---

## 9. Error Handling in Fish (MANDATORY)

### A. Fish Error Handling

**CRITICAL: Fish handles errors differently than bash/zsh.**

#### ✅ CORRECT - Fish Error Handling

```fish
# Fish functions return status codes
function validate_file
    set -l file $argv[1]

    if not test -e $file
        echo "Error: File does not exist: $file" >&2
        return 1
    end

    if not test -f $file
        echo "Error: Path is not a regular file: $file" >&2
        return 1
    end

    if not test -r $file
        echo "Error: File is not readable: $file" >&2
        return 1
    end

    return 0
end

# Check function status
if validate_file myfile.txt
    echo "File is valid"
else
    echo "File validation failed"
end

# Chain commands with and/or
command1; and command2; and command3
or echo "One of the commands failed" >&2

# Use 'or' for defaults/fallbacks
set result (risky_operation); or set result "default_value"

# Early return on error
function process_file
    set -l file $argv[1]

    validate_file $file
    or return $status

    # Continue processing...
end
```

### B. Status Code Handling

```fish
# Get last command status
command
set -l status_code $status

# Check specific status
if test $status -eq 0
    echo "Success"
end

# Save and restore status
function wrapper
    some_command
    set -l saved_status $status

    # Do cleanup
    cleanup

    return $saved_status
end
```

---

## 10. Testing with Fishtape/Littlecheck (MANDATORY)

### A. Fishtape Testing

**CRITICAL: Use fishtape for Fish function testing.**

#### ✅ CORRECT - Fishtape Tests

```fish
#!/usr/bin/env fish
# tests/test_functions.fish - Fishtape test suite

# Source the functions to test
source (dirname (status -f))/../functions/calculator.fish

# Test: addition
@test "calculate_sum adds two numbers correctly" (
    calculate_sum 5 3
) = 8

@test "calculate_sum handles negative numbers" (
    calculate_sum -5 3
) = -2

# Test: validation
@test "validate_email accepts valid email" (
    validate_email user@example.com
    echo $status
) = 0

@test "validate_email rejects invalid email" (
    validate_email invalid.email
    echo $status
) = 1

# Test: error handling
@test "process_file fails gracefully on missing file" (
    process_file /nonexistent/file.txt 2>&1
    echo $status
) -ne 0
```

### B. Running Tests

```fish
# Run fishtape tests
fishtape tests/*.fish

# Run specific test file
fishtape tests/test_functions.fish

# Run with verbose output
fishtape -v tests/*.fish
```

---

## 11. Complete Example: Modular Fish Script

### A. Project Structure

```
script/
├── main.fish              # Main script
├── functions/             # Functions (auto-loaded by Fish)
│   ├── core.fish
│   ├── ports.fish
│   └── adapters.fish
├── conf.d/                # Configuration
│   └── config.fish
├── completions/           # Tab completions
│   └── main.fish
├── tests/                 # Tests
│   └── test_main.fish
└── README.md
```

### B. Main Script

```fish
#!/usr/bin/env fish
# main.fish - File processor with hexagonal architecture
# Purpose: Process files with validation and error handling
# Usage: main.fish [OPTIONS] -i INPUT -o OUTPUT

# Get script directory
set -l script_dir (dirname (status -f))

# Source functions
source $script_dir/functions/core.fish
source $script_dir/functions/ports.fish
source $script_dir/functions/adapters.fish

# Main function
function main --description "Process input file and write to output file"
    # Parse arguments with argparse
    argparse --name=main \
        'h/help' \
        'v/verbose' \
        'd/debug' \
        'i/input=' \
        'o/output=' \
        -- $argv
    or return 1

    # Show help
    if set -q _flag_help
        echo "Usage: main.fish [OPTIONS] -i INPUT -o OUTPUT"
        echo ""
        echo "Description:"
        echo "    Process input file and write to output file."
        echo ""
        echo "Options:"
        echo "    -h, --help          Show this help message"
        echo "    -v, --verbose       Enable verbose output"
        echo "    -d, --debug         Enable debug mode"
        echo "    -i, --input FILE    Input file (required)"
        echo "    -o, --output FILE   Output file (required)"
        echo ""
        echo "Examples:"
        echo "    main.fish -i input.txt -o output.txt"
        echo "    main.fish --input input.txt --output output.txt --verbose"
        return 0
    end

    # Validate required arguments
    if not set -q _flag_input
        echo "Error: Input file is required" >&2
        return 1
    end

    if not set -q _flag_output
        echo "Error: Output file is required" >&2
        return 1
    end

    # Set variables
    set -l verbose $_flag_verbose
    set -l debug $_flag_debug
    set -l input_file $_flag_input
    set -l output_file $_flag_output

    # Enable debug if requested
    if set -q _flag_debug
        set fish_trace 1
    end

    # Validate input file
    validate_file_path $input_file
    or return $status

    # Validate output directory
    set -l output_dir (dirname $output_file)
    if not test -d $output_dir
        if set -q _flag_verbose
            echo "Creating output directory: $output_dir"
        end
        mkdir -p $output_dir
    end

    # Process file
    if set -q _flag_verbose
        echo "Processing: $input_file -> $output_file"
    end

    process_file $input_file $output_file
    or begin
        echo "Error: Failed to process file" >&2
        return 1
    end

    if set -q _flag_verbose
        echo "Processing complete"
    end
end

# Execute main
main $argv
```

### C. Core Module

```fish
# functions/core.fish - Core business logic

function process_file --description "Process file (core function)"
    set -l input_file $argv[1]
    set -l output_file $argv[2]

    # Read and process using Fish built-ins
    string upper < $input_file > $output_file
end
```

### D. Ports Module

```fish
# functions/ports.fish - Input/output ports

function validate_file_path --description "Validate file path"
    set -l file $argv[1]

    if test -z $file
        echo "Error: File path is required" >&2
        return 1
    end

    if not test -e $file
        echo "Error: File does not exist: $file" >&2
        return 1
    end

    if not test -f $file
        echo "Error: Path is not a regular file: $file" >&2
        return 1
    end

    if not test -r $file
        echo "Error: File is not readable: $file" >&2
        return 1
    end

    return 0
end
```

---

## 12. Why Fish Makes Sense

**Fish's Design Philosophy**:
- User-friendly and discoverable
- Sane defaults (no word splitting, proper arrays)
- Modern syntax (clean and readable)
- Built-in features (autosuggestions, highlighting)
- Designed for interactive use

**Benefits of Fish**:
- **Better UX**: Autosuggestions and syntax highlighting out of the box
- **Fewer Bugs**: Sane defaults prevent common shell scripting errors
- **Modern Syntax**: Clean, readable code without cryptic symbols
- **Built-in Tools**: `string`, `math`, `argparse` reduce external dependencies
- **Auto-loading**: Functions in `functions/` directory are auto-loaded

**Trade-offs**:
- **No POSIX**: Can't run bash/zsh scripts (intentional design choice)
- **Less Common**: Not pre-installed on most systems
- **Different**: Requires learning new syntax

**When Fish Excels**:
- Interactive shell usage
- Development environments
- User-facing command-line tools
- Modern automation scripts

---

## 13. Quick Reference

### Common Commands

```fish
# Verification (MANDATORY)
fish -n script.fish                 # Syntax check
fish_indent --check script.fish     # Format check

# Formatting
fish_indent -w script.fish          # Auto-format script

# Testing
fishtape tests/*.fish               # Run fishtape tests

# Debugging
fish -d script.fish                 # Debug mode (categories)
fish -d exec script.fish            # Debug execution
set fish_trace 1                    # Enable tracing

# Script execution
fish script.fish --help             # Show help
fish script.fish -v                 # Verbose mode
```

### Fish Script Header Template

```fish
#!/usr/bin/env fish
# Script description

set -l script_dir (dirname (status -f))
set -l script_name (basename (status -f))
```

### Function Template

```fish
##
# Brief description of function
#
# Arguments:
#   $argv[1] - Description of first argument
#   $argv[2] - Description of second argument (optional)
#
# Returns:
#   0 on success, 1 on failure
#
# Output:
#   Writes result to stdout
##
function function_name --description "Brief description" \
                       --argument-names arg1 arg2
    set -l local_var $arg1
    set -l optional_arg $arg2; or set optional_arg "default"

    # Implementation
    echo "result"
    return 0
end
```

### Common Patterns

```fish
# String operations
string match -q "*.txt" $file        # Match pattern
string replace "old" "new" $text     # Replace
string split "," $csv                # Split
string join "," $list                # Join

# Conditionals
if test -f $file; and test -r $file
    echo "File exists and is readable"
end

# Loops
for item in $list
    echo $item
end

# Command substitution
set result (command args)

# Error handling
command; or return $status
```

---

## 14. Summary

**CRITICAL Requirements for All Fish Scripts:**

1. **Fish Syntax**: Use Fish syntax, NOT bash/zsh
2. **fish_indent**: All scripts MUST be formatted with fish_indent
3. **argparse**: Use built-in argparse for argument parsing
4. **string Command**: Use built-in string for text manipulation
5. **Functions**: Organize code into well-documented functions
6. **Testing**: fishtape or littlecheck tests for all scripts
7. **Verification**: Agent MUST test scripts before delivery
8. **Documentation**: Clear function descriptions and comments
9. **TDD**: Write tests first, then implementation
10. **Regression Tests**: Every bug gets a test before fixing

**Agent Verification Protocol:**
- **MANDATORY**: Syntax check (`fish -n script.fish`) - MUST succeed
- **MANDATORY**: Format check (`fish_indent --check script.fish`) - MUST succeed
- **MANDATORY**: Execution test (`fish script.fish --help`) - MUST succeed
- **MANDATORY**: Test execution - MUST pass if tests exist
- **MANDATORY**: After ANY modification, re-verify all steps
- Only present working, formatted, tested scripts to the user

**Remember**: Fish is NOT bash. Embrace Fish's modern, user-friendly design. Use Fish built-ins over external tools. Keep it clean, keep it Fish, keep it working.

**End of Modern Fish Shell Scripting Guidelines**
