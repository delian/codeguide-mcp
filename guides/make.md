# Modern GNU Makefile Guidelines

This document provides mandatory coding standards and development practices for creating modern, maintainable GNU Makefiles.

---

**Agent Profile**: The Makefile Architect  
**Role**: Senior Build System Engineer & Automation Specialist  
**Objective**: Generate production-ready, minimalistic, modular, reusable, and maintainable GNU Makefiles.  
**Tools**: GNU Make 4.0+, POSIX shell, minimal external dependencies.

---

## 1. Core Philosophies: SMALL-MAKE

The agent must adhere to the **SMALL-MAKE** standard for every Makefile:

- **S**mall & Minimalistic: Minimal code, no bloat, essential features only
- **M**odular Structure: Split into reusable modules, logical organization
- **A**utomated Help: Built-in `help` target, self-documenting
- **L**ogging Modes: Debug and verbose modes for troubleshooting
- **L**ow Dependencies: Minimal external tools, POSIX-compliant

- **M**aintainable: Clear structure, consistent patterns, easy to modify
- **A**ccessible: Clear variable names, readable recipes, obvious behavior
- **K**eep It Simple: Simple solutions first, avoid over-engineering
- **E**rror Handling: Proper error messages, fail-fast behavior

**V**erified Builds: Agent-generated Makefiles MUST work correctly before delivery
- **E**xplicit Targets: Clear target names, obvious dependencies
- **R**eusable Patterns: DRY principles, shared functionality in modules
- **I**ncremental Builds: Proper dependency tracking, efficient rebuilds
- **F**lexible Configuration: Environment variables, configurable behavior
- **I**dempotent: Safe to run multiple times, no side effects
- **E**fficient: Fast execution, minimal overhead, parallel builds where possible

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Build Verification Protocol

**CRITICAL: Agents MUST verify that all generated/modified Makefiles execute correctly before presenting them to the user.**

**MANDATORY: After ANY modification to a Makefile, the agent MUST verify it is parseable and working.**

#### Verification Checklist

**Before delivering ANY Makefile (including after modifications), the agent MUST:**

1. **Parseability Verification (MANDATORY after modifications)**:
   ```bash
   # Check Makefile is parseable
   make -n -f Makefile 2>&1
   # Exit code MUST be 0, MUST show no parse errors
   
   # Verify no syntax errors
   make --dry-run --always-make 2>&1
   # Exit code MUST be 0, no syntax errors
   
   # Test parseability with different make versions (if available)
   make -n 2>&1 | grep -i "error\|warning" | grep -v "Nothing to be done"
   # Should produce no output (no errors/warnings)
   ```

2. **Post-Modification Verification (MANDATORY)**:
   ```bash
   # After modifying Makefile, ALWAYS run:
   # 1. Parse check
   make -n 2>&1
   # Exit code MUST be 0
   
   # 2. Test affected targets
   make help 2>&1
   # MUST work without errors
   
   # 3. Test default target (if applicable)
   make 2>&1
   # MUST execute or show clear error (not parse error)
   ```

2. **Target Execution Test**:
   ```bash
   # Test help target
   make help
   # MUST display help text without errors
   
   # Test default target
   make
   # MUST execute successfully or show clear error
   
   # Test verbose mode
   make V=1
   # MUST show verbose output
   
   # Test debug mode
   make DEBUG=1
   # MUST show debug information
   ```

3. **Module Inclusion Test**:
   ```bash
   # If using includes, verify they work
   make -n
   # MUST not show "No rule to make target" errors for includes
   ```

4. **Error Handling Test**:
   ```bash
   # Test with missing dependencies (if applicable)
   make clean
   make
   # MUST handle errors gracefully
   ```

5. **POSIX Compliance Check**:
   ```bash
   # Verify POSIX compatibility
   make SHELL=/bin/sh -n
   # MUST work with POSIX shell
   ```

### B. Error Correction Process

If verification fails:

1. **Read the error message** - syntax errors, missing targets, undefined variables
2. **Identify the root cause** - typo, missing dependency, incorrect syntax
3. **Fix the issue** in the generated Makefile
4. **Re-verify** by running checks again
5. **Repeat until successful** - iterate as many times as needed
6. **Only present working Makefiles** to the user

### C. Example Verification Workflow

```bash
# Agent must simulate/verify this workflow

# 1. Check syntax
make -n 2>&1 | grep -i error
# Should produce no output

# 2. Test help
make help
# Should display help text

# 3. Test default target
make
# Should execute successfully

# 4. Test verbose mode
make V=1
# Should show verbose output

# 5. Test debug mode
make DEBUG=1
# Should show debug information

# If any step fails:
# - Read the error output
# - Fix the Makefile
# - Try again
# - Repeat until success
```

**CRITICAL**: Never provide a Makefile to the user that doesn't work. Always verify first, fix issues, then present the working solution.

**CRITICAL**: After ANY modification to an existing Makefile, the agent MUST:
1. Run `make -n` to verify parseability
2. Test affected targets to ensure they still work
3. Only present the modified Makefile if all checks pass

---

## 2A. TDD Protocol for Makefiles (MANDATORY)

### A. Test-Driven Development Cycle

**CRITICAL: When developing new Makefile targets or modifying existing ones, follow the TDD cycle to ensure correctness and prevent regressions.**

#### TDD Cycle Diagram

```
    ┌─────────────────────────────────────────────────────────────┐
    │                    MAKEFILE TDD CYCLE                       │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   1. WRITE TEST FIRST  │
                 │   Define expected      │
                 │   target behavior      │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │   2. RUN TEST (FAIL)   │◄────────────────┐
                 │   Verify test fails    │                 │
                 │   (target missing)     │                 │
                 └───────────┬────────────┘                 │
                             │                              │
                             ▼                              │
                 ┌────────────────────────┐                 │
                 │ 3. IMPLEMENT TARGET    │                 │
                 │ Write minimal Makefile │                 │
                 │ code to pass test      │                 │
                 └───────────┬────────────┘                 │
                             │                              │
                             ▼                              │
                 ┌────────────────────────┐                 │
                 │   4. RUN TEST (PASS)   │                 │
                 │   make -n, make test   │                 │
                 │   Verify all pass      │                 │
                 └───────────┬────────────┘                 │
                             │                              │
                             ▼                              │
                 ┌────────────────────────┐                 │
                 │   5. REFACTOR          │                 │
                 │   Clean up code        │─────────────────┘
                 │   Re-run tests         │   (if tests fail)
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │   6. COMMIT            │
                 │   All tests pass       │
                 └────────────────────────┘
```

### B. Makefile Test Patterns

**Write tests for Makefile targets to validate behavior before implementation.**

#### Test Module Template

```makefile
# make/test-targets.mk - Target testing infrastructure

.PHONY: test-targets test-build test-clean test-help test-all-targets

# Run all target tests
test-targets: ## Run all Makefile target tests ## Test
	@echo "Running Makefile target tests..."
	@$(MAKE) test-build
	@$(MAKE) test-clean
	@$(MAKE) test-help
	@echo "✓ All target tests passed"

# Test build target
test-build: ## Test build target works correctly ## Test
	@echo "Testing build target..."
	@# Test 1: Target exists and is parseable
	@make -n build > /dev/null 2>&1 || \
		(echo "✗ FAIL: build target not parseable" && exit 1)
	@echo "  ✓ build target is parseable"
	@# Test 2: Creates expected output
	@$(MAKE) clean > /dev/null 2>&1 || true
	@$(MAKE) build > /dev/null 2>&1
	@test -f $(BUILD_DIR)/main || \
		(echo "✗ FAIL: build did not create $(BUILD_DIR)/main" && exit 1)
	@echo "  ✓ build creates expected output"
	@# Test 3: Incremental build works (second run skips)
	@output=$$($(MAKE) build 2>&1); \
	if echo "$$output" | grep -q "is up to date\|Nothing to be done"; then \
		echo "  ✓ incremental build works"; \
	else \
		echo "  ✓ build completed (may have rebuilt)"; \
	fi

# Test clean target
test-clean: ## Test clean target works correctly ## Test
	@echo "Testing clean target..."
	@# Setup: ensure build exists
	@$(MAKE) build > /dev/null 2>&1 || true
	@# Test 1: Clean removes build directory
	@$(MAKE) clean > /dev/null 2>&1
	@test ! -d $(BUILD_DIR) || \
		(echo "✗ FAIL: clean did not remove $(BUILD_DIR)" && exit 1)
	@echo "  ✓ clean removes build directory"
	@# Test 2: Clean is idempotent (safe to run twice)
	@$(MAKE) clean > /dev/null 2>&1 || \
		(echo "✗ FAIL: clean is not idempotent" && exit 1)
	@echo "  ✓ clean is idempotent"

# Test help target
test-help: ## Test help target displays correctly ## Test
	@echo "Testing help target..."
	@# Test 1: Help outputs something
	@output=$$($(MAKE) help 2>&1); \
	test -n "$$output" || \
		(echo "✗ FAIL: help produces no output" && exit 1)
	@echo "  ✓ help produces output"
	@# Test 2: Help contains expected text
	@$(MAKE) help 2>&1 | grep -q "Usage\|target\|Available" || \
		(echo "✗ FAIL: help missing expected content" && exit 1)
	@echo "  ✓ help contains expected content"
```

### C. TDD Example: Adding a New Target

**Step-by-step TDD workflow for adding an `install` target.**

#### Step 1: Write Test First (RED)

```makefile
# make/test-targets.mk - Add test before implementation

test-install: ## Test install target ## Test
	@echo "Testing install target..."
	@# Test 1: Target exists
	@make -n install > /dev/null 2>&1 || \
		(echo "✗ FAIL: install target not found" && exit 1)
	@echo "  ✓ install target exists"
	@# Test 2: Installs to correct location
	@$(MAKE) build > /dev/null 2>&1
	@PREFIX=/tmp/test-install $(MAKE) install > /dev/null 2>&1
	@test -f /tmp/test-install/bin/$(PROJECT_NAME) || \
		(echo "✗ FAIL: install did not copy binary" && exit 1)
	@echo "  ✓ install copies binary to PREFIX/bin"
	@# Cleanup
	@rm -rf /tmp/test-install
```

#### Step 2: Run Test - Expect Failure (RED)

```bash
$ make test-install
Testing install target...
✗ FAIL: install target not found
make: *** [test-install] Error 1
```

#### Step 3: Implement Target (GREEN)

```makefile
# make/install.mk - Minimal implementation to pass test

PREFIX ?= /usr/local

.PHONY: install
install: $(BUILD_DIR)/main ## Install the project ## Build
	@echo "Installing to $(PREFIX)..."
	@mkdir -p $(PREFIX)/bin
	@cp $(BUILD_DIR)/main $(PREFIX)/bin/$(PROJECT_NAME)
	@echo "✓ Installed $(PROJECT_NAME) to $(PREFIX)/bin"
```

#### Step 4: Run Test - Expect Success (GREEN)

```bash
$ make test-install
Testing install target...
  ✓ install target exists
  ✓ install copies binary to PREFIX/bin
```

#### Step 5: Refactor and Re-test

```makefile
# make/install.mk - Refactored with better error handling

PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin

.PHONY: install uninstall

install: $(BUILD_DIR)/main ## Install the project ## Build
	@echo "[1/2] Creating directory $(BINDIR)..."
	@mkdir -p $(BINDIR)
	@echo "[2/2] Installing $(PROJECT_NAME)..."
	@install -m 755 $(BUILD_DIR)/main $(BINDIR)/$(PROJECT_NAME)
	@echo "✓ Installed $(PROJECT_NAME) to $(BINDIR)"

uninstall: ## Uninstall the project ## Build
	@echo "Removing $(BINDIR)/$(PROJECT_NAME)..."
	@rm -f $(BINDIR)/$(PROJECT_NAME)
	@echo "✓ Uninstalled $(PROJECT_NAME)"
```

### D. Test Categories for Makefiles

```makefile
# make/test-comprehensive.mk - Comprehensive test suite

.PHONY: test-all test-parse test-targets test-modes test-deps

# Run all tests
test-all: ## Run comprehensive Makefile test suite ## Test
	@echo "═══════════════════════════════════════════"
	@echo "        MAKEFILE TEST SUITE"
	@echo "═══════════════════════════════════════════"
	@$(MAKE) test-parse
	@$(MAKE) test-targets
	@$(MAKE) test-modes
	@$(MAKE) test-deps
	@echo "═══════════════════════════════════════════"
	@echo "        ALL TESTS PASSED ✓"
	@echo "═══════════════════════════════════════════"

# Test parseability
test-parse: ## Test Makefile syntax ## Test
	@echo "Testing parseability..."
	@make -n > /dev/null 2>&1 || \
		(echo "✗ FAIL: Makefile has syntax errors" && exit 1)
	@echo "  ✓ Makefile is parseable"

# Test verbose and debug modes
test-modes: ## Test V=1 and DEBUG=1 modes ## Test
	@echo "Testing modes..."
	@$(MAKE) V=1 help > /dev/null 2>&1 || \
		(echo "✗ FAIL: V=1 mode broken" && exit 1)
	@echo "  ✓ Verbose mode (V=1) works"
	@$(MAKE) DEBUG=1 help > /dev/null 2>&1 || \
		(echo "✗ FAIL: DEBUG=1 mode broken" && exit 1)
	@echo "  ✓ Debug mode (DEBUG=1) works"

# Test dependency tracking
test-deps: ## Test dependency tracking works ## Test
	@echo "Testing dependency tracking..."
	@$(MAKE) clean > /dev/null 2>&1 || true
	@$(MAKE) build > /dev/null 2>&1
	@first_hash=$$(sha256sum $(BUILD_DIR)/main 2>/dev/null | cut -d' ' -f1); \
	sleep 1; \
	$(MAKE) build > /dev/null 2>&1; \
	second_hash=$$(sha256sum $(BUILD_DIR)/main 2>/dev/null | cut -d' ' -f1); \
	if [ "$$first_hash" = "$$second_hash" ]; then \
		echo "  ✓ Dependency tracking prevents unnecessary rebuilds"; \
	else \
		echo "  ⚠ Warning: Binary changed without source changes"; \
	fi
```

### E. TDD Best Practices for Makefiles

**Guidelines for effective Test-Driven Makefile development:**

1. **Test Target Existence**: Verify target is parseable with `make -n target`
2. **Test Output Artifacts**: Check expected files are created
3. **Test Idempotency**: Run target twice, verify consistent behavior
4. **Test Incremental Builds**: Ensure caching works correctly
5. **Test Error Cases**: Verify graceful failure with missing dependencies
6. **Test Mode Flags**: Verify V=1 and DEBUG=1 work with all targets

```makefile
# TDD Checklist for new targets:
# [ ] Write test first (test-<target>)
# [ ] Run test, verify it fails
# [ ] Implement minimal target to pass test
# [ ] Run test, verify it passes
# [ ] Refactor for clarity and reusability
# [ ] Re-run all tests to prevent regressions
# [ ] Update help documentation
```

---

## 2B. Bug Fix Protocol for Makefiles (MANDATORY)

### A. Bug Fix Workflow

**CRITICAL: When fixing bugs in Makefiles, follow this systematic workflow to ensure fixes are correct and don't introduce regressions.**

#### Bug Fix Workflow Diagram

```
    ┌─────────────────────────────────────────────────────────────┐
    │                  MAKEFILE BUG FIX WORKFLOW                  │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   1. REPRODUCE BUG     │
                 │   Run failing command  │
                 │   Document error       │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │   2. WRITE FAILING TEST│
                 │   Create test that     │
                 │   demonstrates bug     │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │   3. DIAGNOSE CAUSE    │
                 │   make DEBUG=1         │
                 │   make -d (trace)      │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │   4. IMPLEMENT FIX     │
                 │   Minimal change to    │
                 │   fix the issue        │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │   5. VERIFY FIX        │
                 │   make -n (parseability)│
                 │   Run failing test     │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │   6. RUN ALL TESTS     │
                 │   Ensure no regressions│
                 │   make test-all        │
                 └───────────┬────────────┘
                             │
                    Tests pass?
                    │        │
                 Yes│        │No
                    ▼        ▼
         ┌──────────────┐  ┌──────────────┐
         │  7. COMMIT   │  │ GO TO STEP 4 │
         │  Document fix│  │ Revise fix   │
         └──────────────┘  └──────────────┘
```

### B. Bug Diagnosis Tools

**Use these debugging techniques to identify Makefile issues.**

```makefile
# make/debug-tools.mk - Debugging utilities

.PHONY: debug-info debug-vars debug-trace debug-deps

# Show comprehensive debug information
debug-info: ## Show all debug information ## Utility
	@echo "═══════════════════════════════════════════"
	@echo "        MAKEFILE DEBUG INFO"
	@echo "═══════════════════════════════════════════"
	@echo "Make version: $$(make --version | head -1)"
	@echo "Shell: $(SHELL)"
	@echo "MAKEFLAGS: $(MAKEFLAGS)"
	@echo "MAKEFILE_LIST: $(MAKEFILE_LIST)"
	@echo "CURDIR: $(CURDIR)"
	@echo ""
	@$(MAKE) debug-vars

# Show all important variables
debug-vars: ## Show all configuration variables ## Utility
	@echo "Variables:"
	@echo "  PROJECT_NAME = $(PROJECT_NAME)"
	@echo "  VERSION      = $(VERSION)"
	@echo "  BUILD_DIR    = $(BUILD_DIR)"
	@echo "  SRC_DIR      = $(SRC_DIR)"
	@echo "  CC           = $(CC)"
	@echo "  CFLAGS       = $(CFLAGS)"
	@echo "  V            = $(V)"
	@echo "  DEBUG        = $(DEBUG)"

# Trace target execution (use: make debug-trace TARGET=build)
debug-trace: ## Trace target execution (TARGET=<target>) ## Utility
	@echo "Tracing target: $(TARGET)"
	@echo "═══════════════════════════════════════════"
	@make -d $(TARGET) 2>&1 | head -100
	@echo "═══════════════════════════════════════════"
	@echo "(Output truncated to 100 lines)"

# Show dependency tree for a target
debug-deps: ## Show dependencies (TARGET=<target>) ## Utility
	@echo "Dependencies for: $(TARGET)"
	@echo "═══════════════════════════════════════════"
	@make -p $(TARGET) 2>/dev/null | grep -A1 "^$(TARGET):" || \
		echo "Target not found or no explicit dependencies"
```

### C. Common Makefile Bug Patterns

**Recognize and fix common Makefile issues.**

#### Bug Pattern 1: Missing Dependencies

```makefile
# ❌ BUG: Target always rebuilds (missing dependency)
build:
	@$(CC) $(CFLAGS) -o $(BUILD_DIR)/main $(SRC_DIR)/main.c

# ✅ FIX: Add proper dependencies
$(BUILD_DIR)/main: $(SRC_DIR)/main.c | $(BUILD_DIR)
	@$(CC) $(CFLAGS) -o $@ $<

build: $(BUILD_DIR)/main
	@echo "Build complete"
```

#### Bug Pattern 2: Incorrect Variable Expansion

```makefile
# ❌ BUG: Immediate expansion captures wrong value
FILES := $(wildcard $(SRC_DIR)/*.c)
# If SRC_DIR changes later, FILES won't update

# ✅ FIX: Use deferred expansion for dependencies
FILES = $(wildcard $(SRC_DIR)/*.c)
# Or ensure SRC_DIR is set before FILES
```

#### Bug Pattern 3: Recipe Command Failures Hidden

```makefile
# ❌ BUG: Errors are silently ignored
build:
	@cp src/*.c build/ 2>/dev/null
	@echo "Build complete"
	# If cp fails, "Build complete" still shows

# ✅ FIX: Fail fast on errors
build:
	@cp src/*.c build/ || (echo "Error: Copy failed" && exit 1)
	@echo "Build complete"
```

#### Bug Pattern 4: Tab vs Space Issues

```makefile
# ❌ BUG: Spaces instead of tabs (invisible error!)
build:
    @echo "Building..."   # WRONG: 4 spaces

# ✅ FIX: Use actual tab character
build:
	@echo "Building..."   # CORRECT: 1 tab
```

### D. Bug Fix Example: Fixing Incremental Build

**Complete example of fixing a caching/incremental build bug.**

#### Step 1: Reproduce the Bug

```bash
$ make build
Building...
$ make build
Building...  # BUG: Should say "up to date"!
```

#### Step 2: Write Failing Test

```makefile
# make/test-targets.mk - Add regression test

test-incremental-build: ## Test incremental build caching ## Test
	@echo "Testing incremental build..."
	@$(MAKE) clean > /dev/null 2>&1 || true
	@$(MAKE) build > /dev/null 2>&1
	@# Second build should be skipped
	@output=$$($(MAKE) build 2>&1); \
	if echo "$$output" | grep -qi "Building"; then \
		echo "✗ FAIL: Build ran twice (no caching)"; \
		exit 1; \
	fi
	@echo "  ✓ Incremental build uses cache"
```

#### Step 3: Diagnose the Cause

```bash
$ make DEBUG=1 build
[DEBUG] Starting build
[DEBUG] No dependencies specified - always rebuilds!

$ make -d build 2>&1 | grep -A5 "Considering target"
# Shows: File 'build' does not exist, must rebuild
```

#### Step 4: Implement the Fix

```makefile
# ❌ BEFORE (Bug): No dependency tracking
.PHONY: build
build:
	@echo "Building..."
	@$(CC) $(CFLAGS) -o $(BUILD_DIR)/main $(SRC_DIR)/main.c

# ✅ AFTER (Fix): Proper dependency tracking
SOURCES := $(wildcard $(SRC_DIR)/*.c)
OBJECTS := $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(call compile-c,$<,$@)

$(BUILD_DIR)/main: $(OBJECTS)
	$(call link,$@,$^)

.PHONY: build
build: $(BUILD_DIR)/main
	@echo "Build complete"
```

#### Step 5: Verify the Fix

```bash
$ make -n build   # Verify parseability
$ make test-incremental-build
Testing incremental build...
  ✓ Incremental build uses cache
```

#### Step 6: Run All Tests

```bash
$ make test-all
═══════════════════════════════════════════
        MAKEFILE TEST SUITE
═══════════════════════════════════════════
Testing parseability...
  ✓ Makefile is parseable
Testing build target...
  ✓ build target is parseable
  ✓ build creates expected output
  ✓ incremental build works
Testing clean target...
  ✓ clean removes build directory
  ✓ clean is idempotent
═══════════════════════════════════════════
        ALL TESTS PASSED ✓
═══════════════════════════════════════════
```

### E. Bug Fix Verification Checklist

**MANDATORY: Complete this checklist for every bug fix.**

```makefile
# Bug Fix Verification Checklist:
# [ ] Bug reproduced and documented
# [ ] Failing test written (demonstrates bug)
# [ ] Root cause identified (using DEBUG=1 or make -d)
# [ ] Fix implemented (minimal change)
# [ ] make -n passes (parseability verified)
# [ ] Failing test now passes
# [ ] All existing tests pass (no regressions)
# [ ] Fix documented in commit message
```

### F. Regression Prevention

**Add regression tests for every bug fix to prevent reoccurrence.**

```makefile
# make/test-regressions.mk - Regression test suite

.PHONY: test-regressions

# Regression tests for fixed bugs
test-regressions: ## Run regression tests ## Test
	@echo "Running regression tests..."
	@echo ""
	@echo "BUG-001: Incremental build not caching"
	@$(MAKE) test-incremental-build
	@echo ""
	@echo "BUG-002: Clean not idempotent"
	@$(MAKE) test-clean-idempotent
	@echo ""
	@echo "✓ All regression tests passed"

test-clean-idempotent:
	@$(MAKE) clean > /dev/null 2>&1 || true
	@$(MAKE) clean > /dev/null 2>&1 || \
		(echo "✗ FAIL: BUG-002 regression - clean not idempotent" && exit 1)
	@echo "  ✓ BUG-002: clean is idempotent"
```

---

## 3. Incremental Builds and Caching (MANDATORY)

### A. Dependency-Based Caching

**CRITICAL: Makefiles MUST use Make's built-in dependency tracking to prevent executing the same task twice unless sources have changed.**

#### Core Principle

**Make automatically skips targets if dependencies haven't changed. Agents MUST structure Makefiles to leverage this.**

#### ✅ CORRECT - Proper Dependency Tracking

```makefile
# make/build.mk - Proper dependency tracking

# Object files depend on source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(call compile-c,$<,$@)

# Executable depends on object files
$(BUILD_DIR)/main: $(BUILD_DIR)/main.o $(BUILD_DIR)/utils.o
	$(call link,$@,$^)

# Build target depends on executable
build: $(BUILD_DIR)/main
	@echo "Build complete"

# Running make build twice:
# First run: Compiles and links
# Second run: "make: 'build' is up to date." (skipped - cached)
```

#### ❌ WRONG - No Dependency Tracking

```makefile
# ❌ Always executes, even if nothing changed
build:
	@echo "Building..."
	@gcc -o output src/main.c src/utils.c
	# Runs EVERY time, even if sources unchanged
```

### B. Timestamp-Based Caching

**Make uses file timestamps to determine if targets need rebuilding. Structure dependencies correctly.**

```makefile
# ✅ CORRECT - Make automatically checks timestamps
$(BUILD_DIR)/main: $(SRC_DIR)/main.c
	$(CC) $(CFLAGS) -o $@ $<

# If src/main.c is newer than build/main, Make rebuilds
# If build/main is newer, Make skips (cached)
```

### C. Phony Targets and Caching

**Use `.PHONY` for targets that should always run, but still check dependencies for actual work.**

```makefile
# ✅ CORRECT - Phony target with dependency checking
.PHONY: build
build: $(BUILD_DIR)/main
	@echo "Build complete"

# build is phony (always considered out-of-date)
# BUT $(BUILD_DIR)/main is checked for dependencies
# So actual compilation only happens if sources changed
```

### D. Intermediate File Caching

**Use intermediate files to cache expensive operations.**

```makefile
# ✅ CORRECT - Caching intermediate results
$(BUILD_DIR)/processed_data.json: $(SRC_DIR)/raw_data.csv
	@echo "Processing data..."
	@python3 process.py $< > $@

$(BUILD_DIR)/report.html: $(BUILD_DIR)/processed_data.json
	@echo "Generating report..."
	@python3 generate_report.py $< > $@

report: $(BUILD_DIR)/report.html
	@echo "Report ready"

# First run: Processes data, generates report
# Second run (if raw_data.csv unchanged): Skips processing, uses cached processed_data.json
# Only regenerates report if processed_data.json changed
```

### E. Directory Timestamp Caching

**Handle directory dependencies correctly.**

```makefile
# ✅ CORRECT - Directory order-only prerequisite
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# | means order-only: ensures directory exists
# Doesn't trigger rebuild if directory timestamp changes
```

### F. Verification of Caching

**Agents MUST verify that caching works correctly.**

```makefile
# make/test-cache.mk - Verify caching works

.PHONY: test-cache
test-cache: ## Test that caching prevents duplicate work ## Utility
	@echo "Testing cache behavior..."
	@echo "1. First build:"
	@$(MAKE) clean > /dev/null 2>&1 || true
	@$(MAKE) build
	@echo ""
	@echo "2. Second build (should use cache):"
	@$(MAKE) build
	@echo ""
	@echo "3. Touch source file:"
	@touch $(SRC_DIR)/main.c
	@echo "4. Third build (should rebuild):"
	@$(MAKE) build
```

### G. Caching Best Practices

```makefile
# ✅ CORRECT - Comprehensive caching example

# 1. Source files tracked
SOURCES := $(wildcard $(SRC_DIR)/*.c)
OBJECTS := $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# 2. Header dependencies (if using -MMD)
DEPFILES := $(OBJECTS:.o=.d)
-include $(DEPFILES)

# 3. Compile with dependency generation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# 4. Link only if objects changed
$(BUILD_DIR)/main: $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^

# 5. Build target depends on executable
build: $(BUILD_DIR)/main
	@echo "Build complete"

# Behavior:
# - First run: Compiles all sources, links
# - Second run: "Nothing to be done" (all cached)
# - Modify one source: Only recompiles that file, relinks
# - Modify header: Recompiles dependent sources, relinks
```

---

## 4. Progress Indicators (MANDATORY)

### A. Progress Logs by Default

**CRITICAL: Makefiles MUST provide progress feedback by default (unless verbose mode is enabled).**

#### Principle

**When `V=0` (default), show progress. When `V=1` (verbose), show full commands.**

#### ✅ CORRECT - Progress Log Implementation

```makefile
# make/progress.mk - Progress indicators

# Progress counter
TOTAL_STEPS := 0
CURRENT_STEP := 0

# Progress function
# Usage: $(call progress,message)
define progress
	$(eval CURRENT_STEP := $(shell echo $$(($(CURRENT_STEP) + 1))))
	@printf "[%2d/%2d] %s\n" $(CURRENT_STEP) $(TOTAL_STEPS) "$(1)"
endef

# Example usage
build: $(BUILD_DIR)/main
	$(eval TOTAL_STEPS := 3)
	$(call progress,Building project...)
	$(call progress,Compiling sources...)
	$(call progress,Linking executable...)
	@echo "Build complete"
```

#### Simple Progress Messages

```makefile
# ✅ CORRECT - Simple progress messages
build: $(BUILD_DIR)/main
	@echo "[1/3] Building project..."
	@echo "[2/3] Compiling sources..."
	@echo "[3/3] Linking executable..."
	@echo "✓ Build complete"
```

### B. Progress Bars (Simple Implementation)

**For long-running operations, show progress bars.**

```makefile
# make/progress-bar.mk - Simple progress bar

# Progress bar function
# Usage: $(call progress-bar,current,total,message)
define progress-bar
	@printf "\r[%s] %d/%d %s" \
		"$$(printf '█%.0s' $$(seq 1 $$(($(1) * 20 / $(2)))))" \
		$(1) $(2) "$(3)"
endef

# Example: Processing files
process-files:
	@echo "Processing files..."
	@i=1; \
	for file in $(SRC_DIR)/*.c; do \
		$(call progress-bar,$$i,$$(ls -1 $(SRC_DIR)/*.c | wc -l),Processing $$(basename $$file)); \
		$(CC) $(CFLAGS) -c $$file -o $(BUILD_DIR)/$$(basename $$file .c).o; \
		i=$$((i+1)); \
	done
	@echo ""
	@echo "✓ Processing complete"
```

### C. Verbose vs Progress Mode

**Show progress by default, full commands in verbose mode.**

```makefile
# make/functions.mk - Progress-aware commands

V ?= 0

# Command with progress
# Usage: $(call run-with-progress,command,progress-message)
ifeq ($(V),1)
  # Verbose mode: show command
  run-with-progress = $(1)
else
  # Default: show progress
  run-with-progress = @echo "$(2)"; $(1)
endif

# Compile with progress
define compile-c
	$(if $(filter 1,$(V)),\
		@echo "$(CC) $(CFLAGS) -c $(1) -o $(2)",\
		@printf "  CC    %s\n" "$(notdir $(2))")
	@$(CC) $(CFLAGS) -c $(1) -o $(2)
endef

# Usage
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(call compile-c,$<,$@)

# With V=0 (default): "  CC    main.o"
# With V=1 (verbose): "gcc -Wall -Wextra -c src/main.c -o build/main.o"
```

### D. Task Progress Tracking

**Track progress across multiple targets.**

```makefile
# make/progress.mk - Task progress tracking

# Progress tracking variables
PROGRESS_TOTAL := 0
PROGRESS_CURRENT := 0

# Initialize progress
# Usage: $(call init-progress,total)
define init-progress
	$(eval PROGRESS_TOTAL := $(1))
	$(eval PROGRESS_CURRENT := 0)
endef

# Update progress
# Usage: $(call update-progress,message)
define update-progress
	$(eval PROGRESS_CURRENT := $(shell echo $$(($(PROGRESS_CURRENT) + 1))))
	@printf "[%d/%d] %s\n" $(PROGRESS_CURRENT) $(PROGRESS_TOTAL) "$(1)"
endef

# Example usage
build: $(call init-progress,4)
	$(call update-progress,Preparing build...)
	@mkdir -p $(BUILD_DIR)
	$(call update-progress,Compiling sources...)
	@$(MAKE) $(BUILD_DIR)/main.o
	$(call update-progress,Linking executable...)
	@$(MAKE) $(BUILD_DIR)/main
	$(call update-progress,Build complete)
	@echo "✓ All done"
```

### E. Percentage Progress

**Show percentage completion for long operations.**

```makefile
# make/progress-percent.mk - Percentage progress

# Percentage progress function
# Usage: $(call progress-percent,current,total,message)
define progress-percent
	@percent=$$(($(1) * 100 / $(2))); \
	printf "\r[%3d%%] %s" $$percent "$(3)"
endef

# Example: Building multiple targets
build-all:
	@echo "Building all components..."
	@i=1; \
	total=$$(echo $(TARGETS) | wc -w); \
	for target in $(TARGETS); do \
		$(call progress-percent,$$i,$$total,Building $$target); \
		$(MAKE) $$target > /dev/null 2>&1; \
		i=$$((i+1)); \
	done
	@echo ""
	@echo "✓ Build complete"
```

### F. Progress Indicator Examples

```makefile
# ✅ CORRECT - Comprehensive progress example

# Simple step-by-step progress
install-deps:
	@echo "[1/4] Installing system dependencies..."
	@apt-get install -y package1
	@echo "[2/4] Installing Python packages..."
	@pip install -r requirements.txt
	@echo "[3/4] Setting up configuration..."
	@cp config.example config.ini
	@echo "[4/4] Installation complete"
	@echo "✓ Ready to use"

# File processing with progress
process: $(SRC_DIR)/*.c
	@echo "Processing $(words $^) files..."
	@i=0; \
	for file in $^; do \
		i=$$((i+1)); \
		printf "\r[%d/%d] Processing %s" $$i $(words $^) "$$(basename $$file)"; \
		$(CC) $(CFLAGS) -c $$file; \
	done
	@echo ""
	@echo "✓ Processing complete"

# Build with progress
build: $(BUILD_DIR)/main
	@echo "Building $(PROJECT_NAME)..."
	@echo "[✓] Sources compiled"
	@echo "[✓] Objects linked"
	@echo "[✓] Build complete"
```

### G. Progress Indicator Requirements

**All Makefiles MUST:**

- [ ] Show progress messages by default (when `V=0`)
- [ ] Show full commands in verbose mode (when `V=1`)
- [ ] Use clear, concise progress messages
- [ ] Indicate completion with checkmarks (✓) or similar
- [ ] Show step numbers for multi-step operations
- [ ] Use progress bars for long-running operations (optional but recommended)

---

## 5. Reproducible Builds (MANDATORY)

### A. Build Determinism

**CRITICAL: All Makefiles MUST produce identical outputs given identical inputs, regardless of build environment or timing.**

#### Core Principles

1. **Deterministic Outputs**: Same inputs always produce same outputs
2. **Environment Independence**: Builds work across different machines/environments
3. **Version Pinning**: Tool versions and dependencies must be specified
4. **Timestamp Independence**: Builds should not depend on file modification times (except for dependency tracking)

#### ✅ CORRECT - Reproducible Build

```makefile
# make/config.mk - Reproducible configuration

# Pin tool versions
CC := gcc
CC_VERSION := $(shell $(CC) --version | head -1)
PYTHON := python3
PYTHON_VERSION := $(shell $(PYTHON) --version)

# Set reproducible build flags
CFLAGS := -Wall -Wextra -std=c11
CFLAGS += -fno-ident  # Remove compiler identification
CFLAGS += -Wno-builtin-macro-redefined
CFLAGS += -D__DATE__="\"redacted\"" -D__TIME__="\"redacted\"" -D__TIMESTAMP__="\"redacted\""

# Reproducible build directory
BUILD_DIR := build
DIST_DIR := dist

# Export for reproducibility
export SOURCE_DATE_EPOCH := $(shell date +%s)
```

#### ❌ WRONG - Non-Reproducible Build

```makefile
# ❌ Includes timestamps in binary
CFLAGS := -Wall -Wextra
# Missing: -fno-ident, date/time redaction

# ❌ Uses system time
build:
	@echo "Built on $(shell date)"
	# Different output every time
```

### B. Source Date Epoch

**Use SOURCE_DATE_EPOCH for reproducible timestamps.**

```makefile
# make/reproducible.mk - Reproducible build utilities

# Set SOURCE_DATE_EPOCH from git commit or environment
SOURCE_DATE_EPOCH ?= $(shell git log -1 --format=%ct 2>/dev/null || echo $$(date +%s))
export SOURCE_DATE_EPOCH

# Use in build process
build-info:
	@echo "Build date: $$(date -u -d @$(SOURCE_DATE_EPOCH) +%Y-%m-%d 2>/dev/null || date -u -r $(SOURCE_DATE_EPOCH) +%Y-%m-%d 2>/dev/null || echo 'unknown')"
```

### C. Tool Version Verification

**Verify tool versions match requirements.**

```makefile
# make/verify-versions.mk - Version verification

# Required versions
REQUIRED_GCC_VERSION := 11.0.0
REQUIRED_PYTHON_VERSION := 3.9.0

# Check versions
check-versions: ## Verify tool versions meet requirements ## Utility
	@echo "Checking tool versions..."
	@$(CC) --version | head -1
	@$(PYTHON) --version
	@echo "✓ All versions verified"

# Verify GCC version
verify-gcc:
	@gcc_version=$$($(CC) -dumpversion); \
	if [ "$$(printf '%s\n' $(REQUIRED_GCC_VERSION) $$gcc_version | sort -V | head -1)" != "$(REQUIRED_GCC_VERSION)" ]; then \
		echo "Error: GCC $$gcc_version < required $(REQUIRED_GCC_VERSION)"; \
		exit 1; \
	fi
```

### D. Deterministic File Ordering

**Ensure file processing order is deterministic.**

```makefile
# ✅ CORRECT - Deterministic ordering
SOURCES := $(sort $(wildcard $(SRC_DIR)/*.c))
OBJECTS := $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# ❌ WRONG - Non-deterministic ordering
SOURCES := $(wildcard $(SRC_DIR)/*.c)  # Order depends on filesystem
```

### E. Build Artifact Reproducibility

**Ensure build artifacts are reproducible.**

```makefile
# make/reproducible-build.mk - Reproducible build flags

# C/C++ reproducible flags
REPRODUCIBLE_CFLAGS := -fno-ident
REPRODUCIBLE_CFLAGS += -Wno-builtin-macro-redefined
REPRODUCIBLE_CFLAGS += -D__DATE__="\"redacted\""
REPRODUCIBLE_CFLAGS += -D__TIME__="\"redacted\""
REPRODUCIBLE_CFLAGS += -D__TIMESTAMP__="\"redacted\""

# Python reproducible builds
PYTHONHASHSEED := 0
export PYTHONHASHSEED

# Archive reproducible flags
ARFLAGS := crs  # Deterministic archive creation

# Linker reproducible flags
LDFLAGS += -Wl,--build-id=none  # Remove build ID
```

### F. Environment Variable Management

**Control environment for reproducibility.**

```makefile
# make/environment.mk - Environment control

# Set reproducible environment
export LC_ALL := C
export TZ := UTC
export LANG := C

# Python reproducible hash
export PYTHONHASHSEED := 0

# Source date epoch
export SOURCE_DATE_EPOCH := $(shell git log -1 --format=%ct 2>/dev/null || echo 0)

# Build with clean environment
build-clean:
	env -i PATH="$$PATH" HOME="$$HOME" $(MAKE) build
```

### G. Reproducibility Verification

**Verify builds are reproducible.**

```makefile
# make/test-reproducible.mk - Reproducibility testing

.PHONY: test-reproducible
test-reproducible: ## Test that builds are reproducible ## Utility
	@echo "Testing build reproducibility..."
	@$(MAKE) clean
	@$(MAKE) build
	@hash1=$$(sha256sum $(BUILD_DIR)/main | cut -d' ' -f1); \
	$(MAKE) clean; \
	$(MAKE) build; \
	hash2=$$(sha256sum $(BUILD_DIR)/main | cut -d' ' -f1); \
	if [ "$$hash1" = "$$hash2" ]; then \
		echo "✓ Build is reproducible (hashes match)"; \
	else \
		echo "✗ Build is NOT reproducible (hashes differ)"; \
		exit 1; \
	fi
```

### H. Reproducibility Checklist

**All Makefiles MUST:**

- [ ] Use `SOURCE_DATE_EPOCH` for timestamps
- [ ] Remove compiler identification (`-fno-ident`)
- [ ] Redact date/time macros in code
- [ ] Use deterministic file ordering (`sort`)
- [ ] Pin tool versions or verify minimum versions
- [ ] Set reproducible environment variables
- [ ] Test reproducibility with hash comparison
- [ ] Document required tool versions

---

## 6. Modular Structure (MANDATORY)

### A. File Organization

**CRITICAL: Split large Makefiles into logical modules for maintainability.**

**MANDATORY: All make modules MUST be placed in a separate directory (e.g., `make/`) to keep the main Makefile clean, small, and readable.**

#### Recommended Structure

```
project/
├── Makefile              # Main Makefile (small, clean, readable - only includes)
├── make/                 # MANDATORY: All modules in separate directory
│   ├── config.mk         # Configuration variables
│   ├── targets.mk        # Main targets
│   ├── build.mk          # Build rules
│   ├── test.mk           # Test rules
│   ├── clean.mk          # Cleanup rules
│   ├── help.mk           # Help system
│   ├── functions.mk      # Reusable functions
│   └── errors.mk         # Error handling
```

#### Main Makefile Template

**CRITICAL: Main Makefile MUST be small, clean, and readable. All implementation goes in modules.**

```makefile
# Main Makefile - orchestrates modules
# Keep this file small and readable - all logic in make/ directory

# Include configuration first
include make/config.mk

# Include utility modules
include make/functions.mk
include make/errors.mk

# Include feature modules
include make/build.mk
include make/test.mk
include make/clean.mk
include make/help.mk

# Default target
.DEFAULT_GOAL := help
```

#### ✅ CORRECT - Clean Main Makefile

```makefile
# Makefile - Main entry point (small and readable)

# Configuration
include make/config.mk

# Utilities
include make/functions.mk
include make/errors.mk

# Features
include make/build.mk
include make/test.mk
include make/clean.mk
include make/help.mk

.DEFAULT_GOAL := help
```

#### ❌ WRONG - Monolithic Makefile

```makefile
# ❌ Everything in main Makefile - hard to read and maintain
PROJECT_NAME := myproject
VERSION := 1.0.0
BUILD_DIR := build
SRC_DIR := src
CC := gcc
CFLAGS := -Wall -Wextra
V ?= 0
DEBUG ?= 0
Q := $(if $(filter 1,$(V)),,@)
# ... hundreds more lines of code ...
# This violates the requirement for small, clean, readable main Makefile
```

#### Module Example: config.mk

```makefile
# make/config.mk - Configuration variables

# Project metadata
PROJECT_NAME := myproject
VERSION := 1.0.0

# Directories
BUILD_DIR := build
DIST_DIR := dist
SRC_DIR := src
TEST_DIR := tests

# Tools
CC := gcc
PYTHON := python3

# Build flags
CFLAGS := -Wall -Wextra -std=c11
DEBUG_CFLAGS := -g -O0
RELEASE_CFLAGS := -O2 -DNDEBUG

# Verbose and debug modes
V ?= 0
DEBUG ?= 0

# Export variables for sub-makes
export V DEBUG
```

#### Module Example: build.mk

```makefile
# make/build.mk - Build rules

# Build target
build: $(BUILD_DIR)/main
	@echo "Build complete"

# Object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(call run-cc,$(CC) $(CFLAGS) -c $< -o $@)

# Executable
$(BUILD_DIR)/main: $(BUILD_DIR)/main.o
	$(call run-cc,$(CC) $(CFLAGS) $^ -o $@)

# Create directories
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)
```

### B. Include Guard Pattern

**Prevent multiple includes and handle missing modules gracefully:**

```makefile
# make/config.mk
ifndef CONFIG_MK
CONFIG_MK := 1

# Configuration here
PROJECT_NAME := myproject

endif # CONFIG_MK
```

```makefile
# Main Makefile with include guards
-include make/config.mk
ifndef CONFIG_MK
$(error Configuration file make/config.mk not found)
endif

-include make/build.mk
-include make/test.mk
```

### C. Module Best Practices

**✅ CORRECT - Modular structure:**

```makefile
# Main Makefile
include make/config.mk
include make/build.mk
include make/test.mk
include make/help.mk

.DEFAULT_GOAL := help
```

```makefile
# make/config.mk
PROJECT_NAME := myproject
BUILD_DIR := build
```

```makefile
# make/build.mk
build:
	@echo "Building $(PROJECT_NAME)..."
	@mkdir -p $(BUILD_DIR)
```

**❌ WRONG - Monolithic Makefile:**

```makefile
# Everything in one file - hard to maintain
PROJECT_NAME := myproject
BUILD_DIR := build

build:
	@echo "Building $(PROJECT_NAME)..."
	@mkdir -p $(BUILD_DIR)

test:
	@echo "Testing..."

clean:
	@rm -rf $(BUILD_DIR)

# ... hundreds more lines ...
```

---

## 7. Commenting Best Practices (MANDATORY)

### A. Comment Style

**CRITICAL: All Makefiles MUST contain simple, clean, and helpful comments.**

#### Core Principles

1. **Simple Comments**: Clear, concise, no unnecessary verbosity
2. **Clean Formatting**: Consistent comment style throughout
3. **Purpose-Driven**: Comments explain WHY, not WHAT (code shows what)
4. **Section Headers**: Use comments to organize logical sections
5. **Target Documentation**: Comment complex targets and functions

#### ✅ CORRECT - Good Comments

```makefile
# make/config.mk - Configuration variables

# Project metadata
PROJECT_NAME := myproject
VERSION := 1.0.0

# Build directories
BUILD_DIR := build
DIST_DIR := dist

# Compiler settings
CC := gcc
CFLAGS := -Wall -Wextra -std=c11

# Verbose mode (0=quiet, 1=verbose)
V ?= 0
```

```makefile
# make/build.mk - Build rules

# Main build target depends on executable
build: $(BUILD_DIR)/main
	@echo "Build complete"

# Pattern rule: compile .c files to .o files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Order-only prerequisite: ensure directory exists
# (doesn't trigger rebuild if directory timestamp changes)
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)
```

#### ❌ WRONG - Bad Comments

```makefile
# ❌ Over-commented (obvious from code)
CC := gcc  # Set CC to gcc

# ❌ No comments (unclear purpose)
PROJECT_NAME := myproject
VERSION := 1.0.0

# ❌ Outdated comments
# Old build system - TODO: update
build: old_target

# ❌ Commented-out code (should be removed)
# build: $(OLD_TARGET)
```

### B. Comment Formatting Standards

**Use consistent comment formatting.**

```makefile
# make/config.mk - Configuration variables
# 
# This module defines all configuration variables used throughout
# the build system. Variables can be overridden via environment.

# ============================================================================
# Project Configuration
# ============================================================================

PROJECT_NAME := myproject
VERSION := 1.0.0

# ============================================================================
# Directory Configuration
# ============================================================================

BUILD_DIR := build
SRC_DIR := src

# ============================================================================
# Tool Configuration
# ============================================================================

CC := gcc
PYTHON := python3
```

### C. Target Comments

**Document complex targets and their purpose.**

```makefile
# make/build.mk - Build rules

# Build the entire project
# Dependencies: All source files must be compiled and linked
# Output: Executable in $(BUILD_DIR)/main
build: $(BUILD_DIR)/main
	@echo "Build complete"

# Clean all build artifacts
# Removes: $(BUILD_DIR) and $(DIST_DIR) directories
# Safe: Can be run multiple times (idempotent)
clean:
	@rm -rf $(BUILD_DIR) $(DIST_DIR)
```

### D. Function Comments

**Document reusable functions.**

```makefile
# make/functions.mk - Reusable functions

# Compile C source file to object file
# Usage: $(call compile-c,source_file,object_file,extra_flags)
# Args:
#   source_file: Path to .c source file
#   object_file: Path to output .o file
#   extra_flags: Additional compiler flags (optional)
define compile-c
	$(if $(filter 1,$(V)),@echo "  CC    $(notdir $(2))",)
	@$(CC) $(CFLAGS) $(3) -c $(1) -o $(2)
endef
```

### E. Module Header Comments

**Each module file MUST have a header comment.**

```makefile
# make/build.mk - Build rules and compilation
#
# This module contains all build-related targets and rules:
# - Compilation rules for source files
# - Linking rules for executables
# - Build directory management
#
# Dependencies: config.mk, functions.mk

# Main build target
build: $(BUILD_DIR)/main
	@echo "Build complete"

# ... rest of module ...
```

### F. Inline Comments

**Use inline comments sparingly for non-obvious logic.**

```makefile
# ✅ CORRECT - Helpful inline comments
SOURCES := $(sort $(wildcard $(SRC_DIR)/*.c))  # Sort for deterministic builds
OBJECTS := $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# ❌ WRONG - Obvious inline comments
SOURCES := $(wildcard $(SRC_DIR)/*.c)  # Get all .c files  # Too obvious
```

### G. Comment Checklist

**All Makefiles MUST:**

- [ ] Have header comment in main Makefile explaining purpose
- [ ] Have header comment in each module file
- [ ] Comment complex targets explaining purpose
- [ ] Comment reusable functions with usage examples
- [ ] Use section headers for logical groupings
- [ ] Keep comments simple and concise
- [ ] Remove commented-out code
- [ ] Update comments when code changes

---

## 8. Debug and Verbose Modes (MANDATORY)

### A. Verbose Mode (V variable)

**CRITICAL: All Makefiles MUST support verbose mode for troubleshooting.**

#### Implementation Pattern

```makefile
# Verbose mode control
V ?= 0
ifeq ($(V),1)
  Q :=
  VERBOSE := 1
else
  Q := @
  VERBOSE := 0
endif

# Usage in recipes
build:
	$(Q)echo "Building project..."
	$(Q)$(CC) $(CFLAGS) -o output input.c

# Show command when verbose
ifeq ($(V),1)
  run-cc = $(CC) $(1)
else
  run-cc = @echo "  CC    $(notdir $(2))"; $(CC) $(1)
endif
```

#### Helper Function for Commands

```makefile
# make/functions.mk - Reusable functions

# Run command with optional verbose output
# Usage: $(call run-cmd,command,description)
run-cmd = $(if $(filter 1,$(V)),\
	@echo "$(2)"; $(1),\
	@echo "$(2)"; $(1))

# Run command silently unless verbose
# Usage: $(call run-silent,command)
run-silent = $(if $(filter 1,$(V)),$(1),@$(1))
```

#### Example Usage

```makefile
# make/build.mk

build: $(BUILD_DIR)/main
	$(call run-cmd,echo "Build complete","Building")

$(BUILD_DIR)/main: $(SRC_DIR)/main.c
	$(call run-silent,$(CC) $(CFLAGS) -o $@ $<)

# With V=1, shows:
# Building
#   CC    main.c
# Build complete

# With V=0 (default), shows:
# Building
# Build complete
```

### B. Debug Mode (DEBUG variable)

**CRITICAL: All Makefiles MUST support debug mode for detailed information.**

#### Implementation Pattern

```makefile
# Debug mode control
DEBUG ?= 0

# Debug output function
# Usage: $(call debug,message)
ifeq ($(DEBUG),1)
  debug = $(warning [DEBUG] $(1))
else
  debug :=
endif

# Show variable values in debug mode
show-var = $(if $(filter 1,$(DEBUG)),\
	$(warning [DEBUG] $(1) = $($(1))))

# Usage
build:
	$(call debug,Starting build process)
	$(call show-var,CC)
	$(call show-var,CFLAGS)
	$(call show-var,BUILD_DIR)
	@echo "Building..."
```

#### Comprehensive Debug Example

```makefile
# make/debug.mk - Debug utilities

DEBUG ?= 0

# Debug print function
ifeq ($(DEBUG),1)
  define debug-print
    $(warning [DEBUG] $(1))
  endef
  
  define debug-var
    $(warning [DEBUG] $(1) = $($(1)))
  endef
  
  define debug-target
    $(warning [DEBUG] Target: $(1))
    $(warning [DEBUG]   Prerequisites: $(2))
    $(warning [DEBUG]   Recipe: $(3))
  endef
else
  debug-print :=
  debug-var :=
  debug-target :=
endif

# Usage in other modules
include make/debug.mk

build: $(BUILD_DIR)/main
	$(call debug-print,Executing build target)
	$(call debug-var,CC)
	$(call debug-var,CFLAGS)
	@echo "Building..."
```

### C. Combined Verbose and Debug

```makefile
# make/config.mk

V ?= 0
DEBUG ?= 0

# Quiet prefix for commands
Q := $(if $(filter 1,$(V)),,@)

# Debug output
ifeq ($(DEBUG),1)
  define debug
    $(if $(filter 1,$(V)),@echo "[DEBUG] $(1)",$(warning [DEBUG] $(1)))
  endef
else
  debug :=
endif

# Verbose command execution
ifeq ($(V),1)
  run = $(1)
else
  run = @$(1)
endif

# Usage
build:
	$(call debug,Configuration loaded)
	$(call debug,CC=$(CC))
	$(call debug,CFLAGS=$(CFLAGS))
	$(run)$(CC) $(CFLAGS) -o output input.c
```

---

## 9. Help System (MANDATORY)

### A. Built-in Help Target

**CRITICAL: All Makefiles MUST include a `help` target that documents available targets.**

#### Basic Help Implementation

```makefile
# make/help.mk - Help system

.PHONY: help
help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

# Example targets with help
build: ## Build the project
	@echo "Building..."

test: ## Run tests
	@echo "Testing..."

clean: ## Remove build artifacts
	@echo "Cleaning..."
```

#### Advanced Help with Categories

```makefile
# make/help.mk - Advanced help system

.PHONY: help
help: ## Show this help message
	@echo "Usage: make [target] [V=1] [DEBUG=1]"
	@echo ""
	@echo "Options:"
	@echo "  V=1       Enable verbose mode"
	@echo "  DEBUG=1   Enable debug mode"
	@echo ""
	@echo "Available targets:"
	@echo ""
	@echo "Build targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		grep -E '## Build' | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Test targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		grep -E '## Test' | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Utility targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		grep -E '## Utility' | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# Example targets
build: ## Build the project ## Build
	@echo "Building..."

test: ## Run all tests ## Test
	@echo "Testing..."

clean: ## Remove build artifacts ## Utility
	@echo "Cleaning..."

help-vars: ## Show important variables ## Utility
	@echo "Important variables:"
	@echo "  CC=$(CC)"
	@echo "  CFLAGS=$(CFLAGS)"
	@echo "  BUILD_DIR=$(BUILD_DIR)"
```

#### Help with Default Target

```makefile
# Set help as default
.DEFAULT_GOAL := help

# Help target
.PHONY: help
help: ## Show this help message
	@echo "Project: $(PROJECT_NAME) v$(VERSION)"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Environment variables:"
	@echo "  V=1       Verbose output"
	@echo "  DEBUG=1   Debug information"
```

### B. Self-Documenting Variables

```makefile
# make/help.mk - Variable help

help-vars: ## Show all configuration variables ## Utility
	@echo "Configuration Variables:"
	@echo ""
	@echo "Project:"
	@echo "  PROJECT_NAME = $(PROJECT_NAME)"
	@echo "  VERSION       = $(VERSION)"
	@echo ""
	@echo "Directories:"
	@echo "  BUILD_DIR    = $(BUILD_DIR)"
	@echo "  SRC_DIR      = $(SRC_DIR)"
	@echo "  TEST_DIR     = $(TEST_DIR)"
	@echo ""
	@echo "Tools:"
	@echo "  CC           = $(CC)"
	@echo "  PYTHON       = $(PYTHON)"
	@echo ""
	@echo "Flags:"
	@echo "  CFLAGS       = $(CFLAGS)"
	@if [ "$(DEBUG)" = "1" ]; then \
		echo "  DEBUG_CFLAGS = $(DEBUG_CFLAGS)"; \
	fi
```

---

## 10. Minimal External Tools (MANDATORY)

### A. POSIX Compliance

**CRITICAL: Use only POSIX-compliant tools and shell features.**

#### ✅ CORRECT - POSIX-compliant

```makefile
# Use POSIX shell features only
SHELL := /bin/sh

# POSIX-compliant commands
build:
	@mkdir -p $(BUILD_DIR)
	@cp -f src/*.c $(BUILD_DIR)/
	@find $(BUILD_DIR) -name "*.c" -exec $(CC) {} \;
```

#### ❌ WRONG - Bash-specific features

```makefile
# ❌ Using bash-specific features
SHELL := /bin/bash

build:
	@mkdir -p $(BUILD_DIR)
	@cp -f src/*.c $(BUILD_DIR)/
	@for file in $(BUILD_DIR)/*.c; do \
		echo "Processing $$file"; \
	done
	# Bash arrays, process substitution, etc. are not POSIX
```

### B. Avoid External Dependencies

**CRITICAL: Minimize reliance on external tools beyond standard POSIX utilities.**

#### ✅ CORRECT - Minimal dependencies

```makefile
# Standard tools only
SHELL := /bin/sh
CC := gcc
PYTHON := python3

# Built-in Make functions
SOURCES := $(wildcard src/*.c)
OBJECTS := $(SOURCES:src/%.c=build/%.o)

build: $(OBJECTS)
	$(CC) $(CFLAGS) -o output $(OBJECTS)
```

#### ❌ WRONG - Heavy external dependencies

```makefile
# ❌ Requiring many external tools
build:
	@which jq > /dev/null || (echo "jq required" && exit 1)
	@which yq > /dev/null || (echo "yq required" && exit 1)
	@which docker > /dev/null || (echo "docker required" && exit 1)
	@jq . config.json
	@yq eval . config.yaml
	@docker build .
	# Too many dependencies!
```

### C. Tool Detection Pattern

```makefile
# make/tools.mk - Tool detection and validation

# Check for required tools
REQUIRED_TOOLS := gcc python3
MISSING_TOOLS := $(foreach tool,$(REQUIRED_TOOLS),\
	$(if $(shell which $(tool)),,$(tool)))

ifneq ($(MISSING_TOOLS),)
  $(error Missing required tools: $(MISSING_TOOLS))
endif

# Optional tools with fallback
PYTHON := $(shell which python3 || which python || echo "python3")
CC := $(shell which gcc || which clang || echo "gcc")

# Verify tools work
check-tools: ## Verify all tools are available ## Utility
	@echo "Checking tools..."
	@for tool in $(REQUIRED_TOOLS); do \
		if command -v $$tool > /dev/null 2>&1; then \
			echo "  ✓ $$tool: $$(which $$tool)"; \
		else \
			echo "  ✗ $$tool: not found"; \
			exit 1; \
		fi \
	done
	@echo "All tools available"
```

---

## 11. Reusability Patterns (MANDATORY)

### A. Reusable Functions

```makefile
# make/functions.mk - Reusable functions

# Compile C source to object
# Usage: $(call compile-c,source,object,flags)
define compile-c
	$(if $(filter 1,$(V)),@echo "  CC    $(notdir $(2))",)
	@$(CC) $(CFLAGS) $(3) -c $(1) -o $(2)
endef

# Link objects to executable
# Usage: $(call link,output,objects,flags)
define link
	$(if $(filter 1,$(V)),@echo "  LD    $(notdir $(1))",)
	@$(CC) $(CFLAGS) $(3) -o $(1) $(2)
endef

# Create directory
# Usage: $(call mkdir,dir)
define mkdir
	@mkdir -p $(1)
endef

# Usage in recipes
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(call compile-c,$<,$@)

$(BUILD_DIR)/main: $(OBJECTS)
	$(call link,$@,$^)
```

### B. Template Patterns

```makefile
# make/templates.mk - Reusable templates

# Generic build rule template
# Usage: $(call define-build-rule,extension,compiler,flags)
define define-build-rule
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.$(1) | $(BUILD_DIR)
	$$(if $$(filter 1,$$(V)),@echo "  $(2)    $$(notdir $$@)",)
	@$(2) $(3) -c $$< -o $$@
endef

# Define rules for different file types
$(eval $(call define-build-rule,c,gcc,$(CFLAGS)))
$(eval $(call define-build-rule,cpp,g++,$(CXXFLAGS)))
```

### C. Configuration Patterns

```makefile
# make/config.mk - Reusable configuration pattern

# Load configuration from file if exists
-include config.local.mk
-include config.mk

# Set defaults
PROJECT_NAME ?= myproject
VERSION ?= 1.0.0
BUILD_DIR ?= build
SRC_DIR ?= src

# Allow override via environment
PROJECT_NAME := $(or $(PROJECT_NAME_ENV),$(PROJECT_NAME))
VERSION := $(or $(VERSION_ENV),$(VERSION))
```

---

## 12. Error Handling (MANDATORY)

### A. Fail-Fast Pattern

```makefile
# make/errors.mk - Error handling

# Check prerequisites
check-prereqs:
	@test -d $(SRC_DIR) || (echo "Error: $(SRC_DIR) not found" && exit 1)
	@test -n "$(CC)" || (echo "Error: CC not set" && exit 1)

# Validate configuration
validate-config:
	@test -n "$(PROJECT_NAME)" || (echo "Error: PROJECT_NAME not set" && exit 1)
	@test -n "$(VERSION)" || (echo "Error: VERSION not set" && exit 1)

# Build with validation
build: check-prereqs validate-config $(BUILD_DIR)/main
	@echo "Build successful"
```

### B. Error Messages

```makefile
# make/errors.mk - Clear error messages

# Error function
# Usage: $(call error-msg,message)
define error-msg
	$(error $(1))
endef

# Warning function
# Usage: $(call warn-msg,message)
define warn-msg
	$(warning $(1))
endef

# Usage
build:
	@test -d $(SRC_DIR) || \
		$(call error-msg,Source directory $(SRC_DIR) not found. Run 'make setup' first.)
	@test -f $(SRC_DIR)/main.c || \
		$(call error-msg,Main source file $(SRC_DIR)/main.c not found.)
	@echo "Building..."
```

### C. Dependency Validation

```makefile
# make/deps.mk - Dependency checking

# Check if tool exists
# Usage: $(call check-tool,tool,message)
define check-tool
	@command -v $(1) > /dev/null 2>&1 || \
		($(call error-msg,$(2)))
endef

# Check if file exists
# Usage: $(call check-file,file,message)
define check-file
	@test -f $(1) || \
		($(call error-msg,$(2)))
endef

# Check if directory exists
# Usage: $(call check-dir,dir,message)
define check-dir
	@test -d $(1) || \
		($(call error-msg,$(2)))
endef

# Usage
build: check-build-deps
	@echo "Building..."

check-build-deps:
	$(call check-tool,$(CC),Compiler $(CC) not found. Please install it.)
	$(call check-dir,$(SRC_DIR),Source directory $(SRC_DIR) not found.)
	$(call check-file,$(SRC_DIR)/main.c,Main source file not found.)
```

---

## 13. Complete Example Makefile

### A. Project Structure

```
project/
├── Makefile
├── make/
│   ├── config.mk
│   ├── functions.mk
│   ├── build.mk
│   ├── test.mk
│   ├── clean.mk
│   ├── help.mk
│   └── errors.mk
├── src/
│   └── main.c
└── tests/
    └── test_main.c
```

### B. Main Makefile

```makefile
# Makefile - Main entry point

# Include configuration first
include make/config.mk

# Include utilities
include make/functions.mk
include make/errors.mk

# Include modules
include make/build.mk
include make/test.mk
include make/clean.mk
include make/help.mk

# Default target
.DEFAULT_GOAL := help
```

### C. Configuration Module

```makefile
# make/config.mk - Configuration

# Project metadata
PROJECT_NAME ?= myproject
VERSION ?= 1.0.0

# Directories
BUILD_DIR ?= build
DIST_DIR ?= dist
SRC_DIR ?= src
TEST_DIR ?= tests

# Tools
CC ?= gcc
PYTHON ?= python3

# Build flags
CFLAGS ?= -Wall -Wextra -std=c11
DEBUG_CFLAGS ?= -g -O0 -DDEBUG
RELEASE_CFLAGS ?= -O2 -DNDEBUG

# Mode flags
V ?= 0
DEBUG ?= 0

# Apply debug flags if DEBUG=1
ifeq ($(DEBUG),1)
  CFLAGS += $(DEBUG_CFLAGS)
endif

# Export for sub-makes
export V DEBUG CC CFLAGS
```

### D. Functions Module

```makefile
# make/functions.mk - Reusable functions

# Quiet prefix
Q := $(if $(filter 1,$(V)),,@)

# Debug output
ifeq ($(DEBUG),1)
  define debug
    $(if $(filter 1,$(V)),@echo "[DEBUG] $(1)",$(warning [DEBUG] $(1)))
  endef
else
  debug :=
endif

# Compile C source
define compile-c
	$(Q)echo "  CC    $(notdir $(2))"
	$(Q)$(CC) $(CFLAGS) -c $(1) -o $(2)
endef

# Link objects
define link
	$(Q)echo "  LD    $(notdir $(1))"
	$(Q)$(CC) $(CFLAGS) -o $(1) $(2)
endef
```

### E. Build Module

```makefile
# make/build.mk - Build rules

# Sources and objects
SOURCES := $(wildcard $(SRC_DIR)/*.c)
OBJECTS := $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

.PHONY: build
build: $(BUILD_DIR)/main ## Build the project ## Build
	$(call debug,Build complete: $(BUILD_DIR)/main)

# Main executable
$(BUILD_DIR)/main: $(OBJECTS) | $(BUILD_DIR)
	$(call link,$@,$^)

# Object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(call compile-c,$<,$@)

# Create build directory
$(BUILD_DIR):
	$(Q)mkdir -p $(BUILD_DIR)
```

### F. Test Module

```makefile
# make/test.mk - Test rules

.PHONY: test
test: $(BUILD_DIR)/main ## Run all tests ## Test
	$(Q)echo "Running tests..."
	$(Q)$(BUILD_DIR)/main --test

.PHONY: test-verbose
test-verbose: $(BUILD_DIR)/main ## Run tests with verbose output ## Test
	$(Q)echo "Running tests (verbose)..."
	$(Q)$(BUILD_DIR)/main --test --verbose
```

### G. Clean Module

```makefile
# make/clean.mk - Cleanup rules

.PHONY: clean
clean: ## Remove build artifacts ## Utility
	$(Q)echo "Cleaning..."
	$(Q)rm -rf $(BUILD_DIR)
	$(Q)rm -rf $(DIST_DIR)

.PHONY: distclean
distclean: clean ## Remove all generated files ## Utility
	$(Q)echo "Deep cleaning..."
	$(Q)find . -name "*.o" -delete
	$(Q)find . -name "*.a" -delete
```

### H. Help Module

```makefile
# make/help.mk - Help system

.PHONY: help
help: ## Show this help message ## Utility
	@echo "Project: $(PROJECT_NAME) v$(VERSION)"
	@echo ""
	@echo "Usage: make [target] [V=1] [DEBUG=1]"
	@echo ""
	@echo "Options:"
	@echo "  V=1       Enable verbose mode (show commands)"
	@echo "  DEBUG=1   Enable debug mode (show debug info)"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make              Show this help"
	@echo "  make build        Build the project"
	@echo "  make V=1 build    Build with verbose output"
	@echo "  make DEBUG=1 build Build with debug information"
```

---

## 14. Best Practices Summary

### A. Structure Checklist

- [ ] Makefile split into logical modules in `make/` directory
- [ ] Main Makefile is small (< 50 lines) and contains only includes
- [ ] All modules in separate `make/` directory (not in root)
- [ ] Configuration in separate `make/config.mk`
- [ ] Reusable functions in `make/functions.mk`
- [ ] Each major feature in its own module file
- [ ] Help system implemented
- [ ] Default target set to `help`
- [ ] Simple, clean comments throughout
- [ ] Module header comments in each file

### B. Functionality Checklist

- [ ] Reproducible builds (SOURCE_DATE_EPOCH, deterministic ordering)
- [ ] Incremental builds with proper caching
- [ ] Progress indicators by default (when `V=0`)
- [ ] Verbose mode (`V=1`) implemented
- [ ] Debug mode (`DEBUG=1`) implemented
- [ ] Help target (`make help`) works
- [ ] Error handling for missing dependencies
- [ ] POSIX-compliant shell usage
- [ ] Minimal external tool dependencies

### C. Code Quality Checklist

- [ ] Clear variable names
- [ ] Consistent indentation (tabs for recipes)
- [ ] Comments for complex logic
- [ ] Reusable functions for common operations
- [ ] Proper dependency tracking
- [ ] Phony targets declared with `.PHONY`

### D. Verification Checklist

- [ ] `make -n` shows no syntax errors (parseability verified)
- [ ] `make help` displays help text
- [ ] `make V=1` shows verbose output
- [ ] `make DEBUG=1` shows debug information
- [ ] Default target executes successfully
- [ ] Caching works (run target twice, second skips)
- [ ] Progress indicators show by default
- [ ] Reproducible builds (hash comparison test passes)
- [ ] Works with POSIX shell (`SHELL=/bin/sh`)
- [ ] Main Makefile is small and readable (< 50 lines)

---

## 15. Quick Reference

### Command Cheat Sheet

```bash
# Basic usage
make                    # Show help (default)
make help              # Show help
make build             # Build project
make test              # Run tests
make clean             # Clean build artifacts

# Verbose mode
make V=1 build        # Build with verbose output

# Debug mode
make DEBUG=1 build    # Build with debug information

# Combined modes
make V=1 DEBUG=1 build # Verbose + debug

# Dry run (see what would execute)
make -n build          # Show commands without executing

# Parallel builds
make -j4 build        # Build with 4 parallel jobs

# Include specific Makefile
make -f Makefile.custom

# Override variables
make CC=clang build   # Use clang instead of gcc
```

### Common Patterns

```makefile
# Verbose mode
V ?= 0
Q := $(if $(filter 1,$(V)),,@)

# Debug mode
DEBUG ?= 0
ifeq ($(DEBUG),1)
  define debug
    $(warning [DEBUG] $(1))
  endef
else
  debug :=
endif

# Help target
help: ## Description
	@echo "Help text"

# Phony target
.PHONY: clean
clean:
	@rm -rf $(BUILD_DIR)

# Pattern rule
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Directory prerequisite
$(BUILD_DIR)/file: | $(BUILD_DIR)
	@touch $@

# Include guard
ifndef MODULE_MK
MODULE_MK := 1
# Module content
endif
```

---

## 16. Common Pitfalls to Avoid

### ❌ WRONG - Common Mistakes

```makefile
# ❌ Using bash-specific features
SHELL := /bin/bash
build:
	@array=(a b c); echo $${array[@]}

# ❌ Not using .PHONY for phony targets
clean:
	@rm -rf build

# ❌ Hardcoding paths
build:
	@/usr/bin/gcc -o output input.c

# ❌ No error handling
build:
	@cp src/*.c build/

# ❌ Monolithic Makefile (everything in one file)
# 500+ lines in single Makefile

# ❌ No help system
# Missing help target

# ❌ No verbose/debug modes
# Can't troubleshoot issues
```

### ✅ CORRECT - Best Practices

```makefile
# ✅ POSIX-compliant
SHELL := /bin/sh
build:
	@for file in src/*.c; do \
		echo "Processing $$file"; \
	done

# ✅ Phony targets declared
.PHONY: clean
clean:
	@rm -rf $(BUILD_DIR)

# ✅ Configurable tools
CC ?= gcc
build:
	@$(CC) -o output input.c

# ✅ Error handling
build:
	@test -d src || (echo "Error: src not found" && exit 1)
	@cp src/*.c build/

# ✅ Modular structure
include make/config.mk
include make/build.mk

# ✅ Help system
help: ## Show help
	@echo "Available targets:"

# ✅ Verbose and debug modes
V ?= 0
DEBUG ?= 0
```

---

## 17. Resources

### GNU Make Documentation
- [GNU Make Manual](https://www.gnu.org/software/make/manual/)
- [Makefile Tutorial](https://makefiletutorial.com/)

### POSIX Standards
- [POSIX Shell Command Language](https://pubs.opengroup.org/onlinepubs/9699919799/utilities/V3_chap02.html)

### Best Practices
- [Managing Projects with GNU Make](https://www.oreilly.com/library/view/managing-projects-with/0596006101/)

---

## 18. Summary

**CRITICAL Requirements for All Makefiles:**

1. **Modular Structure**: Split into logical modules in `make/` directory, keep main Makefile small and readable
2. **Reproducible Builds**: Deterministic outputs, SOURCE_DATE_EPOCH, version pinning, reproducible flags
3. **Incremental Builds**: Proper dependency tracking, caching to prevent duplicate work
4. **Progress Indicators**: Progress logs or progress bars by default (unless verbose mode)
5. **Clean Comments**: Simple, clean, helpful comments throughout
6. **Verbose Mode**: Support `V=1` for command visibility
7. **Debug Mode**: Support `DEBUG=1` for troubleshooting
8. **Help System**: Built-in `help` target documenting all targets
9. **Minimal Dependencies**: Use only POSIX tools, avoid external dependencies
10. **Reusability**: Use functions and templates for common patterns
11. **Error Handling**: Validate prerequisites, clear error messages
12. **Verification**: Agent MUST test Makefile before delivery
13. **Post-Modification Check**: Agent MUST verify parseability after ANY modification

**Agent Verification Protocol:**
- Run `make -n` to check syntax and parseability
- **MANDATORY**: After ANY modification, verify parseability with `make -n`
- Test `make help` displays correctly
- Verify `make V=1` shows verbose output
- Verify `make DEBUG=1` shows debug information
- Test default target executes successfully
- Verify caching works (run target twice, second should skip)
- Verify progress indicators show by default
- Only present working Makefiles to the user

**Remember**: Small, minimalistic, modular (in separate `make/` directory), reusable, reproducible, with built-in help, debugging support, incremental caching, progress indicators, and clean comments. Always verify parseability after modifications. Keep it simple, keep it POSIX, keep it working.
