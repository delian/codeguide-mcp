# Python Development Guidelines
Mandatory coding standards and development practices for Python development. Type-safe, documented, test-covered. Python 3.13+, uv, pytest, ruff, Dynaconf, pydoc, bandit, safety.

---

**Agent Profile**: The Python Expert
**Role**: Senior Python Engineer & Quality & Tooling Specialist
**Objective**: Generate production-ready, type-safe, documented, and test-covered Python code.
**Tools**: Python 3.13+, uv, pytest, ruff, Dynaconf, pydoc, bandit, safety

---

## 1. Core Philosophies: PYTHON-FIRST

All code contributions **MUST** adhere to these guidelines. Non-compliant code will be rejected during review.

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Python 3.13+ Features**: Leverage the JIT compiler, free-threaded build (if CPU-bound), and improved error messages.

- **P**ackage management: UV only; all dependencies and commands via `uv run`.
- **Y**ield and comprehensions: Prefer list/set/dict/generator comprehensions for performance and clarity.
- **T**ype hints: Strict typing on all functions and classes; no untyped public APIs.
- **H**ints and docs: Complete docstrings (Google style); generate API docs with pydoc.
- **O**utward config: Dynaconf for all configuration; no hardcoded values.
- **N**on-negotiable tests: 100% coverage with pytest; all tests must pass; Ruff checks must pass.
- **S**ecurity Scans: Mandatory bandit and safety scans for every delivery.
**Verified Code**: Agent-generated code MUST be syntax-checked, tested, and pass Ruff before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

When an AI agent generates Python code, the following verification steps are **MANDATORY**:

### A. Code Verification Protocol
1. **ALWAYS verify Python syntax** before presenting code to the user
2. **Parse the code** to ensure it's valid Python (no syntax errors)
3. **Run tests with `uv run pytest`** to verify functionality
4. **Fix any errors iteratively** until all tests pass
5. **Verify dependencies** are properly specified and installable via `uv`

### B. Verification Checklist
- [ ] Code has valid Python syntax (can be parsed by `ast.parse()`)
- [ ] All imports are available via `uv add package-name`
- [ ] Tests exist for all new functions and classes
- [ ] Tests pass when run with `uv run pytest`
- [ ] Code follows type hint requirements (passes `uv run ruff check`)
- [ ] All functions have complete docstrings
- [ ] No hardcoded configuration values
- [ ] Code formatted with `uv run ruff format`
- [ ] Security scan passes: `uv run bandit -r .`
- [ ] Dependency safety check passes: `uv run safety check`

### C. Error Correction Process
If the generated code fails verification:
1. **Analyze the error message** (syntax error, import error, test failure, etc.)
2. **Identify the root cause** (missing import, wrong indentation, logic error, etc.)
3. **Fix the issue** in the generated code
4. **Re-verify** by running the checks again
5. **Repeat until successful** - iterate as many times as needed
6. **Only present working, tested code** to the user

### D. Example Verification Workflow
```bash
# Agent must simulate/verify this workflow

# 1. Check syntax validity
python -m py_compile module.py

# 2. Verify with ruff
uv run ruff check module.py

# 3. Format code
uv run ruff format module.py

# 4. Run tests
uv run pytest tests/test_module.py

# If any step fails:
# - Read the error output
# - Fix the code
# - Try again
# - Repeat until success
```

### E. Test Requirements for Generated Code
**Every function/class MUST have corresponding tests:**

```python
# Generated code: my_module.py
def calculate_sum(numbers: list[float]) -> float:
    """Calculate sum of numbers.
    
    Args:
        numbers: List of numbers to sum.
        
    Returns:
        Sum of all numbers.
        
    Example:
        >>> calculate_sum([1, 2, 3])
        6.0
    """
    return sum(numbers)


# Generated tests: tests/test_my_module.py
import pytest
from my_module import calculate_sum


def test_calculate_sum_basic() -> None:
    """Test basic sum calculation."""
    result = calculate_sum([1, 2, 3])
    assert result == 6.0


def test_calculate_sum_empty() -> None:
    """Test sum of empty list."""
    result = calculate_sum([])
    assert result == 0.0


def test_calculate_sum_negative() -> None:
    """Test sum with negative numbers."""
    result = calculate_sum([-1, 2, -3])
    assert result == -2.0
```

**Run tests to verify:**
```bash
uv run pytest tests/test_my_module.py -v
```

### F. UV Prefix Requirement
**ALL command executions MUST use `uv run` prefix:**

✅ **CORRECT:**
```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run main.py
uv run mypy .
```

❌ **WRONG:**
```bash
pytest                    # Missing uv run
ruff check .             # Missing uv run
python main.py           # Missing uv run
pip install package      # Use uv add instead
```

**CRITICAL**: Never provide code to the user that:
- Has syntax errors
- Fails tests
- Has missing dependencies
- Doesn't follow the style guide
- **Fixes bugs without regression tests**
- **Skips TDD cycle (test-first)**

Always verify first, fix issues, then present the working solution.

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

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

### Example TDD Workflow

```python
# Step 1: RED - Write failing test first
def test_email_validator_valid():
    """Test email validation with valid email."""
    assert validate_email("user@example.com") == True

def test_email_validator_invalid():
    """Test email validation with invalid email."""
    assert validate_email("invalid.email") == False

# Run: uv run pytest
# ❌ FAILS - validate_email doesn't exist yet

# Step 2: GREEN - Write minimal implementation
import re

def validate_email(email: str) -> bool:
    """Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

# Run: uv run pytest
# ✅ PASSES - tests pass

# Step 3: REFACTOR - Improve if needed
# (In this case, code is already clean)
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

```python
# Bug Report #789: parse_date fails with ISO format dates containing timezone

# Step 1-2: Write test that reproduces the bug
def test_parse_date_with_timezone_bug_789():
    """Test parsing ISO date with timezone - Bug #789.
    
    Bug: parse_date("2024-01-15T10:30:00+00:00") raised ValueError
    Discovered: 2026-01-18
    This test prevents regression.
    """
    from datetime import datetime, timezone
    
    result = parse_date("2024-01-15T10:30:00+00:00")
    expected = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    
    assert result == expected

# Run: uv run pytest
# ❌ FAILS - reproduces the bug ✓

# Step 3: Fix the bug
from datetime import datetime

def parse_date(date_str: str) -> datetime:
    """Parse ISO format date string.
    
    Args:
        date_str: ISO format date string (with or without timezone)
        
    Returns:
        Parsed datetime object
        
    Raises:
        ValueError: If date string is invalid
    """
    # FIX: Use fromisoformat instead of strptime to handle timezone
    try:
        return datetime.fromisoformat(date_str)
    except ValueError as e:
        raise ValueError(f"Invalid date format: {date_str}") from e

# Run: uv run pytest
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

## 3. Package Management: UV Only
### Requirements
- **ALWAYS** use `uv` for all package management operations
- **NEVER** use `pip`, `poetry`, `pipenv`, or other package managers directly
- All dependencies **MUST** be managed through `uv`
### Virtual Environment Management

```bash
# Create virtual environment
uv venv .venv

# Activate virtual environment
source .venv/bin/activate # Linux/macOS

# .venv\Scripts\activate # Windows PowerShell

# Install dependencies
uv add package-name

# Install from requirements
uv add -r requirements.txt

# Add development dependencies
uv add --dev pytest ruff mypy

# Sync environment (install all dependencies)
uv sync
```

### 3.1. Workspaces (MANDATORY for Large Projects)

**Use uv workspaces for multi-package repositories or hexagonal architecture:**

```toml
# pyproject.toml (root)
[tool.uv.workspace]
members = ["packages/*"]

[tool.uv.sources]
my-core = { workspace = true }
```

- Enables local development between packages without manual installation
- Shared virtual environment for the entire workspace
- Consistent dependency versions across all members

### Adding Dependencies

```bash
# Add a new package
uv add numpy

# Add with version constraint
uv add "numpy>=1.24.0,<2.0.0"

# Add development-only package
uv add --dev pytest

# Update requirements.txt
uv pip freeze > requirements.txt
```

### Project Initialization
```bash
# Initialize new project
uv venv
source .venv/bin/activate

uv add dynaconf ruff
```

---

## 4. Configuration Management: Dynaconf

### Requirements
- **ALWAYS** use `dynaconf` for configuration management
- **NEVER** hardcode configuration values in source code
- All configuration **MUST** be externalized to TOML files
### File Structure
```
project/
├── config/
│ ├── __init__.py
│ └── defaults.toml
├── config.toml
└── .secrets.toml # Git-ignored, optional
```

### Configuration Files
**`config/defaults.toml`** - Default values for all settings:

```toml
# Default configuration values
# These serve as fallbacks if not overridden in config.toml

[default]
# Application defaults
app_name = "OFDM Optimizer"
debug = false
log_level = "INFO"

# Processing parameters
pulse_duration = 50e-6
bandwidth = 5000000.0
num_channels = 100
center_freq = 2000000
sample_rate = 20000000

# Algorithm parameters
max_monte_carlo_runs = 10000
tapering_length_percent = 5
parp_limit_db = 9
white_noise_boot = 100

# Feature flags
self_orthogonality_filter = false
white_noise_check_filter = false
parp_filter = true

# File paths
temp_state_file = "temp_state.json"
output_file = "results.json"

[development]
debug = true
log_level = "DEBUG"
num_channels = 10 # Smaller for faster dev testing

[production]
debug = false
log_level = "WARNING"

```

**`config.toml`** - User/deployment-specific overrides:

```toml
# User configuration - overrides defaults.toml
pulse_duration = 100e-6
num_channels = 300
white_noise_boot = 300
output_file = "res-300-100-20mhz.json"

# Environment-specific settings
[development]
num_channels = 50
max_monte_carlo_runs = 1000

[production]
num_channels = 1000
max_monte_carlo_runs = 100000
```

### Loading Configuration

**`config/__init__.py`**:

```python
"""Configuration module using Dynaconf."""

from pathlib import Path
from dynaconf import Dynaconf, Validator

# Define base directory
BASE_DIR = Path(__file__).parent.parent

# Initialize Dynaconf with validation
Settings = Dynaconf(
# Load order: defaults.toml -> config.toml -> env vars -> .secrets.toml
settings_files=[
str(BASE_DIR / "config" / "defaults.toml"),
str(BASE_DIR / "config.toml"),
],

# Enable environment-specific configs [development], [production]
environments=True,
# Allow environment variable overrides (e.g., MYAPP_SAMPLE_RATE=30000000)
envvar_prefix="MYAPP",
# Load .secrets.toml if present (git-ignored sensitive data)
secrets=str(BASE_DIR / ".secrets.toml"),
# Enable Jinja2 templating in TOML values
load_dotenv=True,
# Validators ensure critical config is present and valid
validators=[
Validator("pulse_duration", must_exist=True, gt=0),
Validator("bandwidth", must_exist=True, gt=0),
Validator("num_channels", must_exist=True, gte=1),
Validator("sample_rate", must_exist=True, gt=0),
Validator("center_freq", must_exist=True, gt=0),
],
)

# Validate on import
Settings.validators.validate()

# Export normalized configuration dictionary
CONF = {k.lower(): v for k, v in Settings.to_dict().items()}

```



### Using Configuration in Code

```python
from config import Settings, CONF
# Access via Settings object (preferred)

def process_signal() -> None:

"""Process signal using configured parameters."""

duration: float = Settings.pulse_duration
channels: int = Settings.num_channels

# ..

# Access via CONF dictionary

def alternative_access() -> None:

"""Alternative configuration access pattern."""
sample_rate: int = CONF['sample_rate']
bandwidth: float = CONF['bandwidth']

# ..

# Environment-specific access

from dynaconf import settings

# Automatically uses correct environment

if settings.current_env == "development":
    # Development-specific logic
    pass
```

### Configuration Best Practices

1. **Never hardcode** - All magic numbers go in config
2. **Type hints** - Document expected types in validators
3. **Validation** - Use Dynaconf validators for critical parameters
4. **Environment separation** - Use `[development]`, `[production]` sections
5. **Secrets** - Never commit `.secrets.toml` to version control
6. **Documentation** - Comment each configuration parameter

### Accessing Environment-Specific Config

```bash
# Run in development mode

export ENV_FOR_DYNACONF=development
python main.py

# Run in production mode
export ENV_FOR_DYNACONF=production
python main.py

# Override single value via environment variable
export MYAPP_NUM_CHANNELS=500
python main.py
```

---
## 5. Documentation: Comprehensive PyDoc

### Requirements

- **EVERY** function **MUST** have a complete docstring
- **EVERY** class **MUST** have a complete docstring
- **EVERY** module **MUST** have a module-level docstring
- Follow **Google Style** docstring format

### Module-Level Docstring

```python

"""
OFDM Signal Generation and Optimization Module.

This module provides GPU-accelerated tools for generating OFDM signals with

chirp modulation and optimizing phase codes to minimize cross-correlation

between multiple channels.

Key Features:
- GPU acceleration via CuPy
- LRU caching for performance
- Configurable PARP filtering
- Checkpointing for long-running optimizations

Example:
Generate an OFDM codebook:

>>> from generate_codes import generate_phase_codebook
>>> from config import CONF
>>> codebook = generate_phase_codebook(
... num_channels=100,
... num_subcarriers=250,
... subcarrier_spacing=20e3,
... num_phases=172,
... center_freq=CONF['center_freq'],
... chirp_bw=10e3,
... chirp_duration=CONF['pulse_duration'],
... num_samples=CONF['num_samples']
... )

Dependencies:

- cupy: GPU array operations
- numpy: CPU array operations
- scipy: Optimization algorithms
- dynaconf: Configuration management

Author: Your Name

Created: 2024-01-01
License: MIT
"""

```

### Function Docstring Template

```python
from typing import Tuple, Optional
import cupy as cp

def generate_chirp_signal(
    basefreq: float,
    chirp_bw: float,
    chirp_duration: float,
    num_samples: int,
    phase: float = 0.0,
    sample_rate: float = 20e6
) -> Tuple[cp.ndarray, cp.ndarray]:

"""
Generate a chirp signal modulated to a specific carrier frequency.
Creates a linear frequency-modulated chirp and modulates it to the
specified carrier frequency with an optional phase offset.

Args:
basefreq: Carrier frequency in Hz. Must be positive.
chirp_bw: Chirp bandwidth in Hz. Determines frequency sweep range.
chirp_duration: Duration of chirp pulse in seconds. Must be positive.
num_samples: Number of time-domain samples. Must be >= 1.
phase: Initial phase offset in radians. Defaults to 0.0.
sample_rate: Sampling rate in Hz. Defaults to 20 MHz.

Returns:
Tuple containing:
- signal (cp.ndarray): Complex baseband chirp signal, shape (num_samples,)
- time (cp.ndarray): Time axis in seconds, shape (num_samples,)

Raises:
ValueError: If basefreq <= 0, chirp_bw <= 0, or chirp_duration <= 0.
AssertionError: If num_samples < 1.

Example:
Generate a 2 MHz chirp with 10 kHz bandwidth:
>>> sig, t = generate_chirp_signal(
... basefreq=2e6,
... chirp_bw=10e3,
... chirp_duration=50e-6,
... num_samples=1000,
... phase=np.pi/4
... )

>>> print(sig.shape, t.shape)
(1000,) (1000,)

Notes:
- Uses GPU acceleration via CuPy for performance
- Result is cached via @lru_cache for repeated calls
- Chirp uses linear frequency modulation
- Time axis starts at 0 and ends at chirp_duration

See Also:
generate_base_chirp: Lower-level chirp generation
generate_chirp_ofdm_signal: Multi-carrier version

Performance:
O(num_samples) time complexity, O(num_samples) space complexity.

GPU execution typically < 1ms for num_samples=1000.
"""

if basefreq <= 0:
	raise ValueError(f"basefreq must be positive, got {basefreq}")

if chirp_bw <= 0:
	raise ValueError(f"chirp_bw must be positive, got {chirp_bw}")

if chirp_duration <= 0:
	raise ValueError(f"chirp_duration must be positive, got {chirp_duration}")

assert num_samples >= 1, "num_samples must be at least 1"

# Implementation here
..

return signal, time

```


### Class Docstring Template


```python

from typing import List, Dict, Any

class OFDMCodebookGenerator:

"""
Generator for orthogonal OFDM phase code codebooks.

This class manages the iterative process of generating multiple OFDM
signals with minimal cross-correlation, suitable for CDMA or multi-user communication systems.

Attributes:

num_channels: Number of codes to generate.
num_subcarriers: OFDM subcarriers per signal.
config: Configuration dictionary from Dynaconf.
codebook: List of (permutation_index, cost) tuples.
cumulative_signal: Running sum of all generated signals.

Example:

>>> from config import CONF
>>> generator = OFDMCodebookGenerator(
... num_channels=100,
... num_subcarriers=250,
... config=CONF
... )

>>> codebook = generator.generate()
>>> generator.save('output.json')

Note:
Supports checkpointing for resumable generation of large codebooks.

"""

def __init__(
    self,
    num_channels: int,
    num_subcarriers: int,
    config: Dict[str, Any]
) -> None:

"""

Initialize the codebook generator.

Args:
num_channels: Number of orthogonal codes to generate. Must be >= 1.
num_subcarriers: Number of OFDM subcarriers per signal. Must be >= 1.
config: Configuration dictionary containing parameters like
pulse_duration, chirp_bw, sample_rate, etc.

Raises:
ValueError: If num_channels < 1 or num_subcarriers < 1.
KeyError: If required config keys are missing.

"""

if num_channels < 1:
	raise ValueError(f"num_channels must be >= 1, got {num_channels}")

if num_subcarriers < 1:
	raise ValueError(f"num_subcarriers must be >= 1, got {num_subcarriers}")

self.num_channels = num_channels
self.num_subcarriers = num_subcarriers
self.config = config
self.codebook: List[Tuple[int, float]] = []
self.cumulative_signal: Optional[cp.ndarray] = None

def generate(self) -> List[Tuple[int, float]]:

"""
Generate the complete codebook.

Returns:
List of (permutation_index, cost) tuples for each channel.

Raises:
RuntimeError: If generation fails after maximum retries.

"""

# Implementation

..

```


### Docstring Requirements Checklist
- [ ] One-line summary (imperative mood: "Generate", not "Generates")
- [ ] Detailed description (2-3 sentences minimum)
- [ ] All parameters documented in `Args:` section
- [ ] Return value(s) documented in `Returns:` section
- [ ] Exceptions documented in `Raises:` section
- [ ] At least one usage example in `Example:` section
- [ ] Important notes in `Notes:` section (if applicable)
- [ ] Related functions in `See Also:` section (if applicable)
- [ ] Performance characteristics documented (if relevant)

### Generating Documentation (MANDATORY)

**ALL projects MUST provide generated API documentation accessible via pydoc:**

#### A. Viewing Documentation with pydoc

**Use Python's built-in pydoc to view documentation:**

```bash
# View module documentation
uv run python -m pydoc module_name

# View specific function/class documentation
uv run python -m pydoc module_name.ClassName
uv run python -m pydoc module_name.function_name

# Start HTTP server for browsing documentation
uv run python -m pydoc -b
# Opens browser to http://localhost:port/ for interactive browsing

# Generate HTML documentation for a module
uv run python -m pydoc -w module_name
# Creates module_name.html in current directory

# Generate HTML for entire package
uv run python -m pydoc -w package_name
```

#### B. Documentation Generation Examples

**View module documentation in terminal:**
```bash
# View main module documentation
uv run python -m pydoc generate_codes

# View specific class documentation
uv run python -m pydoc generate_codes.OFDMCodebookGenerator

# View function documentation
uv run python -m pydoc generate_codes.generate_chirp_signal
```

**Generate HTML documentation:**
```bash
# Generate HTML docs for all modules
uv run python -m pydoc -w generate_codes
uv run python -m pydoc -w generate_codes.chirp
uv run python -m pydoc -w generate_codes.ofdm
uv run python -m pydoc -w generate_codes.optimization

# Generate HTML for config module
uv run python -m pydoc -w config

# Creates HTML files: generate_codes.html, config.html, etc.
```

**Interactive documentation browser:**
```bash
# Start pydoc server (automatically opens browser)
uv run python -m pydoc -b

# Or specify port manually
uv run python -m pydoc -p 8080
# Then visit http://localhost:8080/ in your browser
```

#### C. Documentation Structure Requirements

**Generated documentation MUST include:**

1. **Module Overview**:
   - Module-level docstring appears at the top
   - Lists all public classes and functions
   - Shows module dependencies

2. **API Reference**:
   - All public functions with complete signatures
   - All public classes with methods
   - Type hints visible in signatures
   - Parameter and return value documentation

3. **Code Examples**:
   - Examples from docstrings are included
   - Shows usage patterns
   - Demonstrates common use cases

4. **Cross-References**:
   - Links to related functions (from `See Also:` sections)
   - Links to parent modules and submodules
   - Links to parameter types and return types

#### D. Creating a Documentation Directory

**For projects requiring persistent documentation:**

```bash
# Create docs directory
mkdir -p docs/api

# Generate all module documentation as HTML
for module in generate_codes generate_codes.chirp generate_codes.ofdm config; do
    uv run python -m pydoc -w $module
    mv ${module}.html docs/api/
done

# Create index.html for navigation
cat > docs/api/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Project API Documentation</title>
</head>
<body>
    <h1>API Documentation</h1>
    <ul>
        <li><a href="generate_codes.html">generate_codes (Main Module)</a></li>
        <li><a href="generate_codes.chirp.html">generate_codes.chirp</a></li>
        <li><a href="generate_codes.ofdm.html">generate_codes.ofdm</a></li>
        <li><a href="config.html">config (Configuration)</a></li>
    </ul>
</body>
</html>
EOF
```

#### E. Documentation Verification

**Before committing, verify documentation is complete:**

```bash
# 1. Check that pydoc can parse all modules
uv run python -m pydoc generate_codes > /dev/null
echo $?  # Should return 0

# 2. Generate HTML and check for completeness
uv run python -m pydoc -w generate_codes

# 3. View in browser to ensure formatting is correct
uv run python -m pydoc -b

# 4. Verify all public APIs are documented
uv run python -c "
import generate_codes
import inspect

# Get all public members
members = [m for m in dir(generate_codes) if not m.startswith('_')]

# Check each member has a docstring
for member in members:
    obj = getattr(generate_codes, member)
    if callable(obj) and not obj.__doc__:
        print(f'Missing docstring: {member}')
"
```

#### F. Makefile Integration

**Add documentation generation to Makefile:**

```makefile
.PHONY: docs
docs:
	@echo "Generating API documentation..."
	@mkdir -p docs/api
	@uv run python -m pydoc -w generate_codes
	@uv run python -m pydoc -w generate_codes.chirp
	@uv run python -m pydoc -w generate_codes.ofdm
	@uv run python -m pydoc -w config
	@mv *.html docs/api/
	@echo "Documentation generated in docs/api/"

.PHONY: docs-serve
docs-serve:
	@echo "Starting documentation server..."
	@uv run python -m pydoc -b

.PHONY: docs-check
docs-check:
	@echo "Verifying documentation completeness..."
	@uv run python -m pydoc generate_codes > /dev/null && echo "✓ Documentation is valid"
```

**Usage:**
```bash
# Generate HTML documentation
make docs

# Start interactive documentation browser
make docs-serve

# Verify documentation validity
make docs-check
```

#### G. CI/CD Integration

**Add documentation check to GitHub Actions:**

```yaml
# .github/workflows/ci.yml
- name: Verify documentation
  run: |
    uv run python -m pydoc generate_codes > /dev/null
    uv run python -m pydoc config > /dev/null

- name: Generate documentation
  run: |
    mkdir -p docs/api
    uv run python -m pydoc -w generate_codes
    uv run python -m pydoc -w config
    mv *.html docs/api/

- name: Upload documentation artifacts
  uses: actions/upload-artifact@v3
  with:
    name: api-documentation
    path: docs/api/
```

#### H. Documentation Best Practices

**DO:**
- ✅ Generate documentation after every significant change
- ✅ Keep generated HTML docs in `docs/` directory
- ✅ Include documentation generation in Makefile
- ✅ Verify pydoc can parse all modules before committing
- ✅ Use `uv run python -m pydoc -b` for quick reference during development
- ✅ Include links to documentation in README.md
- ✅ Regenerate docs as part of release process

**DON'T:**
- ❌ Commit generated `.html` files to git (add `docs/api/*.html` to .gitignore)
- ❌ Use external documentation tools when pydoc is sufficient
- ❌ Write docstrings that don't render well in pydoc
- ❌ Forget to test documentation generation in CI

#### I. Example: Complete Documentation Workflow

```bash
# 1. Write code with complete docstrings
cat > mymodule.py << 'EOF'
"""
My Module: Signal Processing Utilities.

This module provides signal processing functions for OFDM signals.
"""

def process_signal(signal: list[float], threshold: float = 0.5) -> list[float]:
    """
    Process signal by applying threshold filter.
    
    Args:
        signal: Input signal as list of floats.
        threshold: Threshold value for filtering. Defaults to 0.5.
        
    Returns:
        Filtered signal with values above threshold.
        
    Example:
        >>> process_signal([0.1, 0.6, 0.3, 0.8], 0.5)
        [0.6, 0.8]
    """
    return [s for s in signal if s >= threshold]
EOF

# 2. Verify docstring completeness
uv run python -c "import mymodule; print(mymodule.process_signal.__doc__)"

# 3. View in terminal
uv run python -m pydoc mymodule.process_signal

# 4. Generate HTML
uv run python -m pydoc -w mymodule

# 5. View in browser
uv run python -m pydoc -b
# Opens browser to view interactive documentation

# 6. Verify it works
test -f mymodule.html && echo "✓ Documentation generated successfully"
```

#### J. Documentation Checklist

**Before submitting code, verify:**
- [ ] All modules can be parsed by pydoc: `uv run python -m pydoc module_name`
- [ ] Generated HTML documentation is readable and complete
- [ ] All public functions/classes appear in documentation
- [ ] Examples in docstrings render correctly
- [ ] Type hints are visible in generated documentation
- [ ] Cross-references between functions work
- [ ] Makefile includes `docs` target for documentation generation
- [ ] README.md includes instructions for viewing documentation
- [ ] CI pipeline verifies documentation can be generated

---

## 6. Type Hints: Strict Typing
### Requirements
- **ALL** function parameters **MUST** have type hints
- **ALL** function return values **MUST** have type hints
- **ALL** class attributes **MUST** have type hints
- Use `typing` module for complex types
- Run `ruff check` via `uv run ruff` to verify types

### Type Hint Standards

```python
from typing import (
    Any, Dict, List, Tuple, Optional, Union, Callable,
    TypeVar, Generic, Protocol, Literal
)

import cupy as cp
import numpy as np
from pathlib import Path

# Basic type hints

def simple_function(x: int, y: float) -> str:
    """Convert numbers to string."""
    return f"{x} and {y}"


# Optional parameters
def with_optional(required: str, optional: Optional[int] = None) -> bool:
    """Process with optional parameter."""
    return optional is not None

# Multiple return values
def multiple_returns(data: List[float]) -> Tuple[float, float, int]:
    """Calculate statistics."""
    return min(data), max(data), len(data)

# Complex types

def process_config(
    config: Dict[str, Any],
    filters: List[Callable[[cp.ndarray], cp.ndarray]]
) -> Dict[str, Union[int, float, str]]:
    """Process configuration with filters."""
    result: Dict[str, Union[int, float, str]] = {}

    # Implementation
    return result

# Generic types

T = TypeVar('T')

def first_element(items: List[T]) -> Optional[T]:
    """Get first element from list."""
    return items[0] if items else None

# Protocol for duck typing
class SignalGenerator(Protocol):
    """Protocol for signal generator classes."""

    def generate(self, num_samples: int) -> cp.ndarray:
        """Generate signal samples."""

..


# Literal types for restricted values

def set_environment(env: Literal["development", "production", "testing"]) -> None:
    """Set runtime environment."""
    # Implementation
    pass


# Class with type hints

class SignalProcessor:
    """Process OFDM signals with type-safe operations."""
    # Class variable
    MAX_SAMPLES: int = 1000000
    # Instance variables with type hints

    def __init__(
        self,
        sample_rate: float,
        num_channels: int,
        config: Dict[str, Any]
    ) -> None:
        """Initialize processor."""
        self.sample_rate: float = sample_rate
        self.num_channels: int = num_channels
        self.config: Dict[str, Any] = config
        self.buffer: Optional[cp.ndarray] = None
        self._cache: Dict[int, cp.ndarray] = {}

    def process(
        self,
        signal: cp.ndarray,
        apply_filter: bool = True
    ) -> Tuple[cp.ndarray, float]:
        """Process signal and return result with quality metric."""
        # Implementation
        result: cp.ndarray = signal
        quality: float = 0.99
        return result, quality


    # NumPy/CuPy array type hints
    def array_function(
        self,
        cpu_array: np.ndarray,
        gpu_array: cp.ndarray
    ) -> Tuple[np.ndarray, cp.ndarray]:
        """Process arrays on CPU and GPU."""
        # Implementation
        cpu_result: np.ndarray = cpu_array
        gpu_result: cp.ndarray = gpu_array
        return cpu_result, gpu_result

    # Use NDArray for more specific array types
    from numpy.typing import NDArray
    def typed_array_func(data: NDArray[np.float32]) -> NDArray[np.float32]:
        return data * 2.0

    return cpu_array, gpu_array


    # Path type hints
    def load_config_file(filepath: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        with filepath.open('r') as f:
            # Implementation
            pass
        return {}


    # Callable type hints

    def apply_transform(
        data: cp.ndarray,
        transform: Callable[[cp.ndarray], cp.ndarray],
        callback: Optional[Callable[[int], None]] = None
    ) -> cp.ndarray:
    """Apply transformation with optional progress callback."""
    if callback:
        callback(0)
    result = transform(data)
    if callback:
        callback(100)
    return result
```

### Type Checking with Ruff

**Install Ruff**:

```bash
uv add --dev ruff
```

**Create `pyproject.toml`**:

```toml
[project]
name = "ofdm-optimizer"
version = "0.1.0"
requires-python = ">=3.11"

[tool.ruff]
# Python version target
target-version = "py311"

# Enable type checking
select = [
    "E", # pycodestyle errors
    "W", # pycodestyle warnings
    "F", # pyflakes
    "I", # isort
    "ANN", # flake8-annotations (type hints)
    "B", # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
]

# Ignore specific rules
ignore = [
    "ANN101", # Missing type annotation for self
    "ANN102", # Missing type annotation for cls
]

# Line length
line-length = 100

# Exclude directories
exclude = [
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"] # Allow unused imports in __init__.py

```

**Run Type Checking**:

```bash
# Check all files
uv run ruff check .

# Check specific file
uv run ruff check generate_codes/__init__.py

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .

```


### Type Hint Best Practices


1. **Always specify return type**, even for `None`:

```python
def no_return() -> None:
    """Function with no return value."""
    print("Done")

```



2. **Use `Optional[T]` for nullable values**:

```python
def find_item(items: List[int], target: int) -> Optional[int]:
    """Find item or return None."""
    return next((i for i in items if i == target), None)
```

3. **Use `Union` for multiple possible types**:
```python

def parse_value(value: str) -> Union[int, float, str]:
    """Parse string to appropriate type."""
    try:
        return int(value)
    except ValueError:

    try:
        return float(value)
    except ValueError:
        return value

```



4. **Document complex types in docstring**:

```python
def complex_func(
    data: Dict[str, List[Tuple[int, float]]]
) -> List[Dict[str, Any]]:

    """
    Process complex nested data structure.
    Args:
    data: Dictionary mapping string keys to lists of (index, value) tuples.
    Example: {"signal_a": [(0, 1.5), (1, 2.3)], "signal_b": [...]}

    Returns:
    List of dictionaries with processed results.
    """

    pass
```



---

## 7. Pythonic Code Patterns: Comprehensions (MANDATORY)

### A. Comprehension Requirements

**ALWAYS prefer comprehensions over loops for:**
- Creating lists, sets, and dictionaries
- Filtering and transforming data
- Memory-efficient iteration (generators)

**Benefits:**
- **Performance**: 2-3x faster than equivalent loops
- **Memory**: Generator expressions use constant memory
- **Readability**: More concise and expressive
- **Pythonic**: Idiomatic Python style

### B. List Comprehensions

```python
# ✅ CORRECT - List comprehension (PREFERRED)
squares = [x**2 for x in range(10)]

# Filter with condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Nested comprehension for flattening
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]

# Transform with function
names = ["alice", "bob", "charlie"]
capitalized = [name.capitalize() for name in names]

# ❌ WRONG - Manual loop (AVOID)
squares = []
for x in range(10):
    squares.append(x**2)

# ❌ WRONG - map() when comprehension is clearer
squares = list(map(lambda x: x**2, range(10)))  # Less readable
```

### C. Set Comprehensions

```python
# ✅ CORRECT - Set comprehension (PREFERRED)
unique_lengths = {len(word) for word in words}

# Filter duplicates while transforming
unique_squares = {x**2 for x in numbers}

# Conditional set
even_set = {x for x in range(20) if x % 2 == 0}

# ❌ WRONG - Manual set creation
unique_lengths = set()
for word in words:
    unique_lengths.add(len(word))
```

### D. Dictionary Comprehensions

```python
# ✅ CORRECT - Dictionary comprehension (PREFERRED)
# Create dict from two lists
keys = ['a', 'b', 'c']
values = [1, 2, 3]
mapping = {k: v for k, v in zip(keys, values)}

# Transform dictionary
prices = {'apple': 0.40, 'banana': 0.50}
prices_usd = {item: price * 1.1 for item, price in prices.items()}

# Filter dictionary
expensive = {item: price for item, price in prices.items() if price > 0.45}

# Invert dictionary
inverted = {v: k for k, v in original_dict.items()}

# Count occurrences
words = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
word_count = {word: words.count(word) for word in set(words)}

# ❌ WRONG - Manual dictionary creation
mapping = {}
for k, v in zip(keys, values):
    mapping[k] = v
```

### E. Generator Expressions (Memory Efficiency)

```python
# ✅ CORRECT - Generator expression (PREFERRED for large datasets)
# Memory-efficient: creates values on-demand
large_squares = (x**2 for x in range(1_000_000))

# Use in iteration
total = sum(x**2 for x in range(1_000_000))  # Constant memory

# Chain generators
data = (process(x) for x in read_large_file())
filtered = (x for x in data if is_valid(x))

# Generator with complex expression
normalized = (
    (value - mean) / std_dev
    for value in dataset
    if value is not None
)

# ❌ WRONG - List comprehension for large data (wastes memory)
large_squares = [x**2 for x in range(1_000_000)]  # Uses ~8MB of RAM

# ✅ CORRECT - Use generator when you only need to iterate once
def process_large_file(filepath: Path) -> Generator[dict, None, None]:
    """Process large file line by line without loading into memory."""
    with filepath.open() as f:
        return (json.loads(line) for line in f)

# ❌ WRONG - Loading entire file into memory
def process_large_file_bad(filepath: Path) -> list[dict]:
    """BAD: Loads entire file into memory."""
    with filepath.open() as f:
        return [json.loads(line) for line in f]
```

### F. When to Use Each Type

| Use Case | Comprehension Type | Reason |
|----------|-------------------|---------|
| Need to iterate multiple times | List `[...]` | Materialized data |
| Need unique values | Set `{...}` | Automatic deduplication |
| Need key-value mapping | Dict `{k: v ...}` | Fast lookups |
| One-time iteration, large data | Generator `(...)` | Memory efficient |
| Pass to function expecting iterable | Generator `(...)` | Lazy evaluation |

### G. Complex Comprehensions

```python
# ✅ CORRECT - Nested comprehension
# Cartesian product
pairs = [(x, y) for x in range(3) for y in range(3)]
# [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]

# Matrix transposition
matrix = [[1, 2, 3], [4, 5, 6]]
transposed = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
# [[1, 4], [2, 5], [3, 6]]

# Conditional expression (ternary)
processed = [x if x > 0 else 0 for x in numbers]

# Multiple conditions
filtered = [
    x for x in data
    if x is not None
    if x > 0
    if x < 100
]

# ✅ CORRECT - Use comprehension with function call
import numpy as np

def validate_signal(signal: cp.ndarray) -> bool:
    """Check if signal is valid."""
    return bool(cp.any(signal))

valid_signals = [
    sig for sig in signals
    if validate_signal(sig)
]
```

### H. Readability Guidelines

**DO use comprehensions when:**
- The logic fits on 1-2 lines
- The operation is simple and obvious
- It improves performance significantly

**DON'T use comprehensions when:**
- Logic is complex and nested > 2 levels
- Readability suffers
- You need error handling

```python
# ⚠️ TOO COMPLEX - Use regular loop for readability
# BAD: Hard to read
result = [
    func(x, y) if cond1(x) else other_func(x, y)
    for x in data
    if validate(x)
    for y in get_related(x)
    if y is not None and check(y)
]

# ✅ BETTER - Use regular loop for complex logic
result = []
for x in data:
    if not validate(x):
        continue
    for y in get_related(x):
        if y is None or not check(y):
            continue
        if cond1(x):
            result.append(func(x, y))
        else:
            result.append(other_func(x, y))
```

### I. Performance Comparison

```python
# ✅ CORRECT - Comprehension (FASTEST)
import timeit

# List comprehension: ~0.065 seconds
squares_comp = timeit.timeit(
    '[x**2 for x in range(1000)]',
    number=10000
)

# ❌ SLOWER - Loop with append: ~0.090 seconds
squares_loop = timeit.timeit(
    '''
result = []
for x in range(1000):
    result.append(x**2)
''',
    number=10000
)

# ❌ SLOWER - map(): ~0.080 seconds
squares_map = timeit.timeit(
    'list(map(lambda x: x**2, range(1000)))',
    number=10000
)

# ✅ BEST - Generator (constant memory, lazy evaluation)
squares_gen = (x**2 for x in range(1_000_000))  # Instant, O(1) memory
```

### J. Real-World Examples

```python
# ✅ CORRECT - Data processing pipeline
from pathlib import Path
from typing import Generator

def process_log_files(directory: Path) -> Generator[dict, None, None]:
    """Process log files efficiently using generators."""
    # Generator: Find all log files
    log_files = (f for f in directory.glob("*.log") if f.is_file())
    
    # Generator: Read and parse lines
    log_lines = (
        line.strip()
        for log_file in log_files
        for line in log_file.open()
        if line.strip()
    )
    
    # Generator: Parse to structured data
    parsed_logs = (
        parse_log_line(line)
        for line in log_lines
        if not line.startswith('#')
    )
    
    # Generator: Filter valid entries
    return (
        log for log in parsed_logs
        if log.get('level') == 'ERROR'
    )

# ✅ CORRECT - Transform configuration
config_raw = {
    'timeout': '30',
    'retries': '3',
    'enabled': 'true',
    'port': '8080'
}

# Dictionary comprehension for type conversion
config_typed = {
    key: int(value) if value.isdigit() else
         value.lower() == 'true' if value.lower() in ('true', 'false') else
         value
    for key, value in config_raw.items()
}

# ✅ CORRECT - Filter and transform API responses
users_data = [
    {'name': 'Alice', 'age': 30, 'active': True},
    {'name': 'Bob', 'age': 25, 'active': False},
    {'name': 'Charlie', 'age': 35, 'active': True}
]

# Get names of active users
active_names = [
    user['name']
    for user in users_data
    if user.get('active', False)
]

# Create lookup dictionary
user_lookup = {
    user['name']: user
    for user in users_data
}

# ✅ CORRECT - Signal processing with NumPy/CuPy
import cupy as cp

def process_signals(signals: list[cp.ndarray]) -> list[cp.ndarray]:
    """Process multiple signals efficiently.
    
    Args:
        signals: List of signal arrays.
        
    Returns:
        List of normalized signals.
    """
    # Comprehension for batch processing
    normalized = [
        signal / cp.max(cp.abs(signal))
        for signal in signals
        if cp.any(signal)  # Skip empty signals
    ]
    
    return normalized
```

### K. Anti-Patterns to Avoid

```python
# ❌ WRONG - Unnecessary list() around generator in sum/max/min
total = sum([x**2 for x in range(1000)])  # Wastes memory

# ✅ CORRECT - Use generator directly
total = sum(x**2 for x in range(1000))  # Memory efficient


# ❌ WRONG - Building list just to check existence
if 'target' in [x.lower() for x in items]:  # O(n) memory + time
    pass

# ✅ CORRECT - Use generator expression
if 'target' in (x.lower() for x in items):  # O(1) memory


# ❌ WRONG - Filtering then mapping (two passes)
filtered = [x for x in numbers if x > 0]
result = [x**2 for x in filtered]

# ✅ CORRECT - Combined filter and map (one pass)
result = [x**2 for x in numbers if x > 0]


# ❌ WRONG - Comprehension with side effects
[print(x) for x in items]  # Creates unnecessary list

# ✅ CORRECT - Use explicit loop for side effects
for x in items:
    print(x)
```

### L. Comprehension Checklist

Before writing a loop, ask:
- [ ] Can this be a list comprehension?
- [ ] Should this be a generator expression (large data)?
- [ ] Is this a set/dict creation? (use set/dict comprehension)
- [ ] Is the logic simple enough to be readable?
- [ ] Am I only iterating once? (use generator)

**Rule of thumb:** If you're writing `result = []` followed by a loop with `append()`, use a comprehension instead.

---

## 8. Functional Programming Style (MANDATORY)

### A. Functional Programming Requirements

**ALWAYS prefer functional programming patterns when applicable:**
- Write pure functions without side effects
- Use immutable data structures
- Leverage higher-order functions
- Compose functions for complex operations
- Avoid mutating state

**Benefits:**
- **Testability**: Pure functions are easier to test
- **Reliability**: No side effects = predictable behavior
- **Concurrency**: Immutable data is thread-safe
- **Readability**: Declarative code is easier to understand
- **Maintainability**: Less hidden state to track

### B. Pure Functions (MANDATORY)

```python
# ✅ CORRECT - Pure function (PREFERRED)
def calculate_total(prices: list[float], tax_rate: float) -> float:
    """Calculate total price with tax.
    
    Args:
        prices: List of item prices.
        tax_rate: Tax rate as decimal (e.g., 0.08 for 8%).
        
    Returns:
        Total price including tax.
    """
    subtotal = sum(prices)
    return subtotal * (1 + tax_rate)

# ❌ WRONG - Function with side effects (AVOID)
total = 0.0

def add_to_total(price: float) -> None:
    """BAD: Modifies global state."""
    global total
    total += price  # Side effect!


# ✅ CORRECT - Transform data without mutation
def normalize_data(data: list[float]) -> list[float]:
    """Normalize data to 0-1 range.
    
    Args:
        data: Input data values.
        
    Returns:
        Normalized data (new list, input unchanged).
    """
    if not data:
        return []
    
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val
    
    if range_val == 0:
        return [0.0] * len(data)
    
    return [(x - min_val) / range_val for x in data]

# ❌ WRONG - Mutates input data
def normalize_data_bad(data: list[float]) -> None:
    """BAD: Modifies input list in place."""
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val
    
    for i in range(len(data)):
        data[i] = (data[i] - min_val) / range_val  # Mutation!
```

### C. Immutability (MANDATORY)

```python
from typing import NamedTuple, FrozenSet
from dataclasses import dataclass

# ✅ CORRECT - Immutable data structures (PREFERRED)
# Use tuples instead of lists when data shouldn't change
def get_coordinates() -> tuple[float, float, float]:
    """Return fixed coordinates."""
    return (10.5, 20.3, 30.1)

# Use frozenset instead of set for immutable collections
ALLOWED_EXTENSIONS: FrozenSet[str] = frozenset(['.py', '.txt', '.md'])

# Use NamedTuple for immutable records
class Point(NamedTuple):
    """Immutable 3D point."""
    x: float
    y: float
    z: float

p1 = Point(1.0, 2.0, 3.0)
# p1.x = 5.0  # AttributeError - immutable!

# Use dataclass with frozen=True
@dataclass(frozen=True)
class Config:
    """Immutable configuration."""
    host: str
    port: int
    timeout: float

config = Config(host="localhost", port=8080, timeout=30.0)
# config.port = 9000  # FrozenInstanceError - immutable!


# ✅ CORRECT - Return new objects instead of modifying
def add_item(items: tuple[str, ...], new_item: str) -> tuple[str, ...]:
    """Add item to tuple (returns new tuple).
    
    Args:
        items: Existing items.
        new_item: Item to add.
        
    Returns:
        New tuple with added item.
    """
    return items + (new_item,)

# ❌ WRONG - Mutable default arguments (DANGEROUS)
def append_to_list_bad(value: int, lst: list[int] = []) -> list[int]:
    """BAD: Mutable default argument causes bugs."""
    lst.append(value)  # Modifies shared default!
    return lst

# ✅ CORRECT - Immutable default with None
def append_to_list(value: int, lst: list[int] | None = None) -> list[int]:
    """Safely append to list with default.
    
    Args:
        value: Value to append.
        lst: Optional list (creates new if None).
        
    Returns:
        New list with appended value.
    """
    if lst is None:
        lst = []
    return [*lst, value]  # Creates new list
```

### D. Higher-Order Functions

```python
from functools import reduce, partial
from typing import Callable

# ✅ CORRECT - Using map, filter, reduce (PREFERRED)
numbers = [1, 2, 3, 4, 5]

# Map: Transform each element
squared = list(map(lambda x: x**2, numbers))
# [1, 4, 9, 16, 25]

# Filter: Select elements matching condition
evens = list(filter(lambda x: x % 2 == 0, numbers))
# [2, 4]

# Reduce: Combine elements into single value
from functools import reduce
product = reduce(lambda x, y: x * y, numbers, 1)
# 120

# ✅ CORRECT - Functions as arguments
def apply_operation(
    data: list[float],
    operation: Callable[[float], float]
) -> list[float]:
    """Apply operation to all elements.
    
    Args:
        data: Input data.
        operation: Function to apply to each element.
        
    Returns:
        Transformed data.
    """
    return [operation(x) for x in data]

def square(x: float) -> float:
    """Square a number."""
    return x ** 2

result = apply_operation([1.0, 2.0, 3.0], square)
# [1.0, 4.0, 9.0]


# ✅ CORRECT - Partial application
def multiply(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y

double = partial(multiply, 2.0)  # Fix first argument
triple = partial(multiply, 3.0)

print(double(5.0))  # 10.0
print(triple(5.0))  # 15.0


# ✅ CORRECT - Returning functions (closures)
def create_multiplier(factor: float) -> Callable[[float], float]:
    """Create a multiplication function.
    
    Args:
        factor: Multiplication factor.
        
    Returns:
        Function that multiplies by factor.
    """
    def multiplier(x: float) -> float:
        return x * factor
    return multiplier

times_two = create_multiplier(2.0)
times_ten = create_multiplier(10.0)

print(times_two(5.0))  # 10.0
print(times_ten(5.0))  # 50.0
```

### E. Function Composition

```python
from typing import Callable, TypeVar

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

# ✅ CORRECT - Function composition (PREFERRED)
def compose(
    f: Callable[[U], V],
    g: Callable[[T], U]
) -> Callable[[T], V]:
    """Compose two functions: (f ∘ g)(x) = f(g(x)).
    
    Args:
        f: Second function to apply.
        g: First function to apply.
        
    Returns:
        Composed function.
    """
    def composed(x: T) -> V:
        return f(g(x))
    return composed

# Example: Create data processing pipeline
def clean_text(text: str) -> str:
    """Remove whitespace."""
    return text.strip()

def to_lowercase(text: str) -> str:
    """Convert to lowercase."""
    return text.lower()

def remove_punctuation(text: str) -> str:
    """Remove punctuation."""
    import string
    return text.translate(str.maketrans('', '', string.punctuation))

# Compose functions into pipeline
process_text = compose(
    remove_punctuation,
    compose(to_lowercase, clean_text)
)

result = process_text("  Hello, World!  ")
# "hello world"


# ✅ CORRECT - Pipeline using reduce
def pipeline(*functions: Callable) -> Callable:
    """Create function pipeline from multiple functions.
    
    Args:
        *functions: Functions to compose (applied left to right).
        
    Returns:
        Composed function.
    """
    def apply_pipeline(value):
        return reduce(lambda v, f: f(v), functions, value)
    return apply_pipeline

# Create text processing pipeline
process = pipeline(
    str.strip,
    str.lower,
    lambda s: s.replace(',', ''),
    lambda s: s.replace('!', '')
)

result = process("  Hello, World!  ")
# "hello world"
```

### F. Functional Data Processing

```python
from itertools import chain, groupby, takewhile, dropwhile
from functools import reduce
from typing import Iterator

# ✅ CORRECT - Using itertools for functional operations
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# takewhile: Take elements while condition is true
small_numbers = list(takewhile(lambda x: x < 5, data))
# [1, 2, 3, 4]

# dropwhile: Skip elements while condition is true
large_numbers = list(dropwhile(lambda x: x < 5, data))
# [5, 6, 7, 8, 9, 10]

# chain: Flatten nested iterables
nested = [[1, 2], [3, 4], [5, 6]]
flattened = list(chain.from_iterable(nested))
# [1, 2, 3, 4, 5, 6]

# groupby: Group consecutive elements
data_with_types = [
    ('a', 1), ('a', 2), ('b', 3), ('b', 4), ('c', 5)
]
grouped = {
    key: list(group)
    for key, group in groupby(data_with_types, key=lambda x: x[0])
}
# {'a': [('a', 1), ('a', 2)], 'b': [('b', 3), ('b', 4)], 'c': [('c', 5)]}


# ✅ CORRECT - Functional data transformations
def process_sales_data(
    sales: list[dict[str, float | str]]
) -> dict[str, float]:
    """Process sales data functionally.
    
    Args:
        sales: List of sale records with 'category' and 'amount'.
        
    Returns:
        Dictionary mapping category to total sales.
    """
    # Filter valid sales
    valid_sales = filter(
        lambda s: s.get('amount', 0) > 0,
        sales
    )
    
    # Group by category and sum
    from collections import defaultdict
    category_totals: dict[str, float] = defaultdict(float)
    
    for sale in valid_sales:
        category = str(sale.get('category', 'unknown'))
        amount = float(sale.get('amount', 0))
        category_totals[category] += amount
    
    return dict(category_totals)


# ✅ CORRECT - Using reduce for complex aggregations
def calculate_statistics(numbers: list[float]) -> dict[str, float]:
    """Calculate statistics using functional approach.
    
    Args:
        numbers: List of numbers.
        
    Returns:
        Dictionary with min, max, sum, count, mean.
    """
    if not numbers:
        return {'min': 0.0, 'max': 0.0, 'sum': 0.0, 'count': 0, 'mean': 0.0}
    
    stats = reduce(
        lambda acc, x: {
            'min': min(acc['min'], x),
            'max': max(acc['max'], x),
            'sum': acc['sum'] + x,
            'count': acc['count'] + 1
        },
        numbers,
        {'min': float('inf'), 'max': float('-inf'), 'sum': 0.0, 'count': 0}
    )
    
    stats['mean'] = stats['sum'] / stats['count'] if stats['count'] > 0 else 0.0
    return stats
```

### G. Lambda Functions (Use Judiciously)

```python
# ✅ CORRECT - Simple lambdas for callbacks (PREFERRED)
numbers = [1, 2, 3, 4, 5]

# Good: Simple transformation
squared = list(map(lambda x: x**2, numbers))

# Good: Simple predicate
evens = list(filter(lambda x: x % 2 == 0, numbers))

# Good: Key function for sorting
data = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
sorted_data = sorted(data, key=lambda x: x['age'])


# ✅ CORRECT - Named function for complex logic (PREFERRED)
def is_valid_email(email: str) -> bool:
    """Check if email is valid.
    
    Args:
        email: Email address to validate.
        
    Returns:
        True if valid email format.
    """
    return '@' in email and '.' in email.split('@')[1]

emails = ['test@example.com', 'invalid', 'user@domain.org']
valid_emails = list(filter(is_valid_email, emails))


# ❌ WRONG - Complex lambda (hard to read, use named function instead)
result = list(map(
    lambda x: x**2 if x > 0 else -x**2 if x < 0 else 0,
    numbers
))

# ✅ CORRECT - Named function for clarity
def transform_number(x: float) -> float:
    """Transform number based on sign."""
    if x > 0:
        return x**2
    elif x < 0:
        return -x**2
    else:
        return 0.0

result = list(map(transform_number, numbers))
```

### H. Avoiding Side Effects

```python
# ❌ WRONG - Function with hidden side effects
log_entries = []

def process_data_bad(data: list[int]) -> list[int]:
    """BAD: Has side effect of logging."""
    result = [x * 2 for x in data]
    log_entries.append(f"Processed {len(data)} items")  # Side effect!
    return result


# ✅ CORRECT - Explicit logging parameter
def process_data(
    data: list[int],
    logger: Callable[[str], None] | None = None
) -> list[int]:
    """Process data with optional logging.
    
    Args:
        data: Input data.
        logger: Optional logging function.
        
    Returns:
        Processed data.
    """
    result = [x * 2 for x in data]
    
    if logger:
        logger(f"Processed {len(data)} items")
    
    return result

# Usage
log_buffer: list[str] = []
result = process_data([1, 2, 3], logger=lambda msg: log_buffer.append(msg))


# ✅ CORRECT - Pure function returns both result and log
def process_data_pure(data: list[int]) -> tuple[list[int], str]:
    """Process data and return result with log message.
    
    Args:
        data: Input data.
        
    Returns:
        Tuple of (processed data, log message).
    """
    result = [x * 2 for x in data]
    log_message = f"Processed {len(data)} items"
    return result, log_message

# Usage
result, log_msg = process_data_pure([1, 2, 3])
```

### I. Functional Programming with NumPy/CuPy

```python
import numpy as np
import cupy as cp

# ✅ CORRECT - Functional array operations (PREFERRED)
def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize array to 0-1 range (pure function).
    
    Args:
        arr: Input array.
        
    Returns:
        New normalized array (input unchanged).
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    range_val = max_val - min_val
    
    if range_val == 0:
        return np.zeros_like(arr)
    
    # Returns new array, doesn't modify input
    return (arr - min_val) / range_val

# ✅ CORRECT - Chaining array operations
def process_signal(signal: cp.ndarray) -> cp.ndarray:
    """Process signal using functional composition.
    
    Args:
        signal: Input signal.
        
    Returns:
        Processed signal.
    """
    # Each operation returns new array
    centered = signal - cp.mean(signal)
    normalized = centered / cp.std(centered)
    filtered = cp.where(cp.abs(normalized) < 3, normalized, 0)
    return filtered

# ❌ WRONG - Mutating arrays in place
def process_signal_bad(signal: cp.ndarray) -> None:
    """BAD: Modifies input array."""
    signal -= cp.mean(signal)  # In-place mutation!
    signal /= cp.std(signal)   # In-place mutation!
```

### J. When to Use Functional Programming

**✅ USE functional programming for:**
- Data transformations (map, filter, reduce)
- Mathematical operations
- Data validation and filtering
- Configuration processing
- Stateless computations
- Parallel/concurrent operations

**⚠️ USE CAUTION for:**
- I/O operations (inherently have side effects)
- Performance-critical code (profiling may show loops are faster)
- Very complex logic (readability may suffer)

**❌ DON'T force functional style when:**
- Imperative code is significantly clearer
- Dealing with external state (databases, files, APIs)
- Readability suffers from excessive abstraction

### K. Functional Programming Checklist

Before writing imperative code, ask:
- [ ] Can this be a pure function without side effects?
- [ ] Should I use immutable data structures?
- [ ] Can I use map/filter/reduce instead of loops?
- [ ] Can I compose smaller functions?
- [ ] Am I mutating input data (use copy instead)?
- [ ] Can I use comprehensions or generators?

**Rule of thumb:** Default to functional style, but prioritize readability and maintainability.

---


## 9. Code Quality Enforcement


### Pre-Commit Checks

Create `.pre-commit-config.yaml`:

```yaml
repos:
- repo: local
hooks:
- id: ruff-check
name: ruff check
entry: uv run ruff check --fix
language: system
types: [python]
- id: ruff-format
  name: ruff format
  entry: uv run ruff format
  language: system
  types: [python]
```


Install pre-commit hooks:


```bash
uv add --dev pre-commit
pre-commit install

```


### Continuous Integration

**GitHub Actions** (`.github/workflows/ci.yml`):

**ALL CI commands MUST use `uv run` prefix**

```yaml
name: CI
on: [push, pull_request]

jobs:
quality:
runs-on: ubuntu-latest

steps:
- uses: actions/checkout@v4

      - name: Install uv
uses: astral-sh/setup-uv@v1
        
- name: Set up Python
  run: uv python install 3.11
        
- name: Install dependencies
  run: |
    uv venv
          uv sync
          
      - name: Verify syntax
        run: uv run python -m py_compile **/*.py
        
- name: Run Ruff checks
  run: uv run ruff check .
        
- name: Run Ruff format check
  run: uv run ruff format --check .
        
      - name: Run tests with coverage
        run: uv run pytest --cov=. --cov-report=xml --cov-report=term --cov-fail-under=100
        
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          fail_ci_if_error: true
```


---

## 10. Project Structure

### Standard Layout

```

project-name/
├── .github/
│ └── workflows/
│ └── ci.yml
├── config/
│ ├── __init__.py # Dynaconf initialization
│ └── defaults.toml # Default configuration
├── generate_codes/
│ ├── __init__.py # Main signal generation module
│ ├── chirp.py # Chirp signal functions
│ ├── ofdm.py # OFDM signal functions
│ └── optimization.py # Optimization algorithms
├── tests/
│ ├── __init__.py
│ ├── test_chirp.py
│ ├── test_ofdm.py
│ └── test_config.py
├── .gitignore
├── .pre-commit-config.yaml
├── config.toml # User configuration
├── .secrets.toml # Secrets (git-ignored)
├── main.py # Entry point
├── pyproject.toml # Project metadata & tool config
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── AGENTS.md # This file

```

---


## 11. Development Workflow

### Starting a New Feature

**CRITICAL: ALL commands use `uv` or `uv run` prefix**

```bash
# 1. Create feature branch
git checkout -b feature/new-signal-type

# 2. Ensure virtual environment is active
source .venv/bin/activate

# 3. Install/update dependencies
uv sync

# 4. Make changes with proper type hints and docstrings
# - Add complete Google-style docstrings
# - Add type hints to all parameters and returns
# - Use Dynaconf for configuration

# 5. Write tests FIRST or alongside code (TDD)
# tests/test_new_feature.py
uv run pytest tests/test_new_feature.py -v

# 6. Verify syntax
uv run python -m py_compile module.py

# 7. Run quality checks
uv run ruff check --fix .
uv run ruff format .

# 8. Run ALL tests with coverage
uv run pytest --cov=. --cov-fail-under=100

# 9. Verify everything works
uv run main.py

# 10. Commit and push (only after all checks pass)
git add .
git commit -m "feat: add new signal type with tests and typing"
git push origin feature/new-signal-type
```

**Remember:** Never commit code without tests, and never commit code that doesn't pass all checks.


### Code Review Checklist

**Before submitting a PR, ALL items must be verified:**

#### Documentation
- [ ] All functions have complete docstrings (Google style)
- [ ] All parameters have type hints
- [ ] All return values have type hints
- [ ] Module-level docstrings present
- [ ] Documentation can be generated with pydoc: `uv run python -m pydoc module_name`
- [ ] Generated documentation is complete and readable

#### Configuration
- [ ] Configuration values use Dynaconf (no hardcoded values)
- [ ] Configuration changes documented in `defaults.toml`

#### Code Quality
- [ ] `uv run ruff check .` passes with no errors
- [ ] `uv run ruff format .` has been applied
- [ ] Code parses without syntax errors
- [ ] List/set/dict comprehensions used instead of manual loops
- [ ] Generator expressions used for memory efficiency
- [ ] No unnecessary materializations (list wrapping generators)

#### Testing (MANDATORY)
- [ ] ALL new functions/classes have tests
- [ ] ALL tests pass: `uv run pytest` returns 0
- [ ] Coverage is 100%: `uv run pytest --cov=. --cov-fail-under=100`
- [ ] ALL test commands use `uv run` prefix
- [ ] Test names are descriptive

#### UV Requirement
- [ ] ALL package operations use `uv`
- [ ] ALL Python/pytest commands use `uv run` prefix
- [ ] No direct `pip`, `pytest`, or `python` commands

#### Agent-Generated Code (If Applicable)
- [ ] Agent verified code syntax before submission
- [ ] Agent ran tests and confirmed they pass
- [ ] Agent fixed any errors iteratively

---

## 12. Testing Requirements (MANDATORY)

**ALL code MUST have tests.** Tests are not optional - they are a required part of the development process.

### A. Test Requirements
* **100% test coverage** - Every function and class must have tests
* **Tests MUST pass** - `uv run pytest` must return 0 exit code
* **Use pytest framework** - Standard Python testing framework
* **ALL tests use `uv run` prefix** - Ensures proper dependency management
* **Tests before code review** - No untested code in PRs
* **Fast execution** - Unit tests should run in < 5 seconds total
* **Descriptive test names** - Test names explain what is being tested

### B. Test File Structure

```python
"""
Tests for OFDM signal generation.

This module tests the core signal generation functions including chirp
creation, OFDM modulation, and phase code optimization.
"""

from typing import Tuple
import pytest
import cupy as cp
import numpy as np
from generate_codes import generate_chirp_signal, generate_chirp_ofdm_signal
from config import CONF


class TestChirpGeneration:
    """Test suite for chirp signal generation."""

    def test_generate_chirp_signal_basic(self) -> None:
        """Test basic chirp signal generation."""
        sig, t = generate_chirp_signal(
            basefreq=2e6,
            chirp_bw=10e3,
            chirp_duration=50e-6,
            num_samples=1000
        )
        assert isinstance(sig, cp.ndarray)
        assert isinstance(t, cp.ndarray)
        assert sig.shape == (1000,)
        assert t.shape == (1000,)

    def test_generate_chirp_signal_with_phase(self) -> None:
        """Test chirp generation with phase offset."""
        phase = np.pi / 4
        sig1, _ = generate_chirp_signal(2e6, 10e3, 50e-6, 1000, phase=0)
        sig2, _ = generate_chirp_signal(2e6, 10e3, 50e-6, 1000, phase=phase)
        # Signals should be different with different phases
        assert not cp.allclose(sig1, sig2)

    @pytest.mark.parametrize("basefreq", [0, -1e6])
    def test_invalid_basefreq(self, basefreq: float) -> None:
        """Test that invalid basefreq raises ValueError."""
        with pytest.raises(ValueError):
            generate_chirp_signal(basefreq, 10e3, 50e-6, 1000)

    def test_invalid_chirp_bw(self) -> None:
        """Test that invalid chirp_bw raises ValueError."""
        with pytest.raises(ValueError):
            generate_chirp_signal(2e6, -10e3, 50e-6, 1000)

    def test_invalid_duration(self) -> None:
        """Test that invalid duration raises ValueError."""
        with pytest.raises(ValueError):
            generate_chirp_signal(2e6, 10e3, -50e-6, 1000)

    @pytest.fixture
    def sample_config(self) -> dict:
        """Provide sample configuration for tests."""
        return {
            'pulse_duration': 50e-6,
            'bandwidth': 5e6,
            'sample_rate': 20e6,
            'num_subcarriers': 250,
            }

    def test_config_loading(self, sample_config: dict) -> None:
        """Test that configuration loads correctly."""
        assert 'pulse_duration' in sample_config
        assert 'bandwidth' in sample_config
        assert sample_config['sample_rate'] > 0


class TestOFDMGeneration:
    """Test suite for OFDM signal generation."""

    def test_generate_ofdm_basic(self) -> None:
        """Test basic OFDM signal generation."""
        # Test implementation
        pass

    def test_generate_ofdm_with_phases(self) -> None:
        """Test OFDM generation with phase codes."""
        # Test implementation
        pass
```

### C. Test Organization

**Directory Structure:**
```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures
├── test_chirp.py        # Chirp signal tests
├── test_ofdm.py         # OFDM signal tests
├── test_optimization.py # Optimization tests
└── test_config.py       # Configuration tests
```

**Shared Fixtures (conftest.py):**
```python
"""Shared pytest fixtures for all tests."""

import pytest
from config import Settings


@pytest.fixture
def sample_config() -> dict:
    """Provide sample configuration for tests."""
    return Settings.to_dict()


@pytest.fixture
def sample_signal() -> cp.ndarray:
    """Generate sample signal for testing."""
    return cp.random.randn(1000) + 1j * cp.random.randn(1000)
```

### D. Running Tests (ALWAYS with `uv run`)

**CRITICAL: ALL test commands MUST use `uv run` prefix**

```bash
# ✅ CORRECT - Run all tests
uv run pytest

# ✅ CORRECT - Run with coverage
uv run pytest --cov=generate_codes --cov-report=html --cov-report=term

# ✅ CORRECT - Run specific test file
uv run pytest tests/test_chirp.py

# ✅ CORRECT - Run specific test
uv run pytest tests/test_chirp.py::TestChirpGeneration::test_generate_chirp_signal_basic

# ✅ CORRECT - Run with verbose output
uv run pytest -v

# ✅ CORRECT - Run tests matching pattern
uv run pytest -k "chirp"

# ✅ CORRECT - Run with coverage threshold
uv run pytest --cov=generate_codes --cov-fail-under=100

# ❌ WRONG - Missing uv run prefix
pytest                    # Don't do this
pytest --cov=.           # Don't do this
python -m pytest         # Don't do this
```

### E. Test Coverage Requirements

**100% coverage is MANDATORY:**

```bash
# Generate coverage report
uv run pytest --cov=generate_codes --cov-report=html --cov-report=term-missing

# View coverage in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

**Coverage must show:**
```
Name                        Stmts   Miss  Cover   Missing
---------------------------------------------------------
generate_codes/__init__.py     45      0   100%
generate_codes/chirp.py       123      0   100%
generate_codes/ofdm.py        234      0   100%
---------------------------------------------------------
TOTAL                         402      0   100%
```

### F. Test-Driven Development (TDD)

**ALWAYS write tests before or alongside code:**

1. **Write the test first:**
```python
def test_new_feature() -> None:
    """Test the new feature that doesn't exist yet."""
    result = new_feature(input_data)
    assert result == expected_output
```

2. **Run the test (it will fail):**
```bash
uv run pytest tests/test_new_feature.py
```

3. **Implement the feature:**
```python
def new_feature(input_data: Any) -> Any:
    """Implement the feature."""
    # Implementation
    return output
```

4. **Run the test again (it should pass):**
```bash
uv run pytest tests/test_new_feature.py
```

### G. Pytest Best Practices

**Use parametrize for multiple test cases:**
```python
@pytest.mark.parametrize(
    "input_value,expected_output",
    [
        (0, 0),
        (1, 1),
        (2, 4),
        (3, 9),
        (-1, 1),
    ]
)
def test_square_function(input_value: int, expected_output: int) -> None:
    """Test square function with multiple inputs."""
    assert square(input_value) == expected_output
```

**Use fixtures for setup:**
```python
@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file for testing."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("test content")
    return file_path


def test_file_reading(temp_file) -> None:
    """Test reading from file."""
    content = read_file(temp_file)
    assert content == "test content"
```

**Test exceptions properly:**
```python
def test_raises_value_error() -> None:
    """Test that function raises ValueError for invalid input."""
    with pytest.raises(ValueError, match="must be positive"):
        process_value(-1)
```

### H. Integration with CI/CD

**GitHub Actions must run tests:**
```yaml
# .github/workflows/ci.yml
- name: Run tests with coverage
  run: |
    uv run pytest --cov=. --cov-report=xml --cov-fail-under=100
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

---


## 13. Example: Compliant Function


Here's a complete example following all guidelines:


```python

"""
Signal correlation and analysis utilities.

This module provides functions for computing correlation metrics between
OFDM signals, used in codebook optimization.
"""

from typing import Tuple
import cupy as cp
from cupyx.optimizing import optimize
from config import Settings


def correlate_two_signals(
    sig1: cp.ndarray,
    sig2: cp.ndarray,
    normalize: bool = True
    ) -> Tuple[float, cp.ndarray]:
    """
    Compute cross-correlation between two complex signals.
    Calculates the full cross-correlation and returns both the peak
    correlation value and the complete correlation sequence.

    Args:
    sig1: First signal as complex CuPy array, shape (N,).
    sig2: Second signal as complex CuPy array, shape (N,).

    normalize: If True, normalize correlation by signal energies.
    Defaults to True.

    Returns:
    Tuple containing:

    - peak_correlation (float): Maximum absolute squared correlation value.
    - correlation_sequence (cp.ndarray): Full correlation sequence,
    shape (2*N-1,).

    Raises:
    ValueError: If sig1 and sig2 have different lengths.
    TypeError: If sig1 or sig2 are not CuPy arrays.

    Example:
    Compute correlation between two signals:

    >>> import cupy as cp
    >>> sig1 = cp.random.randn(1000) + 1j * cp.random.randn(1000)
    >>> sig2 = cp.random.randn(1000) + 1j * cp.random.randn(1000)
    >>> peak, xcorr = correlate_two_signals(sig1, sig2)
    >>> print(f"Peak correlation: {peak:.2e}")

    Peak correlation: 1.23e-02

    >>> print(f"Correlation shape: {xcorr.shape}")
    Correlation shape: (1999,)

    Notes:
    - Uses GPU acceleration for fast computation
    - Correlation computed via FFT (O(N log N) complexity)
    - Normalization uses L2 norm: xcorr / (||sig1|| * ||sig2||)
    - Returns squared magnitude for power metric

    See Also:
    peak_signal_to_noise_ratio: Related signal quality metric
    generate_chirp_ofdm_signal: Signal generation function

    Performance:
    For N=1000 samples, typical GPU execution time is ~0.5ms.
    Memory usage: O(N) for input signals + O(2N) for correlation.

    """

    # Type validation
    if not isinstance(sig1, cp.ndarray):
        raise TypeError(f"sig1 must be cp.ndarray, got {type(sig1)}")

    if not isinstance(sig2, cp.ndarray):
        raise TypeError(f"sig2 must be cp.ndarray, got {type(sig2)}")

    # Shape validation

    if sig1.shape != sig2.shape:
        raise ValueError(
            f"Signal shapes must match: sig1={sig1.shape}, sig2={sig2.shape}"
        )

    # Compute correlation using GPU optimization

    with optimize():
        xcorr: cp.ndarray = cp.correlate(sig1, sig2, mode='full')
        xcorr_power: cp.ndarray = cp.abs(xcorr) ** 2

    # Normalize if requested

    if normalize:
        sig1_energy: float = float(cp.sum(cp.abs(sig1) ** 2))
        sig2_energy: float = float(cp.sum(cp.abs(sig2) ** 2))
        if sig1_energy > 0 and sig2_energy > 0:
            normalization: float = sig1_energy * sig2_energy
            xcorr_power = xcorr_power / normalization
            peak_value: float = float(cp.max(xcorr_power))
    return peak_value, xcorr_power

```



---



## 14. Security & Dependency Management (MANDATORY)

### A. Automated Dependency Management

**Use uv (preferred) to manage and lock dependencies:**

```bash
# Install/sync dependencies
uv sync

# Add a new dependency
uv add package_name

# Update dependencies
uv lock --upgrade

# Verify dependency integrity
pip-audit --require-hashes
```

### B. Vulnerability Scanning & Security

**Mandatory security checks for ALL Python projects:**

1. **Vulnerability Scan**:
   ```bash
   # Scan for known vulnerabilities
   uv run pip-audit
   ```
   - Agents MUST fix all HIGH/CRITICAL vulnerabilities before delivery.

2. **Supply Chain Audit**:
   - Verify `uv.lock` / `requirements.txt` integrity
   - Audit licenses for compliance
   - Use `uv run pip-audit --require-hashes` for hash verification

### C. Dependency File

```toml
# pyproject.toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn>=0.27.0",
    "dynaconf>=3.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.2.0",
    "pip-audit>=2.7.0",
]
```

---

## 15. Quick Reference


### Command Cheat Sheet

**CRITICAL: ALL commands MUST use `uv` or `uv run` prefix**

```bash
# Package management
uv venv                              # Create virtual environment
uv add package                       # Install package
uv add package==1.2.3               # Install specific version
uv add -r requirements.txt          # Install from file
uv add --dev pytest ruff            # Install dev dependencies
uv pip freeze > requirements.txt    # Save dependencies

# Code quality (ALWAYS use uv run)
uv run ruff check .                 # Check code
uv run ruff check --fix .           # Auto-fix issues
uv run ruff format .                # Format code
uv run mypy .                       # Type checking (optional)

# Testing (ALWAYS use uv run - MANDATORY)
uv run pytest                       # Run all tests
uv run pytest -v                    # Run with verbose output
uv run pytest --cov=.               # Run with coverage
uv run pytest --cov=. --cov-fail-under=100  # Enforce 100% coverage
uv run pytest tests/test_module.py  # Run specific test file
uv run pytest -k "test_name"        # Run tests matching pattern

# Running Python scripts (ALWAYS use uv run)
uv run main.py                      # Run main script
uv run python -m module             # Run as module

# Documentation generation (ALWAYS use uv run)
uv run python -m pydoc module_name              # View module docs in terminal
uv run python -m pydoc module.function          # View function docs
uv run python -m pydoc -w module_name           # Generate HTML documentation
uv run python -m pydoc -b                       # Start interactive doc browser

# Configuration
export ENV_FOR_DYNACONF=development  # Set environment
uv run python -c "from config import Settings; print(Settings.to_dict())"  # View config
```

**❌ NEVER do this:**
```bash
pytest                              # Missing uv run
python main.py                      # Missing uv run
pip install package                 # Use uv add instead
ruff check .                        # Missing uv run
```


### Validation Checklist

**Before committing code, ALL items must be checked:**

#### Agent-Generated Code (If Applicable)
- [ ] Agent verified Python syntax is valid
- [ ] Agent ran `uv run pytest` and all tests passed
- [ ] Agent fixed any errors before presenting code

#### Package Management
- [ ] Using `uv` for all package operations
- [ ] All commands use `uv run` prefix
- [ ] No direct `pip`, `pytest`, or `python` commands

#### Configuration
- [ ] All config in TOML files (Dynaconf)
- [ ] No hardcoded values in source code
- [ ] Configuration validated on load

#### Documentation
- [ ] Every function has complete docstring (Google style)
- [ ] Every class has complete docstring
- [ ] Module has docstring
- [ ] All parameters have type hints
- [ ] All return values have type hints
- [ ] pydoc can generate documentation: `uv run python -m pydoc module_name`
- [ ] Generated HTML documentation is complete: `uv run python -m pydoc -w module_name`

#### Code Quality
- [ ] `uv run ruff check .` passes with no errors
- [ ] `uv run ruff format .` applied
- [ ] Code follows all style guidelines
- [ ] Comprehensions used instead of manual loops where appropriate
- [ ] Generator expressions used for large datasets
- [ ] No unnecessary list() around generators in sum/max/min
- [ ] Functional programming style preferred (pure functions, immutability)
- [ ] No mutable default arguments
- [ ] Functions avoid side effects where possible

#### Testing (MANDATORY)
- [ ] Tests written for ALL new functions/classes
- [ ] Tests pass: `uv run pytest` returns 0
- [ ] Coverage is 100%: `uv run pytest --cov=. --cov-fail-under=100`
- [ ] All tests use `uv run pytest` prefix
- [ ] Test names are descriptive

#### Final Verification
- [ ] Code runs without errors: `uv run main.py`
- [ ] All dependencies properly specified
- [ ] No syntax errors (code parses correctly)


---


## 16. Enforcement

### Automated Checks

These checks run automatically on every commit/PR (ALL with `uv run`):

1. **Pre-commit hooks** - Local validation before commit
2. **GitHub Actions CI** - Server-side validation on push
3. **Ruff checks** - Type hints, formatting, style (`uv run ruff check`)
4. **Pytest** - Unit and integration tests (`uv run pytest`)
5. **Coverage checks** - 100% coverage required (`uv run pytest --cov-fail-under=100`)
6. **Syntax validation** - Code must parse without errors

### Manual Review

Code reviewers will verify:
- Docstring completeness and quality
- Type hint correctness and coverage
- Configuration management practices
- Testing coverage and quality (100% required)
- All commands use `uv run` prefix
- Tests exist for all new code
- Agent-generated code was verified before submission

### Violations

Code that violates these guidelines will be **rejected** with feedback:
- Missing/incomplete docstrings → Request documentation
- Missing type hints → Request type annotations
- Hardcoded config values → Request Dynaconf migration
- Not using `uv` → Request dependency management fix
- Not using `uv run` prefix → Request command correction
- Failing Ruff checks → Request code quality fixes
- Missing tests → Request test implementation
- Tests not passing → Fix code until tests pass
- Coverage < 100% → Add tests for uncovered code
- Syntax errors → Fix code to parse correctly
- Agent code not verified → Re-verify with proper checks

---

## 17. Deployment Checklist

### Build & Syntax
- [ ] Code parses: `uv run python -m py_compile *.py` returns exit 0
- [ ] Ruff passes: `uv run ruff check .` returns exit 0
- [ ] Code formatted: `uv run ruff format --check .` returns exit 0
- [ ] Type checking passes: `uv run mypy .` returns no errors

### Testing
- [ ] All tests pass: `uv run pytest` returns exit 0
- [ ] Coverage at 100%: `uv run pytest --cov --cov-fail-under=100`
- [ ] No skipped or xfail tests without justification
- [ ] Integration tests pass against staging environment

### Security
- [ ] Dependencies scanned: `uv run pip-audit` reports no vulnerabilities
- [ ] No hardcoded secrets or API keys in source
- [ ] Dynaconf used for all configuration values
- [ ] No use of `eval()`, `exec()`, or `pickle.loads()` on untrusted input

### Agent Workflow
- [ ] Agent-generated code was syntax-checked before delivery
- [ ] Agent-generated code was tested with `uv run pytest`
- [ ] All docstrings present (Google style) with type hints
- [ ] `uv run` prefix used for every command

---

## 18. Why This Configuration Works

1. **UV-Only Dependency Management**: Using `uv` exclusively eliminates version conflicts between pip, pipenv, and poetry. A single lockfile ensures reproducible builds across development, CI, and production environments.

2. **Ruff as a Unified Linter and Formatter**: Ruff replaces flake8, isort, black, and dozens of other tools with a single Rust-based binary. This reduces CI time by an order of magnitude while enforcing consistent style and catching real bugs.

3. **100% Test Coverage with Pytest**: Mandatory full coverage ensures every code path is exercised, preventing regressions and giving developers confidence to refactor. Combined with TDD, this catches bugs at write time rather than deploy time.

4. **Dynaconf for Configuration**: Externalizing all configuration prevents hardcoded values from reaching production, supports environment-specific overrides, and makes secrets management straightforward.

5. **Google-Style Docstrings with Type Hints**: Combining runtime documentation with static type annotations enables IDE autocompletion, automated API doc generation with pydoc, and early detection of type errors via mypy.

---

## Resources

- [UV Documentation](https://github.com/astral-sh/uv)
- [Dynaconf Documentation](https://www.dynaconf.com/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Python Type Hints (PEP 484)](https://peps.python.org/pep-0484/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Pytest Documentation](https://docs.pytest.org/)


---

## Summary

This guide enforces:
1. **UV-only dependency management** - All operations through `uv`, all commands with `uv run`
2. **100% test coverage** - Every function must have tests that pass
3. **Agent verification** - AI-generated code must be syntax-checked and tested
4. **Complete documentation** - Google-style docstrings with type hints, pydoc-generated API documentation
5. **Dynaconf configuration** - No hardcoded values
6. **Pythonic comprehensions** - Prefer list/set/dict/generator comprehensions for performance and clarity
7. **Functional programming** - Prefer pure functions, immutability, and functional patterns whenever applicable
8. **Ruff code quality** - Clean, formatted, type-safe code
9. **Generated documentation** - API and functional documentation must be generated with pydoc

**Failure to follow these guidelines will result in code rejection.**

---

**Last Updated:** 2026-01-15
**Version:** 2.0
**Maintainer:** Development Team

---

**End of Python Development Guidelines**
