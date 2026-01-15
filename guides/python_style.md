# Python Development Guidelines
This document provides mandatory coding standards and development practices for AI agents and human developers working on this Python project.

## Core Principles
All code contributions **MUST** adhere to these guidelines. Non-compliant code will be rejected during review.

### Mandatory Requirements
- **UV Only**: All package management through `uv`
- **Dynaconf**: All configuration externalized to TOML files
- **Type Hints**: Strict typing on all functions and classes
- **Documentation**: Complete docstrings (Google style) for all code
- **Comprehensions**: Prefer list/set/dict/generator comprehensions for performance and memory efficiency
- **Testing**: 100% test coverage with pytest, all tests must pass
- **Code Quality**: Ruff checks must pass without errors
- **Agent Verification**: AI-generated code must be syntax-checked and tested before delivery

---

## 1. Agent Code Generation Requirements (MANDATORY)

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

Always verify first, fix issues, then present the working solution.

---

## 2. Package Management: UV Only
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

## 3. Configuration Management: Dynaconf

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

# ...

# Access via CONF dictionary

def alternative_access() -> None:

"""Alternative configuration access pattern."""
sample_rate: int = CONF['sample_rate']
bandwidth: float = CONF['bandwidth']

# ...

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
## 4. Documentation: Comprehensive PyDoc

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
...

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

...

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


---

## 5. Type Hints: Strict Typing
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

...


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

## 6. Pythonic Code Patterns: Comprehensions (MANDATORY)

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


## 7. Code Quality Enforcement


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

## 8. Project Structure

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


## 9. Development Workflow

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

## 10. Testing Requirements (MANDATORY)

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


## 11. Example: Compliant Function


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



## 12. Quick Reference


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

#### Code Quality
- [ ] `uv run ruff check .` passes with no errors
- [ ] `uv run ruff format .` applied
- [ ] Code follows all style guidelines
- [ ] Comprehensions used instead of manual loops where appropriate
- [ ] Generator expressions used for large datasets
- [ ] No unnecessary list() around generators in sum/max/min

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


## 13. Enforcement

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
4. **Complete documentation** - Google-style docstrings with type hints
5. **Dynaconf configuration** - No hardcoded values
6. **Pythonic comprehensions** - Prefer list/set/dict/generator comprehensions for performance and clarity
7. **Ruff code quality** - Clean, formatted, type-safe code

**Failure to follow these guidelines will result in code rejection.**

---

**Last Updated:** 2026-01-15
**Version:** 2.0
**Maintainer:** Development Team
