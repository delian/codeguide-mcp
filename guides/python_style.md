# Python Development Guidelines
This document provides mandatory coding standards and development practices for AI agents and human developers working on this Python project.
## Core Principles
All code contributions **MUST** adhere to these guidelines. Non-compliant code will be rejected during review.

---
## 1. Package Management: UV Only
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

## 2. Configuration Management: Dynaconf

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
## 3. Documentation: Comprehensive PyDoc

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

## 4. Type Hints: Strict Typing
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


## 5. Code Quality Enforcement


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
    uv pip install -r requirements.txt
    uv pip install ruff pytest
- name: Run Ruff checks
  run: uv run ruff check .
- name: Run Ruff format check
  run: uv run ruff format --check .
- name: Run tests
  run: uv run pytest

```


---

## 6. Project Structure

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


## 7. Development Workflow

### Starting a New Feature

```bash
# 1. Create feature branch
git checkout -b feature/new-signal-type

# 2. Ensure virtual environment is active
source .venv/bin/activate

# 3. Install/update dependencies
uv add -r requirements.txt

# 4. Make changes with proper type hints and docstrings

# 5. Run quality checks
uv run ruff check --fix .
uv run ruff format .

# 6. Run tests
uv run pytest

# 7. Commit and push
git add .
git commit -m "feat: add new signal type with proper typing"
git push origin feature/new-signal-type

```


### Code Review Checklist

Before submitting a PR, verify:
- [ ] All functions have complete docstrings (Google style)
- [ ] All parameters have type hints
- [ ] All return values have type hints
- [ ] Configuration values use Dynaconf (no hardcoded values)
- [ ] `uv run ruff check .` passes with no errors
- [ ] `uv run ruff format .` has been applied
- [ ] All tests pass (`uv run pytest`)
- [ ] New functionality has tests
- [ ] Configuration changes documented in `defaults.toml`

---

## 8. Testing Requirements

### Test File Structure

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

    @pytest.fixture
    def sample_config(self) -> dict:
        """Provide sample configuration for tests."""

        return {
            'pulse_duration': 50e-6,
            'bandwidth': 5e6,
            'sample_rate': 20e6,
            'num_subcarriers': 250,
            }

    def test_config_loading(sample_config: dict) -> None:
        """Test that configuration loads correctly."""
        assert 'pulse_duration' in sample_config
        assert 'bandwidth' in sample_config
        assert sample_config['sample_rate'] > 0

```



### Running Tests


```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=generate_codes --cov-report=html

# Run specific test file
uv run pytest tests/test_chirp.py

# Run specific test
uv run pytest tests/test_chirp.py::TestChirpGeneration::test_generate_chirp_signal_basic

# Run with verbose output
uv run pytest -v

# Run tests matching pattern
uv run pytest -k "chirp"

```

---


## 9. Example: Compliant Function


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



## 10. Quick Reference


### Command Cheat Sheet


```bash
# Package management
uv venv # Create virtual environment
uv add package # Install package
uv add -r requirements.txt # Install from file
uv pip freeze > requirements.txt # Save dependencies

# Code quality

uv run ruff check . # Check code
uv run ruff check --fix . # Auto-fix issues
uv run ruff format . # Format code

# Testing

uv run pytest # Run tests
uv run pytest --cov=. # Run with coverage



# Configuration
export ENV_FOR_DYNACONF=development # Set environment
python -c "from config import Settings; print(Settings.to_dict())" # View config

```


### Validation Checklist


Before committing code, ensure:

- [ ] Using `uv` for all package operations
- [ ] All config in TOML files (Dynaconf)
- [ ] Every function has complete docstring
- [ ] All parameters have type hints
- [ ] `uv run ruff check .` passes
- [ ] `uv run ruff format .` applied
- [ ] Tests written for new code
- [ ] Tests pass: `uv run pytest`


---


## 11. Enforcement

### Automated Checks

These checks run automatically on every commit/PR:

1. **Pre-commit hooks** - Local validation before commit
2. **GitHub Actions CI** - Server-side validation on push
3. **Ruff checks** - Type hints, formatting, style
4. **Pytest** - Unit and integration tests

### Manual Review

Code reviewers will verify:
- Docstring completeness and quality
- Type hint correctness and coverage
- Configuration management practices
- Testing coverage and quality

### Violations

Code that violates these guidelines will be **rejected** with feedback:
- Missing/incomplete docstrings → Request documentation
- Missing type hints → Request type annotations
- Hardcoded config values → Request Dynaconf migration
- Not using `uv` → Request dependency management fix
- Failing Ruff checks → Request code quality fixes

---

## Resources

- [UV Documentation](https://github.com/astral-sh/uv)
- [Dynaconf Documentation](https://www.dynaconf.com/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Python Type Hints (PEP 484)](https://peps.python.org/pep-0484/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Pytest Documentation](https://docs.pytest.org/)


---

**Last Updated:** 2024-11-03
**Version:** 1.0
**Maintainer:** Development Team
