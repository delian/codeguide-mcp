# PyTorch Development Guidelines
Mandatory coding standards and best practices for modern PyTorch development. Type-safe, reproducible, GPU-optimized, production-ready deep learning. Python 3.11+, PyTorch 2.x, torch.compile, mixed precision, distributed training, safetensors, pytest, ruff.

---

**Agent Profile**: The PyTorch Expert
**Role**: Senior Deep Learning Engineer & Production ML Specialist
**Objective**: Generate production-ready, reproducible, GPU-optimized, and secure PyTorch code.
**Tools**: Python 3.11+, PyTorch 2.x, torch.compile, CUDA/XPU, uv, pytest, ruff, torchmetrics, safetensors, torch.profiler.

---

## 1. Core Philosophies: TENSOR-FIRST

The agent must adhere to the **TENSOR-FIRST** principles for every PyTorch implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Security-First**: NEVER use `torch.load()` without `weights_only=True`. Prefer safetensors for all model serialization.

- **T**orch.compile by Default: Use `torch.compile` for all production models — inference and training.
- **E**ager-First Development: Develop and debug in eager mode, compile for production.
- **N**ative Mixed Precision: Use `torch.autocast` and `GradScaler` — never manually cast dtypes.
- **S**afe Serialization: Use safetensors or `weights_only=True` — pickle-based formats are attack vectors.
- **O**ptimize Data Pipeline: `DataLoader` with `num_workers>0`, `pin_memory=True`, `persistent_workers=True`.
- **R**eproducible by Default: Seed all RNGs, use deterministic algorithms, version everything.

**Additional Principles:**

- **Profile Before Optimizing**: Use `torch.profiler` — never guess where bottlenecks are.
- **Minimize Host-Device Transfers**: Keep tensors on GPU. Every `.cpu()`, `.item()`, or `.numpy()` call is a synchronization point.
- **Immutable Configs**: Use frozen dataclasses or Dynaconf for hyperparameters — never magic numbers in code.
- **Checkpoint Everything**: Save model, optimizer, scheduler, scaler, epoch, and RNG states.
- **Fail Loudly**: Shape mismatches, NaN losses, and gradient explosions must raise, never silently continue.

**Verified Code**: Agent-generated PyTorch code MUST pass syntax checks, tests, shape validation, and security scans before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated PyTorch code is correct, reproducible, and safe before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY PyTorch code, the agent MUST:**

1. **Syntax and Type Checking**:
   ```bash
   # Verify syntax
   uv run python -m py_compile model.py

   # Lint and type-check
   uv run ruff check .
   uv run ruff format --check .
   ```
   - **MUST** compile with zero errors
   - **MUST** pass ruff with zero warnings
   - All type hints present on public functions

2. **Shape Validation**:
   ```python
   # Verify tensor shapes through the model
   model = MyModel()
   x = torch.randn(2, 3, 224, 224)  # batch, channels, height, width
   out = model(x)
   assert out.shape == (2, num_classes), f"Expected (2, {num_classes}), got {out.shape}"
   ```
   - **MUST** validate input/output shapes with dummy data
   - **MUST** verify batch dimension is preserved
   - **MUST** check shapes at every layer boundary in complex architectures

3. **Test Execution**:
   ```bash
   # Run all tests
   uv run pytest tests/ -v

   # Run with coverage
   uv run pytest --cov=src --cov-fail-under=80
   ```
   - **MUST** pass all existing tests
   - **MUST** include tests for new modules
   - Coverage MUST NOT decrease

4. **Security Verification (CRITICAL)**:
   ```bash
   # Scan for unsafe deserialization
   uv run ruff check --select S301,S302 .  # pickle-related rules

   # Verify no torch.load without weights_only=True
   grep -rn "torch.load" . --include="*.py" | grep -v "weights_only=True"
   # MUST return empty — all torch.load calls must use weights_only=True
   ```
   - **MUST** use `weights_only=True` in ALL `torch.load()` calls
   - **MUST** prefer safetensors over pickle-based formats
   - **MUST** have zero hardcoded secrets or API keys

5. **Reproducibility Check**:
   ```python
   # Verify all seeds are set
   import torch, random, numpy as np
   def set_seed(seed: int = 42) -> None:
       random.seed(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       if torch.cuda.is_available():
           torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
   ```
   - **MUST** include seed-setting in all training scripts
   - **MUST** log all hyperparameters
   - **MUST** pin dependency versions

#### Error Correction Process

If verification fails:

1. **Shape Mismatch**:
   - Trace tensor shapes through each layer
   - Print intermediate shapes with a forward hook
   - Fix dimension mismatches
   - Re-verify with multiple batch sizes (1, 2, 8)

2. **NaN/Inf in Loss**:
   - Check for division by zero in custom losses
   - Enable anomaly detection: `torch.autograd.set_detect_anomaly(True)`
   - Verify gradient scaling with mixed precision
   - Check learning rate and initialization

3. **CUDA Out of Memory**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision (float16/bfloat16)
   - Check for tensor accumulation in training loop (detach losses)

4. **Compilation Failure (torch.compile)**:
   - Identify graph breaks with `TORCH_LOGS="graph_breaks"`
   - Replace unsupported operations
   - Fall back to eager for debugging

### B. Agent Workflow Example

**Complete PyTorch generation workflow:**

1. **Generate Code Structure**:
   ```
   project/
   ├── src/
   │   ├── model.py          # Model architecture
   │   ├── dataset.py         # Data loading
   │   ├── train.py           # Training loop
   │   ├── evaluate.py        # Evaluation
   │   └── utils.py           # Utilities
   ├── tests/
   │   ├── test_model.py
   │   ├── test_dataset.py
   │   └── test_train.py
   ├── configs/
   │   └── defaults.toml
   └── pyproject.toml
   ```

2. **Write Tests First (TDD)**:
   ```python
   def test_model_forward_shape():
       model = MyModel(num_classes=10)
       x = torch.randn(4, 3, 224, 224)
       out = model(x)
       assert out.shape == (4, 10)
   ```

3. **Implement Model**:
   ```python
   class MyModel(nn.Module):
       ...
   ```

4. **Verify**:
   ```bash
   uv run pytest tests/test_model.py -v
   uv run ruff check .
   # ✓ All checks pass
   ```

5. **Present Code**: Only after ALL checks pass

### C. Prohibited Practices

**NEVER deliver PyTorch code that:**
- [ ] Uses `torch.load()` without `weights_only=True`
- [ ] Uses pickle for model serialization (use safetensors)
- [ ] Has no seed setting for reproducibility
- [ ] Accumulates loss tensors without `.item()` or `.detach()`
- [ ] Calls `.cpu()` or `.numpy()` inside a training loop
- [ ] Uses `model.eval()` without `torch.no_grad()` for inference
- [ ] Has hardcoded magic numbers instead of config values
- [ ] Lacks shape validation in forward methods
- [ ] Uses deprecated APIs (`torch.cuda.amp.autocast` → `torch.autocast`)
- [ ] **Fixes bugs without adding regression tests first**
- [ ] **Writes implementation before writing tests (violates TDD)**
- [ ] **Skips Red-Green-Refactor cycle for new features**

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new PyTorch code.**

### TDD Cycle

```
1. RED: Write a failing test (shape check, loss convergence, metric threshold)
   ↓
2. GREEN: Write minimal model/training code to make it pass
   ↓
3. REFACTOR: Optimize (compile, fuse ops, reduce memory) while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for PyTorch

```python
# Step 1: RED — Write failing test first
import torch
import pytest

def test_classifier_output_shape() -> None:
    """Test that classifier produces correct output shape."""
    from src.model import ImageClassifier
    model = ImageClassifier(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)

def test_classifier_output_probabilities() -> None:
    """Test that softmax output sums to 1."""
    from src.model import ImageClassifier
    model = ImageClassifier(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    probs = torch.softmax(model(x), dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

# Run: uv run pytest tests/test_model.py
# FAILS — ImageClassifier doesn't exist yet

# Step 2: GREEN — Write minimal implementation
import torch.nn as nn

class ImageClassifier(nn.Module):
    """Simple image classifier."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

# Run: uv run pytest tests/test_model.py
# PASSES

# Step 3: REFACTOR — Add torch.compile, keep tests green
compiled_model = torch.compile(ImageClassifier(num_classes=10))
# Tests still pass
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug in PyTorch code MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. Bug Discovered (e.g., NaN loss, shape error, wrong metric)
   ↓
2. Write a test that REPRODUCES the bug (test FAILS)
   ↓
3. Verify the test fails for the right reason
   ↓
4. Fix the bug (make the test pass)
   ↓
5. Verify no regressions
   ↓
6. Document the root cause
```

### Example Bug Fix

```python
# Bug Report #42: Model produces NaN loss with large learning rate

# Step 1-2: Write test that reproduces the bug
def test_no_nan_loss_large_lr_bug_42() -> None:
    """Regression test: NaN loss with lr=1.0 — Bug #42.

    Root cause: Missing gradient clipping caused exploding gradients.
    """
    from src.model import ImageClassifier
    from src.train import train_one_epoch

    model = ImageClassifier(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    dummy_loader = create_dummy_loader(batch_size=4, num_batches=5)

    loss = train_one_epoch(model, dummy_loader, optimizer)
    assert not torch.isnan(torch.tensor(loss)), f"Loss is NaN: {loss}"
    assert not torch.isinf(torch.tensor(loss)), f"Loss is Inf: {loss}"

# Run: uv run pytest — FAILS (reproduces NaN)

# Step 3: Fix — add gradient clipping
def train_one_epoch(model, loader, optimizer, max_grad_norm: float = 1.0):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # FIX
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Run: uv run pytest — PASSES
```

---

## 3. Project Structure & Organization (MANDATORY)

### A. Standard PyTorch Project Layout

```
project/
├── src/
│   ├── __init__.py
│   ├── model.py               # nn.Module definitions
│   ├── layers.py              # Custom layers/blocks
│   ├── dataset.py             # Dataset and DataLoader setup
│   ├── transforms.py          # Data augmentation / preprocessing
│   ├── train.py               # Training loop
│   ├── evaluate.py            # Evaluation / inference
│   ├── losses.py              # Custom loss functions
│   ├── metrics.py             # Custom metrics (or use torchmetrics)
│   └── utils.py               # Seed setting, checkpointing, logging
├── tests/
│   ├── conftest.py            # Shared fixtures (seed, dummy data, devices)
│   ├── test_model.py          # Model shape/output tests
│   ├── test_dataset.py        # Dataset loading/transform tests
│   ├── test_train.py          # Training step/loop tests
│   └── test_losses.py         # Loss function correctness tests
├── configs/
│   ├── defaults.toml          # Default hyperparameters
│   └── experiment_001.toml    # Experiment overrides
├── scripts/
│   ├── train.sh               # Training launch script
│   └── eval.sh                # Evaluation launch script
├── checkpoints/               # Saved model checkpoints (git-ignored)
├── logs/                      # TensorBoard / W&B logs (git-ignored)
├── pyproject.toml             # Project metadata, ruff, pytest config
├── AGENTS.md                  # AI agent instructions
└── README.md
```

### B. Module Separation Principles

1. **Model vs. System**: Keep `nn.Module` definitions pure — no training logic, no data loading, no logging inside model files.

2. **Config vs. Code**: Hyperparameters live in config files (Dynaconf/TOML), never hardcoded:
   ```python
   # BAD
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

   # GOOD
   from config import Settings
   optimizer = torch.optim.Adam(model.parameters(), lr=Settings.learning_rate)
   ```

3. **One Module per File**: Complex architectures split into `model.py` (top-level), `layers.py` (reusable blocks), `losses.py` (custom objectives).

---

## 4. Model Architecture Patterns (MANDATORY)

### A. Standard nn.Module Pattern

```python
import torch
import torch.nn as nn
from torch import Tensor


class ResidualBlock(nn.Module):
    """Residual block with pre-activation BatchNorm.

    Args:
        channels: Number of input and output channels.
        dropout: Dropout probability. Defaults to 0.1.
    """

    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H, W).
        """
        return x + self.block(x)
```

**Rules for nn.Module:**
- Always type-hint `forward()` with `Tensor` input/output
- Document expected shapes in docstrings: `(B, C, H, W)`
- Use `bias=False` before BatchNorm (BN absorbs the bias)
- Prefer `nn.Sequential` for linear chains
- Initialize weights explicitly for custom layers

### B. Weight Initialization

```python
def init_weights(module: nn.Module) -> None:
    """Initialize model weights.

    Uses Kaiming normal for Conv/Linear layers, constant for BatchNorm.

    Args:
        module: Module to initialize.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# Apply to model
model = MyModel()
model.apply(init_weights)
```

### C. Compile-Friendly Patterns

**Write models that work with `torch.compile`:**

```python
# GOOD — torch.compile friendly
class CompileFriendlyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(256, 256)
        self.norm = nn.LayerNorm(256)

    def forward(self, x: Tensor) -> Tensor:
        # Static control flow — no graph breaks
        return self.norm(self.linear(x))


# BAD — causes graph breaks
class CompileUnfriendlyModel(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if x.shape[0] > 32:          # Data-dependent control flow
            return self.big_branch(x)
        else:
            return self.small_branch(x)
        # Also bad: print(), breakpoint(), data-dependent Python ops
```

**torch.compile modes:**

| Mode | Use Case | Compile Time | Runtime |
|------|----------|-------------|---------|
| `"default"` | Balanced | Medium | Good |
| `"reduce-overhead"` | Small models, latency-sensitive | Low | Good |
| `"max-autotune"` | Large models, throughput | High | Best |

```python
# Production inference
model = torch.compile(model, mode="max-autotune")

# Quick iteration during development
model = torch.compile(model, mode="reduce-overhead")

# Full graph capture (no graph breaks allowed)
model = torch.compile(model, fullgraph=True)
```

---

## 5. Training Loop Patterns (MANDATORY)

### A. Standard Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration — immutable."""

    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    use_compile: bool = True
    compile_mode: str = "default"
    mixed_precision: bool = True
    device: str = "cuda"


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
) -> dict[str, float]:
    """Train model with modern PyTorch best practices.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration.

    Returns:
        Dictionary with final training and validation metrics.
    """
    device = torch.device(config.device)
    model = model.to(device)

    # Compile for production
    if config.use_compile:
        model = torch.compile(model, mode=config.compile_mode)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=config.mixed_precision)

    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=config.mixed_precision,
            ):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()  # .item() detaches from graph

        scheduler.step()

        # --- Validation ---
        val_loss = evaluate(model, val_loader, criterion, device, config)

        # --- Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss)

    return {"train_loss": train_loss / len(train_loader), "val_loss": best_val_loss}
```

### B. Evaluation Pattern

```python
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: TrainConfig,
) -> float:
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate.
        loader: Data loader.
        criterion: Loss function.
        device: Target device.
        config: Training configuration.

    Returns:
        Average loss over the dataset.
    """
    model.eval()
    total_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=config.mixed_precision,
        ):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item()

    return total_loss / len(loader)
```

### C. Key Training Loop Rules

| Practice | Why |
|----------|-----|
| `optimizer.zero_grad(set_to_none=True)` | Faster than setting to zero — avoids memset |
| `loss.item()` for logging | Prevents GPU memory leak from graph accumulation |
| `.to(device, non_blocking=True)` | Overlaps data transfer with computation |
| `torch.autocast` context manager | Modern mixed precision API (replaces `torch.cuda.amp.autocast`) |
| `scaler.unscale_()` before clip | Required for correct gradient clipping with AMP |
| `model.eval()` + `@torch.no_grad()` | Both are needed — `eval()` changes BN/Dropout, `no_grad()` saves memory |

---

## 6. Data Loading & Pipelines (MANDATORY)

### A. Optimized DataLoader Configuration

```python
from torch.utils.data import DataLoader, Dataset
import os


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int | None = None,
) -> DataLoader:
    """Create an optimized DataLoader.

    Args:
        dataset: Dataset to load.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of worker processes. Defaults to min(8, cpu_count).

    Returns:
        Configured DataLoader.
    """
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,               # Faster host→device transfer
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,                 # Consistent batch sizes for compile
    )
```

### B. Custom Dataset Pattern

```python
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torch import Tensor
import torchvision.transforms.v2 as T


class ImageDataset(Dataset):
    """Image classification dataset.

    Args:
        root: Path to image directory.
        transform: Image transforms to apply.
    """

    def __init__(
        self,
        root: Path,
        transform: T.Compose | None = None,
    ) -> None:
        self.root = Path(root)
        self.samples = sorted(self.root.glob("**/*.jpg"))
        self.transform = transform or T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        path = self.samples[idx]
        image = Image.open(path).convert("RGB")
        label = self._extract_label(path)
        return self.transform(image), label

    def _extract_label(self, path: Path) -> int:
        """Extract class label from directory structure."""
        return int(path.parent.name)
```

### C. Data Loading Rules

| Rule | Reason |
|------|--------|
| `num_workers = min(8, cpu_count)` | More workers than CPUs causes contention |
| `pin_memory=True` | Enables async CUDA transfers |
| `persistent_workers=True` | Avoids worker restart overhead each epoch |
| `drop_last=True` for training | Prevents small last batch (helps torch.compile) |
| `drop_last=False` for validation | Evaluate on all samples |
| Use `torchvision.transforms.v2` | Modern API, composable, supports bboxes/masks |
| `non_blocking=True` on `.to(device)` | Overlaps transfer with compute |

---

## 7. Mixed Precision Training (MANDATORY)

### A. Standard AMP Pattern

```python
# Modern API (PyTorch 2.x)
device = torch.device("cuda")

# Use bfloat16 on Ampere+ GPUs, float16 on older
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))
# bfloat16 doesn't need GradScaler (wider dynamic range)

with torch.autocast(device_type="cuda", dtype=dtype):
    output = model(inputs)
    loss = criterion(output, targets)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

### B. Precision Selection Guide

| Precision | GPU Requirement | GradScaler | Best For |
|-----------|----------------|------------|----------|
| `torch.float32` | Any | No | Debugging, numeric precision |
| `torch.float16` | Volta+ (V100) | Yes | Training with AMP |
| `torch.bfloat16` | Ampere+ (A100) | No | Large model training, LLMs |

### C. Anti-Patterns

```python
# BAD — manual casting
x = x.half()  # Don't do this
output = model(x.float())  # Don't cast manually

# GOOD — let autocast handle it
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(x)  # Autocast selects dtype per operation

# BAD — backward inside autocast
with torch.autocast(device_type="cuda"):
    loss = criterion(output, targets)
    loss.backward()  # Backward under autocast is NOT recommended

# GOOD — backward outside autocast
with torch.autocast(device_type="cuda"):
    loss = criterion(output, targets)
loss.backward()  # Or scaler.scale(loss).backward()
```

---

## 8. Checkpointing & Serialization (MANDATORY)

### A. Safe Checkpoint Pattern

```python
from pathlib import Path
from safetensors.torch import save_file, load_file


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    val_loss: float,
    path: Path = Path("checkpoints/best.pt"),
) -> None:
    """Save training checkpoint with full state.

    Args:
        model: Trained model.
        optimizer: Optimizer state.
        scheduler: LR scheduler state.
        scaler: GradScaler state.
        epoch: Current epoch number.
        val_loss: Validation loss at this checkpoint.
        path: Path to save checkpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save model weights with safetensors (secure, no pickle)
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    save_file(model_to_save.state_dict(), path.with_suffix(".safetensors"))

    # Save training state (optimizer, scheduler, etc.)
    torch.save(
        {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_loss": val_loss,
            "rng_state": torch.random.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    path: Path = Path("checkpoints/best.pt"),
    device: str = "cuda",
) -> int:
    """Load training checkpoint.

    Args:
        model: Model to load weights into.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        scaler: Optional GradScaler to restore.
        path: Path to checkpoint.
        device: Device to load tensors to.

    Returns:
        Epoch number from checkpoint.
    """
    # Load model weights (safetensors — safe, no code execution)
    state_dict = load_file(path.with_suffix(".safetensors"), device=device)
    model.load_state_dict(state_dict)

    # Load training state
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint["epoch"]
```

### B. Serialization Security Rules

| Format | Security | Speed | Use Case |
|--------|----------|-------|----------|
| **safetensors** | Safe (no code exec) | Fast | Model weights (PREFERRED) |
| **torch.save + weights_only=True** | Safe (PyTorch 2.6+) | Medium | Training state |
| **torch.save (default)** | UNSAFE (RCE via pickle) | Medium | NEVER use |
| **pickle** | UNSAFE | Medium | NEVER use for models |

```python
# GOOD — safetensors for model weights
from safetensors.torch import save_file, load_file
save_file(model.state_dict(), "model.safetensors")
state_dict = load_file("model.safetensors")

# GOOD — weights_only for training state
torch.save(training_state, "state.pt")
state = torch.load("state.pt", weights_only=True)

# BAD — NEVER do this (CVE-2025-32434)
state = torch.load("state.pt")  # Arbitrary code execution risk!
```

---

## 9. Distributed Training (MANDATORY)

### A. DDP (Data-Parallel) Pattern

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def setup_distributed(rank: int, world_size: int) -> None:
    """Initialize distributed training.

    Args:
        rank: Process rank.
        world_size: Total number of processes.
    """
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)


def train_distributed(rank: int, world_size: int) -> None:
    """Distributed training entry point.

    Args:
        rank: Process rank.
        world_size: Total number of processes.
    """
    setup_distributed(rank, world_size)

    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Compile AFTER wrapping with DDP
    ddp_model = torch.compile(ddp_model)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, sampler=sampler, batch_size=32)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # CRITICAL: shuffle differently each epoch
        train_one_epoch(ddp_model, loader)

    dist.destroy_process_group()


# Launch with torchrun (PREFERRED over mp.spawn)
# torchrun --nproc_per_node=4 train.py
```

### B. FSDP2 (Fully Sharded Data Parallel) Pattern

**Use FSDP2 for models that don't fit on a single GPU:**

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

def train_fsdp(rank: int, world_size: int) -> None:
    """FSDP2 training — shard model across GPUs.

    Args:
        rank: Process rank.
        world_size: Total number of processes.
    """
    setup_distributed(rank, world_size)

    model = LargeModel().to(rank)

    # FSDP2 mixed precision policy
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    # Shard each transformer block individually
    for layer in model.transformer_blocks:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)

    # Compile AFTER sharding
    model = torch.compile(model)

    # Training loop — standard after setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for epoch in range(num_epochs):
        train_one_epoch(model, loader, optimizer)

    dist.destroy_process_group()
```

### C. Distributed Training Rules

| Rule | Details |
|------|---------|
| Use `torchrun` for launching | `torchrun --nproc_per_node=N script.py` — handles env vars |
| Compile AFTER DDP/FSDP wrapping | `torch.compile(ddp_model)`, not `DDP(torch.compile(model))` |
| Set epoch on sampler | `sampler.set_epoch(epoch)` — ensures proper shuffling |
| Use NCCL backend for GPU | `backend="nccl"` for GPU training |
| Destroy process group | Always call `dist.destroy_process_group()` on exit |
| Save from rank 0 only | `if rank == 0: save_checkpoint(...)` |

---

## 10. Performance Optimization (MANDATORY)

### A. Profiling First

**ALWAYS profile before optimizing:**

```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler


def profile_training(model: nn.Module, loader: DataLoader) -> None:
    """Profile a training loop.

    Args:
        model: Model to profile.
        loader: Data loader.
    """
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=tensorboard_trace_handler("./logs/profile"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step, (inputs, targets) in enumerate(loader):
            if step >= 5:
                break
            train_step(model, inputs, targets)
            prof.step()
```

### B. Performance Checklist

```markdown
## GPU Optimization Checklist (Priority Order)

### 1. Data Pipeline
- [ ] num_workers > 0 (use min(8, cpu_count))
- [ ] pin_memory=True
- [ ] persistent_workers=True
- [ ] non_blocking=True on .to(device)
- [ ] Profile: DataLoader is NOT the bottleneck

### 2. Mixed Precision
- [ ] torch.autocast enabled
- [ ] bfloat16 on Ampere+ GPUs
- [ ] GradScaler for float16

### 3. Compilation
- [ ] torch.compile applied to model
- [ ] No graph breaks (check with TORCH_LOGS="graph_breaks")
- [ ] Static shapes where possible (drop_last=True)

### 4. Memory Optimization
- [ ] optimizer.zero_grad(set_to_none=True)
- [ ] loss.item() for logging (not raw loss tensor)
- [ ] torch.utils.checkpoint for large models
- [ ] No unnecessary .cpu()/.numpy() in loop

### 5. Gradient Accumulation (for effective large batch)
- [ ] Accumulate over N steps before optimizer.step()
- [ ] Scale loss by 1/N during accumulation
- [ ] Use DDP no_sync() for first N-1 steps
```

### C. Gradient Accumulation Pattern

```python
accumulation_steps = 4

for batch_idx, (inputs, targets) in enumerate(loader):
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    # Use no_sync for intermediate steps (DDP only)
    context = model.no_sync if (batch_idx + 1) % accumulation_steps != 0 else nullcontext

    with context():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(inputs)
            loss = criterion(output, targets) / accumulation_steps  # Scale loss

        scaler.scale(loss).backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

### D. Memory-Efficient Patterns

```python
# Gradient checkpointing — trade compute for memory
from torch.utils.checkpoint import checkpoint

class LargeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(24)])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            # Recompute activations during backward (saves ~60% memory)
            x = checkpoint(block, x, use_reentrant=False)
        return x


# Channels-last memory format — faster convolutions on modern GPUs
model = model.to(memory_format=torch.channels_last)
inputs = inputs.to(memory_format=torch.channels_last)
```

---

## 11. Testing for PyTorch (MANDATORY)

### A. Test Fixtures

```python
# tests/conftest.py
"""Shared PyTorch test fixtures."""

import pytest
import torch
import random
import numpy as np


@pytest.fixture(autouse=True)
def seed_everything() -> None:
    """Set all seeds for reproducible tests."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture
def device() -> torch.device:
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a dummy batch of images and labels."""
    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))
    return images, labels
```

### B. Model Tests

```python
# tests/test_model.py
"""Tests for model architecture."""

import pytest
import torch
from src.model import ImageClassifier


class TestImageClassifier:
    """Test suite for ImageClassifier."""

    def test_forward_shape(self, device: torch.device) -> None:
        """Test output shape matches (batch, num_classes)."""
        model = ImageClassifier(num_classes=10).to(device)
        x = torch.randn(4, 3, 32, 32, device=device)
        out = model(x)
        assert out.shape == (4, 10)

    def test_single_sample(self, device: torch.device) -> None:
        """Test model works with batch size 1."""
        model = ImageClassifier(num_classes=10).to(device)
        x = torch.randn(1, 3, 32, 32, device=device)
        out = model(x)
        assert out.shape == (1, 10)

    def test_output_is_finite(self, device: torch.device) -> None:
        """Test no NaN or Inf in output."""
        model = ImageClassifier(num_classes=10).to(device)
        x = torch.randn(4, 3, 32, 32, device=device)
        out = model(x)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_backward_pass(self, device: torch.device) -> None:
        """Test gradients flow through the model."""
        model = ImageClassifier(num_classes=10).to(device)
        x = torch.randn(4, 3, 32, 32, device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"NaN gradient in {name}"

    def test_num_parameters(self) -> None:
        """Test parameter count is as expected."""
        model = ImageClassifier(num_classes=10)
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0
        # Optionally assert exact count for regression detection
        # assert num_params == 12345

    @pytest.mark.parametrize("num_classes", [2, 10, 100, 1000])
    def test_variable_num_classes(self, num_classes: int) -> None:
        """Test model works with different class counts."""
        model = ImageClassifier(num_classes=num_classes)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, num_classes)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_compile(self) -> None:
        """Test model compiles without graph breaks."""
        model = ImageClassifier(num_classes=10).cuda()
        compiled = torch.compile(model, fullgraph=True)
        x = torch.randn(4, 3, 32, 32, device="cuda")
        out = compiled(x)
        assert out.shape == (4, 10)
```

### C. Training Tests

```python
# tests/test_train.py
"""Tests for training loop."""

import torch
from src.model import ImageClassifier
from src.train import train_one_epoch


def test_loss_decreases() -> None:
    """Test that loss decreases over multiple steps."""
    model = ImageClassifier(num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Overfit on a single batch
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))

    losses = []
    model.train()
    for _ in range(20):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], "Loss did not decrease"


def test_no_nan_in_training() -> None:
    """Test training produces no NaN values."""
    model = ImageClassifier(num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    model.train()
    for _ in range(5):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        assert not torch.isnan(torch.tensor(loss.item()))
```

### D. Test Coverage Requirements

- **Model tests**: 100% coverage for `forward()`, shape checks, gradient flow
- **Loss tests**: Verify correctness with known inputs/outputs
- **Data tests**: Verify dataset length, sample shapes, transform correctness
- **Training tests**: Loss decreases, no NaN, checkpoint save/load round-trip
- **Minimum overall**: 80% coverage for all PyTorch code

---

## 12. Reproducibility (MANDATORY)

### A. Seed Everything

```python
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers for reproducibility.

    Args:
        worker_id: Worker process ID.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Usage
g = torch.Generator()
g.manual_seed(42)

loader = DataLoader(
    dataset,
    batch_size=32,
    worker_init_fn=seed_worker,
    generator=g,
)
```

### B. Reproducibility Checklist

```markdown
## Reproducibility Requirements

- [ ] All random seeds set (Python, NumPy, PyTorch, CUDA)
- [ ] DataLoader workers seeded via worker_init_fn
- [ ] DataLoader generator seeded
- [ ] Deterministic algorithms enabled
- [ ] Dependencies pinned in pyproject.toml / uv.lock
- [ ] Hyperparameters in config files (not hardcoded)
- [ ] Dataset version tracked (hash or version tag)
- [ ] GPU model and CUDA version logged
- [ ] PyTorch version logged
- [ ] Checkpoint includes RNG states
```

---

## 13. Logging & Observability (MANDATORY)

### A. TorchMetrics for Metrics

```python
import torchmetrics


class MetricTracker:
    """Track training and validation metrics.

    Args:
        num_classes: Number of classes for classification metrics.
        device: Device for metric tensors.
    """

    def __init__(self, num_classes: int, device: torch.device) -> None:
        self.train_metrics = torchmetrics.MetricCollection({
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
            "f1": torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro"),
        }).to(device)

        self.val_metrics = self.train_metrics.clone(prefix="val_")

    def update_train(self, preds: Tensor, targets: Tensor) -> None:
        """Update training metrics."""
        self.train_metrics.update(preds, targets)

    def update_val(self, preds: Tensor, targets: Tensor) -> None:
        """Update validation metrics."""
        self.val_metrics.update(preds, targets)

    def compute_and_reset(self) -> dict[str, float]:
        """Compute all metrics and reset state."""
        results = {}
        results.update({k: v.item() for k, v in self.train_metrics.compute().items()})
        results.update({k: v.item() for k, v in self.val_metrics.compute().items()})
        self.train_metrics.reset()
        self.val_metrics.reset()
        return results
```

### B. Structured Logging

```python
import logging
from pathlib import Path

def setup_logging(log_dir: Path, rank: int = 0) -> logging.Logger:
    """Configure structured logging.

    Args:
        log_dir: Directory for log files.
        rank: Process rank (only rank 0 logs to console).

    Returns:
        Configured logger.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)

    # File handler — all ranks
    fh = logging.FileHandler(log_dir / f"train_rank{rank}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # Console handler — rank 0 only
    if rank == 0:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(ch)

    return logger
```

---

## 14. Security & Dependency Management (MANDATORY)

### A. Model Security

**CVE-2025-32434 (CVSS 9.3)**: `torch.load()` with `weights_only=True` was bypassed for RCE in PyTorch < 2.6.0. Always update to PyTorch >= 2.6.0.

```python
# MANDATORY: Safe model loading patterns

# Option 1: safetensors (PREFERRED — no code execution possible)
from safetensors.torch import load_file
state_dict = load_file("model.safetensors")

# Option 2: torch.load with weights_only (PyTorch >= 2.6.0)
state_dict = torch.load("model.pt", weights_only=True, map_location="cpu")

# NEVER do this — arbitrary code execution
state_dict = torch.load("model.pt")  # CVE-2025-32434
```

### B. Dependency Management

```bash
# Use uv for all package management
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv add safetensors torchmetrics
uv add --dev pytest ruff

# Verify dependencies
uv run pip-audit            # Check for CVEs
uv run ruff check .         # Lint
```

### C. Supply Chain Rules

| Risk | Mitigation |
|------|-----------|
| Pickle RCE in model files | Use safetensors; never pickle |
| Malicious pretrained models | Verify source; scan with safetensors |
| Dependency vulnerabilities | Pin versions; run pip-audit in CI |
| Hugging Face model poisoning | Check model cards; verify checksums |

---

## 15. Configuration & Hyperparameters (MANDATORY)

### A. Frozen Dataclass Config

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture configuration."""

    num_classes: int = 10
    hidden_dim: int = 256
    num_layers: int = 6
    dropout: float = 0.1
    activation: str = "gelu"


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters."""

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    compile: bool = True
    compile_mode: str = "default"
    seed: int = 42


@dataclass(frozen=True)
class DataConfig:
    """Data pipeline configuration."""

    train_path: str = "data/train"
    val_path: str = "data/val"
    num_workers: int = 8
    image_size: int = 224
    augment: bool = True
```

### B. Dynaconf Integration

```toml
# configs/defaults.toml
[default]
seed = 42
device = "cuda"

[default.model]
num_classes = 10
hidden_dim = 256
num_layers = 6
dropout = 0.1

[default.training]
epochs = 100
batch_size = 64
learning_rate = 3e-4
weight_decay = 0.01
mixed_precision = true
compile = true

[default.data]
num_workers = 8
image_size = 224

[development]
training.epochs = 5
training.compile = false  # Faster iteration

[production]
training.compile = true
training.compile_mode = "max-autotune"
```

---

## 16. Deployment Checklist

### Agent-Generated PyTorch Code Verification (MANDATORY)

**Before delivering ANY PyTorch code:**

#### Build & Syntax
- [ ] Code compiles: `uv run python -m py_compile *.py` returns exit 0
- [ ] Ruff passes: `uv run ruff check .` returns exit 0
- [ ] Code formatted: `uv run ruff format --check .` returns exit 0

#### Model Correctness
- [ ] Forward pass produces correct shapes with dummy data
- [ ] Backward pass produces finite gradients for all parameters
- [ ] Model works with batch size 1 and batch size > 1
- [ ] Loss decreases on overfit test (single batch, 20 steps)
- [ ] No NaN/Inf in outputs or gradients

#### Security
- [ ] No `torch.load()` without `weights_only=True`
- [ ] No pickle-based serialization for models (use safetensors)
- [ ] No hardcoded secrets or API keys
- [ ] PyTorch >= 2.6.0 (CVE-2025-32434 patched)
- [ ] Dependencies scanned: `uv run pip-audit`

#### Performance
- [ ] `torch.compile` applied to production model
- [ ] Mixed precision enabled (`torch.autocast`)
- [ ] DataLoader optimized (num_workers, pin_memory, persistent_workers)
- [ ] No `.cpu()` / `.item()` / `.numpy()` inside training loop (except logging)
- [ ] `optimizer.zero_grad(set_to_none=True)` used

#### Reproducibility
- [ ] All seeds set (Python, NumPy, PyTorch, CUDA)
- [ ] DataLoader workers seeded
- [ ] Hyperparameters in config files (not hardcoded)
- [ ] Dependencies pinned
- [ ] Checkpoint saves full state (model, optimizer, scheduler, scaler, RNG)

#### Testing
- [ ] All tests pass: `uv run pytest` returns exit 0
- [ ] Coverage ≥ 80%: `uv run pytest --cov --cov-fail-under=80`
- [ ] Model shape tests present
- [ ] Gradient flow tests present
- [ ] Loss decrease test present

---

## 17. Quick Reference

### Common Commands

```bash
# Setup
uv add torch torchvision torchaudio
uv add safetensors torchmetrics
uv add --dev pytest ruff

# Development
uv run python -m py_compile src/model.py    # Syntax check
uv run ruff check .                          # Lint
uv run ruff format .                         # Format
uv run pytest tests/ -v                      # Test
uv run pytest --cov=src --cov-fail-under=80  # Coverage

# Training
uv run python src/train.py                   # Single GPU
torchrun --nproc_per_node=4 src/train.py     # Multi-GPU DDP
TORCH_LOGS="graph_breaks" uv run python src/train.py  # Debug compile

# Profiling
uv run python -c "
import torch
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # ... training step ...
    pass
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
"
```

### PyTorch 2.x Migration Cheat Sheet

| Old API | New API |
|---------|---------|
| `torch.cuda.amp.autocast()` | `torch.autocast(device_type="cuda")` |
| `torch.cuda.amp.GradScaler()` | `torch.amp.GradScaler()` |
| `torch.load(path)` | `torch.load(path, weights_only=True)` |
| Manual optimization | `torch.compile(model)` |
| `torchvision.transforms` | `torchvision.transforms.v2` |
| `DistributedDataParallel` only | `fully_shard` (FSDP2) for large models |

### Model Serialization Cheat Sheet

```python
# SAVE model weights (safetensors — PREFERRED)
from safetensors.torch import save_file
save_file(model.state_dict(), "model.safetensors")

# LOAD model weights (safetensors)
from safetensors.torch import load_file
model.load_state_dict(load_file("model.safetensors"))

# SAVE training state (torch — with weights_only)
torch.save({"epoch": e, "optimizer": opt.state_dict()}, "state.pt")

# LOAD training state
state = torch.load("state.pt", weights_only=True)
```

### Compilation Debugging

```bash
# Show graph breaks
TORCH_LOGS="graph_breaks" python train.py

# Show compiled graphs
TORCH_LOGS="output_code" python train.py

# Disable compile for debugging
model = MyModel()  # Don't compile — debug in eager mode
```

---

**End of PyTorch Development Guidelines**
