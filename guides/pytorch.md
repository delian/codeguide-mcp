# PyTorch Development Guidelines
Mandatory standards for modern PyTorch: reproducible, GPU-optimized, production-ready deep learning. PyTorch 2.x, torch.compile, mixed precision (autocast/GradScaler), DDP/FSDP2, safetensors, CUDA.

---
name: pytorch
title: PyTorch Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: framework
tools: [pytorch@2.7, torchvision, torchmetrics, safetensors, torch.compile, cuda@12.4, pytest, ruff]
requires:
  - tdd
recommends:
  - mlops
  - python
  - performance
  - cuda
  - observability
provides:
  - tensors-autograd
  - nn-modules
  - training-loops
  - mixed-precision
  - torch-compile
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to PyTorch. The host language is owned by [`python.md`](guides://python.md); experiment tracking / model registry / serving by [`mlops.md`](guides://mlops.md); GPU kernels & memory by [`cuda.md`](guides://cuda.md).

---

## 0. Prerequisites & References

Fetch and apply these **before** generating PyTorch code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(PyTorch binding: runner is `uv run pytest`; assertion targets are shapes, finiteness, gradient flow, and loss-decreases-on-overfit — see §9.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`python.md`](guides://python.md) — the language: `uv` toolchain, typing, ruff, project layout, packaging, secure-coding/dependency policy. *(PyTorch is a Python library; all language gates apply.)*
> - [`mlops.md`](guides://mlops.md) — experiment tracking, model registry, dataset/model versioning, CI for models, serving. *(PyTorch binding: log via the mlops tool; export with safetensors / `torch.export` / ONNX.)*
> - [`performance.md`](guides://performance.md) — profile-before-optimize, benchmarking discipline, perf budgets.
> - [`cuda.md`](guides://cuda.md) — GPU memory model, kernels, streams, occupancy. *(PyTorch binding: device placement, `channels_last`, custom kernels.)*
> - [`observability.md`](guides://observability.md) — metrics/tracing policy *(binding: `torchmetrics`, `torch.profiler`, TensorBoard/W&B handler).*

> 📎 **SEE ALSO:** [`secure-coding.md`](guides://secure-coding.md) *(deserialization / supply chain — the policy owner for §8)* · [`env-config.md`](guides://env-config.md) *(hyperparameter config)* · [`logging.md`](guides://logging.md)

---

## 1. Core Philosophies: TENSOR-FIRST

PyTorch-specific principles only. TDD, security, error handling, and the Python toolchain come from §0.

- **T**orch.compile for production: develop and debug in eager mode; ship `torch.compile`d models. Eager is the source of truth; compile is an optimization layer.
- **E**nd-to-end device discipline: keep tensors on the accelerator; every `.cpu()`, `.item()`, `.numpy()`, or `.tolist()` is a host-device **sync point** — none belong in the hot loop except scalar logging.
- **N**ative mixed precision: cast via `torch.autocast` + `torch.amp.GradScaler`, never by hand (`.half()`/`.float()`).
- **S**afe serialization: weights via `safetensors`; any `torch.load` MUST pass `weights_only=True` (PyTorch ≥ 2.6 — see §8).
- **O**ptimized input pipeline: `DataLoader` with `num_workers>0`, `pin_memory=True`, `persistent_workers=True`, `non_blocking=True` transfers.
- **R**eproducible by default: seed Python/NumPy/Torch/CUDA, enable deterministic algorithms, checkpoint full RNG state.

**Fail loudly**: shape mismatches, NaN/Inf losses, and exploding gradients MUST raise, never silently continue.

**Verified Code**: Agent-generated PyTorch MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `PT-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| PT-TST-01 | Every model/loss/training feature MUST be test-first (see `tdd.md`) | `uv run pytest` | exit 0, 0 skips |
| PT-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `uv run pytest` | failing→passing |
| PT-TST-03 | Every `nn.Module` MUST have a forward-shape + finiteness + gradient-flow test | `uv run pytest -k model` | exit 0 |
| PT-TST-04 | Training step MUST pass an overfit-on-one-batch convergence test | `uv run pytest -k overfit` | loss decreases |
| PT-FMT-01 | Code MUST be formatted & linted (see `python.md`) | `uv run ruff format --check . && uv run ruff check .` | no diff, exit 0 |
| PT-TYP-01 | `forward()` and public APIs MUST be typed with `Tensor` in/out | `uv run mypy --strict src/` | exit 0 |
| PT-SEC-01 | No `torch.load` without `weights_only=True`; weights use safetensors (see `secure-coding.md`) | `grep -rn "torch.load" src/ \| grep -v weights_only=True` | empty |
| PT-SEC-02 | PyTorch ≥ 2.6 and 0 known CVEs in deps (see `secure-coding.md`) | `uv run pip-audit` | 0 vulnerabilities |
| PT-RPRO-01 | All RNGs seeded; deterministic algorithms set; checkpoint stores RNG state | review / `grep manual_seed` | seed util present |
| PT-AMP-01 | Mixed precision via `torch.autocast`; no manual `.half()`/`.float()` casts | `grep -rn "\.half()" src/` | empty (or justified) |
| PT-PERF-01 | Production model `torch.compile`d; DataLoader uses workers+pin_memory (see `performance.md`) | review | compile applied |
| PT-MEM-01 | No `.item()`/`.cpu()`/`.numpy()` in the hot loop except scalar logging | review | no sync in loop |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`); `torch.load` without `weights_only=True`; pickle for model weights; accumulating un-detached loss tensors; `model.eval()` inference without `torch.inference_mode()`/`no_grad()`; deprecated APIs (`torch.cuda.amp.autocast` → `torch.autocast`).

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green. The Python gates (ruff/mypy/pip-audit) are owned by [`python.md`](guides://python.md); below adds the PyTorch-specific checks.

```bash
uv run ruff format --check . && uv run ruff check .   # PT-FMT-01
uv run mypy --strict src/                             # PT-TYP-01
uv run pytest -v                                      # PT-TST-01..04
grep -rn "torch.load" src/ | grep -v weights_only=True   # PT-SEC-01 (must be empty)
uv run pip-audit                                      # PT-SEC-02
TORCH_LOGS="graph_breaks" uv run python -m src.smoke  # PT-PERF-01 (no unexpected breaks)
```

---

## 4. Project Structure

Keep `nn.Module` definitions **pure** — no data loading, training, logging, or device decisions inside model files. The `src/` layout and import-boundary policy are owned by [`python.md`](guides://python.md); below is the PyTorch mapping.

```
project/
├── src/
│   ├── model.py        # nn.Module definitions only (pure: tensors in → tensors out)
│   ├── layers.py       # reusable blocks
│   ├── losses.py       # custom objectives
│   ├── dataset.py      # Dataset / DataLoader factory
│   ├── transforms.py   # torchvision.transforms.v2 pipelines
│   ├── train.py        # training loop, optimizer, scheduler, AMP, checkpointing
│   ├── evaluate.py     # inference / metrics
│   └── utils.py        # seed, checkpoint, device helpers
├── tests/              # mirrors src/ (see tdd.md); conftest.py seeds + dummy batches
├── configs/            # hyperparameters (see env-config.md) — never hardcoded
├── checkpoints/        # git-ignored
└── pyproject.toml
```

Hyperparameters live in config (frozen dataclass or TOML), never as magic numbers in code; the config policy is owned by [`env-config.md`](guides://env-config.md).

---

## 5. Tensors, Autograd & Modules

### A. Tensor & device hygiene

```python
import torch
from torch import Tensor

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():   # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")
```

- Create tensors **on the target device** (`torch.zeros(n, device=dev)`), don't allocate on CPU then `.to()`.
- `x.to(dev, non_blocking=True)` overlaps transfer with compute (requires `pin_memory` source).
- Prefer `torch.as_tensor` / `.from_numpy` (zero-copy) over `torch.tensor(...)` (always copies).
- Use views (`view`, `reshape`, `permute`, `unsqueeze`) over copies; remember `permute`/`transpose` return non-contiguous tensors — call `.contiguous()` before ops that require it.

### B. Autograd

- The graph is built on the forward pass and freed by `.backward()`; retain it only with `retain_graph=True` (rarely needed).
- Detach to stop gradients: `x.detach()` or `with torch.no_grad():`. For **inference**, prefer `torch.inference_mode()` — stricter and faster than `no_grad()` (it also disables version counter bookkeeping).
- Accumulating metrics across steps: store `loss.item()` (a Python float), **never** the live `loss` tensor — the latter keeps the whole graph alive and leaks memory.
- Debug a NaN/Inf gradient with `torch.autograd.set_detect_anomaly(True)` (slow — debug only).

### C. nn.Module pattern

```python
import torch.nn as nn
from torch import Tensor

class ResidualBlock(nn.Module):
    """Pre-activation residual block. Shapes: (B, C, H, W) → (B, C, H, W)."""

    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),  # bias=False before BN
            nn.BatchNorm2d(channels), nn.ReLU(), nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)
```

Rules: type-hint `forward()` as `Tensor → Tensor`; document expected shapes in the docstring; `bias=False` on conv/linear feeding a norm layer; register submodules as attributes or in `nn.ModuleList`/`nn.ModuleDict` (a plain Python list hides params from `.parameters()`); use `register_buffer` for non-trainable state (running stats, masks) so it moves with `.to(device)` and is saved.

### D. Initialization

Apply explicit init to custom layers via `model.apply(fn)` — e.g. `nn.init.kaiming_normal_` for conv/linear (`mode="fan_out"` with ReLU), constant 1/0 for norm weight/bias. Don't rely on default init for non-trivial architectures.

---

## 6. Training Loop, Optimizers & Mixed Precision

### A. Canonical training step (AMP + clipping + scheduler)

```python
scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))  # bf16 needs no scaler

for inputs, targets in loader:
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)            # cheaper than zeroing

    with torch.autocast(device_type=device.type, dtype=amp_dtype):
        loss = criterion(model(inputs), targets)     # forward + loss INSIDE autocast

    scaler.scale(loss).backward()                    # backward OUTSIDE autocast
    scaler.unscale_(optimizer)                       # unscale BEFORE clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    running += loss.item()                           # scalar sync — the only one allowed
scheduler.step()                                     # per-epoch schedulers step here
```

Key bindings:

| Practice | Why |
|---|---|
| `zero_grad(set_to_none=True)` | Frees grad tensors; faster than memset-to-zero |
| forward+loss inside `autocast`, backward outside | Autograd records the autocast region; backward must run in fp32 master grads |
| `scaler.unscale_()` before `clip_grad_norm_` | Clipping must see true-scale gradients |
| `loss.item()` for logging | Detaches; raw tensor would pin the graph and leak |
| `bfloat16` on Ampere+ → no GradScaler | Wider dynamic range; fp16 still needs the scaler |

### B. Precision selection

| dtype | HW floor | GradScaler | Use |
|---|---|---|---|
| `float32` | any | no | debugging, numeric precision |
| `float16` | Volta+ | **yes** | AMP on older GPUs |
| `bfloat16` | Ampere+ | no | default for large models / LLMs |

Pick `torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16`. GPU memory & kernel-level detail are owned by [`cuda.md`](guides://cuda.md).

### C. Optimizers & schedulers

- Default to `torch.optim.AdamW` (decoupled weight decay); `fused=True` on CUDA fuses the step into one kernel.
- Pair with a scheduler (`CosineAnnealingLR`, `OneCycleLR`, or a warmup+cosine via `SequentialLR`); step per-epoch or per-batch as the scheduler requires — `OneCycleLR` steps per batch.
- Checkpoint the scheduler and scaler state, not just the optimizer.

### D. Gradient accumulation (effective large batch)

Scale loss by `1/N`, call `backward()` each micro-step, and run `optimizer.step()` every `N` steps. Under DDP, wrap the first `N-1` micro-steps in `model.no_sync()` to skip all-reduce until the final step.

### E. torch.compile

```python
model = torch.compile(model, mode="max-autotune")   # production throughput
# modes: "default" (balanced) · "reduce-overhead" (latency, small models) · "max-autotune"
# fullgraph=True forbids graph breaks — use to assert a clean capture in tests
```

- Compile **once**, after moving to device and (for distributed) after DDP/FSDP wrapping — `torch.compile(ddp_model)`, never `DDP(torch.compile(model))`.
- Graph breaks kill speedups: avoid data-dependent Python control flow on tensor values, `.item()`/`print()`/`breakpoint()` in `forward`, and dynamic shapes (use `drop_last=True` for static batch). Diagnose with `TORCH_LOGS="graph_breaks"`; inspect generated code with `TORCH_LOGS="output_code"`.
- The compiled module wraps the original as `model._orig_mod` — unwrap it before saving `state_dict` to keep keys clean.

---

## 7. Data Loading

```python
from torch.utils.data import DataLoader, Dataset
import os

def make_loader(ds: Dataset, batch_size: int, *, train: bool) -> DataLoader:
    workers = min(8, os.cpu_count() or 1)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=train,
        num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        drop_last=train,                 # static shapes help torch.compile; eval keeps all samples
    )
```

| Rule | Reason |
|---|---|
| `num_workers = min(8, cpu_count)` | More workers than CPUs → contention |
| `pin_memory=True` + `non_blocking=True` | Enables async H→D copy overlapping compute |
| `persistent_workers=True` | Avoids per-epoch worker respawn |
| `drop_last=True` (train only) | Constant batch size → fewer recompiles |
| `torchvision.transforms.v2` | Current API; composable; supports bbox/mask/video |

Custom `Dataset`: implement `__len__` and `__getitem__` returning tensors; do heavy decode/augment in `__getitem__` (runs in workers), keep `__init__` to indexing. Map-style is default; use `IterableDataset` only for true streams (and shard by `worker_id` to avoid duplication).

---

## 8. Checkpointing & Serialization (Security)

Deserialization safety is owned by [`secure-coding.md`](guides://secure-coding.md). PyTorch binding: pickle-based `torch.save`/`torch.load` is an **RCE vector** (CVE-2025-32434 bypassed `weights_only` in PyTorch < 2.6 — require ≥ 2.6).

```python
from safetensors.torch import save_file, load_file

# Weights → safetensors (no code execution, mmap-fast, safe to share)
raw = model._orig_mod if hasattr(model, "_orig_mod") else model   # unwrap compiled module
save_file(raw.state_dict(), "model.safetensors")
model.load_state_dict(load_file("model.safetensors", device=str(device)))

# Training state (optimizer/scheduler/scaler/epoch/RNG) → torch.save, load with weights_only
torch.save({
    "epoch": epoch,
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "scaler": scaler.state_dict(),
    "rng": torch.get_rng_state(),
    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
}, "state.pt")
state = torch.load("state.pt", weights_only=True, map_location="cpu")   # MANDATORY flag
```

| Format | Safety | Use |
|---|---|---|
| safetensors | safe (no exec) | model weights — **preferred** |
| `torch.save` + `weights_only=True` | safe (≥ 2.6) | training state (tensors/primitives only) |
| `torch.save` default / pickle | **UNSAFE (RCE)** | never for untrusted or shared artifacts |

Model **registry, versioning, and serving** of these artifacts is owned by [`mlops.md`](guides://mlops.md). For deployment, export via `torch.export` / ONNX / TorchScript and register the artifact through the mlops pipeline.

---

## 9. Testing PyTorch Models

Test-first policy and coverage gates are owned by [`tdd.md`](guides://tdd.md). PyTorch-specific assertion targets:

- **Shape**: `forward` output is exactly `(B, …)` for batch sizes 1 and >1, across parametrized config (e.g. `num_classes`).
- **Finiteness**: `torch.isfinite(out).all()` — no NaN/Inf in outputs.
- **Gradient flow**: after `out.sum().backward()`, every `requires_grad` param has a finite, non-`None` `.grad`.
- **Convergence (overfit test)**: a model trained on one fixed batch for ~20 steps MUST drive loss strictly down — catches dead layers, detached graphs, wrong LR.
- **Compile**: `torch.compile(model, fullgraph=True)` runs without a graph break (mark `skipif` when no GPU).
- **Determinism**: seeded run reproduces identical outputs.

```python
# tests/conftest.py
@pytest.fixture(autouse=True)
def _seed() -> None:
    set_seed(42)                       # seeds python/numpy/torch/cuda + deterministic algos

@pytest.fixture
def dummy_batch() -> tuple[Tensor, Tensor]:
    return torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))
```

Keep tests CPU-runnable by default; gate GPU/compile tests with `@pytest.mark.skipif(not torch.cuda.is_available(), ...)`. A bug (NaN loss, shape error, wrong metric) gets a failing regression test **before** the fix (see `tdd.md`).

---

## 10. Reproducibility

```python
import os, random, numpy as np, torch

def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False          # benchmark picks nondeterministic kernels
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # required for deterministic cuBLAS

def seed_worker(_: int) -> None:                    # DataLoader(worker_init_fn=seed_worker, generator=g)
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)
```

Checklist: seed Python/NumPy/Torch/CUDA; seed DataLoader workers + generator; deterministic algorithms on; pin deps (`uv.lock`); hyperparameters in config; log dataset hash, GPU/CUDA, and PyTorch version; checkpoint RNG state. Dataset/run versioning and experiment metadata are owned by [`mlops.md`](guides://mlops.md).

---

## 11. Distributed Training (DDP / FSDP2)

For multi-GPU. Launch with `torchrun --nproc_per_node=N script.py` (handles env/rank), never hand-rolled `mp.spawn`.

- **DDP** (data parallel, model fits one GPU): wrap `DDP(model, device_ids=[rank])`; use `DistributedSampler` and call `sampler.set_epoch(epoch)` each epoch so shuffling differs; backend `"nccl"` for GPU; save from **rank 0 only**; `dist.destroy_process_group()` on exit.
- **FSDP2** (model too big for one GPU): shard with `fully_shard(layer, mp_policy=...)` per transformer block, then the whole model; `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`.
- Apply `torch.compile` **after** DDP/FSDP wrapping.
- Gradient accumulation: `model.no_sync()` on micro-steps to skip premature all-reduce (see §6.D).

Collective-level GPU/interconnect concerns are owned by [`cuda.md`](guides://cuda.md).

---

## 12. Performance & Memory

Profile-before-optimize is owned by [`performance.md`](guides://performance.md); GPU memory mechanics by [`cuda.md`](guides://cuda.md). PyTorch bindings:

- **Profile** with `torch.profiler.profile(activities=[CPU, CUDA], profile_memory=True, record_shapes=True)` + a `schedule` and `tensorboard_trace_handler`; read `prof.key_averages().table(sort_by="cuda_time_total")`. Never optimize on a guess.
- **Priority order**: (1) data pipeline not the bottleneck → (2) AMP on → (3) `torch.compile` clean → (4) memory (set_to_none, no stray syncs) → (5) gradient accumulation for batch size.
- **Memory levers**: `torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)` trades compute for ~60% activation memory; `model.to(memory_format=torch.channels_last)` (+ inputs) speeds convolutions on Tensor Cores; reduce batch / enable AMP for OOM.
- **Sync points** are the silent killer: avoid `.item()`, `.cpu()`, `.numpy()`, `tensor.tolist()`, `print(tensor)`, and Python `if` on a GPU scalar inside the loop.

### Inference

```python
model.eval()                                    # toggles BN/Dropout
with torch.inference_mode():                    # faster & stricter than no_grad()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(batch.to(device, non_blocking=True))
```

Both `eval()` **and** `inference_mode()`/`no_grad()` are required — the first changes layer behavior, the second drops autograd bookkeeping.

---

## 13. Observability

Metrics/tracing policy is owned by [`observability.md`](guides://observability.md); structured logging by [`logging.md`](guides://logging.md). PyTorch bindings:

- Compute metrics with **`torchmetrics`** (`MetricCollection`, `.clone(prefix="val_")`); keep metric objects on-device and `.update()` per batch, `.compute()`/`.reset()` per epoch — avoids manual sync-and-accumulate bugs.
- Stream scalars/histograms to TensorBoard or W&B; in distributed runs log from **rank 0 only**.
- Surface profiler traces (§12) and per-epoch metrics through the experiment tracker — owned by [`mlops.md`](guides://mlops.md).

---

## 14. Quick Reference

```bash
# Setup (see python.md for the uv toolchain)
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv add safetensors torchmetrics && uv add --dev pytest ruff

# Verify
uv run ruff check . && uv run mypy --strict src/ && uv run pytest    # gates
grep -rn "torch.load" src/ | grep -v weights_only=True               # PT-SEC-01 (empty)

# Train / debug
torchrun --nproc_per_node=4 src/train.py                # multi-GPU DDP
TORCH_LOGS="graph_breaks" uv run python src/train.py    # diagnose compile
```

### PyTorch 2.x migration cheat sheet

| Old | New |
|---|---|
| `torch.cuda.amp.autocast()` | `torch.autocast(device_type="cuda")` |
| `torch.cuda.amp.GradScaler()` | `torch.amp.GradScaler()` |
| `torch.load(path)` | `torch.load(path, weights_only=True)` |
| `torch.no_grad()` for inference | `torch.inference_mode()` |
| hand-tuned kernels | `torch.compile(model)` |
| `torchvision.transforms` | `torchvision.transforms.v2` |
| FSDP1 wrapper | `fully_shard` (FSDP2) |

---

## 15. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] PT-FMT-01 — ruff format + check clean (see `python.md`)
- [ ] PT-TYP-01 — `mypy --strict` clean; `forward()` typed `Tensor→Tensor`
- [ ] PT-TST-01/02 — tests pass, bugs have regression tests first
- [ ] PT-TST-03 — every module has shape + finiteness + gradient-flow tests
- [ ] PT-TST-04 — overfit-on-one-batch convergence test present
- [ ] PT-SEC-01 — no `torch.load` without `weights_only=True`; weights via safetensors
- [ ] PT-SEC-02 — PyTorch ≥ 2.6, `pip-audit` 0 CVEs
- [ ] PT-RPRO-01 — seeds + deterministic algos set; checkpoint stores RNG state
- [ ] PT-AMP-01 — mixed precision via `torch.autocast`; no manual casts
- [ ] PT-PERF-01 — production model compiled; DataLoader workers+pin_memory
- [ ] PT-MEM-01 — no host-device syncs in the hot loop (except scalar logging)
- [ ] Agent ran every §3 command and documented any fixes

---
**End of PyTorch Guidelines**
