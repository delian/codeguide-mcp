# CUDA Programming Guidelines
Mandatory standards for CUDA GPU programming: correct kernels, coalesced memory, profiler-driven optimization. CUDA Toolkit 12.x, nvcc, Nsight Compute (ncu), Nsight Systems (nsys), compute-sanitizer, CMake.

---
name: cuda
title: CUDA Programming Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: tooling
tools: [cuda@12.x, nvcc, nsight-compute, nsight-systems, compute-sanitizer, cmake]
requires: []
recommends:
  - cpp
  - c
  - performance
  - parallelism
  - pytorch
provides:
  - cuda-programming-model
  - gpu-memory-hierarchy
  - kernel-optimization
  - cuda-streams
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns only what is unique to CUDA — the GPU execution and memory model. Host-language rules (C/C++ style, build, tests) live in their owners.

---

## 0. Prerequisites & References

CUDA is C/C++ that targets the GPU. Fetch the host-language and methodology guides for everything that is *not* GPU-specific; this guide does not repeat them.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`cpp.md`](guides://cpp.md) · [`c.md`](guides://c.md) — host language: style, RAII, build, headers. CUDA `.cu`/`.cuh` files obey these for all host code. *(Compile host TUs as C++17+.)*
> - [`performance.md`](guides://performance.md) — perf methodology: **measure first, optimize the proven bottleneck**. CUDA binding: the profiler is `ncu`/`nsys` (§7).
> - [`parallelism.md`](guides://parallelism.md) — concurrency concepts (data races, synchronization, dependencies). CUDA binding: warps, `__syncthreads`, streams, atomics (§4–§5).
> - [`pytorch.md`](guides://pytorch.md) — the dominant CUDA consumer; read it when writing custom ops/extensions that interop with PyTorch tensors and CUDA streams.

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) — Red-Green-Refactor & regression-before-fix (CUDA binding in §2/§6: CPU reference is the gold standard). · [`secure-coding.md`](guides://secure-coding.md) · [`comments.md`](guides://comments.md) · [`cmake.md`](guides://cmake.md) · [`conan.md`](guides://conan.md)

---

## 1. Core Philosophies: PERFORMANCE-FIRST

CUDA-specific principles only. Test-first, error-handling strategy, and C/C++ style come from §0.

- **P**rofile before optimizing: never claim a speedup without `ncu`/`nsys` evidence (methodology: `performance.md`).
- **E**fficient memory: minimize host↔device transfers; keep data resident on the GPU; maximize **coalesced** global access; exploit shared memory and registers.
- **R**eference-validated: every kernel has a CPU reference; GPU output is checked against it within a documented tolerance (test-first per `tdd.md`).
- **F**used & library-first: prefer NVIDIA libraries (cuBLAS, cuFFT, Thrust/CUB) over hand-rolled kernels; fuse passes to cut launch overhead.
- **O**ccupancy-aware: size blocks/grids and bound register/shared usage so latency is hidden — but occupancy is a means, not the goal (throughput is).
- **R**obust & checked: every CUDA API call is error-checked; `compute-sanitizer` is clean before delivery.
- **M**inimal divergence: avoid divergent branches within a warp; organize work at warp granularity.

**Verified Code**: agent-generated CUDA MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `CUDA-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| CUDA-BLD-01 | MUST compile for every target arch, warnings as errors | `nvcc -arch=sm_XX -Xcompiler -Wall,-Wextra,-Werror` / `cmake --build` | exit 0, no warnings |
| CUDA-ERR-01 | Every CUDA API call MUST be error-checked; every launch followed by `cudaGetLastError()` | grep / review for unwrapped `cuda*` calls | none unchecked |
| CUDA-MEM-01 | MUST be free of OOB, races, leaks, uninit reads | `compute-sanitizer --tool memcheck` (+`racecheck`,`initcheck`) | 0 errors |
| CUDA-ACC-01 | Every kernel MUST validate against a CPU reference within a stated tolerance | run accuracy test | max_err < tol |
| CUDA-TST-01 | Each kernel MUST be test-first (CPU ref + failing GPU test) (see `tdd.md`) | `ctest --output-on-failure` | exit 0, 0 skips |
| CUDA-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `ctest` | failing→passing |
| CUDA-PERF-01 | Optimization claims MUST be backed by profiler data (see `performance.md`) | `ncu`/`nsys` report attached | report exists |
| CUDA-SEC-01 | 0 high/critical CVEs in deps; track NVIDIA security bulletins (see `secure-coding.md`) | dep audit (e.g. `conan audit scan .`) | 0 high/critical |
| CUDA-DOC-01 | Kernels MUST document contract + perf characteristics (see `comments.md`) | review / doc build | documented |

> **Forbidden**: shipping a kernel before its test (violates `tdd.md`); fixing a bug without a regression test first; unchecked `cudaMalloc`/launches; missing `cudaFree`; suppressing `compute-sanitizer` findings; deprecated APIs (texture *references*, `__syncthreads_count` over cooperative groups); optimization claims without a profile.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
nvcc -std=c++17 -O3 -arch=sm_90 -Xcompiler -Wall,-Wextra,-Werror -o app app.cu  # CUDA-BLD-01
ctest --output-on-failure                       # CUDA-TST-01/02, CUDA-ACC-01
compute-sanitizer --tool memcheck ./app         # CUDA-MEM-01
compute-sanitizer --tool racecheck ./app        # CUDA-MEM-01 (data races)
ncu --set full -o profile ./app                 # CUDA-PERF-01 (when optimizing)
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. The CUDA Programming Model

The unique core of this guide.

### A. Host vs device, kernels, the launch
Host (CPU) code orchestrates; device (GPU) code runs in **kernels** (`__global__`, launched with `<<<grid, block>>>`, return `void`). `__device__` functions run on and are callable from the device only; `__host__ __device__` compiles for both. A launch is **asynchronous** — it returns immediately; synchronize before reading results.

```cuda
__global__ void saxpy(float* y, const float* x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;     // global thread index
    if (i < n) y[i] = a * x[i] + y[i];                 // ALWAYS guard the tail
}

int block = 256;                                       // threads/block (multiple of 32)
int grid  = (n + block - 1) / block;                   // cover all n elements
saxpy<<<grid, block>>>(d_y, d_x, 2.0f, n);
CUDA_CHECK(cudaGetLastError());                         // catch launch-config errors
CUDA_CHECK(cudaDeviceSynchronize());                    // wait + catch async errors
```

### B. Thread hierarchy: threads → warps → blocks → grid
- A **block** is up to 1024 threads, 1–3D (`blockDim`/`threadIdx`), co-resident on one SM, sharing its shared memory and able to `__syncthreads()`.
- A **grid** is up to 3D of blocks (`gridDim`/`blockIdx`); blocks are independent and unordered — **never** assume a block execution order or cross-block sync within a launch.
- A **warp** is 32 threads executing in lockstep (SIMT). Divergent control flow within a warp serializes the paths — keep branches warp-coherent.
- Use `dim3` for 2D/3D problems: `dim3 block(16,16); dim3 grid((W+15)/16,(H+15)/16);`.

### C. Launch-config rules
- Block size a multiple of 32 (warp size); 128–256 is a safe default — confirm with occupancy (§6).
- Compute grid to cover the domain; the tail guard (`if (i < n)`) is mandatory.
- Pass dynamic shared memory as the 3rd launch arg, a stream as the 4th: `k<<<g,b,smem_bytes,stream>>>()`.

---

## 5. Memory Hierarchy & Access

Getting memory right is where almost all CUDA performance lives.

| Space | Scope / lifetime | Latency | Use for |
|-------|------------------|---------|---------|
| Registers | per-thread | fastest | scalars, accumulators (spills → "local"/DRAM: avoid) |
| Shared (`__shared__`) | per-block | ~L1 | block-level reuse, tiling, staging for coalescing |
| Global (`cudaMalloc`) | grid, app | slow (DRAM) | bulk data; **must be coalesced** |
| Constant (`__constant__`) | grid, read-only | cached | small broadcast values (same address per warp) |
| Local | per-thread | DRAM | register spills, large per-thread arrays (avoid) |
| Texture/`__ldg`/read-only | grid, read-only | cached | irregular/spatially-local read-only access |

### A. Minimize host↔device transfers
PCIe/NVLink copies dominate naive code. Transfer once, compute many times, keep intermediates on the GPU. Use **pinned** host memory (`cudaMallocHost`) for faster, async-capable copies.

### B. Coalesced global access
Threads in a warp should touch **contiguous, aligned** global addresses so the hardware merges them into the fewest transactions. Strided/transposed access wastes bandwidth — stage through shared memory instead.

```cuda
// Transpose: coalesce both read and write via a shared tile (pad +1 to kill bank conflicts).
__global__ void transpose(float* out, const float* in, int n) {
    __shared__ float tile[32][33];                 // 33 avoids 32-way bank conflicts
    int x = blockIdx.x*32 + threadIdx.x, y = blockIdx.y*32 + threadIdx.y;
    if (x < n && y < n) tile[threadIdx.y][threadIdx.x] = in[y*n + x];  // coalesced load
    __syncthreads();
    x = blockIdx.y*32 + threadIdx.x; y = blockIdx.x*32 + threadIdx.y;
    if (x < n && y < n) out[y*n + x] = tile[threadIdx.x][threadIdx.y]; // coalesced store
}
```

### C. Shared memory & bank conflicts
Shared memory has 32 banks (4-byte stride). A warp accessing distinct banks is conflict-free; multiple threads hitting the *same bank, different word* serialize. Pad the leading dimension (`[N][N+1]`) for column access. Use shared memory for intra-block reuse (tiling, reductions, scans) — it is the single most effective optimization for memory-bound kernels.

### D. Unified Memory
`cudaMallocManaged` gives one pointer valid on host and device; the driver migrates pages on demand. It simplifies prototyping and oversubscription; for steady-state performance, guide it with `cudaMemPrefetchAsync` and `cudaMemAdvise`, or use explicit `cudaMalloc` + copies on hot paths.

### E. Async allocation (CUDA 11.2+)
Prefer stream-ordered allocation to recycle device memory cheaply: `cudaMallocAsync`/`cudaFreeAsync` against a `cudaMemPool_t` (set `cudaMemPoolAttrReleaseThreshold` to retain memory and cut allocation latency).

---

## 6. Synchronization, Streams, Events & Graphs

Concurrency *concepts* (races, happens-before, dependencies) are owned by [`parallelism.md`](guides://parallelism.md); below is the CUDA binding.

- **Intra-block:** `__syncthreads()` is a barrier for all threads in a block — every thread must reach it (never inside a divergent branch). Warp-level exchange uses `__shfl_*_sync`/`__ballot_sync` with an explicit mask. Prefer **cooperative groups** (`cg::thread_block`, `cg::coalesced_threads`) over legacy warp intrinsics.
- **Atomics** (`atomicAdd`, `atomicCAS`, …) for cross-thread accumulation; minimize contention (e.g. reduce in shared memory first, then one global atomic per block).
- **Streams:** work in one stream is ordered; different streams may overlap. Use multiple streams + pinned memory + `cudaMemcpyAsync` to overlap copies with compute. The default (null) stream serializes against others unless created with `cudaStreamNonBlocking`.
- **Events:** `cudaEventRecord`/`cudaEventElapsedTime` for GPU timing (always warm up first); `cudaStreamWaitEvent` for cross-stream dependencies.

```cuda
cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s);   // pinned h overlaps...
kernel<<<g, b, 0, s>>>(d, n);                              // ...with this kernel
cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, s);
cudaStreamSynchronize(s);
```

- **CUDA Graphs:** for a fixed, repeated sequence of many small kernels, capture once and replay to cut per-launch overhead (~µs each) to near zero. Capture a stream with `cudaStreamBeginCapture`/`EndCapture`, `cudaGraphInstantiate`, then `cudaGraphLaunch` in the loop; update parameters in place with `cudaGraphExecUpdate` instead of re-instantiating. Use when profiling shows launch overhead matters.

---

## 7. Occupancy, Profiling & Optimization

**Measure first (see [`performance.md`](guides://performance.md)).** Optimize only the bottleneck `ncu` identifies.

### A. Occupancy
Occupancy = active warps / max warps per SM; it hides latency. It is bounded by registers/thread, shared memory/block, and block size. Let the runtime pick a block size, and bound resources when needed:

```cuda
int grid, block;
cudaOccupancyMaxPotentialBlockSize(&grid, &block, kernel, 0, 0);
// Cap registers to raise occupancy: __launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM)
// Inspect usage: nvcc --ptxas-options=-v  → "registers=.., smem=.., occupancy=.."
```
Higher occupancy is not always faster — high-ILP kernels run well at low occupancy. Tune against measured throughput, not the occupancy number.

### B. Profiling workflow
```bash
ncu --set full -o profile ./app          # per-kernel: roofline, memory vs compute
nsys profile -o timeline --stats=true ./app   # whole-app timeline: gaps, overlap, H2D/D2H
```
Read `ncu` to classify each kernel, then act:
- **Memory-bound** (high DRAM throughput) → improve coalescing, stage in shared memory, raise arithmetic intensity, lower precision (fp16/bf16/TF32).
- **Compute-bound** (high SM throughput) → use tensor cores (WMMA / cuBLAS / CUTLASS), cut redundant work, reduce register pressure.
- **Latency-bound** (both low) → raise occupancy/ILP, fuse kernels, hide latency with more in-flight work.
- **Divergence** (`smsp__sass_branch_targets_threads_divergent`) → predicate, or reorganize data so warps are coherent.
- **Bank conflicts** (`l1tex__data_bank_conflicts_pipe_lsu`) → pad shared arrays.

Document the result on the kernel (target arch, achieved TFLOPS/bandwidth vs peak, occupancy, tolerance vs CPU ref).

### C. Optimization idioms
- **Tiling**: stage reused global data into shared memory (matmul, stencils, convolution).
- **Reduction/scan**: reduce in shared memory, finish in-warp with `__shfl_down_sync`; or just use **CUB/Thrust** (`cub::BlockReduce`, `thrust::reduce`).
- **Vectorized I/O**: `float4`/`int4` loads to raise bytes-per-instruction and bandwidth.
- **Fusion**: combine element-wise passes into one kernel (templates/lambdas) to avoid extra launches and round-trips to DRAM.
- **Avoid over-modularization** on the hot path: `__forceinline__ __device__` and templates let the compiler optimize across boundaries.

---

## 8. Error Checking & Memory Safety

CUDA APIs return codes and kernels fail asynchronously — checking is mandatory (CUDA-ERR-01, CUDA-MEM-01).

```cpp
#define CUDA_CHECK(call) do {                                            \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,          \
                cudaGetErrorString(err));                                \
        std::abort();                                                    \
    } } while (0)

CUDA_CHECK(cudaMalloc(&d, bytes));
kernel<<<g, b>>>(d, n);
CUDA_CHECK(cudaGetLastError());        // launch/config errors (sync, immediate)
CUDA_CHECK(cudaDeviceSynchronize());   // execution errors (async) — in debug/tests
```

- Wrap **every** runtime call; after each launch check `cudaGetLastError()` and, when validating, `cudaDeviceSynchronize()`.
- Pair every `cudaMalloc`/`cudaMallocHost` with a free; prefer RAII wrappers (per `cpp.md`) or `cudaMallocAsync`.
- `compute-sanitizer` (`memcheck`/`racecheck`/`initcheck`/`synccheck`) MUST be clean. Treat its findings as build failures.
- Validate transfer sizes and directions; never read GPU results without validation.

---

## 9. Libraries First

Prefer NVIDIA's tuned libraries over custom kernels; drop to custom only when no library fits or a profile proves it wins.

| Need | Use |
|------|-----|
| Dense linear algebra / GEMM | **cuBLAS**, cuBLASLt; **CUTLASS** for fused/custom tiles |
| FFT | **cuFFT**; **cuFFTDx** for block-level, fusible device FFTs |
| Sparse | cuSPARSE | 
| RNG | cuRAND |
| Scan/reduce/sort/select | **Thrust** (host-side) / **CUB** (device/block/warp primitives) |
| Solvers / tensor contractions | cuSOLVER / cuTENSOR |
| Deep learning | cuDNN (and via [`pytorch.md`](guides://pytorch.md)) |

Tensor cores (fp16/bf16/TF32/fp8) are reached through cuBLAS/CUTLASS/cuDNN or the WMMA API — prefer the libraries.

---

## 10. Project Structure & Build

Host-side structure follows [`cpp.md`](guides://cpp.md); CMake policy follows [`cmake.md`](guides://cmake.md). CUDA binding:

```
project/
├── CMakeLists.txt
├── include/   *.cuh         # kernel + utility declarations (CUDA_CHECK, device queries)
├── src/       *.cu          # kernel implementations, host orchestration
├── tests/     *.cu          # GoogleTest: CPU-reference accuracy + regression (see tdd.md)
└── benchmarks/ *.cu         # ncu/nsys harnesses, PERFORMANCE notes
```

```cmake
cmake_minimum_required(VERSION 3.24)
project(app LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 89 90)          # A100, RTX 4090, H100 — target real GPUs
find_package(CUDAToolkit 12.0 REQUIRED)
target_link_libraries(app PRIVATE CUDA::cudart CUDA::cublas)
```

- Compile for every architecture you ship to (`-gencode`/`CMAKE_CUDA_ARCHITECTURES`); a JIT-only PTX fallback is a last resort.
- Pin the CUDA Toolkit version in CI; driver/toolkit mismatches cause subtle failures.
- Manage third-party deps (GoogleTest, fmt, …) and CVE audits via Conan/vcpkg (see [`conan.md`](guides://conan.md), [`secure-coding.md`](guides://secure-coding.md)).

---

## 11. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] CUDA-BLD-01 — compiles for every target arch, warnings-as-errors clean
- [ ] CUDA-ERR-01 — every API call wrapped; `cudaGetLastError()` after each launch
- [ ] CUDA-MEM-01 — `compute-sanitizer` (memcheck/racecheck/initcheck) clean
- [ ] CUDA-ACC-01 — GPU output matches CPU reference within stated tolerance
- [ ] CUDA-TST-01/02 — kernels test-first; bugs have regression tests (see `tdd.md`)
- [ ] CUDA-PERF-01 — optimization claims backed by `ncu`/`nsys` (see `performance.md`)
- [ ] CUDA-SEC-01 — 0 high/critical CVEs; NVIDIA bulletins tracked (see `secure-coding.md`)
- [ ] CUDA-DOC-01 — kernel contracts + perf characteristics documented (see `comments.md`)
- [ ] Agent ran every §3 command and documented any fixes

---
**End of CUDA Programming Guidelines**
