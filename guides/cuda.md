# Modern CUDA Programming Guidelines (C/C++)

This document provides mandatory coding style and practices for CUDA programming in C and C++.

---

**Agent Profile**: The GPU Performance Engineer  
**Role**: Senior CUDA Developer & GPU Architecture Specialist  
**Objective**: Generate production-ready, high-performance, memory-efficient CUDA code.  
**Tools**: CUDA Toolkit 12.x+, NVIDIA Nsight Compute (ncu), NVIDIA Nsight Systems, cuFFTdx, cuBLAS, CUDA-X Libraries, NVPL.

---

## 1. Core Philosophies: PERFORMANCE-FIRST

The agent must adhere to the **PERFORMANCE-FIRST** standard for every CUDA implementation:

- **Test-Driven Development (TDD)**: ALWAYS write tests (CPU reference + GPU accuracy) BEFORE kernel implementation (Red-Green-Refactor cycle mandatory)
- **Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression
- **P**erformance Optimized: Maximum throughput, minimal latency, profiler-verified bottlenecks
- **E**fficient Memory: Minimize host-device transfers, maximize coalesced access, leverage shared memory
- **R**eusable Templates: Prefer compile-time fusion over multi-pass algorithms
- **F**used Kernels: Use templates to fuse operations, avoid kernel launch overhead
- **O**ccupancy Optimized: Maximize GPU utilization, tune block/grid dimensions
- **R**obust Testing: Unit test every kernel, validate GPU vs CPU results, use reference code as gold standard when porting
- **M**inimal Branching: Avoid divergent warps, minimize conditional logic
- **A**synchronous Execution: Use CUDA streams for concurrency, CUDA Graphs for pipelines
- **N**VIDIA Libraries: Prefer cuFFTdx > cuFFT, use cuBLAS, CUDA-X, NVPL for proven performance
- **C**lean Code: Minimalistic, readable, avoid over-modularization when it hurts performance
- **E**rror Checked: Every CUDA call wrapped with error checking, validated before delivery

**F**irst-Class Verification: Agent-generated code MUST compile, run tests, and validate GPU accuracy
- **I**nstrumented: Profile with ncu/Nsight Systems before optimization claims
- **R**eadable: Clear naming, documented performance characteristics
- **S**treaming: Overlap compute and memory transfers with streams
- **T**exture Cache: Use tex1Dfetch/tex2D for non-coalesced or read-only data with locality

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Build & Execution Verification Protocol

**CRITICAL: Agents MUST verify that all generated/modified CUDA code compiles, executes, and passes accuracy tests before presenting to the user.**

#### Verification Checklist

**Before delivering ANY CUDA code, the agent MUST:**

1. **Compilation Verification**:
   ```bash
   # Compile CUDA code with appropriate architecture
   nvcc -std=c++17 -O3 -arch=sm_80 -o test_kernel kernel.cu
   
   # OR with CMake
   mkdir build && cd build
   cmake -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" ..
   cmake --build . --config Release
   ```
   - **MUST** compile without errors (exit code 0)
   - Target appropriate GPU architecture (sm_80 for A100, sm_89 for RTX 4090, sm_90 for H100)
   - Use `-Werror` or equivalent to treat warnings as errors
   - Enable CUDA-specific warnings

2. **Runtime Execution Verification**:
   ```bash
   # Run the compiled binary
   ./test_kernel
   
   # OR with CTest
   ctest --output-on-failure
   ```
   - **MUST** execute without CUDA errors
   - Check `cudaGetLastError()` after kernel launches
   - Verify `cudaDeviceSynchronize()` returns success
   - No segmentation faults or crashes

3. **Accuracy Verification (MANDATORY)**:
   ```cpp
   // EVERY kernel must have CPU reference implementation
   void cpu_reference_kernel(float* output, const float* input, int n) {
       // CPU implementation
   }
   
   void test_gpu_vs_cpu() {
       // 1. Allocate host and device memory
       float *h_input, *h_output_gpu, *h_output_cpu;
       float *d_input, *d_output;
       
       // 2. Initialize input data
       init_data(h_input, n);
       
       // 3. Run GPU kernel
       cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
       my_kernel<<<grid, block>>>(d_output, d_input, n);
       cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost);
       
       // 4. Run CPU reference
       cpu_reference_kernel(h_output_cpu, h_input, n);
       
       // 5. Compare results
       float max_error = 0.0f;
       for (int i = 0; i < n; i++) {
           float error = fabs(h_output_gpu[i] - h_output_cpu[i]);
           max_error = fmax(max_error, error);
       }
       
       // 6. Assert accuracy
       assert(max_error < 1e-5f && "GPU output does not match CPU reference!");
       printf("✓ Accuracy verified: max_error = %e\n", max_error);
   }
   ```
   - Compare GPU results against CPU reference implementation
   - Define acceptable error thresholds (e.g., 1e-5 for float, 1e-10 for double)
   - Test with multiple input sizes and edge cases
   - Validate boundary conditions

4. **Performance Profiling (Required for optimization claims)**:
   ```bash
   # Profile with NVIDIA Nsight Compute
   ncu --set full --target-processes all -o profile_report ./test_kernel
   
   # View metrics
   ncu --import profile_report.ncu-rep --page details
   
   # Key metrics to check:
   # - Memory Bound vs Compute Bound
   # - Occupancy (target: >50% for compute-bound, >25% for memory-bound)
   # - Coalesced memory access (>80%)
   # - Branch divergence (<5%)
   # - Shared memory bank conflicts (0 ideally)
   ```
   - Identify bottlenecks with ncu before optimization
   - Document memory-bound vs compute-bound classification
   - Measure achieved occupancy and bandwidth utilization

5. **Memory Safety Verification**:
   ```bash
   # Run with cuda-memcheck
   cuda-memcheck --tool memcheck ./test_kernel
   
   # OR with compute-sanitizer (CUDA 11.4+)
   compute-sanitizer --tool memcheck ./test_kernel
   ```
   - **MUST** pass without memory errors
   - No out-of-bounds access
   - No uninitialized memory reads
   - No memory leaks

#### Error Correction Process

If verification fails:

1. **Compilation Errors**:
   - Read full compiler error message
   - Check CUDA API version compatibility
   - Verify architecture flags match target GPU
   - Fix syntax, type mismatches, or undefined symbols
   - Re-compile and verify

2. **Runtime Errors**:
   - Check `cudaGetLastError()` and print error string
   - Verify kernel launch configuration (grid/block dimensions)
   - Check for insufficient resources (shared memory, registers)
   - Validate memory allocation sizes
   - Add `cudaDeviceSynchronize()` after kernels for debugging

3. **Accuracy Errors**:
   - Add debug prints for intermediate values
   - Reduce problem size to isolate issue
   - Check for floating-point precision issues
   - Verify atomic operations don't cause race conditions
   - Test with deterministic inputs
   - Compare individual elements, not just aggregates

4. **Performance Issues**:
   - Profile with ncu to identify bottleneck
   - Check occupancy: `ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active`
   - Check memory throughput: `ncu --metrics dram__bytes.sum.per_second`
   - Look for branch divergence: `ncu --metrics smsp__sass_branch_targets_threads_divergent.sum`
   - Optimize based on data: memory-bound needs better access patterns, compute-bound needs more operations per byte

### B. Agent Workflow Example

**Complete CUDA kernel generation workflow:**

1. **Generate Kernel Code**:
   ```cuda
   __global__ void vector_add(float* c, const float* a, const float* b, int n) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < n) {
           c[idx] = a[idx] + b[idx];
       }
   }
   ```

2. **Add Error Checking**:
   ```cpp
   #define CUDA_CHECK(call) \
       do { \
           cudaError_t err = call; \
           if (err != cudaSuccess) { \
               fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                       __FILE__, __LINE__, cudaGetErrorString(err)); \
               exit(EXIT_FAILURE); \
           } \
       } while(0)
   ```

3. **Add CPU Reference**:
   ```cpp
   void vector_add_cpu(float* c, const float* a, const float* b, int n) {
       for (int i = 0; i < n; i++) {
           c[i] = a[i] + b[i];
       }
   }
   ```

4. **Create Test**:
   ```cpp
   void test_vector_add() {
       const int n = 1024 * 1024;
       size_t size = n * sizeof(float);
       
       // Allocate and initialize
       float *h_a, *h_b, *h_c_gpu, *h_c_cpu;
       h_a = (float*)malloc(size);
       h_b = (float*)malloc(size);
       h_c_gpu = (float*)malloc(size);
       h_c_cpu = (float*)malloc(size);
       
       for (int i = 0; i < n; i++) {
           h_a[i] = (float)i;
           h_b[i] = (float)(i * 2);
       }
       
       // GPU execution
       float *d_a, *d_b, *d_c;
       CUDA_CHECK(cudaMalloc(&d_a, size));
       CUDA_CHECK(cudaMalloc(&d_b, size));
       CUDA_CHECK(cudaMalloc(&d_c, size));
       
       CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
       CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
       
       int blockSize = 256;
       int gridSize = (n + blockSize - 1) / blockSize;
       vector_add<<<gridSize, blockSize>>>(d_c, d_a, d_b, n);
       CUDA_CHECK(cudaGetLastError());
       CUDA_CHECK(cudaDeviceSynchronize());
       
       CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost));
       
       // CPU reference
       vector_add_cpu(h_c_cpu, h_a, h_b, n);
       
       // Verify
       float max_error = 0.0f;
       for (int i = 0; i < n; i++) {
           float error = fabs(h_c_gpu[i] - h_c_cpu[i]);
           max_error = fmax(max_error, error);
       }
       
       printf("✓ Test passed: max_error = %e\n", max_error);
       assert(max_error < 1e-5f);
       
       // Cleanup
       CUDA_CHECK(cudaFree(d_a));
       CUDA_CHECK(cudaFree(d_b));
       CUDA_CHECK(cudaFree(d_c));
       free(h_a); free(h_b); free(h_c_gpu); free(h_c_cpu);
   }
   ```

5. **Compile and Verify**:
   ```bash
   nvcc -std=c++17 -O3 -arch=sm_80 -o test test.cu
   ./test
   # ✓ Test passed: max_error = 0.000000e+00
   ```

6. **Profile Performance**:
   ```bash
   ncu --set full -o profile ./test
   # Check: Memory-bound, coalesced access, good occupancy
   ```

7. **Optimize (if needed)** based on profiling data:
   - Memory-bound → improve access patterns, use shared memory
   - Compute-bound → increase arithmetic intensity, use tensor cores
   - Low occupancy → adjust block size, reduce register usage

8. **Re-verify** after optimization:
   - Compile, run tests, check accuracy
   - Profile again to confirm improvements

9. **Present Code**: Only after ALL checks pass

### C. Reference Code Porting Requirements (MANDATORY)

**When reference code is provided for porting to CUDA:**

**CRITICAL: If the user provides existing CPU/sequential code to be ported to CUDA, the agent MUST use that reference implementation as the gold standard for accuracy validation.**

#### Porting Workflow

1. **Analyze Reference Implementation**:
   ```cpp
   // Example: User provides C++ reference code
   void matrix_multiply_reference(float* C, const float* A, const float* B, int N) {
       for (int i = 0; i < N; i++) {
           for (int j = 0; j < N; j++) {
               float sum = 0.0f;
               for (int k = 0; k < N; k++) {
                   sum += A[i * N + k] * B[k * N + j];
               }
               C[i * N + j] = sum;
           }
       }
   }
   ```
   - Understand the algorithm, input/output specifications, edge cases
   - Identify numerical precision requirements
   - Note any special handling (NaN, Inf, boundary conditions)

2. **Design Comprehensive Unit Tests Using Reference Code**:
   ```cpp
   class MatrixMultiplyPortingTest : public ::testing::Test {
   protected:
       void SetUp() override {
           // Setup test data
       }
       
       void validate_against_reference(int N) {
           // Allocate memory
           float *h_A, *h_B, *h_C_cuda, *h_C_reference;
           h_A = new float[N * N];
           h_B = new float[N * N];
           h_C_cuda = new float[N * N];
           h_C_reference = new float[N * N];
           
           // Initialize with test data
           for (int i = 0; i < N * N; i++) {
               h_A[i] = static_cast<float>(rand()) / RAND_MAX;
               h_B[i] = static_cast<float>(rand()) / RAND_MAX;
           }
           
           // Run reference implementation (PROVIDED BY USER)
           matrix_multiply_reference(h_C_reference, h_A, h_B, N);
           
           // Run CUDA implementation
           float *d_A, *d_B, *d_C;
           CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(float)));
           CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(float)));
           CUDA_CHECK(cudaMalloc(&d_C, N * N * sizeof(float)));
           
           CUDA_CHECK(cudaMemcpy(d_A, h_A, N * N * sizeof(float), 
                                cudaMemcpyHostToDevice));
           CUDA_CHECK(cudaMemcpy(d_B, h_B, N * N * sizeof(float), 
                                cudaMemcpyHostToDevice));
           
           // CUDA kernel (agent-generated)
           dim3 block(16, 16);
           dim3 grid((N + 15) / 16, (N + 15) / 16);
           matrix_multiply_cuda<<<grid, block>>>(d_C, d_A, d_B, N);
           CUDA_CHECK(cudaGetLastError());
           CUDA_CHECK(cudaDeviceSynchronize());
           
           CUDA_CHECK(cudaMemcpy(h_C_cuda, d_C, N * N * sizeof(float),
                                cudaMemcpyDeviceToHost));
           
           // Compare against reference
           float max_error = 0.0f;
           int error_count = 0;
           for (int i = 0; i < N * N; i++) {
               float error = fabs(h_C_cuda[i] - h_C_reference[i]);
               if (error > 1e-4f) {  // Tolerance based on reference precision
                   error_count++;
                   if (error_count <= 10) {  // Print first 10 errors
                       std::cerr << "Error at index " << i 
                                 << ": CUDA=" << h_C_cuda[i]
                                 << ", Reference=" << h_C_reference[i]
                                 << ", diff=" << error << std::endl;
                   }
               }
               max_error = fmaxf(max_error, error);
           }
           
           EXPECT_EQ(error_count, 0) << "Found " << error_count 
                                     << " mismatches against reference";
           EXPECT_LT(max_error, 1e-4f) << "Max error: " << max_error;
           
           // Cleanup
           delete[] h_A; delete[] h_B; delete[] h_C_cuda; delete[] h_C_reference;
           CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
       }
   };
   
   // Test cases based on reference implementation
   TEST_F(MatrixMultiplyPortingTest, SmallMatrix) {
       validate_against_reference(16);
   }
   
   TEST_F(MatrixMultiplyPortingTest, MediumMatrix) {
       validate_against_reference(128);
   }
   
   TEST_F(MatrixMultiplyPortingTest, LargeMatrix) {
       validate_against_reference(1024);
   }
   
   TEST_F(MatrixMultiplyPortingTest, EdgeCases) {
       // Test edge cases specific to reference implementation
       validate_against_reference(1);    // Minimum size
       validate_against_reference(33);   // Non-power-of-2
       validate_against_reference(127);  // Prime number
   }
   
   TEST_F(MatrixMultiplyPortingTest, SpecialValues) {
       // Test special input values
       const int N = 16;
       float *h_A, *h_B, *h_C_cuda, *h_C_reference;
       h_A = new float[N * N];
       h_B = new float[N * N];
       h_C_cuda = new float[N * N];
       h_C_reference = new float[N * N];
       
       // Test with zeros
       std::fill(h_A, h_A + N * N, 0.0f);
       std::fill(h_B, h_B + N * N, 1.0f);
       
       matrix_multiply_reference(h_C_reference, h_A, h_B, N);
       // Run CUDA version and compare...
       
       // Test with identity matrix
       std::fill(h_A, h_A + N * N, 0.0f);
       for (int i = 0; i < N; i++) h_A[i * N + i] = 1.0f;  // Identity
       
       matrix_multiply_reference(h_C_reference, h_A, h_B, N);
       // Run CUDA version and compare...
       
       // Cleanup
       delete[] h_A; delete[] h_B; delete[] h_C_cuda; delete[] h_C_reference;
   }
   ```

3. **Implement CUDA Version**:
   - Port algorithm to CUDA
   - Optimize for GPU (coalescing, shared memory, etc.)
   - **MAINTAIN** exact numerical behavior of reference unless user specifies otherwise

4. **Validate Every Change Against Reference**:
   ```bash
   # After ANY modification to CUDA code, re-run tests
   ctest --output-on-failure
   
   # All tests MUST pass before delivery
   # Example output:
   # Test project /path/to/build
   #     Start 1: MatrixMultiplyPortingTest.SmallMatrix
   # 1/5 Test #1: MatrixMultiplyPortingTest.SmallMatrix ......   Passed    0.15 sec
   #     Start 2: MatrixMultiplyPortingTest.MediumMatrix
   # 2/5 Test #2: MatrixMultiplyPortingTest.MediumMatrix .....   Passed    0.23 sec
   #     Start 3: MatrixMultiplyPortingTest.LargeMatrix
   # 3/5 Test #3: MatrixMultiplyPortingTest.LargeMatrix ......   Passed    1.34 sec
   #     Start 4: MatrixMultiplyPortingTest.EdgeCases
   # 4/5 Test #4: MatrixMultiplyPortingTest.EdgeCases ........   Passed    0.45 sec
   #     Start 5: MatrixMultiplyPortingTest.SpecialValues
   # 5/5 Test #5: MatrixMultiplyPortingTest.SpecialValues ....   Passed    0.18 sec
   #
   # 100% tests passed, 0 tests failed out of 5
   ```

5. **Document Differences** (if any):
   ```cpp
   /**
    * CUDA Implementation Notes:
    * 
    * Changes from reference implementation:
    * - Uses tiled algorithm for better memory access (preserves numerical accuracy)
    * - Processes multiple elements per thread (maintains reference behavior)
    * 
    * Numerical accuracy:
    * - Matches reference implementation within 1e-5 for float32
    * - Tested against reference for N ∈ [1, 16, 128, 1024]
    * - All edge cases validated (zeros, identity, special values)
    * 
    * Performance improvements:
    * - 45x speedup vs reference on A100 for N=1024
    * - 85% of peak GEMM throughput
    */
   ```

#### Reference Code Testing Requirements

**MANDATORY when reference code is provided:**

- [ ] Reference implementation integrated into test suite
- [ ] All CUDA outputs validated against reference implementation
- [ ] Test suite covers all input sizes used in reference code
- [ ] Edge cases from reference implementation tested
- [ ] Special value handling (NaN, Inf, zeros) matches reference
- [ ] Numerical tolerance documented and justified
- [ ] Any deviations from reference behavior explicitly documented and justified
- [ ] Tests run and pass after EVERY change to CUDA code
- [ ] Performance comparison included (CUDA speedup vs reference)

**Exception Handling:**
- If user explicitly requests deviation from reference behavior (e.g., "use lower precision", "approximate this function"), document it clearly
- Maintain reference test as baseline, add separate test for modified behavior
- Clearly mark which behavior is intentional vs reference

#### Example: Complete Porting Test Suite

```cpp
// reference_port_tests.cu
#include <gtest/gtest.h>
#include "reference_implementation.h"  // User-provided reference code
#include "cuda_implementation.cuh"     // Agent-generated CUDA code

class ReferencePortTest : public ::testing::Test {
protected:
    // Helper to compare CUDA output against reference
    template<typename T>
    void compare_with_reference(
        std::function<void(T*, const T*, int)> reference_func,
        std::function<void(T*, const T*, int)> cuda_func,
        int N, T tolerance = 1e-5
    ) {
        size_t size = N * sizeof(T);
        
        // Allocate host memory
        T *h_input = new T[N];
        T *h_output_ref = new T[N];
        T *h_output_cuda = new T[N];
        
        // Initialize input
        for (int i = 0; i < N; i++) {
            h_input[i] = static_cast<T>(rand()) / RAND_MAX;
        }
        
        // Run reference
        reference_func(h_output_ref, h_input, N);
        
        // Run CUDA
        T *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, size));
        CUDA_CHECK(cudaMalloc(&d_output, size));
        CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
        
        cuda_func(d_output, d_input, N);
        
        CUDA_CHECK(cudaMemcpy(h_output_cuda, d_output, size, cudaMemcpyDeviceToHost));
        
        // Compare
        T max_error = 0;
        for (int i = 0; i < N; i++) {
            T error = std::abs(h_output_cuda[i] - h_output_ref[i]);
            max_error = std::max(max_error, error);
        }
        
        EXPECT_LT(max_error, tolerance) 
            << "CUDA output differs from reference by " << max_error;
        
        // Cleanup
        delete[] h_input; delete[] h_output_ref; delete[] h_output_cuda;
        CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_output));
    }
};

TEST_F(ReferencePortTest, ValidateAgainstReference) {
    compare_with_reference<float>(
        reference_implementation,  // User-provided
        cuda_wrapper_function,     // Agent-generated
        1024 * 1024,
        1e-5f
    );
}
```

#### Agent Responsibilities for Reference Code Porting

**The agent MUST:**
1. **Ask** for the reference implementation if not provided
2. **Use** the reference as the authoritative source of correctness
3. **Design** comprehensive tests comparing CUDA vs reference
4. **Run** tests after every CUDA code modification
5. **Pass** all tests before delivering code
6. **Document** any intentional deviations (with user approval)
7. **Measure** and report speedup vs reference

**The agent MUST NOT:**
- Deliver CUDA code without reference-based validation
- Modify reference behavior without explicit user request
- Skip tests during iterative development
- Claim "equivalence" without testing proof

**Unless the user explicitly states:**
- "Don't worry about matching the reference exactly"
- "Use approximate/lower precision"
- "Modify the algorithm for GPU"
- In these cases, document the deviation and maintain reference test as baseline

### D. Prohibited Practices

**NEVER deliver CUDA code that:**
- ❌ Fails to compile for target GPU architecture
- ❌ Crashes or produces CUDA runtime errors
- ❌ Lacks CPU reference implementation for validation
- ❌ Has untested accuracy (GPU vs CPU comparison required)
- ❌ **When porting: Fails to validate against provided reference implementation**
- ❌ **When porting: Lacks comprehensive tests based on reference code**
- ❌ Lacks error checking on CUDA API calls
- ❌ Uses `cudaMalloc` without checking return value
- ❌ Has memory leaks (missing `cudaFree`)
- ❌ Uses synchronous copies (`cudaMemcpy`) when async would work
- ❌ Ignores profiling data and makes optimization claims
- ❌ Has excessive branching without profiler evidence it's not a bottleneck
- ❌ Over-modularizes at the expense of kernel fusion
- ❌ Uses deprecated APIs (e.g., `__syncthreads_count()` instead of cooperative groups)
- ❌ Lacks documentation of performance characteristics
- ❌ Uses `printf` in kernels without warning about performance impact
- ❌ **When porting: Delivers code with known deviations from reference without user approval**
- ❌ **Fixes bugs without adding regression tests first**
- ❌ **Writes kernel implementation before writing tests (violates TDD)**
- ❌ **Skips Red-Green-Refactor cycle for new kernels**

---

## 2A. Test-Driven Development (TDD) Protocol for CUDA (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new CUDA kernels.**

### TDD Cycle for CUDA

```
1. 🔴 RED: Write CPU reference + failing GPU test first
   ↓
2. 🟢 GREEN: Write minimal GPU kernel to make it pass
   ↓
3. 🔵 REFACTOR: Optimize kernel while keeping tests green
   ↓
4. 📊 PROFILE: Verify performance with ncu/Nsight
   ↓
   Repeat
```

### Example TDD Workflow for CUDA Kernel

```cpp
// Step 1: RED - Write CPU reference and failing test first
// tests/test_vector_add.cu
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// CPU reference implementation (write this FIRST)
void vector_add_cpu(float* c, const float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// Forward declaration of GPU kernel (doesn't exist yet)
__global__ void vector_add_gpu(float* c, const float* a, const float* b, int n);

TEST(VectorAdd, GPUMatchesCPU) {
    // Test will fail - kernel doesn't exist yet
    const int n = 1024;
    const size_t bytes = n * sizeof(float);
    
    // Host memory
    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    std::vector<float> h_c_gpu(n);
    std::vector<float> h_c_cpu(n);
    
    // Device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy to device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
    
    // Run GPU kernel
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    vector_add_gpu<<<grid_size, block_size>>>(d_c, d_a, d_b, n);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Run CPU reference
    vector_add_cpu(h_c_cpu.data(), h_a.data(), h_b.data(), n);
    
    // Compare results
    float max_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        float error = std::abs(h_c_gpu[i] - h_c_cpu[i]);
        max_error = std::max(max_error, error);
    }
    
    EXPECT_LT(max_error, 1e-5f);
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// Run: ctest
// ❌ FAILS - vector_add_gpu doesn't exist yet

// Step 2: GREEN - Write minimal GPU kernel implementation
// src/vector_add.cu
/**
 * @brief Adds two vectors element-wise on GPU.
 * 
 * Performs c[i] = a[i] + b[i] for all elements.
 * 
 * @param c Output vector (device pointer)
 * @param a First input vector (device pointer)
 * @param b Second input vector (device pointer)
 * @param n Number of elements
 * 
 * @note Each thread processes one element
 * @note Requires n threads total
 * 
 * @par Example
 * @code
 * int block_size = 256;
 * int grid_size = (n + block_size - 1) / block_size;
 * vector_add_gpu<<<grid_size, block_size>>>(d_c, d_a, d_b, n);
 * cudaDeviceSynchronize();
 * @endcode
 */
__global__ void vector_add_gpu(float* c, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Run: ctest
// ✅ PASSES - tests pass

// Step 3: REFACTOR - Optimize with vectorized loads
/**
 * @brief Optimized vector addition using float4 vectorization.
 * 
 * Performs vectorized element-wise addition: c[i] = a[i] + b[i]
 * Uses float4 for coalesced memory access and improved bandwidth.
 * 
 * @param c Output vector (device pointer, must be 16-byte aligned)
 * @param a First input vector (device pointer, must be 16-byte aligned)
 * @param b Second input vector (device pointer, must be 16-byte aligned)
 * @param n Number of elements (must be multiple of 4)
 * 
 * @pre n must be divisible by 4
 * @pre All pointers must be 16-byte aligned
 * @post c[i] = a[i] + b[i] for all i in [0, n)
 * 
 * @note Each thread processes 4 elements using float4
 * @note Achieves ~2x bandwidth compared to naive version
 * 
 * @par Performance
 * - Memory bandwidth: ~800 GB/s on A100
 * - Occupancy: 100% with block_size=256
 * 
 * @par Example
 * @code
 * // Ensure n is multiple of 4
 * int n_aligned = (n + 3) & ~3;
 * 
 * int block_size = 256;
 * int grid_size = (n_aligned / 4 + block_size - 1) / block_size;
 * vector_add_gpu_optimized<<<grid_size, block_size>>>(d_c, d_a, d_b, n_aligned);
 * cudaDeviceSynchronize();
 * @endcode
 */
__global__ void vector_add_gpu_optimized(float* c, const float* a, const float* b, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx < n) {
        // Vectorized load (4 floats at once)
        float4 a4 = reinterpret_cast<const float4*>(a)[idx / 4];
        float4 b4 = reinterpret_cast<const float4*>(b)[idx / 4];
        
        // Vectorized compute
        float4 c4;
        c4.x = a4.x + b4.x;
        c4.y = a4.y + b4.y;
        c4.z = a4.z + b4.z;
        c4.w = a4.w + b4.w;
        
        // Vectorized store
        reinterpret_cast<float4*>(c)[idx / 4] = c4;
    }
}
// Tests still pass ✓

// Step 4: PROFILE - Verify performance
// Run: ncu --set full ./test_vector_add
// Verify: Memory bandwidth, occupancy, warp efficiency
```

### Example TDD for Complex CUDA Kernel (Matrix Multiply)

```cpp
// Step 1: RED - Write CPU reference and failing test
// tests/test_matmul.cu
#include <gtest/gtest.h>
#include <cuda_runtime.h>

// CPU reference implementation (NAIVE - for correctness, not speed)
void matmul_cpu(float* C, const float* A, const float* B, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Forward declaration (doesn't exist yet)
__global__ void matmul_gpu(float* C, const float* A, const float* B, int M, int N, int K);

TEST(MatMul, SmallMatrix_GPUMatchesCPU) {
    // Test will fail - kernel doesn't exist yet
    const int M = 64, N = 64, K = 64;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    
    // Host memory
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 2.0f);
    std::vector<float> h_C_gpu(M * N);
    std::vector<float> h_C_cpu(M * N);
    
    // Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);
    
    // Run GPU kernel
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_gpu<<<grid, block>>>(d_C, d_A, d_B, M, N, K);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C_gpu.data(), d_C, size_C, cudaMemcpyDeviceToHost);
    
    // Run CPU reference
    matmul_cpu(h_C_cpu.data(), h_A.data(), h_B.data(), M, N, K);
    
    // Compare results
    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C_gpu[i] - h_C_cpu[i]);
        max_error = std::max(max_error, error);
    }
    
    EXPECT_LT(max_error, 1e-3f);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Run: ctest
// ❌ FAILS - matmul_gpu doesn't exist yet

// Step 2: GREEN - Write minimal GPU kernel
/**
 * @brief Naive matrix multiplication kernel.
 * 
 * Computes C = A * B where:
 * - A is M x K
 * - B is K x N
 * - C is M x N
 * 
 * @param C Output matrix (device pointer)
 * @param A First input matrix (device pointer)
 * @param B Second input matrix (device pointer)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * 
 * @note Each thread computes one element of C
 * @note This is a naive implementation for correctness, not performance
 */
__global__ void matmul_gpu(float* C, const float* A, const float* B, 
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Run: ctest
// ✅ PASSES - tests pass

// Step 3: REFACTOR - Optimize with shared memory tiling
/**
 * @brief Optimized matrix multiplication with shared memory tiling.
 * 
 * Uses shared memory to cache tiles of A and B, reducing global memory access.
 * Achieves significantly higher performance than naive version.
 * 
 * @param C Output matrix (device pointer)
 * @param A First input matrix (device pointer)
 * @param B Second input matrix (device pointer)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * 
 * @tparam TILE_SIZE Size of shared memory tile (typically 16 or 32)
 * 
 * @note Requires shared memory: 2 * TILE_SIZE * TILE_SIZE * sizeof(float)
 * @note Achieves ~10x speedup over naive version
 * 
 * @par Performance
 * - TFLOPS: ~15 on A100 for large matrices
 * - Memory bandwidth: ~1.2 TB/s
 * - Occupancy: 75-100% depending on TILE_SIZE
 */
template<int TILE_SIZE = 16>
__global__ void matmul_gpu_tiled(float* C, const float* A, const float* B,
                                  int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile of A into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile of B into shared memory
        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
// Tests still pass ✓

// Step 4: PROFILE
// Run: ncu --set full ./test_matmul
// Verify: FLOPS, memory bandwidth, shared memory efficiency
```

---

## 2B. Bug Fix Protocol for CUDA (MANDATORY)

**CRITICAL: Every CUDA bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow for CUDA

```
1. 🐛 Bug Reported/Discovered (incorrect results, crash, performance issue)
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
7. 📊 PROFILE to ensure fix doesn't hurt performance
   ↓
8. 🚀 Deploy with confidence (regression prevented)
```

### Example Bug Fix: Incorrect Results

```cpp
// Bug Report #2341: reduce_sum gives wrong results for large arrays

// Step 1-2: Write test that reproduces the bug
// tests/test_reduce.cu
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <numeric>

// CPU reference
float reduce_sum_cpu(const float* data, int n) {
    return std::accumulate(data, data + n, 0.0f);
}

// Forward declaration
__global__ void reduce_sum_gpu(float* output, const float* input, int n);

TEST(Reduce, LargeArray_Bug2341) {
    // Bug #2341: reduce_sum gives wrong results for n > 1024
    // Discovered: 2026-01-18
    // This test prevents regression
    
    const int n = 4096;  // Larger than single block
    std::vector<float> h_input(n, 1.0f);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    
    cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run GPU kernel
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    reduce_sum_gpu<<<grid_size, block_size>>>(d_output, d_input, n);
    cudaDeviceSynchronize();
    
    float h_output_gpu;
    cudaMemcpy(&h_output_gpu, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    // CPU reference
    float h_output_cpu = reduce_sum_cpu(h_input.data(), n);
    
    // Compare
    float error = std::abs(h_output_gpu - h_output_cpu);
    EXPECT_LT(error, 1e-3f) << "GPU: " << h_output_gpu << ", CPU: " << h_output_cpu;
    
    cudaFree(d_input);
    cudaFree(d_output);
}

// Run: ctest
// ❌ FAILS - GPU returns wrong value (only sums first 1024 elements)

// Step 3: Fix the bug
/**
 * @brief Two-stage reduction for arrays larger than block size.
 * 
 * Stage 1: Each block reduces its portion to a single value
 * Stage 2: Reduce the per-block results (if needed)
 * 
 * @param output Output sum (device pointer)
 * @param input Input array (device pointer)
 * @param n Number of elements
 * 
 * @note Handles arrays of any size
 * @note Uses shared memory for efficient reduction
 * 
 * @par Bug Fix
 * Fixed Bug #2341: Now correctly handles arrays > 1024 elements
 * by using multi-block reduction instead of assuming single block.
 */
__global__ void reduce_sum_gpu_stage1(float* block_sums, const float* input, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_sum_gpu_stage2(float* output, const float* block_sums, int num_blocks) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    
    // Load block sums
    sdata[tid] = (tid < num_blocks) ? block_sums[tid] : 0.0f;
    __syncthreads();
    
    // Final reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *output = sdata[0];
    }
}

// Wrapper function
void reduce_sum_gpu(float* output, const float* input, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    // Allocate temporary storage for block sums
    float* d_block_sums;
    cudaMalloc(&d_block_sums, grid_size * sizeof(float));
    
    // Stage 1: Reduce each block
    reduce_sum_gpu_stage1<<<grid_size, block_size, block_size * sizeof(float)>>>(
        d_block_sums, input, n
    );
    
    // Stage 2: Reduce block sums
    reduce_sum_gpu_stage2<<<1, block_size, block_size * sizeof(float)>>>(
        output, d_block_sums, grid_size
    );
    
    cudaFree(d_block_sums);
}

// Run: ctest
// ✅ PASSES - bug fixed, regression prevented ✓
```

### Example Bug Fix: Memory Corruption

```cpp
// Bug Report #2342: Kernel crashes with large input sizes

// Step 1-2: Write test that reproduces the bug
TEST(Convolution, LargeImage_Bug2342) {
    // Bug #2342: Kernel crashes for images > 2048x2048
    // Discovered: 2026-01-18
    // This test prevents regression
    
    const int width = 4096;
    const int height = 4096;
    const int kernel_size = 5;
    
    std::vector<float> h_input(width * height, 1.0f);
    std::vector<float> h_output(width * height);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));
    
    cudaMemcpy(d_input, h_input.data(), width * height * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // This should not crash
    EXPECT_NO_THROW({
        convolution_2d<<<grid, block>>>(d_output, d_input, width, height, kernel_size);
        cudaError_t err = cudaDeviceSynchronize();
        EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
    });
    
    cudaFree(d_input);
    cudaFree(d_output);
}

// Run: ctest
// ❌ FAILS - Kernel crashes with "illegal memory access"

// Step 3: Fix the bug
/**
 * @brief 2D convolution with bounds checking.
 * 
 * @param output Output image (device pointer)
 * @param input Input image (device pointer)
 * @param width Image width
 * @param height Image height
 * @param kernel_size Convolution kernel size (must be odd)
 * 
 * @par Bug Fix
 * Fixed Bug #2342: Added proper bounds checking to prevent
 * out-of-bounds memory access for large images.
 * 
 * @note Now handles images of any size safely
 */
__global__ void convolution_2d(float* output, const float* input,
                                int width, int height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // FIX: Add bounds check BEFORE accessing memory
    if (x >= width || y >= height) {
        return;  // Out of bounds, exit early
    }
    
    int half_kernel = kernel_size / 2;
    float sum = 0.0f;
    
    for (int ky = -half_kernel; ky <= half_kernel; ++ky) {
        for (int kx = -half_kernel; kx <= half_kernel; ++kx) {
            int nx = x + kx;
            int ny = y + ky;
            
            // FIX: Check bounds for each neighbor access
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += input[ny * width + nx];
            }
        }
    }
    
    output[y * width + x] = sum / (kernel_size * kernel_size);
}

// Run: ctest
// ✅ PASSES - bug fixed, no crashes, regression prevented ✓
```

### Prohibited Practices for CUDA Bug Fixes

**NEVER:**
- ❌ Fix a CUDA bug without adding a regression test first
- ❌ Write kernel implementation before writing tests (violates TDD)
- ❌ Skip the Red-Green-Refactor cycle
- ❌ Commit code with failing tests
- ❌ Remove tests to make code pass
- ❌ Disable CUDA error checking to hide bugs
- ❌ Ignore cuda-memcheck or compute-sanitizer warnings

**ALWAYS:**
- ✅ Write a test that reproduces the bug first (CPU reference + GPU test)
- ✅ Verify the test fails before fixing
- ✅ Document bug ID in test comments
- ✅ Run with cuda-memcheck or compute-sanitizer
- ✅ Profile after fix to ensure no performance regression
- ✅ Keep tests in codebase permanently
- ✅ Test with multiple input sizes and edge cases

---

## 3. CUDA Libraries & Performance Hierarchy (MANDATORY)

### A. Library Preference Order

**ALWAYS prefer optimized CUDA libraries over custom implementations:**

1. **CUDA-X Libraries** (Highest Priority):
   - **cuFFTdx** (ALWAYS prefer over cuFFT): Thread-block-level FFTs, fused kernels, lower overhead
   - **cuBLAS**: Matrix operations, GEMM, optimized for tensor cores
   - **cuBLASLt**: Low-level tile operations, custom epilogues
   - **cuSPARSE**: Sparse matrix operations
   - **cuRAND**: Random number generation on GPU
   - **Thrust**: GPU-accelerated STL-like algorithms
   - **CUB**: Warp-level and block-level primitives

2. **NVPL (NVIDIA Performance Libraries)**: CPU-side optimized math libraries

3. **cuFFT** (Fallback): Only if cuFFTdx doesn't support the use case (e.g., very large FFTs that need multi-GPU)

4. **Custom Kernels** (Last Resort): Only when no library exists or profiling proves custom is faster

### B. cuFFTdx over cuFFT (MANDATORY)

**ALWAYS prefer cuFFTdx for FFT operations:**

```cpp
// ❌ WRONG - Using cuFFT (legacy approach)
#include <cufft.h>

cufftHandle plan;
cufftPlan1d(&plan, N, CUFFT_C2C, batch);
cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD);
cufftDestroy(plan);

// ✅ CORRECT - Using cuFFTdx (modern, fused, higher performance)
#include <cufftdx.hpp>

using FFT = decltype(cufftdx::Block() + 
                     cufftdx::Size<N>() + 
                     cufftdx::Type<cufftdx::fft_type::c2c>() +
                     cufftdx::Direction<cufftdx::fft_direction::forward>() +
                     cufftdx::Precision<float>() +
                     cufftdx::SM<800>());

template<class FFT>
__global__ void fft_kernel(typename FFT::value_type* data) {
    using complex_type = typename FFT::value_type;
    extern __shared__ complex_type shared_mem[];
    
    // Load data to shared memory
    unsigned int offset = blockIdx.x * FFT::storage_size;
    complex_type thread_data[FFT::storage_size];
    
    // Execute FFT
    FFT().execute(thread_data, shared_mem);
    
    // Store results (fused with other operations if needed)
}
```

**Why cuFFTdx > cuFFT:**
- Thread-block-level execution (can fuse with other operations)
- Lower overhead (no CPU-GPU sync)
- Better for small-to-medium FFTs (N ≤ 8192)
- Enables kernel fusion for complex pipelines
- Higher performance for batched operations

**When to use cuFFT:**
- Very large FFTs (N > 8192) requiring global memory
- Multi-GPU FFTs
- Legacy codebase integration with compelling reason

### C. cuBLAS for Matrix Operations

```cpp
// ✅ CORRECT - Using cuBLAS for GEMM
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

// Matrix multiplication: C = alpha * A * B + beta * C
float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            d_A, m,
            d_B, k,
            &beta,
            d_C, m);

cublasDestroy(handle);

// ❌ WRONG - Custom naive matrix multiply (unless profiling proves necessary)
__global__ void naive_matmul(float* C, float* A, float* B, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}
```

---

## 4. Memory Optimization (MANDATORY)

### A. Minimize Host-Device Transfers

**Memory copies are expensive. Minimize them aggressively:**

```cpp
// ❌ WRONG - Excessive transfers
for (int i = 0; i < iterations; i++) {
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);  // ← Avoid!
    kernel<<<grid, block>>>(d_data);
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);  // ← Avoid!
}

// ✅ CORRECT - Transfer once, compute many times
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
for (int i = 0; i < iterations; i++) {
    kernel<<<grid, block>>>(d_data);
}
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

// ✅ EVEN BETTER - Keep data on GPU entirely
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
for (int i = 0; i < iterations; i++) {
    kernel1<<<grid, block>>>(d_data);
    kernel2<<<grid, block>>>(d_data);  // ← No transfers between kernels
}
cudaMemcpy(h_result, d_data, size, cudaMemcpyDeviceToHost);
```

### B. Coalesced Memory Access

**Ensure threads in a warp access contiguous memory:**

```cpp
// ❌ WRONG - Strided access (poor coalescing)
__global__ void transpose_bad(float* out, float* in, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < n && y < n) {
        out[x * n + y] = in[y * n + x];  // ← Non-coalesced write
    }
}

// ✅ CORRECT - Use shared memory for coalescing
__global__ void transpose_good(float* out, float* in, int n) {
    __shared__ float tile[32][33];  // +1 to avoid bank conflicts
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];  // Coalesced read
    }
    __syncthreads();
    
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < n && y < n) {
        out[y * n + x] = tile[threadIdx.x][threadIdx.y];  // Coalesced write
    }
}
```

### C. Shared Memory Usage

**Use shared memory for data reuse within a block:**

```cpp
// ✅ CORRECT - Shared memory for reduction
template<unsigned int blockSize>
__global__ void reduce_sum(float* out, const float* in, int n) {
    __shared__ float sdata[blockSize];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
    
    // Load from global to shared memory
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  sdata[tid] += sdata[tid + 64];  __syncthreads(); }
    
    // Warp-level reduction (no __syncthreads needed)
    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockSize >= 64) smem[tid] += smem[tid + 32];
        if (blockSize >= 32) smem[tid] += smem[tid + 16];
        if (blockSize >= 16) smem[tid] += smem[tid + 8];
        if (blockSize >= 8)  smem[tid] += smem[tid + 4];
        if (blockSize >= 4)  smem[tid] += smem[tid + 2];
        if (blockSize >= 2)  smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) out[blockIdx.x] = sdata[0];
}
```

### D. Texture Memory for Non-Coalesced Access

**Use texture cache for read-only data with non-coalesced access patterns:**

```cpp
// ✅ CORRECT - Texture memory for irregular access
texture<float, 1, cudaReadModeElementType> tex_data;

__global__ void kernel_with_texture(float* out, const int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int lookup_idx = indices[idx];  // Irregular access pattern
        out[idx] = tex1Dfetch(tex_data, lookup_idx);  // Use texture cache
    }
}

// Setup texture
cudaBindTexture(0, tex_data, d_input, n * sizeof(float));
kernel_with_texture<<<grid, block>>>(d_out, d_indices, n);
cudaUnbindTexture(tex_data);

// ✅ MODERN - Use texture objects (CUDA 5.0+)
cudaTextureObject_t tex_obj;
cudaResourceDesc res_desc = {};
res_desc.resType = cudaResourceTypeLinear;
res_desc.res.linear.devPtr = d_input;
res_desc.res.linear.sizeInBytes = n * sizeof(float);
res_desc.res.linear.desc = cudaCreateChannelDesc<float>();

cudaTextureDesc tex_desc = {};
tex_desc.readMode = cudaReadModeElementType;

cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr);

__global__ void kernel_with_texture_obj(float* out, cudaTextureObject_t tex, 
                                        const int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int lookup_idx = indices[idx];
        out[idx] = tex1Dfetch<float>(tex, lookup_idx);
    }
}

kernel_with_texture_obj<<<grid, block>>>(d_out, tex_obj, d_indices, n);
cudaDestroyTextureObject(tex_obj);
```

---

## 5. Performance Optimization Patterns (MANDATORY)

### A. Kernel Fusion with Templates

**Prefer template-based kernel fusion over multiple passes:**

```cpp
// ❌ WRONG - Multiple kernel launches (high overhead)
__global__ void kernel_add(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] + 1.0f;
}

__global__ void kernel_multiply(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * 2.0f;
}

// Three kernel launches
kernel_add<<<grid, block>>>(d_temp, d_in, n);
kernel_multiply<<<grid, block>>>(d_out, d_temp, n);

// ✅ CORRECT - Fused kernel with templates
template<typename Op1, typename Op2>
__global__ void kernel_fused(float* out, const float* in, int n, Op1 op1, Op2 op2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        val = op1(val);
        val = op2(val);
        out[idx] = val;
    }
}

// Single kernel launch
auto add_one = [](float x) __device__ { return x + 1.0f; };
auto mul_two = [](float x) __device__ { return x * 2.0f; };
kernel_fused<<<grid, block>>>(d_out, d_in, n, add_one, mul_two);

// ✅ EVEN BETTER - Generic pipeline with variadic templates
template<typename T, typename... Ops>
__device__ T apply_pipeline(T val, Ops... ops) {
    return (ops(val), ...);  // C++17 fold expression
}

template<typename... Ops>
__global__ void kernel_pipeline(float* out, const float* in, int n, Ops... ops) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = apply_pipeline(in[idx], ops...);
    }
}

// Usage: unlimited fusion
kernel_pipeline<<<grid, block>>>(d_out, d_in, n, add_one, mul_two, sqrt_op, exp_op);
```

### B. Minimize Branching

**Avoid divergent branches within warps:**

```cpp
// ❌ WRONG - Divergent branching
__global__ void divergent_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (in[idx] > 0.5f) {  // ← Causes warp divergence
            out[idx] = in[idx] * 2.0f;
        } else {
            out[idx] = in[idx] * 0.5f;
        }
    }
}

// ✅ CORRECT - Predicated execution (no branch)
__global__ void predicated_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        float mask = (val > 0.5f) ? 1.0f : 0.0f;
        out[idx] = mask * (val * 2.0f) + (1.0f - mask) * (val * 0.5f);
    }
}

// ✅ EVEN BETTER - Use fminf/fmaxf for conditional logic
__global__ void branchless_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        // Branchless: result = (val > 0.5) ? val * 2 : val * 0.5
        float high = val * 2.0f;
        float low = val * 0.5f;
        float cond = fminf(1.0f, fmaxf(0.0f, (val - 0.5f) * 1e6f));
        out[idx] = cond * high + (1.0f - cond) * low;
    }
}
```

**When branching is unavoidable, organize by warp:**

```cpp
// ✅ ACCEPTABLE - Branching at warp granularity
__global__ void warp_coherent_kernel(float* out, const float* in, 
                                     const int* types, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / 32;
    int lane_id = idx % 32;
    
    // Assume input is sorted/grouped by type for each warp
    // All threads in a warp take the same branch
    if (idx < n) {
        int type = types[warp_id];  // Same for entire warp
        if (type == 0) {
            out[idx] = in[idx] * 2.0f;
        } else {
            out[idx] = in[idx] * 0.5f;
        }
    }
}
```

### C. Optimize Occupancy

**Tune block size and resource usage for maximum occupancy:**

```cpp
// ✅ CORRECT - Occupancy-aware kernel design
#include <cuda_runtime.h>

// Query optimal block size
int min_grid_size, block_size;
cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, my_kernel, 0, 0);

int grid_size = (n + block_size - 1) / block_size;
my_kernel<<<grid_size, block_size>>>(args);

// Manual optimization: reduce register usage
template<int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE, 4)  // 4 blocks per SM
optimized_kernel(float* out, const float* in, int n) {
    // Kernel code
    // __launch_bounds__ hints compiler to limit register usage
}

// Check occupancy with nvcc
// nvcc --ptxas-options=-v kernel.cu
// Look for: "registers=32, smem=1024, occupancy=100%"
```

**Profiling occupancy:**

```bash
# Check achieved occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./app

# Target occupancy:
# - Compute-bound: >50% is good, >75% is excellent
# - Memory-bound: >25% is often sufficient
```

### D. CUDA Streams for Concurrency

**Use streams to overlap compute and memory transfers:**

```cpp
// ✅ CORRECT - Asynchronous execution with streams
const int num_streams = 4;
cudaStream_t streams[num_streams];
for (int i = 0; i < num_streams; i++) {
    cudaStreamCreate(&streams[i]);
}

// Allocate pinned host memory for async transfers
float *h_data;
cudaMallocHost(&h_data, size);

// Process data in chunks with overlapped transfers and compute
for (int i = 0; i < num_streams; i++) {
    int offset = i * chunk_size;
    
    // Async H2D transfer
    cudaMemcpyAsync(d_data + offset, h_data + offset, chunk_bytes,
                    cudaMemcpyHostToDevice, streams[i]);
    
    // Kernel execution (overlaps with transfer of next chunk)
    kernel<<<grid, block, 0, streams[i]>>>(d_data + offset, chunk_size);
    
    // Async D2H transfer
    cudaMemcpyAsync(h_result + offset, d_data + offset, chunk_bytes,
                    cudaMemcpyDeviceToHost, streams[i]);
}

// Wait for all streams
for (int i = 0; i < num_streams; i++) {
    cudaStreamSynchronize(streams[i]);
}

// Cleanup
for (int i = 0; i < num_streams; i++) {
    cudaStreamDestroy(streams[i]);
}
cudaFreeHost(h_data);
```

### E. CUDA Graphs for Non-Trivial Pipelines

**Use CUDA Graphs to minimize kernel launch overhead:**

```cpp
// ✅ CORRECT - CUDA Graphs for repeated execution
#include <cuda_runtime.h>

// Define the operation sequence
cudaGraph_t graph;
cudaGraphExec_t graph_exec;

// 1. Capture graph
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// Execute operations to capture
kernel1<<<grid, block, 0, stream>>>(d_data1);
kernel2<<<grid, block, 0, stream>>>(d_data2);
cudaMemcpyAsync(d_data3, d_data2, size, cudaMemcpyDeviceToDevice, stream);
kernel3<<<grid, block, 0, stream>>>(d_data3);

// End capture
cudaStreamEndCapture(stream, &graph);

// 2. Instantiate executable graph
cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);

// 3. Execute graph multiple times (very low overhead)
for (int i = 0; i < iterations; i++) {
    cudaGraphLaunch(graph_exec, stream);
    cudaStreamSynchronize(stream);
}

// Cleanup
cudaGraphExecDestroy(graph_exec);
cudaGraphDestroy(graph);

// Performance benefit:
// - Reduces kernel launch overhead from ~5-10 μs to <1 μs
// - Better for pipelines with many small kernels
// - Enables whole-pipeline optimizations by the CUDA driver
```

**When to use CUDA Graphs:**
- Pipeline executed multiple times with same structure
- Many small kernel launches (< 10 μs execution time)
- Need predictable low-latency execution
- Profiling shows kernel launch overhead is significant

### F. Avoid Over-Modularization

**Minimize abstraction overhead when performance matters:**

```cpp
// ❌ WRONG - Over-modularized (function call overhead)
__device__ float add_one(float x) { return x + 1.0f; }
__device__ float multiply(float x, float y) { return x * y; }
__device__ float complex_function(float x) {
    return multiply(add_one(x), 2.0f);  // Function calls
}

__global__ void kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex_function(in[idx]);
    }
}

// ✅ CORRECT - Inlined for performance
__forceinline__ __device__ float complex_function(float x) {
    return (x + 1.0f) * 2.0f;  // Inlined, no call overhead
}

// ✅ EVEN BETTER - Use templates and let compiler optimize
template<typename T>
__forceinline__ __device__ T fused_operation(T x) {
    return (x + T(1)) * T(2);
}

// Compiler will fully inline and optimize
__global__ void optimized_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fused_operation(in[idx]);
    }
}
```

**Balance readability and performance:**
- Use `__forceinline__` for hot path functions
- Template metaprogramming for compile-time decisions
- Profile before and after modularization changes
- Document performance-critical sections

---

## 6. Profiling & Optimization Workflow (MANDATORY)

### A. Profile Before Optimizing

**ALWAYS profile with ncu to identify bottlenecks:**

```bash
# Full profiling report
ncu --set full -o profile_report ./app

# Specific metrics for memory-bound analysis
ncu --metrics dram__bytes.sum.per_second,dram__bytes_read.sum,dram__bytes_write.sum ./app

# Specific metrics for compute-bound analysis
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_active,smsp__inst_executed.avg.per_cycle ./app

# Check occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./app

# Check branch divergence
ncu --metrics smsp__sass_branch_targets_threads_divergent.sum ./app

# Check shared memory bank conflicts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu.sum ./app

# Memory coalescing efficiency
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./app
# Efficiency = (requested / sectors) * 100%
# Goal: >80% efficiency
```

### B. Optimization Decision Tree

```
1. Profile kernel with ncu
   ↓
2. Is it memory-bound?
   - Check: dram__throughput.avg.pct_of_peak_sustained_active > 60%
   - If YES:
     → Optimize memory access patterns (coalescing)
     → Use shared memory to reduce global memory accesses
     → Consider texture cache for non-coalesced reads
     → Reduce memory bandwidth (compression, lower precision)
   ↓
3. Is it compute-bound?
   - Check: sm__throughput.avg.pct_of_peak_sustained_active > 60%
   - If YES:
     → Increase arithmetic intensity (more ops per byte)
     → Use tensor cores (wmma API or cuBLAS)
     → Optimize instruction mix
     → Reduce register pressure to increase occupancy
   ↓
4. Is occupancy low?
   - Check: sm__warps_active < 50%
   - If YES:
     → Reduce register usage (__launch_bounds__)
     → Reduce shared memory usage
     → Adjust block size
     → Check for resource limitations (registers/shared mem per block)
   ↓
5. Is there branch divergence?
   - Check: smsp__sass_branch_targets_threads_divergent > 0
   - If YES:
     → Refactor to predicated execution
     → Reorganize data for warp-coherent branching
     → Use __ballot_sync or __any_sync for warp-wide decisions
   ↓
6. Is kernel launch overhead high?
   - Check: Kernel execution time < 50 μs
   - If YES:
     → Use CUDA Graphs
     → Fuse multiple kernels
     → Increase work per kernel
   ↓
7. Re-profile and compare
   → Document improvements
   → Ensure accuracy still passes
```

### C. Profiling Example Workflow

```bash
# 1. Baseline profile
ncu --set full -o baseline ./app

# 2. Identify bottleneck
ncu --import baseline.ncu-rep --page details
# Output: Memory-bound, 85% DRAM throughput, 45% occupancy

# 3. Optimize memory access (use shared memory)
# ... modify code ...

# 4. Re-profile
ncu --set full -o optimized ./app

# 5. Compare
ncu --import baseline.ncu-rep,optimized.ncu-rep --page diff
# Output: DRAM throughput reduced to 60%, occupancy increased to 75%
# Result: 2.5x speedup

# 6. Document
echo "Optimization: Added shared memory tiling" >> PERFORMANCE.md
echo "Speedup: 2.5x (3.2ms -> 1.3ms)" >> PERFORMANCE.md
echo "Bottleneck changed: Memory-bound -> Compute-bound" >> PERFORMANCE.md
```

### D. Performance Documentation

**ALWAYS document performance characteristics:**

```cpp
/**
 * Matrix Multiplication Kernel (Shared Memory Tiled)
 * 
 * Performance Characteristics:
 * - Compute-bound on A100 (sm_80)
 * - Achieves 85% of peak GEMM throughput for N > 2048
 * - Occupancy: 75% (limited by shared memory)
 * - Memory efficiency: 92% coalesced access
 * 
 * Profiling Results (A100, N=4096):
 * - Kernel time: 1.23 ms
 * - Throughput: 21.5 TFLOPS (85% of peak 25.6 TFLOPS)
 * - DRAM bandwidth: 850 GB/s (55% of peak 1555 GB/s)
 * 
 * Tuning Parameters:
 * - TILE_SIZE=32: Best for sm_80 (A100)
 * - TILE_SIZE=16: Best for sm_75 (T4, RTX 2080)
 * 
 * Tested on: A100, V100, RTX 4090
 * Accuracy: max_error < 1e-5 vs CPU double precision
 */
template<int TILE_SIZE>
__global__ void matmul_tiled(float* C, const float* A, const float* B, int N);
```

---

## 7. Testing Requirements (MANDATORY)

### A. Unit Test Structure

**EVERY kernel MUST have corresponding unit tests:**

```cpp
// test_kernels.cu
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        ASSERT_EQ(err, cudaSuccess) << "CUDA Error: " << cudaGetErrorString(err); \
    } while(0)

// Test fixture for CUDA kernels
class VectorAddTest : public ::testing::Test {
protected:
    void SetUp() override {
        n = 1024 * 1024;
        size = n * sizeof(float);
        
        // Allocate host memory
        h_a = new float[n];
        h_b = new float[n];
        h_c_gpu = new float[n];
        h_c_cpu = new float[n];
        
        // Initialize with random data
        for (int i = 0; i < n; i++) {
            h_a[i] = static_cast<float>(rand()) / RAND_MAX;
            h_b[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_a, size));
        CUDA_CHECK(cudaMalloc(&d_b, size));
        CUDA_CHECK(cudaMalloc(&d_c, size));
    }
    
    void TearDown() override {
        delete[] h_a;
        delete[] h_b;
        delete[] h_c_gpu;
        delete[] h_c_cpu;
        
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }
    
    void compare_results(float tolerance = 1e-5f) {
        float max_error = 0.0f;
        int error_count = 0;
        const int max_errors_to_print = 10;
        
        for (int i = 0; i < n; i++) {
            float error = fabs(h_c_gpu[i] - h_c_cpu[i]);
            if (error > tolerance) {
                if (error_count < max_errors_to_print) {
                    std::cerr << "Mismatch at index " << i 
                              << ": GPU=" << h_c_gpu[i] 
                              << ", CPU=" << h_c_cpu[i] 
                              << ", error=" << error << std::endl;
                }
                error_count++;
            }
            max_error = fmaxf(max_error, error);
        }
        
        EXPECT_EQ(error_count, 0) << "Found " << error_count << " errors";
        EXPECT_LT(max_error, tolerance) << "Max error: " << max_error;
    }
    
    int n;
    size_t size;
    float *h_a, *h_b, *h_c_gpu, *h_c_cpu;
    float *d_a, *d_b, *d_c;
};

// CPU reference implementation
void vector_add_cpu(float* c, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// GPU kernel (declared elsewhere)
__global__ void vector_add_gpu(float* c, const float* a, const float* b, int n);

// Test case: Basic functionality
TEST_F(VectorAddTest, BasicFunctionality) {
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    vector_add_gpu<<<grid_size, block_size>>>(d_c, d_a, d_b, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy back
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost));
    
    // CPU reference
    vector_add_cpu(h_c_cpu, h_a, h_b, n);
    
    // Compare
    compare_results(1e-5f);
}

// Test case: Edge cases
TEST_F(VectorAddTest, EdgeCases) {
    // Test with size 1
    int n_small = 1;
    vector_add_gpu<<<1, 1>>>(d_c, d_a, d_b, n_small);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Test with non-power-of-2 size
    int n_odd = 12345;
    int grid = (n_odd + 255) / 256;
    vector_add_gpu<<<grid, 256>>>(d_c, d_a, d_b, n_odd);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Test case: Special values
TEST_F(VectorAddTest, SpecialValues) {
    // Test with zeros
    std::fill(h_a, h_a + n, 0.0f);
    std::fill(h_b, h_b + n, 0.0f);
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    vector_add_gpu<<<grid_size, block_size>>>(d_c, d_a, d_b, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost));
    
    vector_add_cpu(h_c_cpu, h_a, h_b, n);
    compare_results(1e-5f);
}

// Performance test
TEST_F(VectorAddTest, Performance) {
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    
    // Warm-up
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    vector_add_gpu<<<grid_size, block_size>>>(d_c, d_a, d_b, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed run
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int iterations = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        vector_add_gpu<<<grid_size, block_size>>>(d_c, d_a, d_b, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time = milliseconds / iterations;
    
    // Calculate bandwidth
    float bandwidth = (3.0f * size) / (avg_time * 1e-3) / 1e9;  // GB/s
    
    std::cout << "Average kernel time: " << avg_time << " ms" << std::endl;
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;
    
    // Sanity check: bandwidth should be reasonable
    EXPECT_GT(bandwidth, 100.0f) << "Bandwidth too low, check implementation";
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
```

### B. CMakeLists.txt for CUDA Testing

```cmake
cmake_minimum_required(VERSION 3.18)
project(CUDAKernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)  # A100, RTX 3090, RTX 4090, H100

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find GTest
find_package(GTest REQUIRED)
enable_testing()

# Kernel library
add_library(kernels STATIC
    kernels.cu
)
target_link_libraries(kernels
    CUDA::cudart
    CUDA::cufft
    CUDA::cublas
)

# Optional: cuFFTdx (requires separate download)
# target_include_directories(kernels PRIVATE ${CUFFTDX_INCLUDE_DIR})

# Test executable
add_executable(test_kernels
    test_kernels.cu
)
target_link_libraries(test_kernels
    kernels
    GTest::GTest
    GTest::Main
    CUDA::cudart
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(test_kernels)

# Custom target to run tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_kernels
)

# Release build should fail if tests disabled
if(CMAKE_BUILD_TYPE MATCHES Release)
    if(NOT BUILD_TESTING)
        message(FATAL_ERROR "Release builds must have testing enabled")
    endif()
endif()
```

### C. Running Tests

```bash
# Build and run tests
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 ..
cmake --build . --config Release
ctest --output-on-failure

# Run with cuda-memcheck
compute-sanitizer ./test_kernels

# Run specific test
./test_kernels --gtest_filter=VectorAddTest.BasicFunctionality

# Run with profiling
ncu --set full -o profile ./test_kernels
```

---

## 8. Project Structure (MANDATORY)

### A. Recommended Directory Layout

```
project/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── kernels.cuh         # Kernel declarations
│   ├── cuda_utils.cuh      # Error checking, utilities
│   └── types.h             # Common types
├── src/
│   ├── kernels.cu          # Kernel implementations
│   ├── cuda_utils.cu       # Utility implementations
│   └── main.cu             # Application entry point
├── tests/
│   ├── test_kernels.cu     # GTest unit tests
│   ├── test_accuracy.cu    # GPU vs CPU validation
│   └── test_performance.cu # Performance benchmarks
├── benchmarks/
│   └── profile_kernels.cu  # Profiling harness
└── docs/
    ├── PERFORMANCE.md      # Profiling results, optimization notes
    └── API.md              # Kernel API documentation
```

### B. cuda_utils.cuh (Required)

```cpp
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while(0)

// Check last CUDA error (for kernel launches)
#define CUDA_CHECK_LAST()                                                      \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Kernel Error at %s:%d - %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while(0)

// CUDA device query
inline void print_device_info() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %.2f GB\n", 
           prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("Shared Memory per Block: %zu KB\n", 
           prop.sharedMemPerBlock / 1024);
    printf("Registers per Block: %d\n", prop.regsPerBlock);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Grid Size: (%d, %d, %d)\n", 
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
}

// Timing utilities
class CUDATimer {
public:
    CUDATimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    
    ~CUDATimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
    }
    
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return milliseconds;
    }
    
private:
    cudaEvent_t start_, stop_;
};
```

---

## 9. Deployment Checklist

### Agent-Generated CUDA Code Verification (MANDATORY)

**If CUDA code was generated/modified by an agent, verify BEFORE delivery:**

#### Compilation & Execution
- [ ] Code compiles without errors for target GPU architecture
- [ ] No compiler warnings with `-Wall -Wextra`
- [ ] CUDA runtime version compatible with code
- [ ] All CUDA API calls have error checking (`CUDA_CHECK`)
- [ ] Kernel launches followed by `cudaGetLastError()` check
- [ ] Code executes without CUDA errors or crashes
- [ ] No segmentation faults or memory access violations

#### Accuracy & Testing
- [ ] CPU reference implementation provided for every kernel
- [ ] GPU results compared against CPU reference
- [ ] **If porting existing code: Reference implementation used as gold standard**
- [ ] **If porting: Comprehensive test suite designed using reference code**
- [ ] **If porting: All tests validate CUDA output against reference implementation**
- [ ] **If porting: Tests pass for all input sizes and edge cases from reference**
- [ ] Accuracy within acceptable tolerance (e.g., 1e-5 for float)
- [ ] Unit tests created using GTest
- [ ] All tests pass: `ctest --output-on-failure`
- [ ] Edge cases tested (size=1, non-power-of-2, special values)
- [ ] Special value handling tested (NaN, Inf, zeros)
- [ ] No memory leaks: `compute-sanitizer --tool memcheck` passes
- [ ] Any deviations from reference documented and user-approved

#### Performance & Optimization
- [ ] Profiled with ncu to identify bottleneck (memory vs compute)
- [ ] Occupancy measured and documented (target: >50% for compute-bound)
- [ ] Memory access patterns are coalesced (>80% efficiency)
- [ ] Branch divergence minimal (<5% of instructions)
- [ ] No shared memory bank conflicts (or documented if unavoidable)
- [ ] Performance characteristics documented in comments
- [ ] Bandwidth or throughput measured and reasonable
- [ ] Optimization claims backed by profiling data

#### Memory Management
- [ ] All `cudaMalloc` calls have corresponding `cudaFree`
- [ ] Host-device transfers minimized
- [ ] Asynchronous transfers used where possible (with streams)
- [ ] Memory copy directions correct (H2D, D2H, D2D)
- [ ] No unnecessary intermediate buffers

#### Code Quality
- [ ] Clean, minimalistic code (no over-modularization)
- [ ] Kernels fused where appropriate (template-based fusion)
- [ ] CUDA streams used for concurrency
- [ ] CUDA Graphs used for non-trivial pipelines
- [ ] Texture cache used for non-coalesced read-only data (if applicable)
- [ ] Comments explain non-obvious optimizations
- [ ] GPU architecture-specific code documented

#### Library Usage
- [ ] cuFFTdx preferred over cuFFT (unless very large FFTs)
- [ ] cuBLAS used for matrix operations (not naive custom kernels)
- [ ] CUDA-X libraries used where applicable
- [ ] Library versions documented

#### Agent Workflow Completed
- [ ] Agent compiled code successfully
- [ ] Agent ran tests and verified accuracy
- [ ] **If porting: Agent validated against reference implementation**
- [ ] **If porting: Agent designed and executed comprehensive reference-based tests**
- [ ] **If porting: All CUDA changes tested against reference before delivery**
- [ ] Agent profiled code (if optimization claims made)
- [ ] Agent documented any fixes made during verification
- [ ] Agent documented any deviations from reference (with user approval)

### General Best Practices
- [ ] Project follows recommended directory structure
- [ ] CMakeLists.txt configured for multiple GPU architectures
- [ ] README documents build instructions and dependencies
- [ ] PERFORMANCE.md documents profiling results (if applicable)
- [ ] Code follows C++17 standards
- [ ] No deprecated CUDA APIs used

---

## 10. Why This Configuration Works

**Performance-First Philosophy**: Modern CUDA development prioritizes:
- **Verified Builds**: Agent verification ensures code compiles and runs before delivery, eliminating broken code and reducing debugging time.
- **Accuracy Testing**: GPU vs CPU validation catches numerical errors, race conditions, and memory corruption early.
- **Reference Code Porting**: When porting existing code to CUDA, using the original implementation as the gold standard ensures correctness is never compromised during optimization. Comprehensive reference-based tests catch subtle bugs (floating-point differences, edge case handling, special value behavior) that might otherwise go unnoticed.
- **Profiler-Driven Optimization**: Using ncu to identify bottlenecks prevents premature optimization and wasted effort. 5-10x speedups are common when optimizing the right thing.
- **Kernel Fusion**: Template-based fusion eliminates kernel launch overhead (5-10 μs per launch). For pipelines with 10 kernels, this saves 50-100 μs = 20-50% of total time for small kernels.
- **Memory Optimization**: Coalesced access patterns achieve 80-90% of peak bandwidth vs 10-20% for naive implementations—a 4-8x speedup.
- **Occupancy Tuning**: Higher occupancy hides memory latency. 75% occupancy vs 25% can yield 2-3x performance for memory-bound kernels.
- **CUDA Graphs**: Reduce kernel launch overhead to <1 μs, enabling efficient execution of complex pipelines. Critical for latency-sensitive applications.
- **Library Preference**: cuFFTdx, cuBLAS, and CUDA-X libraries are heavily optimized by NVIDIA engineers and leverage hardware features (tensor cores, async copy) that are difficult to replicate manually.
- **Minimal Branching**: Eliminating warp divergence can improve performance by 2-4x for branching-heavy code.
- **Asynchronous Execution**: CUDA streams enable overlapping compute and memory transfers, hiding transfer latency entirely (10-30% overall speedup).
- **Clean Code**: Minimalistic, non-over-modularized code compiles to fewer instructions and is easier to optimize by the compiler. Balance readability with performance.

**Modern CUDA**: This guide emphasizes CUDA 12.x features, C++17 templates, and contemporary best practices. Legacy approaches (synchronous execution, no profiling, custom implementations of library functions) are explicitly discouraged.

**Hardware Awareness**: Different GPU architectures (sm_80 A100, sm_89 RTX 4090, sm_90 H100) have different characteristics. Always compile for your target architecture and profile on actual hardware.

---

## 11. Quick Reference

### Compilation Commands

```bash
# Basic compilation
nvcc -std=c++17 -O3 -arch=sm_80 -o app kernel.cu

# Multiple architectures (for portability)
nvcc -std=c++17 -O3 -gencode arch=compute_80,code=sm_80 \
                     -gencode arch=compute_89,code=sm_89 \
                     -gencode arch=compute_90,code=sm_90 \
     -o app kernel.cu

# With cuBLAS and cuFFT
nvcc -std=c++17 -O3 -arch=sm_80 -o app kernel.cu \
     -lcublas -lcufft

# Enable all warnings
nvcc -std=c++17 -O3 -arch=sm_80 -Xcompiler -Wall,-Wextra -o app kernel.cu

# View PTX/SASS register usage
nvcc --ptxas-options=-v -arch=sm_80 kernel.cu
```

### Profiling Commands

```bash
# Full profile
ncu --set full -o profile ./app

# Memory-bound analysis
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_active,\
              l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
              l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
    ./app

# Compute-bound analysis
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_active,\
              smsp__inst_executed.avg.per_cycle \
    ./app

# Occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./app

# Branch divergence
ncu --metrics smsp__sass_branch_targets_threads_divergent.sum ./app

# System-wide timeline
nsys profile -o timeline --stats=true ./app
```

### Testing Commands

```bash
# Run tests
ctest --output-on-failure

# Memory check
compute-sanitizer --tool memcheck ./test_app

# Race condition detection
compute-sanitizer --tool racecheck ./test_app

# Run specific test
./test_app --gtest_filter=MyKernelTest.Accuracy
```

### Common Patterns

```cpp
// Error checking wrapper
#define CUDA_CHECK(call) /* ... as defined above ... */

// Kernel launch with error check
my_kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());

// Optimal block size
int block_size, grid_size;
cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, my_kernel, 0, 0);

// Asynchronous execution
cudaStream_t stream;
cudaStreamCreate(&stream);
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
my_kernel<<<grid, block, 0, stream>>>(d_data);
cudaMemcpyAsync(h_result, d_result, size, cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);
```

---

**End of CUDA Programming Guidelines**
