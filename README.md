# CUDA SGEMM Optimization

Progressive optimization of SGEMM (Single-Precision General Matrix Multiplication) in CUDA to study:

- GPU memory hierarchy
- Arithmetic intensity
- Global memory traffic
- Shared memory tiling
- Warp-level execution
- CUDA performance optimization

Inspired by:
https://siboehm.com/articles/22/CUDA-MMM

---

## Optimization Roadmap

| Step | Focus |
|---|---|
| 01 | Naive SGEMM |
| 02 | Global Memory Coalescing |
| 03 | Shared Memory Tiling |
| 04 | 1D Block Tiling |
| 05 | 2D Block Tiling |
| 06 | Vectorized Memory Access |
| 07 | Warp Tiling |

---

## Benchmark Environment

| Component | Details |
|---|---|
| GPU | NVIDIA L4 |
| NVIDIA Driver Version | 580.126.20 |
| CUDA Runtime Version | 13.0 |
| NVCC Version | CUDA 12.6 (V12.6.77) |
| Operating Environment | Lightning AI GPU Instance |

---

## Benchmark Results

| Kernel | Time (ms) | GFLOP/s | Arithmetic Intensity | Key Bottleneck |
|---|---|---|---|---|
| Naive SGEMM | 1.17 | 1822.12 | 0.249 | High global memory traffic |

---

## Key Learnings

- Correct CUDA kernels are not automatically high-performance.
- Naive SGEMM suffers from repeated global memory accesses.
- Low arithmetic intensity leads to memory-bound execution.
- Optimization focuses on improving data reuse and reducing DRAM traffic.
