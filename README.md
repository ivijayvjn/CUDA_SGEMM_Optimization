# CUDA SGEMM Optimization

Progressive optimization of SGEMM (Single-Precision General Matrix Multiplication) in CUDA to study:

- GPU memory hierarchy
- Arithmetic intensity
- Global memory traffic
- Shared memory tiling
- CUDA thread mapping
- Data reuse in GPU kernels

Inspired by:
https://siboehm.com/articles/22/CUDA-MMM

---

## Implemented Optimizations

| Step | Focus |
|---|---|
| 01 | Naive SGEMM |
| 02 | Thread Mapping / Coalescing Study |
| 03 | Shared Memory Tiling |

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

| Kernel | Time (ms) | GFLOP/s | Estimated Memory Traffic | Arithmetic Intensity | Key Observation |
|---|---:|---:|---:|---:|---|
| Naive SGEMM | 1.178 | 1822.12 | 8.594 GB | 0.249 | High global memory traffic |
| Coalesced Thread Mapping | 1.180 | 1818.86 | 8.594 GB | 0.249 | Similar performance since baseline already used relatively coalesced-friendly thread mapping |
| Shared Memory Tiling | 0.910 | 2359.00 | 0.273 GB | 7.877 | Reused A/B tiles through shared memory to reduce global memory traffic |

---

## Key Learnings

- Correct CUDA kernels are not automatically high-performance.
- Naive SGEMM repeatedly fetches A/B values from global memory.
- Arithmetic intensity is strongly tied to data reuse.
- Shared memory tiling improves throughput by reducing repeated DRAM accesses.
- CUDA optimization is largely about improving memory access efficiency and reuse.
