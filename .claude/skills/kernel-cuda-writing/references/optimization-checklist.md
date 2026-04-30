<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-ThirdParty-Unlicensed
-->

# Optimization Checklist

## Essential Optimizations (Apply First)

- [ ] **Memory Coalescing** -- Consecutive threads access consecutive addresses.
- [ ] **Kernel Fusion** -- Combine operations to reduce memory traffic.
- [ ] **Shared Memory** -- Cache frequently accessed data.
- [ ] **Grid-Stride Loops** -- Handle data larger than grid size.
- [ ] **Boundary Checks** -- Validate all array accesses (`tid < size`).

## Performance Optimizations (Apply as Needed)

- [ ] **Vectorized Memory** -- Use `float2`/`float4` for higher throughput.
- [ ] **Warp Primitives** -- `__shfl_sync` for inter-thread communication.
- [ ] **Occupancy Tuning** -- Balance block size and resource usage.
- [ ] **Bank Conflict Avoidance** -- Pad shared memory arrays.
- [ ] **Loop Unrolling** -- Increase instruction-level parallelism.

## Advanced Optimizations (Final Tuning)

- [ ] **Tensor Cores** -- Use WMMA/MMA for eligible GEMM operations.
- [ ] **Mixed Precision** -- FP16/TF32 where appropriate.
- [ ] **Persistent Kernels** -- Keep data in registers across iterations.
- [ ] **CUDA Graphs** -- Reduce launch overhead.
- [ ] **Double Buffering** -- Overlap computation with memory transfers.

## Correctness Checklist (Always Verify)

- [ ] **Thread Bounds** -- Check `tid < N` before array access.
- [ ] **Synchronization** -- `__syncthreads()` before shared memory reuse.
- [ ] **Data Types** -- Ensure correct types and conversions.
- [ ] **Memory Safety** -- No out-of-bounds access.
- [ ] **Numerical Stability** -- Handle NaN/Inf, use stable algorithms.
