<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-ThirdParty-Unlicensed
-->

# Optimization Strategy and Parameter Tuning

## Optimization Priority

### Priority 1: Algorithmic (>50% impact)

- **Kernel fusion** -- Reduce memory traffic by combining operations.
- **Shared memory tiling** -- Improve data reuse across threads.
- **Memory coalescing** -- Ensure consecutive threads access consecutive addresses.

### Priority 2: Hardware Utilization (20-50% impact)

- **Vectorized loads** -- Use `float2`/`float4` for higher memory throughput.
- **Warp-level primitives** -- `__shfl_sync`, `__ballot_sync` for inter-thread communication.
- **Occupancy tuning** -- Balance block size, register usage, and shared memory.

### Priority 3: Fine-Tuning (<20% impact)

- **Instruction-level parallelism** -- Overlap independent operations.
- **Mixed precision** -- FP16/TF32 where accuracy permits.
- **Prefetching and double buffering** -- Overlap computation with memory transfers.

## Parameter Tuning (Last Resort)

Use only when within 1.2x of target and algorithmic options are exhausted. This approach requires no recompilation -- it selects among pre-compiled template configurations at runtime.

```python
# tune_kernel.py -- NO recompilation needed
import time, torch, cuda_extension

configs = [
    (0, "256_threads_16_tile"),
    (1, "128_threads_32_tile"),
    (2, "512_threads_8_tile"),
]

# Test input
x = torch.randn(batch_size, features).cuda()

# Benchmark each config
best_config, best_time = 0, float("inf")
for config_id, name in configs:
    # Warmup
    for _ in range(10):
        cuda_extension.my_kernel_forward(x, config=config_id)
    torch.cuda.synchronize()

    # Measure
    start = time.perf_counter()
    for _ in range(100):
        cuda_extension.my_kernel_forward(x, config=config_id)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"Config {name}: {elapsed:.4f}s")
    if elapsed < best_time:
        best_time, best_config = elapsed, config_id

print(f"Best: config {best_config} ({best_time:.4f}s)")
# Update model_new.py with best_config
```
