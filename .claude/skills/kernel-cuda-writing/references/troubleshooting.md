<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-ThirdParty-Unlicensed
-->

# Troubleshooting

## Compilation Errors

| Error | Solution |
|-------|----------|
| undefined symbol | Check `extern "C"` declarations match between `.cu` and `_binding.cpp` |
| no kernel image | Verify `TORCH_CUDA_ARCH_LIST` matches the target GPU |

## Correctness Failures

| Issue | Debug Steps |
|-------|-------------|
| Wrong output values | 1. Check kernel math. 2. Verify indexing. 3. Test with simple inputs. |
| NaN/Inf results | 1. Check division by zero. 2. Verify numerical stability. 3. Add bounds checking. |
| Mismatched shapes | 1. Print tensor shapes. 2. Check dimension calculations. 3. Verify reduction logic. |

## Performance Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Slower than baseline | No fusion | Combine kernels |
| Low SM efficiency | Poor occupancy | Tune block size |
| Low memory throughput | Uncoalesced access | Restructure memory pattern |
| High kernel count | Missing fusion | Implement compound operations |
