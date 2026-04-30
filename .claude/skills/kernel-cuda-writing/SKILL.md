<!-- Ported from https://gitlab-master.nvidia.com/wkong/perf-bot -->
---
name: kernel-cuda-writing
description: >-
  ONLY for writing raw CUDA C/C++ (.cu/.cpp) files compiled via
  torch.utils.cpp_extension into custom PyTorch C++ extensions. NEVER use
  for Triton kernels, CuDeepy kernels, TileIR kernels, or any other kernel
  DSL/framework — those have their own dedicated skills. Also NOT for CUDA
  Graphs, nsight profiling, torch.compile, memory analysis, or distributed
  training. Use when the user asks to write a CUDA C++ extension, create a
  .cu kernel, fuse PyTorch ops into a custom CUDA kernel, or build a custom
  CUDA extension for PyTorch.
tags: [cuda, optimization, kernels]
license: LicenseRef-ThirdParty-Unlicensed
metadata:
  author: Third-Party
  source: https://github.com/BytedTsinghua-SIA/CUDA-Agent/tree/86a10af3bd8b64f9e2f8e75c0f09fa67c8ce5d95
  documentation: https://github.com/BytedTsinghua-SIA/CUDA-Agent/tree/86a10af3bd8b64f9e2f8e75c0f09fa67c8ce5d95
---

Accelerate the given PyTorch model by creating a high-performance CUDA C++
extension.

**Step 0 (Arena Search):** If `PERFBOT_ARENA_ENABLED` is set, the calling agent searches the Kernel Arena for existing exemplars before writing from scratch. Adapt any matching exemplar rather than reimplementing.

## Arena-Compatible Kernel Format

For standalone kernels (arena-eligible), produce a `kernel.py` wrapper alongside `.cu`/`.cpp` files:

```python
import torch
from torch.utils.cpp_extension import load
from pathlib import Path

_dir = Path(__file__).parent
_ext = load(
    name="arena_<operation>_<variant>",  # unique name to avoid cache collision
    sources=[str(_dir / "kernel.cu"), str(_dir / "kernel_binding.cpp")],
    verbose=False,
)

def kernel_fn(x):
    return _ext.my_cuda_op(x)

def reference_fn(x):
    return torch.nn.functional.silu(x)  # PyTorch reference

def get_inputs():
    return [torch.randn(1024, 1024, device="cuda")]
```

The `name` parameter in `load()` MUST be unique per kernel (e.g., `arena_unary_elementwise_cuda`) to avoid `torch.utils.cpp_extension` cache collisions.

This format enables verification via the **kernel-cute-writing** skill's `verify_kernel.py` and benchmarking via `benchmark_kernel.py` — same as Triton/CuTe kernels.

For **model-level optimization** (not standalone kernels), use the traditional `model_new.py` pattern below. Model-level work is NOT arena-eligible.

### Final Step: Arena Upload

After verification (and benchmarking if performed), if `PERFBOT_ARENA_ENABLED` is set, the calling agent handles arena upload. Include companion files (`.cu`, `.cpp`) in the upload payload.

## 1. Critical Restrictions

### Strictly Forbidden

- NO torch operators in C++ -- never use `torch::*` or `torch::nn::functional::*` in binding.cpp or .cu files.
- NO torch operations in model_new.py -- only tensor creation and custom ops allowed.
- NO third-party libraries except cuBLAS (GEMM only) and cuDNN (Conv only).
- NO modifications to `utils/`, `binding.cpp`, or `binding_registry.h`.

### Allowed Only

- **C++**: Raw CUDA kernels (custom ops), cuBLAS (GEMM), cuDNN (mandatory for Conv/ConvTranspose).
- **Python**: `torch.tensor` creation, custom extension ops, tensor properties (`.shape`, `.device`).
- **Memory**: `torch::empty_like` for allocation only.
- **Focus**: Implement kernels in `kernels/` directory only.

## 2. Workspace Structure

```
.
├── binding_registry.h    # Do NOT modify - registration system
├── binding.cpp           # Do NOT modify - main module binding
├── kernels/              # YOUR WORK: implement all kernels here
├── utils/                # Do NOT modify - compilation, verification, profiling
├── model.py              # Do NOT modify - original PyTorch model
└── model_new.py          # YOUR WORK: optimized model using custom ops
```

### File Types

- **`.cu` files** -- CUDA kernels with `__global__` functions (custom implementations).
- **`.cpp` files** -- cuDNN/cuBLAS API calls (no custom kernels).
- **`_binding.cpp` files** -- PyTorch tensor handling and Python bindings.

## 3. Workflow

### Step 1: Implement Kernels

Create paired files in `kernels/`:

1. **`kernels/<name>.cu`** -- Pure CUDA implementation with a C-interface launcher. No PyTorch dependencies. Use template parameters for runtime-tunable block/tile sizes.
2. **`kernels/<name>_binding.cpp`** -- PyTorch wrapper that validates inputs, allocates output with `torch::empty_like`, obtains the CUDA stream via `c10::cuda::getCurrentCUDAStream().stream()`, calls the launcher, and registers the binding with `REGISTER_BINDING`.
3. **`model_new.py`** -- `ModelNew` class matching the original `Model` constructor signature. Use only `cuda_extension.<op>_forward(...)` calls in `forward()`.

See `examples/` for a complete axpby reference implementation.

### Step 2: Compile and Test

```bash
# Set TORCH_CUDA_ARCH_LIST to match the target GPU (e.g., 8.0 for A100, 9.0 for H100)
TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh
sudo python3 -m utils.verification
sudo python3 -m utils.profiling
```

### Step 3: Optimize (If Needed)

Refer to `references/tuning.md` for the three-tier optimization priority and parameter tuning workflow.

### Step 4: Iterate

#### Correctness Failures

Iterate until correctness passes -- no exceptions.

1. Debug the specific failing kernel.
2. Common issues: boundary conditions (`tid < size`), synchronization (`__syncthreads` placement), data types, memory alignment.
3. Fix in `kernels/*.cu` and `*_binding.cpp` only.
4. Recompile and test.

#### Performance Optimization

Goal: produce a correct, competitive CUDA extension. Stop when one of
these conditions is met (check after each compile-verify-benchmark cycle):

1. **Within 5% of torch.compile** — the kernel is already well-optimized;
   further gains require micro-tuning that rarely pays off.
2. **Two consecutive iterations with <5% improvement** — diminishing
   returns; declare the current version final.
3. **Three total optimization iterations** — hard cap to avoid unbounded
   tuning loops on bandwidth-bound kernels.

For each iteration:

1. Document expectation -- e.g., "Fusion will eliminate 3 kernels, expect ~20% speedup."
2. Apply optimization aggressively -- do not revert to slow versions.
3. Debug if correctness fails -- fix the optimized version.
4. Measure and analyze -- understand why performance changed.

### Step 5: Final Cleanup (Mandatory)

Before declaring completion, clean the `kernels/` directory to contain only the final optimized version. Remove all intermediate attempts (`*_v[0-9].cu`, `*_old.cu`, `*_test.cu`, `*.bak`).

## 4. Success Criteria

- **Correctness**: verification must pass (atol=1e-2, rtol=1e-2).
- **Performance**: competitive with torch.compile (within 5% is good; faster is great).
- **Clean code**: `kernels/` contains only the final optimized version.

## 5. Key Reminders

1. Keep `.cu` and `_binding.cpp` files separate -- faster compilation.
2. Pass config parameters through bindings -- enable runtime tuning without recompilation.
3. Document performance expectations before each optimization attempt.

## Additional Resources

Only read reference materials or scripts when needed for the current step.
Do not proactively read files you are not about to use.

- **`references/optimization-checklist.md`** -- Essential, performance, and advanced optimization checklists.
- **`references/troubleshooting.md`** -- Common compilation, correctness, and performance issue tables.
- **`references/tuning.md`** -- Three-tier optimization priority and parameter tuning workflow.
- **`examples/`** -- Complete axpby reference implementation (kernel, binding, model, model_new).
- **`assets/`** -- Template infrastructure files (`binding.cpp`, `binding_registry.h`) and tool scripts (`utils/`).
