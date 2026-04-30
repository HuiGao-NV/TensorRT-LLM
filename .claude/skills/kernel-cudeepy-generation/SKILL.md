<!-- Ported from https://gitlab-master.nvidia.com/wkong/perf-bot -->
---
name: kernel-cudeepy-generation
tags: [cudeepy]
description: >
  Generate optimized GPU kernels via CuDeepy CLI using CuTe DSL or LIR.
  Triggers: SOL% below 60%, operator needs custom kernel, user requests
  CuTe DSL or LIR lowering. Covers GEMM and attention on Blackwell+ (SM100+),
  element-wise and reduction on Ampere+ (SM80+). Includes two-stage validation,
  CUDA-event benchmarking, and workload integration via companion script.
compatibility: Requires cudeepy CLI
license: LicenseRef-NvidiaProprietary
metadata:
  author: NVIDIA Corporation
  documentation: https://gitlab-master.nvidia.com/wkong/perf-bot
---

# CuDeepy Kernel Generation

**Step 0 (Arena Search):** If `PERFBOT_ARENA_ENABLED` is set, the calling agent searches the Kernel Arena for existing exemplars before writing from scratch. Adapt any matching exemplar rather than reimplementing.

**Important:** CuDeePy-generated kernels must be wrapped in the fixed-name contract (`kernel_fn`, `reference_fn`, `get_inputs()`) before the Post-Gate to enable safety-net detection.

## Principles

### CuDeepy over Triton

Prefer CuDeepy when:
- Target is Blackwell+ (SM100+) for GEMM or attention
- Target is Ampere+ (SM80+) for element-wise or reduction
- Workload needs CuTe DSL's hardware-aware abstractions (TMA, TMEM, MMA)
- SOL% < 60% on profiled operators

Prefer Triton when:
- Portability across GPU vendors matters
- Operation is a simple element-wise fusion without GEMM
- No Blackwell-specific features needed

### DSL Selection

| Operation | DSL | Hardware |
|-----------|-----|----------|
| GEMM, attention | CuTe DSL | Blackwell+ (SM100+) |
| Element-wise, reduction | CuTe DSL | Ampere+ (SM80+) |
| Warp specialization, custom pipelines | LIR | Blackwell+ (SM100+) |

Default to **CuTe DSL**. Use LIR only for explicit buffer management, warp specialization, or non-standard pipeline configurations.

### Never Write Kernels Manually

CuDeepy has accumulated learnings about CuTe DSL and LIR patterns. If generation times out or fails, report the error -- do not attempt to hand-write the kernel.

Exception: pure element-wise operations can be written directly using element-wise patterns from the `kernel-cute-writing` skill.

## Workflow

### Step 1: Extract Operator

Read the workload source and locate the target operator. Extract:
- Operator code (class or function definition)
- Input/output shapes and dtypes
- Reference implementation location

### Step 2: Select DSL

Apply the DSL selection table above. Default to CuTe DSL unless the operation requires warp specialization or custom pipelines.

### Step 3: Generate Kernel

Create an output directory, write the operator code to `{output_dir}/operator.py`, then invoke cudeepy:

```bash
cudeepy generate --operator <name> --dsl cute --input {output_dir}/operator.py --output {output_dir}
```

Generated output files:
- `kernel.py` -- CuTe/LIR kernel implementation
- `test_harness.py` -- correctness and benchmark tests
- `README.md` -- performance results and tuning notes
- `agentcontext.md` -- generation metadata

Always use output directory paths for subsequent operations.

### Step 4: Validate (Two Stages)

**Stage 1 -- CuDeepy harness** (correctness + performance metrics):

```bash
python {output_dir}/test_harness.py
```

**Stage 2 -- Independent verification** using the **kernel-cute-writing** skill's `verify_kernel.py`:

```bash
python scripts/verify_kernel.py \
    --kernel {output_dir}/kernel.py \
    --reference-code 'def ref(x): return torch.nn.functional.gelu(x)' \
    --input-shapes '{"x": [B, M, K]}' \
    --input-dtypes '{"x": "bfloat16"}'
```

Both stages must pass before proceeding.

### Step 5: Benchmark

Write benchmark code using CUDA event timing patterns from the **perf-workload-profiling** skill.

**CRITICAL**: Pre-compile with `cute.compile()` before timing. Without it, the kernel recompiles every iteration and timing is inaccurate.

### Step 6: Integrate

Use the companion script to integrate the kernel into the workload:

```bash
python scripts/integrate_kernel.py \
    --workload path/to/model.py \
    --kernel {output_dir}/kernel.py \
    --operator <name> \
    --method import
```

Methods:
- `import` -- add import statement after existing imports
- `patch` -- add monkey-patching code at file beginning

The script creates a backup automatically (`<workload>.bak`).

### Step 7: Full Workload Validation

Delegate workload-level benchmarking to perf-profiling-specialist via the parent optimization-agent. If regression detected, restore backup:

```bash
cp path/to/model.py.bak path/to/model.py
```

## Companion Script: integrate_kernel.py

```
python scripts/integrate_kernel.py \
    --workload <path> --kernel <path> --operator <name> --method import|patch [--mock]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--workload` | Yes | Workload Python file to modify |
| `--kernel` | Yes | Generated kernel.py path |
| `--operator` | Yes | Operator name being replaced |
| `--method` | Yes | `import` or `patch` |
| `--mock` | No | Return mock JSON for testing |

Output JSON:

```json
{
  "success": true,
  "workload_path": "/abs/path/model.py",
  "changes": ["Added import: from ...kernel import optimized_FFN"],
  "method": "import",
  "backup_path": "/abs/path/model.py.bak"
}
```

### Final Step: Arena Upload

After verification (and benchmarking if performed), if `PERFBOT_ARENA_ENABLED` is set, the calling agent handles arena upload to evaluate whether to upload this kernel to the arena.

## References

- [CuTe DSL Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html) -- Official CuTe DSL API, programming model, and examples

## Reporting

Include in final report:
- Kernel path and DSL used (CuTe or LIR)
- Correctness: numerical error, pass/fail
- Performance: TFLOPS, speedup vs baseline, new SOL%
- Integration status: success/failure with changes made

## Error Handling

### Generation Timeout

Do NOT write the kernel manually. Report the timeout and suggest:
1. Increase `generation_timeout` in perfbot.yaml (default: 900s)
2. Simplify the operator (break into smaller parts)
3. Check cudeepy logs for issues

### Generation Failure

- Verify operator has a clear class/function definition
- Confirm input shapes are detectable
- Try a simpler operator first to validate the pipeline

### Validation Failure

- Match dtypes between reference and kernel
- Use tolerances appropriate for precision (1e-3 for BF16/FP16)
- Ensure input shapes include batch dimension (3D)

### Integration Failure

- Confirm backup exists before modification
- Verify import path is correct relative to workload
- Check for circular imports
