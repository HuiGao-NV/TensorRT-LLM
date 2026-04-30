---
name: modeling-module-implement
description: >
  Shared workflow contract for TensorRT-LLM modeling module specialists.
  Defines common phase lifecycle, state-file protocol, status transitions, and
  module-level verification/reporting conventions for specialist agents invoked
  by modeling-bringup.
tags:
  - tensorrt-llm
  - modeling
  - workflow
  - module
license: LicenseRef-NvidiaProprietary
metadata:
  author: NVIDIA Corporation
---

# Modeling Module Common Workflow

## Purpose

This skill defines the shared process for module specialists used by
`modeling-bringup`.

Use this as the base contract. Domain-specific specialists (for example MoE or
non-MoE module specialists) must add their own analysis/implementation details
on top of this workflow without violating these rules.

## Common Inputs

The orchestrator should provide:

| Input | Required | Description |
|-------|----------|-------------|
| `workspace_path` | Yes | Shared bring-up workspace (absolute path). The orchestrator must always provide this. |
| `reference_code_path` | Yes | Path to HuggingFace `modeling_*.py` reference implementation. |
| `checkpoint_path` | Yes | Checkpoint directory containing `config.json` and weight files. |
| `trtllm_repo_path` | Yes | TensorRT-LLM source tree path. |
| `model_name` | Yes | Model identifier (e.g. `llama`, `mixtral`). |
| `module_name` | Yes | Module identifier (e.g. `moe`, `embedding`). Used for sub-directory naming under `workspace_path`. |
| `task` | Yes | `plan`, `implement`, `test`, combined tasks (for example `plan+implement+test`), or bug-fix requests. |
| `auxiliary_info` | No | Extra context from orchestrator. |

Specialists may require additional fields. Those fields are defined in the
specialist prompt and do not replace this base input contract.

## State and Status Protocol

Each specialist works directly under `workspace_path` and must keep:

- `module_plan.md` - source of truth produced in planning.
- `status.md` - current state and latest error notes.

Status values:

- `planning`
- `implementing`
- `testing`
- `done`
- `blocked`

Rules:

1. Read `module_plan.md` and `status.md` at the start of every invocation if
   they exist.
2. Update `status.md` at phase start and phase end.
3. On failures, append/update an `Errors` section with:
   - failing test name
   - error type
   - root cause
   - suggested fix

## Shared Workspace Protocol

Unless a specialist defines a stricter layout, use this module-oriented default:

- `module_plan.md`: `<workspace_path>/module_plan.md`
- `status.md`: `<workspace_path>/status.md`
- `test file`: `<workspace_path>/test_module.py`
- `test log`: `<workspace_path>/test_output.txt`

The orchestrator can override paths, but specialists should keep this structure
when possible because downstream automation assumes it.

## Shared Phase Lifecycle

All specialists execute the same three-phase lifecycle.

### Phase 1: Plan

Required outcomes:

> **Important principles:**
> 1. Prefer direct reuse of already-implemented TensorRT-LLM modules with
>    the largest valid scope before considering smaller-scope composition
>    or custom implementations.
> 2. Scope is strictly limited to `<module_name>` and its in-scope
>    sub-modules; do not expand analysis or implementation to unrelated
>    modules.

1. Analyze reference code, identify integration points, and enumerate all
   in-scope sub-modules (name, responsibility, and execution order).
2. Read checkpoint config and extract module-relevant fields, including:
   - core dimensions (`hidden_size`, `intermediate_size`, `num_hidden_layers`)
   - head/layout fields (`num_attention_heads`, `num_key_value_heads`,
     `head_dim`)
   - activation and normalization fields (`hidden_act`, `rms_norm_eps`)
   - positional fields (`rope_theta`, `rope_scaling`,
     `max_position_embeddings`)
   - module-specific feature flags (bias toggles, cache options, tied weights)
3. Inspect checkpoint weights and infer layout/transform requirements,
   including fused-vs-split storage, transpose/permute rules, bias coverage,
   and quantization scale tensor mapping.
   Optional command template:

   ```bash
   python3 -c "
   import glob
   module_keywords = ['<module_name>', 'attn', 'mlp', 'norm', 'embed', 'lm_head', 'rope']
   try:
       from safetensors import safe_open
       files = sorted(glob.glob('<checkpoint_path>/*.safetensors'))
       for f in files:
           with safe_open(f, framework='pt') as st:
               for k in st.keys():
                   if any(x in k for x in module_keywords):
                       t = st.get_tensor(k)
                       print(f'{k}: {tuple(t.shape)} {t.dtype}')
   except ImportError:
       import torch
       files = sorted(glob.glob('<checkpoint_path>/*.bin'))
       for f in files:
           d = torch.load(f, map_location='cpu', weights_only=True)
           for k, v in d.items():
               if any(x in k for x in module_keywords):
                   print(f'{k}: {tuple(v.shape)} {v.dtype}')
   "
   ```
4. Inspect TRT-LLM reusable infrastructure and matching patterns from:
   - `<trtllm_repo_path>/tensorrt_llm/_torch/modules/`
   - `<trtllm_repo_path>/tensorrt_llm/_torch/models/modeling_*.py`
   - related loader/config/cache/tensor-parallel helpers
5. Analyze functional equivalence constraints:
   - exact forward operation order
   - residual/norm/scaling order
   - cache or state update semantics
   - cast points and precision-sensitive paths
6. Choose implementation strategy (reuse / wrapper / custom) and justify it.
7. Define weight loading + transform rules (key renaming, reshape/permute,
   fused/split handling, missing/unexpected key policy).
8. Analyze checkpoint quantization strategy and describe how it should be
   implemented in TensorRT-LLM modeling code.
9. Write `module_plan.md` with a JSON section that accurately captures
   module/sub-module containment relationships, and explicitly lists every
   initialization parameter for each sub-module.
10. Include a dedicated API/parameter mapping section in `module_plan.md`
    (reference symbol -> TRT-LLM symbol, with config source for each
    constructor argument).
11. Transition `status.md` from `planning` to `done` (or `implementing` if
   continuing immediately).

### Phase 2: Implement

> **CRITICAL — File Placement Rule (MUST NOT violate):**
>
> ALL modeling code files (model classes, decoder layers, config classes,
> weight mappers, helper modules, etc.) MUST be placed under
> `workspace_path`. You MUST NOT create or modify files
> under `<trtllm_repo_path>/tensorrt_llm/_torch/models/` — this path is
> FORBIDDEN for writes.
>
> You MAY read files under `_torch/models/` for reference and reuse
> existing infrastructure via imports. You MAY also modify other parts of
> the repo (e.g. `_torch/configs/`, `_torch/pyexecutor/`, `__init__.py`
> registration files) when necessary for wiring up the new model.
>
> The orchestrator or a later integration step is responsible for moving
> the validated modeling code from `workspace_path` into `_torch/models/`.
> Module specialists never do this.
>
> If you catch yourself about to write a file whose path falls under
> `<trtllm_repo_path>/tensorrt_llm/_torch/models/`, STOP and write it
> under `workspace_path` instead. No exceptions.

Required outcomes:

1. Treat `module_plan.md` as the single source of truth.
2. Before coding, read existing target modeling file (if present) and align
   edits with current model conventions.
3. Create `<module_name>.py` under `workspace_path` and
   implement this module's modeling code there. NEVER place it under
   `<trtllm_repo_path>/tensorrt_llm/`.
4. Implement parent-layer construction exactly from plan parameter mapping.
5. Implement forward integration in parent model/layer while preserving
   reference operation order for residual/norm/cache behavior.
6. Implement required weight transforms with minimal deterministic operations.
7. If the plan chooses `custom`, isolate module logic in a dedicated class and
   keep integration glue in parent classes.
8. Keep dtype/device handling explicit for runtime tensors and constants.
9. Transition `status.md` from `implementing` to `done` (or `testing` if
   continuing immediately).

### Phase 3: Test

> **CRITICAL — Test Import Rule (MUST NOT violate):**
>
> Test code in `<workspace_path>/test_module.py` MUST NOT import from
> `tensorrt_llm._torch.models` or any other repo source tree modeling
> path. The TRT-LLM side of the test MUST import the module
> implementation from the `workspace_path` files produced
> in Phase 2 — for example via `importlib` or `sys.path` manipulation
> pointing at `workspace_path`.
>
> Rationale: module-level tests validate the bring-up code in isolation
> before it is integrated into the repo. Importing from the repo source
> tree defeats this purpose and creates a false dependency on code that
> does not yet exist there.
>
> If a test file contains `from tensorrt_llm._torch.models import ...`
> or `from tensorrt_llm._torch.models.modeling_* import ...`, that is a
> violation. STOP and fix the import to point at the workspace copy.

Required outcomes:

1. Create `<workspace_path>/test_module.py` for module-level parity tests between
   HF reference and TensorRT-LLM implementation from Phase 2.
2. Implement test setup that guarantees fair comparison:
   - import the HF implementation from `reference_code_path`
   - import the TRT-LLM implementation from `workspace_path`
     (NEVER from `tensorrt_llm._torch.models`)
   - load matched checkpoint weights into both sides (after TRT-LLM transform)
   - use identical seeded inputs and identical runtime arguments
   - keep eval/inference mode and disable nondeterministic behavior when needed
3. Add forward correctness tests:
   - compare primary outputs (`hf_output` vs `trt_output`)
   - compare module-relevant side effects (for example cache tensors, branch
     outputs, normalized states)
   - report `max_abs_diff` and `mean_abs_diff` when assertion fails
4. Add structural/invariant tests:
   - output shape and dtype checks
   - module-specific invariants from `module_plan.md` (for example cache shape,
     residual update contract, norm-stat constraints, projection split/fuse
     consistency)
   - optional branch/feature-path checks when the module has conditional logic
5. Add weight-loading tests:
   - call model/module load path used in production (`load_weights` or
     equivalent)
   - assert no module-related missing keys
   - assert no module-related unexpected keys
   - verify transformed tensor shapes/layout match plan expectations
6. Use dtype-aware tolerances for numerical checks:
   - `float32`: start with `atol=1e-5`, `rtol=1e-5`
   - `bfloat16` / `float16`: start with `atol=1e-3`, `rtol=1e-3`
   - quantized paths: include max/mean error metrics in addition to
     `torch.allclose`, with plan-specific thresholds
7. Run tests and capture output to `<workspace_path>/test_output.txt`.
8. If all pass, mark `status.md` as `done`.
9. If failures exist, record an actionable `Errors` section and set status to
   `blocked` or `testing` (depending on orchestrator expectations), including:
   - failed test name
   - error type (shape mismatch, numerical divergence, missing weights, runtime
     error)
   - root cause hypothesis
   - next concrete fix suggestion

Suggested test function set in `test_module.py`:

- `test_forward_correctness()`
- `test_shape_dtype_invariants()`
- `test_weight_loading()`
- `test_module_side_effects_parity()` (when module has cache/state updates)
- `test_optional_branches_parity()` (when module has feature-flag branches)

Minimal `pytest` skeleton example:

```python
import torch


def _pick_tolerance(dtype):
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-3, 1e-3
    # Quantized or other paths should use plan-specific thresholds.
    return 1e-3, 1e-3


def _build_modules(reference_code_path, target_modeling_file, module_name, model_config, checkpoint_weights):
    # Pseudocode:
    # 1) import HF module from reference_code_path
    # 2) import TRT-LLM module/model from target_modeling_file
    # 3) load equivalent weights into both implementations
    # 4) set both to eval mode
    # Return: hf_module, trt_module
    raise NotImplementedError


def _sample_inputs(batch_size, seq_len, hidden_size, dtype, device="cuda"):
    torch.manual_seed(0)
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size, device=device, dtype=dtype
    )
    return {"hidden_states": hidden_states}


def test_forward_correctness():
    dtype = torch.float16
    atol, rtol = _pick_tolerance(dtype)
    hf_module, trt_module = _build_modules(
        reference_code_path="<reference_code_path>",
        target_modeling_file="<target_modeling_file>",
        module_name="<module_name>",
        model_config="<model_config>",
        checkpoint_weights="<checkpoint_weights>",
    )
    inputs = _sample_inputs(batch_size=2, seq_len=16, hidden_size=4096, dtype=dtype)
    with torch.no_grad():
        hf_out = hf_module(**inputs)
        trt_out = trt_module(**inputs)
    max_abs_diff = (hf_out - trt_out).abs().max().item()
    mean_abs_diff = (hf_out - trt_out).abs().mean().item()
    assert torch.allclose(hf_out, trt_out, atol=atol, rtol=rtol), (
        f"output mismatch: max_abs_diff={max_abs_diff}, mean_abs_diff={mean_abs_diff}, "
        f"atol={atol}, rtol={rtol}"
    )


def test_shape_dtype_invariants():
    dtype = torch.float16
    hf_module, trt_module = _build_modules(
        reference_code_path="<reference_code_path>",
        target_modeling_file="<target_modeling_file>",
        module_name="<module_name>",
        model_config="<model_config>",
        checkpoint_weights="<checkpoint_weights>",
    )
    inputs = _sample_inputs(batch_size=2, seq_len=16, hidden_size=4096, dtype=dtype)
    with torch.no_grad():
        hf_out = hf_module(**inputs)
        trt_out = trt_module(**inputs)
    assert trt_out.shape == hf_out.shape
    assert trt_out.dtype == hf_out.dtype
    # Add module-specific invariants from module_plan.md here.


def test_weight_loading():
    # Pseudocode:
    # model = <Model>ForCausalLM(<model_config>)
    # missing, unexpected = model.load_weights(<checkpoint_weights>)
    # module_missing = [k for k in missing if "<module_name>" in k or "<module_alias>" in k]
    # module_unexpected = [k for k in unexpected if "<module_name>" in k or "<module_alias>" in k]
    # assert not module_missing, f"Missing module weights: {module_missing}"
    # assert not module_unexpected, f"Unexpected module weights: {module_unexpected}"
    # Also verify transformed tensor layout/shape against module_plan.md.
    pass
```

Recommended command template:

```bash
pytest <test_file> -v 2>&1 | tee <workspace_path>/test_output.txt
echo "EXIT_CODE:$?"
```

## Verification and Quality Guardrails

- Preserve reference operation ordering in forward paths.
- Keep residual/norm/cache semantics consistent with reference.
- Keep weight transforms explicit and auditable.
- Prefer reuse of existing TRT-LLM modules over custom reimplementation when
  behavior matches.
- Keep transforms and tests focused on the assigned module scope.
- Ensure module constructor argument mapping is traceable back to config fields.
- Ensure test assertions cover both numerical parity and structural invariants.
