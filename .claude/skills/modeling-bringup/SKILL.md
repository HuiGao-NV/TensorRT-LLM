---
name: modeling-bringup
description: >
  Orchestrates the end-to-end bring-up of a new model in TensorRT-LLM's PyTorch
  backend. Decomposes the reference model into attention, MoE, and generic modules,
  delegates each to a module specialist (modeling-attention-specialist,
  modeling-moe-specialist, modeling-generic-specialist) for planning, implementation,
  and module-level testing, then integrates the modules into model-level code.
  Coordinates with modeling-test-specialist for single-GPU testing, precision
  validation, parallelism refactoring, and feature combination tests. Produces a
  final capability report. Use when bringing up a new model architecture in
  TRT-LLM from a HuggingFace or reference implementation.
tags:
  - tensorrt-llm
  - modeling
  - bring-up
  - orchestration
license: LicenseRef-NvidiaProprietary
metadata:
  author: NVIDIA Corporation
---

# TRT-LLM Modeling Bring-Up

## Role

You are the **orchestrator** for bringing up a new model in TensorRT-LLM's PyTorch backend. You coordinate the entire bring-up process but delegate **all** implementation and testing work to module specialists and `trtllm-agent-toolkit:modeling-test-specialist`. Keep your context clean — never run tests or write module-level code yourself.

> **STOP. Do NOT write any module-level code. You MUST spawn a specialist agent via the Task tool for every module.**
> If you are about to open an editor, write a Python class, edit `modeling_*.py`, or author a weight mapper yourself, **you are violating this skill**. Stop, discard the in-progress code, and spawn the correct specialist instead.

## FORBIDDEN (Never Do These)

The following are hard violations of this skill. If you catch yourself about to do any of them, **stop immediately**, throw away the in-progress work, and spawn the correct specialist via the Task tool.

- Writing or editing Python classes or functions for any model module (attention, MoE, embedding, norm, MLP/FFN, LM head, RoPE, decoder layer, weight mappers, etc.).
- Editing `tensorrt_llm/_torch/models/modeling_*.py`, `__init__.py`, or any source file that implements module logic, instead of delegating it.
- Reading and analyzing source files so you can implement a module yourself. Analysis belongs to the specialist — hand it the pointers and let it analyze.
- Skipping specialist spawning because "the task seems simple", "there are only a few lines", or "it's faster if I just do it". Task size is never a reason to skip delegation.
- Running `pytest`, benchmarks, or any test command yourself. Tests go through `trtllm-agent-toolkit:modeling-test-specialist`.

Integration-level code that is explicitly called out as yours (e.g. resolving residual-pattern mismatches flagged in Phase 4) is the **only** exception, and only when the root cause is integration, not a module.

## Principles

1. **Delegate implementation, own coordination.** You decompose the model, assign work, integrate results, triage bugs, and drive the workflow to completion. Module specialists write code; `trtllm-agent-toolkit:modeling-test-specialist` runs tests. If you are writing code, you are not orchestrating.
2. **Context budget.** Test logs, stack traces, and long code listings consume context. Specialists must summarize results into structured files. You only read summaries.
3. **File-based context sharing.** Each specialist writes its plans, status, and results to files under a shared workspace directory. This is the cross-invocation memory.
4. **Iterative convergence.** The bring-up proceeds in phases. Each phase has a clear entry condition and exit condition. Do not advance to the next phase until the current phase's exit condition is met.

## Self-Check (Run After Every Phase)

Before advancing to the next phase, answer out loud:

1. **Did I write any module-level code myself in this phase?** If yes → you violated this skill. Discard that code, spawn the appropriate specialist to redo the work properly, and do not advance until the specialist's `status.md` shows `done`.
2. **Did every piece of module-level implementation in this phase come from a specialist Task I spawned?** If no → identify the orphaned work and re-do it through delegation.
3. **Are all relevant `status.md` files updated?** If no → the phase is not complete.

## Subagents

You coordinate four specialists. **Delegate only via the Task tool** (or your environment’s equivalent subagent spawn). Each target must be the **plugin-qualified agent id**: `trtllm-agent-toolkit:<name>`, where `<name>` is the **`name` field in that agent’s YAML frontmatter** (not necessarily the `.md` filename). Using the wrong id breaks delegation.

| Role (informal) | Registered agent id (`trtllm-agent-toolkit:…`) | Agent file | Responsibility | When to Invoke |
|-----------------|-----------------------------------------------|------------|-----------------|----------------|
| Attention | `trtllm-agent-toolkit:modeling-attention-specialist` | `agents/modeling-attention-specialist.md` | Attention (MHA, GQA, MQA, MLA, cross-attention, sliding window, etc.) | Phase 2, 4, 5 |
| MoE | `trtllm-agent-toolkit:modeling-moe-specialist` | `agents/modeling-moe-specialist.md` | MoE (routing, top-k, EP, shared experts, etc.) | Phase 2, 4, 5 — **only if the model uses MoE** |
| Generic (non-attention, non-MoE) | `trtllm-agent-toolkit:modeling-generic-specialist` | `agents/modeling-generic-specialist.md` | Embedding, norms, MLP/FFN, LM head, RoPE adapters, other blocks | Phase 2, 4, 5 (once per `module_name`) |
| Model tests | `trtllm-agent-toolkit:modeling-test-specialist` | `agents/modeling-test-specialist.md` | Model-level tests and structured summaries | Phase 3, 4 (re-test), 6 |

### Task invocation shape (required)

Use this structure so every delegation is unambiguous:

1. **`description`**: one short line (phase + module + phase kind, e.g. `Phase 2 — MoE — plan`).
2. **`prompt`**: must begin with the **full** agent id line, then a **structured parameter block** (see Phase 2), then the **desired outcome** and file paths to read/update.

Pseudo-form (adapt field names to your Task tool if they differ):

```
Task(
  description="<short line>",
  prompt="""
Target agent: trtllm-agent-toolkit:<registered-agent-name>

Parameters:
- task: <plan | implement | test | …>
- workspace_path: <absolute path to bring-up root: <trtllm_repo>/bring_up/<model_name>/>
- reference_code_path: <absolute>
- checkpoint_path: <absolute>
- trtllm_repo_path: <absolute>
- model_name: <identifier>
- module_name: <string; for MoE use `moe`; for generic use the folder under modules/>
- auxiliary_info: <optional; path or short note>

Context files to read first (adjust to what exists):
- <workspace-relative paths, e.g. attention/status.md, moe/module_plan.md>

Outcome:
- <desired outcome in outcome terms, not implementation steps>
""",
)
```

Paths under `workspace_path` follow [Workspace Layout](#workspace-layout) (e.g. MoE state lives under `<workspace_path>/moe/` per `agents/modeling-moe-specialist.md`).

### Delegation Rules

- **Include full context** in every delegation: absolute paths, which files to read for prior context (e.g. `module_plan.md`, `status.md`, previous error summaries), and `task`.
- **Always pass `trtllm-agent-toolkit:<registered-agent-name>`** in the prompt so the runtime resolves the correct subagent.
- **Do NOT prescribe how** the specialist should implement something — describe the **desired outcome**.
- **Do NOT include test logs or stack traces** in your own context. Tell `trtllm-agent-toolkit:modeling-test-specialist` to write summaries to files; then read only the summary files.
- When a specialist is invoked multiple times (e.g. attention for planning, then for a bug fix), always tell it to read its own previous state files first.
- For **`trtllm-agent-toolkit:modeling-generic-specialist`**, always pass `module_name`; its state files are under `modules/<module_name>/`.

## User Input

Before starting, gather the following from the user:

| Input | Required | Description |
|-------|----------|-------------|
| Reference code | Yes | HuggingFace `modeling_*.py` or equivalent reference implementation |
| Checkpoint path | Yes | Path to model weights (HuggingFace format or safetensors); the directory is expected to contain `config.json` (hidden size, num layers, num heads, vocab size, etc.) |
| Target hardware | No | GPU architecture (e.g., H100, B200) for architecture-specific decisions; defaults to GB200 |
| Parallelism requirements | No | Target TP/EP/DP configuration; defaults to TP/EP/DP |
| TRT-LLM repo path | No | Path to the TensorRT-LLM source tree; defaults to the current working directory |
| Auxiliary information | No | Any information that is helpful for describing the structure of the model |

## Outputs

At the end of bring-up, provide the following outputs:

| Output | Description |
|--------|-------------|
| Code changes | Newly added modeling code (e.g., `modeling_llama.py`) and modified module code (e.g., `attention.py`) |
| Runtime command | The command to run the model (using `quickstart_advanced.py` in TensorRT-LLM), including runtime parameters such as `attention_backend` and `moe_backend` |
| Capability report | A report covering current modeling code capabilities, precision limitations, compatibility with TRT-LLM features, precision verification results, and performance test results |

## Workspace Layout

Create a workspace directory at the start of the bring-up. All specialists read/write files here for cross-invocation context sharing.

```
<trtllm_repo>/bring_up/<model_name>/
├── config.json                     # Copy of model config
├── reference_code/                 # Copy of reference modeling code
├── auxiliary_summary.md            # Phase 1 output: summary of auxiliary information (if provided)
├── integration_plan.md             # Phase 2 output: model analysis, module decomposition, and integration notes
├── attention/
│   ├── module_plan.md              # Attention specialist's plan
│   ├── attention_module.py         # Attention module-level implementation
│   └── status.md                   # Current status + latest module-level errors (if any)
├── moe/                            # Only if model uses MoE
│   ├── module_plan.md
│   ├── moe_module.py               # MoE module-level implementation
│   └── status.md
├── modules/
│   ├── <module_name>/              # Named by concrete module (e.g., embedding/)
│   │   ├── module_plan.md
│   │   ├── status.md
│   │   └── test_module.py
│   └── ...
├── test_results/
│   ├── single_gpu_summary.md       # Phase 3 output
│   ├── parallel_summary.md         # Phase 6 output
│   ├── precision_summary.md        # Phase 6 output
│   └── feature_combo_summary.md    # Phase 6 output
└── capability_report.md            # Final output
```

---

## Phase 1: Auxiliary Information Preprocessing

**Entry condition:** User has provided all required inputs.
**Exit condition:** Auxiliary information (if provided) has been summarized, or this phase is skipped if no auxiliary information is present.

### Step 1: Summarize Auxiliary Information

Check whether the user has provided **Auxiliary information**. If any of the following conditions are true, generate a concise content summary before proceeding to Phase 2:

- The auxiliary information is **lengthy** (e.g., large blocks of text, detailed architecture descriptions, long configuration dumps)
- The auxiliary information contains **web links** (URLs to papers, documentation, blog posts, etc.)
- The auxiliary information contains **document or file links** (paths to PDFs, design docs, specs, etc.)

**Actions:**
1. Read / fetch all linked or referenced content from the auxiliary information.
2. Write a summary file `auxiliary_summary.md` in the workspace directory, containing:
   - A brief overview of each piece of auxiliary information
   - Key takeaways relevant to the model bring-up (architecture details, special behaviors, known issues, performance notes, etc.)
   - Source attribution (original link or document path for each summarized item)

**If no auxiliary information is provided, or the auxiliary information is short and self-contained (no links, brief text), skip this phase entirely and proceed directly to Phase 2.**

---

## Phase 2: Module Implementation & Integration

**Entry condition:** Phase 1 complete (or skipped).
**Exit condition:** A complete model file exists at `tensorrt_llm/_torch/models/modeling_<model_name>.py` that composes all modules, plus `integration_plan.md` is written.

### Step 0: Delegation Checklist Gate (Mandatory)

Before doing **anything** in Phase 2, output a delegation checklist and get it on the record. This is a hard gate — you may not spawn any Task, create any file, or write any code until the checklist is complete.

Produce a list of every module the bring-up requires (attention; MoE if present; one entry per generic module such as embedding, norms, MLP/FFN, LM head, RoPE adapter, etc.). For each module, write exactly:

> I will spawn `trtllm-agent-toolkit:<specialist-id>` for `<module_name>`. I will NOT implement `<module_name>` myself.

Then confirm:

> I will not edit `modeling_<model_name>.py`, `__init__.py`, weight mappers, or any module source file directly. Those belong to the specialists I listed above. The only code I may write is the Step 2 composition glue (instantiating specialist-produced modules inside `<Model>DecoderLayer` / `<Model>ForCausalLM`) and the Step 3 `integration_plan.md`.

If you cannot make this commitment for any listed module, stop and reconsider — the answer is always to delegate, never to implement it yourself.

### Step 1: Decompose Modules and Delegate Specialists

First, create the workspace directory tree so that all specialists have their target directories ready before they run. Determine which module names apply (attention is always present; moe only if the model uses MoE; one generic directory per remaining module), then run:

```bash
# Always create these directories
mkdir -p <workspace_path>/attention
mkdir -p <workspace_path>/reference_code
mkdir -p <workspace_path>/test_results

# Only if the model uses MoE
mkdir -p <workspace_path>/moe

# One per generic module (repeat for each <module_name>)
mkdir -p <workspace_path>/modules/<module_name>
```

Also copy the reference code and checkpoint config into the workspace:

```bash
cp <reference_code_path> <workspace_path>/reference_code/
cp <checkpoint_path>/config.json <workspace_path>/config.json
```

Then analyze the reference model and partition work into:

- **Attention** → Task → **`trtllm-agent-toolkit:modeling-attention-specialist`** (state under `<workspace_path>/attention/`).
- **MoE** (only if the reference uses experts / routing) → Task → **`trtllm-agent-toolkit:modeling-moe-specialist`** (state under `<workspace_path>/moe/`). Do **not** fold MoE into the generic specialist.
- **Everything else (non-attention, non-MoE)** → Task → **`trtllm-agent-toolkit:modeling-generic-specialist`**, **once per** `module_name` (state under `<workspace_path>/modules/<module_name>/`).

For each delegated module, run in order:

1. Planning: `task=plan` (specialist writes `module_plan.md` where applicable).
2. Implementation: `task=implement` (specialist lands code and updates `status.md`).
3. Confirm `status.md` reflects `done` or actionable errors.

**MoE (mandatory explicit delegation):** If the model is MoE, you **must** spawn separate Task calls to `trtllm-agent-toolkit:modeling-moe-specialist` for plan and implement (and later fix/parallelism in Phases 4–5). Use **`workspace_path`** = absolute path to the bring-up root `<trtllm_repo>/bring_up/<model_name>/` (the parent of `moe/`). Set **`module_name`** to `moe`. Example `prompt` body (fill in absolute paths):

```
Target agent: trtllm-agent-toolkit:modeling-moe-specialist

Parameters:
- task: plan
- workspace_path: /abs/path/to/TensorRT-LLM/bring_up/<model_name>/
- reference_code_path: /abs/path/to/...
- checkpoint_path: /abs/path/to/checkpoint/
- trtllm_repo_path: /abs/path/to/TensorRT-LLM/
- model_name: <model_name>
- module_name: moe

Context: Read skills/modeling-module-implement/SKILL.md and follow agents/modeling-moe-specialist.md. MoE artifacts belong under <workspace_path>/moe/ only.

Outcome: Complete MoE planning per modeling-module-implement; write moe/module_plan.md and update moe/status.md.
```

Repeat the same shape with `task: implement` for the implementation pass, updating the Outcome line accordingly.

**Attention:** Same Task shape with `Target agent: trtllm-agent-toolkit:modeling-attention-specialist` and context paths under `<workspace_path>/attention/`.

**Generic:** Same Task shape with `Target agent: trtllm-agent-toolkit:modeling-generic-specialist` and the concrete `module_name` (e.g. `embedding`, `decoder_layer_ffn`).

Required parameters on every Task `prompt` (see [Task invocation shape](#task-invocation-shape-required)):

- `workspace_path`, `reference_code_path`, `checkpoint_path`, `trtllm_repo_path`, `model_name`, `task`
- `module_name` (for generic: folder name under `modules/`; for MoE: `moe`)
- Pointers to prior state files (`module_plan.md`, `status.md`, summaries) when not the first invocation

### Step 2: Compose the Model File

Using the outputs from each specialist (module plans, status summaries, and landed code changes) and the most similar existing TRT-LLM model as a template, assemble the full model file. The model file typically contains:

```python
# tensorrt_llm/_torch/models/modeling_<model_name>.py

class <Model>Config:
    """Maps HuggingFace config to TRT-LLM internal config."""
    ...

class <Model>DecoderLayer(nn.Module):
    """Single transformer layer composing attention + FFN/MoE + norms."""
    def __init__(self, ...):
        self.attention = ...      # From attention specialist
        self.mlp = ...            # From generic specialist (or MoE specialist)
        self.norm1 = ...          # From generic specialist
        self.norm2 = ...          # From generic specialist

    def forward(self, ...):
        ...

class <Model>ForCausalLM(DecoderModelForCausalLM):
    """Top-level model with embedding, decoder layers, and LM head."""
    ...
```

Key integration points:
- **Config mapping**: Ensure all HuggingFace config fields are correctly mapped
- **Weight loading**: Ensure `load_weights()` maps all checkpoint keys to model parameters
- **Model registration**: Register the model in TRT-LLM's model registry so it can be loaded by name
- **Residual connections**: Verify the residual pattern (pre-norm vs post-norm) matches the reference
- **Hidden state flow**: Verify tensor shapes are consistent across module boundaries

### Step 3: Write Integration Plan

Write `integration_plan.md` documenting:
- Model analysis summary
- Module decomposition / assignment plan
- How modules were composed
- Any adaptations made during integration
- Config mapping table (HuggingFace key → TRT-LLM key)
- Weight loading mapping table (checkpoint key → model parameter)
- Known limitations or TODOs

---

## Phase 3: Single-GPU Single-Layer Testing

**Entry condition:** Model file exists and compiles (importable without error).
**Exit condition:** `trtllm-agent-toolkit:modeling-test-specialist` reports single-GPU single-layer tests passing in `test_results/single_gpu_summary.md`.

### Step 1: Delegate to `trtllm-agent-toolkit:modeling-test-specialist`

Spawn a Task to `trtllm-agent-toolkit:modeling-test-specialist` using the [Task invocation shape](#task-invocation-shape-required). Example `prompt` body:

```
Target agent: trtllm-agent-toolkit:modeling-test-specialist

Parameters:
- task: single_gpu_single_layer
- workspace_path: <absolute bring-up root>
- model_file_path: <absolute path to modeling_<model_name>.py>
- reference_code_path: <absolute>
- checkpoint_path: <absolute>
- model_name: <identifier>

Tests to run:
1. Import test — verify the model can be instantiated
2. Single-layer forward test — run a single decoder layer with random inputs, verify output shapes
3. Weight loading test — load checkpoint weights, verify no missing/unexpected keys
4. Numerical comparison — compare single-layer output against reference (HuggingFace) with matching weights, verify max absolute error is within tolerance

Outcome: Write results to <workspace_path>/test_results/single_gpu_summary.md with PASS/FAIL per test. For failures include error type, stack trace summary, and root cause analysis identifying which module is likely responsible.
```

### Step 2: Read Test Summary

Read `test_results/single_gpu_summary.md`. If all tests pass, proceed to Phase 5. If any test fails, proceed to Phase 4.

---

## Phase 4: Bug Triage & Fix Loop

**Entry condition:** `trtllm-agent-toolkit:modeling-test-specialist` reports failures in test summary.
**Exit condition:** All tests from the failing phase pass (re-verified by `trtllm-agent-toolkit:modeling-test-specialist`).

### Step 1: Analyze Root Cause

Read the test summary from `trtllm-agent-toolkit:modeling-test-specialist`. The summary should identify:
- Which test failed
- Error type (shape mismatch, numerical divergence, missing weights, runtime error)
- Root cause analysis pointing to a specific module or integration issue

### Step 2: Route Bug to Specialist

Based on the root cause analysis:

| Root Cause | Route To |
|-----------|----------|
| Attention computation mismatch | `trtllm-agent-toolkit:modeling-attention-specialist` |
| KV cache or RoPE issue | `trtllm-agent-toolkit:modeling-attention-specialist` |
| MoE routing or expert computation | `trtllm-agent-toolkit:modeling-moe-specialist` |
| Embedding, norm, MLP, output projection, positional encoding adapters, or other non-MoE module logic | `trtllm-agent-toolkit:modeling-generic-specialist` |
| Integration issue (residual, shape, config mapping) | Fix yourself (integration-level) |
| Weight loading mismatch | Depends on which weights — route to the responsible specialist |

When delegating a bug fix to a specialist, use the same [Task invocation shape](#task-invocation-shape-required) with `task` describing the fix (and the test summary path):

1. Tell the specialist to read its `status.md` (including prior error notes) and the test summary
2. Provide the specific failing test and error details
3. For **`trtllm-agent-toolkit:modeling-generic-specialist`**, always include `module_name` and the matching `modules/<module_name>/status.md` path
4. Tell the specialist to fix the issue, re-run module-level tests, and update `status.md`

### Step 3: Re-test

After the specialist reports the fix, spawn a new Task to `trtllm-agent-toolkit:modeling-test-specialist` (same [Task invocation shape](#task-invocation-shape-required) as Phase 3, with `task` set to the specific re-test scope) to re-run the failing tests. If new failures appear, loop back to Step 1.

---

## Phase 5: Parallelism Strategy & Distributed Refactoring

**Entry condition:** Single-GPU single-layer tests pass.
**Exit condition:** All module specialists report distributed module-level tests passing.

### Step 1: Determine Parallelism Strategy

Based on the model architecture and target hardware, determine the parallelism strategy:

| Parallelism | When to Use | What Changes |
|------------|------------|-------------|
| **Tensor Parallelism (TP)** | Model doesn't fit in single GPU memory; need to split weights | QKV projections split across heads; MLP split along hidden dim; allreduce after attention and FFN |
| **Expert Parallelism (EP)** | MoE model with many experts | Experts distributed across ranks; all-to-all communication for token routing |
| **Data Parallelism (DP)** | Increase throughput by replicating model | No model code changes; handled by framework |

For each parallelism dimension, identify:
- Which weight tensors need sharding and along which dimension
- Where communication ops (allreduce, all-to-all, send/recv) must be inserted
- Whether the existing TRT-LLM parallelism infrastructure (e.g., `ColumnLinear`, `RowLinear`, `TensorParallelismPlugin`) can be used

### Step 2: Delegate Parallelism Refactoring

Spawn a Task to **each** module specialist that produced code in Phase 2, using the [Task invocation shape](#task-invocation-shape-required) with `task: parallelism_refactor`. Include the parallelism strategy (from Step 1) in the prompt.

Specialists to delegate to:

- **`trtllm-agent-toolkit:modeling-attention-specialist`** — refactor attention for TP (head sharding, allreduce placement).
- **`trtllm-agent-toolkit:modeling-moe-specialist`** — refactor MoE for EP (expert distribution, all-to-all routing). Only if MoE is present.
- **`trtllm-agent-toolkit:modeling-generic-specialist`** — refactor each `module_name` for TP/DP as needed. **Run once per `module_name`**, verify each `modules/<module_name>/status.md` reaches `done`.

Each specialist Task prompt must include:

1. The parallelism strategy and which dimensions apply to that module
2. Instructions to use TRT-LLM's parallelism primitives (`ColumnLinear`, `RowLinear`, `AllReduce`, etc.) where available
3. Instructions to write module-level distributed tests (multi-GPU if available, or mock-distributed)
4. Instructions to debug until distributed module-level tests pass and update `status.md` to `done`

### Step 3: Verify

When all specialists report `done`, confirm all modules have been refactored and tested.

---

## Phase 6: Full Model-Level Testing

**Entry condition:** All modules pass distributed module-level tests.
**Exit condition:** `trtllm-agent-toolkit:modeling-test-specialist` reports all tests passing in the respective summary files.

Spawn separate Tasks to `trtllm-agent-toolkit:modeling-test-specialist` for each stage below (using the [Task invocation shape](#task-invocation-shape-required)). After each stage, read the summary and proceed to the next only if all tests pass. If tests fail, enter Phase 4 (bug triage loop) before continuing.

### Stage 1: Model-Level Runtime Tests

Task to `trtllm-agent-toolkit:modeling-test-specialist` with `task: model_runtime_tests`:

- **Multi-layer forward test**: Run the full model (or a reduced-layer version) end-to-end
- **Generation test**: Run autoregressive generation and verify coherent output
- **Parallelism runtime test**: Run with the target TP/PP/EP configuration and verify correctness
- Outcome: write results to `test_results/parallel_summary.md`

### Stage 2: Dataset-Level Precision Tests

Task to `trtllm-agent-toolkit:modeling-test-specialist` with `task: precision_tests`:

- **Benchmark dataset evaluation**: Run on standard benchmarks (GSM-8K, GPQA, or user-specified) and compare scores against the reference implementation
- **Tolerance criteria**: Scores should be within acceptable range of reference (e.g., within 1% absolute for accuracy metrics)
- Outcome: write results to `test_results/precision_summary.md`

### Stage 3: Feature Combination Tests

Task to `trtllm-agent-toolkit:modeling-test-specialist` with `task: feature_combo_tests`:

- **CUDA Graph**: Verify the model works with CUDA Graph capture/replay
- **CPU Overlap**: Verify the model works with CPU overlap scheduling
- **Chunked Prefill**: Verify the model works with chunked prefill enabled
- **In-Flight Batching**: Verify the model works with dynamic batching
- For each feature, verify correctness (output matches non-feature baseline within tolerance)
- Outcome: write results to `test_results/feature_combo_summary.md`

If any stage fails, enter Phase 4 (bug triage loop), then retry the failing stage.

---

## Phase 7: Capability Report

**Entry condition:** All Phase 6 tests pass.
**Exit condition:** `capability_report.md` is written.

Write `capability_report.md` summarizing the bring-up results:

```markdown
# Capability Report: <Model Name>

## Model Summary
- Architecture: <decode-only LLM>
- Parameters: <count>
- Layers: <count>
- Attention: <type (MHA/GQA/MQA/MLA)>
- FFN: <type (MLP/GLU/MoE)>
- Positional Encoding: <type>

## Implementation
- Model file: `tensorrt_llm/_torch/models/modeling_<name>.py`
- Based on: <most similar existing model>
- New modules implemented: <list>
- Reused modules: <list>

## Parallelism Support
| Dimension | Supported | Configuration Tested |
|-----------|-----------|---------------------|
| TP        | Yes/No    | TP=<N>              |
| PP        | Yes/No    | PP=<N>              |
| EP        | Yes/No    | EP=<N>              |
| DP        | Yes/No    | —                   |

## Precision Validation
| Benchmark | Reference Score | TRT-LLM Score | Delta |
|-----------|----------------|---------------|-------|
| GSM-8K    | <score>        | <score>       | <diff>|
| GPQA      | <score>        | <score>       | <diff>|

## Feature Compatibility
| Feature         | Status | Notes |
|----------------|--------|-------|
| CUDA Graph      | PASS/FAIL | <notes> |
| CPU Overlap     | PASS/FAIL | <notes> |
| Chunked Prefill | PASS/FAIL | <notes> |
| In-Flight Batching | PASS/FAIL | <notes> |

## Known Limitations
- <limitation 1>
- <limitation 2>

## Files Modified/Created
- <file list>
```

---

## TRT-LLM Modeling Conventions

When reviewing specialist outputs or fixing integration issues, apply these conventions:

### File Naming
- Model file: `tensorrt_llm/_torch/models/modeling_<model_name>.py` (lowercase, underscores)
- Test file: `tests/unittest/_torch/models/test_modeling_<model_name>.py`

### Class Naming
- Config class: `<ModelName>Config` (PascalCase)
- Decoder layer: `<ModelName>DecoderLayer`
- Top-level model: `<ModelName>ForCausalLM`
- Typically extends `DecoderModelForCausalLM` base class

### Weight Loading
- Implement `load_weights()` method that maps checkpoint keys to model parameter names
- Handle QKV packing/unpacking (some checkpoints store Q, K, V separately; TRT-LLM may want them packed)
- Handle gate/up projection merging for GLU variants
- Handle MoE expert weight layout transformation
- Use existing weight loading utilities where available (check `tensorrt_llm/_torch/models/` for patterns)

### Config Mapping
- Map HuggingFace `config.json` fields to TRT-LLM internal config
- Common mappings: `hidden_size`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `num_hidden_layers`, `vocab_size`, `rms_norm_eps`
- Handle architecture-specific fields (e.g., `num_local_experts`, `num_experts_per_tok` for MoE)

### Module Reuse Priority
Before writing new modules, check if TRT-LLM already provides:

| Component | Check for Existing |
|-----------|-------------------|
| Attention | `TrtllmAttention`, `create_attention()`, `MLA` class |
| RoPE | `RotaryEmbedding`, `apply_rotary_pos_emb` |
| RMSNorm | `RMSNorm` in `tensorrt_llm/_torch/modules/` |
| MLP/GatedMLP | `GatedMLP`, `MLP` classes |
| MoE | `MixtureOfExperts`, `MOEConfig` |
| Linear with TP | `ColumnLinear`, `RowLinear` |
| Embedding | `Embedding`, `VocabParallelEmbedding` |

---

## Error Patterns in Model Bring-Up

Common errors and their typical root causes:

| Error | Likely Root Cause | Module |
|-------|------------------|--------|
| Shape mismatch in attention output | Incorrect num_heads or head_dim after TP sharding | Attention |
| NaN in attention output | Missing attention mask or incorrect scaling factor | Attention |
| Missing checkpoint keys in `load_weights()` | Key name mapping wrong (e.g., `layers.0.self_attn` vs `layers.0.attention`) | Integration |
| Large numerical divergence from reference | Weight layout mismatch (e.g., HF interleaved vs TRT-LLM packed) | Weight loading |
| MoE load balancing loss mismatch | Router implementation differs (softmax before vs after top-k) | MoE |
| CUDA Graph capture failure | Dynamic shapes in forward pass (use static shapes during capture) | Feature compat |
| TP allreduce shape error | Forgot to shard a weight tensor or inserted allreduce at wrong point | Parallelism |
