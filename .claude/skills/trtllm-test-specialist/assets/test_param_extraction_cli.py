#!/usr/bin/env python3
"""
Test parameter extraction from natural-language prompts for trtllm-test-specialist.

Each test case defines:
  - expected: dict of parameters the prompt should yield
  - prompt:   natural-language user request that embeds those parameters

The script calls Claude via `claude -p` CLI with the skill's Inputs section
injected and an instruction to extract parameters then stop — outputting only JSON.
It then compares the extracted mapping against the expected dict.

Usage:
    python3 test_param_extraction_cli.py [--model MODEL_ID] [--verbose]
    python3 test_param_extraction_cli.py --list         # print test case IDs
    python3 test_param_extraction_cli.py --id tc03      # run a single case

Exit codes:
    0  all tests passed
    1  one or more tests failed
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SKILL_MD = Path(__file__).parent.parent / "SKILL.md"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# ── Numeric parameters (must be JSON numbers, not strings) ─────────────────────

_NUMERIC_KEYS = {
    "tp_size", "pp_size", "ep_size", "dp_size", "required_devices",
    "num_requests", "input_mean", "output_mean", "concurrency",
    "max_batch_size", "max_num_tokens",
}

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_TMPL = """\
Invoke the trtllm-test-specialist skill:

{skill_section}

## Override: stop after parameter extraction

Stop work immediately after extracting parameters from the user's request.
Do NOT execute any workflow step, do NOT ask for clarification, and do NOT
proceed past extraction.

Output ONLY a single raw JSON object — no prose, no markdown code fences:
  {{"extracted_params": {{"param_name": value, ...}}}}

If nothing is found, output: {{"extracted_params": {{}}}}
"""

# ── Test cases ─────────────────────────────────────────────────────────────────
# Each case has:
#   id          — unique identifier
#   description — human-readable label (test scope, params count)
#   expected    — ground-truth parameter dict used for comparison
#   prompt      — natural-language user request that encodes those parameters
#
# Groups
#   tc01–tc12   original cases (mixed scopes)
#   tc13–tc32   module test variations
#   tc33–tc52   partial model test variations
#   tc53–tc92   benchmark test variations
#   tc93–tc122  evaluation test variations
#   tc123–tc142 functionality test variations
#   tc143–tc150 misc / edge cases
#   tc201–tc210 config_file override cases (other params must be dropped)

TEST_CASES = [
    # ── Original 12 ────────────────────────────────────────────────────────────
    {
        "id": "tc01",
        "description": "model_name + checkpoint_path only (2 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
        },
        "prompt": (
            "Run tests for llama_3_1_8b using the checkpoint at"
            " /models/Llama-3.1-8B-Instruct."
        ),
    },
    {
        "id": "tc02",
        "description": "benchmark with tp_size + pp_size (5 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "tp_size": 4,
            "pp_size": 1,
        },
        "prompt": (
            "Benchmark llama_3_1_8b at /models/Llama-3.1-8B-Instruct"
            " with tensor parallelism 4 and pipeline parallelism 1."
        ),
    },
    {
        "id": "tc03",
        "description": "evaluation with eval_tasks, no model_name (5 params)",
        "expected": {
            "checkpoint_path": "/checkpoints/llama-70b",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k,gpqa",
            "tp_size": 8,
            "pp_size": 1,
        },
        "prompt": (
            "Evaluate the model at /checkpoints/llama-70b on gsm8k and gpqa"
            " tasks using tp_size=8 and pp_size=1."
        ),
    },
    {
        "id": "tc04",
        "description": "module test: test_cmd + class_name (2 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_attention.py -v",
            "class_name": "Attention",
        },
        "prompt": (
            "Run the following module test:"
            " pytest tests/unittest/_torch/modules/test_attention.py -v."
            " The class under test is Attention."
        ),
    },
    {
        "id": "tc05",
        "description": "partial_model with layer_ids + tp_size (5 params)",
        "expected": {
            "model_name": "mistral_7b",
            "checkpoint_path": "/models/Mistral-7B-v0.1",
            "test_type": "partial_model",
            "layer_ids": "0,1,2,7,8,9",
            "tp_size": 2,
        },
        "prompt": (
            "Run a partial model test for mistral_7b at /models/Mistral-7B-v0.1"
            " on decoder layers 0,1,2,7,8,9 with tensor parallelism 2."
        ),
    },
    {
        "id": "tc06",
        "description": "benchmark with all optional bench params (13 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "benchmark",
            "backend": "pytorch",
            "bench_subcommand": "throughput",
            "num_requests": 512,
            "input_mean": 1024,
            "output_mean": 512,
            "concurrency": 32,
            "max_batch_size": 64,
            "max_num_tokens": 8192,
            "tp_size": 4,
            "pp_size": 2,
        },
        "prompt": (
            "Benchmark llama_3_1_70b (checkpoint /data/Llama-3.1-70B):"
            " throughput benchmark, pytorch backend, tp_size=4, pp_size=2,"
            " 512 requests, input_mean=1024, output_mean=512,"
            " concurrency=32, max_batch_size=64, max_num_tokens=8192."
        ),
    },
    {
        "id": "tc07",
        "description": "functionality test with report_file (5 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "functionality",
            "tp_size": 1,
            "report_file": "results/smoke_test.md",
        },
        "prompt": (
            "Run a functionality smoke test for llama_3_1_8b"
            " at /models/Llama-3.1-8B-Instruct with tp_size=1."
            " Write the report to results/smoke_test.md."
        ),
    },
    {
        "id": "tc08",
        "description": "MoE model with ep_size + dp_size, no test_type (5 params)",
        "expected": {
            "model_name": "mixtral_8x7b",
            "checkpoint_path": "/checkpoints/Mixtral-8x7B-Instruct-v0.1",
            "tp_size": 4,
            "ep_size": 8,
            "dp_size": 2,
        },
        "prompt": (
            "Test mixtral_8x7b at /checkpoints/Mixtral-8x7B-Instruct-v0.1"
            " with tp_size=4, ep_size=8, dp_size=2."
        ),
    },
    {
        "id": "tc09",
        "description": "module test with required_devices (3 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_moe.py -v",
            "class_name": "MixtralMoE",
            "required_devices": 4,
        },
        "prompt": (
            "Run pytest tests/unittest/_torch/modules/test_moe.py -v."
            " This test requires 4 GPUs. The class under test is MixtralMoE."
        ),
    },
    {
        "id": "tc10",
        "description": "evaluation with dataset_path + report_file (7 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k",
            "dataset_path": "/data/gsm8k.json",
            "tp_size": 2,
            "report_file": "eval_results/report.md",
        },
        "prompt": (
            "Evaluate llama_3_1_8b at /models/Llama-3.1-8B-Instruct on gsm8k"
            " using the dataset at /data/gsm8k.json, tp_size=2."
            " Write the report to eval_results/report.md."
        ),
    },
    {
        "id": "tc11",
        "description": "checkpoint_path only (1 param)",
        "expected": {
            "checkpoint_path": "/models/qwen2_72b",
        },
        "prompt": "Test the model at /models/qwen2_72b.",
    },
    {
        "id": "tc12",
        "description": "latency benchmark with extra_llm_api_options_yaml (6 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "tp_size": 2,
            "bench_subcommand": "latency",
            "extra_llm_api_options_yaml": "/tmp/pytorch_extra.yaml",
        },
        "prompt": (
            "Run a latency benchmark for llama_3_1_8b"
            " at /models/Llama-3.1-8B-Instruct with tp_size=2,"
            " passing extra LLM API options from /tmp/pytorch_extra.yaml."
        ),
    },

    # ── Module test variations (tc13–tc32) ─────────────────────────────────────
    {
        "id": "tc13",
        "description": "module test: test_cmd only (1 param)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_linear.py",
        },
        "prompt": "Run pytest tests/unittest/_torch/modules/test_linear.py.",
    },
    {
        "id": "tc14",
        "description": "module test: test_cmd + model_name (2 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_rope.py -v",
            "model_name": "llama_3_1_8b",
        },
        "prompt": (
            "Run pytest tests/unittest/_torch/modules/test_rope.py -v"
            " for model llama_3_1_8b."
        ),
    },
    {
        "id": "tc15",
        "description": "module test: test_cmd + required_devices=2 (2 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_allreduce.py -v",
            "required_devices": 2,
        },
        "prompt": (
            "Run pytest tests/unittest/_torch/modules/test_allreduce.py -v."
            " Needs 2 GPUs."
        ),
    },
    {
        "id": "tc16",
        "description": "module test: test_cmd + class_name + required_devices=4 (3 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_tp_linear.py -v",
            "class_name": "TensorParallelLinear",
            "required_devices": 4,
        },
        "prompt": (
            "Run pytest tests/unittest/_torch/modules/test_tp_linear.py -v."
            " The class under test is TensorParallelLinear. Requires 4 GPUs."
        ),
    },
    {
        "id": "tc17",
        "description": "module test: test_cmd + required_devices=8 (2 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_pp.py -v",
            "required_devices": 8,
        },
        "prompt": (
            "Execute pytest tests/unittest/_torch/modules/test_pp.py -v"
            " on 8 GPUs."
        ),
    },
    {
        "id": "tc18",
        "description": "module test: test_cmd + class_name + model_name (3 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_attention.py -v -k LlamaAttention",
            "class_name": "LlamaAttention",
            "model_name": "llama_3_1_8b",
        },
        "prompt": (
            "Run pytest tests/unittest/_torch/modules/test_attention.py -v -k LlamaAttention"
            " for the llama_3_1_8b model. The class being tested is LlamaAttention."
        ),
    },
    {
        "id": "tc19",
        "description": "module test: test_cmd + class_name + report_file (3 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_mlp.py -v",
            "class_name": "GatedMLP",
            "report_file": "module_reports/mlp_test.md",
        },
        "prompt": (
            "Run pytest tests/unittest/_torch/modules/test_mlp.py -v."
            " Class under test: GatedMLP."
            " Save the report to module_reports/mlp_test.md."
        ),
    },
    {
        "id": "tc20",
        "description": "module test: test_cmd + class_name + required_devices=1 (3 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_embedding.py -v",
            "class_name": "VocabParallelEmbedding",
            "required_devices": 1,
        },
        "prompt": (
            "Run pytest tests/unittest/_torch/modules/test_embedding.py -v."
            " The class under test is VocabParallelEmbedding. 1 GPU is sufficient."
        ),
    },
    {
        "id": "tc21",
        "description": "module test: complex pytest command with -k flag (2 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_norm.py -v -k RMSNorm",
            "class_name": "RMSNorm",
        },
        "prompt": (
            "Run pytest tests/unittest/_torch/modules/test_norm.py -v -k RMSNorm."
            " Testing the RMSNorm class."
        ),
    },
    {
        "id": "tc22",
        "description": "module test: test_cmd with --tb=short flag + class_name (2 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_decoder.py -v --tb=short",
            "class_name": "DecoderLayer",
        },
        "prompt": (
            "Please run: pytest tests/unittest/_torch/modules/test_decoder.py -v --tb=short."
            " The class under test is DecoderLayer."
        ),
    },
    {
        "id": "tc23",
        "description": "module test: test_cmd + model_name + required_devices (3 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_moe_routing.py -v",
            "model_name": "mixtral_8x7b",
            "required_devices": 2,
        },
        "prompt": (
            "Run pytest tests/unittest/_torch/modules/test_moe_routing.py -v"
            " for mixtral_8x7b on 2 GPUs."
        ),
    },
    {
        "id": "tc24",
        "description": "module test: different pytest invocation, no class (1 param)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/ -v --timeout=300",
        },
        "prompt": (
            "Run the full module test suite:"
            " pytest tests/unittest/_torch/modules/ -v --timeout=300."
        ),
    },
    {
        "id": "tc25",
        "description": "module test: test_cmd + repo_path (2 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_kv_cache.py -v",
            "repo_path": "/workspace/TensorRT-LLM",
        },
        "prompt": (
            "In the TensorRT-LLM repo at /workspace/TensorRT-LLM, run"
            " pytest tests/unittest/_torch/modules/test_kv_cache.py -v."
        ),
    },
    {
        "id": "tc26",
        "description": "module test: test_cmd + class_name + repo_path (3 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_attention.py -v",
            "class_name": "GroupedQueryAttention",
            "repo_path": "/home/user/TensorRT-LLM",
        },
        "prompt": (
            "Repo is at /home/user/TensorRT-LLM."
            " Run pytest tests/unittest/_torch/modules/test_attention.py -v."
            " Class under test: GroupedQueryAttention."
        ),
    },
    {
        "id": "tc27",
        "description": "module test: test_cmd + required_devices=16 (2 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_expert_parallel.py -v",
            "required_devices": 16,
        },
        "prompt": (
            "Run pytest tests/unittest/_torch/modules/test_expert_parallel.py -v."
            " This needs 16 GPUs."
        ),
    },
    {
        "id": "tc28",
        "description": "module test: all 4 module params (4 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_moe.py -v",
            "class_name": "MixtralMoE",
            "required_devices": 8,
            "model_name": "mixtral_8x22b",
        },
        "prompt": (
            "For mixtral_8x22b, run pytest tests/unittest/_torch/modules/test_moe.py -v."
            " Class under test: MixtralMoE. Requires 8 GPUs."
        ),
    },
    {
        "id": "tc29",
        "description": "module test: RoPE module, minimal (1 param)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_rope.py",
        },
        "prompt": "Execute the RoPE module test: pytest tests/unittest/_torch/modules/test_rope.py.",
    },
    {
        "id": "tc30",
        "description": "module test: class_name + required_devices + report_file (3 params, no test_cmd inferred)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_gelu.py -v",
            "class_name": "GeluActivation",
            "report_file": "reports/gelu_module.md",
        },
        "prompt": (
            "Run pytest tests/unittest/_torch/modules/test_gelu.py -v."
            " The class being tested is GeluActivation."
            " Output report to reports/gelu_module.md."
        ),
    },
    {
        "id": "tc31",
        "description": "module test: test_cmd + all optional module params (4 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_speculative.py -v",
            "class_name": "EagleSpeculativeDecoder",
            "required_devices": 4,
            "report_file": "reports/speculative_module.md",
        },
        "prompt": (
            "Run pytest tests/unittest/_torch/modules/test_speculative.py -v."
            " Class under test: EagleSpeculativeDecoder. Requires 4 GPUs."
            " Write the report to reports/speculative_module.md."
        ),
    },
    {
        "id": "tc32",
        "description": "module test: test_cmd + class_name + model_name + required_devices + report_file (5 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_lora.py -v",
            "class_name": "LoraLayer",
            "model_name": "llama_3_1_70b",
            "required_devices": 2,
            "report_file": "reports/lora_module.md",
        },
        "prompt": (
            "For llama_3_1_70b, run pytest tests/unittest/_torch/modules/test_lora.py -v."
            " Testing LoraLayer. Needs 2 GPUs. Report to reports/lora_module.md."
        ),
    },

    # ── Partial model test variations (tc33–tc52) ──────────────────────────────
    {
        "id": "tc33",
        "description": "partial_model: checkpoint only, no layer_ids (1 param)",
        "expected": {
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
        },
        "prompt": (
            "Run a partial model test using checkpoint /models/Llama-3.1-8B-Instruct."
        ),
    },
    {
        "id": "tc34",
        "description": "partial_model: checkpoint + model_name (2 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "partial_model",
        },
        "prompt": (
            "Do a partial model test for llama_3_1_8b"
            " at /models/Llama-3.1-8B-Instruct."
        ),
    },
    {
        "id": "tc35",
        "description": "partial_model: checkpoint + layer_ids=0,1,2 (3 params)",
        "expected": {
            "checkpoint_path": "/models/Mistral-7B-v0.1",
            "test_type": "partial_model",
            "layer_ids": "0,1,2",
        },
        "prompt": (
            "Partial model test for /models/Mistral-7B-v0.1 on layers 0,1,2."
        ),
    },
    {
        "id": "tc36",
        "description": "partial_model: checkpoint + layer_ids=4,5,6 (3 params)",
        "expected": {
            "checkpoint_path": "/checkpoints/Qwen2-7B-Instruct",
            "test_type": "partial_model",
            "layer_ids": "4,5,6",
        },
        "prompt": (
            "Run a partial layer test for the model at /checkpoints/Qwen2-7B-Instruct."
            " Test decoder layers 4, 5, and 6."
        ),
    },
    {
        "id": "tc37",
        "description": "partial_model: model_name + checkpoint + layer_ids (4 params)",
        "expected": {
            "model_name": "qwen2_7b",
            "checkpoint_path": "/checkpoints/Qwen2-7B-Instruct",
            "test_type": "partial_model",
            "layer_ids": "10,11,12",
        },
        "prompt": (
            "Partial model test for qwen2_7b at /checkpoints/Qwen2-7B-Instruct,"
            " layers 10,11,12."
        ),
    },
    {
        "id": "tc38",
        "description": "partial_model: model_name + checkpoint + layer_ids + tp_size=2 (5 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "partial_model",
            "layer_ids": "0,1,2",
            "tp_size": 2,
        },
        "prompt": (
            "Run partial model test for llama_3_1_70b at /data/Llama-3.1-70B"
            " on layers 0,1,2 with tensor parallelism 2."
        ),
    },
    {
        "id": "tc39",
        "description": "partial_model: checkpoint + tp_size=4 + pp_size=2 (4 params)",
        "expected": {
            "checkpoint_path": "/models/Llama-3.1-70B-Instruct",
            "test_type": "partial_model",
            "tp_size": 4,
            "pp_size": 2,
        },
        "prompt": (
            "Partial model test on /models/Llama-3.1-70B-Instruct"
            " with TP=4 and PP=2."
        ),
    },
    {
        "id": "tc40",
        "description": "partial_model: checkpoint + layer_ids + tp_size=1 (4 params)",
        "expected": {
            "checkpoint_path": "/models/phi-3-mini-4k-instruct",
            "test_type": "partial_model",
            "layer_ids": "0,1,2",
            "tp_size": 1,
        },
        "prompt": (
            "Run a single-GPU partial model test on /models/phi-3-mini-4k-instruct,"
            " checking layers 0,1,2. tp_size=1."
        ),
    },
    {
        "id": "tc41",
        "description": "partial_model: model_name + checkpoint + layer_ids + pp_size=2 (5 params)",
        "expected": {
            "model_name": "llama_3_1_405b",
            "checkpoint_path": "/scratch/Llama-3.1-405B-Instruct",
            "test_type": "partial_model",
            "layer_ids": "0,1,2,3",
            "pp_size": 2,
        },
        "prompt": (
            "Partial model test for llama_3_1_405b at /scratch/Llama-3.1-405B-Instruct."
            " Test layers 0,1,2,3 with pipeline parallelism 2."
        ),
    },
    {
        "id": "tc42",
        "description": "partial_model: checkpoint + layer_ids + report_file (4 params)",
        "expected": {
            "checkpoint_path": "/models/Mistral-7B-v0.1",
            "test_type": "partial_model",
            "layer_ids": "1,2,3",
            "report_file": "partial_reports/mistral_partial.md",
        },
        "prompt": (
            "Partial model test at /models/Mistral-7B-v0.1 on layers 1,2,3."
            " Save output to partial_reports/mistral_partial.md."
        ),
    },
    {
        "id": "tc43",
        "description": "partial_model: model_name + checkpoint + tp_size=8 (4 params)",
        "expected": {
            "model_name": "llama_3_1_405b",
            "checkpoint_path": "/models/Llama-3.1-405B",
            "test_type": "partial_model",
            "tp_size": 8,
        },
        "prompt": (
            "Run partial model test for llama_3_1_405b at /models/Llama-3.1-405B"
            " using 8-way tensor parallelism."
        ),
    },
    {
        "id": "tc44",
        "description": "partial_model: checkpoint + repo_path (3 params)",
        "expected": {
            "checkpoint_path": "/models/gemma-2-9b-it",
            "test_type": "partial_model",
            "repo_path": "/workspace/TensorRT-LLM",
        },
        "prompt": (
            "In the TensorRT-LLM repo at /workspace/TensorRT-LLM,"
            " run a partial model test on /models/gemma-2-9b-it."
        ),
    },
    {
        "id": "tc45",
        "description": "partial_model: many layer_ids (5 params)",
        "expected": {
            "model_name": "deepseek_r1_70b",
            "checkpoint_path": "/models/DeepSeek-R1-70B",
            "test_type": "partial_model",
            "layer_ids": "0,1,2,10,20,30,40,50,60,70",
            "tp_size": 4,
        },
        "prompt": (
            "Partial model test for deepseek_r1_70b at /models/DeepSeek-R1-70B."
            " Test layers 0,1,2,10,20,30,40,50,60,70 with tp_size=4."
        ),
    },
    {
        "id": "tc46",
        "description": "partial_model: model_name + checkpoint + single layer (4 params)",
        "expected": {
            "model_name": "gemma2_9b",
            "checkpoint_path": "/models/gemma-2-9b-it",
            "test_type": "partial_model",
            "layer_ids": "0",
        },
        "prompt": (
            "Run partial model test for gemma2_9b at /models/gemma-2-9b-it,"
            " testing only layer 0."
        ),
    },
    {
        "id": "tc47",
        "description": "partial_model: checkpoint + layer_ids + tp=4 + pp=2 (5 params)",
        "expected": {
            "checkpoint_path": "/models/Llama-3.1-70B-Instruct",
            "test_type": "partial_model",
            "layer_ids": "0,1,2",
            "tp_size": 4,
            "pp_size": 2,
        },
        "prompt": (
            "Partial layer test at /models/Llama-3.1-70B-Instruct on layers 0,1,2"
            " using tp_size=4 and pp_size=2."
        ),
    },
    {
        "id": "tc48",
        "description": "partial_model: model_name + checkpoint + layer_ids + device_type (5 params)",
        "expected": {
            "model_name": "nemotron_70b",
            "checkpoint_path": "/models/Nemotron-70B-Instruct",
            "test_type": "partial_model",
            "layer_ids": "0,1,2",
            "device_type": "GB200",
        },
        "prompt": (
            "Run partial model test for nemotron_70b at /models/Nemotron-70B-Instruct"
            " on layers 0,1,2 targeting GB200 GPUs."
        ),
    },
    {
        "id": "tc49",
        "description": "partial_model: full params with report_file + tp_size (6 params)",
        "expected": {
            "model_name": "mistral_7b",
            "checkpoint_path": "/models/Mistral-7B-v0.1",
            "test_type": "partial_model",
            "layer_ids": "0,1,2,3,4",
            "tp_size": 2,
            "report_file": "reports/mistral_partial.md",
        },
        "prompt": (
            "Partial model test for mistral_7b at /models/Mistral-7B-v0.1."
            " Layers 0,1,2,3,4. TP=2. Report: reports/mistral_partial.md."
        ),
    },
    {
        "id": "tc50",
        "description": "partial_model: model_name + checkpoint + layer_ids + repo_path + tp_size (6 params)",
        "expected": {
            "model_name": "qwen2_72b",
            "checkpoint_path": "/models/Qwen2-72B-Instruct",
            "test_type": "partial_model",
            "layer_ids": "0,1,2",
            "tp_size": 4,
            "repo_path": "/home/user/TensorRT-LLM",
        },
        "prompt": (
            "TensorRT-LLM repo: /home/user/TensorRT-LLM."
            " Partial model test for qwen2_72b at /models/Qwen2-72B-Instruct,"
            " layers 0,1,2, tp_size=4."
        ),
    },
    {
        "id": "tc51",
        "description": "partial_model: mixtral checkpoint + layer_ids (4 params)",
        "expected": {
            "model_name": "mixtral_8x7b",
            "checkpoint_path": "/checkpoints/Mixtral-8x7B-Instruct-v0.1",
            "test_type": "partial_model",
            "layer_ids": "0,1,2",
        },
        "prompt": (
            "Run partial model test for mixtral_8x7b"
            " at /checkpoints/Mixtral-8x7B-Instruct-v0.1 on layers 0,1,2."
        ),
    },
    {
        "id": "tc52",
        "description": "partial_model: HF model id as checkpoint + layer_ids + tp_size (4 params)",
        "expected": {
            "checkpoint_path": "meta-llama/Llama-3.1-8B-Instruct",
            "test_type": "partial_model",
            "layer_ids": "0,1,2",
            "tp_size": 1,
        },
        "prompt": (
            "Partial model test using HuggingFace model meta-llama/Llama-3.1-8B-Instruct."
            " Layers 0,1,2. tp_size=1."
        ),
    },

    # ── Benchmark test variations (tc53–tc92) ──────────────────────────────────
    {
        "id": "tc53",
        "description": "benchmark: model + checkpoint + test_type only (3 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
        },
        "prompt": "Benchmark llama_3_1_8b at /models/Llama-3.1-8B-Instruct.",
    },
    {
        "id": "tc54",
        "description": "benchmark: model + checkpoint + tp=1 (4 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "tp_size": 1,
        },
        "prompt": (
            "Run benchmark for llama_3_1_8b at /models/Llama-3.1-8B-Instruct"
            " with tp_size=1."
        ),
    },
    {
        "id": "tc55",
        "description": "benchmark: model + checkpoint + tp=2 (4 params)",
        "expected": {
            "model_name": "mistral_7b",
            "checkpoint_path": "/models/Mistral-7B-v0.1",
            "test_type": "benchmark",
            "tp_size": 2,
        },
        "prompt": (
            "Benchmark mistral_7b at /models/Mistral-7B-v0.1 using TP=2."
        ),
    },
    {
        "id": "tc56",
        "description": "benchmark: model + checkpoint + tp=8 + pp=1 (5 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "benchmark",
            "tp_size": 8,
            "pp_size": 1,
        },
        "prompt": (
            "Performance benchmark for llama_3_1_70b at /data/Llama-3.1-70B."
            " Use tp_size=8, pp_size=1."
        ),
    },
    {
        "id": "tc57",
        "description": "benchmark: MoE model with tp + ep (5 params)",
        "expected": {
            "model_name": "mixtral_8x7b",
            "checkpoint_path": "/checkpoints/Mixtral-8x7B-Instruct-v0.1",
            "test_type": "benchmark",
            "tp_size": 2,
            "ep_size": 4,
        },
        "prompt": (
            "Benchmark mixtral_8x7b at /checkpoints/Mixtral-8x7B-Instruct-v0.1"
            " with tp_size=2 and ep_size=4."
        ),
    },
    {
        "id": "tc58",
        "description": "benchmark: large MoE with tp=8 + ep=8 (5 params)",
        "expected": {
            "model_name": "mixtral_8x22b",
            "checkpoint_path": "/models/Mixtral-8x22B-Instruct-v0.1",
            "test_type": "benchmark",
            "tp_size": 8,
            "ep_size": 8,
        },
        "prompt": (
            "Run a performance benchmark for mixtral_8x22b"
            " at /models/Mixtral-8x22B-Instruct-v0.1 with TP=8 and EP=8."
        ),
    },
    {
        "id": "tc59",
        "description": "benchmark: backend=tensorrt (4 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "backend": "tensorrt",
        },
        "prompt": (
            "Benchmark llama_3_1_8b at /models/Llama-3.1-8B-Instruct"
            " using the tensorrt backend."
        ),
    },
    {
        "id": "tc60",
        "description": "benchmark: backend=_autodeploy (4 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "backend": "_autodeploy",
        },
        "prompt": (
            "Benchmark llama_3_1_8b at /models/Llama-3.1-8B-Instruct"
            " with backend=_autodeploy."
        ),
    },
    {
        "id": "tc61",
        "description": "benchmark: bench_subcommand=latency only (4 params)",
        "expected": {
            "model_name": "qwen2_7b",
            "checkpoint_path": "/models/Qwen2-7B-Instruct",
            "test_type": "benchmark",
            "bench_subcommand": "latency",
        },
        "prompt": (
            "Run a latency benchmark for qwen2_7b at /models/Qwen2-7B-Instruct."
        ),
    },
    {
        "id": "tc62",
        "description": "benchmark: latency + tp=2 + pp=1 (6 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "bench_subcommand": "latency",
            "tp_size": 2,
            "pp_size": 1,
        },
        "prompt": (
            "Latency benchmark for llama_3_1_8b at /models/Llama-3.1-8B-Instruct."
            " tp_size=2, pp_size=1."
        ),
    },
    {
        "id": "tc63",
        "description": "benchmark: throughput + num_requests=1000 (5 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "bench_subcommand": "throughput",
            "num_requests": 1000,
        },
        "prompt": (
            "Throughput benchmark for llama_3_1_8b at /models/Llama-3.1-8B-Instruct"
            " with 1000 requests."
        ),
    },
    {
        "id": "tc64",
        "description": "benchmark: input_mean=2048 + output_mean=1024 (6 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "benchmark",
            "tp_size": 4,
            "input_mean": 2048,
            "output_mean": 1024,
        },
        "prompt": (
            "Benchmark llama_3_1_70b at /data/Llama-3.1-70B with tp_size=4,"
            " input_mean=2048 and output_mean=1024."
        ),
    },
    {
        "id": "tc65",
        "description": "benchmark: short sequences input_mean=128 + output_mean=64 (6 params)",
        "expected": {
            "model_name": "phi3_mini",
            "checkpoint_path": "/models/Phi-3-mini-4k-instruct",
            "test_type": "benchmark",
            "tp_size": 1,
            "input_mean": 128,
            "output_mean": 64,
        },
        "prompt": (
            "Benchmark phi3_mini at /models/Phi-3-mini-4k-instruct, tp_size=1,"
            " input_mean=128, output_mean=64."
        ),
    },
    {
        "id": "tc66",
        "description": "benchmark: concurrency=16 (5 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "tp_size": 1,
            "concurrency": 16,
        },
        "prompt": (
            "Benchmark llama_3_1_8b at /models/Llama-3.1-8B-Instruct."
            " tp_size=1, concurrency=16."
        ),
    },
    {
        "id": "tc67",
        "description": "benchmark: concurrency=64 + max_batch_size=128 (6 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "benchmark",
            "tp_size": 4,
            "concurrency": 64,
            "max_batch_size": 128,
        },
        "prompt": (
            "Benchmark llama_3_1_70b at /data/Llama-3.1-70B, tp_size=4,"
            " concurrency=64, max_batch_size=128."
        ),
    },
    {
        "id": "tc68",
        "description": "benchmark: max_batch_size=256 only (4 params)",
        "expected": {
            "model_name": "mistral_7b",
            "checkpoint_path": "/models/Mistral-7B-v0.1",
            "test_type": "benchmark",
            "max_batch_size": 256,
        },
        "prompt": (
            "Run benchmark for mistral_7b at /models/Mistral-7B-v0.1"
            " with max_batch_size=256."
        ),
    },
    {
        "id": "tc69",
        "description": "benchmark: max_num_tokens=16384 (5 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "tp_size": 2,
            "max_num_tokens": 16384,
        },
        "prompt": (
            "Benchmark llama_3_1_8b at /models/Llama-3.1-8B-Instruct."
            " tp_size=2, max_num_tokens=16384."
        ),
    },
    {
        "id": "tc70",
        "description": "benchmark: dataset_path provided (5 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "tp_size": 2,
            "dataset_path": "/data/bench_dataset.txt",
        },
        "prompt": (
            "Benchmark llama_3_1_8b at /models/Llama-3.1-8B-Instruct, tp_size=2,"
            " using dataset /data/bench_dataset.txt."
        ),
    },
    {
        "id": "tc71",
        "description": "benchmark: dataset_path + report_file (6 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "benchmark",
            "tp_size": 4,
            "dataset_path": "/data/bench.txt",
            "report_file": "bench_reports/llama70b.md",
        },
        "prompt": (
            "Benchmark llama_3_1_70b at /data/Llama-3.1-70B, tp_size=4."
            " Dataset: /data/bench.txt. Report: bench_reports/llama70b.md."
        ),
    },
    {
        "id": "tc72",
        "description": "benchmark: extra_llm_api_options_yaml (5 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "tp_size": 1,
            "extra_llm_api_options_yaml": "/tmp/llm_options.yaml",
        },
        "prompt": (
            "Benchmark llama_3_1_8b at /models/Llama-3.1-8B-Instruct, tp_size=1."
            " Use extra LLM API options from /tmp/llm_options.yaml."
        ),
    },
    {
        "id": "tc73",
        "description": "benchmark: with report_file (5 params)",
        "expected": {
            "model_name": "qwen2_72b",
            "checkpoint_path": "/models/Qwen2-72B-Instruct",
            "test_type": "benchmark",
            "tp_size": 8,
            "report_file": "results/qwen2_72b_bench.md",
        },
        "prompt": (
            "Run benchmark for qwen2_72b at /models/Qwen2-72B-Instruct, tp_size=8."
            " Output report to results/qwen2_72b_bench.md."
        ),
    },
    {
        "id": "tc74",
        "description": "benchmark: device_type=GB200 (5 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "tp_size": 1,
            "device_type": "GB200",
        },
        "prompt": (
            "Benchmark llama_3_1_8b at /models/Llama-3.1-8B-Instruct"
            " on GB200 GPUs with tp_size=1."
        ),
    },
    {
        "id": "tc75",
        "description": "benchmark: device_type=B200 (5 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "benchmark",
            "tp_size": 4,
            "device_type": "B200",
        },
        "prompt": (
            "Benchmark llama_3_1_70b at /data/Llama-3.1-70B"
            " with tp_size=4 on B200 hardware."
        ),
    },
    {
        "id": "tc76",
        "description": "benchmark: tp + pp + ep + dp all set (7 params)",
        "expected": {
            "model_name": "mixtral_8x22b",
            "checkpoint_path": "/models/Mixtral-8x22B-Instruct-v0.1",
            "test_type": "benchmark",
            "tp_size": 4,
            "pp_size": 2,
            "ep_size": 8,
            "dp_size": 2,
        },
        "prompt": (
            "Benchmark mixtral_8x22b at /models/Mixtral-8x22B-Instruct-v0.1."
            " tp_size=4, pp_size=2, ep_size=8, dp_size=2."
        ),
    },
    {
        "id": "tc77",
        "description": "benchmark: with repo_path (5 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "tp_size": 1,
            "repo_path": "/workspace/TensorRT-LLM",
        },
        "prompt": (
            "Benchmark llama_3_1_8b at /models/Llama-3.1-8B-Instruct, tp_size=1."
            " TensorRT-LLM repo is at /workspace/TensorRT-LLM."
        ),
    },
    {
        "id": "tc78",
        "description": "benchmark: HF model id as checkpoint (4 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "meta-llama/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "tp_size": 1,
        },
        "prompt": (
            "Benchmark llama_3_1_8b using HuggingFace model"
            " meta-llama/Llama-3.1-8B-Instruct with tp_size=1."
        ),
    },
    {
        "id": "tc79",
        "description": "benchmark: tp=2 + backend=pytorch + num_requests=256 (6 params)",
        "expected": {
            "model_name": "mistral_7b",
            "checkpoint_path": "/models/Mistral-7B-v0.1",
            "test_type": "benchmark",
            "tp_size": 2,
            "backend": "pytorch",
            "num_requests": 256,
        },
        "prompt": (
            "Benchmark mistral_7b at /models/Mistral-7B-v0.1:"
            " tp_size=2, pytorch backend, 256 requests."
        ),
    },
    {
        "id": "tc80",
        "description": "benchmark: latency + concurrency=32 (6 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "bench_subcommand": "latency",
            "tp_size": 2,
            "concurrency": 32,
        },
        "prompt": (
            "Latency benchmark for llama_3_1_8b at /models/Llama-3.1-8B-Instruct,"
            " tp_size=2, concurrency=32."
        ),
    },
    {
        "id": "tc81",
        "description": "benchmark: throughput + max_batch_size + max_num_tokens (6 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "benchmark",
            "bench_subcommand": "throughput",
            "max_batch_size": 64,
            "max_num_tokens": 4096,
        },
        "prompt": (
            "Throughput benchmark for llama_3_1_70b at /data/Llama-3.1-70B."
            " max_batch_size=64, max_num_tokens=4096."
        ),
    },
    {
        "id": "tc82",
        "description": "benchmark: very large TP=16 for 405B model (5 params)",
        "expected": {
            "model_name": "llama_3_1_405b",
            "checkpoint_path": "/scratch/Llama-3.1-405B-Instruct",
            "test_type": "benchmark",
            "tp_size": 16,
            "pp_size": 1,
        },
        "prompt": (
            "Benchmark llama_3_1_405b at /scratch/Llama-3.1-405B-Instruct"
            " with tp_size=16, pp_size=1."
        ),
    },
    {
        "id": "tc83",
        "description": "benchmark: all four parallelism params (7 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "benchmark",
            "tp_size": 4,
            "pp_size": 2,
            "ep_size": 0,
            "dp_size": 1,
        },
        "prompt": (
            "Benchmark llama_3_1_70b at /data/Llama-3.1-70B."
            " tp_size=4, pp_size=2, ep_size=0, dp_size=1."
        ),
    },
    {
        "id": "tc84",
        "description": "benchmark: input/output means + num_requests (7 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "tp_size": 1,
            "num_requests": 500,
            "input_mean": 512,
            "output_mean": 256,
        },
        "prompt": (
            "Benchmark llama_3_1_8b at /models/Llama-3.1-8B-Instruct."
            " tp_size=1. Generate 500 requests with input_mean=512 tokens"
            " and output_mean=256 tokens."
        ),
    },
    {
        "id": "tc85",
        "description": "benchmark: dataset_path + extra_llm_api_options_yaml (6 params)",
        "expected": {
            "model_name": "qwen2_72b",
            "checkpoint_path": "/models/Qwen2-72B-Instruct",
            "test_type": "benchmark",
            "tp_size": 8,
            "dataset_path": "/data/qwen_bench.txt",
            "extra_llm_api_options_yaml": "/tmp/qwen_options.yaml",
        },
        "prompt": (
            "Benchmark qwen2_72b at /models/Qwen2-72B-Instruct, tp_size=8."
            " Use dataset /data/qwen_bench.txt and extra options from /tmp/qwen_options.yaml."
        ),
    },
    {
        "id": "tc86",
        "description": "benchmark: device_type + tp + pp (6 params)",
        "expected": {
            "model_name": "nemotron_70b",
            "checkpoint_path": "/models/Nemotron-70B-Instruct",
            "test_type": "benchmark",
            "tp_size": 8,
            "pp_size": 2,
            "device_type": "GB200",
        },
        "prompt": (
            "Benchmark nemotron_70b at /models/Nemotron-70B-Instruct"
            " on GB200 with tp_size=8 and pp_size=2."
        ),
    },
    {
        "id": "tc87",
        "description": "benchmark: benchmark_config_yaml (perf-sanity fallback) (4 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "benchmark",
            "benchmark_config_yaml": "examples/disaggregated/slurm/benchmark/config.yaml",
        },
        "prompt": (
            "Run a benchmark for llama_3_1_70b at /data/Llama-3.1-70B"
            " using the config file examples/disaggregated/slurm/benchmark/config.yaml."
        ),
    },
    {
        "id": "tc88",
        "description": "benchmark: pytorch + throughput + tp=4 + input/output + concurrency + report (9 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "backend": "pytorch",
            "bench_subcommand": "throughput",
            "tp_size": 4,
            "input_mean": 1024,
            "output_mean": 512,
            "concurrency": 16,
        },
        "prompt": (
            "Benchmark llama_3_1_8b at /models/Llama-3.1-8B-Instruct:"
            " throughput, pytorch backend, tp_size=4,"
            " input_mean=1024, output_mean=512, concurrency=16."
        ),
    },
    {
        "id": "tc89",
        "description": "benchmark: latency + backend=tensorrt + tp=2 + report_file (6 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "bench_subcommand": "latency",
            "backend": "tensorrt",
            "tp_size": 2,
        },
        "prompt": (
            "Run a latency benchmark for llama_3_1_8b at /models/Llama-3.1-8B-Instruct"
            " with tensorrt backend and tp_size=2."
        ),
    },
    {
        "id": "tc90",
        "description": "benchmark: num_requests=100 + input_mean=64 + output_mean=64 (7 params)",
        "expected": {
            "model_name": "phi3_mini",
            "checkpoint_path": "/models/Phi-3-mini-4k-instruct",
            "test_type": "benchmark",
            "tp_size": 1,
            "num_requests": 100,
            "input_mean": 64,
            "output_mean": 64,
        },
        "prompt": (
            "Quick benchmark for phi3_mini at /models/Phi-3-mini-4k-instruct."
            " tp_size=1, 100 requests, input_mean=64, output_mean=64."
        ),
    },
    {
        "id": "tc91",
        "description": "benchmark: deepseek with ep=8 + tp=8 (5 params)",
        "expected": {
            "model_name": "deepseek_r1_70b",
            "checkpoint_path": "/models/DeepSeek-R1-70B",
            "test_type": "benchmark",
            "tp_size": 8,
            "ep_size": 8,
        },
        "prompt": (
            "Benchmark deepseek_r1_70b at /models/DeepSeek-R1-70B"
            " with tp_size=8 and ep_size=8."
        ),
    },
    {
        "id": "tc92",
        "description": "benchmark: HF checkpoint + tp=4 + backend=pytorch + num_requests (6 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "meta-llama/Llama-3.1-70B-Instruct",
            "test_type": "benchmark",
            "tp_size": 4,
            "backend": "pytorch",
            "num_requests": 200,
        },
        "prompt": (
            "Benchmark llama_3_1_70b using HuggingFace model"
            " meta-llama/Llama-3.1-70B-Instruct."
            " pytorch backend, tp_size=4, 200 requests."
        ),
    },

    # ── Evaluation test variations (tc93–tc122) ────────────────────────────────
    {
        "id": "tc93",
        "description": "evaluation: checkpoint + eval_tasks=gsm8k only (2 params)",
        "expected": {
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "eval_tasks": "gsm8k",
        },
        "prompt": "Evaluate /models/Llama-3.1-8B-Instruct on gsm8k.",
    },
    {
        "id": "tc94",
        "description": "evaluation: model_name + checkpoint + eval_tasks=gsm8k (3 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k",
        },
        "prompt": (
            "Run an accuracy evaluation for llama_3_1_8b"
            " at /models/Llama-3.1-8B-Instruct on the gsm8k benchmark."
        ),
    },
    {
        "id": "tc95",
        "description": "evaluation: eval_tasks=gpqa (3 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "evaluation",
            "eval_tasks": "gpqa",
        },
        "prompt": (
            "Evaluate llama_3_1_70b at /data/Llama-3.1-70B on the gpqa benchmark."
        ),
    },
    {
        "id": "tc96",
        "description": "evaluation: eval_tasks=mmlu (3 params)",
        "expected": {
            "model_name": "mistral_7b",
            "checkpoint_path": "/models/Mistral-7B-v0.1",
            "test_type": "evaluation",
            "eval_tasks": "mmlu",
        },
        "prompt": "Evaluate mistral_7b at /models/Mistral-7B-v0.1 on mmlu.",
    },
    {
        "id": "tc97",
        "description": "evaluation: multi-task eval_tasks=gsm8k,gpqa,mmlu (4 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k,gpqa,mmlu",
        },
        "prompt": (
            "Evaluate llama_3_1_8b at /models/Llama-3.1-8B-Instruct"
            " on gsm8k, gpqa, and mmlu."
        ),
    },
    {
        "id": "tc98",
        "description": "evaluation: checkpoint + eval_tasks + tp=2 (4 params)",
        "expected": {
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k",
            "tp_size": 2,
        },
        "prompt": (
            "Evaluate /models/Llama-3.1-8B-Instruct on gsm8k with tp_size=2."
        ),
    },
    {
        "id": "tc99",
        "description": "evaluation: tp=4 + pp=2 (6 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k",
            "tp_size": 4,
            "pp_size": 2,
        },
        "prompt": (
            "Evaluate llama_3_1_70b at /data/Llama-3.1-70B on gsm8k."
            " Use tp_size=4, pp_size=2."
        ),
    },
    {
        "id": "tc100",
        "description": "evaluation: tp=8 for large model (5 params)",
        "expected": {
            "model_name": "llama_3_1_405b",
            "checkpoint_path": "/scratch/Llama-3.1-405B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "gpqa",
            "tp_size": 8,
        },
        "prompt": (
            "Evaluate llama_3_1_405b at /scratch/Llama-3.1-405B-Instruct"
            " on gpqa with tp_size=8."
        ),
    },
    {
        "id": "tc101",
        "description": "evaluation: MoE with ep=8 (6 params)",
        "expected": {
            "model_name": "mixtral_8x7b",
            "checkpoint_path": "/checkpoints/Mixtral-8x7B-Instruct-v0.1",
            "test_type": "evaluation",
            "eval_tasks": "mmlu",
            "tp_size": 2,
            "ep_size": 8,
        },
        "prompt": (
            "Evaluate mixtral_8x7b at /checkpoints/Mixtral-8x7B-Instruct-v0.1"
            " on mmlu with tp_size=2 and ep_size=8."
        ),
    },
    {
        "id": "tc102",
        "description": "evaluation: dp=2 (5 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k",
            "dp_size": 2,
        },
        "prompt": (
            "Evaluate llama_3_1_8b at /models/Llama-3.1-8B-Instruct on gsm8k"
            " with dp_size=2."
        ),
    },
    {
        "id": "tc103",
        "description": "evaluation: dataset_path provided (6 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k",
            "tp_size": 1,
            "dataset_path": "/data/gsm8k_test.json",
        },
        "prompt": (
            "Evaluate llama_3_1_8b at /models/Llama-3.1-8B-Instruct on gsm8k."
            " tp_size=1. Dataset at /data/gsm8k_test.json."
        ),
    },
    {
        "id": "tc104",
        "description": "evaluation: dataset_path + report_file (7 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "evaluation",
            "eval_tasks": "gpqa",
            "tp_size": 4,
            "dataset_path": "/data/gpqa.json",
            "report_file": "eval_results/llama70b_gpqa.md",
        },
        "prompt": (
            "Evaluate llama_3_1_70b at /data/Llama-3.1-70B on gpqa, tp_size=4."
            " Dataset: /data/gpqa.json. Report: eval_results/llama70b_gpqa.md."
        ),
    },
    {
        "id": "tc105",
        "description": "evaluation: device_type=GB200 (5 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k",
            "device_type": "GB200",
        },
        "prompt": (
            "Evaluate llama_3_1_8b at /models/Llama-3.1-8B-Instruct"
            " on gsm8k on GB200 hardware."
        ),
    },
    {
        "id": "tc106",
        "description": "evaluation: device_type=B200 + tp=4 (6 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "evaluation",
            "eval_tasks": "mmlu",
            "tp_size": 4,
            "device_type": "B200",
        },
        "prompt": (
            "Evaluate llama_3_1_70b at /data/Llama-3.1-70B on mmlu,"
            " tp_size=4 on B200 GPUs."
        ),
    },
    {
        "id": "tc107",
        "description": "evaluation: report_file only extra param (5 params)",
        "expected": {
            "model_name": "qwen2_72b",
            "checkpoint_path": "/models/Qwen2-72B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k,mmlu",
            "report_file": "eval_results/qwen2_72b.md",
        },
        "prompt": (
            "Evaluate qwen2_72b at /models/Qwen2-72B-Instruct on gsm8k and mmlu."
            " Save report to eval_results/qwen2_72b.md."
        ),
    },
    {
        "id": "tc108",
        "description": "evaluation: repo_path + tp (6 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k",
            "tp_size": 2,
            "repo_path": "/workspace/TensorRT-LLM",
        },
        "prompt": (
            "Evaluate llama_3_1_8b at /models/Llama-3.1-8B-Instruct on gsm8k."
            " tp_size=2. TensorRT-LLM repo: /workspace/TensorRT-LLM."
        ),
    },
    {
        "id": "tc109",
        "description": "evaluation: all parallelism + eval_tasks (7 params)",
        "expected": {
            "model_name": "mixtral_8x22b",
            "checkpoint_path": "/models/Mixtral-8x22B-Instruct-v0.1",
            "test_type": "evaluation",
            "eval_tasks": "gpqa,mmlu",
            "tp_size": 8,
            "pp_size": 1,
            "ep_size": 8,
        },
        "prompt": (
            "Evaluate mixtral_8x22b at /models/Mixtral-8x22B-Instruct-v0.1"
            " on gpqa and mmlu. tp_size=8, pp_size=1, ep_size=8."
        ),
    },
    {
        "id": "tc110",
        "description": "evaluation: HF model as checkpoint (4 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "meta-llama/Llama-3.1-8B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k",
        },
        "prompt": (
            "Evaluate llama_3_1_8b using HuggingFace model"
            " meta-llama/Llama-3.1-8B-Instruct on gsm8k."
        ),
    },
    {
        "id": "tc111",
        "description": "evaluation: benchmark_config_yaml multi-node fallback (4 params)",
        "expected": {
            "model_name": "llama_3_1_405b",
            "checkpoint_path": "/scratch/Llama-3.1-405B-Instruct",
            "test_type": "evaluation",
            "benchmark_config_yaml": "configs/accuracy_multinode.yaml",
        },
        "prompt": (
            "Evaluate llama_3_1_405b at /scratch/Llama-3.1-405B-Instruct"
            " using accuracy config configs/accuracy_multinode.yaml."
        ),
    },
    {
        "id": "tc112",
        "description": "evaluation: eval_tasks=humaneval (4 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "humaneval",
        },
        "prompt": (
            "Evaluate llama_3_1_8b at /models/Llama-3.1-8B-Instruct"
            " on the humaneval coding benchmark."
        ),
    },
    {
        "id": "tc113",
        "description": "evaluation: eval_tasks=winogrande,arc (4 params)",
        "expected": {
            "model_name": "mistral_7b",
            "checkpoint_path": "/models/Mistral-7B-v0.1",
            "test_type": "evaluation",
            "eval_tasks": "winogrande,arc",
        },
        "prompt": (
            "Evaluate mistral_7b at /models/Mistral-7B-v0.1"
            " on winogrande and arc."
        ),
    },
    {
        "id": "tc114",
        "description": "evaluation: 8-param full eval (8 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k,mmlu",
            "dataset_path": "/data/eval_data.json",
            "tp_size": 2,
            "pp_size": 1,
            "report_file": "reports/full_eval.md",
        },
        "prompt": (
            "Full evaluation for llama_3_1_8b at /models/Llama-3.1-8B-Instruct."
            " Tasks: gsm8k,mmlu. Dataset: /data/eval_data.json."
            " tp_size=2, pp_size=1. Report to reports/full_eval.md."
        ),
    },
    {
        "id": "tc115",
        "description": "evaluation: gemma2 on gsm8k (3 params)",
        "expected": {
            "model_name": "gemma2_9b",
            "checkpoint_path": "/models/gemma-2-9b-it",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k",
        },
        "prompt": (
            "Evaluate gemma2_9b at /models/gemma-2-9b-it on gsm8k."
        ),
    },
    {
        "id": "tc116",
        "description": "evaluation: deepseek with tp=8 + ep=8 + eval_tasks (6 params)",
        "expected": {
            "model_name": "deepseek_r1_70b",
            "checkpoint_path": "/models/DeepSeek-R1-70B",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k,gpqa",
            "tp_size": 8,
            "ep_size": 8,
        },
        "prompt": (
            "Evaluate deepseek_r1_70b at /models/DeepSeek-R1-70B"
            " on gpqa and gsm8k with tp_size=8, ep_size=8."
        ),
    },
    {
        "id": "tc117",
        "description": "evaluation: phi3_mini single GPU (3 params)",
        "expected": {
            "model_name": "phi3_mini",
            "checkpoint_path": "/models/Phi-3-mini-4k-instruct",
            "test_type": "evaluation",
            "eval_tasks": "mmlu",
        },
        "prompt": "Evaluate phi3_mini at /models/Phi-3-mini-4k-instruct on mmlu.",
    },
    {
        "id": "tc118",
        "description": "evaluation: nemotron + report_file (5 params)",
        "expected": {
            "model_name": "nemotron_70b",
            "checkpoint_path": "/models/Nemotron-70B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "gpqa,mmlu",
            "report_file": "results/nemotron_eval.md",
        },
        "prompt": (
            "Evaluate nemotron_70b at /models/Nemotron-70B-Instruct on gpqa and mmlu."
            " Report to results/nemotron_eval.md."
        ),
    },
    {
        "id": "tc119",
        "description": "evaluation: qwen2 + tp=4 + dp=2 (6 params)",
        "expected": {
            "model_name": "qwen2_72b",
            "checkpoint_path": "/models/Qwen2-72B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k,gpqa",
            "tp_size": 4,
            "dp_size": 2,
        },
        "prompt": (
            "Evaluate qwen2_72b at /models/Qwen2-72B-Instruct on gsm8k and gpqa."
            " tp_size=4, dp_size=2."
        ),
    },
    {
        "id": "tc120",
        "description": "evaluation: eval_tasks=truthfulqa (4 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "truthfulqa",
        },
        "prompt": (
            "Run accuracy evaluation for llama_3_1_8b"
            " at /models/Llama-3.1-8B-Instruct on truthfulqa."
        ),
    },
    {
        "id": "tc121",
        "description": "evaluation: 5-task eval (4 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k,gpqa,mmlu,arc,winogrande",
        },
        "prompt": (
            "Evaluate llama_3_1_70b at /data/Llama-3.1-70B"
            " on gsm8k, gpqa, mmlu, arc, and winogrande."
        ),
    },
    {
        "id": "tc122",
        "description": "evaluation: device_type + tp + eval_tasks + report_file (7 params)",
        "expected": {
            "model_name": "llama_3_1_405b",
            "checkpoint_path": "/scratch/Llama-3.1-405B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "gpqa",
            "tp_size": 16,
            "device_type": "GB200",
            "report_file": "reports/405b_gpqa_gb200.md",
        },
        "prompt": (
            "Evaluate llama_3_1_405b at /scratch/Llama-3.1-405B-Instruct on gpqa."
            " Run on GB200 with tp_size=16."
            " Report to reports/405b_gpqa_gb200.md."
        ),
    },

    # ── Functionality test variations (tc123–tc142) ────────────────────────────
    {
        "id": "tc123",
        "description": "functionality: model + checkpoint only (3 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "functionality",
        },
        "prompt": (
            "Run a functionality test for llama_3_1_8b"
            " at /models/Llama-3.1-8B-Instruct."
        ),
    },
    {
        "id": "tc124",
        "description": "functionality: tp=2 (4 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "functionality",
            "tp_size": 2,
        },
        "prompt": (
            "Smoke test llama_3_1_8b at /models/Llama-3.1-8B-Instruct with tp_size=2."
        ),
    },
    {
        "id": "tc125",
        "description": "functionality: tp=4 (4 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "functionality",
            "tp_size": 4,
        },
        "prompt": (
            "Run a functionality smoke test for llama_3_1_70b at /data/Llama-3.1-70B"
            " with tensor parallel 4."
        ),
    },
    {
        "id": "tc126",
        "description": "functionality: tp=1 + pp=1 (5 params)",
        "expected": {
            "model_name": "mistral_7b",
            "checkpoint_path": "/models/Mistral-7B-v0.1",
            "test_type": "functionality",
            "tp_size": 1,
            "pp_size": 1,
        },
        "prompt": (
            "Functionality test for mistral_7b at /models/Mistral-7B-v0.1."
            " tp_size=1, pp_size=1."
        ),
    },
    {
        "id": "tc127",
        "description": "functionality: tp=2 + pp=2 (5 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "functionality",
            "tp_size": 2,
            "pp_size": 2,
        },
        "prompt": (
            "Functionality test for llama_3_1_70b at /data/Llama-3.1-70B."
            " Use tensor parallelism 2 and pipeline parallelism 2."
        ),
    },
    {
        "id": "tc128",
        "description": "functionality: ep=4 (MoE) (5 params)",
        "expected": {
            "model_name": "mixtral_8x7b",
            "checkpoint_path": "/checkpoints/Mixtral-8x7B-Instruct-v0.1",
            "test_type": "functionality",
            "tp_size": 2,
            "ep_size": 4,
        },
        "prompt": (
            "Functionality test for mixtral_8x7b"
            " at /checkpoints/Mixtral-8x7B-Instruct-v0.1 with tp_size=2, ep_size=4."
        ),
    },
    {
        "id": "tc129",
        "description": "functionality: dp=2 (4 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "functionality",
            "dp_size": 2,
        },
        "prompt": (
            "Run functionality test for llama_3_1_8b at /models/Llama-3.1-8B-Instruct"
            " with dp_size=2."
        ),
    },
    {
        "id": "tc130",
        "description": "functionality: tp + ep combined (5 params)",
        "expected": {
            "model_name": "mixtral_8x22b",
            "checkpoint_path": "/models/Mixtral-8x22B-Instruct-v0.1",
            "test_type": "functionality",
            "tp_size": 4,
            "ep_size": 8,
        },
        "prompt": (
            "Smoke test mixtral_8x22b at /models/Mixtral-8x22B-Instruct-v0.1"
            " with tp_size=4 and ep_size=8."
        ),
    },
    {
        "id": "tc131",
        "description": "functionality: report_file (4 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "functionality",
            "report_file": "smoke_test_results/llama8b.md",
        },
        "prompt": (
            "Functionality test for llama_3_1_8b at /models/Llama-3.1-8B-Instruct."
            " Write report to smoke_test_results/llama8b.md."
        ),
    },
    {
        "id": "tc132",
        "description": "functionality: device_type=GB200 (4 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "functionality",
            "device_type": "GB200",
        },
        "prompt": (
            "Run a functionality test for llama_3_1_8b at /models/Llama-3.1-8B-Instruct"
            " on GB200 hardware."
        ),
    },
    {
        "id": "tc133",
        "description": "functionality: device_type=B200 + tp=8 (5 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "functionality",
            "tp_size": 8,
            "device_type": "B200",
        },
        "prompt": (
            "Functionality test for llama_3_1_70b at /data/Llama-3.1-70B"
            " on B200 GPUs with TP=8."
        ),
    },
    {
        "id": "tc134",
        "description": "functionality: repo_path (4 params)",
        "expected": {
            "model_name": "qwen2_7b",
            "checkpoint_path": "/checkpoints/Qwen2-7B-Instruct",
            "test_type": "functionality",
            "repo_path": "/home/user/TensorRT-LLM",
        },
        "prompt": (
            "Functionality test for qwen2_7b at /checkpoints/Qwen2-7B-Instruct."
            " TensorRT-LLM is at /home/user/TensorRT-LLM."
        ),
    },
    {
        "id": "tc135",
        "description": "functionality: tp + report_file + device_type (6 params)",
        "expected": {
            "model_name": "nemotron_70b",
            "checkpoint_path": "/models/Nemotron-70B-Instruct",
            "test_type": "functionality",
            "tp_size": 8,
            "device_type": "GB200",
            "report_file": "results/nemotron_smoke.md",
        },
        "prompt": (
            "Smoke test nemotron_70b at /models/Nemotron-70B-Instruct."
            " tp_size=8 on GB200. Report: results/nemotron_smoke.md."
        ),
    },
    {
        "id": "tc136",
        "description": "functionality: HF model as checkpoint (3 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "meta-llama/Llama-3.1-8B-Instruct",
            "test_type": "functionality",
        },
        "prompt": (
            "Functionality test for llama_3_1_8b using HuggingFace model"
            " meta-llama/Llama-3.1-8B-Instruct."
        ),
    },
    {
        "id": "tc137",
        "description": "functionality: all parallelism + report_file (7 params)",
        "expected": {
            "model_name": "mixtral_8x7b",
            "checkpoint_path": "/checkpoints/Mixtral-8x7B-Instruct-v0.1",
            "test_type": "functionality",
            "tp_size": 2,
            "pp_size": 2,
            "ep_size": 4,
            "dp_size": 1,
        },
        "prompt": (
            "Functionality test for mixtral_8x7b"
            " at /checkpoints/Mixtral-8x7B-Instruct-v0.1."
            " tp_size=2, pp_size=2, ep_size=4, dp_size=1."
        ),
    },
    {
        "id": "tc138",
        "description": "functionality: phi3_mini single GPU (3 params)",
        "expected": {
            "model_name": "phi3_mini",
            "checkpoint_path": "/models/Phi-3-mini-4k-instruct",
            "test_type": "functionality",
        },
        "prompt": (
            "Run functionality test for phi3_mini at /models/Phi-3-mini-4k-instruct."
        ),
    },
    {
        "id": "tc139",
        "description": "functionality: gemma2 (3 params)",
        "expected": {
            "model_name": "gemma2_9b",
            "checkpoint_path": "/models/gemma-2-9b-it",
            "test_type": "functionality",
        },
        "prompt": (
            "Smoke test gemma2_9b at /models/gemma-2-9b-it."
        ),
    },
    {
        "id": "tc140",
        "description": "functionality: deepseek with ep=8 (5 params)",
        "expected": {
            "model_name": "deepseek_r1_70b",
            "checkpoint_path": "/models/DeepSeek-R1-70B",
            "test_type": "functionality",
            "tp_size": 8,
            "ep_size": 8,
        },
        "prompt": (
            "Run a functionality test for deepseek_r1_70b at /models/DeepSeek-R1-70B"
            " with tp_size=8 and ep_size=8."
        ),
    },
    {
        "id": "tc141",
        "description": "functionality: qwen2_72b + tp=8 + report (5 params)",
        "expected": {
            "model_name": "qwen2_72b",
            "checkpoint_path": "/models/Qwen2-72B-Instruct",
            "test_type": "functionality",
            "tp_size": 8,
            "report_file": "results/qwen72b_smoke.md",
        },
        "prompt": (
            "Functionality test for qwen2_72b at /models/Qwen2-72B-Instruct."
            " TP=8. Report to results/qwen72b_smoke.md."
        ),
    },
    {
        "id": "tc142",
        "description": "functionality: tp + pp + ep + dp + repo_path + report_file (8 params)",
        "expected": {
            "model_name": "llama_3_1_405b",
            "checkpoint_path": "/scratch/Llama-3.1-405B-Instruct",
            "test_type": "functionality",
            "tp_size": 8,
            "pp_size": 2,
            "ep_size": 0,
            "dp_size": 1,
            "report_file": "reports/405b_smoke.md",
        },
        "prompt": (
            "Functionality test for llama_3_1_405b at /scratch/Llama-3.1-405B-Instruct."
            " tp_size=8, pp_size=2, ep_size=0, dp_size=1."
            " Write report to reports/405b_smoke.md."
        ),
    },

    # ── Misc / edge cases (tc143–tc150) ────────────────────────────────────────
    {
        "id": "tc143",
        "description": "edge: model_name + checkpoint + repo_path (3 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "repo_path": "/workspace/TensorRT-LLM",
        },
        "prompt": (
            "Test llama_3_1_8b at /models/Llama-3.1-8B-Instruct."
            " The TRT-LLM repo is at /workspace/TensorRT-LLM."
        ),
    },
    {
        "id": "tc144",
        "description": "edge: checkpoint + device_type only (2 params)",
        "expected": {
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "device_type": "GB200",
        },
        "prompt": "Test /models/Llama-3.1-8B-Instruct on GB200 GPUs.",
    },
    {
        "id": "tc145",
        "description": "edge: model_name + checkpoint + device_type (3 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "device_type": "B200",
        },
        "prompt": "Test llama_3_1_70b at /data/Llama-3.1-70B on B200.",
    },
    {
        "id": "tc146",
        "description": "edge: checkpoint + report_file (2 params)",
        "expected": {
            "checkpoint_path": "/models/Mistral-7B-v0.1",
            "report_file": "reports/mistral_test.md",
        },
        "prompt": (
            "Test the model at /models/Mistral-7B-v0.1."
            " Report to reports/mistral_test.md."
        ),
    },
    {
        "id": "tc147",
        "description": "edge: model_name + checkpoint + report_file + repo_path (4 params)",
        "expected": {
            "model_name": "gemma2_9b",
            "checkpoint_path": "/models/gemma-2-9b-it",
            "report_file": "results/gemma_report.md",
            "repo_path": "/home/user/TensorRT-LLM",
        },
        "prompt": (
            "Test gemma2_9b at /models/gemma-2-9b-it."
            " TRT-LLM repo: /home/user/TensorRT-LLM."
            " Report to results/gemma_report.md."
        ),
    },
    {
        "id": "tc148",
        "description": "edge: checkpoint + tp=1 + pp=1 (3 params, parallelism defaults stated)",
        "expected": {
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "tp_size": 1,
            "pp_size": 1,
        },
        "prompt": (
            "Run tests on /models/Llama-3.1-8B-Instruct with tp_size=1 and pp_size=1."
        ),
    },
    {
        "id": "tc149",
        "description": "edge: model_name + checkpoint + tp=4 + ep=0 + dp=1 (5 params, ep and dp stated explicitly as 0 and 1)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "tp_size": 4,
            "ep_size": 0,
            "dp_size": 1,
        },
        "prompt": (
            "Test llama_3_1_8b at /models/Llama-3.1-8B-Instruct."
            " tp_size=4, ep_size=0, dp_size=1."
        ),
    },
    {
        "id": "tc150",
        "description": "edge: all common params set (5 params, no test-specific params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "repo_path": "/workspace/TensorRT-LLM",
            "report_file": "results/llama8b_report.md",
            "device_type": "GB200",
        },
        "prompt": (
            "Model: llama_3_1_8b at /models/Llama-3.1-8B-Instruct."
            " TRT-LLM repo: /workspace/TensorRT-LLM."
            " Target device: GB200."
            " Write report to results/llama8b_report.md."
        ),
    },

    # ── Additional benchmark variations (tc151–tc165) ──────────────────────────
    {
        "id": "tc151",
        "description": "benchmark: pp=4 only, no tp specified (4 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "benchmark",
            "pp_size": 4,
        },
        "prompt": (
            "Benchmark llama_3_1_70b at /data/Llama-3.1-70B"
            " with pipeline parallelism 4."
        ),
    },
    {
        "id": "tc152",
        "description": "benchmark: tp=2 + pp=4 + throughput subcommand (6 params)",
        "expected": {
            "model_name": "llama_3_1_405b",
            "checkpoint_path": "/scratch/Llama-3.1-405B-Instruct",
            "test_type": "benchmark",
            "bench_subcommand": "throughput",
            "tp_size": 2,
            "pp_size": 4,
        },
        "prompt": (
            "Run a throughput benchmark for llama_3_1_405b"
            " at /scratch/Llama-3.1-405B-Instruct with tp_size=2 and pp_size=4."
        ),
    },
    {
        "id": "tc153",
        "description": "benchmark: high concurrency=128 + tp=4 (5 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "tp_size": 4,
            "concurrency": 128,
        },
        "prompt": (
            "Benchmark llama_3_1_8b at /models/Llama-3.1-8B-Instruct."
            " tp_size=4. Set concurrency to 128."
        ),
    },
    {
        "id": "tc154",
        "description": "benchmark: max_batch_size=512 (4 params)",
        "expected": {
            "model_name": "qwen2_72b",
            "checkpoint_path": "/models/Qwen2-72B-Instruct",
            "test_type": "benchmark",
            "max_batch_size": 512,
        },
        "prompt": (
            "Benchmark qwen2_72b at /models/Qwen2-72B-Instruct"
            " with max_batch_size=512."
        ),
    },
    {
        "id": "tc155",
        "description": "benchmark: long-context input_mean=4096 (6 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "benchmark",
            "tp_size": 4,
            "input_mean": 4096,
            "output_mean": 256,
        },
        "prompt": (
            "Long-context benchmark for llama_3_1_70b at /data/Llama-3.1-70B,"
            " tp_size=4, input_mean=4096, output_mean=256."
        ),
    },
    {
        "id": "tc156",
        "description": "benchmark: large output_mean=2048 (5 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "tp_size": 1,
            "output_mean": 2048,
        },
        "prompt": (
            "Benchmark llama_3_1_8b at /models/Llama-3.1-8B-Instruct."
            " Single GPU (tp_size=1), output_mean=2048."
        ),
    },
    {
        "id": "tc157",
        "description": "benchmark: max_num_tokens=32768 (4 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "max_num_tokens": 32768,
        },
        "prompt": (
            "Benchmark llama_3_1_8b at /models/Llama-3.1-8B-Instruct"
            " with max_num_tokens=32768."
        ),
    },
    {
        "id": "tc158",
        "description": "benchmark: small model phi3 latency (4 params)",
        "expected": {
            "model_name": "phi3_mini",
            "checkpoint_path": "/models/Phi-3-mini-4k-instruct",
            "test_type": "benchmark",
            "bench_subcommand": "latency",
        },
        "prompt": "Latency benchmark for phi3_mini at /models/Phi-3-mini-4k-instruct.",
    },
    {
        "id": "tc159",
        "description": "benchmark: concurrency + max_batch_size + max_num_tokens (7 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "benchmark",
            "tp_size": 4,
            "concurrency": 64,
            "max_batch_size": 256,
            "max_num_tokens": 16384,
        },
        "prompt": (
            "Benchmark llama_3_1_70b at /data/Llama-3.1-70B, tp_size=4."
            " concurrency=64, max_batch_size=256, max_num_tokens=16384."
        ),
    },
    {
        "id": "tc160",
        "description": "benchmark: dataset + concurrency + report_file (7 params)",
        "expected": {
            "model_name": "mistral_7b",
            "checkpoint_path": "/models/Mistral-7B-v0.1",
            "test_type": "benchmark",
            "tp_size": 2,
            "dataset_path": "/data/bench_data.txt",
            "concurrency": 32,
            "report_file": "bench_reports/mistral_bench.md",
        },
        "prompt": (
            "Benchmark mistral_7b at /models/Mistral-7B-v0.1, tp_size=2."
            " Dataset: /data/bench_data.txt. concurrency=32."
            " Report to bench_reports/mistral_bench.md."
        ),
    },
    {
        "id": "tc161",
        "description": "benchmark: ep only (no tp stated) for MoE (4 params)",
        "expected": {
            "model_name": "mixtral_8x7b",
            "checkpoint_path": "/checkpoints/Mixtral-8x7B-Instruct-v0.1",
            "test_type": "benchmark",
            "ep_size": 8,
        },
        "prompt": (
            "Benchmark mixtral_8x7b at /checkpoints/Mixtral-8x7B-Instruct-v0.1"
            " with expert parallelism 8."
        ),
    },
    {
        "id": "tc162",
        "description": "benchmark: pp=4 + backend=pytorch (5 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "benchmark",
            "pp_size": 4,
            "backend": "pytorch",
        },
        "prompt": (
            "Benchmark llama_3_1_70b at /data/Llama-3.1-70B"
            " with pipeline parallelism 4 using pytorch backend."
        ),
    },
    {
        "id": "tc163",
        "description": "benchmark: ep=4 + tp=4 + pytorch + num_requests (7 params)",
        "expected": {
            "model_name": "mixtral_8x22b",
            "checkpoint_path": "/models/Mixtral-8x22B-Instruct-v0.1",
            "test_type": "benchmark",
            "tp_size": 4,
            "ep_size": 4,
            "backend": "pytorch",
            "num_requests": 300,
        },
        "prompt": (
            "Benchmark mixtral_8x22b at /models/Mixtral-8x22B-Instruct-v0.1."
            " pytorch backend, tp_size=4, ep_size=4, 300 requests."
        ),
    },
    {
        "id": "tc164",
        "description": "benchmark: throughput + large num_requests=2000 (5 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "bench_subcommand": "throughput",
            "num_requests": 2000,
        },
        "prompt": (
            "Run a throughput benchmark for llama_3_1_8b"
            " at /models/Llama-3.1-8B-Instruct with 2000 requests."
        ),
    },
    {
        "id": "tc165",
        "description": "benchmark: latency + concurrency=4 (small) (5 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "benchmark",
            "bench_subcommand": "latency",
            "concurrency": 4,
        },
        "prompt": (
            "Latency benchmark for llama_3_1_8b at /models/Llama-3.1-8B-Instruct"
            " with concurrency=4."
        ),
    },

    # ── Additional evaluation variations (tc166–tc175) ─────────────────────────
    {
        "id": "tc166",
        "description": "evaluation: tp=4 + dp=2 combined (6 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k",
            "tp_size": 4,
            "dp_size": 2,
        },
        "prompt": (
            "Evaluate llama_3_1_70b at /data/Llama-3.1-70B on gsm8k."
            " tp_size=4, dp_size=2."
        ),
    },
    {
        "id": "tc167",
        "description": "evaluation: tp + pp + dp all set (7 params)",
        "expected": {
            "model_name": "llama_3_1_405b",
            "checkpoint_path": "/scratch/Llama-3.1-405B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "gpqa",
            "tp_size": 8,
            "pp_size": 2,
            "dp_size": 2,
        },
        "prompt": (
            "Evaluate llama_3_1_405b at /scratch/Llama-3.1-405B-Instruct on gpqa."
            " tp_size=8, pp_size=2, dp_size=2."
        ),
    },
    {
        "id": "tc168",
        "description": "evaluation: pp=4 (5 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "evaluation",
            "eval_tasks": "mmlu",
            "pp_size": 4,
        },
        "prompt": (
            "Evaluate llama_3_1_70b at /data/Llama-3.1-70B on mmlu"
            " with pipeline parallelism 4."
        ),
    },
    {
        "id": "tc169",
        "description": "evaluation: eval_tasks=hellaswag (4 params)",
        "expected": {
            "model_name": "mistral_7b",
            "checkpoint_path": "/models/Mistral-7B-v0.1",
            "test_type": "evaluation",
            "eval_tasks": "hellaswag",
        },
        "prompt": "Evaluate mistral_7b at /models/Mistral-7B-v0.1 on hellaswag.",
    },
    {
        "id": "tc170",
        "description": "evaluation: eval_tasks=piqa (4 params)",
        "expected": {
            "model_name": "phi3_mini",
            "checkpoint_path": "/models/Phi-3-mini-4k-instruct",
            "test_type": "evaluation",
            "eval_tasks": "piqa",
        },
        "prompt": "Evaluate phi3_mini at /models/Phi-3-mini-4k-instruct on piqa.",
    },
    {
        "id": "tc171",
        "description": "evaluation: repo_path + device_type (6 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k",
            "repo_path": "/workspace/TensorRT-LLM",
            "device_type": "GB200",
        },
        "prompt": (
            "Evaluate llama_3_1_8b at /models/Llama-3.1-8B-Instruct on gsm8k."
            " TRT-LLM repo: /workspace/TensorRT-LLM. Target: GB200."
        ),
    },
    {
        "id": "tc172",
        "description": "evaluation: dataset + report + repo (8 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k,gpqa",
            "tp_size": 4,
            "dataset_path": "/data/eval_data.json",
            "report_file": "reports/llama70b_eval.md",
            "repo_path": "/workspace/TensorRT-LLM",
        },
        "prompt": (
            "Evaluate llama_3_1_70b at /data/Llama-3.1-70B on gsm8k and gpqa."
            " tp_size=4. Dataset: /data/eval_data.json."
            " Report: reports/llama70b_eval.md."
            " TRT-LLM repo: /workspace/TensorRT-LLM."
        ),
    },
    {
        "id": "tc173",
        "description": "evaluation: HF checkpoint + tp (5 params)",
        "expected": {
            "model_name": "qwen2_72b",
            "checkpoint_path": "Qwen/Qwen2-72B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "mmlu",
            "tp_size": 8,
        },
        "prompt": (
            "Evaluate qwen2_72b using Qwen/Qwen2-72B-Instruct on mmlu with tp_size=8."
        ),
    },
    {
        "id": "tc174",
        "description": "evaluation: pp=2 + dp=2 (6 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k",
            "pp_size": 2,
            "dp_size": 2,
        },
        "prompt": (
            "Evaluate llama_3_1_70b at /data/Llama-3.1-70B on gsm8k."
            " pp_size=2, dp_size=2."
        ),
    },
    {
        "id": "tc175",
        "description": "evaluation: ep=4 + tp=4 for MoE (6 params)",
        "expected": {
            "model_name": "mixtral_8x7b",
            "checkpoint_path": "/checkpoints/Mixtral-8x7B-Instruct-v0.1",
            "test_type": "evaluation",
            "eval_tasks": "mmlu,gsm8k",
            "tp_size": 4,
            "ep_size": 4,
        },
        "prompt": (
            "Evaluate mixtral_8x7b at /checkpoints/Mixtral-8x7B-Instruct-v0.1"
            " on mmlu and gsm8k with tp_size=4 and ep_size=4."
        ),
    },

    # ── Additional partial model variations (tc176–tc185) ──────────────────────
    {
        "id": "tc176",
        "description": "partial_model: MoE with ep_size (5 params)",
        "expected": {
            "model_name": "mixtral_8x7b",
            "checkpoint_path": "/checkpoints/Mixtral-8x7B-Instruct-v0.1",
            "test_type": "partial_model",
            "layer_ids": "0,1,2",
            "ep_size": 4,
        },
        "prompt": (
            "Partial model test for mixtral_8x7b"
            " at /checkpoints/Mixtral-8x7B-Instruct-v0.1 on layers 0,1,2"
            " with ep_size=4."
        ),
    },
    {
        "id": "tc177",
        "description": "partial_model: last few layers 28,29,30 (4 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "partial_model",
            "layer_ids": "28,29,30",
        },
        "prompt": (
            "Run a partial model test for llama_3_1_8b"
            " at /models/Llama-3.1-8B-Instruct on the last layers 28,29,30."
        ),
    },
    {
        "id": "tc178",
        "description": "partial_model: full 7-param set (7 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "partial_model",
            "layer_ids": "0,1,2,3",
            "tp_size": 4,
            "pp_size": 2,
            "report_file": "reports/llama70b_partial.md",
        },
        "prompt": (
            "Partial model test for llama_3_1_70b at /data/Llama-3.1-70B."
            " Layers 0,1,2,3. tp_size=4, pp_size=2."
            " Report: reports/llama70b_partial.md."
        ),
    },
    {
        "id": "tc179",
        "description": "partial_model: device_type + layer_ids (5 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "partial_model",
            "layer_ids": "0,1,2",
            "device_type": "GB200",
        },
        "prompt": (
            "Partial layer test for llama_3_1_8b at /models/Llama-3.1-8B-Instruct"
            " on GB200, checking layers 0,1,2."
        ),
    },
    {
        "id": "tc180",
        "description": "partial_model: tp=16 large model (4 params)",
        "expected": {
            "model_name": "llama_3_1_405b",
            "checkpoint_path": "/scratch/Llama-3.1-405B-Instruct",
            "test_type": "partial_model",
            "tp_size": 16,
        },
        "prompt": (
            "Partial model test for llama_3_1_405b"
            " at /scratch/Llama-3.1-405B-Instruct with tp_size=16."
        ),
    },
    {
        "id": "tc181",
        "description": "partial_model: HF id + layer_ids + tp_size (4 params)",
        "expected": {
            "checkpoint_path": "mistralai/Mistral-7B-Instruct-v0.3",
            "test_type": "partial_model",
            "layer_ids": "0,1,2",
            "tp_size": 1,
        },
        "prompt": (
            "Partial model test using HuggingFace model"
            " mistralai/Mistral-7B-Instruct-v0.3 on layers 0,1,2, tp_size=1."
        ),
    },
    {
        "id": "tc182",
        "description": "partial_model: with dp_size (5 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "partial_model",
            "layer_ids": "0,1,2",
            "dp_size": 2,
        },
        "prompt": (
            "Partial model test for llama_3_1_8b at /models/Llama-3.1-8B-Instruct."
            " Test layers 0,1,2 with dp_size=2."
        ),
    },
    {
        "id": "tc183",
        "description": "partial_model: device_type=B200 + tp_size (5 params)",
        "expected": {
            "model_name": "qwen2_72b",
            "checkpoint_path": "/models/Qwen2-72B-Instruct",
            "test_type": "partial_model",
            "tp_size": 8,
            "device_type": "B200",
        },
        "prompt": (
            "Partial model test for qwen2_72b at /models/Qwen2-72B-Instruct"
            " on B200 hardware with tp_size=8."
        ),
    },
    {
        "id": "tc184",
        "description": "partial_model: report + layer_ids + tp + pp + repo (7 params)",
        "expected": {
            "model_name": "llama_3_1_8b",
            "checkpoint_path": "/models/Llama-3.1-8B-Instruct",
            "test_type": "partial_model",
            "layer_ids": "0,1,2",
            "tp_size": 2,
            "pp_size": 2,
            "report_file": "partial_reports/llama8b_partial.md",
        },
        "prompt": (
            "Partial model test for llama_3_1_8b at /models/Llama-3.1-8B-Instruct."
            " Layers 0,1,2. tp_size=2, pp_size=2."
            " Write report to partial_reports/llama8b_partial.md."
        ),
    },
    {
        "id": "tc185",
        "description": "partial_model: gemma2 model + layer_ids (4 params)",
        "expected": {
            "model_name": "gemma2_9b",
            "checkpoint_path": "/models/gemma-2-9b-it",
            "test_type": "partial_model",
            "layer_ids": "0,1,2,3,4",
        },
        "prompt": (
            "Run partial model test for gemma2_9b at /models/gemma-2-9b-it"
            " on decoder layers 0,1,2,3,4."
        ),
    },

    # ── Additional module test variations (tc186–tc195) ────────────────────────
    {
        "id": "tc186",
        "description": "module test: command uses python -m pytest style (2 params)",
        "expected": {
            "test_cmd": "python -m pytest tests/unittest/_torch/modules/test_attention.py -v",
            "class_name": "CausalSelfAttention",
        },
        "prompt": (
            "Run: python -m pytest tests/unittest/_torch/modules/test_attention.py -v."
            " Class under test: CausalSelfAttention."
        ),
    },
    {
        "id": "tc187",
        "description": "module test: pytest with -x (stop on first failure) + class (2 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_mlp.py -v -x",
            "class_name": "FusedMLP",
        },
        "prompt": (
            "Run pytest tests/unittest/_torch/modules/test_mlp.py -v -x."
            " Testing the FusedMLP class."
        ),
    },
    {
        "id": "tc188",
        "description": "module test: all params including repo_path + report_file (6 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_attention.py -v",
            "class_name": "MultiHeadAttention",
            "required_devices": 4,
            "model_name": "llama_3_1_70b",
            "repo_path": "/workspace/TensorRT-LLM",
            "report_file": "reports/mha_module.md",
        },
        "prompt": (
            "For llama_3_1_70b, run:"
            " pytest tests/unittest/_torch/modules/test_attention.py -v."
            " Class: MultiHeadAttention. Requires 4 GPUs."
            " TRT-LLM repo: /workspace/TensorRT-LLM."
            " Report to reports/mha_module.md."
        ),
    },
    {
        "id": "tc189",
        "description": "module test: required_devices=16 + class_name (2 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_dist_attn.py -v",
            "required_devices": 16,
        },
        "prompt": (
            "Run pytest tests/unittest/_torch/modules/test_dist_attn.py -v."
            " Needs 16 GPUs."
        ),
    },
    {
        "id": "tc190",
        "description": "module test: class_name + required_devices=2 + model_name (4 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_kv_cache.py -v",
            "class_name": "KVCacheManager",
            "required_devices": 2,
            "model_name": "mistral_7b",
        },
        "prompt": (
            "For mistral_7b run: pytest tests/unittest/_torch/modules/test_kv_cache.py -v."
            " Class under test: KVCacheManager. Requires 2 GPUs."
        ),
    },
    {
        "id": "tc191",
        "description": "module test: pytest with --log-cli-level=DEBUG (2 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_norm.py -v --log-cli-level=DEBUG",
            "class_name": "LayerNorm",
        },
        "prompt": (
            "Run pytest tests/unittest/_torch/modules/test_norm.py -v --log-cli-level=DEBUG."
            " Testing LayerNorm."
        ),
    },
    {
        "id": "tc192",
        "description": "module test: terse single-sentence prompt (1 param)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_embedding.py",
        },
        "prompt": "pytest tests/unittest/_torch/modules/test_embedding.py.",
    },
    {
        "id": "tc193",
        "description": "module test: model_name + required_devices, no class (3 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_rope.py -v",
            "model_name": "llama_3_1_8b",
            "required_devices": 1,
        },
        "prompt": (
            "For llama_3_1_8b, run pytest tests/unittest/_torch/modules/test_rope.py -v"
            " on 1 GPU."
        ),
    },
    {
        "id": "tc194",
        "description": "module test: pytest -k filter with multiple markers (2 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_attention.py -v -k 'not slow and not multi_gpu'",
            "class_name": "FlashAttention",
        },
        "prompt": (
            "Run: pytest tests/unittest/_torch/modules/test_attention.py -v"
            " -k 'not slow and not multi_gpu'."
            " The class under test is FlashAttention."
        ),
    },
    {
        "id": "tc195",
        "description": "module test: all 5 distinct module params (5 params)",
        "expected": {
            "test_cmd": "pytest tests/unittest/_torch/modules/test_speculative.py -v",
            "class_name": "DraftModelSpeculator",
            "required_devices": 2,
            "model_name": "llama_3_1_8b",
            "report_file": "reports/speculator_test.md",
        },
        "prompt": (
            "Test llama_3_1_8b speculative decoding module:"
            " pytest tests/unittest/_torch/modules/test_speculative.py -v."
            " Class: DraftModelSpeculator. 2 GPUs needed."
            " Report to reports/speculator_test.md."
        ),
    },

    # ── Complex / edge cases (tc196–tc200) ─────────────────────────────────────
    {
        "id": "tc196",
        "description": "complex: near-maximal benchmark (12 params)",
        "expected": {
            "model_name": "llama_3_1_70b",
            "checkpoint_path": "/data/Llama-3.1-70B",
            "test_type": "benchmark",
            "backend": "pytorch",
            "bench_subcommand": "throughput",
            "tp_size": 4,
            "pp_size": 2,
            "ep_size": 0,
            "num_requests": 512,
            "input_mean": 1024,
            "output_mean": 512,
            "concurrency": 32,
        },
        "prompt": (
            "Benchmark llama_3_1_70b at /data/Llama-3.1-70B."
            " Throughput benchmark, pytorch backend."
            " tp_size=4, pp_size=2, ep_size=0."
            " 512 requests, input_mean=1024, output_mean=512, concurrency=32."
        ),
    },
    {
        "id": "tc197",
        "description": "complex: maximal evaluation with 9 params",
        "expected": {
            "model_name": "llama_3_1_405b",
            "checkpoint_path": "/scratch/Llama-3.1-405B-Instruct",
            "test_type": "evaluation",
            "eval_tasks": "gsm8k,gpqa,mmlu",
            "tp_size": 16,
            "pp_size": 1,
            "ep_size": 0,
            "dataset_path": "/data/combined_eval.json",
            "report_file": "reports/405b_full_eval.md",
        },
        "prompt": (
            "Full evaluation for llama_3_1_405b at /scratch/Llama-3.1-405B-Instruct."
            " Tasks: gsm8k, gpqa, mmlu. tp_size=16, pp_size=1, ep_size=0."
            " Dataset: /data/combined_eval.json."
            " Report to reports/405b_full_eval.md."
        ),
    },
    {
        "id": "tc198",
        "description": "edge: tp_size + pp_size only, no model or checkpoint (2 params)",
        "expected": {
            "tp_size": 4,
            "pp_size": 2,
        },
        "prompt": "Run with tp_size=4 and pp_size=2.",
    },
    {
        "id": "tc199",
        "description": "edge: model_name only, no checkpoint or test type (1 param)",
        "expected": {
            "model_name": "llama_3_1_8b",
        },
        "prompt": "Run a test for llama_3_1_8b.",
    },
    {
        "id": "tc200",
        "description": "edge: all four parallelism defaults stated explicitly (4 params)",
        "expected": {
            "tp_size": 1,
            "pp_size": 1,
            "ep_size": 0,
            "dp_size": 1,
        },
        "prompt": (
            "Run with tp_size=1, pp_size=1, ep_size=0, dp_size=1."
        ),
    },

    # ── config_file override cases (tc201–tc210) ───────────────────────────────
    # When config_file is present, ALL other parameters must be dropped.
    # Expected always contains only config_file regardless of what else is stated.
    {
        "id": "tc201",
        "description": "config_file only (1 param)",
        "expected": {
            "config_file": "configs/benchmark_llama_8b.yaml",
        },
        "prompt": "Use config file configs/benchmark_llama_8b.yaml.",
    },
    {
        "id": "tc202",
        "description": "config_file + model_name + checkpoint — only config_file extracted (1 param)",
        "expected": {
            "config_file": "configs/llama_eval.yaml",
        },
        "prompt": (
            "Use config file configs/llama_eval.yaml."
            " Model: llama_3_1_8b at /models/Llama-3.1-8B-Instruct."
        ),
    },
    {
        "id": "tc203",
        "description": "config_file + test_type + tp_size — only config_file extracted (1 param)",
        "expected": {
            "config_file": "configs/bench_tp4.yaml",
        },
        "prompt": (
            "Run a benchmark with config file configs/bench_tp4.yaml."
            " tp_size=4, test_type=benchmark."
        ),
    },
    {
        "id": "tc204",
        "description": "config_file + evaluation params — only config_file extracted (1 param)",
        "expected": {
            "config_file": "configs/accuracy_gsm8k.yaml",
        },
        "prompt": (
            "Evaluate on gsm8k using config file configs/accuracy_gsm8k.yaml."
            " tp_size=8, eval_tasks=gsm8k, report_file=reports/eval.md."
        ),
    },
    {
        "id": "tc205",
        "description": "config_file + partial model params — only config_file extracted (1 param)",
        "expected": {
            "config_file": "configs/partial_model.yaml",
        },
        "prompt": (
            "Partial layer test using config file configs/partial_model.yaml."
            " Layers 0,1,2. tp_size=2."
        ),
    },
    {
        "id": "tc206",
        "description": "config_file + module test params — only config_file extracted (1 param)",
        "expected": {
            "config_file": "configs/module_test.yaml",
        },
        "prompt": (
            "Run module tests using config file configs/module_test.yaml."
            " test_cmd: pytest tests/unittest/_torch/modules/test_attention.py -v."
            " required_devices=4."
        ),
    },
    {
        "id": "tc207",
        "description": "config_file + all common params — only config_file extracted (1 param)",
        "expected": {
            "config_file": "configs/full_run.yaml",
        },
        "prompt": (
            "Run using config file configs/full_run.yaml."
            " model_name=llama_3_1_70b, checkpoint_path=/data/Llama-3.1-70B,"
            " tp_size=8, pp_size=2, ep_size=4, dp_size=1."
            " Report to reports/run.md."
        ),
    },
    {
        "id": "tc208",
        "description": "config_file with .yml extension + other params — only config_file extracted (1 param)",
        "expected": {
            "config_file": "configs/bench_config.yml",
        },
        "prompt": (
            "Benchmark llama_3_1_8b using configs/bench_config.yml."
            " tp_size=4, bench_subcommand=latency."
        ),
    },
    {
        "id": "tc209",
        "description": "config_file in deeply nested path + other params — only config_file extracted (1 param)",
        "expected": {
            "config_file": "examples/disaggregated/slurm/configs/accuracy_multinode.yaml",
        },
        "prompt": (
            "Evaluate using config file"
            " examples/disaggregated/slurm/configs/accuracy_multinode.yaml."
            " tp_size=16, eval_tasks=mmlu,gsm8k."
        ),
    },
    {
        "id": "tc210",
        "description": "config_file + device_type + repo_path — only config_file extracted (1 param)",
        "expected": {
            "config_file": "configs/gb200_bench.yaml",
        },
        "prompt": (
            "Use config file configs/gb200_bench.yaml on GB200."
            " TRT-LLM repo: /workspace/TensorRT-LLM."
        ),
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────────


def build_system_prompt(skill_md: str) -> str:
    return _SYSTEM_TMPL.format(skill_section=skill_md.strip())


def call_claude(model: str, system: str, user: str) -> str:
    result = subprocess.run(
        ["claude", "-p", "--model", model, "--system-prompt", system, user],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def parse_response(text: str) -> dict:
    """Extract the extracted_params dict from Claude's raw response text."""
    # Strip markdown fences if present
    text = re.sub(r"^```[a-z]*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```$", "", text.strip(), flags=re.MULTILINE)
    text = text.strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "extracted_params" in data:
            return data["extracted_params"]
        return data  # already a flat param dict
    except json.JSONDecodeError:
        pass

    # Find first JSON object in text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and "extracted_params" in data:
                return data["extracted_params"]
            return data
        except json.JSONDecodeError:
            pass

    return {}


def coerce_value(key: str, value):
    """Coerce a value to int for known numeric keys if it arrived as a string."""
    if key in _NUMERIC_KEYS and isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
    return value


def compare(expected: dict, extracted: dict) -> tuple[bool, list[str]]:
    """
    Compare extracted params against expected.

    Returns (passed, issues) where issues is a list of annotated lines.
    MISSING and MISMATCH entries cause failure; EXTRA entries are warnings.
    """
    issues: list[str] = []

    for key, exp_val in expected.items():
        if key not in extracted:
            issues.append(f"  MISSING   {key!r}  (expected {exp_val!r})")
        else:
            got = coerce_value(key, extracted[key])
            if got != exp_val:
                issues.append(f"  MISMATCH  {key!r}  expected={exp_val!r}  got={got!r}")

    for key in sorted(set(extracted) - set(expected)):
        issues.append(f"  EXTRA     {key!r} = {extracted[key]!r}  (not in expected — warning only)")

    passed = not any(line.startswith("  MISSING") or line.startswith("  MISMATCH") for line in issues)
    return passed, issues


# ── Runner ─────────────────────────────────────────────────────────────────────


def run_tests(
    model: str,
    system_prompt: str,
    cases: list[dict],
    verbose: bool,
    log_fp=None,
) -> bool:
    n_pass = n_fail = 0

    for tc in cases:
        print(f"\n[{tc['id']}] {tc['description']}")
        if verbose:
            print(f"  prompt   : {tc['prompt']!r}")

        if log_fp is not None:
            log_fp.write(f"\n[{tc['id']}] {tc['description']}\n")
            log_fp.write(f"  prompt   : {tc['prompt']!r}\n")
            log_fp.flush()

        raw = call_claude(model, system_prompt, tc["prompt"])

        if verbose:
            print(f"  response : {raw!r}")

        extracted = parse_response(raw)
        passed, issues = compare(tc["expected"], extracted)

        status = "PASS" if passed else "FAIL"
        print(f"  status   : {status}")
        if verbose or not passed:
            print(f"  extracted: {json.dumps(extracted)}")
        for line in issues:
            print(line)

        if log_fp is not None:
            log_fp.write(f"  response : {raw!r}\n")
            log_fp.write(f"  status   : {status}\n")
            log_fp.write(f"  extracted: {json.dumps(extracted)}\n")
            for line in issues:
                log_fp.write(line + "\n")
            log_fp.flush()

        if passed:
            n_pass += 1
        else:
            n_fail += 1

    total = n_pass + n_fail
    print(f"\n{'=' * 60}")
    print(f"Results: {n_pass}/{total} passed, {n_fail}/{total} failed")
    if log_fp is not None:
        log_fp.write(f"\n{'=' * 60}\n")
        log_fp.write(f"Results: {n_pass}/{total} passed, {n_fail}/{total} failed\n")
        log_fp.flush()
    return n_fail == 0


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, metavar="MODEL_ID", help=f"Claude model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show prompts and raw responses")
    parser.add_argument("--list", action="store_true", help="Print all test case IDs and exit")
    parser.add_argument("--id", metavar="ID", help="Run only the test case with this ID")
    parser.add_argument("--log-file", metavar="PATH", help="Write system prompt and per-case prompts/responses to this log file")
    args = parser.parse_args()

    if args.list:
        for tc in TEST_CASES:
            print(f"  {tc['id']:8s}  {tc['description']}")
        return

    cases = TEST_CASES
    if args.id:
        cases = [tc for tc in TEST_CASES if tc["id"] == args.id]
        if not cases:
            sys.exit(f"ERROR: no test case with id={args.id!r}")

    if not SKILL_MD.exists():
        sys.exit(f"ERROR: SKILL.md not found at {SKILL_MD}")

    skill_md = SKILL_MD.read_text()
    system_prompt = build_system_prompt(skill_md)

    if args.verbose:
        print("=== System prompt ===")
        print(system_prompt)
        print("=" * 60)

    log_fp = None
    if args.log_file:
        log_fp = open(args.log_file, "w")
        log_fp.write("=== System prompt ===\n")
        log_fp.write(system_prompt + "\n")
        log_fp.write("=" * 60 + "\n")
        log_fp.flush()

    try:
        ok = run_tests(args.model, system_prompt, cases, args.verbose, log_fp=log_fp)
    finally:
        if log_fp is not None:
            log_fp.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
