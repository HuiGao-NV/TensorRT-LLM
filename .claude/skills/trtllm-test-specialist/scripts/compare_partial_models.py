#!/usr/bin/env python3
"""Compare outputs of HuggingFace and TensorRT-LLM for a partial model.

Workflow:
1. Dump a new partial HF checkpoint containing only the selected layers.
2. Load the partial checkpoint as a HuggingFace model and run forward to get
   context logits.
3. Initialize a TensorRT-LLM ``LLM`` instance from the same partial checkpoint
   and generate with ``return_context_logits=True`` to get context logits.
4. Compare the context logits from both frameworks.

Usage:
    python compare_partial_models.py \
        --checkpoint_path /path/to/hf/model \
        --layer_ids 1,2,3,7,8,9

    python compare_partial_models.py \
        --checkpoint_path /path/to/hf/model \
        --layer_ids 1,2,3,7,8,9 \
        --atol 1e-2 --rtol 1e-2 \
        --prompt "Hello, my name is"

    # Multi-GPU: tensor parallel 4, pipeline parallel 1
    python compare_partial_models.py \
        --checkpoint_path /path/to/hf/model \
        --layer_ids 1,2,3,7,8,9 \
        --tp_size 4
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Ensure sibling scripts are importable
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from dump_hf_partial_model import dump_partial_model


def _parse_layer_ids(value: str) -> List[int]:
    """Parse a comma-separated string of layer IDs into a list of ints."""
    try:
        return [int(x.strip()) for x in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid layer_ids format: '{value}'. Expected comma-separated integers (e.g., 0,5,31)."
        )


def compare_partial_models(
    checkpoint_path: str,
    layer_ids: List[int],
    prompt: str = "Hello, my name is",
    max_tokens: int = 16,
    torch_dtype: Optional[torch.dtype] = None,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    output_dir: Optional[str] = None,
    keep_dump: bool = False,
    tp_size: int = 1,
    pp_size: int = 1,
) -> Dict:
    """Dump partial model, run HF and TRT-LLM, and compare context logits.

    Args:
        checkpoint_path: Path to the full HuggingFace checkpoint directory.
        layer_ids: Original layer indices to keep (e.g., [1, 2, 3, 7, 8, 9]).
        prompt: Text prompt for generation.
        max_tokens: Max new tokens for TRT-LLM generation.
        torch_dtype: Model dtype for HF model. If None, inferred from config.
        atol: Absolute tolerance for torch.allclose.
        rtol: Relative tolerance for torch.allclose.
        output_dir: Directory for the dumped partial checkpoint. If None, uses
            a temporary directory.
        keep_dump: If True, keep the dumped checkpoint after comparison.
        tp_size: Tensor parallel size for TRT-LLM. When ``tp_size * pp_size > 1``
            the HF reference model is sharded across the visible GPUs via
            ``device_map="auto"``.
        pp_size: Pipeline parallel size for TRT-LLM.

    Returns:
        A dict with comparison results.
    """
    from tensorrt_llm import LLM, SamplingParams
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_path = str(Path(checkpoint_path).resolve())
    layer_ids = sorted(set(layer_ids))

    if tp_size < 1 or pp_size < 1:
        raise ValueError(
            f"tp_size and pp_size must be >= 1 (got tp_size={tp_size}, pp_size={pp_size})."
        )
    world_size = tp_size * pp_size
    gpus_per_node = torch.cuda.device_count()
    num_nodes = int(os.environ.get('SLURM_NNODES', 1))
    available_gpus = gpus_per_node * num_nodes
    if world_size > available_gpus:
        raise RuntimeError(
            f"Requested tp_size*pp_size={world_size} but only {available_gpus} "
            f"CUDA device(s) are visible ({gpus_per_node} GPU(s) x {num_nodes} node(s)).")

    # --- Step 1: Dump partial checkpoint ---
    if output_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="partial_model_")
        partial_dir = tmp_dir
    else:
        partial_dir = str(Path(output_dir).resolve())
        tmp_dir = None

    print("=" * 60)
    print("Step 1: Dumping partial checkpoint")
    print("=" * 60)
    layer_id_map = dump_partial_model(
        checkpoint_path=checkpoint_path,
        layer_ids=layer_ids,
        output_dir=partial_dir,
    )

    try:
        # --- Step 2: Load HF model and get context logits ---
        print()
        print("=" * 60)
        print("Step 2: Loading HuggingFace model from partial checkpoint")
        print("=" * 60)

        tokenizer = AutoTokenizer.from_pretrained(partial_dir)
        seq_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        print(f"  Prompt: {prompt}")
        print(f"  Tokenized length: {seq_len}")
        print(f"  Parallelism: tp_size={tp_size}, pp_size={pp_size} "
              f"(world_size={world_size}, visible_gpus={available_gpus})")

        hf_kwargs = {}
        if torch_dtype is not None:
            hf_kwargs["torch_dtype"] = torch_dtype
        if world_size > 1:
            # Shard the reference model across the same GPUs TRT-LLM will use.
            hf_kwargs["device_map"] = "auto"
            hf_model = AutoModelForCausalLM.from_pretrained(
                partial_dir, **hf_kwargs).eval()
            hf_input_device = next(hf_model.parameters()).device
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(
                partial_dir, **hf_kwargs).cuda().eval()
            hf_input_device = torch.device("cuda")

        input_ids = tokenizer(prompt,
                              return_tensors="pt").input_ids.to(hf_input_device)

        with torch.no_grad():
            hf_out = hf_model(input_ids)
            hf_logits = hf_out.logits  # (1, seq_len, vocab_size)

        print(
            f"  HF logits shape: {list(hf_logits.shape)}, dtype: {hf_logits.dtype}"
        )

        # Free HF model memory before loading TRT-LLM
        del hf_model
        torch.cuda.empty_cache()

        # --- Step 3: Initialize TRT-LLM LLM and generate ---
        print()
        print("=" * 60)
        print("Step 3: Initializing TensorRT-LLM LLM instance")
        print("=" * 60)

        llm_kwargs = {"model": partial_dir, "backend": "pytorch"}
        if tp_size > 1:
            llm_kwargs["tensor_parallel_size"] = tp_size
        if pp_size > 1:
            llm_kwargs["pipeline_parallel_size"] = pp_size
        llm = LLM(**llm_kwargs)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            return_context_logits=True,
        )

        outputs = llm.generate([prompt], sampling_params)
        output = outputs[0]

        trtllm_context_logits = output.context_logits  # (seq_len, vocab_size)
        generated_text = output.outputs[0].text

        print(
            f"  TRT-LLM context logits shape: {list(trtllm_context_logits.shape)}, dtype: {trtllm_context_logits.dtype}"
        )
        print(f"  Generated text: {generated_text!r}")

        # Free TRT-LLM
        del llm
        torch.cuda.empty_cache()

        # --- Step 4: Compare context logits ---
        print()
        print("=" * 60)
        print("Step 4: Comparing context logits")
        print("=" * 60)

        # HF logits: (1, seq_len, vocab_size) -> (seq_len, vocab_size)
        hf_ctx = hf_logits.squeeze(0)
        trtllm_ctx = trtllm_context_logits

        # Ensure same device
        if hf_ctx.device != trtllm_ctx.device:
            trtllm_ctx = trtllm_ctx.to(hf_ctx.device)

        # Cast to same dtype for comparison
        if hf_ctx.dtype != trtllm_ctx.dtype:
            compare_dtype = torch.float32
            hf_cmp = hf_ctx.float()
            trtllm_cmp = trtllm_ctx.float()
        else:
            compare_dtype = hf_ctx.dtype
            hf_cmp = hf_ctx
            trtllm_cmp = trtllm_ctx

        # Align shapes if needed (truncate to shorter length)
        min_len = min(hf_cmp.shape[0], trtllm_cmp.shape[0])
        min_vocab = min(hf_cmp.shape[1], trtllm_cmp.shape[1])
        hf_cmp = hf_cmp[:min_len, :min_vocab]
        trtllm_cmp = trtllm_cmp[:min_len, :min_vocab]

        match = torch.allclose(hf_cmp, trtllm_cmp, atol=atol, rtol=rtol)
        max_diff = (hf_cmp - trtllm_cmp).abs().max().item()
        mean_diff = (hf_cmp - trtllm_cmp).abs().mean().item()

        result = {
            "match": match,
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "atol": atol,
            "rtol": rtol,
            "layer_ids": layer_ids,
            "layer_id_map": {
                str(k): v
                for k, v in layer_id_map.items()
            },
            "prompt": prompt,
            "generated_text": generated_text,
            "seq_len": seq_len,
            "hf_logits_shape": list(hf_logits.shape),
            "trtllm_context_logits_shape": list(trtllm_context_logits.shape),
            "hf_dtype": str(hf_logits.dtype),
            "trtllm_dtype": str(trtllm_context_logits.dtype),
            "compare_dtype": str(compare_dtype),
            "partial_checkpoint_dir": partial_dir,
            "tp_size": tp_size,
            "pp_size": pp_size,
            "world_size": world_size,
        }

        # Print summary
        status = "PASS" if match else "FAIL"
        print(f"  Status:               {status}")
        print(f"  Layer IDs:            {layer_ids}")
        print(
            f"  Parallelism:          tp={tp_size}, pp={pp_size}, world={world_size}"
        )
        print(f"  Prompt:               {prompt}")
        print(f"  Prompt tokens:        {seq_len}")
        print(
            f"  HF logits shape:      {list(hf_logits.shape)} (dtype: {hf_logits.dtype})"
        )
        print(
            f"  TRT-LLM logits shape: {list(trtllm_context_logits.shape)} (dtype: {trtllm_context_logits.dtype})"
        )
        print(f"  Max abs diff:         {max_diff:.6e}")
        print(f"  Mean abs diff:        {mean_diff:.6e}")
        print(f"  Tolerance:            atol={atol}, rtol={rtol}")
        print("=" * 60)

        return result

    finally:
        # Clean up temp directory if not keeping
        if tmp_dir and not keep_dump:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"\nCleaned up temporary checkpoint: {tmp_dir}")
        elif partial_dir:
            print(f"\nPartial checkpoint kept at: {partial_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=
        "Compare HF and TRT-LLM outputs using a dumped partial model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Examples:\n"
                "  python compare_partial_models.py \\\n"
                "      --checkpoint_path /models/Llama-2-7b-hf \\\n"
                "      --layer_ids 1,2,3,7,8,9\n"
                "\n"
                "  python compare_partial_models.py \\\n"
                "      --checkpoint_path /models/Llama-2-7b-hf \\\n"
                "      --layer_ids 0,15,31 \\\n"
                "      --prompt 'The capital of France is' \\\n"
                "      --atol 1e-3 --rtol 1e-3\n"
                "\n"
                "  python compare_partial_models.py \\\n"
                "      --checkpoint_path /models/Llama-2-7b-hf \\\n"
                "      --layer_ids 1,2,3,7,8,9 \\\n"
                "      --output_dir /tmp/partial-ckpt \\\n"
                "      --keep_dump\n"),
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Path to the full HuggingFace checkpoint directory.",
    )
    parser.add_argument(
        "--layer_ids",
        type=_parse_layer_ids,
        required=True,
        help="Comma-separated layer indices to keep (e.g., 1,2,3,7,8,9).",
    )
    parser.add_argument(
        "--prompt",
        default="Hello, my name is",
        help="Text prompt for generation (default: 'Hello, my name is').",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16,
        help="Max new tokens for TRT-LLM generation (default: 16).",
    )
    parser.add_argument(
        "--torch_dtype",
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype for HF model (default: inferred from config).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for torch.allclose (default: 1e-2).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for torch.allclose (default: 1e-2).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for the partial checkpoint (default: temp dir).",
    )
    parser.add_argument(
        "--keep_dump",
        action="store_true",
        help="Keep the dumped partial checkpoint after comparison.",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help=
        "Tensor parallel size for TRT-LLM (default: 1). When tp_size*pp_size>1 "
        "the HF reference model is sharded with device_map='auto'.",
    )
    parser.add_argument(
        "--pp_size",
        type=int,
        default=1,
        help="Pipeline parallel size for TRT-LLM (default: 1).",
    )
    parser.add_argument(
        "--report_file",
        default=None,
        help="Optional path to write JSON report to.",
    )

    args = parser.parse_args()

    dtype = None
    if args.torch_dtype:
        dtype = getattr(torch, args.torch_dtype)

    result = compare_partial_models(
        checkpoint_path=args.checkpoint_path,
        layer_ids=args.layer_ids,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        torch_dtype=dtype,
        atol=args.atol,
        rtol=args.rtol,
        output_dir=args.output_dir,
        keep_dump=args.keep_dump,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
    )

    if args.report_file:
        Path(args.report_file).write_text(json.dumps(result, indent=2))
        print(f"Report written to: {args.report_file}")

    sys.exit(0 if result["match"] else 1)


if __name__ == "__main__":
    main()
