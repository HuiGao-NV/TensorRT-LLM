#!/usr/bin/env python3
"""Dump a new HuggingFace checkpoint containing only the specified decoder layers.

Reads an existing HuggingFace checkpoint, extracts weights for the requested
layer IDs (remapping them to consecutive indices), updates the config, and
saves everything as a new standalone checkpoint directory. The output can be
loaded directly with ``AutoModelForCausalLM.from_pretrained(output_dir)``.

Usage:
    python dump_hf_partial_model.py \
        --checkpoint_path /models/Llama-2-7b-hf \
        --layer_ids 1,2,3,7,8,9 \
        --output_dir /tmp/llama-partial

    python dump_hf_partial_model.py \
        --checkpoint_path /models/Llama-2-7b-hf \
        --layer_ids 0,15,31 \
        --output_dir /tmp/llama-3layers \
        --save_format safetensors
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import torch


def _parse_layer_ids(value: str) -> List[int]:
    """Parse a comma-separated string of layer IDs into a list of ints."""
    try:
        return [int(x.strip()) for x in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid layer_ids format: '{value}'. Expected comma-separated integers (e.g., 0,5,31)."
        )


# Regex to match layer index patterns like ".layers.5." or ".layers.31."
_LAYER_PATTERN = re.compile(r"(\.layers\.)(\d+)(\.)")


def dump_partial_model(
    checkpoint_path: str,
    layer_ids: List[int],
    output_dir: str,
    save_format: str = "safetensors",
) -> Dict[int, int]:
    """Extract selected layers from a checkpoint and save as a new checkpoint.

    Args:
        checkpoint_path: Path to the source HuggingFace checkpoint directory.
        layer_ids: Original layer indices to keep (e.g., [1, 2, 3, 7, 8, 9]).
        output_dir: Directory to write the new partial checkpoint into.
        save_format: Weight format — "safetensors" (default) or "bin".

    Returns:
        The layer_id_map: original layer index -> new consecutive index.

    Raises:
        ValueError: If any layer_id is out of range.
        RuntimeError: If no weight files are found in the source checkpoint.
    """
    checkpoint_path = str(Path(checkpoint_path).resolve())
    output_dir = str(Path(output_dir).resolve())
    layer_ids = sorted(set(layer_ids))

    # --- Load and validate config ---
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found in {checkpoint_path}")

    with open(config_path) as f:
        config = json.load(f)

    original_num_layers = config.get("num_hidden_layers")
    if original_num_layers is None:
        raise ValueError("config.json does not contain 'num_hidden_layers'.")

    for lid in layer_ids:
        if lid < 0 or lid >= original_num_layers:
            raise ValueError(
                f"Layer ID {lid} is out of range [0, {original_num_layers}). "
                f"The model has {original_num_layers} layers."
            )

    layer_id_map = {orig: new for new, orig in enumerate(layer_ids)}
    num_partial_layers = len(layer_ids)

    print(
        f"Extracting {num_partial_layers} layer(s) "
        f"(original indices: {layer_ids}) from {original_num_layers} total."
    )

    # --- Load weights ---
    print("Loading checkpoint weights...")
    weights = _load_checkpoint_weights(checkpoint_path)

    # --- Remap layer weights ---
    print("Remapping layer weights...")
    remapped = _remap_layer_weights(weights, layer_id_map)
    del weights

    # --- Prepare output directory ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Save weights ---
    weight_file: str
    if save_format == "safetensors":
        import safetensors.torch

        weight_file = os.path.join(output_dir, "model.safetensors")
        print(f"Saving weights to {weight_file} ...")
        safetensors.torch.save_file(remapped, weight_file)
    else:
        weight_file = os.path.join(output_dir, "pytorch_model.bin")
        print(f"Saving weights to {weight_file} ...")
        torch.save(remapped, weight_file)

    del remapped

    # --- Update and save config ---
    config["num_hidden_layers"] = num_partial_layers

    # Slice per-layer config lists
    _PER_LAYER_ATTRS = [
        "layer_types",
        "head_dim_list",
        "num_attention_heads_list",
        "num_key_value_heads_list",
    ]
    for attr in _PER_LAYER_ATTRS:
        if attr in config and isinstance(config[attr], list):
            if len(config[attr]) >= original_num_layers:
                config[attr] = [config[attr][i] for i in layer_ids]

    out_config_path = os.path.join(output_dir, "config.json")
    with open(out_config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {out_config_path}")

    # --- Copy tokenizer and other auxiliary files ---
    _copy_auxiliary_files(checkpoint_path, output_dir)

    # --- Summary ---
    total_params = sum(t.numel() for t in torch.load(weight_file, weights_only=True, map_location="cpu").values()) if save_format == "bin" else _count_safetensors_params(weight_file)
    print()
    print("=" * 60)
    print("DUMP COMPLETE")
    print("=" * 60)
    print(f"  Output dir:       {output_dir}")
    print(f"  Layers kept:      {layer_ids}")
    print(f"  Layer mapping:    {layer_id_map}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Format:           {save_format}")
    print("=" * 60)

    return layer_id_map


def _count_safetensors_params(path: str) -> int:
    """Count total parameters in a safetensors file without loading tensors."""
    import safetensors

    with safetensors.safe_open(path, framework="pt") as f:
        return sum(
            1
            for _ in range(0)  # placeholder
        ) or sum(
            f.get_tensor(key).numel() for key in f.keys()
        )


def _load_checkpoint_weights(checkpoint_path: str) -> dict:
    """Load all weights from checkpoint into a flat dict on CPU."""
    import safetensors.torch

    weight_files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
    weight_files = [
        f for f in weight_files if "consolidated" not in os.path.basename(f)
    ]

    if weight_files:
        weights = {}
        for wf in weight_files:
            weights.update(safetensors.torch.load_file(wf))
        return weights

    bin_files = glob.glob(os.path.join(checkpoint_path, "*.bin"))
    if not bin_files:
        bin_files = glob.glob(os.path.join(checkpoint_path, "*.pth"))

    if bin_files:
        weights = {}
        for bf in bin_files:
            weights.update(
                torch.load(bf, weights_only=True, map_location="cpu")
            )
        return weights

    raise RuntimeError(f"No weight files found in {checkpoint_path}")


def _remap_layer_weights(weights: dict, layer_id_map: Dict[int, int]) -> dict:
    """Remap checkpoint weight keys so that only selected layers are included."""
    remapped = {}
    kept_layers = set(layer_id_map.keys())

    for key, value in weights.items():
        match = _LAYER_PATTERN.search(key)
        if match:
            original_idx = int(match.group(2))
            if original_idx not in kept_layers:
                continue
            new_idx = layer_id_map[original_idx]
            new_key = (
                key[: match.start()]
                + match.group(1)
                + str(new_idx)
                + match.group(3)
                + key[match.end() :]
            )
            remapped[new_key] = value
        else:
            remapped[key] = value

    return remapped


def _copy_auxiliary_files(src_dir: str, dst_dir: str):
    """Copy tokenizer and other non-weight files needed for from_pretrained."""
    auxiliary_patterns = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "preprocessor_config.json",
        "chat_template.json",
    ]
    copied = []
    for pattern in auxiliary_patterns:
        src = os.path.join(src_dir, pattern)
        if os.path.isfile(src):
            dst = os.path.join(dst_dir, pattern)
            if not os.path.isfile(dst):
                shutil.copy2(src, dst)
                copied.append(pattern)

    # Also copy any .tiktoken or sentencepiece files
    for ext in ("*.tiktoken", "*.spm"):
        for src_file in glob.glob(os.path.join(src_dir, ext)):
            name = os.path.basename(src_file)
            dst = os.path.join(dst_dir, name)
            if not os.path.isfile(dst):
                shutil.copy2(src_file, dst)
                copied.append(name)

    if copied:
        print(f"Copied auxiliary files: {', '.join(copied)}")


def main():
    parser = argparse.ArgumentParser(
        description="Dump a new HuggingFace checkpoint with only specified decoder layers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python dump_hf_partial_model.py \\\n"
            "      --checkpoint_path /models/Llama-2-7b-hf \\\n"
            "      --layer_ids 1,2,3,7,8,9 \\\n"
            "      --output_dir /tmp/llama-partial\n"
            "\n"
            "  python dump_hf_partial_model.py \\\n"
            "      --checkpoint_path /models/Llama-2-7b-hf \\\n"
            "      --layer_ids 0,15,31 \\\n"
            "      --output_dir /tmp/llama-3layers \\\n"
            "      --save_format bin\n"
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Path to source HuggingFace checkpoint directory.",
    )
    parser.add_argument(
        "--layer_ids",
        type=_parse_layer_ids,
        required=True,
        help="Comma-separated layer indices to keep (e.g., 1,2,3,7,8,9).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write the partial checkpoint to.",
    )
    parser.add_argument(
        "--save_format",
        default="safetensors",
        choices=["safetensors", "bin"],
        help="Weight file format (default: safetensors).",
    )

    args = parser.parse_args()

    dump_partial_model(
        checkpoint_path=args.checkpoint_path,
        layer_ids=args.layer_ids,
        output_dir=args.output_dir,
        save_format=args.save_format,
    )


if __name__ == "__main__":
    main()
