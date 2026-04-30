#!/usr/bin/env python3
"""Instantiate a HuggingFace model with only a subset of decoder layers.

This script creates a HuggingFace transformers model from a checkpoint, but
only instantiates and loads weights for the specified layer indices. This is
useful for memory-efficient module-level testing and debugging, where you
don't need the full model.

Usage:
    # As a CLI tool — instantiate layers 0 and 5
    python instantiate_hf_partial_model.py \
        --checkpoint_path /path/to/hf/model \
        --layer_ids 0,5

    # As a library
    from instantiate_hf_partial_model import load_hf_partial_model
    model, layer_id_map = load_hf_partial_model(
        checkpoint_path="/path/to/hf/model",
        layer_ids=[0, 5],
    )
    # layer_id_map = {0: 0, 5: 1}  — original layer 5 is at model index 1
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def _parse_layer_ids(value: str) -> List[int]:
    """Parse a comma-separated string of layer IDs into a list of ints."""
    try:
        return [int(x.strip()) for x in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid layer_ids format: '{value}'. Expected comma-separated integers (e.g., 0,5,31)."
        )


def load_hf_partial_model(
    checkpoint_path: str,
    layer_ids: List[int],
    device: str = "cuda",
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple["PreTrainedModel", Dict[int, int]]:
    """Instantiate a HuggingFace model with only the specified decoder layers.

    Args:
        checkpoint_path: Path to HuggingFace checkpoint directory.
        layer_ids: List of original layer indices to include (e.g., [0, 5, 31]).
        device: Device to place the model on.
        torch_dtype: Model dtype. If None, inferred from checkpoint config.

    Returns:
        A tuple of (model, layer_id_map) where:
        - model is the instantiated HuggingFace model with len(layer_ids) layers.
        - layer_id_map maps original layer index -> new model layer index.

    Raises:
        ValueError: If any requested layer_id is out of range.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    checkpoint_path = str(Path(checkpoint_path).resolve())
    layer_ids = sorted(set(layer_ids))

    # --- Load config ---
    config = AutoConfig.from_pretrained(checkpoint_path)
    original_num_layers = config.num_hidden_layers

    # Validate layer IDs
    for lid in layer_ids:
        if lid < 0 or lid >= original_num_layers:
            raise ValueError(
                f"Layer ID {lid} is out of range [0, {original_num_layers}). "
                f"The model has {original_num_layers} layers."
            )

    # Build mapping: original layer index -> new consecutive index
    layer_id_map = {orig: new for new, orig in enumerate(layer_ids)}
    num_partial_layers = len(layer_ids)

    # --- Override num_hidden_layers ---
    config.num_hidden_layers = num_partial_layers

    # Handle per-layer config lists (e.g., layer_types for sliding window)
    _slice_per_layer_configs(config, layer_ids)

    # --- Resolve dtype ---
    if torch_dtype is None:
        config_dtype = getattr(config, "torch_dtype", None)
        if isinstance(config_dtype, str) and config_dtype != "auto":
            torch_dtype = getattr(torch, config_dtype, torch.float32)
        elif isinstance(config_dtype, torch.dtype):
            torch_dtype = config_dtype
        else:
            torch_dtype = torch.float32

    # --- Instantiate model with reduced layer count ---
    print(
        f"Instantiating HF model with {num_partial_layers} layer(s) "
        f"(original indices: {layer_ids}) out of {original_num_layers} total."
    )

    try:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch_dtype,
            )
        # Materialize meta tensors to real device
        model.to_empty(device=device)
    except Exception:
        # Fallback: instantiate directly on device without meta init
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch_dtype,
            )
        model.to(device)

    # --- Load and remap weights ---
    print("Loading checkpoint weights...")
    weights = _load_checkpoint_weights(checkpoint_path)

    print("Remapping layer weights...")
    remapped_weights = _remap_layer_weights(weights, layer_id_map)

    # Load weights into model
    missing_keys, unexpected_keys = model.load_state_dict(
        remapped_weights, strict=False
    )

    if missing_keys:
        # Filter out expected missing keys (tied weights like lm_head.weight)
        truly_missing = [
            k for k in missing_keys if not _is_tied_weight_key(k)
        ]
        if truly_missing:
            print(
                f"Warning: {len(truly_missing)} missing weight key(s): "
                f"{truly_missing[:5]}{'...' if len(truly_missing) > 5 else ''}"
            )

    if unexpected_keys:
        print(
            f"Warning: {len(unexpected_keys)} unexpected weight key(s): "
            f"{unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}"
        )

    # Re-establish tied weights (e.g., lm_head.weight -> embed_tokens.weight)
    model.tie_weights()

    print(
        f"Model ready: {num_partial_layers} layer(s) loaded on {device}. "
        f"Layer mapping: {layer_id_map}"
    )
    return model, layer_id_map


def _is_tied_weight_key(key: str) -> bool:
    """Check if a weight key is commonly tied (and thus expected to be missing)."""
    tied_patterns = ["lm_head.weight"]
    return any(pattern in key for pattern in tied_patterns)


def _slice_per_layer_configs(config, layer_ids: List[int]):
    """Slice any per-layer config lists to match the selected layers.

    Some models have per-layer attributes like `layer_types` (list of length
    num_hidden_layers). When we reduce the layer count, these must be sliced
    to match the selected layer indices.
    """
    per_layer_attrs = [
        "layer_types",
        "head_dim_list",
        "num_attention_heads_list",
        "num_key_value_heads_list",
    ]
    for attr in per_layer_attrs:
        value = getattr(config, attr, None)
        if isinstance(value, (list, tuple)) and len(value) > max(layer_ids):
            sliced = [value[i] for i in layer_ids]
            setattr(config, attr, type(value)(sliced))


def _load_checkpoint_weights(checkpoint_path: str) -> dict:
    """Load all weights from checkpoint into a flat dict on CPU."""
    import safetensors.torch

    weight_files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
    # Filter out consolidated files
    weight_files = [
        f for f in weight_files if "consolidated" not in os.path.basename(f)
    ]

    if weight_files:
        weights = {}
        for wf in weight_files:
            weights.update(safetensors.torch.load_file(wf))
        return weights

    # Fallback to .bin / .pth
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


# Regex to match layer index patterns like ".layers.5." or ".layers.31."
_LAYER_PATTERN = re.compile(r"(\.layers\.)(\d+)(\.)")


def _remap_layer_weights(
    weights: dict, layer_id_map: Dict[int, int]
) -> dict:
    """Remap checkpoint weight keys so that only selected layers are included.

    For each weight key containing a layer index (e.g., "model.layers.5.attn.weight"),
    keep it only if layer 5 is in layer_id_map, and rename it to the new index
    (e.g., "model.layers.1.attn.weight" if layer 5 maps to index 1).

    Non-layer weights (embeddings, final norm, lm_head) are passed through as-is.
    """
    remapped = {}
    kept_layers = set(layer_id_map.keys())

    for key, value in weights.items():
        match = _LAYER_PATTERN.search(key)
        if match:
            original_idx = int(match.group(2))
            if original_idx not in kept_layers:
                continue  # Skip layers we don't need
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
            # Non-layer weight (embedding, norm, lm_head, etc.)
            remapped[key] = value

    return remapped


def main():
    parser = argparse.ArgumentParser(
        description="Instantiate a HuggingFace model with only specified decoder layers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Load only layer 0 and layer 5\n"
            "  python instantiate_hf_partial_model.py \\\n"
            "      --checkpoint_path /models/Llama-2-7b-hf \\\n"
            "      --layer_ids 0,5\n"
            "\n"
            "  # Load first layer on CPU with float32\n"
            "  python instantiate_hf_partial_model.py \\\n"
            "      --checkpoint_path /models/Llama-2-7b-hf \\\n"
            "      --layer_ids 0 \\\n"
            "      --device cpu \\\n"
            "      --torch_dtype float32\n"
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Path to HuggingFace checkpoint directory.",
    )
    parser.add_argument(
        "--layer_ids",
        type=_parse_layer_ids,
        required=True,
        help="Comma-separated original layer indices to instantiate (e.g., 0,5,31).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to place model on (default: cuda).",
    )
    parser.add_argument(
        "--torch_dtype",
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: inferred from config).",
    )
    args = parser.parse_args()

    dtype = None
    if args.torch_dtype:
        dtype = getattr(torch, args.torch_dtype)

    model, layer_id_map = load_hf_partial_model(
        checkpoint_path=args.checkpoint_path,
        layer_ids=args.layer_ids,
        device=args.device,
        torch_dtype=dtype,
    )

    print(f"\nModel type: {type(model).__name__}")
    print(f"Layer ID map: {layer_id_map}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
