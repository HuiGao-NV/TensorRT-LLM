#!/usr/bin/env python3
"""Instantiate a TensorRT-LLM model with only a subset of decoder layers.

This script creates a TRT-LLM model from a HuggingFace checkpoint, but only
instantiates and loads weights for the specified layer indices. This is useful
for memory-efficient module-level testing and debugging, where you don't need
the full model.

Usage:
    # As a CLI tool — instantiate layers 0 and 5, then run an interactive shell
    python instantiate_partial_model.py \
        --checkpoint_path /path/to/hf/model \
        --layer_ids 0,5 \
        --interactive

    # As a library
    from instantiate_partial_model import load_partial_model
    model, layer_id_map = load_partial_model(
        checkpoint_path="/path/to/hf/model",
        layer_ids=[0, 5],
    )
    # layer_id_map = {0: 0, 5: 1}  — original layer 5 is at model index 1
"""

import argparse
import json
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


def load_partial_model(
    checkpoint_path: str,
    layer_ids: List[int],
    device: str = "cuda",
) -> Tuple["DecoderModelForCausalLM", Dict[int, int]]:
    """Instantiate a TRT-LLM model with only the specified decoder layers.

    Args:
        checkpoint_path: Path to HuggingFace checkpoint directory.
        layer_ids: List of original layer indices to include (e.g., [0, 5, 31]).
        device: Device to place the model on.

    Returns:
        A tuple of (model, layer_id_map) where:
        - model is the instantiated DecoderModelForCausalLM with len(layer_ids) layers.
        - layer_id_map maps original layer index → new model layer index.

    Raises:
        ValueError: If any requested layer_id is out of range.
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
    from tensorrt_llm.mapping import Mapping

    checkpoint_path = str(Path(checkpoint_path).resolve())
    layer_ids = sorted(set(layer_ids))

    # --- Load config ---
    config = ModelConfig.from_pretrained(
        checkpoint_path,
        mapping=Mapping(tp_size=1, pp_size=1),
    )
    pretrained_config = config.pretrained_config
    original_num_layers = pretrained_config.num_hidden_layers

    # Validate layer IDs
    for lid in layer_ids:
        if lid < 0 or lid >= original_num_layers:
            raise ValueError(
                f"Layer ID {lid} is out of range [0, {original_num_layers}). "
                f"The model has {original_num_layers} layers."
            )

    # Build mapping: original layer index → new consecutive index
    layer_id_map = {orig: new for new, orig in enumerate(layer_ids)}
    num_partial_layers = len(layer_ids)

    # --- Override num_hidden_layers ---
    config._frozen = False
    pretrained_config.num_hidden_layers = num_partial_layers

    # Handle per-layer config lists (e.g., layer_types for sliding window)
    _slice_per_layer_configs(pretrained_config, layer_ids)

    config._frozen = True

    # --- Instantiate model with reduced layer count ---
    print(
        f"Instantiating model with {num_partial_layers} layer(s) "
        f"(original indices: {layer_ids}) out of {original_num_layers} total."
    )

    try:
        with MetaInitMode():
            model = AutoModelForCausalLM.from_config(config)
        # Move meta tensors to real device
        memo = {}

        def init_meta_tensor(t: torch.Tensor):
            if t.device != torch.device("meta"):
                return t
            if t not in memo:
                memo[t] = torch.empty_like(t, device=device)
            return memo[t]

        model._apply(init_meta_tensor)
        del memo
    except Exception:
        model = AutoModelForCausalLM.from_config(config)

    model.to(device)

    # --- Load and remap weights ---
    print("Loading checkpoint weights...")
    weights = _load_checkpoint_weights(checkpoint_path)

    print("Remapping layer weights...")
    remapped_weights = _remap_layer_weights(weights, layer_id_map)

    # Load weights into model. Try with allow_partial_loading first; some
    # models override load_weights with a custom signature that doesn't
    # accept this kwarg, so fall back to the basic call.
    import inspect

    load_sig = inspect.signature(model.load_weights)
    if "allow_partial_loading" in load_sig.parameters:
        model.load_weights(remapped_weights, allow_partial_loading=True)
    else:
        model.load_weights(remapped_weights)

    # Post-load hooks
    for module in model.modules():
        if hasattr(module, "post_load_weights") and not getattr(
            module, "_weights_removed", False
        ):
            module.post_load_weights()

    torch.cuda.current_stream().synchronize()

    print(
        f"Model ready: {num_partial_layers} layer(s) loaded on {device}. "
        f"Layer mapping: {layer_id_map}"
    )
    return model, layer_id_map


def _slice_per_layer_configs(pretrained_config, layer_ids: List[int]):
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
        value = getattr(pretrained_config, attr, None)
        if isinstance(value, (list, tuple)) and len(value) > max(layer_ids):
            sliced = [value[i] for i in layer_ids]
            setattr(pretrained_config, attr, type(value)(sliced))


def _load_checkpoint_weights(checkpoint_path: str) -> dict:
    """Load all weights from checkpoint into a flat dict on CPU."""
    import glob

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
        description="Instantiate a TRT-LLM model with only specified decoder layers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Load only layer 0 and layer 5\n"
            "  python instantiate_partial_model.py \\\n"
            "      --checkpoint_path /models/Llama-2-7b-hf \\\n"
            "      --layer_ids 0,5\n"
            "\n"
            "  # Load first layer and drop into interactive Python shell\n"
            "  python instantiate_partial_model.py \\\n"
            "      --checkpoint_path /models/Llama-2-7b-hf \\\n"
            "      --layer_ids 0 \\\n"
            "      --interactive\n"
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
        "--interactive",
        action="store_true",
        help="Drop into an interactive Python shell after loading.",
    )

    args = parser.parse_args()

    model, layer_id_map = load_partial_model(
        checkpoint_path=args.checkpoint_path,
        layer_ids=args.layer_ids,
        device=args.device,
    )

    if args.interactive:
        print("\nVariables available: model, layer_id_map")
        print("Access decoder layers via: model.model.layers[new_idx]")
        import code

        code.interact(local={**globals(), **locals()})


if __name__ == "__main__":
    main()
