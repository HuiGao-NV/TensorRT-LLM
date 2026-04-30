#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
"""Benchmark a CUDA C++ extension model against the original PyTorch model.

Standalone script -- outputs structured JSON to stdout.

Contract:
    The workspace directory must contain:
    - ``model.py``: original PyTorch model with ``Model``, ``get_inputs()``,
      ``get_init_inputs()``
    - ``model_new.py``: optimized model with ``ModelNew`` using compiled
      ``cuda_extension``
    - ``cuda_extension.so``: compiled CUDA extension (or available via
      build directory)

Usage:
    python benchmark_model.py /path/to/workspace [--warmup 5] [--iters 20] \
        [--timeout 120] [--mock]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap


def _build_benchmark_script(
    workspace_dir: str,
    warmup: int,
    iters: int,
) -> str:
    """Generate a benchmark harness script that runs inside the workspace."""
    return textwrap.dedent(f"""\
        import sys
        import os
        import torch

        os.chdir({workspace_dir!r})
        sys.path.insert(0, {workspace_dir!r})

        # Also add build directory to path for cuda_extension import
        build_dir = os.path.join({workspace_dir!r}, "build", "forced_compile")
        if os.path.isdir(build_dir):
            sys.path.insert(0, build_dir)

        from model import Model, get_inputs, get_init_inputs
        from model_new import ModelNew

        def gpu_benchmark(fn, inputs, warmup, iters):
            with torch.no_grad():
                for _ in range(warmup):
                    fn(*inputs)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iters):
                    fn(*inputs)
                end.record()
                torch.cuda.synchronize()
            return start.elapsed_time(end) / iters

        def clone_inputs(inputs):
            return [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]

        # Initialize models
        init_inputs = get_init_inputs()
        if not isinstance(init_inputs, (list, tuple)):
            init_inputs = [init_inputs]
        torch_model = Model(*init_inputs).eval().cuda()
        cuda_model = ModelNew(*init_inputs).eval().cuda()
        cuda_model.load_state_dict(torch_model.state_dict())

        # Get inputs
        inputs = get_inputs()
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

        # Benchmark
        cuda_ms = gpu_benchmark(
            cuda_model, clone_inputs(inputs),
            warmup={warmup}, iters={iters},
        )
        torch_ms = gpu_benchmark(
            torch_model, clone_inputs(inputs),
            warmup={warmup}, iters={iters},
        )

        speedup = torch_ms / cuda_ms if cuda_ms > 0 else 0
        print(
            f"BENCHMARK:kernel_ms={{cuda_ms:.6f}},"
            f"ref_ms={{torch_ms:.6f}},"
            f"speedup={{speedup:.4f}}"
        )
    """)


def _find_file(name: str, search_root: str) -> str | None:
    """Recursively search for a file under search_root."""
    from pathlib import Path

    matches = list(Path(search_root).rglob(name))
    return str(matches[0]) if matches else None


def benchmark_model(
    workspace_dir: str,
    warmup: int = 5,
    iters: int = 20,
    timeout: int = 120,
    search_root: str | None = None,
) -> dict:
    """Benchmark CUDA extension model performance.

    Args:
        workspace_dir: Path to workspace containing model.py and build
            infrastructure.
        warmup: Number of warmup iterations.
        iters: Number of measured iterations.
        timeout: Execution timeout in seconds.
        search_root: Optional root to search for model_new.py if not found
            in workspace_dir.  When the agent writes to an artifact
            subdirectory, this finds it automatically.

    Returns:
        Dict with ``kernel_time_ms``, ``reference_time_ms``, ``speedup``.
    """
    workspace_dir = os.path.abspath(workspace_dir)

    # model.py must be in workspace_dir (provided by eval setup)
    model_path = os.path.join(workspace_dir, "model.py")
    if not os.path.exists(model_path):
        print(f"Required file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # model_new.py: check workspace_dir first, then search_root
    model_new_path = os.path.join(workspace_dir, "model_new.py")
    if not os.path.exists(model_new_path) and search_root:
        found = _find_file("model_new.py", search_root)
        if found:
            # Copy model_new.py and any compiled extension to workspace_dir
            import shutil

            shutil.copy2(found, model_new_path)
            # Copy all sibling files so JIT compilation works
            # (without-skill agents use torch.utils.cpp_extension.load
            # which needs .cu/.cpp sources alongside model_new.py)
            found_dir = os.path.dirname(found)
            for item in os.listdir(found_dir):
                if item == "model_new.py":
                    continue
                src = os.path.join(found_dir, item)
                dst = os.path.join(workspace_dir, item)
                if os.path.exists(dst):
                    continue
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

    if not os.path.exists(model_new_path):
        print(f"Required file not found: {model_new_path}", file=sys.stderr)
        sys.exit(1)

    script = _build_benchmark_script(workspace_dir, warmup, iters)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                     delete=False) as script_file:
        script_file.write(script)
        script_path = script_file.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workspace_dir,
        )

        output = result.stdout + result.stderr

        if "BENCHMARK:" not in output:
            print(f"Benchmark failed:\n{output[:2000]}", file=sys.stderr)
            sys.exit(1)

        result_line = next(line for line in output.split("\n")
                           if "BENCHMARK:" in line)
        parts = result_line.split("BENCHMARK:")[1].split(",")
        try:
            parsed = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in parts}
        except (IndexError, ValueError) as exc:
            return {
                "kernel_time_ms": None,
                "error": f"Failed to parse BENCHMARK line: {exc}",
            }

        result_dict = {"kernel_time_ms": parsed["kernel_ms"]}
        if "ref_ms" in parsed:
            result_dict["reference_time_ms"] = parsed["ref_ms"]
            result_dict["speedup"] = parsed["speedup"]

        return result_dict

    except subprocess.TimeoutExpired:
        print(f"Benchmark timed out after {timeout} seconds", file=sys.stderr)
        sys.exit(1)
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)


def _mock_data() -> dict:
    """Return realistic mock benchmark data for testing."""
    return {
        "kernel_time_ms": 1.23,
        "reference_time_ms": 2.45,
        "speedup": 1.99,
    }


def main() -> None:
    """Entry point for CLI invocation."""
    parser = argparse.ArgumentParser(
        description="Benchmark a CUDA extension model.")
    parser.add_argument(
        "workspace_dir",
        nargs="?",
        help="Path to workspace with model.py, model_new.py, cuda_extension.",
    )
    parser.add_argument("--warmup",
                        type=int,
                        default=5,
                        help="Warmup iterations (default: 5).")
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Measured iterations (default: 20).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Execution timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--search-root",
        default=None,
        help="Root directory to search for model_new.py if not in workspace.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Return mock data for testing (no GPU required).",
    )
    args = parser.parse_args()

    if args.mock:
        data = _mock_data()
    elif args.workspace_dir:
        data = benchmark_model(
            workspace_dir=args.workspace_dir,
            warmup=args.warmup,
            iters=args.iters,
            timeout=args.timeout,
            search_root=args.search_root,
        )
    else:
        parser.error("Either --mock or workspace_dir is required.")

    json.dump(data, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
