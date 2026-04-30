#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
"""Integrate a CuDeepy-generated kernel into a workload file.

Standalone script -- only Python stdlib dependency.
Modifies a workload to use a CuDeepy-generated kernel via import or
monkey-patching.

Usage:
    python integrate_kernel.py --workload train.py --kernel kernel.py \
        --operator gelu --method import|patch [--mock]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_import_end(content: str) -> int:
    """Find the byte offset after the last top-level import statement.

    Args:
        content: Python source file contents.

    Returns:
        Byte offset after the last import line, or 0 if no imports found.
    """
    import_end = 0
    for match in re.finditer(r"^(?:import|from)\s+.*$", content, re.MULTILINE):
        import_end = max(import_end, match.end())
    return import_end


# ---------------------------------------------------------------------------
# Integration methods
# ---------------------------------------------------------------------------


def integrate_import(
    code: str,
    import_path: str,
    operator_name: str,
) -> tuple[str, list[str]]:
    """Add an import statement for the generated kernel.

    Inserts an import statement after the last existing import in the file.

    Args:
        code: Source file contents.
        import_path: Dot-separated import path to the kernel module.
        operator_name: Name of the operator being replaced.

    Returns:
        Tuple of (modified code, list of change descriptions).
    """
    changes: list[str] = []
    import_stmt = f"from {import_path}.kernel import optimized_{operator_name}"

    import_end = _find_import_end(code)

    if import_end > 0:
        new_code = code[:import_end] + "\n" + import_stmt + code[import_end:]
    else:
        new_code = import_stmt + "\n\n" + code

    changes.append(f"Added import: {import_stmt}")
    return new_code, changes


def integrate_patch(
    code: str,
    kernel_dir: str,
    operator_name: str,
) -> tuple[str, list[str]]:
    """Add monkey-patching code to use the generated kernel.

    Prepends sys.path manipulation and import at the beginning of the file.

    Args:
        code: Source file contents.
        kernel_dir: Absolute path to the kernel directory.
        operator_name: Name of the operator being replaced.

    Returns:
        Tuple of (modified code, list of change descriptions).
    """
    changes: list[str] = []
    patch_code = (f"\n# CuDeepy kernel integration (auto-generated)\n"
                  f"import sys\n"
                  f"sys.path.insert(0, '{kernel_dir}')\n"
                  f"from kernel import optimized_{operator_name}\n"
                  f"# Patch the original operator\n")
    new_code = patch_code + code
    changes.append(f"Added monkey-patch for {operator_name}")
    return new_code, changes


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------


def _mock_metadata(
    workload_path: str,
    kernel_path: str,
    operator_name: str,
    method: str,
) -> dict:
    """Return realistic mock metadata for testing without real integration."""
    if method == "import":
        changes = [
            f"Added import: from generated.kernel import optimized_{operator_name}"
        ]
    else:
        changes = [f"Added monkey-patch for {operator_name}"]

    return {
        "success": True,
        "workload_path": workload_path,
        "changes": changes,
        "method": method,
        "backup_path": workload_path + ".bak",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrate a CuDeepy-generated kernel into a workload.")
    parser.add_argument(
        "--workload",
        required=True,
        help="Path to the workload Python file to modify.",
    )
    parser.add_argument(
        "--kernel",
        required=True,
        help="Path to the generated kernel.py file.",
    )
    parser.add_argument(
        "--operator",
        required=True,
        help="Name of the operator being replaced.",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["import", "patch"],
        help="Integration method: import (add import) or patch (monkey-patch).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Return mock metadata for testing without real integration.",
    )
    args = parser.parse_args()

    workload_path = os.path.abspath(args.workload)
    kernel_path = os.path.abspath(args.kernel)

    # --- Mock mode --------------------------------------------------------
    if args.mock:
        meta = _mock_metadata(workload_path, kernel_path, args.operator,
                              args.method)
        json.dump(meta, sys.stdout, indent=2)
        print()
        return

    # --- Validate inputs --------------------------------------------------
    if not os.path.exists(workload_path):
        result = {
            "success": False,
            "workload_path": workload_path,
            "error": f"Workload file not found: {workload_path}",
        }
        json.dump(result, sys.stdout, indent=2)
        print()
        return

    if not os.path.exists(kernel_path):
        result = {
            "success": False,
            "workload_path": workload_path,
            "error": f"Kernel file not found: {kernel_path}",
        }
        json.dump(result, sys.stdout, indent=2)
        print()
        return

    # --- Create backup ----------------------------------------------------
    backup_path = workload_path + ".bak"
    shutil.copy2(workload_path, backup_path)

    # --- Read workload ----------------------------------------------------
    with open(workload_path) as f:
        code = f.read()

    # --- Calculate import path --------------------------------------------
    kernel_dir = os.path.dirname(kernel_path)
    workload_dir = os.path.dirname(workload_path)

    try:
        rel_path = os.path.relpath(kernel_dir, workload_dir)
        import_path = rel_path.replace(os.sep, ".")
    except ValueError:
        import_path = kernel_dir.replace(os.sep, ".")

    # --- Integrate --------------------------------------------------------
    if args.method == "import":
        new_code, changes = integrate_import(code, import_path, args.operator)
    else:
        new_code, changes = integrate_patch(code, kernel_dir, args.operator)

    # --- Write modified workload ------------------------------------------
    with open(workload_path, "w") as f:
        f.write(new_code)

    # --- Output result ----------------------------------------------------
    result = {
        "success": True,
        "workload_path": workload_path,
        "changes": changes,
        "method": args.method,
        "backup_path": backup_path,
    }
    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
