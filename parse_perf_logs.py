#!/usr/bin/env python3
"""Parse perf-sanity logs to extract memory profiling fractions and map to config files."""

import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "perf-sanity-logs")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "case_fraction_map.txt")
AGG_CONFIG_FOLDER = os.path.join(SCRIPT_DIR, "tests/scripts/perf-sanity/aggregated")
DISAGG_CONFIG_FOLDER = os.path.join(SCRIPT_DIR, "tests/scripts/perf-sanity/disaggregated")

# Regex for the Peak memory line — try RANK 0 first, fall back to any line without RANK tag
PEAK_RE_RANK0 = re.compile(
    r"\[RANK 0\].*Peak memory during memory usage profiling \(torch \+ non-torch\):\s*([\d.]+)\s*GiB,\s*"
    r"available KV cache memory when calculating max tokens:\s*([\d.]+)\s*GiB,\s*"
    r"fraction is set\s*([\d.]+),\s*"
    r"kv size per token is\s*[\d.]+\.\s*"
    r"device total memory\s*([\d.]+)\s*GiB,\s*"
    r"temporary kv cache memory during profiling\s*([\d.]+)\s*GiB"
)
PEAK_RE_ANY = re.compile(
    r"Peak memory during memory usage profiling \(torch \+ non-torch\):\s*([\d.]+)\s*GiB,\s*"
    r"available KV cache memory when calculating max tokens:\s*([\d.]+)\s*GiB,\s*"
    r"fraction is set\s*([\d.]+),\s*"
    r"kv size per token is\s*[\d.]+\.\s*"
    r"device total memory\s*([\d.]+)\s*GiB,\s*"
    r"temporary kv cache memory during profiling\s*([\d.]+)\s*GiB"
)

# Regex for Memory used after loading model weights — try RANK 0 first, fall back to any
WEIGHTS_INSIDE_RE_RANK0 = re.compile(
    r"\[RANK 0\].*Memory used after loading model weights \(inside torch\) in memory usage profiling:\s*([\d.]+)\s*GiB"
)
WEIGHTS_OUTSIDE_RE_RANK0 = re.compile(
    r"\[RANK 0\].*Memory used after loading model weights \(outside torch\) in memory usage profiling:\s*([\d.]+)\s*GiB"
)
WEIGHTS_INSIDE_RE_ANY = re.compile(
    r"Memory used after loading model weights \(inside torch\) in memory usage profiling:\s*([\d.]+)\s*GiB"
)
WEIGHTS_OUTSIDE_RE_ANY = re.compile(
    r"Memory used after loading model weights \(outside torch\) in memory usage profiling:\s*([\d.]+)\s*GiB"
)


def parse_log(log_path):
    """Parse a log file and return memory metrics dict or None."""
    if not os.path.isfile(log_path):
        return None

    with open(log_path, "r", errors="replace") as f:
        content = f.read()

    peak_match = PEAK_RE_RANK0.search(content) or PEAK_RE_ANY.search(content)
    inside_match = WEIGHTS_INSIDE_RE_RANK0.search(content) or WEIGHTS_INSIDE_RE_ANY.search(content)
    outside_match = WEIGHTS_OUTSIDE_RE_RANK0.search(content) or WEIGHTS_OUTSIDE_RE_ANY.search(content)

    if not peak_match:
        return None

    peak_memory = float(peak_match.group(1))
    kv_cache_memory = float(peak_match.group(2))
    fraction_set = float(peak_match.group(3))
    device_total_memory = float(peak_match.group(4))
    temporary_kv = float(peak_match.group(5))

    denominator = device_total_memory - peak_memory + kv_cache_memory
    computed_fraction = kv_cache_memory / denominator if denominator != 0 else 0.0

    weights_inside = float(inside_match.group(1)) if inside_match else None
    weights_outside = float(outside_match.group(1)) if outside_match else None

    return {
        "peak_memory": peak_memory,
        "kv_cache_memory": kv_cache_memory,
        "fraction_set": fraction_set,
        "device_total_memory": device_total_memory,
        "temporary_kv": temporary_kv,
        "weights_inside_torch": weights_inside,
        "weights_outside_torch": weights_outside,
        "computed_fraction": computed_fraction,
    }


def parse_test_case_name(subfolder):
    """Parse test case subfolder name to extract config_base, benchmark_mode, runtime_mode.

    Follows the logic in jenkins/scripts/perf/local/submit.py::parse_test_string().

    Returns:
        (config_base_name, benchmark_mode, runtime_mode)
        - benchmark_mode: "e2e", "gen_only", "ctx_only", or None (normal aggr)
        - runtime_mode: "aggregated" or "disaggregated"
    """
    labels = subfolder.split("-")
    prefix = labels[0]
    is_disagg_prefix = "disagg" in prefix
    is_aggr_prefix = "aggr" in prefix

    if is_disagg_prefix:
        # disagg_upload-{e2e|gen_only}-{config_base}
        if len(labels) < 3:
            return None, None, None
        benchmark_mode = labels[1]  # e2e or gen_only
        config_base_name = "-".join(labels[2:])
        return config_base_name, benchmark_mode, "disaggregated"
    elif is_aggr_prefix:
        if len(labels) > 2 and labels[1] == "ctx_only":
            # aggr_upload-ctx_only-{config_base}
            config_base_name = "-".join(labels[2:])
            return config_base_name, "ctx_only", "aggregated"
        else:
            # aggr_upload-{config_yml}-{server_config_name}
            config_base_name = labels[1] if len(labels) > 1 else None
            return config_base_name, None, "aggregated"
    return None, None, None


def resolve_config_path(config_base_name, benchmark_mode):
    """Resolve the config YAML file path from the config base name and benchmark mode.

    For disagg/ctx_only/e2e/gen_only: look in disaggregated config folder.
    For normal aggr: look in aggregated config folder.

    Returns:
        Relative config path (from repo root) or "NOT_FOUND" if missing.
    """
    if config_base_name is None:
        return "NOT_FOUND"

    if benchmark_mode in ("e2e", "gen_only", "ctx_only"):
        config_dir = DISAGG_CONFIG_FOLDER
        rel_dir = "tests/scripts/perf-sanity/disaggregated"
    else:
        config_dir = AGG_CONFIG_FOLDER
        rel_dir = "tests/scripts/perf-sanity/aggregated"

    config_path = os.path.join(config_dir, f"{config_base_name}.yaml")
    if os.path.exists(config_path):
        return os.path.join(rel_dir, f"{config_base_name}.yaml")
    return "NOT_FOUND"


def format_result(archive_folder, test_type, subfolder, server_label, metrics, config_file):
    """Format a single result line."""
    if metrics is None:
        return (
            f"{archive_folder} | {test_type} | {server_label} | {subfolder} | "
            f"config_file={config_file} | DATA_MISSING"
        )

    m = metrics
    wi = f"{m['weights_inside_torch']:.2f} GiB" if m['weights_inside_torch'] is not None else "N/A"
    wo = f"{m['weights_outside_torch']:.2f} GiB" if m['weights_outside_torch'] is not None else "N/A"
    return (
        f"{archive_folder} | {test_type} | {server_label} | {subfolder} | "
        f"config_file={config_file} | "
        f"fraction_set={m['fraction_set']} | "
        f"computed_fraction={m['computed_fraction']:.6f} | "
        f"peak_memory={m['peak_memory']:.2f} GiB | "
        f"kv_cache_memory={m['kv_cache_memory']:.2f} GiB | "
        f"device_total_memory={m['device_total_memory']:.2f} GiB | "
        f"temporary_kv={m['temporary_kv']:.2f} GiB | "
        f"weights_inside_torch={wi} | "
        f"weights_outside_torch={wo}"
    )


def main():
    results = []

    # Each tar.gz extracts to: results-X/inner_dir/case_subfolders/
    for top_entry in sorted(os.listdir(LOGS_DIR)):
        top_path = os.path.join(LOGS_DIR, top_entry)
        if not os.path.isdir(top_path) or not top_entry.startswith("results-"):
            continue

        # Find the inner directory (usually one level deep)
        inner_dirs = [
            d for d in os.listdir(top_path)
            if os.path.isdir(os.path.join(top_path, d))
        ]
        if not inner_dirs:
            continue

        for inner_dir in inner_dirs:
            inner_path = os.path.join(top_path, inner_dir)

            for subfolder in sorted(os.listdir(inner_path)):
                subfolder_path = os.path.join(inner_path, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                # Resolve config file for this test case
                config_base, benchmark_mode, runtime_mode = parse_test_case_name(subfolder)
                config_file = resolve_config_path(config_base, benchmark_mode)

                if subfolder.startswith("disagg"):
                    test_type = "disagg"
                    ctx_log = os.path.join(subfolder_path, "ctx_server_0.log")
                    gen_log = os.path.join(subfolder_path, "gen_server_0.log")
                    ctx_metrics = parse_log(ctx_log)
                    gen_metrics = parse_log(gen_log)
                    results.append(format_result(top_entry, test_type, subfolder, "ctx_server_0", ctx_metrics, config_file))
                    results.append(format_result(top_entry, test_type, subfolder, "gen_server_0", gen_metrics, config_file))
                elif subfolder.startswith("aggr"):
                    test_type = "aggr"
                    log_file = os.path.join(subfolder_path, "trtllm-serve.0.log")
                    metrics = parse_log(log_file)
                    results.append(format_result(top_entry, test_type, subfolder, "trtllm-serve.0", metrics, config_file))
                else:
                    continue

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(results) + "\n")

    print(f"Wrote {len(results)} entries to {OUTPUT_FILE}")
    data_found = sum(1 for r in results if "DATA_MISSING" not in r)
    data_missing = len(results) - data_found
    print(f"  Data found: {data_found}, Data missing: {data_missing}")
    not_found = sum(1 for r in results if "NOT_FOUND" in r)
    print(f"  Config NOT_FOUND: {not_found}")


if __name__ == "__main__":
    main()
