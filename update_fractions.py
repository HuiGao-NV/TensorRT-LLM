#!/usr/bin/env python3
"""Update free_gpu_memory_fraction in perf-sanity config files using computed fractions."""

import math
import os
import re
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRACTION_MAP = os.path.join(SCRIPT_DIR, "case_fraction_map.txt")
AGG_CONFIG_FOLDER = os.path.join(SCRIPT_DIR, "tests/scripts/perf-sanity/aggregated")
DISAGG_CONFIG_FOLDER = os.path.join(SCRIPT_DIR, "tests/scripts/perf-sanity/disaggregated")

FRAC_RE = re.compile(r"^(\s*free_gpu_memory_fraction:\s*)[\d.]+(.*)$")


def parse_fraction_map():
    """Parse case_fraction_map_1.txt and build update plan.

    Returns:
        disagg_updates: {abs_config_path: {"ctx": [fractions], "gen": [fractions]}}
        aggr_updates: {abs_config_path: {server_name: [fractions]}}
    """
    disagg_updates = defaultdict(lambda: defaultdict(list))
    aggr_updates = defaultdict(lambda: defaultdict(list))

    with open(FRACTION_MAP) as f:
        for line in f:
            line = line.strip()
            if not line or "DATA_MISSING" in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            server_label = parts[2]
            subfolder = parts[3]

            kv = {}
            for p in parts[4:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    kv[k.strip()] = v.strip()

            config_rel = kv.get("config_file", "")
            computed = float(kv.get("computed_fraction", "0"))
            if not config_rel or config_rel == "NOT_FOUND":
                continue

            config_abs = os.path.join(SCRIPT_DIR, config_rel)

            labels = subfolder.split("-")
            prefix = labels[0]

            if "disagg" in prefix:
                # disagg_upload-gen_only-{config_base} or disagg_upload-e2e-{config_base}
                section = "ctx" if server_label == "ctx_server_0" else "gen"
                disagg_updates[config_abs][section].append(computed)
            elif "aggr" in prefix:
                if len(labels) > 2 and labels[1] == "ctx_only":
                    # aggr_upload-ctx_only-{config_base} → updates ctx in disagg config
                    disagg_updates[config_abs]["ctx"].append(computed)
                else:
                    # aggr_upload-{config_yml}-{server_name}
                    server_name = "-".join(labels[2:]) if len(labels) > 2 else None
                    if server_name:
                        aggr_updates[config_abs][server_name].append(computed)

    return disagg_updates, aggr_updates


def avg_floor2(values):
    """Average values and floor to 2 decimal places."""
    avg = sum(values) / len(values)
    return math.floor(avg * 100) / 100.0


def update_disagg_config(config_path, section_fractions):
    """Update a disagg config file's free_gpu_memory_fraction for ctx and/or gen sections.

    Uses line scanning to find sections and targeted replacement.
    """
    with open(config_path, "r") as f:
        lines = f.readlines()

    # Parse to know the section order under worker_config
    current_section = None  # "gen" or "ctx"
    in_worker_config = False
    updates = []  # list of (line_index, new_fraction)

    for i, line in enumerate(lines):
        stripped = line.rstrip()

        # Detect worker_config: at top level
        if stripped == "worker_config:":
            in_worker_config = True
            continue

        if in_worker_config:
            # Detect gen: or ctx: as direct children of worker_config (2-space indent)
            if re.match(r"^  gen:", stripped):
                current_section = "gen"
                continue
            elif re.match(r"^  ctx:", stripped):
                current_section = "ctx"
                continue
            # A top-level key (no indent) exits worker_config
            if stripped and not stripped.startswith(" ") and stripped.endswith(":"):
                in_worker_config = False
                current_section = None
                continue

        if current_section and current_section in section_fractions:
            m = FRAC_RE.match(stripped)
            if m:
                new_frac = avg_floor2(section_fractions[current_section])
                updates.append((i, m.group(1), new_frac, m.group(2)))

    if not updates:
        return False

    for line_idx, prefix, new_frac, suffix in updates:
        lines[line_idx] = f"{prefix}{new_frac}{suffix}\n"

    with open(config_path, "w") as f:
        f.writelines(lines)
    return True


def update_aggr_config(config_path, server_fractions):
    """Update an aggr config file's free_gpu_memory_fraction per server_config name.

    Uses line scanning to track current server_config name and targeted replacement.
    """
    with open(config_path, "r") as f:
        lines = f.readlines()

    current_server_name = None
    in_server_configs = False
    updates = []

    for i, line in enumerate(lines):
        stripped = line.rstrip()

        if stripped == "server_configs:":
            in_server_configs = True
            continue

        if in_server_configs:
            # A top-level key exits server_configs
            if stripped and not stripped.startswith(" ") and not stripped.startswith("#"):
                in_server_configs = False
                current_server_name = None
                continue

            # Detect server_config name (- name: "xxx" or name: xxx)
            name_match = re.match(r'^\s+-\s+name:\s*["\']?([^"\']+)["\']?\s*$', stripped)
            if name_match:
                current_server_name = name_match.group(1).strip()
                continue

            if current_server_name and current_server_name in server_fractions:
                m = FRAC_RE.match(stripped)
                if m:
                    new_frac = avg_floor2(server_fractions[current_server_name])
                    updates.append((i, m.group(1), new_frac, m.group(2)))

    if not updates:
        return False

    for line_idx, prefix, new_frac, suffix in updates:
        lines[line_idx] = f"{prefix}{new_frac}{suffix}\n"

    with open(config_path, "w") as f:
        f.writelines(lines)
    return True


def main():
    disagg_updates, aggr_updates = parse_fraction_map()

    print(f"Disagg configs to update: {len(disagg_updates)}")
    print(f"Aggr configs to update: {len(aggr_updates)}")

    total_updated = 0
    total_skipped = 0

    for config_path, section_fractions in sorted(disagg_updates.items()):
        if not os.path.exists(config_path):
            print(f"  SKIP (missing): {config_path}")
            total_skipped += 1
            continue

        avg_fracs = {s: avg_floor2(v) for s, v in section_fractions.items()}
        rel_path = os.path.relpath(config_path, SCRIPT_DIR)
        if update_disagg_config(config_path, section_fractions):
            print(f"  UPDATED: {rel_path} -> {avg_fracs}")
            total_updated += 1
        else:
            print(f"  NO CHANGE: {rel_path}")

    for config_path, server_fractions in sorted(aggr_updates.items()):
        if not os.path.exists(config_path):
            print(f"  SKIP (missing): {config_path}")
            total_skipped += 1
            continue

        avg_fracs = {s: avg_floor2(v) for s, v in server_fractions.items()}
        rel_path = os.path.relpath(config_path, SCRIPT_DIR)
        if update_aggr_config(config_path, server_fractions):
            print(f"  UPDATED: {rel_path} -> {avg_fracs}")
            total_updated += 1
        else:
            print(f"  NO CHANGE: {rel_path}")

    print(f"\nTotal updated: {total_updated}, skipped: {total_skipped}")


if __name__ == "__main__":
    main()
