#!/usr/bin/env python3
"""Parse sync_manifest.txt and emit git or scp section paths."""
import re
import sys

section = sys.argv[1]  # 'git' or 'scp'
manifest_path = sys.argv[2]

target_header = f"[{section}]"
in_section = False
with open(manifest_path) as f:
    for line in f:
        s = line.rstrip("\n")
        if s == target_header:
            in_section = True
            continue
        if s.startswith("[") and s != target_header:
            in_section = False
            continue
        if in_section and s and not s.startswith("#"):
            print(re.sub(r"^[0-9]+\t", "", s))
