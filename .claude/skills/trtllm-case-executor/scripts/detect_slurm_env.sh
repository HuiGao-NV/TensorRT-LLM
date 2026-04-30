#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Detect SLURM account(s) and available partitions for the current user.
#
# Usage:
#   bash detect_slurm_env.sh                  # JSON output to stdout
#   bash detect_slurm_env.sh --format=text    # human-readable text
#
# Output (JSON, default):
#   {
#     "user": "<whoami>",
#     "default_account": "<first-account-or-null>",
#     "accounts": ["<acct1>", "<acct2>", ...],
#     "default_partition": "<partition-marked-with-*-or-null>",
#     "partitions": [
#       {"name": "<part>", "state": "<up|down|...>", "time_limit": "<HH:MM:SS|infinite>", "nodes": "<count>"},
#       ...
#     ],
#     "errors": ["<diagnostic>", ...]
#   }
#
# Exit codes:
#   0  success (account and at least one partition found)
#   1  SLURM tools not available
#   2  partial success (one of account/partition could not be resolved)

set -u
set -o pipefail

FORMAT="json"
for arg in "$@"; do
    case "$arg" in
        --format=json) FORMAT="json" ;;
        --format=text) FORMAT="text" ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

ERRORS=()

if ! command -v sacctmgr >/dev/null 2>&1 || ! command -v sinfo >/dev/null 2>&1; then
    ERRORS+=("SLURM client tools (sacctmgr/sinfo) not found in PATH")
    if [ "$FORMAT" = "json" ]; then
        printf '{"user":"%s","default_account":null,"accounts":[],"default_partition":null,"partitions":[],"errors":["SLURM client tools not found"]}\n' "$(whoami)"
    else
        echo "ERROR: SLURM client tools (sacctmgr / sinfo) not found in PATH" >&2
    fi
    exit 1
fi

USER_NAME="$(whoami)"

# --- Accounts -----------------------------------------------------------------
# `sacctmgr -nP show assoc where user=$USER format=account` returns one
# account per line. Some sites duplicate entries (per-cluster, per-partition);
# de-duplicate while preserving order.
ACCOUNTS_RAW="$(sacctmgr -nP show assoc where user="$USER_NAME" format=account 2>/dev/null || true)"
ACCOUNTS=()
if [ -n "$ACCOUNTS_RAW" ]; then
    while IFS= read -r line; do
        line="${line//[[:space:]]/}"
        [ -z "$line" ] && continue
        # de-dupe
        local_seen=0
        for a in "${ACCOUNTS[@]:-}"; do
            [ "$a" = "$line" ] && local_seen=1 && break
        done
        [ "$local_seen" -eq 0 ] && ACCOUNTS+=("$line")
    done <<< "$ACCOUNTS_RAW"
fi
DEFAULT_ACCOUNT="${ACCOUNTS[0]:-}"
if [ -z "$DEFAULT_ACCOUNT" ]; then
    ERRORS+=("no SLURM account associations for user $USER_NAME")
fi

# --- Partitions ---------------------------------------------------------------
# `sinfo -h -o "%P|%a|%l|%D"` — partition (default has a trailing '*'),
# avail (up/down), time limit, node count.
DEFAULT_PARTITION=""
PART_LINES=()
PART_RAW="$(sinfo -h -o '%P|%a|%l|%D' 2>/dev/null || true)"
if [ -n "$PART_RAW" ]; then
    # de-dupe by partition name (sinfo can emit one line per state group)
    declare -A SEEN_PART
    while IFS='|' read -r pname pavail ptime pnodes; do
        pname_trim="${pname%[*]}"   # strip default '*'
        is_default=0
        [ "$pname" != "$pname_trim" ] && is_default=1
        [ -z "$pname_trim" ] && continue
        if [ -z "${SEEN_PART[$pname_trim]:-}" ]; then
            SEEN_PART["$pname_trim"]=1
            PART_LINES+=("$pname_trim|$pavail|$ptime|$pnodes")
            [ "$is_default" -eq 1 ] && [ -z "$DEFAULT_PARTITION" ] && DEFAULT_PARTITION="$pname_trim"
        fi
    done <<< "$PART_RAW"
fi
if [ "${#PART_LINES[@]}" -eq 0 ]; then
    ERRORS+=("sinfo returned no partitions")
fi
# If sinfo did not flag a default, fall back to the first listed partition.
if [ -z "$DEFAULT_PARTITION" ] && [ "${#PART_LINES[@]}" -gt 0 ]; then
    first="${PART_LINES[0]}"
    DEFAULT_PARTITION="${first%%|*}"
fi

# --- Emit ---------------------------------------------------------------------
emit_json() {
    # Build accounts JSON array
    local acc_json="[]"
    if [ "${#ACCOUNTS[@]}" -gt 0 ]; then
        acc_json="["
        local first=1
        for a in "${ACCOUNTS[@]}"; do
            if [ "$first" -eq 1 ]; then first=0; else acc_json+=","; fi
            acc_json+="\"$a\""
        done
        acc_json+="]"
    fi
    # Build partitions JSON array
    local part_json="[]"
    if [ "${#PART_LINES[@]}" -gt 0 ]; then
        part_json="["
        local first=1
        for line in "${PART_LINES[@]}"; do
            IFS='|' read -r p_name p_avail p_time p_nodes <<< "$line"
            if [ "$first" -eq 1 ]; then first=0; else part_json+=","; fi
            part_json+="{\"name\":\"$p_name\",\"state\":\"$p_avail\",\"time_limit\":\"$p_time\",\"nodes\":\"$p_nodes\"}"
        done
        part_json+="]"
    fi
    # Build errors JSON array
    local err_json="[]"
    if [ "${#ERRORS[@]}" -gt 0 ]; then
        err_json="["
        local first=1
        for e in "${ERRORS[@]}"; do
            if [ "$first" -eq 1 ]; then first=0; else err_json+=","; fi
            # escape quotes & backslashes
            esc="${e//\\/\\\\}"
            esc="${esc//\"/\\\"}"
            err_json+="\"$esc\""
        done
        err_json+="]"
    fi
    local def_acc_json="null"
    [ -n "$DEFAULT_ACCOUNT" ] && def_acc_json="\"$DEFAULT_ACCOUNT\""
    local def_part_json="null"
    [ -n "$DEFAULT_PARTITION" ] && def_part_json="\"$DEFAULT_PARTITION\""
    printf '{"user":"%s","default_account":%s,"accounts":%s,"default_partition":%s,"partitions":%s,"errors":%s}\n' \
        "$USER_NAME" "$def_acc_json" "$acc_json" "$def_part_json" "$part_json" "$err_json"
}

emit_text() {
    echo "user: $USER_NAME"
    echo "default_account: ${DEFAULT_ACCOUNT:-<none>}"
    echo "accounts: ${ACCOUNTS[*]:-<none>}"
    echo "default_partition: ${DEFAULT_PARTITION:-<none>}"
    echo "partitions:"
    if [ "${#PART_LINES[@]}" -gt 0 ]; then
        printf '  %-20s %-6s %-12s %s\n' NAME STATE TIME_LIMIT NODES
        for line in "${PART_LINES[@]}"; do
            IFS='|' read -r p_name p_avail p_time p_nodes <<< "$line"
            printf '  %-20s %-6s %-12s %s\n' "$p_name" "$p_avail" "$p_time" "$p_nodes"
        done
    else
        echo "  <none>"
    fi
    if [ "${#ERRORS[@]}" -gt 0 ]; then
        echo "errors:"
        for e in "${ERRORS[@]}"; do echo "  - $e"; done
    fi
}

if [ "$FORMAT" = "json" ]; then
    emit_json
else
    emit_text
fi

# Exit code
if [ -z "$DEFAULT_ACCOUNT" ] || [ -z "$DEFAULT_PARTITION" ]; then
    exit 2
fi
exit 0
