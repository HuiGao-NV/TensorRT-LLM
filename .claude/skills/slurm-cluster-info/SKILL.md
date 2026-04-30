<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary
-->

---
name: slurm-cluster-info
description: >-
  Look up internal SLURM cluster hardware, partitions, constraints, storage,
  and GPU topology. Use when you need to know which cluster has a specific GPU
  type, what partitions are available, how NVLink/InfiniBand topology works,
  or how to configure a cluster connection. Triggers on: "which cluster",
  "cluster info", "partition", "GPU type", "GB200", "B200", "B300", "GB300",
  "lyris", "prenyx", "ptyche", "bia", "NVLink topology", "cluster storage".
tags: [infrastructure, slurm, cluster]
license: LicenseRef-NvidiaProprietary
metadata:
  author: NVIDIA Corporation
---

# SLURM Cluster Info

Quick-reference skill for internal NVIDIA SLURM cluster details. Use this
skill when you need to answer questions about cluster hardware, partitions,
constraints, storage, or GPU topology — or when routing a workload to the
right cluster based on GPU requirements.

## When to Use

| Question | Action |
|----------|--------|
| Which cluster has GB200 / B200 / B300 / GB300? | Read `references/device_cluster_mapping.md` |
| What partitions does Lyris have? | Read `references/lyris.md` |
| How does NVLink topology work on Pre-Tyche? | Read `references/ptyche.md` |
| What storage paths should I use? | Read the cluster-specific reference |
| What constraint syntax do I use? | Read `references/pre-cluster-common.md` + cluster reference |
| How do I set up a new cluster connection? | Read `references/slurm-cluster-template.md` |

## Procedure

### Step 1: Identify the Cluster

If the user asks about a **specific cluster by name** (e.g., "lyris", "prenyx"),
go directly to the matching reference file in `references/`. → **single-cluster mode**.

If the user asks about a **GPU type or hardware** (e.g., "GB200", "B300 SXM7"),
read `references/device_cluster_mapping.md` to find the matching cluster, then
read that cluster's reference file. (Informal lookup for human questions.)

If a caller skill (e.g., `trtllm-remote-slurm-executor`) passes **hardware
constraints** (`device_type` + `required_devices_per_node` [+ optional
`total_required_devices`]) and expects a single resolved cluster back, use
**constraint-based mode** (see Output section). This skill — not the caller —
owns the filter rules.

### Step 2: Read Cluster Reference

Each cluster file contains:
- **Authentication**: MFA-style login flag (true = SSH requires interactive MFA / one-time-password / Duo prompt; false = key-based or passwordless SSH is acceptable)
- **GPU Board**: GPU model, memory, GPUs per node/rack
- **Multi-Node GPU Communication**: NVLink vs InfiniBand topology, MNNVL support
- **Constraints**: SLURM `--constraint` syntax for topology-aware scheduling
- **Storage**: Filesystem types, mount paths, recommended working directories
- **Model Repository**: path to the shared model weights directory — surfaced in the output template as `default_models_repo`
- **User Root directory**: the per-user working root on cluster storage (e.g., `/lustre/fsw/coreai_comparch_trtllm/<user_name>`) — surfaced in the output template as `default_user_root_dir`. Extract from the `## User Root directory` section; substitute `<user_name>` with the actual SLURM username when returning the value.


## Output (Returned Value)

This skill supports three modes:

- **Single-cluster mode** — invoked with a `cluster` name. Returns the output template below for that one cluster (plus the supplementary fields).
- **Catalog mode** — invoked without a cluster name (or with `mode: catalog`). Returns a list of entries, one per cluster discovered under `references/`. Each entry conforms to the output template. Order matches `references/device_cluster_mapping.md`.
- **Constraint-based mode** — invoked with hardware constraints (`device_type`, `required_devices_per_node`, optional `total_required_devices`). Returns a single resolved cluster entry, or stops and asks the user when no cluster qualifies. This is the entry point used by `trtllm-remote-slurm-executor` for auto-selection — that skill does not implement the filter rules itself.

### Output Template

Every mode's returned entry conforms to this template. These are the **core fields** every caller can rely on; do not rename them at the call site. Set any field to `null` if it cannot be resolved.

```yaml
cluster_name:          <cluster identifier, e.g. lyris | prenyx | ptyche | bia | oci>
device_type:           <short GPU/device name, e.g. B200 | B300>
default_user_root_dir: <per-user working root on cluster storage, e.g. /lustre/fsw/coreai_comparch_trtllm/<user_name>>
default_models_repo:   <host path to the shared llm-models directory>
gpus_per_node:         <integer; default GPUs per node on this cluster>
```

| Field | Source in cluster reference | Notes |
|-------|------------------------------|-------|
| `cluster_name` | filename / `# <Cluster> Cluster Reference` heading | The cluster identifier |
| `device_type` | `## GPU Board` table — `GPU` row | Strip vendor prefix; return short name (e.g., `B200`) |
| `default_user_root_dir` | `## Storage` / `## User Root directory` section | Per-user working root; substitute `<user_name>` with the actual SLURM user. Callers may override per-job. |
| `default_models_repo` | `## Storage` / `## Model Repository` section | Default location of the shared llm-models directory. Callers may override per-job. |
| `gpus_per_node` | `## GPU Board` table — `gpus_per_node` row | Integer |

### Single-cluster mode

When invoked with a cluster name, return the output template fields **plus** the two supplementary fields below. Each value is extracted from the matching cluster reference file in `references/<cluster>.md`. Set any field to `null` if it cannot be resolved.

```yaml
# Output template (core fields)
cluster_name:          <e.g. lyris | prenyx | ptyche | bia | oci | null>
device_type:           <e.g. B200 | B300 | null>
default_user_root_dir: <path or null>
default_models_repo:   <path or null>
gpus_per_node:         <N or null>

# Supplementary fields
mfa_style:   <true | false | null>
config_path: <relative path to the cluster connection config, or null>
```

| Supplementary field | Source in cluster reference | Notes |
|---------------------|------------------------------|-------|
| `mfa_style` | `## Authentication` table — `MFA-style login` row | `true` means SSH requires interactive MFA (e.g., a one-time password / Duo prompt) and key-only / non-interactive login is not sufficient; `false` means key-based or passwordless SSH is acceptable |
| `config_path` | `references/slurm-cluster-<cluster>.md` if it exists, else `null` | Connection config consumed by `trtllm-remote-slurm-executor` |

If no cluster matches the requested name, return all fields as `null` (and `cluster_name` as `null`).

### Catalog mode

When invoked without a cluster name, enumerate every cluster reference file in `references/` (excluding `device_cluster_mapping.md`, `pre-cluster-common.md`, `slurm-cluster-template.md`, and any `slurm-cluster-*.md` connection configs). Return a list ordered by `references/device_cluster_mapping.md` (clusters not listed there appear afterwards in alphabetical order). Each entry has exactly the same fields as single-cluster mode.

```yaml
clusters:
  - cluster_name: lyris
    device_type: B200
    default_user_root_dir: /lustre/fsw/coreai_comparch_trtllm/<user_name>
    default_models_repo: /lustre/fsw/coreai_comparch_trtllm/<user_name>/llm-models
    gpus_per_node: 4
    mfa_style: true
    config_path: references/slurm-cluster-lyris.md
  - cluster_name: ptyche
    ...
```

If a `references/<cluster>.md` documents two hardware variants in one file (e.g., Lyris hosts both B200 and B300), emit one catalog entry per variant. Use the variant-specific `device_type` and `gpus_per_node`.

### Constraint-based mode

When invoked with hardware constraints, return exactly one resolved cluster entry (same field shape as single-cluster mode), or stop and ask the user when nothing qualifies.

**Inputs:**

| Input | Description |
|-------|-------------|
| `device_type` | Required short GPU/device name (e.g., `B200`, `B300`). Hardware product strings like `GB200-NVL72` are not used here. |
| `required_devices_per_node` | Minimum GPUs per node the workload needs. |
| `total_required_devices` (optional) | Total GPU processes across all nodes. Not used as a filter unless a cluster declares an explicit total-GPU ceiling. |

**Procedure:**

1. Run catalog mode internally to enumerate every cluster (including per-variant entries).
2. Filter by `entry.device_type == device_type` (exact short-name match).
3. From the survivors, keep entries with `entry.gpus_per_node >= required_devices_per_node`.
4. Return the **first** surviving entry in catalog order (`device_cluster_mapping.md` order). The returned entry has the same field shape as single-cluster mode (including `config_path`).
5. If no entry survives, **stop and ask the user** which cluster to use. List the candidates that matched `device_type` so the user can see why the per-node filter failed. Do not silently fall back.

This is the only place where the filter rules live. Other skills must call this mode rather than re-implementing the filter.

## Reference Files

| File | Content |
|------|---------|
| `references/device_cluster_mapping.md` | Hardware type → cluster routing table |
| `references/pre-cluster-common.md` | Shared ASE pre-cluster patterns (constraints, GPU clocks, container save, power, IMEX) |
| `references/lyris.md` | Lyris cluster: GB200-NVL72 and GB300-NVL72 (Theia) |
| `references/prenyx.md` | Pre-Nyx cluster: B200 SXM6 |
| `references/ptyche.md` | Pre-Tyche cluster: GB200-NVL36 |
| `references/bia.md` | Bia cluster: B300 SXM7 |
| `references/slurm-cluster-template.md` | Template for creating cluster connection configs |
| `references/slurm-cluster-lyris.md` | Example cluster connection config |
