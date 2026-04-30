<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary
-->

---
name: trtllm-env-check
description: >-
  Check the local execution environment for GPU availability, Docker support,
  and Slurm access. Returns the execution scenario (`satisfied, local, docker`,
  `satisfied, local, direct`, `satisfied, slurm, local`, or `not_satisfied`),
  the number of available GPUs, and the GPU type. For known clusters (lyris,
  prenyx, ptyche, bia), GPU type and gpus_per_node are resolved from the
  slurm-cluster-info skill without probing compute nodes via srun.
tags: [infrastructure, slurm, environment]
license: LicenseRef-NvidiaProprietary
metadata:
  author: NVIDIA Corporation
---

# TensorRT-LLM Environment Check

Detect whether the current machine can run a GPU workload locally (Docker) or via Slurm, and report hardware details.

## Input

| Field | Description | Required |
|-------|-------------|----------|
| `required_devices` | Minimum number of GPUs needed | Yes |
| `account` | Slurm account (unused for GPU probing, kept for compatibility) | No |

## Procedure

### 1. Check local GPUs

```bash
timeout 5 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
```

If this succeeds:
- Count the number of GPU lines → `available_gpus`
- Extract the GPU name from the first line → `device_type` (e.g., `NVIDIA B200`, `NVIDIA H100 80GB HBM3`)
- Normalize `device_type`: strip `NVIDIA ` prefix and trailing memory info to get the short name (e.g., `B200`, `H100`, `A100`, `L40S`, `RTX 6000`)

If `nvidia-smi` fails or returns no GPUs → `available_gpus = 0`, `device_type = null`.

### 2. Check if local GPUs are sufficient

If `available_gpus >= required_devices`, continue to step 2a to determine whether Docker is available on this host.

### 2a. Check Docker availability

```bash
command -v docker >/dev/null 2>&1 && timeout 3 docker info >/dev/null 2>&1
```

- If the command succeeds (Docker CLI exists **and** the daemon responds) → **Result**: `satisfied, local, docker`
- Otherwise (Docker CLI missing, daemon not running, or permission denied) → **Result**: `satisfied, local, direct`

Include `available_gpus` in the result. Do NOT include `device_type` — local execution does not need it.

### 3. Check Slurm availability

If local GPUs are insufficient (or as additional detection), check for Slurm:

```bash
which squeue 2>/dev/null && squeue --version 2>/dev/null
```

If Slurm is NOT available → go to step 5.

### 4. Resolve GPU type

When Slurm is available but local `nvidia-smi` returned no GPUs or `device_type` is null (login nodes typically have no GPUs), derive the cluster name from the hostname and look up the GPU type:

```bash
hostname -f 2>/dev/null || hostname
```

- Extract the cluster name from the hostname (e.g., `login-lyris.nvidia.com` → `lyris`, `draco-oci-iad-login-001` → `oci`, `login01.aioli.nvidia.com` → `aioli`)
- Invoke the `slurm-cluster-info` skill with the cluster name to resolve `device_type`, `gpus_per_node`, `default_models_repo`, and `default_user_root_dir`.
- Set `available_gpus` = `gpus_per_node` (if resolved).
- Set `cluster_name` = the extracted cluster name (e.g., `lyris`, `prenyx`). This allows the orchestrator to fetch the full cluster config (mounts, container_image, partition, etc.) from `slurm-cluster-info`.
- Set `default_user_root_dir` = the user root directory from the cluster reference (with `<user_name>` substituted with the actual SLURM username). Set to `null` if not found.
- If no match found → `device_type = null`, `cluster_name = null`, `default_user_root_dir = null`

**Result**: `satisfied, slurm, local` with `device_type`, `gpus_per_node`, `cluster_name`, and `default_user_root_dir`.

### 5. Not satisfied

If neither local GPUs nor Slurm is available:
- **Result**: `not_satisfied`

## Output

Return a single structured result:

```
scenario: <satisfied, local, docker | satisfied, local, direct | satisfied, slurm, local | not_satisfied>
available_gpus: <N>
device_type: <short GPU/device name or null>
gpus_per_node: <N or null>
cluster_name: <cluster name or null>
default_models_repo: <host path to llm-models directory, or null>
default_user_root_dir: <per-user root directory on cluster storage, or null>
```

**Examples:**

```
scenario: satisfied, local, docker
available_gpus: 4
```

```
scenario: satisfied, local, direct
available_gpus: 4
```

```
scenario: satisfied, slurm, local
available_gpus: 4
device_type: B200
gpus_per_node: 4
cluster_name: lyris
default_user_root_dir: /lustre/fsw/coreai_comparch_trtllm/myuser
```

```
scenario: not_satisfied
available_gpus: 0
device_type: null
cluster_name: null
```

## Rules

- Never install drivers or modify the system
- If `nvidia-smi` hangs, use a 5-second timeout: `timeout 5 nvidia-smi ...`
- GPU type is derived from the hostname by matching the cluster name via the `slurm-cluster-info` skill — no `srun` allocation needed
- Report the GPU type exactly as found in the mapping (do not guess or fabricate)
