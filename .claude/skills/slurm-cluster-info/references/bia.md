<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary
-->

# Bia Cluster Reference

## GPU Board

| Component | Detail |
|-----------|--------|
| GPU Board | Umbriel B300 SXM7 |
| GPUs per node | 8x B300 SXM7 |
| **gpus_per_node** | **8** |
| GPU Memory | 2.3 TB total per node |
| CPU arch | x86_64 |

## Multi-Node GPU Communication

- NVLink5 connects GPUs **within a single node only** (8 GPUs per NVL8 domain)
- **MNNVL (Multi-Node NVLink) is NOT supported on Bia.** Multi-node GPU communication uses SpectrumX (RoCE) only.
- 800 Gb/s total bandwidth per GPU for cross-node communication

## Constraints

- Available features: `cleaf[01-02]-[01-02]` (compute leaf group), `sleaf[01-02]` (storage leaf), `esvt` (ESVT hardware), `ts6` (TS6 hardware revision)
- Use `--constraint=<feature>` to target specific hardware

```bash
# Check partition configuration
scontrol -a show partition <partition>

# List available nodes with features
sinfo -p <partition> -t idle,alloc -o "%15P|%55b|%10t|%N"
```

## Storage

| Filesystem | Type | Path | Notes |
|------------|------|------|-------|
| Home | NFS (Isilon) | `/home/<username>` | Small quota |
| Project | Lustre (DDN Exascaler) | `/project` | Recommended working directory |
| Fast scratch | Lustre (DDN Exascaler) | `/lustre/fsw/coreai_comparch_trtllm/<user_name>` | High-speed parallel I/O |

- Lustre is available on all compute nodes

## User Root directory
`/lustre/fsw/coreai_comparch_trtllm/<user_name>` (not `/home`)

## Model Repository

| Path | Notes |
|------|-------|
| `/lustre/fsw/coreai_comparch_trtllm/llm-models` | Default shared model weights directory |
