<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary
-->

# Lyris Cluster Reference

## GPU Board

| Property | Lyris (GB200-NVL72) | Theia (GB300-NVL72) |
|----------|---------------------|---------------------|
| GPU | B200 Blackwell | B300 Blackwell |
| GPU memory | 192 GB HBM3e per GPU | 288 GB HBM3e per GPU |
| GPUs per compute tray | 4 (2 per Bianca board) | 4 (2 per Bianca board) |
| **gpus_per_node** | **4** | **4** |
| GPUs per rack | 72 (18 compute trays) | 72 (18 compute trays) |
| Total GPUs | 1,152 B200 | 1,152 B300 |
| CPU arch | arm | x86_64 |

## Multi-Node GPU Communication

- **NVLink5** connects GPUs within a rack: 72 GPUs fully connected (MNNVL domain = 1 rack = 18 nodes)
- Up to 1,800 GB/s per GPU bandwidth
- **SLURM block = 1 NVL domain = 1 rack = 18 nodes = 72 GPUs**
- Cross-rack communication uses InfiniBand (not NVLink)

## Constraints

- **NVL block:** `--constraint=nvlblk01` through `nvlblk32` (01-16 = GB200, 17-32 = GB300)
- **IB leaf:** `--constraint=ibcleaf01-01` (POD-leaf format)
- **Hardware revision (GB300 only):** `--constraint=ts2`, `--constraint=cr`, `--constraint=ts3`
- Query available features: `sinfo -p <partition> -t idle,alloc -o "%15P|%55b|%10t|%N"`
- Query partition config: `scontrol -a show partition <partition>`

## Storage

| Filesystem | Type | Purpose | Path |
|------------|------|---------|------|
| Lustre (DDN EXAScaler) | Parallel | Fast read/write for workloads | `/lustre/fsw/coreai_comparch_trtllm/<user_name>` |
| NFS (Isilon) | Network | Home directories | `/home/<user>` |

- Lustre is available on all compute nodes

## User Root directory
`/lustre/fsw/coreai_comparch_trtllm/<user_name>` (not `/home`)

## Model Repository

| Path | Notes |
|------|-------|
| `/lustre/fsw/coreai_comparch_trtllm/llm-models` | Default shared model weights directory |
