<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary
-->

# Pre-Nyx Cluster Reference

## GPU Board

| Component | Detail |
|-----------|--------|
| GPU Board | Umbriel B200 SXM6 (PG525) |
| GPUs per node | 8x NVIDIA B200 |
| **gpus_per_node** | **8** |
| GPU Memory | 180 GB HBM3e per GPU |
| CPU arch | x86_64 |

## Multi-Node GPU Communication

- **MNNVL (Multi-Node NVLink) is NOT supported on Pre-Nyx.** Multi-node GPU communication uses InfiniBand only.
- NVLink5 connects GPUs **within a single node only** (8 GPUs per tray, 14.4 TB/s aggregate)
- Cross-node communication via NDR InfiniBand, 400 Gb/s per port, 8x CX-7 HCAs per node

## Constraints

- Available features: `viking`, `emr`, `b200`, `ndr`, `dhb`, `dhc`, `aircool`, `cac1`-`cac4`, `ibcleaf03-01` through `ibcleaf03-04`, `ibcleaf01-01` through `ibcleaf01-04`.
- `dhb` = Data Hall B (prenyx[0001-0160]), `dhc` = Data Hall C (prenyx[0161-0288]).
- `ibcleafXX-YY` = IB leaf group identifier; nodes in same leaf group share leaf switches (lowest latency for same-rail traffic).

```bash
# Request B200 nodes in a specific leaf group (topology-aware)
--constraint="b200,ibcleaf03-03"

# Request nodes in Data Hall B only
--constraint="b200,dhb"

# Check a node's features
scontrol show node prenyx0023 | grep Features

# List available nodes with features
sinfo -p batch -t idle,alloc -o "%15P|%55b|%10t|%N"

# Check leaf group topology
scontrol show topo
```

## Storage

| Filesystem | Type | Path | Notes |
|------------|------|------|-------|
| Home | NFS | `/home/<username>` | Small quota. Kerberized. |
| Project | Lustre (DDN Exascaler) | `/project` | Recommended working directory. High-speed, all compute nodes. |
| FSW | Lustre (DDN Exascaler) | `/lustre/fsw/portfolios/coreai/users/<user_name>` | Recommended working directory. High-speed, all compute nodes. |

- All Lustre mounts are available on all compute nodes.
- All filesystems require a valid Kerberos ticket; expired tickets block access.
- Kerberos tickets are valid for 10 hours, auto-renewed up to 7 days.
- Re-authenticate with `kinit` if ticket expires; log in at least once every 7 days for long-queued jobs.

## User Root directory
`/lustre/fsw/coreai_comparch_trtllm/<user_name>` (not `/home`)

## Model Repository

| Path | Notes |
|------|-------|
| `/lustre/fsw/coreai_comparch_trtllm/llm-models` | Default shared model weights directory |
