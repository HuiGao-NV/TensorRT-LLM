<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary
-->

# Pre-Tyche Cluster Reference

## GPU Board

| Component | Spec |
|-----------|------|
| GPU | B200 Blackwell, 4 per compute tray (2 per Bianca board) |
| **gpus_per_node** | **4** |
| GPU Memory | 192 GB HBM3e per GPU |
| GPUs per rack | 36 (9 compute trays) |
| Rack types | 8 ESVT racks (A01P) + 32 DSVT racks (A01R) |
| CPU arch | arm |

## Multi-Node GPU Communication

- **NVLink5** connects GPUs within MNNVL domains:
  - **36x1 config**: 36 GPUs within one rack
  - **36x2 config**: 72 GPUs across two adjacent racks (NVLink cables between switch trays)
- Up to 1,800 GB/s per GPU bandwidth
- Cross-domain communication uses InfiniBand (not NVLink)

## Constraints

- Block constraints: `--constraint="nvlblk17-01"` (force job onto a specific NVL block)
- Hardware type: `--constraint="a01r"` or `--constraint="a01p"`
- Configuration: `--constraint="36x2"` or `--constraint="36x1"`
- IB leaf: `--constraint="ibcleaf01-01"`
- Rack constraints: `--constraint="a02"`, `--constraint="a02p"`
- Mitigation flag: `--constraint="mitigated"`
- Generally not needed; prefer `--segment` and let the scheduler decide placement

## Storage

| Filesystem | Type | Path | Notes |
|------------|------|------|-------|
| Home | NFS (Isilon) | `/home/<username>` | Small quota; not for workloads |
| Project | Lustre (DDN Exascaler) | `/project/<project>/` | Recommended working directory |
| Fast scratch | Lustre (DDN Exascaler) | `/lustre/fsw/coreai_comparch_trtllm/<user_name>/` | High-speed parallel I/O; recommended for workloads |

- Lustre available on ALL compute nodes
- Kerberos ticket required for filesystem access; ticket valid 10 hours, auto-renewed up to 7 days
- Expired ticket = no access to NFS, SLURM, or Lustre (`Permission denied`)
- Renew expired ticket: `kinit` on login node; destroy and reinit if multiple tickets: `kdestroy -A && kinit`
- Log in at least once every 7 days to keep queued job tickets valid

## User Root directory
`/lustre/fsw/coreai_comparch_trtllm/<user_name>` (not `/home`)

## Model Repository

| Path | Notes |
|------|-------|
| `/lustre/fsw/coreai_comparch_trtllm/llm-models` | Default shared model weights directory |
