<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary
-->

---
cluster: <cluster-name>
ssh_host: <ssh-destination>       # what you type after 'ssh' (e.g., lyris, wkong-mfa@login-lyris)
user: <slurm-username>            # SLURM username for squeue -u (may differ from SSH user on MFA clusters)
repo_root: <absolute-path>         # TensorRT-LLM repo root on this cluster (used as REPO_ROOT for both local and remote slurm)
remote_cwd: <absolute-path>       # working directory on the cluster (defaults to repo_root if not set)
account: <slurm-account>          # run: sacctmgr -nP show assoc where user=$(whoami) format=account
subproject: <subproject-name>     # for job naming: <account>-<subproject>.<detail>
partition: <default-partition>    # run: sinfo -o "%P %l %D" --noheader
container_image: <image-uri>      # e.g., nvcr.io/nvidia/pytorch:25.01-py3
mounts:
  - <host-path>:<container-path>
gpus_per_node:                         # optional: set if cluster requires --gpus-per-node (e.g., 4, 8)
packages:                              # optional: extra pip packages not in the container
  # - nvidia-cutlass-dsl
  # - apache-tvm-ffi
---

# Cluster Notes

<!-- This section is updated automatically with cluster quirks discovered during sessions -->
