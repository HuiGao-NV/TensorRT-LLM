<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary
-->

# ASE Pre-Cluster Common Reference

Patterns shared across all ASE pre-clusters. Cluster-specific hardware, partitions, and IB topology are in separate files.

## Constraint Syntax

- Constraints select node groups via SLURM features: `--constraint=<feature>`
- Discover available features: `sinfo -p batch -o "%P %a %l %D %f"`
- Per-node features: `scontrol show nodes | grep -E "NodeName=|Features="`

| Scope | Syntax | Description |
|-------|--------|-------------|
| Block (rack) | `--constraint=nvlblk[01-02]-[01-16]` | 1 rack, 32 total groups sharing same rack |
| Leaf group | `--constraint=ibcleaf[01-02]-[01-08]` | 2 racks per leaf-group, 16 total groups sharing same leaf switches |
| Leaf group (alt) | `--constraint=cleaf01-01` or `--constraint=sleaf01` | Cluster-specific leaf naming (check `sinfo`) |
| Pod | `--constraint=pod[1-2]` | 16 racks per pod, 2 total pods |

## Job Comments

- Comments are set at job submission time and cannot be changed after submission.
- Multiple comments: separate with commas in a single `--comment=` value.

| Comment Flag | Effect |
|-------------|--------|
| `--comment=metrics` | Enable Prometheus/Grafana metrics collection (CPU, memory, disk, network I/O, GPU utilization) |
| `--comment=transparent_hugepage=always` | Set THP mode to always |
| `--comment=transparent_hugepage=never` | Set THP mode to never |
| `--comment=transparent_hugepage_defrag=always` | Set THP defrag (values: `always`, `defer+madvise`, `defer`, `never`) |
| `--comment=sysctl-sys.kernel.numa_balancing=1` | Enable NUMA balancing |
| `--comment=sysctl-sys.vm.zone_reclaim_mode=1` | Enable zone reclaim mode |
| `--comment=lustreclient=nopcc` | Disable Lustre Persistent Cache on Client (PCC) |

```bash
# Example: THP always with metrics
sbatch --comment=transparent_hugepage=always,metrics run.sub
```

## GPU Clock Sudo

- All sudo commands must run OUTSIDE the container, as a separate srun step.
- Check available sudo commands on a node: `sudo -l`

```bash
# Set application clocks (memory, graphics) — requires root via sudo
srun -N ${SLURM_NNODES} --ntasks-per-node=1 sudo nvidia-smi -ac <memory_clock>,<graphics_clock>

# Example: set clocks before running workload
srun -N ${SLURM_NNODES} --ntasks-per-node=1 sudo nvidia-smi -ac 2619,1980
srun ./app.sh

# Lock graphics clocks to a range
sudo nvidia-smi -lgc <min_clock_MHz>,<max_clock_MHz>
# Lock specific GPU
sudo nvidia-smi -lgc <min>,<max> -i <gpu#>

# Reset graphics clocks
sudo nvidia-smi -rgc
# Reset specific GPU
sudo nvidia-smi -rgc -i <gpu#>

# List supported clock combinations
nvidia-smi -q -d SUPPORTED_CLOCKS
```

## Power Limit Sudo

- Must run OUTSIDE the container, as a separate srun step.

```bash
# Set power limit in watts
sudo nvidia-smi -pl <power_limit_in_watts>
# Set for specific GPU
sudo nvidia-smi -pl <watts> -i <gpu#>
```

## Container Save

- `--container-save=<path>` persists container modifications to a squashfs file.
- The saved `.sqsh` file can be reused as a container image in future jobs.

```bash
# Install packages and save the modified container
srun -A <account> -N1 -p <partition> \
  --container-image=nvidia/cuda:<ver>-cudnn-devel-ubuntu<rel> \
  --container-save=/project/<project>/<user>/custom.sqsh \
  apt install -y vim

# Reuse the saved container in a future job
srun -A <account> -N1 -p <partition> \
  --container-image=/project/<project>/<user>/custom.sqsh \
  ./run.sh
```

## Power Data Collection

- Collects CPU and GPU power data from Prometheus for a completed or running job.
- Can be run at end of job or after job completes.

```bash
# Collect power data for current job
/project/share/admin-ops/download-job-power-metrics/download-job-power-metrics \
  -job "${SLURM_JOB_ID}" -out <OUTPUT_DIR>

# Default output path (if -out omitted):
#   $(pwd)/slurm-${SLURM_JOB_ID}-power/
```

- **Single-node NVLink systems**: one CSV per node with columns: `timestamp`, `bmc_PWR_CPU0`, `bmc_PWR_CPU1`, `pdu_apparentPower_[0-11]`, `pdu_powerFactor_[0-11]`
- **MNNVL systems**: one CSV per node (`bmc_cpu_0_power`, `bmc_cpu_1_power`, `bmc_gpu_[0-3]_power`) plus one CSV per rack with powershelf outputs (`<rack>-pwr[01-08]_output_power`)
- BMC data may be missing for some nodes (warning: `found 0 series instead of 8`); PDU data is still written when this occurs.

## IMEX Channels

- IMEX channels are only created during an `srun` step.
- `salloc` alone does NOT start IMEX.
- Workaround for interactive sessions: use `salloc` then `srun --pty bash` to get IMEX channels initialized.

```bash
# IMEX will NOT be available after this alone:
salloc -A <account> -p <partition> -N1

# IMEX IS available after this:
srun --pty bash
```

## Multiple MPI in Same Job

- Run multiple MPI commands as separate job steps (sequential `srun` calls).
- Each `srun` is a distinct job step with its own MPI context.

```bash
#!/bin/bash
#SBATCH -A $PROJECT
#SBATCH -p batch
#SBATCH --ntasks-per-node=$GPUS

srun --mpi=pmix ./run1.sub
srun --mpi=pmix ./run2.sub
```

- To restrict job steps to different GPUs, use `NVIDIA_VISIBLE_DEVICES`:

```bash
NVIDIA_VISIBLE_DEVICES=0,1 srun ./run1.sub
NVIDIA_VISIBLE_DEVICES=2,3 srun ./run2.sub
```

## Other Sudo Commands

| Command | Purpose |
|---------|---------|
| `sudo sysctl vm.drop_caches=[1-3]` | Drop page/inode/dentry caches (1=pages, 2=dentries+inodes, 3=all) |

## Miscellaneous Facts

- ASE pre-clusters do NOT support `--gres=gpu:8`; all jobs run exclusively on each node with all GPUs.
- Hash Based Forwarding (HBF) is enabled on all clusters by default.
- Driver regkeys are static and not user-configurable; list with `cat /proc/driver/nvidia/params` from a compute node.
- Email notifications (`--mail-type`) are not supported.
- Login node thread limit is dynamic (based on cores); check with: `cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/pids.{current,max}`
