---
name: remote-slurm
description: >-
  Remote SLURM cluster development via SSH. Use when running jobs,
  profiling, or developing on a remote SLURM cluster with pyxis/enroot
  containers. Covers SSH connection management, srun/sbatch/salloc job
  patterns, tmux-based allocation persistence, file transfer, and safe
  remote file access. Triggers on: "run on cluster", "submit job",
  "srun", "sbatch", "salloc", "remote GPU", "SSH to cluster",
  "SLURM job", "enroot", "pyxis". Works with any SLURM cluster
  accessible via SSH.
tags: [infrastructure, remote, slurm]
license: LicenseRef-NvidiaProprietary
metadata:
  author: NVIDIA Corporation
  documentation: https://gitlab-master.nvidia.com/wkong/perf-bot
---

# Remote SLURM Development

Develop and execute on remote SLURM clusters via SSH commands. No filesystem
sync needed — all remote operations go through SSH.

## Step 1: Load Config

This agent receives a cluster config file path from the invoking skill.
The config file is a markdown file with YAML frontmatter located in the
skill's `references/` directory.

Parse the YAML frontmatter to extract: `ssh_host`, `user`, `remote_cwd`,
`account`, `subproject`, `partition`, `container_image`, `mounts`, and
optionally `gpus_per_node` (required on shared/QOS clusters) and `packages`
(extra pip packages not in the container image).

If a cluster-specific reference exists in `references/` matching the `cluster`
field (e.g., `references/lyris.md`), read it for partition limits, GPU counts,
and topology-aware scheduling guidance. Also read
`references/pre-cluster-common.md` for shared constraint and tuning patterns.

## Step 2: SSH Preflight

Verify SSH connectivity. The goal is to confirm you can reach the cluster —
not to force a specific connection method.

**Check 1:** Test if any existing ControlMaster is active — either the user's
own (from `~/.ssh/config`) or a PerfBot-managed one:

```bash
ssh -O check <ssh_host> 2>&1 || ssh -o ControlPath=/tmp/ssh-perfbot-%r@%h:%p -O check <ssh_host> 2>&1
```

If either succeeds, a ControlMaster is already running. Skip to Step 3.

**Check 2:** If no ControlMaster exists, test basic connectivity:

```bash
ssh -o ConnectTimeout=5 <ssh_host> "echo ok"
```

If this succeeds, SSH works (e.g., key-based auth without multiplexing).
**Skip to Step 3.** Do NOT create a PerfBot-specific ControlMaster; reuse
whatever SSH config already works.

**Check 3 (MFA fallback):** If the connectivity check fails with
`Permission denied (keyboard-interactive)`, the cluster requires MFA. MFA
uses browser-based device code verification that needs an interactive SSH
session — the agent cannot complete this because the SSH session must stay
alive while the user authenticates in their browser and presses ENTER.

Tell the user to run the ControlMaster command (with MFA, it will prompt
interactively instead of backgrounding immediately):

> Run this in your terminal:
>
>     ssh -fNM -o ControlMaster=auto -o ControlPath=/tmp/ssh-perfbot-%r@%h:%p -o ControlPersist=24h <ssh_host>
>
> You will see a PIN and a URL like:
>
>     Authenticate with PIN XXXXXX at https://login.microsoft.com/device
>
> Open that URL, enter the PIN, approve, then press ENTER. The SSH session
> will background itself as a persistent ControlMaster. Let me know when done.

After the user confirms, verify with `ssh -o ControlPath=/tmp/ssh-perfbot-%r@%h:%p -O check <ssh_host>`.

**If SSH fails for other reasons**, ask the user to verify their `ssh_host` value
in `slurm-cluster-<name>.md` (it should be exactly what they type after `ssh` to connect).

## Step 3: Remote Environment Check

Verify the remote environment is ready:

```bash
ssh <ssh_host> "test -d <remote_cwd> && echo ok"
ssh <ssh_host> "which tmux"
ssh <ssh_host> "squeue --version"
```

If `remote_cwd` does not exist, create it or ask the user. If tmux or SLURM
commands are not available, stop and inform the user.

## Recipes

Pick the recipe that matches your workflow. Each recipe is self-contained —
copy it, substitute the `<placeholders>` from config, and run.

**Placeholders** (all from `slurm-cluster-<name>.md` unless noted):

| Placeholder | Source |
|-------------|--------|
| `<ssh_host>`, `<account>`, `<partition>`, `<image>`, `<mounts>`, `<subproject>` | Config |
| `<detail>` | You generate: short task descriptor (`nccl`, `train`, `profile`, `debug`) |
| `<nodes>` | User request or task requirement |
| `<gpus>` | GPUs per node — probe if unknown (see bottom of this section) |
| `<time>` | Walltime in HH:MM:SS |

**Job naming:** Every recipe includes `-J <account>-<subproject>.<detail>`.
This is required — without `-J`, SLURM emits warnings and jobs are hard to
identify in `squeue`.

**GPU allocation:** Cluster behavior varies:

- **Exclusive-access clusters** (most NVIDIA internal): Nodes are allocated
  exclusively with all GPUs available. No `--gpus-per-node` needed.
- **Shared/QOS clusters**: Require `--gpus-per-node=<gpus_per_node>`. Without
  it, you may get a CPU-only node (no GPUs at all), a "Cannot find GPU
  specification" error, or a "QOSMinGRES" error (below minimum).

If `gpus_per_node` is set in the cluster config, add `--gpus-per-node=<gpus_per_node>`
to every `srun`, `salloc`, and `sbatch` command. If not set, omit it.

If a job fails with GPU-related errors and `gpus_per_node` is not configured,
probe the node's GPU count, set `gpus_per_node` in the cluster config, and retry.

Do not set `CUDA_VISIBLE_DEVICES` for MPI workloads — the MPI runtime handles
GPU assignment.

---

### Recipe 1: Single-process command (srun)

One-off commands on a single node: profiling, inspecting, lightweight scripts.

```bash
ssh <ssh_host> "srun -A <account> -p <partition> -N 1 --ntasks-per-node=1 \
  -J <account>-<subproject>.<detail> -t <time> \
  --container-image=<image> --container-mounts=<mounts> \
  <command>"
```

---

### Recipe 2: MPI workload (srun)

MPI applications across nodes: NCCL tests, HPCX benchmarks. The binary must
be MPI-aware (e.g., `all_reduce_perf_mpi`, not `all_reduce_perf`).

```bash
ssh <ssh_host> "srun -A <account> -p <partition> -N <nodes> \
  --ntasks-per-node=<gpus> --mpi=pmix \
  -J <account>-<subproject>.<detail> -t <time> \
  --container-image=<image> --container-mounts=<mounts> \
  <command>"
```

---

### Recipe 3: Distributed training (srun)

Framework-launched distributed training (PyTorch DDP, DeepSpeed). SLURM starts
`<nodes> × <gpus>` processes. Pyxis/enroot sets `RANK`, `LOCAL_RANK`, and
`WORLD_SIZE`. Each process runs the training script directly.

```bash
ssh <ssh_host> "srun -A <account> -p <partition> -N <nodes> \
  --ntasks-per-node=<gpus> \
  -J <account>-<subproject>.<detail> -t <time> \
  --container-image=<image> --container-mounts=<mounts> \
  python train.py"
```

---

### Recipe 4: Launcher-based training (srun)

One process per node, an in-container launcher forks locally (torchrun,
deepspeed launcher).

```bash
ssh <ssh_host> "srun -A <account> -p <partition> -N <nodes> \
  --ntasks-per-node=1 \
  -J <account>-<subproject>.<detail> -t <time> \
  --container-image=<image> --container-mounts=<mounts> \
  torchrun --nproc-per-node=<gpus> train.py"
```

---

### Recipe 5: Batch job (sbatch)

Fire-and-forget background job:

```bash
ssh <ssh_host> "sbatch -A <account> -p <partition> -N <nodes> \
  -J <account>-<subproject>.<detail> -t <time> \
  --container-image=<image> --container-mounts=<mounts> \
  <script-path>"
```

Check status:

```bash
ssh <ssh_host> "squeue -j <jobid>"
ssh <ssh_host> "sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed -n"
```

---

### Recipe 6: Multi-command session (salloc + srun)

Interactive development with persistent container state. Use this when you need
multiple commands on the same allocation — installing packages, compiling,
running experiments iteratively.

**Step 1: Allocate via tmux**

tmux keeps the allocation alive on the frontend — invisible plumbing.

```bash
ssh <ssh_host> "tmux new-session -d -s perfbot-<id> \
  'salloc -A <account> -p <partition> -N <nodes> \
  -J <account>-<subproject>.<detail> -t <time>'"
```

**Step 2: Check allocation status**

Wait ~30 seconds for the scheduler, then get job ID and status:

```bash
ssh <ssh_host> "squeue -u <user> -h -o '%i %T %S' --state=RUNNING,PENDING"
```

If RUNNING → proceed to Step 3.

If PENDING → follow the **walltime negotiation flow** (see Walltime strategy
section). Only proceed to Step 3 once the job is RUNNING.

**Step 3: Create a persistent container**

Import the image once. This creates a named container on each node that
persists across `srun` calls (container persistence):

```bash
ssh <ssh_host> "srun --jobid=<jobid> --ntasks-per-node=1 \
  --container-image=<image> --container-name=perfbot-<jobid> true"
```

**Container packages:** If `packages` is defined in config, install them
using the exact names listed — do not guess or substitute package names:

```bash
ssh <ssh_host> "srun --jobid=<jobid> --ntasks-per-node=1 \
  --container-name=perfbot-<jobid> --container-writable \
  pip install <packages>"
```

Only install when the workload actually needs them — not for simple
commands like `nvidia-smi` or `hostname`.

**Step 4: Run commands**

Use `--container-name` (without `--container-image`) to reuse the persistent
container. All in-container state is preserved between commands.

Single-process command:

```bash
ssh <ssh_host> "srun --jobid=<jobid> \
  --container-name=perfbot-<jobid> --container-mounts=<mounts> \
  <command>"
```

Per-node operation (mkdir, pip install) — use `--ntasks-per-node=1`:

```bash
ssh <ssh_host> "srun --jobid=<jobid> --ntasks-per-node=1 \
  --container-name=perfbot-<jobid> --container-mounts=<mounts> \
  <command>"
```

MPI workload within the allocation:

```bash
ssh <ssh_host> "srun --jobid=<jobid> \
  --ntasks-per-node=<gpus> --mpi=pmix \
  --container-name=perfbot-<jobid> --container-mounts=<mounts> \
  <command>"
```

Distributed training within the allocation:

```bash
ssh <ssh_host> "srun --jobid=<jobid> \
  --ntasks-per-node=<gpus> \
  --container-name=perfbot-<jobid> --container-mounts=<mounts> \
  python train.py"
```

Add `--container-writable` if the workload writes to container paths outside
of mounted directories (e.g., `pip install` into site-packages).

**Step 5: Release the allocation**

Clean up the named container, then kill tmux. If the allocation has already
expired (walltime exceeded), the `srun` will fail — that is fine, the container
is already gone. Just kill the tmux session.

```bash
ssh <ssh_host> "srun --jobid=<jobid> --ntasks-per-node=1 \
  bash -c 'enroot remove -f perfbot-<jobid> 2>/dev/null; true'" 2>/dev/null
ssh <ssh_host> "tmux kill-session -t perfbot-<id>"
```

---

### Probing GPUs per node

If you do not know `<gpus>`, probe once and record in cluster notes:

```bash
ssh <ssh_host> "srun --jobid=<jobid> --ntasks-per-node=1 \
  --container-name=perfbot-<jobid> bash -c 'nvidia-smi -L | wc -l'"
```

## Cluster-Specific Flags

After substituting recipe placeholders, check the cluster reference file
(`references/<cluster>.md`) for additional flags:

- **SHARP**: Add `--network=sharp` if the cluster supports it
- **Topology constraints**: Add `--constraint=<feature>` for topology-aware
  placement (e.g., `--constraint=nvlblk01` for a single rack)
- **GPU clocks**: Run `sudo nvidia-smi -ac <mem>,<graphics>` as a separate
  `srun --ntasks-per-node=1` pre-step in sbatch scripts
- **Comments**: Add `--comment=metrics` for Prometheus/Grafana collection

## File Transfer

Use SSH pipes for file transfer — no extra tools needed, works on every cluster.

### Local → Remote (writing files)

Pipe file content directly via SSH heredoc. Never write locally then scp:

```bash
ssh <ssh_host> "cat > <remote_cwd>/script.py" << 'PYEOF'
import torch
# ... file content ...
PYEOF
```

For multiple files, repeat the pattern — each file is one SSH command.

### Remote → Local (reading results)

Read remote files via SSH stdout:

```bash
ssh <ssh_host> "cat <remote_cwd>/output.json"
```

For large files, read only what you need:

```bash
ssh <ssh_host> "tail -n 100 <remote_cwd>/slurm-<jobid>.out"
ssh <ssh_host> "head -n 50 <remote_cwd>/results.csv"
```

### When to use scp

Fall back to `scp` only for binary files or large directories that cannot be
piped through SSH:

```bash
scp <ssh_host>:<remote_cwd>/profile.nsys-rep .         # binary profile
scp -r local_dir/ <ssh_host>:<remote_cwd>/             # whole directory
```

**Decision rule:** text files → SSH pipe. Binary files or directories → scp.
Never use rsync or sshfs — not reliably available on HPC clusters.

## Rules

### Never read full remote files

Remote output can be gigabytes. Always use `tail`, `head`, or `grep`:

```bash
ssh <ssh_host> "tail -n 100 <remote_cwd>/output.log"
ssh <ssh_host> "head -n 50 <remote_cwd>/script.py"
ssh <ssh_host> "grep -n 'error\|Error\|ERROR' <remote_cwd>/output.log"
```

Never use `cat` on remote files unless you are certain the file is small.

### Never run heavy computation on the frontend

The frontend node is shared. Only use it for SLURM commands, lightweight file
operations, and tmux management.

### Check before creating

Before creating resources, check if they already exist:
- SSH ControlMaster: `ssh -O check <ssh_host>`
- tmux sessions: `ssh <ssh_host> "tmux has-session -t perfbot-<id> 2>/dev/null && echo exists"`
- Existing allocations: `ssh <ssh_host> "squeue -u <user> -h -o '%i %j' --state=RUNNING"`

### Clean up when done

- Remove named containers: `enroot remove -f` (see Recipe 6, Step 5)
- Kill tmux sessions: `ssh <ssh_host> "tmux kill-session -t perfbot-<id>"`
- Cancel unneeded jobs: `ssh <ssh_host> "scancel <jobid>"`

### Walltime strategy

- **Production (sbatch, srun):** Tight walltime with ~20% margin — gets
  scheduled sooner.
- **Dev/debug (salloc):** Use **walltime negotiation** — submit with max
  walltime, then reduce if the queue wait is too long:

**Walltime negotiation flow:**

1. Submit `salloc` with **max walltime** for the partition.
2. Wait ~30 seconds for the scheduler to process, then check:
   ```bash
   ssh <ssh_host> "squeue -j <jobid> -h -o '%i %T %S'"
   ```
   - `%S` = estimated start time. Parse it to calculate wait.
3. **If RUNNING** → proceed to next step.
4. **If PENDING and wait ≤ 10 minutes** → wait for it, proceed.
5. **If PENDING and wait > 10 minutes** (or no start time shown):
   a. Infer how long you actually need from context (profiling → ~30 min,
      iterative dev → ~2h, training run → match expected duration).
      Round up to the next hour boundary.
   b. Reduce walltime in-place:
      ```bash
      ssh <ssh_host> "scontrol update JobId=<jobid> TimeLimit=<new_time>"
      ```
   c. Tell the user: "Reduced walltime from Xh to Yh for faster scheduling —
      you'll have less time on the allocation."
   d. Wait ~15 seconds, re-check estimated start time.
   e. **If still > 10 minutes:** inform user with the ETA and suggest
      alternatives — different partition (e.g., backfill), fewer nodes.

### Keep slurm-cluster-<name>.md up to date

Update the active cluster's config file when you discover cluster information:
partition limits, queue behavior, container quirks, GPU counts, mount
requirements.

## Recovering from Failures

**SSH connection drops:** Re-run SSH preflight (Step 2). tmux-held allocations
survive disconnects.

**Job stuck in PENDING:**

```bash
ssh <ssh_host> "squeue -j <jobid> -o '%i %T %r %S'"
```

Check the reason (`%r`) and estimated start (`%S`). If wait > 10 minutes,
follow the **walltime negotiation flow** (see Walltime strategy). If already
negotiated and still stuck, suggest alternatives: backfill partition, fewer
nodes, or a different cluster.

**Job fails:** Read last lines, never dump full output:

```bash
ssh <ssh_host> "tail -n 100 <remote_cwd>/slurm-<jobid>.out"
ssh <ssh_host> "tail -n 50 <remote_cwd>/slurm-<jobid>.err"
```
