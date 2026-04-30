<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary
-->

---
name: trtllm-remote-slurm-executor
description: >-
  Remote SLURM cluster development via SSH. Use when running jobs,
  profiling, or developing on a remote SLURM cluster with pyxis/enroot
  containers. Covers SSH connection management, srun/sbatch/salloc job
  patterns, tmux-based allocation persistence, file transfer, and safe
  remote file access. Works with any SLURM cluster accessible via SSH.
tags: [infrastructure, remote, slurm]
license: LicenseRef-NvidiaProprietary
metadata:
  author: NVIDIA Corporation
---

# Remote SLURM Development

Develop and execute on remote SLURM clusters via SSH commands. No filesystem
sync needed — all remote operations go through SSH.

## Step 1: Resolve Cluster

The orchestrator supplies one of two input shapes; pick the right path:

- **Explicit cluster** — `cluster_config_path` (or `slurm_cluster` name) is
  provided. Skip auto-selection and go straight to **Load the config**.
- **Auto-select cluster** — only hardware constraints are provided
  (`device_type`, `total_required_devices`, `required_devices_per_node`).
  Resolve a cluster from the constraints, then load its config.

### 1a. Auto-select (only when no explicit cluster was provided)

Invoke the `slurm-cluster-info` skill in **constraint-based mode**, passing
`device_type`, `required_devices_per_node`, and (optionally)
`total_required_devices`. That skill owns the filter rules and either
returns a single resolved cluster entry (with `config_path`) or stops and
asks the user. Do not implement the filter here.

Use `node_count` from `job_spec.json` directly as `--nodes` and
`required_devices_per_node` as `--ntasks-per-node`. **Do not recompute
`node_count`** — `trtllm-case-executor` is the single source of truth and
writes the resolved value into `job_spec.json` for every executor to read.

The returned entry's `config_path` feeds Step 1b.

### 1b. Load the config

The cluster connection config is a markdown file with YAML frontmatter
located in the `slurm-cluster-info` skill's `references/` directory
(`skills/slurm-cluster-info/references/slurm-cluster-<cluster>.md`).

Parse the YAML frontmatter to extract: `ssh_host`, `user`, `remote_cwd`,
`account`, `partition`, `mounts`, and optionally `gpus_per_node`
(required on shared/QOS clusters) and `packages` (extra pip packages not
in the container image). The cluster YAML may also list `container_image`
informationally, but **the runtime image comes from `job_spec.json`**
(resolved by `trtllm-case-executor`); do not consume the cluster YAML's
`container_image` field here.

Invoke the `slurm-cluster-info` skill in single-cluster mode with the
selected `cluster` name to retrieve partition limits, GPU counts,
topology-aware scheduling guidance, `default_models_repo`, `mfa_style`
(used by Step 2 to choose the SSH path — see below), and shared
constraint and tuning patterns.

## Step 2: SSH Preflight

The cluster's `mfa_style` (returned by `slurm-cluster-info` in Step 1b) is
the **single source of truth for which SSH path to use**. Switch on the
value — do **not** probe-and-fall-back on the SSH error string.

| `mfa_style` | Path | Why |
|-------------|------|-----|
| `false` | Direct (Step 2a) | Cluster accepts key-based / passwordless SSH; no special handling needed. |
| `true`  | MFA (Step 2b)    | Cluster requires interactive MFA (Duo / device-code / one-time PIN). |
| `null`  | Unknown (Step 2c)| `mfa_style` not declared in the cluster reference. Probe direct first, fall back to MFA, and ask the user to update the reference file so future runs skip the probe. |

### 2.0. Build `<ssh_host>` from `mfa_style` + `cluster_name`

Before any SSH attempt, derive the SSH destination from the resolved
inputs. NVIDIA-internal clusters follow this convention:

```bash
MFA_FLAG=""
[ "$mfa_style" = "true" ] && MFA_FLAG="-mfa"
SSH_HOST="${slurm_user}${MFA_FLAG}@login-${cluster_name}.nvidia.com"
```

Where:

- `slurm_user` is case-executor's `slurm_user` input (default `$(whoami)`).
- `cluster_name` is `cluster_name` from `slurm-cluster-info` (Step 1b).
- `mfa_style` is from `slurm-cluster-info` (Step 1b). The `-mfa` suffix is
  appended to the username only when MFA is required.

The result is the canonical `<ssh_host>` placeholder used everywhere in
the rest of this skill (Steps 2a/2b/2c, Step 3, recipes, file-transfer,
hang detection).

**Override:** if the caller passed an explicit `ssh_host` input to
case-executor, use that value verbatim and skip the construction. The
constructed value is the default; the input is the escape hatch.

### 2a. Direct path — `mfa_style: false`

Verify connectivity using the user's existing SSH config:

```bash
ssh -o ConnectTimeout=5 <ssh_host> "echo ok"
```

If this succeeds, **skip to Step 3**. Do not create a trtllm-managed
ControlMaster — reuse whatever the user already has.

If it fails, do **not** silently switch to the MFA path. Stop and ask the
user to verify their `ssh_host` value or their key-based SSH setup.
A direct-path cluster failing here is a configuration issue, not an MFA
prompt to interpret.

### 2b. MFA path — `mfa_style: true`

First check whether a ControlMaster is already live (the user may have set
one up earlier in this session):

```bash
ssh -O check <ssh_host> 2>&1 || ssh -o ControlPath=/tmp/ssh-trtllm-%r@%h:%p -O check <ssh_host> 2>&1
```

If either succeeds, **skip to Step 3** — reuse it.

Otherwise the user must complete an interactive MFA handshake. MFA uses
browser-based device-code verification that needs an interactive SSH
session — the skill cannot complete this on the user's behalf because the
session must stay alive while the user authenticates in their browser and
presses ENTER.

Tell the user to run the ControlMaster command:

> Run this in your terminal:
>
>     ssh -fNM -o ControlMaster=auto -o ControlPath=/tmp/ssh-trtllm-%r@%h:%p -o ControlPersist=24h <ssh_host>
>
> You will see a PIN and a URL like:
>
>     Authenticate with PIN XXXXXX at https://login.microsoft.com/device
>
> Open that URL, enter the PIN, approve, then press ENTER. The SSH session
> will background itself as a persistent ControlMaster. Let me know when done.

After the user confirms, verify with
`ssh -o ControlPath=/tmp/ssh-trtllm-%r@%h:%p -O check <ssh_host>` and
proceed to Step 3.

### 2c. Unknown — `mfa_style: null`

The cluster reference (`references/<cluster>.md`) does not declare an
`mfa_style` value. Try the direct path (Step 2a) first. If it fails with
`Permission denied (keyboard-interactive)` — and only that — switch to
the MFA path (Step 2b). Any other failure stops here and is reported to
the user.

After connecting successfully, **ask the user to update the cluster
reference file** with the discovered value (`mfa_style: true` or
`false`, in the `## Authentication` table) so that future runs skip
this probe and use the deterministic switch in Step 2a / 2b.

## SSH Command Shorthand

Every SSH and SCP invocation in the rest of this skill uses the placeholders below. Both pin `ControlPath` to the trtllm-managed socket so they reuse the ControlMaster set up in Step 2 — without this, OpenSSH starts a fresh handshake on every call, which on MFA clusters re-prompts for keyboard-interactive auth (and defeats the whole point of multiplexing):

| Placeholder | Expansion |
|-------------|-----------|
| `<ssh_cmd>` | `ssh -o ControlPath=/tmp/ssh-trtllm-%r@%h:%p` |
| `<scp_cmd>` | `scp -o ControlPath=/tmp/ssh-trtllm-%r@%h:%p` |

When the user has their own ControlMaster configured in `~/.ssh/config` (no trtllm-managed socket needed), the explicit `-o ControlPath` is harmless: if the path has no live master, OpenSSH falls back to a fresh connection. The connectivity probe in Step 2 Check 2 (`ssh -o ConnectTimeout=5 <ssh_host> "echo ok"`) deliberately uses bare `ssh` to test un-multiplexed access; everywhere else use `<ssh_cmd>` / `<scp_cmd>`.

## Step 3: Remote Environment Check

Verify the remote environment is ready:

```bash
<ssh_cmd> <ssh_host> "test -d <remote_cwd> && echo ok"
<ssh_cmd> <ssh_host> "which tmux"
<ssh_cmd> <ssh_host> "squeue --version"
```

If `remote_cwd` does not exist, create it or ask the user. If tmux or SLURM
commands are not available, stop and inform the user.

## Step 4: Prepare Remote Environment

This step sets up the user's working directory and clones the TensorRT-LLM
repository on the remote cluster. It runs once before any job submission.

> **MANDATORY: temp-branch sync only.** Code is always synced from local to remote
> via the temp-branch flow in **4c–4f**: create a unique session-scoped branch on
> the local repo, push it to `origin`, fetch it on the remote, check it out, then
> delete the temp branch (local + origin) once the remote checkout succeeds.
>
> **Do NOT try to fetch/checkout the user's current branch (`repo_branch`) on the
> remote directly.** The current branch may not exist on `origin` (the user may not
> have pushed it), may be stale, or may not include uncommitted/untracked work.
> Skipping 4c–4f to take a shortcut like `git fetch origin <repo_branch> && git checkout <repo_branch>`
> is forbidden — it has caused failures (e.g., `fatal: couldn't find remote ref ...`).
> Always run 4c–4f, even when the local current branch already exists on `origin`.

**Inputs received from `trtllm-case-executor`:**

| Field | Description |
|-------|-------------|
| `default_user_root_dir` | Per-user root directory on the remote cluster (from `trtllm-env-check` / `slurm-cluster-info`) |
| `repo_url` | Git remote URL of the local TensorRT-LLM repo (from `git remote get-url origin`). **Used to clone the bare repository on the remote (Step 4b) and to push the temp branch to (Step 4d).** |
| `repo_branch` | **Informational only** — the local current branch name (from `git rev-parse --abbrev-ref HEAD`). Used in Step 4c to remember which branch to restore locally after the temp branch is deleted. **Never used as the ref the remote checks out** — the remote always checks out the temp branch from Step 4c. |
| `device_type` | Required short GPU name (e.g., `B200`, `B300`). Used by Step 1a auto-selection. Ignored when an explicit cluster was provided. |
| `total_required_devices` | Total GPU processes across all nodes. Used by Step 1a auto-selection only. SLURM job sizing reads `node_count` from `job_spec.json` (precomputed by `trtllm-case-executor`) — do not recompute it here. |
| `required_devices_per_node` | Minimum GPUs per node. Used by Step 1a auto-selection (cluster must have `gpus_per_node >= required_devices_per_node`) and as `--ntasks-per-node`. |

**Procedure:**

### 4a. Create user root directory

```bash
<ssh_cmd> <ssh_host> "mkdir -p <default_user_root_dir>"
```

If `default_user_root_dir` is null or empty, skip this step and use `remote_cwd` directly.

### 4b. Clone or update the repository

Determine the repo directory name from `repo_url` (strip `.git` suffix if present):

```bash
REPO_DIR_NAME=$(basename <repo_url> .git)
REMOTE_REPO_PATH="<default_user_root_dir>/${REPO_DIR_NAME}"
```

Check if it already exists:

```bash
<ssh_cmd> <ssh_host> "test -d <remote_repo_path>/.git && echo exists"
```

If the repository does **not** exist, clone it:

```bash
<ssh_cmd> <ssh_host> "cd <default_user_root_dir> && git clone <repo_url>"
```

If the repository **already exists**, refresh remote refs only — do **not**
checkout or reset to any branch here:

```bash
<ssh_cmd> <ssh_host> "cd <remote_repo_path> && git fetch origin --prune"
```

The actual ref the remote will check out is the temp branch created in 4c
and pushed in 4d; the remote checkout happens in 4e. **Do not** add a
`git checkout <repo_branch>` or `git reset --hard origin/<repo_branch>`
here as a shortcut — that path is forbidden (see the MANDATORY callout
at the top of Step 4).

### 4c. Create a temporary local branch capturing all local changes

Generate a unique temp branch name tied to this session. The branch name
**must start with `<user_name>_<model_name>`** so it is easy to identify
the owner and the workload it was created for. A timestamp suffix keeps
it unique across sessions:

```bash
# Resolve the user name (prefer the slurm_user/ssh_user input over whoami,
# so the temp branch is attributed to the cluster user rather than the
# host that happens to dispatch the job).
USER_NAME="${slurm_user:-${ssh_user:-$(whoami)}}"

# Resolve the workload name. job_spec.json's `model_name` is authoritative;
# fall back to the repo basename if it is missing.
MODEL_NAME="${model_name:-$(basename "<REPO_ROOT>")}"

# Sanitize to a git-ref-safe identifier (alphanum + `_`/`-` only).
# Any other character is replaced with `_`.
sanitize() { printf '%s' "$1" | tr -c 'A-Za-z0-9_-' '_'; }
USER_NAME_SAFE=$(sanitize "$USER_NAME")
MODEL_NAME_SAFE=$(sanitize "$MODEL_NAME")

TEMP_BRANCH="${USER_NAME_SAFE}_${MODEL_NAME_SAFE}_$(date +%Y%m%d%H%M%S)"
```

Example resolved value: `huig_dwdp_deepseek_v3_lite_20260430145200`.

Stash the current branch name so it can be restored later:

```bash
ORIGINAL_BRANCH=$(git -C "<REPO_ROOT>" rev-parse --abbrev-ref HEAD)
```

Create the temp branch at the current HEAD and stage everything (committed
changes, uncommitted edits, and new untracked files):

```bash
git -C "<REPO_ROOT>" checkout -b "$TEMP_BRANCH"
git -C "<REPO_ROOT>" add -A
git -C "<REPO_ROOT>" commit --allow-empty -m "sync: temp snapshot for remote execution"
```

If `git commit` reports "nothing to commit", the branch still exists at HEAD
and the push in the next step will carry the committed state.

### 4d. Push temp branch to the remote origin

Push the temp branch so the remote cluster can fetch it:

```bash
git -C "<REPO_ROOT>" push origin "$TEMP_BRANCH"
```

### 4e. Fetch and check out temp branch on the remote cluster

On the remote, fetch all refs from origin (so the temp branch pushed in 4d
becomes visible) and check it out by name. This is the **only** sanctioned
way to land local code on the remote — never substitute `repo_branch` for
`$TEMP_BRANCH` here:

```bash
<ssh_cmd> <ssh_host> "cd <remote_repo_path> && \
    git fetch origin --prune && \
    git checkout -B $TEMP_BRANCH origin/$TEMP_BRANCH"
```

`-B` (force-create-or-reset the local branch from the remote ref) ensures
the remote working tree is reset to exactly the snapshot that was pushed
in 4d, even if a prior session left a same-named branch around. After this
step, the remote working tree is byte-for-byte identical to the full local
state (committed history + uncommitted edits + new files).

`remote_cwd` is updated to `<remote_repo_path>` so that all subsequent recipe
steps operate inside the cloned repository.

### 4f. Delete the temp branch (local and remote)

After the remote checkout succeeds, immediately delete the temp branch from
the local repo and the remote origin, then restore the original local branch:

```bash
git -C "<REPO_ROOT>" checkout "$ORIGINAL_BRANCH"
git -C "<REPO_ROOT>" branch -D "$TEMP_BRANCH"
git -C "<REPO_ROOT>" push origin --delete "$TEMP_BRANCH"
```

The remote cluster keeps the checked-out working tree from the deleted branch
— only the ref is removed, not the files.

### 4g. Copy build scripts to remote node

The skill ships two build scripts in `scripts/` alongside this SKILL.md.
Copy them to a dedicated directory on the remote so they are accessible
from inside the SLURM container:

```bash
REMOTE_SCRIPTS_DIR="<default_user_root_dir>/trtllm-build-scripts"
SKILL_SCRIPTS_DIR="<LOCAL_SKILL_DIR>/scripts"   # local path to this skill's scripts/

<ssh_cmd> <ssh_host> "mkdir -p ${REMOTE_SCRIPTS_DIR}/log"

<ssh_cmd> <ssh_host> "cat > ${REMOTE_SCRIPTS_DIR}/build.slurm" < "${SKILL_SCRIPTS_DIR}/build.slurm"
<ssh_cmd> <ssh_host> "cat > ${REMOTE_SCRIPTS_DIR}/build.sh"    < "${SKILL_SCRIPTS_DIR}/build.sh"
<ssh_cmd> <ssh_host> "chmod +x ${REMOTE_SCRIPTS_DIR}/build.slurm ${REMOTE_SCRIPTS_DIR}/build.sh"
```

`<LOCAL_SKILL_DIR>` is the directory containing this SKILL.md file
(`.claude/skills/trtllm-remote-slurm-executor/`).

---

## Step 5: Build on Remote SLURM Node

Submit a SLURM batch job that compiles TensorRT-LLM on the remote cluster
inside the official container. Wait for it to complete before proceeding to
job submission.

### 5a. Read the container image

The container image is resolved by `trtllm-case-executor` (Step 3) and
forwarded in `job_spec.json` as `container_image`. Read it from there —
**do not grep `current_image_tags.properties` and do not re-derive from
the cluster config**. case-executor is the single source of truth for the
image URI; this skill only consumes it.

Convert `urm.nvidia.com/...` style URLs to `urm.nvidia.com#...` at use-time
to avoid enroot credential lookup issues (replace the first `/` after the
hostname with `#`):

```bash
ENROOT_IMAGE=$(echo "<container_image>" | sed 's|^urm.nvidia.com/|urm.nvidia.com#|')
```

Use `<ENROOT_IMAGE>` as the value passed to `--container-image=` in the
build sbatch below and in every `srun`/`sbatch` recipe later in this skill.
The canonical (un-rewritten) URI in `job_spec.json` is unchanged; the
rewrite is purely a transport adaptation for enroot.

### 5b. Derive build architecture

Map `device_type` (from the env-check output) to the `build_wheel.py` `-a` flag:

| `device_type` | `-a` value |
|------------|------------|
| B200, GB200, GB300, B300 | `100-real` |
| H100, H200 | `90-real` |
| L40S | `89-real` |
| A100 | `80-real` |

Default to `100-real` if `device_type` is unknown.

### 5c. Submit the build job

```bash
<ssh_cmd> <ssh_host> "sbatch \
    --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --partition=<partition> \
    --account=<account> \
    --job-name=<account>-trtllm.build \
    --time=02:00:00 \
    ${REMOTE_SCRIPTS_DIR}/build.slurm \
    <ENROOT_IMAGE> <default_user_root_dir> ${REMOTE_SCRIPTS_DIR} <remote_repo_path> \
    --trt_root /usr/local/tensorrt --benchmarks -a \"<arch>\" --nvtx"
```

Capture the job ID from `sbatch` stdout (format: `Submitted batch job <id>`).

If `gpus_per_node` is set in the cluster config, add
`--gpus-per-node=<gpus_per_node>` to the `sbatch` flags.

### 5d. Monitor the build job

Poll every 30–60 seconds until the job leaves the queue:

```bash
<ssh_cmd> <ssh_host> "squeue -j <job_id> -o '%.18i %.9P %.30j %.8u %.2t %.10M %R'"
```

Once running, tail the log to track compile progress:

```bash
<ssh_cmd> <ssh_host> "tail -30 ${REMOTE_SCRIPTS_DIR}/log/build_<job_id>.srun.log"
```

Progress indicators in the log:
- `[XX%] Building CXX object ...` — compilation in progress
- `Linking CXX ...` — link phase
- `FAILED:` / `error:` / `fatal error:` — build failure
- `Successfully built` — success

If the job disappears from `squeue`, proceed to Step 5e.

### 5e. Verify the build

```bash
<ssh_cmd> <ssh_host> "sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed -n"
<ssh_cmd> <ssh_host> "tail -50 ${REMOTE_SCRIPTS_DIR}/log/build_<job_id>.srun.log"
```

A successful build shows exit code `0:0` and ends with
`Successfully built tensorrt_llm` or no error lines.

If the build fails, report the last 100 lines of the log to the user and
stop — do not proceed to job submission.

## Build Scripts Reference

| Script | Location | Purpose |
|--------|----------|---------|
| `build.slurm` | `scripts/build.slurm` | SLURM batch script — launches container, calls `build.sh` |
| `build.sh` | `scripts/build.sh` | Runs inside container — calls `build_wheel.py` with defaults |

Both scripts accept the same positional arguments:
`<container_image> <mount_dir> <scripts_dir> <repo_dir> [build_wheel_args...]`

---

## Recipes

Pick the recipe that matches your workflow. Each recipe is self-contained —
copy it, substitute the `<placeholders>` from config, and run.

**Placeholders** (all from `slurm-cluster-<name>.md` unless noted):

| Placeholder | Source |
|-------------|--------|
| `<ssh_host>`, `<account>`, `<partition>`, `<mounts>` | Config |
| `<image>` | `container_image` from `job_spec.json` (resolved by `trtllm-case-executor` Step 3). Apply enroot rewrite (`/` → `#`) at use-time per Step 5a — substitute the rewritten `<ENROOT_IMAGE>` here. Do not re-resolve from the cluster config or `current_image_tags.properties`. |
| `<job_name>` | Pre-resolved by `trtllm-case-executor` and forwarded as `job_name` in `job_spec.json`. Substitute directly into `-J <job_name>`; do not reconstruct from `account` / `subproject` / `detail`. |
| `<nodes>` | `node_count` from `job_spec.json` (computed by `trtllm-case-executor`). Do not recompute. |
| `<gpus>` | `required_devices_per_node` from `job_spec.json` — probe only if absent (see bottom of this section) |
| `<time>` | Walltime in HH:MM:SS |

**Job naming:** Every recipe includes `-J <job_name>` using the
pre-resolved `job_name` value from `job_spec.json` (built by
`trtllm-case-executor`). This is required — without `-J`, SLURM emits
warnings and jobs are hard to identify in `squeue`. Do not reconstruct
the name from `account` / `subproject` / `detail` here; that convention
lives in `trtllm-case-executor`.

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
<ssh_cmd> <ssh_host> "srun -A <account> -p <partition> -N 1 --ntasks-per-node=1 \
  -J <job_name> -t <time> \
  --container-image=<image> --container-mounts=<mounts> \
  <command>"
```

---

### Recipe 2: MPI workload (srun)

MPI applications across nodes: NCCL tests, HPCX benchmarks. The binary must
be MPI-aware (e.g., `all_reduce_perf_mpi`, not `all_reduce_perf`).

```bash
<ssh_cmd> <ssh_host> "srun -A <account> -p <partition> -N <nodes> \
  --ntasks-per-node=<gpus> --mpi=pmix \
  -J <job_name> -t <time> \
  --container-image=<image> --container-mounts=<mounts> \
  <command>"
```

---

### Recipe 3: Distributed training (srun)

Framework-launched distributed training (PyTorch DDP, DeepSpeed). SLURM starts
`<nodes> × <gpus>` processes. Pyxis/enroot sets `RANK`, `LOCAL_RANK`, and
`WORLD_SIZE`. Each process runs the training script directly.

```bash
<ssh_cmd> <ssh_host> "srun -A <account> -p <partition> -N <nodes> \
  --ntasks-per-node=<gpus> \
  -J <job_name> -t <time> \
  --container-image=<image> --container-mounts=<mounts> \
  python train.py"
```

---

### Recipe 4: Launcher-based training (srun)

One process per node, an in-container launcher forks locally (torchrun,
deepspeed launcher).

```bash
<ssh_cmd> <ssh_host> "srun -A <account> -p <partition> -N <nodes> \
  --ntasks-per-node=1 \
  -J <job_name> -t <time> \
  --container-image=<image> --container-mounts=<mounts> \
  torchrun --nproc-per-node=<gpus> train.py"
```

---

### Recipe 5: Batch job (sbatch)

Fire-and-forget background job:

```bash
<ssh_cmd> <ssh_host> "sbatch -A <account> -p <partition> -N <nodes> \
  -J <job_name> -t <time> \
  --container-image=<image> --container-mounts=<mounts> \
  <script-path>"
```

Check status:

```bash
<ssh_cmd> <ssh_host> "squeue -j <jobid>"
<ssh_cmd> <ssh_host> "sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed -n"
```

---

### Recipe 6: Multi-command session (salloc + srun)

Interactive development with persistent container state. Use this when you need
multiple commands on the same allocation — installing packages, compiling,
running experiments iteratively.

**Step 1: Allocate via tmux**

tmux keeps the allocation alive on the frontend — invisible plumbing.

```bash
<ssh_cmd> <ssh_host> "tmux new-session -d -s trtllm-<id> \
  'salloc -A <account> -p <partition> -N <nodes> \
  -J <job_name> -t <time>'"
```

**Step 2: Check allocation status**

Wait ~30 seconds for the scheduler, then get job ID and status:

```bash
<ssh_cmd> <ssh_host> "squeue -u <user> -h -o '%i %T %S' --state=RUNNING,PENDING"
```

If RUNNING → proceed to Step 3.

If PENDING → follow the **walltime negotiation flow** (see Walltime strategy
section). Only proceed to Step 3 once the job is RUNNING.

**Step 3: Create a persistent container**

Import the image once. This creates a named container on each node that
persists across `srun` calls (container persistence):

```bash
<ssh_cmd> <ssh_host> "srun --jobid=<jobid> --ntasks-per-node=1 \
  --container-image=<image> --container-name=trtllm-<jobid> true"
```

**Container packages:** If `packages` is defined in config, install them
using the exact names listed — do not guess or substitute package names:

```bash
<ssh_cmd> <ssh_host> "srun --jobid=<jobid> --ntasks-per-node=1 \
  --container-name=trtllm-<jobid> --container-writable \
  pip install <packages>"
```

Only install when the workload actually needs them — not for simple
commands like `nvidia-smi` or `hostname`.

**Step 4: Run commands**

Use `--container-name` (without `--container-image`) to reuse the persistent
container. All in-container state is preserved between commands.

Single-process command:

```bash
<ssh_cmd> <ssh_host> "srun --jobid=<jobid> \
  --container-name=trtllm-<jobid> --container-mounts=<mounts> \
  <command>"
```

Per-node operation (mkdir, pip install) — use `--ntasks-per-node=1`:

```bash
<ssh_cmd> <ssh_host> "srun --jobid=<jobid> --ntasks-per-node=1 \
  --container-name=trtllm-<jobid> --container-mounts=<mounts> \
  <command>"
```

MPI workload within the allocation:

```bash
<ssh_cmd> <ssh_host> "srun --jobid=<jobid> \
  --ntasks-per-node=<gpus> --mpi=pmix \
  --container-name=trtllm-<jobid> --container-mounts=<mounts> \
  <command>"
```

Distributed training within the allocation:

```bash
<ssh_cmd> <ssh_host> "srun --jobid=<jobid> \
  --ntasks-per-node=<gpus> \
  --container-name=trtllm-<jobid> --container-mounts=<mounts> \
  python train.py"
```

Add `--container-writable` if the workload writes to container paths outside
of mounted directories (e.g., `pip install` into site-packages).

**Step 5: Release the allocation**

Clean up the named container, then kill tmux. If the allocation has already
expired (walltime exceeded), the `srun` will fail — that is fine, the container
is already gone. Just kill the tmux session.

```bash
<ssh_cmd> <ssh_host> "srun --jobid=<jobid> --ntasks-per-node=1 \
  bash -c 'enroot remove -f trtllm-<jobid> 2>/dev/null; true'" 2>/dev/null
<ssh_cmd> <ssh_host> "tmux kill-session -t trtllm-<id>"
```

---

### Probing GPUs per node

If you do not know `<gpus>`, probe once and record in cluster notes:

```bash
<ssh_cmd> <ssh_host> "srun --jobid=<jobid> --ntasks-per-node=1 \
  --container-name=trtllm-<jobid> bash -c 'nvidia-smi -L | wc -l'"
```

## Cluster-Specific Flags

After substituting recipe placeholders, apply the cluster-specific flags
returned by the `slurm-cluster-info` skill in Step 1:

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
<ssh_cmd> <ssh_host> "cat > <remote_cwd>/script.py" << 'PYEOF'
import torch
# ... file content ...
PYEOF
```

For multiple files, repeat the pattern — each file is one SSH command.

### Remote → Local (reading results)

Read remote files via SSH stdout:

```bash
<ssh_cmd> <ssh_host> "cat <remote_cwd>/output.json"
```

For large files, read only what you need:

```bash
<ssh_cmd> <ssh_host> "tail -n 100 <remote_cwd>/slurm-<jobid>.out"
<ssh_cmd> <ssh_host> "head -n 50 <remote_cwd>/results.csv"
```

### When to use scp

Fall back to `scp` only for binary files or large directories that cannot be
piped through SSH:

```bash
<scp_cmd> <ssh_host>:<remote_cwd>/profile.nsys-rep .    # binary profile
<scp_cmd> -r local_dir/ <ssh_host>:<remote_cwd>/        # whole directory
```

**Decision rule:** text files → SSH pipe. Binary files or directories → scp.
Never use rsync or sshfs — not reliably available on HPC clusters.

## Rules

### Never read full remote files

Remote output can be gigabytes. Always use `tail`, `head`, or `grep`:

```bash
<ssh_cmd> <ssh_host> "tail -n 100 <remote_cwd>/output.log"
<ssh_cmd> <ssh_host> "head -n 50 <remote_cwd>/script.py"
<ssh_cmd> <ssh_host> "grep -n 'error\|Error\|ERROR' <remote_cwd>/output.log"
```

Never use `cat` on remote files unless you are certain the file is small.

### Never run heavy computation on the frontend

The frontend node is shared. Only use it for SLURM commands, lightweight file
operations, and tmux management.

### Check before creating

Before creating resources, check if they already exist:
- SSH ControlMaster: `ssh -O check <ssh_host> 2>&1 || <ssh_cmd> -O check <ssh_host>` (probe both the user's default socket and the trtllm-managed socket from Step 2)
- tmux sessions: `<ssh_cmd> <ssh_host> "tmux has-session -t trtllm-<id> 2>/dev/null && echo exists"`
- Existing allocations: `<ssh_cmd> <ssh_host> "squeue -u <user> -h -o '%i %j' --state=RUNNING"`

### Clean up when done

- Remove named containers: `enroot remove -f` (see Recipe 6, Step 5)
- Kill tmux sessions: `<ssh_cmd> <ssh_host> "tmux kill-session -t trtllm-<id>"`
- Cancel unneeded jobs: `<ssh_cmd> <ssh_host> "scancel <jobid>"`

### Walltime strategy

- **Production (sbatch, srun):** Tight walltime with ~20% margin — gets
  scheduled sooner.
- **Dev/debug (salloc):** Use **walltime negotiation** — submit with max
  walltime, then reduce if the queue wait is too long:

**Walltime negotiation flow:**

1. Submit `salloc` with **max walltime** for the partition.
2. Wait ~30 seconds for the scheduler to process, then check:
   ```bash
   <ssh_cmd> <ssh_host> "squeue -j <jobid> -h -o '%i %T %S'"
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
      <ssh_cmd> <ssh_host> "scontrol update JobId=<jobid> TimeLimit=<new_time>"
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

## Hang Detection Monitoring

When monitoring a running job (polling via `squeue`), also check the remote log
file every 60 seconds for the pattern `hang detected` (case-insensitive). This
catches GPU hangs, NCCL deadlocks, and other stalls that keep the job running
indefinitely without making progress.

**Check for hang (run alongside each `squeue` poll once job is RUNNING):**

```bash
<ssh_cmd> <ssh_host> "grep -qi 'hang detected' <remote_work_dir>/slurm-<jobid>.out 2>/dev/null && echo HANG_DETECTED"
```

Also check the `.err` file if the output file has no match:

```bash
<ssh_cmd> <ssh_host> "grep -qi 'hang detected' <remote_work_dir>/slurm-<jobid>.err 2>/dev/null && echo HANG_DETECTED"
```

**If `HANG_DETECTED` is printed:**

1. Immediately cancel the job:
   ```bash
   <ssh_cmd> <ssh_host> "scancel <jobid>"
   ```
2. Read the last 200 lines of the log file for context:
   ```bash
   <ssh_cmd> <ssh_host> "tail -n 200 <remote_work_dir>/slurm-<jobid>.out"
   ```
3. Report status as `HANG_DETECTED` with the relevant log lines. Do not
   continue polling — the job is terminated.

## Recovering from Failures

**SSH connection drops:** Re-run SSH preflight (Step 2). tmux-held allocations
survive disconnects.

**Job stuck in PENDING:**

```bash
<ssh_cmd> <ssh_host> "squeue -j <jobid> -o '%i %T %r %S'"
```

Check the reason (`%r`) and estimated start (`%S`). If wait > 10 minutes,
follow the **walltime negotiation flow** (see Walltime strategy). If already
negotiated and still stuck, suggest alternatives: backfill partition, fewer
nodes, or a different cluster.

**Job fails:** Read last lines, never dump full output:

```bash
<ssh_cmd> <ssh_host> "tail -n 100 <remote_cwd>/slurm-<jobid>.out"
<ssh_cmd> <ssh_host> "tail -n 50 <remote_cwd>/slurm-<jobid>.err"
```

**Hang detected:** If the log file contains `hang detected`, the job is
auto-cancelled (see Hang Detection Monitoring section above). Report
`HANG_DETECTED` with the last 200 lines of log context.
