---
name: trtllm-test-script-builder
description: >-
  Build Slurm scripts or Docker commands for TensorRT-LLM workloads. Resolves
  all parameters (docker image, mounts, parallelism,
  MPI mode), generates the complete script from Category templates, and writes
  both the script and a job_spec.json manifest to the work directory.
---

# TensorRT-LLM Script Builder

Resolve parameters and generate execution scripts for all workflow types.

## Input (from orchestrator prompt)

The orchestrator passes these fields. Not all fields are present for every workflow.

| Field | Description | Required |
|-------|-------------|----------|
| `workflow_type` | `pytest`, `eval`, `bench`, `custom`, or `perf_sanity` | Always |
| `work_dir` | Absolute path to local work directory (already created) | Always |
| `skill_dir` | Absolute path to the test-script-builder skill directory (`.claude/skills/trtllm-test-script-builder`) | Always |
| `repo_root` | Absolute path to the TensorRT-LLM repo root | Always |
| `test_cmd` | Pytest command string | pytest |
| `eval_cmd` | `trtllm-eval` command string, or `evaluation`, `accuracy`, `score`, `dataset` | eval |
| `bench_cmd` | `trtllm-bench` command (may chain `prepare-dataset` and `throughput`/`latency` with `&&`). Auto-parses `--tp`/`--pp`/`--ep` from the last `trtllm-bench` invocation. | bench |
| `custom_cmd` | Custom command string, `test command`, non-pytest command | custom |
| `custom_script` | Path to custom script file | custom (optional) |
| `custom_env` | Space-separated `KEY=VALUE` pairs | custom (optional) |
| `perf_config_yaml` | Path to pre-built perf-sanity or benchmark config YAML | perf_sanity (required) |
| `perf_test_name` | Server config name for aggregated mode | pytest (optional) |
| `model_name` | Short model name (auto-derived if not provided) | Optional |
| `required_devices` | Number of GPUs needed (auto-derived if not provided) | Optional |
| `node_count` | Number of Slurm nodes (auto-derived if not provided) | Optional |
| `partition` | Slurm partition. **For slurm scenarios, this MUST be the value resolved by `trtllm-case-executor` Step 2.5 against the cluster (validated to exist in `slurm_env.partitions[].name`).** Do not detect or default it here. | Required (slurm) |
| `account` | Slurm account. **For slurm scenarios, this MUST be the value resolved by `trtllm-case-executor` Step 2.5 against the cluster (validated to exist in `slurm_env.accounts[]`).** Do not detect or default it here. | Required (slurm) |
| `slurm_env` | The full parsed JSON from Step 2.5 (`accounts[]`, `partitions[]`, `default_account`, `default_partition`, `errors[]`). Used to populate the rendered `partition`/`account` and to forward into `job_spec.json` for downstream consumers. | Required (slurm) |
| `time_limit` | Slurm time limit (default: `02:00:00`) | Optional |
| `llm_models_root` | Models root inside container (default: `/scratch.trt_llm_data/llm-models/`) | Optional |
| `checkpoint_path` | Host path to model checkpoint | Optional |
| `models_path` | Host path to LLM models directory | Optional |
| `device_type` | Required GPU/device hardware type | Optional |
| `execution_scenario` | `local_docker`, `local_slurm`, or `remote_slurm` | Always |
| `cluster_config` | Parsed cluster config fields (`container_image`, `mounts`, `gpus_per_node`, `partition`, `account`; remote also includes `ssh_host`, `user`, `remote_cwd`). For both `local_slurm` (known clusters) and `remote_slurm`, fetched from `slurm-cluster-info` by the orchestrator. | local_slurm, remote_slurm |
| `persistent_mode` | `true` or `false`. When true, add persistent-mode fields to `job_spec.json` for the local-slurm-executor's allocation reuse. Default: `true` for `local_slurm`. | Optional |

## Procedure

### Step 1: Resolve Docker Image

1. If `cluster_config.container_image` is set, use it directly — skip the remaining steps.
2. Search `<repo_root>` for an image-tags properties file (e.g., under `jenkins/` or a similar CI config directory). Extract the relevant key based on arch (see step 3).
3. Determine compute node architecture:

   **Local Docker** (`execution_scenario == local_docker`):
   - Run `uname -m` via Bash tool on the local machine.

   **Local Slurm** (`execution_scenario == local_slurm`):
   - For each relevant partition (use `cluster_config.partition`; if multiple partitions are known from `cluster_config`, check each):
     1. Query a representative node's CPU arch:
        ```bash
        NODE=$(sinfo -p <partition> -h -o "%n" | head -1)
        scontrol show node $NODE | awk -F= '/^ *Arch=/{print $2; exit}'
        ```
     2. Query whether the partition has GPU GRES configured:
        ```bash
        sinfo -p <partition> -h -o "%G" | head -1
        ```
   - Collect a **partition → {arch, gres}** mapping from the results.

   **Remote Slurm** (`execution_scenario == remote_slurm`):
   - For each relevant partition, run both queries over SSH:
     1. CPU arch:
        ```bash
        ssh <cluster_config.ssh_host> \
          'NODE=$(sinfo -p <partition> -h -o "%n" | head -1); scontrol show node $NODE | awk -F= "/^ *Arch=/{print \$2; exit}"'
        ```
     2. GPU GRES:
        ```bash
        ssh <cluster_config.ssh_host> 'sinfo -p <partition> -h -o "%G" | head -1'
        ```
   - Collect a **partition → {arch, gres}** mapping from the results.

   **Arch → properties key mapping** (apply to each partition's result):

   | `Arch=` value | Properties key |
   |---------------|----------------|
   | `aarch64` | `LLM_SBSA_DOCKER_IMAGE` |
   | `x86_64` | `LLM_DOCKER_IMAGE` |

   Use the arch for `cluster_config.partition` (the target partition for this job) to select the Docker image. If `scontrol`/`sinfo` fails or returns no output, fall back to the `slurm-cluster-info` skill to look up the cluster's CPU arch (Grace-Blackwell clusters → `aarch64`; Viking/SXM clusters → `x86_64`).

   **GRES interpretation** — applied in Step 3:
   - `gres` starts with `gpu:` (e.g., `gpu:4`, `gpu:4(IDX:0-3)`) → partition requires `--gres`
   - `gres` is `(null)` → partition does not use GRES scheduling; omit `--gres` from the script

### Step 2: Resolve Container Mounts

Build comma-separated `<CONTAINER_MOUNTS>`:

**Resolve models host path** — evaluate rules in priority order and stop at the first match:

1. **`checkpoint_path` provided** → use its **parent directory** as the models host path. Construct one mount entry: `<parent_of_checkpoint_path>:/scratch.trt_llm_data/llm-models/`. This takes priority over `cluster_config.mounts`.
2. **`models_path` provided** → use that as the models host path. Construct: `<models_path>:/scratch.trt_llm_data/llm-models/`.
3. **`cluster_config.mounts` available** → use all entries verbatim (pre-configured for the cluster; includes the models directory and any symlink targets). Do **not** construct a separate models mount entry.
4. **Fallback** → use the default path `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/common/llm-models-repo/llm-models` and construct: `<default>:/scratch.trt_llm_data/llm-models/`.

**Home directory**: For local, `$HOME`. For remote, resolve from cluster config or `ssh <host> 'echo $HOME'`.

**Project path**: For local, `<repo_root>`. For remote, `cluster_config.remote_cwd`.

**Additional mounts** (custom workflow only): If `custom_script` parent directory is not covered by existing mounts, add it.

Assemble: `<home>:<home>,<project>:<project>,<models_mount or cluster_config.mounts entries>[,<extra>:<extra>]`

### Step 3: Resolve Parallelism & Slurm Parameters

**By workflow type:**

**Pytest**:
- Detect sub-type: perf-sanity vs non-perf-sanity (check if `test_cmd` contains `perf-sanity` or a perf sanity test file path, or if `perf_config_yaml` is provided)
- Non-perf: inspect test file for `skip_less_device(N)`, `@pytest.mark.parametrize` GPU markers → `required_devices`. If not derivable, use `required_devices` input (default: 1).
- Non-perf: pytest is always single-node. If `required_devices` > `cluster_config.gpus_per_node` (or `available_gpus` from env-check if no cluster config), **stop and report an error** — pytest does not support multi-node execution.
- Non-perf: **Hard limit — `required_devices` must be `<= 8`.** If `required_devices > 8`, stop and report an error: `required_devices = N exceeds the maximum of 8 GPUs for pytest.`
- Non-perf: **Hard limit — `node_count` must be `<= 1`.** Pytest is always single-node; `node_count` is always set to `1`. This is a redundant safety check — if any derivation produces `node_count > 1`, stop and report an error.
- Perf-sanity: all Slurm parameters (nodes, ntasks-per-node, gpus_per_node) are derived by `scripts/submit.py` exclusively from the config YAML hardware section — do not use `cluster_config.gpus_per_node`, user input, or any default value

**Eval**:
- Parse from `eval_cmd`: `--tp_size N` (default: 1), `--pp_size N` (default: 1), `--gpus_per_node N` (overrides cluster config)
- `world_size` = `tp_size * pp_size`; `required_devices` = `world_size`
- **Single-node only**: `trtllm-eval` does not support multi-node execution. If `world_size > gpus_per_node` (i.e., the job would need more than one node), **do not generate a Category 5 script** — fall back to the perf-sanity path (Category 4) by generating a benchmark config with `accuracy.enable_accuracy_test: true` and submitting via `scripts/submit.py`. Same fallback applies to disaggregated accuracy testing.
- Auto-derive `model_name` from last component of `--model` arg (e.g., `Llama-3.1-8B-Instruct` → `llama_3_1_8b_instruct`)
- Eval task subcommand (e.g., `gsm8k`, `mmlu`) passes through as-is in `eval_cmd`
- **Model path resolution**: If `--model` is a HuggingFace model ID (contains `/` but does NOT start with `/`), resolve it to a local path under `<llm_models_root>` (default `/scratch.trt_llm_data/llm-models/`). Search subdirectories (max depth 3) for a directory matching the model name component (case-insensitive). If found, rewrite `--model` in `eval_cmd` to the local absolute path. If not found, keep the original and warn that HF download will be attempted.

**Bench** (trtllm-bench):
- Parse from the last `trtllm-bench` invocation in `bench_cmd`: `--tp N` (default: 1), `--pp N` (default: 1), `--ep N` (default: 0)
- `world_size = tp * pp`; `required_devices = world_size`
- Split `bench_cmd` on `&&` into `PREP_CMD` (the `prepare-dataset` step) and `BENCH_RUN_CMD` (the `throughput`/`latency` step). If no `&&` is present (single command only), set `PREP_CMD` to empty and `BENCH_RUN_CMD` to the full command.
- Single-node only: if `world_size > gpus_per_node`, stop and report an error — multi-node `trtllm-bench` is not supported.
- Auto-derive `model_name` from the `--model` or `--model_path` argument
- `NTASKS = world_size` (used for the benchmark run srun; the `prepare-dataset` srun always overrides to `--ntasks=1`)

**Benchmark**:
- Derived from config YAML hardware section by `scripts/submit.py` — skip manual derivation

**Custom**:
- Use `required_devices` input (default: 1)
- **Hard limit — `required_devices` must be `<= 8`.** If `required_devices > 8`, stop and report an error: `required_devices = N exceeds the maximum of 8 GPUs for custom commands.`
- **Hard limit — `node_count` must be `<= 1`.** Compute `node_count = ceil(required_devices / gpus_per_node)`. If `node_count > 1`, stop and report an error: `required_devices = N requires N nodes (gpus_per_node = M), but custom commands are limited to 1 node. Reduce required_devices to at most M.`

**Common Slurm parameters:**
- `node_count` = 1 for pytest (non-perf only, always single-node); derived from config YAML by `submit.py` for perf-sanity and benchmark; 1 for eval and bench (single-node only); `ceil(required_devices / gpus_per_node)` for custom, where `gpus_per_node` = `cluster_config.gpus_per_node` or 4
- `NTASKS` = 1 for pytest; `world_size` for eval and bench; `required_devices` for custom
- `NTASKS_PER_NODE` = 1 for pytest; `NTASKS / node_count` for other workflows
- `PARTITION` = the `partition` input — the value resolved and validated by `trtllm-case-executor` Step 2.5 against the cluster's `slurm_env.partitions[].name`. **Do not fall back to `cluster_config.partition` or any default** for slurm scenarios; if `partition` is not provided, stop and report that case-executor did not run Step 2.5 before invoking the script builder.
- `ACCOUNT` = the `account` input — the value resolved and validated by `trtllm-case-executor` Step 2.5 against the cluster's `slurm_env.accounts[]`. **Do not fall back to `cluster_config.account` or `coreai_comparch_trtllm`** for slurm scenarios; if `account` is not provided, stop and report that case-executor did not run Step 2.5 before invoking the script builder.
- `TIME_LIMIT` = `time_limit` input or `02:00:00`
- `GRES_FLAG`: look up the `gres` value for `<PARTITION>` in the partition → {arch, gres} mapping from Step 1:
  - `gres` starts with `gpu:` → `GRES_FLAG = --gres=gpu:<GPUS_PER_NODE>`
  - `gres` is `(null)` or mapping unavailable → `GRES_FLAG = ` (empty string; omit the `#SBATCH --gres` line entirely from the generated script)

**MPI mode:**
- Pytest: empty (no `--mpi`), clear MPI env vars in script (always single-node)
- Eval: `world_size > 1` → `--mpi=pmix`; `world_size == 1` → empty
- Bench (trtllm-bench): `world_size > 1` → `--mpi=pmix` on the benchmark run srun only; `world_size == 1` → empty; `prepare-dataset` srun never uses `--mpi`
- Custom single-node: empty; multi-node: `--mpi=pmix`
- Benchmark/perf-sanity: handled by `submit.py`

### Step 4: Generate Script

Select the Category template from `<skill_dir>/references/trtllm_test_template.md` and generate:

**Category 1 (Local Docker)** — `execution_scenario == local_docker`:
- Read the Category 1 template
- Substitute placeholders
- Write Docker command to `<work_dir>/docker_cmd.sh`

**Category 2 (Non-perf pytest)** — `workflow_type == pytest` and non-perf-sanity:

Run `build_slurm_script.py` to generate the script:
```bash
python3 <skill_dir>/scripts/build_slurm_script.py pytest \
  --model-name <MODEL_NAME> \
  --log-dir <LOG_DIR> \
  --partition <PARTITION> --account <ACCOUNT> --time-limit <TIME_LIMIT> \
  --nodes <NODES_NUM> --ntasks <NTASKS> --ntasks-per-node <NTASKS_PER_NODE> \
  [--gres <GRES>] \
  --docker-image <DOCKER_IMAGE> \
  --container-mounts <CONTAINER_MOUNTS> \
  --container-name <CONTAINER_NAME> \
  --project-path <PROJECT_PATH> \
  --llm-models-root <LLM_MODELS_ROOT> \
  --test-cmd "<TEST_CMD>" \
  --output <work_dir>/<model_name>_auto_test.slurm
```
- Omit `--gres` entirely when `GRES_FLAG` is empty (script omits the `#SBATCH --gres` line when `--gres` is not provided or empty)
- For remote: use remote paths for `--project-path`, `--log-dir`, `--container-mounts`

**Category 3 (Custom script)** — `workflow_type == custom`:

Resolve `CUSTOM_CMD` first: `bash <custom_script>` if `custom_script` was given, otherwise the literal `custom_cmd`. Then run:
```bash
python3 <skill_dir>/scripts/build_slurm_script.py custom \
  --model-name <MODEL_NAME> \
  --log-dir <LOG_DIR> \
  --partition <PARTITION> --account <ACCOUNT> --time-limit <TIME_LIMIT> \
  --nodes <NODES_NUM> --ntasks <NTASKS> --ntasks-per-node <NTASKS_PER_NODE> \
  [--gres <GRES>] \
  --docker-image <DOCKER_IMAGE> \
  --container-mounts <CONTAINER_MOUNTS> \
  --container-name <CONTAINER_NAME> \
  --project-path <PROJECT_PATH> \
  --custom-cmd "<CUSTOM_CMD>" \
  --custom-env "<CUSTOM_ENV>" \
  --custom-workdir <CUSTOM_WORKDIR> \
  --run-script <skill_dir>/scripts/slurm_run_custom.sh \
  --output <work_dir>/<model_name>_custom.slurm
```
- For remote: also transfer `slurm_run_custom.sh`, `slurm_install.sh`, `bash_utils.sh` to the cluster; use remote path for `--run-script`

**Category 4 (Perf-sanity / Benchmark)** — `workflow_type == perf_sanity` or pytest perf-sanity:

> Benchmark requires Slurm — local Docker is not supported.

The caller (trtllm-case-executor) always provides a pre-built `perf_config_yaml`. Config YAML generation is not done here. Use `perf_config_yaml` directly with `submit.py`.

Generate the launch script via `submit.py`. Always pass **absolute paths** for `--run-sh`, `--install-sh`, `--draft-launch-sh`:
```bash
python3 <skill_dir>/scripts/submit.py \
  <--test-list "<pytest_test_string>" | --config-file <yaml> [--test-name <name>] [--benchmark-mode <mode>]> \
  --partition <PARTITION> --account <ACCOUNT> \
  --job-name <model_name>_<perf|benchmark> \
  --image <DOCKER_IMAGE> --mounts <CONTAINER_MOUNTS> \
  --work-dir <work_dir_or_remote_work_dir> \
  --llm-src <repo_root_or_remote_cwd> \
  --llm-models-root <llm_models_root> \
  --run-sh <skill_dir_or_remote>/scripts/slurm_run.sh \
  --install-sh <skill_dir_or_remote>/scripts/slurm_install.sh \
  --draft-launch-sh <skill_dir>/scripts/<draft_template> \
  --launch-sh <work_dir>/slurm_launch.sh \
  --time <TIME_LIMIT>
```
- `--test-list`: perf-sanity with bracket string; `--config-file` + optional `--test-name`/`--benchmark-mode`: explicit config file
- Draft template: aggregated → `slurm_launch_draft.sh`; disaggregated → `slurm_launch_disagg_draft.sh`
- For remote: `--work-dir`, `--llm-src`, `--run-sh`, `--install-sh` use **remote paths**; `--draft-launch-sh`, `--launch-sh` use **local paths**

**Category 6 (Bench)** — `workflow_type == bench`:

```bash
python3 <skill_dir>/scripts/build_slurm_script.py bench \
  --model-name <MODEL_NAME> \
  --log-dir <LOG_DIR> \
  --partition <PARTITION> --account <ACCOUNT> --time-limit <TIME_LIMIT> \
  --nodes 1 --ntasks <NTASKS> --ntasks-per-node <NTASKS> \
  [--gres <GRES>] \
  --docker-image <DOCKER_IMAGE> \
  --container-mounts <CONTAINER_MOUNTS> \
  --container-name <CONTAINER_NAME> \
  --project-path <PROJECT_PATH> \
  --bench-cmd "<bench_cmd>" \
  --output <work_dir>/<model_name>_bench.slurm
```
- `<NTASKS>` = `world_size` (tp * pp); single node so `--ntasks` == `--ntasks-per-node`
- `bench_cmd` passed verbatim — the script splits on `&&` internally; if no `&&`, prep srun runs install only

**Category 5 (Eval)** — `workflow_type == eval`:

> **Single-node only.** `trtllm-eval` does not support multi-node execution. If Step 3 computes `node_count > 1` (i.e., `world_size > gpus_per_node`), **do not generate a Category 5 script** — fall back to Category 4 (perf-sanity) with `accuracy.enable_accuracy_test: true` in the benchmark config. Same fallback applies to disaggregated accuracy testing.

```bash
python3 <skill_dir>/scripts/build_slurm_script.py eval \
  --model-name <MODEL_NAME> \
  --log-dir <LOG_DIR> \
  --partition <PARTITION> --account <ACCOUNT> --time-limit <TIME_LIMIT> \
  --nodes 1 --ntasks <NTASKS> --ntasks-per-node <NTASKS> \
  [--gres <GRES>] \
  --docker-image <DOCKER_IMAGE> \
  --container-mounts <CONTAINER_MOUNTS> \
  --container-name <CONTAINER_NAME> \
  --project-path <PROJECT_PATH> \
  --eval-cmd "<EVAL_CMD>" \
  --output <work_dir>/<model_name>_eval.slurm
```
- `<NTASKS>` = `world_size` (tp_size * pp_size); MPI mode is `pmix` when `world_size > 1`, omitted for single-GPU
- The script installs any required eval dependencies and, when `world_size > 1`, prepends `trtllm-llmapi-launch` to the eval command

### Step 5: Write job_spec.json

Write `<work_dir>/job_spec.json` with all resolved values:

```json
{
  "workflow_type": "pytest|eval|custom|benchmark",
  "model_name": "<MODEL_NAME>",
  "execution_scenario": "local_docker|local_slurm|remote_slurm",
  "docker_image": "<resolved image>",
  "script_path": "<work_dir>/<script_name>",
  "script_name": "<script_name>",
  "log_file_pattern": "<MODEL_NAME>_<type>_%j.out",
  "success_patterns": ["passed", "accuracy:"],
  "failure_patterns": ["FAILED", "Error", "AssertionError"],
  "extra_files": [
    "<skill_dir>/scripts/slurm_run.sh",
    "<skill_dir>/scripts/slurm_install.sh"
  ],
  "slurm_params": {
    "partition": "<PARTITION>",
    "account": "<ACCOUNT>",
    "time_limit": "<TIME_LIMIT>",
    "nodes": 1,
    "ntasks": 4,
    "ntasks_per_node": 4,
    "gpus_per_node": 4
  }
}
```

**Persistent mode fields:** When `persistent_mode=true` and `execution_scenario=local_slurm`, add these fields to `job_spec.json` (all already resolved during script generation):

```json
{
  "persistent_mode": true,
  "container_name": "<CONTAINER_NAME>",
  "container_mounts": "<CONTAINER_MOUNTS>"
}
```

`container_name` is the resolved `<CONTAINER_NAME>` placeholder value (e.g., `llama_auto_test`, `gpt_custom`, `llama_eval`). `container_mounts` is the resolved comma-separated mount string. `docker_image` and `slurm_params` are already present in the base schema. The `.slurm` script generation is completely unchanged — persistent mode reuses the same scripts.

**Pattern selection by workflow type:**

| Workflow | `log_file_pattern` | `success_patterns` | `failure_patterns` |
|----------|-------------------|-------------------|-------------------|
| pytest | `<name>_auto_test_%j.out` | `passed` | `FAILED`, `ERROR` |
| eval | `<name>_eval_%j.out` | `accuracy:`, `score:` | `AssertionError`, `Error` |
| bench | `<name>_bench_%j.out` | `token/s`, `throughput` | `Error`, `Traceback` |
| perf_sanity | `slurm-%j.out` | `8_done_` | `FAILED`, `Error` |
| custom | `<name>_custom_%j.out` | *(none)* | `Error`, `Traceback` |

For **local Docker**, use `docker_cmd` field instead of `script_path`:
```json
{
  "docker_cmd": "<complete docker run command>",
  "log_file": "<work_dir>/<model_name>_local.log"
}
```

### Step 6: Report

Print a summary of what was generated:
```
Generated: <work_dir>/job_spec.json
Script: <work_dir>/<script_name>
Workflow: <workflow_type>
Model: <model_name>
Docker image: <image>
Devices: <required_devices> GPUs across <node_count> nodes
```

## Rules

- Always use **absolute paths** in generated scripts and job_spec.json
- For remote execution, the script must contain **remote paths** for container-workdir, mounts, and log directories — but `job_spec.json` records the **local** script path (the executor skill transfers it)
- Never execute the script — only generate it. Execution is the executor skill's job.
- If `scripts/submit.py` fails, report the error and stop
- Read `trtllm_test_template.md` for the exact template content — do not hardcode templates
- Validate that required files exist before referencing them (docker image properties, test files, config YAMLs)
