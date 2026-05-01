#!/bin/bash
#SBATCH --job-name=coreai_comparch_trtllm-test.dwdp_deepseek_v3_lite
#SBATCH --output=/lustre/fsw/coreai_comparch_trtllm/huig/work_dirs/dwdp_deepseek_v3_lite_20260430_143536/dwdp_deepseek_v3_lite_auto_test_%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_trtllm/huig/work_dirs/dwdp_deepseek_v3_lite_20260430_143536/dwdp_deepseek_v3_lite_auto_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --partition=gb200
#SBATCH --account=coreai_comparch_trtllm

set -euo pipefail

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Nodes: $SLURM_JOB_NUM_NODES, Tasks: $SLURM_NTASKS"

# Step 1: Install TRT-LLM (one task per node)
srun -l \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --container-image=urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-26.02-py3-sbsa-ubuntu24.04-trt10.15.1.29-skip-tritondevel-202604011104-12600 \
  --container-name=dwdp_deepseek_v3_lite_auto_test \
  --container-mounts=/home/huig:/home/huig,/lustre/fsw/coreai_comparch_trtllm/huig:/lustre/fsw/coreai_comparch_trtllm/huig,/lustre/fsw/coreai_comparch_trtllm/llm-models:/scratch.trt_llm_data/llm-models/ \
  --container-workdir=/lustre/fsw/coreai_comparch_trtllm/huig/TensorRT-LLM \
  --kill-on-bad-exit=1 \
  bash -c '
    set -euo pipefail
    echo "Node ${SLURM_NODEID:-0}: Installing TRT-LLM from source..."
    cd /lustre/fsw/coreai_comparch_trtllm/huig/TensorRT-LLM
    pip install -e . 2>&1 | tail -5
    pip install -r requirements-dev.txt 2>&1 | tail -5
    echo "Node ${SLURM_NODEID:-0}: Install complete"
  '

# Step 2: Run tests
srun -l \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --container-image=urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-26.02-py3-sbsa-ubuntu24.04-trt10.15.1.29-skip-tritondevel-202604011104-12600 \
  --container-name=dwdp_deepseek_v3_lite_auto_test \
  --container-mounts=/home/huig:/home/huig,/lustre/fsw/coreai_comparch_trtllm/huig:/lustre/fsw/coreai_comparch_trtllm/huig,/lustre/fsw/coreai_comparch_trtllm/llm-models:/scratch.trt_llm_data/llm-models/ \
  --container-workdir=/lustre/fsw/coreai_comparch_trtllm/huig/TensorRT-LLM \
  --kill-on-bad-exit=1 \
  bash -c '
    nvidia-smi
    # Single-node jobs: clear MPI/SLURM env vars to prevent false MPI init
    if [ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ]; then
      for v in ${!PMI@} ${!PMIX@} ${!MPI@} ${!OMPI@} ${!SLURM@}; do
        if [ "$v" != "SLURM_PROCID" ]; then
          unset "$v"
        fi
      done
    fi
    export LLM_MODELS_ROOT=/scratch.trt_llm_data/llm-models/
    cd /lustre/fsw/coreai_comparch_trtllm/huig/TensorRT-LLM
    pytest tests/integration/defs/accuracy/test_dwdp_disaggregated_serving.py::TestDwdpDeepSeekV3Lite::test_dwdp_accuracy -sv
  '

exit_code=$?
echo "Job $SLURM_JOB_ID finished at $(date) with exit code $exit_code"
exit $exit_code
