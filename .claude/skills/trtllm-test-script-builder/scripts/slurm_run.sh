#!/bin/bash

# Set up error handling
set -xEeuo pipefail
trap 'rc=$?; echo "Error in file ${BASH_SOURCE[0]} on line $LINENO: $BASH_COMMAND (exit $rc)"; exit $rc' ERR

# Aggregated mode will run install together with pytest in slurm_run.sh
# Disaggregated mode will run install separately in slurm_install.sh
if [[ -z "${DISAGG_SERVING_TYPE:-}" ]]; then
    installScriptPath="$(dirname "${BASH_SOURCE[0]}")/slurm_install.sh"
    source "$installScriptPath"
    slurm_install_setup
fi

cd $llmSrcNode/tests/integration/defs

# Turn off "exit on error" so the following lines always run
set +e

pytest_exit_code=0

# Strip MPI/SLURM env to prevent phantom MPI init for:
#   - single-node runs (all roles): trtllm-llmapi-launch manages its own multiprocessing
#   - disagg BENCHMARK/DISAGG_SERVER roles: these are single-task orchestrators in
#     a multi-node job — they must not inherit the job-level pmi2/pmix context
# Multi-node GEN/CTX workers keep the env so pmi2/pmix can coordinate across nodes.
if [ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ] || \
   [ "${DISAGG_SERVING_TYPE:-}" == "BENCHMARK" ] || \
   [ "${DISAGG_SERVING_TYPE:-}" == "DISAGG_SERVER" ]; then
    for v in ${!PMI@} ${!PMIX@} ${!MPI@} ${!OMPI@} ${!SLURM@}; do
        if [ "$v" != "SLURM_PROCID" ]; then
            unset "$v"
        fi
    done
fi

eval $pytestCommand
pytest_exit_code=$?
echo "Rank${SLURM_PROCID} Pytest finished execution with exit code $pytest_exit_code"

exit $pytest_exit_code
