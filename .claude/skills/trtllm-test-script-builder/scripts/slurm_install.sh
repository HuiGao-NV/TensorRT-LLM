#!/bin/bash

# Set up error handling
set -xEeuo pipefail
trap 'rc=$?; echo "Error in file ${BASH_SOURCE[0]} on line $LINENO: $BASH_COMMAND (exit $rc)"; exit $rc' ERR

# Source bash utilities for retry_command
source "$(dirname "${BASH_SOURCE[0]}")/bash_utils.sh"

slurm_install_setup() {
    lock_file="install_lock_job_${SLURM_JOB_ID:-local}_node_${SLURM_NODEID:-0}.lock"
    if [ "${SLURM_LOCALID:-0}" -eq 0 ]; then
        cd /tmp
        if [ -f "$lock_file" ]; then
            rm -f "$lock_file"
        fi

        echo "(Installing TensorRT-LLM and requirements) Install mode: ${INSTALL_MODE:-source}"

        # Support two installation modes: source (default) and wheel
        if [ "${INSTALL_MODE:-source}" = "wheel" ]; then
            # Wheel installation mode
            echo "Installing from wheel..."
            WHEEL_FILE=$(find "$llmSrcNode/build" -name "tensorrt_llm-*.whl" -type f 2>/dev/null | head -1)

            if [ -n "$WHEEL_FILE" ]; then
                echo "Found wheel: $WHEEL_FILE"
                retry_command pip install --retries 10 "$WHEEL_FILE"
                retry_command pip install --retries 10 -r "$llmSrcNode/requirements-dev.txt"
            else
                echo "ERROR: No wheel file found in $llmSrcNode/build, falling back to source install"
                retry_command bash -c "cd $llmSrcNode && pip install --retries 10 -e . && pip install --retries 10 -r requirements-dev.txt"
            fi
        else
            # Source installation mode (default)
            retry_command bash -c "cd $llmSrcNode && pip install --retries 10 -e . && pip install --retries 10 -r requirements-dev.txt"
        fi

        cd /tmp
        echo "(Writing install lock) Current directory: $(pwd)"
        touch "$lock_file"
    else
        cd /tmp
        echo "(Waiting for install lock) Current directory: $(pwd)"
        while [ ! -f "$lock_file" ]; do
            sleep 10
        done
    fi
    echo "Install completed"
}

# Only run when script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    slurm_install_setup
fi
