
cleanup_on_failure() {
    echo "Error: $1"
    scancel ${SLURM_JOB_ID}
    exit 1
}

mkdir -p $jobWorkspace
chmod +x $runScript

# Run aggregated test
echo "Starting aggregated test..."
world_size=$((totalNodes * gpusPerNodePerServer))
echo "Test output: $jobWorkspace/aggregated_run.log"
if ! srun "${srunArgs[@]}" --kill-on-bad-exit=1 \
    -N $totalNodes \
    --ntasks=$world_size \
    --ntasks-per-node=$gpusPerNodePerServer \
    $runScript &> $jobWorkspace/aggregated_run.log; then
    cleanup_on_failure "Aggregated test failed. Check $jobWorkspace/aggregated_run.log for details"
fi

echo "Aggregated test completed successfully"

# Run accuracy evaluation if configured. Mirrors the disagg accuracy block in
# slurm_launch_disagg_draft.sh. ACCURACY_CONFIG_JSON is set by submit.py when
# accuracy.enable_accuracy_test=true; accuracy_runner.py runs lm_eval against
# the trtllm-serve instance started by the aggregated perf-sanity test. The
# test is responsible for keeping trtllm-serve alive past the benchmark phase;
# otherwise this step will fail to connect.
if [ -n "${ACCURACY_CONFIG_JSON:-}" ]; then
    echo "Starting accuracy evaluation..."
    server_host=$(scontrol show hostname "$SLURM_NODELIST" | head -1)
    export DISAGG_SERVER_HOST="$server_host"
    echo "Accuracy server endpoint: ${server_host}:${DISAGG_SERVER_PORT:-8333}"
    # DISAGG_SERVING_TYPE=ACCURACY_TEST tells slurm_run.sh to skip its inline
    # install step (install already ran on the main aggregated srun).
    export DISAGG_SERVING_TYPE="ACCURACY_TEST"
    export pytestCommand="$pytestCommandAccuracy"
    if ! srun "${srunArgs[@]}" --kill-on-bad-exit=1 --overlap \
        -N 1 \
        --ntasks=1 \
        --ntasks-per-node=1 \
        $runScript &> $jobWorkspace/accuracy_test.log; then
        echo "Accuracy evaluation failed. Check $jobWorkspace/accuracy_test.log"
    else
        echo "Accuracy evaluation completed. Results in $jobWorkspace/accuracy_eval_*"
    fi
fi

echo "Total runtime: $SECONDS seconds"
