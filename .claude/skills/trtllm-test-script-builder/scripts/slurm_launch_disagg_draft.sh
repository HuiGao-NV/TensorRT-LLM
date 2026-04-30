
cleanup_on_failure() {
    echo "Error: $1"
    scancel ${SLURM_JOB_ID}
    exit 1
}

mkdir -p $jobWorkspace
chmod +x $runScript
chmod +x $installScript

# Run installation on all nodes
echo "Running installation on all nodes..."
if ! srun "${srunArgs[@]}" $installScript &> $jobWorkspace/install.log; then
    cleanup_on_failure "Failed to run installation. Check $jobWorkspace/install.log"
fi
echo "Installation completed on all nodes"

# Start gen servers
echo "Starting gen servers..."
for i in $(seq 0 $((numGenServers - 1))); do
    gen_world_size=$((nodesPerGenServer * gpusPerNodePerGenServer))
    export DISAGG_SERVING_TYPE="GEN_$i"
    export pytestCommand="$pytestCommandGENWorker"
    srun "${srunArgs[@]}" --mpi=pmix --kill-on-bad-exit=1 \
        -N $nodesPerGenServer \
        --ntasks=$gen_world_size \
        --ntasks-per-node=$gpusPerNodePerGenServer \
        $runScript &> $jobWorkspace/gen_server_$i.log &
    echo "Started gen server $i"
done

# Start ctx servers (skip if gen_only_no_context mode)
if [ "${TRTLLM_DISAGG_BENCHMARK_GEN_ONLY:-0}" != "1" ]; then
    echo "Starting ctx servers..."
    for i in $(seq 0 $((numCtxServers - 1))); do
        ctx_world_size=$((nodesPerCtxServer * gpusPerNodePerCtxServer))
        export DISAGG_SERVING_TYPE="CTX_$i"
        export pytestCommand="$pytestCommandCTXWorker"
        srun "${srunArgs[@]}" --mpi=pmix --kill-on-bad-exit=1 \
            -N $nodesPerCtxServer \
        --ntasks=$ctx_world_size \
        --ntasks-per-node=$gpusPerNodePerCtxServer \
            $runScript &> $jobWorkspace/ctx_server_$i.log &
        echo "Started ctx server $i"
    done
else
    echo "Skipping ctx servers (gen_only_no_context mode)"
fi


# Start disagg server
echo "Starting disagg server..."
export DISAGG_SERVING_TYPE="DISAGG_SERVER"
export pytestCommand="$pytestCommandDisaggServer"
srun "${srunArgs[@]}" --kill-on-bad-exit=1 --overlap \
    -N 1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    $runScript &> $jobWorkspace/disagg_server.log &
echo "Started disagg server"

# Start benchmark
echo "Starting benchmark..."
export DISAGG_SERVING_TYPE="BENCHMARK"
export pytestCommand="$pytestCommandBenchmark"
echo "Benchmark output: $jobWorkspace/benchmark.log"
if ! srun "${srunArgs[@]}" --kill-on-bad-exit=1 --overlap \
    -N 1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    $runScript &> $jobWorkspace/benchmark.log; then
    cleanup_on_failure "Benchmark failed. Check $jobWorkspace/benchmark.log for details"
fi

echo "Disagg server and benchmark completed successfully"

# Run accuracy evaluation if configured (mirrors client_cmds flow in
# examples/disaggregated/slurm/benchmark/disaggr_torch.slurm).
# ACCURACY_CONFIG_JSON is set by submit.py when accuracy.enable_accuracy_test=true.
# accuracy_runner.py (pytestCommandAccuracy) reads lm_eval task configs from it and
# runs lm_eval against the still-running disagg server.
if [ -n "${ACCURACY_CONFIG_JSON:-}" ]; then
    echo "Starting accuracy evaluation..."
    # Derive disagg server hostname: first node in the job (matches examples' GEN-0 node choice).
    disagg_server_host=$(scontrol show hostname "$SLURM_NODELIST" | head -1)
    export DISAGG_SERVER_HOST="$disagg_server_host"
    echo "Accuracy server endpoint: ${disagg_server_host}:${DISAGG_SERVER_PORT:-8333}"
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
