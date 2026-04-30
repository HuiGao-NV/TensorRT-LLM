# Performance Optimization Commits

Total: 283 commits

## Category Summary

- **General Performance**: 84 commits
- **Fusion**: 72 commits
- **Throughput/Latency**: 50 commits
- **Kernel Optimization**: 36 commits
- **Parallelism/Async**: 33 commits
- **Quantization Optimization**: 33 commits
- **Memory Optimization**: 8 commits
- **Cache Optimization**: 8 commits
- **Host-side Optimization**: 7 commits

## All Performance Commits

| # | Hash | Date | Category | Message |
|---|------|------|----------|--------|
| 1 | 01423ac183 | 2025-10-02 | General Performance | [None][feat] perf_metrics endpoint functionality improvement (#8005) |
| 2 | 028fc877a5 | 2025-11-20 | Fusion | [#9096][feature] Auto Deploy: configurable fused MoE backend (#9194) |
| 3 | 03b38e9fbf | 2026-02-07 | General Performance | [TRTLLM-10030][perf] avoid sync in PyTorchModelEngine when using beam search (#11341) |
| 4 | 03cdf5804f | 2026-01-16 | Kernel Optimization, Memory Optimization, Fusion | [None][fix] impl fused triton kernel for e8m0 resmooth to reduce memory footprint (#10327) |
| 5 | 040103ab56 | 2025-10-13 | Parallelism/Async | [None] [blog] Scaling Expert Parallelism in TensorRT LLM (Part 3: Pushing the Performance Boundary) (#8323) |
| 6 | 040f4c70d3 | 2025-08-27 | General Performance | [None][perf] Accelerate global scale calculations for deepEP fp4 combine (#7126) |
| 7 | 0788c5d0d6 | 2025-06-26 | General Performance | [perf] improve XQA-MLA perf (#5468) |
| 8 | 09d9878385 | 2026-01-08 | General Performance | [TRTLLM-9661][chore] Further reduce tuning time for cuteDSL nvFP4 dense gemm. (#10339) |
| 9 | 0a0f93d4a8 | 2025-10-27 | Quantization Optimization | [None][fix] Fix the performance issue of FP8 blockwise grouped GEMM when using attention DP (#8501) |
| 10 | 0b81173efa | 2025-11-11 | Fusion | [TRTLLM-9259][perf] Use torch.compile to fuse copy + layernorm within the LayerNorm module (#9052) |
| 11 | 0c19c8e96e | 2026-03-12 | Parallelism/Async, Cache Optimization | [None][test] Add e2e tests for KV cache connector async loading path (#12053) |
| 12 | 0c31502fbc | 2025-12-15 | Fusion | [None][feat] disable fused gemm for sm121 (#9916) |
| 13 | 0ce36516b9 | 2026-03-12 | Throughput/Latency, Quantization Optimization | [TRTLLM-11257][infra] Unwaive TestDeepSeekR1::test_fp8_blockscale[throughput_mtp] test case (#12059) |
| 14 | 0dc4b4e699 | 2025-08-11 | Quantization Optimization | [#4403][autodeploy] Refactor: Move more transformations to new inf optimizer, Add quantization_source to factory interface (#6760) |
| 15 | 0dcf47f1c2 | 2025-05-09 | Throughput/Latency | [TRTLLM-4717][perf] Set CUDA graph max batch size and padding in throughput benchmark. (#3875) |
| 16 | 0fc0cbd1cf | 2026-03-13 | Fusion | [None][feat] Add flashinfer api for TRTLLMGenFusedMoE (#10453) |
| 17 | 0fee8cd028 | 2025-09-07 | Parallelism/Async | [TRTLLM-7153] [feat] Move stop_criteria to sample_async (#7041) |
| 18 | 10348f80fd | 2026-03-06 | Kernel Optimization, Quantization Optimization | [None][perf] Add Triton FP8 blockwise quant kernel and autotuner bucket-skip for visual gen (#11854) |
| 19 | 1074aa91b8 | 2026-03-09 | Host-side Optimization | [TRTLLM-11148][perf] _prepare_inputs host time optimization (#11704) |
| 20 | 109f27265c | 2025-09-03 | General Performance | [None][perf] Add MOE support for dynamic cluster shapes and custom epilogue schedules (#6126) |
| 21 | 1191555cce | 2025-07-07 | Throughput/Latency, Fusion | [ci] speedup fused moe tests (#5726) |
| 22 | 11d79aa875 | 2026-02-12 | Throughput/Latency | [https://nvbugs/5832481][test] Add gpt-oss-120b-Eagle3-throughput case on DGX-Spark (#11419) |
| 23 | 1260e2f33f | 2025-07-07 | General Performance | feat: Optimize TRTLLM Sampler perf single beam single step (#5550) |
| 24 | 12f339f3bf | 2025-11-14 | Throughput/Latency, Fusion | [None][fix] Fix the aux_stream in Llama4MinLatencyFusedMoE (#9035) |
| 25 | 147ad69368 | 2025-08-01 | Parallelism/Async | [None][doc] blog: Scaling Expert Parallelism in TensorRT-LLM (Part 2: Performance Status and Optimization) (#6547) |
| 26 | 1745102e72 | 2025-09-04 | Kernel Optimization, Fusion | [TRTLLM-7027][feat] Fuse d2t to logitsBitmaskKernel and fix a race condition in one-model spec (#7481) |
| 27 | 1902d73eb5 | 2025-04-14 | Fusion | fix: llama4: add an option `apply_router_weight_on_input` for in FusedMoE (#3492) |
| 28 | 196d94a419 | 2026-02-09 | General Performance | [TRTLLM-10030][perf] avoid syncs in beam search + other improvements (#11349) |
| 29 | 19a0ea363b | 2025-08-24 | General Performance | [TRTLLM-6743][feat] Optimize and refactor alltoall in WideEP (#6973) |
| 30 | 1bab9000a6 | 2025-06-26 | General Performance | perf: Optimize swizzle_sf, unswizzle_sf, reswizzle_sf (#5318) |
| 31 | 1c69aad850 | 2026-01-09 | General Performance | [TRTLLM-10309] [feat] Optimize qk rope/nope concat for DSA (#10571) |
| 32 | 1d3b98b920 | 2025-04-15 | Kernel Optimization, Quantization Optimization | perf: Optimize quantization kernels used in DeepSeek on Hopper (#3466) |
| 33 | 1d68fab49c | 2026-01-24 | Fusion | [https://nvbugs/5814215][fix] Unwaive test_trtllm_flashinfer_symbol_collision.py::test_flashinfer_fused_moe_matches_torch_moe (#10930) |
| 34 | 1e317c98c6 | 2025-04-30 | Throughput/Latency | [feat]: Allow for a settable end-of-sequence/padding token in max throughput benchmark. (#3776) |
| 35 | 1e5e71aa42 | 2025-07-25 | General Performance | Mtp optimizations round1 (#5689) |
| 36 | 1f292ff2a0 | 2025-06-25 | Kernel Optimization, Throughput/Latency | [https://jirasw.nvidia.com/browse/TRTLLM-4645] support mutliCtasKvMode for high-throughput MLA kernels (#5426) |
| 37 | 1fef88e95d | 2026-03-10 | General Performance | [None][chore] Improve sampler performance by replacing torch.where with masked_fill_ (#11949) |
| 38 | 211c44b951 | 2026-01-15 | Kernel Optimization, Fusion, Quantization Optimization | [None][feat] Adding torch ext API for FusedAddRMSNormQuant kernel (#9905) |
| 39 | 215fb20567 | 2025-04-10 | General Performance | chore : split GptExecutor tests out of gpt tests to reduce single test time (#3412) |
| 40 | 21a696b671 | 2026-03-12 | Kernel Optimization | [None][feat] Optimize the q3n decode kernel with IO read (#11344) |
| 41 | 21a93fbf9d | 2025-12-20 | Kernel Optimization | [TRTLLM-9992][perf] Enable PDL for CuteDSL kernels and overlap MoeOutputMemset (#10043) |
| 42 | 225d3a9001 | 2026-01-06 | General Performance | [None][perf] TRTLLM MoE maps to lower tuning buckets when ep>1 (#9998) |
| 43 | 22b45ff9c7 | 2025-09-25 | General Performance | [TRTLLM-7758][feat] Phi4-mm image modality inference optimization (#7918) |
| 44 | 22c1748b80 | 2025-11-13 | Kernel Optimization | [TRTLLM-8816][feat] add optimized trtllm-gen attention kernels on sm103 (#9081) |
| 45 | 255779a91d | 2025-05-29 | Fusion | Chore: fuse _merge_requests method into _fetch_new_requests method (#4689) |
| 46 | 25cd4f215e | 2025-07-31 | General Performance | [PERF] Move calculation Qwen2-VL's rotary_cos_sin to LLM worker process (#6004) |
| 47 | 264d38e6c5 | 2025-11-12 | Parallelism/Async | [TRTLLM-9175][test] ensure sampling is async (#9076) |
| 48 | 2967d299fb | 2026-01-13 | General Performance | [TRTLLM-10271][test] Add Spark QA functional and performance cases (#10564) |
| 49 | 29e63d3bc2 | 2025-09-24 | Memory Optimization, Fusion | [https://nvbugs/5532248][fix] Fix fused_moe OOM (#7931) |
| 50 | 2b58dba0f6 | 2025-10-19 | Fusion, Quantization Optimization | [https://nvbugs/5524714][fix] Fix TP sharding of fused-QKV weight scales in W4A16 AWQ (#8432) |
| 51 | 2bc2acda4f | 2026-03-10 | General Performance | [https://nvbugs/5708901][perf] reduce logprobs=0 overhead in TorchSampler (#11983) |
| 52 | 2db3d7eeba | 2026-01-20 | Parallelism/Async | [None][chore] Async Transfer Manager (#9891) |
| 53 | 2ea4077993 | 2025-07-14 | Throughput/Latency | [Model load] Fix llama min-latency model load (#5883) |
| 54 | 2f3b2a3172 | 2026-01-21 | Throughput/Latency | [None][fix] Add a timeout in MNNVL throughput to prevent hangs if one rank crashes (#9532) |
| 55 | 2f725eae08 | 2026-03-02 | Fusion, Quantization Optimization | [https://nvbugs/5775256] [fix] Reopen fp8_dsl_fused_moe ut. (#11779) |
| 56 | 31cdbdfd72 | 2026-02-12 | Throughput/Latency | [https://nvbugs/5808500][chore] Move DeepEPLowLatency tests to machines that support IBGDA with GPU handles (#11178) |
| 57 | 320195dc0d | 2025-06-02 | Fusion | [Architecture] Refactor FusedMoE (#4790) |
| 58 | 326a201473 | 2025-11-07 | Parallelism/Async | [https://nvbugs/5508536][fix] Take Over (#8627): Reintroduce: Move stop_criteria to sample_async (#7041) (#8794) |
| 59 | 32dfdfba30 | 2025-07-02 | Fusion, Quantization Optimization | feat: fuse w4a8 moe pre-quant scale on Hopper (#5613) |
| 60 | 336593cac5 | 2025-11-25 | Kernel Optimization | [None][fix] Fix topk outIndices when using vectorized_process (#9404) |
| 61 | 34a730aaf7 | 2026-01-29 | Fusion | [None][fix] Fix enable_alltoall passed to CutlassFusedMoE (#11016) |
| 62 | 361132b98a | 2026-03-02 | Throughput/Latency | [https://nvbugs/5885070][fix] fix deepeplowlatency with cutedsl moe backend (#11769) |
| 63 | 37a1bd810f | 2025-08-29 | Fusion | [https://nvbugs/5481385][fix] Fix max_seq_len in cuda graph warmup and intermediate_size in fused_moe_deepgemm (#7345) |
| 64 | 390a7fd6b1 | 2026-03-13 | General Performance | [None][feat] Qwen3.5 perf optimizations (#11581) |
| 65 | 3991aa9c72 | 2025-12-02 | Throughput/Latency | [https://nvbugs/5688388][fix] fix: Reducing num request in disagg test to speed up (#9598) |
| 66 | 3ac11a6180 | 2025-11-18 | Fusion | [#9152][fix] AutoDeploy fused_allreduce_residual_rmsnorm to support demollm mode (#9197) |
| 67 | 3b7120d60e | 2025-05-30 | General Performance | DeepSeek R1 throughut optimization tech blog for Blackwell GPUs (#4791) |
| 68 | 3c5aec19c2 | 2025-08-10 | Host-side Optimization | [#5048][enhance] AutoDeploy: Optimize prepare_inputs (#6634) |
| 69 | 3d0e38e074 | 2025-10-31 | General Performance | [None][perf] AutoDeploy optimize _get_unique_value (#8822) |
| 70 | 3dbb087292 | 2025-05-12 | Fusion | [TRTLLM-5188] fix: [AutoDeploy] update output shape of prepare_fused_mha_metadata_fake (#4199) |
| 71 | 3ddc9d2b48 | 2025-12-23 | General Performance | [https://nvbugs/5729697][fix] MNNVL Allreduce: use CUDA runtime instead of Macro to get SM version. (#10062) |
| 72 | 3fd5fafb58 | 2026-02-26 | Parallelism/Async | [https://nvbugs/5911143][fix] add async worker to MTP/Eagle3 sampler,… (#11573) |
| 73 | 3ff4f503ad | 2025-08-06 | General Performance | [None][opt] ADP schedule balance optimization (#6061) |
| 74 | 4018806742 | 2025-05-21 | Fusion | feat: large-scale EP(part 3 - refactor: FusedMoe for redundant expert) (#4495) |
| 75 | 411fa9ff87 | 2026-02-10 | Memory Optimization | [TRTLLM-10030][perf] pin host memory and batch sampler setup in beam search (#11390) |
| 76 | 421eb9e39c | 2026-02-12 | Fusion | [None][feat] Optimize NemotronH model with elementwise and nvfp4 fusion (#11273) |
| 77 | 448bb1a44f | 2025-12-08 | General Performance | [TRTLLM-9431][perf] Enable multistream for Linear Attention in Qwen3-… (#9696) |
| 78 | 4632a8642d | 2026-01-09 | General Performance | [None][doc] blog: Optimizing DeepSeek-V3.2 on NVIDIA Blackwell GPUs (#10565) |
| 79 | 46e4af5688 | 2025-12-25 | General Performance | [TRTLLM-9831][perf] Enable 2CTA with autotune for CuteDSL MoE and Grouped GEMM optimizations (#10201) |
| 80 | 4742c130db | 2025-11-25 | Throughput/Latency | [None][feat] Improve TRTLLM MoE in small hidden size throughput cases (#9377) |
| 81 | 497a07021d | 2025-11-03 | Kernel Optimization | [None][update] optimized sparse mla kernels && fix unspecified cuda launch (#8866) |
| 82 | 498b25cb60 | 2026-03-06 | Parallelism/Async | [TRTLLM-11259][perf] Parallel VAE harness and implementation for WAN (#11875) |
| 83 | 49f2f1f8eb | 2025-05-30 | Throughput/Latency | Expose new tech blog about DSR1 throughput optimization to the main R… (#4803) |
| 84 | 4ae46b6714 | 2025-06-13 | Fusion, Quantization Optimization | fix: [nvbugs/5324229] Fix broken WInt4AFP8FusedMoEMethod since FusedMoE refactor. (#4930) |
| 85 | 4b82b8b4c7 | 2025-06-17 | Kernel Optimization | [TRTLLM-5330] perf: Optimize MoE supplementary kernels for large-scale EP (#5215) |
| 86 | 4c15db0bfa | 2026-03-09 | Throughput/Latency | [https://nvbugs/5732958][bug] Fix TestLlama4MinLatency::test_llama_allclose_to_hf failure (#10191) |
| 87 | 4dc7bc525f | 2026-03-06 | General Performance | [None][fix] Refine tests/unittest/_torch/flashinfer/test_trtllm_flashinfer_symbol_collision.py to reduce jit-compile time (#11890) |
| 88 | 4e55b83101 | 2025-12-18 | Kernel Optimization | [None][perf] Add more optimization options for MOE CuteDSL finalized kernel (#10042) |
| 89 | 4f6d4da035 | 2025-12-11 | General Performance | [None][perf] Fix TPOT when `min_tokens` set (#9862) |
| 90 | 52110e8ca7 | 2026-02-23 | Quantization Optimization | [#11529][perf] Replace Python-traced FP8 quantization with optimized CUDA op in AD MoE (#11626) |
| 91 | 5339d367ce | 2025-05-30 | General Performance | [perf] Reduce the workspace size of FP4 activation scales for MoE (#4303) |
| 92 | 53491ffdb1 | 2025-11-12 | General Performance | [#9023][feat] reduce AD graph optimization time for non-participating passes (#9024) |
| 93 | 53adb3cb4e | 2025-03-25 | Parallelism/Async, Cache Optimization | test: waive flaky test_kv_cache_event_async_api (#3062) |
| 94 | 55fed1873c | 2025-10-17 | General Performance | [None][chore] AutoDeploy: cleanup old inference optimizer configs (#8039) |
| 95 | 572551b586 | 2025-09-03 | General Performance | [None][perf] Autotune TRT-LLM Gen MoE when using CUDA graphs (#7285) |
| 96 | 5a99c9734d | 2025-11-25 | General Performance | [TRTLLM-8777][feat] Update DeepGEMM to the latest commit to include optimizations for DeepSeek-v3.2 (#9380) |
| 97 | 5ddeaf9990 | 2026-02-27 | Kernel Optimization, Quantization Optimization | [None][perf] Vectorize quantize_fp8_blockwise with CUDA kernel (#11724) |
| 98 | 5e272eef81 | 2025-03-26 | General Performance | feat : reduce trt engine build time in testing (#3014) |
| 99 | 5ef65872a3 | 2025-11-07 | Fusion | [None][fix] type annotations in fuse_input_embeds (#8976) |
| 100 | 5f737b8dbe | 2025-10-28 | Kernel Optimization, Quantization Optimization | [None][perf] Use fp8 quant kernel in DS3.2 indexer module (#8701) |
| 101 | 607bf4c395 | 2025-07-08 | Throughput/Latency | Doc: Add llama4 Maverick eagle3 and max-throughput and low_latency benchmark guide (#5810) |
| 102 | 6151a4c9d6 | 2025-11-17 | General Performance | [None][feat] Add simple optimizations for MTP 2-model (#9176) |
| 103 | 6157f30b06 | 2026-02-17 | Kernel Optimization, Fusion | [#11318][infra] AutoDeploy: Add fused rope kernel - triton_rope_on_interleaved_qk_inputs (#11327) |
| 104 | 61c5a53642 | 2025-07-01 | General Performance | [#5403][perf] Conditionally enable SWAP AB for speculative decoding (#5404) |
| 105 | 6313c9799c | 2025-09-17 | Kernel Optimization, Fusion | [https://nvbugs/5488582][fix] Cherry-pick 7495: Avoid unexpected Triton recompilation in DG fused_moe (#7708) |
| 106 | 64db7d27f6 | 2025-06-30 | Kernel Optimization | [feat] Optimizations on weight-only batched gemv kernel (#5420) |
| 107 | 655d0f48d0 | 2025-08-19 | Throughput/Latency | [https://nvbugs/5455140][fix] unwaive DSR1-fp4 throughput_tp8 (#7022) |
| 108 | 6711ad9cf3 | 2025-06-18 | Quantization Optimization | [TRTLLM-5589] feat: Minor optimizations for tunable FP8 batched GEMM op. (#5139) |
| 109 | 68687a9f56 | 2025-06-19 | Parallelism/Async | [WAR][nvbug/5321947] Add an async sleep to unblock event loop. (#5342) |
| 110 | 695d7a0bdd | 2026-03-03 | General Performance | [TRTLLM-9939][perf] Short-sequence MHA optimization for DSA MLA prefill (#11677) |
| 111 | 696f754ef4 | 2025-12-23 | Parallelism/Async | [None][fix] avoid implicit cudaStreamSynchronize in sample_async. (#10120) |
| 112 | 6b3242654e | 2025-06-05 | Fusion | fix: Fix broken vanilla moe since FusedMoE refactor. (#4897) |
| 113 | 6da95f29a9 | 2025-08-05 | Fusion, Quantization Optimization | [None][feat] Add support for fused gate_up_proj scales for FP8 blockwise (#6496) |
| 114 | 6e1aee6fd6 | 2025-07-16 | Kernel Optimization | [fix] Performance Optimization for MNNVL TwoShot Kernel (#5934) |
| 115 | 6e470aab72 | 2025-12-01 | General Performance | [None] [feat] Optimize the algorithm part of RocketKV (#9333) |
| 116 | 6f3acc0614 | 2026-03-10 | Parallelism/Async | [https://nvbugs/5892646][perf] Long-sequence token-parallel optimization for DSA indexer prefill (#11871) |
| 117 | 6fc6f70a68 | 2025-10-13 | Throughput/Latency | [https://nvbugs/5441729][test] Fix test_modeling_llama_min_latency.py failures (#7478) |
| 118 | 6fe89ea00f | 2025-12-19 | General Performance | [TRTLLM-9819][perf] Reuse alltoall workspace for CuteDSL MoE output (#9840) |
| 119 | 7081f254cf | 2025-11-07 | Cache Optimization | [None][perf] Add custom indexer k cache scatter op (#8960) |
| 120 | 719e82c429 | 2026-02-05 | General Performance | [TRTLLM-10030][perf] beam search (remove GPU sync + fix batching + refactor) (#11276) |
| 121 | 72ef732bcf | 2026-01-25 | Kernel Optimization | [TRTLLM-10147][perf] Balanced random MoE workload generator for CuteDSL kernel UT, autotuner and layerwise benchmark (#10279) |
| 122 | 73b8a95049 | 2025-06-27 | General Performance | feat: Use inference mode in update_requests to improve perf of TRTLLM Sampler (#5538) |
| 123 | 73fca4e0bd | 2026-03-11 | Quantization Optimization | [None][feat] Mamba optimization and mixed quantization support for nemotron-h (#11972) |
| 124 | 7588029763 | 2025-12-15 | Parallelism/Async | [None][feat] Async pp send for PPCommTorch. (#9976) |
| 125 | 763bce523b | 2026-03-03 | Throughput/Latency | [None][test] Enable DeepGemm + DeepEPLowLatency MoE test combination (#11876) |
| 126 | 77288d3671 | 2025-07-01 | Parallelism/Async | fix [nvbug5351244]: test_mpi_session submit sync/async (#5608) |
| 127 | 775c2736d9 | 2026-02-19 | Parallelism/Async, Host-side Optimization | [TRTLLM-9040][perf] Make preprocessing async (#11459) |
| 128 | 7a5e0fd300 | 2025-06-15 | Throughput/Latency | [fix] Fix Llama4 min-latency import error (#5209) |
| 129 | 7aeac97e4e | 2025-11-11 | Parallelism/Async | [https://nvbugs/5622938][fix] Use async send_requests_to_next_pp. (#9041) |
| 130 | 7b210ae9c3 | 2025-06-11 | Throughput/Latency | test: add unit tests for Llama4 min_latency code (#4980) |
| 131 | 7c8ba71b49 | 2025-10-27 | Parallelism/Async | [TRTLLM-8832][feat] fully async _select_generated_logits with tests (#8628) |
| 132 | 7ceb5e5ab6 | 2025-11-10 | Parallelism/Async, Cache Optimization | [TRTLLM-9198][perf] Add torch.compile + multi-stream support for k-cache scatter and weight scaling (#8988) |
| 133 | 7d31532850 | 2026-01-29 | General Performance | [TRTLLM-10312][perf] Improve performance of _write_finish_reasons in TorchSampler (#10459) |
| 134 | 7deefb3d2b | 2025-09-15 | Quantization Optimization | [TRTLLM-7192][feat] optimize MLA chunked prefill && support fp8 mla chunked prefill (#7477) |
| 135 | 7e033c392e | 2025-07-17 | Kernel Optimization | Feat: Add vectorized loading for finalize kernel in MoE Trtllm backend (#5919) |
| 136 | 7eee9a9d28 | 2025-04-22 | Throughput/Latency | doc: Update doc for Deepseek min latency (#3717) |
| 137 | 8039ef45d3 | 2025-06-01 | General Performance | CI: Performance regression tests update (#3531) |
| 138 | 80f261ea36 | 2026-01-09 | Parallelism/Async | [https://nvbugs/5622938][feat] Run sample_async on extra stream. (#10215) |
| 139 | 80f9989a1e | 2025-06-03 | Throughput/Latency | [enhanchment] Add beam width to low latency. (#4812) |
| 140 | 812d2ce938 | 2026-03-02 | Fusion | [#11726][feat] AutoDeploy: Fuse gemms of mixed children (#11793) |
| 141 | 81c0764012 | 2025-07-04 | Fusion, Quantization Optimization | Cherry pick "[NVBUG:5355009] Modify check for fuse_fp4_quant on SM120 (#5724) |
| 142 | 81f878c279 | 2026-01-08 | Fusion, Quantization Optimization | [https://nvbugs/5707392][fix] unwaive test_fused_moe_fp8_blockwise_wide_ep[NotEnabled] (#10428) |
| 143 | 822cb0115b | 2025-09-21 | Memory Optimization | [TRTLLM-6286] [perf] Add NoSmem epilogue schedule and dynamic cluster shape for sm10x group gemm (#7757) |
| 144 | 8282d6c1a7 | 2025-06-11 | Throughput/Latency | [fix] Fix llama4 min latency (#5117) |
| 145 | 83b36ebecd | 2025-04-17 | Fusion | Fix fused_moe fallback issue. (#3652) |
| 146 | 841608f35e | 2026-03-01 | General Performance | [None][perf] Use F.rms_norm for per-head QK normalization in visual gen (#11798) |
| 147 | 88076eecd0 | 2025-07-21 | Fusion | [fix] Fix can_use_alltoall in fused_moe_wide_ep.py (#6173) |
| 148 | 89dabf5aa1 | 2025-12-11 | Parallelism/Async | [TRTLLM-9736][feat] AsyncLLM and verl integ (#9353) |
| 149 | 8a04c05079 | 2026-01-06 | Throughput/Latency | [None][fix] Only Use Throughput Metrics to Check Regression (#10404) |
| 150 | 8cec2da375 | 2025-12-10 | Kernel Optimization, Quantization Optimization | [None][feat] Port fp4 quantization kernel optimization from FlashInfer (#9854) |
| 151 | 8e6eead6a5 | 2025-04-29 | Fusion | refactor: (part1) Add contraints doc for fusedMoe module. (#3882) |
| 152 | 90145cf557 | 2025-08-08 | Memory Optimization | [None][feat] Optimize CUDA graph memory usage for spec decode cases (#6718) |
| 153 | 908463a5f5 | 2025-06-18 | General Performance | [feat]: improve performance of XQA-MLA for sm120 (#5087) |
| 154 | 91528365a9 | 2026-01-29 | General Performance | [None][feat] Add performance alignment to layer-wise benchmarks (#11018) |
| 155 | 93a54457ac | 2025-05-26 | General Performance | [nvbugs/5274894] fix: Sort requests for functional correctness and performance (adapted from #4608) (#4621) |
| 156 | 946ffcd2eb | 2025-09-24 | General Performance | [None][ci] optimize test cases of dgx b200 (#7948) |
| 157 | 94e6167879 | 2025-04-29 | General Performance | optimize cudaMemGetInfo for TllmGenFmhaRunner (#3907) |
| 158 | 9667ea3fff | 2026-02-25 | Host-side Optimization | [#11529][perf] AD host time attention MD optimization for large context (#11624) |
| 159 | 97657bfda2 | 2025-06-14 | General Performance | optimize memset before alltoall communication (#5188) |
| 160 | 97b38ac403 | 2025-12-25 | General Performance | [None] [doc] Update IFB performance guide & GPTOSS deployment guide (#10283) |
| 161 | 97f7e12588 | 2025-07-28 | Throughput/Latency | [fix] Fix perf regression caused by MoE autotuner when using DeepEPLowLatency (#6288) |
| 162 | 985b79ca82 | 2025-09-29 | Throughput/Latency | [TRTLLM-8348][feat] Speed up concat k and copy k_nope in context phase using torch.compile (#8044) |
| 163 | 9879400479 | 2026-01-18 | General Performance | [#10642][feat] AutoDeploy: optimized canonicalize_graph utilities [1/2] (#10675) |
| 164 | 992781dc7b | 2025-12-03 | Kernel Optimization | [None][feat] update trtllm-gen nvfp4 kernels with better performance (#9510) |
| 165 | 99b98f1374 | 2025-09-06 | Fusion | [TRTLLM-7440][fix] Split `fused_input_embed` to separate out host sync (#7280) |
| 166 | 9a070ed709 | 2026-03-10 | Kernel Optimization, Fusion, Quantization Optimization | [TRTLLM-10421][perf] Add fused cat+fp8_quantize CUDA kernel for DSA indexer (#11899) |
| 167 | 9a1750c8f9 | 2025-12-14 | Kernel Optimization, Fusion | [TRTLLM-9493][noop] Refactor fusedMoeCommKernels to enable code sharing (#9922) |
| 168 | 9ae705af1b | 2025-05-23 | Fusion | perf: Add fused q_norm/k_norm/RoPE for Qwen3. (#4482) |
| 169 | 9c4432f8a4 | 2025-10-28 | Throughput/Latency | [TRTLLM-7318][feat] MnnvlThroughput AlltoAll implementation. (#7499) |
| 170 | 9c4b8f66b4 | 2025-05-28 | Fusion | feat: Integration of Fused QKNorm+RoPE. (#4611) |
| 171 | 9cb5410067 | 2025-09-09 | Fusion | [https://nvbugs/5454559][fix] handle bias term in fuse_gate_mlp (#7449) |
| 172 | 9ee33605bb | 2025-06-26 | Throughput/Latency | [TRTLLM-6019] feat: Remove cutlass min latency code from AutoTuner. (#5394) |
| 173 | a030a898d1 | 2025-05-21 | Fusion | perf: Fuse gemm setup function for SM90/SM100 MOE plugin path (#4146) |
| 174 | a0d489a8d5 | 2025-09-29 | General Performance | [TRTLLM-7728][perf] improve batched sampling perf for contiguous batches (#7908) |
| 175 | a1e03af0f4 | 2025-08-25 | General Performance | [TRTLLM-7346][fix] Improve performance of PyTorchModelEngine._get_lora_params_from_requests (#7033) |
| 176 | a20ab5cbdb | 2025-08-01 | Fusion | [https://nvbugs/5381276][fix] fix warning for fused_a_gemm (#6402) |
| 177 | a28def9020 | 2026-03-02 | General Performance | [TRTLLM-9687][feat] Improve are_stop_words performance (#11196) |
| 178 | a5768ce316 | 2026-02-11 | Host-side Optimization | [https://nvbugs/5820922][perf] Improve TorchSampler performance by reducing host overhead (#11315) |
| 179 | a5a37227d6 | 2025-12-13 | Kernel Optimization, Fusion | [None][feat] Fused kernels (qknormrope + moe routing) and two-model MTP support for glm4moe (#9852) |
| 180 | a5cfc8368f | 2025-09-18 | Parallelism/Async | [https://nvbugs/5508536][fix] Revert #7041: Move stop_criteria to sample_async (#7041) (#7796) |
| 181 | a6f2a1e918 | 2025-05-16 | Fusion, Quantization Optimization | Fix test_fused_moe_w4afp8 (#4393) |
| 182 | a891013e3c | 2025-06-13 | Cache Optimization | [feat] Optimize KV Cache Reuse for MLA (#4869) |
| 183 | a966644a71 | 2025-10-27 | Parallelism/Async | [None][fix] Change Ray submit() to use async RPC (#8636) |
| 184 | ac2ab9ba36 | 2025-05-05 | General Performance | [AutoDeploy][perf] Further optimize flashinfer backend in AutoDeploy (#4024) |
| 185 | b10704428d | 2026-01-14 | General Performance | [https://nvbugs/5787566][fix] Only keep a limited number of performance statistic data (#10569) |
| 186 | b168adba70 | 2025-04-11 | General Performance | feat: Add NVFP4 UB pattern optimization pass in torch compile (#3371) |
| 187 | b2095aa074 | 2025-09-29 | Memory Optimization, Fusion | [#4674][bugfix] AutoDeploy Fix memory leak in fuse_moe (#7844) |
| 188 | b4d17d1a4c | 2025-11-03 | General Performance | [TRTLLM-8991][test] Add Llama 3.3 70B model with different performance config (#8753) |
| 189 | b4dab23e7b | 2025-06-30 | Kernel Optimization | [TRTLLM-5965] perf: Optimize MoE sort kernels for large-scale EP (#5435) |
| 190 | b4e9669d2c | 2026-02-13 | General Performance | [None][chore] Optimize MOE export by tracing with reduced experts and expanding graph (#11504) |
| 191 | b558232ce1 | 2025-06-19 | Fusion | Refactor CutlassFusedMoE (#5344) |
| 192 | b5b83009ff | 2025-04-02 | Parallelism/Async | chore: Reenabling get_stats_async test which seems to have been fixed by recent commit (#3246) |
| 193 | b622cde5d5 | 2025-09-25 | General Performance | [None][perf] Fix the tactic sorting in TrtllmGenBatchedGemmRunner::getValidConfigIndices (#7419) |
| 194 | b6acd96616 | 2026-01-16 | Fusion | [None][fix] AutoDeploy: Fix the nvfp4 fused_moe (#10727) |
| 195 | b99c5ce8c1 | 2025-06-14 | Throughput/Latency, Fusion | Feat/ds r1 min latency opt round3, add router gemm, fused a gemm, PDL  (#4560) |
| 196 | b9b1c1368c | 2025-04-17 | Fusion | feat: Support unfused rope in MLA. (#3610) |
| 197 | ba8abeab10 | 2025-09-30 | Fusion, Quantization Optimization | [OMNIML-2336][feat] add W4A8 NVFP4 FP8 fused moe (#7968) |
| 198 | be2065755c | 2026-03-11 | Throughput/Latency | [None][fix] Enforce minimum NVSHMEM_QP_DEPTH of 128 for DeepEP low latency (#12100) |
| 199 | bf1b958f1a | 2025-08-26 | Fusion | [TRTLLM-7319][perf] Fuse slicing into MoE. (#6728) |
| 200 | bf5b2a2e0a | 2025-05-09 | Throughput/Latency | test: amend regex match for perf throughput (#4186) |
| 201 | c0cf5a3706 | 2026-03-13 | Quantization Optimization | [None][feat] Optimize 6KD fp8 blockscale gemm (#11502) |
| 202 | c12e67bb66 | 2025-12-01 | Fusion | [TRTLLM-8958][feat] and [TRTLLM-8960]: create ConfigurableMoE and support TRTLLMGenFusedMoE as backend (#9486) |
| 203 | c2d3c6cdba | 2026-02-26 | Throughput/Latency | [https://nvbugs/5884735][fix] fix deepeplowlatency with DeepGEMM (#11700) |
| 204 | c3729dbd7d | 2025-07-30 | Throughput/Latency | infra: [TRTLLM-5873] Use build stage wheels to speed up docker release image build (#4939) |
| 205 | c37531c3f7 | 2026-01-28 | Throughput/Latency | [TRTLLM-10669][fix] Fix Eagle3 draft model weight loading for throughput checkpoint (#11010) |
| 206 | c39bbb2d1a | 2026-02-25 | Kernel Optimization, Quantization Optimization | [TRTLLM-11090][perf] Improve fp8 (per-tensor) quant kernel by vectorized load/store (#11662) |
| 207 | c4da4fd462 | 2026-01-14 | Throughput/Latency | [https://nvbugs/5637220][ci] unwaive TestQwen3_235B_A22B::test_nvfp4[latency_moe_trtllm_attention_dp] (#9870) |
| 208 | c53bc19f5e | 2025-06-16 | Throughput/Latency | [infra] Make test_chunked_prefill faster (#5248) |
| 209 | c59aa8bec5 | 2025-12-28 | General Performance | [TRTLLM-9962][feat] Some optimizations for two-model spec dec (#10208) |
| 210 | c5f52ab304 | 2025-11-25 | General Performance | [TRTLLM-8376][feat] top-p optimization (removes redundant softmax) (#9411) |
| 211 | c678774c99 | 2025-04-08 | Fusion | feat: Apply the new torch-flow compatible AutoTuner to both Fused MoE and NVFP4 Linear operators. (#3151) |
| 212 | c7548ad72c | 2025-04-02 | Throughput/Latency | perf: Add optimizations for deepseek in min latency mode (#3093) |
| 213 | c7d8cc1f34 | 2026-02-23 | Kernel Optimization, Quantization Optimization | [None][perf] Use UE8M0 FP8 quant kernel for DeepGemm blockwise GEMM (#11607) |
| 214 | c8b9998acb | 2025-10-20 | Kernel Optimization | [TRTLLM-8637][feat] Optimize the routing kernel for DeepseekV3 (MoE CUTLASS backend); Add support for KimiK2 and Qwen-next (MoE TRTLLM backend) (#7761) |
| 215 | cb1d8d130f | 2026-02-13 | Host-side Optimization | [TRTLLM-10791][feat] TorchSampler general host time optimization (#11141) |
| 216 | cc16289dfe | 2026-03-07 | Fusion, Quantization Optimization | [None][feat] Optimize by fuse nvfp4_quant to layernorm_gated for mamba2_mixer (#11473) |
| 217 | cc989ea49f | 2025-04-30 | Fusion | perf: Optimise MOE prologue to use fused setup function (#3790) |
| 218 | ccc64da287 | 2025-12-23 | Fusion | [TRTLLM-9847][fix] WAR fix hanging fused allreduce. (#10087) |
| 219 | cd4e639536 | 2025-12-13 | Parallelism/Async | [None][feat] Async pp send. (#9952) |
| 220 | cea5dd1e38 | 2025-06-16 | General Performance | [TRTLLM-5835][feat] Optimized Mamba2Mixer prefill (#5128) |
| 221 | d12cb9436d | 2025-11-13 | Kernel Optimization | [None][feat] Autodeploy add triton configs and optimize mamba prefill (#9083) |
| 222 | d1ba3b8620 | 2026-03-06 | Fusion | [TRTLLM-11093][feat] add 5D A2A for fused ulysses (#11787) |
| 223 | d252101a76 | 2025-12-07 | General Performance | [OMNIML-3036][doc] Re-branding TensorRT-Model-Optimizer as Nvidia Model-Optimizer (#9679) |
| 224 | d2a04abb95 | 2025-07-28 | Throughput/Latency | [fix] Fixes to parameter usage and low latency configuration. (#6343) |
| 225 | d3f4fbb742 | 2026-01-14 | Parallelism/Async | [None][fix] Avoid write-write race for async pp send. (#10488) |
| 226 | d548b29a41 | 2026-01-24 | Parallelism/Async | [None][fix] Bugfix/mtp with async scheduler (#10941) |
| 227 | d643aef73c | 2025-08-08 | General Performance | [Perf] Improve Llama4 performance for small max_seqlen cases (#6306) |
| 228 | d6e49542bd | 2026-02-10 | Throughput/Latency, Quantization Optimization | [https://nvbugs/5848377][fix] fix deepeplowlatency with trtllm moe backend running fp8 DS_R1 (#11266) |
| 229 | d6f95a4363 | 2025-12-05 | General Performance | [None][feat] AutoDeploy: Perf optimization for Attention and rmsnorm (#9719) |
| 230 | d7087015f1 | 2025-09-26 | General Performance | [TRTLLM-8271][fix] Fix CDL overlap scheduling performance (#7971) |
| 231 | d7c51c953b | 2025-05-08 | Throughput/Latency | test: add INTEGRATION_TEST env var to speed up integration test (#3618) |
| 232 | d8b05894ee | 2025-11-19 | General Performance | [None][perf] Adjust select_alltoall_method_type. (#8950) |
| 233 | d913955952 | 2025-08-08 | Fusion | [TRTLLM-6898][feat] make fused_moe_cute_dsl work on blackwell (#6616) |
| 234 | da6cb541a2 | 2025-09-09 | Kernel Optimization | [None][feat] Optimize MLA kernels with separate reduction kernels (#7597) |
| 235 | dbad94715b | 2026-01-29 | General Performance | [None][feat] Add gRPC server for high-performance external router integration (#11037) |
| 236 | dee6644ed9 | 2025-07-08 | Parallelism/Async | feat(scaffolding): add streaming scaffolding_llm.generate_async support (#5345) |
| 237 | df0b976b99 | 2026-01-06 | Throughput/Latency, Quantization Optimization | [https://nvbugs/5785206][infra] Waive TestQwen3_30B_A3B::test_fp8[latency-torch_compile=False]. (#10441) |
| 238 | df3484ddfa | 2026-02-23 | General Performance | [#11529][perf] AD NemotronH topk router to use the model default dtype (#11623) |
| 239 | e0253ee805 | 2025-08-29 | General Performance | [None][perf] Disable Swap AB when num tokens exceeds N dimension (#7104) |
| 240 | e12868bc00 | 2025-08-27 | Fusion, Quantization Optimization | [None][fix] Remove and fuse some element-wise ops in the ds-r1-fp8 model (#7238) |
| 241 | e134a52e07 | 2025-07-04 | Memory Optimization, Throughput/Latency | Perf: reduce DeepEPLowLatency memory and time (#5712) |
| 242 | e18dacc931 | 2025-08-21 | Fusion, Cache Optimization | [#4403][refactor] Move fusion, kvcache, and compile to modular inference optimizer (#7057) |
| 243 | e405468230 | 2026-01-26 | Fusion | [TRTLLM-10048][feat] Fuse the AllGather for expert statistics required by the EPLB. (#10885) |
| 244 | e4bf29bc66 | 2025-11-04 | Throughput/Latency | [None][feat] Integrate MnnvlThroughput into TRTLLM MoE. (#8728) |
| 245 | e57d83c5dc | 2025-11-05 | Fusion | [TRTLLM-8768][chore] Fuse QK down_proj with indexer K + weight_proj for FP4 ckpt (#8771) |
| 246 | e5d4305c04 | 2025-12-04 | Fusion | [https://nvbugs/5467531][fix] Unwaive fused_moe all to all test with … (#9617) |
| 247 | e67f4da9b5 | 2025-07-30 | General Performance | [Perf]: Add residual, norm for nemotron_nas models (#6455) |
| 248 | e6b482ef47 | 2025-04-29 | Parallelism/Async | fix: change the seq_lens sync copy to an async one (#3786) |
| 249 | e8ad899f93 | 2026-03-01 | Kernel Optimization | [None][feat] TRT-LLM Gen MoE finalize kernel optimization (#11501) |
| 250 | ea3739ee62 | 2025-03-26 | Fusion | Fix: fuse message not aligned on different processes (#3067) |
| 251 | ead89a0e40 | 2025-08-12 | General Performance | [None][perf] Improve the performance of online EPLB on Hopper by better overlapping (#6624) |
| 252 | eb2d51a429 | 2025-06-01 | Throughput/Latency | [fix] Fix llama4 min-latency mode (#4810) |
| 253 | ec6b1821c7 | 2025-06-10 | Fusion, Quantization Optimization | [fix] Fix W4A8 weight loading error in WInt4AFP8FusedMoEMethod (#5026) |
| 254 | ec9cf715a2 | 2025-11-11 | General Performance | [None][feat] AutoDeploy: Perf improvement for mamba layers (#8991) |
| 255 | ed297d7c2e | 2025-11-04 | General Performance | [None][chore] Optimize perf for the RPC executor and add some profile utilities to llm-api (#8415) |
| 256 | edab7532dd | 2025-07-15 | Throughput/Latency | feat/add latency support for trtllm bench (#3730) |
| 257 | ee471df07c | 2025-08-07 | Cache Optimization | [None][chore] optimize kv cache transfer for context TEP and  gen DEP (#6657) |
| 258 | eeb555e37b | 2025-06-06 | Throughput/Latency | chore: memoize weight shuffle index to speed up weight preproc in moe_backend=TRTLLM (#4826) |
| 259 | efd503751f | 2025-11-24 | Parallelism/Async | [#9271][perf] Enable multi-stream MOE optimization in AutoDeploy (#9322) |
| 260 | f02948d956 | 2026-01-16 | Throughput/Latency | [https://nvbugs/5803813][fix] Fix llama 4 min latency (#10724) |
| 261 | f03053b4dd | 2025-08-20 | General Performance | [None][fix] update accelerate dependency to 1.7+ for AutoDeploy (#7077) |
| 262 | f0b68e4c66 | 2025-11-18 | General Performance | [None][feat] AutoDeploy: Perf improvement for small batch size (#9163) |
| 263 | f2aee0db03 | 2025-12-15 | Parallelism/Async, Host-side Optimization | [TRTLLM-9854][feat] Optimize the host overhead of _sample_async (#9935) |
| 264 | f2ebaf288a | 2025-11-21 | Kernel Optimization, Quantization Optimization | [None][feat] TRT-LLM Gen MoE optimize DeepSeek Fp8 activation kernel (#9175) |
| 265 | f39e1a8603 | 2026-02-24 | Quantization Optimization | [https://nvbugs/5846489][perf] Apply TE's FP8 per-tensor quantization (#11057) |
| 266 | f3a985ce27 | 2026-01-20 | Kernel Optimization | [TRTLLM-10296][fix] Fix the potential misaligned access due to vectorized ld/st instructions in NVLinkOneSided A2A. (#10539) |
| 267 | f3d784c6f6 | 2026-02-15 | Parallelism/Async | [#10345][perf] Enable multi-stream MOE for super. Also adds multi-stream MLA attn (#11520) |
| 268 | f512ddaeef | 2025-10-24 | Kernel Optimization, Fusion | [None][feat] add skip condition in AutoDeploy's triton fused moe kernel (#8632) |
| 269 | f5b6d453aa | 2025-05-16 | Throughput/Latency | doc： DS r1 min latency blog (#4386) |
| 270 | f6654f26a4 | 2025-10-05 | Fusion | [#5255][autodeploy] Update FuseAllreduceResidualRMSNorm to use pattern matcher utility; remove fuse_collective (#7545) |
| 271 | f670a036df | 2025-05-07 | Fusion | [Qwen3] chore: fix bug of fused_moe on tp > 1 (#4093) |
| 272 | f77aca9f2c | 2025-09-22 | General Performance | [TRTLLM-7385][feat] Optimize Qwen2/2.5-VL performance (#7250) |
| 273 | f7c597ec40 | 2025-08-22 | Fusion | [None][perf] Make finalize fusion part of the tactic selection logic (#6915) |
| 274 | f7e245668b | 2025-12-17 | General Performance | [TRTLLM-9680][perf] Optimize TRTLLMSampler log_probs performance (Core fix has been merged via #9353) (#9655) |
| 275 | f8a4cc0629 | 2025-04-04 | Throughput/Latency | perf: Add total token throughput metric. (#3212) |
| 276 | f8dd494536 | 2025-11-28 | General Performance | [None][perf] Helix: improve all-to-all perf for large CP size (#9494) |
| 277 | f98fa0cf8b | 2025-09-26 | Cache Optimization | [None][feat] Optimize kv cache transfer TEP (#7613) |
| 278 | f9a455651b | 2025-07-01 | General Performance | perf: Use tokenizers API to optimize incremental detokenization perf (#5574) |
| 279 | f9e6045f39 | 2026-02-03 | General Performance | [#11086][feat] Optimize Auto Deploy weight loading by preloading weights to CPU (#11059) |
| 280 | fac47e2826 | 2025-10-12 | Fusion | [https://nvbugs/5510879][fix] Fix pytorch & TRT-python flows fused LoRA adapter modules weight split with TP>1 (#8063) |
| 281 | fae4985797 | 2026-01-27 | Memory Optimization | [TRTLLM-9831][perf] Use TMA.RED to improve effective memory bandwidth (#10987) |
| 282 | fcda1a1442 | 2025-12-14 | Parallelism/Async | [None][fix] disable async pp send for ray cases. (#9959) |
| 283 | ff82aef99b | 2025-04-11 | Fusion | Fix the issues related to fused moe path. (#3435) |
