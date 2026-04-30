# Commit Section 9

Commits 4001 to 4500 (total: 5612)

| # | Commit Hash | Author | Date | Message |
|---|------------|--------|------|--------|
| 4001 | 3e75320fe8 | Shunkangz | 2025-07-02 | Add pd dynamic scaling readme (#5540) |
| 4002 | caf27ca0f6 | Yiteng Niu | 2025-07-02 | [chore] 2025-07-02 update github CI allowlist (#5661) |
| 4003 | 32dfdfba30 | Xiaowei Wang | 2025-07-02 | feat: fuse w4a8 moe pre-quant scale on Hopper (#5613) |
| 4004 | 7992869798 | Void | 2025-07-02 | perf: better heuristic for allreduce (#5432) |
| 4005 | 10c50515c2 | HuiGao-NV | 2025-07-02 | fix: Add back allreduce_strategy parameter into TorchLlmArgs (#5637) |
| 4006 | ba2ab5098b | Perkz Zheng | 2025-07-02 | [Bug] attention DP doesn't work with embedding TP (#5642) |
| 4007 | efef911f5e | Aurelien Chartier | 2025-07-01 | fix: add missing self. from PR #5346 (#5653) |
| 4008 | 1341ffdfaa | Po-Wei (Vincent) | 2025-07-01 | [TRTLLM-5644][infra] Update the community action to more appropriate api (#4883) |
| 4009 | fa95e402a5 | Aurelien Chartier | 2025-07-01 | feat: add LLmArgs option to force using dynamic quantization (#5346) |
| 4010 | c345f5876c | liji-nv | 2025-07-02 | [feat] Support torch compile for attention dp (#5086) |
| 4011 | f9a455651b | Kaiyu Xie | 2025-07-01 | perf: Use tokenizers API to optimize incremental detokenization perf (#5574) |
| 4012 | d68fa728d8 | Robin Kobus | 2025-07-01 | refactor: Clean up DecodingInput and DecodingOutput (#5617) |
| 4013 | 3bc703d450 | Yan Chunwei | 2025-06-27 | ci: unwaive llmapi launch test (#5281) |
| 4014 | 178fc3f655 | Emma Qiao | 2025-06-27 | [Infra][release/0.21] - waive failed tests (#5537) |
| 4015 | 48eee338bf | ixlmar | 2025-06-26 | fix: constrain grepping in docker/Makefile (#5493) |
| 4016 | 4b3f2dbb45 | ixlmar | 2025-06-26 | fix: fix regression in LOCAL_USER (#5517) |
| 4017 | 93edfea2b8 | Anurag Mukkara | 2025-06-25 | [nvbug/5354825] Fix nougat test image url (#5496) |
| 4018 | ee7fcbf20e | Yan Chunwei | 2025-06-26 | [nvbug 5273941] fix: broken cyclic reference detect (#5417) |
| 4019 | be5ddb0533 | Martin Marciniszyn Mehringer | 2025-06-25 | Fix permission for local user issues in NGC docker container. (#5373) |
| 4020 | ded203d8aa | ruodil | 2025-06-25 | test: set enable_attention_dp=True in default deepseek settings (#5461) |
| 4021 | 3789ba1d37 | Wanli Jiang | 2025-06-25 | feat: TRTLLM-5941 Upgrade xgrammar to 0.1.18 (#5364) |
| 4022 | 4ef60d5fbb | brb-nv | 2025-06-24 | nvbugs-5331031; nvbugs-5344203 - address intermittent issues with Mistral Small multimodal for BS=8 (#5453) |
| 4023 | 61213e3562 | Ivy Zhang | 2025-06-25 | tests: fix typos in qa test (#5421) |
| 4024 | 872610a048 | Martin Marciniszyn Mehringer | 2025-06-19 | doc: cherry pick #5334 (#5368) |
| 4025 | a5eff139f1 | Yan Chunwei | 2025-07-01 | [TRTLLM-5277] chore: refine llmapi examples for 1.0 (part1) (#5431) |
| 4026 | 61c5a53642 | 杨凯旋 | 2025-07-01 | [#5403][perf] Conditionally enable SWAP AB for speculative decoding (#5404) |
| 4027 | 65c2b93284 | Emma Qiao | 2025-07-01 | [Infra] - Add some timeout and unwaive a test which dev fixed (#5631) |
| 4028 | 071ad758c4 | Pamela Peng | 2025-07-01 | [https://nvbugs/5318059][test] Unwaive test (#5624) |
| 4029 | 5f77d212ef | Robin Kobus | 2025-07-01 | test: Reduce number of C++ test cases (#5437) |
| 4030 | 7a617ad1fe | danielafrimi | 2025-07-01 | feat: W4A16 GEMM (#4232) |
| 4031 | 19c56f0374 | xinhe-nv | 2025-06-30 | test: [CI] Add failed cases into waives.txt (#5582) |
| 4032 | 34212e2e36 | Vivian Chen | 2025-06-30 | [TRTLLM-6104] feat: add request_perf_metrics to triton LLMAPI backend (#5554) |
| 4033 | 7135b27284 | Stanley Sun | 2025-07-01 | rcca: test default kv_cache_reuse option for pytorch multimodal (#5544) |
| 4034 | a8cf611baa | xinhe-nv | 2025-07-01 | test: [CI] Add failed cases into waives.txt (#5569) |
| 4035 | 9b17b29b6e | xinhe-nv | 2025-07-01 | test: [CI] remove closed bugs (#5572) |
| 4036 | 82547f733d | QI JUN | 2025-07-01 | add feature support matrix for PyTorch backend (#5037) |
| 4037 | 8caaf6871d | Erin | 2025-06-30 | chores: [TRTLLM-6072] 1.0 LLMAPI doc updates (#5629) |
| 4038 | 7cf1209a19 | Yi Zhang | 2025-07-01 | [fix]: Fix main test skip issue (#5503) |
| 4039 | 6ee94c7ac8 | Netanel Haber | 2025-06-30 | Reintroduce with perf fixes: feature: unify new_tokens format sample state to trtllm samper tokens format (#5513) |
| 4040 | f28cd3056e | Wei-Ming Chen | 2025-06-30 | feat: AutoDeploy fp8 quantization support for bmm (#3849) |
| 4041 | 6e48ac25a6 | nv-guomingz | 2025-07-01 | chore: remove cuda_graph_ prefix from cuda_graph_config filed members. (#5585) |
| 4042 | 16fc99391f | Li Min | 2025-06-30 | refactor: [TRTLLM-6150] Refactor moe permute and finalize op by removing duplicated code (#5557) |
| 4043 | 3b19634a5c | Omer Ullman Argov | 2025-06-30 | [fix][ci] missing class names in post-merge test reports (#5603) |
| 4044 | 98a7c24062 | Yan Chunwei | 2025-06-30 | chore [TRTLLM-6009]: remove ptuning knobs from TorchLlmArgs (#5595) |
| 4045 | 42134b8b84 | Omer Ullman Argov | 2025-06-30 | [ci] move eagle1 and medusa tests to post-merge (#5604) |
| 4046 | 38a39772ce | ixlmar | 2025-06-30 | [TRTLLM-5989, TRTLLM-5991, TRTLLM-5993] doc: Update container instructions (#5490) (#5605) |
| 4047 | b8a568d3c6 | Emma Qiao | 2025-06-30 | [Infra][main] Cherry-pick from release/0.21: Update nccl to 2.27.5 (#5539) (#5587) |
| 4048 | 9bdc5951f8 | Robin Kobus | 2025-06-30 | refactor: decoder state setup (#5093) |
| 4049 | 6cbc9a5297 | Fanrong Li | 2025-06-30 | [nvbug/5354946][fix] Fix mtp vanilla draft inputs (#5568) |
| 4050 | 2ce200fbbb | Kaiyu Xie | 2025-06-30 | doc: Minor update to DeepSeek R1 best practice (#5600) |
| 4051 | 42a9385d02 | WeiHaocheng | 2025-06-30 | [TRTLLM-5331] perf: Replace allgaher with AllToAllPrepare (#5570) |
| 4052 | 852b79053d | dongjiyingdjy | 2025-06-30 | feat : support duplicate_kv_weight for qwen3 blockwise scale (#5459) |
| 4053 | 1db63c2546 | Omer Ullman Argov | 2025-06-30 | [fix] speedup modeling unittests (#5579) |
| 4054 | 4fef14da56 | Yiqing Yan | 2025-06-30 | Deduplicate waive list (#5546) |
| 4055 | 578430e64c | nv-guomingz | 2025-06-30 | [TRTLLM-5530][BREAKING CHANGE]: enhance the llm args pytorch config part 1(cuda_graph_config) (#5014) |
| 4056 | 2780fc27a7 | Omer Ullman Argov | 2025-06-30 | [ci] remove MMLU if followed by GSM8K (#5578) |
| 4057 | 64db7d27f6 | Cheng Hang | 2025-06-30 | [feat] Optimizations on weight-only batched gemv kernel (#5420) |
| 4058 | b4dab23e7b | Enwei Zhu | 2025-06-30 | [TRTLLM-5965] perf: Optimize MoE sort kernels for large-scale EP (#5435) |
| 4059 | 94dc97ab10 | Omer Ullman Argov | 2025-06-29 | [feat][test] reuse MPI pool executor across tests (#5566) |
| 4060 | 6000380a0c | Bo Li | 2025-06-29 | perf: Avoid reswizzle_sf after allgather. (#5504) |
| 4061 | a1c1c6b504 | tomeras91 | 2025-06-29 | [CI] reduce mamba2 ssm test parameterization (#5571) |
| 4062 | 70e34a3291 | Talor Abramovich | 2025-06-29 | [TRTLLM-5831][feat] Add LoRA support for pytorch backend in trtllm-serve (#5376) |
| 4063 | de9779900c | amirkl94 | 2025-06-29 | feat: Add support for YARN in NemotronNAS models (#4906) |
| 4064 | a985c0b7e6 | amirkl94 | 2025-06-29 | tests: Move stress tests to be Post-Merge only (#5166) |
| 4065 | 9db769ee62 | Emma Qiao | 2025-06-29 | [Infra] - Add import pytest (#5565) |
| 4066 | 619709fc33 | Lucas Liebenwein | 2025-06-28 | [AutoDeploy] merge feat/ad-2025-06-13 (#5556) |
| 4067 | 6021a439ab | Li Min | 2025-06-28 | Make moe permute and final as custom op (#5412) |
| 4068 | 5773cfdcf2 | Daniel Stokes | 2025-06-28 | feat: Add support for per expert activation scaling factors (#5013) |
| 4069 | 26b953e29a | Iman Tabrizian | 2025-06-27 | [nvbugs/5309940] Add support for input output token counts (#5445) |
| 4070 | 5437075def | Darragh Hanley | 2025-06-27 | ReDrafter support for Qwen (#4875) |
| 4071 | a8141a4513 | Robin Kobus | 2025-06-27 | refactor: Speculative decoding buffers part 2 (#5316) |
| 4072 | 833c0dea4a | Aurelien Chartier | 2025-06-27 | [TRTLLM-6104] feat: add request_perf_metrics to LLMAPI (#5497) |
| 4073 | 56cdfe5c6c | wili | 2025-06-27 | [TRTLLM-5000][feat] NGrams V2 (#4569) |
| 4074 | cb58073ab7 | peaceh-nv | 2025-06-27 | Fix : fix build for sm120 (#5265) |
| 4075 | 6fc1c6fd7b | Omer Ullman Argov | 2025-06-27 | [fix][ci] correct unittests test prefix (#5547) |
| 4076 | a608b00d38 | ChristinaZ | 2025-06-27 | Fix mPtrExpertCounts allocation in MoE TRT-LLM backend (nvfp4) (#5519) |
| 4077 | 7f1893f54c | Enwei Zhu | 2025-06-27 | ci: waive flaky test test_llama_eagle3 (#5548) |
| 4078 | 73b8a95049 | Daniel Cámpora | 2025-06-27 | feat: Use inference mode in update_requests to improve perf of TRTLLM Sampler (#5538) |
| 4079 | 980030c816 | Emma Qiao | 2025-06-27 | [Infra] - Waive failed case in post-merge (#5536) |
| 4080 | 83a1f60556 | Daniel Stokes | 2025-06-27 | feat: Expose bias and FP8_MXFP4 MOE CUTLASS backend features to pytorch  (#5410) |
| 4081 | ef43b95aa1 | Tailing Yuan | 2025-06-27 | Fix execute_process: check results using EQUAL (#5481) |
| 4082 | 49af791f66 | Iman Tabrizian | 2025-06-26 | Add testing for trtllm-llmapi-launch with tritonserver (#5528) |
| 4083 | dc36228f52 | Yuxian Qiu | 2025-06-27 | fix: Fix block scale fp8 support for deepseek v3 on Blackwell. (#5514) |
| 4084 | a3494bebec | xinhe-nv | 2025-06-27 | tests: waive failed tests on main (#5512) |
| 4085 | 0f3bd7800e | Yibin Li | 2025-06-26 | [TRTLLM-4971]: Use safe deserialization in ParallelConfig (#4630) |
| 4086 | aa6e015ef8 | Frank | 2025-06-26 | Update trtllm-bench to support new Pytorch default. (#5491) |
| 4087 | 0083228d2a | Venky | 2025-06-26 | fix: Mapping rank boundary check bug (#4935) |
| 4088 | 69c4ef2e0e | yuanjingx87 | 2025-06-26 | Update allow list 2025_06_26 (#5526) |
| 4089 | de7cd0de05 | Anthony Chang | 2025-06-27 | fix: MoE autotune fallback failed to query default heuristic (#5520) |
| 4090 | 8836990bde | jmydurant | 2025-06-26 | [TRTLLM-3602][feat] support nvfp4 model and fp8 kv cache for MLA chunked prefill (Blackwell) (#5475) |
| 4091 | 8dfa31c71d | Robin Kobus | 2025-06-26 | refactor: remove batch_manager::KvCacheConfig and use executor::KvCacheConfig instead (#5384) |
| 4092 | 6bae76d7ca | Omer Ullman Argov | 2025-06-26 | [fix][ci] move torch tests to run under torch stage (#5473) |
| 4093 | 1633bd2bef | Omer Ullman Argov | 2025-06-26 | [CI] move flashinfer llama tests to post merge (#5506) |
| 4094 | baf7eaa1cc | Frank | 2025-06-26 | Add trtllm-bench reviewers. (#5452) |
| 4095 | 3a1f4d4001 | Rashid Kaleem | 2025-06-26 | [feat] Add progress bar to benchmark (#5173) |
| 4096 | 2eb6502b1d | Kaiyu Xie | 2025-06-26 | feat: Add support for TRTLLM CustomDataset (#5511) |
| 4097 | 0788c5d0d6 | Yao Yao | 2025-06-26 | [perf] improve XQA-MLA perf (#5468) |
| 4098 | 749393ec9f | Kaiyu Xie | 2025-06-26 | doc: Fix benchmark cmd in disagg scripts (#5515) |
| 4099 | ff2dd72df4 | xinhe-nv | 2025-06-26 | tests: waive tests (#5458) |
| 4100 | fa0ea92dfd | Omer Ullman Argov | 2025-06-26 | [fix][ci] trigger multigpu tests for deepseek changes (#5423) |
| 4101 | 1bab9000a6 | Bo Li | 2025-06-26 | perf: Optimize swizzle_sf, unswizzle_sf, reswizzle_sf (#5318) |
| 4102 | 7e681fbe52 | Alessio Netti | 2025-06-26 | [chore] Allow configuring linking of NVRTC wrapper (#5189) |
| 4103 | 490d2e5819 | dongxuy04 | 2025-06-25 | feat: large-scale EP(part 8: Online EP load balancer integration for PCIe fp8) (#5226) |
| 4104 | e0bb123ae7 | amitz-nv | 2025-06-26 | [TRTLLM-5921][feat] Prevent serialization of entire LoRA adapters in each request (#5080) |
| 4105 | 9ee33605bb | Yukun He | 2025-06-26 | [TRTLLM-6019] feat: Remove cutlass min latency code from AutoTuner. (#5394) |
| 4106 | 942841417e | Daniel Stokes | 2025-06-26 | opensource: Opensource MOE MXFP8-MXFP4 implementation (#5222) |
| 4107 | e9cd810071 | qsang-nv | 2025-06-26 | keep sm90 headsize 128 cubins (#5320) |
| 4108 | 6aef14943c | Netanel Haber | 2025-06-26 | Revert "feature: unify new_tokens format sample state to trtllm samper new_tokens format (#4401)" (#5474) |
| 4109 | 32d1573c43 | Emma Qiao | 2025-06-26 | [Infra] - Add timeout setting for long tests found in post-merge (#5501) |
| 4110 | d9b75f83fd | Venky | 2025-06-25 | [CI] Waive `test_fp8_block_scales_4gpus[ep4-mtp_nextn=0-fp8kv=True-attention_dp=True-cuda_graph=True-overlap_scheduler=True-torch_compile=False]` (#5494) |
| 4111 | d135f5993d | ChristinaZ | 2025-06-26 | Add unit test for routing kernels (#5405) |
| 4112 | 578dbc8d9a | jmydurant | 2025-06-26 | feat: chunked prefill for MLA (Blackwell) (#4651) |
| 4113 | 3fc57543e2 | Yukun He | 2025-06-26 | [5356427] fix: Remove the seq_len of 4096 from FP8 block scale MoE tuning configs. (#5485) |
| 4114 | 74ae15a26b | HuiGao-NV | 2025-06-26 | CI: enable test cases on single device type (#5484) |
| 4115 | 1e4fa13d33 | Xianjie Qiao | 2025-06-26 | Add sleep function for disagg gen-only benchmarking (#5398) |
| 4116 | feaf789342 | QI JUN | 2025-06-26 | CI: reduce BF16 test cases in B200 (#5482) |
| 4117 | bdc8dfebc3 | Omer Ullman Argov | 2025-06-26 | [fix][ci] dont build wheel for cpp tests (#5443) |
| 4118 | 61bb71fd1b | Omer Ullman Argov | 2025-06-25 | [fix][test] remove test in global scope (#5470) |
| 4119 | 3a2c4ca77b | QI JUN | 2025-06-26 | chore: split _build_model method for TorchLlm and TrtLlm (#5418) |
| 4120 | 5bc8c894f7 | Mike Iovine | 2025-06-25 | [chore] Disable block reuse when draft model speculation is being used (#5448) |
| 4121 | 205c97a4ae | Daniel Cámpora | 2025-06-25 | [TRTLLM-5974][feat] Support disaggregated serving in TRTLLM Sampler (#5328) |
| 4122 | c5ae3272b9 | Kaiyu Xie | 2025-06-25 | feat: Make benchmark_serving part of the library (#5428) |
| 4123 | 314f15f0a7 | HuiGao-NV | 2025-06-25 | Fix:  fix nvbug 5356427 (#5464) |
| 4124 | cc3c2b3be2 | HuiGao-NV | 2025-06-25 | Move 3 disaggregated cases from 4 GPUs devices to 1 GPU device (#5457) |
| 4125 | d6ada5ffce | Kaiyu Xie | 2025-06-25 | [nvbug/5354956] fix: unexpected keyword argument 'streaming' (#5436) |
| 4126 | b3a4c1f404 | HuiGao-NV | 2025-06-25 | feat: Remove not used padding_idx in models (#5385) |
| 4127 | 2901c5a5bc | QI JUN | 2025-06-25 | CI: waive test_ad_build_small_multi (#5471) |
| 4128 | 1f292ff2a0 | Perkz Zheng | 2025-06-25 | [https://jirasw.nvidia.com/browse/TRTLLM-4645] support mutliCtasKvMode for high-throughput MLA kernels (#5426) |
| 4129 | f3cfe86dd1 | Yiqing Yan | 2025-06-25 | chore: bump version to 1.0.0rc1 (#5460) |
| 4130 | 3ca2f6ac51 | Netanel Haber | 2025-06-25 | start OAIServer with `max_beam_width=1` for TorchSampler (#5427) |
| 4131 | 478f668dcc | QI JUN | 2025-06-25 | CI: update multi gpu test triggering file list (#5466) |
| 4132 | fc7a81ceb0 | Enwei Zhu | 2025-06-25 | test: Add LLGuidance test and refine guided decoding (#5348) |
| 4133 | 76da7fed86 | Enwei Zhu | 2025-06-25 | fix (NvBug 5354925): Fix static EPLB (#5411) |
| 4134 | da98e03747 | HuiGao-NV | 2025-06-25 | tests: Set kv cache free memory fraction in test case (#5433) |
| 4135 | d5354897c0 | Shunkangz | 2025-06-25 | feat: Dynamically remove servers in PD (#5270) |
| 4136 | 5cffb7e0ec | Lucas Liebenwein | 2025-06-25 | [AutoDeploy] Merge feat/ad_2025_06_13 feature branch (#5454) |
| 4137 | 73ba4fc320 | bhsueh_NV | 2025-06-25 | fix: fix bug of qwen3 + eagle3 + finalize_moe_fusion (#5369) |
| 4138 | 241f921800 | QI JUN | 2025-06-25 | waive test_moe.py::test_moe_fp8[autotune] (#5455) |
| 4139 | 699520082b | dongxuy04 | 2025-06-25 | Add MTP support for Online EPLB (#5213) |
| 4140 | 846bbf1edc | Iman Tabrizian | 2025-06-24 | Fix test Pytorch model engine (#5416) |
| 4141 | d93a5e04b5 | QI JUN | 2025-06-24 | Chore: remove unused variables (#5314) |
| 4142 | 35a92f6bab | HuiGao-NV | 2025-06-24 | Add debug hook to support dump tensor data and add new debug functions easily (#5182) |
| 4143 | 475272046a | Emma Qiao | 2025-06-24 | [Infra] - Waive failed tests in post-merge and increase some timeout setting (#5424) |
| 4144 | d26040e5d9 | Luis Vega | 2025-06-24 | chore: delete mamba hybrid, since it is now called NemotronH (#5409) |
| 4145 | 658fb5b54e | xinhe-nv | 2025-06-24 | tests: update benchmark test lists (#5365) |
| 4146 | e2a8cbc80b | Robin Kobus | 2025-06-24 | refactor: manage cache indirection in decoder state (#5315) |
| 4147 | 4b32a3f1a7 | xinhe-nv | 2025-06-24 | test: [CI] remove closed bugs (#5400) |
| 4148 | e16c1bef6e | HuiGao-NV | 2025-06-24 | [fix] Add 1 and draft_token_num to seq_len when overlap scheduling is enabled during memory estimation (#5343) |
| 4149 | 58a8a8fd37 | Netanel Haber | 2025-06-23 | feature: unify new_tokens format sample state to trtllm sampler new_tokens format (#4401) |
| 4150 | ebadc13086 | Fanrong Li | 2025-06-21 | [doc] update mtp documents (#5387) |
| 4151 | b3045c44b9 | Robin Kobus | 2025-06-20 | refactor: remove TrtGptModelOptionalParams (#5165) |
| 4152 | 4f0f17ac8a | dongxuy04 | 2025-06-20 | feat: Misc Opt for large scale EP (#5374) |
| 4153 | 5d4ab47d5b | Fanrong Li | 2025-06-20 | fix: refactor and fix mtp vanilla (#4762) |
| 4154 | b1878eabeb | Adamz-nvidia | 2025-06-20 | Add Wechat_Group_QR_Code.png to docs/source/media and main page of TR… (#5142) |
| 4155 | 9bd42ecf9b | Yan Chunwei | 2025-06-20 | [TRTLLM-5208][BREAKING CHANGE] chore: make pytorch LLM the default (#5312) |
| 4156 | 113f6fbadd | Kaiyu Xie | 2025-06-19 | Fix: missing clientId when serialize and deserialize response (#5231) |
| 4157 | 7246fd75d1 | Kaiyu Xie | 2025-06-19 | feat: Support stream_interval (#5284) |
| 4158 | 1e35be5840 | Shi Xiaowei | 2025-06-19 | doc: subsequent modifications of blog 5 (#5366) |
| 4159 | c7af650d5a | Fanrong Li | 2025-06-19 | Fix: fix the deterministic issue in the MTP Eagle path (#5285) |
| 4160 | 9a53e58a58 | Shi Xiaowei | 2025-06-19 | blog: Disaggregated Serving in TensorRT-LLM (#5353) |
| 4161 | 68687a9f56 | Frank | 2025-06-19 | [WAR][nvbug/5321947] Add an async sleep to unblock event loop. (#5342) |
| 4162 | bca758fce1 | Enwei Zhu | 2025-06-19 | fix: Fix DS-R1 nvfp4 test case naming (#5361) |
| 4163 | 493f268b1c | Emma Qiao | 2025-06-19 | [Infra]Fix l0_sanity_check.yml which also has gb202 and gb203 (#5360) |
| 4164 | b558232ce1 | hlu1 | 2025-06-19 | Refactor CutlassFusedMoE (#5344) |
| 4165 | e22e884b02 | ruodil | 2025-06-19 | test: amend test case name in perf cluster test (#5356) |
| 4166 | 21ce9b6749 | ruodil | 2025-06-19 | test: add qwen3 cases (#5302) |
| 4167 | 1753202b61 | amitz-nv | 2025-06-19 | [TRTLLM-5825][fix] Fix torch LoRA TP (#5338) |
| 4168 | 7f68de3e3f | Emma Qiao | 2025-06-19 | Refactor test timeout for individual long case (#4757) |
| 4169 | b3e886074e | yunruis | 2025-06-19 | Fix CI build time increase (#5337) |
| 4170 | dce8620013 | bhsueh_NV | 2025-06-19 | chore: enable moe_backend on Qwen3 test (#5230) |
| 4171 | e5400eeae0 | xinhe-nv | 2025-06-19 | tests: add ds r1 tp4 test (#5197) |
| 4172 | dedce8ab0e | Yiqing Yan | 2025-06-19 | chore: bump version to 1.0.0rc0 (#5326) |
| 4173 | da576bcafa | Yiqing Yan | 2025-06-19 | Waive L0 test (#5349) |
| 4174 | 6c3210a8be | Fanrong Li | 2025-06-19 | [test] add nvfp4 DeepSeek-V3-Lite-mtp tests (#5125) |
| 4175 | 6a388b105a | nv-guomingz | 2025-06-19 | chore: remove torch_compile prefix for TorchCompileConfig field members (#5261) |
| 4176 | 2b23cd56ce | Zongfei Jing | 2025-06-19 | [feat] Fusion finalize and allreduce for qwenmoe model (#5223) |
| 4177 | 1a7c6e7974 | Robin Kobus | 2025-06-19 | ci: Split long running jobs into multiple jobs (#5268) |
| 4178 | 3946e798db | Yan Chunwei | 2025-06-19 | fix[nvbug5298640]: trtllm-llmapi-launch multiple LLM instances (#4727) |
| 4179 | 0b6d005ef6 | Omer Ullman Argov | 2025-06-19 | [fix][test] clear cuda cache before unittests automatically (#5121) |
| 4180 | d25f93c07f | Aurelien Chartier | 2025-06-18 | chore: skip test_llm_gpt2_medium_fp8 for fp8_pc_pt + quant_lm_head (#5293) |
| 4181 | 5010f8719d | Omer Ullman Argov | 2025-06-18 | [fix][test] remove duplicate test runs (#5241) |
| 4182 | a28a152001 | Omer Ullman Argov | 2025-06-18 | [fix][test] remove some cpp test cases from h100 (#5335) |
| 4183 | a1c5704055 | yuanjingx87 | 2025-06-18 | [feat] Multi-node CI testing support via Slurm (#4771) |
| 4184 | e5ee5c5352 | Iman Tabrizian | 2025-06-18 | Unwaive disaggregated serving accuracy tests (#5095) |
| 4185 | 857108aeca | Xianjie Qiao | 2025-06-18 | Add disagg slurm scripts (#5243) |
| 4186 | d13d2f460d | HuiGao-NV | 2025-06-18 | Remove duplicated test cases (#5323) |
| 4187 | 00bdd39b96 | juney-nvidia | 2025-06-18 | chore: Update README.md to expose meet-up info (#5329) |
| 4188 | b29ac5b561 | Emma Qiao | 2025-06-18 | [Infra] Update 5080 and 5090 case condition due to the driver update (#5317) |
| 4189 | 0623ffe3bc | jellysnack | 2025-06-18 | feat: Add LLGuidance Support for PyTorch Backend (#5214) |
| 4190 | 610a49f117 | xinhe-nv | 2025-06-18 | tests: add multi nodes tests (#5196) |
| 4191 | 375dd0b971 | Yi Zhang | 2025-06-18 | Waive L0 (#5311) |
| 4192 | a3a48410f3 | Yiqing Yan | 2025-06-18 | Fix rerun step (#5319) |
| 4193 | f599ee63c1 | Yuan Tong | 2025-06-18 | test: correct unittest rerun behavior (#5273) |
| 4194 | 516bd4dc05 | Zhanrui Sun | 2025-06-18 | chore: bump version to 0.21.0rc3 (#5309) |
| 4195 | 38547b92f3 | Robin Kobus | 2025-06-18 | refactor: Introduce ResourceManagerType enum for resource management (#5246) |
| 4196 | d76bda7f2c | Bo Li | 2025-06-18 | chore: Refine printed info of CHECK_TYPE. (#5295) |
| 4197 | 3a02489e86 | Wanli Jiang | 2025-06-18 | [TRTLLM-5758] test: Add Bielik-11B-v2.2 Model Support (#5159) |
| 4198 | 9ea7bb67a4 | QI JUN | 2025-06-18 | CI: fix TensorRT H200 tests (#5301) |
| 4199 | 6711ad9cf3 | Yukun He | 2025-06-18 | [TRTLLM-5589] feat: Minor optimizations for tunable FP8 batched GEMM op. (#5139) |
| 4200 | 3b5d916250 | ruodil | 2025-06-18 | test: cherry-pick deepseek rcca cases in main branch (#5307) |
| 4201 | ee26965054 | nv-guomingz | 2025-06-18 | doc:update contributing md for internal developers (#5250) |
| 4202 | 908463a5f5 | Yao Yao | 2025-06-18 | [feat]: improve performance of XQA-MLA for sm120 (#5087) |
| 4203 | 724e495254 | Yan Chunwei | 2025-06-18 | chore: partition LLM class into TorchLLM and TrtLLM (#4900) |
| 4204 | e44f7687af | Yi Zhang | 2025-06-18 | feat: Add no_kv_cache_reuse option and streaming support for trtllm serve bench (#4971) |
| 4205 | 8f67e3604d | Yiqing Yan | 2025-06-18 | Waive L0 tests (#5308) |
| 4206 | f501ce57b1 | Omer Ullman Argov | 2025-06-18 | [fix][test] move deepseek single gpu tests to post merge (#5280) |
| 4207 | 3c0fecbf42 | dominicshanshan | 2025-06-18 | CI: extend model weights load time for dsv3 in stress test. (#5275) |
| 4208 | 41cfcaa964 | Ivy Zhang | 2025-06-18 | test: update qa test list (#5305) |
| 4209 | 855036d8ee | QI JUN | 2025-06-18 | update LlmRequest.is_dummy property (#5283) |
| 4210 | e1e5f725fc | Aurelien Chartier | 2025-06-17 | fix: only set _mpi_session if world_size is > 1 (#5253) |
| 4211 | 627062c265 | Robin Kobus | 2025-06-18 | refactor: Update decoder buffer and logits management (#4450) |
| 4212 | 7d55c381fa | tburt-nv | 2025-06-17 | Revert "[infra] Report CI authorization errors to PR" (#5298) |
| 4213 | 2df9f875cf | tburt-nv | 2025-06-17 | [infra] Report CI authorization errors to PR (#5175) |
| 4214 | 9bf69c9fdb | Mike Iovine | 2025-06-17 | [chore] Remove BaseDraftTokenManager (#5251) |
| 4215 | ff32caf4d7 | Emma Qiao | 2025-06-17 | [Infra] - Update dependencies with NGC PyTorch 25.05 and TRT 10.11 (#4885) |
| 4216 | dcf18c4bcf | Yiteng Niu | 2025-06-17 | infra[TRTLLM-5635] remove package stage in CI build (#5075) |
| 4217 | 5236bb9084 | qsang-nv | 2025-06-17 | delete cubins (#5274) |
| 4218 | f899c4d294 | QI JUN | 2025-06-17 | Re-implement LlmResponse in Python to reduce host overhead of pybind (#5224) |
| 4219 | f4cdbfcdf0 | Yanchao Lu | 2025-06-17 | None - Some clean-ups for the automation pipeline (#5245) |
| 4220 | 44fb3c1673 | Dom Brown | 2025-06-17 | [TRTLLM-5770] feat: Integrate TRT-LLM Gen FP8 block scale MoE with Pytorch workflow kernel autotuner (#5207) |
| 4221 | 8451a87742 | amirkl94 | 2025-06-17 | chore: Mass integration of release/0.20 (#5082) |
| 4222 | 13eef642e6 | liji-nv | 2025-06-17 | [feat] Piecewise cuda graph support for MLA (#4467) |
| 4223 | dc3861b4aa | Robin Kobus | 2025-06-17 | refactor: Unify decoder test with e2e worklfow (#5239) |
| 4224 | ccd9adbe33 | QI JUN | 2025-06-17 | CI: move multi-gpu test cases of tensorrt backend to h200 (#5272) |
| 4225 | 2ad8758ecc | Ivy Zhang | 2025-06-17 | [TRTLLM-5786][https://nvbugspro.nvidia.com/bug/5310520][test] Add QA test cases (#5073) |
| 4226 | 498fadceb4 | Yilin Fan | 2025-06-17 | [feat] Add EAGLE3 support for Qwen3 (#5206) |
| 4227 | 517c1ecf72 | QI JUN | 2025-06-17 | move some test cases of TensorRT backend back (#5232) |
| 4228 | faca19c2f0 | qsang-nv | 2025-06-17 | update setup.py for special cases (#5227) |
| 4229 | 6a6b9d2594 | bhsueh_NV | 2025-06-17 | doc: add document of benchmarking for Qwen3 (#5158) |
| 4230 | 134cb66a53 | qsang-nv | 2025-06-17 | fix mla test (#5240) |
| 4231 | 4b82b8b4c7 | Enwei Zhu | 2025-06-17 | [TRTLLM-5330] perf: Optimize MoE supplementary kernels for large-scale EP (#5215) |
| 4232 | a49ad790b3 | xinhe-nv | 2025-06-17 | test: [CI] remove closed bugs (#5218) |
| 4233 | 546274d40e | QI JUN | 2025-06-17 | fix ci (#5259) |
| 4234 | bb2348372c | ruodil | 2025-06-17 | test: add more pytorch cases in perf test (#5237) |
| 4235 | a2e8ae1120 | Tracin | 2025-06-17 | Update internal cutlass commit. (#5228) |
| 4236 | c53bc19f5e | Mike Iovine | 2025-06-16 | [infra] Make test_chunked_prefill faster (#5248) |
| 4237 | 5c18160d27 | Simeng Liu | 2025-06-16 | chore: Waive CI failure. (#5252) |
| 4238 | e607768e45 | Izzy Putterman | 2025-06-16 | Speculation: Draft Target in new FW (#4558) |
| 4239 | cea5dd1e38 | tomeras91 | 2025-06-16 | [TRTLLM-5835][feat] Optimized Mamba2Mixer prefill (#5128) |
| 4240 | dd29063538 | Yilin Fan | 2025-06-16 | [feat] Add llm args to tune python gc threshold (#5141) |
| 4241 | 03f1a6a3d8 | Tao Li @ NVIDIA | 2025-06-16 | Update DeepSeek R1 perf numbers to latest release/0.20 results (#5235) |
| 4242 | 64b7f04fdc | Ivy Zhang | 2025-06-16 | [test] split nemotron test cases from examples_test_list (#5238) |
| 4243 | 802f22cd12 | xinhe-nv | 2025-06-16 | test: [CI] Add failed cases into waives.txt (#5221) |
| 4244 | 8445416c39 | Yiqing Yan | 2025-06-16 | Waive L0 tests (#5233) |
| 4245 | b6ca677741 | Robin Kobus | 2025-06-16 | refactor: remove decoder request from decoder interface (#5129) |
| 4246 | 4f9fa9f21d | Anthony Chang | 2025-06-16 | feat: MoE trtllm backend kernel update (#5183) |
| 4247 | 1d2b0d3d80 | Chuang Zhu | 2025-06-16 | use file lock to avoid port conflict (#5123) |
| 4248 | dda64166cd | Robin Kobus | 2025-06-16 | refactor: Scheduling based on KV cache state (#4865) |
| 4249 | 0acf23185e | Wanli Jiang | 2025-06-16 | [Stress test] Add DeepSeek-R1 stress test (#5033) |
| 4250 | ef3fdc8051 | Tracin | 2025-06-16 | feat: Add w4a8_mxfp4_fp8 quantization recipe. (#4867) |
| 4251 | 9b616db13b | Yi Zhang | 2025-06-16 | test: Add fixture to skip tests based on MPI world size (#5028) |
| 4252 | 2848e012ae | ruodil | 2025-06-16 | test: add llama4 models for perf test (#5187) |
| 4253 | 3d22f27063 | ruodil | 2025-06-16 | test: add more cases for llama_v3.3/3.1 70b fp8 and set enable_attention_dp to false to non-deepseek models (#5155) |
| 4254 | babdd9ce06 | Enwei Zhu | 2025-06-16 | test: Add json_mode_eval for guided decoding evaluation (#5179) |
| 4255 | 7a5e0fd300 | Yilin Fan | 2025-06-15 | [fix] Fix Llama4 min-latency import error (#5209) |
| 4256 | c84e41fd9d | Yan Chunwei | 2025-06-16 | fix: build_config in TorchLlmArgs and avoid arbitrary args  (#4972) |
| 4257 | 109c426077 | amitz-nv | 2025-06-15 | Enable trtllm-bench to run LoRA and add basic e2e perf testing capability for LoRA in PyT flow (#5130) |
| 4258 | 39bba63758 | Fanrong Li | 2025-06-15 | [TRTLLM-4983] feat: enable overlap scheduler between draft forwards (#4802) |
| 4259 | 5a01ba5260 | qsang-nv | 2025-06-15 | use cu for fmha_v2 (#4694) |
| 4260 | 4eade3ae33 | Omer Ullman Argov | 2025-06-15 | [fix][test] Speedup Nemotron NAS unittests (#5202) |
| 4261 | 159ffc584e | Fanrong Li | 2025-06-15 | fix: fix cuda graph max batch size for spec decoding cases. (#5076) |
| 4262 | dce1dcc4f9 | Kaiyu Xie | 2025-06-15 | feat: Support post_proc for bench (#5122) |
| 4263 | 63bc62ddf4 | Enwei Zhu | 2025-06-15 | feat: Enable EPLB to existing MoE models (#5203) |
| 4264 | 6bce7337a9 | Yuan Tong | 2025-06-15 | perf: avoid dynamic import overhead in is_llm_response with duck typing (#5110) |
| 4265 | e055af1bc9 | ixlmar | 2025-06-14 | chore: improve disagg test failure detection (#4738) |
| 4266 | 1389f5a4d3 | Aurelien Chartier | 2025-06-14 | feat: Add support for fp8 rowwise quantization (#4876) |
| 4267 | dc52b67492 | 2ez4bz | 2025-06-14 | linting(python): Enable ruff on more files (wave 1/N) (#5140) |
| 4268 | 0b60da2c45 | Tailing Yuan | 2025-06-14 | feat: large-scale EP(part 7: DeepEP integration) (#4792) |
| 4269 | 443b2eb51f | Robin Kobus | 2025-06-14 | refactor: Speculative decoding buffers (#5091) |
| 4270 | b99c5ce8c1 | yunruis | 2025-06-14 | Feat/ds r1 min latency opt round3, add router gemm, fused a gemm, PDL  (#4560) |
| 4271 | 3b7b5a5ad5 | nv-guomingz | 2025-06-14 | refactor [BREAKING CHANGE]: enhance the llm args pytorch config part 3(torch_compile_config) (#5032) |
| 4272 | 97657bfda2 | dongxuy04 | 2025-06-14 | optimize memset before alltoall communication (#5188) |
| 4273 | 82e280f6f3 | Aurelien Chartier | 2025-06-13 | feat: add multi-node support for Triton with pytorch backend (#5172) |
| 4274 | 5f2785fb90 | Enwei Zhu | 2025-06-13 | fix: Fix waive list (#5205) |
| 4275 | 06342ffb4d | Yilin Fan | 2025-06-13 | [feat] Implement model-agnostic one-engine eagle3 (#4778) |
| 4276 | 25aa3881d7 | Mike Iovine | 2025-06-13 | [nvbug/5319281][fix] Stop drafting when we hit the draft model's max seq len (#4879) |
| 4277 | 3d87770e15 | Perkz Zheng | 2025-06-13 | [https://nvbugspro.nvidia.com/bug/5295470] support headDim 256 for blackwell fmha kernels (#5164) |
| 4278 | 952f33dcad | QI JUN | 2025-06-13 | CI: move all test cases of TensorRT backend into post merge (#5186) |
| 4279 | 8e9937081d | Chuang Zhu | 2025-06-13 | ucxx only use ucp_feature_tag to aviod some issuse on some platform (#4994) |
| 4280 | e5be3a95b3 | yunruis | 2025-06-13 | fix: fix license bug (#5200) |
| 4281 | e96d6863d8 | yunruis | 2025-06-13 | add doc for open-sourced cutlass kernels (#5194) |
| 4282 | 089be8912a | brb-nv | 2025-06-13 | feat: Basic skeleton for Gemma3 VLM (#5108) |
| 4283 | 30d9d0fa71 | xinhe-nv | 2025-06-13 | test: [CI] Add failed cases into waives.txt (#5178) |
| 4284 | b959618579 | nv-guomingz | 2025-06-13 | refactor [BREAKING CHANGE]:: remove the redundant use_kv_cache field from PytorchConfig  (#5031) |
| 4285 | 30c5b4183a | yunruis | 2025-06-13 | refactoring: port customized kernels with public cutlass version (#5027) |
| 4286 | 12e075eb70 | Yao Yao | 2025-06-13 | [nvbug 5333996 ][fix] Unload XQA cubins early to avoid static lifetime (#5133) |
| 4287 | 514baf1287 | Matthias Jouanneaux | 2025-06-13 | [fix] Fix comment to pass guardwords check (#5191) |
| 4288 | 4d0a5ad384 | Zheng Duan | 2025-06-13 | chore: gracefully exit disagg process in tests; better startup and logging (#5109) |
| 4289 | 28cd536bd6 | Ivy Zhang | 2025-06-13 | [test] Update timeout params in QA test list (#5124) |
| 4290 | 01bd4c00b4 | Iman Tabrizian | 2025-06-13 | Add two MTP disaggregated test (#4546) |
| 4291 | dec326ba7d | Daniel Cámpora | 2025-06-13 | [fix] Reenable test return logits (#5160) |
| 4292 | b79eb34bfe | Yibin Li | 2025-06-12 | [fix]: Fall back to HMAC to Avoid IPC Serialization Churn (#5074) |
| 4293 | d9be419f45 | xinhe-nv | 2025-06-13 | tests: update tests for b200 (#5180) |
| 4294 | fa582cbe9a | ruodil | 2025-06-13 | test: add more cases for rtx_pro_6000_se and add option kv_cache_dtype in perf test (#5083) |
| 4295 | a891013e3c | zhhuang-nv | 2025-06-13 | [feat] Optimize KV Cache Reuse for MLA (#4869) |
| 4296 | 4ae46b6714 | Yuxian Qiu | 2025-06-13 | fix: [nvbugs/5324229] Fix broken WInt4AFP8FusedMoEMethod since FusedMoE refactor. (#4930) |
| 4297 | 38a907aaca | Fanrong Li | 2025-06-13 | [TRTLLM-5278][feat] Add attention dp support to MTP relaxed acceptance (#5119) |
| 4298 | a0b6c635b1 | Matthias Jouanneaux | 2025-06-13 | [feat] trtllmGen MoE routing: added support for top groups and top K bounds (#4063) |
| 4299 | cc2a1344be | Xiaodong (Vincent) Huang | 2025-06-12 | None: fix OOM because of unnecessary mha workspace (#5056) |
| 4300 | 3a04c9fa7b | pcastonguay | 2025-06-12 | chore: Include prompt_token_ids only for context-only disagg requests (#5055) |
| 4301 | 655bce0b19 | Omer Ullman Argov | 2025-06-12 | [fix][test] report individual unittests results to jenkins (#5116) |
| 4302 | 690873ba1a | Mike Iovine | 2025-06-12 | [nvbug/5334370][fix] Fix one model EAGLE3 (#5134) |
| 4303 | dfeeaf6746 | HuiGao-NV | 2025-06-12 | Move allreduce_strategy from committed api to reference (#5147) |
| 4304 | 8cfb567182 | brb-nv | 2025-06-12 | fix: Updates to yarn implementation (#5105) |
| 4305 | cf35a079f9 | nv-guomingz | 2025-06-12 | fix:https://nvbugs/5298661 (#5022) |
| 4306 | 58d4ca2385 | nv-guomingz | 2025-06-12 | fix:remove duplicated trust_remote_code knob from trtllm-serve (#5143) |
| 4307 | 22281cfc55 | Daniel Cámpora | 2025-06-12 | doc: Added documentation for enable_trtllm_sampler. (#4990) |
| 4308 | 59c9588e9a | Venky | 2025-06-12 | enh(doc): Add `ci-overview` in `docs/source/reference/` (#5137) |
| 4309 | 88cba5f354 | Shi Xiaowei | 2025-06-12 | test: waive the NIXL related tests (#5153) |
| 4310 | b563696dee | nv-guomingz | 2025-06-12 | doc:fix invalid links for trtllm-serve doc (#5145) |
| 4311 | a97f4581d2 | Zhanrui Sun | 2025-06-12 | infra: upload imageTag info to artifactory and add ngc_staging to save ngc image (#4764) |
| 4312 | 10ab9791ec | liji-nv | 2025-06-12 | [fix] Do not reuse dummy request KVCache (#4804) |
| 4313 | 4d070d3862 | Fanrong Li | 2025-06-12 | chore: fix typo in tests (#5092) |
| 4314 | e46267765f | Daniel Cámpora | 2025-06-12 | Fix logprobs issues. (#5136) |
| 4315 | 53983ad273 | Michal Guzek | 2025-06-12 | [TRTLLM-4932] Add Llama-3.1-Nemotron-Nano-8B-v1-FP8 accuracy tests (#4933) |
| 4316 | d021cc5126 | ruodil | 2025-06-12 | test: set enable_attention_dp to False for non-deepseek models and add more cases for llama_v3.1/3.3 70b fp8 models (#5149) |
| 4317 | 06d9f1e2f6 | tomeras91 | 2025-06-12 | [test] Use LLM API for Nemotron-H correctness test (#5097) |
| 4318 | 505678a286 | bhsueh_NV | 2025-06-12 | update the free_gpu_mem_fraction for H100 qwen3 qa test (#5114) |
| 4319 | 0daa70999a | Michal Guzek | 2025-06-11 | Fix Llama-3_3-Nemotron-Super-49B-v1 FP8 accuracy threshold configs (#4961) |
| 4320 | c3b2eb6dab | Venky | 2025-06-11 | test(perf): Add remaining Llama-Nemotron perftests (nano, super, ultra) + extras ✨ (#5066) |
| 4321 | 49d7268acc | Lucas Liebenwein | 2025-06-12 | [nvbugs/5331013] fix AutoDeploy for PyTorch 25.05 dependency upgrade (#5106) |
| 4322 | e692779ead | Netanel Haber | 2025-06-12 | Solve underallocation in VSWA+/VGQA  (#4667) |
| 4323 | 43192379af | HuiGao-NV | 2025-06-12 | Use backend to replace macro to control enablement of MNNVL all reduce (#4635) |
| 4324 | c592798f64 | Zheng Duan | 2025-06-12 | fix: limit process pool size when prefetching (#5088) |
| 4325 | ee44fa00f8 | Zheng Duan | 2025-06-12 | chore: rename IOFormatter to BaseCacheFormatter (#5068) |
| 4326 | ad99a08fa2 | Po-Wei (Vincent) | 2025-06-11 | [TRTLLM-5581][infra] Update Module Owners (#5052) |
| 4327 | ddfe4fceb3 | tburt-nv | 2025-06-11 | [chore] 2025-06-10 update allowlist (#5102) |
| 4328 | 11b94feff8 | xinhe-nv | 2025-06-11 | test: skip disaggregated tests on arm (#5070) |
| 4329 | a90dd571f8 | Yiqing Yan | 2025-06-11 | [TRTLLM-5082] - Add a bot run option for detailed logs (#4390) |
| 4330 | 8282d6c1a7 | liji-nv | 2025-06-11 | [fix] Fix llama4 min latency (#5117) |
| 4331 | 56abae0835 | ruodil | 2025-06-11 | test: add more llama_v3.3_70b cases in perf test (#4979) |
| 4332 | e2863a3159 | Zhanrui Sun | 2025-06-11 | chore: bump version to 0.21.0rc2 (#5112) |
| 4333 | fdf1c47d1d | Daniel Cámpora | 2025-06-11 | [TRTLLM-4995][feat] TRTLLM Sampler log probs support (#4836) |
| 4334 | 00991d1520 | Enwei Zhu | 2025-06-11 | chore: Merge remaining changes from feat/large-ep branch to main (#5039) |
| 4335 | 0a9f105931 | Yiqing Yan | 2025-06-11 | Waive L0 tests (#5111) |
| 4336 | 035b048a65 | Zhanrui Sun | 2025-06-11 | infra: Add timeout and retry for wget in docker image build (#5035) |
| 4337 | 273c6b9355 | ChristinaZ | 2025-06-11 | [https://nvbugspro.nvidia.com/bug/5332927][fix] Fix the bug in the routing unit test (#5065) |
| 4338 | 580a92521e | Zheng Duan | 2025-06-11 | test: conditional disagg and cache aware balancing for deepseek v3 (#4522) |
| 4339 | 1b79041f5d | Bo Li | 2025-06-11 | fix: XQA is not enabled when history_length < kMinHistoryTokensPerBlock. (#4264) |
| 4340 | fcd71921f1 | Mike Iovine | 2025-06-10 | [fix] Unwaive test_llama_eagle3 (#5042) |
| 4341 | 6cb2b7d370 | Izzy Putterman | 2025-06-10 | CI: Allow run (#5101) |
| 4342 | 194a708d83 | Jinyang Yuan | 2025-06-11 | [fix] Fix test_attention_mla (#5084) |
| 4343 | 50f576172b | Linda | 2025-06-10 | doc: add info about stop words appearing in output (#4956) |
| 4344 | 7b210ae9c3 | nvpohanh | 2025-06-11 | test: add unit tests for Llama4 min_latency code (#4980) |
| 4345 | 7ddc4d6282 | Lucas Liebenwein | 2025-06-10 | [AutoDeploy] Merge Feature Branch Week 3 (#5054) |
| 4346 | 6c91f1c7ac | Tracin | 2025-06-10 | Mxfp8xmxfp4 quant mode(#4978) |
| 4347 | f6a49a9343 | liji-nv | 2025-06-10 | [CI] waive failing L0 test (#5089) |
| 4348 | 6d1f2d0fd7 | Zongfei Jing | 2025-06-10 | [TRTLLM-3927] [feat] Finalize + Allreduce + add + rmsnorm fusion (#4756) |
| 4349 | 08dc369a4d | Yuxian Qiu | 2025-06-10 | fix: pytorch_backend_config is deprecated in update_llm_args_with_extra_dict. (#4890) |
| 4350 | dcf72c6ad3 | Aurelien Chartier | 2025-06-10 | chore: cleanup GDS Cmake interface (#4928) |
| 4351 | 8ec8e4559d | Yiqing Yan | 2025-06-10 | Waive L0 test (#5077) |
| 4352 | f121f13ddf | tomeras91 | 2025-06-10 | [nvbug 5325284][fix] Increase Nemotron-H warmup request robustness (#4954) |
| 4353 | fdfc711261 | Yiqing Yan | 2025-06-10 | Waive L0 test (#5067) |
| 4354 | 7137cc8f67 | dongxuy04 | 2025-06-10 | fix cuda driver link issue with driver version less than 12.3 (#5025) |
| 4355 | ec6b1821c7 | Xiaowei Wang | 2025-06-10 | [fix] Fix W4A8 weight loading error in WInt4AFP8FusedMoEMethod (#5026) |
| 4356 | 12ffdcbf53 | QI JUN | 2025-06-10 | CI: waive test_ad_build_small_multi (#5071) |
| 4357 | 86959ef1e4 | Simeng Liu | 2025-06-09 | chore: Waive CI failure. (#5069) |
| 4358 | 87c56ab024 | pcastonguay | 2025-06-09 | perf: Removing initializing ptuning buffers to zero (#4915) |
| 4359 | 74b0e71ef4 | Stanley Sun | 2025-06-10 | test: add more disaggregated serving tests into QA testlist (#5036) |
| 4360 | d68b8180d3 | Daniel Cámpora | 2025-06-10 | feat: port MakeDecodingBatchInputOutput to python in TRTLLMSampler (#4828) |
| 4361 | 6d4d179cac | pcastonguay | 2025-06-09 | [TRTLLM-5518] doc: Adding disaggregated serving section to models doc (#4877) |
| 4362 | e2bd01fa18 | tburt-nv | 2025-06-09 | [https://nvbugs/5332927] Waive new tests (#5051) |
| 4363 | f70815c945 | Chang Liu | 2025-06-10 | [TRTLLM-5007][feat] Add multimodal hashing support (image hashing) (#4145) |
| 4364 | 5097c86168 | Yukun He | 2025-06-09 | chore: Change cutlass version back to 4.0 (#5041) |
| 4365 | e79527d195 | Yuxian Qiu | 2025-06-09 | chore: Refine weight prefetching. (#4893) |
| 4366 | 5b84fd9201 | pcastonguay | 2025-06-09 | [nvbug 5283506] fix: Fix spec decode triton test (#4845) |
| 4367 | f4d9c87c51 | Mike Iovine | 2025-06-09 | [nvbug/5314469][feat] Include the executor's max batch size in CUDA g… (#4843) |
| 4368 | 137fe35539 | Yukun He | 2025-06-09 | fix: Fix warmup phase batch size out of range. (#4986) |
| 4369 | 88480197da | Yuxian Qiu | 2025-06-09 | ci: [nvbugs/5280806] Unwaive unittests/_torch. (#4951) |
| 4370 | 9c012d5bf8 | Dom Brown | 2025-06-09 | [TRTLLM-5589] feat: Integrate TRT-LLM Gen FP8 Batched GEMM with Pytorch workflow kernel autotuner (#4872) |
| 4371 | 1d4f748773 | liji-nv | 2025-06-09 | [fix] Fix illegal mem access and possible accuracy lose. Cherry-pick … (#5017) |
| 4372 | f45aff2b7d | ChristinaZ | 2025-06-09 | Add customized renormalized moe routing kernel for moe cutlass backend (#4955) |
| 4373 | c104388d37 | Bo Li | 2025-06-09 | chore: Refactor apply_rope. (#4918) |
| 4374 | 6b17dff2f1 | Yiqing Yan | 2025-06-09 | Waive L0 test (#5024) |
| 4375 | 9a874760c1 | Chuang Zhu | 2025-06-09 | Kv cache transfer support duplicate heads (#4929) |
| 4376 | 947571c311 | Chuang Zhu | 2025-06-09 | Fix buffer count (#5007) |
| 4377 | f4bfb8e49d | Yan Chunwei | 2025-06-09 | ci: unwaive llmapi launch test (#4991) |
| 4378 | 3a4851b7c3 | Daniel Stokes | 2025-06-09 | feat: Add Mixture of Experts FP8xMXFP4 support (#4750) |
| 4379 | 77e8d739f1 | amitz-nv | 2025-06-09 | [TRTLLM-4987][feat] Support generation logits in TRTLLMSampler (#4819) |
| 4380 | 8b4104d34a | Yechan Kim | 2025-06-09 | feat: add HyperCLOVAX-SEED-Vision support in refactored way (#4799) |
| 4381 | bb79ba7c35 | Julien Demouth | 2025-06-09 | Edits for tech blog 4 (#5006) |
| 4382 | 78472339b3 | nv-guomingz | 2025-06-09 | fix:https://nvbugs/5324252 (#4925) |
| 4383 | 8731f5f14f | Omer Ullman Argov | 2025-06-08 | chore: Mass integration of release/0.20 (#4898) |
| 4384 | ec0d984656 | Mike Iovine | 2025-06-08 | [nvbug/5280806][fix] Fix 2 model spec decode flow (#4807) |
| 4385 | 9e05613679 | Yanchao Lu | 2025-06-08 | [Infra] - Update JNLP container config (#5008) |
| 4386 | 786e32d56f | nv-guomingz | 2025-06-08 | chore:update modelopt to 0.31 (#5003) |
| 4387 | 1e369658f1 | dongxuy04 | 2025-06-08 | feat: large-scale EP(part 6: Online EP load balancer integration for GB200 nvfp4) (#4818) |
| 4388 | 5ee0de7f2a | QI JUN | 2025-06-08 | Resubmit #4894 (#4969) |
| 4389 | d8abb91dc8 | nv-guomingz | 2025-06-07 | chore:set the flashinfer to 0.2.5. (#5004) |
| 4390 | f414a079ad | Bo Li | 2025-06-07 | chore: Change the type annotations of input_ids and position_ids to int32. (#4632) |
| 4391 | 7dce328ad6 | Ivy Zhang | 2025-06-07 | [TRTLLM-5692][tests] Add speculative decoding test cases on torch flow (#4940) |
| 4392 | 0c7dd660d8 | nv-guomingz | 2025-06-07 | fix:https://nvbugs/5324248 (#4973) |
| 4393 | 20d0649f19 | Jinyang Yuan | 2025-06-06 | [feat] Support XQA-based MLA on SM120 (#4858) |
| 4394 | 75d020cf07 | Fanrong Li | 2025-06-06 | fix: fix cuda graph padding for spec decoding (#4853) |
| 4395 | eeb555e37b | Anthony Chang | 2025-06-06 | chore: memoize weight shuffle index to speed up weight preproc in moe_backend=TRTLLM (#4826) |
| 4396 | 1b963c17c0 | QI JUN | 2025-06-06 | CI: waive test_llm_multi_node_with_postproc (#4977) |
| 4397 | 564472168e | xinhe-nv | 2025-06-06 | test: [CI] Add failed cases into waives.txt (#4966) |
| 4398 | a761cc2f8d | juney-nvidia | 2025-06-06 | doc: refinement based on Julien's feedbacks (#4967) |
| 4399 | ec50684d80 | QI JUN | 2025-06-06 | Revert "fix a bug of global cuda graph dummy request" (#4970) |
| 4400 | 37ac564190 | juney-nvidia | 2025-06-05 | doc: expose Large-scale EP design and implementation tech blog in the main… (#4960) |
| 4401 | d2c311c9d3 | Yiteng Niu | 2025-06-05 | infra: update jnlp version in container image (#4944) |
| 4402 | 5a5427f86e | Kaiyu Xie | 2025-06-05 | blog: Scaling Expert Parallelism in TensorRT-LLM (Part 1: Design and Implementation of Large-scale EP) (#4958) |
| 4403 | 180b91f957 | qsang-nv | 2025-06-05 | update fmha_v2 (#4895) |
| 4404 | 51652b9b2b | dongjiyingdjy | 2025-06-05 | feat : add PositionEmbeddingType=0 to xqa support (#4934) |
| 4405 | bfa877a22e | QI JUN | 2025-06-05 | Fix: fix autodeploy (#4957) |
| 4406 | 154f7cc40a | QI JUN | 2025-06-05 | fix a bug of global cuda graph dummy request (#4894) |
| 4407 | 7e921c78b5 | Yiqing Yan | 2025-06-05 | Waive L0 tests (#4953) |
| 4408 | 3eae58ca36 | Shunkangz | 2025-06-05 | Add disaggregated unittest (#4899) |
| 4409 | a1526356aa | ixlmar | 2025-06-05 | [TRTLLM-5630] restore free_gpu_memory_fraction=0.9 in tests (#4859) |
| 4410 | b8c5e3892b | QI JUN | 2025-06-05 | Revert "fix: build_config in TorchLlmArgs and avoid invalid args" (#4949) |
| 4411 | d5a8079eb6 | QI JUN | 2025-06-05 | Revert "[infra] Unwaive unittests/_torch" (#4950) |
| 4412 | 743fb0a159 | Lucas Liebenwein | 2025-06-05 | [AutoDeploy] _AutoDeployLlmArgs as primary config object (#4891) |
| 4413 | 91e8d43d66 | QI JUN | 2025-06-05 | CI: waive test_llm_get_queued_stats (#4945) |
| 4414 | 6437756da8 | ixlmar | 2025-06-05 | fix: handle OOMs during KV cache estimation (#4690) |
| 4415 | 1c3091c63b | xinhe-nv | 2025-06-05 | tests: [TRTQA-2906] add benchmark serving tests (#4901) |
| 4416 | ddbaa5ef80 | Netanel Haber | 2025-06-05 | Only pass `fast_build=true` to non-pytorch backend (#4920) |
| 4417 | 9ceef983c0 | Yiqing Yan | 2025-06-05 | Waive L0 tests (#4927) |
| 4418 | 50a74a1daa | xinhe-nv | 2025-06-05 | tests: fix 5273697 (#4685) |
| 4419 | b0d287c9b7 | Shiyu Li | 2025-06-04 | [TRTLLM-4647][fix] Fix the no fusion allreduce hanging (#4594) |
| 4420 | 8433091630 | Mike Iovine | 2025-06-04 | [infra] Unwaive unittests/_torch (#4919) |
| 4421 | f9d45e03a4 | Lucas Liebenwein | 2025-06-04 | [AutoDeploy] deprecate CI post-merge tests and keep them for local testing (#4892) |
| 4422 | 8e0d96fcc6 | Yan Chunwei | 2025-06-05 | fix: LLM invalid arg in a test (#4922) |
| 4423 | 6b3242654e | Yuxian Qiu | 2025-06-05 | fix: Fix broken vanilla moe since FusedMoE refactor. (#4897) |
| 4424 | 1fca654bfd | Yi Zhang | 2025-06-04 | tests: Update gb200 test case (#4754) |
| 4425 | 2bbb6b5976 | ixlmar | 2025-06-04 | chore: introduce KvCacheCreator (#4581) |
| 4426 | 325ccaae3d | Xianjie Qiao | 2025-06-04 | Fix trtllm-bench iter_stats and cuda_graph_batch_sizes error errors. (#4827) |
| 4427 | dd2191c5b3 | Zheng Duan | 2025-06-04 | fix: correct the order of llm request state (#4781) |
| 4428 | 4954780649 | Joosung Yoon | 2025-06-04 | Fix: draft target README and set exclude_input_in_output to False (#4882) |
| 4429 | 35e87b99f3 | Zhanrui Sun | 2025-06-04 | chore: bump version to 0.21.0rc1 (#4896) |
| 4430 | 8d31e16877 | tomeras91 | 2025-06-04 | [TRTLLM-4923][feat] Paged mamba cache (#4822) |
| 4431 | e71de2a13e | Omer Ullman Argov | 2025-06-04 | chore: Mass integration of release/0.20. (#4871) |
| 4432 | ac20159d32 | Yan Chunwei | 2025-06-04 | fix: build_config in TorchLlmArgs and avoid invalid args (#4600) |
| 4433 | e2eea80c1d | QI JUN | 2025-06-04 | Chore: refine comments of prepare inputs method of model engine (#4837) |
| 4434 | 5fa6fbd989 | Yukun He | 2025-06-04 | feat: Enhance AutoTuner inference path and code readability (#4466) |
| 4435 | ded694b1aa | Zheng Duan | 2025-06-04 | feat: cache reuse support (selective cache transfer) in mla cache formatter (#4749) |
| 4436 | b13f8c9cba | Shi Xiaowei | 2025-06-04 | Fix: NVBug 5302895 (#4835) |
| 4437 | c835f06371 | Shunkangz | 2025-06-04 | Refactor the first token response in PD (#4692) |
| 4438 | d64af85e8c | ChristinaZ | 2025-06-04 | Replace memset with data initialization within kernels (#4851) |
| 4439 | 73389d6531 | Mike Iovine | 2025-06-03 | [fix] Fix llama 4 long context (#4809) |
| 4440 | a089aa3225 | Perkz Zheng | 2025-06-04 | [https://nvbugspro.nvidia.com/bug/5300080] Fix the bug of setting attention_chunk_size and enable chunked-attention in the generation-phase by default (#4693) |
| 4441 | 8043d7a03c | Nikita Korobov | 2025-06-03 | feat: update DeepSeek FP8 TRT-LLM Gen cubins (#4643) |
| 4442 | d0eb47d33a | rakib-hasan | 2025-06-03 | [TRTLLM-5053] Refactoring and Unifying the Multimodal input preparation (#4506) |
| 4443 | b4ed4b22f3 | hlu1 | 2025-06-03 | [Arch] Freeze model_config (#4814) |
| 4444 | 2384655c3a | Simeng Liu | 2025-06-03 | chore: Waive examples/test_mistral.py::test_llm_mistral_v1_1gpu. (#4873) |
| 4445 | 19786a7961 | Rashid Kaleem | 2025-06-03 | [Doc] Fix readme for disaggregated serving (#4846) |
| 4446 | 80b4026775 | Yan Chunwei | 2025-06-03 | chore: remove request_error ipc in LLM.submit (#4763) |
| 4447 | 01f29ce38b | pcastonguay | 2025-06-03 | [nvbug 5294316] fix: Fix queued request stats (#4714) |
| 4448 | ae9a6cf24f | Shunkangz | 2025-06-03 | feat: Add integration of etcd (#3738) |
| 4449 | 3fe4a1842a | Enwei Zhu | 2025-06-03 | fix: Register MoeLoadBalancerConfig to serialization.py (#4864) |
| 4450 | 80f9989a1e | Frank | 2025-06-03 | [enhanchment] Add beam width to low latency. (#4812) |
| 4451 | 3de02582dd | Robin Kobus | 2025-06-03 | refactor: Separate DecoderState from GptDecoderBatched (#4700) |
| 4452 | b9263a8e10 | Robin Kobus | 2025-06-03 | fix: max_num_sequences calculation with overlap scheduling (#4532) |
| 4453 | 320195dc0d | hlu1 | 2025-06-02 | [Architecture] Refactor FusedMoE (#4790) |
| 4454 | 141467d4b6 | Iman Tabrizian | 2025-06-03 | Add pre-merge Triton backend tests (#4842) |
| 4455 | fa93eeee84 | ruodil | 2025-06-03 | shorten reqs in con:1 cases and add streaming cases, and add l2 perf … (#4849) |
| 4456 | 8686868531 | Ivy Zhang | 2025-06-03 | tests: [TRTQA-2905] improve timeout report for qa test cases (#4753) |
| 4457 | ec796e44e4 | Yuxian Qiu | 2025-06-03 | feat: add heuristics for checkpoint files prefetching. (#4765) |
| 4458 | 7ce1e1311f | WeiHaocheng | 2025-06-03 | [TRTLLM-5340] fix: remove the accuracy assert on run_majority_vote_aime24.py (#4784) |
| 4459 | e34a1beb72 | Robin Kobus | 2025-06-03 | [nvbugs/5303555] ci: unwaive test_fp8_block_scales_cuda_graph_padding (#4735) |
| 4460 | e013c8cbc2 | Yan Chunwei | 2025-06-03 | fix [nvbug5256044]: bench hang due to llmapi ipc (#4798) |
| 4461 | 380a5d1690 | Fanrong Li | 2025-06-03 | [https://nvbugs/5271281][fix] fix a pd+mtp accuracy issue (#4536) |
| 4462 | 9832787050 | Tian Zheng | 2025-06-03 | [feat] Enable NVFP4 output for TRTLLM attention kernels (#4737) |
| 4463 | 4e2fefc076 | yunruis | 2025-06-03 | upgrade cutlass to 4.0 (#4794) |
| 4464 | 9ae2ce6665 | Po-Wei (Vincent) | 2025-06-02 | [TRTLLM-5502][infra] Add github action to identify if PR is from community (#4824) |
| 4465 | 90aab0596e | Yilin Fan | 2025-06-02 | [fix] Fix Llama4 guradwords failures (#4844) |
| 4466 | 13f68338d2 | Fanrong Li | 2025-06-02 | fix: [https://nvbugspro.nvidia.com/bug/5273945] Unwaive tests for bug-5273945 (#4832) |
| 4467 | 8166649d03 | Yanchao Lu | 2025-06-02 | [Infra] - Minor clean-up and test Ubuntu mirrors (#4829) |
| 4468 | eb2d51a429 | Yilin Fan | 2025-06-01 | [fix] Fix llama4 min-latency mode (#4810) |
| 4469 | 5b4852b7b5 | Enwei Zhu | 2025-06-02 | feat: large-scale EP(part 5: Static EP load balancer with offline statistics) (#4695) |
| 4470 | 7d356efc7d | Fanrong Li | 2025-06-02 | fix: fix accuracy and illegal memory access issues when using mtp + attention dp (#4379) |
| 4471 | 2ce05c3ab4 | Netanel Haber | 2025-06-01 | 'entered copyBlock' format string expects %s, pass string rather than int (#4820) |
| 4472 | bf9cd11fd4 | tomeras91 | 2025-06-01 | [TRTLLM-4783][feat] Mamba2 kernel updates for Nemotron-H (#4494) |
| 4473 | 8039ef45d3 | amirkl94 | 2025-06-01 | CI: Performance regression tests update (#3531) |
| 4474 | 491a09b0c6 | Lucas Liebenwein | 2025-05-31 | [AutoDeploy] Increased Model Coverage Mass Migration Week 2 (#4817) |
| 4475 | 202813f054 | Emma Qiao | 2025-06-01 | Check test names in waive list (#4292) |
| 4476 | 0087bd27ba | Enwei Zhu | 2025-06-01 | [fix] Fix SamplingParams check on n and best_of (#4655) |
| 4477 | 69c7fe8905 | Daniel Cámpora | 2025-05-31 | [TRTLLM-4987][feat] Partial support of context logits in TRTLLMSampler (#4538) |
| 4478 | 25dde49c28 | Enwei Zhu | 2025-06-01 | fix: EP load balancer with MTP layer and route offset by EP rank (#4767) |
| 4479 | 338d6e9f95 | Dom Brown | 2025-05-31 | [nvbug 5305210] fix: Resolve nvbug 5305210 (#4759) |
| 4480 | a02df6aa4b | Yuxian Qiu | 2025-05-31 | fix: re-enable tp/pp for quickstart_advanced.py. (#4766) |
| 4481 | 93c0632ee4 | Yan Chunwei | 2025-05-31 | opt: the perormance for dist-agg streaming generation (#4214) |
| 4482 | c945e92fdb | Emma Qiao | 2025-05-31 | [Infra]Remove some old keyword (#4552) |
| 4483 | 8cb6163a57 | Mike Iovine | 2025-05-30 | [fix] Fix Llama 3.3 70b EAGLE (#4772) |
| 4484 | 49f2f1f8eb | juney-nvidia | 2025-05-30 | Expose new tech blog about DSR1 throughput optimization to the main R… (#4803) |
| 4485 | 3b7120d60e | Tao Li @ NVIDIA | 2025-05-30 | DeepSeek R1 throughut optimization tech blog for Blackwell GPUs (#4791) |
| 4486 | f82e44bbb9 | Yuxian Qiu | 2025-05-30 | fix: [nvbugs/5310520] disable embed_tokens's TP when DP enabled for llama model. (#4758) |
| 4487 | bac22ff7b5 | Pengyun Lin | 2025-05-30 | [feat] support sharegpt downloading in benchmark_serving (#4578) |
| 4488 | 99fdef20c4 | QI JUN | 2025-05-30 | [TRTLLM-5516] perf: replicate dummy request for cuda graph padding (#4729) |
| 4489 | c026dda400 | ixlmar | 2025-05-30 | fix: iteration logging and typing in PyExecutor (#4734) |
| 4490 | 7e6d06d5d7 | ixlmar | 2025-05-30 | feat: estimate GPU mem. usage w/ minimal KV cache (#4574) |
| 4491 | 54200ee8ac | Zheng Duan | 2025-05-30 | fix: random fail of cache router test (#4597) |
| 4492 | f117d6abe9 | Chuang Zhu | 2025-05-30 | Fabric Memory for KV Cache Transfer (#4717) |
| 4493 | ee916da8f1 | Enwei Zhu | 2025-05-30 | test: Waive test_llm_loading_from_ckpt_for_tp2 (#4797) |
| 4494 | 53794b26f8 | xinhe-nv | 2025-05-30 | test: skip test_llm_hf_gemma_quantization_1gpu_vswa on A100 (#4779) |
| 4495 | 55d56f8155 | Thor Johnsen | 2025-05-30 | [JIRA-5226219][fix] Fix Bug in KV cache manager (#4596) |
| 4496 | 36b87b8671 | Aurelien Chartier | 2025-05-29 | chore: fix llm_root when LLM_ROOT is not set (#4741) |
| 4497 | fe359d9df9 | juney-nvidia | 2025-05-30 | Added code owners for AutoDeploy (#4769) |
| 4498 | 5339d367ce | Jinyang Yuan | 2025-05-30 | [perf] Reduce the workspace size of FP4 activation scales for MoE (#4303) |
| 4499 | 3093c747b7 | hlu1 | 2025-05-29 | [Architecture] Redesign Linear module (#4721) |
| 4500 | 31bb650298 | Yilin Fan | 2025-05-29 | Cherry pick feat/llama4 to main (#4739) |
