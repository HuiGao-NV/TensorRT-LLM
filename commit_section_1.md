# Commit Section 1

Commits 1 to 500 (total: 5612)

| # | Commit Hash | Author | Date | Message |
|---|------------|--------|------|--------|
| 1 | 93b0dc7afb | Robin Kobus | 2026-03-16 | [None][refactor] Improve request management in sampler (#11861) |
| 2 | e2df4f487c | dominicshanshan | 2026-03-16 | [https://nvbugs/5955927][fix] Add warm up before aiperf to fix timeout issue. (#12178) |
| 3 | 1797e3fbb9 | Yiyun Lu | 2026-03-16 | [TRTLLM-11288][fix] Adapt LTX2 pipeline to CompilationConfig warmup interface (#12232) |
| 4 | 7fd1c7609f | Xianjie Qiao | 2026-03-16 | [None][fix] Export computed env vars to env_vars.json and fix port allocation in disagg benchmark (#12140) |
| 5 | 1221dffa1e | Qi Zhang (qizh) | 2026-03-16 | [None][fix] Fix W4A16 AWQ bias not applied on SM100 (Blackwell) (#12190) |
| 6 | 8a89887f64 | xinhe-nv | 2026-03-16 | [None][chore] Add failed cases into waives.txt (#12220) |
| 7 | 87658f2256 | Bo Li | 2026-03-16 | [None][chore] Alltoall benchmark script refine (second time). (#12192) |
| 8 | 7686debc3c | Leslie Fang | 2026-03-16 | [None][chore] fix deepep trtllm backend MXFP4 (#12219) |
| 9 | 3de31517be | xinhe-nv | 2026-03-16 | [None][chore] Add failed cases into waives.txt (#12218) |
| 10 | fe77b4075c | Bo Deng | 2026-03-16 | [https://nvbugs/5948878][fix] fix lost requests (#12197) |
| 11 | 80871a21cd | Yiqing Yan | 2026-03-16 | [TRTLLM-10569][infra] Only check the changed files in pre-commit in pre-merge CI (#11379) |
| 12 | 503d6785ba | Bala Marimuthu | 2026-03-15 | [None][doc] AutoDeploy: ad-model-onboard skill updates (#12234) |
| 13 | fc2bf2790d | Yi Zhang | 2026-03-16 | [None][fix] Fix KV cache V2 OOM with separate draft KV cache (EAGLE3/MTP) (#12188) |
| 14 | 9d723146a3 | xinhe-nv | 2026-03-16 | [None][chore] Remove closed bugs (#12222) |
| 15 | fb3ad3104b | Bo Li | 2026-03-16 | [None][doc] Blog18 for NVLinkOneSided AlltoAll. (#12195) |
| 16 | ff07459eae | JunyiXu-nv | 2026-03-16 | [https://nvbugs/5823135][fix] Fix min_tokens not respected when prompt is long (#12166) |
| 17 | b72ee4fd89 | Yi Zhang | 2026-03-16 | [https://nvbugs/5973536][fix] Route DSA attention through MLA custom op for torch.compile compatibility (#12186) |
| 18 | fe9e1a334c | dongfengy | 2026-03-15 | [https://nvbugs/5955188][fix] Fix harmony parsers for agentic coding use cases (#12045) |
| 19 | 5a7076c84f | chenfeiz0326 | 2026-03-16 | [None][fix] Fix Disagg Perf Test No result.xml Bug (#12211) |
| 20 | 8e13bf319e | TensorRT LLM | 2026-03-16 | [None][infra] Check in most recent lock file from nightly pipeline |
| 21 | 677cdf673a | Guoming Zhang | 2026-03-16 | [TRTLLM-9767][feat] Enable attention dp for qwen3-next. (#10218) |
| 22 | b802a3818c | Zhenhuan Chen | 2026-03-16 | [TRTLLM-10929][feat] add fp8 combine in moe_a2a (#11844) |
| 23 | 73cca36b4e | Eran Geva | 2026-03-16 | [None][fix] port retry loop and exception handling (#12225) |
| 24 | 8766e81aa6 | Zhanrui Sun | 2026-03-15 | [None][infra] Waive 9 failed cases for main in post-merge 2593 (#12224) |
| 25 | 4be2fda970 | tcherckez-nvidia | 2026-03-15 | [None][fix] remove test_llm_api_autodeploy.py::TestNemotronSuperV3::t… (#12193) |
| 26 | b20dd452ec | Lizhi Zhou | 2026-03-15 | [TRTLLM-8922][feat] py cache transceiver for gen-first workflow (#11941) |
| 27 | e3e6a38285 | Leslie Fang | 2026-03-15 | [https://nvbugs/5973316][fix] fix deepep with trtllm moe backend and seqlen one (#12158) |
| 28 | afe731482c | TensorRT LLM | 2026-03-15 | [None][infra] Check in most recent lock file from nightly pipeline |
| 29 | 267396cba9 | Leslie Fang | 2026-03-15 | [None][chore] Add explicit error for intermediate size misalignment with fp8 block size (#12101) |
| 30 | 8cdcce9b3d | Joyjit Daw | 2026-03-14 | [None][fix] Add streaming support to no </think> for nemotron model (#12176) |
| 31 | 338088987c | yuanjingx87 | 2026-03-14 | [None][infra] Waive failed A10-PyTorch-1 test in pre-merge (#12207) |
| 32 | 1e0b9eae99 | TensorRT LLM | 2026-03-14 | [None][infra] Check in most recent lock file from nightly pipeline |
| 33 | 9a9dc3c678 | Joyjit Daw | 2026-03-13 | [https://nvbugs/5944411][fix] Handle anyOf parameter schemas in Qwen3Coder tool parser (#12173) |
| 34 | 7754c661d2 | Frida Hou | 2026-03-13 | [None][feat] Add mix-precision checkpoint support in AutoDeploy (#12175) |
| 35 | 390a7fd6b1 | Suyog Gupta | 2026-03-13 | [None][feat] Qwen3.5 perf optimizations (#11581) |
| 36 | 1a2849af63 | Wanli Jiang | 2026-03-13 | [None][fix] Fixed mamba cache issue for pp>1 (#12146) |
| 37 | 3fb931a4bd | Perkz Zheng | 2026-03-13 | [None][feat] add trtllm-gen kernels for glm4.7 and support groupsTokensHeadsQ + e2m1 output (#11643) |
| 38 | 18d02df53a | Yiyun Lu | 2026-03-13 | [TRTLLM-11288][feat] Configurable warmup shapes for VisualGen (#12107) |
| 39 | fb7c6341fe | Zheyu Fu | 2026-03-13 | [TRTLLM-10319][feat] Dynamic draft length on spec decode one-model path (#10860) |
| 40 | b65c9ac82a | Yao Yao | 2026-03-13 | [None][chore] Fix KVCacheManagerV2 shrink for last level and improve init_ratio (#12112) |
| 41 | 27270929e0 | Robin Kobus | 2026-03-13 | [TRTLLM-11207][requirements] Update numpy version to 2 (#11280) |
| 42 | 60091fffc2 | JennyLiu | 2026-03-13 | [None][Test] Add multinode e2e and accuracy cases on DGX-Spark (#12110) |
| 43 | 0fc0cbd1cf | Song Rong | 2026-03-13 | [None][feat] Add flashinfer api for TRTLLMGenFusedMoE (#10453) |
| 44 | b0cec7f529 | TensorRT LLM | 2026-03-13 | [None][infra] Check in most recent lock file from nightly pipeline |
| 45 | e226930194 | xxi | 2026-03-13 | [TRTLLM-11037][bug] Fix MoE DeepEP hang caused by non-deterministic GC (#12060) |
| 46 | 94f94897cf | chenfeiz0326 | 2026-03-13 | [https://nvbugs/5949033][fix] Add 3 Disagg gen_only tests back (#12159) |
| 47 | c0cf5a3706 | CarstyYou | 2026-03-13 | [None][feat] Optimize 6KD fp8 blockscale gemm (#11502) |
| 48 | 0507609b30 | Yan Chunwei | 2026-03-13 | [TRTLLM-10695][ci] add verl stage in CI (#11306) |
| 49 | 2eee70102b | Yibin Li | 2026-03-13 | [TRTLLM-10617][feat] LTX-2 Model Support (#12009) |
| 50 | 5cc0ccd1d1 | tcherckez-nvidia | 2026-03-12 | [https://nvbugs/5973199][fix] support attn-dp TRTLLM-Gen NVFP4 MoE fu… (#12156) |
| 51 | 7c64f4bbd8 | dpitman-nvda | 2026-03-12 | [TRTLLMINF-11][chore] Change image used for Preparation step of CI (#12086) |
| 52 | 0c19c8e96e | Iman Tabrizian | 2026-03-12 | [None][test] Add e2e tests for KV cache connector async loading path (#12053) |
| 53 | dc377983d1 | Chang Su | 2026-03-12 | [#11800][fix] Add keepalive ping tolerance and context.abort to gRPC server (#11992) |
| 54 | f11eea7515 | JunyiXu-nv | 2026-03-12 | [TRTLLM-10303][feat] Deprecate trtllm-serve CLI options (#12106) |
| 55 | adfc542cf5 | Eden | 2026-03-12 | [None][fix] Narrow bare except clause and use identity check for None (#12041) |
| 56 | a4e6745c24 | Guoming Zhang | 2026-03-12 | [TRTLLM-10244][doc] Add deployment guide for Nemotron 3 Super (#12129) |
| 57 | 5dc4318ce3 | chenfeiz0326 | 2026-03-12 | [https://nvbugs/5846166][bug] Fix Disagg Perf Test's MPI Issue and Port Conflict (#12020) |
| 58 | 8de01ac09f | Aurelien Chartier | 2026-03-11 | [None][feat] Add shared expert LoRA support for MoE models in PyTorch backend (#11760) |
| 59 | 1160f19d55 | Stanley Sun | 2026-03-12 | [None][test] Add speculative decoding test with exclude_input_in_output=true (#12080) |
| 60 | a865b6c369 | JunyiXu-nv | 2026-03-12 | [https://nvbugs/5955173][fix] Add abort method for GenerationResultBase (#11970) |
| 61 | e824264306 | o-stoner | 2026-03-12 | [TRTLLM-11092][feat] add support for visual gen FA4 attention backend (#11697) |
| 62 | 803510d6f9 | xinhe-nv | 2026-03-12 | [None][chore] Add failed cases into waives.txt (#12093) |
| 63 | 21a696b671 | JadoTu | 2026-03-12 | [None][feat] Optimize the q3n decode kernel with IO read (#11344) |
| 64 | 959306c0fe | TensorRT LLM | 2026-03-12 | [None][infra] Check in most recent lock file from nightly pipeline |
| 65 | 3f9f7573a4 | Zhanrui Sun | 2026-03-12 | [None][infra] Waive 2 failed cases for main in post-merge 2586 (#12134) |
| 66 | 0ce36516b9 | zhaoyangwang-nvidia | 2026-03-12 | [TRTLLM-11257][infra] Unwaive TestDeepSeekR1::test_fp8_blockscale[throughput_mtp] test case (#12059) |
| 67 | 6739c1182a | Enwei Zhu | 2026-03-12 | [https://nvbugs/5826604][test] Remove test waive for Llama3.1 8B bfloat16 4gpu timeout … (#12092) |
| 68 | 2578637810 | NVShreyas | 2026-03-11 | [None][refactor] parallel vae refactor (#12123) |
| 69 | be2065755c | Iman Tabrizian | 2026-03-11 | [None][fix] Enforce minimum NVSHMEM_QP_DEPTH of 128 for DeepEP low latency (#12100) |
| 70 | bf7142f8d1 | dongfengy | 2026-03-11 | [https://nvbugs/5955170][fix] Disable TRTLLM GEN Routing PDL due to nan issue (#11994) |
| 71 | 7479423391 | Patrice Castonguay | 2026-03-11 | [None][chore] Unwaiving disagg tests failing with address in use error (#12085) |
| 72 | ae2bd166b8 | yuanjingx87 | 2026-03-11 | [None][infra] Update CI allow list (#12119) |
| 73 | ea4d4d1bde | Lain | 2026-03-11 | [None][feat] Enable non-gated activation to the new MoE test (#11996) |
| 74 | b79f4c7700 | Ludwig Schneider | 2026-03-11 | [https://nvbugs/5923949][fix] Improve NCCL library load stability (#12015) |
| 75 | 0350b7f229 | jthomson04 | 2026-03-11 | [None][fix] Enable more KV connector priority tests in CI (#11892) |
| 76 | f90b35fc26 | Guiju Zhang | 2026-03-11 | [None][fix] Fix ValueError and missing decoding statistics for MTP (#12063) |
| 77 | 873a01e7a2 | Hrithvik Alex | 2026-03-11 | [None][fix] Split mContextChunkSize into per-target/draft fields (#12058) |
| 78 | 906781bf4c | Iman Tabrizian | 2026-03-11 | [https://nvbugs/5948539][fix] Fix disagg gen-only benchmark (#12091) |
| 79 | be57adba48 | Grzegorz Kwasniewski | 2026-03-11 | [TRTLLM-11928][feat] Fix sharding overwrite with multiple graph module (#12051) |
| 80 | d3628467e1 | Shi Xiaowei | 2026-03-11 | [TRTLLM-9523][feat] Adapting the transceiver to manager v2 (step 6) (#11978) |
| 81 | 9b545e2c47 | Eran Geva | 2026-03-11 | [https://nvbugs/5936322][fix] Fix sporadic port collision in multigpu AutoDeploy tests (#11913) |
| 82 | 9c26e2217a | William Zhang | 2026-03-11 | [None][fix] Various fixes for agentic flow (#12061) |
| 83 | 73fca4e0bd | Wanli Jiang | 2026-03-11 | [None][feat] Mamba optimization and mixed quantization support for nemotron-h (#11972) |
| 84 | 627c96cf88 | Zhenhua Wang | 2026-03-11 | [None][chore] re-enable benchmark test in post merge (#12035) |
| 85 | 5319f83e3d | Jie Li | 2026-03-11 | [None][chore] Waive mpi hang test case (#12077) |
| 86 | b8c96d41a9 | Zhanrui Sun | 2026-03-11 | [None][infra] Waive 2 failed cases for main in post-merge 2584 (#12108) |
| 87 | 611ca65100 | xinhe-nv | 2026-03-11 | [None][test] add Perf sanity gb200 test into QA test db (#11882) |
| 88 | 298b6c8ee6 | bhsueh_NV | 2026-03-11 | [https://nvbugs/5961430][fix] Fix CI issue of Mistral Large3  (#12073) |
| 89 | fa31ce067f | Yuan Tong | 2026-03-11 | [TRTLLM-11366][feat] Add dedicated virtual memory tag for model weights, configurable restore mode (#11889) |
| 90 | e03e3611b1 | Abby Wei | 2026-03-11 | [None][fix] Fix Upload Build Info branch and run in post always (#12025) |
| 91 | f7255e0e1a | Qi Zhang (qizh) | 2026-03-11 | [None][feat] 2FP4 / Arcquant. (#11333) |
| 92 | ecf0057bbb | Bo Li | 2026-03-11 | [None][chore] Refine AlltoAll benchmark scripts. (#11649) |
| 93 | 8dddfb41c4 | yuanjingx87 | 2026-03-10 | [None][chore] Bump version to 1.3.0rc8 (#12090) |
| 94 | 71ffd8b1ae | jthomson04 | 2026-03-10 | [None][fix] Revert "[None][chore] KV Connector Refactor (#11078)" (#11872) |
| 95 | bf897e6446 | Izzy Putterman | 2026-03-10 | [None][fix] MTP Advanced Sampling Topk IMA (#12088) |
| 96 | 4dee46edfe | chenfeiz0326 | 2026-03-11 | [https://nvbugs/5919026][fix] Pass sparse_attn_config from effective_draft_config for one-model draft KV cache (#12032) |
| 97 | a154da62cc | Liao Lanyu | 2026-03-11 | [TRTLLM-11276][fix] Fix Kimi-K2.5 accuracy test skip condition and reference configs (#11930) |
| 98 | 97e98ef51d | TensorRT LLM | 2026-03-11 | [None][infra] Check in most recent lock file from nightly pipeline |
| 99 | bba2981f01 | xinhe-nv | 2026-03-11 | [None][chore] Add failed cases into waives.txt (#12047) |
| 100 | 2afe11ddbd | jthomson04 | 2026-03-10 | [None][fix] Improve KV Event Batching (#11883) |
| 101 | 04cefe271d | Grzegorz Kwasniewski | 2026-03-11 | [TRTLLM-11535][feat] Fixed NVFP4 sharding (#11618) |
| 102 | 885fedd7a7 | Chang Liu | 2026-03-10 | [https://nvbugs/5963896][fix] Remove test `test_visual_gen_quickstart` on A10 (#12048) |
| 103 | c96c68f9b8 | tcherckez-nvidia | 2026-03-10 | [None][feat] NVFP4 TRTLLM-Gen MoE for AutoDeploy (Nemotron Super) (#11652) |
| 104 | 2bc2acda4f | mpikulski | 2026-03-10 | [https://nvbugs/5708901][perf] reduce logprobs=0 overhead in TorchSampler (#11983) |
| 105 | 3ce0ec8e20 | William Zhang | 2026-03-10 | [TRTLLM-11265][feat] Implement dynamic resolution for Nemotron VL (#11894) |
| 106 | 6f3acc0614 | nvxuanyuc | 2026-03-10 | [https://nvbugs/5892646][perf] Long-sequence token-parallel optimization for DSA indexer prefill (#11871) |
| 107 | f20346c38d | fredricz-20070104 | 2026-03-10 | [None][test] Fix disagg sku (#12065) |
| 108 | 9a070ed709 | Kaiyu Xie | 2026-03-10 | [TRTLLM-10421][perf] Add fused cat+fp8_quantize CUDA kernel for DSA indexer (#11899) |
| 109 | 72598decdc | Zhanrui Sun | 2026-03-10 | [None][infra] Waive 1 failed cases for main in post-merge 2582 (#12069) |
| 110 | 1fef88e95d | Stefan Niebler | 2026-03-10 | [None][chore] Improve sampler performance by replacing torch.where with masked_fill_ (#11949) |
| 111 | 81350b7045 | yingguo-trt | 2026-03-10 | [None][chore] Align perf benchmark output format (#12067) |
| 112 | 39d294b859 | Yiqing Yan | 2026-03-10 | [TRTLLM-11135][fix] Fix vulnerabilities protobuf and aiohttp (#11898) |
| 113 | a20de88324 | JunyiXu-nv | 2026-03-10 | [https://nvbugs/5937478][fix] Fix DS v32 tool calling type and parse error (#11935) |
| 114 | b411e149bf | JunyiXu-nv | 2026-03-10 | [TRTLLM-11246][feat] Add tool parser support for GLM-4 models (#11986) |
| 115 | 51e3508765 | fredricz-20070104 | 2026-03-10 | [None][test] Add QA's perf test cases with L0 local mode (#12022) |
| 116 | 3139ffa798 | TensorRT LLM | 2026-03-10 | [None][infra] Check in most recent lock file from nightly pipeline |
| 117 | 8d8b84bc49 | Yao Yao | 2026-03-10 | [TRTLLM-7784][feat] Basic SSM support in KVCacheManagerV2 (#11976) |
| 118 | e3d4ba7ba7 | Kaiyu Xie | 2026-03-10 | [None][chore] Clarify DCO sign-off and co-author guidelines in AGENTS.md (#12034) |
| 119 | 35ccedde58 | tcherckez-nvidia | 2026-03-09 | [None][feat] Add AD model list validation checks to pre-commit and PR… (#12036) |
| 120 | 2fe7b1474e | Pamela Peng | 2026-03-09 | [https://nvbugs/5820511][fix] Upgrade Cutlass version (#11956) |
| 121 | 7747f255b8 | tcherckez-nvidia | 2026-03-09 | [None][feat] Add Auto-Deploy dashboard failures analysis skill (#12033) |
| 122 | 69de4a60e7 | NVShreyas | 2026-03-09 | [None][feat] NIXL support for hybrid model cache transfer (#11608) |
| 123 | 3b82b6cac0 | tburt-nv | 2026-03-09 | [None][chore] waive test_visual_gen_quickstart (#12043) |
| 124 | d3a16b3298 | Guiju Zhang | 2026-03-09 | [TRTLLM-11045][feat] Integrate SA with EAGLE3 and PARD (#11878) |
| 125 | ae2dc3d671 | Lain | 2026-03-09 | [None][feat] Add silu to trtllm-gen MoE (#11663) |
| 126 | 7a68c42a23 | tburt-nv | 2026-03-09 | [None][chore] limit tileiras to CUDA13.1 (#12042) |
| 127 | 27cab47f15 | Robin Kobus | 2026-03-09 | [https://nvbugs/5924144][test] unwaive cpp/test_unit_tests.py::test_unit_tests[kernels-80] (#11902) |
| 128 | 9fe24dbee0 | sunnyqgg | 2026-03-09 | [None][feat] Upgrade xgrammar from 0.1.25 to 0.1.32 (#12016) |
| 129 | 34a915377f | Yiyun Lu | 2026-03-09 | [https://nvbugs/5863806][fix] Fix Python string truthiness bug in FMHA cubin selection (#11909) |
| 130 | a176d83478 | tcherckez-nvidia | 2026-03-09 | [None][fix] Fix the model list as it had a dup model (#12029) |
| 131 | d704b5e889 | Zhenhua Wang | 2026-03-09 | [None][chore] Remove visual_gen benchmark test from YAML (#12027) |
| 132 | 5b4ff40ac5 | Gal Hubara-Agam | 2026-03-09 | [None][chore] AutoDeploy: re-enable nvfp4 superv3 accuracy test (#11945) |
| 133 | d17046d4fc | Zhanrui Sun | 2026-03-09 | [None][infra] Waive 5 failed cases for main in post-merge 2578 (#12023) |
| 134 | cf484a0cb8 | yufeiwu-nv | 2026-03-09 | [None][test] Fix model_name starcoder_15b is not in allowed_models issue (#11981) |
| 135 | 1074aa91b8 | Yukun He | 2026-03-09 | [TRTLLM-11148][perf] _prepare_inputs host time optimization (#11704) |
| 136 | 91233d5c41 | Li Min | 2026-03-09 | [TRTLLM-10407][feat] Integrate CuTE DSL top-k kernel for Blackwell (#11900) |
| 137 | 165b61cb5d | yingguo-trt | 2026-03-09 | [https://nvbugs/5948878][fix] Implement workaround for ClientPayloadError (#12018) |
| 138 | d1ac900c4a | Zhanrui Sun | 2026-03-09 | [None][infra] Waive 7 failed cases for main in post-merge 2576 (#12014) |
| 139 | 76a70b4322 | Liao Lanyu | 2026-03-09 | [TRTLLM-11276][chore] Expose use_python_scheduler in SchedulerConfig and add UTs/ITs for python scheduler (#11884) |
| 140 | 18af5efd8f | Emma Qiao | 2026-03-09 | [None][infra] Unwaive 2 cases on rtx-pro-6000d (#12003) |
| 141 | ba0ad133c1 | Wanli Jiang | 2026-03-09 | [None][fix] Use try/except fallback for Pydantic ValidatorIterator in chat message parsing (#11903) |
| 142 | 7b4da2bb99 | Kanghwan | 2026-03-08 | [TRTLLM-11342][fix] Fix FLUX.1 TeaCache polynomial coefficients and default t… (#12007) |
| 143 | 357378b766 | Jiagan Cheng | 2026-03-09 | [None][feat] Use max_gpu_total_bytes to control v2's capacity (#11907) |
| 144 | 066ca48660 | Ivy Zhang | 2026-03-09 | [https://nvbugs/5823783][test] add qa test case for trust-remote-code on multinode failure (#11905) |
| 145 | 06ec49235a | TensorRT LLM | 2026-03-09 | [None][infra] Check in most recent lock file from nightly pipeline |
| 146 | 02c8a94820 | Zhenhua Wang | 2026-03-09 | [TRTLLM-11134][feat] export VisualGen API and update doc (#11911) |
| 147 | db533cff87 | Leslie Fang | 2026-03-09 | [None][chore] Unwaive some skip for trtllm moe backend (#11975) |
| 148 | 4c15db0bfa | Po-Han Huang (NVIDIA) | 2026-03-09 | [https://nvbugs/5732958][bug] Fix TestLlama4MinLatency::test_llama_allclose_to_hf failure (#10191) |
| 149 | 6ec0aad7ca | Lucas Liebenwein | 2026-03-08 | [None][infra] Update AutoDeploy CODEOWNERS coverage (#12013) |
| 150 | 89acff31cb | tcherckez-nvidia | 2026-03-09 | Model update 260308 (#12011) |
| 151 | 595a51dbea | Lucas Liebenwein | 2026-03-08 | [#11166][infra] AutoDeploy: improve test organization in CI and add overview doc (#11291) |
| 152 | f931f4e9e2 | Simeng Liu | 2026-03-08 | [TRTLLM-11159][feat] Wire KVCacheBlock to UnifiedBlockTree, replacing mPrevBlock/mNextBlocks with lookup-node pointers. (#11919) |
| 153 | 69b6203a42 | tcherckez-nvidia | 2026-03-08 | [None][feat] add ReLU2 NVFP4 fusion for AutoDeploy with tests (#11957) |
| 154 | 5eb8eab4f8 | Abby Wei | 2026-03-08 | [TRTLLM-10956][infra] Skip updating gitlab status for GenPostMergeBuilds (#11954) |
| 155 | b9bd3d47ad | TensorRT LLM | 2026-03-08 | [None][infra] Check in most recent lock file from nightly pipeline |
| 156 | 6b04973331 | chenfeiz0326 | 2026-03-07 | [None][fix] Fix Collect Perf Sanity Result's import requests Error (#12002) |
| 157 | a0a9e330eb | Yanchao Lu | 2026-03-07 | Update tests/integration/test_lists/waives.txt |
| 158 | 0579ac65ee | yingguo-trt | 2026-03-05 | [None][chore] Fix/disagg perf failure detection (#11904) |
| 159 | f4593cf31f | Guoming Zhang | 2026-03-05 | [None][doc] Replace the TensorRT-LLM with TensorRT LLM (#11914) |
| 160 | 9c6ce75b38 | Patrice Castonguay | 2026-03-04 | [https://nvbugs/5949098][doc] Fixing docs links (#11912) |
| 161 | 2d9ed59241 | bhsueh_NV | 2026-03-04 | [https://nvbugs/5936273][fix] Fix bugs of Mistral Large3 (#11885) |
| 162 | 07fbb5d1c5 | heyuhhh | 2026-03-02 | [https://nvbugs/5762822][chore] Unwaive longbenchV2 test (#11647) |
| 163 | 2f725eae08 | Li Min | 2026-03-02 | [https://nvbugs/5775256] [fix] Reopen fp8_dsl_fused_moe ut. (#11779) |
| 164 | b548320b81 | Iman Tabrizian | 2026-02-27 | [https://nvbugs/5875522][docs] Add known issue for disaggregated serving hang with asymmetric PP/TP (#11789) |
| 165 | 05bb5c1203 | Frank | 2026-02-26 | [https://nvbugs/5889841][fix] Add custom option class to allow subcommand help to work. (#11722) |
| 166 | 75e038ef8b | yingguo-trt | 2026-02-26 | [None][feat] add sanity tests for release1.2 version (#11738) |
| 167 | 039b06f6d2 | Yukun He | 2026-02-26 | [https://nvbugs/5859881][fix] Unwaive test (#11716) |
| 168 | 1dcb6ec697 | peaceh-nv | 2026-02-26 | [https://nvbugs/5809169][unwaive] Unwaive TestGPTOSS test (#11416) |
| 169 | 656091bf00 | yuanjingx87 | 2026-03-06 | [None][infra] Update CI allow list 20260305 (#11965) |
| 170 | 86e0282c65 | Chenghao Zhang | 2026-03-06 | [None][chore] Autodeploy: add models for sprint (#11999) |
| 171 | dd8ffbdd96 | TensorRT LLM | 2026-03-07 | [None][infra] Check in most recent lock file from nightly pipeline |
| 172 | cc16289dfe | Wanli Jiang | 2026-03-07 | [None][feat] Optimize by fuse nvfp4_quant to layernorm_gated for mamba2_mixer (#11473) |
| 173 | 2eb332cf5a | JunyiXu-nv | 2026-03-07 | [TRTLLM-11290][feat] Enable trtllm-serve E2E tests (#11985) |
| 174 | 10348f80fd | Chang Liu | 2026-03-06 | [None][perf] Add Triton FP8 blockwise quant kernel and autotuner bucket-skip for visual gen (#11854) |
| 175 | 2087b247f1 | Robin Kobus | 2026-03-07 | [None][refactor] Request management in ScheduledRequests (#11784) |
| 176 | 7dbda08444 | Kanghwan | 2026-03-06 | [TRTLLM-11189][fix] Fix TeaCache broken caching for FLUX.1 and FLUX.2 (#11868) |
| 177 | d1ba3b8620 | NVShreyas | 2026-03-06 | [TRTLLM-11093][feat] add 5D A2A for fused ulysses (#11787) |
| 178 | 5918348b14 | Chang Su | 2026-03-06 | [#11578][feat] support multimodal image input in gRPC server (#11800) |
| 179 | 498b25cb60 | NVShreyas | 2026-03-06 | [TRTLLM-11259][perf] Parallel VAE harness and implementation for WAN (#11875) |
| 180 | 427369e8fc | Hiroyoshi Komatsu | 2026-03-07 | [#2912][feat] Support Cohere Command A model  (#11505) |
| 181 | ac8bc6ed11 | Balaram Buddharaju | 2026-03-06 | [TRTLLM-11057][feat] Add Helix CP support for DSV3.2 (#11507) |
| 182 | 22c47069d0 | chenfeiz0326 | 2026-03-07 | [https://nvbugs/5846166][fix] Update Perf Triage Scripts to Fix gen_only issue (#11802) |
| 183 | 5b0c956bcb | o-stoner | 2026-03-06 | [TRTLLM-11189][fix] VisualGen isolated TeaCache Wan fix (#11964) |
| 184 | b5a4e34218 | Chenghao Zhang | 2026-03-06 | [#11422][feat] AutoDeploy: Piecewise cudagraph support Prototype (#11515) |
| 185 | 4dc7bc525f | Yihan Wang | 2026-03-06 | [None][fix] Refine tests/unittest/_torch/flashinfer/test_trtllm_flashinfer_symbol_collision.py to reduce jit-compile time (#11890) |
| 186 | dc740c20a9 | Yiqing Yan | 2026-03-06 | [TRTLLM-11155][infra] Run multi-GPU tests even single-GPU tests are failed when use --disable-fail-fast (#11740) |
| 187 | b94656c0bc | Emma Qiao | 2026-03-06 | [TRTLLM-11284][infra] Move large models test to post-merge (#11933) |
| 188 | 7f458abc1e | Bala Marimuthu | 2026-03-05 | [#10245][feat] AutoDeploy: Support Finegrained FP8 quantization (#10897) |
| 189 | 191e349e4b | Pengbo Wang | 2026-03-06 | [https://nvbugs/5624818][fix] Add unittest for GPT-OSS non-paged_context_fmha (#11415) |
| 190 | c6c6dc118a | Balaram Buddharaju | 2026-03-05 | [None][feat] Avoid duplicated computation with ADP + Helix CP in GQA (#11891) |
| 191 | f639e8b88e | bhsueh_NV | 2026-03-06 | [https://nvbugs/5819048][fix] unwaive test of qwen3-235b eagle3 (#11969) |
| 192 | a7c0af5652 | bhsueh_NV | 2026-03-06 | [https://nvbugs/5896577][fix] fix bug of mistral large3 with eagle (#11942) |
| 193 | a018c48ef0 | Yuxian Qiu | 2026-03-06 | [None][fix] Remove incorrect Python import style rule from AGENTS.md (#11940) |
| 194 | 93ac4a0926 | yufeiwu-nv | 2026-03-06 | [None][test] Fix deepseek-r1 OOM issue for H100 perf test (#11948) |
| 195 | 93a62dc900 | Zhanrui Sun | 2026-03-06 | [None][infra] Waive 4 failed cases for main in post-merge 2571 (#11968) |
| 196 | dd61fd5eeb | xxi | 2026-03-06 | [TRTLLM-11036][feat] Enable new moe test and clean the legacy moe test in the CI (#11817) |
| 197 | c6dbef27f0 | TensorRT LLM | 2026-03-06 | [None][infra] Check in most recent lock file from nightly pipeline |
| 198 | e699f23251 | Daniel Stokes | 2026-03-06 | [None][feat] Add support for bidirectional sliding window attention mask to fmha_v2 (#11212) |
| 199 | 5f1fb7cf32 | Chang Su | 2026-03-05 | [#11578][fix] Use string stop/bad words in gRPC proto instead of pre-tokenized TokenSequence (#11888) |
| 200 | 497b07d6cf | tcherckez-nvidia | 2026-03-05 | [None][chore] Update model list (#11827) |
| 201 | 4786834382 | Izzy Putterman | 2026-03-05 | [None][feat] External Drafter One Model (#11758) |
| 202 | 6062df4f44 | Ethan Kou | 2026-03-05 | [None][chore] Use cluster service discover in disagg CI tests (#11242) |
| 203 | 517ee94938 | sunnyqgg | 2026-03-06 | [None][fix] Fix nemotron super MTP crash on SM90 (#11807) |
| 204 | 2ee7dbae9d | Jin Li | 2026-03-05 | [None][feat] Run extra general warmup to warm up memory pool (#10340) |
| 205 | 2f4ed7dc84 | Zhenhua Wang | 2026-03-05 | [TRTLLM-11101][feat] VisualGen benchmarking script (#11651) |
| 206 | 5b0e8a9290 | 노란토끼 | 2026-03-05 | [None][fix] Prevent RuntimeError from dict mutation during iteration in EXAONE MoE weight mapper (#11862) |
| 207 | 9da717a495 | Bala Marimuthu | 2026-03-04 | [#11755][feat] AutoDeploy onboarding agent + Kimi K2.5 AD modeling code (#11780) |
| 208 | 17921f8d5e | Abby Wei | 2026-03-05 | [TRTLLM-10956][infra] Support build-only mode for GenPostMergeBuilds job (#11895) |
| 209 | 12f2f3938d | Lizhi Zhou | 2026-03-05 | [https://nvbugs/5907477][chore] unwaive test (#11896) |
| 210 | 22007ca14d | xinhe-nv | 2026-03-05 | [None][fix] remove leak check for kimi (#11825) |
| 211 | 559828c5bb | TensorRT LLM | 2026-03-05 | [None][infra] Check in most recent lock file from nightly pipeline |
| 212 | 4e735db8fb | Zhanrui Sun | 2026-03-05 | [None][infra] Waive 1 failed cases for main in pre-merge 29212 (#11929) |
| 213 | e01c38f83a | ChristinaZ | 2026-03-05 | [None][feat] Add support for expert_number<=2048 and K<=32 (#11510) |
| 214 | 011af4c72c | Thor Johnsen | 2026-03-04 | [None][feat] Separate radix search tree implementation (#10862) |
| 215 | 460889fa0c | Chien-Chun Hung | 2026-03-04 | [None][chore] Fix format issue in tensorrt_llm/serve/openai_server.py (#11920) |
| 216 | 5b81307785 | Taylor Yeonbok Lee | 2026-03-04 | [#11819][fix] Disable preload for Llama4 scout (#11873) |
| 217 | e3788f3870 | Mike Iovine | 2026-03-04 | [None][chore] Deprecate eagle3 2-model (#11761) |
| 218 | 234eb83111 | Gal Hubara-Agam | 2026-03-04 | [None][fix] Refactor nanoV3+superV3 accuracy tests to load example config (#11458) |
| 219 | e4332e0831 | Bala Marimuthu | 2026-03-04 | [None][fix] Qwen3.5 fix positions ids input for text-only usage (#11877) |
| 220 | cb231c5b08 | peihengh | 2026-03-04 | [https://nvbugs/5930934][fix] Fix OOM hang with NCCL_SYMMETRIC fallback during long-context inference (#11870) |
| 221 | e0f54b38da | Zhenhua Wang | 2026-03-04 | [None][chore] Handle failure in auto-assign author workflow (#11906) |
| 222 | b15062ed81 | jiahanc | 2026-03-04 | [None][chore] Update autotuner (#11859) |
| 223 | 941a245281 | Yiyun Lu | 2026-03-04 | [https://nvbugs/5946303][fix] Fix incorrect GPU timing in time breakdown under overlap scheduler (#11860) |
| 224 | 2bae787849 | Joosung Yoon | 2026-03-04 | [None][fix] Update check_is_moe into support mlp_layer_types after config.json update (#11477) |
| 225 | e3d1e55606 | Kaiyu Xie | 2026-03-04 | [None][fix] Fix typos, grammar, and formatting in comments and docstrings (#11826) |
| 226 | d3536f15fa | Stefan Niebler | 2026-03-04 | [TRTLLM-10852][feat] Enhance logprobs functionality to always return prompt token logprobs in prompt logprobs (#11235) |
| 227 | 2dbd154534 | Bo Li | 2026-03-04 | [TRTLLM-9392][feat] Support MoE output to alltoall's workspace for all the quantization recipe of trtllm-gen. (#11449) |
| 228 | a9d247fe24 | heyuhhh | 2026-03-04 | [None][doc] Add sparse attention tech blog (#11644) |
| 229 | 77d48e0989 | Zhanrui Sun | 2026-03-04 | [None][infra] Waive 3 failed cases for main in post-merge 2566 (#11881) |
| 230 | a106419b28 | capyun007 | 2026-03-04 | [#11666][fix] Fix inmemory model dir detection (#11753) |
| 231 | 72091b37e7 | Tri Dao | 2026-03-03 | [None][feat] Support mix quantization between shared experts and routed experts for dsv3 (#11215) |
| 232 | 4f2a230db3 | Ziyi Xiong | 2026-03-04 | [https://nvbugs/5927620][fix] Override mMaxAttentionWindow with the actual largest window size (#11842) |
| 233 | 0518cc4556 | TensorRT LLM | 2026-03-04 | [None][infra] Check in most recent lock file from nightly pipeline |
| 234 | 1c2afe9b2f | JunyiXu-nv | 2026-03-04 | [#10009][fix] Fix json_schema response_format to support OpenAI API w… (#11497) |
| 235 | 763bce523b | Iman Tabrizian | 2026-03-03 | [None][test] Enable DeepGemm + DeepEPLowLatency MoE test combination (#11876) |
| 236 | 54d14df3bc | JunyiXu-nv | 2026-03-04 | [TRTLLM-11184][feat] Explicit video encode format support (#11830) |
| 237 | 0a5a5e78a2 | Ziyi Xiong | 2026-03-04 | [https://nvbugs/5919026][fix] Fix AttributeError when DSA indexer accesses non-DSA kv_cache_manager (#11858) |
| 238 | c2c9a42de0 | yuanjingx87 | 2026-03-03 | [None][chore] Bump version to 1.3.0rc7 (#11864) |
| 239 | 7c745395b8 | Liao Lanyu | 2026-03-04 | [None][feat] Add Kimi-K2.5 text model support (NVFP4) (#11777) |
| 240 | 6041a78b61 | Ziyi Xiong | 2026-03-04 | [https://nvbugs/5941681][fix] Handle dict type for speculative_config (#11828) |
| 241 | e99d74f1da | Guiju Zhang | 2026-03-03 | [TRTLLM-11042][feat] Implement suffix automaton on device for spec and support one model (#11434) |
| 242 | e6374a85bc | Venky | 2026-03-03 | [None][chore] Refresh inferenceX configs in recipes (#11595) |
| 243 | 3606afa7a2 | dpitman-nvda | 2026-03-03 | [TRTLLMINF-9][chore] Use checkoutFile in mergeWaiveList to avoid full clone (#11794) |
| 244 | 2ba9140dcb | Chenjie Luo | 2026-03-03 | [None][feat] Add a flag in trtllm serve to support overriding kv cache dtype (#11487) |
| 245 | 3d348ab32b | Lucas Liebenwein | 2026-03-03 | [None][refactor] Revisit attention interface for AutoDeploy (#11796) |
| 246 | 695d7a0bdd | Kaiyu Xie | 2026-03-03 | [TRTLLM-9939][perf] Short-sequence MHA optimization for DSA MLA prefill (#11677) |
| 247 | 5d13ebb1b7 | xinhe-nv | 2026-03-03 | [None][test] update waive list (#11815) |
| 248 | fd1738a926 | Yi Zhang | 2026-03-03 | [None][fix] Fix OOM issue/dummy request allocation/chunked prefill/pp for KV Cache Manager V2 (#11710) |
| 249 | 11fc7cb799 | xinhe-nv | 2026-03-03 | [None][test] fix flaky issues (#11814) |
| 250 | 6367dc23c5 | yufeiwu-nv | 2026-03-03 | [None][test] Fix wrong lora config (#11818) |
| 251 | e9c495eca9 | Emma Qiao | 2026-03-03 | [None][infra] Fix a typo in waives.txt (#11852) |
| 252 | d5f6053f48 | Zhenhua Wang | 2026-03-03 | [None][chore] a GitHub Action to assign the PR to the author (#11673) |
| 253 | fa76c44041 | Bo Deng | 2026-03-03 | [https://nvbugs/5936502][fix] remove dead codes (#11813) |
| 254 | 58e9f1b8f0 | Zhanrui Sun | 2026-03-03 | [None][infra] Waive failed cases for main for post-merge 2564 (#11848) |
| 255 | ae6bc675d1 | Wanli Jiang | 2026-03-03 | [None][fix] Fix SM120 issue for rms_norm with nvfp4_quant_fusion (#11774) |
| 256 | 9afa7a43a2 | xinhe-nv | 2026-03-03 | [None][test] add b200 multi nodes tests db (#11783) |
| 257 | ec2d8f4122 | TensorRT LLM | 2026-03-03 | [None][infra] Check in most recent lock file from nightly pipeline |
| 258 | 998e939d20 | NVShreyas | 2026-03-02 | [None][fix] remove torch compile models arg (#11836) |
| 259 | 8553560c4b | ruodil | 2026-03-03 | [None][test] add deepseek RCCA perf test case (#11736) |
| 260 | 9c9d00dcaa | William Zhang | 2026-03-02 | [https://nvbugs/5938603][fix] Fix E/PD disagg chunked prefill bug (#11805) |
| 261 | c6b1ed4d9d | benzh-2025 | 2026-03-03 | [https://nvbugs/5863912][fix] Fix with move launch_dependent_grids after tmem free (#11812) |
| 262 | a632f0ffd6 | Kazuhiro Yamasaki | 2026-03-03 | [https://nvbugs/5689262][fix] use proper tokens when exclude_input_in_output is true (#9453) |
| 263 | 971c4f0956 | jthomson04 | 2026-03-02 | [None][fix] Fix overly aggressive capacity scheduler (#11731) |
| 264 | 812d2ce938 | Taylor Yeonbok Lee | 2026-03-02 | [#11726][feat] AutoDeploy: Fuse gemms of mixed children (#11793) |
| 265 | 788b868fac | Balaram Buddharaju | 2026-03-02 | [https://nvbugs/5934461][fix] Propagate logits from prefill to decode in disagg (#11767) |
| 266 | 3b8b91fcdc | Stefan Niebler | 2026-03-02 | [https://nvbugs/5764627][fix] Fix generation logits with streaming and improve runtime of logits testcase. Also fixes https://nvbugs/5573238 (#10637) |
| 267 | 4b4de8110f | Marina Yanovskiy | 2026-03-02 | [#10693][chore] AutoDeploy: Add L1 tests from coverage dashboard (#11530) |
| 268 | f44984528b | sunnyqgg | 2026-03-02 | [https://nvbugs/5883738][fix] fix bug for illegal memory access on Qwen3-235B-A22B-Thinking-2507-NVFP4 + Eagle3 (#11474) |
| 269 | a28def9020 | Stefan Niebler | 2026-03-02 | [TRTLLM-9687][feat] Improve are_stop_words performance (#11196) |
| 270 | 9013b580bd | Kanghwan | 2026-03-02 | [None][fix] Fix FP8 per-tensor torch.compile graph break in dynamic quantization (#11759) |
| 271 | 361132b98a | Leslie Fang | 2026-03-02 | [https://nvbugs/5885070][fix] fix deepeplowlatency with cutedsl moe backend (#11769) |
| 272 | 3ab7770fa1 | nvyocox | 2026-03-02 | [None][feat] Extract embeding as .savetensors and support float8 quantized model (#11180) |
| 273 | 85dc52acae | Tailing Yuan | 2026-03-02 | [https://nvbugs/5823212][fix] Warmup maybe_compiled_cat in forward_context_with_chunked_prefill (#11743) |
| 274 | 50713d8a58 | TensorRT LLM | 2026-03-02 | [None][infra] Check in most recent lock file from nightly pipeline |
| 275 | 4e9aa861ee | JunyiXu-nv | 2026-03-02 | [TRTLLM-10962][feat] Refactor video encoding to use ffmpeg CLI or pur… (#11672) |
| 276 | aa7632e939 | Liwei Ma | 2026-03-02 | [None][chore] pass nsight options to ray_executor and trigger profiling through collective_rpc (#11493) |
| 277 | 17e03fae5f | Iman Tabrizian | 2026-03-01 | [None][test] Add E2E test for cancelled disagg gen request with overlap scheduler (#11795) |
| 278 | e8ad899f93 | Nikita Korobov | 2026-03-01 | [None][feat] TRT-LLM Gen MoE finalize kernel optimization (#11501) |
| 279 | 17eaed5bc3 | Emma Qiao | 2026-03-01 | [None][infra] Waive failed cases for main on 03/01 (#11811) |
| 280 | a20745a0c3 | Gal Hubara-Agam | 2026-03-01 | [None][fix] AutoDeploy: Fix shape handling for singleton prefill (#11679) |
| 281 | ea7a708ea4 | Lucas Liebenwein | 2026-03-01 | [None][chore] Update AGENTS.md (#11809) |
| 282 | a413f2182c | Simo Lin | 2026-03-01 | [None][feat] Add --served-model-name option to serve command (#11711) |
| 283 | 37343d4cdc | Erin | 2026-03-01 | [None][fix] cleanup mem in rollout process (#11658) |
| 284 | 1a349cd7a3 | Kaiyu Xie | 2026-03-01 | [None][doc] Fix typos, grammar, and accuracy across documentation (#11766) |
| 285 | 0df2ec6f9c | HuiGao-NV | 2026-03-01 | [TRTLLM-9782][feat] Support to skip KV cache memory estimation (#11714) |
| 286 | f6acde1638 | Chang Liu | 2026-03-01 | [TRTLLM-11185][test] Add back WAN VBench test in CI (#11804) |
| 287 | 49b9e1bec2 | TensorRT LLM | 2026-03-01 | [None][infra] Check in most recent lock file from nightly pipeline |
| 288 | 841608f35e | Kanghwan | 2026-03-01 | [None][perf] Use F.rms_norm for per-head QK normalization in visual gen (#11798) |
| 289 | 7bd01d284c | Grzegorz Kwasniewski | 2026-02-28 | [TRTLLM-11568][feat] Fix collective calls (#11632) |
| 290 | b8bf27afbe | Kaiyu Xie | 2026-03-01 | [None][fix] Fix typo: avaiable_blocks -> available_blocks in scheduler (#11801) |
| 291 | 1d576c3a77 | Lucas Liebenwein | 2026-02-28 | [None][chore] Add CI trigger and test failure retrieval instructions to AGENTS.md (#11803) |
| 292 | bb5cf9ba62 | Min Yu | 2026-02-28 | [https://nvbugs/5868616][fix] Fix warnings when building moe_kernels.cu (#11703) |
| 293 | 5ddeaf9990 | Kanghwan | 2026-02-27 | [None][perf] Vectorize quantize_fp8_blockwise with CUDA kernel (#11724) |
| 294 | e396442185 | TensorRT LLM | 2026-02-28 | [None][infra] Check in most recent lock file from nightly pipeline |
| 295 | 3fe0908b10 | Balaram Buddharaju | 2026-02-27 | [TRTLLM-11058][feat] Support Helix CP with GQA (#11570) |
| 296 | ab7a20a348 | Zhu Yang | 2026-02-27 | [None][fix] enable separate draft KV cache pool for aggregated + KVBM… (#11689) |
| 297 | b5921b15e3 | Zheyu Fu | 2026-02-27 | [https://nvbugs/5685010][fix] Delete test_eagle3_output_repetition_4gpus flaky assertions. (#11725) |
| 298 | d42911a1a6 | dpitman-nvda | 2026-02-27 | [TRTLLMINF-9][chore] Remove submodule pulls from TRT-LLM git checkouts (#11693) |
| 299 | 2220d4880c | Balaram Buddharaju | 2026-02-27 | [https://nvbugs/5926823][fix] Propagate logprobs from prefill to decode in disagg (#11727) |
| 300 | 63c33c7c9a | dhansen-nvidia | 2026-02-27 | [None][feat] add globaltimer-based timing backend for autotuner profi… (#11657) |
| 301 | 985f81d82e | Jin Li | 2026-02-27 | [https://nvbugs/5911788][test] Waive test_llm_partial_update_weights[Qwen3/Qwen3-8B] (#11785) |
| 302 | c2d766b579 | Ziyi Xiong | 2026-02-27 | [https://nvbugs/5879614][test] Waive test_guided_decoding_with_eagle3 xgrammar in disaggregated serving (#11773) |
| 303 | 7a06614d1c | Jiagan Cheng | 2026-02-27 | [None][feat] Refactor cache manager v2 to simplify new model support (#11749) |
| 304 | cb1a872692 | fredricz-20070104 | 2026-02-27 | [None][test] local wheel installation support and add gb300 cases demo (#11742) |
| 305 | ab99ddf0d8 | xinhe-nv | 2026-02-27 | [None][chore] Remove closed bugs (#11527) |
| 306 | 55077fead9 | Yao Yao | 2026-02-27 | [None][feat] Support heterogeneous tokens_per_block (#11751) |
| 307 | 37ab642c99 | Yiyun Lu | 2026-02-27 | [TRTLLM-10386][fix] torch.compile: register add+norm fallback pass in multi-GPU mode (#11739) |
| 308 | 2237d7d71d | Balaram Buddharaju | 2026-02-26 | [TRTLLM-11064][fix] Remove duplicated MoE Computation with Helix CP+DP (#11167) |
| 309 | 57c2904377 | Emma Qiao | 2026-02-27 | [None][infra] Waive failed cases for main on 02/27 (#11770) |
| 310 | c65aee9bd7 | yuanjingx87 | 2026-02-26 | [None][infra] Move B200 test stage to AIHub (#11692) |
| 311 | 50b48c1f4c | TensorRT LLM | 2026-02-27 | [None][infra] Check in most recent lock file from nightly pipeline |
| 312 | 6f7138ab43 | Qi Zhang (qizh) | 2026-02-27 | [None][chore] Minor fix in w4a8 mxfp4 mxfp8 test. (#11745) |
| 313 | 097ecea344 | NVShreyas | 2026-02-26 | [TRTLLM-11115][feat] enable autotuner for visual gen + Compilation Config (#11660) |
| 314 | 3e1207164d | Yao Yao | 2026-02-27 | [None][fix] Make KVCacheManagerV2 release mem immediately on shutdown (#11746) |
| 315 | a7b17f3658 | Mike Iovine | 2026-02-26 | [None][fix] Use prefer_pinned() in pard.py (#11762) |
| 316 | cde2592bca | Iman Tabrizian | 2026-02-26 | [None][fix] Fix disagg cancellation (#11730) |
| 317 | 80aa8caf44 | Ziyi Xiong | 2026-02-27 | [TRTLLM-10886][feat] Support PARD(Parallel Draft Model) in one-model spec dec (#11438) |
| 318 | 3fd5fafb58 | dhansen-nvidia | 2026-02-26 | [https://nvbugs/5911143][fix] add async worker to MTP/Eagle3 sampler,… (#11573) |
| 319 | 41dd9e0729 | Fadi Saady | 2026-02-26 | [None][test] Add tests for all database configs. (#11653) |
| 320 | e56397dd5a | Wanli Jiang | 2026-02-26 | [None][feat] Support tensor parallelism of trtllm moe backend for nemotron-h model (#11470) |
| 321 | 617440d385 | yufeiwu-nv | 2026-02-26 | [None][test] Remove A100 test cases from QA perf scope (#11712) |
| 322 | 3f4c42d30e | Ziyi Xiong | 2026-02-26 | [https://nvbugs/5821053][fix] Preventing drift accumulation on kv_lens_cuda (#11696) |
| 323 | e3a4f43e33 | zhhuang-nv | 2026-02-26 | [https://nvbugs/5612438][fix] add timeout 14400 for SeedOSS (#11269) |
| 324 | a93c56e9f6 | Balaram Buddharaju | 2026-02-26 | [https://nvbugs/5915550][fix] Fix illegal memory access when max_seq_len > max_position_embeddings (#11598) |
| 325 | eba1b5444e | TensorRT LLM | 2026-02-26 | [None][infra] Check in most recent lock file from nightly pipeline |
| 326 | 8c9d9af367 | Bala Marimuthu | 2026-02-25 | [None][doc] Added Qwen3.5 Cookbook (#11728) |
| 327 | 1754dccf12 | Yanchao Lu | 2026-02-26 | [None][docs] Update PR template (#11735) |
| 328 | b103625ebf | dongfengy | 2026-02-25 | [https://nvbugs/5914691][fix] WAR F.linear perf regression for GPTOSS (#11668) |
| 329 | 7bb371553f | Anthony Chang | 2026-02-26 | [https://nvbugs/5799917][fix] Recover from CUTLASS MoE doActivation perf regression for MXFP4/NVFP4 dtype (#11165) |
| 330 | b195d9e48d | TensorRT LLM | 2026-02-26 | [None][infra] Check in most recent lock file from nightly pipeline |
| 331 | 4ca2a441f5 | Pengbo Wang | 2026-02-26 | [None][feat] Remove non flash attetnion style fmha_v2 kernel for hopper (#11381) |
| 332 | c2d3c6cdba | Leslie Fang | 2026-02-26 | [https://nvbugs/5884735][fix] fix deepeplowlatency with DeepGEMM (#11700) |
| 333 | 5ac888f582 | yuanjingx87 | 2026-02-25 | [None][infra] Update TRTLLM PLC pipeline (#11684) |
| 334 | c39bbb2d1a | Chang Liu | 2026-02-25 | [TRTLLM-11090][perf] Improve fp8 (per-tensor) quant kernel by vectorized load/store (#11662) |
| 335 | 9667ea3fff | Eran Geva | 2026-02-25 | [#11529][perf] AD host time attention MD optimization for large context (#11624) |
| 336 | a73afc08a7 | Mike Iovine | 2026-02-25 | [TRTLLM-11087][doc] Update speculative decoding docs (#11604) |
| 337 | 56001aba9d | Iman Tabrizian | 2026-02-25 | [https://nvbugs/5845901][fix] Fix cancelled disagg requests stuck in gen server (#11695) |
| 338 | 043765be97 | Chien-Chun Hung | 2026-02-25 | [https://nvbugs/5822983][fix] Update waives.txt to remove skipped tests for TestDeepSeekV3Lite in accuracy module (#11591) |
| 339 | 0d8fd3f5c6 | h-guo18 | 2026-02-25 | [None][fix] Quantized Eagle3 support: quantizing self.fc (#11699) |
| 340 | 3d3c41f201 | Wanli Jiang | 2026-02-26 | [https://nvbugs/5866619][fix] Support PEFT-saved safetensors file loading (#11339) |
| 341 | 9e64f2fac2 | Emma Qiao | 2026-02-25 | [None][infra] Waive failed cases for main on 02/25 (#11719) |
| 342 | c3f1e07511 | JadoTu | 2026-02-25 | [https://nvbugs/5734983][doc] update Qwen3-Next readme of server arg (#11682) |
| 343 | 3fce67f345 | Ahmet Inci | 2026-02-25 | [TRTLLM-10948][feat] Add GPU energy monitoring to trtllm-bench (#11397) |
| 344 | 8b0892c076 | peihengh | 2026-02-25 | [https://nvbugs/5875514][fix] Fix WideEP gen-only benchmark hang in disaggregated serving (#11521) |
| 345 | 0338bb2bcc | jthomson04 | 2026-02-25 | [None][chore] KV Connector Refactor (#11078) |
| 346 | 0848a4c84b | Chuang Zhu | 2026-02-25 | [TRTLLM-9527][feat] E2E Python KV transceiver for current KV manager (step 5) (#11136) |
| 347 | 69498aa6cd | Liao Lanyu | 2026-02-25 | [TRTLLM-11106][chore] Abstract ADPRouter interface and RankState (#11633) |
| 348 | 7c05ad0942 | fredricz-20070104 | 2026-02-25 | [None][chore] Add feature for enhance perf dashboard (#11506) |
| 349 | 16e2c71456 | Yiteng Niu | 2026-02-25 | [TRTLLM-8828][infra] export HF_TOKEN in tests (#9382) |
| 350 | f548b53bb8 | Kaiyu Xie | 2026-02-25 | [None][docs] Fix 60+ broken links across docs, blogs, and examples (#11676) |
| 351 | 2ec5204e0d | Grzegorz Kwasniewski | 2026-02-25 | [TRTLLM-11614][feat] Fixing multigpu tests (#11615) |
| 352 | dc63612484 | ruodil | 2026-02-25 | [None][test] support short test case matcher in disagg test (#11707) |
| 353 | bab71927d7 | Jin Li | 2026-02-25 | [TRTLLM-9904][feat] KVCache V2 MTP support (#11346) |
| 354 | d5c513d14b | ruodil | 2026-02-25 | [None][test] add concurrency override and fix for 128k8k cases (#11669) |
| 355 | c6308ba2d4 | Wojciech Wais | 2026-02-25 | [#4666][fix] Handle None priority in KVCacheEventSerializer._event_diff_to_json (#11576) |
| 356 | 4789dfdab3 | Yao Yao | 2026-02-25 | [TRTLLM-7836][feat] Implement dynamic quota resize for KVCacheManager v2 (#11503) |
| 357 | 270b815bed | Bo Li | 2026-02-25 | [None][fix] Fix FP8 + Skip Softmax Attention accuracy issue on fmha_v2. (#11448) |
| 358 | 8d19b68b91 | TensorRT LLM | 2026-02-25 | [None][infra] Check in most recent lock file from nightly pipeline |
| 359 | ab8d7cbd50 | yuanjingx87 | 2026-02-24 | [None][chore] Bump version to 1.3.0rc6 (#11688) |
| 360 | c55b190750 | xd-nv | 2026-02-24 | [None] [fix] Restructure kv cache memory ratio parameters in curated .yaml config files (#11511) |
| 361 | 455c29d954 | Kanghwan | 2026-02-24 | [None][fix] rename svd-nvfp4 to trtllm-nvfp4 in visual gen examples (#11664) |
| 362 | b1dd89baad | tburt-nv | 2026-02-24 | [TRTINFRA-7367][infra] Automatically generate attributions file (#11323) |
| 363 | 730797461e | Fadi Saady | 2026-02-24 | [None][chore] Moving kimi-k2-thinking deployment guide configs to config files. (#11645) |
| 364 | e64d2c7d69 | Stanley Sun | 2026-02-24 | [None][test] Add wideep DS-R1 nvfp4 test with attn_dp and kv_cache_reuse (#11670) |
| 365 | 53562ee488 | chenfeiz0326 | 2026-02-24 | [None][fix] Add comparison operators for perf regression triage (#11675) |
| 366 | c96a9dd052 | xxi | 2026-02-24 | [TRTLLM-9108][feat] refactor MoE unit tests: add unified ConfigurableMoE test framework (#11648) |
| 367 | 15682c494f | JadoTu | 2026-02-24 | [https://nvbugs/5606178][fix] unwaive mamba2 two tests (#11479) |
| 368 | b5953b9e88 | tcherckez-nvidia | 2026-02-24 | [None][fix] Accept **kwargs in DynamicYamlWithDeepMergeSettingsSource… (#11621) |
| 369 | debbe3ecb8 | TensorRT LLM | 2026-02-24 | [None][infra] Check in most recent lock file from nightly pipeline |
| 370 | 92342ba4bc | Yiqing Yan | 2026-02-24 | [TRTLLM-9781][infra] Don't create timeout xml if the stage is aborted (#9777) |
| 371 | 951a467c7a | dominicshanshan | 2026-02-23 | [None][chore] Fix gpu memory requirement in stress test (#11404) |
| 372 | 457025d0ca | JunyiXu-nv | 2026-02-23 | [https://nvbugs/5823783][fix] Fix multi-node trust_remote_code hang i… (#11383) |
| 373 | e3c607db21 | Patrice Castonguay | 2026-02-19 | [https://nvbugs/839137][fix] Unwaive disagg unexpected ucx error (#11543) |
| 374 | c20bbedfe6 | dongfengy | 2026-02-16 | [https://nvbugs/5875296][fix] Fix TritonMOE test for Qwen3_30B_A3B (#11495) |
| 375 | 4e505522b1 | dongfengy | 2026-02-13 | [https://nvbugs/5833795][fix] Remove test waive and try CI (#11464) |
| 376 | c24393cc61 | Lizhi Zhou | 2026-02-13 | [https://nvbugs/5889564][fix] fix kwargs name (#11496) |
| 377 | 6923357e0a | tburt-nv | 2026-02-23 | [None][fix] Fix test prefix generation for per-sm waives (#11519) |
| 378 | f39e1a8603 | Min Yu | 2026-02-24 | [https://nvbugs/5846489][perf] Apply TE's FP8 per-tensor quantization (#11057) |
| 379 | de5621dd69 | TensorRT LLM | 2026-02-24 | [None][infra] Check in most recent lock file from nightly pipeline |
| 380 | fc0fa5d424 | Emma Qiao | 2026-02-24 | [None][infra] Waive failed cases for main on 02/24 (#11665) |
| 381 | c7d8cc1f34 | Chang Liu | 2026-02-23 | [None][perf] Use UE8M0 FP8 quant kernel for DeepGemm blockwise GEMM (#11607) |
| 382 | e987a1cee8 | Anish Shanbhag | 2026-02-23 | [None][chore] Align LlmArgs with some Pydantic best practices (#11158) |
| 383 | a5768ce316 | Stefan Niebler | 2026-02-11 | [https://nvbugs/5820922][perf] Improve TorchSampler performance by reducing host overhead (#11315) |
| 384 | bf1d64ec54 | Emma Qiao | 2026-02-10 | [None][infra] Enable spark stage for release since the spark cloud migration is done (#11408) |
| 385 | 87ccb27f18 | Emma Qiao | 2026-02-10 | [None][infra] Disable release spark stage due to migration of spark cloud (#11402) |
| 386 | f3a692e058 | Yechan Kim | 2026-02-10 | [https://nvbugs/5814504][fix] Add skip_pre_hopper flag on NVILA & Nano V2 VLMs (#11275) |
| 387 | 37b59e8ff3 | Pengbo Wang | 2026-02-10 | [https://nvbugs/5624818][fix] Fix GPT-OSS with non-paged_context_fmha (#11309) |
| 388 | 566f39821b | Ziyi Xiong | 2026-02-06 | [None][chore] Resolve a conflict in the md file (#11255) |
| 389 | 8e27a87e19 | Lucas Liebenwein | 2026-02-05 | [https://nvbugs/5688721][fix] AutoDeploy: unwaive fixed NemotronH accuracy test (#11290) |
| 390 | ba2a442b28 | Balaram Buddharaju | 2026-02-05 | [https://nvbugs/5863443][fix] Fix message truncation in Helix CP cache transmission (#11252) |
| 391 | a0e8ef7473 | dongfengy | 2026-02-05 | [https://nvbugs/5830877][fix] Use the best (correct) config for GPTOSS perf test (#11046) |
| 392 | 75fd2420e0 | Yechan Kim | 2026-02-05 | [https://nvbugs/5845769][fix] B300(sm103) support on VLMs (#11274) |
| 393 | 208bf7b3ca | TensorRT LLM | 2026-02-23 | [None][infra] Check in most recent lock file from nightly pipeline |
| 394 | d8d11695c0 | Eran Geva | 2026-02-23 | [#10243][chore] switched the default AD attention backend to trtllm (#11627) |
| 395 | 6bed44ec2b | Kanghwan | 2026-02-23 | [TRTLLM-10616][feat] Add FLUX.1 and FLUX.2 text-to-image pipeline support (#11556) |
| 396 | ce088f903d | Chang Liu | 2026-02-23 | [https://nvbugs/5919025][fix] Disable warmup steps for some WAN unit tests (#11616) |
| 397 | c53b8fc2f1 | NVShreyas | 2026-02-23 | [None][fix] Nemotron H fp4 and MTP (#11601) |
| 398 | f636e121fc | Grzegorz Kwasniewski | 2026-02-23 | [TRTLLM-11567][feat] Added GatedDeltaNet sharding from config (#11599) |
| 399 | c4beca8f65 | Emma Qiao | 2026-02-23 | [None][infra] Waive failed cases for main for post-merge 2550 (#11650) |
| 400 | 062c800c2e | Taylor Yeonbok Lee | 2026-02-23 | [#11398][feat] AutoDeploy: flashinfer rope for GLM4.7-Flash (#11524) |
| 401 | c2394bf81e | Yao Yao | 2026-02-23 | [https://nvbugs/5921273][fix] Fix an issue where sync is missing before cuMemUnmap (#11641) |
| 402 | 505d0f939f | Jie Li | 2026-02-23 | [#9907][infra] Add Python builds tests to CI pre-merge pipeline (#9943) |
| 403 | 6943087bc1 | Robin Kobus | 2026-02-23 | [None][fix] numpy v2 preparations (#11389) |
| 404 | df3484ddfa | Eran Geva | 2026-02-23 | [#11529][perf] AD NemotronH topk router to use the model default dtype (#11623) |
| 405 | 8f542b9c4f | Emma Qiao | 2026-02-23 | [None][infra] Waive failed cases for main branch on 2/23 (#11635) |
| 406 | a9faf894f7 | Yiyun Lu | 2026-02-23 | [TRTLLM-10514][feat] Refactor time breakdown tool (visualization, generation breakdown, etc.) (#11340) |
| 407 | 52110e8ca7 | Eran Geva | 2026-02-23 | [#11529][perf] Replace Python-traced FP8 quantization with optimized CUDA op in AD MoE (#11626) |
| 408 | a37d9d0466 | TensorRT LLM | 2026-02-23 | [None][infra] Check in most recent lock file from nightly pipeline |
| 409 | 538a9dfb2e | TensorRT LLM | 2026-02-23 | [None][infra] Check in most recent lock file from nightly pipeline |
| 410 | 3a01a96e5b | tcherckez-nvidia | 2026-02-23 | [None][chore] Enable Nemotron Super nvfp4 tests (#11172) |
| 411 | 4f49f6ef64 | TensorRT LLM | 2026-02-23 | [None][infra] Check in most recent lock file from nightly pipeline |
| 412 | 630fccb3ca | TensorRT LLM | 2026-02-22 | [None][infra] Check in most recent lock file from nightly pipeline |
| 413 | 56d54fa009 | Patrice Castonguay | 2026-02-22 | [None][chore] Multi gpu sbsa waives (#11629) |
| 414 | d8622c328a | Patrice Castonguay | 2026-02-22 | [None][chore] Waiving more tests (#11613) |
| 415 | f86656ffc6 | TensorRT LLM | 2026-02-22 | [None][infra] Check in most recent lock file from nightly pipeline |
| 416 | e8a4f1f1ae | Chang Liu | 2026-02-21 | [None][chore] Waive failing WAN tests due to missing warmup_steps (#11617) |
| 417 | 4d3f947153 | TensorRT LLM | 2026-02-21 | [None][infra] Check in most recent lock file from nightly pipeline |
| 418 | cf2254fbd9 | yuanjingx87 | 2026-02-20 | [None][infra] PLC pipeline update (#11597) |
| 419 | f4b316a9b9 | TensorRT LLM | 2026-02-21 | [None][infra] Check in most recent lock file from nightly pipeline |
| 420 | 5ae5adedfc | yuanjingx87 | 2026-02-20 | [None][infra] Waive unittest that timed out (#11605) |
| 421 | f3705426c6 | Venky | 2026-02-20 | [None][infra] add visual_gen codeowners paths (#11606) |
| 422 | a7f33ec035 | TensorRT LLM | 2026-02-20 | [None][infra] Check in most recent lock file from nightly pipeline |
| 423 | b090473cb7 | Yuewei Na | 2026-02-20 | [https://nvbugs/5896216][fix] Prevent NIXL agent name collision in containerized disaggregated serving (#11552) |
| 424 | e4277e44d8 | yuanjingx87 | 2026-02-20 | [None][fix] fix testdb file for l0_b200_multi_gpus_perf_sanity (#11603) |
| 425 | acc43e19cb | Patrice Castonguay | 2026-02-20 | [None][chore] Waive failing post merge (#11600) |
| 426 | 42ffb47e5b | Izzy Putterman | 2026-02-20 | [None][fix] SpecDec: Sampling seed fix (#11081) |
| 427 | d33c1e3b09 | Izzy Putterman | 2026-02-20 | [None][fix] Nemotron Super fix (#11425) |
| 428 | 0c239d0cbe | Chang Liu | 2026-02-20 | [None][feat] Add NVFP4 dynamic quantization support for visual_gen models (#11563) |
| 429 | 1950be317e | mpikulski | 2026-02-20 | [None][fix] correct chunked prefill handling in TorchSampler (#11544) |
| 430 | 1b922dbbd1 | mpikulski | 2026-02-20 | [TRTLLM-11069][fix] validate requests outside sampling loop (#11584) |
| 431 | bcd6f5c3d4 | NVShreyas | 2026-02-20 | [TRTLLM-10197][feat] Cache Transfer Setup for Mamba States (#10934) |
| 432 | fa2bfa5dd4 | Mike Iovine | 2026-02-20 | [TRTLLM-10857][chore] Move SaveHiddenStates spec dec mode to 1 model (#11241) |
| 433 | d0b2f03484 | Eran Geva | 2026-02-20 | [#10243][feat] Add TRT-LLM attention backend to AutoDeploy (#11430) |
| 434 | 6b70df2483 | Balaram Buddharaju | 2026-02-20 | [https://nvbugs/5914959][fix] Fix illegal memory access with Helix CP=64 (#11593) |
| 435 | 3dd67c19b0 | tomeras91 | 2026-02-20 | [None][fix] Read mamba_ssm_cache_dtype from HF config when set to auto (#11582) |
| 436 | 4759f74214 | mpikulski | 2026-02-20 | [None][chore] split up TorchSampler.Store (#11566) |
| 437 | c2f840da37 | William Zhang | 2026-02-19 | [#11569][fix] Fix broken LLMAPI config (#11571) |
| 438 | b5029d3d50 | Chenghao Zhang | 2026-02-19 | [None][feat] AutoDeploy: Add nemotron v2 acc test (#11429) |
| 439 | 69ff36a741 | Bala Marimuthu | 2026-02-19 | [None][doc] Add Qwen3.5, GLM 4.7 Flash to support matrix (#11594) |
| 440 | c172acf5bf | Chang Su | 2026-02-19 | [#11292][feat] use smg-grpc-proto package for gRPC proto definitions (#11578) |
| 441 | d45f03e481 | TensorRT LLM | 2026-02-20 | [None][infra] Check in most recent lock file from nightly pipeline |
| 442 | a22835b918 | Bala Marimuthu | 2026-02-19 | [#11440] [feat] AutoDeploy : Support Qwen3.5 (#11394) |
| 443 | 775c2736d9 | William Zhang | 2026-02-19 | [TRTLLM-9040][perf] Make preprocessing async (#11459) |
| 444 | 4bee075e52 | NVShreyas | 2026-02-19 | [None][feat] Visual Gen: add cuda graphs; torch compile; nvtx; warmup (#11554) |
| 445 | f7511f11d7 | TensorRT LLM | 2026-02-19 | [None][infra] Check in most recent lock file from nightly pipeline |
| 446 | 353fd33fba | Simeng Liu | 2026-02-19 | [TRTLLM-1543][feat] Account for reusable KV cache blocks in capacity … (#11490) |
| 447 | 19672e418c | yuanjingx87 | 2026-02-19 | [None][infra] Waive unittest that consistently timed out (#11580) |
| 448 | fa6c0f241a | Iman Tabrizian | 2026-02-18 | [https://nvbugs/5880313][fix] Fix pp + disagg (#11509) |
| 449 | b41bf57172 | TensorRT LLM | 2026-02-19 | [None][infra] Check in most recent lock file from nightly pipeline |
| 450 | c87c80043e | yijingl-nvidia | 2026-02-18 | [TRTLLM-10827][feat] Add KV Cache metrics to MetricsCollector for more Prometheus metrics (#11243) |
| 451 | c2f5f43ebf | yuanjingx87 | 2026-02-18 | [None][infra] PLC pipeline update (#11547) |
| 452 | 66861a0364 | jthomson04 | 2026-02-18 | [None][fix] Fix silent MPI failures on models with custom tokenizers (#11399) |
| 453 | 0ff3a6db38 | Aurelien Chartier | 2026-02-18 | [None][feat] Add support for multi instances in Triton backend with pytorch backend (#11153) |
| 454 | 53dced1907 | dpitman-nvda | 2026-02-18 | [TRTLLM-10037][chore] Re-upgrade GHA for blossom-ci workflow (#11483) |
| 455 | 80dbb64f1a | chenfeiz0326 | 2026-02-18 | [TRTLLM-8263][feat] Add ctx-only and gen-only Disagg Perf Tests (#11361) |
| 456 | 682def9bfd | Grzegorz Kwasniewski | 2026-02-18 | [TRTLLM-10064][feat] MoE all-to-all paradigm (#10985) |
| 457 | e49c0e66fd | Yibin Li | 2026-02-18 | [None][chore] TAVA architecture diagram updates for visual gen flow and auto deploy flow (#11523) |
| 458 | 324ac4f104 | tburt-nv | 2026-02-18 | [https://nvbugs/5888464][fix] Stop using remotes in the Conan install build step (#11516) |
| 459 | 776132f72a | Venky | 2026-02-17 | [TRTLLM-10845][feat] Add dynamic llmapi defaults system (#11035) |
| 460 | d23b0ca14b | TensorRT LLM | 2026-02-18 | [None][infra] Check in most recent lock file from nightly pipeline |
| 461 | fce8f1811a | yuanjingx87 | 2026-02-17 | [None][infra] bump version (#11557) |
| 462 | c64bc14719 | Balaram Buddharaju | 2026-02-17 | [None][chore] Waive moe fp4 test (#11558) |
| 463 | 957f803dd2 | Balaram Buddharaju | 2026-02-17 | [None][chore] Waive failing pre-merge test (#11551) |
| 464 | 6157f30b06 | Bala Marimuthu | 2026-02-17 | [#11318][infra] AutoDeploy: Add fused rope kernel - triton_rope_on_interleaved_qk_inputs (#11327) |
| 465 | ab941afa2e | Bo Li | 2026-02-17 | [None][doc] Update media files path in Skip Softmax blog. (#11540) |
| 466 | 1c065fbb3e | Bala Marimuthu | 2026-02-17 | [#11109][feat] AutoDeploy: GLM 4.7 Flash Improvements (#11414) |
| 467 | fedd7178d1 | TensorRT LLM | 2026-02-17 | [None][infra] Check in most recent lock file from nightly pipeline |
| 468 | 2450188808 | jthomson04 | 2026-02-16 | [None][fix] Better error message for mismatched MPI world size (#11294) |
| 469 | cc4511997a | Yanchao Lu | 2026-02-16 | [None][revert] - Revert "[TRTLLM-9108][feat] refactor MoE unit tests: add unified ConfigurableMoE test framework" (#11532) |
| 470 | 08c7103fc4 | mpikulski | 2026-02-16 | [TRTLLM-10030][test] ensure that TorchSampler does not sync (#11508) |
| 471 | d72f8098fe | TensorRT LLM | 2026-02-16 | [None][infra] Check in most recent lock file from nightly pipeline |
| 472 | f3d784c6f6 | Suyog Gupta | 2026-02-15 | [#10345][perf] Enable multi-stream MOE for super. Also adds multi-stream MLA attn (#11520) |
| 473 | fcb7bea07f | tcherckez-nvidia | 2026-02-15 | [#11455][bug] Use the torch_dtype set by ModelOpt (#11525) |
| 474 | 361ff36784 | Yi Zhang | 2026-02-15 | [None][feat] Use new index api, add block scale support, fix max_seq_len esitmation, add flash mla support (#11334) |
| 475 | 59b6bee7e6 | yingguo-trt | 2026-02-05 | [None][chore] Fix slurm job name (#11265) |
| 476 | 17e6062690 | Ivy Zhang | 2026-02-05 | [https://nvbugs/5821433][fix] complete WAR for popen in QA env (#11214) |
| 477 | 2b4ef3a014 | Pengbo Wang | 2026-02-04 | [https://nvbugs/5815025][fix] Fix spec-dec mode flag and related cpp requirements (#10996) |
| 478 | ebd859cf61 | Yechan Kim | 2026-02-03 | [https://nvbugs/5854419][fix] Fix Qwen3-VL-Dense/MoE accuracy drop (#11134) |
| 479 | 435ea36977 | Mike Iovine | 2026-02-02 | [None][chore] Add warning about 2-model MTP deprecation (#11043) |
| 480 | 5e47e6970b | Emma Qiao | 2026-02-02 | [None][infra] Waive failed cases for release branch on 02/02 (#11182) |
| 481 | 592988ebdb | Pengyun Lin | 2026-02-02 | [https://nvbugs/5819444][fix] Unwaive gpt-oss test (#10927) |
| 482 | 80708ba231 | xinhe-nv | 2026-02-02 | [https://nvbugs/5787904][fix] update mig tests (#11014) |
| 483 | d8e7c61ea9 | dominicshanshan | 2026-02-02 | [https://nvbugs/5823465][fix] Add CUTEDSL moe backend for deepseek r1 nvfp4 checkpoint in stress test (#10920) |
| 484 | 80235e53cf | dhansen-nvidia | 2026-01-30 | [None][feat] Add documentation on configuring CPU affinity in TRT-LLM (#10678) |
| 485 | 5d73194ffb | Ziyi Xiong | 2026-01-30 | [https://nvbugs/5829830][fix] Declare the var in the correct scope (#11066) |
| 486 | d9f787a8d2 | Patrice Castonguay | 2026-01-29 | [None][doc] Hardware support update (#10719) |
| 487 | ed404f9298 | Yukun He | 2026-02-15 | [TRTLLM-10851][feat] Add line_profiler tool for host overhead analysis. (#11232) |
| 488 | b003355050 | Chang Liu | 2026-02-14 | [None][doc] Add doc for TRTLLM AIGV initial release (#11489) |
| 489 | 144188c2c4 | TensorRT LLM | 2026-02-15 | [None][infra] Check in most recent lock file from nightly pipeline |
| 490 | 0a9ddf8c17 | Chuang Zhu | 2026-02-15 | [https://nvbugs/5880261][fix] fix cacheTransceiver (#11409) |
| 491 | 29e44dd749 | Thor Johnsen | 2026-02-13 | [None][fix] Add cacheSaltID property to BlockKey serialization code (#11457) |
| 492 | 2989bf5b39 | Balaram Buddharaju | 2026-02-13 | [None][feat] Add new helix kernels for MNNVL-based codepath (#11433) |
| 493 | 4debf153d8 | William Zhang | 2026-02-13 | [#11170][fix] Fix for mm placeholder counts (#11461) |
| 494 | b4e9669d2c | Suyog Gupta | 2026-02-13 | [None][chore] Optimize MOE export by tracing with reduced experts and expanding graph (#11504) |
| 495 | f164669c04 | tburt-nv | 2026-02-13 | [None][chore] Adjust waive to avoid sm parsing (#11518) |
| 496 | 26901e4aa0 | Chang Liu | 2026-02-13 | [TRTLLM-10612][feat] Initial support of AIGV models in TRTLLM (#11462) |
| 497 | 19a3031ecb | Pamela Peng | 2026-02-13 | [TRTLLM-10329][feat] Fix weight loading for Nemotron 3 models on DGX Spark (#11405) |
| 498 | 052fe2f7f6 | dpitman-nvda | 2026-02-13 | [None][chore] Update allowlist 2026-02-13 (#11512) |
| 499 | 37c53425c1 | mpikulski | 2026-02-13 | [TRTLLM-10030][chore] improve assert in sampler (#11475) |
| 500 | b67dcd8fef | Venky | 2026-02-13 | [None][docs] enable Deepwiki docs (#11492) |
