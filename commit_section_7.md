# Commit Section 7

Commits 3001 to 3500 (total: 5612)

| # | Commit Hash | Author | Date | Message |
|---|------------|--------|------|--------|
| 3001 | f156221c27 | Izzy Putterman | 2025-09-03 | [None][doc] add GPT OSS Eagle3 blog (#7140) |
| 3002 | 7c73c2ff4b | Lizhi Zhou | 2025-09-03 | [https://nvbugs/5485593][fix] improve accuracy/test_disaggregated_serving.py (#7366) |
| 3003 | cebbf48b74 | Stanley Sun | 2025-09-03 | [TRTLLM-7363][test] Add 8-GPU test cases for RTX6000 (#7083) |
| 3004 | ae5136831f | Anurag Mukkara | 2025-09-03 | [https://nvbugs/5472947][fix] wait on isend handles before reusing buffers (#7462) |
| 3005 | 79d93f9419 | Mike Iovine | 2025-09-03 | [https://nvbugs/5488141][fix] Unwaive llama3 test_eagle3 (#7486) |
| 3006 | 9a4f60687f | YueWeng | 2025-09-03 | [https://nvbugs/5480289][fix] release slot manager in mtp MTPHiddenStatesManager (#7340) |
| 3007 | 4223a9aada | Wanli Jiang | 2025-09-03 | [TRTLLM-7261][feat] Support phi-4 model in pytorch backend (#7371) |
| 3008 | 572551b586 | Jinyang Yuan | 2025-09-03 | [None][perf] Autotune TRT-LLM Gen MoE when using CUDA graphs (#7285) |
| 3009 | 109f27265c | Daniel Stokes | 2025-09-03 | [None][perf] Add MOE support for dynamic cluster shapes and custom epilogue schedules (#6126) |
| 3010 | 42697ea32a | Leslie Fang | 2025-09-03 | [None][chore] rm executor config in kv cache connector (#7372) |
| 3011 | b4340ecb62 | Martin Marciniszyn Mehringer | 2025-09-02 | [None][chore] Add note about trtllm-serve to the devel container (#7483) |
| 3012 | 75c1bb6389 | Eran Geva | 2025-09-02 | [https://nvbugs/5458798][fix] Disabled test_trtllm_bench_backend_comparison due to timeout (#7397) |
| 3013 | bcc55bcdf3 | Simeng Liu | 2025-09-02 | [https://nvbugs/5470782][fix] Add specific test names for test_deepseek.py (#7318) |
| 3014 | f58a183c6e | Kanghwan | 2025-09-02 | [None][chore] Fix formatting error in Gemma3 readme (#7352) |
| 3015 | aae5d22bfe | Emma Qiao | 2025-09-02 | [None][infra] Waive failed tests on main branch 0902 (#7482) |
| 3016 | 90479c50fb | peaceh-nv | 2025-09-02 | [https://nvbugs/5453992][unwaive] Unwaive llama quickstart test (#7242) |
| 3017 | eefe5f2093 | JunyiXu-nv | 2025-09-02 | [TRTLLM-7208][feat] Implement basic functionalities for Responses API (#7341) |
| 3018 | 7279297717 | HuiGao-NV | 2025-09-02 | [None][infra] waive test case failed on post-merge (#7471) |
| 3019 | c3c95736a1 | aalanwyr | 2025-09-02 | [TRTLLM-6643][feat] Add DeepSeek-v3-0324 e2e torch test (#7413) |
| 3020 | 9c8d2161d0 | tomeras91 | 2025-09-02 | [None][doc] fix example in docstring (#7410) |
| 3021 | 3799e5d460 | Ivy Zhang | 2025-09-02 | [None][test] auto reuse torch empty cache on qa test (#7421) |
| 3022 | f90375f37c | Yan Chunwei | 2025-09-02 | [https://nvbugs/5476580][fix] unwaive test_nvfp4_4gpus (#7454) |
| 3023 | a07bb163f7 | Yanchao Lu | 2025-09-02 | [None][ci] Correct docker args for GPU devices and remove some stale CI codes (#7417) |
| 3024 | ff2439ff48 | Yiqing Yan | 2025-09-02 | [None][infra] Using local variables in rerun function (#7198) |
| 3025 | 60df6b2826 | Jiagan Cheng | 2025-09-02 | [https://nvbugs/5485430][fix] Copy the nanobind file when using precompiled package (#7334) |
| 3026 | e81c50dbd2 | Leslie Fang | 2025-09-02 | [None][chore] Use llm args in create_py_executor (#7239) |
| 3027 | 1b9c4cc2f7 | Tian Zheng | 2025-09-02 | [None][fix] Fix nanobind failure (#7425) |
| 3028 | 9f2dc3069d | jiahanc | 2025-09-01 | [None] [doc] Update DeepSeek example doc (#7358) |
| 3029 | b3c57a7042 | Mike Iovine | 2025-09-01 | [TRTLLM-7353][feat] Implement capturable drafting loops for speculation (#7100) |
| 3030 | 01dfd3af1b | Emma Qiao | 2025-09-01 | [None][infra] Waive failed case on main 0901 (#7447) |
| 3031 | 16e9d1121c | bhsueh_NV | 2025-09-01 | [https://nvbugs/5481087][fix] fix bug of ci when we use mocker (#7332) |
| 3032 | 2b286ae613 | yuanjingx87 | 2025-08-31 | [None][infra] Disable GB200-PyTorch-1 due to OOM issue (#7386) |
| 3033 | efaefca2c8 | nvamyt | 2025-09-01 | [None][test] Update case that not support passing quantization fp8 for pytorch backend (#7302) |
| 3034 | b0558c73fc | Dimitrios Bariamis | 2025-08-20 | [None][fix] Fix build of tritonbuild/tritonrelease image (#7003) |
| 3035 | 44cc308e6a | Dimitrios Bariamis | 2025-08-22 | [https://nvbugs/5474037][fix] Fix building tritonbuild/tritonrelease images (#7157) |
| 3036 | ed4087a295 | QI JUN | 2025-08-19 | [https://nvbugs/5374016][fix] improve error message (#6893) |
| 3037 | 93e623b455 | Aurelien Chartier | 2025-08-18 | [https://nvbugs/5449155][fix] Fix DeepSeek R1 weight loading for TP16 (#6913) |
| 3038 | 21291f3d8e | Yiqing Yan | 2025-08-18 | [None][chore] Remove duplicate test waives (#6999) |
| 3039 | 09bca7ca82 | Emma Qiao | 2025-08-18 | [None][infra] Waive failed tests for release branch 0818 (#6993) |
| 3040 | f4dc1ed39c | peaceh-nv | 2025-08-18 | [https://nvbugs/5449218][fix] Fix KvCacheConfig error in test_perf (#6937) |
| 3041 | 29cdcdb56a | Ivy Zhang | 2025-08-18 | [None][fix] update skip config (#6891) |
| 3042 | d5bc5cd4f2 | Guoming Zhang | 2025-08-18 | [https://nvbugs/5375646][fix] update waives.txt for nvbug 5375646 (#6847) |
| 3043 | d15dcdc4ae | William Zhang | 2025-08-17 | [https://nvbugs/5448525][fix] Mistral Small 3.1 accuracy tests (#6909) |
| 3044 | 704fca4178 | Liao Lanyu | 2025-08-18 | [TRTLLM-6835][fix] Fix potential hang caused by python multiprocessing when prefetching weights (#6927) |
| 3045 | 261ffacfa4 | Yilin Fan | 2025-08-17 | [https://nvbugs/5412562][feat] Allocate MoE workspace only when necessary (release/1.0 retargeted) (#6955) |
| 3046 | 093a03796f | Venky | 2025-08-15 | [None][infra] update CODEOWNERS for release (#6905) |
| 3047 | de55763f13 | Mike Iovine | 2025-08-15 | [https://nvbugs/5455836][fix] Fix llama 4 FP4 (#6911) |
| 3048 | ac07418968 | Yan Chunwei | 2025-08-15 | [None][ci] unwaive test_ptp_star_attention_example (#6943) |
| 3049 | 665a1a7c36 | Iman Tabrizian | 2025-08-14 | [https://nvbugs/5451434][fix] Fix triton docker build (#6898) |
| 3050 | b4d41d6604 | xinhe-nv | 2025-08-15 | [TRTLLM-7048][feat] add benchmark TRT flow test for MIG (#6884) |
| 3051 | 612c26be22 | Yan Chunwei | 2025-08-15 | [None][doc] add legacy section for tensorrt engine (#6724) |
| 3052 | 0253036a4e | brb-nv | 2025-08-14 | [None][chore] Add docs for Gemma3 VLMs (#6880) |
| 3053 | e106045fda | Yukun He | 2025-08-15 | [None][fix] Complete the last missing allreduce op in Llama3/4. (#6850) |
| 3054 | b821883b25 | Anurag Mukkara | 2025-08-14 | [None][fix] Revert phi4-mm aggregate mode (#6907) |
| 3055 | cf0c47ca2d | 2ez4bz | 2025-08-13 | [None][fix] Fix batching bug in Mistral3 model (#6841) |
| 3056 | 3aeee19f9c | Yiqing Yan | 2025-08-14 | [None][infra] Setup the code review rule on the release branch (#6725) |
| 3057 | 2480aedb73 | 2ez4bz | 2025-08-13 | [TRTLLM-5252][feat] Add fp8 support for Mistral Small 3.1 (#6731) |
| 3058 | 3e99744201 | Guoming Zhang | 2025-08-13 | [https://nvbugs/5375594][fix] fix oom issue on structural_tag test case (#6838) |
| 3059 | deba2885c1 | Ivy Zhang | 2025-08-13 | [None][fix] fix Llama3 eagle3 test case OOM (#6832) |
| 3060 | 7841ea6255 | xinhe-nv | 2025-08-13 | [None][chore] waive GB300 known issues (#6812) |
| 3061 | c7147d25dc | Ivy Zhang | 2025-08-13 | [TRTLLM-6975][test] Add multi-turn test cases for VLM models (#6749) |
| 3062 | c5148f52d5 | Yanchao Lu | 2025-09-01 | [None][ci] Some improvements for Slurm CI setup (#7407) |
| 3063 | e257cb3533 | Tian Zheng | 2025-09-01 | [None][feat] Support NVFP4 KV Cache (#6244) |
| 3064 | a7ed26dd8b | Zongfei Jing | 2025-09-01 | [TRTLLM-6747][feat] Merge add sparse exp and shared exp into local reduction (#7369) |
| 3065 | ec595a8e29 | Yiqing Yan | 2025-08-31 | [None][chore] Bump version to 1.1.0rc2 (#7394) |
| 3066 | 5f939b9121 | xinhe-nv | 2025-08-30 | [None][chore] Add failed cases into waives.txt (#7342) |
| 3067 | e09c025ffb | Robin Kobus | 2025-08-30 | [None] [fix] store blog 10 media via lfs (#7375) |
| 3068 | 9bb0c9500e | Zhongdongming Dai | 2025-08-29 | [None][docs] Update Dynasor paper info (#7137) |
| 3069 | 43cb50f788 | brb-nv | 2025-08-29 | [None][feat] Update TargetInfo to accommodate CP in disagg (#7224) |
| 3070 | 642ff13710 | juney-nvidia | 2025-08-30 | [None][doc] Exposing the ADP balance strategy tech blog (#7380) |
| 3071 | 15ec2b855d | Emma Qiao | 2025-08-29 | [None][infra] Waive failed tests on main branch 08/29 (#7370) |
| 3072 | 62459d533d | Pengbo Wang @ NVIDIA | 2025-08-29 | [None][chore] Update pre-merge test to add DeepSeek/LLaMA and gpt-oss (#7192) |
| 3073 | 37a1bd810f | Fanrong Li | 2025-08-29 | [https://nvbugs/5481385][fix] Fix max_seq_len in cuda graph warmup and intermediate_size in fused_moe_deepgemm (#7345) |
| 3074 | f617b03bfc | yunruis | 2025-08-29 | [None][fix] fix doc formula (#7367) |
| 3075 | 091b67ad2f | fredricz-20070104 | 2025-08-29 | [TRTLLM-7280][test] Add beam search CudaGraph + Overlap Scheduler tests (#7326) |
| 3076 | 31b0f0fb0c | Chang Liu | 2025-08-28 | [https://nvbugs/5445466][fix] Eliminate race when loading HF dynamic modules (#7268) |
| 3077 | 2e437536b7 | Venky | 2025-08-28 | [None] [chore] Update .coderabbit.yaml review configuration (#7351) |
| 3078 | ce580ce4f5 | Richard Huo | 2025-08-28 | [None][feat] KV Cache Connector API (#7228) |
| 3079 | 085dc19bfa | aalanwyr | 2025-08-29 | [TRTLLM-6646][test] NIM migration to TRT-LLM LLMAPI : Add QWQ-32b torch test (#7284) |
| 3080 | e0253ee805 | Daniel Stokes | 2025-08-29 | [None][perf] Disable Swap AB when num tokens exceeds N dimension (#7104) |
| 3081 | ccb800f909 | Yuan Tong | 2025-08-29 | [TRTLLM-7457][ci] Update unittest parallel config (#7297) |
| 3082 | b093d94d34 | Shiyu Li | 2025-08-28 | [https://nvbugs/5445466][fix] Bypass MLP TP split for MNNVL in DeepSeek V3 to avoid hanging. (#6886) |
| 3083 | 367ff88a5e | dongfengy | 2025-08-28 | [None][feat] Refactor llama4 for multimodal encoder IFB (#6844) |
| 3084 | 460a34c671 | Yanchao Lu | 2025-08-29 | [None][chore] Some improvements for CI stability (#7199) |
| 3085 | a419b77fb5 | Nikita Korobov | 2025-08-28 | [None][fix] mxfp4 padding bug for TRT-LLM and CUTLASS MoE backends (#7214) |
| 3086 | 1e644fa28a | Emma Qiao | 2025-08-29 | [None][infra] Waive failed tests on main branch 08/26 (#7346) |
| 3087 | c4f823319b | yunruis | 2025-08-28 | [None][doc] add adp balance blog (#7213) |
| 3088 | 08f935681d | Neta Zmora | 2025-08-28 | [https://nvbugs/5474453][fix] fix path to tested model (#7272) |
| 3089 | 23f72c8bbd | Kaiyu Xie | 2025-08-28 | [None] [feat] Use numa to bind CPU (#7304) |
| 3090 | 53163bf1df | Zongfei Jing | 2025-08-28 | [TRTLLM-6876][feat] Add low precision all2all for mnnvl (#7155) |
| 3091 | ae89163368 | QI JUN | 2025-08-28 | [None][ci] skip TestGPTOSS (#7333) |
| 3092 | 4541655e5f | William Zhang | 2025-08-27 | [https://nvbugs/5430124][ci] Unwaive Mistral 3.1 Small tests (#7274) |
| 3093 | 7f4adca8b8 | Venky | 2025-08-27 | [None][fix] Disable mandatory PR checklist enforcement (#7325) |
| 3094 | 39c9ffda5a | QI JUN | 2025-08-28 | [None][ci] fix test list name (#7321) |
| 3095 | c1e7fb9042 | Pengyun Lin | 2025-08-28 | [TRTLLM-7207][feat] Chat completions API for gpt-oss (#7261) |
| 3096 | f30768e70d | Venky | 2025-08-27 | [TRTLLM-6822][infra] Add PR-Checklist github action and modify PR template (#6029) |
| 3097 | 8a619be828 | Kaiyu Xie | 2025-08-27 | [None] [chore] Make disagg example compatible with recommended usage (#7121) |
| 3098 | 7cfa475e05 | Martin Marciniszyn Mehringer | 2025-08-27 | [None][fix] Remove the wheel from intermediate docker storage (#7175) |
| 3099 | 9d345b31c0 | bhsueh_NV | 2025-08-27 | [https://nvbugs/5453727][fix] unwaive qwen3 CI tests (#7293) |
| 3100 | 462169bfc9 | Eran Geva | 2025-08-27 | [https://nvbugs/5458798][fix] AD perf test outliers handling, tightened threshold, re-enabled in CI, fixed mem threshold (#7189) |
| 3101 | d09add5ede | QI JUN | 2025-08-27 | [None][ci] parallelize unit tests of auto deploy in B200 (#7291) |
| 3102 | 8dc62ffac4 | Emma Qiao | 2025-08-27 | [None][infra] Waive failed tests on main (#7300) |
| 3103 | f082e4857c | xinhe-nv | 2025-08-27 | [TRTLLM-7250][fix] waive failed cases (#7292) |
| 3104 | 8b216135f0 | Mike Iovine | 2025-08-27 | [None][refactor] Move draft token padding out of Drafter (#7134) |
| 3105 | dbd4f21687 | nvamyt | 2025-08-27 | [None][fix] Update maxnt of llama_v3.2_1b bench (#7279) |
| 3106 | f167b1fd99 | bhsueh_NV | 2025-08-27 | [https://nvbugs/5453727][fix] Fix bug of how GPT-OSS setup the parameters in CI (#7151) |
| 3107 | e08c7cf17b | QI JUN | 2025-08-27 | [None][ci] remove test_llm_api_autodeploy from B200 test db (#7282) |
| 3108 | abdb2735be | dongxuy04 | 2025-08-27 | [None][fix] Fix possible hang issue in WideEP and move some tests to pre-merge (#7262) |
| 3109 | bed5bc9f2e | Yukun He | 2025-08-27 | [None][chore] Wrap the swiglu into custom op to avoid redundant device copy. (#7021) |
| 3110 | 82bd1871ea | Raayan Dhar | 2025-08-26 | [None][chore] update disagg readme and scripts for pipeline parallelism (#6875) |
| 3111 | 6c7813e821 | Yuan Tong | 2025-08-27 | [TRTLLM-7457][ci] Update & cleanup unittest parallel config (#7254) |
| 3112 | bc84758626 | Iman Tabrizian | 2025-08-26 | [None][feat] Add logging for OAI disagg server (#7232) |
| 3113 | d0d8903a7f | Zhenhuan Chen | 2025-08-27 | [TRTLLM-6960][fix] replace flasky scaled_mm test with more stable config (#7089) |
| 3114 | ff4047414b | Shunkangz | 2025-08-27 | [None][opt] Balance the request based on number of tokens in AttentionDP (#7183) |
| 3115 | e12868bc00 | Fanrong Li | 2025-08-27 | [None][fix] Remove and fuse some element-wise ops in the ds-r1-fp8 model (#7238) |
| 3116 | ccb6aadea8 | Zhou Yuxin | 2025-08-27 | [https://nvbugs/5412456][fix] Remove from waives.txt (#7248) |
| 3117 | 028235404b | Jin Li | 2025-08-27 | [TRTLLM-6633][feat] Padding for piecewise cudagraph (#6750) |
| 3118 | 87d1d3ab06 | Iman Tabrizian | 2025-08-26 | [None][update] Update disagg code owners (#7266) |
| 3119 | 0f947c64cb | Fridah-nv | 2025-08-26 | [None][doc] Update autodeploy README.md, deprecate lm_eval in examples folder (#7233) |
| 3120 | 78ecfbb4a4 | Frank | 2025-08-26 | [None][fix] Fix data type of KV Cache percentage in bench. (#7230) |
| 3121 | 040f4c70d3 | Void | 2025-08-27 | [None][perf] Accelerate global scale calculations for deepEP fp4 combine (#7126) |
| 3122 | baef70e67e | QI JUN | 2025-08-26 | [None][ci] move qwen3 tests from b200 to gb200 (#7257) |
| 3123 | 2d0c9b383f | Maurits de Groot | 2025-08-26 | [None][fix] Updated blog9_Deploying_GPT_OSS_on_TRTLLM (#7260) |
| 3124 | 80043affb5 | xinhe-nv | 2025-08-26 | [None][chore] Add failed cases into waives.txt (#7251) |
| 3125 | a142c0c4de | Emma Qiao | 2025-08-26 | [None][infra] Add retry 3 times if ssh cluster failed (#6859) |
| 3126 | f01101f687 | Zhou Yuxin | 2025-08-26 | [None][feat] Hopper Fp8 context mla (#7116) |
| 3127 | 23ed0c892d | amitz-nv | 2025-08-26 | [https://nvbugs/5477332][fix] Relax atol in test_mamba2_chunk_scan_combined_prefill_chunking (#7215) |
| 3128 | bf377d0b8e | Guoming Zhang | 2025-08-26 | [None][doc] Display tech blog for nvidia.github.io domain. (#7241) |
| 3129 | cf50ba2980 | Zheng Duan | 2025-08-26 | [TRTLLM-6549][feat] add perf metrics endpoint to openai server and openai disagg server (#6985) |
| 3130 | 1a929a1490 | Zheng Duan | 2025-08-26 | [https://nvbugs/5457504][fix] fix kv cache event test in disaggregated worker tests (#7028) |
| 3131 | d8bd8843fc | nvamyt | 2025-08-26 | [None][test] Update qwen3 timeout to 60 minutes (#7200) |
| 3132 | bbc1478627 | yuanjingx87 | 2025-08-25 | [None][chore] Update CI allowlist 2025-08-25 (#7229) |
| 3133 | b165f8bc97 | qixiang-99 | 2025-08-25 | fix/improve kvcache allocation in PyTorch runtime (#5933) |
| 3134 | 92576488d3 | William Zhang | 2025-08-25 | [None][feat] Skip prefetching consolidated safetensors when appropriate (#7013) |
| 3135 | 4f84a45899 | Zheng Duan | 2025-08-26 | [https://nvbugs/5452463][doc] update disagg doc about UCX_MAX_RNDV_RAILS (#7205) |
| 3136 | 20922b7d1f | Leslie Fang | 2025-08-26 | [None][chore] Create PyExecutor from TorchLlmArgs Part 1 (#7105) |
| 3137 | b845eb7a3a | ruodil | 2025-08-26 | [None][test] add kv cache size in bench metric and fix failed cases (#7160) |
| 3138 | 9df15b2104 | Leslie Fang | 2025-08-26 | [None][doc] update feature_combination_matrix doc (#6691) |
| 3139 | 2101d46d68 | Grzegorz Kwasniewski | 2025-08-26 | [TRTLLM-6342][feat] TP Sharding read from the model config (#6972) |
| 3140 | 97d550b4ba | Lucas Liebenwein | 2025-08-25 | [None] [AutoDeploy] canonicalize_graph before shape prop for consistent state_dict (#7223) |
| 3141 | bf1b958f1a | Bo Li | 2025-08-26 | [TRTLLM-7319][perf] Fuse slicing into MoE. (#6728) |
| 3142 | e8e7e52892 | Daniel Cámpora | 2025-08-25 | [None][chore] Refactored the handle logits pp communication (#7154) |
| 3143 | 788fc62d23 | Frank | 2025-08-25 | [None][fix] Update to pull LLM from a central location. (#6458) |
| 3144 | 6a44e5b9d1 | chenfeiz0326 | 2025-08-25 | [https://nvbugs/5440241][fix] Fix 70B GSM8K Accuracy drop (#6967) |
| 3145 | 200db3b809 | Emma Qiao | 2025-08-25 | [None][infra] Waive failed tests on main branch (#7201) |
| 3146 | bea5e07fb7 | QI JUN | 2025-08-25 | [None][refactor] refactor the CUDA graph runner to manage all CUDA graphs (#6846) |
| 3147 | b32e00e9fd | shaharmor98 | 2025-08-25 | [None][chore] remove CLI support for mamba cache dtype setting (#7119) |
| 3148 | a1e03af0f4 | amitz-nv | 2025-08-25 | [TRTLLM-7346][fix] Improve performance of PyTorchModelEngine._get_lora_params_from_requests (#7033) |
| 3149 | be6d92f09f | Enwei Zhu | 2025-08-25 | [None][fix] Fix MoE load balancer config loading (#7150) |
| 3150 | f61b74f796 | Ivy Zhang | 2025-08-25 | [None][test] add l20 specific qa test list (#7067) |
| 3151 | 630e67b845 | QI JUN | 2025-08-25 | [None][ci] waive test_mamba2_chunk_scan_combined_prefill_chunking[seqlens1-8] (#7194) |
| 3152 | 9c5b464fe0 | Yukun He | 2025-08-25 | [None][feat] Apply AutoTuner to fp8_block_scale_deep_gemm to trigger JIT ahead of time. (#7113) |
| 3153 | c038fb3ef4 | Bo Deng | 2025-08-25 | [None][chore] cherry-pick 6940 (#7097) |
| 3154 | 3ba9afcc7b | xinhe-nv | 2025-08-25 | [None][feat] add gpt-osss tests to sanity list (#7158) |
| 3155 | 6e131602b2 | Bo Deng | 2025-08-25 | [TRTLLM-7096][infra] Testing cache transmission functionality in Python (#7025) |
| 3156 | 486bc763c3 | Yiqing Yan | 2025-08-25 | [None][infra] Split DGX_B200 stage into multiple parts and pre-/post-merge (#7074) |
| 3157 | 31979aefac | Robin Kobus | 2025-08-24 | [None] [ci] Reorganize CMake and Python integration test infrastructure for C++ tests (#6754) |
| 3158 | 068056677f | ajrasane | 2025-08-24 | [None][chore] Enable auto deploy accuracy test in CI (#7179) |
| 3159 | ec35481b0a | Yanchao Lu | 2025-08-24 | [None][infra] Prepare for single GPU GB200 test pipeline (#7073) |
| 3160 | 48155f52bf | dongfengy | 2025-08-24 | [TRTLLM-7321][doc] Refine GPT-OSS doc (#7180) |
| 3161 | 19a0ea363b | dongxuy04 | 2025-08-24 | [TRTLLM-6743][feat] Optimize and refactor alltoall in WideEP (#6973) |
| 3162 | 35e0ae484a | amitz-nv | 2025-08-24 | [https://nvbugs/5467232][fix] Fix load_torch_hf_lora to override lora_config.trtllm_modules_to_hf_modules with default only when it has no value (#7132) |
| 3163 | 96ff82e77a | Iman Tabrizian | 2025-08-23 | [None][fix] Waive test (#7185) |
| 3164 | 3d54a1a521 | Grace Ho | 2025-08-22 | [None] [feat] nsys profile output kernel classifier (#7020) |
| 3165 | 81fd468fec | Frank | 2025-08-22 | [None][fix] Correct KV cache percentage report out. (#7102) |
| 3166 | b36460d7b5 | Izzy Putterman | 2025-08-22 | [None][feat] Deepseek: Start Eagle work (#6210) |
| 3167 | 37543a9ad7 | Robin Kobus | 2025-08-22 | [None][refactor] Simplify decoder state initialization for speculative decoding (#6869) |
| 3168 | c232ba8157 | tomeras91 | 2025-08-22 | [TRTLLM-4921][feat] Enable chunked prefill for Nemotron-H (#6334) |
| 3169 | e3de5758a3 | Suyog Gupta | 2025-08-22 | [#7136][feat] trtllm-serve + autodeploy integration (#7141) |
| 3170 | 907bc22fcb | Yiqing Yan | 2025-08-22 | [None][chore] Bump version to 1.1.0rc2 (#7167) |
| 3171 | 1388e84793 | QI JUN | 2025-08-22 | [None][ci] move all B200 TensorRT test cases to post merge (#7165) |
| 3172 | b8b2bd4a0a | xinhe-nv | 2025-08-22 | [TRTLLM-7245][feat] add test_multi_nodes_eval tests (#7108) |
| 3173 | d94cc3fa3c | dongfengy | 2025-08-22 | [TRTLLM-7321][doc] Add GPT-OSS Deployment Guide into official doc site (#7143) |
| 3174 | 898f37faa0 | Linda | 2025-08-22 | [None][feat] Enable nanobind as the default binding library (#6608) |
| 3175 | a49cf684f8 | Emma Qiao | 2025-08-22 | [TRTLLM-5801][infra] Add more RTX Pro 6000 test stages (#5126) |
| 3176 | 099f081e03 | Daniel Cámpora | 2025-08-22 | [TRTLLM-7155][feat] Unify sampler handle logits implementation. (#6867) |
| 3177 | 983dd7e57c | Yukun He | 2025-08-22 | [None][fix] Fix mm_placholder_counts extraction issue. (#7118) |
| 3178 | 4017f7cd6b | xinhe-nv | 2025-08-22 | [None][chore] Add failed cases into waives.txt (#7109) |
| 3179 | 07c711eb1f | Wanli Jiang | 2025-08-22 | [TRTLLM-6825][fix] Update lora for phi4-mm (#6817) |
| 3180 | c5036cb536 | Suyog Gupta | 2025-08-21 | [None][docs] update stale link for AutoDeploy (#7135) |
| 3181 | 6f245ec78b | dominicshanshan | 2025-08-22 | [None][chore] Mass integration of release/1.0 (#6864) |
| 3182 | f7c597ec40 | Daniel Stokes | 2025-08-22 | [None][perf] Make finalize fusion part of the tactic selection logic (#6915) |
| 3183 | e18dacc931 | Fridah-nv | 2025-08-21 | [#4403][refactor] Move fusion, kvcache, and compile to modular inference optimizer (#7057) |
| 3184 | 344bc4575d | Emma Qiao | 2025-08-22 | [None][infra] Waive failed case for main branch (#7129) |
| 3185 | f49dafe0da | Dimitrios Bariamis | 2025-08-21 | [https://nvbugs/5394409][feat] Support Mistral Small 3.1 multimodal in Triton Backend (#6714) |
| 3186 | 9a2b44d0f2 | brb-nv | 2025-08-21 | [None][chore] No-op changes to support context parallelism in disaggregated serving later (#7063) |
| 3187 | 90bfc8cc29 | Yuan Tong | 2025-08-21 | [https://nvbugs/5453827][fix] Fix RPATH of th_common shared library to find pip-installed NCCL (#6984) |
| 3188 | c7269ea93a | ChristinaZ | 2025-08-21 | [https://nvbugs/5392414] [fix] Add customized default routing method (#6818) |
| 3189 | 2d40e8750b | Farshad Ghodsian | 2025-08-21 | [None][doc] Update gpt-oss deployment guide to latest release image (#7101) |
| 3190 | ba0a86e0bb | bhsueh_NV | 2025-08-21 | [https://nvbugs/5437405][fix] qwen3 235b eagle3 ci (#7000) |
| 3191 | 647a52698a | Fridah-nv | 2025-08-20 | [https://nvbugs/5443039][fix] Fix AutoDeploy pattern matcher for torch 2.8 (#7076) |
| 3192 | cbcea33279 | Yao Yao | 2025-08-21 | [fix]: use safeInitRowMax instead of fp32_lowest to avoid NaN (#7087) |
| 3193 | 21f4434404 | xinhe-nv | 2025-08-21 | [None][chore] waive failed cases on H100 (#7084) |
| 3194 | 41ff4901ee | Fan - Yunfan | 2025-08-21 | [None][fix] Fix const modifier inconsistency in log function declaration/implementation (#6679) |
| 3195 | f03053b4dd | Fridah-nv | 2025-08-20 | [None][fix] update accelerate dependency to 1.7+ for AutoDeploy (#7077) |
| 3196 | 9f51f8d20c | BatshevaBlack | 2025-08-21 | [None][infra] Upgrade UCX to v1.19.x and NIXL to 0.5.0 (#7024) |
| 3197 | 75b8a90816 | Chang Liu | 2025-08-20 | [None][fix] Fix llama4 multimodal by skipping request validation (#6957) |
| 3198 | 0893afae3d | Yechan Kim | 2025-08-21 | [TRTLLM-6771][feat] Support MMMU for multimodal models (#6828) |
| 3199 | 73d2daa386 | bhsueh_NV | 2025-08-21 | [https://nvbugs/5457489][fix] unwaive some tests (#6991) |
| 3200 | a918de710a | QI JUN | 2025-08-21 | [None][ci] move some tests of b200 to post merge (#7093) |
| 3201 | e5e417019b | Jin Li | 2025-08-21 | [None][chore] Only check the bindings lib for current build (#7026) |
| 3202 | 92daec1115 | Dom Brown | 2025-08-20 | [TRTLLM-7348] [feat] Enable Cross-Attention to use XQA kernels for Whisper (#7035) |
| 3203 | 8ac7dec623 | Yuhao Yao | 2025-08-20 | [None][fix] Fix W4A8 MoE kernel issue (#7072) |
| 3204 | f84dd64250 | Emma Qiao | 2025-08-20 | [None][infra] Waive failed tests on main branch 8/20 (#7092) |
| 3205 | b95cab2a7c | Robin Kobus | 2025-08-20 | [None][ci] move unittests to sub-directories (#6635) |
| 3206 | 983fb8e607 | Kanghwan | 2025-08-20 | [None][chore] Update namelist in blossom-ci (#7015) |
| 3207 | 20f54cb272 | Zhenhuan Chen | 2025-08-20 | [None][fix] fix scaffolding dynasor test (#7070) |
| 3208 | 020fed97b6 | Yueh-Ting (eop) Chen | 2025-08-20 | [TRTLLM-6341][chore] Preliminary refactors on the kv cache manager before supporting swa kv cache reuse (#6767) |
| 3209 | e27088421e | Iman Tabrizian | 2025-08-19 | [None][infra] "[TRTLLM-6960][fix] enable scaled_mm tests (#6936)" (#7059) |
| 3210 | 9e71b4fda4 | xinhe-nv | 2025-08-20 | [TRTLLM-7205][feat] add llama4 tp4 tests (#6989) |
| 3211 | 3f6a9267f1 | Leslie Fang | 2025-08-20 | [None][infra] update feature_combination_matrix of disaggregated and chunked prefill (#6661) |
| 3212 | ce53832610 | Chang Liu | 2025-08-19 | [TRTLLM-7326][feat] Add standalone multimodal encoder (#6743) |
| 3213 | fc85e3db1c | Ivy Zhang | 2025-08-20 | [None][fix] fix llmapi import error (#7030) |
| 3214 | 30da5d3cc4 | Bo Deng | 2025-08-20 | [None][chore] unwaive test_disaggregated_genbs1 (#6944) |
| 3215 | c02592d051 | Fridah-nv | 2025-08-19 | [None][autodeploy] Add group attention pattern for solar-pro-preview (#7054) |
| 3216 | 0e30fe4372 | Jinyang Yuan | 2025-08-20 | [None][fix] Fix assertion errors of quantization when using online EPLB (#6922) |
| 3217 | 7334f9390c | Michal Guzek | 2025-08-19 | [None][fix] Accommodate Phi3/4 to work with ModelOpt's FP8 ckpts in Torch (#6761) |
| 3218 | d26a5a93ad | Yanchao Lu | 2025-08-19 | [https://nvbugs/5451296][bug] Cherry-pick #7017 from release/1.0 branch (#7043) |
| 3219 | e07fcc3a22 | pcastonguay | 2025-08-19 | [https://nvbugs/5444937][chore] Fixing KV events tests (#7004) |
| 3220 | 7e135d2ea7 | zhhuang-nv | 2025-08-19 | [None][feat] Use Separate QKV Input Layout for Context MLA (#6538) |
| 3221 | 8f95f35503 | Emma Qiao | 2025-08-19 | [None][infra] Waive failed tests on main (#7037) |
| 3222 | 07506bccbe | Yiqing Yan | 2025-08-19 | [None][chore] Remove duplicate test waives (#7044) |
| 3223 | 655d0f48d0 | Fanrong Li | 2025-08-19 | [https://nvbugs/5455140][fix] unwaive DSR1-fp4 throughput_tp8 (#7022) |
| 3224 | f0bfb49219 | tomeras91 | 2025-08-19 | [https://nvbugs/5458874][fix] Fix Nemotron-H flaky CUDA graph / overlap scheduler test (#6996) |
| 3225 | a54c53652b | amitz-nv | 2025-08-19 | [TRTLLM-7263][fix] Prevent recreation of cublas handles in lora_grouped_gemm every call (#6968) |
| 3226 | 19667304b5 | Xianjie Qiao | 2025-08-19 | [None] [chore] Update wide-ep genonly scripts (#6995) |
| 3227 | 9a74ee9dae | Kaiyu Xie | 2025-08-19 | [None] [doc] Add more documents for large scale EP (#7029) |
| 3228 | 953f4fd69e | Zero Zeng | 2025-08-19 | [None][fix] acceptance rate calculation fix in benchmark_serving (#6746) |
| 3229 | 2c86cee38c | xinhe-nv | 2025-08-19 | [None][chore] Remove closed bugs (#6969) |
| 3230 | 54ec2c1af1 | Shunkangz | 2025-08-19 | [None][opt] Add batch wait timeout in fetching requests (#6923) |
| 3231 | 636c622bb8 | Eran Geva | 2025-08-19 | [https://nvbugs/5458798][fix] Relaxed test threshold, added documentation (#6997) |
| 3232 | bff5fdf6df | Ivy Zhang | 2025-08-19 | [TRTLLM-6541][test] Add NIM Related Cases Part 1 (#6684) |
| 3233 | daa2a65d37 | William Zhang | 2025-08-18 | [https://nvbugs/5454875][ci] Unwaive Mistral Small 3.1 test (#7011) |
| 3234 | e90280a84d | fredricz-20070104 | 2025-08-19 | [TRTLLM-6541][test] Add NIM Related Cases [StarCoder2_7B] and [Codestral_22B_V01] (#6939) |
| 3235 | 816a120af6 | Fanrong Li | 2025-08-19 | [TRTLLM-6991][chore] add DeepSeek-R1 FP8 accuracy tests on Blackwell (#6710) |
| 3236 | 2bb90ba002 | Zhenhuan Chen | 2025-08-19 | [TRTLLM-6960][fix] enable scaled_mm tests (#6936) |
| 3237 | 06911c0173 | Venky | 2025-08-18 | [None] [infra] stricter coderabbit pr title generation instructions (#6918) |
| 3238 | a15af879ec | Yi Zhang | 2025-08-19 | [None][refactor] Refactor Torch Compile Backend, MoeLoadBalancer and warmup Logic (#6615) |
| 3239 | 71e28eab36 | Lizhi Zhou | 2025-08-19 | [TRTLLM-7014][chore] Add accuracy test for ctx and gen workers with different models (#6741) |
| 3240 | dabebb2c7a | Wanli Jiang | 2025-08-19 | [https://nvbugs/5371480][fix] Enable test_phi3_small_8k (#6938) |
| 3241 | 97ba0eb879 | Fridah-nv | 2025-08-18 | [None][autodeploy] Doc: fix link path in trtllm bench doc (#7007) |
| 3242 | e76e5c640f | Leslie Fang | 2025-08-19 | [None][infra] Enable accuracy test for mtp and chunked prefill (#6314) |
| 3243 | d16af87d03 | Daniel Cámpora | 2025-08-19 | [TRTLLM-7158][feat] Introduce sampler options in trtllm bench (#6855) |
| 3244 | d1d17dbeba | Yanchao Lu | 2025-08-19 | [None][infra] Cherry-pick #6836 from main branch and improve SSH connection (#6971) (#7005) |
| 3245 | 425dad01fd | Martin Marciniszyn Mehringer | 2025-08-18 | [None][fix] Clean up linking to CUDA stub libraries in build_wheel.py (#6823) |
| 3246 | 1ce23545fc | Yiqing Yan | 2025-08-18 | [None][chore] Remove duplicate test waives (#6998) |
| 3247 | 69ff32f9b1 | Emma Qiao | 2025-08-18 | [None][infra] Waive failed tests on main 0818 (#6992) |
| 3248 | 55f4f2d80c | ChristinaZ | 2025-08-18 | [None] [fix] Fix the macro name (#6983) |
| 3249 | 5ec15b98f0 | Shi Xiaowei | 2025-08-18 | [TRTLLM-7030][fix] uppercase def value in pd-config (#6981) |
| 3250 | e88cb92f24 | Kaiyu Xie | 2025-08-18 | [None] [feat] Support accurate device iter time (#6906) |
| 3251 | 8b05b5d801 | Bo Li | 2025-08-18 | [None][doc] Update gpt oss doc (#6954) |
| 3252 | ce0b13ea02 | Leslie Fang | 2025-08-18 | [None][infra] update feature_combination_matrix of disaggregated and Eagle3 (#6945) |
| 3253 | d6322f70b7 | Naveassaf | 2025-08-17 | [https://nvbugs/5451028][fix] Constrain NemotronSuper test parameters to prevent OOMs (#6970) |
| 3254 | 3a49b47081 | amitz-nv | 2025-08-17 | [https://nvbugs/5390853][fix] Fix _test_openai_lora.py - disable cuda graph (#6965) |
| 3255 | cc6d763824 | Emma Qiao | 2025-08-17 | [None][infra]Waive failed cases in main branch (#6951) |
| 3256 | 1e72721e8c | ChristinaZ | 2025-08-17 | [None][feat] Add single block version renormalized routing kernel (#6756) |
| 3257 | 85cbd0263b | bhsueh_NV | 2025-08-17 | [None][feat] Support Yarn on Qwen3 (#6785) |
| 3258 | 22d59a6f61 | Fan - Yunfan | 2025-08-16 | [None][fix] Using RAII to automatically manage the allocation and release of va_list for potential resource leak (#6758) |
| 3259 | f6ff0e3311 | Izzy Putterman | 2025-08-15 | [None][fix] Skip Topk if 0 (#6934) |
| 3260 | 53312eeebd | Daniel Cámpora | 2025-08-16 | [TRTLLM-7157][feat] BREAKING CHANGE Introduce sampler_type, detect sampler according to options (#6831) |
| 3261 | ec3d9f8052 | Yiqing Yan | 2025-08-16 | [None][chore] Bump version to 1.1.0rc1 (#6953) |
| 3262 | 9505727d31 | brb-nv | 2025-08-15 | [https://nvbugs/5401114][fix] Unwaive Gemma3 tests (#6952) |
| 3263 | 1f8ae2b2db | Yuening Li | 2025-08-16 | [TRTLLM-5863][feat] Support MoE INT8 Weight-Only-Quantization in PyTorch Workflow (#6629) |
| 3264 | 0ad0b967bb | dongfengy | 2025-08-15 | [None][fix] Make TP working for Triton MOE (in additional to EP we are using) (#6722) |
| 3265 | 4162d2d746 | ajrasane | 2025-08-15 | [None][test] Add accuracy evaluation for AutoDeploy (#6764) |
| 3266 | 4127d77678 | yifeizhang-c | 2025-08-16 | [https://nvbugs/5394392][fix] Enlarge scheduler capacity under disagg bs == 1 (#6537) |
| 3267 | 6037fe3716 | Perkz Zheng | 2025-08-15 | [https://nvbugs/5394685][fix] proper fix for the accuracy issue in 2CTA MLA kernels (#6941) |
| 3268 | 18ccd053d3 | liji-nv | 2025-08-15 | [https://nvbugs/5427801][fix] Torch compile support for Llama4 and Ea… (#6858) |
| 3269 | f7dbc1435a | tomeras91 | 2025-08-15 | [None] [chore] Mamba cache in separate file (#6796) |
| 3270 | c2fe8b03a2 | Xianjie Qiao | 2025-08-15 | [https://nvbugs/5405041][fix] Update wide-ep doc (#6933) |
| 3271 | 1c1d5d2495 | peaceh-nv | 2025-08-15 | [https://nvbugs/5451373][fix] : Fix the accuracy issue when using FP8 context MLA (#6881) |
| 3272 | fadb5e75dd | Zhenhua Wang | 2025-08-15 | [None][chore] add a EditorConfig config (#6897) |
| 3273 | b23fdfc62f | xinhe-nv | 2025-08-15 | [None][chore] Add failed cases into waives.txt (#6914) |
| 3274 | 8e252256f5 | jmydurant | 2025-08-15 | [None][doc] Modify the description for mla chunked context (#6929) |
| 3275 | 3a987891d8 | Yanchao Lu | 2025-08-15 | [TRTLLM-7141][infra] Use repo mirrors to avoid intermittent network failures (#6836) |
| 3276 | e54ba75dac | Bo Deng | 2025-08-15 | [None][fix] Update tests to use standardized uppercase backend identifiers (#6921) |
| 3277 | 9a133e9b41 | Wanli Jiang | 2025-08-15 | [https://nvbugs/5415862][fix] Update cublas as 12.9.1 and cuda memory alignment as 256 (#6501) |
| 3278 | 15aabc1540 | Bo Li | 2025-08-15 | [None][fix] Fix perfect router. (#6797) |
| 3279 | 2cc59aacb3 | Frank | 2025-08-14 | [None][fix] Correct reporting of torch_dtype for ModelConfig class. (#6800) |
| 3280 | 11d08c33af | Yunfan Fan | 2025-08-15 | [None][fix] Fix responsibility boundary between the assert and tllmException files (#6723) |
| 3281 | 70e352a6f7 | JunyiXu-nv | 2025-08-15 | [https://nvbugs/5437106][fix] Add L4 Scout benchmarking WAR option in deploy guide (#6829) |
| 3282 | 11d89a3732 | Perkz Zheng | 2025-08-15 | [https://nvbugs/5394685][fix] using static scheduler 2CTA MLA as WAR for an accuracy issue (#6896) |
| 3283 | 5346eb7bc5 | hlu1 | 2025-08-14 | [None][doc] Update gpt-oss doc on MoE support matrix (#6908) |
| 3284 | b13a5a99b2 | Aurelien Chartier | 2025-08-14 | [None][chore] Add tests for non-existent and completed request cancellation (#6840) |
| 3285 | 5c2f0fd03d | qianbiao | 2025-08-15 | [None] [feat] Add Tencent HunYuanMoEV1 model support (#5521) |
| 3286 | 8b237b943b | Raayan Dhar | 2025-08-14 | [https://nvbugs/5441714][chore] remove skip on disagg n-gram test (#6872) |
| 3287 | 078e907b16 | Mike Iovine | 2025-08-14 | [https://nvbugs/5455651][fix] Make ngram use XQA attention on Blackwell (#6873) |
| 3288 | 26f413ad90 | Bo Li | 2025-08-15 | [https://nvbugs/5450262][fix] Fix unsupported alltoall use case (#6882) |
| 3289 | 69574ad730 | Matthias Jouanneaux | 2025-08-14 | [TRTLLM-5966][feat] Helix: extend mapping to support different CP types (#6816) |
| 3290 | 96339c69a9 | Emma Qiao | 2025-08-14 | [None][infra] Waive failed cases on main (#6902) |
| 3291 | afb116f703 | Jiagan Cheng | 2025-08-14 | [None][fix] Fix python-only build that uses TRTLLM_USE_PRECOMPILED (#6825) |
| 3292 | 4aed7a7d19 | kris1025 | 2025-08-14 | [TRTLLM-6853][feat] refactor deepseekv3 model (#6698) |
| 3293 | ffc976ceaf | Pengbo Wang @ NVIDIA | 2025-08-14 | [https://nvbugs/5445466][fix] fix deepseek r1 hang by not enabling mnnvl by default (#6860) |
| 3294 | 1095dfd03c | Shi Xiaowei | 2025-08-14 | [None][fix] BREAKING CHANGE: Mismatch between docs and actual commands (#6323) |
| 3295 | 5cd8c0f6cc | chenfeiz0326 | 2025-08-14 | [None][test] Add perf-sweep scripts (#6738) |
| 3296 | 345d3d3524 | Tao Li @ NVIDIA | 2025-08-14 | [None][doc] update moe support matrix for DS R1 (#6883) |
| 3297 | a700646132 | NVJiangShao | 2025-08-14 | [None][fix] Add FP4 all2all unitest and fix a bug for module WideEPMoE (#6784) |
| 3298 | 0132c1db84 | Yan Chunwei | 2025-08-14 | [https://nvbugs/5427043][fix] request length exceeds max_num_tokens (#6821) |
| 3299 | d8acca495b | Bo Deng | 2025-08-14 | [TRTLLM-6675][infra] Cherry-pick https://github.com/NVIDIA/TensorRT-LLM/pull/6623 (#6735) |
| 3300 | 4200fa46d1 | jmydurant | 2025-08-14 | [None][feat] Add support for Hopper MLA chunked prefill (#6655) |
| 3301 | 868c5d166e | Zhenhua Wang | 2025-08-14 | [None][chore] fix markdown format for the deployment guide (#6879) |
| 3302 | ef53de8eef | Izzy Putterman | 2025-08-13 | [None][feat] Add test for speculative rejection sampler (2-model) (#6542) |
| 3303 | eb4ed18a63 | Linda | 2025-08-14 | [None][fix] max_num_sequences argument in nanobind (#6862) |
| 3304 | 7cba883932 | Mike Iovine | 2025-08-13 | [https://nvbugs/5410399][chore] Unwaive mtp llmapi test (#6833) |
| 3305 | 58f7783ea4 | Perkz Zheng | 2025-08-14 | [https://nvbugs/5394685][fix] the bug with spec-decoding + SWA && an accuracy issue related to 2CTA MLA (#6834) |
| 3306 | 6c52bb07ff | Tin-Yin Lai | 2025-08-13 | [https://nvbugs/5302040][feat] Add whisper support (Bert Attention on SM100 and GPTAttention for cross attention on SM100) (#5527) |
| 3307 | bda42f8c3a | danielafrimi | 2025-08-13 | [None][feat] Support running heterogeneous model execution for Nemotron-H (#6866) |
| 3308 | c7e6145409 | Emma Qiao | 2025-08-13 | [None][infra] Waive failed cases on main (#6863) |
| 3309 | 2198587b35 | Anthony Chang | 2025-08-13 | [https://nvbugs/5378031] [feat] Hopper W4A8 MoE supports ModelOpt ckpt for PyT backend (#6200) |
| 3310 | 8416d7fea8 | Zhenhua Wang | 2025-08-13 | [https://nvbugs/5412885][doc] Add the workaround doc for H200 OOM (#6853) |
| 3311 | 0fad6029f7 | Perkz Zheng | 2025-08-13 | [TRTLLM-7093][fix] the perf regression to cvt_fp4 kernels (#6851) |
| 3312 | fe7dda834d | Shi Xiaowei | 2025-08-13 | [TRTLLM-7030][fix] Refactor the example doc of dist-serving (#6766) |
| 3313 | bc5f766e0e | Yukun He | 2025-08-13 | [TRTLLM-4501][feat] AutoTuner tuning config refactor and valid tactic generalization. (#6545) |
| 3314 | 1d80df0955 | Void | 2025-08-13 | [None][feat] DeepEP LL combine FP4 (#6822) |
| 3315 | 50e5e725e9 | Zhou Yuxin | 2025-08-13 | [https://nvbugs/5412456][fix] Fix an illegal instruction was encountered (#6776) |
| 3316 | 2e0081b53e | Aurelien Chartier | 2025-08-12 | [#6530][fix] Fix script when using calibration tensors from modelopt (#6803) |
| 3317 | f68e03e646 | Mike Iovine | 2025-08-12 | [https://nvbugs/5452167][fix] Fix ngram padding issue (#6837) |
| 3318 | 12102e2d48 | Yechan Kim | 2025-08-13 | [TRTLLM-6772][feat] Multimodal benchmark_serving support (#6622) |
| 3319 | 1bbc0e323b | Fanrong Li | 2025-08-13 | [None][fix] Pre-allocate workspaces for DeepGEMM MoE to avoid frequent cudaFree/cudaMalloc (#6811) |
| 3320 | 47806f09d9 | Kaiyu Xie | 2025-08-13 | feat: Support custom repo_dir for SLURM script  (#6546) |
| 3321 | 2923eb88a1 | rakib-hasan | 2025-08-12 | [None][fix] Refactoring input prep to allow out-of-tree models (#6497) |
| 3322 | bd9a6dd9ab | dongxuy04 | 2025-08-13 | [TRTLLM-7008][fix] fix wideEP weights loading and args (#6789) |
| 3323 | 45c7518032 | Robin Kobus | 2025-08-12 | [None][refactor] Simplify decoder state initialization (#6559) |
| 3324 | dd11e08d26 | Robin Kobus | 2025-08-12 | [#6187][feat] add LayerNorm module (#6625) |
| 3325 | 81f0ded1c4 | nvchenghaoz | 2025-08-12 | [None][feat] Add GPT OSS support for AutoDeploy (#6641) |
| 3326 | a060e12041 | Jhao-Ting Chen | 2025-08-12 | [https://nvbugs/5438869][fix] Set nvfp4 expert w1 w3 weight scale to the same value if they're not (#6656) |
| 3327 | e35fca4272 | xinhe-nv | 2025-08-12 | [TRTQA-2920][chore] improve hang tests (#6781) |
| 3328 | 8845e0f065 | QI JUN | 2025-08-12 | [None][fix] fix ci (#6814) |
| 3329 | ab0d768acf | Shunkangz | 2025-08-12 | [None][fix] Fix attention dp log (#6570) |
| 3330 | f7c13a4aa7 | Liao Lanyu | 2025-08-12 | [TRTLLM-6906][chore] Using pybind to bind functions in thop/attentionOp (#6745) |
| 3331 | 27fc35175e | Sergey Klevtsov | 2025-08-12 | [None][feat] CUTLASS MoE FC2+Finalize fusion (#3294) |
| 3332 | 0dc4b4e699 | Fridah-nv | 2025-08-11 | [#4403][autodeploy] Refactor: Move more transformations to new inf optimizer, Add quantization_source to factory interface (#6760) |
| 3333 | 7c686ba8de | Enwei Zhu | 2025-08-12 | [TRTLLM-2285][feat] Enable guided decoding with CUDA graph padding and draft model chunked prefill (#6774) |
| 3334 | b4fcd5f592 | Ziyi Xiong | 2025-08-12 | [https://nvbugs/5441438][fix] Set correct draft length for the cuda graph dummy request (#6701) |
| 3335 | ead89a0e40 | Jinyang Yuan | 2025-08-12 | [None][perf] Improve the performance of online EPLB on Hopper by better overlapping (#6624) |
| 3336 | be9dd4713c | Chang Liu | 2025-08-11 | [https://nvbugs/5385987][fix] Fix Qwen2 quantization issue by pinning transformers version (#6673) |
| 3337 | 56bfc3a6d2 | Aurelien Chartier | 2025-08-11 | [None][chore] Find LLM_ROOT and LLM_BACKEND_ROOT dynamically (#6763) |
| 3338 | 7ab8112450 | rakib-hasan | 2025-08-11 | [None][fix] Refactoring to avoid circular import when importing torch models (#6720) |
| 3339 | c9fe07ede6 | Venky | 2025-08-11 | [TRTLLM-6812][feat] Add standardized GitHub issue templates and disable blank issues (#6494) |
| 3340 | 7e33ed6d61 | Zhenhua Wang | 2025-08-11 | [None][chore] always try-catch when clear build folder in build_wheel.py (#6748) |
| 3341 | 5145e9d40e | Emma Qiao | 2025-08-11 | [None][infra] Unwaive an updated case to test (#6791) |
| 3342 | a2e9153cb0 | Liao Lanyu | 2025-08-11 | [None][doc] Add K2 tool calling examples (#6667) |
| 3343 | 83dbc6c75d | bhsueh_NV | 2025-08-11 | [TRTLLM-5532][feat] store the block of context request into kv cache (#6683) |
| 3344 | 9a8195ef88 | Martin Marciniszyn Mehringer | 2025-08-11 | fix: Ensure that Python stub generation works against libnvidia-ml stubs (#6188) |
| 3345 | d6ad4a9d5b | Emma Qiao | 2025-08-11 | [None][infra] Waive failed tests on main 0811 (#6778) |
| 3346 | 9c358c26e4 | xinhe-nv | 2025-08-11 | [None][chore] remove closed bugs (#6772) |
| 3347 | 62d6c98d68 | Yiqing Yan | 2025-08-11 | [TRTLLM-5633][infra] Force set changed file diff to empty string for post-merge CI (#6777) |
| 3348 | b3e8fa2960 | Eran Geva | 2025-08-11 | [None][test] Test trtllm-bench AD vs, PT BEs on H100 single gpu (#6487) |
| 3349 | 49bcaa4e95 | Tracin | 2025-08-11 | Add gpt-oss GSM8K test. (#6732) |
| 3350 | c566a8d2a2 | Chuang Zhu | 2025-08-11 | [None][fix] fix same pp disagg (#6730) |
| 3351 | 767879ef85 | Bo Deng | 2025-08-11 | [https://nvbugs/5431127][fix] Run test_disaggregated_deepseek_v3_lite_fp8_nixl[DeepSeek-V3-Lite-fp8] only on hopper (#6736) |
| 3352 | 4b4b91ab51 | Zero Zeng | 2025-08-11 | [None][feat] improve dataloading for benchmark_dataset by using batch… (#6548) |
| 3353 | 60073a7ad9 | Yechan Kim | 2025-08-11 | [None][feat] Support SharedTensor on MultimodalParams (#6254) |
| 3354 | b6baa9ed9b | shaharmor98 | 2025-08-11 | [TRTLLM-6823][doc] Add checkpoint refactor docs (#6592) |
| 3355 | 4142320e53 | pcastonguay | 2025-08-10 | [https://nvbugs/5444937][fix] Fixing kv_cache_event unit test (#6753) |
| 3356 | 14b36e07d7 | shaharmor98 | 2025-08-10 | [TRTLLM-6174][feat] Enable FP32 mamba ssm cache (#6574) |
| 3357 | 199f306984 | Yueh-Ting (eop) Chen | 2025-08-10 | [None][chore][kv cache manager] Dead code elimination, we no longer record/fetch through WindowBlockManager:: mContextBlocksByHash (#6249) |
| 3358 | 3c5aec19c2 | Gal Hubara-Agam | 2025-08-10 | [#5048][enhance] AutoDeploy: Optimize prepare_inputs (#6634) |
| 3359 | ee19ca5e58 | Emma Qiao | 2025-08-10 | [None][infra] Waive test main 0808 (#6751) |
| 3360 | de472828b9 | Ziyi Xiong | 2025-08-09 | [TRTLLM-6637][feat] Resolve KV cache divergence issue (#6628) |
| 3361 | d643aef73c | Yilin Fan | 2025-08-08 | [Perf] Improve Llama4 performance for small max_seqlen cases (#6306) |
| 3362 | bcf5ec0c9a | Ye Zhang | 2025-08-09 | [None][feat] Core Metrics Implementation (#5785) |
| 3363 | 97787883c3 | Yibin Li | 2025-08-08 | [TRTLLM-6420][feat] add support for Eclairv2 model - cherry-pick changes and minor fix (#6493) |
| 3364 | d06675071e | dongfengy | 2025-08-08 | [None][fix] WAR GPT OSS on H20 with Triton MOE (#6721) |
| 3365 | cc0f4c87d4 | Fridah-nv | 2025-08-08 | [None][doc] Move AutoDeploy README.md to torch docs (#6528) |
| 3366 | efcb8f7f16 | Venky | 2025-08-08 | [TRTLLM-7025] [infra] Reorganize CODEOWNERS to rectify `examples` mapping (#6762) |
| 3367 | 90145cf557 | Mike Iovine | 2025-08-08 | [None][feat] Optimize CUDA graph memory usage for spec decode cases (#6718) |
| 3368 | d45236b253 | Wanli Jiang | 2025-08-08 | [TRTLLM-6308][feat] Support Aggregate mode for phi4-mm (#6184) |
| 3369 | b8f036f264 | Stefan Niebler | 2025-08-08 | [TRTLLM-6650][fix] Enhance CUDA graph + Beam search to correctly handle padding (#6665) |
| 3370 | e251f7c00b | Chuang Zhu | 2025-08-08 | [None][fix]revert kvcache transfer (#6709) |
| 3371 | ebdc43e69d | Zheng Duan | 2025-08-08 | [None][feat] move kv cache measure into transfer session (#6633) |
| 3372 | 32ad7f3c12 | Liao Lanyu | 2025-08-08 | [None][fix] Remove lock related typo in py_executor (#6653) |
| 3373 | 5f45227a93 | JunyiXu-nv | 2025-08-08 | [https://nvbugs/5437106][fix] Fix llama4 scout TRTLLM attn_backend (#6690) |
| 3374 | 9ff4e75f14 | Yuxian Qiu | 2025-08-08 | [None][refactor] Combine resmooth_to_fp8_e8m0 and transform_sf_into_required_layout (#6654) |
| 3375 | 294e0d3dab | Leslie Fang | 2025-08-08 | [https://nvbugs/5436461][infra] Adjust free_gpu_memory_fraction of test_eagle3 to prevent OOM on CI (#6631) |
| 3376 | d913955952 | Li Min | 2025-08-08 | [TRTLLM-6898][feat] make fused_moe_cute_dsl work on blackwell (#6616) |
| 3377 | 9687bb42b5 | Chang Liu | 2025-08-07 | [None][doc] Add doc for multimodal feature support matrix (#6619) |
| 3378 | b15d6fb145 | ruodil | 2025-08-08 | [None][test] fix yml condition error under qa folder (#6734) |
| 3379 | 064eb7a70f | 2ez4bz | 2025-08-07 | [TRTLLM-5252][fix] Propagate mapping to intermediate layers (#6611) |
| 3380 | aee828d98a | Enwei Zhu | 2025-08-08 | [TRTLLM-6854][feat] Enable guided decoding with disagg serving (#6704) |
| 3381 | 1cf669496a | zhanghaotong | 2025-08-08 | [None][fix] Fix unnecessary GPU synchronization in torch sampler caused by incorrect tensor reference (#6626) |
| 3382 | 2f2f5cc72c | NVJiangShao | 2025-08-08 | [TRTLLM-6744][feat] Remove input_sf swizzle for module WideEPMoE (#6231) |
| 3383 | 22f45a0e19 | ruodil | 2025-08-08 | [TRTLLM-5252][test] add for mistral_small_3.1_24b perf test (#6685) |
| 3384 | 88ced50ca7 | xinhe-nv | 2025-08-08 | [TRTQA-2920][fix] Add failed cases into waives.txt (#6719) |
| 3385 | efca359b66 | Daniel Cámpora | 2025-08-08 | [TRTLLM-6785][feat] BREAKING CHANGE Enable TRTLLM sampler by default (#6216) |
| 3386 | 82276167e6 | Iman Tabrizian | 2025-08-07 | [None][feat] Add NCCL Symmetric Integration for All Reduce (#4500) |
| 3387 | 980929e1a9 | Haohang Huang | 2025-08-07 | [https://nvbugs/5410687][fix] Hopper w4a8 groupwise MoE interleave (#6708) |
| 3388 | db8dc97b7b | Yuan Tong | 2025-08-08 | [None][fix] Migrate to new cuda binding package name (#6700) |
| 3389 | 4ecda91ecc | Andrew Chen | 2025-08-07 | [https://nvbugs/5423962][fix] Address broken links (#6531) |
| 3390 | 3b2dd40d50 | pcastonguay | 2025-08-07 | [None][chore] Remove py_executor from disagg gh team (#6716) |
| 3391 | e968f98b43 | Mike Iovine | 2025-08-07 | [None][feat] Clean up ngram auto mode, add max_concurrency to configs (#6676) |
| 3392 | 4055b764db | Raayan Dhar | 2025-08-07 | [None][fix] disagg ctx pp4 + gen pp4 integ test (#6489) |
| 3393 | 0223de0727 | Guoming Zhang | 2025-08-07 | [None][doc] Add deployment guide section for VDR task (#6669) |
| 3394 | 46357e7869 | Yiqing Yan | 2025-08-07 | [None][package] Pin cuda-python version to >=12,<13 (#6702) |
| 3395 | 3c44b44e45 | Emma Qiao | 2025-08-07 | [None][infra] Fix guardwords (#6711) |
| 3396 | 453a06e6ab | pcastonguay | 2025-08-07 | [TRTLLM-6881][feat] Include attention dp rank info with KV cache events (#6563) |
| 3397 | 1b9781e8e7 | Enwei Zhu | 2025-08-07 | [TRTLLM-6409][feat] Enable guided decoding with speculative decoding (part 1: two-model engine) (#6300) |
| 3398 | c23e8e7b05 | shaharmor98 | 2025-08-07 | [TRTLLM-6092][doc] Add LoRA feature usage doc (#6603) |
| 3399 | 8ec3b1de10 | peaceh-nv | 2025-08-07 | [None][feat] : Add FP8 context MLA support for SM120 (#6059) |
| 3400 | 0a467b00cc | xinhe-nv | 2025-08-07 | [https://nvbugs/5409414][fix] fix Not registered specs (#6660) |
| 3401 | 8207d5fd39 | hlu1 | 2025-08-07 | [None] [feat] Add model gpt-oss (#6645) |
| 3402 | 6c1f7d8b91 | ruodil | 2025-08-07 | [None][test] correct test-db context for perf yaml file (#6686) |
| 3403 | 85af62184b | amitz-nv | 2025-08-07 | [TRTLLM-6683][feat] Support LoRA reload CPU cache evicted adapter (#6510) |
| 3404 | 5fa1914cab | Yiqing Yan | 2025-08-07 | [None][chore] Bump version to 1.1.0rc0 (#6651) |
| 3405 | ee471df07c | Chuang Zhu | 2025-08-07 | [None][chore] optimize kv cache transfer for context TEP and  gen DEP (#6657) |
| 3406 | 3e41e6c077 | Yiqing Yan | 2025-08-07 | [TRTLLM-6892][infra] Run guardwords scan first in Release Check stage (#6659) |
| 3407 | 157ea77549 | YueWeng | 2025-08-07 | [https://nvbugs/5375966][chore] Unwaive test_disaggregated_deepseek_v3_lite_fp8_attention_dp_one (#6658) |
| 3408 | f7f46a5017 | Guoming Zhang | 2025-08-07 | doc: remove the outdated features which marked as Experimental (#5995) |
| 3409 | 2e90b0b550 | Pengbo Wang @ NVIDIA | 2025-08-07 | [None][fix] Explicitly add tiktoken as required by kimi k2 (#6663) |
| 3410 | 780d7507f9 | ruodil | 2025-08-07 | [None][test] remove trt backend cases in release perf test and move NIM cases to llm_perf_nim.yml (#6662) |
| 3411 | f30398470d | ruodil | 2025-08-07 | [None][chore] update readme for perf release test (#6664) |
| 3412 | 2a946859a7 | Yibin Li | 2025-08-06 | [None][fix] Upgrade dependencies version to avoid security vulnerability (#6506) |
| 3413 | 7e0158b583 | Izzy Putterman | 2025-08-06 | Qwen3: Fix eagle hidden states (#6199) |
| 3414 | a16ba6445c | chenfeiz0326 | 2025-08-06 | [None][doc] Create deployment guide for Llama4 Scout FP8 and NVFP4 (#6550) |
| 3415 | 3a71ddfe09 | Yuxian Qiu | 2025-08-06 | [TRTLLM-6859][doc] Add DeepSeek R1 deployment guide. (#6579) |
| 3416 | 5eae3184fa | Yan Chunwei | 2025-08-06 | [None][chore] add missing tests to test list (#6590) |
| 3417 | 1aed7511fe | Yechan Kim | 2025-08-06 | [https://nvbugs/5430124][fix] Mistral mixture_text_image test case fix (#6648) |
| 3418 | 13ecb4aced | Iman Tabrizian | 2025-08-06 | [https://nvbugs/5328160][fix] Unwaive disaggregated serving tests (#6644) |
| 3419 | 79fc2f48c0 | Pengyun Lin | 2025-08-06 | [None][chore] Enhance trtllm-serve example test (#6604) |
| 3420 | b7347ce7d1 | Yanchao Lu | 2025-08-06 | [https://nvbugs/5433581][fix] Revert deep_gemm installation workaround for SBSA (#6666) |
| 3421 | 98424f3186 | Yiqing Yan | 2025-08-06 | [TRTLLM-5633][infra] Change the TOT repo to default-llm-repo for merge waive list (#6605) |
| 3422 | 80f918cc22 | Hanjun Cho | 2025-08-06 | [None][feat] Add Qwen3 MoE support to TensorRT backend (#6470) |
| 3423 | 0ff8df95b7 | Zongfei Jing | 2025-08-06 | [https://nvbugs/5433581][fix] DeepGEMM installation on SBSA (#6588) |
| 3424 | 907c180eb2 | ruodil | 2025-08-06 | [None][test] align kv_frac in perf test with perflab and add more cases for 4 gpus GB200 (#6632) |
| 3425 | 43bd861ce1 | Iman Tabrizian | 2025-08-05 | Update allreduce benchmark for torch (#6271) |
| 3426 | 83ee91e17b | Netanel Haber | 2025-08-06 | [None][fix] Fix 6522 mpi.pkl5.intracomm.Request has wait not Wait (#6646) |
| 3427 | 3036d49071 | Guoming Zhang | 2025-08-06 | [None][doc] Unify the tech blogs naming. (#6649) |
| 3428 | 0bd99b5d6d | ruodil | 2025-08-06 | [TRTLLM-6764][test] add new feature cases in cluster(B200/GB200) and sanity test (#6650) |
| 3429 | 3170039e36 | jiahanc | 2025-08-05 | [None][doc] Add llama4 hybrid guide (#6640) |
| 3430 | da072277d1 | juney-nvidia | 2025-08-06 | [None][doc] Exposing the GPT OSS model support blog (#6647) |
| 3431 | 13e0214fe0 | JunyiXu-nv | 2025-08-06 | [TRTLLM-6263][feat] Enable fp8 SwiGLU to minimize host overhead (#6540) |
| 3432 | 9a01934dbf | brb-nv | 2025-08-05 | [None][feat] Switch to internal version of MMProjector in Gemma3 (#6572) |
| 3433 | 3ff4f503ad | yunruis | 2025-08-06 | [None][opt] ADP schedule balance optimization (#6061) |
| 3434 | 19b7524ff6 | Ransiki | 2025-08-06 | [None][feat] Add vLLM KV Pool support for XQA kernel (#6013) |
| 3435 | c17f4984e2 | Yechan Kim | 2025-08-06 | [None][feat] Refactor Llava-Next (#6478) |
| 3436 | f92397493e | Venky | 2025-08-05 | [TRTLLM-5500][infra] Update CODEOWNERS with new ownership rules for additional paths (#6564) |
| 3437 | 6da95f29a9 | Aurelien Chartier | 2025-08-05 | [None][feat] Add support for fused gate_up_proj scales for FP8 blockwise (#6496) |
| 3438 | 46df8712c8 | Wanli Jiang | 2025-08-06 | [https://nvbugs/5355007][fix] Set `enable_chunked_context` as True by default in trtllm bench (#6582) |
| 3439 | 1ebceb790d | ixlmar | 2025-08-05 | [TRTLLM-5508][feat] check input tokens + improve error handling (#5170) |
| 3440 | 6af1514dc3 | Farshad Ghodsian | 2025-08-05 | [None][doc] Adding GPT-OSS Deployment Guide documentation (#6637) |
| 3441 | dcbfa7e509 | liji-nv | 2025-08-05 | [https://nvbugs/5252313][fix] Fix torch compile + MTP (#6554) |
| 3442 | 61da2daeb4 | Venky | 2025-08-05 | [TRTLLM-6761][refactor] Replace LogitBiasLogitsProcessor with embedding bias tensor system (#6464) |
| 3443 | 6a9b4b11be | Zhanrui Sun | 2025-08-05 | [https://nvbugs/5433581][infra] Temporarily disable Docker Image use wheel from build stage (#6630) |
| 3444 | 78a75c2990 | Emma Qiao | 2025-08-05 | [None][Infra] - Split gb200 stages for each test (#6594) |
| 3445 | c32584125e | xinhe-nv | 2025-08-05 | [TRTQA-2920][fix] Add failed cases into waives.txt (#6600) |
| 3446 | c289880afb | Pengbo Wang @ NVIDIA | 2025-08-05 | [None][fix] fix kimi k2 serving and add test for Kimi-K2 (#6589) |
| 3447 | 08ed9d7305 | Ivy Zhang | 2025-08-05 | [None][doc] add introduction doc on qa test (#6535) |
| 3448 | d101a6cebc | Ivy Zhang | 2025-08-05 | [https://nvbugs/5410279][test] resubmit timeout refactor (#6337) |
| 3449 | 7cbe30e17d | Zhanrui Sun | 2025-08-05 | [TRTLLM-6893][infra] fix Build Docker Image tag issue (#6555) |
| 3450 | dc84695520 | amitz-nv | 2025-08-05 | [TRTLLM-6826][feat] Allow sending more than 2GiB through MPI by using mpi4py.util.pkl5 (#6522) |
| 3451 | ed801ff74b | danielafrimi | 2025-08-05 | [None][fix] Remove expand  configuration from mamba2 mixer (#6521) |
| 3452 | c9eebcb454 | Haohang Huang | 2025-08-05 | [TRTLLM-6674][feat] (Breaking Change) Hopper SWA non-cyclic kernels + KV reuse + Spec Dec (#6379) |
| 3453 | 4d040b50b7 | Chuang Zhu | 2025-08-05 | [None][chore] ucx establish connection with zmq (#6090) |
| 3454 | 164acfa31e | Leslie Fang | 2025-08-05 | [None][infra] Skip test_eagle3 test with device memory check (#6617) |
| 3455 | 7625845365 | ruodil | 2025-08-05 | test: add README_release_test.md for perf test (#6443) |
| 3456 | db51ab11a9 | Guoming Zhang | 2025-08-05 | [TRTLLM-5990][doc] trtllm-serve doc improvement. (#5220) |
| 3457 | d53cc2374b | Yanchao Lu | 2025-08-05 | [https://nvbugs/5433581][infra] Update install docs and CI script for SBSA deep_gemm workaround (#6607) |
| 3458 | a178cea324 | xinhe-nv | 2025-08-05 | [TRTLLM-6856][feat] add disaggregated serving tests to QA list (#6536) |
| 3459 | fe3d607c4b | xinhe-nv | 2025-08-05 | [TRTQA-2920][fix] Add failed cases into waives.txt (#6581) |
| 3460 | 899b74c357 | Enwei Zhu | 2025-08-05 | [None][doc] Fix blog4 typo (#6612) |
| 3461 | 6a3a921284 | kris1025 | 2025-08-05 | [TRTLLM-6685][feat] Add speculative metrics for trt llm bench (#6476) |
| 3462 | 6135f75f87 | brb-nv | 2025-08-04 | [None][chore] Update Gemma3 closeness check to mitigate flakiness (#6591) |
| 3463 | 13cc1c4878 | Olya Kozlova | 2025-08-04 | [TRTLLM-5271][feat] best_of/n for pytorch workflow (#5997) |
| 3464 | f3651adea8 | Ivy Zhang | 2025-08-04 | [None][test] update invalid test name (#6596) |
| 3465 | 5d8a5a0cb8 | Emma Qiao | 2025-08-04 | [None][Infra]Waive failed case in post-merge on main (#6602) |
| 3466 | a4e518de51 | Yiteng Niu | 2025-08-04 | [TRTLLM-6364] [fix] Update PR title regex to allow optional spaces between ticket and type (#6598) |
| 3467 | 87e4e9f468 | brb-nv | 2025-08-04 | [None][chore] Add unit test for Gemma3 lora (#6560) |
| 3468 | 3916dbd98b | Yiqing Yan | 2025-08-04 | [None][chore] Bump version to 1.0.0rc6 (#6597) |
| 3469 | a15e33351d | Pengyun Lin | 2025-08-04 | [None][fix] Revert commit 48ddc3d & add test for disagg server with different max_num_tokens (#6259) |
| 3470 | 8c82ee2803 | Bruce-Lee-LY | 2025-08-04 | [fix] xqa precision for fp16/bf16 kv cache (#6573) |
| 3471 | a54972e463 | xinhe-nv | 2025-08-04 | [None][fix] remove closed bugs (#6576) |
| 3472 | a2f271c8e0 | Yuan Tong | 2025-08-04 | [TRTLLM-4406][feat] LLM sleep & wakeup Part 1: virtual device memory (#5034) |
| 3473 | b9fe0fa7ec | Leslie Fang | 2025-08-04 | [None][infra] Enable test of chunked prefill with logit post processor (#6483) |
| 3474 | a60190836c | Leslie Fang | 2025-08-04 | [None][infra] Enable accuracy test for eagle3 and chunked prefill (#6386) |
| 3475 | 4763e94156 | Yiqing Yan | 2025-08-04 | [TRTLLM-5563][infra] Move test_rerun.py to script folder (#6571) |
| 3476 | 6459725bf9 | ruodil | 2025-08-04 | test: move ministral_8b_fp8 to fp8_specific gpu list(exclude Ampere) (#6533) |
| 3477 | 59d91b8b94 | Zhenhua Wang | 2025-08-04 | [None][chore] add online help to build_wheel.py and fix a doc link (#6391) |
| 3478 | 2279cec4ce | Yiteng Niu | 2025-08-04 | [https://nvbugs/5430932][infra] update namelist (#6585) |
| 3479 | 7bf0a48899 | Yiteng Niu | 2025-08-04 | [None][infra] update namelist (#6465) |
| 3480 | 18d1941083 | Zac Patel | 2025-07-30 | [doc] Update perf_overview.md for release 0.21 (#6270) |
| 3481 | 03430ed379 | Perkz Zheng | 2025-07-30 | [https://nvbugspro.nvidia.com/bug/5415268] fix illegal smem access with chunked attention (#6401) |
| 3482 | 5913282e17 | QI JUN | 2025-07-29 | doc: update release notes (#6438) |
| 3483 | 5eefdf2c75 | Ivy Zhang | 2025-07-29 | tests: Add llama4 functional cases (#6392) |
| 3484 | e1eca33dfc | QI JUN | 2025-07-28 | doc: update release notes (#6324) |
| 3485 | 3f47117870 | QI JUN | 2025-07-22 | doc: update known issues (#6247) |
| 3486 | 8d82ccca63 | ruodil | 2025-08-04 | test: modify max_lora_rank of phi4_multimodal to 320 (#6474) |
| 3487 | ee6ab5be96 | Yechan Kim | 2025-08-04 | chore: add EXAONE4 accuracy test (#6397) |
| 3488 | df90202b51 | Jinyang Yuan | 2025-08-04 | [fix] Fix DeepSeek w4a8 weight loading (#6498) |
| 3489 | 7547a7d0a2 | Ivy Zhang | 2025-08-04 | [TRTLLM-6473][test] add speculative decoding and ep load balance cases into QA test list (#6436) |
| 3490 | 6edaa23c1c | Jhao-Ting Chen | 2025-08-03 | [None][feat] Multi-block mode for Hopper spec dec XQA kernel (#4416) |
| 3491 | 542f552d0b | Chuang Zhu | 2025-08-04 | use cudaSetDevice to create context ,fix nvbug 5394497 (#6403) |
| 3492 | 3f7abf87bc | Yiqing Yan | 2025-08-03 | [TRTLLM-6224][infra] Upgrade dependencies to DLFW 25.06 and CUDA 12.9.1 (#5678) |
| 3493 | 4da5cfc511 | Jhao-Ting Chen | 2025-08-02 | [None][infra] add eagle3 one model accuracy tests (#6264) |
| 3494 | 918fedf952 | Robin Kobus | 2025-08-02 | [None][refactor] Simplify finish reasons handling in DecoderState (#6524) |
| 3495 | 67a3fd858b | Shunkangz | 2025-08-02 | [None][feat] Add support of scheduling attention dp request (#6246) |
| 3496 | 31802de0b0 | Richard Huo | 2025-08-01 | [None][fix] Serialize the window_size in the kv event (#6526) |
| 3497 | 6f34f3489b | Lizhi Zhou | 2025-08-02 | [TRTLLM-6357][test] Add accuracy tests for Qwen3 (#6177) |
| 3498 | 263c6c0ad0 | xinhe-nv | 2025-08-02 | test: skip post blackwell (#6357) |
| 3499 | 5247df6ae2 | Lucas Liebenwein | 2025-08-01 | [AutoDeploy] merge feat/ad-2025-07-22 (#6520) |
| 3500 | 16febefee0 | Emma Qiao | 2025-08-01 | [None][Infra] - Skip failed tests in post-merge (#6558) |
