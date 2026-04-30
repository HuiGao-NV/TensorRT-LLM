# Commit Section 11

Commits 5001 to 5500 (total: 5612)

| # | Commit Hash | Author | Date | Message |
|---|------------|--------|------|--------|
| 5001 | e6c14ca97a | yuxianq | 2025-04-26 | fix: Detect pmix and raise error when mpirun is not used. (#3858) |
| 5002 | 362a8272f8 | milesial | 2025-04-25 | feat: llama4 input processor (#3383) |
| 5003 | d7472231f9 | Kaiyu Xie | 2025-04-26 | TRTLLM-4875 feat: Add version switcher to doc (#3846) |
| 5004 | 7ff9fd345c | Dom Brown | 2025-04-25 | Test: Split C++ unit tests for CI granularity (#3868) |
| 5005 | 6ac1a54f57 | QI JUN | 2025-04-25 | chore: update pytorch only change file list (#3873) |
| 5006 | ecd621fb0a | qixiang-99 | 2025-04-25 | feat: Add head size 72 support for QKV Preprocessing kernel (#3743) |
| 5007 | 5b9897a8cd | sugunav14 | 2025-04-25 | fix: [AutoDeploy] update hf loading for e_score_correction_bias (#3847) |
| 5008 | 68e774ff9e | Mike Iovine | 2025-04-25 | [chore] Add Llama 4 Maverick to quickstart README (#3848) |
| 5009 | 238fefc659 | Yiqing Yan | 2025-04-25 | [infra] Waive L0 tests (#3853) |
| 5010 | 16535991b2 | dongxuy04 | 2025-04-25 | feat: Add MNNVL MoE A2A support (#3504) |
| 5011 | 57944206ba | Yuan Tong | 2025-04-25 | feat: return logits in PyTorch flow (#3221) |
| 5012 | 991939a0f4 | QI JUN | 2025-04-24 | chore: increase A30 for cpp test (#3811) |
| 5013 | d72add1794 | hlu1 | 2025-04-24 | [Deepseek] Pass hidden_states_fp4 to shared_experts (#3819) |
| 5014 | ccd1eb67ec | rakib-hasan | 2025-04-24 | Adding local paths to the datasets to make them loadable in offline mode (#3750) |
| 5015 | f95dbbb6cb | Luis Vega | 2025-04-24 | added nemotron-h to supported models (#3663) |
| 5016 | 7420ddc3d0 | HuiGao-NV | 2025-04-24 | fix: fix lora case failure (#3838) |
| 5017 | 3fc2a16920 | WeiHaocheng | 2025-04-24 | feat(part 2): Enhance the integrated robustness of scaffolding with __init__.py #3305 (#3731) |
| 5018 | ae34d60108 | Zhanrui Sun | 2025-04-24 | chore: bump version to 0.20.0rc1 (#3834) |
| 5019 | 1d5178814b | Shi Xiaowei | 2025-04-24 | Fix: Revert commit 25f9669 (#3832) |
| 5020 | 3f67a4c9d8 | qixiang-99 | 2025-04-23 | fix: Set default prompts and media for multimodal quickstart example (#3792) |
| 5021 | 777c40e5fa | Enwei Zhu | 2025-04-24 | [https://nvbugspro.nvidia.com/bug/5238599][fix] Normalize example path in accuracy tests |
| 5022 | 90b708f851 | Enwei Zhu | 2025-04-24 | [https://nvbugspro.nvidia.com/bug/5238602][fix] Package lm_eval configuration files |
| 5023 | 476d7003f8 | xinhe-nv | 2025-04-24 | test: [CI] Add failed cases into waives.txt (#3777) |
| 5024 | cd2bcdc1a9 | hlu1 | 2025-04-23 | Fix create_weights in attention (#3692) |
| 5025 | bc5fe7800d | Mike Iovine | 2025-04-23 | [chore] Fix KV cache block reuse flag name in quickstart_advanced (#3781) |
| 5026 | d0d19e81ca | QI JUN | 2025-04-23 | chore: fix some invalid paths of contrib models (#3818) |
| 5027 | dfbcb543ce | Kaiyu Xie | 2025-04-24 | doc: fix path after examples migration (#3814) |
| 5028 | 635dcdcb9e | QI JUN | 2025-04-23 | chore: reorganize some unit tests of PyTorch (#3780) |
| 5029 | 0c6c8eaffd | Julien Debache | 2025-04-23 | fix: 5197419 and removed unused runtime kernels (#3631) |
| 5030 | 1299f27c74 | Daniel Cámpora | 2025-04-23 |  fix: Fix C++ decoder synchronization in PyTorch (#3106) |
| 5031 | 0bc520f15e | Mike Iovine | 2025-04-23 | fix: Limit llama4 context length to 8k (#3778) |
| 5032 | 49262a62a5 | shaharmor98 | 2025-04-23 | add passing E2E LoRA flow (#3788) |
| 5033 | a51b3cf7a6 | Enwei Zhu | 2025-04-23 | [TRTLLM-4763][test] Accuracy test improvement (Part 3.6): Deprecate mmlu_llmapi.py (#3802) |
| 5034 | bfc4e55ded | Zhanrui Sun | 2025-04-23 | infra: [TRTLLM-4417]Support auto trigger special test stage for special file change (#3478) |
| 5035 | 8f2b2eaf83 | Enwei Zhu | 2025-04-23 | test: Add DeepSeek-V3-Lite GSM8K tests (#3771) |
| 5036 | bc1c4ddcb5 | Fanrong Li | 2025-04-23 | fix: remove the unnecessary metadata changes in mtp. (#3787) |
| 5037 | 25f96697ad | Shi Xiaowei | 2025-04-23 | fix: Intercept the error of multi-rank binding to a single card (#3525) |
| 5038 | b82d72bc37 | xinhe-nv | 2025-04-23 | update waive list (#3696) |
| 5039 | 11d35656bf | Yechan Kim | 2025-04-23 | fix: nvbugs/5234029 fix Qwen2.5-VL image test (#3726) |
| 5040 | 80d8fdefd6 | xinhe-nv | 2025-04-23 | add test_mistral_large_hidden_vocab_size tests (#3716) |
| 5041 | 1e5af736ea | Zongfei Jing | 2025-04-23 | Add smart router for moe (#3641) |
| 5042 | cc161dd83d | Yiqing Yan | 2025-04-23 | Waive L0 tests (#3784) |
| 5043 | 5fff8f0935 | shaharmor98 | 2025-04-23 | Add running E2E LoRA flow (#3648) |
| 5044 | c4d86b267c | bhsueh_NV | 2025-04-23 | chore: add pull request template (#3760) |
| 5045 | 0324a7389d | Perkz Zheng | 2025-04-23 | add QMMA-based MLA kernels (#3752) |
| 5046 | 44bff85e08 | William Tambellini | 2025-04-22 | Fix double link to fp8_blockscale_gemm_src (#3707) |
| 5047 | 257abfbc51 | QI JUN | 2025-04-22 | move pytorch tests of LLM API into separate test files (#3745) |
| 5048 | b16a127026 | rakib-hasan | 2025-04-22 | fixing the metric fmeasure access (#3774) |
| 5049 | 4728256bb6 | Alessio Netti | 2025-04-22 | chore: Move cv2 import inside load_video() function (#3768) |
| 5050 | 06b914e0f9 | Lucas Liebenwein | 2025-04-22 | feat: [AutoDeploy] generalizing cudagraph to multiple dynamic inputs (#3589) |
| 5051 | 442386d302 | Emma Qiao | 2025-04-23 | infra: Add test stages for sm120 (#3533) |
| 5052 | ba4131f176 | Xianjie Qiao | 2025-04-23 | Add log_level for disaggregated_mpi_worker (#3765) |
| 5053 | 7eee9a9d28 | Zongfei Jing | 2025-04-22 | doc: Update doc for Deepseek min latency (#3717) |
| 5054 | 0ae7017342 | Yukun He | 2025-04-22 | Unify two versions of AllReduce custom op (#3032) |
| 5055 | b87f26ee2a | bhsueh_NV | 2025-04-22 | chore: remove useless allgather (#3751) |
| 5056 | 47d2f16bb8 | Ivy Zhang | 2025-04-22 | waive gemma on L20 (#3767) |
| 5057 | 9223000765 | ruodil | 2025-04-22 | waive failed case in perf test, change default max_batch_size to 512 and write config.json to output log (#3657) |
| 5058 | 353699a3b3 | Enwei Zhu | 2025-04-22 | fix: fnmatch usage in modeling_utils.py (#3754) |
| 5059 | 8340657ae4 | Robin Kobus | 2025-04-22 | refactor: Introduce DecoderOutputBuffers per batch (#3506) |
| 5060 | ba216341f4 | xinhe-nv | 2025-04-22 | update waive list (#3683) |
| 5061 | 98966cb45e | Yi Zhang | 2025-04-22 | test: Unwaive Llama 3.1 with torch compile test (#3475) |
| 5062 | a32389b4cd | Kaiyu Xie | 2025-04-22 | fix: Remove unnecessary max call (#3574) |
| 5063 | 74c13ea84f | rakib-hasan | 2025-04-21 | datasets API change : datasets.load_metric => evaluate.load (#3741) |
| 5064 | 3fa19ffa4e | Enwei Zhu | 2025-04-22 | test [TRTLLM-4477,TRTLLM-4481]: Accuracy test improvement (Part 3.5): Support GSM8K and GPQA (#3483) |
| 5065 | 0c07d4dc21 | bhsueh_NV | 2025-04-22 | Fix/executor bugs (#3681) |
| 5066 | 943f3ff8f6 | Kaiyu Xie | 2025-04-22 | Revert "Report number of context tokens in one iteration (#3691)" (#3740) |
| 5067 | 231b39015c | Yan Chunwei | 2025-04-21 | unwaive multi_node test (#3715) |
| 5068 | d87b009d8d | Barry Kang | 2025-04-21 | Fix ModelOpt Mixtral AWQ OOM (#3714) |
| 5069 | ae48abefc1 | Zheng Duan | 2025-04-21 | bind block key and hasher (#3712) |
| 5070 | af04b6f6aa | Iman Tabrizian | 2025-04-21 | bug: Fix hang bug when context server doesn't have enough capacity for KV Cache (#3095) |
| 5071 | 852dd0c1be | Stanley Sun | 2025-04-21 | test: add llama3.2 ptp test case (#3363) |
| 5072 | bc2b01d1dd | Jinyang Yuan | 2025-04-21 | chore: update FMHA cubin files (#3680) |
| 5073 | 2672f13d77 | Zhenhuan Chen | 2025-04-21 | test: fix cublas_scaled_mm with aligned workspace size (#3600) |
| 5074 | eeb605abd6 | katec846 | 2025-04-20 | feat: Offloading Multimodal embedding table to CPU in Chunked Prefill Mode (#3380) |
| 5075 | faef37782a | yuxianq | 2025-04-21 | fix: Remove ParallelConfig. (#3678) |
| 5076 | e0446a4dc0 | HuiGao-NV | 2025-04-21 | Report number of context tokens in one iteration (#3691) |
| 5077 | 591f3d2be8 | yuxianq | 2025-04-21 | fix: Support TLLM_OVERRIDE_LAYER_NUM for llama4. (#3679) |
| 5078 | a51f7559a3 | liji-nv | 2025-04-21 | fix: update test_user_buffers_mm_add_prologue atol (#3711) (#3713) |
| 5079 | 6f7f262779 | Yiqing Yan | 2025-04-21 | Waive L0 tests (#3709) |
| 5080 | 31624b079a | hlu1 | 2025-04-20 | feat: [Deepseek] Add trtllm-gen MOE FP4 MOE backend (#3387) |
| 5081 | 48db263d9a | Emma Qiao | 2025-04-20 | infra: Add test list name check (#3097) |
| 5082 | f7c2eb4fa2 | Naveassaf | 2025-04-20 | Update Nemotron Super and Ultra in Supported Models and add an example (#3632) |
| 5083 | 17eba98445 | hlu1 | 2025-04-19 | Refactor Deepseek tp_size calculation (#3695) |
| 5084 | d51ae53940 | QI JUN | 2025-04-19 | move the reset models into `examples/models/core` directory (#3555) |
| 5085 | c35d2a7532 | brb-nv | 2025-04-19 | test: Get Eagle tests working (#3593) |
| 5086 | e70961f541 | nv-guomingz | 2025-04-19 | test:update waives.txt for nvbug 5219532 (#3672) |
| 5087 | 5346f53250 | yuxianq | 2025-04-19 | feat: Introduce feature properties for attention backend. (#3659) |
| 5088 | 61ee983488 | Iman Tabrizian | 2025-04-18 | fix: Fix disaggregated load balance test (#3689) |
| 5089 | c861b6cf17 | hlu1 | 2025-04-18 | Clean up modeling_deepseek.py (#3640) |
| 5090 | a2f190f306 | Iman Tabrizian | 2025-04-18 | chore: Waive disaggregated load balance (#3687) |
| 5091 | 5460d18b10 | Yechan Kim | 2025-04-19 | feat: trtllm-serve multimodal support (#3590) |
| 5092 | ce8329646f | mayani-nv | 2025-04-18 | Update run.py for draft_target_model (#3615) |
| 5093 | ae5671644a | pcastonguay | 2025-04-18 | feat: Disaggregated router class (#3584) |
| 5094 | b9fce42717 | QI JUN | 2025-04-18 | enable test_ptp_quickstart_advanced_mixed_precision (#3667) |
| 5095 | bce7ea8c38 | Zheng Duan | 2025-04-18 | test: add kv cache event tests for disagg workers  (#3602) |
| 5096 | 2a09826ec4 | Yan Chunwei | 2025-04-18 | fix hmac in remote mpi session (#3649) |
| 5097 | d3608d6818 | HuiGao-NV | 2025-04-18 | Remove dummy forward path (#3669) |
| 5098 | dbd9a83b0d | Dom Brown | 2025-04-18 | feat: Integrate GPUDirect Storage (GDS) into Executor API (#3582) |
| 5099 | 90a28b917f | Zheyu Fu | 2025-04-18 | feat: Add Dynasor-CoT in scaffolding examples. (#3501) |
| 5100 | 4fedf0be5c | Erin | 2025-04-18 | unwaive test for nvbug_5150466 (#3552) |
| 5101 | 0b0e6d8a0a | Yuan Tong | 2025-04-18 | refactor: Clean up CMakeLists.txt (#3479) |
| 5102 | 2f48985b9c | Emma Qiao | 2025-04-18 | infra: Add step to generate new duration file (#3298) |
| 5103 | 88cff61fa1 | peaceh-nv | 2025-04-18 | chore : Split more tests out of gpt tests (#3524) |
| 5104 | b71a0f76b4 | dongfengy | 2025-04-17 | test: Add llama 4 to ci (#3520) |
| 5105 | fc88d67675 | Iman Tabrizian | 2025-04-17 | chore: Refactor test_disaggregated.py (#3154) |
| 5106 | b8818b45be | Chang Liu | 2025-04-17 | fix: llama4: address couple of issues in llama4 attention module (#3491) |
| 5107 | 1b2b112d44 | Jackch-NV | 2025-04-18 | fix sage attention headsize check error in bertAttentionPlugin.cpp (#3660) |
| 5108 | ff3b741045 | rakib-hasan | 2025-04-17 | feat: adding multimodal (only image for now) support in trtllm-bench (#3490) |
| 5109 | 26ebd95302 | QI JUN | 2025-04-17 | chore: update multi gpu trigger file list (#3665) |
| 5110 | 91660939fd | QI JUN | 2025-04-17 | tests: waive test_llm_multi_node (#3664) |
| 5111 | 5a6cb2b985 | Frank | 2025-04-17 | fix: Correct reporting of text dtype for Llama 4 (#3494) |
| 5112 | 83b36ebecd | Yukun He | 2025-04-17 | Fix fused_moe fallback issue. (#3652) |
| 5113 | b9b1c1368c | yuxianq | 2025-04-17 | feat: Support unfused rope in MLA. (#3610) |
| 5114 | ad19ca3cbf | Ivy Zhang | 2025-04-17 | remove benchmark test list (#3644) |
| 5115 | 3c52ac098f | Netanel Haber | 2025-04-17 | feat: allocate minimal blocks per window size (#3028) |
| 5116 | 1c6f3debbb | Yiqing Yan | 2025-04-17 | Waive L0 tests (#3651) |
| 5117 | b82a4e8d01 | xinhe-nv | 2025-04-17 | test: [CI] Add failed cases into waives.txt (#3627) |
| 5118 | e4476bf521 | Tao Li @ NVIDIA | 2025-04-17 | update fp8 doc (#3647) (#3650) |
| 5119 | 0f084d9566 | danielafrimi | 2025-04-17 | added loraOp into lora layer + test for mlp and comparison  to lora plugin (#3455) |
| 5120 | 239fe0ff26 | yuxianq | 2025-04-17 | chore: Use ellipsis as default value to detect whether residual argument is provided (#3626) |
| 5121 | a06bff5052 | Luis Vega | 2025-04-16 | Fix rotary_emb param in NemotronH attention (#3646) |
| 5122 | 950cadf2bd | Void | 2025-04-17 | add support for smaller hidden_dim (#3609) |
| 5123 | b2fb0fe843 | Ivy Zhang | 2025-04-17 | test: add quickstart test for nemotron-ultra (#3596) |
| 5124 | 5e2ebebe76 | ruodil | 2025-04-17 | tests: change qa perf test to trtllm-bench (#3189) |
| 5125 | f4ddc304f2 | Chuang Zhu | 2025-04-17 | disable ib for ucx test (#3613) |
| 5126 | 57cafe7f9b | QI JUN | 2025-04-16 | waive test_fp8_scaled_mm (#3637) |
| 5127 | 0bda1f9780 | Luis Vega | 2025-04-16 | feat: Nemotron-H model support (#3430) |
| 5128 | 41a6c98544 | Mike Iovine | 2025-04-16 | Support CUDA graphs for EAGLE3 (#3176) |
| 5129 | b6bae33453 | hlu1 | 2025-04-16 | Clean up linear.py, mlp.py, gated_mlp.py (#3553) |
| 5130 | 351808efeb | Yibin Li | 2025-04-16 | fix: Use hmac authentication for pickle encryption (#3384) |
| 5131 | fac1a905e9 | QI JUN | 2025-04-16 | waive test_llm_multi_node_with_postproc (#3628) |
| 5132 | b3e6723dbc | Olya Kozlova | 2025-04-16 | feat: Adding FP8 BMM from Codegen (#3541) |
| 5133 | ca88674210 | Yiteng Niu | 2025-04-16 | update user list (#3614) |
| 5134 | 2e0cd7922e | Gabriel Wu | 2025-04-16 | fix: add SM90 guard for FP8 Blockscale GEMM (#3575) |
| 5135 | fd8ded2b2b | yuxianq | 2025-04-16 | feat: Support cos_sin_cache in all cases. (#3517) |
| 5136 | ab29348db2 | QI JUN | 2025-04-15 | waive test_llm_phi_quantization_1gpu (#3603) |
| 5137 | efabf6b443 | Jinyang Yuan | 2025-04-16 | chore: Add comments to modifications that fix TP size of DeepSeek-V3/R1 when using more than 16 GPUs (#3572) |
| 5138 | 9d88ee3e45 | Zhanrui Sun | 2025-04-16 | chore: bump version to 0.20.0rc0 (#3561) |
| 5139 | ccd73c71a5 | narutolhy | 2025-04-15 | feat: Add stream generation task scaffolding examples (#3527) |
| 5140 | 409c294c4e | Yan Chunwei | 2025-04-16 | fix trtllm-bench mgmn (#3563) |
| 5141 | 63f3fba679 | Yan Chunwei | 2025-04-16 | waive test_llm_multi_node_pytorch (#3592) |
| 5142 | 44da0e8d60 | Enwei Zhu | 2025-04-16 | fix: LLM API _hf_model_dir for non-cached case (#3562) |
| 5143 | 41ce5440fe | Daniel Cámpora | 2025-04-16 | chore: Mass integration of release/0.18 (#3421) |
| 5144 | da47d5f27e | xiweny | 2025-04-16 | fix: nvbugs/5075538: fix cross attention mask when decoder input len > 1 (#3585) |
| 5145 | f5f68ded26 | Kaiyu Xie | 2025-04-16 | Minor fixes for documents (#3577) |
| 5146 | fffb403125 | Robin Kobus | 2025-04-15 | fix: disable KV cache reuse if using attention sink (#3021) |
| 5147 | 1899e71364 | Pengyun Lin | 2025-04-16 | doc: add genai-perf benchmark & slurm multi-node for trtllm-serve doc (#3407) |
| 5148 | e037d3e99b | Kaiyu Xie | 2025-04-15 | chore: Unify Python NVTX call (#3450) |
| 5149 | 258ae9c58c | Kaiyu Xie | 2025-04-15 | Revert "infra: move nvrtc_wrapper to conan (#3282)" (#3573) |
| 5150 | d35db254e2 | HuiGao-NV | 2025-04-15 | test: Enable 4 multi-gpu test cases for deepseek (#3569) |
| 5151 | c27e130be0 | Yan Chunwei | 2025-04-15 | unwaive test (#3559) |
| 5152 | 1d3b98b920 | jiahanc | 2025-04-15 | perf: Optimize quantization kernels used in DeepSeek on Hopper (#3466) |
| 5153 | 5cfa927132 | xinhe-nv | 2025-04-15 | update waive list (#3503) |
| 5154 | 3aa37e6b72 | bhsueh_NV | 2025-04-15 | fix bug (#3570) |
| 5155 | d4c0423cdb | Yuan Tong | 2025-04-15 | refactor: collect executor and decoder states into dataclass (#3234) |
| 5156 | b7a38feb14 | Robin Kobus | 2025-04-15 | chore: Clean up cpp runtime (#3537) |
| 5157 | ede7058544 | shaharmor98 | 2025-04-15 | Feat/ Integrate peftCacheManager in PyExecutor creation (#3372) |
| 5158 | 5881a65374 | hlu1 | 2025-04-14 | Fix test_fp4_quantize_gemm_torch (#3551) |
| 5159 | 668a0335e4 | Yuan Tong | 2025-04-15 | fix: Proper error bubbling for PyExecutor (#3321) |
| 5160 | 0e152910f5 | xinhe-nv | 2025-04-15 | update waive list (#3498) |
| 5161 | cfc6f242dd | Yukun He | 2025-04-15 | Chore: Remove profile test. (#3565) |
| 5162 | 0305942808 | Jinyang Yuan | 2025-04-15 | chore: Modifications that should have been included but were mistakenly overwritten in PR #3467 (#3557) |
| 5163 | 39bdb1fe1c | nv-guomingz | 2025-04-15 | docs:update llm api examples and customizations sections' links. (#3566) |
| 5164 | 0e7e949feb | yuxianq | 2025-04-15 | refactor: Split llama4 model from llama model. (#3530) |
| 5165 | e1e068d4f3 | tburt-nv | 2025-04-15 | fix local user (#3550) |
| 5166 | 5eae397b3b | Bo Li | 2025-04-15 | doc: Update instructions to enable FP8 MLA for Deepseek. (#3488) |
| 5167 | b0cb963199 | Zheng Duan | 2025-04-15 | test: torch-flow conditional disagg test (#3410) |
| 5168 | 175adb94ab | Jinyang Yuan | 2025-04-15 | chore: Log memory sizes of weights and activations separately (#3467) |
| 5169 | b32ae7ac92 | nv-guomingz | 2025-04-15 | test:add fp8_kv_cache functionality test case. (#3457) |
| 5170 | 112f716155 | QI JUN | 2025-04-15 | chore: move all distributed related codes into _torch.distributed directory (#3511) |
| 5171 | 098ca7f68c | brb-nv | 2025-04-14 | test: Fix breaking Phi3 multimodal tests (#3544) |
| 5172 | bad55e99bb | Iman Tabrizian | 2025-04-14 | test: Add MTP + overlap + Attention DP disaggregated test (#3542) |
| 5173 | 6cdfc54883 | Pamela Peng | 2025-04-14 | feat: Add FP8 support for SM 120 (#3248) |
| 5174 | c0dd6cbce0 | tburt-nv | 2025-04-15 | infra: move nvrtc_wrapper to conan (#3282) |
| 5175 | 8cf2785bc6 | Aurelien Chartier | 2025-04-14 | chore: unify pp_layers helpers (#3429) |
| 5176 | 01cb3ccb04 | Chang Liu | 2025-04-14 | use global expert idx to load expert weights (#3386) |
| 5177 | 1902d73eb5 | Chang Liu | 2025-04-14 | fix: llama4: add an option `apply_router_weight_on_input` for in FusedMoE (#3492) |
| 5178 | b286b51118 | Kaiyu Xie | 2025-04-14 | feat: Support torch profiler (#3470) |
| 5179 | 5985d362a9 | Yuan Tong | 2025-04-14 | fix: install RTC headers with linking when using --linking_install_binary (#3484) |
| 5180 | 714ff3eedd | Zhanrui Sun | 2025-04-14 | chore: bump version to 0.19.0rc0 (#3535) |
| 5181 | f58d4698c8 | Robin Kobus | 2025-04-14 | chore: Clean up cpp runtime (#3505) |
| 5182 | ee4ce0379d | Zhanrui Sun | 2025-04-14 | chore: bump version to 0.19.0rc0 (#3514) |
| 5183 | 170bc22139 | Ivy Zhang | 2025-04-14 | fix test name (#3534) |
| 5184 | c0493523d0 | Zhenhuan Chen | 2025-04-14 | infra: add more codespell ignore words, prevent ans->and (#3513) |
| 5185 | f99be2726f | Kaiyu Xie | 2025-04-14 | doc: Add example section for multi-node DeepSeek R1 benchmark on GB200 (#3519) |
| 5186 | 2e669133c2 | Chuang Zhu | 2025-04-14 | disagg perf tune doc (#3516) |
| 5187 | b1d8495b3d | xinhe-nv | 2025-04-14 | update waive list (#3510) |
| 5188 | 2fb1d65d43 | dongjiyingdjy | 2025-04-14 | fix: fix max_seq_len in executor_config (#3487) |
| 5189 | 9f41e826bf | HuiGao-NV | 2025-04-14 | fix: remove one duplicated line of code (#3523) |
| 5190 | 9d7d48faeb | bhsueh_NV | 2025-04-14 | fix: disable the kv cache reuse for prompt tuning test (#3474) |
| 5191 | 44090a5388 | brb-nv | 2025-04-13 | Add support for Phi-4-MM (#3296) |
| 5192 | 8d3e449a8d | QI JUN | 2025-04-14 | reduce num layers in attention test (#3509) |
| 5193 | 19d296b4b2 | Yiqing Yan | 2025-04-14 | chore: add dgx_h200 tests (#3451) |
| 5194 | 9d64b6b890 | yuxianq | 2025-04-14 | Cache sin cos in model instead of global LRU cache. (#3378) |
| 5195 | fe6f14b2b1 | pcastonguay | 2025-04-13 | fix: Fixing issue with first gen token being returned twice in streaming (#3427) |
| 5196 | af67bf00a8 | William Tambellini | 2025-04-13 | feat: register ENABLE_MULTI_DEVICE and ENABLE_UCX as CMake options (#3343) |
| 5197 | 75e13f4f88 | Chuang Zhu | 2025-04-14 | chore: disable some env for disagg defaultly (#3415) |
| 5198 | baeec63dda | yuxianq | 2025-04-14 | refactor: Remove _pp_forward. (#3496) |
| 5199 | 65d1591fbf | Yiqing Yan | 2025-04-14 | Waive L0 test (#3508) |
| 5200 | 6ee021a90d | Chuang Zhu | 2025-04-14 | chore: exchange connection id with tagSend/tagRecv (#3320) |
| 5201 | d0f83d19f1 | HuiGao-NV | 2025-04-13 | fix: add kv memory size per token of draft model to calculate max number of tokens of kv cache (#3497) |
| 5202 | b37c5c0a4d | Yan Chunwei | 2025-04-13 | make LLM-API slurm examples executable (#3402) |
| 5203 | 7b38018fa0 | Aurelien Chartier | 2025-04-13 | feat: Add numNodes to ParallelConfig (#3346) |
| 5204 | 5d3180be82 | dominicshanshan | 2025-04-13 | feat: Add stress test for TRT-LLM (#3250) |
| 5205 | 74850c61e9 | Yan Chunwei | 2025-04-13 | fix: switch ZMQ from file socket to tcp socket in RemoteMpiCommSession (#3462) |
| 5206 | ceec4924d9 | Robin Kobus | 2025-04-12 | refactor: batch slot management in decoder classes (#3300) |
| 5207 | 145a126a28 | pcastonguay | 2025-04-12 | chore: Unwaive DS + overlap disagg test (#3339) |
| 5208 | c6081abb0e | WeiHaocheng | 2025-04-12 | feat: Make scaffolding Controller more generic #3408 (#3416) |
| 5209 | 012fb9a1c4 | QI JUN | 2025-04-12 | remove useless max_num_tokens member in PyTorchConfig (#3493) |
| 5210 | 2ab71f9a80 | Robin Kobus | 2025-04-12 | refactor: decoder buffers (#3307) |
| 5211 | 29c5085400 | yuxianq | 2025-04-12 | fix: Fix PP for llama. (#3449) |
| 5212 | 1bd84c6d8c | Robin Kobus | 2025-04-12 | feat: Allow individual gatherContext for each additional output (#3374) |
| 5213 | adf60a8723 | nv-guomingz | 2025-04-12 | fix:update the default excluded_modules value for fp8rowwise recipe. (#3477) |
| 5214 | 4855431d3d | hlu1 | 2025-04-11 | [Deepseek] Redesign multi-stream API (#3459) |
| 5215 | 3041bbdab3 | Iman Tabrizian | 2025-04-11 | fix: Fix disagg MTP with overlap (#3406) |
| 5216 | c51e90d7d7 | HuiGao-NV | 2025-04-12 | fix: don't perform memory estimation for start_attention (#3485) |
| 5217 | 5e2923bb92 | Enwei Zhu | 2025-04-12 | test: Automatically clean checkpoints and engines (#3468) |
| 5218 | c539750d42 | Iman Tabrizian | 2025-04-11 | fix: Allow context_and_generation request type in disagg overlap (#3489) |
| 5219 | d167cbd5bb | QI JUN | 2025-04-12 | refactor: remove ParallelConfig in tensorrt_llm._torch.distributed module (#3370) |
| 5220 | cf9ceea890 | Enwei Zhu | 2025-04-12 | test: Add DeepSeek-V3-Lite PP=4 cases (#3454) |
| 5221 | ec723fa993 | Fridah-nv | 2025-04-11 | feat:[AutoDeploy] Enhance RoPE support (#3115) |
| 5222 | 11b0091863 | Bo Li | 2025-04-11 | docs: Update perf-benchmarking doc on GPU configuration for consistent benchmarking. (#3458) |
| 5223 | aeecdb0ab9 | Robin Kobus | 2025-04-11 | fix: Eagle decoding (#3456) |
| 5224 | ff82aef99b | Yukun He | 2025-04-11 | Fix the issues related to fused moe path. (#3435) |
| 5225 | b168adba70 | liji-nv | 2025-04-11 | feat: Add NVFP4 UB pattern optimization pass in torch compile (#3371) |
| 5226 | ea050084ad | Shunkangz | 2025-04-11 | feat: Add support of chat completion in PD (#2985) |
| 5227 | 5bc6f093c8 | Yechan Kim | 2025-04-11 | fix: mllama e2e pytorch flow fix (#3397) |
| 5228 | 20e54e5c89 | Ivy Zhang | 2025-04-11 | test: add cuda visible device constraint for phi_1gpu test (#3364) |
| 5229 | d998832b33 | Ivy Zhang | 2025-04-11 | test: add torch flow test case in qa test list (#3404) |
| 5230 | 143edc8153 | pansicheng | 2025-04-11 | fix partialMatch (#3413) |
| 5231 | 0d351317c2 | Yiqing Yan | 2025-04-11 | Waive failure post-merge tests (#3472) |
| 5232 | a139eae425 | Yuan Tong | 2025-04-11 | chore: Stabilize ABI boundary for internal kernel library (#3117) |
| 5233 | 410f56357e | Enwei Zhu | 2025-04-11 | test: Waive torch compile tests (#3471) |
| 5234 | 16ca45747b | QI JUN | 2025-04-11 | always trigger multi gpu test to protect modeling_llama.py and modeling_deepseekv3.py (#3434) |
| 5235 | 5142c783c0 | wili | 2025-04-11 | fix: Beam Search Diversity (#3375) |
| 5236 | 1e2a339642 | QI JUN | 2025-04-11 | waive unittest/_torch/multi_gpu (#3464) |
| 5237 | 6cef10068a | QI JUN | 2025-04-11 | waive a test case of llama 3.1 with torch compile (#3461) |
| 5238 | 5616c0d232 | tburt-nv | 2025-04-11 | add precommit check to github actions (#3129) |
| 5239 | a8310b01dc | Dom Brown | 2025-04-10 | feat: trtllm-gen fp4 GEMM for pytorch workflow (#3423) |
| 5240 | d7f45e50c6 | Iman Tabrizian | 2025-04-10 | test: disable attention DP tests for single GPU (#3395) |
| 5241 | 8300218d21 | Zhihan Jiang | 2025-04-10 | feat: support llama4 nope layers; support FP8 checkpoint loading; (#3382) |
| 5242 | a6a2ae6cc1 | amitz-nv | 2025-04-10 | chore: Rename nvsmall to nemotron nas (#3447) |
| 5243 | af05749e90 | wm2012011492 | 2025-04-10 | feat: add qwen2 moe to torch flow; fix wrong imported KvCacheConfig in gpqa… (#3369) |
| 5244 | f5281fffaa | QI JUN | 2025-04-10 | waive some test cases of test_llm_multi_gpu.py (#3452) |
| 5245 | c5e803ba48 | Yan Chunwei | 2025-04-10 | chore: code cleanup for error logging and SharedMemory in proxy.py (#3432) |
| 5246 | d7a0bf934c | Julien Debache | 2025-04-10 | fix: updating ucxx, which appears to avoid occasional segfaults when profiling (#3420) |
| 5247 | 3ade9375ba | HuiGao-NV | 2025-04-10 | feat: Run PyExecutor's inference flow to estimate max_num_tokens for kv_cache_manager (#3092) |
| 5248 | 10d2d16247 | Yiqing Yan | 2025-04-10 | Waive L0 test (#3442) |
| 5249 | 5023e0d0f4 | Emma Qiao | 2025-04-10 | infra: Update some test description which is out of date (#3437) |
| 5250 | 67949f7c39 | Kefeng-Duan | 2025-04-10 | Update README and add benchmarking blog for DeepSeek-R1 (#3232) |
| 5251 | cec65bd09a | bhsueh_NV | 2025-04-10 | clean the waive.txt (#3441) |
| 5252 | 863d023fd0 | xinhe-nv | 2025-04-10 | test: fix memory leak of tests (#3392) |
| 5253 | b331d62f98 | tburt-nv | 2025-04-10 | add sqlite to rocky container (#3114) |
| 5254 | 16c8f39fc5 | yuxianq | 2025-04-10 | feat: Support TLLM_OVERRIDE_LAYER_NUM and TLLM_TRACE_MODEL_FORWARD for debugging (#3417) |
| 5255 | fbcf954d9c | hlu1 | 2025-04-09 | [MLA] Deallocate tensors after use (#3286) |
| 5256 | c59abae436 | brb-nv | 2025-04-09 | feat: Add Gemma3 text-only model support (#3247) |
| 5257 | b5473f7eca | QI JUN | 2025-04-10 | waive llama3.1 8B test cases with pipeline parallelism (#3433) |
| 5258 | 9307ff95ae | Frank | 2025-04-09 | fix: Add nested aliases for Llama 4 (#3381) |
| 5259 | 215fb20567 | peaceh-nv | 2025-04-10 | chore : split GptExecutor tests out of gpt tests to reduce single test time (#3412) |
| 5260 | 8d164f40d7 | tburt-nv | 2025-04-10 | update allowlist (#3428) |
| 5261 | 943218b54a | Yechan Kim | 2025-04-10 | feat: Add Qwen2.5-VL and refactor Qwen2-VL (#3156) |
| 5262 | 996696203f | Maximiliano Levi | 2025-04-09 | fix: #3137 speculative decoding and multimodal input support (#3276) |
| 5263 | 47f5cf6c0d | danielafrimi | 2025-04-09 | lora_tests (#3201) |
| 5264 | 6eee15900e | WeiHaocheng | 2025-04-09 | feat: Enhance the integrated robustness of scaffolding with __init__.py #3305 (#3312) |
| 5265 | c069abc7d8 | 石晓伟 | 2025-04-09 | Update gh pages build script (#3405) |
| 5266 | 4d78f51608 | Gabriel Wu | 2025-04-09 | fix: remove DeepGEMM line info (#3411) |
| 5267 | 6f1b2cdb83 | wili | 2025-04-09 | Doc: update steps of using Draft-Target-Model (DTM) in the documents. (#3366) |
| 5268 | d0671494cd | QI JUN | 2025-04-09 | chore: fix wheel version <= 0.45.1 (#3391) |
| 5269 | 64abb01a36 | sugunav14 | 2025-04-08 | Fix failing DSV3 unit tests (#3385) |
| 5270 | 3a8443f1e1 | tburt-nv | 2025-04-09 | extend allowlist (#3379) |
| 5271 | 8401722245 | Iman Tabrizian | 2025-04-08 | test: Add single gpu disaggregated tests (#3295) |
| 5272 | 2a2b7bfc66 | Tracin | 2025-04-09 | Fix miss bias add for FP4Linear. (#3361) |
| 5273 | 5bdf997963 | Mike Iovine | 2025-04-08 | Add Llama 4 (#3302) |
| 5274 | 7225bd8b91 | yuxianq | 2025-04-09 | chore: Refine attention backend interface. (#3271) |
| 5275 | 7199588796 | Zhanrui Sun | 2025-04-09 | infra: [TRTLLM-4450] Support more files for pytorch only mode (#3365) |
| 5276 | 54ad95eaa8 | wili | 2025-04-08 | Feat: Variable-Beam-Width-Search (VBWS) part3 (#3338) |
| 5277 | 84fc07b011 | sugunav14 | 2025-04-08 | feat: [TRTLLM-3510] DeepseekV3 support in AutoDeploy (#3281) |
| 5278 | 02f446a9ff | pcastonguay | 2025-04-08 | chore: Adding DS V3-lite tests with overlap + cuda graph (#3342) |
| 5279 | 63b0194c50 | Zhanrui Sun | 2025-04-08 | chore: bump version to 0.19.0.dev2025041500 (#3360) |
| 5280 | 316e5c3be3 | Void | 2025-04-08 | feat: fix and improve allreduce and fusion kernels (#3064) |
| 5281 | 7b03350527 | yuxianq | 2025-04-08 | Add thread leak check and fix thread/memory leak issues. (#3270) |
| 5282 | dca6397d1e | liji-nv | 2025-04-08 | feat: Introduce UB allocator for pytorch flow (#3257) |
| 5283 | c692474b59 | Zhanrui Sun | 2025-04-08 | infra: Fix bot help error when " in bot command (#3314) |
| 5284 | cdb0906be4 | Chuang Zhu | 2025-04-08 | disagg test single h100 (#3353) |
| 5285 | e04f6a1b9b | amirkl94 | 2025-04-08 | fix: Fix p-tuning test bug (#3326) |
| 5286 | deb876ecdb | Yan Chunwei | 2025-04-08 | clean up trtllm-llmapi-launch logs (#3358) |
| 5287 | 8ee019f8c4 | Enwei Zhu | 2025-04-08 | test: Accuracy test improvement (Part 3.4): Move LLaMA tests (#3350) |
| 5288 | 60e02a3684 | Pengyun Lin | 2025-04-08 | Use llm.tokenizer in OpenAIServer (#3199) |
| 5289 | c678774c99 | Yukun He | 2025-04-08 | feat: Apply the new torch-flow compatible AutoTuner to both Fused MoE and NVFP4 Linear operators. (#3151) |
| 5290 | f1655afb0d | Gabriel Wu | 2025-04-08 | feat: enable DeepGEMM by default (#3341) |
| 5291 | 62e0876e39 | Fanrong Li | 2025-04-08 | Waive unittest/trt/model/test_mamba.py::TestMamba::test_loaders_mamba_130m_hf_from_checkpoint. Will fix it later. (#3356) |
| 5292 | 31422e7e46 | MinaHuai | 2025-04-08 | add tp=2 ci test for vision encoder (#3319) |
| 5293 | 42c8574e93 | Gabriel Wu | 2025-04-08 | fix: revert extra cmake var (#3351) |
| 5294 | 1c88af1378 | Chuang Zhu | 2025-04-08 | feat: use cudaMalloc to allocate kvCache (#3303) |
| 5295 | 0a4e1d5a55 | Kaiyu Xie | 2025-04-08 | breaking change: perf: Make ipc_periodically the default responses_handler (#3102) |
| 5296 | add5e5cd93 | pcastonguay | 2025-04-07 | feat: Add option to run disaggregated serving without ctx servers,… (#3243) |
| 5297 | efe2ecfb37 | Void | 2025-04-08 | fix: runtime error in est_deepseek_allreduce.py (#3226) |
| 5298 | d40fce474a | Ivan Sorokin | 2025-04-08 | fix: redrafter sampling (#3278) |
| 5299 | ba019a43d6 | Enwei Zhu | 2025-04-08 | test: Accuracy test improvement (Part 3.3): Move DeepSeek tests (#3260) |
| 5300 | f3237e52ed | Chuang Zhu | 2025-04-07 | update readme for disaggregated (#3323) |
| 5301 | 376731013d | Gabriel Wu | 2025-04-07 | feat: use NVRTC for DeepGEMM JIT compilation (#3239) |
| 5302 | aab6214801 | YueWeng | 2025-04-07 | test: fix conflicting test names (#3316) |
| 5303 | 3545d59635 | Yao Yao | 2025-04-07 | Support speculative decoding with Hopper XQA (#3269) |
| 5304 | e5407ea89a | amitz-nv | 2025-04-07 | Fix torch nvsmall through pyexecutor and fix its TP support (#3238) |
| 5305 | ef1ba468a1 | pansicheng | 2025-04-07 | feat: support abort disconnected requests (#3214) |
| 5306 | e232d037a2 | Yiqing Yan | 2025-04-07 | chore: Blossom debug hook (#3091) |
| 5307 | 515dd0d78f | Bo Li | 2025-04-07 | feat: Add support for FP8 MLA on Hopper and Blackwell. (#3190) |
| 5308 | a2fad51011 | QI JUN | 2025-04-07 | chore: waive a timeout multi-GPU test case (#3310) |
| 5309 | a6a4920b1d | nv-guomingz | 2025-04-07 | chore: update internal cutlass library base #2981 and #3165. (#3308) |
| 5310 | 62bc13430e | Shunkangz | 2025-04-07 | fix: fix attentionDP padding request type (#3299) |
| 5311 | e8b97341de | Fanrong Li | 2025-04-07 | fix the py_decoding_iter update in decoder. (#3297) |
| 5312 | 017361c26c | brb-nv | 2025-04-06 | test: Waive non-Llama Eagle tests (#3309) |
| 5313 | 5aeef6d4c7 | Chuang Zhu | 2025-04-07 | ucx interface (#3306) |
| 5314 | 4735b87f1f | Nick Comly | 2025-04-06 | L4 added to readme (#3301) |
| 5315 | 7a659885e3 | tburt-nv | 2025-04-05 | chore: remove usernames from comments (#3291) |
| 5316 | b21cfcfed1 | Yan Chunwei | 2025-04-05 | chore: refactor the LlmArgs with Pydantic and migrate remaining pybinding configs to python (#3025) |
| 5317 | f8a4cc0629 | Frank | 2025-04-04 | perf: Add total token throughput metric. (#3212) |
| 5318 | e12e7a753d | Robin Kobus | 2025-04-05 | refactor: Expose DecoderState via bindings and integrate in TRTLLMDecoder (#3139) |
| 5319 | 0d4d50a745 | qixiang-99 | 2025-04-04 | feat: no-cache attention in PyTorch workflow (#3085) |
| 5320 | 1128dc2a5a | Jinyang Yuan | 2025-04-04 | perf: Use pinned H2D to reduce bubbles (#3147) |
| 5321 | 77724b0fcb | Robin Kobus | 2025-04-04 | Reapply "refactor: Replace DecoderFinishedEvent with CudaEvent in decoder clas…" (#3183) (#3195) |
| 5322 | 059a34468c | QI JUN | 2025-04-04 | fix deepseek multi gpu tests timeout (#3285) |
| 5323 | d96c4e3379 | tburt-nv | 2025-04-04 | update internal_cutlass version.txt to d03df7b27 (#3279) |
| 5324 | 5776b99b70 | yuanjings-nvda | 2025-04-03 | fix vila test (#3042) |
| 5325 | ee4aab72ec | shaharmor98 | 2025-04-04 | feat: Support PeftCacheManager in Torch (#3186) |
| 5326 | f25c7cefb4 | Pengyun Lin | 2025-04-04 | doc: refactor trtllm-serve examples and doc (#3187) |
| 5327 | bb6c338730 | Tracin | 2025-04-04 | AWQ support Modelopt ckpts. (#3258) |
| 5328 | b763051ba4 | pcastonguay | 2025-04-03 | chore: Refactor disaggregated serving scripts (#3073) |
| 5329 | 32ae1564bd | Yibin Li | 2025-04-03 | update FP4 quantize layout (#3045) |
| 5330 | 385a01055c | Kaiyu Xie | 2025-04-03 | doc: Add serving section for DS V3 document (#3262) |
| 5331 | bd75ec02f2 | Zhanrui Sun | 2025-04-03 | Fix bot check error when triggered by pull request (#3268) |
| 5332 | 11624a8e96 | Fanrong Li | 2025-04-03 | fix deepseek-v3 mtp doc. (#3272) |
| 5333 | c7533d271f | Yechan Kim | 2025-04-03 | doc: add supported-models on PyTorch example (#3179) |
| 5334 | d138795485 | Yukun He | 2025-04-03 | Fix minor issues in test_autotuner.py and loose the cache check for test gemms. (#3261) |
| 5335 | 67e9f99d46 | Zhanrui Sun | 2025-04-03 | infra: [TRTLLM-4308] Add Bot help (#3192) |
| 5336 | 587a36db96 | Zhanrui Sun | 2025-04-03 | infra: [TRTLLM-4370] Fix the build error when build GH200 image (#3229) |
| 5337 | 2005e5aaaf | xinhe-nv | 2025-04-03 | remove tests from qa test lists (#3256) |
| 5338 | 174a5af779 | xiweny | 2025-04-03 | doc: refine integration test guide (#3215) |
| 5339 | 1fe64b90be | Fanrong Li | 2025-04-03 | fix: fix the acceptance rate of pytorch workflow in trtllm-bench (#3240) |
| 5340 | 2d80db4c36 | Frank | 2025-04-03 | chore: Remove build config from Pytorch kwargs. (#3210) |
| 5341 | 7f03125098 | Zhanrui Sun | 2025-04-03 | test: [TRTLLM-3994] Support only run pytorch tests (#3013) |
| 5342 | dcc0ebd273 | Zongfei Jing | 2025-04-03 | Fix warning (#3254) |
| 5343 | b5b83009ff | pcastonguay | 2025-04-02 | chore: Reenabling get_stats_async test which seems to have been fixed by recent commit (#3246) |
| 5344 | 2fdfa39ea8 | Jinyang Yuan | 2025-04-03 | fix: Fix an error related to dummy request when MTP is used (#3146) |
| 5345 | 664f428476 | QI JUN | 2025-04-03 | set test timeout threshold to 5400 second (#3249) |
| 5346 | ca6615d800 | Ming Wei | 2025-04-03 | Remove gen_cuda_headers_for_xqa.py (#3222) |
| 5347 | f5bf74bc7f | Chuang Zhu | 2025-04-03 | enable some disagg test (#3203) |
| 5348 | d998339855 | Anurag Mukkara | 2025-04-02 | Raise error for PP + MTP (#3244) |
| 5349 | 5fc2f63fec | Lucas Liebenwein | 2025-04-02 | infra: Devcontainer productivity improvements (#3075) |
| 5350 | abcb0486dc | QI JUN | 2025-04-02 | fix deepseek failure with pipeline parallelism (#3225) |
| 5351 | b5bc0a9fcd | Robin Kobus | 2025-04-02 | chore: Add output of first token to additional generation outputs (#3205) |
| 5352 | c9e94ec807 | Zheng Duan | 2025-04-02 | fix: remove test relies on timing (#3228) |
| 5353 | 228e453780 | WeiHaocheng | 2025-04-02 | doc: add doc ahout developent on cloud or runpod (#3194) |
| 5354 | 3cf7066350 | Enwei Zhu | 2025-04-02 | test: Accuracy test improvement (Part 3.2): Move Qwen tests (NvBug 5135332) (#3219) |
| 5355 | d3948cd9b2 | Enwei Zhu | 2025-04-02 | fix: GPT-Next convert failure (#3220) |
| 5356 | e64c565750 | WeiHaocheng | 2025-04-02 | doc: add a directory for scaffolding contributors (#3224) |
| 5357 | 5a72945eec | Zheng Duan | 2025-04-02 | fix: conditional disagg test name (#3161) |
| 5358 | dbc0496f37 | William Tambellini | 2025-04-02 | fix: upgrade cmake minimum from 3.18 to 3.27 (#3208) |
| 5359 | 76a6a62073 | Julien Debache | 2025-04-02 | fix: segfault in cudaDriverWrapper (#3017) |
| 5360 | 8d48b96545 | Zongfei Jing | 2025-04-02 | reduce test cases for deepseek (#3211) |
| 5361 | 34e63d07e6 | wili | 2025-04-02 | feat: Variable-Beam-Width-Search (VBWS) Part2 (#3133) |
| 5362 | 05b50b297f | Gabriel Wu | 2025-04-02 | [feat] open source fp8_blockscale_gemm (#3071) |
| 5363 | c19b7f7c2a | Yiqing Yan | 2025-04-02 | waive L0 test (#3217) |
| 5364 | bb10cdcfb8 | QI JUN | 2025-04-02 | chore: refine fetch new requests method (#3213) |
| 5365 | c5199c0b3d | Zhanrui Sun | 2025-04-02 | infra: Support get file change for github PR (#3098) |
| 5366 | 35b828ca2d | Zheng Duan | 2025-04-02 | fix streaming in dist-serving (#3087) |
| 5367 | bc5811da65 | Chuang Zhu | 2025-04-02 | chore: Ucx ip port remove mpi depend (#3101) |
| 5368 | c7548ad72c | Zongfei Jing | 2025-04-02 | perf: Add optimizations for deepseek in min latency mode (#3093) |
| 5369 | 1fe3e30356 | brb-nv | 2025-04-01 | Add support for Phi-4-mini (#2990) |
| 5370 | 42963baacd | Zhanrui Sun | 2025-04-02 | chore: bump version to 0.19.0.dev2025040800 (#3171) |
| 5371 | 8fe2e5865e | QI JUN | 2025-04-02 | refine broadcast new requests method (#3198) |
| 5372 | a5f32f46fd | Fridah-nv | 2025-04-01 | fix: [AutoDeploy] Update README.md (#3072) |
| 5373 | 1d3a5d38af | Chang Liu | 2025-04-01 | fix: Update FP8 sf layout for Blackwell and relax blockwise GEMM assertions (#3144) |
| 5374 | d880f4a7c6 | Robin Kobus | 2025-04-01 | chore: Cursor ignore cubin in headers (#3202) |
| 5375 | b2f69db507 | Enwei Zhu | 2025-04-01 | test: Accuracy test improvement (Part 3.1): Extend accuracy test suite with LLM API and initial implementation of `trtllm-eval` (#3167) |
| 5376 | bf02b9144f | amirkl94 | 2025-04-01 | feature: Add LoRA support for gemma (#3068) |
| 5377 | d7386d14a8 | Robin Kobus | 2025-04-01 | refactor: Simplify disableLookahead and improve numDecodingEngineTokens handling (#3103) |
| 5378 | ff35af77ea | WeiHaocheng | 2025-04-01 | feat: refactor scaffolding worker and support openai api worker (#3166) |
| 5379 | d34202273b | bhsueh_NV | 2025-04-01 | fix bug of glm-4-9b ci (#3184) bug nvbug_5196515 |
| 5380 | c725f1043f | Yiteng Niu | 2025-04-01 | update user list (#3193) |
| 5381 | 992d513bc6 | Jinyang Yuan | 2025-04-01 | feat: Optionally split MoE inputs into chunks to reduce GPU memory usage (#3104) |
| 5382 | 727d78e785 | brb-nv | 2025-03-31 | Support prequantized fp8 ckpt for nemotron-mini-4b-instruct (#3046) |
| 5383 | 7575dd00e7 | Yan Chunwei | 2025-04-01 | add slurm script examples for llm-api (#3135) |
| 5384 | 2994527110 | Yuan Tong | 2025-04-01 | chore: cutlass cleanup (#3165) |
| 5385 | 22ff81b047 | dongjiyingdjy | 2025-04-01 | fix：fix illeagel memory access when mtp >= 2 (#3006) |
| 5386 | 75495730bc | QI JUN | 2025-04-01 | Revert "refactor: Replace DecoderFinishedEvent with CudaEvent in decoder clas…" (#3183) |
| 5387 | dda7354d1a | Shunkangz | 2025-04-01 | Refactor return of first gen token in PD (#2986) |
| 5388 | 1901bfcf76 | brb-nv | 2025-03-31 | test: Add Eagle tests with untrained heads (#2991) |
| 5389 | c4ee14e43a | jiahanc | 2025-03-31 | fix: Reverse cuda graph size order (#3116) |
| 5390 | 68bcd0ac07 | Erin | 2025-03-31 | doc: update README (#3162) |
| 5391 | 14e194433c | Aurelien Chartier | 2025-03-31 | chore: cleanup py_executor code (#3132) |
| 5392 | 435cd2983d | Anurag Mukkara | 2025-03-31 | perf: Optimisations for PP + attention DP (#3134) |
| 5393 | 8bb3eea285 | Frank | 2025-03-31 | perf: Readd iteration logging for trtllm-bench. (#3039) |
| 5394 | e8731ba3b7 | Iman Tabrizian | 2025-03-31 | fix: disable cuda graph and MTP for overlap tests (#3155) |
| 5395 | f665f83256 | WeiHaocheng | 2025-03-31 | feat: improve scaffolding shutdown process (#3084) |
| 5396 | 36ac5e78ed | Zhanrui Sun | 2025-03-31 | chore: bump version to 0.19.0.dev2025040100 (#3152) |
| 5397 | 839aad4d6e | Quanfeng Li | 2025-03-31 | fix: Add missing parameter for WeightOnlyQuantRowLinear module (#2768) |
| 5398 | 9560fcd5ec | QI JUN | 2025-03-31 | Chore: waive tests and fix multi-GPU tests (#3157) |
| 5399 | 322ac565fc | bhsueh_NV | 2025-03-31 | chore: clean some ci of qa test (#3083) |
| 5400 | 1e1116ccfc | Zhanrui Sun | 2025-03-31 | infra: Switch to urm.nvidia.com as a WAR for urm-rn.nvidia.com connection issue |
| 5401 | 86f3b59f81 | xinhe-nv | 2025-03-31 | update waive list (#3094) |
| 5402 | e0d0dde058 | liji-nv | 2025-03-31 | None - Add one-shot version for UB AR NORM FP16/BF16 (#2995) |
| 5403 | 794f61c997 | Yan Chunwei | 2025-03-31 | fix: fix single-node cannot quit issue on slurm (#3140) |
| 5404 | 88e1c90fd0 | musvaage | 2025-03-31 | doc: use alert formatting (#3153) |
| 5405 | 3aae124a00 | Yiteng Niu | 2025-03-30 | infra: update concurrency control (#3120) |
| 5406 | 5416966ddb | Mike Iovine | 2025-03-29 | Add initial EAGLE-3 implementation (#3035) |
| 5407 | 9c484b24e6 | William Tambellini | 2025-03-29 | fix #3109: early exit cmake if find_library() does not find any lib (#3113) |
| 5408 | c75d7cd684 | Erin | 2025-03-28 | move BuildConfig functional args to llmargs (#3036) |
| 5409 | 3ee4332fb1 | Robin Kobus | 2025-03-28 | refactor: Replace DecoderFinishedEvent with CudaEvent in decoder classes (#3078) |
| 5410 | 45134d7095 | Robin Kobus | 2025-03-28 | refactor: Improve decoder finalize function (#3077) |
| 5411 | 3e37531c6a | BatshevaBlack | 2025-03-28 | feat: Add BW measurement (#3070) |
| 5412 | 3de82c41cd | Aurelien Chartier | 2025-03-27 | Pytorch PP + attention DP support (#3044) |
| 5413 | ec03159e60 | Fanrong Li | 2025-03-27 | fix: Waive twoshot to fix acc issue (#3066) |
| 5414 | 644a01cbbe | Fanrong Li | 2025-03-27 | test: Add gpqa tests for DeepSeek models (#3063) |
| 5415 | 87ab794aa2 | Yan Chunwei | 2025-03-27 | fix: fix hang in mgmn with trtllm-llmapi-launch command (#3119) |
| 5416 | 0976360204 | Fanrong Li | 2025-03-27 | add support for MTP+cuda_graph_padding. (#3096) |
| 5417 | 6979afa6f2 | xiweny | 2025-03-27 | test: reorganize tests folder hierarchy (#2996) |
| 5418 | 82edd90350 | Yan Chunwei | 2025-03-27 | fix gpus_per_node in trtllm-bench when world_size < device_count (#3007) |
| 5419 | 60d4dacc47 | Dom Brown | 2025-03-26 | Port multi GPU changes to GitHub (#3027) |
| 5420 | 047f2b234d | Suyog Gupta | 2025-03-26 | perf: [AutoDeploy] Enable AutoDeploy as a backend in trtllm-bench (#3041) |
| 5421 | 3e035f2219 | wili | 2025-03-26 | v1.2 (#3082) |
| 5422 | d9522c5906 | Robin Kobus | 2025-03-26 | feat: Update cutlass (#2981) |
| 5423 | 6b583f6f83 | Jinyang Yuan | 2025-03-26 | perf: Enable CUDA graphs when attention DP is used and active requests on different GPUs are uneven (#3010) |
| 5424 | 3c3629c52a | Robin Kobus | 2025-03-26 | refactor: simplify forward methods in GptDecoderBatched (#3076) |
| 5425 | 94dd456bd0 | Robin Kobus | 2025-03-26 | refactor: Remove speculative decoding parameters from stateful decoders (#3024) |
| 5426 | f995a92a31 | Dom Brown | 2025-03-26 | CI: Waive for https://nvbugspro.nvidia.com/bug/5189673 (#3100) |
| 5427 | 224469b096 | Enwei Zhu | 2025-03-26 | test: [TRTLLM-4334] Create 1.0 criteria scope from API stability references (#3069) |
| 5428 | ea3739ee62 | Kaiyu Xie | 2025-03-26 | Fix: fuse message not aligned on different processes (#3067) |
| 5429 | d70ff79d1d | Zheng Duan | 2025-03-26 | conditional disagg test (#3012) |
| 5430 | 3e116c9687 | Ivy Zhang | 2025-03-26 | test: add random image test for llama-3.2-11b-vision (#3055) |
| 5431 | f70b439503 | Enwei Zhu | 2025-03-26 | bitmask v3 (#3009) |
| 5432 | 0ec7b5701f | Aurelien Chartier | 2025-03-26 | chore: Handle qwen2audio inputs ids expansion during processing (#3080) |
| 5433 | 3c7cb6629c | Yechan Kim | 2025-03-26 | Add EXAONE-Deep (#3054) |
| 5434 | e6cb34d921 | kxdc | 2025-03-26 | test: fix QA TRT integration testlist mismatch issue (#3090) |
| 5435 | 5e272eef81 | peaceh-nv | 2025-03-26 | feat : reduce trt engine build time in testing (#3014) |
| 5436 | 1ac0566a93 | DylanChen-NV | 2025-03-26 | fix: fix for cp > kvHeadNum (#3002) |
| 5437 | 25f2434495 | HuiGao-NV | 2025-03-26 | fix: Set correct draft_token_nums to dummy requests for torch compilation with MTP (#3053) |
| 5438 | 268933b5cc | yuxianq | 2025-03-26 | Refactor imports inside tensorrt_llm._torch. (#3015) |
| 5439 | e68749ca1e | tburt-nv | 2025-03-26 | 2025-03-25 update CI allowlist (#3074) |
| 5440 | 7361c7d401 | Anurag Mukkara | 2025-03-25 | Add second possible output (#3043) |
| 5441 | f93ac9672e | Enwei Zhu | 2025-03-25 | clean (#3061) |
| 5442 | 8ee840159b | Shunkangz | 2025-03-25 | Add updateKVCacheTransfer (#2984) |
| 5443 | 110c6fc0f0 | Chuang Zhu | 2025-03-25 | wait long time for disagg test (#2998) |
| 5444 | 53adb3cb4e | Yuan Tong | 2025-03-25 | test: waive flaky test_kv_cache_event_async_api (#3062) |
| 5445 | d9acce72bb | Xiaowei Wang | 2025-03-25 | doc: Update DeepSeekV3 doc (#3052) |
| 5446 | e9df23f815 | Perkz Zheng | 2025-03-25 | fix: [MLA] fix the bug with fp8 MLA kernels on Blackwell. (#3008) |
| 5447 | 5724c61934 | bhsueh_NV | 2025-03-25 | chore: fix bug of model paths in confset.py (#3011) |
| 5448 | aacb8d66f4 | xiweny | 2025-03-25 | doc: document running CI stage locally (#3060) |
| 5449 | a8ec1cc4ea | QI JUN | 2025-03-25 | remove examples/test_gptj.py::test_llm_gptj_fp8_manage_weights_summary test case (#3057) |
| 5450 | 69feafc947 | Yan Chunwei | 2025-03-25 | fix: amend the test list (#3056) |
| 5451 | 7ac04ada2a | WeiHaocheng | 2025-03-25 | doc: Add README.md for scaffolding (#3048) |
| 5452 | ed84f8f923 | bhsueh_NV | 2025-03-25 | fix bug of test_phi (#3050) |
| 5453 | ef78518310 | Aurelien Chartier | 2025-03-24 | Only gather responses on rank 0 (#3040) |
| 5454 | a33c595c88 | Aurelien Chartier | 2025-03-24 | Fix logits dtype in assert (#3038) |
| 5455 | c2ffce7dbd | Zhanrui Sun | 2025-03-25 | chore: bump version to "0.19.0.dev2025032500" (#3019) |
| 5456 | c29cebf79d | Yan Chunwei | 2025-03-25 | Deprecate model_api examples (#2999) |
| 5457 | 11f9ecb2fd | bhsueh_NV | 2025-03-25 | chore: remove useless param (#3023) |
| 5458 | 59deb8b06e | Kaiyu Xie | 2025-03-25 | doc: Update CONTRIBUTING.md (#3033) |
| 5459 | 705eef68c2 | Enwei Zhu | 2025-03-25 | test: Accuracy test improvement (Part 2): Incorporate mmlu to accuracy test suite (#2982) |
| 5460 | dc0463b0e2 | nv-guomingz | 2025-03-24 | doc:add version.txt for internal cutlass library and nvrtc_wrapper so files (#3030) |
| 5461 | 5b4a5014d1 | Pradeep Raj Prabhu Raj | 2025-03-24 | Fix: wrong path to constraints.txt in bloom/requirements.txt (#3003) |
| 5462 | da0b0e0ee3 | Netanel Haber | 2025-03-24 | fix: disable kv cache reuse when minimum window size is reached, instead of maximum window size (#2983) |
| 5463 | 531b98ed62 | Yan Chunwei | 2025-03-24 | feat: Add several pure python configs to LlmArgs (#2997) |
| 5464 | cb11c10719 | Yiteng Niu | 2025-03-24 | add ratelimit in workflow (#3001) |
| 5465 | 832ea997f6 | QI JUN | 2025-03-24 | chore: Simplify quickstart of PyTorch flow (#3000) |
| 5466 | ec4f43a0ab | nv-guomingz | 2025-03-24 | test:remove opt/mpt/gptj/gptneox/bloom/falcon/baichuan/internlm/deep_… (#2987) |
| 5467 | 08b45d1bb9 | Michael Gschwind | 2025-03-23 | Update README.md (#2862) |
| 5468 | 7413cb555a | bhsueh_NV | 2025-03-24 | relax the limitation of setuptools (#2992) |
| 5469 | c3c5a07dca | Oguz Vuruskaner | 2025-03-24 | Update setup.py (#2876) |
| 5470 | 456a850e66 | Laikh Tewari | 2025-03-23 | Claim support for QwQ 32B (#2877) |
| 5471 | 37644e22bc | Yiteng Niu | 2025-03-24 | update approver list (#2994) |
| 5472 | c03d59817f | Enwei Zhu | 2025-03-24 | fix: LLM API logits processor example comments (#2962) |
| 5473 | a570578c7f | juney-nvidia | 2025-03-23 | Update the CONTRIBUTING.md as the ramp-up for TensorRT-LLM github firstly (#2980) |
| 5474 | 2631f21089 | Kaiyu Xie | 2025-03-23 | Update (#2978) |
| 5475 | c2ac9e6269 | tburt-nv | 2025-03-19 | update github workflow (#2943) |
| 5476 | 3aa6b11d13 | Kaiyu Xie | 2025-03-18 | Update TensorRT-LLM (#2936) |
| 5477 | aa1c52fa26 | niukuo | 2025-03-13 | update github workflow |
| 5478 | 9b931c0f63 | Kaiyu Xie | 2025-03-11 | Update TensorRT-LLM (#2873) |
| 5479 | c384d26736 | Yiteng Niu | 2025-03-06 | migrate to l0-test.yml (#2858) |
| 5480 | 225b77667c | Kaiyu Xie | 2025-03-04 | Fix .gitmodules (#2852) |
| 5481 | 77d7fe1eb2 | Kaiyu Xie | 2025-03-04 | Update TensorRT-LLM (#2849) |
| 5482 | 0bcfdca6aa | tburt-nv | 2025-02-27 | Use NVIDIA-gha runners to collect test results (#2830) |
| 5483 | d2b7b64b25 | Laikh Tewari | 2025-02-25 | Add R1 perf data to latest news page (#2823) |
| 5484 | ab5b19e027 | Kaiyu Xie | 2025-02-25 | Update TensorRT-LLM (#2820) |
| 5485 | 5c794e3714 | tburt-nv | 2025-02-20 | allow build command arguments (#2808) |
| 5486 | 2ea17cdad2 | Kaiyu Xie | 2025-02-18 | Update TensorRT-LLM (#2792) |
| 5487 | e88da961c5 | Kaiyu Xie | 2025-02-13 | Update TensorRT-LLM (#2783) |
| 5488 | 16d2467ea8 | Dan Blanaru | 2025-02-06 | Update TensorRT-LLM (#2755) |
| 5489 | d93a2dde84 | Denis Kayshev | 2025-01-20 | Fix kwarg name (#2691) |
| 5490 | 0d0583a639 | Kaiyu Xie | 2025-01-08 | Update README.md (#2668) |
| 5491 | be17881062 | Kaiyu Xie | 2024-12-16 | Update TensorRT-LLM (#2582) |
| 5492 | b171e87956 | Kaiyu Xie | 2024-12-11 | Add issue triage workflows (#2566) |
| 5493 | aaacc9bd68 | Kaiyu Xie | 2024-12-11 | Update TensorRT-LLM (#2562) |
| 5494 | 340a1b62fc | Kevin Chen | 2024-12-04 | Add issue triage workflows (#2498) |
| 5495 | 548b5b7310 | 石晓伟 | 2024-12-04 | Update TensorRT-LLM (#2532) |
| 5496 | 4420547017 | Kyungmin Lee | 2024-12-02 | Fix typo (#2473) |
| 5497 | c994b69731 | niukuo | 2024-11-29 | blossom-ci.yml: run vulnerability scan on ubuntu |
| 5498 | af3d49ce53 | niukuo | 2024-11-28 | update blossom-ci.yml |
| 5499 | ae640fd376 | niukuo | 2024-11-29 | Add blossom-ci.yml (#2512) |
| 5500 | 385626572d | Kaiyu Xie | 2024-11-26 | Update TensorRT-LLM (#2502) |
