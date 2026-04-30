# Commit Section 3

Commits 1001 to 1500 (total: 5612)

| # | Commit Hash | Author | Date | Message |
|---|------------|--------|------|--------|
| 1001 | ccdfa43a6e | Balaram Buddharaju | 2026-01-13 | [https://nvbugs/5791900][fix] Fix HelixCpMnnvlMemory init with PP (#10533) |
| 1002 | bf16fbd86c | Frida Hou | 2026-01-13 | [#9283][feat] AutoDeploy: separate rms pattern detection from fusion (#9969) |
| 1003 | 7b7f1e2ba1 | Neta Zmora | 2026-01-13 | [None][feat] AutoDeploy: refactor memory usage logging (#8505) |
| 1004 | 6ee8dbfe0b | dongfengy | 2026-01-14 | [https://nvbugs/5772396][fix] WAR: Disable TinyGEMM PDL due to accuracy issues (#10619) |
| 1005 | 7a47e29dcb | Yiteng Niu | 2026-01-13 | [None][infra] support overriding nspect version (#10402) |
| 1006 | 6df2c8a074 | benzh-2025 | 2026-01-13 | [None][feat] add fp4 gemm + allreduce (#9729) |
| 1007 | c1b0b7350f | Guoming Zhang | 2026-01-13 | [None][test] Unwaive qwen3 next test case. (#9877) |
| 1008 | 38296a472b | Tailing Yuan | 2026-01-13 | [None][feat] Layer-wise benchmarks: make model init more general and support weights loading (#10562) |
| 1009 | 50c78179dd | mpikulski | 2026-01-13 | [TRTLLM-8425][doc] document Torch Sampler details (#10606) |
| 1010 | 55580f8ec1 | Erin | 2026-01-13 | [NVBUG-5670458][chore] Unwaive lp tests (#10524) |
| 1011 | 7d16f3a28b | Void | 2026-01-13 | [https://nvbugs/5788127][fix] Use uint64_t as the dtype of lamport_buffer_size to avoid overflow (#10499) |
| 1012 | bdaee87895 | Guoming Zhang | 2026-01-13 | [TRTLLM-10060][feat] Enable attention dp for Nemotron Super v3. (#10347) |
| 1013 | e291a834db | JunyiXu-nv | 2026-01-13 | [TRTLLM-8462][feat] Support GET/DELETE v1/responses/{response_id} (#9937) |
| 1014 | 04b112651b | Yuxian Qiu | 2026-01-13 | [None][feat] Hang detection for executor loop and worker. (#10480) |
| 1015 | 50c22b80d7 | Yiteng Niu | 2026-01-13 | [None][infra] Update allowlist 2026.01.08 (#10535) |
| 1016 | 7d41475954 | tburt-nv | 2026-01-13 | [None][infra] try removing shared cache dir mount (#10609) |
| 1017 | 2967d299fb | JennyLiu | 2026-01-13 | [TRTLLM-10271][test] Add Spark QA functional and performance cases (#10564) |
| 1018 | ba1cb6831d | TensorRT LLM | 2026-01-13 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1019 | bbe535fddf | fredricz-20070104 | 2026-01-13 | [None][chore] Fix disagg assert (#10596) |
| 1020 | ba1037ca4a | xxi | 2026-01-13 | [https://nvbugs/5762336][fix] support to parse the keyword modules_to_not_convert of the HF model config" (#10527) |
| 1021 | 48b09e5a25 | Iman Tabrizian | 2026-01-12 | [https://nvbugs/5689235][fix] Fix cancellation+chunked prefill+disagg (#10111) |
| 1022 | 18a33764b5 | Gal Hubara-Agam | 2026-01-12 | [None][chore] Print correct backend name in benchmark report (#10597) |
| 1023 | dacc881993 | Anish Shanbhag | 2026-01-12 | [https://nvbugs/5761391][fix] Use correct model names for config database regression tests (#10192) |
| 1024 | a1385243e1 | Suyog Gupta | 2026-01-12 | [#10580][fix] re-enable NemotronH MOE MMLU test (#10594) |
| 1025 | 9f044b9dd9 | Emma Qiao | 2026-01-12 | [None][infra] Waive failed tests for main 01/12 (#10604) |
| 1026 | bf7998f1b8 | mpikulski | 2026-01-12 | [TRTLLM-9522][test] cover LLM API `multi_modal_embeddings` (#9963) |
| 1027 | 11da7e3605 | Wanli Jiang | 2026-01-12 | [None][fix] Solve pillow version conflict (#10537) |
| 1028 | 3bd319dc8e | Zhenhuan Chen | 2026-01-12 | [https://nvbugs/5794796][chore] waive test blocking premerge (#10593) |
| 1029 | 8e806abac3 | yufeiwu-nv | 2026-01-12 | [None][test] Remove most TRT-backend test cases in llm_perf_nim.yml (#10572) |
| 1030 | c5914f9085 | yingguo-trt | 2026-01-12 | [None][chore] update deepseekv3.2 test parameter (#10595) |
| 1031 | 54459377d2 | chenfeiz0326 | 2026-01-12 | [TRTLLM-10248][feat] Support Bot to Send Perf Regression Msg to Slack Channel (#10489) |
| 1032 | 3a9a00b544 | Xianjie Qiao | 2026-01-12 | [None][feat] Add ExpertStatistic and DUMMY_ALLREDUCE for configurable_moe (#10401) |
| 1033 | 5e0dbba0c9 | Jie Li | 2026-01-12 | [None][chore]: update waive list (#10577) |
| 1034 | 2de22f1a70 | TensorRT LLM | 2026-01-12 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1035 | c0e25e5418 | Pengbo Wang | 2026-01-12 | [TRTLLM-10022][feat] Add hopper xqa decode support for skip softmax attention (#10264) |
| 1036 | c5d5af9e7f | Eran Geva | 2026-01-11 | [#8391][chore] removed llama and added deepseek to AutoDeploy's L0 perf test (#10585) |
| 1037 | 7f018c89e9 | Ivy Zhang | 2026-01-12 | [None][test] update core test list (#10538) |
| 1038 | 8e0d20d901 | Yechan Kim | 2026-01-12 | [TRTLLM-10195][feat] K-EXAONE support (#10355) |
| 1039 | 80649a8b78 | Yanchao Lu | 2026-01-11 | [None][ci] Workaround OCI-NRT slowdown issue (#10587) |
| 1040 | 0371cbfd88 | Guoming Zhang | 2026-01-11 | [None][doc] Update Qwen3-Next doc by adding known issues section (#10582) |
| 1041 | b2e2538fcd | TensorRT LLM | 2026-01-11 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1042 | 3c65ec3c55 | HuiGao-NV | 2026-01-11 | [None][chore] waive test case (#10581) |
| 1043 | f6045fac09 | fredricz-20070104 | 2026-01-10 | [None][chore] Fix Gitlab CI termination issues (#10576) |
| 1044 | f6c4dd885f | tcherckez-nvidia | 2026-01-10 | [None][chore] Update AutoDeploy model list (#10505) |
| 1045 | 6ab996d635 | TensorRT LLM | 2026-01-10 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1046 | ff7eb93f31 | William Zhang | 2026-01-09 | [https://nvbugs/5669097][tests] Add MMMU test for mistral small (#10530) |
| 1047 | 38f249b479 | Chenghao Zhang | 2026-01-09 | [https://nvbugs/5548861][fix] AutoDeploy: Fix the test (#10521) |
| 1048 | 82dfef2e56 | Linda | 2026-01-09 | [https://nvbugs/5628848][fix] Fix nanobind stub generation (#10516) |
| 1049 | fdbdbba540 | Faraz | 2026-01-09 | [https://nvbugs/5752687][fix] Choose register model config over root config for VLM (#10553) |
| 1050 | d80f01d205 | yingguo-trt | 2026-01-09 | [None][feat] Add support for DeepSeek v3.2 tests (#10561) |
| 1051 | 7295af68ba | Yechan Kim | 2026-01-10 | [None][fix] Enable AttentionDP on Qwen3-VL and fix test (#10435) |
| 1052 | 1c69aad850 | Kaiyu Xie | 2026-01-09 | [TRTLLM-10309] [feat] Optimize qk rope/nope concat for DSA (#10571) |
| 1053 | ced88424ef | Iman Tabrizian | 2026-01-09 | [https://nvbugs/5756008][fix] unwaive test (#10523) |
| 1054 | 627d306df9 | Jie Li | 2026-01-09 | [None][chore] remove some model support; add device constraint (#10563) |
| 1055 | 2b72d33fdc | ruodil | 2026-01-09 | [TRTLLM-9932][test] add kimi_k2 single node perf test (#10436) |
| 1056 | 4632a8642d | Fanrong Li | 2026-01-09 | [None][doc] blog: Optimizing DeepSeek-V3.2 on NVIDIA Blackwell GPUs (#10565) |
| 1057 | 80f261ea36 | Yuxian Qiu | 2026-01-09 | [https://nvbugs/5622938][feat] Run sample_async on extra stream. (#10215) |
| 1058 | 78bb245554 | Chang Liu | 2026-01-09 | [https://nvbugs/5787453][fix] Better align MLA chunking with indexer chunking when chunked prefill enabled for DSV32 (#10552) |
| 1059 | 4a09acd012 | bhsueh_NV | 2026-01-09 | [https://nvbugs/5785206][infra] unwaive the accuracy/test_llm_api_pytorch.py::TestQwen3_30B_A3B (#10560) |
| 1060 | 4c498bfe58 | JadoTu | 2026-01-09 | [TRTLLM-9676][fix] Fix mamba_cache_manager when enabling cuda_graph_padding and let test cover this case (#9873) |
| 1061 | c5331e6dbb | Yukun He | 2026-01-09 | [None][fix] Setup dist for AutoTuner in Layerwise benchmarking. (#10534) |
| 1062 | 6fcd4e7099 | Jie Li | 2026-01-09 | [None][chore] Add failed cases into waives.txt (#10541) |
| 1063 | 5df03b2ea7 | TensorRT LLM | 2026-01-09 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1064 | d707286ca8 | ruodil | 2026-01-09 | [None][test] restrict max_num_tokens in disagg mtp config (#10442) |
| 1065 | afa55c12b6 | Yuxian Qiu | 2026-01-09 | [None][fix] revert https://github.com/NVIDIA/TensorRT-LLM/pull/10445. (#10547) |
| 1066 | 56e779d09f | Balaram Buddharaju | 2026-01-08 | [None][chore] Waive tests blocking premerge 01/08 (#10555) |
| 1067 | 4092a87b6f | Mike Iovine | 2026-01-08 | [https://nvbugs/5740075][fix] Fix sm120 speculation (#10049) |
| 1068 | 489dd60312 | Eran Geva | 2026-01-08 | [#10513][fix] AutoDeploy: removed self.mlp_type leftovers from last moe refactor (#10512) |
| 1069 | e0331297a6 | mpikulski | 2026-01-08 | [TRTLLM-9522][fix] broken cast (#9975) |
| 1070 | c0ae6bbdbe | William Zhang | 2026-01-08 | [None][feat] EPD for Qwen3 VL (#10470) |
| 1071 | 6511dbaea0 | Eran Geva | 2026-01-08 | [#10417][fix] AutoDepoloy - Reverted to direct computation of minusA (#10509) |
| 1072 | bea61bb17d | bhsueh_NV | 2026-01-08 | [None][fix] Mistral large 3 few code refine (#10405) |
| 1073 | dc6b743fb6 | Yiqing Yan | 2026-01-08 | [None][chore] Bump version to 1.2.0rc8 (#10542) |
| 1074 | 43839c7d9b | Emma Qiao | 2026-01-08 | [TRTLLM-9642][infra] Increase pytest verbosity for failed tests (#9657) |
| 1075 | 8d4b09dac6 | dongfengy | 2026-01-08 | [None][doc] Update GPTOSS Doc (#10536) |
| 1076 | 22c81cb5fa | HuiGao-NV | 2026-01-08 | [None][chore] Enable seg fault cases since one race condition is fixed (#10398) |
| 1077 | f57aab5255 | Barry Kang | 2026-01-08 | [https://nvbugs/5775402][fix] Fix concurrency list in Wide-EP perf tests (#10529) |
| 1078 | 30f8455d29 | Lucas Liebenwein | 2026-01-07 | [https://nvbugs/5747878][fix] unwaive llama4 scout tests (#10468) |
| 1079 | 342a47bf47 | TensorRT LLM | 2026-01-08 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1080 | f8b2a8fd30 | yingguo-trt | 2026-01-08 | [None][chore] Support multiple job submission at the same time (#10492) |
| 1081 | b85c447ceb | Yuxian Qiu | 2026-01-08 | [https://nvbugs/5784543][fix] Setup dist before using autotuner. (#10491) |
| 1082 | 09d9878385 | Yukun He | 2026-01-08 | [TRTLLM-9661][chore] Further reduce tuning time for cuteDSL nvFP4 dense gemm. (#10339) |
| 1083 | 81f878c279 | xxi | 2026-01-08 | [https://nvbugs/5707392][fix] unwaive test_fused_moe_fp8_blockwise_wide_ep[NotEnabled] (#10428) |
| 1084 | d736c7f290 | Lucas Liebenwein | 2026-01-07 | [https://nvbugs/5761665][fix] AutoDeploy: handle bugs for 25.12 dlfw upgrade (#10511) |
| 1085 | 7187afe7b9 | Ziyi Xiong | 2026-01-08 | [https://nvbugs/5781589][fix] Skip spec dec for non-last rank (#10445) |
| 1086 | e8cceb06b2 | Patrice Castonguay | 2026-01-07 | [None][doc] Adding parallelism types in feature combination matrix (#9849) |
| 1087 | b130d58c88 | yufeiwu-nv | 2026-01-07 | [None][test] Remove most TRT-backend test cases in llm_perf_nim.yml (#10487) |
| 1088 | 7e88212d24 | tcherckez-nvidia | 2026-01-07 | [None][bug] fix export for microsoft/Phi-3-medium-128k-instruct (#10455) |
| 1089 | 872210468b | xinhe-nv | 2026-01-07 | [TRTLLM-8638][fix] Add failed cases into waives.txt (#10474) |
| 1090 | dc32bac9fc | Kanghwan | 2026-01-06 | [#4745][fix] Pass lora_params through Qwen2/3 model forward (#10174) |
| 1091 | cbf8357e5f | yingguo-trt | 2026-01-07 | [https://nvbugs/5726086][fix] update kimi-k2-1k1k dataset (#10473) |
| 1092 | be5579633e | xinhe-nv | 2026-01-07 | [TRTLLM-8638][fix] Add failed cases into waives.txt (#10457) |
| 1093 | a34aa63685 | Fanrong Li | 2026-01-07 | [https://nvbugs/5767223][feat] add pp support for DeepSeek-v3.2 (#10449) |
| 1094 | 3fec7e411c | TensorRT LLM | 2026-01-07 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1095 | 1fbadd2dde | xinhe-nv | 2026-01-07 | [None][chore] Add failed cases into waives.txt (#10365) |
| 1096 | 4a1b2e23b3 | Ivy Zhang | 2026-01-07 | [https://nvbugs/5698434][test] add qwen3-4b accuracy test case (#10382) |
| 1097 | 6095c80e56 | Lucas Liebenwein | 2026-01-06 | [https://nvbugs/5721907][fix] AutoDeploy: improve numerical stability of flashinfer attention test (#10467) |
| 1098 | bb2f883296 | Zongfei Jing | 2026-01-07 | [None] [feat] Add test script and raster M for gather fc1 kernel (#10429) |
| 1099 | bb6a3973aa | Lucas Liebenwein | 2026-01-06 | [https://nvbugs/5732942][fix] AutoDeploy: handle transformers 4.57.1 upgrade fixes (#10466) |
| 1100 | 00355b24b7 | Lucas Liebenwein | 2026-01-06 | [None][feat] precompiled installation from local src dir with fnmatch only (#10430) |
| 1101 | 77be1b7572 | Mike Iovine | 2026-01-06 | [https://nvbugs/5749988][fix] Remove redundant qwen3 spec dec test (#10387) |
| 1102 | 037753f65b | Enwei Zhu | 2026-01-07 | [https://nvbugs/5748600][ci] Unwaive disagg guided decoding test (#10409) |
| 1103 | 6a4bebcd01 | Lizhi Zhou | 2026-01-06 | [None][chore] remove redundant retries while binding to arbitrary port (#10452) |
| 1104 | 7d62773c6c | JunyiXu-nv | 2026-01-06 | [https://nvbugs/5760726][fix] Use random port in container port section (#10432) |
| 1105 | 704f58dfbe | xinhe-nv | 2026-01-06 | [TRTLLM-8638][fix] Add failed cases into waives.txt (#10427) |
| 1106 | 6507087c3f | Emma Qiao | 2026-01-06 | [None][infra] Waive failed cases on 1/6 (#10440) |
| 1107 | df0b976b99 | Bo Li | 2026-01-06 | [https://nvbugs/5785206][infra] Waive TestQwen3_30B_A3B::test_fp8[latency-torch_compile=False]. (#10441) |
| 1108 | ab58d7cac1 | William Zhang | 2026-01-05 | [https://nvbugs/5772361][ci] Unwaive tests that have been fixed (#10424) |
| 1109 | 2eaabd7461 | Kaiyu Xie | 2026-01-06 | [None] [fix] Fix undefined tokens_per_block (#10438) |
| 1110 | 1e828587e5 | Ivy Zhang | 2026-01-06 | [TRTLLM-9896][test] add vswa test cases coverage (#10146) |
| 1111 | 5108a69fc0 | Yiqing Yan | 2026-01-06 | [TRTLLM-9622][infra] Enable DGX_B300 multi-gpu testing in pre-merge pipeline (#9699) |
| 1112 | 998527724c | xinhe-nv | 2026-01-06 | [TRTLLM-8638][fix] Add failed cases into waives.txt (#10367) |
| 1113 | 810249c304 | Kaiyu Xie | 2026-01-06 | [https://nvbugs/5769926] [fix] Add no container mount home WAR (#10431) |
| 1114 | 22a1d31a27 | Ivy Zhang | 2026-01-06 | [None][test] update test case constraint (#10381) |
| 1115 | 1b1058279c | xinhe-nv | 2026-01-06 | [TRTLLM-8638][fix] Add failed cases into waives.txt (#10384) |
| 1116 | 3e98265682 | kris1025 | 2026-01-06 | [None][chore] unwaive qwen3 30b test (#10115) |
| 1117 | 596d4f16fb | TensorRT LLM | 2026-01-06 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1118 | 617f728903 | Karthik | 2026-01-05 | [#8460][feat] Revive and simplify Model Explorer visualization integration (#10150) |
| 1119 | aa1fe931de | Venky | 2026-01-06 | [None][docs] Add `--config` preference over `--extra_llm_api_options` in CODING_GUIDELINES.md (#10426) |
| 1120 | 46f035befe | Xiao Xuan | 2026-01-06 | [#2511][fix] eagle: qwen2 capture hidden states (#10091) |
| 1121 | 9cae7277ea | Min Yu | 2026-01-06 | [https://nvbugs/5726962][feat] Apply fusion for W4AFP8_AWQ MoE (#9838) |
| 1122 | 6b8ae6fa81 | alel | 2026-01-06 | [None][feat] CuteDSL MOE FC1 Enhancement (#10088) |
| 1123 | 77712ed4ab | Mike Iovine | 2026-01-05 | [None][chore] Update SWA + spec dec support matrix (#10421) |
| 1124 | 82aaf98070 | JadoTu | 2026-01-06 | [None][feat] add the eos tokens in generation config to stop words in the sampler (#10389) |
| 1125 | 8a04c05079 | chenfeiz0326 | 2026-01-06 | [None][fix] Only Use Throughput Metrics to Check Regression (#10404) |
| 1126 | 536a8f6a9c | Chuang Zhu | 2026-01-06 | [TRTLLM-9527][feat] Add transferAgent binding (step 1) (#10113) |
| 1127 | 846e54aa09 | Lucas Liebenwein | 2026-01-05 | [None][feat] precompiled installation from local src dir (#10419) |
| 1128 | 3b56548fcf | Simeng Liu | 2026-01-05 | [https://nvbugs/5777044][chore] Remove solved bugs from waives.txt (#10422) |
| 1129 | 4e50cb5708 | Karthik | 2026-01-05 | [#10170][fix] Add export patch for GraniteMoe MoE models to enable torch.export compatibility (#10169) |
| 1130 | 91ff46d418 | Mike Iovine | 2026-01-05 | [https://nvbugs/5745152][fix] Unwaive gpt oss spec decode test (#10370) |
| 1131 | 7a2dab8e85 | Mike Iovine | 2026-01-05 | [https://nvbugs/5695984][fix] Unwaive llama3 eagle test (#10092) |
| 1132 | 6b71b03947 | Yan Chunwei | 2026-01-06 | [TRTLLM-9551][infra] Partition test_llm_pytorch.py for parallel execution (#10400) |
| 1133 | ea380ff45c | Grzegorz Kwasniewski | 2026-01-05 | [TRTLLM-9767][feat] Fixed recursive node traversals (#10379) |
| 1134 | db2614ef10 | Mike Iovine | 2026-01-05 | [https://nvbugs/5772414][fix] Fix draft token tree depth=1 corner case (#10385) |
| 1135 | bedfff4f00 | Mike Iovine | 2026-01-05 | [https://nvbugs/5772521][fix] Fix draft token tree chain crash (#10386) |
| 1136 | e98c27ee4f | Gal Hubara-Agam | 2026-01-05 | [TRTLLM-10053][feat] AutoDeploy: Add Super v3 config file, improve test runtime (#10397) |
| 1137 | 225d3a9001 | Anthony Chang | 2026-01-06 | [None][perf] TRTLLM MoE maps to lower tuning buckets when ep>1 (#9998) |
| 1138 | a792c23dcf | Balaram Buddharaju | 2026-01-05 | [TRTLLM-9465][fix] Swap TP-CP grouping order (#10350) |
| 1139 | 3749a2ce1c | Eran Geva | 2026-01-05 | [#10374][fix] fixed race condition in AutoDeploy's mp tests port acquisition (#10366) |
| 1140 | b1733d56f6 | xinhe-nv | 2026-01-05 | [TRTLLM-9381][test] add disag-serving kimi k2 thinking tests (#10357) |
| 1141 | 4931c5eb3a | Fanrong Li | 2026-01-05 | [None][feat] update deepgemm to the DeepGEMM/nv_dev branch (#9898) |
| 1142 | d272f1a9bc | Yukun He | 2026-01-05 | [TRTLLM-8821][feat] Apply AutoTuner to AllReduce Op for strategy tuning. (#8531) |
| 1143 | 2f768b76f8 | HuiGao-NV | 2026-01-05 | [https://nvbugs/5715568][fix] Force release torch memory when LLM is destroyed (#10314) |
| 1144 | c63fad7d96 | Emma Qiao | 2026-01-05 | [None][infra] Waive failed cases again on 1/5 (#10403) |
| 1145 | e7a4486294 | Yihan Wang | 2026-01-05 | [https://nvbugs/5752521][fix] Unwaive test_trtllm_flashinfer_symbol_collision.py (#10227) |
| 1146 | c04cf4334e | Pengyun Lin | 2026-01-05 | [TRTLLM-8242][feat] Add stability tags for serve subcommand (#10012) |
| 1147 | 0937df2c68 | Yukun He | 2026-01-05 | [TRTLLM-10185][feat] AutoTuner Cache: Support cache file lock and merge all ranks into one (#10336) |
| 1148 | 5a8bfcbb50 | Emma Qiao | 2026-01-05 | [None][infra]Waive failed cases in post-merge on 1/5 (#10399) |
| 1149 | a7fe043b13 | Tailing Yuan | 2026-01-05 | [None][feat] Layer-wise benchmarks: support TEP balance, polish slurm scripts (#10237) |
| 1150 | aaf80be0f3 | TensorRT LLM | 2026-01-05 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1151 | 5773a4d775 | Yuxian Qiu | 2026-01-05 | [https://nvbugs/5701425][chore] Unwaive tests. (#10269) |
| 1152 | 656c705ff1 | Cheng Hang | 2026-01-05 | [None][feat] sm100 weight-only kernel (#10190) |
| 1153 | b5a1e10bc0 | Fanrong Li | 2026-01-05 | [https://nvbugs/5779534][fix] fix buffer reuse for CUDA graph attention metadata (#10393) |
| 1154 | da0830670a | Wanli Jiang | 2026-01-05 | [TRTLLM-10065][feat] Add accuracy tests for super-v3 with multiple-gpus (#10234) |
| 1155 | 82c1ba84a7 | Lizhi Zhou | 2026-01-05 | [https://nvbugs/5649010][fix] use 0 port as arbitrary port when disagg service discovery is enabled (#10383) |
| 1156 | 0517b62789 | bhsueh_NV | 2026-01-05 | [https://nvbugs/5772363][fix] fix bug of Mistral-Small-3.1-24B-Instruct-2503 (#10394) |
| 1157 | 8e2065b4d9 | Faraz | 2026-01-04 | [https://nvbugs/5670469][fix] Filter 0s and choose min of kv_head for Nemotron model (#10206) |
| 1158 | e2f5455533 | Eran Geva | 2026-01-04 | [#8391][chore] added deepseek_r1_distill_qwen_32b AutoDeploy perf test to L0 (#10377) |
| 1159 | a65b0d4efa | chenfeiz0326 | 2026-01-05 | [None][fix] Decrease Pre Merge Perf Tests (#10390) |
| 1160 | c4f27fa4c0 | Yanchao Lu | 2026-01-05 | [None][ci] Some tweaks for the CI pipeline (#10359) |
| 1161 | afc533193d | dongfengy | 2026-01-04 | [None][feat] Support nvfp4 for gptoss (#8956) |
| 1162 | a4dcc6a711 | Jaedeok Kim | 2026-01-04 | [TRTLLM-10171][fix] Correct attention handling in ModelConfig and KVCacheManager (#10330) |
| 1163 | 6ba04eba06 | Yuxian Qiu | 2026-01-04 | [https://nvbugs/5748683][fix] Use get_free_port_in_ci to avoid port conflict. (#10392) |
| 1164 | 71b4a8aa60 | TensorRT LLM | 2026-01-04 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1165 | 5bd37ce41e | yuanjingx87 | 2026-01-03 | [None][infra] add retry logic to get slurm sbatch job log when ssh dropped (#9167) |
| 1166 | 0d1f5ad7a2 | Grzegorz Kwasniewski | 2026-01-03 | [TRTLLM-10358][feat] Added proper rescaling of FP4 weights (#10378) |
| 1167 | c0b3c2b919 | Yanchao Lu | 2026-01-03 | [None][ci] Remove an invalid test waive |
| 1168 | 59045a0e41 | Ludwig Schneider | 2026-01-03 | [None][fix] [fix] Make NCCL resource manager destructor exception-safe (#10166) |
| 1169 | 865992b86b | Emma Qiao | 2026-01-03 | [None][infra] Waive failed cases on 1/3 (#10391) |
| 1170 | 9e7b50aefb | Bo Deng | 2026-01-03 | [TRTLLM-9752][fix] WAR: Disable PDL for quant kernels to fix accuracy issues (#10285) |
| 1171 | 45ffbf1f21 | TensorRT LLM | 2026-01-03 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1172 | 937f8f78a1 | Lucas Liebenwein | 2026-01-02 | [None][doc] promote AutoDeploy to beta feature in docs (#10372) |
| 1173 | bdf6953ddc | Izzy Putterman | 2026-01-02 | [None][feat] Eagle: MLA Based Eagle (#9677) |
| 1174 | f3dd6da080 | Gal Hubara-Agam | 2026-01-02 | [#10056][chore] AutoDeploy: Enable Nemo SuperV3 accuracy test (#10308) |
| 1175 | 5e0e48144f | chenfeiz0326 | 2026-01-02 | [None][fix] Minor updates on Perf Test System (#10375) |
| 1176 | 098251648d | TensorRT LLM | 2026-01-02 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1177 | f631b25c85 | fredricz-20070104 | 2026-01-02 | [None][test] Unified slurm extra args management and session collection logic (#10332) |
| 1178 | 4a1b742aa0 | Balaram Buddharaju | 2026-01-01 | [TRTLLM-9467][fix] Fix PP+CP combination with helix parallelism (#10312) |
| 1179 | 5845951538 | Gal Hubara-Agam | 2026-01-01 | [#10056][fix] AutoDeploy: Handle deletion of nested params in sharding (#10376) |
| 1180 | 4868772ad7 | tcherckez-nvidia | 2026-01-01 | [None][feat] Add export data to build and run script for AD (#10299) |
| 1181 | 9f5b750a93 | Balaram Buddharaju | 2026-01-01 | [None][chore] Waive tests blocking pre-merge 12/31 (#10373) |
| 1182 | 0b75340223 | Balaram Buddharaju | 2025-12-31 | [https://nvbugs/5744427][fix] Make Gemma3 multimodal test fp8 (#10368) |
| 1183 | edbcff0257 | TensorRT LLM | 2026-01-01 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1184 | ff836d4f41 | Yuxian Qiu | 2026-01-01 | [https://nvbugs/5740359][chore] Unwaive tests. (#10260) |
| 1185 | 1bbe71b3ed | Lucas Liebenwein | 2025-12-31 | [#10244][feat] AutoDeploy: separate prefill/decode in flashinfer (#10252) |
| 1186 | 9085021aa4 | Mike Iovine | 2025-12-31 | [None][feat] Implement sampling for MTP 1-model (#10019) |
| 1187 | 84d107b2f0 | Simeng Liu | 2025-12-31 | [https://nvbugs/5717993][fix] Add execution_stream across PyExecutor, KVCacheManager, PeftCacheManager to ensure proper CUDA stream synchronization between KV cache transfer operations and model forward kernels. (#10060) |
| 1188 | 0d2e2718ce | xinhe-nv | 2025-12-31 | [None][chore] Add failed cases into waives.txt (#10354) |
| 1189 | a23c6f1092 | chenfeiz0326 | 2025-12-31 | [TRTLLM-9834][feat] Transfer to TRTLLM-INFRA Database and Fail post-merge tests if regression (#10282) |
| 1190 | 464847c6be | tcherckez-nvidia | 2025-12-31 | [#9717][chore] Standardize MoE weights interface (#10295) |
| 1191 | ef1d4a40b5 | Jin Li | 2025-12-31 | [https://nvbugs/5727475][fix] Avoid use property with setter in nn.Mo… (#10212) |
| 1192 | d944430f96 | Emma Qiao | 2025-12-31 | [None][infra] Waive failed cases on 12/31 (#10353) |
| 1193 | 73870ae4ad | Necofish | 2025-12-31 | [None][feat] support Qwen3-VL dense model in pytorch backend (#9060) |
| 1194 | 827d12caaf | xinhe-nv | 2025-12-31 | [https://nvbugs/5558516][test] add disaggregated stress test (#9354) |
| 1195 | 910a633066 | Yuxian Qiu | 2025-12-31 | [https://nvbugs/5774869][chore] waive tests. (#10356) |
| 1196 | fdc03684cc | Yiqing Yan | 2025-12-31 | [TRTLLM-10016][infra] Use SlurmPatition attribute time as timeout threshold (#10254) |
| 1197 | fad000589d | Pengyun Lin | 2025-12-31 | [None][chore] Unify DS tool parser names (#10239) |
| 1198 | 1e9c153b4c | xinhe-nv | 2025-12-31 | [None][fix] disable thread leak check for kimi (#10337) |
| 1199 | 6c1abf2d45 | xinhe-nv | 2025-12-31 | [None][chore] Add failed cases into waives.txt (#10344) |
| 1200 | ed3a3097a4 | TensorRT LLM | 2025-12-31 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1201 | 34c2fd50a9 | Jin Li | 2025-12-31 | [https://nvbugs/5707359][fix] Unwaive OOM case that should be fixed by #9446 (#10334) |
| 1202 | 1f3afb8e6f | Yuxian Qiu | 2025-12-31 | [None][feat] Implement send_object for TorchDist. (#10213) |
| 1203 | ec8a388c25 | Yuxian Qiu | 2025-12-31 | [https://nvbugs/5769890][fix] Import get_free_port. (#10341) |
| 1204 | 74832a1895 | Eran Geva | 2025-12-30 | [https://nvbugs/5766986][fix] fixed the shard_all_unprocessed default value to align with the default.yml (#10271) |
| 1205 | 1f0365da36 | Bo Li | 2025-12-30 | [None][infra] Add LongBenchV1 to trtllm-eval. (#10265) |
| 1206 | 6732c76414 | Emma Qiao | 2025-12-30 | [None][infra] Waive failed cases for main on 12/30 (#10338) |
| 1207 | fb05cd769a | Emma Qiao | 2025-12-30 | [None][infra] Enable single-gpu CI on spark (#9304) |
| 1208 | cce7247815 | Emma Qiao | 2025-12-30 | [https://nvbugs/5594703][infra] Unwaive the failed case to test (#10275) |
| 1209 | 6accdbc6a6 | xinhe-nv | 2025-12-30 | [None][chore] Add failed cases into waives.txt (#10302) |
| 1210 | 0f4ed90560 | ruodil | 2025-12-30 | [TRTLLM-9965][test] add long-context disagg test for GB300/GB200 and remove config_index in yaml (#10225) |
| 1211 | 692d8f2023 | binghanc | 2025-12-30 | [TRTLLM-9455][feat] support for new checkpoint (#10082) |
| 1212 | 3e0344a53d | xinhe-nv | 2025-12-30 | [None][chore] Add failed cases into waives.txt (#10301) |
| 1213 | 48fee8d0f6 | xinhe-nv | 2025-12-30 | [None][chore] Add failed cases into waives.txt (#10321) |
| 1214 | f396ad83b0 | Emma Qiao | 2025-12-30 | [None][infra] Remove duplicates in waives.txt (#10333) |
| 1215 | fa4c7997c5 | TensorRT LLM | 2025-12-30 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1216 | 4944192eae | Balaram Buddharaju | 2025-12-29 | [None][chore] Waive tests failing in pre-merge 12/28 (#10311) |
| 1217 | 966231d29c | Neta Zmora | 2025-12-29 | [#9626][feat] Add an auto-deploy transform for using cutlass FP4 MoE kernels (#10304) |
| 1218 | 965578ca21 | Yanchao Lu | 2025-12-29 | [None][infra] Some improvements for Slurm execution path in the CI (#10316) |
| 1219 | 9cee32ab39 | Yueh-Ting (eop) Chen | 2025-12-29 | [https://nvbugs/5625990][fix] Respect VSWA scheme when doing block store for reuse and load block for reuse in KV cache manager (#10183) |
| 1220 | 2f8d6d25a8 | Yanchao Lu | 2025-12-29 | [None][ci] Waive an intermittent test hang case (#10324) |
| 1221 | 223411e988 | TensorRT LLM | 2025-12-29 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1222 | 270be801aa | Yanchao Lu | 2025-12-28 | [None][ci] Move remaining DGX-B200 tests to LBD (#9876) |
| 1223 | c59aa8bec5 | Ziyi Xiong | 2025-12-28 | [TRTLLM-9962][feat] Some optimizations for two-model spec dec (#10208) |
| 1224 | ae6d5766ed | TensorRT LLM | 2025-12-28 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1225 | 55bc6a5ff8 | JunyiXu-nv | 2025-12-28 | [https://nvbugs/5753250][fix] Fix undefined local variable in responses utils (#10154) |
| 1226 | ee07a7c55e | shivghai | 2025-12-27 | [None][fix] [Gemma3] Fix RoPE for local attention for Gemma3 (#9961) |
| 1227 | 1865020b6f | Guoming Zhang | 2025-12-27 | [TRTLLM-8577][feat] Clean the Qwen3-next code by removing Qwen3NextCo… (#10228) |
| 1228 | 93ac0bc1dc | Guoming Zhang | 2025-12-27 | [TRTLLM-10126][feat] Increase topk upper limit to 22 for NVLinkOneSid… (#10229) |
| 1229 | 27976fce9c | TensorRT LLM | 2025-12-27 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1230 | 55f3cda66d | Olya Kozlova | 2025-12-26 | [None][fix] Fix request_id for best_of/n case (#8368) |
| 1231 | c04563657e | Jin Li | 2025-12-27 | [TRTLLM-7735][feat] Attention NVFP4 out support for torch compile (#9740) |
| 1232 | d70aeddc7f | chenfeiz0326 | 2025-12-26 | [TRTLLM-8952][feat] Support Multi-Node Disagg Perf Test in CI (#9138) |
| 1233 | 684b37df02 | Pengyun Lin | 2025-12-26 | [https://nvbugs/5747938][fix] Use local tokenizer (#10230) |
| 1234 | c5b0f9e436 | Pengyun Lin | 2025-12-26 | [https://nvbugs/5633700][fix] Cache tiktoken vocab for gpt-oss (#10219) |
| 1235 | bfc591994c | dongfengy | 2025-12-26 | [https://nvbugs/5745152][fix] Fix some GPTOSS test setups (#10085) |
| 1236 | 4a5ef84dc2 | Jatin Gangani | 2025-12-26 | [None] [doc] Document perfect MoE router feature for perf analysis (#10303) |
| 1237 | 14554ab3f3 | Wanli Jiang | 2025-12-26 | [None][feat] Support multi-gpu running for nemotron-v3-nano and super (#10118) |
| 1238 | 819d03fa88 | TensorRT LLM | 2025-12-26 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1239 | 13ffe52ad0 | Enwei Zhu | 2025-12-26 | [None][fix] Allow YAML config overwriting CLI args for trtllm-eval (#10296) |
| 1240 | f3f02315df | Neta Zmora | 2025-12-25 | [None][chore]: small refactoring to auto-deploy MoE operator (#10300) |
| 1241 | db3430f589 | bhsueh_NV | 2025-12-26 | [None][feat] Support VLM part for Mistral Large 3 (#10188) |
| 1242 | 7e4cef9def | Jin Li | 2025-12-25 | [None][fix] Cherry-pick conflict changes for PR 7999 PR 8515 (#9446) |
| 1243 | d8b5aeb061 | Ziyi Xiong | 2025-12-25 | [https://nvbugs/5652062][fix] Rewind kv_cache and reset draft tokens (#10160) |
| 1244 | 46e4af5688 | ZhichenJiang | 2025-12-25 | [TRTLLM-9831][perf] Enable 2CTA with autotune for CuteDSL MoE and Grouped GEMM optimizations (#10201) |
| 1245 | fe12faef81 | Lizhi Zhou | 2025-12-25 | [https://nvbugs/5752516][chore] unwaive test; fix port conflicts in CI (#10152) |
| 1246 | cd5cd60ee4 | Iman Tabrizian | 2025-12-25 | [None][infra] Move install_boost from install_triton.sh to install_base.sh (#10055) |
| 1247 | 8462cf6c96 | Zhenhuan Chen | 2025-12-25 | [TRTLLM-9578][feat] make PDL enabled by default (#9695) |
| 1248 | 97b38ac403 | Jatin Gangani | 2025-12-25 | [None] [doc] Update IFB performance guide & GPTOSS deployment guide (#10283) |
| 1249 | 0ecdb69b93 | Emma Qiao | 2025-12-25 | [None][infra] Waive failed tests for main on 12/25 (#10298) |
| 1250 | 53b81783b1 | Xianjie Qiao | 2025-12-25 | [None][fix] Fix pageable H2D memcopy issue on GB200 (#10289) |
| 1251 | 83e02ee335 | Jie Li | 2025-12-25 | [None][chore] Remove NIM TRT-Backend Test Lists (#10232) |
| 1252 | 182b3eb633 | Enwei Zhu | 2025-12-25 | [None][ci] Waive TestLlama3_1_8B::test_auto_dtype[False-2] for timeout (#10293) |
| 1253 | 1d01214ff0 | Gabriel Wu | 2025-12-25 | [None][feat] Drop non-deepgemm fp8 block scale gemm (#10256) |
| 1254 | 4ae6f6a46c | xinhe-nv | 2025-12-25 | [None][chore] Add failed cases into waives.txt (#10249) |
| 1255 | 7395ca93b6 | heyuhhh | 2025-12-25 | [None][doc] Add Sparse Attention feature doc (#9648) |
| 1256 | c059e6caa1 | Venky | 2025-12-25 | [TRTC-121] [feat] Add recipe selector UI to complement the recipe database (#10125) |
| 1257 | a9eb5afc9f | gramnarayan | 2025-12-24 | [#9241][feat] AutoDeploy: Support Eagle3 Speculative Decoding (#9869) |
| 1258 | 1f8ed71d5f | TensorRT LLM | 2025-12-25 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1259 | 16fd781e42 | Emma Qiao | 2025-12-25 | [TRTLLM-9862][infra] Move single-gpu tests on rtxpro6000d to pre-merge (#9897) |
| 1260 | 43178590d1 | Ziyi Xiong | 2025-12-25 | [TRTLLM-10143][feat] Reuse previous draft requests if possible (#10263) |
| 1261 | c4b36d31ff | Neta Zmora | 2025-12-24 | [#10137][feat] AutoDeploy FP8 MoE refactor (#10138) |
| 1262 | 8614cd3439 | Necofish | 2025-12-24 | [None][fix] fix: resolve GPU memory imbalance in concurrent weight loading (#6472) |
| 1263 | e2891a6c77 | Suyog Gupta | 2025-12-24 | [#10052][feat] AutoDeploy enable cudagraphs for flashinfer BatchDecode (#10193) |
| 1264 | ddac4d7379 | Stanley Sun | 2025-12-24 | [None][test] Add disag-serving auto scaling qa test (#10262) |
| 1265 | 69152c4e7c | Yiqing Yan | 2025-12-24 | [None][infra] Check GB200 coherent GPU mapping (#10253) |
| 1266 | 56ef97e06e | tcherckez-nvidia | 2025-12-24 | [#10246][feature] Move AD dashboard to use cudagraph compile backend (#10267) |
| 1267 | ecea71ca7a | Jonas Li | 2025-12-24 | [None][chore] Update tinygemm kernel name (#10248) |
| 1268 | f4f0fe85e9 | shuyixiong | 2025-12-24 | [TRTLLM-9737][chore] Add rl perf reproduce script and enhance the robustness of Ray tests (#9939) |
| 1269 | 534700ecd9 | xinhe-nv | 2025-12-24 | [None][chore] Add failed cases into waives.txt (#10240) |
| 1270 | 595daa5089 | Yukun He | 2025-12-24 | [TRTLLM-9615][feat] Support synchronization through PP ranks in the distributed tuning system (#10011) |
| 1271 | 156f6453dc | Fanrong Li | 2025-12-24 | [TRTLLM-9798][feat] Change to use new DeepGEMM MQA sm100 kernel for MTP-3 (#10226) |
| 1272 | f6c3bc16b9 | zackyoray | 2025-12-24 | [None][docs] Add NIXL-Libfabric Usage to Documentation (#10205) |
| 1273 | 7b84e48e0f | Emma Qiao | 2025-12-24 | [None][infra] Waive failed cases om 12/24 (#10257) |
| 1274 | 68cf5c7924 | TensorRT LLM | 2025-12-24 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1275 | fc1f77eafc | xinhe-nv | 2025-12-24 | [None][chore] Add failed cases into waives.txt (#10204) |
| 1276 | 8c1cfc872b | Balaram Buddharaju | 2025-12-23 | [TRTLLM-9493][feat] Custom AllToAll for helix parallelism (#9986) |
| 1277 | 92d90fa29a | Jhao-Ting Chen | 2025-12-23 | [None][feat] Expose enable_trt_overlap in Triton_backend brings 1.05x OTPS (#10018) |
| 1278 | 0027a01ad5 | Grzegorz Kwasniewski | 2025-12-23 | [https://nvbugs/5680312][fix] Updated test waiving (#9630) |
| 1279 | 06900a7f19 | Grzegorz Kwasniewski | 2025-12-23 | [TRTLLM-9565][fix] Fix deepseek sharding (#9984) |
| 1280 | 984c20e0b2 | Emma Qiao | 2025-12-23 | [None][infra] Waive failed cases on 12/23 (#10236) |
| 1281 | e284d0bf80 | dongfengy | 2025-12-23 | [None][infra] Waive flaky unittest/executor/test_rpc_proxy.py and unittest/executor/test_rpc_worker.py tests (#10209) |
| 1282 | 64bb1a5155 | tcherckez-nvidia | 2025-12-23 | [None][chore] Update AD coverage to use torch-cudagraph (#10233) |
| 1283 | 8408c40d8b | Roey Azran | 2025-12-23 | [https://nvbugs/5702786][fix] Fix race conditions in KV cache communication during unexpected termination (#10076) |
| 1284 | 871c6b435c | Xianjie Qiao | 2025-12-23 | [None] [feat] skip batch_tokenize_prompts in CustomDataset (#10214) |
| 1285 | 522f1d2bc3 | Yukun He | 2025-12-23 | [https://nvbugs/5764627][chore] waive the time-out test (#10222) |
| 1286 | f2e00a75de | Balaram Buddharaju | 2025-12-23 | [None][chore] Remove helix test from rtx test list (#10224) |
| 1287 | 3ddc9d2b48 | Shiyu Li | 2025-12-23 | [https://nvbugs/5729697][fix] MNNVL Allreduce: use CUDA runtime instead of Macro to get SM version. (#10062) |
| 1288 | 48c875f8ea | chenfeiz0326 | 2025-12-23 | [None][fix] Add OpenSearch URL in slurm_launch.sh for Multinode Perf Sanity Test (#9990) |
| 1289 | cc1323be24 | Bo Li | 2025-12-23 | [None][fix] Fix the bug for top_k=10 in NVLinkOneSided AlltoAll. (#10197) |
| 1290 | 59b05dc0a8 | Yiqing Yan | 2025-12-23 | [None][chore] Bump version to 1.2.0rc7 (#10216) |
| 1291 | 53db3b2612 | Chuang Zhu | 2025-12-23 | [https://nvbugs/5741884][fix] unwaive disagg sampler (#10189) |
| 1292 | 77b591f73b | xinhe-nv | 2025-12-23 | [None][chore] Add failed cases into waives.txt (#10177) |
| 1293 | d691371eaf | Harshini Komali | 2025-12-22 | [TRTLLM-9091] [feat] Replace GenAI-Perf with AIPerf (#9310) |
| 1294 | 5bc7ffe379 | Pamela Peng | 2025-12-22 | [None][test] Add qa tests for RTX 6K (#10210) |
| 1295 | 18f8b22956 | TensorRT LLM | 2025-12-23 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1296 | 621156ad44 | fredricz-20070104 | 2025-12-23 | [None][chore] Fix GB300 support issues (#10196) |
| 1297 | 1e82ff7a0c | Li Min | 2025-12-23 | [TRTLLM-9989][fix] Fix tvm_ffi aaarch64 issue. (#10199) |
| 1298 | 696f754ef4 | Yuxian Qiu | 2025-12-23 | [None][fix] avoid implicit cudaStreamSynchronize in sample_async. (#10120) |
| 1299 | 648196f8ae | Tailing Yuan | 2025-12-23 | [TRTLLM-9432][feat] Reduce synchronization and recompilation for qwen3-next (#9691) |
| 1300 | f05af48bca | Faraz | 2025-12-22 | [https://nvbugs/5747674][fix] Add contiguous() before view() in load_expert_w3_w1_weight and load (#10136) |
| 1301 | 0d2500c631 | Fanrong Li | 2025-12-23 | [TRTLLM-9677][feat] Support DeepSeek-V3.2 tool parser (#10126) |
| 1302 | ccc64da287 | Grzegorz Kwasniewski | 2025-12-23 | [TRTLLM-9847][fix] WAR fix hanging fused allreduce. (#10087) |
| 1303 | 12e1cb8d7e | tcherckez-nvidia | 2025-12-22 | [#9717][chore] Refactor MoE code to use enums (#9910) |
| 1304 | aaa87abf41 | JunyiXu-nv | 2025-12-23 | [TRTLLM-7906][feat] Support multiple post process for Responses API (#9908) |
| 1305 | ba14a9308e | Emma Qiao | 2025-12-23 | [None][infra] Waive failed cases on 12/22 (#10200) |
| 1306 | 0f308e95f9 | Pengyun Lin | 2025-12-22 | [None][chore] Remove logprobs constraint on trtllm-serve pytorch backend (#9911) |
| 1307 | a6a88985cf | William Zhang | 2025-12-22 | [TRTLLM-9409][feat] Pass MRoPE tensors for EPD disagg (#9758) |
| 1308 | 472fe497dc | Bo Li | 2025-12-22 | [None][chore] NVLinkOneSided AlltoAll Support zero local_num_tokens. (#9822) |
| 1309 | ea6cd76c55 | Yan Chunwei | 2025-12-22 | [None][refactor] simplify get_stats and get_kvcache_events with rpc (#9980) |
| 1310 | c87f1a6b39 | Perkz Zheng | 2025-12-22 | [https://nvbugs/5503479][fix] update trtllm-gen kernels to address few bugs (#10089) |
| 1311 | 9e9523c3cc | shuyixiong | 2025-12-22 | [https://nvbugs/5762016][chore] Skip a ray test (#10194) |
| 1312 | 7421224d69 | JadoTu | 2025-12-22 | [None][fix] NVFP4 linear method's weight and weight_scale padding (#10148) |
| 1313 | d30ee8101e | xinhe-nv | 2025-12-22 | [None][chore] Remove closed bugs (#10182) |
| 1314 | 237fd0eae4 | Yuxian Qiu | 2025-12-22 | [https://nvbugs/5666821][chore] unwaive tests. (#9958) |
| 1315 | f8501f3cc8 | TensorRT LLM | 2025-12-22 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1316 | f0bd60a395 | Fanrong Li | 2025-12-22 | [https://nvbugs/5684820][fix] fix the detokenizer issue for DeepSeek-v3.2 (#10106) |
| 1317 | 066b653940 | Jin Li | 2025-12-22 | [TRTLLM-9880][feat] Include torch compile tests in QA test list (#10149) |
| 1318 | 2f139ee07e | Yuxian Qiu | 2025-12-22 | [https://nvbugs/5701445][chore] unwaive test. (#9949) |
| 1319 | 914dd39127 | Chuang Zhu | 2025-12-22 | [None][fix] disable cuda ipc on device without nvlink (L40s) for disagg test (#9735) |
| 1320 | d274a4c5d3 | dominicshanshan | 2025-12-22 | [https://nvbugs/5701457][fix] Unwaive ray test. (#10175) |
| 1321 | 5549067966 | Enwei Zhu | 2025-12-22 | [None][ci] Waive GPTOSS test case (#10155) |
| 1322 | 5266475014 | Balaram Buddharaju | 2025-12-21 | [None][feat] Cudagraph updates for helix parallelism (#10141) |
| 1323 | 4fc6036276 | shuyixiong | 2025-12-22 | [https://nvbugs/5702793][fix] Fix view operation on uncontiguous tensor (#10147) |
| 1324 | cd4b4f43fa | bhsueh_NV | 2025-12-21 | [None][feat] Support Eagle3 on Mistral Large3 (#9971) |
| 1325 | 5a611cb8f5 | Kaiyu Xie | 2025-12-21 | [None] [feat] Enhancements to slurm scripts (#10112) |
| 1326 | aa5dbb7ca5 | Emma Qiao | 2025-12-21 | [None][infra] Waive failed tests for main branch on 12/21 (#10184) |
| 1327 | 5ae154022a | xxi | 2025-12-21 | [TRTLLM-9872][fix] clear the failed test at CI when enalbe_configurab… (#10067) |
| 1328 | b15f987972 | Eran Geva | 2025-12-21 | [None][chore] removed duplicated test from l0_b200.yml (#10090) |
| 1329 | a66eeab537 | Bo Li | 2025-12-21 | [TRTLLM-9805][feat] Skip Softmax Attention. (#9821) |
| 1330 | dcd3f7b5ea | Balaram Buddharaju | 2025-12-20 | [https://nvbugs/5744427][fix] Fix accuracy test OOM (#10173) |
| 1331 | 6c76148b56 | TensorRT LLM | 2025-12-21 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1332 | 77e37d9dd0 | Bo Li | 2025-12-20 | [https://nvbugs/5753250][infra] Further waive all tests in _test_openai_responses.py (#10176) |
| 1333 | 2ce785f39a | Enwei Zhu | 2025-12-20 | [https://nvbugs/5643631][fix] Fix hostfunc seg fault (#10028) |
| 1334 | 21a93fbf9d | Enwei Zhu | 2025-12-20 | [TRTLLM-9992][perf] Enable PDL for CuteDSL kernels and overlap MoeOutputMemset (#10043) |
| 1335 | 3f25db9d3e | TensorRT LLM | 2025-12-20 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1336 | 3b3069b390 | Yuxian Qiu | 2025-12-20 | [https://nvbugs/5747930][fix] Use offline tokenizer for whisper models. (#10121) |
| 1337 | e75331480f | Yuxian Qiu | 2025-12-20 | [None][fix] fix draft_lengths for CUDA graph capture. (#10004) |
| 1338 | 7c82605327 | Anish Shanbhag | 2025-12-19 | [None][fix] enable KV cache reuse for config database (#10094) |
| 1339 | bee9051484 | Balaram Buddharaju | 2025-12-19 | [None][chore] Waive timing out pre-merge test (#10167) |
| 1340 | 20b69a982a | Gal Hubara-Agam | 2025-12-19 | [#10056][test] AutoDeploy: Add accuracy test for Nemotron SuperV3 (#10131) |
| 1341 | 5489d188a4 | Chang Liu | 2025-12-19 | [None][fix] Revert the change and remove device count guard for DSv32 (#9631) |
| 1342 | b882393d69 | longcheng-nv | 2025-12-20 | [https://nvbugs/5720357][fix] Fix indice offset overflow in custom Top-K kernel and corresponding UT case (#10027) |
| 1343 | dfa11d810e | Venky | 2025-12-20 | [TRTC-102][docs] `--extra_llm_api_options`->`--config` in docs/examples/tests (#10005) |
| 1344 | 7b71ff6b8a | JunyiXu-nv | 2025-12-20 | [https://nvbugs/5722653][fix] Unwaive fixed test (#10157) |
| 1345 | 27e49e2904 | xxi | 2025-12-19 | [None][fix] waive the failed test test_service_discovery[etcd-load_ba… (#10161) |
| 1346 | 9f6abaf59f | tcherckez-nvidia | 2025-12-19 | [#9640][feat] Migrate model registry to v2.0 format with composable configs (#9836) |
| 1347 | 7b51e3cedb | xinhe-nv | 2025-12-19 | [TRTLLM-8638][fix] Add failed cases into waives.txt (#10129) |
| 1348 | dd8ce68c94 | Emma Qiao | 2025-12-19 | [None][infra] Update waive and waive failed tests for main branch on 12/19 (#10151) |
| 1349 | ac03915dc3 | Pengyun Lin | 2025-12-19 | [TRTLLM-9604][feat] DS R1 & V3.1 tool parser (#10010) |
| 1350 | 31bc14b350 | Chang Liu | 2025-12-19 | [TRTLLM-9654][feat] Support DeepSeek-V32 chat template (#9814) |
| 1351 | 52cee573ad | yufeiwu-nv | 2025-12-19 | [TRTLLM-8830][test] Overlap scheduler enhancement perf test: Add qwen3_0,8b and llama3.1 test cases (#10114) |
| 1352 | cb0444b1b5 | xinhe-nv | 2025-12-19 | [TRTLLM-8638][fix] Add failed cases into waives.txt (#10132) |
| 1353 | 356ad4fe3a | JunyiXu-nv | 2025-12-19 | [https://nvbugs/5722653][fix] Address port conflict by assigning different port section in the same node. (#10035) |
| 1354 | 70b4d282c6 | Ziyi Xiong | 2025-12-19 | [TRTLLM-7736][feat] Incrementally update the inputs of target and draft models (#9708) |
| 1355 | 48dbc61129 | Larry Xu | 2025-12-19 | [None][chore] Update CODEOWNERS for test cases and test list (#10119) |
| 1356 | 478b6b20a1 | William Zhang | 2025-12-18 | [#9230][refactor] Replace nemotron patches with custom model implementation (#9751) |
| 1357 | 72c5480dfb | Balaram Buddharaju | 2025-12-18 | [None][chore] Waive test blocking pre-merge 12/18 (#10145) |
| 1358 | 00f70c30a6 | TensorRT LLM | 2025-12-19 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1359 | 9aa40871c2 | Ivy Zhang | 2025-12-19 | [TRTLLM-9840][test] switch ucx backend to default backend (#10101) |
| 1360 | a7ac5a6bca | TensorRT LLM | 2025-12-19 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1361 | 9f283f330b | Wangjue Yao | 2025-12-19 | [None][feat] Support Mooncake transfer engine as a cache transceiver backend (#8309) |
| 1362 | e0b2a94309 | Chuang Zhu | 2025-12-19 | [None][fix] Fix ready signal in NIXL backend (#10000) |
| 1363 | 2e88c86f10 | yuanjingx87 | 2025-12-18 | [None][infra] Fix issue that lock file geneartion will skip dependency with comment (#10144) |
| 1364 | bd5b3c2ac0 | Yukun He | 2025-12-19 | [https://nvbugs/5721912][chore] Unwaive the test (#10108) |
| 1365 | 91a9ae42d2 | Anish Shanbhag | 2025-12-18 | [TRTC-71][feat] Add regression testing for config database (#9832) |
| 1366 | 799a2ae311 | Balaram Buddharaju | 2025-12-18 | [https://nvbugs/5741331][fix] Fix helix accuracy test (#10021) |
| 1367 | a97e411b44 | Chang Liu | 2025-12-18 | [https://nvbugs/5747911][fix] Use offline data path for the unit test of mmencoder server (#10135) |
| 1368 | f02782a6f2 | Lizhi Zhou | 2025-12-19 | [https://nvbugs/5726066][fix] fix auto-scaling related failures (#9845) |
| 1369 | 6fe89ea00f | Enwei Zhu | 2025-12-19 | [TRTLLM-9819][perf] Reuse alltoall workspace for CuteDSL MoE output (#9840) |
| 1370 | 0b279f4ad4 | CarstyYou | 2025-12-18 | [https://nvbugs/5456493][feat] Add fp8 bmm on sm120 (#9687) |
| 1371 | 4e55b83101 | ZhichenJiang | 2025-12-18 | [None][perf] Add more optimization options for MOE CuteDSL finalized kernel (#10042) |
| 1372 | 3b4f26e4d1 | Nikita Korobov | 2025-12-18 | [None][feat] update TRT-LLM Gen MoE for NvFp4 + bias with tileN=256 (#9734) |
| 1373 | df15be3fad | yuanjingx87 | 2025-12-18 | [None][infra] Fix slurm job does not catch cancelled jobs (#9722) |
| 1374 | 9d7e038bcb | Bo Li | 2025-12-18 | [https://nvbugs/5753250][infra] Waive _test_openai_responses. (#10110) |
| 1375 | 33a90f2dd2 | Emma Qiao | 2025-12-18 | [None][infra] Waive failed cases for main branch on 12/18 (#10105) |
| 1376 | bec864a78c | Yuxian Qiu | 2025-12-18 | [None][fix] avoid ID conversion for non enable_configurable_moe cases. (#10003) |
| 1377 | 897a38978d | yuanjingx87 | 2025-12-17 | [None][infra] Update allowlist 2025.12.17 (#10097) |
| 1378 | 601c29ca73 | Wanli Jiang | 2025-12-18 | [https://nvbugs/5721644][fix] Update tests for nemotron_h (#9993) |
| 1379 | 76ec820465 | Lucas Liebenwein | 2025-12-17 | [#7532][feat] AutoDeploy: gather logits before lm head (#9962) |
| 1380 | cfe53e7425 | TensorRT LLM | 2025-12-18 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1381 | 4a98f190a8 | xinhe-nv | 2025-12-18 | [None][chore] Add failed cases into waives.txt (#10025) |
| 1382 | c1cfb61b1b | xinhe-nv | 2025-12-18 | [TRTLLM-9381][feat] Add kimi k2 fp4 tests (#9906) |
| 1383 | 50c2b82f24 | TensorRT LLM | 2025-12-17 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1384 | 27064f95c7 | tburt-nv | 2025-12-17 | [None][chore] Clarify copyright header guidance (#9882) |
| 1385 | 5da7879b38 | tburt-nv | 2025-12-17 | [None][fix] Revert GHA upgrade for blossom-ci workflow (#10095) |
| 1386 | 22c6e8a424 | Chenghao Zhang | 2025-12-17 | [None][fix] Autodeploy: fix some legacy flashinfer attention test errors (#9928) |
| 1387 | cb5cd4376e | Salman Chishti | 2025-12-17 | [None][chore] Upgrade GitHub Actions for Node 24 compatibility (#10045) |
| 1388 | f7e245668b | Yuan Tong | 2025-12-17 | [TRTLLM-9680][perf] Optimize TRTLLMSampler log_probs performance (Core fix has been merged via #9353) (#9655) |
| 1389 | 00c0564334 | Yukun He | 2025-12-17 | [None][chore] Remove unnecessary warning log for tuning. (#10077) |
| 1390 | 18b335d584 | Yukun He | 2025-12-17 | [TRTLLM-9989][fix] Disable tvm_ffi for CuteDSL nvFP4 dense GEMM. (#10040) |
| 1391 | 2fd1a23e4c | Yukun He | 2025-12-17 | [TRTLLM-9998][fix] Change trtllm-gen MoE distributed tuning strategy back to INDEPENDENT (#10036) |
| 1392 | 5d71f662c3 | yufeiwu-nv | 2025-12-17 | [https://nvbugs/5698434][test] Add Qwen3-4B-Eagle3 One-model perf test (#10041) |
| 1393 | 47404196fa | Void | 2025-12-17 | [None][fix] Enabled simultaneous support for low-precision combine and MTP. (#9091) |
| 1394 | 0dbf3948cc | Emma Qiao | 2025-12-17 | [None][infra] Waive failed tests due to llm model files (#10068) |
| 1395 | 02fd13448b | Kaiyu Xie | 2025-12-17 | [None] [feat] Enhancements to slurm scripts (#10031) |
| 1396 | 6649c3743c | JunyiXu-nv | 2025-12-17 | [https://nvbugs/5635153][chore] Remove responses tests from waive list (#10026) |
| 1397 | 26fb063076 | shuyixiong | 2025-12-17 | [https://nvbugs/5741060][fix] Fix pg op test (#9989) |
| 1398 | 7175d89b48 | Aurelien Chartier | 2025-12-16 | [None][fix] Fix iteration stats for spec-dec (#9855) |
| 1399 | dba9036072 | QI JUN | 2025-12-11 | [None][doc] remove nano-vl-v2 model support in release notes (#9887) |
| 1400 | 3daca4fea3 | QI JUN | 2025-12-10 | [https://nvbugs/5729847][doc] fix broken links to modelopt (#9868) |
| 1401 | e6ab864066 | QI JUN | 2025-12-10 | [None][doc] Update release notes (#9739) |
| 1402 | 1ffa2c8937 | Zac Patel | 2025-12-09 | [IB-1920][doc] Update Perf_Overview.md with Benchmarking Results for Release 1.1 (#9723) |
| 1403 | 2756a0da60 | xiweny | 2025-12-03 | [TRTLLM-4629][doc] Add B300 & GB300 in documents (#9663) |
| 1404 | 07f307d131 | ruodil | 2025-12-03 | [https://nvbugs/5652552][fix] cherry-pick add printing for llm args (#9206) |
| 1405 | 1fc8bd3cd8 | Iman Tabrizian | 2025-12-02 | [TRTLLM-9082][doc] Address Dynamo Example feedback (#9619) |
| 1406 | e41b060fe6 | Kaiyu Xie | 2025-12-02 | [TRTLLM-9090] [doc] Update online benchmarking docs (#9611) |
| 1407 | bd13957e70 | Lizhi Zhou | 2025-12-16 | [TRTLLM-9181][feat] improve disagg-server prometheus metrics; synchronize workers' clocks when workers are dynamic (#9726) |
| 1408 | 609d1d0383 | Enwei Zhu | 2025-12-16 | [None][fix] Fix Illegal Memory Access for CuteDSL Grouped GEMM (#10008) |
| 1409 | 6a238ca8ad | Enwei Zhu | 2025-12-16 | [None][doc] Update CONTRIBUTING.md (#10023) |
| 1410 | 12727ebd7f | Emma Qiao | 2025-12-16 | [None][infra] Waive failed test for main branch on 12/16 (#10029) |
| 1411 | 064b67e40c | Perkz Zheng | 2025-12-16 | [https://nvbugs/5727952][fix] a pdl bug in trtllm-gen fmha kernels (#9913) |
| 1412 | 0a4c59136a | yuanjingx87 | 2025-12-15 | [None][infra] Fixing credential loading in lockfile generation pipeline (#10020) |
| 1413 | 28b02b4f5a | William Zhang | 2025-12-15 | [None][docs] Add README for Nemotron Nano v3 (#10017) |
| 1414 | 6b5ebaae3e | Yihan Wang | 2025-12-16 | [None][chore] Update internal_cutlass_kernels artifacts (#9992) |
| 1415 | 8af51211c1 | Wanli Jiang | 2025-12-16 | [FMDL-1222][feat] Support weight and weight_scale padding for NVFP4 MoE cutlass (#9358) |
| 1416 | ce7a42f4cf | Eran Geva | 2025-12-16 | [https://nvbugs/5731717][fix] fixed flashinfer build race condition during test (#9983) |
| 1417 | 8ba8699f66 | Yechan Kim | 2025-12-16 | [TRTLLM-8310][feat] Add Qwen3-VL-MoE (#9689) |
| 1418 | dff77efa2a | ChristinaZ | 2025-12-16 | [None][feat] Add routing support for the new model for both cutlass and trtllm moe backend (#9792) |
| 1419 | 4ce35eacf1 | QI JUN | 2025-12-16 | [TRTLLM-9794][ci] move more test cases to gb200 (#9994) |
| 1420 | cdf56c278f | xinhe-nv | 2025-12-16 | [TRTLLM-8638][fix] Add failed cases into waives.txt New activity. (#9979) |
| 1421 | b757ea73ba | Zhanrui Sun | 2025-12-16 | [TRTLLM-9641][infra] Use public triton 3.5.0 in SBSA (#9652) |
| 1422 | e6187d8109 | Michal Guzek | 2025-12-15 | [https://nvbugs/5708810][fix] Fix TRTLLMSampler (#9710) |
| 1423 | 9ba14263db | Patrice Castonguay | 2025-12-15 | [https://nvbugs/5673559][fix] Unwaiving disagg test for nvbug 5673559 (#9957) |
| 1424 | d5d15c06df | Emma Qiao | 2025-12-16 | [None][infra] Waive failed tests for main branch on 12/15 (#10001) |
| 1425 | 0c31502fbc | Faraz | 2025-12-15 | [None][feat] disable fused gemm for sm121 (#9916) |
| 1426 | 44b0f8c3ed | Kaiyu Xie | 2025-12-16 | [None] [fix] Revert "[None] [feat] add eos_token_id in generation_config to sampling params" (#10002) |
| 1427 | 63e7a2fa70 | zackyoray | 2025-12-15 | [None][infra] Update ucx to 1.20.x (#9977) |
| 1428 | 4f75a31a45 | arekay-nv | 2025-12-15 | [https://nvbugs/5540979][fix] Potential fix for 5540979 (#9716) |
| 1429 | 3230fbe79a | Wanli Jiang | 2025-12-15 | [None][feat] Update reasoning parser for nano-v3 (#9944) |
| 1430 | 9e7182b603 | Yukun He | 2025-12-15 | [TRTLLM-9615][feat] Implement a distributed tuning system (#9621) |
| 1431 | ef4ea955b2 | Kaiyu Xie | 2025-12-15 | [None] [fix] Fix slrum scripts (#10007) |
| 1432 | ad12b795c9 | Anthony Chang | 2025-12-15 | [https://nvbugs/5661741][fix] Fix accuracy issue in TRTLLM MoE introduced in #9377 (#9999) |
| 1433 | 9eb5a229dd | Bo Li | 2025-12-15 | [None][infra] Fully waive test_worker_restart test_disagg_server_restart. (#9988) |
| 1434 | 83885c69e7 | Grzegorz Kwasniewski | 2025-12-15 | [TRTLLM-9136][feat] 2D parallel EP TP support (#9459) |
| 1435 | 825025b137 | dominicshanshan | 2025-12-15 | [None][infra] Add multi gpu Ray tests into L0 merge change request list. (#9996) |
| 1436 | 3c98b25005 | xinhe-nv | 2025-12-15 | [None][chore] Add failed cases into waives.txt (#9941) |
| 1437 | 504ede707e | Kaiyu Xie | 2025-12-15 | [None] [fix] Fix nsys_on argument for slurm scripts (#9995) |
| 1438 | dda7658306 | Void | 2025-12-15 | [https://nvbugs/5655885][fix] fix invalid instruction error in 2shot ar kernel on Ampere (#9394) |
| 1439 | 7588029763 | Yuxian Qiu | 2025-12-15 | [None][feat] Async pp send for PPCommTorch. (#9976) |
| 1440 | af899d2fe7 | JunyiXu-nv | 2025-12-15 | [TRTLLM-9860][doc] Add docs and examples for Responses API (#9946) |
| 1441 | f2aee0db03 | Ziyi Xiong | 2025-12-15 | [TRTLLM-9854][feat] Optimize the host overhead of _sample_async (#9935) |
| 1442 | 25db9e7b3e | shuyixiong | 2025-12-15 | [https://nvbugs/5741060][chore] Waive all pg operator tests (#9991) |
| 1443 | dfc8799352 | Balaram Buddharaju | 2025-12-14 | [https://nvbugs/5669114][fix] Switch to MMMU benchmark for Gemma3 27B (#9966) |
| 1444 | 8f144d9282 | Fanrong Li | 2025-12-15 | [TRTLLM-9416][feat] Skip DS-v3.2 indexer MQA and Top-K for short sequences. (#9524) |
| 1445 | 0788635d6c | Kaiyu Xie | 2025-12-15 | [TRTLLM-9762] [doc] Update documents for GB300 NVL72 (#9987) |
| 1446 | b57650f1e6 | QI JUN | 2025-12-15 | [TRTLLM-9794][ci] move test cases of gpt-oss to gb200 (#9934) |
| 1447 | f5696df285 | xxi | 2025-12-15 | [TRTLLM-8961][feat] ConfigurableMoE support DeepGemm (#9858) |
| 1448 | 355e06d66d | Yan Chunwei | 2025-12-15 | [None][doc] update readme for rpc (#9972) |
| 1449 | 4bf42f8fa8 | dominicshanshan | 2025-12-15 | [https://nvbugs/5580297][fix] Skip capture request error test from Ray stage (#9947) |
| 1450 | 3be5f3abcf | Anthony Chang | 2025-12-15 | [None][fix] Fix regex pattern for cubin filtering (#9914) |
| 1451 | bf923a1074 | Zongfei Jing | 2025-12-15 | [None] [chore] Comments cleanup (#9978) |
| 1452 | f21e2b3329 | Simeng Liu | 2025-12-14 | [TRTLLM-9601][feat] Expose mmKeys for multimodal to integrate with dynamo. (#9604) |
| 1453 | 9a1750c8f9 | Balaram Buddharaju | 2025-12-14 | [TRTLLM-9493][noop] Refactor fusedMoeCommKernels to enable code sharing (#9922) |
| 1454 | e0a4b72279 | Emma Qiao | 2025-12-14 | [None][infra] Waive failed tests for main branch on 12/14 (#9982) |
| 1455 | 1375910f1b | Matt Lefebvre | 2025-12-14 | [None][infra] Delete container before attempting import (#9967) |
| 1456 | 96d654029d | Mike Iovine | 2025-12-14 | [https://nvbugs/5666816][fix] Unwaive llama3 eagle3 test (#9964) |
| 1457 | fcda1a1442 | Yuxian Qiu | 2025-12-14 | [None][fix] disable async pp send for ray cases. (#9959) |
| 1458 | f6b0ddd61d | TensorRT LLM | 2025-12-14 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1459 | a5a37227d6 | nvxuanyuc | 2025-12-13 | [None][feat] Fused kernels (qknormrope + moe routing) and two-model MTP support for glm4moe (#9852) |
| 1460 | 64d7796234 | Faraz | 2025-12-13 | [None][chore] Add namespace to header to fix tot failure (#9973) |
| 1461 | 383b13e0e5 | Mike Iovine | 2025-12-13 | [None][feat] Implement sampling on 1-model EAGLE3 (#9885) |
| 1462 | 079ef8ae77 | jellysnack | 2025-12-13 | [None][feat] Graceful Error Handling for Guided Decoder (#9078) |
| 1463 | 85406f9dda | Yan Chunwei | 2025-12-13 | [https://nvbugs/5720482][fix] Fix test rpc streaming (#9902) |
| 1464 | 8cbf2d958c | shuyixiong | 2025-12-13 | [TRTLLM-9738][chore] Guard accuracy with nccl allreduce strategy (#9793) |
| 1465 | 6a6e41f802 | Balaram Buddharaju | 2025-12-12 | [TRTLLM-9468][chore] Update disagg benchmarking scripts to support context parallelism (#9720) |
| 1466 | 7fc720a397 | shuyixiong | 2025-12-13 | [TRTLLM-9784][fix] Resolve port conflicts (#9780) |
| 1467 | e49c70f6df | bhsueh_NV | 2025-12-13 | [None][feat] Support Mistral Large3 LLM part (#9820) |
| 1468 | 98d72c7648 | Faraz | 2025-12-12 | [None][feat] spark cublas LUT table for llama-8b-bf16 perf (#9811) |
| 1469 | e4e09867d1 | TensorRT LLM | 2025-12-13 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1470 | 461446045e | Balaram Buddharaju | 2025-12-12 | [TRTLLM-9493][feat] Add helixPostProcessNative kernel for cp_dim=2 (#9924) |
| 1471 | 6147452158 | tburt-nv | 2025-12-12 | [https://nvbugs/4141427][chore] Add more details to LICENSE file (#9881) |
| 1472 | 246a877571 | yuanjingx87 | 2025-12-12 | [None][infra] Remove generate lockfile schedule for 1.2.0rc4.post1 branch (#9945) |
| 1473 | cd4e639536 | Yuxian Qiu | 2025-12-13 | [None][feat] Async pp send. (#9952) |
| 1474 | 4cc4cbe926 | Chuang Zhu | 2025-12-13 | [https://nvbugs/5716787][fix] terminate nixl running when exiting (#9785) |
| 1475 | 9c59c9f920 | Chuang Zhu | 2025-12-13 | [https://nvbugs/5643787][fix] remove the war path for notify to itself (#9834) |
| 1476 | 2fec53dfa5 | JunyiXu-nv | 2025-12-12 | [TRTLLM-9637][feat] Support tool parser for Kimi K2 (#9830) |
| 1477 | 9df4dad3b6 | Yihan Wang | 2025-12-12 | [None][fix] Introduce inline namespace to avoid symbol collision (#9541) |
| 1478 | af315d8ef1 | Balaram Buddharaju | 2025-12-12 | [TRTLLM-5972][chore] Load balance decode token KV cache with helix parallelism (#9757) |
| 1479 | d5b9ad91c9 | zackyoray | 2025-12-12 | [None][feat] Upgrade NIXL to v0.8.0 (#9707) |
| 1480 | e767fc649a | Lucas Liebenwein | 2025-12-12 | [None][feat] AutoDeploy: prepare_metadata revisited (#9764) |
| 1481 | a6263a127f | Yukun He | 2025-12-12 | [None][chore] Degrade log level in cublas fp4 runner when using default configs (#9951) |
| 1482 | 9b3e5e90ee | ruodil | 2025-12-12 | [None][test] fix a typo in model name in script (#9867) |
| 1483 | 61745f034a | chenfeiz0326 | 2025-12-12 | [https://nvbugs/5727481][ci] Fix Port Conflict in Perf-Sanity CI Test (#9896) |
| 1484 | 2fc94e5dd7 | kris1025 | 2025-12-12 | [None][chore] unwaive qwen3 accuracy test (#9895) |
| 1485 | fd3d3a553d | yufeiwu-nv | 2025-12-12 | [None][chore] Modify python ipc_util to align with C++ path (#9894) |
| 1486 | 711016c799 | Yihan Wang | 2025-12-12 | [https://nvbugs/5736923][infra] Waive timeout disaggregated/test_auto_scaling[http-round_robin] test (#9942) |
| 1487 | eeb03f314a | yuanjingx87 | 2025-12-11 | [None][infra] Replace the deprecated github token (#9915) |
| 1488 | 9d1f2a9925 | Yifei Wang | 2025-12-11 | [#6425][fix] address CUDA stream sync issue in ModelRunnerCPP (#6426) |
| 1489 | fded6c393d | Ivy Zhang | 2025-12-12 | [TRTLLM-9262][test] add groupgemm ada case for rcca (#9833) |
| 1490 | 110820bb15 | Kaiyu Xie | 2025-12-12 | [TRTLLM-9792] [feat] Support multiple instances on single node for slurm scripts (#9900) |
| 1491 | bd441e9822 | Chuang Zhu | 2025-12-12 | [None][infra] revert ucx to 1.19 (#9936) |
| 1492 | 3e39afea9a | Yiteng Niu | 2025-12-12 | [None][infra] update nspect version for api change (#9899) |
| 1493 | 093465ed29 | dominicshanshan | 2025-12-12 | [https://nvbugs/5599176][fix] Unwaive fixed test for Ray (#9861) |
| 1494 | 0132769c22 | TensorRT LLM | 2025-12-12 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1495 | 5065b60cd1 | Yiqing Yan | 2025-12-12 | [None][infra] Fix mergeWaiveList stage (#9892) |
| 1496 | e8efeb765d | xinhe-nv | 2025-12-12 | [TRTLLM-9717][fix] fix multi nodes tests cases (#9736) |
| 1497 | 4670e0c297 | Chuang Zhu | 2025-12-12 | [None][infra] update ucx to 1.20 (#9786) |
| 1498 | 710c592d7c | JunyiXu-nv | 2025-12-12 | [https://nvbugs/5727517][fix] Preserve ip:port for disagg (#9859) |
| 1499 | 98c68c195b | Kanghwan | 2025-12-11 | [None][infra] Ignore comments from bots and CI accounts (#9929) |
| 1500 | 4f6d4da035 | jthomson04 | 2025-12-11 | [None][perf] Fix TPOT when `min_tokens` set (#9862) |
