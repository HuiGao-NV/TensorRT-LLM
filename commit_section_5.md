# Commit Section 5

Commits 2001 to 2500 (total: 5612)

| # | Commit Hash | Author | Date | Message |
|---|------------|--------|------|--------|
| 2001 | 177ba7b0f1 | Kaiyu Xie | 2025-11-13 | [None] [fix] Disable UCC as WAR to MPI allgather issue before NGC PyTorch 25.12 upgrade (#9126) |
| 2002 | 48a27c7bef | Lizhi Zhou | 2025-11-13 | [https://nvbugs/5633340][chore] unwaive test_auto_scaling.py::test_disagg_server_restart (#9131) |
| 2003 | d0ea417ec8 | Emma Qiao | 2025-11-13 | [None][infra] Waive failed tests for main 11/13 (#9132) |
| 2004 | 548f5ce4bc | xinhe-nv | 2025-11-13 | [None][fix] waive failed tests (#9090) |
| 2005 | 8fa3c55c76 | xinhe-nv | 2025-11-13 | [None][chore] Remove closed bugs (#9114) |
| 2006 | c86e36fe38 | ruodil | 2025-11-13 | [None][test] add deepseek and qwen cases for rtx series (#8839) |
| 2007 | c37924f37b | Chang Liu | 2025-11-12 | [None][fix] Clear indexer k cache reference before release cuda memory (#9110) |
| 2008 | cde18c12da | HuiGao-NV | 2025-11-13 | [https://nvbugs/5640873][fix] Move thop tests to pre-merge (#9094) |
| 2009 | 22c1748b80 | Perkz Zheng | 2025-11-13 | [TRTLLM-8816][feat] add optimized trtllm-gen attention kernels on sm103 (#9081) |
| 2010 | 49df731b96 | Zhang Ge | 2025-11-13 | [#6507][fix] Fix precision issue due to KV layout mismatch for split/concat kernels (#6917) |
| 2011 | 4fd93bdc2c | Yan Chunwei | 2025-11-13 | [None][ci] Waive test_llm_rpc and test_llm_rpc_streaming (#9118) |
| 2012 | 3ab24df815 | cheshirekow | 2025-11-12 | [TRTLLM-9209][infra] Upgrade precommit-hooks to v6.0.0 (#9097) |
| 2013 | fc5a28c1db | TensorRT LLM | 2025-11-13 | [None][infra] Check in most recent lock file from nightly pipeline |
| 2014 | c79b27851d | Venky | 2025-11-12 | [None] [infra] Update CODEOWNERS for pre-commit-config.yaml (#9108) |
| 2015 | 8a8883bc73 | Yan Chunwei | 2025-11-13 | [None][chore] Waive test_llm_rpc_streaming (#9113) |
| 2016 | d1b003d31e | QI JUN | 2025-11-13 | [TRTLLM-9212][chore] move MoeLoadBalancerConfig to llm_args.py (#9002) |
| 2017 | 943b05e2d3 | Zhenhuan Chen | 2025-11-13 | [TRTLLM-9179][feat] add pp_partition to customize each rank's layer number (#9003) |
| 2018 | 3416efbc29 | QI JUN | 2025-11-13 | [None][ci] waive test_disaggregated_serving.py::TestQwen3_8B::test_chunked_prefill (#9111) |
| 2019 | f1d637ec69 | Chenghao Zhang | 2025-11-12 | [None][fix] AutoDeploy: Use tmp folder for the load_moe_align (#9101) |
| 2020 | 9241ccaf27 | dongxuy04 | 2025-11-13 | [None][feat] Enable EPLB for trtllm-gen and cutlass backend (#8886) |
| 2021 | 5f26c31954 | Chenghao Zhang | 2025-11-12 | [https://nvbugs/5636912][fix] AutoDeploy: Unwaive the test (#9018) |
| 2022 | 8a751a0e56 | Patrice Castonguay | 2025-11-12 | [None][chore] Remove is_disaggregated param in executor request queue (#9049) |
| 2023 | 780d4f9dc5 | Fanrong Li | 2025-11-13 | [None][feat] Add MTP>1 support for DS-v3.2 (#9045) |
| 2024 | 53491ffdb1 | Neta Zmora | 2025-11-12 | [#9023][feat] reduce AD graph optimization time for non-participating passes (#9024) |
| 2025 | cdde15b275 | Iman Tabrizian | 2025-11-12 | [TRTLLM-8540][feat] Add support for disagg in DSv3.2 (#8735) |
| 2026 | 264d38e6c5 | mpikulski | 2025-11-12 | [TRTLLM-9175][test] ensure sampling is async (#9076) |
| 2027 | b7a2574c60 | yufeiwu-nv | 2025-11-12 | [https://nvbugs/5568991][test] Remove Phi-3 models (#9066) |
| 2028 | 96132b4274 | Timothy Gao | 2025-11-11 | [None] [doc] Add Mixed Precision Context and Generation section to Disagg (#8769) |
| 2029 | 4003dc7574 | QI JUN | 2025-11-12 | [None][ci] waive some test cases of disaggregated serving (#9085) |
| 2030 | bb6eb9510d | Emma Qiao | 2025-11-12 | [None][infra] Waive a failed case of disaggregated/test_disaggregated.py (#9074) |
| 2031 | 0b25d240a1 | Zhanrui Sun | 2025-11-12 | [TRTLLM-9018][infra] add mirror for Build-Docker-Images stage (#9063) |
| 2032 | 1af9b2ec6a | TensorRT LLM | 2025-11-12 | [None][infra] Check in most recent lock file from nightly pipeline |
| 2033 | 1a56722697 | Jiagan Cheng | 2025-11-12 | [None][fix] Remove unnecessary attention workspace memory check (#9064) |
| 2034 | fd703fbb7b | QI JUN | 2025-11-12 | [None][ci] run speculative unit tests serially (#9080) |
| 2035 | 0b81173efa | Chang Liu | 2025-11-11 | [TRTLLM-9259][perf] Use torch.compile to fuse copy + layernorm within the LayerNorm module (#9052) |
| 2036 | aca56097cb | Lucas Liebenwein | 2025-11-11 | [None][fix] AutoDeploy: update nano3 accuracy test (#9061) |
| 2037 | 524754b6fd | QI JUN | 2025-11-12 | [TRTLLM-8521][chore] remove circular dependency between model engine and cuda graph runner (#7572) |
| 2038 | ec9cf715a2 | Chenghao Zhang | 2025-11-11 | [None][feat] AutoDeploy: Perf improvement for mamba layers (#8991) |
| 2039 | ebdd1cc8e0 | Wanli Jiang | 2025-11-11 | [TRTLLM-8119][feat] Update doc/tests/chat_template for nano-v2-vlm (#8840) |
| 2040 | 20fd305bb6 | mpikulski | 2025-11-11 | [None][fix] type annotation (#9071) |
| 2041 | b151de4a8f | mpikulski | 2025-11-11 | [TRTLLM-8377][test] unit tests for TorchSampler batched sampling (#9012) |
| 2042 | b894dc2d70 | Guoming Zhang | 2025-11-11 | [None][fix] Display the GPU memory information in GiB unit. (#9070) |
| 2043 | 979b3ae9ce | mpikulski | 2025-11-11 | [TRTLLM-7723][feat] sampling using FlashInfer.sampling (#8581) |
| 2044 | 23c388c58b | HuiGao-NV | 2025-11-11 | [https://nvbugs/5616189][fix] Make more cases use local cached models (#8935) |
| 2045 | 22f1523f9e | Emma Qiao | 2025-11-11 | [None][infra] Only print and don't fail the check if there are duplicated items in waives.txt (#9068) |
| 2046 | 0ce22ce928 | QI JUN | 2025-11-11 | [None][ci] waive test_disaggregated_serving.py::TestQwen3_8B::test_auto_dtype[False] (#9069) |
| 2047 | 62a30bca25 | elvischenv | 2025-11-11 | [None][chore] Add tensorrt_llm/scripts to .gitignore (#8895) |
| 2048 | b7d51c5549 | Yiqing Yan | 2025-11-11 | [None][chore] Remove duplicated waive test (#9067) |
| 2049 | 7aeac97e4e | Yuxian Qiu | 2025-11-11 | [https://nvbugs/5622938][fix] Use async send_requests_to_next_pp. (#9041) |
| 2050 | 6bf4e59267 | Lucas Liebenwein | 2025-11-10 | [#8763][feature] AutoDeploy: configurable dtype for caching (#8812) |
| 2051 | de6088e363 | jiahanc | 2025-11-10 | [None][doc] update llama and llama4 example doc (#9048) |
| 2052 | 0b9bc5aae8 | Bo Deng | 2025-11-11 | [None][infra] install mooncake in docker images (#8447) |
| 2053 | da1f0e2465 | Emma Qiao | 2025-11-11 | [None][infra] Waive failed tests on main 11/11 (#9058) |
| 2054 | fac522056c | xinhe-nv | 2025-11-11 | [None][chore] Add failed cases into waives.txt (#8998) |
| 2055 | 7ceb5e5ab6 | Chang Liu | 2025-11-10 | [TRTLLM-9198][perf] Add torch.compile + multi-stream support for k-cache scatter and weight scaling (#8988) |
| 2056 | c61b44e594 | TensorRT LLM | 2025-11-11 | [None][infra] Check in most recent lock file from nightly pipeline |
| 2057 | 1ccb799c9a | shuyixiong | 2025-11-11 | [None][chore] Relocate rlhf_utils.py (#8938) |
| 2058 | 972c21c142 | dongfengy | 2025-11-10 | [None][chore] Clean up unused and confusing code in moe test  (#9019) |
| 2059 | 1fd11455d8 | Liao Lanyu | 2025-11-11 | [https://nvbugs/5556998][fix] init_hf_modules in worker_main for models with trust_remote=true (#8931) |
| 2060 | 0938a3ad2a | Yechan Kim | 2025-11-11 | [https://nvbugs/5644187][fix] Llava-Next MMMU bugfix and Phi4 test bugfix (#9034) |
| 2061 | f40e1f7496 | Frida Hou | 2025-11-10 | [https://nvbugs/5625972][fix] Add context manager to fix FakeTensorProp (#9047) |
| 2062 | 50c486367a | xiweny | 2025-11-11 | [https://nvbugs/5619396][fix] Add sm103 to CutlassFP8RowwiseGemm (#9042) |
| 2063 | edc91ba819 | mpikulski | 2025-11-10 | [None][fix] Improve type annotations on ResourceManager.get_resource_manager (#9013) |
| 2064 | 2e7769d1e8 | ChristinaZ | 2025-11-10 | [None][feat] Add customized topk and related unit tests for DSA (#8882) |
| 2065 | f848d844d9 | xinhe-nv | 2025-11-10 | [None][chore] Add failed cases into waives.txt (#9030) |
| 2066 | e8d4a56dd0 | bhsueh_NV | 2025-11-10 | [None][fix] fix eagle3 accuracy issue on sm120 (#8944) |
| 2067 | a7033a9193 | Fanrong Li | 2025-11-10 | [TRTLLM-9001][feat] add TP support for DeepSeek-V3.2 (#8943) |
| 2068 | 78fac1f665 | Yiqing Yan | 2025-11-10 | [None][chore] Lock onnx version <1.20.0 and remove WAR for TRT 10.13 (#9006) |
| 2069 | 67af7c15a5 | Bo Li | 2025-11-10 | [https://nvbugs/5637037][fix] Update unwaive list. (#9001) |
| 2070 | 183778d58a | Emma Qiao | 2025-11-09 | [None][infra] Waive failed tests for main 11/07 (#9008) |
| 2071 | 2af6a537ad | Emma Qiao | 2025-11-08 | [TRTLLM-8999][infra] Reduce gb200 multi-node test stages (#8778) |
| 2072 | 533add5056 | mpikulski | 2025-11-08 | [TRTLLM-8598][feat] enable n > 1 in OpenAI API with PyTorch backend (#8951) |
| 2073 | 6ff82ea24e | hvagadia | 2025-11-07 | [None][feat] Allow env variable to specify spawn process IPC address (#8922) |
| 2074 | 748c56a036 | yuanjingx87 | 2025-11-07 | [None][infra] Update allowed list 2025.11.06 (#8987) |
| 2075 | 7081f254cf | Chang Liu | 2025-11-07 | [None][perf] Add custom indexer k cache scatter op (#8960) |
| 2076 | c232ffd122 | Guoming Zhang | 2025-11-08 | [None][doc] Replace the relative links with absolute links in README.md. (#8995) |
| 2077 | d8ea0b967f | Patrice Castonguay | 2025-11-07 | [None][fix] Moving transfer timeout test to test_llm_pytorch, fixing broken kv transfer timeout (#8892) |
| 2078 | 7b82ba90da | Yuxian Qiu | 2025-11-07 | [https://nvbugs/5629790][chore] unwaive test. (#8967) |
| 2079 | e53be1564a | Zhanrui Sun | 2025-11-07 | [TRTLLM-9213][infra] Fix boost issue (#8996) |
| 2080 | c836ae5aaa | Yiqing Yan | 2025-11-07 | [None][chore] Bump version to 1.2.0rc3 (#9004) |
| 2081 | 1944fb15af | mpikulski | 2025-11-07 | [None][fix] add missing CLI option in multimodal example (#8977) |
| 2082 | 5ef65872a3 | mpikulski | 2025-11-07 | [None][fix] type annotations in fuse_input_embeds (#8976) |
| 2083 | 326a201473 | Stefan Niebler | 2025-11-07 | [https://nvbugs/5508536][fix] Take Over (#8627): Reintroduce: Move stop_criteria to sample_async (#7041) (#8794) |
| 2084 | 1c6e490894 | QI JUN | 2025-11-07 | [TRTLLM-9065][chore] remove PyTorchConfig completely (#8856) |
| 2085 | b26e1617f2 | Lizhi Zhou | 2025-11-07 | [https://nvbugs/5633340][fix] kill processes properly after test (#8970) |
| 2086 | 990e674b71 | Eran Geva | 2025-11-07 | [None][fix] Switch AD AllReduce strategy to NCCL (#8979) |
| 2087 | ee20e679a9 | xiweny | 2025-11-07 | [https://nvbugs/5636986][fix] Fix DeepGemmMoe get_buffer calls (#8939) |
| 2088 | b53961e972 | Cao Dong | 2025-11-07 | [None][feat] Return logprobs incrementally in torch backend (#8785) |
| 2089 | 9f8d93f89a | Simeng Liu | 2025-11-06 | [https://nvbugs/5606136][ci] Remove tests for deprecating triton multimodal models. (#8926) |
| 2090 | 1c19fd6868 | Chang Liu | 2025-11-06 | [https://nvbugspro.nvidia.com/bug/5637012][fix] Bugfix when config is None for MLA (#8978) |
| 2091 | fcae852cef | jthomson04 | 2025-11-06 | [None][fix] Fix KV cache clearing with KV Connector API (#8750) |
| 2092 | 1a78e7a3d6 | Chenghao Zhang | 2025-11-06 | [None][feat] AutoDeploy: Support Latent MOE for Nemotron (#8955) |
| 2093 | ada93f1187 | dhansen-nvidia | 2025-11-06 | [https://nvbugs/5527655][feat] Add NUMA-aware CPU affinity autoconfig (#8805) |
| 2094 | ddf2d010e2 | Chenghao Zhang | 2025-11-06 | [TRTLLM-8814][feat] AutoDeploy: Use TRTLLM kernels for FP8 linear (#8820) |
| 2095 | b275635a9a | DylanChen-NV | 2025-11-06 | [https://nvbugs/5498478][fix] Fix eagle3 fp8 kv target model + bf16 draft model + chunked prefill (#8910) |
| 2096 | c73efe12e7 | shuyixiong | 2025-11-06 | [None][chore] Use cached model in all ray tests (#8962) |
| 2097 | d246f62868 | Fanrong Li | 2025-11-06 | [https://nvbugs/5630345] [chore] skip deepseek-v3.2 fp8 kv tests on pre-Blackwell architectures (#8973) |
| 2098 | 51545560da | yunruis | 2025-11-06 | [TRTLLM-8803][feat] Add rope and uk-bgemm overlap for mla generation (#8495) |
| 2099 | b7798bfab8 | Yilin Fan | 2025-11-05 | [None][feat] Add `trtllm_` prefix for exposed metrics (#8845) |
| 2100 | e822184cd7 | xinhe-nv | 2025-11-06 | [None][feat] add waive by sm version (#8928) |
| 2101 | 1c8c771974 | TensorRT LLM | 2025-11-06 | [None][infra] Check in most recent lock file from nightly pipeline |
| 2102 | 18a4b985f1 | yuanjingx87 | 2025-11-05 | [None][infra] allow to choose repo when generate lock files (#8659) |
| 2103 | cc12d33393 | Yi Sun | 2025-11-06 | [None][feat] Deep Research Implemented with Scaffolding (#8452) |
| 2104 | 6bbb43f2b9 | JadoTu | 2025-11-06 | [None][feat] Add qwen3-next nvfp4 support (#8526) |
| 2105 | 7a552c450a | Lucas Liebenwein | 2025-11-05 | [https://nvbugs/5606166][fix] AutoDeploy: unwaive test for use tuples for cudagraph shape lookup (#8957) |
| 2106 | fb7f9831d3 | Frida Hou | 2025-11-05 | [#8924][fix] Fix AutoDeploy pattern matcher for torch 2.9 (#8920) |
| 2107 | b181568d6f | Lucas Liebenwein | 2025-11-05 | [TRTLLM-8201][feat] Nemotron H MoE Sharding (#8744) |
| 2108 | 222bc911cd | Perkz Zheng | 2025-11-06 | [None][feat] add swapsMmaAb sparseMla kernels (#8913) |
| 2109 | e57d83c5dc | Chang Liu | 2025-11-05 | [TRTLLM-8768][chore] Fuse QK down_proj with indexer K + weight_proj for FP4 ckpt (#8771) |
| 2110 | fdd9e4fe00 | fredricz-20070104 | 2025-11-05 | [TRTLLM-7251][test] Get submit eplb slots empty key work (#8945) |
| 2111 | c2feed798a | Fanrong Li | 2025-11-05 | [https://nvbugs/5630345][chore] unwaive DS-v32 nvfp4 and fp8 tests (#8887) |
| 2112 | 595f78078c | Chuang Zhu | 2025-11-05 | [https://nvbugs/5624367][fix] Fix disagg GPT-OSS test (#8870) |
| 2113 | 1ce83582f9 | Yiteng Niu | 2025-11-05 | [None][infra] update github token name (#8907) |
| 2114 | b9e5315dfb | Yukun He | 2025-11-05 | [https://nvbugs/5623960][fix] Fix the logger once key issue and further compress log in AutoTuner. (#8873) |
| 2115 | 31116825b3 | Emma Qiao | 2025-11-05 | [None][infra] Waive failed cases on main 11/05 (#8936) |
| 2116 | cc4aa29523 | xinhe-nv | 2025-11-05 | [None][chore] Add failed cases into waives.txt (#8865) |
| 2117 | eeb56c2848 | Shiyu Li | 2025-11-04 | [None][feat] MNNVLAllreduce Kernel Refactor (#8018) |
| 2118 | ed81173c55 | Yechan Kim | 2025-11-05 | [None][ci] Add test on waives (#8915) |
| 2119 | 871ea244a3 | Yibin Li | 2025-11-04 | [None][chore] Design diagram review process change (#8748) |
| 2120 | 782824533e | Patrice Castonguay | 2025-11-04 | [https://nvbugs/5587574][fix] Increase server timeout to wait for weight loading (#8806) |
| 2121 | 11ded113cd | Frida Hou | 2025-11-04 | [#8389][fix] Update group attention matching to first map to custom torch attention (#8638) |
| 2122 | 70e4d72ffa | shuyixiong | 2025-11-05 | [TRTLLM-8511][feat] Add update_weights and sleep_wakeup support for rl integration (#8302) |
| 2123 | e2b2675120 | Yanchao Lu | 2025-11-04 | [None][fix] Remove duplicated test waives (#8914) |
| 2124 | e4bf29bc66 | Bo Li | 2025-11-04 | [None][feat] Integrate MnnvlThroughput into TRTLLM MoE. (#8728) |
| 2125 | 7e4b87b17c | Robin Kobus | 2025-11-04 | [None][ci] Remove outdated test entries (#8909) |
| 2126 | dddfcdd3bf | Cao Dong | 2025-11-04 | [None][fix] Fix bug of undefined py_topk_logprobs_vals (#8789) |
| 2127 | cae468cc8e | xiweny | 2025-11-04 | [https://nvbugs/5596343] [test] Waive flaky GPT-OSS cases (#8904) |
| 2128 | 4de31bece2 | Zhanrui Sun | 2025-11-04 | [TRTLLM-8994][infra] upgrade to DLFW 25.10 and pytorch 2.9.0 / triton 3.5.0 (#8838) |
| 2129 | 4296c9553d | CarstyYou | 2025-11-04 | [TRTLLM-1234][feat] Add fp8 blockscaled Gemm for sm120 (#8844) |
| 2130 | 23717cdb3f | Ivy Zhang | 2025-10-20 | [TRTLLM-8580][test] save runtime report periodically (#8312) (#8455) |
| 2131 | 2b58dba0f6 | danielafrimi | 2025-10-19 | [https://nvbugs/5524714][fix] Fix TP sharding of fused-QKV weight scales in W4A16 AWQ (#8432) |
| 2132 | ce23e24123 | xiweny | 2025-10-17 | [https://nvbugs/5565565] [fix] Remove waiver (#8450) |
| 2133 | 6c8ba3be27 | Yukun He | 2025-10-17 | [None][chore] Remove duplicate log outputs in test_perf.py (#8418) |
| 2134 | 102e556863 | ruodil | 2025-10-16 | [None][test] cherry-pick: add test-model-suites in integration conftest.py (#8388) |
| 2135 | 2225745782 | Yukun He | 2025-10-16 | [TRTLLM-8129][feat] Allreduce tuning and benchmark script revising (#7870) |
| 2136 | 34fbc7052c | Zhenhuan Chen | 2025-10-16 | [https://nvbugs/5545522][fix] move PREEXIT in UB kernels to fix accuracy issue (#8318) |
| 2137 | 65c138108e | Patrice Castonguay | 2025-10-15 | [https://nvbugs/5552889][fix] fix: Prevent empty batch when using attention DP with disagg (#8372) |
| 2138 | 9bcd2e6c0a | Ivy Zhang | 2025-10-15 | [None][chore] Update nim test list (#8356) |
| 2139 | def9c0004d | Stanley Sun | 2025-10-15 | [TRTLLM-8113][test] Add pytorch workflow e2e tests with pp enabled (#8357) |
| 2140 | fcac2022e2 | xiweny | 2025-10-15 | [https://nvbugs/5565565] [fix] fp8 wideep support sm103 (#8228) |
| 2141 | bd1c9c0af4 | Yueh-Ting (eop) Chen | 2025-11-04 | [https://nvbugs/5625990][chore] Add test coverage for current incapability of the KV cache manager (#8829) |
| 2142 | 67208f1512 | Yechan Kim | 2025-11-04 | [None][fix] InputProcessor config naming convention fix (#8705) |
| 2143 | 4fe47faf47 | Emma Qiao | 2025-11-04 | [None][infra] Waive failed tests for main branch (#8897) |
| 2144 | 9ec6a6b68f | Zhanrui Sun | 2025-11-04 | [None][infra] waive failed test on main 11/4 (#8896) |
| 2145 | 97674c3114 | HuiGao-NV | 2025-11-04 | [TRTLLM-8690][feat] add more tensors to share buffers (#8691) |
| 2146 | ed297d7c2e | Yan Chunwei | 2025-11-04 | [None][chore] Optimize perf for the RPC executor and add some profile utilities to llm-api (#8415) |
| 2147 | 6a6317727b | Anish Shanbhag | 2025-11-03 | [TRTLLM-8680][doc] Add table with one-line deployment commands to docs (#8173) |
| 2148 | d0f107e4dd | Matthias Jouanneaux | 2025-11-04 | [TRTLLM-5966][feat] Helix: add full MLA support for Helix (#8104) |
| 2149 | 5e6f1bcd24 | Mike Iovine | 2025-11-03 | [TRTLLM-8979][test] Improve qwen3 spec dec test coverage (#8767) |
| 2150 | 0f6763680a | Matt Lefebvre | 2025-11-03 | [TRTINFRA-7215][infra] - Move half of the DGX H100 premerge tests to SLURM (#8849) |
| 2151 | db2a42f641 | Kaiyu Xie | 2025-11-03 | [None][chore] Add sample yaml for wide-ep example and minor fixes (#8825) |
| 2152 | 89336fbf07 | Li Min | 2025-11-03 | [None][fix] Fix cute dsl nvfp4 gemm autotune issue (#8761) |
| 2153 | f48968b6cc | Yechan Kim | 2025-11-03 | [TRTLLM-6928][fix] Refactor multimodal unittest (#8453) |
| 2154 | 14bc8571ae | Emma Qiao | 2025-11-03 | [TRTLLM-8435][infra] Test existing rtxpro6000 stages on rtxpro6000d (#8319) |
| 2155 | d7176768cd | Emma Qiao | 2025-11-03 | [None][infra] Waive the failed test for main on 11/3 (#8875) |
| 2156 | 8303cfa477 | Tailing Yuan | 2025-11-03 | [None][fix] Fix import issues in layer-wise benchmarks (#8827) |
| 2157 | 4873ca04cc | xinhe-nv | 2025-11-03 | [https://nvbugs/5521799][fix] add harmony channel validation (#8837) |
| 2158 | 65b793c77e | Guoming Zhang | 2025-11-03 | [None][doc] Add the missing content for model support section and fix valid links for long_sequence.md (#8869) |
| 2159 | 271a981f1f | Yan Chunwei | 2025-11-03 | [None][doc] Add LLM-API API change principle (#8350) |
| 2160 | 64540451e7 | xinhe-nv | 2025-11-03 | [None][chore] Add failed cases into waives.txt (#8872) |
| 2161 | e9f78c687a | Fanrong Li | 2025-11-03 | [https://nvbugs/5625962][chore] unwaive DS-v32-fp4 tests (#8853) |
| 2162 | 00c0e6c440 | Yechan Kim | 2025-11-03 | [https://nvbugs/5523315][fix] Fix serve benchmark test (#8255) |
| 2163 | cc4ab8d9d1 | chenfeiz0326 | 2025-11-03 | [TRTLLM-8825][feat] Support Pytest Perf Results uploading to Database (#8653) |
| 2164 | 2ff772ef71 | Cao Dong | 2025-11-03 | [None][feat] Add benchmark to DeepConf (#8776) |
| 2165 | 497a07021d | Perkz Zheng | 2025-11-03 | [None][update] optimized sparse mla kernels && fix unspecified cuda launch (#8866) |
| 2166 | b4d17d1a4c | yufeiwu-nv | 2025-11-03 | [TRTLLM-8991][test] Add Llama 3.3 70B model with different performance config (#8753) |
| 2167 | f57dc01e6f | Chang Liu | 2025-11-02 | [https://nvbugs/5625380][chore] Remove multimodal related fields from decoder llm input (#8846) |
| 2168 | 0f42a24f45 | qsang-nv | 2025-11-03 | [None][feat] Fix attention sink load in xqa (#8836) |
| 2169 | 6d6797c792 | dongfengy | 2025-11-02 | [None][test] Enhance GPT-OSS CI with GPQA Diamond and additional Spec Decoding Test (#8661) |
| 2170 | f8778230e3 | Eran Geva | 2025-11-02 | [#8781][fix] Cache the AllReduce wrapper to avoid re-allocating workspace which caused a hang (#8803) |
| 2171 | da73410d3b | Yanchao Lu | 2025-11-02 | [None][fix] WAR for tensorrt depending on the archived nvidia-cuda-runtime-cu13 package (#8857) |
| 2172 | 1b3ad7259d | Robin Kobus | 2025-11-01 | [None][feat] Use ruff for formatting and linting new files by default (#8629) |
| 2173 | 1551ed8e5f | Yan Chunwei | 2025-11-01 | [https://nvbugs/5437384][test] CHERRY-PICK: fix trtllm-llmapi-launch multi tests  (#8567) |
| 2174 | 4c5a8f4ec6 | Bo Li | 2025-11-01 | [None][fix] Rename: slot_count -> invalid_expert_id (#8783) |
| 2175 | 89e0117097 | QI JUN | 2025-11-01 | [TRTLLM-8836][chore] Create ModelEngine from LlmArgs (#8600) |
| 2176 | d798d66976 | brb-nv | 2025-10-31 | [TRTLLM-7731][feat] Avoid over-allocation of KV cache for transmission in disagg with CP (#8145) |
| 2177 | bba2519726 | dongxuy04 | 2025-11-01 | [TRTLLM-7008][fix] Enable GDRCopy and unwaive online eplb tests (#8720) |
| 2178 | f0dc746738 | Fanrong Li | 2025-11-01 | [TRTLLM-8541][feat] Add trtllm-gen sparse MLA kernels to support per-Tensor FP8 KV Cache (#8692) |
| 2179 | da2dca58aa | Matt Lefebvre | 2025-10-31 | [TRTINFRA-7215][infra] Add support for enroot SLURM clusters (#8770) |
| 2180 | 0edba5a7e2 | dongfengy | 2025-10-31 | [https://nvbugs/5474119][fix] Re-enable test (#8809) |
| 2181 | 6424f7e55f | dongfengy | 2025-10-31 | [None][doc] Clarify the perf best practice and supported hardware for gptoss (#8665) |
| 2182 | afa75c9494 | Patrice Castonguay | 2025-10-31 | [https://nvbugs/5614506][chore] Adding e+p+d e2e test (#8801) |
| 2183 | 3d0e38e074 | Suyog Gupta | 2025-10-31 | [None][perf] AutoDeploy optimize _get_unique_value (#8822) |
| 2184 | 852e5060aa | Anthony Chang | 2025-10-31 | [https://nvbugs/5558117][fix] Allow per-layer quant config from hf_quant_config.json (#8617) |
| 2185 | 98453d2bb7 | Tailing Yuan | 2025-10-31 | [None][fix] Waive layer-wise benchmark tests (#8823) |
| 2186 | 3a79d03874 | Chang Liu | 2025-10-30 | [https://nvbugs/5617275][fix] Extract py files from prebuilt wheel for editable installs (#8738) |
| 2187 | aecc9655a0 | Emma Qiao | 2025-10-31 | [None][info] Waive failed case for main (#8826) |
| 2188 | 1a338e1a05 | HuiGao-NV | 2025-10-31 | [None][chore] use cached vila model (#8788) |
| 2189 | 1d4a186ace | Yukun He | 2025-10-31 | [https://nvbugs/5623960][fix] Compress the warning log of AutoTuner when encountering tactic failures. (#8793) |
| 2190 | a6a3de8e35 | Zhanrui Sun | 2025-10-31 | [TRTLLM-9003][infra] Add python OpenSearchDB query / push. (#8506) |
| 2191 | 025d2926df | Yuxian Qiu | 2025-10-31 | [https://nvbugs/5599515][fix] Fix PP bubbles. (#8687) |
| 2192 | f3224ccd32 | Yilin Fan | 2025-10-30 | [None][feat] Add disagg relay time to time breakdown tool (#8465) |
| 2193 | 603ec03fb1 | Zhenhuan Chen | 2025-10-31 | [https://nvbugs/5575687][fix] fix moe_gemm's preexit position that cause illegal memory access (#8786) |
| 2194 | fe670af65f | yuanjingx87 | 2025-10-30 | [None][infra] Update allow list 20251030 (#8808) |
| 2195 | b87448b009 | Mike Iovine | 2025-10-30 | [TRTLLM-8978][test] Remove llama 4 spec dec tests (#8766) |
| 2196 | 71c5576a44 | Chenghao Zhang | 2025-10-30 | [TRTLLM-8734][feat] AutoDeploy: Enable the nvfp4 for Nemotron MOE (#8737) |
| 2197 | ec31363a86 | Tailing Yuan | 2025-10-31 | [None][fix] Layer wise benchmarks: use local models, lint (#8799) |
| 2198 | 9112cffaf3 | Emma Qiao | 2025-10-30 | [None][infra] Waive failed case for main branch (#8797) |
| 2199 | 547d799111 | Zhanrui Sun | 2025-10-30 | [TRTLLM-8930][infra] Force Blossom perf test stages to use 'tensorrt/test_type: perf' in the K8S template (#8752) |
| 2200 | f9c7786dc8 | Tailing Yuan | 2025-10-30 | [None][feat] Add layer wise benchmarks (#8777) |
| 2201 | f666ad2f6b | Anthony Chang | 2025-10-30 | [None][feat] Autotuner can iterate through all tactics for test purposes (#8663) |
| 2202 | a5cc9fe0aa | Emma Qiao | 2025-10-30 | [TRTLLM-5453][infra] Check all steps for test name and also check the test in waives.txt also exists in l0 or qa test list. (#6256) |
| 2203 | 13cfd70f57 | ChristinaZ | 2025-10-30 | [None][feat] Add unit tests and revision in block_level kernel for invalid input (#8718) |
| 2204 | cc286687c4 | WeiHaocheng | 2025-10-30 | [None][feat] Refactor scaffolding streaming feature and fix openai wo… (#8622) |
| 2205 | a4f75399b9 | xinhe-nv | 2025-10-30 | [https://nvbugs/5481206][fix] update waives (#8774) |
| 2206 | 2072185d76 | Leslie Fang | 2025-10-30 | [https://nvbugs/5608461][fix] exclude InductorSubproc from thread leak check (#8704) |
| 2207 | 6b755fd9f8 | Void | 2025-10-30 | [None][fix] fix runtime error that bf16 input is not quantized to nvfp4 when use bf16 dispatch (#8507) |
| 2208 | e689a73c83 | yuanjingx87 | 2025-10-29 | [None][infra] fix slurm results path (#8751) |
| 2209 | 7d3cebf34e | Emma Qiao | 2025-10-30 | [None][infra] Unwaive the tests passed in latest CI and disable a perf stage (#8775) |
| 2210 | 496b419791 | Yi Zhang | 2025-10-30 | [None][doc] Add doc for torch.compile & piecewise cuda graph (#8527) |
| 2211 | db99a936b0 | Emma Qiao | 2025-10-30 | [TRTLLM-8971][infra] Update gpu key for B300/GB300 (#8724) |
| 2212 | 3176bd3815 | Yuxian Qiu | 2025-10-30 | [None][fix] Fix UnboundLocalError. (#8756) |
| 2213 | ae57738bae | HuiGao-NV | 2025-10-30 | [https://nvbugs/5547414][fix] Use cached models (#8755) |
| 2214 | a2e964d9a8 | Sharan Chetlur | 2025-10-29 | [None][doc] Minor doc update to disagg-serving (#8768) |
| 2215 | 834a780655 | Simeng Liu | 2025-10-29 | [https://nvbugs/5599086][fix] Fix FP8 Linear module for spark (#8707) |
| 2216 | 45b36cc069 | yuanjingx87 | 2025-10-29 | [None][infra] Check in most recent lock file from nightly pipeline (#8739) |
| 2217 | ae6875fe10 | Iman Tabrizian | 2025-10-29 | [TRTLLM-8976][feat] Move indexer-k-cache to KVCacheManager (#8699) |
| 2218 | 579e1067bf | Emma Qiao | 2025-10-29 | [None][infra] Waive failed tests on main (#8759) |
| 2219 | 451959c60d | Leslie Fang | 2025-10-29 | [TRTLLM-8763][chore] Deprecate pybind based GuidedDecodingConfig usage in torch backend (#8717) |
| 2220 | fc3b6f5331 | Yan Chunwei | 2025-10-29 | [None][ci] waive test_rpc.py (#8745) |
| 2221 | a21697ead9 | Fanrong Li | 2025-10-29 | [None][fix] fix config loading for DeepSeek-V3.2 in trtllm-bench (#8729) |
| 2222 | e2c5a38879 | kris1025 | 2025-10-29 | [https://nvbugs/5534574][fix] disable spec decoding forever once the request spec decoding is disabled (#8446) |
| 2223 | 81eb861df0 | Chang Liu | 2025-10-29 | [None][chore] Enable GPQA in CI for DeepSeek V3.2 (#8712) |
| 2224 | a69bd2a6fa | Yi Zhang | 2025-10-29 | [https://nvbugs/5550409][fix] Disable torch compile in piecewise attention part to Avoid host overhead (#8708) |
| 2225 | d626d13d37 | Zheng Duan | 2025-10-29 | [https://nvbugs/5607238][test] fix working dir in disagg worker test (#8648) |
| 2226 | 2aade46d18 | Pengyun Lin | 2025-10-29 | [TRTLLM-8214][feat] Support Qwen3 tool parser (#8216) |
| 2227 | 741183917c | Yiteng Niu | 2025-10-29 | [None][infra] update ci allow list 2025/10/29 (#8749) |
| 2228 | 585733f113 | Faraz | 2025-10-29 | [None][fix] add readme copy to wheel stage to avoid setup.py failure (#8736) |
| 2229 | 00eaf5f883 | dongxuy04 | 2025-10-29 | [None][feat] add flag for EPLB to force using GDRCopy (#8650) |
| 2230 | 19ca7b15c7 | Stefan Niebler | 2025-10-29 | [https://nvbugs/5593199][test] Enhance beam search tests deterministic dummy model (#8625) |
| 2231 | 5f737b8dbe | Chang Liu | 2025-10-28 | [None][perf] Use fp8 quant kernel in DS3.2 indexer module (#8701) |
| 2232 | 15c293a90b | Cheng Hang | 2025-10-29 | [None][feat] Enable nvfp4 cuda core for sm120 (#8620) |
| 2233 | bc26f4ce7c | Yechan Kim | 2025-10-29 | [https://nvbugs/5549829][fix] Qwen2.5-VL TP > 1 + Quantized weight load fix (#8680) |
| 2234 | 7ba98a6b20 | xinhe-nv | 2025-10-29 | [None][chore] Add failed cases into waives.txt (#8684) |
| 2235 | f2faf2809f | Yan Chunwei | 2025-10-29 | [None][ci] waive test_rpc.py temporarily (#8743) |
| 2236 | fea5bfbda7 | Zheng Duan | 2025-10-29 | [None][feat] add detailed KV cache transfer time breakdown (#8521) |
| 2237 | f444fe2deb | ruodil | 2025-10-29 | [None][test] fix a typo in perf test sampler config (#8726) |
| 2238 | b828b6445b | Chuang Zhu | 2025-10-29 | [https://nvbugs/5612529][fix] Fix transferAgent_test (#8710) |
| 2239 | cf8a1d2ef9 | Yechan Kim | 2025-10-29 | [https://nvbugs/5596377][fix] Fix mm dummy calculation (#8498) |
| 2240 | 24167d00eb | Lizhi Zhou | 2025-10-29 | [TRTLLM-8431][doc] update public doc and example, add etcd auto-scaling tests (#8602) |
| 2241 | 227c288441 | Kaiyu Xie | 2025-10-29 | [TRTLLM-8827] [feat] Enable low precision alltoall for Cutlass and TRTLLMGen backends (#8675) |
| 2242 | 00161b315f | Mike Iovine | 2025-10-28 | [https://nvbugs/5549111][fix] Fix 2-model overlap scheduler accuracy on very long prompts (#8076) |
| 2243 | 083f3637f1 | dongfengy | 2025-10-28 | [https://nvbugs/5596343][test] Update test waive to get back some coverage (#8702) |
| 2244 | 0ee71d95ec | Lucas Liebenwein | 2025-10-28 | [https://nvbugs/5606166][fix] AutoDeploy: use tuples for cudagraph shape lookup (#8658) |
| 2245 | a09b38a862 | Anish Shanbhag | 2025-10-28 | [TRTLLM-8684][chore] Migrate BuildConfig to Pydantic, add a Python wrapper for KVCacheType enum (#8330) |
| 2246 | cdc9e5e645 | William Zhang | 2025-10-28 | [None][fix] Properly raise error for nemotron H models (#8697) |
| 2247 | 5a01f382c1 | dongfengy | 2025-10-28 | [https://nvbugs/5575913][fix] Use separate thresholds for 120b/20b gptoss (#8664) |
| 2248 | e8e2b0697a | Robin Kobus | 2025-10-28 | [None][chore] Revert "[TRTLLM-7835][test] add default sample config for perf test (#8523) (#8725) |
| 2249 | e051a05e6c | Eran Geva | 2025-10-28 | [#8694][fix] fix AutoDeploy cuda memory access failure in nvidia/NVIDIA-Nemotron-Nano-31B-A3-v3 (#8696) |
| 2250 | b37a8a9a74 | dongxuy04 | 2025-10-28 | [None][fix] fix EPLB init hang (#8649) |
| 2251 | 6b9b73ee27 | ruodil | 2025-10-28 | [https://nvbugs/5564465][test] ensure deepseek_v3_lite isl + osl < max_seq_len (#8565) |
| 2252 | bf72eb045e | ruodil | 2025-10-28 | [TRTLLM-7835][test] add default sample config for perf test (#8523) |
| 2253 | 0e36484fba | yufeiwu-nv | 2025-10-28 | [None][test] Add gpt_oss_20b Model to Sanity Perf Test (#8265) |
| 2254 | a966644a71 | Erin | 2025-10-27 | [None][fix] Change Ray submit() to use async RPC (#8636) |
| 2255 | 08134cbca0 | Sai Kiran Polisetty | 2025-10-28 | [https://nvbugs/5556475] [fix] Fix the `tensorrt_llm_bls` model to correctly return the outputs for `num_input_tokens` and `num_output_tokens` (#8150) |
| 2256 | 0a02f5f25d | Aurelien Chartier | 2025-10-27 | [None][chore] Use a cached model path for Ray integration test (#8660) |
| 2257 | 49974eed75 | HuiGao-NV | 2025-10-28 | [None][chore] ISOLATE some cases (#8690) |
| 2258 | f5265a087b | chenfeiz0326 | 2025-10-28 | [None][infra] Minor Update on Perf Sanity Testdb Files (#8607) |
| 2259 | 88b0fbc8ff | gramnarayan | 2025-10-27 | [#8245][feat] Autodeploy: Guided Decoding Support (#8551) |
| 2260 | a6017f6266 | Yechan Kim | 2025-10-28 | [https://nvbugs/5608723][fix] Use local data on multimodal tests and unwaive tests (#8673) |
| 2261 | 73a5479b26 | Emma Qiao | 2025-10-28 | [None][infra] Skip failed tests for main 10/27 (#8686) |
| 2262 | 1401a3c09c | Aurelien Chartier | 2025-10-27 | [None][feat] Add FP8 rowwise GEMMs for B200 (#8332) |
| 2263 | 9c4432f8a4 | Bo Li | 2025-10-28 | [TRTLLM-7318][feat] MnnvlThroughput AlltoAll implementation. (#7499) |
| 2264 | d1398c05e6 | nvxuanyuc | 2025-10-27 | [None][feat] Support ignored prompt length for penalties via new sampling config parameter (#8127) |
| 2265 | b9b2802599 | Chenghao Zhang | 2025-10-27 | [None][feat] Autodeploy: Update the ssm to use slice (#8667) |
| 2266 | 7c8ba71b49 | mpikulski | 2025-10-27 | [TRTLLM-8832][feat] fully async _select_generated_logits with tests (#8628) |
| 2267 | 4fd58137a1 | QI JUN | 2025-10-27 | [TRTLLM-8933][chore] remove unused update_executor_config function (#8678) |
| 2268 | c9b08790c2 | Kaiyu Xie | 2025-10-27 | [None] [test] Add MNNVL AlltoAll tests to pre-merge (#8601) |
| 2269 | 0019d99e6d | Chao Ni | 2025-10-27 | [None][test] Add longbench v2 for long context evaluation (#8604) |
| 2270 | 1026069a2b | zhanghaotong | 2025-10-27 | [None][feat] Add opentelemetry tracing (#5897) |
| 2271 | ce0d76135d | Jie Li | 2025-10-27 | [https://nvbugs/5546507][fix] skip TRT-Flow test case due to CMake Error in building (#8677) |
| 2272 | 990b0c0c47 | Robin Kobus | 2025-10-27 | [TRTLLM-7159][docs] Add documentation for additional outputs (#8325) |
| 2273 | 8090c9641c | xinhe-nv | 2025-10-27 | [TRTLLM-8638][fix] Add failed cases into waives.txt (#8672) |
| 2274 | 1614624beb | Yanchao Lu | 2025-10-27 | [None][docs] Update Python wheel's short-/long-descriptions (#8676) |
| 2275 | 0ac5cbcac4 | xinhe-nv | 2025-10-27 | [None][chore] Add failed cases into waives.txt (#8669) |
| 2276 | 858d6437c1 | Tailing Yuan | 2025-10-27 | [None][fix] Fix ModelConfig.from_pretrained get quant config file (#8647) |
| 2277 | cc5b8b6d28 | QI JUN | 2025-10-27 | [None][ci] move some time-consuming benchmark test cases to post merge (#8641) |
| 2278 | 0a0f93d4a8 | Jinyang Yuan | 2025-10-27 | [None][fix] Fix the performance issue of FP8 blockwise grouped GEMM when using attention DP (#8501) |
| 2279 | e0728ba8a7 | Emma Qiao | 2025-10-26 | [None][infra] Waive failed case on main 10/26 (#8668) |
| 2280 | a6d20f6f9b | Chenghao Zhang | 2025-10-25 | [None][feat] AutoDeploy: Add FP8 MOE for Nemotron (#8599) |
| 2281 | 95be56e56b | Wanli Jiang | 2025-10-25 | [TRTLLM-8238][feat] Add EVS support for nano-v2-vlm (#8024) |
| 2282 | 2b27810198 | Simeng Liu | 2025-10-24 | [https://nvbugs/5494718][fix] Fix Single GPU Multi-node issue and OOM on DGX Spark (#8514) |
| 2283 | 812bc8c954 | Erin | 2025-10-24 | [TRTLLM-8513][feat] Add back worker extension (#8482) |
| 2284 | 02081e2390 | jthomson04 | 2025-10-24 | [None][feat] Support KV Connector with Disagg Prefill Worker (#8246) |
| 2285 | e47c787dd7 | Chang Liu | 2025-10-24 | [TRTLLM-8535][feat] Support DeepSeek V3.2 with FP8 + BF16 KV cache/NVFP4 + BF16 KV cache (#8405) |
| 2286 | 2d86d6be40 | Yechan Kim | 2025-10-25 | [TRTLLM-8737][feat] Support media_io_kwargs on trtllm-serve (#8528) |
| 2287 | cdf0403c64 | Aurelien Chartier | 2025-10-24 | [None][feat] Pass KvCacheRetentionConfig to torch LlmRequest (#8634) |
| 2288 | 2420918e5b | Chuang Zhu | 2025-10-24 | [TRTLLM-7078][chore] optimal kvcache transfer for VWSA (#7952) |
| 2289 | f512ddaeef | Suyog Gupta | 2025-10-24 | [None][feat] add skip condition in AutoDeploy's triton fused moe kernel (#8632) |
| 2290 | 602b059180 | Yiqing Yan | 2025-10-24 | [None][chore] Disable GB300 stages due to nodes will be offline temporarily (#8643) |
| 2291 | 35e35db422 | Emma Qiao | 2025-10-24 | [None][infra] Waive tests on main and remove lines which missed in MI (#8639) |
| 2292 | 2aaedd08cd | xinhe-nv | 2025-10-24 | [TRTLLM-8638][fix] fix test issues (#8557) |
| 2293 | 9a9d647292 | xinhe-nv | 2025-10-24 | [None][chore] Add failed cases into waives.txt (#8630) |
| 2294 | 07a957e5cb | ruodil | 2025-10-24 | [None][test] remove redunctant runtime backend in perf test (#8358) |
| 2295 | 6b793d5c3d | Stanley Sun | 2025-10-24 | [TRTLLM-8738][test] Add end-to-end trtllm-serve negative tests (#8580) |
| 2296 | e7ad5e4d6a | yuanjingx87 | 2025-10-23 | [None][infra] enable lfs for generateLockFile pipeline (#8547) |
| 2297 | 59375e8bed | xinhe-nv | 2025-10-24 | [TRTLLM-8638][fix] Add failed cases into waives.txt (#8590) |
| 2298 | 95d39e6e76 | xinhe-nv | 2025-10-24 | [TRTLLM-8638][fix] Add failed cases into waives.txt (#8588) |
| 2299 | f448043d88 | Wanli Jiang | 2025-10-24 | [None][feat] Support base64 video input (#8458) |
| 2300 | e666a704f5 | Zheng Duan | 2025-10-24 | [None][doc] add visualization of perf metrics in time breakdown tool doc (#8530) |
| 2301 | 6ee1c87595 | QI JUN | 2025-10-24 | [TRTLLM-8817][chore] Set default value of KvCacheConfig.free_gpu_memory_fraction explicitly (#8561) |
| 2302 | 23920223ab | h-guo18 | 2025-10-23 | [#4585][feat] Replace unified attention before export (#8303) |
| 2303 | 32e1ad68e1 | Aurelien Chartier | 2025-10-23 | [None][chore] Cleanup GDS code (#8475) |
| 2304 | cc81028547 | QI JUN | 2025-10-23 | [TRTLLM-8812][chore] Limit the scope of pybind based CacheTransceiverConfig (#8558) |
| 2305 | ee21ea3e91 | Emma Qiao | 2025-10-23 | [None][infra] Disable rtxpro6000 stages due to nodes will be offline (#8613) |
| 2306 | 7c1bca4563 | Emma Qiao | 2025-10-23 | [None][infra] Fix slurm exitcode (#8585) |
| 2307 | 3a5845e293 | Robin Kobus | 2025-10-23 | [TRTLLM-8714][fix] update create_input_processor to handle custom checkpoint format (#7811) |
| 2308 | 928247a3f9 | Shijie | 2025-10-23 | [https://nvbugs/5451205][feat] Add cuBLASLt NVFP4 GEMM backend support (#7943) |
| 2309 | 04e2b2752a | xinhe-nv | 2025-10-23 | [None][feat] add Nemotron-Ultra multi nodes eval tests (#8577) |
| 2310 | 2956978da3 | Suyog Gupta | 2025-10-22 | [None][feat] Enable rms norm fusion for Nemotron MOE (#8563) |
| 2311 | a7c2c8c212 | dongxuy04 | 2025-10-23 | [None][fix] Allow multi-threaded copy for GDRCopy wrapper (#8535) |
| 2312 | 77fa5dfee9 | Lucas Liebenwein | 2025-10-22 | [https://nvbugs/5604136][fix] AutoDeploy: correct import for mxfp4_moe unit test (#8593) |
| 2313 | ea3e0eea51 | sunnyqgg | 2025-10-23 | [TRTLLM-7954][feat] Target model KV cache rellocation (#8421) |
| 2314 | 8a3b870e09 | Anthony Chang | 2025-10-23 | [None][feat] Update TRTLLM MoE MxFP4 cubins; autotune tileN (#8156) |
| 2315 | 15de45d782 | Anish Shanbhag | 2025-10-22 | [TRTLLM-8682][chore] Remove auto_parallel module (#8329) |
| 2316 | e5865de518 | Leslie Fang | 2025-10-23 | [TRTLLM-8754][chore] Refine PyTorchModelEngine with llm args (#8493) |
| 2317 | 00c2b81037 | brb-nv | 2025-10-22 | [None][chore] Skip failing import of mxfp4_moe (#8591) |
| 2318 | df689f8fed | dongxuy04 | 2025-10-22 | [None][fix] Fix EPLB CPU thread NUMA binding (#8579) |
| 2319 | 879039f6d5 | Patrice Castonguay | 2025-10-22 | [https://nvbugs/5429636][feat] Kv transfer timeout (#8459) |
| 2320 | b8b2c9efb4 | xinhe-nv | 2025-10-22 | [None][chore] add precommit hook to remove redundant tab and white space (#8534) |
| 2321 | 910e6b9684 | Eran Geva | 2025-10-22 | [None][fix] fixed cached model path in test (#8549) |
| 2322 | 40a9c61a89 | mpikulski | 2025-10-22 | [None][fix] generate nanobind stubs for submodules (#8539) |
| 2323 | f81caf5491 | Yan Chunwei | 2025-10-22 | [None][chore] replace print_colored_debug with logger_debug (#8417) |
| 2324 | d4b3bae5af | Eran Geva | 2025-10-22 | [#8391][fix] check perf by device subtype (#8428) |
| 2325 | 3f9dbc76c0 | Yan Chunwei | 2025-10-22 | [None][fix] fix rpc unique addr related issue (#8419) |
| 2326 | 912cf4f603 | Ivy Zhang | 2025-10-22 | [TRTLLM-8785][fix] fix conflicts between periodic-junit and store-durations (#8518) |
| 2327 | 92e99b6545 | Emma Qiao | 2025-10-22 | [None][infra] Waive failed cases for main branch 10/22 (#8573) |
| 2328 | 8c9fda4b85 | yunruis | 2025-10-22 | [None][doc] Paragraph adjustment and fix statistic (#8568) |
| 2329 | b04e51291a | Yiqing Yan | 2025-10-22 | [None][chore] Bump version to 1.2.0rc2 (#8562) |
| 2330 | 77940635bb | Shi Xiaowei | 2025-10-22 | [https://nvbugs/5451272][fix] unwaive the test (#8537) |
| 2331 | 07edac2818 | qsang-nv | 2025-10-22 | [None][feat] Add vLLM KV Pool support for XQA mla kernel (#8560) |
| 2332 | 187cf12d8f | xinhe-nv | 2025-10-22 | [TRTLLM-8638][fix] Add failed cases into waives.txt (#8554) |
| 2333 | 2b4e812aea | Emma Qiao | 2025-10-22 | [None][infra] Let CI continue running other isolation tests when an isolation test get hanging (#8471) |
| 2334 | 6cf1c3fba4 | chenfeiz0326 | 2025-10-22 | [TRTLLM-8260][feat] Add Server-Client Perf Test in pytest for B200 and B300 (#7985) |
| 2335 | 50149ac2bd | Shi Xiaowei | 2025-10-22 | [None][doc] Fix the incorrect doc figure (#8536) |
| 2336 | 90080e0e09 | sunnyqgg | 2025-10-22 | [https://nvbugs/5556020][fix] test_disaggregated_serving.py::TestLlama3_1_8BInstruct::test_eagle3 dimension mismatch (#8517) |
| 2337 | 50d4e5bc06 | Leslie Fang | 2025-10-22 | [TRTLLM-8483][chore] Refine scheduler_config and peft_cache_config in create_py_executor (#8451) |
| 2338 | bac9e8c2ad | Chenghao Zhang | 2025-10-21 | [None][feat] AutoDeploy: Add Nemotron MOE support for AutoDeploy (#8469) |
| 2339 | 23d5280a90 | Lizhi Zhou | 2025-10-22 | [TRTLLM-7843][feat] implement disagg cluster auto-scaling (#8215) |
| 2340 | 9b54b3bfaf | Lucas Liebenwein | 2025-10-21 | [None][chore] AutoDeploy: replace HF's deprecated keyword torch_dtype --> dtype (#8510) |
| 2341 | 8dc4aac5b6 | YueWeng | 2025-10-21 | [TRTLLM-8160][feat] Add max_total_draft_tokens (#8366) |
| 2342 | a0024f4d34 | Shi Xiaowei | 2025-10-21 | [None][doc] Facilitates the integration of the transfer agent (#7867) |
| 2343 | 653aa6b6dc | Emma Qiao | 2025-10-21 | [None][infra] Waive failed tests for main 10/21 (#8524) |
| 2344 | 9ba5959e8e | Yan Chunwei | 2025-10-21 | [None][fix] the api_stability unify default values of None and inspect._empty (#8496) |
| 2345 | 85088dce05 | Yueh-Ting (eop) Chen | 2025-10-21 | [None][chore] Update feature combination matrix for SWA kv cache reuse (#8529) |
| 2346 | c566890624 | xinhe-nv | 2025-10-21 | [TRTLLM-8638][fix] Remove closed bugs (#8478) |
| 2347 | c72f6d1dcc | Emma Qiao | 2025-10-21 | [None][infra] Add split algorithm for slurm (#8516) |
| 2348 | a4227cf1b0 | Pengyun Lin | 2025-10-21 | [None][feat] Support Qwen3 reasoning parser (#8000) |
| 2349 | 0acd10e3de | QI JUN | 2025-10-21 | [None][ci] rebalance H100 stages (#8491) |
| 2350 | 3264d605fb | xinhe-nv | 2025-10-21 | [TRTLLM-8638][fix] Add failed cases into waives.txt (#8486) |
| 2351 | ebb62e17d8 | Bo Li | 2025-10-21 | [None][feat] Add alltoall to trtllm-gen MoE backend. (#8481) |
| 2352 | ab4b9966b2 | ruodil | 2025-10-21 | [TRTLLM-7287][test] add multimodal chunked_prefill cases (#8011) |
| 2353 | 4545700fcf | Zero Zeng | 2025-10-21 | [None][chore] Move submit.sh to python and use yaml configuration (#8003) |
| 2354 | 87eb5086fb | mpikulski | 2025-10-21 | [None][fix] restore list[list[list[int]]] in add_token (#8502) |
| 2355 | 85d5aa7763 | Yechan Kim | 2025-10-21 | [None][feat] Support kv_cahce_reuse for HyperCLOVAX-Vision model (#7789) |
| 2356 | 984d4fe0fe | Ruoqian Guo | 2025-10-21 | [None][feat] Update 3rdparty/DeepGEMM to latest commit (#8488) |
| 2357 | 7050b1ea49 | Suyog Gupta | 2025-10-20 | [#8272][feat] Enable chunked prefill for SSMs in AutoDeploy (#8477) |
| 2358 | 3e681e2a80 | Venky | 2025-10-20 | [None] [chore] Add architecture-specific ATTRIBUTIONS files (#8468) |
| 2359 | 55c468b218 | Lucas Liebenwein | 2025-10-20 | [#8461][feat] AutoDeploy: trtllm-serve bug fix + unit test (#8462) |
| 2360 | 9b289d5230 | dongfengy | 2025-10-20 | [https://nvbugs/5568676][fix] Remove test waive (#8437) |
| 2361 | 1e3e1474c6 | yuanjingx87 | 2025-10-20 | [TRTLLM-6055][infra] Slurm Test refactor (#7176) |
| 2362 | d0663e16e0 | HuiGao-NV | 2025-10-20 | [https://nvbugs/5492250][fix] Remove isolated cases and unwaive cases (#8492) |
| 2363 | b818a912d7 | Pamela Peng | 2025-10-20 | [https://nvbugs/5540752][fix] Support quantized Phi4 MM models (#8190) |
| 2364 | 18c7a520b3 | Robin Kobus | 2025-10-20 | [None][feat] Update devcontainer configuration to include additional extensions (#8369) |
| 2365 | 97ce0ecefe | mpikulski | 2025-10-20 | [TRTLLM-8436][feat] batched sampling and top-k logprobs improvements (#8398) |
| 2366 | d05079ba4b | QI JUN | 2025-10-20 | [None][ci] move some test cases from H100 to A10 (#8449) |
| 2367 | 3c2b3bd4d4 | Yi Zhang | 2025-10-20 | [TRTLLM-7255][feat] Add iteration log parser script for benchmark log (#6942) |
| 2368 | 8124a62b74 | Zhanrui Sun | 2025-10-20 | [TRTLLM-8669][infra] Use artifactory mirror for install python (#8394) |
| 2369 | ec32711b1e | Yuxian Qiu | 2025-10-20 | [https://nvbugs/5542862][fix] Upgrade fmha_v2. (#8364) |
| 2370 | c8b9998acb | ChristinaZ | 2025-10-20 | [TRTLLM-8637][feat] Optimize the routing kernel for DeepseekV3 (MoE CUTLASS backend); Add support for KimiK2 and Qwen-next (MoE TRTLLM backend) (#7761) |
| 2371 | f7722e2b65 | xiweny | 2025-10-20 | [TRTLLM-4866] [test] Support waiving unit tests by waives.txt (#8359) |
| 2372 | 128a351bdc | Yueh-Ting (eop) Chen | 2025-10-20 | [None][fix] Avoid overwrite of `kv_cache_config.max_tokens` for VSWA scheme for the KVCacheManager (#8219) |
| 2373 | 9aa086d3bb | xinhe-nv | 2025-10-20 | [None][chore] update test duration (#8377) |
| 2374 | 796891ba2a | Emma Qiao | 2025-10-19 | [None][infra] Skip a failed case in pre-merge for main on 10/19 (#8479) |
| 2375 | dd25595ae8 | Bo Deng | 2025-10-19 | [TRTLLM-7964][infra] Set nixl to default cache transceiver backend (#7926) |
| 2376 | e185173240 | Emma Qiao | 2025-10-19 | [None][infra] Waive test for main branch on 10/18 (#8472) |
| 2377 | 852316886e | jthomson04 | 2025-10-18 | [None][fix] Fix KV event consumption (#6346) |
| 2378 | 7cc65a6296 | brb-nv | 2025-10-18 | [None][chore] Waive failing transceiver test (#8473) |
| 2379 | 41169fb20c | Lucas Liebenwein | 2025-10-18 | [None][feat] AutoDeploy: chunked prefill support (#8158) |
| 2380 | 4a8ac8dd62 | QI JUN | 2025-10-18 | [TRTLLM-8480][chore] clean create_py_executor API (#8412) |
| 2381 | 58b43a6dab | Wanli Jiang | 2025-10-18 | [None][fix] Fix get_num_tokens_per_image for nano-v2-vlm (#8425) |
| 2382 | 136e0e6882 | Kyle McGill | 2025-10-17 | [None][feat] Enable CUDA graph support for KvConnectorWorker API (#8275) |
| 2383 | 5ff4f88be6 | Anish Shanbhag | 2025-10-17 | [TRTLLM-8683][chore] Migrate PluginConfig to Pydantic (#8277) |
| 2384 | 55fed1873c | h-guo18 | 2025-10-17 | [None][chore] AutoDeploy: cleanup old inference optimizer configs (#8039) |
| 2385 | bb7fdcebf4 | Grzegorz Kwasniewski | 2025-10-17 | [TRTLLM-8201][feat] Topological graph helpers (#8457) |
| 2386 | 8d07580c95 | Venky | 2025-10-17 | [None] [chore] Add ATTRIBUTIONS-{CPP,Python}.md + Update in wheels setup (#8438) |
| 2387 | 56f697be2e | Wanli Jiang | 2025-10-17 | [None][feat] Add fmha_v2 kernel for head_dim=80 and sm=100 to support VLM (#8392) |
| 2388 | bc833d3de3 | xinhe-nv | 2025-10-17 | [TRTLLM-8638][fix] add waives tests (#8445) |
| 2389 | 0722717ec0 | Perkz Zheng | 2025-10-17 | [None][fix] trtllm-gen regression in PR 8301 (#8426) |
| 2390 | 7a2bab93f0 | zhhuang-nv | 2025-10-17 | [None][test] Add post merge test for Seed-OSS-36B-Instruct (#8321) |
| 2391 | e72ade33c2 | Yanchao Lu | 2025-10-17 | [None][chore] Update commit msg for adding lock files (#8448) |
| 2392 | 023e515d33 | Leslie Fang | 2025-10-17 | [None][chore] Combine two documents of feature combination matrix (#8442) |
| 2393 | 1e1f430163 | yufeiwu-nv | 2025-10-17 | [None][test] Filter out all fp8 test case for A100. (#8420) |
| 2394 | 70a0f5beb6 | Ivy Zhang | 2025-10-17 | [TRTLLM-8580][test] save runtime report periodically (#8312) |
| 2395 | dd06612d0e | Tracin | 2025-10-17 | [https://nvbugs/5540138][fix] Fix shape error when duplicating kv. (#8390) |
| 2396 | 85deacf117 | yuanjingx87 | 2025-10-16 | [None][infra] Update CI allowed list 2025_10_15 (#8403) |
| 2397 | 3481d03470 | yuanjingx87 | 2025-10-16 | [None][infra] Fix for generate lockfile pipeline (#7820) |
| 2398 | 22eb1633ae | Iman Tabrizian | 2025-10-16 | [None][bug] Set NCCL_GRAPH_REGISTER to false to avoid hang (#8413) |
| 2399 | 46ee7acb33 | John Calderon | 2025-10-16 | [TRTLLM-6780][fix] Add multimodal data to dummy requests during memory profiling (#7539) |
| 2400 | bde606f82d | Yanchao Lu | 2025-10-16 | Update Dockerfile.multi |
| 2401 | d594c2d0ff | Jin Li | 2025-10-14 | [https://nvbugs/5537348][fix] Use device tensor index for MTP (#8062) |
| 2402 | 05dd437084 | Yiqing Yan | 2025-10-14 | [https://nvbugs/5565541][fix] Add timeout threshold for H100 FHMA test (#8354) |
| 2403 | 69325e1aa3 | bhsueh_NV | 2025-10-14 | [https://nvbugs/5574556][fix] fix bug of Qwen3_235B_A22B::test_fp8 CI (#8351) |
| 2404 | 982d4b65e8 | Lizhi Zhou | 2025-10-14 | [https://nvbugs/5550671][fix] fix disagg-serving multinodes test failure (#8307) |
| 2405 | 18a534d2b4 | Chuang Zhu | 2025-10-14 | [https://nvbugs/5465642][fix] Increase server timeout to wait weight loading (#8297) |
| 2406 | 47e6eea3fa | Jin Li | 2025-10-14 | [https://nvbugs/5543770][fix] Update to Cutlass v4.2.1 (#8055) |
| 2407 | b7602f7bd4 | Patrice Castonguay | 2025-10-13 | [https://nvbugs/5534837][fix] Fix KV cache split on long context (#8247) |
| 2408 | 526cad37d7 | Enwei Zhu | 2025-10-13 | [https://nvbugs/5568951][fix] Fix guided decoding disagg tests (#8311) |
| 2409 | 19241626d0 | Zhanrui Sun | 2025-10-13 | [https://nvbugs/5563653][infra] reduce docker image layers (#8250) |
| 2410 | 4230639370 | Yechan Kim | 2025-10-13 | [https://nvbugs/5550722][fix] Fix image load (#8093) |
| 2411 | 9587f099ac | Yechan Kim | 2025-10-13 | [https://nvbugs/5547434][fix] Fix Qwen2.5-VL device_path error (#8057) |
| 2412 | 1b559ba91d | Ivy Zhang | 2025-10-13 | [None][chore] Update test configs for release (#8224) |
| 2413 | 4789c1e588 | Ivy Zhang | 2025-10-13 | [TRTLLM-8246][test] add multimodal kvcache+chunked_prefil cases in to QA test list (#8212) |
| 2414 | be2ab98233 | Ivy Zhang | 2025-10-13 | [None][chore] Update constaintfor release (#8211) |
| 2415 | 4e51148088 | Yan Chunwei | 2025-10-11 | [https://nvbugs/5532023][fix] unwaive GenerationExecutor tests (#8251) |
| 2416 | 179c7dc501 | Yukun He | 2025-10-08 | [https://nvbugs/5536131][fix] Fix illegal access issue when scale is not provided in Llama3/4. (#7960) |
| 2417 | 57a4ef870a | Enwei Zhu | 2025-09-30 | [None][fix] Fix chunked prefill state of draft request (#8067) |
| 2418 | dd61454d5f | sunnyqgg | 2025-10-16 | [https://nvbugs/5461761][fix] Unwaive eagle3 test (#8363) |
| 2419 | 9865d3d770 | Wangjue Yao | 2025-10-16 | [None][feat] Support cached tokens for Openai server (#7637) |
| 2420 | f70eff30b3 | xinhe-nv | 2025-10-16 | [TRTLLM-8638][fix] waive llam4 tests on H20 (#8416) |
| 2421 | 89d03d7668 | xiweny | 2025-10-16 | [https://nvbugs/5532789] [doc] Add documents about CUDA 12.9 (#8411) |
| 2422 | 4e6a492aa3 | HuiGao-NV | 2025-10-16 | [None][chore] Isolate several intermittent cases (#8408) |
| 2423 | 42ab473bb0 | Yan Chunwei | 2025-10-16 | [https://nvbugs/5583261][ci] waive test_fetch_responses_streaming_sync (#8407) |
| 2424 | ee588a73ac | chinamaoge | 2025-10-16 | [None][fix] Fix the error where checkpoint_dir is assigned as NONE wh… (#8401) |
| 2425 | 0a0159fdd8 | Min Yu | 2025-10-16 | [https://nvbugs/5378031] [feat] W4A8 AWQ MoE supports Per Expert Pre-quant Scale Factor for PyT backend (#7286) |
| 2426 | e75b4f9f65 | Cao Dong | 2025-10-16 | [None][feat] Dev DeepConf (#8362) |
| 2427 | 4143887370 | xiweny | 2025-10-16 | [https://nvbugs/5541494] [fix] Remove waivers (#8353) |
| 2428 | ebf0e51206 | Wanli Jiang | 2025-10-16 | [TRTLLM-8579][feat] Support quantized model for nano-v2-vlm (#8304) |
| 2429 | db1c271bc6 | ChristinaZ | 2025-10-16 | [None][feat] Revise the calculation related to TileN in routing of MOE TRTLLM backend (#8148) |
| 2430 | 206cf31705 | Yan Chunwei | 2025-10-16 | [https://nvbugs/5560921][fix] GenerationExecutor RPC (#8209) |
| 2431 | 40d129a415 | Chuang Zhu | 2025-10-16 | [None][fix] Fix cache buffer size for window (#8320) |
| 2432 | e265eb5fe9 | HuiGao-NV | 2025-10-16 | [None][feat] reuse cudagraph memory pool in normal forward flow (#8095) |
| 2433 | 7a0aa64973 | dongfengy | 2025-10-15 | [None][fix] Refactor triton paddings (#6980) |
| 2434 | 65ec01b257 | QI JUN | 2025-10-15 | [TRTLLM-8532][chore] clean warmup method of ModelEngine (#8264) |
| 2435 | 7efaa5216f | Venky | 2025-10-15 | [None] [chore] Add OSS compliance to CODEOWNERS (#8375) |
| 2436 | 56c20665a9 | Yukun He | 2025-10-15 | [TRTLLM-4501][feat] Add input tensor pre-hook function API for the tuning process. (#6924) |
| 2437 | 0510b34588 | mpikulski | 2025-10-15 | [TRTLLM-8551][feat] add cache_salt in LLM.generate and refactor test_return_logits.py (#8317) |
| 2438 | 1a1c9a29ab | QI JUN | 2025-10-15 | [None][ci] move all llama4 test cases to post merge (#8387) |
| 2439 | 93a4b7f1b6 | mpikulski | 2025-10-15 | [None][chore] update torch_dtype -> dtype in 'transformers' (#8263) |
| 2440 | 616d1df7a0 | QI JUN | 2025-10-15 | [None][chore] set the default value of max_num_tokens explicitly (#8208) |
| 2441 | 6a6124dcb5 | sychen52 | 2025-10-14 | [OMNIML-2336][feat] w4a8 nvfp4 fp8 exports scale factor properly (#8180) |
| 2442 | f4e7738f65 | Erin | 2025-10-14 | [None][doc] Ray orchestrator initial doc (#8373) |
| 2443 | c822c117ce | Kaiyu Xie | 2025-10-15 | [None] [docs] Update TPOT/ITL docs (#8378) |
| 2444 | 206a9930df | Jin Li | 2025-10-15 | [https://nvbugs/5547435][fix] Fix a merge conflict (#8365) |
| 2445 | 493da020c1 | Emma Qiao | 2025-10-15 | [TRTLLM-7351][infra] Add isolate marker for L0 (#7497) |
| 2446 | 9d855f47ad | dongfengy | 2025-10-14 | [None][fix] Remove outdated test waives for GPTOSS (#8183) |
| 2447 | 22471ecc67 | Lizhi Zhou | 2025-10-15 | [TRTLLM-7846][feat] implement etcd storage for disagg cluster (#8210) |
| 2448 | 8444a50d3a | Tailing Yuan | 2025-10-15 | [None][fix] Fix is_post_quant_all2all_supported for MNNVL (#8355) |
| 2449 | 43c46a09db | Lucas Liebenwein | 2025-10-14 | [None][chore] AutoDeplopy: Update expert section on yaml configuration in README (#8370) |
| 2450 | 1cdb0b62c3 | Michal Guzek | 2025-10-14 | [https://nvbugs/5563469][fix] Temporarily disable test_nemotron_nano_8b_lora_torch in L0 due to Torch non-determinism (#8206) |
| 2451 | 6776caaad1 | shuyixiong | 2025-10-14 | [TRTLLM-8507][fix] Fix ray resource cleanup and error handling in LoRA test (#8175) |
| 2452 | 0d20a8fd61 | Fanrong Li | 2025-10-14 | [TRTLLM-8536][feat] Add the sparse attention framework and one use case--RocketKV support (#8086) |
| 2453 | 7291cdc422 | Aurelien Chartier | 2025-10-14 | [https://nvbugs/5404000][fix] Ensure consistency between firstTokenTime and lastTokenTime (#8294) |
| 2454 | 8733e830fc | Chuang Zhu | 2025-10-14 | [None][fix] Add lock for request_to_session in sendReadySingal (#8310) |
| 2455 | 86be06bda4 | Yan Chunwei | 2025-10-14 | [None][ci] waive several rpc tests (#8349) |
| 2456 | 62cea877b1 | Cao Dong | 2025-10-14 | [None][feat] Move StreamGeneration to scaffolding main directory (#8347) |
| 2457 | 72d65d079a | William Zhang | 2025-10-13 | [https://nvbugs/5542878][fix] Unwaive test (#8027) |
| 2458 | 371fcb0338 | xinhe-nv | 2025-10-14 | [TRTLLM-8366][feat] add kimi multi nodes case (#8025) |
| 2459 | d90b4c57cc | yuanjingx87 | 2025-10-13 | [None][infra] Pin numexpr in requirements.txt (#8343) |
| 2460 | 3450fe9944 | Yuxian Qiu | 2025-10-14 | [None][fix] Fix dummy load format for key models. (#7993) |
| 2461 | 9bc055faf1 | Aurelien Chartier | 2025-10-13 | [None][fix] Disable DeepGEMM for Qwen3 MoE Attention layers (#8087) |
| 2462 | 22aa4ac08c | Lucas Liebenwein | 2025-10-13 | [None][feat] AutoDeploy: VLMs with subgraphs + cudagraph/compile (#8203) |
| 2463 | bac665e650 | Zheyu Fu | 2025-10-13 | [TRTLLM-7412][feat] Turn off spec decode when the rolling average acceptance length drops below threshold. (#7283) |
| 2464 | ea4658197f | Grzegorz Kwasniewski | 2025-10-13 | [TRTLLM-6342][feat] Factory TP sharding of quantized models (#8123) |
| 2465 | bd740c9ba6 | Yuxian Qiu | 2025-10-14 | [None][fix] Avoid unnecessary concat in attn_output_gate case. (#8094) |
| 2466 | 6c4cc4c8b2 | mpikulski | 2025-10-13 | [None][fix] workaround for numexpr issue (#8327) |
| 2467 | 4882815fa1 | Yueh-Ting (eop) Chen | 2025-10-14 | [TLLM-6777][feature] Support SWA KV cache reuse OOW block detach (#7922) |
| 2468 | 9ff9fa6413 | Kaiyu Xie | 2025-10-13 | [None] [doc] Update README (#8326) |
| 2469 | 040103ab56 | Kaiyu Xie | 2025-10-13 | [None] [blog] Scaling Expert Parallelism in TensorRT LLM (Part 3: Pushing the Performance Boundary) (#8323) |
| 2470 | db8c63b9b1 | Robin Kobus | 2025-10-13 | [TRTLLM-4517] [feat] Additional model outputs (#7206) |
| 2471 | bbae7a05f0 | amitz-nv | 2025-10-13 | [https://nvbugs/5521949][fix] Replace test_codellama_fp8_with_bf16_lora with test_llama_3_1_8b_fp8_with_bf16_lora (#8199) |
| 2472 | 1e0fbb776d | Fanrong Li | 2025-10-13 | [TRTLLM-8536][feat] Update trtllm gen fmha kernels to support block sparse attention (#8301) |
| 2473 | d145e87f6f | Xianjie Qiao | 2025-10-13 | [None][chore] Update disagg benchmark configs (#8289) |
| 2474 | d882c92a84 | Cao Dong | 2025-10-13 | [None][fix] Fix EventLoopShutdownError (#8260) |
| 2475 | 6fc6f70a68 | Po-Han Huang (NVIDIA) | 2025-10-13 | [https://nvbugs/5441729][test] Fix test_modeling_llama_min_latency.py failures (#7478) |
| 2476 | 9fe63dd8db | xinhe-nv | 2025-10-13 | [None][chore] Add failed cases into waives.txt (#8290) |
| 2477 | fe17e78f27 | Emma Qiao | 2025-10-13 | [None][infra] Add back gb200 multi-node test stage to pre-merge (#8281) |
| 2478 | 8d1b068b1a | Leslie Fang | 2025-10-13 | [TRTLLM-8477][chore] Replace KvCacheConfigCpp with KvCacheConfig inside PyExecutor (#8259) |
| 2479 | 1a9044949f | Yilin Fan | 2025-10-12 | [None][fix] Fix bench_serving import error (#8296) |
| 2480 | 5ce9719759 | xiweny | 2025-10-13 | [https://nvbugs/5503138] [fix] Remove compile warnings (#8167) |
| 2481 | 72fcff1044 | xinhe-nv | 2025-10-13 | [None][fix] add timeout for llama4 (#8254) |
| 2482 | d6e315e9ff | DylanChen-NV | 2025-10-13 | [None][feat] Add torch compile support for cuda core GEMM OP (#8261) |
| 2483 | 989c25fcba | Guoming Zhang | 2025-10-13 | [None][doc] Add qwen3-next doc into deployment guid and test case into L0. (#8288) |
| 2484 | 656d73087e | Guoming Zhang | 2025-10-13 | [None][doc] Fix several invalid ref links in deployment guide sections. (#8287) |
| 2485 | fac47e2826 | amitz-nv | 2025-10-12 | [https://nvbugs/5510879][fix] Fix pytorch & TRT-python flows fused LoRA adapter modules weight split with TP>1 (#8063) |
| 2486 | a1ed03fe8a | Eran Geva | 2025-10-12 | [None][fix] AD test_trtllm_bench to use small model config and skip loading weights (#8149) |
| 2487 | fdbeea51d3 | Emma Qiao | 2025-10-12 | [None][infra] Skip failed cases for main branch (#8293) |
| 2488 | a7ea544dbe | kris1025 | 2025-10-12 | [TRTLLM-7384][feat] enable rejection sampling for CDL (#7731) |
| 2489 | 5798a12199 | Zhanrui Sun | 2025-10-12 | [None][infra] Remove WAR code for GH200 node (#8266) |
| 2490 | 56a539cd37 | brb-nv | 2025-10-10 | [None][chore] Waive failing pre-merge test on main (#8282) |
| 2491 | efd4ffa03b | Ziyi Xiong | 2025-10-11 | [https://nvbugs/5534705][fix] Skip unnecessary CUDA graph capture (#8050) |
| 2492 | 84d2f12818 | Zhenhuan Chen | 2025-10-11 | [TRTLLM-6748][feat] add PDL support for more kernels (#7977) |
| 2493 | 2695d70d42 | Yilin Fan | 2025-10-10 | [None][feat] Add request timing breakdown option in benchmark_serving (#8128) |
| 2494 | 85f157f389 | Chuang Zhu | 2025-10-10 | [None][fix] Add Lock to protect mReqeustToSession (#8085) |
| 2495 | 48c15d805c | QI JUN | 2025-10-10 | [https://nvbugs/5558167][fix] update canceled_req_ids correctly for canceled requests (#8207) |
| 2496 | 2655995a09 | xinhe-nv | 2025-10-10 | [None][fix] add gc for test fixture (#8220) |
| 2497 | d3059dbd8a | bhsueh_NV | 2025-10-10 | [https://nvbugs/5547416][fix] unwaive no_cache test (#8213) |
| 2498 | b555f1ff98 | xinhe-nv | 2025-10-10 | [None][chore] Add failed cases into waives.txt (#8229) |
| 2499 | 795a051765 | HuiGao-NV | 2025-10-10 | [None][chore] Print log with time for starting to load safetensor weights (#8218) |
| 2500 | e8c9bae37e | xinhe-nv | 2025-10-10 | [None][chore] Remove closed bugs (#8151) |
