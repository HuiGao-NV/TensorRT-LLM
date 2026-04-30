# Commit Section 6

Commits 2501 to 3000 (total: 5612)

| # | Commit Hash | Author | Date | Message |
|---|------------|--------|------|--------|
| 2501 | 76a47c7bef | Jonas Li | 2025-10-10 | [None][fix] Enable FP8 ContextMLA on GB300 (#8080) |
| 2502 | 7da4b05289 | Pengbo Wang | 2025-10-10 | [https://nvbugs/5501820][fix] Add requirements for numba-cuda version to WAR mem corruption (#7992) |
| 2503 | 7b6803b6e9 | mpikulski | 2025-10-09 | [TRTLLM-7769][chore] document the role of 'd2t' (#8174) |
| 2504 | ccd949ea5b | Emma Qiao | 2025-10-09 | [None][infra] Waive failed tests on main 10/09 (#8230) |
| 2505 | d560054e1b | amitz-nv | 2025-10-09 | [None][chore] Restore asserts in pytorch flow LoRA tests (#8227) |
| 2506 | e10121345e | QI JUN | 2025-10-09 | [None][ci] pin flashinfer-python version (#8217) |
| 2507 | a193867f8f | Guoming Zhang | 2025-10-09 | [None][doc] Refine deployment guide by renaming TRT-LLM to TensorRT L… (#8214) |
| 2508 | 27677a36f5 | bhsueh_NV | 2025-10-09 | [https://nvbugs/5516666][fix] unwaive some Qwen3 CI tests (#8130) |
| 2509 | fdf29ab8fa | Lizhi Zhou | 2025-10-09 | [TRTLLM-7846][feat] Http disagg-cluster management implemention (#7869) |
| 2510 | 6884d06aed | QI JUN | 2025-10-09 | [None][ci] move some llama4 test cases to pre merge (#8189) |
| 2511 | 9f2a3ae88c | dongfengy | 2025-10-08 | [None][fix] Restrict tinygemm use to certain SMs (#8182) |
| 2512 | ed8e00ad4a | Liao Lanyu | 2025-10-09 | [https://nvbugs/5522746][fix] unwaive tests caused by node issues after rebooting (#8193) |
| 2513 | c88913dc03 | Mike Iovine | 2025-10-08 | [https://nvbugs/5541545][fix] Remove test_llama4 (#8031) |
| 2514 | 80517b7812 | brb-nv | 2025-10-08 | [None][chore] Waive some tests failing on main post merge (#8186) |
| 2515 | 8298e93bd8 | mpikulski | 2025-10-08 | [TRTLLM-8414][chore] BREAKING CHANGE: refine sampling strategy selection (#8132) |
| 2516 | e98616512f | xxi | 2025-10-08 | [https://nvbugs/5550283][fix] update test case to the latest MoE API (#8165) |
| 2517 | d57b8f0951 | Liao Lanyu | 2025-10-08 | [https://nvbugs/5455140][fix] unwaive tests related to GB200 OOM (#8159) |
| 2518 | 971610e3ff | ruodil | 2025-10-08 | [None][test] add test-model-suites option in integration conftest.py (#8016) |
| 2519 | 017583a949 | Sergey Klevtsov | 2025-10-07 | [https://nvbugs/5488576][fix] Propagate disable_finalize_fusion config flag in WIDEEP MoE backend (#8141) |
| 2520 | 7facac077b | Mike Iovine | 2025-10-07 | [None][fix] Fix MTP illegal memory access (#8161) |
| 2521 | ca9da1f1c2 | Emma Qiao | 2025-10-07 | [None][infra] Skip failed cases for main (#8176) |
| 2522 | 9298f1bdcc | xiweny | 2025-10-07 | [None] [test] Add B300 cases to CI (#8056) |
| 2523 | 2b8722b671 | Kanghwan | 2025-10-06 | [None][chore] Increase operations-per-run to 1000 for stale action (#8162) |
| 2524 | 27a5091fcb | Faraz | 2025-10-06 | [None][feat] GPT-OSS Sm120/Sm121 Support (#7937) |
| 2525 | f2657c1ae9 | Izzy Putterman | 2025-10-06 | [None][fix] Eagle: Attention DP (#7939) |
| 2526 | 3492391feb | Lucas Liebenwein | 2025-10-06 | [None][chore] AutoDeploy: clean up accuracy test configs (#8134) |
| 2527 | 98b3af4d4e | mpikulski | 2025-10-06 | [TRTLLM-8413][chore] resolve sampling defaults in OpenAI API backend (#8121) |
| 2528 | 54ab9767b5 | Yan Chunwei | 2025-10-06 | [None][chore] fix llmargs conflict (#8152) |
| 2529 | fba351a211 | Patrice Castonguay | 2025-10-05 | [None][fix] Adding docker folder to Dockerfile (#8138) |
| 2530 | 8060aad239 | amitz-nv | 2025-10-05 | [https://nvbugs/5521949][fix] Re-enable test_bielik_11b_v2_2_instruct_multi_lora, fix its API use with pytorch flow LoRA (#8146) |
| 2531 | fb51de6c2e | Yan Chunwei | 2025-10-05 | [TRTLLM-8189][chore] enhance GenerationExecutor with RPC (part1) (#5543) |
| 2532 | f6654f26a4 | Frida Hou | 2025-10-05 | [#5255][autodeploy] Update FuseAllreduceResidualRMSNorm to use pattern matcher utility; remove fuse_collective (#7545) |
| 2533 | 744246d316 | Frida Hou | 2025-10-03 | [None][autodeploy] small refactors on attention matching (#8079) |
| 2534 | 88ea2c4ee9 | Jonas Yang CN | 2025-10-04 | [TRTLLM-7349][feat] Adding new orchestrator type --  ray (#7520) |
| 2535 | 9d098e3142 | Lucas Liebenwein | 2025-10-03 | [None][feat] AutoDeploy: graph/module inputs with kwargs instead of args (#8137) |
| 2536 | 2c454e8003 | Lucas Liebenwein | 2025-10-03 | [None][feat] AutoDeploy: Nemotron-H accuracy test (#8133) |
| 2537 | 38da871db3 | Michal Guzek | 2025-10-03 | [TRTLLM-6496][feat] Add LoRa Torch tests for the latest NIM model list (#6806) |
| 2538 | ca8291133a | Mike Iovine | 2025-10-03 | [None][fix] Fix MTP 2-model (#8115) |
| 2539 | aaf2c3c2e5 | Lucas Liebenwein | 2025-10-03 | [None][feat] AutoDeploy: compiler backends based on nn.Module (#8126) |
| 2540 | 7bc2d9e993 | Ziyi Xiong | 2025-10-03 | [https://nvbugs/5537878][fix] Reserve an extra slot for padded batch (#7998) |
| 2541 | d8215241d8 | Suyog Gupta | 2025-10-03 | [None][feat] AutoDeploy add autotuning when capturing cudagraphs (#8120) |
| 2542 | 9db4366903 | Aurelien Chartier | 2025-10-03 | [None][fix] Fix Qwen3 FP8 per-tensor when requesting TRTLLM-GEN MoE backend (#8075) |
| 2543 | 5faa5e9dd8 | Lucas Liebenwein | 2025-10-03 | [None][feat] AutoDeploy: dive deeper into token generation bugs + enable_block_reuse (#8108) |
| 2544 | e2f69c5c23 | Robin Kobus | 2025-10-03 | [None] [refactor] Minor cleanup and improvements (#7619) |
| 2545 | ba3dbb6c94 | Erin | 2025-10-02 | [https://nvbugs/5548098][fix] Fix flakey unit test for dynamic spec d… (#8129) |
| 2546 | 9b3d7cc3e6 | Nikita Korobov | 2025-10-03 | [None][feat] Update TRT-LLM Gen MoE kernels (#7970) |
| 2547 | 01423ac183 | Yilin Fan | 2025-10-02 | [None][feat] perf_metrics endpoint functionality improvement (#8005) |
| 2548 | a5b59fd31d | Grzegorz Kwasniewski | 2025-10-03 | [TRTLLM-6342][bug] Patched incorrect starcoder tp config (#8118) |
| 2549 | 4136942436 | Eran Geva | 2025-10-02 | [#7588][fix] fixed the kv cache size parsing in test_perf.py AD backend (#8092) |
| 2550 | 08a47918cf | Patrice Castonguay | 2025-10-02 | [None][chore] Adding install_tensorrt.sh script to pip wheel (#8116) |
| 2551 | ab433b7228 | Daniel Cámpora | 2025-10-02 | [None][fix] Fix access to new tokens in sampler. (#7958) |
| 2552 | fefa7d8fa3 | Patrice Castonguay | 2025-10-02 | [None][feat] Support for cancelling requests with disaggregation (#8114) |
| 2553 | 6568e565db | dongfengy | 2025-10-02 | [TRTLLM-7775][feat] Integrate tinygemm2 for gpt-oss (#7916) |
| 2554 | 34d158b6da | yifeizhang-c | 2025-10-03 | [TRTLLM-6589][feat] Support CUDA graph for DeepEP (#7514) |
| 2555 | 293637e0a1 | Erin | 2025-10-02 | [https://nvbugs/5556020][chore] waive test_eagle3 (#8119) |
| 2556 | fc7f78c400 | mpikulski | 2025-10-02 | [TRTLLM-8269][test] do not explicitly pass temperature=0 to select greedy sampling (#8110) |
| 2557 | 32c7f8c36f | Eran Geva | 2025-10-02 | [#7588][feat] lock gpu clocks in test_perf.py to reliably detect perf regressions (#8099) |
| 2558 | 726ac07cc0 | Chang Liu | 2025-10-01 | [https://nvbugs/5549081][fix] Fix device id assignment for some vision models (#8070) |
| 2559 | bd3d0ad233 | brb-nv | 2025-10-01 | [TRTLLM-7733][feat] Executor changes to support helix parallelism (#7972) |
| 2560 | 1ad7bc4c78 | Izzy Putterman | 2025-10-01 | [None][feat] Draft: Save state first pass (#7012) |
| 2561 | e107749a69 | Bo Deng | 2025-10-02 | [None][fix] fix patchelf version issue (#8112) |
| 2562 | de99e23696 | Frida Hou | 2025-10-01 | [#5860][feat] Add ModelOPT INT4 awq fake quant support in AutoDeploy (#7770) |
| 2563 | d7581bb551 | Yibin Li | 2025-10-01 | [TRTLLM-8031][feat] Add chunked return_generation_logits logic (#7831) |
| 2564 | 6fd225833c | Grzegorz Kwasniewski | 2025-10-01 | [TRTLLM-6342][bug] Fix shape propagation after TP sharding (#7912) |
| 2565 | ba8abeab10 | sychen52 | 2025-09-30 | [OMNIML-2336][feat] add W4A8 NVFP4 FP8 fused moe (#7968) |
| 2566 | b77f19f4ff | Patrice Castonguay | 2025-10-01 | [https://nvbugs/5434320][fix] fix: Unwaiving disagg pp tests (#8069) |
| 2567 | b1e3fef8aa | Emma Qiao | 2025-10-01 | [None][infra] Skip failed tests in post-merge for main (#8102) |
| 2568 | e9e4632e44 | Yechan Kim | 2025-10-01 | [None][doc] Add more description on EXAONE usage (#8089) |
| 2569 | 808e556c79 | peaceh-nv | 2025-10-01 | [None][fix] : Fix OOM issue when dp padding is enabled (#8052) |
| 2570 | 84aa3c981e | brb-nv | 2025-09-30 | [None][chore] Waive failing MNNVL alltoall multi-gpu test (#8106) |
| 2571 | ee5ae49337 | mpikulski | 2025-09-30 | [TRTLLM-8269][fix] Revert "do not explicitly pass temperature=0 to select greedy sampling" (#8103) |
| 2572 | b4be0d2e4c | Guoming Zhang | 2025-10-01 | [None][chore] Refine qwen3-next implementation. (#8064) |
| 2573 | c510b67fa0 | Iman Tabrizian | 2025-09-30 | [https://nvbugs/5547414][fix] avoid downloading Tiny llama from HF (#8071) |
| 2574 | 1560cca227 | Yiqing Yan | 2025-09-30 | [None][chore] Bump version to 1.2.0rc1 (#8097) |
| 2575 | 948b8b9569 | Yechan Kim | 2025-09-30 | [None][fix] Fix CUDA graph for Qwen2.5-VL (#8047) |
| 2576 | 1dba9fa89e | xinhe-nv | 2025-09-30 | [TRTLLM-6239][feat] add test cases into QA test list (#8081) |
| 2577 | b0cb9ca50e | Kaiyu Xie | 2025-09-30 | [None] [test] Add MNNVL AlltoAll tests to pre-merge (#7466) |
| 2578 | dcfd3ef81c | Lucas Liebenwein | 2025-09-29 | [#4593][feat] AutoDeploy: Linear Attention Support (SSM + causal_conv + Bamba + Nemotron-H) (#8068) |
| 2579 | 62010c0ab7 | Cao Dong | 2025-09-30 | [None][feat] Return topk logprobs in torch backend (#7976) |
| 2580 | cdce68c3e0 | Cheng Hang | 2025-09-30 | [TRTLLM-6741][fix] Add heuristics for lm head tp size when `enable_lm_head_tp_in_adp=True` (#7891) |
| 2581 | 6396cb9208 | Patrice Castonguay | 2025-09-29 | [https://nvbugs/5538098][fix] Checking connection to etcd server in unit test (#8006) |
| 2582 | 334e2cab0d | Chang Liu | 2025-09-29 | [https://nvbugs/5542867][fix] Fix the non-determinism issue in the mm_encoder test (#8033) |
| 2583 | e5f9b6aaa0 | amitz-nv | 2025-09-29 | [None][fix] Fix TRT-python multi LoRA TP=2 test arguments (#8059) |
| 2584 | 31a1a5ff80 | mpikulski | 2025-09-29 | [TRTLLM-8269][test] do not explicitly pass temperature=0 to select greedy sampling (#7909) |
| 2585 | 38d6e4e60b | bhsueh_NV | 2025-09-29 | [None][feat] Support Qwen3 next (#7892) |
| 2586 | a0d489a8d5 | mpikulski | 2025-09-29 | [TRTLLM-7728][perf] improve batched sampling perf for contiguous batches (#7908) |
| 2587 | 560ded5450 | Yiqing Yan | 2025-09-29 | [None][chore] Bump version to 1.2.0rc0 (#7941) |
| 2588 | 48e779ae8c | xiweny | 2025-09-29 | [https://nvbugs/5541494] [fix] add back missing sm100f bmm kernels (#8051) |
| 2589 | 3ba6727a68 | yufeiwu-nv | 2025-09-29 | [None][test] Update get_sysinfo.py to avoid UnboundLocalError (#7982) |
| 2590 | b2095aa074 | Gal Hubara-Agam | 2025-09-29 | [#4674][bugfix] AutoDeploy Fix memory leak in fuse_moe (#7844) |
| 2591 | 20e6cd39f1 | xinhe-nv | 2025-09-29 | [None][chore] Add failed cases into waives.txt (#8043) |
| 2592 | ce381d6813 | Emma Qiao | 2025-09-29 | [None][infra] Waive failed cases for main on 0929 (#8053) |
| 2593 | 7f1e2dba92 | Void | 2025-09-29 | [None][fix] only support deepep post quant all2all on nvfp4 (#8041) |
| 2594 | 1339beb04e | HuiGao-NV | 2025-09-29 | [None][ci] Disable tensorRT cases in post-merge (#8028) |
| 2595 | 7ac932d45e | HuiGao-NV | 2025-09-29 | [https://nvbugs/5532087][CI] Enable test case (#8029) |
| 2596 | 985b79ca82 | Tailing Yuan | 2025-09-29 | [TRTLLM-8348][feat] Speed up concat k and copy k_nope in context phase using torch.compile (#8044) |
| 2597 | 1e2e851db8 | Ivy Zhang | 2025-09-29 | [None][chore] update test case constraint (#8020) |
| 2598 | 9cea6bfb30 | Eran Geva | 2025-09-29 | [#7288][feat] Added AutoDeploy backend support to test_perf.py (#7588) |
| 2599 | 030254f88a | Kaiyu Xie | 2025-09-29 | [None] [doc] Document hang issue caused by `UnpicklingError` (#8049) |
| 2600 | 4be533183f | Zhenhua Wang | 2025-09-29 | [None][chroe] Update cron schedule for closing inactive issues (#8048) |
| 2601 | 0ecafd84da | Ivy Zhang | 2025-09-29 | [None][chore] Update chunked prefill test case configs (#7868) |
| 2602 | e9f26feeb6 | Zongfei Jing | 2025-09-29 | [None][chore] Cherry-pick from (#7598) Make low_precision_combine as a llm arg (#7898) |
| 2603 | 28b9a81c58 | Yukun He | 2025-09-29 | [TRTLLM-4500][feat] Add serialization/deserialization options for AutoTuner profiling cache (#7738) |
| 2604 | 563e588e56 | WeiHaocheng | 2025-09-28 | [None][doc] Scaffolding tech blog fix a typo (#8042) |
| 2605 | 3ba4bf6e70 | Guoming Zhang | 2025-09-28 | [None][chore] Disable concurrent weights loading for _load_weights_im… (#8034) |
| 2606 | 2be05cbd6e | Emma Qiao | 2025-09-28 | [None][infra] Skip failed test for main branch on 9/28 (#8040) |
| 2607 | 51aefd1bac | Guoming Zhang | 2025-09-28 | [None][doc] Refine perf overview.md and correct the error link in per… (#8035) |
| 2608 | 95eac2cda7 | ChristinaZ | 2025-09-28 | [https://nvbugs/5537738][fix] Add fp8 post-quant allgather support (#8008) |
| 2609 | 77b68d9d7d | Aurelien Chartier | 2025-09-27 | [https://nvbugs/5461712] [fix] Use DG for Qwen3 Linear layers (#8030) |
| 2610 | c8f98b3065 | Xianjie Qiao | 2025-09-28 | [None] [feat] Update disagg gen-only benchmark. (#7917) |
| 2611 | 33282351a2 | Iman Tabrizian | 2025-09-27 | [TRTLLM-6106][feat] Add support for KVCache transfer from KVCache reuse path (#6348) |
| 2612 | a36b48bcab | Frida Hou | 2025-09-26 | [#5860][autodeploy] GPT-OSS MXFP4 support (#7451) |
| 2613 | c33f43e13a | Jhao-Ting Chen | 2025-09-26 | [https://nvbugs/5518713][fix] Trtllm-gen moe backend for blockwise fp8 ckpt (Qwen3-235B-A22B-FP8) (#7856) |
| 2614 | d7087015f1 | Mike Iovine | 2025-09-26 | [TRTLLM-8271][fix] Fix CDL overlap scheduling performance (#7971) |
| 2615 | c8bef27ebb | Emma Qiao | 2025-09-27 | [None][infra] Waive failed cases in post-merge 2305 (#8019) |
| 2616 | a4243f0da5 | YueWeng | 2025-09-27 | [TRTLLM-6393][feat] add static tree sampling and verification (#7161) |
| 2617 | f4d3be4bbc | HuiGao-NV | 2025-09-26 | [None][feat] Add a standalone buffer cache class and reuse buffers between cduagraph and no-graph flow (#7669) |
| 2618 | b11ee868c5 | Tailing Yuan | 2025-09-26 | [https://nvbugs/5495789][feat] Optionally disable server GC and worker GC (#7995) |
| 2619 | 6dc50ebcdd | Martin Marciniszyn Mehringer | 2025-09-26 | [None][chore] Require NVIDIA developers to use their full name or NVIDIA account in GitHub profiles (#8022) |
| 2620 | 35edad37f9 | WeiHaocheng | 2025-09-26 | [None][doc] Add scaffolding tech blog to cover (#8021) |
| 2621 | ba6ab62bd1 | xinhe-nv | 2025-09-26 | [None][chore] Add failed cases into waives.txt (#8004) |
| 2622 | f32f5730b2 | xinhe-nv | 2025-09-26 | [None][chore] Add failed cases into waives.txt (#7986) |
| 2623 | 2db22fb4e5 | Yueh-Ting (eop) Chen | 2025-09-26 | [None][feature] Add environment variable to adjust block pool allocation ration under kv cache manager (#7923) |
| 2624 | a9965d84e0 | HuiGao-NV | 2025-09-26 | [None][chore] Report NCCL error message but not OOM when NCCL error happens (#8009) |
| 2625 | 55ce70060e | peaceh-nv | 2025-09-26 | [https://nvbugs/5451740][fix] Add DP padding back on SM120 (#7965) |
| 2626 | 3a96d75a3c | Lucas Liebenwein | 2025-09-26 | [https://nvbugs/5527956][fix] AutoDeploy: fix IMA due to outdated metadata (#8002) |
| 2627 | 2e5850c28a | sunnyqgg | 2025-09-26 | [TRTLLM-7330][feat] Eagle3 cuda graph support for the first draft model inference (#7363) |
| 2628 | f98fa0cf8b | Chuang Zhu | 2025-09-26 | [None][feat] Optimize kv cache transfer TEP (#7613) |
| 2629 | 4c0f8482f1 | QI JUN | 2025-09-26 | [None][ci] Waive test_mm_encoder_standalone.py::test_multi_request_batch_chat[llava-v1.6-mistral-7b-hf]  (#8010) |
| 2630 | fae83c387b | Yuan Tong | 2025-09-26 | [#6102][fix] support non-system python installation (#7763) |
| 2631 | d650320de4 | Enwei Zhu | 2025-09-26 | [None][infra] Improve the failure message for accuracy test suite (#7994) |
| 2632 | 108248ece1 | Yiqing Yan | 2025-09-26 | [TRTLLM-7999][infra] Add B300/GB300 single gpu test (#7951) |
| 2633 | 7e2521a7f0 | Yanchao Lu | 2025-09-26 | [None][chore] Some clean-ups for CUDA 13.0 dependencies (#7979) |
| 2634 | 1eb653146a | dongfengy | 2025-09-25 | [https://nvbugs/5525951][fix] Clarify that PP is not supported for GPTOSS (#7911) |
| 2635 | 1529a6f22d | QI JUN | 2025-09-26 | [None][chore] extract weights loading related logic to model loader (#7579) |
| 2636 | 2dc93c6371 | Emma Qiao | 2025-09-25 | [None][infra] Waive failed tests on main (#8001) |
| 2637 | 4b0570a0d6 | WeiHaocheng | 2025-09-25 | [None][doc] Add acknowledgements in scaffolding tech blog (#7983) |
| 2638 | 57ff5f4c0d | xxi | 2025-09-25 | [None][fix] fix a bug in wideEp use DeepEP with num_chunks > 1 (#7954) |
| 2639 | eda1467061 | Matthias Jouanneaux | 2025-09-25 | [TRTLLM-5966][feat] Helix: add alltoall op (#6815) |
| 2640 | 396c0ea677 | PeganovAnton | 2025-09-25 | [None][chore] relax version constraints on fastapi (#7935) |
| 2641 | c5012423f5 | Yueh-Ting (eop) Chen | 2025-09-25 | [None][chore] Remove developer name in comment (#7981) |
| 2642 | 40c6103ef8 | Yan Chunwei | 2025-09-24 | [None][doc] add Llama PP known issue to release note (#7959) |
| 2643 | 663ce3a4de | Guoming Zhang | 2025-09-23 | [None][doc] fix invalid links in perf benchmarking. (#7933) |
| 2644 | 202bed4574 | Guoming Zhang | 2025-09-23 | [None][chroe] Rename TensorRT-LLM to TensorRT LLM for source code. (#7851) |
| 2645 | 961418908c | QI JUN | 2025-09-22 | [https://nvbugs/5531963][fix] cherry pick #7725 (#7907) |
| 2646 | 5999fab146 | Yan Chunwei | 2025-09-22 | [https://nvbugs/5427043][fix] cherrypick: request length exceeds max_num_tokens (#7718) |
| 2647 | cb466a846d | Yan Chunwei | 2025-09-22 | [None][fix] api stability bug in status label (#7861) |
| 2648 | 9d48898def | Yan Chunwei | 2025-09-22 | [None][doc] add stable label to all the un-labelled arguments in LLM class (#7863) |
| 2649 | c38d4cf6a6 | Zac Patel | 2025-09-21 | [None][doc] Update Perf-Overview.md for release/1.0 (#7848) |
| 2650 | 57c098956e | Yan Chunwei | 2025-09-22 | [None][doc] add a guide for modifying APIs (#7866) |
| 2651 | 9f0f52249e | Guoming Zhang | 2025-09-19 | [None][doc] Rename TensorRT-LLM to TensorRT LLM for homepage and the … (#7850) |
| 2652 | 5ecc8d0ee2 | Guoming Zhang | 2025-09-18 | [None][doc] Replace the main in the examples' link with commit id. (#7837) |
| 2653 | 5342c607cd | Yan Chunwei | 2025-09-18 | [https://nvbugs/5516710][fix] fix Llama 3.3 TP PP case (#7717) |
| 2654 | 44d7c3b245 | Tao Li @ NVIDIA | 2025-09-18 | [https://nvbugs/1234567][fix] Revert https://github.com/NVIDIA/TensorRT-LLM/pull/7768/files (#7813) |
| 2655 | 4a09be40f0 | Guoming Zhang | 2025-09-17 | [None][doc] Update docker cmd in quick start guide and trtllm-serve … (#7787) |
| 2656 | e30d9aced9 | xinhe-nv | 2025-09-25 | [https://nvbugs/4955671][fix] update test list (#7980) |
| 2657 | 791e73edf6 | Chuang Zhu | 2025-09-25 | [https://nvbugs/5536141][fix] fix_disagg_single_gpu_test (#7990) |
| 2658 | b622cde5d5 | Jinyang Yuan | 2025-09-25 | [None][perf] Fix the tactic sorting in TrtllmGenBatchedGemmRunner::getValidConfigIndices (#7419) |
| 2659 | cb53261aaf | Emma Qiao | 2025-09-25 | [None][infra] Unwaive some tests since dev already have a PR to collect more info (#7984) |
| 2660 | 22b45ff9c7 | Wanli Jiang | 2025-09-25 | [TRTLLM-7758][feat] Phi4-mm image modality inference optimization (#7918) |
| 2661 | 259cc66c34 | WeiHaocheng | 2025-09-25 | [None][doc] scaffolding tech blog part one (#7835) |
| 2662 | 0945403174 | fredricz-20070104 | 2025-09-25 | [TRTLLM-6541][test] Add NIM perf test cases (#7924) |
| 2663 | bb6067176f | Guoming Zhang | 2025-09-25 | [None][chroe] Update the cuda and tensorrt version in homepage icons. (#7963) |
| 2664 | 98726a3bed | Aurelien Chartier | 2025-09-24 | [None][chore] Update trtllm-bench documentation on setting FP8 KV cache (#7885) |
| 2665 | 336c2ef540 | Void | 2025-09-25 | [None][feat] DeepEP LL fp8 dispatch/combine (#7927) |
| 2666 | be7e51727e | Iman Tabrizian | 2025-09-24 | [https://nvbugs/5456485][bug] unwaive triton test (#7966) |
| 2667 | 342014069e | Leslie Fang | 2025-09-25 | [None][chore] Validate features combination (#7630) |
| 2668 | da30d496b0 | Iman Tabrizian | 2025-09-24 | [None][fix] Revert "[None][feat] Return topk logprobs in torch backend (#7756)" (#7969) |
| 2669 | 5a65af24cd | sychen52 | 2025-09-24 | [OMNIML-2336][feat] Add NVFP4 x FP8 moe kernels (#7821) |
| 2670 | 6d45cd163e | Iman Tabrizian | 2025-09-24 | [None][bug] Fix transformers version for Triton backend (#7964) |
| 2671 | 42c2ec3239 | Mike Iovine | 2025-09-24 | [https://nvbugs/5473781][fix] Fix llama 4 FP8 for PP>1 (#7220) |
| 2672 | b1dc84b4a3 | Pamela Peng | 2025-09-24 | [TRTLLM-7399][test] Add DS-R1/Qwen3 test cases for RTX 6000 (#7662) |
| 2673 | 48fda86c56 | Yuxian Qiu | 2025-09-24 | [None][fix] Fix dummy load format for DeepSeek. (#7874) |
| 2674 | 6e5e8b8a3b | Macrocell | 2025-09-24 | [None][fix] fix get_iteration_stats IndexError (#7216) |
| 2675 | 603517f72a | Eran Geva | 2025-09-24 | [#7675][feat] CapturedGraph to support max_batch_size > max(cuda_graph_batch_sizes) (#7888) |
| 2676 | 51bef1beb0 | Yuan Tong | 2025-09-24 | [None][chore] cleanup build script (#7865) |
| 2677 | 60101eb8a5 | Perkz Zheng | 2025-09-24 | [None][fix] trtllm-gen cubins compiled with wrong arch. (#7953) |
| 2678 | c8bda4b3a9 | HuiGao-NV | 2025-09-24 | [None][ci] Waive some intermittent failures (#7955) |
| 2679 | cfbcf9b9e8 | Necofish | 2025-09-24 | [None][feat] Support Seed-OSS model in pytorch backend (#7496) |
| 2680 | a1a57e83b8 | Enwei Zhu | 2025-09-24 | [TRTLLM-5235][feat] Enable regex and EBNF grammar in trtllm-serve (#7925) |
| 2681 | b8bfa63197 | xinhe-nv | 2025-09-24 | [None][chore] add test_w4_1gpu[True-True-cutlass-fp8] & TestKimiK2::test_fp8_blocks… (#7944) |
| 2682 | 18ff1e31b8 | QI JUN | 2025-09-24 | [None][ci] remove duplicate test cases (#7956) |
| 2683 | f323b74d42 | yufeiwu-nv | 2025-09-24 | [None][test] Update llm_models_root to improve path handling on BareMetal environment (#7876) |
| 2684 | 29e63d3bc2 | HuiGao-NV | 2025-09-24 | [https://nvbugs/5532248][fix] Fix fused_moe OOM (#7931) |
| 2685 | 6654b78c94 | JunyiXu-nv | 2025-09-24 | [https://nvbugs/5521799][fix] Trim incorrectly generated harmony messages (#7849) |
| 2686 | 0252cee4c3 | Li Min | 2025-09-24 | [None][chore] Recover cutlass-dsl pkg install and dsl op testing. (#7945) |
| 2687 | 946ffcd2eb | QI JUN | 2025-09-24 | [None][ci] optimize test cases of dgx b200 (#7948) |
| 2688 | 2f8dc6feb0 | Cao Dong | 2025-09-24 | [None][feat] Return topk logprobs in torch backend (#7756) |
| 2689 | 62563760fb | xinhe-nv | 2025-09-24 | [None][chore] update chunked prefill cases (#7921) |
| 2690 | 929ef4c474 | qsang-nv | 2025-09-24 | [None][chore] remove cubins for ci cases (#7902) |
| 2691 | b890d7fea4 | Pengbo Wang | 2025-09-24 | [None][infra] Skip failed test for nvbugs 5537738 (#7946) |
| 2692 | 276d83c898 | xiweny | 2025-09-24 | [https://nvbugs/5532225] [fix] MoE use stream-dependent workspace (#7940) |
| 2693 | cf100933cc | Yueh-Ting (eop) Chen | 2025-09-24 | [TRTLLM-6341][feature] Support SWA KV cache reuse (#6768) |
| 2694 | 5ccb2dea33 | Daniel Cámpora | 2025-09-24 | [None][chore] Make sampler type beta. (#7934) |
| 2695 | 70c3b100eb | Yuan Tong | 2025-09-24 | [#7692][fix] recognize RequestError as per-request error in background handler (#7726) |
| 2696 | f050b8d871 | Yuan Tong | 2025-09-24 | [None][fix] refine `backend` option handling for commands (#7829) |
| 2697 | e4f1f90202 | Lizhi Zhou | 2025-09-24 | [https://nvbugs/5477404][chore] unwaive test_disaggregated_single_gpu.py::test_disaggregated_llama_context_capacity (#7857) |
| 2698 | 31ef03fd82 | Ziyi Xiong | 2025-09-24 | [https://nvbugs/5528405][fix] Set up draft_tokens before scheduling (#7903) |
| 2699 | 6ff0fad75e | Venky | 2025-09-23 | [TRTLLM-7015] [feat] Enable `prompt_logprobs` in pytorch backend (#7580) |
| 2700 | 7550251988 | Lizhi Zhou | 2025-09-24 | [TRTLLM-7182][test] add multi-nodes test for disagg-serving (#7470) |
| 2701 | 9970345919 | mpikulski | 2025-09-24 | [TRTLLM-7728][feat] batched sampling by strategy (supersedes enable_mixed_sampler, cf. TRTLLM-7156) (#7294) |
| 2702 | 220dc01372 | Jhao-Ting Chen | 2025-09-23 | [None][feat] support JIT mha.cu for SPEC_DEC in runtime (#6078) |
| 2703 | e3c1a9409f | Zheng Duan | 2025-09-24 | [TRTLLM-6549][fix] add kv cache time output back (#7798) |
| 2704 | 7d4d6cc9e0 | Yilin Fan | 2025-09-23 | [TRTLLM-7292][feat] Support multi-threaded tokenizers for trtllm-serve (cherry-pick) (#7776) |
| 2705 | 1f2761e67b | Tracin | 2025-09-24 | [None][feat] Enable gpt oss on DGX H100. (#6775) |
| 2706 | 9f1d9b7b18 | Daniel Cámpora | 2025-09-23 | [None][feat] Use list instead of torch tensor for new tokens in update requests (#7730) |
| 2707 | 6a36349964 | Yanchao Lu | 2025-09-23 | [None][test] Waive another intermittent OOM test (#7930) |
| 2708 | 34963ec39c | Zheyu Fu | 2025-09-23 | [None][fix] Assign [] to req.py_draft_tokens instead of None when spec decode is off (#7511) |
| 2709 | 16bb76c31d | Zero Zeng | 2025-09-23 | [None][chore] Update benchmark script (#7860) |
| 2710 | dd5fb2857a | ChristinaZ | 2025-09-23 | [None][fix] Re-add the import for allgather that was mistakenly removed. (#7920) |
| 2711 | 3ba19b6ff1 | Yan Chunwei | 2025-09-23 | [https://nvbugs/5532023][fix] executor with-statement bug (#7895) |
| 2712 | bb64e7462c | Perkz Zheng | 2025-09-23 | [None][fix] fix a bug with trtllm-gen kernels + attention sinks (#7919) |
| 2713 | f882fb86db | Enwei Zhu | 2025-09-23 | [https://nvbugs/5367180][fix] Fix xgrammar import before loading tensorrt_llm binary (#7906) |
| 2714 | 40820e6711 | Yan Chunwei | 2025-09-23 | [None][fix] CHERRY-PICK trtllm-serve yaml loading (#7551) (#7897) |
| 2715 | 05bec3bf0f | ruodil | 2025-09-23 | [None][test] rename llm_perf_full to llm_perf_core and add missing cases (#7899) |
| 2716 | a4b4ed4535 | Pengbo Wang | 2025-09-23 | [None][fix] Fix and add test for TRTLLM MoE backend (#7755) |
| 2717 | 5792464d37 | Pengbo Wang | 2025-09-23 | [None][fix] Read eos_token_id from generation_config for kimi_k2 (#7120) |
| 2718 | 08cc7a041f | Pengbo Wang | 2025-09-23 | [https://nvbugs/5355128][fix] Add missing wgmma intrinsic for starcoder (#7643) |
| 2719 | 126cd707e3 | yunruis | 2025-09-23 | [None][opt] Add batch waiting when scheduling (#7416) |
| 2720 | 998857bcde | Chang Liu | 2025-09-22 | [TRTLLM-7328][feat] E-PD Disagg Support via llmapi (3/N) (#7577) |
| 2721 | 9da4203e2e | jianweiwu | 2025-09-23 | [None][feat] Add Tencent HunYuanDenseV1 model support (#7081) |
| 2722 | 740340dd17 | Tailing Yuan | 2025-09-23 | [https://nvbugs/5522847][fix] Disable GC on disagg server and client (#7858) |
| 2723 | 8330d5363a | Enwei Zhu | 2025-09-23 | [TRTLLM-8209][feat] Support new structural tag API (upgrade XGrammar to 0.1.25) (#7893) |
| 2724 | d471655242 | xxi | 2025-09-23 | [TRTLLM-7831][feat] Cherry-pick from #7423 Support fp8 block wide ep cherry pick (#7712) |
| 2725 | 59f57598a7 | Enwei Zhu | 2025-09-23 | [https://nvbugs/5504086][fix] Fix MTP vanilla (#7904) |
| 2726 | be576a3152 | ChristinaZ | 2025-09-23 | [None] [feat] Enable run_post_quant_allgather for MoE TRTLLM backend (#6794) |
| 2727 | b5391b4ac6 | Jin Li | 2025-09-23 | [https://nvbugs/5516665][fix] Fix CUTLASS moe fake impl errors (#7714) |
| 2728 | b1738c3f18 | Linda | 2025-09-22 | [https://nvbugs/5477359][fix] Removing test waivers (#7877) |
| 2729 | 2a30f11d63 | Wanli Jiang | 2025-09-22 | [None][chore] Upgrade transformers to 4.56.0 (#7523) |
| 2730 | fadce99af4 | Yan Chunwei | 2025-09-22 | [https://nvbugs/5351244][fix] CHERRY-PICK test_mpi_session (#7501) (#7900) |
| 2731 | 324301ccba | Emma Qiao | 2025-09-22 | [None][infra] Skip failed test for nvbugs 5532023 (#7905) |
| 2732 | f77aca9f2c | Yechan Kim | 2025-09-22 | [TRTLLM-7385][feat] Optimize Qwen2/2.5-VL performance (#7250) |
| 2733 | 0dac1ddb74 | HuiGao-NV | 2025-09-22 | [https://nvbugs/5525849][fix] Cherry-pick to fix mismatch of max seq len between kv cache manager and dummy requests (#7855) |
| 2734 | 8cf95681e6 | Bo Deng | 2025-09-22 | [TRTLLM-7989][infra] Bundle UCX and NIXL libs in the TRTLLM python package (#7766) |
| 2735 | d330d0005c | Emma Qiao | 2025-09-22 | [None][infra] Waive a failed case on main (#7901) |
| 2736 | 9c1b75e978 | xinhe-nv | 2025-09-22 | [TRTLLM-7070][feat] add gpt-oss chunked prefill tests (#7779) |
| 2737 | ab26d21620 | Yukun He | 2025-09-17 | [https://nvbugs/5517023][fix] Pass allreduce strategy and force NCCL on pre-Blackwell arch (#7768) |
| 2738 | edbe270198 | Guoming Zhang | 2025-09-17 | [TRTLLM-7958][doc] add 1.0 release notes (#7605) |
| 2739 | ba2864a2c6 | Yan Chunwei | 2025-09-17 | [None][doc] Enhance api reference doc by labeling stable APIs (#7751) |
| 2740 | f5bfd68a50 | Wanli Jiang | 2025-09-16 | [https://nvbugs/5509024][fix] Print full parsed outputs and update keywords for multimodal model (#7670) |
| 2741 | e8a3e21b87 | Guoming Zhang | 2025-09-16 | [https://nvbugs/5519525][fix] fix doc invalid link for bug 5519525 (#7753) |
| 2742 | f9c9c3f50a | Yi Zhang | 2025-09-16 | [https://nvbugs/5355219][fix] Fix trtllm moe backend  test config and Qwen3 MoE multi node (#7724) |
| 2743 | 022bc96fb6 | Ivy Zhang | 2025-09-15 | [https://nvbugs/5512734][fix] Update kv cache config for maverick (#7710) |
| 2744 | ef557f880b | bhsueh_NV | 2025-09-15 | [https://nvbugs/5437405][fix] cherry-pick PR 7000 (qwen3 235b eagle3 ci) (#7702) |
| 2745 | bc7b50334c | Guoming Zhang | 2025-09-15 | [None][doc] Add labels description note into llm api section (#7696) |
| 2746 | 5c8b022d1e | Yanchao Lu | 2025-09-15 | [None][ci] Test waives for the release/1.0 branch 09/15 (#7700) |
| 2747 | 8879ec4d35 | brb-nv | 2025-09-10 | [https://nvbugs/5501557][fix] Fix out-of-bounds vector access for model with multiple layer types (#7636) |
| 2748 | ab915fb333 | Guoming Zhang | 2025-09-09 | [None][doc] Use hash id for external link (#7641) |
| 2749 | 5c54173054 | Guoming Zhang | 2025-09-09 | [None][doc] Fix a invalid link and a typo. (#7634) |
| 2750 | 8fed8ee066 | Guoming Zhang | 2025-09-02 | [None][doc] add blackwell information into support matrix (#6740) |
| 2751 | 2ffc33921f | Yan Chunwei | 2025-09-08 | [https://nvbugs/5416501][doc] add known issues to llmapi doc (#7560) |
| 2752 | 99995846b3 | Simeng Liu | 2025-09-08 | [https://nvbugs/5470782][chore] Remove the skip statement in 1.0 rele… (#7573) |
| 2753 | 541b7fda89 | peaceh-nv | 2025-09-09 | [https://nvbugs/5503423][waive] Waive Llama3.1-70B-FP8 test on RTX PRO 6000 (#7603) |
| 2754 | af34c9713a | HuiGao-NV | 2025-09-09 | [https://nvbugs/5474169][fix] seq_len mismatch between kv cache manager and graph attn metadata (#7606) |
| 2755 | 3cc16c2438 | Yukun He | 2025-09-04 | [https://nvbugs/5496960][fix] Fix Gemma model forward. (#7509) |
| 2756 | afca2fcbe0 | Yan Chunwei | 2025-09-04 | [https://nvbugs/5351244][fix] test_mpi_session (#7501) |
| 2757 | 2d46dda6a7 | Yuxian Qiu | 2025-09-01 | [https://nvbugs/5448754][fix] Download HF model for all nodes. (#6824) |
| 2758 | 123f5cbbf0 | HuiGao-NV | 2025-09-01 | [https://nvbugs/5474169][fix]Adjust max seq len for kvcache for memory estimation (#7391) |
| 2759 | 293d9fb612 | Lizhi Zhou | 2025-08-29 | [https://nvbugs/5448767][fix] disable kv cache reuse for disagg pp>1 tests (#7354) |
| 2760 | a15f08db3d | Bo Li | 2025-08-29 | [https://nvbugs/5467548][fix] DeepSeek illegal memory access. (#7298) |
| 2761 | 8484aa9858 | Barry Kang | 2025-09-22 | [None][fix] Fix DeepGEMM commit (#7875) |
| 2762 | 8aead224fb | Stefan Niebler | 2025-09-22 | [https://nvbugs/5513423][fix] Correctly respect min_tokens in PyTorch Workflow (#7808) |
| 2763 | 9dc7316b7f | peaceh-nv | 2025-09-22 | [https://nvbugs/5512556][unwaive] Unwaive DeepSeek PP tests (#7828) |
| 2764 | b057fc9593 | dongxuy04 | 2025-09-22 | [None][fix] cherrypick to main: Fix possible mpi broadcast and gather issue on large object (#7854) |
| 2765 | 639d4109a7 | Enwei Zhu | 2025-09-22 | [None][fix] Disable torch.compile for CapturableGuidedDecoder (#7871) |
| 2766 | 9eb8084ca9 | dongxuy04 | 2025-09-22 | [TRTLLM-7008][fix] cherrypick to main Add automatic shared memory delete if already exist (#7727) |
| 2767 | 822cb0115b | xiweny | 2025-09-21 | [TRTLLM-6286] [perf] Add NoSmem epilogue schedule and dynamic cluster shape for sm10x group gemm (#7757) |
| 2768 | 897c4dd23b | Ziyi Xiong | 2025-09-21 | [https://nvbugs/5517404][fix] Use the correct cuda graph for dynamic spec dec (#7728) |
| 2769 | 4509d97780 | Yan Chunwei | 2025-09-20 | [TRTLLM-8188][chore] refactor GenerationExecutorWorker with WorkerBase for better code reusing (#7840) |
| 2770 | e10a027a03 | brb-nv | 2025-09-20 | [TRTLLM-7731][feat] KV cache transmission in disagg with CP on gen side (#7624) |
| 2771 | e943a39cbd | Enwei Zhu | 2025-09-20 | [None][doc] Update tech blog12 (#7884) |
| 2772 | 2e317a7db6 | Chang Liu | 2025-09-19 | [https://nvbugs/5520490][fix] Fix intermittent test failures by avoiding external web data pulls (#7879) |
| 2773 | 8adaf0bb78 | Grzegorz Kwasniewski | 2025-09-19 | [TRTLLM-6342][feat] Support for partial sharding from factory (#7393) |
| 2774 | 8fcd11515d | Kanghwan | 2025-09-19 | [#7704][chore] Enable MathJax to fix formulas in documentation (#7744) |
| 2775 | 8030b540ac | Mike Iovine | 2025-09-19 | [https://nvbugs/5522462][fix] Fix FP8 scout illegal memory access (#7845) |
| 2776 | fbe325ce57 | pcastonguay | 2025-09-19 | [https://nvbugs/5471108][chore] Unwaiving disagg acc test (#7686) |
| 2777 | 1be7faef37 | Matthias Jouanneaux | 2025-09-19 | [TRTLLM-5966][feat] Helix: add custom position ids to MLA kernels (#6904) |
| 2778 | 7d28acdbf0 | Yuxian Qiu | 2025-09-19 | [https://nvbugs/5522332][fix] Pin numpy version for Gemma. (cherry-pick https://github.com/NVIDIA/TensorRT-LLM/pull/7783) (#7797) |
| 2779 | c8cc16d38d | Enwei Zhu | 2025-09-19 | [None][doc] Tech blog: Combining Guided Decoding and Speculative Decoding: Making CPU and GPU Cooperate Seamlessly (#7864) |
| 2780 | 18095a7cb8 | Liao Lanyu | 2025-09-19 | [https://nvbugs/5503440][fix] Fix potential hang due to wrong type of ZMQ socket and protocol for worker_init_status_queue (#7646) |
| 2781 | efb763402f | xinhe-nv | 2025-09-19 | [None][chore] Add failed cases into waives.txt (#7841) |
| 2782 | 0e72e8f7e6 | Gabriel Wu | 2025-09-19 | [None][feat] Support EPLB in Qwen3 MoE (#7443) |
| 2783 | 0ac51487f4 | Ivy Zhang | 2025-09-19 | [None][chore] remove cli cases for rtx6k (#7833) |
| 2784 | 6b33bcced2 | Ivy Zhang | 2025-09-19 | [None][test] Add accuracy benchmark in stress test (#7561) |
| 2785 | 451475e0dc | dominicshanshan | 2025-09-19 | [None][ci] Waive llama3 auto dtype test bug in https://nvbugs/5527956. (#7853) |
| 2786 | ea079fa530 | Emma Qiao | 2025-09-19 | [None][infra] Waive failed tests in post-merge (#7859) |
| 2787 | 6fcc0540f0 | Kyungmin Lee | 2025-09-19 | [None][fix] fix load_model_on_cpu on qwen/convert_checkpoint.py (#2382) |
| 2788 | f1b362faac | QI JUN | 2025-09-19 | [None][chore] polish error message in cute_dsl_utils.py (#7852) |
| 2789 | c5453103d6 | ruodil | 2025-09-19 | [None][test] add deepseek r1/v3 model with chunked prefill cases (#7124) |
| 2790 | a6370fd143 | HuiGao-NV | 2025-09-19 | [https://nvbugs/5481434][feat] cherry-pick fix to reuse pytorch memory segments occupied by cudagraph (#7747) |
| 2791 | fc4e6d3702 | fredricz-20070104 | 2025-09-19 | [TRTLLM-7183][test] Feature fix model issue for disagg serving (#7785) |
| 2792 | c98b9468af | Chuang Zhu | 2025-09-19 | [None][fix] get Local IP by connect remote (#7719) |
| 2793 | 423e5f6a3c | xiweny | 2025-09-19 | [TRTLLM-6286] [feat] Update CUTLASS to 4.2 and enable SM103 group gemm (#7832) |
| 2794 | d6ebcf7c4a | Yuxian Qiu | 2025-09-19 | [TRTLLM-6994][feat] FP8 Context MLA integration (Cherry-pick https://github.com/NVIDIA/TensorRT-LLM/pull/6059 from release/1.1.0rc2) (#7610) |
| 2795 | 420f0fbcf5 | Ziyi Xiong | 2025-09-19 | [https://nvbugs/5522851][fix] Correct the logic to update kv_lens_cuda (#7790) |
| 2796 | 7646da2d85 | QI JUN | 2025-09-19 | [None][ci] set TORCHINDUCTOR_COMPILE_THREADS correctly (#7800) |
| 2797 | 80dd8fe197 | sunnyqgg | 2025-09-19 | [TRTLLM-6746][feat] Enable two-model spec dec for MTP Eagle (#7001) |
| 2798 | 026f22eb50 | dongfengy | 2025-09-18 | [None][doc] Cherry-pick deployment guide update from 1.1.0rc2 branch to main branch (#7774) |
| 2799 | d921fc3352 | Li Min | 2025-09-18 | [TRTLLM-6898][feat] Add swapab, tileN64, cga sync support for cute dsl nvfp4 gemm (#7764) |
| 2800 | c65457db8a | bhsueh_NV | 2025-09-18 | [None][fix] Revert "Revert "[None][feat] support attention dp for qwen3 dense model"" (#7780) |
| 2801 | 7f87b278bc | QI JUN | 2025-09-18 | [None][chore] remove generated fmha_cubin.h from source tree (#7836) |
| 2802 | d3a907131a | xinhe-nv | 2025-09-18 | [https://nvbugs/5519462][fix] Add failed cases into waives.txt (#7817) |
| 2803 | fe104dc20d | Wanli Jiang | 2025-09-18 | [TRTLLM-7918][feat] Support kvcache reuse and chunk prefill for phi4mm (#7723) |
| 2804 | d909f80379 | xinhe-nv | 2025-09-18 | [TRTLLM-7250][fix] Add failed cases into waives.txt (#7807) |
| 2805 | a55251bf75 | Stefan Niebler | 2025-09-18 | [None][fix] Add TP information in weight scale loading in WeightOnlyQuantLinearMethod (#7732) |
| 2806 | a7ca0fff54 | Wanli Jiang | 2025-09-18 | [TRTLLM-6577][feat] Support nano_v2_vlm in pytorch backend (#7207) |
| 2807 | 2ae08bd1b8 | dongfengy | 2025-09-18 | [https://nvbugs/5519530][fix] Fix gptoss 2-gpu test (#7819) |
| 2808 | 236f71ea05 | xinhe-nv | 2025-09-18 | [None][chore] Add failed cases into waives.txt (#7801) |
| 2809 | 870cfcf9a0 | Leslie Fang | 2025-09-18 | [None][chore] Remove executor config in create_py_executor (#7599) |
| 2810 | b6e916b762 | yuanjingx87 | 2025-09-17 | [None][infra] update ci allow list 2025/09/17 (#7816) |
| 2811 | 1c7f601265 | mpikulski | 2025-09-18 | [https://nvbugs/5508890][fix] gen. result cleanup when using PostprocWorker (#7771) |
| 2812 | 14e455da3e | Li Min | 2025-09-18 | [None][fix] Fix CI issue for dsl pkg install (#7784) |
| 2813 | 4f0e6b5f96 | Barry Kang | 2025-09-18 | [None][feat] Cherry-pick DeepGEMM related commits from release/1.1.0rc2 (#7716) |
| 2814 | 28469dbf27 | Ziyi Xiong | 2025-09-18 | [https://nvbugs/5523080][fix] Correct the batch index in device tensors (#7803) |
| 2815 | 26d50eb539 | Ivy Zhang | 2025-09-18 | [TRTLLM-8070][test] add generation logits case for llama3 (#7759) |
| 2816 | e0423bfaab | Guoming Zhang | 2025-09-18 | [https://nvbugs/5519544][fix] fix invalid expression for disabling pa… (#7806) |
| 2817 | f8e811d134 | Yanchao Lu | 2025-09-18 | [None][chore] Version bump for 1.1.0rc6 (#7824) |
| 2818 | cd80e0a7f1 | Yukun He | 2025-09-18 | [None][fix] Make tile_tokens_dim calculation just in time before kernel launching. (#7529) |
| 2819 | 327e5e5eed | Yan Chunwei | 2025-09-18 | [None][ci] restore unwaive list (#7802) |
| 2820 | 39eb120b96 | Lucas Liebenwein | 2025-09-17 | [#7308] [feat] AutoDeploy: graph-less transformers mode for HF (#7635) |
| 2821 | a5cfc8368f | Netanel Haber | 2025-09-18 | [https://nvbugs/5508536][fix] Revert #7041: Move stop_criteria to sample_async (#7041) (#7796) |
| 2822 | 7c03eb9ea2 | yunruis | 2025-09-18 | [https://nvbugs/5516661][fix] Drop waive case 5516661 (#7791) |
| 2823 | 022d77807d | Matthias Jouanneaux | 2025-09-17 | [TRTLLM-5966][feat] Helix: make softmax stats pointer available to attention gen (#6865) |
| 2824 | 2b1472fb0a | Anu | 2025-09-17 | [None][doc] Update Documentation link to point to docs instead of docs source code (#6495) |
| 2825 | c4abca323e | Emma Qiao | 2025-09-17 | [None][infra] Waive failed tests on main (#7812) |
| 2826 | 2614d71994 | William Zhang | 2025-09-17 | [TRTLLM-7410][feat] Enable KV cache reuse and chunked prefill for mistral3.1 (#7628) |
| 2827 | d3467f9f12 | QI JUN | 2025-09-17 | [None][doc] fix section header of llm_kv_cache_offloading example (#7795) |
| 2828 | f918302b3a | xinhe-nv | 2025-09-17 | [TRTLLM-7250][fix] waive block tests (#7782) |
| 2829 | e6073b3911 | ruodil | 2025-09-17 | [None][test] add gpt oss model for trtllm perf test (#7328) |
| 2830 | 7801d0992b | xinhe-nv | 2025-09-17 | [None][chore] Remove closed bugs (#7697) |
| 2831 | d3e680b3c3 | QI JUN | 2025-09-17 | [None][ci] waive test_llama_eagle3[True-FLASHINFER-False-False-False-False-True] (#7788) |
| 2832 | 523a17d990 | Fanrong Li | 2025-09-17 | [https://nvbugs/5485325][fix] Cherry-pick #7373: fix the CUDA graph warmup issue when using speculative decoding (#7734) |
| 2833 | 39248320d4 | QI JUN | 2025-09-17 | [None][feat] add an example of KV cache host offloading (#7767) |
| 2834 | 6983e8a00d | Zhenhuan Chen | 2025-09-17 | [https://nvbugs/5517260][fix] move scaffolding contrib module's import to subdirectory (#7758) |
| 2835 | bd7aad4988 | QI JUN | 2025-09-17 | [None][ci] waive test_llm_gemma_1gpu_summary_vswa (#7781) |
| 2836 | 4c3dc89f84 | Lucas Liebenwein | 2025-09-16 | [None][chore] AutoDeploy: clean up of model unit test configuration (#7742) |
| 2837 | 62042a9733 | Kaiyu Xie | 2025-09-17 | [TRTLLM-6741] [feat] enable LM tp for MTP, under attention dp case (cherry-pick #7128) (#7571) |
| 2838 | 6313c9799c | Yukun He | 2025-09-17 | [https://nvbugs/5488582][fix] Cherry-pick 7495: Avoid unexpected Triton recompilation in DG fused_moe (#7708) |
| 2839 | 8bdbb48264 | Shiyu Li | 2025-09-16 | [https://nvbugs/5489015][fix] Support communicator split in MNNVL allreduce and fix the binding issues. (#7387) |
| 2840 | a91453de34 | Iman Tabrizian | 2025-09-16 | [None][waive] Waive tests (#7775) |
| 2841 | a49cfb3e68 | HuiGao-NV | 2025-09-17 | [https://nvbugs/5516666][fix] cherrypick fix to the CUDA graph warmup issue when using speculative decoding (#7737) |
| 2842 | eeb89a167c | yuanjingx87 | 2025-09-16 | [None][infra] Add nightly pipeline to generate lock files (#5798) |
| 2843 | 88d9d77912 | yuanjingx87 | 2025-09-16 | [None][infra] Update CI allowlist 2025-09-16 (#7773) |
| 2844 | 98f533453a | Chang Liu | 2025-09-16 | [TRTLLM-7398][doc] Add doc for KV cache salting support (#7772) |
| 2845 | 0f30d7dd6f | yuanjingx87 | 2025-09-16 | [None][infra] add nspect allow list for false positive secrets (#5797) |
| 2846 | 471723bce1 | Aurelien Chartier | 2025-09-16 | [None][chore] Remove unused get_quant_scales methods (#7687) |
| 2847 | 9befd1a72f | Lucas Liebenwein | 2025-09-16 | [None][chore] AutoDeploy: neat disablement of transforms in pipeline (#7736) |
| 2848 | 6ce0624208 | Iman Tabrizian | 2025-09-16 | [TRTLLM-8044][refactor] Rename data -> cache for cacheTransceiver (#7659) |
| 2849 | 8226ef23dc | bhsueh_NV | 2025-09-16 | Revert "[None][feat] support attention dp for qwen3 dense model" (#7765) |
| 2850 | e7c1569456 | xinhe-nv | 2025-09-16 | [None][chore] Add failed cases into waives.txt (#7746) |
| 2851 | 905bb26bbd | Ziyi Xiong | 2025-09-16 | [https://nvbugs/5471106][fix] Remove the waivers (#7711) |
| 2852 | c6ab2072b5 | xinhe-nv | 2025-09-16 | [None][fix] waive hang tests on main (#7720) |
| 2853 | 1fbea497ff | xinhe-nv | 2025-09-16 | [TRTLLM-7070][feat] add gpt-oss serve benchmark tests (#7638) |
| 2854 | 750d15bfaa | amitz-nv | 2025-09-16 | [https://nvbugs/5503529][fix] Change test_llmapi_example_multilora to get adapters path from cmd line to avoid downloading from HF (#7740) |
| 2855 | 6eef19297f | Kaiyu Xie | 2025-09-16 | [None] [chore] cherry pick changes on slurm scripts from `release/1.1.0rc2` (#7750) |
| 2856 | b278d06481 | Li Min | 2025-09-16 | [TRTLLM-6898][feat] Add Cute DSL nvfp4 linear op (#7632) |
| 2857 | 085271eceb | Guoming Zhang | 2025-09-16 | [None][doc] Clean the doc folder and move the outdated docs into lega… (#7729) |
| 2858 | 3f4e160cba | Bo Li | 2025-09-16 | [None][chore] Fix error when running trtllm-bench without cuda graph. (#7725) |
| 2859 | 103b554734 | Void | 2025-09-16 | [None][fix] Ensure that the W4A8 custom input scale remains aligned across all ranks (#7614) |
| 2860 | cf55927064 | xinhe-nv | 2025-09-16 | [None][chore] Add failed cases into waives.txt (#7735) |
| 2861 | e5cead1eb9 | Yanchao Lu | 2025-09-16 | [TRTLLM-6295][test] Exit as early as possible and propagate exit status correctly for multi-node testing (#7739) |
| 2862 | c076a02b38 | xiweny | 2025-09-16 | [TRTLLM-4629] [feat] Add support of CUDA13 and sm103 devices (#7568) |
| 2863 | 809c4d20c0 | Shi Xiaowei | 2025-09-16 | [None][doc] Fix the link in the doc (#7713) |
| 2864 | 96f11b10ae | Necofish | 2025-09-16 | [None][feat] support attention dp for qwen3 dense model (#7618) |
| 2865 | 44d5ccfdd9 | QI JUN | 2025-09-16 | [None][ci] move qwen3 tests from GB200 to B200 (#7733) |
| 2866 | 536e8776cd | Ziyi Xiong | 2025-09-16 | [TRTLLM-6668][feat] Enable overlap scheduler for two-model spec decoding (#7651) |
| 2867 | 857c0b45be | Lucas Liebenwein | 2025-09-15 | [None][infra] AutoDeploy: codeowners for autodeploy unit tests (#7743) |
| 2868 | 8097be7e9c | Izzy Putterman | 2025-09-15 | [None][feat] Eagle, use last hidden post norm (#7546) |
| 2869 | 0c9430e5a5 | Yanchao Lu | 2025-09-15 | [None][ci] Test waives for the main branch 09/15 (#7709) |
| 2870 | 7deefb3d2b | jmydurant | 2025-09-15 | [TRTLLM-7192][feat] optimize MLA chunked prefill && support fp8 mla chunked prefill (#7477) |
| 2871 | 24fc1f9acf | Zheng Duan | 2025-09-15 | [None][fix] using arrival time in llmapi when creating LlmRequest in pytorch workflow (#7553) |
| 2872 | e080294725 | Wanli Jiang | 2025-09-15 | [TRTLLM-7918][feat] Revert "Support kvcache reuse for phi4mm (#7563)" (#7722) |
| 2873 | 965a3dab90 | ixlmar | 2025-09-15 | [None][test] add test for min_tokens (#7678) |
| 2874 | fc9f4c9295 | Wanli Jiang | 2025-09-15 | [TRTLLM-7918][feat] Support kvcache reuse for phi4mm (#7563) |
| 2875 | 335c007df8 | HuiGao-NV | 2025-09-15 | [None][chore] move some cases from post-merge to pre-merge to detect errors in early stage (#7699) |
| 2876 | d5df0af017 | DylanChen-NV | 2025-09-15 | [https://nvbugs/5467981][fix] Fix Qwen2.5-VL fails with cuda graph padding (#7122) |
| 2877 | ddfe0320b3 | Ivy Zhang | 2025-09-15 | [TRTLLM-7279][test] add accuracy test for deepseek-r1 with chunked_prefill (#7365) |
| 2878 | a2c45d82c3 | JunyiXu-nv | 2025-09-15 | [None][chore] Enable multiple postprocess workers tests for chat completions api (#7602) |
| 2879 | b69e3e9f99 | xinhe-nv | 2025-09-15 | [None][chore] Add failed cases into waives.txt (#7682) |
| 2880 | 47e37755a3 | Chang Liu | 2025-09-14 | [TRTLLM-6903][feat] Support chunked prefill for multimodal models (#6843) |
| 2881 | 1b29c2e731 | Perkz Zheng | 2025-09-15 | [None][feat] support gpt-oss with fp8 kv cache (#7612) |
| 2882 | 70aa4e28c1 | Yanchao Lu | 2025-09-14 | [None][ci] Test waives for the main branch 09/14 (#7698) |
| 2883 | 89fc136972 | Yanchao Lu | 2025-09-14 | [None][ci] Some improvements for Slurm CI (#7689) |
| 2884 | 1f43854496 | Zhanrui Sun | 2025-09-13 | [TRTLLM-6791][infra] Add check for uploading stage name and avoid overriding test result tar file (#6742) |
| 2885 | 7d73a89ad0 | Zhanrui Sun | 2025-09-12 | [TRTLLM-7169][infra] Fix Slurm multi-node test showing "Submit Test Results" in the test name (#6856) |
| 2886 | c2bc39af63 | Pengyun Lin | 2025-09-12 | [TRTLLM-1302][feat] Topk logprobs for TRT backend and top1 logprob for PyT backend (#6097) |
| 2887 | ef676fc71f | Guoming Zhang | 2025-09-11 | [https://nvbugs/5513192][fix] Add the missing param for kv_cache_tran… (#7679) |
| 2888 | 3a9847eb84 | Chang Liu | 2025-09-10 | [https://nvbugs/5498165][fix] fix permission error for config file lock (#7656) |
| 2889 | e3117731b3 | Fan - Yunfan | 2025-09-11 | [None][fix] Fix the incorrect header file import in dataType.h (#7133) |
| 2890 | 656f229b58 | QI JUN | 2025-09-10 | [None][ci] move some test cases from l40s to a30 (#7684) |
| 2891 | aa152ce8cf | Kanghwan | 2025-09-10 | [None][infra] Adjust labeling llm prompt for bug issues (#7385) |
| 2892 | 9986070044 | Emma Qiao | 2025-09-11 | [None][infra] Waive failed cases on main 0910 (#7676) |
| 2893 | fc9d426589 | Dom Brown | 2025-09-10 | [https://nvbugs/5505402] [fix] Disable deep_gemm for Qwen3 QKNormRoPEAttention and Linear layers due to accuracy issues (#7616) |
| 2894 | 0652514c6d | v-shobhit | 2025-09-10 | [None][feat] Use a shell context to install dependancies (#7383) |
| 2895 | 222e01662c | nvamyt | 2025-09-10 | [https://nvbugs/5488212][waive] Waive failed tests for L20 (#7664) |
| 2896 | d219a4f225 | Leslie Fang | 2025-09-10 | [None][chore] remove executor config in kv cache creator (#7526) |
| 2897 | a4312ba743 | Linda | 2025-09-10 | [https://nvbugs/5477359][fix] Nanobind: Allow none types for fields in result (#7672) |
| 2898 | 207c5258c4 | xinhe-nv | 2025-09-10 | [https://nvbugs/5494698][fix] skip gemma3 27b on blackwell (#7505) |
| 2899 | bf57829acf | Bo Deng | 2025-09-10 | [TRTLLM-7871][infra] Extend test_perf.py to add disagg-serving perf tests. (#7503) |
| 2900 | 76c5e1a12f | Yiqing Yan | 2025-09-10 | [None][infra] Bump version to 1.1.0rc5 (#7668) |
| 2901 | 758c22f832 | Kanghwan | 2025-09-09 | [#7208][fix] Fix config type of MedusaConfig (#7320) |
| 2902 | bbb5ae3349 | Frida Hou | 2025-09-09 | [#5861][autodeploy] Refactor: Quantization Transforms with Inheritance (#7227) |
| 2903 | c353ff342e | Zheyu Fu | 2025-09-09 | [None][feat] Make the should_use_spec_decode logic a bit smarter (#7112) |
| 2904 | f412f5c4b0 | Chuang Zhu | 2025-09-10 | [None][fix]UCX zmq ip support ipv6 (#7530) |
| 2905 | ef620f3579 | fredricz-20070104 | 2025-09-10 | [https://nvbugs/5410687][test] Add deepseek r1-w4afp8 quickstart (#7645) |
| 2906 | beefd6413e | Guoming Zhang | 2025-09-10 | [None][fix] fix post-merge issue raised by #5488 (#7655) |
| 2907 | faa2f46554 | Chang Liu | 2025-09-09 | [TRTLLM-5059][feat] Enable KV-cache reuse and add E2E tests for llava-next (#7349) |
| 2908 | d49374bc45 | Jin Li | 2025-09-10 | [TRTLLM-7408][feat] Wrap MOE with custom op. (#7277) |
| 2909 | a0e1604898 | QI JUN | 2025-09-09 | [None][ci] add DGX_H100-2_GPUs-PyTorch-Others-1 pipeline (#7629) |
| 2910 | 0566df672d | Linda | 2025-09-09 | [TRTLLM-6707][fix] nanobind fix for executor exit call (#7565) |
| 2911 | dcd110cfac | Richard Huo | 2025-09-09 | [None][chore] add TorchLlmArgs to the connector api (#7493) |
| 2912 | cc7593987b | NVJiangShao | 2025-09-09 | [https://nvbugs/5434424][fix] A quick fix for the wrong output issue of SM89 blocked scaling batched GEMM when the input tensor is non-contiguous. (#7615) |
| 2913 | a6ed0d17d6 | William Tambellini | 2025-09-09 | [#6798][fix] fix compilation error in ub_allocator in single device build (#6874) |
| 2914 | af403848d7 | Liao Lanyu | 2025-09-09 | [https://nvbugs/5445466][fix] unwaive DS R1 test cases with bug already fixed (#7429) |
| 2915 | da6cb541a2 | Perkz Zheng | 2025-09-09 | [None][feat] Optimize MLA kernels with separate reduction kernels (#7597) |
| 2916 | 6e712dd1cc | tomeras91 | 2025-09-09 | [None][fix] enable NvFP4/FP8 quantization for Nemotron-H architecture (#7589) |
| 2917 | 9cb5410067 | Linda | 2025-09-09 | [https://nvbugs/5454559][fix] handle bias term in fuse_gate_mlp (#7449) |
| 2918 | 8a52015f50 | xinhe-nv | 2025-09-09 | [None][chore] Remove closed bugs (#7591) |
| 2919 | 62b564ac3c | Guoming Zhang | 2025-09-09 | [None][fix] add the missing import raised by #7607 (#7639) |
| 2920 | c53d1814a7 | William Zhang | 2025-09-08 | [None][feat] Extend VLM factory and add Mistral3 factory (#7583) |
| 2921 | 6ba1c8421c | William Tambellini | 2025-09-08 | [#6529][feat] CMake option to link statically with cublas/curand (#7178) |
| 2922 | 7a62df5f0b | Zhanrui Sun | 2025-09-09 | [TRTLLM-4366][infra] Don't call reinstall_rockylinux_cuda when the base CUDA image is up to dated (#5980) |
| 2923 | ecc0e687c6 | Tomer Shmilovich | 2025-09-09 | [None][feat] Nixl support for GDS (#5488) |
| 2924 | 7f3f658d5f | Guoming Zhang | 2025-09-05 | [None][doc] Rename TensorRT-LLM to TensorRT LLM. (#7554) |
| 2925 | 35dac55716 | Guoming Zhang | 2025-09-05 | [None][doc] Update kvcache part (#7549) |
| 2926 | f53fb4c803 | Guoming Zhang | 2025-09-04 | [TRTLLM-5930][doc] 1.0 Documentation. (#6696) |
| 2927 | 5c616da2fd | Yiqing Yan | 2025-09-09 | [TRTLLM-5877][infra] Add fmha tests and auto trigger rules (#6050) |
| 2928 | 1e0669d27a | Wanli Jiang | 2025-09-09 | [https://nvbugs/5453709][fix] Remove transformers version limit in Qwen2VL (#7152) |
| 2929 | d96c54d8ae | Iman Tabrizian | 2025-09-08 | [None][test] Skip eagle3 test (#7627) |
| 2930 | fdd5bd49fc | dongfengy | 2025-09-08 | [https://nvbugs/5481080][fix] Fix GPTOSS W4A16 reference (#7323) |
| 2931 | 96af324ff1 | zhanghaotong | 2025-09-09 | [None][fix] Add try-catch in stream generator (#7467) |
| 2932 | 1d243a8503 | yuanjingx87 | 2025-09-08 | [None][infra] Try to fix docker container failed to be killed issue (#7388) |
| 2933 | 77657a1c12 | Chuang Zhu | 2025-09-09 | [TRTLLM-7361][feat] KV cache transfer for uneven pp (#7117) |
| 2934 | 3e0073e86b | Leslie Fang | 2025-09-09 | [None][chore] remove executor config in instantiate sampler (#7516) |
| 2935 | 5f2a42b3df | Eran Geva | 2025-09-08 | [TRTLLM-6142][feat] AutoDeploy: set torch recompile_limit based on cuda_graph_batch_sizes and refactored (#7219) |
| 2936 | 4a1e13897f | Chang Liu | 2025-09-08 | [None][feat] Update multimodal utility `get_num_tokens_per_image` for better generalization (#7544) |
| 2937 | dd9627d9f9 | Emma Qiao | 2025-09-08 | [None][infra] Add back rtx-pro-6000 stages since the node is available (#7601) |
| 2938 | ed27a72bcf | Yanchao Lu | 2025-09-08 | [None][ci] Fix a typo in the Slurm command |
| 2939 | 219e95569a | bhsueh_NV | 2025-09-08 | [https://nvbugs/5506683][fix] adjust the CI (#7604) |
| 2940 | c9dca69e1b | dominicshanshan | 2025-09-08 | [None][chore] Mass integration of release/1.0 - 3rd (#7519) |
| 2941 | 504bb7ffa9 | JunyiXu-nv | 2025-09-08 | [TRTLLM-7779][feat] Support multiple postprocess workers for chat completions API (#7508) |
| 2942 | 14ee43e254 | binghanc | 2025-09-08 | [None][docs] refine docs for accuracy evaluation of gpt-oss models (#7252) |
| 2943 | 205c3a144c | Yan Chunwei | 2025-09-08 | [None][chore] expose tokens_per_block into KvCacheConfig (#5911) |
| 2944 | 7c76dde76d | BatshevaBlack | 2025-09-08 | [TRTLLM-7187][fix] Build wheel with NIXL (#7472) |
| 2945 | 8f3121ac81 | Raayan Dhar | 2025-09-07 | [None][fix] chore: fixing the math on asymmetric tp+pp tests (#7098) |
| 2946 | 045d2cf761 | Yanchao Lu | 2025-09-08 | [None][ci] Block some nodes to avoid unstable network access (#7593) |
| 2947 | 0fee8cd028 | Netanel Haber | 2025-09-07 | [TRTLLM-7153] [feat] Move stop_criteria to sample_async (#7041) |
| 2948 | 5c4711fb2b | Emma Qiao | 2025-09-07 | [None][infra] Skip RTX Pro 6000 test stages due to HW are offline (#7592) |
| 2949 | bae9560e62 | Raayan Dhar | 2025-09-07 | [https://nvbugs/5448767][fix] sync termination of requests across PP ranks (#7455) |
| 2950 | aea8ac1649 | Emma Qiao | 2025-09-07 | [TRTLLM-5950][infra] Removing remaining turtle keywords from the code base (#7086) |
| 2951 | 45390402fc | Mike Iovine | 2025-09-06 | [https://nvbugs/5502352][fix] Fix 2-model CDL path (#7543) |
| 2952 | 99b98f1374 | Chang Liu | 2025-09-06 | [TRTLLM-7440][fix] Split `fused_input_embed` to separate out host sync (#7280) |
| 2953 | 0fdc6c7278 | xiweny | 2025-09-07 | [TRTLLM-4629] [feat] trtllm-gen kernels support sm103 (#7570) |
| 2954 | 23500b55c3 | Chang Liu | 2025-09-06 | [TRTLLM-7398][feat] Support KV cache salting for secure KV cache reuse (#7106) |
| 2955 | 12ecb864c2 | QI JUN | 2025-09-06 | [None][chore] share input_ids buffers among different cuda graphs (#7236) |
| 2956 | 12c66f7610 | Anthony Chang | 2025-09-07 | [None][fix] DeepSeek-R1 W4A8 weight loading issue; fixes regression from #6200 (#7123) |
| 2957 | 9a97f0a3b7 | dominicshanshan | 2025-09-06 | [None][ci] Waive qwen3 test for accuracy bug in https://nvbugs/5505402 (#7585) |
| 2958 | caf9b9cd42 | Yanchao Lu | 2025-09-06 | [None][ci] Improve SSH connection stability (#7567) |
| 2959 | 525bb806a9 | QI JUN | 2025-09-05 | [None][ci] move some test cases of DGX H100 to post merge (#7569) |
| 2960 | b8183cac2b | QI JUN | 2025-09-05 | [None][ci] Revert "[https://nvbugs/5461761][fix] Remove the waiver (#7476)" (#7584) |
| 2961 | 74105a45d9 | Lucas Liebenwein | 2025-09-05 | [#6120][feat] AutoDeploy: flexible args for sequence interface + AD multi-modal input processor + llama4 VLM example (#7221) |
| 2962 | 25389c9fe2 | peaceh-nv | 2025-09-06 | [https://nvbugs/5453806][unwaive] Unwaive fp8 kvcache attention test (#7243) |
| 2963 | d8ec546b73 | Emma Qiao | 2025-09-05 | [None][infra] Waive failed tests on main branch 0905 (#7564) |
| 2964 | 9eb3911470 | Leslie Fang | 2025-09-05 | [None][chore] Remove executor_config in create_py_executor_instance (#7463) |
| 2965 | a95d9616ba | Robin Kobus | 2025-09-05 | [#6186][feat] Introduce QKNormRoPEAttention module (#6830) |
| 2966 | 79e0296ca0 | Ziyi Xiong | 2025-09-05 | [https://nvbugs/5461761][fix] Remove the waiver (#7476) |
| 2967 | 163b1fc84f | Yiteng Niu | 2025-09-05 | [None][infra] update nspect version (#7552) |
| 2968 | 4195010e13 | Yanchao Lu | 2025-09-05 | [None][ci] Increase the number of retries in docker image generation (#7557) |
| 2969 | 8e3962d278 | xinhe-nv | 2025-09-05 | [TRTLLM-6642][feat] add gptoss 20g tests (#7361) |
| 2970 | b3ba3d98d2 | xinhe-nv | 2025-09-05 | [None][chore] Remove closed bugs (#7408) |
| 2971 | ff3704897b | QI JUN | 2025-09-04 | [None][ci] remove unnecessary test_modeling_deepseek.py (#7542) |
| 2972 | 2189a2f3ff | Jin Li | 2025-09-05 | [https://nvbugs/5483615][fix] Remove unnecessary assertion to let mai… (#7441) |
| 2973 | 58d1036bb1 | Naveenraj Kamalakannan | 2025-09-04 | [#3325][feat] Add MCTS and TOT tree-based inference controllers to Scaffolding (#7490) |
| 2974 | bddf183e15 | Shunkangz | 2025-09-05 | [None][feat] Add Request specific exception (#6931) |
| 2975 | 89889fb526 | Rashid Kaleem | 2025-09-04 | [https://nvbugs/5369366] [fix] Report failing requests (#7060) |
| 2976 | 08a0e06621 | Chang Liu | 2025-09-04 | [TRTLLM-7410][feat] Support hashing and KV cache reuse for videos (#7360) |
| 2977 | 48a5270868 | Yuxian Qiu | 2025-09-05 | [https://nvbugs/5492485][fix] Use offline dataset from llm-models instead. (#7435) |
| 2978 | 98a1bffb7c | sychen52 | 2025-09-04 | [OMNIML-2336][feat] Add NVFP4 x FP8 (#6809) |
| 2979 | 1745102e72 | Enwei Zhu | 2025-09-04 | [TRTLLM-7027][feat] Fuse d2t to logitsBitmaskKernel and fix a race condition in one-model spec (#7481) |
| 2980 | 26b133f3a7 | Izzy Putterman | 2025-09-04 | [None][feat] MultiLayer Eagle (#7234) |
| 2981 | b46e0ae5d4 | Ivy Zhang | 2025-09-04 | [None][test] update nim and full test list (#7468) |
| 2982 | d38b8e3dd9 | QI JUN | 2025-09-04 | [None][ci] set TORCHINDUCTOR_COMPILE_THREADS for thop/parallel tests (#7489) |
| 2983 | 4e3dded64d | Wanli Jiang | 2025-09-04 | [TRTLLM-6308][feat] Support Aggregate mode for phi4-mm (#7521) |
| 2984 | 5bcda7520b | WeiHaocheng | 2025-09-04 | [https://nvbugs/5477730][fix] Fix the alltoall case when tp_size larger than ep_size (#7331) |
| 2985 | 0de3f83805 | Zhanrui Sun | 2025-09-04 | [TRTLLM-6893][infra] Disable the x86 / SBSA build stage when run BuildDockerImage (#6729) |
| 2986 | cce9556858 | kris1025 | 2025-09-04 | [https://nvbugs/5485886][fix] Fix resource free of Eagle3ResourceManager (#7437) |
| 2987 | ced5512ae4 | Yiqing Yan | 2025-09-04 | [None][chore] Bump version to 1.1.0rc4 (#7525) |
| 2988 | 7090b286b2 | jianweiwu | 2025-09-04 | [None][fix] fix hunyuan_moe init bug (#7502) |
| 2989 | 3755f8ab7d | Grzegorz Kwasniewski | 2025-09-04 | [TRTLLM-6342][fix] Fixed triggering BMM sharding (#7389) |
| 2990 | c622f61609 | Yanchao Lu | 2025-09-04 | [None][fix] Fix a typo in the Slurm CI codes (#7485) |
| 2991 | 931816fee1 | Emma Qiao | 2025-09-04 | [TRTLLM-6199][infra] Update for using open driver from BSL (#7430) |
| 2992 | a117e7a57e | William Zhang | 2025-09-03 | [TRTLLM-7442][model] Remove unnecessary D2H copies (#7273) |
| 2993 | 2a2dfe273b | Jin Li | 2025-09-04 | [https://nvbugs/5485102][fix] Correctly set stride for piecewise outp… (#7442) |
| 2994 | db8eb0a447 | Stanley Sun | 2025-09-04 | [TRTLLM-7876][test] Test trtllm-serve with --extra_llm_api_options (#7492) |
| 2995 | d97c1e6bd9 | Lizhi Zhou | 2025-09-04 | [https://nvbugs/5470769][fix] fix disagg-serving accuracy test case (#7338) |
| 2996 | c1aa7f31d9 | Yao Yao | 2025-09-04 | [None][fix] Fix a numerical stability issue for XQA with spec dec (#7114) |
| 2997 | 51a2b8729e | Frida Hou | 2025-09-03 | [#7222][autodeploy] Separate run_shape_prop as another graph utility (#7313) |
| 2998 | bd9ba97d89 | Leslie Fang | 2025-09-04 | [None][chore] Remove two unused parameters in create_py_executor (#7458) |
| 2999 | 5ff3a65b23 | Enwei Zhu | 2025-09-04 | [TRTLLM-7028][feat] Enable guided decoding with speculative decoding (part 2: one-model engine) (#6948) |
| 3000 | 64e3bfa054 | Mike Iovine | 2025-09-03 | [None][fix] Fix KV cache recompute in draft_target spec decode (#7348) |
