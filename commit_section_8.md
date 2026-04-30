# Commit Section 8

Commits 3501 to 4000 (total: 5612)

| # | Commit Hash | Author | Date | Message |
|---|------------|--------|------|--------|
| 3501 | a20ab5cbdb | yunruis | 2025-08-01 | [https://nvbugs/5381276][fix] fix warning for fused_a_gemm (#6402) |
| 3502 | 7447d6ed85 | brb-nv | 2025-08-01 | [TRTLLM-6657][feat] Add LoRA support for Gemma3 (#6371) |
| 3503 | 1daa8c3232 | liji-nv | 2025-08-01 | [https://nvbugs/5340941][https://nvbugs/5375785] - fix: Wrap attentio… (#6355) |
| 3504 | f39d621c3b | Yanchao Lu | 2025-08-01 | [None][infra] Pin the version for triton to 3.3.1 (#6508) (#6519) (#6549) |
| 3505 | fca0d37798 | xinhe-nv | 2025-08-01 | [None][fix] update nemotron nas tests free_gpu_memory_fraction=0.8 (#6552) |
| 3506 | 137413fbf4 | juney-nvidia | 2025-08-01 | [None][doc] Exposing the latest tech blogs in README.md (#6553) |
| 3507 | ba5bdbb138 | chenfeiz0326 | 2025-08-01 | [None][chore] Disable add special tokens for Llama3.3 70B (#6482) |
| 3508 | 147ad69368 | Kaiyu Xie | 2025-08-01 | [None][doc] blog: Scaling Expert Parallelism in TensorRT-LLM (Part 2: Performance Status and Optimization) (#6547) |
| 3509 | 90856bf97d | Yukun He | 2025-08-01 | [https://nvbugs/5419069][fix] Fix the mismatched layer name components. (#6417) |
| 3510 | ac23f4a80d | Yang Li | 2025-08-01 | [TRTLLM-4279] fix: Add a protection test for checking trtllm custom ops (#6515) |
| 3511 | 71524a1a48 | Ivy Zhang | 2025-08-01 | [https://nvbugs/5419066][fix] Use trt flow LLM (#6467) |
| 3512 | 48768fd720 | Zero Zeng | 2025-08-01 | fix: Fix missing key (#6471) |
| 3513 | aee35e2dbd | Kaiyu Xie | 2025-08-01 | chore: Make example SLURM scripts more parameterized (#6511) |
| 3514 | d3c14682f0 | Robin Kobus | 2025-08-01 | refactor: Remove unused buffers and bindings from sampler (#6484) |
| 3515 | ad5742b105 | Venky | 2025-07-31 | [fix] Update get_trtllm_bench_build_command to handle batch size and tokens (#6313) |
| 3516 | 4472f11bb7 | Yiteng Niu | 2025-08-01 | [TRTLLM-6364][infra] Validate for PR titles to ensure they follow the required format (#6278) |
| 3517 | 942e080415 | Yao Yao | 2025-08-01 | [fix] Fix missing fields in xqa kernel cache key (#6282) |
| 3518 | fbee279909 | Jaedeok Kim | 2025-08-01 | fix: remove duplicate layer multiplication in KV cache size calculation (#6481) |
| 3519 | 7bb0a78631 | Zongfei Jing | 2025-08-01 | Deepseek R1 FP8 Support on Blackwell (#6486) |
| 3520 | 8c165fd27a | Venky | 2025-07-31 | [TRTLLM-6611][feat] Add warnings and stricter validation to LoraManager adapter loading (#6453) |
| 3521 | 00059de380 | Yukun He | 2025-08-01 | chore: Improve the AutoTuner log information. (#6368) |
| 3522 | 2eca0d5925 | brb-nv | 2025-07-31 | fix: Fix poor generation with FP8 Gemma3 1B checkpoint (#6499) |
| 3523 | 8cf3faa26a | Simeng Liu | 2025-07-31 | [feat] Auto-enable ngram with concurrency <= 32. (#6232) |
| 3524 | 8062e0fe7c | Ziyi Xiong | 2025-08-01 | [TRTLLM-6392][feat] Support turning on/off spec decoding dynamically (#6363) |
| 3525 | b8719fe96d | Michal Guzek | 2025-07-31 | [nvbug/5374773] chore: Update nanobind with fail_fast_on_attention_window_too_large changes (#6491) |
| 3526 | 6d5da9f7c2 | tomeras91 | 2025-07-31 | [https://nvbugs/5404046][fix] Fix Nemotron-H flaky CUDA graph / overlap scheduler test (#6485) |
| 3527 | 0c42f54a39 | shaharmor98 | 2025-07-31 | Bugfix/fix nemotron nas lora support (#6380) |
| 3528 | baece56758 | Emma Qiao | 2025-07-31 | [None][infra] Pin the version for triton to 3.3.1 (#6508) |
| 3529 | d38c26bb78 | Yiqing Yan | 2025-07-31 | [Infra][TRTLLM-5633] - Fix merge waive list (#6504) |
| 3530 | 1ee7a08d2b | amitz-nv | 2025-07-31 | [5830][feat] Improve LoRA cache memory control (#6220) |
| 3531 | 83e97659aa | Venky | 2025-07-30 | [infra] Remove auto_assign_reviewers option from .coderabbit.yaml (#6490) |
| 3532 | fcd5706615 | Wanli Jiang | 2025-07-31 | doc: add bielik model to support-matrix (#6480) |
| 3533 | 8e84df74b5 | Faraz | 2025-07-30 | Fix e2e test failure for RTX6000 Pro  (#6420) |
| 3534 | ca534e4798 | xinhe-nv | 2025-07-31 | test: add accuracy reference (#6479) |
| 3535 | 17e0d0fb1a | dongjiyingdjy | 2025-07-31 | fix: fix illeagel memory access (#6437) |
| 3536 | 4b299cb77e | Enwei Zhu | 2025-07-31 | feat: Support structural tag in C++ runtime and upgrade xgrammar to 0.1.21 (#6408) |
| 3537 | ae3a5fc918 | bhsueh_NV | 2025-07-31 | [doc][ci][Qwen3][nvbugs 5374145] Add Qwen3 235B eagle3 CI (#6477) |
| 3538 | 83621e4b80 | Yechan Kim | 2025-07-31 | doc: update multimodal models on support-matrix.md (#6431) |
| 3539 | 25cd4f215e | Vadim Gimpelson | 2025-07-31 | [PERF] Move calculation Qwen2-VL's rotary_cos_sin to LLM worker process (#6004) |
| 3540 | 0e16d1f070 | brb-nv | 2025-07-30 | test: Add time logging for lora tests (#6466) |
| 3541 | f9cf683e39 | shaharmor98 | 2025-07-30 | add propagation of trust_remote_code to OpenAIServer (#6446) |
| 3542 | fac186e3b5 | Anurag Mukkara | 2025-07-30 | [nvbug/5409417] Unwaive llava test case (#6460) |
| 3543 | f6287e4498 | brb-nv | 2025-07-30 | Unwaive Gemma2 LoRA test on H100 (#6461) |
| 3544 | 24e7f4eece | Bo Deng | 2025-07-31 | [nvbug/5410296][fix] Fix OOM in Llama 4 disagg-serve tests (#6439) |
| 3545 | 9632dba02e | Wanli Jiang | 2025-07-31 | feat: TRTLLM-6450 update long rope for phi3.5/phi4-mini/phi4-mm  (#6353) |
| 3546 | e67f4da9b5 | NVShreyas | 2025-07-30 | [Perf]: Add residual, norm for nemotron_nas models (#6455) |
| 3547 | 0f083b9daf | pcastonguay | 2025-07-30 | fix: Unwaive triton cpp test [nvbug 5401088] (#6412) |
| 3548 | 03e38c9087 | nv-guomingz | 2025-07-30 | chore: update trtllm-serve usage doc by removing backend parameter when it use torch as backend. (#6419) |
| 3549 | b4065d8ca6 | Chang Liu | 2025-07-30 | [TRTLLM-6654][feat] Add support for external multimodal embeddings (#6263) |
| 3550 | e7ae5e2824 | pcastonguay | 2025-07-30 | feat: Add support for disaggregation with pp with pytorch backend (#6369) |
| 3551 | a2514d93fc | tomeras91 | 2025-07-30 | [nvbug 5380101][fix] Fix nemotronNAS loading for TP>1 (#6447) |
| 3552 | d980928c96 | Leslie Fang | 2025-07-30 | [doc] update the doc of feature combination matrix (#6441) |
| 3553 | 0cf2f6f154 | Yiqing Yan | 2025-07-30 | [TRTLLM-5633] - Merge current waive list with the TOT waive list (#5198) |
| 3554 | 22b29df38c | Yechan Kim | 2025-07-30 | [nvbugs/5414909] fix: Qwen2-VL keyword on L20 (#6427) |
| 3555 | d9ab3fd35e | xinhe-nv | 2025-07-30 | tests: add TestNemotronH cuda graph tests (#6390) |
| 3556 | a5540acfce | nv-guomingz | 2025-07-30 | chore: add trtllm-serve json schema example into doc. (#6418) |
| 3557 | 2fe9cc0889 | QI JUN | 2025-07-30 | chore: remove draft_model_engine from init parameter list of PyExecutor (#6325) |
| 3558 | 1f39a11af0 | QI JUN | 2025-07-30 | chore: clean code of PyExecutor (#6445) |
| 3559 | d6eed1b624 | 2ez4bz | 2025-07-29 | [fix] Switch placement of image placeholder for mistral 3.1 (#6435) |
| 3560 | a427f5bece | Jinyang Yuan | 2025-07-30 | [fix] Fix wide EP when using DeepEP with online EPLB (#6429) |
| 3561 | c9ed1ab436 | Zheng Duan | 2025-07-30 | [TRTLLM-6549] chore: record delay introduced by disaggregated serving in kv cache measure (#6135) |
| 3562 | c00d6763b2 | xinhe-nv | 2025-07-29 | test: [CI] Add failed cases into waives.txt (#6457) |
| 3563 | 5b420ad267 | peaceh-nv | 2025-07-30 | Rename layer to comply with deepseek (#6393) |
| 3564 | ab40369053 | Venky | 2025-07-29 | [fix] Move kv_cache_free_gpu_mem_fraction arg to benchmark command in tests (#6463) |
| 3565 | d6eb8e2366 | Yechan Kim | 2025-07-30 | fix: support mixture of text & multimodal prompts (#6345) |
| 3566 | 1a8e28d295 | Yunfan Fan | 2025-07-30 | [FIX] fix bugs caused by None attention_bias during Qwen3 model convert engine (#6344) |
| 3567 | ad662ddcdd | Yan Chunwei | 2025-07-30 | chore: disallow arbitrary in llm_args.Configs (#6367) |
| 3568 | 1a6930986a | Yan Chunwei | 2025-07-30 | chore: remove unused kv_cache_dtype in api reference (#6444) |
| 3569 | 7efe3cb0cd | Michal Guzek | 2025-07-29 | [fix] Add detokenization-based stop word logic to LLM API (#5948) |
| 3570 | c3729dbd7d | Zhanrui Sun | 2025-07-30 | infra: [TRTLLM-5873] Use build stage wheels to speed up docker release image build (#4939) |
| 3571 | 7231134996 | nv-guomingz | 2025-07-29 | doc: remove backend parameter for trtllm-bench when backend is set to… (#6428) |
| 3572 | f1086e7d4f | xinhe-nv | 2025-07-29 | test: [CI] remove closed bugs (#6381) |
| 3573 | 4fbb344caf | xinhe-nv | 2025-07-29 | test: [CI] Add failed cases into waives.txt (#6423) |
| 3574 | 0eee2e2850 | Yukun He | 2025-07-29 | [5385981] fix: Update the usage of VisionAttention init API. (#6413) |
| 3575 | 13e24ab1cb | QI JUN | 2025-07-29 | chore: remove unused code in PyExecutor (#6351) |
| 3576 | e11255e9d0 | ruodil | 2025-07-29 | test:[nvbug 5415268] add kv_cache_free_gpu_mem_fraction param and llama4 rcca cases (#6430) |
| 3577 | d2a04abb95 | Frank | 2025-07-28 | [fix] Fixes to parameter usage and low latency configuration. (#6343) |
| 3578 | e58afa510e | Kaiyu Xie | 2025-07-29 | doc: Add README for wide EP  (#6356) |
| 3579 | 64ba483656 | Zhanrui Sun | 2025-07-29 | infra: [TRTLLM-6499] Split L0_Test into two pipeline by single GPU and multi GPU(For SBSA) (#6132) |
| 3580 | ee3cbb073e | Frank | 2025-07-28 | [fix] Add trust_remote_code option to prepare_dataset. (#6338) |
| 3581 | 2d21bca25e | Venky | 2025-07-28 | [infra] Remove auto_apply_labels option from .coderabbit.yaml reviews section (#6416) |
| 3582 | 2573bb729d | Michal Guzek | 2025-07-28 | feat: Add Phi-4-Mini-Instruct in Pytorch backend for LLM API accuracy tests (#6303) |
| 3583 | 738ab61593 | Aurelien Chartier | 2025-07-28 | [nvbugs/5404000] fix: waive request_perf_metrics_draft test on pre-Hopper GPUs (#6339) |
| 3584 | bca14157a9 | Po-Wei (Vincent) | 2025-07-28 | [infra] Add an auto-labeling github action to TRTLLM (#6373) |
| 3585 | 608ed89f96 | yuanjingx87 | 2025-07-28 | [None][infra]Update slurm config keys (#6370) |
| 3586 | cdca541148 | 2ez4bz | 2025-07-28 | [test] Unwaive mistral3.1 small E2E test (#6352) |
| 3587 | 60e4d3a9d4 | 2ez4bz | 2025-07-28 | [test] Add accuracy regression test for Mistral3.1 (#6322) |
| 3588 | 49044733e1 | nv-guomingz | 2025-07-28 | chore: delete useless gitkeep files. (#6400) |
| 3589 | 03632a679f | ruodil | 2025-07-28 | test: organize perf cases and add missing perflab cases in qa test list (#6283) |
| 3590 | 971be1fe86 | xinhe-nv | 2025-07-28 | test: waive failed cases (#6394) |
| 3591 | 4efc6496b7 | QI JUN | 2025-07-28 | chore: add _prepare_and_schedule_batch function in PyExecutor (#6365) |
| 3592 | 413a83ff80 | Yuan Tong | 2025-07-28 | fix: compatibility with CUDA < 12.9 on `__CUDA_ARCH_SPECIFIC__` macro (#5917) |
| 3593 | 45d441e60c | Yan Chunwei | 2025-07-28 | [TRTLLM-5061] chore: add status tags to LLM API reference (#5707) |
| 3594 | 2945817cae | Ivy Zhang | 2025-07-28 | [nvbug/5409414, 5355707] tests: adjust batchsize and decoding name (#6292) |
| 3595 | b3ca159787 | Emma Qiao | 2025-07-28 | [Infa] - waive failed cases and fix a typo (#6384) |
| 3596 | c9b8b6180f | Zero Zeng | 2025-07-28 | Add Acceptance Rate calculation to benchmark_serving (#6240) |
| 3597 | 97f7e12588 | Jinyang Yuan | 2025-07-28 | [fix] Fix perf regression caused by MoE autotuner when using DeepEPLowLatency (#6288) |
| 3598 | dc757799e1 | Chang Liu | 2025-07-27 | [nvbugs/5401156][fix] Avoid import all models when import trtllm._common (#6266) |
| 3599 | f172face98 | Void | 2025-07-28 | DeepEP LL dispatch FP4 (#6296) |
| 3600 | 93a0fd0a23 | Yukun He | 2025-07-28 | [TRTLLM-6445] feat: Enable AllReduce-associated fusion patterns in Llama3/4. (#6205) |
| 3601 | 2dd3186727 | YueWeng | 2025-07-28 | fix: remove cudaStreamSynchronize when using relaxed acceptance (#5262) |
| 3602 | 908f49a4ad | Yan Chunwei | 2025-07-28 | [nvbug/5320234] fix: test_trtllm_bench_llmapi_launch (#6359) |
| 3603 | d853811190 | Ziyi Xiong | 2025-07-27 | [https://nvbugs/5402719][fix]: Add cuda graph dummy requests to the spec_resource_manager (#6258) |
| 3604 | 96d004d800 | Liana Koleva | 2025-07-26 | doc: fix invalid link in llama 4 example documentation (#6340) |
| 3605 | 54f68287fc | Jhao-Ting Chen | 2025-07-25 | fix precompiled multi_query_token kernel not having is_fp8_out hash key (#6279) |
| 3606 | 08d57123f9 | Michal Guzek | 2025-07-25 | [nvbug/5374773] chore: Add a runtime flag to enable fail fast when attn window is too large to fit at least one sequence in KV cache (#5974) |
| 3607 | c35c78ff58 | Iman Tabrizian | 2025-07-25 | [fix][nvbugs/5390810] Improve the check for disaggregated serving test (#6301) |
| 3608 | 1e5e71aa42 | ameynaik-hub | 2025-07-25 | Mtp optimizations round1 (#5689) |
| 3609 | 7bff341553 | Simeng Liu | 2025-07-25 | [doc] Add NGram tech blog (#6311) |
| 3610 | b8d4cb8beb | nv-guomingz | 2025-07-26 | feat: Support JSON Schema in OpenAI-Compatible API (#6321) |
| 3611 | 3805976e90 | pcastonguay | 2025-07-25 | fix: Fixing kv_cache_events unit tests [nvbug 5362412] (#6265) |
| 3612 | a0aecf0476 | xiaoqi | 2025-07-25 | [feat]: support logit_bias (#5354) |
| 3613 | 470544cf17 | xinhe-nv | 2025-07-25 | test: [CI] Add failed cases into waives.txt (#6333) |
| 3614 | e07fff4f78 | liji-nv | 2025-07-25 | [https://nvbugs/5340941] - fix: Correct custom ops used by Qwen3 Moe … (#6285) |
| 3615 | 6268a60ab3 | xinhe-nv | 2025-07-24 | tests: add test_chunked_prefill for llama4 (#5549) |
| 3616 | d97419805b | Yiqing Yan | 2025-07-25 | [TRTLLM-5312] - Add bot run rules for triton tests (#4988) |
| 3617 | 2dcfa90e99 | xinhe-nv | 2025-07-24 | test: skip llama3.3 70b test on cg4 (#6293) |
| 3618 | 0f2f11f90b | Mike Iovine | 2025-07-24 | [TRTLLM-6453][feat] Support chunked prefill on spec decode 2 model (#6104) |
| 3619 | 9a99e6d6d7 | Linda | 2025-07-25 | fix: integration tests with nanobind (#6326) |
| 3620 | 375f74ecb2 | Shiyu Li | 2025-07-24 | [fix][nvbugs/5399355] Fix Lamport buffer clear issue for MNNVL TwoShot Allreduce and add FP16 support. (#6237) |
| 3621 | f8f5ba65fc | Frank | 2025-07-24 | [fix] Update to remove popping of KV cache and other args. (#6310) |
| 3622 | 0df758ec9f | Stefan Niebler | 2025-07-24 | [TRTLLM-6650][feat] Enhance beam search support with CUDA graph integration (#6217) |
| 3623 | ff72ca90de | Bo Deng | 2025-07-24 | Improve TransferAgentTest.SyncMessage (#6250) |
| 3624 | 706f421cb0 | Perkz Zheng | 2025-07-24 | [Fix] the bug in the trtllm-gen heurisitcf for MLA kernels. (#6284) |
| 3625 | 62298bc473 | Zhenhua Wang | 2025-07-24 | perf: customize cublastLt algo for Llamba 3.3 70B TP4 (#6315) |
| 3626 | 7b6aadc800 | bhsueh_NV | 2025-07-24 | [Fix][nvbug 5401163][nvbug 5404726][Qwen3] Fix bug of MoE on tp > 1 with trtllm moe backend (#6235) |
| 3627 | 0cc1f8c03d | Emma Qiao | 2025-07-24 | [Infra] - Wiave failed tests in post-merge (#6331) |
| 3628 | f290108cd8 | Ivy Zhang | 2025-07-24 | tests: only get timeout value from pytest marker (#6287) |
| 3629 | 0ffcf9a863 | Zhou Yuxin | 2025-07-24 | Update fmhaRunner.cpp to fix guardwords scan error (#6327) |
| 3630 | 14d94a3856 | liji-nv | 2025-07-24 | feat: Add non UB AR + Residual + Norm + Quant fusion (#6320) |
| 3631 | a63a1ac7f9 | Lizhi Zhou | 2025-07-24 | [TRTLLM-6444] Add some UCX trouble shooting docs and print UCX related logs (#6085) |
| 3632 | 428e34080f | QI JUN | 2025-07-24 | chore: remove unused variables in pyexecutor (#6280) |
| 3633 | 31d3eff24b | nv-guomingz | 2025-07-24 | doc: fix invalid links related with llm api example (#6317) |
| 3634 | 5fceaa6153 | Iman Tabrizian | 2025-07-23 | Revert "tests: add timeout_manager to tensorrt flow test cases (#5942)" (#6309) |
| 3635 | 82d03ca979 | Emma Qiao | 2025-07-24 | [Infra] - Increase unittest execution time since some test exceeds 1600 (#6277) |
| 3636 | 7740bfa31d | Iman Tabrizian | 2025-07-23 | Waive tests (#6312) |
| 3637 | 19696a6e4f | Venky | 2025-07-23 | [feat] Update .coderabbit.yaml with review settings and code guidelines (#6251) |
| 3638 | cf4f4e8d73 | Lucas Liebenwein | 2025-07-23 | [AutoDeploy] disable flaky MoE nvfp4 test (#6302) |
| 3639 | cb737a5fcd | Emma Qiao | 2025-07-23 | [Infra] - Skip failed cases (#6299) |
| 3640 | 2486eb778e | Stefan Niebler | 2025-07-23 | [TRTLLM-6651][feat]  Enable Overlap scheduler +  Beam Search in TRTLLM Sampler (#6223) |
| 3641 | 2b0fa24175 | xinhe-nv | 2025-07-23 | test: [CI] Add failed cases into waives.txt (#6289) |
| 3642 | ed62a06eef | YueWeng | 2025-07-23 | [nvbug/5322354] fix PD + MTP + overlap scheduler accuracy issue (#6136) |
| 3643 | fca13b8c95 | Zhou Yuxin | 2025-07-23 | hopper-style context MLA (#5713) |
| 3644 | a8253b942f | QI JUN | 2025-07-23 | chore: remove duplicate should_stop_processing check (#6242) |
| 3645 | 83c3ed128b | Yechan Kim | 2025-07-23 | chore: set default device to cpu on Multimodal models (#5994) |
| 3646 | 5636c67388 | Erin | 2025-07-22 | fix: nvbug_5398806 (#6239) |
| 3647 | 2193ad3aac | Perkz Zheng | 2025-07-23 | [https://nvbugs/5387771] fix deadlocks due to insufficient numSemaphores (#6262) |
| 3648 | 9538c8d0e5 | Venky | 2025-07-22 | Add basic Nemo Ckpt Lora Loading in pytorch flow  (#6019) |
| 3649 | f08286c679 | Kaiyu Xie | 2025-07-23 | doc: Refactor documents and examples of disaggregated serving and wide ep (#6054) |
| 3650 | 8ecdeee300 | wili | 2025-07-23 | [refactor] Simplification of Speculative decoding configs - Part 2 (#5936) |
| 3651 | bc2fb29c5e | Iman Tabrizian | 2025-07-22 | [nvbugs/5401261][fix] Fix Triton backend disaggregated serving support (#6224) |
| 3652 | 41fb8aa8b1 | Lucas Liebenwein | 2025-07-22 | [AutoDeploy] merge feat/ad-2025-07-07 (#6196) |
| 3653 | 5234502717 | Raayan Dhar | 2025-07-22 | [nvbug/5361223] doc: Update Llama4 deployment guide: update config & note concurrency (#6222) |
| 3654 | ef4878db05 | yuanjingx87 | 2025-07-22 | set NVIDIA_IMEX_CHANNELS for dlcluster slurm job only (#6234) |
| 3655 | ab7434ac62 | 2ez4bz | 2025-07-22 | [feat] Enable TP and batching for PixtralVisionModel / Mistral3VLM (#6152) |
| 3656 | b7c8a672da | John Calderon | 2025-07-22 | [Issue 6193] Fix gemma3vl weight loader (#6233) |
| 3657 | ff9963978a | danielafrimi | 2025-07-22 | Add register_fake for finegrained_mixed_dtype_gemm torch_op (#6255) |
| 3658 | 60073731ca | Linda | 2025-07-22 | fix: bindings unit tests for nanobind (#6221) |
| 3659 | 04f2d4b2eb | Stanley Sun | 2025-07-22 | test: update test list for RTX6KD (#6213) |
| 3660 | 3e1a0fbac4 | Lizhi Zhou | 2025-07-22 | [TRTLLM-6537][infra] extend multi-gpu tests related file list (#6139) |
| 3661 | 3e18ee5fe1 | Yiqing Yan | 2025-07-22 | chore: bump version to 1.0.0rc5 (#6252) |
| 3662 | b85ab139f9 | Yechan Kim | 2025-07-22 | doc: add supported data modality and types on multimodal serve (#5988) |
| 3663 | 48ddc3d4b9 | Pengyun Lin | 2025-07-18 | [fix]: Revert commit 388b491 (#6143) |
| 3664 | 24ce6b9517 | bhsueh_NV | 2025-07-18 | [Doc][Qwen3] update qwen3 into support-matrix (#6161) |
| 3665 | 310bdd9830 | pcastonguay | 2025-07-16 | fix: Fix triton backend build [nvbug 5396469] (#6098) |
| 3666 | a03c680581 | QI JUN | 2025-07-16 | add release notes for 0.21 release (#6049) |
| 3667 | 34dd071bd6 | nv-guomingz | 2025-07-15 | [TRTLLM-6495] doc: add disclaimer for 3rd party software installation. (#6039) |
| 3668 | eb7d0f84b5 | Yi Zhang | 2025-07-14 | [nvbugs/5368410][fix] Disable moe allreduce for multi node (#5918) |
| 3669 | c66941036f | Fanrong Li | 2025-07-14 | fix: fix index out of bounds error in spec decoding (#5954) |
| 3670 | 9d26b7891a | Nikita Korobov | 2025-07-10 | fix: [5328141] increase tolerance for test_fp8_block_scale_gemm (#5849) |
| 3671 | f194b65f3e | Yan Chunwei | 2025-07-10 | fix [nvbug/5351244]: address remote mpi session submit (#5664) |
| 3672 | f4f2176cd5 | amirkl94 | 2025-07-10 | chore: Port leftover 0.20 (#5907) |
| 3673 | 537757e669 | Bo Li | 2025-07-10 | fix: [nvbugs/5351130] Adjust DSV3-Lite tests free_gpu_memory_fraction to 0.75 to prevent OOM on CI. (#5896) |
| 3674 | db77d83a2a | Bo Li | 2025-07-22 | bug: [https://nvbugs/5368507] Fix test_generate_with_seed. (#6206) |
| 3675 | 37d0b68442 | 2ez4bz | 2025-07-21 | [fix] Fix flaky mistral E2E test (#6230) |
| 3676 | fddb7f1141 | WeiHaocheng | 2025-07-22 | feat: moe prepare support topk % 4 != 0 (#5742) |
| 3677 | eb5cb5b642 | Ivy Zhang | 2025-07-22 | tests: add timeout_manager to tensorrt flow test cases (#5942) |
| 3678 | ee45e0c63f | Shunkangz | 2025-07-22 | feat: Refactor the fetching request logic (#5786) |
| 3679 | 7381f1dba7 | Chang Liu | 2025-07-21 | [TRTLLM-5059][feat] Add KV cache reuse support for multimodal models (#5444) |
| 3680 | 4a0951f85c | Simeng Liu | 2025-07-21 | [Chore] Replace MODEL_CACHE_DIR with LLM_MODELS_ROOT and unwaive triton_server/test_triton.py::test_gpt_ib[gpt-ib] (#5859) |
| 3681 | 9645814bdf | Mike Iovine | 2025-07-21 | [chore] Clean up quickstart_advanced.py (#6021) |
| 3682 | d7f0b0ab68 | Ziyi Xiong | 2025-07-21 | [fix] Correct the returned value of has_spec_drafter (#6178) |
| 3683 | f9b0a911fb | Yi Zhang | 2025-07-21 | test: Enable GB200 torch compile multi gpu tests (#6145) |
| 3684 | 9832bef07d | Pengyun Lin | 2025-07-21 | [BREAKING CHANGE]: change default backend to PyTorch in trtllm-serve (#5717) |
| 3685 | e41507a253 | Emma Qiao | 2025-07-21 | [Infra] - Waive failed cases on recent post-merge (#6212) |
| 3686 | 3e0fb60e50 | liji-nv | 2025-07-21 | [TRTLLM-4279] feat: Multistream initial support for torch compile flow (#5847) |
| 3687 | aea91b2541 | QI JUN | 2025-07-21 | doc: add Deprecation Policy section (#5784) |
| 3688 | 3cbc23f783 | Zhanrui Sun | 2025-07-21 | infra: [TRTLLM-5250] Add sanity check stage for ngc-release images (Build wheels for devel image) (#4656) |
| 3689 | 3efad2e58c | Linda | 2025-07-21 | feat: nanobind bindings (#6185) |
| 3690 | b46fd41026 | xinhe-nv | 2025-07-21 | test: [CI] remove closed bugs (#6201) |
| 3691 | e8c068b4b1 | Yuening Li | 2025-07-21 | [TRTLLM-5863][feat] Support Weight-Only-Quantization in PyTorch Workflow (#5850) |
| 3692 | 88076eecd0 | Jinyang Yuan | 2025-07-21 | [fix] Fix can_use_alltoall in fused_moe_wide_ep.py (#6173) |
| 3693 | b4c7e8c9a5 | nv-guomingz | 2025-07-21 | doc: remove cuda_graph_config: {} from doc since cuda_graph enabled b… (#6150) |
| 3694 | ca9bc5727e | brb-nv | 2025-07-20 | fix: Flush stale `PlanParams` with custom attention mask (#6163) |
| 3695 | 6a3c9f8061 | ruodil | 2025-07-21 | test: add phi-4 multimodel and bielik-11b-v2.2 models for perf test (#5826) |
| 3696 | a433ebad2b | brb-nv | 2025-07-20 | enh: Lift expectation of single image per sample in Gemma3 VLM (#6195) |
| 3697 | 5300a99bd8 | danielafrimi | 2025-07-20 |  W4A8 GEMM  (#6005) |
| 3698 | 98428f330e | amitz-nv | 2025-07-20 | [TRTLLM-5826][feat] Support pytorch LoRA adapter eviction (#5616) |
| 3699 | 943fd418dd | Martin Marciniszyn Mehringer | 2025-07-20 | fix: Ensure mlx5 library is installed for deep_ep and remove deprecated python bindings (#6189) |
| 3700 | 2e14c8f443 | bhsueh_NV | 2025-07-20 | [Fix][Chore][Qwen3] fix bug of using fp4 on sm120 (#6065) |
| 3701 | 118307c224 | Void | 2025-07-20 | DeepEP LL support variable hidden size and tokens num (#6141) |
| 3702 | 69e9f6d489 | Pengyun Lin | 2025-07-19 | [fix]: Skip prompt length checking for generation only requests (#6146) |
| 3703 | 66030ef815 | Ziyi Xiong | 2025-07-19 | [TRTLLM-6452][feat]: Two-model engine KV cache reuse support (#6133) |
| 3704 | 82d3587bb8 | wili | 2025-07-19 | [refactor] Unify name of NGram speculative decoding (#5937) |
| 3705 | 152e2df43b | Rashid Kaleem | 2025-07-18 | [Disaggregated] Add retry knobs and handling (#5808) |
| 3706 | fc8b29c4ff | John Calderon | 2025-07-18 | [Issue 5927][fix] Avoid memory calls during broadcast for single GPU (#6010) |
| 3707 | 0388ff9083 | Bo Deng | 2025-07-19 | [https://nvbugs/5393961][fix] record kv-cache size in MLACacheFormatter (#6181) |
| 3708 | d9a3530048 | Netanel Haber | 2025-07-18 | [nvbug/5393888][nvbug/5393042] Always use `py_seq_slot` (#6147) |
| 3709 | d475c97c82 | Stefan Niebler | 2025-07-18 | [nvbugs/5354884][fix] Update beam search workspace estimation to new upper bound (#5926) |
| 3710 | 6d7874a467 | Stefan Niebler | 2025-07-18 | [nvbugs/5369799] fix: Update disaggregation handling in sampler (#5762) |
| 3711 | 28858c8711 | xiaoqi | 2025-07-19 | feat(eagle3):support qwen3 dense model (#5879) |
| 3712 | 22d4a8c48a | Venky | 2025-07-18 | enh: Add script to map tests <-> jenkins stages & vice-versa (#5177) |
| 3713 | 2c6fa145ee | Bo Deng | 2025-07-19 | [TRTLLM-6471] Infra: unwaive nixl tests and some disagg-serve tests (#6095) |
| 3714 | 07e8813984 | Bo Li | 2025-07-18 | feat: Remove padding in attention DP. (#6064) |
| 3715 | fd6ce7f20e | Stefan Niebler | 2025-07-18 | [ci] Speedup beam search unit tests with fixtures for LLM (#5843) |
| 3716 | 8454640ee1 | Zhanrui Sun | 2025-07-18 | infra: fix single-GPU stage failed will not raise error (#6165) |
| 3717 | 9522cde464 | Erin | 2025-07-18 | fix: NVBug 5385576 py_batch_idx issue (#6153) |
| 3718 | 44040edbf0 | Leslie Fang | 2025-07-18 | update broken link of PyTorchModelEngine in arch_overview (#6171) |
| 3719 | ec2b953e7e | Robin Kobus | 2025-07-18 | refactor: Enhanced handling of decoder requests and logits within the batch manager (#6055) |
| 3720 | 77acb4f753 | Emma Qiao | 2025-07-18 | [Infra] - Waive failed tests in post-merge (#6176) |
| 3721 | a95f31e72a | QI JUN | 2025-07-18 | chore: add more log in FmhaDispatcher (#6170) |
| 3722 | 519a2116b5 | Yiteng Niu | 2025-07-18 | [None][infra] Update the allow list of CI trigger (#6168) |
| 3723 | f32169269a | Yiqing Yan | 2025-07-18 | [TRTLLM-5179] - Update bot help messages (#5277) |
| 3724 | c0e416535e | Chuang Zhu | 2025-07-18 | fix single_disagg_test (#6166) |
| 3725 | 812243bdd6 | Aurelien Chartier | 2025-07-17 | feat: add support for Modelopt fp8_pb_wo quantization scheme (#6106) |
| 3726 | 992b273045 | Zhenhuan Chen | 2025-07-18 | [https://nvbugs/5387375] fix(scaffolding): fix scaffolding aime test in test_e2e (#6140) |
| 3727 | 200ea9ee81 | xavier-nvidia | 2025-07-17 | fix TMA error with GEMM+AR on TP=2 (#6075) |
| 3728 | 0155e7a3a1 | yifeizhang-c | 2025-07-18 | [TRTLLM-6368] Update deepep dispatch API (#6037) |
| 3729 | b75e53ab69 | Iman Tabrizian | 2025-07-17 | Revert "feat: nanobind bindings (#5961)" (#6160) |
| 3730 | ae28b3a664 | Daniel Stokes | 2025-07-18 | feat: Add support for benchmarking individual gemms in MOE benchmark (#6080) |
| 3731 | 2c90203c36 | qixiang-99 | 2025-07-17 | Refactor KVCacheManager: Simplify token availability calculation and … (#6134) |
| 3732 | 161490f039 | Frank | 2025-07-17 | [fix] Fixes KV Cache overrides in trtllm-bench (#6103) |
| 3733 | 8480c120b1 | 2ez4bz | 2025-07-17 | [fix] Fix Mistral3VLM weight-loading & enable in pre-merge (#6105) |
| 3734 | 10dbf4f0f4 | Iman Tabrizian | 2025-07-17 | [fix] Remove duplicated KVCache transmission check (#6022) |
| 3735 | d71c6fe526 | ixlmar | 2025-07-17 | [fix] Update jenkins container images (#6094) |
| 3736 | 5bff317abf | Linda | 2025-07-17 | feat: nanobind bindings (#5961) |
| 3737 | 58d22a72f1 | Ziyi Xiong | 2025-07-17 | [TRTLLM-6352][feat] Migrate EAGLE3 and draft/target speculation to Drafter (#6007) |
| 3738 | 9518e14f69 | Stanley Sun | 2025-07-17 | test: fix PytestUnknownMarkWarning: Unknown pytest.mark.timeout (#6115) |
| 3739 | a718486900 | Yi Zhang | 2025-07-17 | fix: Fix DeepSeek R1 CI (#6129) |
| 3740 | 9b45499caa | nv-guomingz | 2025-07-17 | test: update max_beam_width to 1 due to torchsampler changes. (#6101) |
| 3741 | de60ae47e3 | Erin | 2025-07-17 | chores: unwaive a few tests for v1.0 (#6107) |
| 3742 | 21efb50068 | Enwei Zhu | 2025-07-17 | [TRTLLM-6406] feat: Enable guided decoding with overlap scheduler (#6000) |
| 3743 | 44c70c88f9 | Chuang Zhu | 2025-07-17 | chore:[BREAKING CHANGE] use cacheTransceiverConfig as knobs for disagg service (#5234) |
| 3744 | 1cc49494fe | Emma Qiao | 2025-07-17 | [Infra] - Add wiave list for pytest when using slurm (#6130) |
| 3745 | 8c1c9ef7aa | Zhenhuan Chen | 2025-07-17 | fix: convert venv_prefix to str before comparison with base_prefix (#6121) |
| 3746 | e821c68611 | QI JUN | 2025-07-17 | CI: update multi gpu test trigger file list (#6131) |
| 3747 | 48daa18de3 | Yanchao Lu | 2025-07-17 | [None][infra] Set up the initial config for CodeRabbit (#6128) |
| 3748 | d4d21a106e | Iman Tabrizian | 2025-07-16 | [fix] Release slots with spec decode + disagg (#5975) (#6032) |
| 3749 | 7e033c392e | ChristinaZ | 2025-07-17 | Feat: Add vectorized loading for finalize kernel in MoE Trtllm backend (#5919) |
| 3750 | 4c364b9a73 | Zhanrui Sun | 2025-07-17 | infra: fix SBSA test stage (#6113) |
| 3751 | 6e1aee6fd6 | Shiyu Li | 2025-07-16 | [fix] Performance Optimization for MNNVL TwoShot Kernel (#5934) |
| 3752 | fe070a0168 | chenfeiz0326 | 2025-07-17 | test: Update Llama4 Scout FP4 & FP8 accuracy tests (#5901) |
| 3753 | 28385f6571 | Frank | 2025-07-16 | [TRTLLM-6070] docs: Add initial documentation for trtllm-bench CLI. (#5734) |
| 3754 | 2d2b8bae32 | Wanli Jiang | 2025-07-17 | feat: TRTLLM-5574 Add phi-4-multimodal pytorch-backend support (#5644) |
| 3755 | e09e409dfb | qixiang-99 | 2025-07-16 | Fix: Enhance ModelConfig for kv cache size calculations (#5868) |
| 3756 | fa34cb7234 | Mike Iovine | 2025-07-16 | [refactor] Clean up drafter/resource manager creation logic (#5805) |
| 3757 | e0836f9ca9 | shaharmor98 | 2025-07-16 | [TRTLLM-5493] Add core infrastructure to enable loading of custom checkpoint formats (#5372) |
| 3758 | 9354114f68 | Wanli Jiang | 2025-07-17 | fix: Update trtllm args issues with extra nested config (#5996) |
| 3759 | 301b78bb77 | Iman Tabrizian | 2025-07-16 | Add documentation for eagle3+disagg+dynamo (#6072) |
| 3760 | e30d7bec38 | Emma Qiao | 2025-07-16 | [Infra] - Waive failed cases in post-merge on main  (#6096) |
| 3761 | e42f5a9581 | Zhanrui Sun | 2025-07-16 | infra: [TRTLLM-5879] Spilt single GPU test and multi GPU test into 2 pipelines (#5199) |
| 3762 | fc2347eaf5 | Bo Li | 2025-07-16 | chore: Cleanup disable_fp4_allgather. (#6006) |
| 3763 | 8ef8e73002 | qsang-nv | 2025-07-16 | update spec_dec (#6079) |
| 3764 | 0552a02943 | Tomer Shmilovich | 2025-07-16 | BlockManager copy constructor fix (#5982) |
| 3765 | a02606a9e2 | Yan Chunwei | 2025-07-16 | [TRTLLM-5530][BREAKING CHANGE] refactor: unify KvCacheConfig in LLM class for pytorch backend (#5752) |
| 3766 | 10349b54df | Martin Marciniszyn Mehringer | 2025-07-16 | fix: Add $HOME/.local/bin to PATH when running docker in local user mode (#6062) |
| 3767 | dda91b5117 | Ivy Zhang | 2025-07-16 | tests: add QA test cases  (#5959) |
| 3768 | 7568deb2f1 | Yan Chunwei | 2025-07-16 | [nvbug/5387226] chore: add propogation for trust_remote_code to AutoConfig (#6001) |
| 3769 | 763012a88a | Ivy Zhang | 2025-07-16 | [nvbug/5359218][tests] add test llm api test case on lookahead with chunked prefill (#6051) |
| 3770 | f5f31beee1 | peaceh-nv | 2025-07-16 | feat: Add deepseek-lite tests for RTX pro 6000 (#5903) |
| 3771 | ec3ebae43e | Bo Deng | 2025-07-16 | [TRTLLM-6471] Infra: Upgrade NIXL to 0.3.1 (#5991) |
| 3772 | 38db4bc7fb | Zheng Duan | 2025-07-16 | feat: use session abstraction in data transceiver and cache formatter (#5611) |
| 3773 | 385af53a4d | Zheng Duan | 2025-07-16 | [nvbug/5347489][nvbug/5388036] increase timeout in disagg worker test (#6041) |
| 3774 | 509dc7c831 | nv-guomingz | 2025-07-16 | chroe: upgrade modelopt to 0.33 (#6058) |
| 3775 | e51c541617 | Yiqing Yan | 2025-07-16 | chore: Bump version to 1.0.0rc4 (#6086) |
| 3776 | 8679a058a3 | Wanli Jiang | 2025-07-16 | fix: Unable to load phi4-model with tp_size>1 (#5962) |
| 3777 | 665b4469b3 | Iman Tabrizian | 2025-07-15 | [fix] Fix Triton build (#6076) |
| 3778 | 6a47cac981 | Aurelien Chartier | 2025-07-15 | feat: Add support for Triton request cancellation (#5898) |
| 3779 | edab7532dd | danielafrimi | 2025-07-15 | feat/add latency support for trtllm bench (#3730) |
| 3780 | 9214ac662a | brb-nv | 2025-07-15 | test: Add regression tests for Gemma3 VLM (#6033) |
| 3781 | 7a1af1c738 | Fanrong Li | 2025-07-16 | Cherry-pick https://github.com/NVIDIA/TensorRT-LLM/pull/5947 (#5989) |
| 3782 | 0523f77b36 | Xiaodong (Vincent) Huang | 2025-07-15 | support TRTLLM_DEEP_EP_TOKEN_LIMIT to allow run deep-ep on memory-con… (#5684) |
| 3783 | e761231c0b | Jinyang Yuan | 2025-07-15 | [fix] Move NCCL group in all-gather and reduce-scatter OPs outside the outer loop (#6053) |
| 3784 | 4a26bd6500 | Tailing Yuan | 2025-07-15 | Fix: pad DeepEP fp4 recv tensors if empty (#6048) |
| 3785 | 9ebc3ab9c4 | MinaHuai | 2025-07-15 | [nvbugs/5385972][nvbugs/5387423][Fix] Minor fix for llava_next/llava_onevision (#5998) |
| 3786 | ab1c54709d | Jaedeok Kim | 2025-07-15 | fix: adjust window sizes of VSWA at torch backend (#5880) |
| 3787 | 9e871ca582 | Yiteng Niu | 2025-07-15 | [infra] add more log on reuse-uploading (#6036) |
| 3788 | 2a147c4d01 | ruodil | 2025-07-15 | test: add llama_v3.3_70b_cases in perf test (#6035) |
| 3789 | 2504aa552e | ruodil | 2025-07-15 | test: add recursive updating pytorch config and change MOE backend format in perf test (#6046) |
| 3790 | 4e4d18826f | nv-guomingz | 2025-07-15 | chore: [Breaking Change] Rename cuda_graph_config padding_enabled fie… (#6003) |
| 3791 | d811843a08 | Zhanrui Sun | 2025-07-15 | infra: [TRTLLM-6313] Fix the package sanity stage 'Host Node Name' in… (#5945) |
| 3792 | e499f6c44a | Lucas Liebenwein | 2025-07-15 | [Fix] check for ImportError or ModuleNotFoundError for deep_ep_utils (#6026) |
| 3793 | 6b35afaf1b | Yiqing Yan | 2025-07-15 | [Infra][TRTLLM-6013] - Fix stage name in single stage test rerun report (#5672) |
| 3794 | 01b2def5ef | Zhanrui Sun | 2025-07-15 | infra: [TRTLLM-6331] Support show all stage name list when stage name check failed (#5946) |
| 3795 | 24dfd4cd0b | jiahanc | 2025-07-14 | Doc: Update llama-3.3-70B guide (#6028) |
| 3796 | dd2491f47d | Daniel Stokes | 2025-07-15 | fix: Fix MOE benchmark to rotate buffers to prevent L2 cache reuse (#4135) |
| 3797 | 2ea4077993 | Rashid Kaleem | 2025-07-14 | [Model load] Fix llama min-latency model load (#5883) |
| 3798 | 2320f12321 | Yechan Kim | 2025-07-15 | doc: update EXAONE 4.0 news (#6034) |
| 3799 | f225f5cd2e | ixlmar | 2025-07-15 | [nvbugs-5318143] fix: restrict PyTorch memory usage to avoid OOMs (#5964) |
| 3800 | f277afdd93 | Daniel Stokes | 2025-07-15 | perf: Enable 128x256 tile shapes for FP4 MOE CUTLASS backend (#5986) |
| 3801 | c4ee535afb | Iman Tabrizian | 2025-07-14 | [fix] fix eagle3 two model disaggregated serving test (#6014) |
| 3802 | 6d4b045d1f | Robin Kobus | 2025-07-14 | refactor: Remove enforced sorted order of batch slots (#3502) |
| 3803 | f5f5be9e94 | brb-nv | 2025-07-14 | enh: Bidirectional mask with multiple images for Gemma3 (#5976) |
| 3804 | 1a2d96919c | brb-nv | 2025-07-14 | feat: Update Gemma3 Vision Encoder (#5973) |
| 3805 | 6c30d78b78 | Alex Zhang | 2025-07-14 | [TRTLLM-5653][infra] Run docs build only if PR contains only doc changes (#5184) |
| 3806 | 63139fdcff | Yechan Kim | 2025-07-14 | feat: EXAONE4.0 support (#5696) |
| 3807 | dbf29184dc | Clay | 2025-07-14 | fix #4974: A thread leak issue in scaffolding unittest (#5020) |
| 3808 | aa97fbb2ad | Kaiyu Xie | 2025-07-14 | [Nvbug/5383670] fix: switch test case to non-fp4 ckpt for more GPU coverage (#5882) |
| 3809 | c720d7f779 | Yiqing Yan | 2025-07-14 | Waive L0 test (#6002) |
| 3810 | 3a0ef73414 | Zhanrui Sun | 2025-07-14 | infra: [TRTLLM-6242] install cuda-toolkit to fix sanity check (#5709) |
| 3811 | 30608a5e6d | Zhenhuan Chen | 2025-07-10 | [https://nvbugs/5355316] fix: update torch.compile option to fix triton store_cubin error (#5865) |
| 3812 | 5a61d64b5b | Robin Kobus | 2025-07-09 | [nvbugs/5345391] fix: chunked prefill + overlap scheduling (#5761) |
| 3813 | 3fcaa8a310 | Pengyun Lin | 2025-07-09 | [nvbug 5327706][fix] fix mgmn postprocess error (#5835) |
| 3814 | 347520494b | ruodil | 2025-07-09 | test: remove duplicate cases in perf sanity test (#5870) |
| 3815 | 966e41a900 | Yi Zhang | 2025-07-08 | doc: Update gb200 doc (#5840) |
| 3816 | 6d79559f3e | Bo Li | 2025-07-08 | fix: [https://nvbugs/5351130][https://nvbugs/5333654] Unwaive for bug 5351130 and 5333654. (#5821) |
| 3817 | 2991cf4b80 | Bo Li | 2025-07-08 | fix: [https://nvbugspro.nvidia.com/bug/5345215] Unwaive for bug 5345215. (#5606) |
| 3818 | 4a0b7a0cf1 | Perkz Zheng | 2025-07-08 | [https://nvbugspro.nvidia.com/bug/5355054] fallback to cubins for fp8 fmha kernels on Ada. (#5779) |
| 3819 | 3e1fd983c3 | Yan Chunwei | 2025-07-07 | [nvbug5266240] chore: unwaive test_llm_with_dummy_weights (#5744) |
| 3820 | 388b4919b8 | Pengyun Lin | 2025-07-07 | [nvbug 5304752][fix] enhance _check_arguments to filter illegal requests for pytorch backend (#5541) |
| 3821 | c321fb8f81 | Martin Marciniszyn Mehringer | 2025-07-07 | Fix docker cache mount (#5763) |
| 3822 | 6992616c1f | Pengyun Lin | 2025-07-07 | [nvbug 5004744][fix] rewrite completion API to avoid repetitive tokens (#5201) |
| 3823 | 278a1a7df3 | ruodil | 2025-07-07 | test: fix some test failure and add llama_nemotron models in perf sanity test, add more torch cases (#5693) |
| 3824 | c8874a7f94 | Iman Tabrizian | 2025-07-04 | [nvbug/5337601][fix] Fix disagg + speculative decoding (#5558) |
| 3825 | 9cc4e5d50e | Yi Zhang | 2025-07-04 | [nvbugs/5336321][fix] Enable attention dp = False test case, Fix TRTLLM Gen Moe workspace allocation (#5463) |
| 3826 | e5e87ecf34 | Yi Zhang | 2025-07-04 | test: Move some of the test from post merge to pre-merge, update dgx b200 test case (#5640) |
| 3827 | 869e88304a | brb-nv | 2025-07-03 | [nvbug/5341178][fix] Fix OOM in Llama 4 accuracy test (#5735) |
| 3828 | afaa388bee | Dom Brown | 2025-07-04 | [TRTLLM-6100] fix: Nvbug 5356427: autotuned TRTLLM Gen fp8 block scale MoE illegal memory access (#5676) |
| 3829 | 4d8920982a | WeiHaocheng | 2025-07-14 | fix: set allreduce strategy to model config (#5955) |
| 3830 | c9e7f831dc | dominicshanshan | 2025-07-14 | Breaking change: perf: [TRTLLM-4662] Enable cuda graph by default (#5480) |
| 3831 | c04570a506 | dongxuy04 | 2025-07-14 | Use huge page mapping for host accessible memory on GB200 (#5963) |
| 3832 | 9c673e9707 | Yan Chunwei | 2025-07-14 | [TRTLLM-6160] chore: add sampling examples for pytorch (#5951) |
| 3833 | ed77ef2ff4 | Enwei Zhu | 2025-07-14 | fix: Fix MoE benchmark (#5966) |
| 3834 | c30eead09f | Yan Chunwei | 2025-07-14 | [TRTLLM-6164][TRTLLM-6165] chore: add runtime example for pytorch (#5956) |
| 3835 | cfcb97af0e | wili | 2025-07-14 | [BUG5388075][fix] Fix error in post-merge-tests (#5949) |
| 3836 | c7ffadf692 | Xianjie Qiao | 2025-07-14 | Fix errors in wide-ep scripts (#5992) |
| 3837 | ce39409530 | QI JUN | 2025-07-14 | fix cancel request logic (#5800) |
| 3838 | a36ac45c4d | Yuan Tong | 2025-07-13 | fix: fast redux detection in trtllm gen routing kernel (#5941) |
| 3839 | 3dfc819849 | wili | 2025-07-12 | [BUG5374319][fix] WAR for draft-target-model unit tests error (#5958) |
| 3840 | 8950223f6f | Mike Iovine | 2025-07-12 | [fix] Remove SpecConfig and fix thread leak issues (#5931) |
| 3841 | bc1d4fb5da | Enwei Zhu | 2025-07-12 | [NvBug 5378370] fix: Fix alltoall for llama4 (apply_router_weight_on_input=True) (#5902) |
| 3842 | 308776442a | Chang Liu | 2025-07-11 | [nvbug/5308432] fix: extend triton exit time for test_llava (#5971) |
| 3843 | 63cf929188 | juney-nvidia | 2025-07-12 | Added code owners for LLM API (#5960) |
| 3844 | 041f1fa513 | Thor Johnsen | 2025-07-11 | [TRTLLM-6264] Fix flaky test_e2e.py::test_openai_lora (#5885) |
| 3845 | 6304866ce8 | 2ez4bz | 2025-07-11 | [refactor] Move vision parts from processor to model for Gemma3 (#5888) |
| 3846 | 509363d858 | xinhe-nv | 2025-07-11 | tests: update sanity tests & fix tests (#5906) |
| 3847 | f4e0425a7b | Shi Xiaowei | 2025-07-11 | doc: update the link of the diagram (#5953) |
| 3848 | 49359574c1 | Shi Xiaowei | 2025-07-11 | [TRTLLM-5673] Doc: ensure the disagg doc is up to date (#5938) |
| 3849 | c5fb692a7d | ChristinaZ | 2025-07-11 | Refactor the rest routing part for the routing kernels in the MoE TRT-LLM backend (#5771) |
| 3850 | 37293e4dfd | Shi Xiaowei | 2025-07-11 | blog: add qwen3 disagg perf metrics (#5822) |
| 3851 | fbb4cc7379 | William Tambellini | 2025-07-10 | [TRTLLM-4770][feat] Enhance cpp executor cmake to listen to ENABLE_MU… (#5104) |
| 3852 | 0385f89abc | brb-nv | 2025-07-10 | test: Fix Gemma3 unit tests due to transformers upgrade (#5921) |
| 3853 | 854655f2f7 | Void | 2025-07-11 | deepEP fp4 post quant all2all dispatch (#5881) |
| 3854 | aa4eebe973 | Frank | 2025-07-10 | [enhance] Add the ability to write a request timeline. (#5258) |
| 3855 | 682acd40da | Zhihan Jiang | 2025-07-10 | [nvbugs/5321981] Cherrypick fix: Fix the Llama3.1 405B hanging issue. (#5698) (#5925) |
| 3856 | c19840235d | 2ez4bz | 2025-07-10 | [fix] Fix mistral unit tests due to transformers upgrade (#5904) |
| 3857 | c32c9e2fad | Iman Tabrizian | 2025-07-10 | doc: Add instructions for running gemma in disaggregated serving (#5922) |
| 3858 | 4d071eb2d1 | Linda | 2025-07-10 | feat: binding type build argument (pybind, nanobind) (#5802) |
| 3859 | 2e3cf42e03 | wili | 2025-07-10 | [refactor] Simplification of Speculative decoding configs (#5639) |
| 3860 | 67a39dbd63 | Zhanrui Sun | 2025-07-10 | infra: [TRTLLM-6054][TRTLLM-5804] Fix two known NSPECT high vulnerability issues and reduce image size (#5434) |
| 3861 | 41ef1ade19 | narutolhy | 2025-07-10 | feat:enable kvcache to be reused during request generation (#4028) |
| 3862 | 7b09a415c1 | Kaiyu Xie | 2025-07-10 | fix: Make the bench serving script compatible with different usages (#5905) |
| 3863 | 8b9a030a5c | Jinyang Yuan | 2025-07-10 | [fix] Fix MoE workspace info by storing Torch tensor itself instead of data_ptr (#5900) |
| 3864 | 3aa53ec36c | Yiqing Yan | 2025-07-10 | [None] - Waive L0 tests (#5915) |
| 3865 | 055c4a9fe6 | Enwei Zhu | 2025-07-10 | [NvBug 5370718, 5371538] fix: Fix incremental detokenization (#5825) |
| 3866 | dc32f9ae73 | CarstyYou | 2025-07-10 | [fix] fix tileN cannot % 16==0 & support sm89 deepgemm bmm (#5531) |
| 3867 | 7d21b55b5a | Anthony Chang | 2025-07-10 | [feat] Add TRTLLM MoE nvfp4 cubins for mid-high concurrency; attention_dp for TRTLLM MoE (#5723) |
| 3868 | 3ec3ff1d82 | Aurelien Chartier | 2025-07-09 | chore: remove support for llmapi + TRT backend in Triton (#5856) |
| 3869 | e289a98d5a | QI JUN | 2025-07-10 | avoid nesting NCCL group in allgather and reduce scatter OPs (#5866) |
| 3870 | 07f6da763d | Yan Chunwei | 2025-07-10 | [TRTLLM-5530] chore: rename LLM.autotuner_enabled to enable_autotuner (#5876) |
| 3871 | 6490a27ad7 | Hanjun Cho | 2025-07-10 | [feat] Add TensorRT-Engine Qwen3 (dense) model support (#5650) |
| 3872 | f57b3d6829 | Venky | 2025-07-09 | Waive unittest failures introduced by PR#5345 (removal of `ScaffoldingOutput` class) (#5886) |
| 3873 | 76c3a12bcb | peaceh-nv | 2025-07-10 | [fix] WAR to fix the illegal memory access issue in moe gemm on SM120 (#5636) |
| 3874 | 3209b31665 | brb-nv | 2025-07-09 | feat: Custom masking utils for Gemma3 VLM (#5853) |
| 3875 | 87fe44fd29 | 2ez4bz | 2025-07-09 | feat(models): Mistral3.1 VLM pytorch backend support (#5529) |
| 3876 | b61a717275 | Chang Liu | 2025-07-09 | [1/N][TRTLLM-5195][feat] Share PyTorch tensor between processes (#5396) |
| 3877 | 3f7cedec7c | Wanli Jiang | 2025-07-10 | Update transformers to 4.53.0 (#5747) |
| 3878 | 74dca0aa7b | DylanChen-NV | 2025-07-09 | [NVBUG-5304516/5319741]Qwen2.5VL FP8 support (#5029) |
| 3879 | 52684d79f7 | peaceh-nv | 2025-07-09 | Fix : fix moe regression for sm120 (#5823) |
| 3880 | 5aa958a11a | tomeras91 | 2025-07-09 | [TRTLLM-5838][fix] fix max batch size and max tokens in kv cache estimations for Nemotron-H (#5371) |
| 3881 | 10e686466e | ixlmar | 2025-07-09 | fix: use current_image_tags.properties in rename_docker_images.py (#5846) |
| 3882 | a32f7083b4 | Omer Ullman Argov | 2025-07-09 | [ci] parallelize torch unittests (#5714) |
| 3883 | 3e3b1769ad | Dom Brown | 2025-07-09 | [TRTLLM-5881] feat: Integrate TRT-LLM Gen FP4 block scale MoE with Pytorch workflow kernel autotuner (#5764) |
| 3884 | dd3c736c7e | dongxuy04 | 2025-07-09 | chore: some refactor on WideEP (#5727) |
| 3885 | 64fd64fcf2 | chenfeiz0326 | 2025-07-09 | [TRTLLM-6262] Fix Llama4 Scout FP4 crash issue (#5834) |
| 3886 | 4df5f96c8d | Chang Liu | 2025-07-08 | [Bugfix] LLama4: fix for llama4 multimodal support (#5809) |
| 3887 | e277766f0d | Erin | 2025-07-08 | chores: merge examples for v1.0 doc (#5736) |
| 3888 | 5ab1cf5ae6 | Xianjie Qiao | 2025-07-09 | Remove unnecessary benchmarking results (#5852) |
| 3889 | d14dd2f597 | Lucas Liebenwein | 2025-07-08 | [AutoDeploy] re-enable waive for flaky AD test (#5867) |
| 3890 | 9d894bc0cb | Bo Li | 2025-07-09 | fix: [https://nvbugspro.nvidia.com/bug/5375656] Unwaive for bug 5375656. (#5842) |
| 3891 | 2bd09ed2d4 | brb-nv | 2025-07-08 | fix: Skip rope scaling for local layers in Gemma3 VLM (#5857) |
| 3892 | c24eb67054 | jiahanc | 2025-07-08 | Doc: fix link in llama4 Maverick example (#5864) |
| 3893 | e1fb1de4d9 | Wanli Jiang | 2025-07-09 | feat: TRTLLM-6224 update xgrammar version to 0.1.19 (#5830) |
| 3894 | e4c777df7d | Jhao-Ting Chen | 2025-07-08 | Add is_fp8_output key to XQA kernel cubin hashing (solves Eagle3-one-engine Hopper fp8 bug) (#5813) |
| 3895 | e27215ca03 | Venky | 2025-07-08 | test: Validate and add accuracy& perf tests for Ministral-8B-Instruct[-FP8](pytorch only) (#5654) |
| 3896 | 607bf4c395 | jiahanc | 2025-07-08 | Doc: Add llama4 Maverick eagle3 and max-throughput and low_latency benchmark guide (#5810) |
| 3897 | d6d2ab2c99 | Omer Ullman Argov | 2025-07-09 | [fix] Catch inference failures in `trtllm-bench` (#5841) |
| 3898 | b6013da198 | xavier-nvidia | 2025-07-08 | Fix GEMM+AR fusion on blackwell (#5563) |
| 3899 | a79b73f577 | Fridah-nv | 2025-07-08 | fix: [5376140] [AutoDeploy] Update unit tests: skip all_close assert for dropout in attention, increase tolerance for rope op test (#5855) |
| 3900 | c508b994b6 | Iman Tabrizian | 2025-07-08 | Fix lost requests for disaggregated serving (#5815) |
| 3901 | e50d95c40d | Yan Chunwei | 2025-07-09 | chore [TRTLLM-6161]: add LLM speculative decoding example (#5706) |
| 3902 | da8c7372d4 | Pamela Peng | 2025-07-08 | [TRTLLM-5366][feat]Add support for sm121 (#5524) |
| 3903 | 08a3dfeb2b | Chang Liu | 2025-07-08 | [nvbug/5308432] unwaive test: post-merge-triton_backend-test_llava (#5814) |
| 3904 | e3ccca06e1 | Dom Brown | 2025-07-08 | test: reduce redundant test cases for TRTLLM Gen FP8 MoE (#5845) |
| 3905 | bb5b16fcb9 | Kaiyu Xie | 2025-07-08 | feat: Return context response immediately when stream_interval > 1 (#5836) |
| 3906 | 3079e8cf0c | Yiteng Niu | 2025-07-08 | [TRTLLM-5878] update nspect version (#5832) |
| 3907 | e3268a4221 | Raayan Dhar | 2025-07-08 | [TRTLLM-5847][feat] Support n-gram speculative decoding with disagg (#5732) |
| 3908 | e104f8bbb5 | Yukun He | 2025-07-08 | [5305318] fix: Fix the accuracy issue when reduce_fusion is enabled for GEMMA model. (#5801) |
| 3909 | b01d1c28f7 | Yegor | 2025-07-08 | [feat] Detokenize option in /v1/completions request (#5382) |
| 3910 | ba0aea1da6 | Tailing Yuan | 2025-07-08 | Fix a quote error introduced in #5534 (#5816) |
| 3911 | 541ab77189 | Yiteng Niu | 2025-07-08 | update namelist in blossom-ci (#5838) |
| 3912 | ec0d7e64b9 | Yiqing Yan | 2025-07-08 | [Infra] - Waive L0 test (#5837) |
| 3913 | 89bbb230cc | xinhe-nv | 2025-07-08 | tests: waive failed cases on main (#5781) |
| 3914 | 035155df7c | Tailing Yuan | 2025-07-08 | Fix: ignore nvshmem_src_*.txz from `confidentiality-scan` (#5831) |
| 3915 | eaf8bec88b | xiweny | 2025-07-08 | fix: Disaggregate serving with attention DP (#4993) |
| 3916 | c8fa08da5c | nv-guomingz | 2025-07-08 | doc: update  cuda_graph_config usage part in DS R1 docs   (#5796) |
| 3917 | 5203a0f6df | Yiqing Yan | 2025-07-08 | chore: bump version to 1.0.0rc3 (#5819) |
| 3918 | 55f86ce7ab | Enwei Zhu | 2025-07-08 | [NvBug 5362426] fix: Fix prompt adapter TP2 case (#5782) |
| 3919 | 9258187e98 | Venky | 2025-07-07 | Waive some `test_llama_eagle3` unittests (#5811) |
| 3920 | 864de5b8b2 | Po-Wei (Vincent) | 2025-07-07 | [None][infra] Set the label community action to only run on upstream TRTLLM (#5806) |
| 3921 | dee6644ed9 | Zhenhuan Chen | 2025-07-08 | feat(scaffolding): add streaming scaffolding_llm.generate_async support (#5345) |
| 3922 | 664bf95892 | JieXin Liang | 2025-07-08 | [fix] improve fp4_block_scale_moe_runner type check (#5681) |
| 3923 | 95978e3044 | liji-nv | 2025-07-08 | [fix] https://nvbugs/5333654 Unwaive to check ci status and improve torch compile multi-gpu coverage (#5700) |
| 3924 | 0be41b6524 | nv-guomingz | 2025-07-08 | Revert "chore: [Breaking Change] Rename cuda_graph_config padding_enabled fie…" (#5818) |
| 3925 | 5bc3a15f10 | Yechan Kim | 2025-07-08 | feat: add MultimodalParams & putting all multimodal params into it and refactor HyperCLOVAX & Qwen2/2.5-VL (#5522) |
| 3926 | 5a8173c121 | nv-guomingz | 2025-07-08 | chore: [Breaking Change] Rename cuda_graph_config padding_enabled fie… (#5795) |
| 3927 | a1235ee978 | davidclark-nv | 2025-07-07 | [feat] Adds optional module cache for TRT-LLM Gen Gemm interfaces (#5743) |
| 3928 | 1191555cce | Omer Ullman Argov | 2025-07-07 | [ci] speedup fused moe tests (#5726) |
| 3929 | 30a19fcf7c | Robin Kobus | 2025-07-07 | [TRTLLM-6291] feat: Add user-provided speculative decoding support (#5204) |
| 3930 | 85b4a6808d | Tailing Yuan | 2025-07-07 | Refactor: move DeepEP from Docker images to wheel building (#5534) |
| 3931 | 1260e2f33f | Daniel Cámpora | 2025-07-07 | feat: Optimize TRTLLM Sampler perf single beam single step (#5550) |
| 3932 | 5ca2b9bb15 | DylanChen-NV | 2025-07-07 | [TRTLLM-5812][feat] support FP8 row-wise dense GEMM in torch flow (#5615) |
| 3933 | ed1b3c884a | Yi Zhang | 2025-07-07 | fix: Adjust free GPU memory fraction in KvCacheConfig for DeepSeek R1 tests (#5774) |
| 3934 | dfce61f4b9 | Yan Chunwei | 2025-07-07 | [TRTLLM-5530][BREAKING CHANGE] refactor: LLM arglist rename mixed_sampler to enable_mixed_sampler (#5751) |
| 3935 | ded38ebdbd | xinhe-nv | 2025-07-07 | test: [CI] remove closed bugs (#5770) |
| 3936 | 12d8c7d129 | ChristinaZ | 2025-07-07 | Refactor the topk parallelization part for the routing kernels (#5567) |
| 3937 | 9db2e9ee47 | Bo Li | 2025-07-07 | fix: [nvbug/5368507] Fix test_generate_with_seed CI failure. (#5772) |
| 3938 | de10774c2e | Zheng Duan | 2025-07-07 | chore: log stack trace on error in openai server (#5749) |
| 3939 | 092e0eb86a | Yanchao Lu | 2025-07-07 | [Infra] - Fix a syntax issue in the image check (#5775) |
| 3940 | 85e934a7fe | bhsueh_NV | 2025-07-07 | [Doc] update the document of qwen3 and cuda_graph usage (#5703) |
| 3941 | ec6c7dff1a | Daniel Stokes | 2025-07-07 | feat: Add support for MXFP8xMXFP4 in pytorch (#5535) |
| 3942 | 66f299a205 | Yiteng Niu | 2025-07-06 | [TRTLLM-5878] add stage for image registration to nspect (#5699) |
| 3943 | 2013034948 | Yanchao Lu | 2025-07-06 | [Test] - Waive or fix few known test failures (#5769) |
| 3944 | ae27261094 | Robin Kobus | 2025-07-06 | refactor: decoding inputs (#5679) |
| 3945 | d95ae1378b | Yanchao Lu | 2025-07-06 | [Infra] - Always use x86 image for the Jenkins agent and few clean-ups (#5753) |
| 3946 | 6bddaf6df6 | Julien Debache | 2025-07-05 | chore: Improve documentation of Kv_block_array (#5765) |
| 3947 | b1976c2add | Xianjie Qiao | 2025-07-05 | Add wide-ep benchmarking scripts (#5760) |
| 3948 | 089fd55eda | Xianjie Qiao | 2025-07-05 | Add dummy all_reduce for kernel breakdown (#5745) |
| 3949 | 1b588f8390 | jthomson04 | 2025-07-04 | feat: KV events for sliding window attention (#5580) |
| 3950 | d61893dc77 | Frank | 2025-07-04 | [fix] Update to properly set cuda graphs in trtllm-bench overrides. (#5634) |
| 3951 | d1112aac37 | Stefan Niebler | 2025-07-04 | [TRTLLM-3442] feat: added beam search support to the PyTorch Workflow (#5333) |
| 3952 | ffc0b8f5da | Chuang Zhu | 2025-07-05 | Cache transceiver support  VSWA (#5505) |
| 3953 | 3ed3bbcb5d | HuiGao-NV | 2025-07-04 | Fix: pass allreduce strategy to pytorchConfig (#5746) |
| 3954 | 7f3ea058f0 | Yiqing Yan | 2025-07-04 | [Infra] - Waive L0 flaky test (#5759) |
| 3955 | 32339d1b20 | Shunkangz | 2025-07-04 | Raise shut down error for each request (#4936) |
| 3956 | 471bf0b4fc | ixlmar | 2025-07-04 | fix: check file exists in dev container script (#5755) |
| 3957 | 3869b969a6 | xinhe-nv | 2025-07-04 | test: [CI] Add failed cases into waives.txt (#5718) |
| 3958 | 81c0764012 | Faraz | 2025-07-04 | Cherry pick "[NVBUG:5355009] Modify check for fuse_fp4_quant on SM120 (#5724) |
| 3959 | 07f9cf1519 | Robin Kobus | 2025-07-04 | fix: Improve chunking test and skip empty kernel calls (#5710) |
| 3960 | b8fef809ae | Yiqing Yan | 2025-07-04 | [Infra] - Waive L0 test (#5748) |
| 3961 | e134a52e07 | Tailing Yuan | 2025-07-04 | Perf: reduce DeepEPLowLatency memory and time (#5712) |
| 3962 | c434147366 | nv-guomingz | 2025-07-04 | chore: update doc by replacing use_cuda_graph with cuda_graph_config (#5680) |
| 3963 | 32b244af38 | Yuan Tong | 2025-07-04 | feat: reduce unnecessary kernel generation (#5476) |
| 3964 | a79d8c9f5e | Shunkangz | 2025-07-04 | Fix none response in PD (#5422) |
| 3965 | 134b2383ff | Netanel Haber | 2025-07-04 | [fix: nvbugs/5355493] Correctly clamp max sequence len to max attention window (#5720) |
| 3966 | 94f0252b46 | Linda | 2025-07-03 | Doc: Update invalid hugging face URLs (#5683) |
| 3967 | a0135c0f6f | Emma Qiao | 2025-07-03 | [Infra] - Waive failed cases on release/0.21 (#5674) |
| 3968 | cdaa6abce7 | brb-nv | 2025-07-02 | fix: Investigate Gemma3 1B decoder output discrepancy (#5564) |
| 3969 | 819ae903de | Frank | 2025-07-02 | [https://nvbugspro.nvidia.com/bug/5351333][fix] Update to chunking calculation. (#5625) |
| 3970 | ab488a5a5d | Kaiyu Xie | 2025-07-01 | doc: Fix outdated config in DeepSeek best perf practice doc (#5638) |
| 3971 | 73d30a23c7 | Yi Zhang | 2025-07-01 | test: add more tests for GB200 with 8 GPUs/2 nodes in L0 tests (#5397) |
| 3972 | cb9f596dbe | Zheng Duan | 2025-07-01 | [nvbug 5300551] test: increase block count in eviction test (#5465) |
| 3973 | d0b3d2ac65 | nv-guomingz | 2025-07-01 | fix:https://nvbugs/5362398 (#5609) |
| 3974 | 77288d3671 | Yan Chunwei | 2025-07-01 | fix [nvbug5351244]: test_mpi_session submit sync/async (#5608) |
| 3975 | 7f837b6e8b | xinhe-nv | 2025-07-03 | tests: waive failures on main (#5704) |
| 3976 | 4762e0b244 | Venky | 2025-07-03 | Waive tests : test_openai_lora, test_trtllm_serve_lora_example and test_openai_chat_structural_tag_example (#5740) |
| 3977 | 7a319524da | Clay | 2025-07-04 | feat: support more parameters in openai worker of scaffolding (#5115) |
| 3978 | 24ac9b5f69 | Lucas Liebenwein | 2025-07-03 | [AutoDeploy] merge feat/ad-2025-06-29 (#5737) |
| 3979 | aa72d39b72 | Netanel Haber | 2025-07-03 | MTP and derivatives: Align sample state with trtllm sampler sample state (#5675) |
| 3980 | 0566fa1697 | Po-Wei (Vincent) | 2025-07-03 | [None][infra] Update the auto-community label action to be triggered every hour (#5658) |
| 3981 | 528ff52ef4 | Zhenhuan Chen | 2025-07-03 | [https://nvbugs/5365714] fix(scaffolding): use default LLM rather than trt backend LLM (#5705) |
| 3982 | 2b0c87e613 | Rashid Kaleem | 2025-07-03 | [ModelLoad] Concurrent load model (#5291) |
| 3983 | 8dad22cbe7 | nv-guomingz | 2025-07-03 | chore: refine the default value by using pydantic default instead of … (#5695) |
| 3984 | 1a3bd140ed | Robin Kobus | 2025-07-03 | chore: Remove unused isFullContextRequest method (#5666) |
| 3985 | f91379b7e8 | Netanel Haber | 2025-07-03 | delete duplicate eagle3 and ngram tests (#5711) |
| 3986 | c72856188c | Omer Ullman Argov | 2025-07-03 | [ci] small multigpu speedups (#5643) |
| 3987 | dccbfc8b1e | WeiHaocheng | 2025-07-03 | fix: Set init value for moe expert id (#5660) |
| 3988 | 530897388c | Emma Qiao | 2025-07-03 | [Infra] - Waive a failed case on main (#5702) |
| 3989 | de0b522dfd | Yiqing Yan | 2025-07-03 | [Infra] - Fix test stage check for the package sanity check stage (#5694) |
| 3990 | 7dbecf7272 | tomeras91 | 2025-07-03 | [TRTLLM-4923][feat] Enable CUDA graphs for Nemotron-H (#5646) |
| 3991 | 3c9dd5cd66 | Yiqing Yan | 2025-07-03 | chore: bump version to 1.0.0rc2 (#5645) |
| 3992 | 2a5fdebf10 | Emma Qiao | 2025-07-03 | [Infra] - Waive failed tests for main 0702 (#5671) |
| 3993 | 3a46cf275b | Enwei Zhu | 2025-07-03 | fix: Fix missing arg to alltoall_prepare_maybe_dispatch (#5669) |
| 3994 | afef5127f0 | Fridah-nv | 2025-07-02 | feat:[AutoDeploy] E2E build example for llama4 VLM (#3922) |
| 3995 | 04fa6c0cfc | ixlmar | 2025-07-02 | [TRTLLM-6143] feat: Improve dev container tagging (#5551) |
| 3996 | 31699cbeb1 | Emma Qiao | 2025-07-02 | [Infra] - Set default timeout to 1hr and remove some specific settings (#5667) |
| 3997 | 77082cde38 | Jhao-Ting Chen | 2025-07-02 | [https://nvbugspro.nvidia.com/bug/5329655] [feat] Pytorch path add spec dec param to attention op  (#5146) |
| 3998 | 4cd8543d8c | Robin Kobus | 2025-07-02 | [TRTLLM-1316] refactor: Remove unnecessary pipeline parallelism logic from postProcessRequest (#5489) |
| 3999 | ca7b6ec8d8 | qixiang-99 | 2025-07-02 | Feat/pytorch vswa kvcachemanager (#5151) |
| 4000 | 2d69b55fe8 | Yan Chunwei | 2025-07-02 | chore: enhance yaml loading arbitrary options in LlmArgs (#5610) |
