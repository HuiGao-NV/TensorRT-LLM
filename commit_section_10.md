# Commit Section 10

Commits 4501 to 5000 (total: 5612)

| # | Commit Hash | Author | Date | Message |
|---|------------|--------|------|--------|
| 4501 | 79a94a28f9 | Robin Kobus | 2025-05-29 | refactor: unique_ptr instead of shared_ptr (#4697) |
| 4502 | 2d61174a9b | Mike Iovine | 2025-05-29 | [feat] Support RULER + chunked prefill in lm-eval-harness (#4592) |
| 4503 | fcadce9f8d | Jhao-Ting Chen | 2025-05-29 | [fix] Eagle-2 LLMAPI pybind argument fix. (#3967) |
| 4504 | 255779a91d | QI JUN | 2025-05-29 | Chore: fuse _merge_requests method into _fetch_new_requests method (#4689) |
| 4505 | 2c48ff5898 | yuanjingx87 | 2025-05-28 | [feat] add b200 support via slurm (#4709) |
| 4506 | 33a9ba55f5 | Yan Chunwei | 2025-05-29 | fix: test trtllm-bench mgmn (#4613) |
| 4507 | 500aca4f44 | ruodil | 2025-05-29 | test: remove perf test l40s/l20 oom test cases and unwaive tests (#4755) |
| 4508 | 7b2b657198 | Zhanrui Sun | 2025-05-29 | infra: [TRTLLM-5247][TRTLLM-5248][TRTLLM-5249] Refactor docker build image groovy and support NGC images (#4294) |
| 4509 | 058f83e47b | QI JUN | 2025-05-29 | CI: move post-merge multi GPU test of PyTorch backend to H200 (#4733) |
| 4510 | 7f29a70f53 | Yiqing Yan | 2025-05-29 | Waive L0 test (#4748) |
| 4511 | ac17142495 | Yan Chunwei | 2025-05-29 | chore: rename ExecutorBindingsWorker/Proxy (#4716) |
| 4512 | 2307e91122 | yuanjingx87 | 2025-05-28 | [fix] add back rtx6000pro tests (#4679) |
| 4513 | 812b1abf86 | Arthur Rasmusson | 2025-05-28 | feature: KV Cache GPUDirect Storage (#3209) |
| 4514 | 820c39041f | Erin | 2025-05-28 | chore: [nvbug_5273941] unwaive test_llm_loading_from_ckpt_for_tp2 (#4725) |
| 4515 | bf691b3d28 | Yuxian Qiu | 2025-05-29 | feat: support packed weights in vanilla moe (#4719) |
| 4516 | 6cf1e4d0a9 | Aurelien Chartier | 2025-05-28 | chore: add -f to pkill calls (#4711) |
| 4517 | 12763779c4 | Robin Kobus | 2025-05-28 | chore: Clean up cpp runtime (#4449) |
| 4518 | ed3c67e34a | Ivy Zhang | 2025-05-28 | tests: [https://nvbugspro.nvidia.com/bug/5289908] run maverick bf16 on blackwell (#4722) |
| 4519 | 93283484c2 | xinhe-nv | 2025-05-28 | test: [CI] Add failed cases into waives.txt (#4688) |
| 4520 | 6b96f09ff2 | Aurelien Chartier | 2025-05-28 | chore: remove extra paths to find binaries (#4706) |
| 4521 | 5506f60037 | Yan Chunwei | 2025-05-28 | chore [BREAKING CHANGE]: Flatten PyTorchConfig knobs into TorchLlmArgs (#4603) |
| 4522 | fbe4db207d | ixlmar | 2025-05-28 | feat: forward exceptions to Python and catch OOMs (#4497) |
| 4523 | 66828015a0 | Yiqing Yan | 2025-05-28 | Fix rerun step (#4715) |
| 4524 | c875184f78 | Iman Tabrizian | 2025-05-28 | Add missing serialization classes (#4642) |
| 4525 | fbec0c3552 | amirkl94 | 2025-05-28 | Release 0.20 to main (#4577) |
| 4526 | b800adc65c | Kaiyu Xie | 2025-05-28 | Fix: hang on disagg when MNNVL two-shot AllReduce is enabled (#4678) |
| 4527 | f3fba4cc63 | Martin Marciniszyn Mehringer | 2025-05-28 | doc: Document the docker release image on NGC (#4705) |
| 4528 | 971d16a2ee | Pengyun Lin | 2025-05-28 | [TRTLLM-1658][feat] Enable multiple response in trtllm-serve for TRT backend (#4623) |
| 4529 | 9c4b8f66b4 | Bo Li | 2025-05-28 | feat: Integration of Fused QKNorm+RoPE. (#4611) |
| 4530 | 6493401986 | Shunkangz | 2025-05-28 | Fix handle cancel request for attentionDP (#4648) |
| 4531 | 5700a4ffcd | Yuxian Qiu | 2025-05-28 | feat: Add vanilla MOE. (#4682) |
| 4532 | 06eba1e061 | Martin Marciniszyn Mehringer | 2025-05-27 | Update the description for NGC docker images (#4671) (#4702) |
| 4533 | 29ac4c20e0 | yunruis | 2025-05-27 | fix: fix dsr1 min lat cga ar rate drop(0.2) (#4561) |
| 4534 | e538b0d95e | Yuxian Qiu | 2025-05-27 | refactor: extract and reuse filter_weights. (#4681) |
| 4535 | bb3d998eb1 | xinhe-nv | 2025-05-27 | test: [CI] remove closed bugs (#4638) |
| 4536 | 40a7161f4f | Perkz Zheng | 2025-05-27 | fix: fmha_v2 compilation (#4659) |
| 4537 | 5cdd6bb10f | Lucas Liebenwein | 2025-05-27 | [AutoDeploy] Increased Model Coverage Mass Migration Week 1 (#4468) |
| 4538 | f6c50293d2 | Yiqing Yan | 2025-05-27 | [Infra][TRTLLM-3929] Rerun failure tests (#3264) |
| 4539 | 5cb4f9be33 | Yuan Tong | 2025-05-27 | feat: improve build_wheel.py venv handling (#4525) |
| 4540 | 92a7984945 | Yiqing Yan | 2025-05-27 | Waive L0 tests (#4686) |
| 4541 | 1582361400 | QI JUN | 2025-05-27 | Chore: only pad one dummy request for attention dp scenario (#4664) |
| 4542 | d6e1b71388 | Yanchao Lu | 2025-05-27 | [Test] - Correct waive the Slurm test stage (#4677) |
| 4543 | 268171bc66 | Tracin | 2025-05-27 | [NVBUG 5301980] Fix fp4 gemm padding. (#4662) |
| 4544 | 59f7622281 | xinhe-nv | 2025-05-27 | test: rcca https://nvbugs/5223130 (#4510) |
| 4545 | 157fe62965 | qsang-nv | 2025-05-27 | fix fmha v2 tests (#4661) |
| 4546 | 258d782b0a | Yanchao Lu | 2025-05-27 | [Test] - Waive RTX Pro 6000 Slurm testing (#4672) |
| 4547 | 4318037ca3 | Chuang Zhu | 2025-05-26 | fix disagg config params (#4646) |
| 4548 | 732d92ff62 | yuanjingx87 | 2025-05-26 | [Infra] - Multi-GPU testing support with Slurm (#4454) |
| 4549 | 4fb8df2701 | Yiqing Yan | 2025-05-26 | [Infra] - Add files into the scan ignore list (#4663) |
| 4550 | 88190faa34 | Enwei Zhu | 2025-05-26 | feat: large-scale EP(part 4: Static EP load balancer integration) (#4615) |
| 4551 | 44eb053b95 | QI JUN | 2025-05-26 | introduce RequestQueueItem class instead of using tuple (#4649) |
| 4552 | 93a54457ac | Robin Kobus | 2025-05-26 | [nvbugs/5274894] fix: Sort requests for functional correctness and performance (adapted from #4608) (#4621) |
| 4553 | fd27f89df6 | Shunkangz | 2025-05-26 | fix: Remove duplicate tokenization in generation server (#4492) |
| 4554 | 11fb00783a | Yiqing Yan | 2025-05-26 | [TRTLLM-5327] - Fix guardwords scan step (#4654) |
| 4555 | 502758aaa9 | Robin Kobus | 2025-05-26 | fix: Handle additional model outputs based on pipeline parallel rank (#4498) |
| 4556 | 6f626af386 | Emma Qiao | 2025-05-26 | [TRTLLM-4535][infra]: Add marker TIMEOUT for test level (#3905) |
| 4557 | ce7f5fae5a | Zheng Duan | 2025-05-26 | sort llm request state (#4607) |
| 4558 | 4a81991b65 | QI JUN | 2025-05-26 | Chore: refine shutdown signal of PyExecutor (#4614) |
| 4559 | 2fee408536 | Yiqing Yan | 2025-05-26 | Waive L0 tests (#4645) |
| 4560 | 8f055f5d14 | Yuxian Qiu | 2025-05-26 | feat: Skip sampler for intermediate pp stages. (#4514) |
| 4561 | 4d711be8f4 | Perkz Zheng | 2025-05-26 | Feat: add sliding-window-attention generation-phase kernels on Blackwell (#4564) |
| 4562 | bb2f545729 | Yibin Li | 2025-05-25 | fix pipeline tests due to rebase (#4640) |
| 4563 | 2b8f6d2871 | shaharmor98 | 2025-05-25 | Fix snake case format (#4559) |
| 4564 | 9472c86661 | juney-nvidia | 2025-05-25 | Update main README.md with the LLaMA4 perf news (#4636) |
| 4565 | 5dff0bff8f | Anton | 2025-05-24 | [#4633][doc] Fixed typo in scaffolding README.md (#4634) |
| 4566 | 7a067a8edf | Yiqing Yan | 2025-05-25 | [TRTLLM-5327] - Add scan stage (#4602) |
| 4567 | 4a236d107d | hlu1 | 2025-05-23 | [Fix][Deepseek] Fix bugs in TestDeepSeekR1 (#4413) |
| 4568 | b60846b47d | Chuang Zhu | 2025-05-24 | fix datatype check (#4606) |
| 4569 | 20c15fc04f | Yanchao Lu | 2025-05-24 | Fix invalid testcase name (#4626) |
| 4570 | ef763b0ddc | Yao Yao | 2025-05-23 | fix: rename some terms (#4534) |
| 4571 | 7b2818a47b | Robin Kobus | 2025-05-23 | refactor: CreateNewDecoderRequests (#4452) |
| 4572 | ca3eaf4070 | dominicshanshan | 2025-05-23 | [nvbug/5028235][fix]pytest bindings tokens logtis comparison. (#4424) |
| 4573 | 7b2bb67491 | juney-nvidia | 2025-05-23 | Update CODEOWNERS for PyTorch backend - runtime component (#4620) |
| 4574 | 15a59e57f6 | Robin Kobus | 2025-05-23 | [nvbugs/5301492] ci: waive test_workers_kv_cache_aware_router (#4617) |
| 4575 | 8452775db8 | zhhuang-nv | 2025-05-23 | [TRTLLM-5070][feat] Support FP8 KV Cache Reuse for MLA (#4535) |
| 4576 | bbea2647b1 | Anthony Chang | 2025-05-23 | Qwen3 supports TRTLLM FP4 MoE backend (#4530) |
| 4577 | 419151f358 | juney-nvidia | 2025-05-23 | Update the GH main page to expose tech blogs  (#4610) |
| 4578 | 3ca05330f9 | Yiqing Yan | 2025-05-23 | Waive L0 test (#4609) |
| 4579 | 9ae705af1b | Bo Li | 2025-05-23 | perf: Add fused q_norm/k_norm/RoPE for Qwen3. (#4482) |
| 4580 | 6527c055cf | bhsueh_NV | 2025-05-23 | chore: fix bug of llama lora test (#4566) |
| 4581 | 862bde99b6 | Fanrong Li | 2025-05-23 | draft[doc]: add mtp tech blog (#4580) |
| 4582 | d69c662215 | bhsueh_NV | 2025-05-23 | [Fix][Qwen3] fix bug of qwen3 fp4 workflow with EP (#4575) |
| 4583 | 1cf0e672e7 | coldwaterq | 2025-05-22 | fix: [nvbugs/5066257] serialization improvments (#3869) |
| 4584 | 87f734b563 | djns99 | 2025-05-23 | [https://nvbugs/5297775] fix: Correct memory guard for large MOE tests to account for TP space (#4553) |
| 4585 | 38241b2346 | Yuxian Qiu | 2025-05-23 | fix: Fix moe_ep_groups/moe_cluster_groups in Mapping. (#4555) |
| 4586 | ef280e687e | CarstyYou | 2025-05-23 | [feat] support fp8 blockscale gemm on sm89 (#4481) |
| 4587 | d7443b6068 | Enwei Zhu | 2025-05-23 | [https://nvbugspro.nvidia.com/bug/5181262] [test] Unwaive Mistral Nemo test (#4515) |
| 4588 | e3a534d0ee | nv-guomingz | 2025-05-23 | chore: guardword clean for header file. (#4540) |
| 4589 | d7d455e7ea | pcastonguay | 2025-05-22 | [feat][TRTLLM-5018] Dis serving python runtime trt backend (#4243) |
| 4590 | 60a6c20174 | Kunyao Wu | 2025-05-23 | Scaffoldingllm supports MCP (#4410) |
| 4591 | 338744fba6 | dongxuy04 | 2025-05-23 | fix[nvbug-5295425]: [TRTLLM-5385] fix race condition in MoeLoadBalancer (#4573) |
| 4592 | 1e55d616da | QI JUN | 2025-05-23 | Chore: clean up _gather_dp_requests_num method of PyExecutor (#4571) |
| 4593 | 3549b68c1c | nv-guomingz | 2025-05-23 | chroe:clean useless flag (#4567) |
| 4594 | 9c0de251db | Mike Iovine | 2025-05-22 | [feat] Integrate Hopper chunked attention kernels (#4330) |
| 4595 | 14fc48ada7 | Mike Iovine | 2025-05-22 | [nvbug/5285881][fix] Fix chunked prefill + overlap scheduler (#4402) |
| 4596 | c713eb5799 | Venky | 2025-05-22 | test(perf): Add `Llama-3_1-Nemotron-Ultra-253B-v1` perf tests (cpp) (#4446) |
| 4597 | e5c90883a9 | Robin Kobus | 2025-05-22 | fix: Move cv2 import to load_video function (#4541) |
| 4598 | 558eaecf16 | Chuang Zhu | 2025-05-22 | fix sequence data race (#4565) |
| 4599 | 1e5d526db4 | QI JUN | 2025-05-22 | Chore: clean up _merge_dummy_request method of PyExecutor (#4438) |
| 4600 | 22c01d5b21 | xinhe-nv | 2025-05-22 | test: [CI] Add failed cases into waives.txt (#4549) |
| 4601 | 1a45890dae | ruodil | 2025-05-22 | test: waive hanging cases for perf test (#4562) |
| 4602 | 3410508020 | Chuang Zhu | 2025-05-22 | cache_transceiver_config (#4556) |
| 4603 | e741d2b8d0 | Iman Tabrizian | 2025-05-21 | Add tritonrelease container (#4455) |
| 4604 | 2898d268f9 | Kaiyu Xie | 2025-05-22 | feat: add health_generate route to openai serving (Cherry-pick https://github.com/NVIDIA/TensorRT-LLM/pull/3856) (#4349) |
| 4605 | bc9f1dbede | HuiGao-NV | 2025-05-22 | fix[nvbug-5228840]: Remove test cases of feature not supported anymore (#3972) |
| 4606 | f491244c84 | Aurelien Chartier | 2025-05-21 | feat: add dataset support for benchmark_core_model with LLMAPI (#4457) |
| 4607 | 099cd3ce07 | Kaiyu Xie | 2025-05-22 | chore: Add all_reduce.py benchmark script to test (#4537) |
| 4608 | 9033dd987d | Michal Guzek | 2025-05-21 | [TRTLLM-4932] Add CLI accuracy tests for Phi-4-mini-instruct (#4415) |
| 4609 | 4798d088d9 | Yan Chunwei | 2025-05-22 | chore: Partition LlmArgs into TorchLlmArgs and TrtLlmArgs (#3823) |
| 4610 | 44cfd757b2 | Chuang Zhu | 2025-05-22 | Agent interface impl for NIXL (#4125) |
| 4611 | 1681e9fd1e | Aurelien Chartier | 2025-05-21 | chore: remove extra PYTHONPATH (#4453) |
| 4612 | e1b42be3d1 | Nikita Korobov | 2025-05-21 | fix: TRT-LLM Gen dtype declaration  (#4503) |
| 4613 | 1cffa99792 | Dom Brown | 2025-05-21 | test: Split test_simple into mpi_utils and cache transceiver tests for DGX (#4451) |
| 4614 | dbaddb3a29 | Zongfei Jing | 2025-05-22 | Adding two-shot allreduce kernel and mnnvl multicasting buffer (#4216) |
| 4615 | 0a8461d54c | Venky | 2025-05-21 | test(perf):  Pt.2 Add `Llama-3_3-Nemotron-Super-49B-v1` integration-perf-tests (cpp) (#4499) |
| 4616 | b80b78f87c | Kevin Chen | 2025-05-21 | Add pytorch backend team (#4405) |
| 4617 | 3b12e460e7 | nv-guomingz | 2025-05-21 | chore: clean ucx and nixl mirror. (#4531) |
| 4618 | cd0c826417 | Robin Kobus | 2025-05-21 | refactor: DisaggExecutorTest (#4398) |
| 4619 | 4018806742 | dongxuy04 | 2025-05-21 | feat: large-scale EP(part 3 - refactor: FusedMoe for redundant expert) (#4495) |
| 4620 | 407ef08662 | xinhe-nv | 2025-05-21 | tests: add qwene fp4 tests into QA test list & update sanity test list (#4478) |
| 4621 | 83f1933f0c | ruodil | 2025-05-21 | test: add failed case in waive list and fix some test script issue for perf test (#4527) |
| 4622 | a201ce9d53 | WeiHaocheng | 2025-05-21 | docs: update the introduction for scaffolding (#4360) |
| 4623 | 3d9a2b5eb7 | ruodil | 2025-05-21 | test: remove enable_overlap_schedule in pytorch config and set enable_chunked prefill to be true for isl>2048 cases (#4285) |
| 4624 | 15317ece5a | QI JUN | 2025-05-21 | CI: waive test_fp8_block_scales_4gpus of deepseek v3 lite (#4520) |
| 4625 | 750f412b8f | xinhe-nv | 2025-05-21 | tests: add llama 3.3 70b 2 nodes tests (#4391) |
| 4626 | 6a35c599ef | Perkz Zheng | 2025-05-21 | Clean: fmha codes (#4496) |
| 4627 | ab5bea957d | Chuang Zhu | 2025-05-21 | unwaive some disagg test (#4476) |
| 4628 | db7446fda7 | Ruoqian Guo | 2025-05-21 | Feat: add deep_gemm swapab Kernel (#4430) |
| 4629 | 2372589689 | QI JUN | 2025-05-21 | Chore: waive torch compile test cases of deepseek v3 lite (#4508) |
| 4630 | 3d62727303 | Shi Xiaowei | 2025-05-21 | test: NIXL single process test (#4486) |
| 4631 | 5d438be59a | Thor Johnsen | 2025-05-20 | [TRTLLM-5000][feat] Pytorch implementation of ngram drafter (#3936) |
| 4632 | 9199793848 | Yan Chunwei | 2025-05-21 | fix: llmapi-launch add add trtllm-bench test with engine building (#4091) |
| 4633 | 426f6fd2bc | Perkz Zheng | 2025-05-21 | Feat: add chunked-attention kernels on Blackwell (#4394) |
| 4634 | 62c16b6d37 | Yuxian Qiu | 2025-05-21 | fix: skip weights defined in create_weights for pp. (#4447) |
| 4635 | a030a898d1 | djns99 | 2025-05-21 | perf: Fuse gemm setup function for SM90/SM100 MOE plugin path (#4146) |
| 4636 | 77a0189554 | Zheng Duan | 2025-05-21 | feat: conditional disaggregation in disagg server (#3974) |
| 4637 | 9a8c3ece22 | Venky | 2025-05-20 | test(perf): Add remaining `Phi-4-mini-instruct` perf tests (#4443) |
| 4638 | 19c6e68bec | xinhe-nv | 2025-05-21 | test: [CI] remove closed bugs (#4417) |
| 4639 | e4fa856b9d | Iman Tabrizian | 2025-05-20 | Build Triton for arm (#4456) |
| 4640 | 3d940e77f0 | Rohan Varma | 2025-05-20 | [TRTLLM-5273]feat/Use full attention mask if Llama3 is used as encoder and fix EarlyStopDecoder unsqueeze bug (#4290) |
| 4641 | 8564c5a41f | Robin Kobus | 2025-05-20 | refactor: Unify request order in TRT and PyTorch workflow (#4096) |
| 4642 | f038218f83 | Daniel Cámpora | 2025-05-20 | fix: Fix TRTLLMSampler beam width bug. (#4473) |
| 4643 | a98e7ea26b | Shi Xiaowei | 2025-05-20 | fix: replace the image links in the blog (#4489) |
| 4644 | 174c5188a2 | Yan Chunwei | 2025-05-20 | fix[nvbug/5286515]: trtllm-llmapi-launch on single node single gpu (#4428) |
| 4645 | bc6a69e4cb | Yanchao Lu | 2025-05-20 | [Docs] - Add date and commit info (#4448) |
| 4646 | 7b09cd904d | tomeras91 | 2025-05-20 | [TRTLLM-5085][fix] Nemotron H correctness test (#4444) |
| 4647 | 21aff2e313 | dongxuy04 | 2025-05-20 | feat: large-scale EP(part 2: MoE Load Balancer - core utilities)  (#4384) |
| 4648 | ec4190fb71 | bhsueh_NV | 2025-05-20 | infra: Add qwen3 235B tests into QA (#4483) |
| 4649 | 3485347584 | Martin Marciniszyn Mehringer | 2025-05-20 | doc: [TRTLLM-325]Integrate the NGC image in Makefile automation and document (#4400) |
| 4650 | f2c0565577 | Zhanrui Sun | 2025-05-20 | chore: bump version to 0.21.0rc0 (#4465) |
| 4651 | de409e8468 | Lucas Liebenwein | 2025-05-19 | [AutoDeploy] HF factory improvements (#4371) |
| 4652 | b5edf13b33 | ruodil | 2025-05-20 | test: update test filter in perf test yml file to select cases by gpu name and add cases for RTX 6000 pro (#4282) |
| 4653 | 0a342a42f7 | Michal Guzek | 2025-05-19 | [TRTLLM-4932] Add CLI accuracy tests for Llama-3.3-70B-Instruct and LLM API BF16 variant (#4362) |
| 4654 | 402385588d | xinhe-nv | 2025-05-20 | test: [CI] Add failed cases into waives.txt (#4429) |
| 4655 | 6f3922f318 | kanghui0204 | 2025-05-20 | feat: Low Precision Allreduce for PCIe based GPU  (#4344) |
| 4656 | c8e062bfd3 | Yuxian Qiu | 2025-05-20 | fix: [nvbugs/5287097] Align PP layer distribution between pytorch and TRT flow. (#4399) |
| 4657 | bb02d86b54 | Venky | 2025-05-19 | test(perf): Add some `Llama-3_3-Nemotron-Super-49B-v1` integration-perf-tests (TRT flow, trtllm-bench) (#4128) |
| 4658 | 1c5b0d6a13 | Perkz Zheng | 2025-05-20 | [Feat] add chunked-attention kernels on Hopper (for llama4) (#4291) |
| 4659 | 7656af1b57 | Faraz | 2025-05-19 | [TRTLLM-4618][feat] Fix cutlass MoE GEMM fallback failure on FP8 + add e2e test for Mixtral 8x7B FP8 on RTX6000 Pro (SM120)  (#4335) |
| 4660 | 942ac5c638 | Yanchao Lu | 2025-05-19 | [Docs] - Reapply #4220 (#4434) |
| 4661 | 58e405624a | liji-nv | 2025-05-19 | [https://nvbugs/5123103][fix] Fix torch compile for DeepSeekV3 (#3952) |
| 4662 | c6074c47da | Iman Tabrizian | 2025-05-19 | Add llama4 disagg accuracy tests (#4336) |
| 4663 | 001704cc6a | Shi Xiaowei | 2025-05-19 | fix: temp disable the problem test (#4445) |
| 4664 | c45f414bbf | Dom Brown | 2025-05-19 | Test: Improve model re-use in C++ DGX tests for CI stability (#4263) |
| 4665 | 98018f3bb9 | Yukun He | 2025-05-19 | Downgrade the logger level for fallback tactic warning. (#4440) |
| 4666 | df2798e0c3 | Shi Xiaowei | 2025-05-19 | feat: NIXL interface integration (#3934) |
| 4667 | e70a205dab | Zhenhuan Chen | 2025-05-19 | [TRTLLM-4638] feat(scaffolding): update Reward Controller to PRM specific controller with step split (#4337) |
| 4668 | 62bb7f9286 | Void | 2025-05-19 | fix potential issues in allreduce fusion kernel and ut (#4226) |
| 4669 | 3640fba52e | Adamz-nvidia | 2025-05-19 | Update "Roadmap" link under README.md to the issues with Roadmap label (#4425) |
| 4670 | a43914619f | Kaiyu Xie | 2025-05-19 | fix: wrong argument name `enable_overlap_scheduler` (#4433) |
| 4671 | cf6cd940e5 | Yuxian Qiu | 2025-05-19 | feat: Add pp support for hybrid attn/mamba model (#4358) |
| 4672 | 5b1c88de8d | Yan Chunwei | 2025-05-19 | chore: cleanup perf_evaluator code (#3833) |
| 4673 | 58d2508b89 | Ivy Zhang | 2025-05-19 | tests: Add test cases for rcca cases (#4347) |
| 4674 | a28cf3240c | Yanchao Lu | 2025-05-19 | [Infra] - Always push the release images in the post-merge job (#4426) |
| 4675 | c4a0d768b5 | Ivy Zhang | 2025-05-19 | tests: add qa test mentioned in docs (#4357) |
| 4676 | 791c209006 | Faraz | 2025-05-18 | [TRTLLM-4618][feat] Add Nemotron Super 49B FP8 test on RTX6000 Pro (SM120) (#4363) |
| 4677 | 7de90a66bc | Iman Tabrizian | 2025-05-18 | Remove vila test (#4376) |
| 4678 | ddf01f6266 | juney-nvidia | 2025-05-19 | refine doc (#4422) |
| 4679 | 58e2d6ffa7 | juney-nvidia | 2025-05-19 | Refine doc (#4421) |
| 4680 | ac610b394a | juney-nvidia | 2025-05-19 | Refine doc (#4420) |
| 4681 | 039f7e3118 | Pengyun Lin | 2025-05-19 | [https://nvbugspro.nvidia.com/bug/5243740][fix] deduce default max_tokens for trtllm-serve (#4265) |
| 4682 | 0d7269e2a7 | Yanchao Lu | 2025-05-19 | [Infra][Docs] - Some clean-up for the CI pipeline and docs (#4419) |
| 4683 | 27afcb9928 | shaharmor98 | 2025-05-18 | add changes for fp8, nemotron-nas, API (#4180) |
| 4684 | 3e08cd231c | Kaiyu Xie | 2025-05-18 | fix: Remove real size allocation (#4396) |
| 4685 | 49f993d862 | rakib-hasan | 2025-05-18 | Removing the outdated argument (#4408) |
| 4686 | e87ea745ba | yuanjingx87 | 2025-05-17 | [Infra] - Terminate the Slurm job if node does not come online in 2 hours (#4334) |
| 4687 | 17d48e0009 | Zhanrui Sun | 2025-05-18 | infra: [TRTLLM-5072] Add SBSA release images (#4231) |
| 4688 | fb663b637a | Venky | 2025-05-17 | Extend the Llama-Nemotron-Nano-8B perf-integration-tests (cpp) (#4195) |
| 4689 | cc1bba1686 | Yuxian Qiu | 2025-05-17 | test: Waive tests for nvbugs/5286795. (#4409) |
| 4690 | b618e1f55b | Jinyang Yuan | 2025-05-17 | perf: Eliminate the need for attention DP padding when possible (#3439) |
| 4691 | befb93cbff | hlu1 | 2025-05-16 | [Deepseek] Add accuracy test references for fp8 kvcache (#4374) |
| 4692 | 7c85890ec7 | Lucas Liebenwein | 2025-05-16 | [AutoDeploy] eager pattern matcher new pattern (#4370) |
| 4693 | 0e872ef0b0 | Lucas Liebenwein | 2025-05-16 | [AutoDeploy] fix: proper process group clean up (#4373) |
| 4694 | 9cd8148f28 | Netanel Haber | 2025-05-16 | API Breaking Change + Readability: "decoder"->"sampler" (#4121) |
| 4695 | 13b61405e8 | ixlmar | 2025-05-16 | fix: improve PyExecutor resource allocations (#4299) |
| 4696 | 7b19acfab1 | Tracin | 2025-05-16 | fix: Fix chat template kwargs bug. (#4387) |
| 4697 | 8e4320ede5 | Lucas Liebenwein | 2025-05-16 | [AutoDeploy] configurable cache resize (#4372) |
| 4698 | 4e370a509a | Robin Kobus | 2025-05-16 | refactor: Copy sequence lengths once in decoder setup (#4102) |
| 4699 | bce281d592 | Fridah-nv | 2025-05-16 | feat: [AutoDeploy] update rope matcher with minor variants (Deepseek) (#3638) |
| 4700 | f5b6d453aa | Kefeng-Duan | 2025-05-16 | doc： DS r1 min latency blog (#4386) |
| 4701 | fb437ed709 | liji-nv | 2025-05-16 | [CI] waive accuracy/test_cli_flow.py::TestTinyLlama1_1BChat::test_pp4 (#4397) |
| 4702 | fa3879629e | Nikita Korobov | 2025-05-16 | feat: TRT-LLM Gen integration for BMM and MoE refactoring (#4280) |
| 4703 | 27bdd0c82d | Emma Qiao | 2025-05-16 | [TRTLLM-4886][infra]Try another timeout opt to exit test thread directly instead of gracefully (#4341) |
| 4704 | a6f2a1e918 | NVJiangShao | 2025-05-16 | Fix test_fused_moe_w4afp8 (#4393) |
| 4705 | df19430629 | Daniel Cámpora | 2025-05-16 | chore: Mass Integration 0.19 (#4255) |
| 4706 | f7ad49bb9b | ixlmar | 2025-05-16 | chore: improve log-level setting UX (#4352) |
| 4707 | d5578b37fc | HuiGao-NV | 2025-05-16 | Change the method to calculate kv memory size in tests (#4332) |
| 4708 | f5ddb7ab4a | Yuan Tong | 2025-05-16 | fix: support TensorRT 10.11+ in FindTensorRT.cmake (#4353) |
| 4709 | 500b43e90c | xinhe-nv | 2025-05-16 | test: [CI] remove closed bugs (#4345) |
| 4710 | 0e14941b7f | Barry Kang | 2025-05-16 | [fix] Fixed incorrect mixed precision MoE conversion (#4351) |
| 4711 | 46c5a56444 | Tracin | 2025-05-16 | Support dynamic per-tensor FP8 (#4250) |
| 4712 | 11aa50d1ea | Stanley Sun | 2025-05-16 | test: add kv cache aware test cases to qa test list (#4257) |
| 4713 | 54d28718c7 | WeiHaocheng | 2025-05-16 | feat: support benchmark on scaffolding (#3328) (#4286) |
| 4714 | 23a63ef9c1 | Zhanrui Sun | 2025-05-16 | update README version (#4381) |
| 4715 | c4cd403af9 | QI JUN | 2025-05-16 | [CI] waive test_chunked_prefill test cases (#4380) |
| 4716 | 6cc3f2093a | NVJiangShao | 2025-05-16 | Fix bias shape in weightOnlyGroupwiseQuantMatmulPlugin for TRT workflow (#4348) |
| 4717 | a1daa22970 | yuxianq | 2025-05-16 | doc: Add docstring for Attention and MLA module. (#4354) |
| 4718 | 13cdf98278 | QI JUN | 2025-05-16 | [CI] update multi-gpu test triggering file list (#4378) |
| 4719 | b0f7522c82 | Suyog Gupta | 2025-05-15 | [AutoDeploy]feat: Add an AutoDeploy compile backend that only calls torch.compile (#4240) |
| 4720 | 25407249a5 | rakib-hasan | 2025-05-15 | [TRTLLM-5054][fix] Removing repeated loading of input processor (#4161) |
| 4721 | 4883121477 | Lucas Liebenwein | 2025-05-15 | [AutoDeploy] fix: disable overlap scheduler until supported (#4365) |
| 4722 | c6e2111f4e | Yechan Kim | 2025-05-16 | feat: enhance trtllm serve multimodal (#3757) |
| 4723 | 4c7191af67 | Iman Tabrizian | 2025-05-15 | Move Triton backend to TRT-LLM main (#3549) |
| 4724 | c44cf34373 | Erin | 2025-05-15 | fix: update checks that broke medusa tests when use_py_session=True (#4339) |
| 4725 | 4f8afe4cc6 | yuxianq | 2025-05-16 | feat: [nvbugs/5261055][nvbugs/5170160] non-invasive pipeline parallelism (#4034) |
| 4726 | 5ebe32f06f | Venky | 2025-05-15 | enh: Enable option in trtllm-bench build subcommand to avoid loading weights (#4142) |
| 4727 | adb0839a33 | Venky | 2025-05-15 | test(perf): Add `Phi-4-mini-instruct` to perf tests (#4267) |
| 4728 | 0e87fcc228 | yuxianq | 2025-05-15 | refactor: use x is None instead of x == None. (#4244) |
| 4729 | 5ce1102a02 | Yanchao Lu | 2025-05-15 | Revert "[test] add qa test mentioned in docs" (#4355) |
| 4730 | 9d3e05486b | Stanley Sun | 2025-05-15 | test: add qa test list for rtx5090 and rtx_pro_6000 (#4254) |
| 4731 | d6b741ddfe | zhhuang-nv | 2025-05-15 | [fix] test_no_kv_cache_reuse for overlap_scheduler (#4350) |
| 4732 | 593f65ff6a | Yuan Tong | 2025-05-15 | fix: better method to help torch find nvtx3 (#4110) |
| 4733 | 4ee82fc0fd | ixlmar | 2025-05-15 | chore: reduce code duplication (#4297) |
| 4734 | f0ca60a95d | Zongfei Jing | 2025-05-15 | Add allreduce and rmsnorm fusion for qwen3 (#4304) |
| 4735 | 14bfb5e0d6 | xinhe-nv | 2025-05-15 | test: FIX test_ptp_quickstart_advanced_deepseek_v3_2nodes_8gpus (#4283) |
| 4736 | 97bc680cd8 | zhhuang-nv | 2025-05-15 | feat: support kv cache reuse for MLA (#3571) |
| 4737 | b4e5df0ee0 | Kaiyu Xie | 2025-05-15 | Breaking change: perf: Enable scheduling overlap by default (#4174) |
| 4738 | 404fbe9b32 | dominicshanshan | 2025-05-15 | [https://nvbugs/5277113][fix]genai-perf API change stress test (#4300) |
| 4739 | d008d6412f | Fridah-nv | 2025-05-14 | feat:[AutoDeploy] Update MoE pattern matcher to drop expert selection logic (#3283) |
| 4740 | b0ce1371ee | Ivy Zhang | 2025-05-15 | [test] add qa test mentioned in docs (#4248) |
| 4741 | 3ea42e7519 | hlu1 | 2025-05-14 | [test] Reorganize TestDeepSeekR1::test_nvfp4_8gpus (#4346) |
| 4742 | e76cf9d9fe | nv-guomingz | 2025-05-15 | fix:https://nvbugs/5234033 enable starcoder trt-flow with transforme… (#3909) |
| 4743 | 5dc3b539ba | Zhanrui Sun | 2025-05-15 | infra: Down the gcc toolset version from 13 to 11 (#4114) |
| 4744 | 2681b26e48 | Zeyu WANG | 2025-05-15 | [TRTLLM-2795] feat: Add yarn support for other models in trt-flow (#3840) |
| 4745 | f9adac3dea | Mike Iovine | 2025-05-14 | [feat] Enable chunked context for flashinfer (#4132) |
| 4746 | 0fd59d64ab | qsang-nv | 2025-05-15 | infra: open source fmha v2 kernels (#4185) |
| 4747 | 498ce8a056 | QI JUN | 2025-05-15 | Revert "feat: Low Precision Allreduce for PCIe based GPU" (#4340) |
| 4748 | efe0972efb | Simeng Liu | 2025-05-14 | doc: Add tensorrtllm_backend serving documentation in the Deepseek-V3 README (#4338) |
| 4749 | 7fb0af9320 | hlu1 | 2025-05-14 | [fix] Remove stale cublas heuristics (#4326) |
| 4750 | d31fefde2c | Robin Kobus | 2025-05-14 | [TRTLLM-5171] chore: Remove GptSession/V1 from TRT workflow (#4092) |
| 4751 | 7c828d767f | sugunav14 | 2025-05-14 | feat: [AutoDeploy] DSV3 mla attn ref op (#4272) |
| 4752 | 42de79d49e | Faraz | 2025-05-14 | test: Added tests for Llama3.1-70B-BF16 on SM120 (#4198) |
| 4753 | 504f4bf779 | Yanchao Lu | 2025-05-14 | [Infra] - Update the upstream PyTorch dependency to 2.7.0 (#4235) |
| 4754 | c67da1fbaa | Robin Kobus | 2025-05-14 | fix: Eagle decoding in TRT flow (#4229) |
| 4755 | 6c45586c51 | Kaiyu Xie | 2025-05-14 | chore: Remove deprecated Python runtime benchmark (#4171) |
| 4756 | f4059c6e2e | HuiGao-NV | 2025-05-14 | Add test case for kv memory estimation (#4158) |
| 4757 | f2bfe2f84f | xinhe-nv | 2025-05-14 | test: [CI] remove closed bugs (#4207) |
| 4758 | 206f82115d | DylanChen-NV | 2025-05-14 | [bug/5247505] fix: CP accuracy on Blackwell (#4188) |
| 4759 | b15f57763d | Anurag Mukkara | 2025-05-14 | tests: PyTorch multimodal using keyword match (#4215) |
| 4760 | 5e634dd1bd | kanghui0204 | 2025-05-14 | feat: Low Precision Allreduce for PCIe based GPU (#3851) |
| 4761 | a66a02a75a | Yiqing Yan | 2025-05-14 | [Infra] Waive L0 test (#4295) |
| 4762 | 20b42912ce | Barry Kang | 2025-05-14 | [TRTLLM-3330][feat] Support DeepSeek-R1 W4A8 on Hopper (#4123) |
| 4763 | bb17649517 | Zongfei Jing | 2025-05-14 | test: Add UT for moe trtllmgen (#4258) |
| 4764 | 1a9298bc66 | bhsueh_NV | 2025-05-14 | CI: add fp8/fp4 ci on Qwen3-30B-A3B (#4266) |
| 4765 | 8280c3d4f2 | brb-nv | 2025-05-13 | feat: Support Gemma3-1b-it in Pytorch workflow (#3999) |
| 4766 | 86ae506b9d | Yi Zhang | 2025-05-14 | [fix] Enable pp tests (#3978) |
| 4767 | 58bb34c460 | tburt-nv | 2025-05-13 | [chore] update CI allowlist 2025-05-13 (#4278) |
| 4768 | 21dbd163a7 | Fridah-nv | 2025-05-13 | [TRTLLM-5188] fix: [AutoDeploy] unwaive AD build test (#4273) |
| 4769 | 23b9705bf4 | Zhanrui Sun | 2025-05-14 | chore: bump version to 0.20.0rc3 (#4261) |
| 4770 | 1ef117688c | brb-nv | 2025-05-13 | test: Validate FP8 and LoRA for Gemma3 (#3670) |
| 4771 | b0a03a289c | Anurag Mukkara | 2025-05-13 | fix: Merge PP overlap and non-overlap executor loop (#3878) |
| 4772 | f408de2d99 | Iman Tabrizian | 2025-05-13 | Waive disagg kv cache load balancer test (#4276) |
| 4773 | cd5b3d21a0 | brb-nv | 2025-05-13 | feat: Support Mistral Small 3.1 24B VLM in TRT workflow (#4183) |
| 4774 | 290649b6aa | Yiqing Yan | 2025-05-13 | [Infra] Waive L0 test (#4269) |
| 4775 | bfa16a63d4 | Yiqing Yan | 2025-05-13 | [Infra] Waive L0 test (#4268) |
| 4776 | c0c3c7f68c | Frank | 2025-05-13 | [TRTLLM-5233][feat]: Add chunking to PyT heuristic for trtllm-bench. (#4133) |
| 4777 | 44d6adfb68 | dominicshanshan | 2025-05-13 | Waive stress test. (#4262) |
| 4778 | cbca6505ff | Yukun He | 2025-05-13 | [nvbugs/5268808][fix] Fix the list-out-of-range access issue of AllReduce workspace on multi-node. (#4159) |
| 4779 | 8f68d56cc1 | Enwei Zhu | 2025-05-13 | [https://nvbugs/5220763] [test] Unwaive Mixtral FP8 TP2 test (#4252) |
| 4780 | fda8b0277a | Yiqing Yan | 2025-05-13 | [Infra][TRTLLM-4374] Upgrade TRT 10.10.0 GA, CUDA 12.9 GA and DLFW 25.04 (#4049) |
| 4781 | e8d7834c50 | Perkz Zheng | 2025-05-13 | fix: [https://nvbugspro.nvidia.com/bug/5238626] illegal memory address when running llama 4 with cuda graph enabled (#4101) |
| 4782 | 1770dd96d8 | v-shobhit | 2025-05-12 | Fix Pipeline Parallelism in Llama4 (#4106) |
| 4783 | 13c8e5a8a8 | nvpohanh | 2025-05-13 | feat: Prefetch safetensors files before loading them (#4140) |
| 4784 | 24be357964 | bhsueh_NV | 2025-05-13 | doc: update qwen3 document (#4246) |
| 4785 | d555fe2530 | ruodil | 2025-05-13 | test: fix for perf test script issue (#4230) |
| 4786 | 0cebc16139 | xinhe-nv | 2025-05-13 | test: [CI] Add failed cases into waives.txt (#4205) |
| 4787 | 7ebae4dcaa | xinhe-nv | 2025-05-13 | test: [CI] Add failed cases into waives.txt (#4203) |
| 4788 | 9643be5f20 | pcastonguay | 2025-05-12 | [TRTLLM-5050][feat] Enable per-request stats with PyT backend  (#4156) |
| 4789 | 286a789549 | Simeng Liu | 2025-05-12 | feat: Add heuristic for GroupRMSNorm kernel selection. (#4047) |
| 4790 | 4becf32360 | Erin | 2025-05-12 | fix: reshape token_ids for lp in torch backend (#4239) |
| 4791 | 035d915fea | Enwei Zhu | 2025-05-13 | [TRTLLM-5081] [test] Align parametrize_with_ids to the pytest behavior (#4090) |
| 4792 | eba3623a54 | wili | 2025-05-13 | Feat: Variable-Beam-Width-Search (VBWS) part4 (#3979) |
| 4793 | a4c3359513 | yuxianq | 2025-05-12 | fix: Reset planned states to avoid memory leak in TrtllmAttentionWrapper (#4227) |
| 4794 | 3dbb087292 | Fridah-nv | 2025-05-12 | [TRTLLM-5188] fix: [AutoDeploy] update output shape of prepare_fused_mha_metadata_fake (#4199) |
| 4795 | b1bee9c394 | Robin Kobus | 2025-05-12 | Revert "Add initial list of CODEOWNERS (#4105)" (#4234) |
| 4796 | 31a2e2d08d | Yiteng Niu | 2025-05-12 | doc: update switcher.json config (#4220) |
| 4797 | c31ca1688c | Enwei Zhu | 2025-05-12 | [https://nvbugs/5214229] [fix] Unwaive lm_head quantization case (#4222) |
| 4798 | b35f9a67f9 | yuxianq | 2025-05-12 | refactor: Allow models to override apply_qk_norm. (#4078) |
| 4799 | c9e2a963e0 | Zheng Duan | 2025-05-12 | feat: add kv cache aware router (#3831) |
| 4800 | c90ebadd84 | Yixin Dong | 2025-05-12 | feat: Support the Structural Tag in guided decoding (#4066) |
| 4801 | 3e9bda3a09 | Yechan Kim | 2025-05-12 | [feat] Support HyperCLOVAX-SEED-Text language part (#3902) |
| 4802 | 33977dbd42 | Martin Marciniszyn Mehringer | 2025-05-12 | infra: [TRTLLM-325] Prepare for NGC release - multiplatform build (#4191) |
| 4803 | 3f29d2f006 | Perkz Zheng | 2025-05-12 | Feat: support exporting softmax statistics and update the kernel-selection heuristic (#4155) |
| 4804 | 9212e9a740 | Zhenhuan Chen | 2025-05-12 | [TRTLLM-4911] feat(scaffolding): make sampling_params only setable by controller (#4151) |
| 4805 | ee92edf2b4 | Ivy Zhang | 2025-05-12 | [https://nvbugspro.nvidia.com/bug/5270564][test] skip per-hopper for llama4 (#4211) |
| 4806 | ba13b51a58 | Robin Kobus | 2025-05-12 | chore: Update CODEOWNERS (#4221) |
| 4807 | 9c03a7ab74 | ruodil | 2025-05-12 | test: add llama_3.2_1B model and fix for test lora script issue (#4139) |
| 4808 | 849d9c343c | xinhe-nv | 2025-05-12 | tests: https://nvbugs/5219534 remove failed tests from test list (#4113) |
| 4809 | 186e2b8c38 | xinhe-nv | 2025-05-12 | [TRTQA-2802][fix]: add --host for mgmn serve examples script (#4175) |
| 4810 | 1333f4f5d5 | Chuang Zhu | 2025-05-12 | remove cache_transceiver_prealloc_size (#4153) |
| 4811 | 3c54e84e47 | Yiqing Yan | 2025-05-12 | [Infra] Waive L0 test (#4212) |
| 4812 | 420048205f | nv-guomingz | 2025-05-12 | chore:update modelopt to 0.29 (#4150) |
| 4813 | b050e70779 | QI JUN | 2025-05-12 | [CI] update pytorch only file list (#4210) |
| 4814 | f021afa241 | QI JUN | 2025-05-12 | [CI] waive two multi-gpu test cases (#4206) |
| 4815 | 7db368c72c | Enwei Zhu | 2025-05-10 | test: Remove CNN Dailymail tasks in favor of GSM8K (#4187) |
| 4816 | fe3a993234 | mayani-nv | 2025-05-09 | chore: PR to fix the formatting errors (#4200) |
| 4817 | aa7300e040 | Kevin Chen | 2025-05-09 | Add initial list of CODEOWNERS (#4105) |
| 4818 | 5c1c69cf9c | mayani-nv | 2025-05-09 | fix: draft target README and assertion for logits-based acceptance (#4167) |
| 4819 | 25533a7736 | mayani-nv | 2025-05-09 | Updating the multimodal models README to add steps for running phi-4-multimodal instruct (#3932) |
| 4820 | 2d0f93a054 | Dom Brown | 2025-05-09 | Refactor: Restructure C++ tests for better modularisation of non-shared code (#4027) |
| 4821 | 0dcf47f1c2 | Frank | 2025-05-09 | [TRTLLM-4717][perf] Set CUDA graph max batch size and padding in throughput benchmark. (#3875) |
| 4822 | 4b8ba7ad61 | Mike Iovine | 2025-05-09 | [fix][nvbug/5244009] Fix llama 4 test lists/scout accuracy issue (#4069) |
| 4823 | 446f62bbab | Tracin | 2025-05-09 | chore: Deprecate evaltool (#4173) |
| 4824 | 0a36db0aa4 | zhhuang-nv | 2025-05-09 | [fix] trtllm-gen mla kernel warnings (#4119) |
| 4825 | d0e672f96d | Martin Marciniszyn Mehringer | 2025-05-09 | chore: [TRTLLM-325][infra] Prepare for NGC release - reduce size of the docker images (#3990) |
| 4826 | bf5b2a2e0a | ruodil | 2025-05-09 | test: amend regex match for perf throughput (#4186) |
| 4827 | ffc13bd325 | chenfeiz0326 | 2025-05-09 | Cherry-pick: Use multi-threading to load MoE expert weights (#4137) |
| 4828 | 0f01826dde | WeiHaocheng | 2025-05-09 | feat: support task collection for to collect information (#3328) (#3824) |
| 4829 | 9082411a50 | xinhe-nv | 2025-05-09 | test: [CI] Add failed cases into waives.txt (#4165) |
| 4830 | 0cf0fce5d3 | Fanrong Li | 2025-05-09 | [fix] Fix add_dummy_requests for spec decoding cases (#4084) |
| 4831 | 5ce5b81281 | ruodil | 2025-05-09 | test: amend default pytorch extra-llm-api-config.yml in perf test (#4176) |
| 4832 | 87f0f79554 | Shi Xiaowei | 2025-05-09 | fix: library path of nixl (#4184) |
| 4833 | e30c76c530 | Zhanrui Sun | 2025-05-09 | infra: Fix pipeline step error in post merge (#3948) |
| 4834 | 1d26a3fd7c | xinhe-nv | 2025-05-09 | test: skip tests on b200 (#3913) |
| 4835 | 77f8e43592 | Fanrong Li | 2025-05-09 | [fix] Fix relaxed acceptance to support enabling it in context phase (#4126) |
| 4836 | e3cf3fd15f | Bo Li | 2025-05-09 | test: Add fp8kv to DS-v3-lite integration tests. (#3950) |
| 4837 | c91d03fa0a | Ivy Zhang | 2025-05-09 | test: move mistral / mixtral test cases in QA test list into the new accuracy test suite (#3440) |
| 4838 | c2d4c2adb6 | Ivy Zhang | 2025-05-09 | [https://nvbugspro.nvidia.com/bug/5260676]test: skip fp8 quantization case for pre-ada (#4095) |
| 4839 | c9cac432dc | Yukun He | 2025-05-09 | chore: Fix pipeline break caused by previous PR (#4081) rebase + pipeline reuse (#4169) |
| 4840 | d80dc40135 | Mike Iovine | 2025-05-09 | [nvbug/5262268][fix] Fix trtllm-bench for llama 4 (#4104) |
| 4841 | 57b2fe2019 | NVJiangShao | 2025-05-09 | [#4085][fix] Fix `apply_per_channel_scale` for extremely large input sequence length. (#4089) |
| 4842 | cdf5ae1547 | Erin | 2025-05-08 | fix: change pp broadcast pattern for LPs (#4130) |
| 4843 | 91bf5e6a8e | Yi Zhang | 2025-05-09 | [TRTLLM-3105][feat] Add Piecewise CUDA Graph Support (#3804) |
| 4844 | fb31f91e15 | Stanley Sun | 2025-05-09 | test: add qwen3 and disaggregated serving accuracy tests to qa test list (#4083) |
| 4845 | 5b61486d87 | Yukun He | 2025-05-09 | chore: Clean up the legacy DeepseekAllreudceFusionOp. (#4081) |
| 4846 | 700d09ab65 | bhsueh_NV | 2025-05-09 | [TRTLLM-5147][Qwen3] fix: fix bug of attention dp on qwen3_moe model (#4141) |
| 4847 | 836c142e1b | pcastonguay | 2025-05-08 | [feat] Allow overriding cli args with yaml file in trtllm-serve (#4164) |
| 4848 | 7147efb2e8 | dongxuy04 | 2025-05-09 | fix: alltoall padding for chunked MoE (#4157) |
| 4849 | 9477661f4c | forrestl | 2025-05-09 | Support RingAttention in the BertAttention plugin and the DiT model (#3661) |
| 4850 | 9afe510367 | Mike Iovine | 2025-05-08 | [fix] Fix llama4 + eagle3 (#3998) |
| 4851 | 57afbf6b79 | Frank | 2025-05-08 | Fix incorrect conversion. (#4112) |
| 4852 | 48ed38a2ac | Lucas Liebenwein | 2025-05-08 | [fix] [AutoDeploy] flashinfer usage on H100 (#4162) |
| 4853 | 7f5716ef83 | chenfeiz0326 | 2025-05-09 | Cherry-pick trtllm-gen from feat/llama4 to main (#4086) |
| 4854 | bb7bcc75c2 | Yukun He | 2025-05-09 | feat: Fallback to NCCL for various patterns when input size is large. (#4080) |
| 4855 | 7d94c9561f | shaharmor98 | 2025-05-08 | feat: support multi lora adapters and TP (#3885) |
| 4856 | 99313af242 | Shi Xiaowei | 2025-05-08 | infra: Add NIXL into the Dockerfile (#3981) |
| 4857 | 5b93273156 | Yuan Tong | 2025-05-08 | feat: adopt new logprob definition in PyTorch flow (#4057) |
| 4858 | 74df12bbaa | Enwei Zhu | 2025-05-08 | [TRTLLM-4480][doc] Documentation for new accuracy test suite and trtllm-eval (#3946) |
| 4859 | 4dfa3ccf43 | nv-guomingz | 2025-05-08 | chore: enhance the cmake experience by ignoring the additional semicolon (#3992) |
| 4860 | 7666bec7c4 | Ivy Zhang | 2025-05-08 | [TRTQA-2861][test]: add nemotron and llama4 cases into qa test (#4053) |
| 4861 | 4468158be4 | xinhe-nv | 2025-05-08 | test: [CI] remove closed bugs (#4046) |
| 4862 | b0dd581e6b | Tracin | 2025-05-08 | Fix TP8 for NVFP4 kv dupilcation. (#4143) |
| 4863 | d1fa80dee3 | Kaiyu Xie | 2025-05-08 | doc: TRTLLM-4797 Update perf-analysis.md (#4100) |
| 4864 | ce8832e80f | Yiqing Yan | 2025-05-08 | [Infra] Waive L0 flaky test (#4148) |
| 4865 | 6e1d2a1320 | yuanjingx87 | 2025-05-08 | feat: Add Slurm support and enable RTX Pro 6000 testing pipeline in CI (#4019) |
| 4866 | 179efd45d4 | Zhanrui Sun | 2025-05-08 | infra: WAR for Argument list too long of globalVars[CACHED_CHANGED_FILE_LIST] (#4131) |
| 4867 | dae6781494 | Enwei Zhu | 2025-05-08 | test: Waive disagg accuracy test (#4124) |
| 4868 | 19eeef1ddd | Enwei Zhu | 2025-05-08 | test: Waive test_llm cases (#4136) |
| 4869 | 389614ca99 | Yan Chunwei | 2025-05-08 | chore: remove data stage in serve example on slurm (#4138) |
| 4870 | 7175392206 | Yanchao Lu | 2025-05-08 | [Infra] - Update code ownership rules for public APIs (#4122) |
| 4871 | d7c51c953b | Ivy Zhang | 2025-05-08 | test: add INTEGRATION_TEST env var to speed up integration test (#3618) |
| 4872 | 81cc60a0fd | zihaok | 2025-05-07 | [feat/] enable attention DP in Llama4 maverick model - part 1 (#4065) |
| 4873 | 26a2679217 | hlu1 | 2025-05-07 | [Deepseek] Refactor Deepseek Decoder layer (#4016) |
| 4874 | bb766eca0a | Simeng Liu | 2025-05-07 | feat: Reduce branch overhead in groupRMSNorm kernels (#4067) |
| 4875 | a7c50cc426 | Venky | 2025-05-07 | enh: Update docker Makefile to use only the visible GPUs of machine (#4097) |
| 4876 | 62cfe74f5f | nv-guomingz | 2025-05-07 | chore:update .gitignore for doc building task. (#3993) |
| 4877 | e650d787e4 | Zhanrui Sun | 2025-05-07 | infra: [TRTLLM-4051] Support only run some backend type test (#3578) |
| 4878 | 721f84a0ac | Pengyun Lin | 2025-05-07 | fix: Align default setting & remove unnecessary check for chat and completion (#3888) |
| 4879 | 4d0e462723 | ruodil | 2025-05-07 | tests: skip writing prepare_dataset output to logs, and add llama_v3.1_8b_fp8, llama_v3.3_70b_fp8, llama_v3.1_405b_fp4 models (#3864) |
| 4880 | 0446270f78 | Yanchao Lu | 2025-05-07 | [Infra] - Update code ownership rules (#4109) |
| 4881 | 0c26059703 | Yan Chunwei | 2025-05-07 | chore: Cleanup deprecated APIs from LLM-API (part 1/2) (#3732) |
| 4882 | bf9ac96de3 | rakib-hasan | 2025-05-06 | Adding option to specify a set of token ids for multimodal tokens (#4107) |
| 4883 | f670a036df | bhsueh_NV | 2025-05-07 | [Qwen3] chore: fix bug of fused_moe on tp > 1 (#4093) |
| 4884 | c28b90984f | Enwei Zhu | 2025-05-07 | [TRTLLM-3925, https://nvbugs/5245262] [fix] Normalize LLM.generate API (#3985) |
| 4885 | 09a28becae | Chuang Zhu | 2025-05-07 | fix cache buffer (#3942) |
| 4886 | 52d4302dda | Kaiyu Xie | 2025-05-07 | bench: TRTLLM-4936 Port benchmark_serving.py (#4011) |
| 4887 | 62fea1e885 | Venky | 2025-05-06 | test(perf): Add Llama-3.1-Nemotron-8B-v1 to perf tests (#3822) |
| 4888 | 001e666fc5 | milesial | 2025-05-06 | fix: Pass local dir to processor creation (#4018) |
| 4889 | cba1793cda | Erin | 2025-05-06 | cleanup logprob params (#4039) |
| 4890 | c56a2aca46 | Daniel Cámpora | 2025-05-06 | fix: Properly get decoding mode according to same logic as cpp. (#4026) |
| 4891 | 72057a0a64 | Robin Kobus | 2025-05-06 | [TRTLLM-3429] feat: Overlap scheduling in C++ runtime (#3625) |
| 4892 | 3ac6637005 | dominicshanshan | 2025-05-06 | fix: trtllm-serve hang in stress test and ds v3 stress parameter update (#3836) |
| 4893 | cf37e31919 | Zhanrui Sun | 2025-05-06 | infra: [TRTLLM-4475][TRTLLM-4565] Add pipeline hierarchy and basic info in the Jenkins job page (#3859) |
| 4894 | e943ad5a2a | Robin Kobus | 2025-05-06 | [https://nvbugs/5247414] fix: draft/target probs shape (#4055) |
| 4895 | b6cfe08c52 | yuxianq | 2025-05-06 | fix: Fix NVLink version decoding. (#3996) |
| 4896 | 5a4794b387 | HuiGao-NV | 2025-05-06 | fix: skip add new slot if request has slot 0 (#3991) |
| 4897 | 4b6c19737b | Yuan Tong | 2025-05-06 | feat: support add internal cutlass kernels as subproject (#3658) |
| 4898 | eeb6c0c83f | Jinyang Yuan | 2025-05-06 | [fix] Loosen the thresholds of test_attention_mla (#4074) |
| 4899 | ac2ab9ba36 | Suyog Gupta | 2025-05-05 | [AutoDeploy][perf] Further optimize flashinfer backend in AutoDeploy (#4024) |
| 4900 | 5c0f554b9e | bhsueh_NV | 2025-05-06 | doc: update qwen3 document (#4073) |
| 4901 | e053cb651b | bhsueh_NV | 2025-05-06 | Fix: fix bug of qwen3 moe (#4058) |
| 4902 | e84dc6b3c7 | pansicheng | 2025-05-06 | feat: add deepseek-r1 reasoning parser to trtllm-serve (#3354) |
| 4903 | 5b1aeb6730 | brb-nv | 2025-05-05 | test: Test OOB access issue in penaltyKernel for endId=-1 (#4035) |
| 4904 | 8caf200322 | Mike Iovine | 2025-05-05 | [fix] Skip debugCheckSemaphores in stream capture mode (#4032) |
| 4905 | 85867d76dd | Iman Tabrizian | 2025-05-05 | test: Add disaggregated serving accuracy tests (#4036) |
| 4906 | 5ee38ad92a | Yanchao Lu | 2025-05-05 | [Test]: Clean up stale waives (#4062) |
| 4907 | ccff86068e | Robin Kobus | 2025-05-05 | fix: request termination in pipeline parallelism (#3892) |
| 4908 | ddfb0fe4e2 | Yanchao Lu | 2025-05-05 | [Test]: Waive unsupported tests (#4059) |
| 4909 | 2cfcdbefee | yuxianq | 2025-05-05 | feat: run mmlu and summarize without engine_dir. (#4056) |
| 4910 | aa980dc92f | Daniel Cámpora | 2025-05-05 | fix: instantiate decoder early in pytorch (#4029) |
| 4911 | 017701343e | yuxianq | 2025-05-05 | fix: apply rope twice in Qwen3. (#4040) |
| 4912 | b5c2327aa0 | Yiqing Yan | 2025-05-05 | Waive L0 tests (#4051) |
| 4913 | aa38e28cfa | Yukun He | 2025-05-05 | fix: [nvbug/5241627] Fix AllReduce kernel hang issue when both tp and pp are enabled. (#3988) |
| 4914 | 266fef88f2 | yuxianq | 2025-05-05 | feat: support to trace executor loop. (#3983) |
| 4915 | 80b96cf910 | tburt-nv | 2025-05-04 | update CI allowlist (#3969) |
| 4916 | bc0cf41592 | Yan Chunwei | 2025-05-05 | chore: refactor llmapi e2e tests (#3803) |
| 4917 | 9f9edd783c | Robin Kobus | 2025-05-04 | refactor: Introduce MpiTag enumeration and update MPI function signatures (#3893) |
| 4918 | 2692daad2e | Emma Qiao | 2025-05-04 | infra: Remove the WAR for test items incompletely (#3313) |
| 4919 | 403370af62 | Robin Kobus | 2025-05-03 | refactor: Move ModelSpec to core library (#3980) |
| 4920 | 0aca05514a | tburt-nv | 2025-05-02 | build: keep using system python for dev install (#4014) |
| 4921 | bf4f7ad744 | qixiang-99 | 2025-05-02 | feat: add Pytorch support of Vision Encoder for multimodal models (#3791) |
| 4922 | 906cddffb0 | Mike Iovine | 2025-05-02 | [infra] Improve llama4 parallelism test coverage (#3821) |
| 4923 | cb2c1cc829 | Daniel Cámpora | 2025-05-02 | [https://nvbugs/5248923] fix: Correctly sizes seqslotmanager considering pp. (#3984) |
| 4924 | 061a6209b5 | Yechan Kim | 2025-05-03 | fix: [nvbug/5252057] Fix kv cache reuse on PyTorch multimodal (#4025) |
| 4925 | c7cf032b89 | Daniel Cámpora | 2025-05-02 | fix: Move all casters to customCasters. (#3945) |
| 4926 | 52edabab30 | hlu1 | 2025-05-01 | Fix Deepseek MTP with moe_backend=TRTLLM (#4001) |
| 4927 | 561ee44737 | bhsueh_NV | 2025-05-02 | add ci and doc for qwen3 (#4022) |
| 4928 | 873c7532fd | Simeng Liu | 2025-05-01 | feat: Add group_rms_norm kernel to normalize multiple inputs in a single operator. (#3438) |
| 4929 | be916b19e0 | Lucas Liebenwein | 2025-05-01 | feat: [AutoDeploy] unfusing attention for native support (#3668) |
| 4930 | a1645c922b | Yukun He | 2025-05-02 | Fallback to NCCL for various patterns when input size is large. (#4009) |
| 4931 | 8fe7bdeacf | Erin | 2025-05-01 | feat: LogitsProcessor in PyTorch backend (#3145) |
| 4932 | f94af0fb86 | Suyog Gupta | 2025-05-01 | [AutoDeploy] Make all ranks agree on kv-cache size (#4007) |
| 4933 | 83f37614ef | Erin | 2025-05-01 | feat: Support Top-K logprobs and prompt_logprobs in LLMAPI (#3388) |
| 4934 | 009d5e9fa3 | xinhe-nv | 2025-05-01 | test: [CI] Add failed cases into waives.txt (#3943) |
| 4935 | 129bf19980 | bhsueh_NV | 2025-05-01 | model: support Qwen3 (#4010) |
| 4936 | dc344b6a4f | nv-guomingz | 2025-05-01 | fix:https://nvbugs/5246733 (#3989) |
| 4937 | b1621e8d4e | YueWeng | 2025-05-01 | feat: add relaxed acceptance for DS (#3865) |
| 4938 | 1294ecb12f | hlu1 | 2025-04-30 | Add attention workspace memory check (#3970) |
| 4939 | 6ded5f984b | milesial | 2025-04-30 | Llama4 processor fixes (#3994) |
| 4940 | 7dbe618683 | Kate Cheng | 2025-04-30 | feat: Add multimodal embedding field in LlmRequest (#3855) |
| 4941 | 1e317c98c6 | Frank | 2025-04-30 | [feat]: Allow for a settable end-of-sequence/padding token in max throughput benchmark. (#3776) |
| 4942 | 9cc5922a0b | Yukun He | 2025-05-01 | Clean up allreduce op in Deepseek V3 model. (#3829) |
| 4943 | b40f351b7a | Dom Brown | 2025-04-30 | [TRTLLM-4460] test: Use Llama 3.2 1B for Llama C++ tests (#3206) |
| 4944 | 941e82faa6 | Erin | 2025-04-30 | waive test_tinyllama_guided_decoding (#3997) |
| 4945 | 7053d0ad5a | tburt-nv | 2025-04-30 | infra: add conan (#3744) |
| 4946 | 8c2c969fcb | Mike Iovine | 2025-04-30 | [fix] Pad requests to maximum draft length in spec decode (#3957) |
| 4947 | dd959de0fd | nv-guomingz | 2025-04-30 | chore: update internal_cutlass_kernels. (#3973) |
| 4948 | 83670571dd | Julien Debache | 2025-04-30 | feat: Mistral-Large-2 support in the Pytorch workflow |
| 4949 | ed887940d4 | Ming Wei | 2025-04-30 | infra: open source XQA kernels (#3762) |
| 4950 | 1ada3c9800 | Chuang Zhu | 2025-04-30 | unwaive disagg tests (#3925) |
| 4951 | a80d2373a3 | Bo Li | 2025-04-30 | fix: [https://nvbugspro.nvidia.com/bug/5243482] If FlashMLA is used, the existence of FMHA based MLA kernels should not be checked. (#3862) |
| 4952 | afb7d3adce | tburt-nv | 2025-04-29 | remove release branch codeowners (#3954) |
| 4953 | cc989ea49f | djns99 | 2025-04-30 | perf: Optimise MOE prologue to use fused setup function (#3790) |
| 4954 | 86e7474a9b | Zhanrui Sun | 2025-04-30 | chore: bump version to 0.20.0rc2 (#3949) |
| 4955 | f568cbb671 | yuxianq | 2025-04-30 | chore: Remove duplicated get_sm_version. (#3935) |
| 4956 | a31afcf3a9 | xinhe-nv | 2025-04-30 | update waive list (#3890) |
| 4957 | c6fea946e1 | QI JUN | 2025-04-30 | chore: update multi-gpu trigger file list (#3971) |
| 4958 | f98a80f9d9 | Pamela Peng | 2025-04-29 | sync internal cutlass kernel changes (#3968) |
| 4959 | 99929e724b | QI JUN | 2025-04-29 | ci: skip pipeline parallelism test of pytorch flow (#3947) |
| 4960 | c8649ce3aa | Pamela Peng | 2025-04-29 | skip blackwell tests for sm120 (#3815) |
| 4961 | e6b482ef47 | Fanrong Li | 2025-04-29 | fix: change the seq_lens sync copy to an async one (#3786) |
| 4962 | 35010e8073 | tomeras91 | 2025-04-29 | Support NemotronH FP8 Quantization |
| 4963 | 68a19a33d4 | xiweny | 2025-04-29 | TRTLLM-4624 feat: Add nvfp4 gemm and moe support for SM120 (#3770) |
| 4964 | 0f8ec693b2 | yuxianq | 2025-04-29 | fix: get head_dim from model’s config. (#3916) |
| 4965 | 8e6eead6a5 | HuiGao-NV | 2025-04-29 | refactor: (part1) Add contraints doc for fusedMoe module. (#3882) |
| 4966 | 06e76020d7 | Junhong Liu | 2025-04-29 | feat: parallel q_b_proj and concat (#3917) |
| 4967 | 8709fe8b53 | Dom Brown | 2025-04-29 | chore: bump version to 0.19.0 (#3598) (#3841) |
| 4968 | 94e6167879 | zhhuang-nv | 2025-04-29 | optimize cudaMemGetInfo for TllmGenFmhaRunner (#3907) |
| 4969 | 2e230b73ec | bhsueh_NV | 2025-04-29 | change log level of some text from info to debug (#3930) |
| 4970 | adfa04745e | yuxianq | 2025-04-29 | fix: revert https://github.com/NVIDIA/TensorRT-LLM/pull/3858 (#3928) |
| 4971 | 0610d0ff84 | bhsueh_NV | 2025-04-29 | add num_scheduled_requests into print_log (#3914) |
| 4972 | cf15efa15e | Frank | 2025-04-28 | [TRTLLM-4883][fix]: Update output speed calculation. (#3923) |
| 4973 | c381380ecc | QI JUN | 2025-04-28 | increase H100 CI nodes for PyTorch only pipelines (#3927) |
| 4974 | 35c5e4f1c5 | Perkz Zheng | 2025-04-29 | feat: add CGA reduction fmha kernels on Blackwell. (#3763) |
| 4975 | d2f312b8e4 | hlu1 | 2025-04-28 | Fix fp8 kvcache (#3877) |
| 4976 | 8a994d879f | WeiHaocheng | 2025-04-29 | feat: fix erros on scaffolding README (#3899) |
| 4977 | f370dd0e32 | qixiang-99 | 2025-04-28 | refactor(test): remove random context sequence lengths and set seed for reproducibility in attention tests (#3919) |
| 4978 | b91da764de | yuxianq | 2025-04-29 | chore: remove DummyKvCacheManager. (#3896) |
| 4979 | dafc28fb85 | Jinyang Yuan | 2025-04-29 | fix: Fix FMHA-based MLA in the generation phase and add MLA unit test (#3863) |
| 4980 | 0577ea0155 | Erin | 2025-04-28 | waive test_attention_no_cache (#3921) |
| 4981 | e534bf09cc | Mike Iovine | 2025-04-28 | [fix] Fix flashinfer + speculation issues (#3686) |
| 4982 | f84dd8f815 | xiweny | 2025-04-28 | test: add deepseek v3 & r1 cases (#3528) |
| 4983 | 5502a522d2 | Yukun He | 2025-04-28 | Fixing minor typo in allreduce kernel selection (#3912) |
| 4984 | e6f7ff3a46 | Mike Iovine | 2025-04-28 | [chore] Make llama4 MoE use maybe_execute_in_parallel (#3779) |
| 4985 | 19da82d68f | Zhenhuan Chen | 2025-04-28 | fix(requirements): fix neither 'setup.py' nor 'pyproject.toml' found (#3906) |
| 4986 | 3617e948fd | Xianjie Qiao | 2025-04-28 | Add docs about DeepSeek-R1 long context support. (#3910) |
| 4987 | ad15e45f07 | Zhenhuan Chen | 2025-04-28 | [TRTLLM-4638	][feat] add best of n support with reward model in scaffolding (#3807) |
| 4988 | 2fe35924e3 | Tao Li @ NVIDIA | 2025-04-28 | Fix the link of doc (#3903) |
| 4989 | 82a8e43557 | xinhe-nv | 2025-04-28 | test: [CI] Add failed cases into waives.txt (#3867) |
| 4990 | e20b67e9fd | xinhe-nv | 2025-04-28 | update waives & tests (#3887) |
| 4991 | d5bca18807 | Zhenhuan Chen | 2025-04-28 | infra: add scaffolding paths to pytorch only files (#3835) |
| 4992 | 068c72ebf8 | Yanchao Lu | 2025-04-28 | Test: waive intermittent test hang (#3894) |
| 4993 | f77252e9ff | bhsueh_NV | 2025-04-28 | fix bug of create cuda stream as default parameter which will be init… (#3764) |
| 4994 | 74cc9e26ff | Iman Tabrizian | 2025-04-27 | infra: install Triton in the base image (#3759) |
| 4995 | ad4226d946 | Yan Chunwei | 2025-04-27 | fix: trtllm-bench build trt engine on slurm (#3825) |
| 4996 | 76f2c631fb | bhsueh_NV | 2025-04-27 | fix: add warmup flag into py_executor to prevent enable profiler during wa… (#3852) |
| 4997 | e2318756ed | Chuang Zhu | 2025-04-27 | cacheTransceiver buffer manager (#3798) |
| 4998 | 136aab5c54 | HuiGao-NV | 2025-04-27 | fix: Update num_of_ctx_tokens in iteration stats (#3785) |
| 4999 | a4b483b969 | Emma Qiao | 2025-04-27 | Infra: Remove empty junit xml (#3794) |
| 5000 | e9fab4f3d9 | bhsueh_NV | 2025-04-27 | fix bug of deepseek gropu_size setting (#3860) |
