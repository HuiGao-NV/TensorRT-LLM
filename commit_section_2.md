# Commit Section 2

Commits 501 to 1000 (total: 5612)

| # | Commit Hash | Author | Date | Message |
|---|------------|--------|------|--------|
| 501 | 6837e73219 | Lizhi Zhou | 2026-02-13 | [https://nvbugs/5847284][fix] fix cuda oom error (#11219) |
| 502 | 0ee757e03a | mpikulski | 2026-02-13 | [TRTLLM-10030][chore] use weakref in atexit handler (#11476) |
| 503 | ca499d600d | yuanjingx87 | 2026-02-12 | [None][infra] Waive failed test in Post-Merge (#11491) |
| 504 | d0e7ba102e | Gal Hubara-Agam | 2026-02-13 | [#11455][fix] Fallback to triton_ssm for nvfp4 quantization (#11456) |
| 505 | db35119c7c | Balaram Buddharaju | 2026-02-12 | [None][chore] Waive test blocking pre-merge (#11498) |
| 506 | 2565f0f4e4 | xxi | 2026-02-13 | [TRTLLM-9108][feat] refactor MoE unit tests: add unified ConfigurableMoE test framework (#11437) |
| 507 | 45d3792245 | dpitman-nvda | 2026-02-12 | [TRTINFRA-7648][chore] Add SECURITY.md file to TensorRT-LLM GitHub (#11484) |
| 508 | dd74f90914 | Iman Tabrizian | 2026-02-12 | [https://nvbugs/5887893][fix] Make NVML work with older CUDA driver versions (#11465) |
| 509 | 5130cbd73e | Ludwig Schneider | 2026-02-12 | [None][fix] Pre-Allocation for Auto-Tuning NCCL_SYMMETRIC (#11326) |
| 510 | 9c2d23c2e5 | Balaram Buddharaju | 2026-02-12 | [https://nvbugs/5888410][fix] Enable warmup for Helix CP (#11460) |
| 511 | 07cd3d4ff2 | tburt-nv | 2026-02-12 | [None][chore] Bump version to 1.3.0rc4 (#11485) |
| 512 | cb1d8d130f | Yukun He | 2026-02-13 | [TRTLLM-10791][feat] TorchSampler general host time optimization (#11141) |
| 513 | 4b2b1d146b | Pamela Peng | 2026-02-12 | [https://nvbugs/5810935][test] unwaive RTX 6000 pro tests (#11452) |
| 514 | 421eb9e39c | Wanli Jiang | 2026-02-12 | [None][feat] Optimize NemotronH model with elementwise and nvfp4 fusion (#11273) |
| 515 | ef7830d137 | xinhe-nv | 2026-02-12 | [None][chore] Add failed cases into waives.txt (#11447) |
| 516 | 11d79aa875 | JennyLiu | 2026-02-12 | [https://nvbugs/5832481][test] Add gpt-oss-120b-Eagle3-throughput case on DGX-Spark (#11419) |
| 517 | 31cdbdfd72 | Tailing Yuan | 2026-02-12 | [https://nvbugs/5808500][chore] Move DeepEPLowLatency tests to machines that support IBGDA with GPU handles (#11178) |
| 518 | d0f3c412ff | mpikulski | 2026-02-12 | [TRTLLM-10030][chore] refactor finish reasons tests (#11445) |
| 519 | 3c1323442b | xinhe-nv | 2026-02-12 | [None][chore] Add failed cases into waives.txt (#11451) |
| 520 | 31314b9fed | Eran Geva | 2026-02-12 | [None][chore] added AutoDeploy nano_v3_scale.yaml (#10845) |
| 521 | 219195688c | Lizhi Zhou | 2026-02-12 | [None][chore] fix a bug in PR11336 (#11439) |
| 522 | 12085536df | Simeng Liu | 2026-02-11 | [TRTLLM-10487][feat] Add user-provided UUID support for multimodal KV cache identification. (#11075) |
| 523 | 936220e746 | Mandar Deshpande | 2026-02-11 | [None][fix] glm engine build dtype (#11246) |
| 524 | e0b11d6ea0 | Perkz Zheng | 2026-02-12 | [https://nvbugs/5804923][none] unwaive test (#11005) |
| 525 | ca9537e17c | William Zhang | 2026-02-11 | [TRTLLM-10858][feat] Multi-image support for EPD disagg (#11264) |
| 526 | 42648734b8 | xinhe-nv | 2026-02-12 | [None][chore] Add failed cases into waives.txt (#11392) |
| 527 | 632c039aea | Yukun He | 2026-02-12 | [TRTLLM-10793][feat] Add BOLT compatible build flags for further experimental usage. (#11297) |
| 528 | 58165d5394 | Liao Lanyu | 2026-02-12 | [None][chore] Introduceing an abstract WaitingQueue interface to decouple the request scheduling logic from specific queue implementations (#11330) |
| 529 | 2c4a4c7b94 | Harris Nover | 2026-02-11 | [None][fix] Fix out-of-bounds array access in kernel factory Get() methods (#11373) |
| 530 | 2d5ebb3fe8 | Harris Nover | 2026-02-11 | [None][chore] Merge residual+hidden into layer norm at the end of each NemotronH MTP, and remove a % operation (#11406) |
| 531 | 7a103035be | Robin Kobus | 2026-02-11 | [None][fix] Remove overlap scheduler adjustment for max sequence length in create_py_executor function (#9229) |
| 532 | c47ff4da43 | Guoming Zhang | 2026-02-11 | [None][feat] Remove the hard code for activation type definition in T… (#11164) |
| 533 | eed9c16560 | Emma Qiao | 2026-02-11 | [None][infra] Pin the torchao version (#11444) |
| 534 | e8b860965b | Yihan Wang | 2026-02-11 | [None][feat] Initial PR for trtllm-gen attention backend (#10784) |
| 535 | 18c992efb1 | Bo Li | 2026-02-11 | [None][doc] Update Skip Softmax attention blog. (#11443) |
| 536 | 8ebd6056fa | Emma Qiao | 2026-02-11 | [None][infra] Waive failed cases for main on 2/11 (#11441) |
| 537 | 3741bb2bb4 | Song Rong | 2026-02-11 | [None][chore] Lock FI version to 0.6.3 (#11371) |
| 538 | 5ea6888dda | Bo Li | 2026-02-11 | [https://nvbugs/5810940][fix] Update lm_eval to 4.9.10 and re-enable Skip Softmax Attention tests on CI. (#11176) |
| 539 | a982554190 | peihengh | 2026-02-10 | [https://nvbugs/5868038][fix] Gracefully terminate disagg serving servers to prevent leftover subprocess warnings (#11395) |
| 540 | 860054c859 | Taylor Yeonbok Lee | 2026-02-10 | [#11203][feat] AutoDeploy: Refactor node caching and improve engine build time (#11250) |
| 541 | f320bc8a9c | tburt-nv | 2026-02-10 | [None][chore] Update allowlist 2026-02-10 (#11426) |
| 542 | a7c4005a3d | Matt Lefebvre | 2026-02-10 | [None][infra] Use frontend dgx-h100 and b200 slurm platforms (#11251) |
| 543 | 411fa9ff87 | mpikulski | 2026-02-10 | [TRTLLM-10030][perf] pin host memory and batch sampler setup in beam search (#11390) |
| 544 | 7d992972b2 | Iman Tabrizian | 2026-02-10 | [TRTLLM-10273][feat] Move MambaCacheManager from Python to C++ (#10540) |
| 545 | d6e49542bd | Leslie Fang | 2026-02-10 | [https://nvbugs/5848377][fix] fix deepeplowlatency with trtllm moe backend running fp8 DS_R1 (#11266) |
| 546 | cf02456613 | Yiqing Yan | 2026-02-10 | [TRTLLM-9711][infra] Fix the testcase name in timeout xml (#9781) |
| 547 | c7689df152 | xinhe-nv | 2026-02-10 | [None][chore] Add failed cases into waives.txt (#11396) |
| 548 | 6e0659dc4d | xinhe-nv | 2026-02-10 | [None][chore] Add failed cases into waives.txt (#11363) |
| 549 | eac56b793e | chenfeiz0326 | 2026-02-10 | [https://nvbugs/5853720][fix] Disable cutedsl argmax kernel to fix perf regression (#11403) |
| 550 | be88fe33be | Bo Deng | 2026-02-10 | [None][fix] fix tinygemm accuracy (#11411) |
| 551 | adc0d82500 | mpikulski | 2026-02-10 | [https://nvbugs/5791242][chore] remove obsolete code (#11388) |
| 552 | 21cdc39e83 | Yiqing Yan | 2026-02-10 | [TRTLLM-10331][infra] Upload unittest sub results in slurm (#10834) |
| 553 | 2a4e70b4a9 | dominicshanshan | 2026-02-10 | [None][chore] Unwaive tests after last MI (#11400) |
| 554 | 8a74ccc57e | Emma Qiao | 2026-02-10 | [None][infra] Waive failed cases for main branch on 02/10 (#11413) |
| 555 | 5f4df89109 | Yuxian Qiu | 2026-02-10 | [None][feat] Fully non-blocking pipeline parallelism executor loop. (#10349) |
| 556 | c233692485 | Lizhi Zhou | 2026-02-10 | [None][doc] add multiple-instances section in disaggregated serving doc (#11412) |
| 557 | 17cc1c13d6 | Emma Qiao | 2026-02-10 | [None][infra] Enable sparck ci since spark cloud migration is done (#11407) |
| 558 | c3cdc93211 | shuyixiong | 2026-02-10 | [TRTLLM-9771][feat] Make update_weights compatible with CUDA Graph (#11267) |
| 559 | 8b2dc57823 | Jonas Li | 2026-02-10 | [None][chore] Mass merge commits from release/1.2.0rc6.post1 branch (#11384) |
| 560 | 0c8b5221b4 | Venky | 2026-02-09 | [TRTC-264][doc] Add CLAUDE.md and AGENTS.md (#11358) |
| 561 | a2fb5afecf | Lucas Liebenwein | 2026-02-09 | [#11032][feat] MLA revisited and GLM 4.7 Flash support (#11324) |
| 562 | d50f010fa9 | Venky | 2026-02-09 | [TRTC-265][chore] Add CODEOWNERS coverage for serve/ and commands/ directories (#11359) |
| 563 | 85919d9517 | Emma Qiao | 2026-02-10 | [None][infra] Disable spark stages due to migration of spark cloud (#11401) |
| 564 | 4fc3644705 | Yuan Tong | 2026-02-10 | [None][fix] Avoid reserved filename on Windows (#11382) |
| 565 | b5508ed75b | JennyLiu | 2026-02-10 | [None][test] Add DGX-Spark multinode perf cases including eagle3 (#11184) |
| 566 | f33086914f | Mike Iovine | 2026-02-09 | [https://nvbugs/5843112][chore] Unwaive ngram test (#11320) |
| 567 | af68c29d3d | Yuxian Qiu | 2026-02-10 | [None][chore] Reduce attention module repeated warnings. (#11335) |
| 568 | fe4c690b6c | Lucas Liebenwein | 2026-02-09 | [https://nvbugs/5855540][fix] AutoDeploy: thread cleanup of eagle test (#11289) |
| 569 | e76b634251 | Ziyi Xiong | 2026-02-10 | [TRTLLM-10321][feat] Support different KV cache layout for one-model spec dec (#10502) |
| 570 | 092f4ce774 | Mike Iovine | 2026-02-09 | [https://nvbugs/5853997][chore] Unwaive gpt-oss test (#11287) |
| 571 | c68d916b6f | Patrice Castonguay | 2026-02-09 | [None][chore] Unit test for disagg gen cancellation (#11108) |
| 572 | ea81a03dd1 | tcherckez-nvidia | 2026-02-09 | [None][chore] update model list (#11364) |
| 573 | 4a743338c3 | Bala Marimuthu | 2026-02-09 | [None][infra] AutoDeploy: Dump graph IR after every transform (#11045) |
| 574 | e719721a60 | Lizhi Zhou | 2026-02-10 | [TRTLLM-10866][feat] implement disaggregated harmony chat (#11336) |
| 575 | 100bfdc516 | Harris Nover | 2026-02-09 | [None][fix] Respect CUDA_LAUNCH_BLOCKING by fixing doCheckError (#11261) |
| 576 | c37531c3f7 | Guiju Zhang | 2026-01-28 | [TRTLLM-10669][fix] Fix Eagle3 draft model weight loading for throughput checkpoint (#11010) |
| 577 | 9384cf8458 | Ivy Zhang | 2026-01-28 | [https://nvbugs/5839569][test] update test constraint (#11054) |
| 578 | 03b635bb08 | Emma Qiao | 2026-01-28 | [None][infra] Waive failed case for release on 1/28 (#11055) |
| 579 | 1524c172a4 | Lizhi Zhou | 2026-01-28 | [https://nvbugs/5821433][fix] WAR for popen in QA env (#10989) |
| 580 | 5f8b1b8cbb | Balaram Buddharaju | 2026-01-27 | [https://nvbugs/5811087][chore] Unwaive Gemma3 27B multimodal test (#11049) |
| 581 | 1ba039f044 | Enwei Zhu | 2026-01-28 | [https://nvbugs/5819452][ci] Unwaive LLaMA2 7B FP8 case (#10997) |
| 582 | abb8106c01 | William Zhang | 2026-01-27 | [https://nvbugs/5835925][fix] Add EPD disagg support for Qwen3 VL MoE (#10962) |
| 583 | 0ead17bb85 | Jin Li | 2026-01-27 | [https://nvbugs/5800646][fix] Fix hang issue by avoid exposing UB buf… (#10842) |
| 584 | d348dd95a7 | yingguo-trt | 2026-01-27 | [None][feat] support Lyris GB200 and increase disagg test timeout (#11019) |
| 585 | fd4e6132e5 | yufeiwu-nv | 2026-01-27 | [None][test] Fix missing test cases (#10881) |
| 586 | d50010cd1f | Stefan Niebler | 2026-01-26 | [https://nvbugs/5769815][fix] Fix offset calculation in _are_stop_words when using speculative decoding (#10854) |
| 587 | 6c4e0c3dbe | Lizhi Zhou | 2026-01-26 | [https://nvbugs/5826689][fix] replace etcd3 with etcd-sdk-python (#10886) |
| 588 | c659280445 | Emma Qiao | 2026-01-26 | [None][infra] Waive failed cases for release branch on 01/26 (#10999) |
| 589 | 59f59efb83 | Pengbo Wang | 2026-01-26 | [https://nvbugs/5779536][fix] Unwaive DeepSeekR1 nvfp4 pp4 mtp test case (#10902) |
| 590 | 90ea6c1e09 | JunyiXu-nv | 2026-01-24 | [https://nvbugs/5804146][fix] Enable responses tests and remove ds to… (#10925) |
| 591 | 196d94a419 | mpikulski | 2026-02-09 | [TRTLLM-10030][perf] avoid syncs in beam search + other improvements (#11349) |
| 592 | 2b60cc181c | Gal Hubara-Agam | 2026-02-09 | [#10780][feat] AutoDeploy: Support per-expert scales in FP8 and NVFP4 MoE (#11322) |
| 593 | 540fb0f29e | Lizhi Zhou | 2026-02-09 | [https://nvbugs/5834212][chore] unwaive test_disaggregated_mixed (#11372) |
| 594 | b3e4ddc953 | Robin Kobus | 2026-02-09 | [None][test] Enhance multi-GPU tests for IFB stats (#11239) |
| 595 | 31db399042 | Robin Kobus | 2026-02-09 | [https://nvbugs/5829097][fix] Disaggregated serving: Only send finished context requests to the KV cache transceiver (#11354) |
| 596 | ab73f6ebc6 | Bo Li | 2026-02-09 | [None][chore] Add microbench for MoE Comm methods. (#10317) |
| 597 | 635d65f9fe | Yihan Wang | 2026-02-09 | [None][chore] Move test_trtllm_flashinfer_symbol_collision.py to tests/unittest/_torch (#11168) |
| 598 | ad8f6748a3 | Emma Qiao | 2026-02-09 | [None][infra] Waive failed case for main branch on 02/09 (#11369) |
| 599 | fe9192f120 | TensorRT LLM | 2026-02-09 | [None][infra] Check in most recent lock file from nightly pipeline |
| 600 | b464c75056 | Yanchao Lu | 2026-02-08 | [None][ci] Waive test failures on main 02/08 (#11365) |
| 601 | f7cf25748b | TensorRT LLM | 2026-02-08 | [None][infra] Check in most recent lock file from nightly pipeline |
| 602 | 03b38e9fbf | mpikulski | 2026-02-07 | [TRTLLM-10030][perf] avoid sync in PyTorchModelEngine when using beam search (#11341) |
| 603 | ffc0f54959 | William Zhang | 2026-02-06 | [https://nvbugs/5848756][fix] Re-take ownership of mrope tensors in prefill worker (#11217) |
| 604 | 408d610877 | TensorRT LLM | 2026-02-07 | [None][infra] Check in most recent lock file from nightly pipeline |
| 605 | 18e611da77 | Iman Tabrizian | 2026-02-06 | [https://nvbugs/5863392][fix] fix partial reuse disabled for disagg (#11247) |
| 606 | f9eed3ecc2 | Gal Hubara-Agam | 2026-02-06 | [None][chore] AutoDeploy update SuperV3 checkpoints and accuracy thresholds (#11107) |
| 607 | b1268e1b37 | Shi Xiaowei | 2026-02-06 | [TRTLLM-9527][feat] Modularization of the transceiver for KV manager v2 (step 4) (#11225) |
| 608 | 66caa67357 | Bo Li | 2026-02-06 | [None][doc] Add sparse attention docs to index. (#11342) |
| 609 | 383c5921c2 | Yueh-Ting (eop) Chen | 2026-02-06 | [https://nvbugs/5756028][fix] Fix VSWA initialization with spec-dec and boundary condition in context input preparation (#10798) |
| 610 | 09807918c7 | Emma Qiao | 2026-02-06 | [None][infra] Waive failed case and delete the redundent waives (#11331) |
| 611 | df1c1a23d4 | Zongfei Jing | 2026-02-06 | [https://nvbugs/5722629] [fix] Remove waive for nvbug 5722629 (#11278) |
| 612 | 9644f024bd | Chenghao Zhang | 2026-02-05 | [None][feat] AutoDeploy: add triton backend for causal conv (#11124) |
| 613 | d160439ef9 | Chenghao Zhang | 2026-02-05 | [#11148][feat] AutoDeploy: Better structure the custom op (#11152) |
| 614 | 639051e98b | Bo Li | 2026-02-06 | [TRTLLM-10021][docs] Skip Softmax Attention blog and docs. (#10592) |
| 615 | 2e6d9350fa | TensorRT LLM | 2026-02-06 | [None][infra] Check in most recent lock file from nightly pipeline |
| 616 | b98f3fca20 | Yan Chunwei | 2026-02-06 | [https://nvbugs/5744432][fix] fix bench script test (#10483) |
| 617 | 86e867297e | Simeng Liu | 2026-02-05 | [https://nvbugs/5856637][ci] Remove the skip for fixed tests. (#11285) |
| 618 | 5521c7b7e7 | yifeizhang-c | 2026-02-06 | [TRTLLM-9457][feat] Add cute dsl fp8 gemm for Blackwell (#10130) |
| 619 | 712dcd31a9 | Lucas Liebenwein | 2026-02-05 | [https://nvbugs/5859869][fix] remove test waive since test is already deprecated (#11288) |
| 620 | a9d4927235 | Chuang Zhu | 2026-02-06 | [TRTLLM-10752][chore] set default val of max_num_tokens_in_buffer as max_seq_len or max_input_len (#11082) |
| 621 | a7494a5ff4 | Harris Nover | 2026-02-05 | [None][chore] Remove outdated comment in model_engine.py (#11240) |
| 622 | d778b26062 | jthomson04 | 2026-02-05 | [None][fix] Reduce host memory usage during model loading (#11119) |
| 623 | e52eb82780 | nvyocox | 2026-02-06 | [#11234][test] Move test_ad_export_onnx to integration examples (#11260) |
| 624 | d3d951d837 | Yuxian Qiu | 2026-02-06 | [None][fix] Fix amax to avoid NaN issue in fp8_blockscale_gemm_kernel. (#11256) |
| 625 | 7d235cfb23 | mpikulski | 2026-02-05 | [TRTLLM-10030][chore] promote SampleState to TypeVar + typing fixes (#11281) |
| 626 | eae480b713 | chenfeiz0326 | 2026-02-05 | [https://nvbugs/5820874][fix] Adjust deepgemm tuning buckets to cover larger num_tokens's scope (#11259) |
| 627 | 719e82c429 | mpikulski | 2026-02-05 | [TRTLLM-10030][perf] beam search (remove GPU sync + fix batching + refactor) (#11276) |
| 628 | e483c7263d | Jiayu Chang | 2026-02-05 | [None][docs] Add CUDA Graph + LoRA in Feature Combination Matrix (#11187) |
| 629 | 0d18b2d7a4 | Yuewei Na | 2026-02-05 | [None][feat] Add priority-based KV cache offload filtering support (#10751) |
| 630 | 9601b17459 | Chang Su | 2026-02-05 | [#11037][fix] Fix proto-to-SamplingParams conversion bugs and add gRPC tests (#11292) |
| 631 | d9b936be94 | Yao Yao | 2026-02-05 | [None][feat] Enhance support for complex models (#11254) |
| 632 | 4c1d9d0c10 | xxi | 2026-02-05 | [None][chore] Pass without_comm to cutlass and deepgemm (#11229) |
| 633 | 36cb5f8c93 | Yechan Kim | 2026-02-05 | [https://nvbugs/5747920][fix] Fix multimodal serve test (#11296) |
| 634 | 8447a96c29 | xinhe-nv | 2026-02-05 | [None][chore] Add failed cases into waives.txt (#11223) |
| 635 | ada4a3a28e | dongfengy | 2026-02-04 | [https://nvbugs/5800679][fix] Re-enable test after bug fixed (#11249) |
| 636 | 9091a193a8 | Jin Li | 2026-02-05 | [https://nvbugs/5837275][fix] Unwaive the failing case that cannot be… (#11137) |
| 637 | ada463d15d | Yi Zhang | 2026-02-05 | [None][fix] Fix comments for kv cache manager v2 (#11207) |
| 638 | 4adf76d860 | TensorRT LLM | 2026-02-05 | [None][infra] Check in most recent lock file from nightly pipeline |
| 639 | 0bd4630cd1 | dongfengy | 2026-02-04 | [https://nvbugs/5854860][fix] Fix cutedsl argmax on sm120 (#11181) |
| 640 | ad2d1df4a9 | dongfengy | 2026-02-04 | [https://nvbugs/5849697][fix] Refine QA Test List for SM120 (#11248) |
| 641 | d9fd8cc951 | Simeng Liu | 2026-02-04 | [https://nvbugs/5674665][fix] Fix accuracy drop in VSWA with KV cache block reuse (#10875) |
| 642 | 767b8dcab3 | Gal Hubara-Agam | 2026-02-04 | [None][chore] AutoDeploy: Set nanov3 and superv3 configs to use flashinfer ssm (#11183) |
| 643 | d90a8e5700 | Grzegorz Kwasniewski | 2026-02-04 | [TRTLLM-10673][feat] Improved layer classification for sharding (#10718) |
| 644 | 925d911fc0 | Lucas Liebenwein | 2026-02-04 | [#10966][feat] AutoDeploy: kv cache manager integration [2/2] (#11149) |
| 645 | e2bd9cce1e | Xianjie Qiao | 2026-02-04 | [None][feat] Support disagg slurm jobs rescheduling (#11218) |
| 646 | f6fff18142 | Yueh-Ting (eop) Chen | 2026-02-04 | [https://nvbugs/5624818][fix] Work around accuracy issue by enforcing paged_context_fmha on Hopper for fmha_v2 (#11192) |
| 647 | 3d8c1a51bd | Zhenhuan Chen | 2026-02-04 | [None][feat] move some disagg script's env configs from bash to submit.py (#10223) |
| 648 | f0ca62b175 | mpikulski | 2026-02-04 | [None][fix] make health_generate work with beam search (#11097) |
| 649 | 02b80bfd58 | xxi | 2026-02-04 | [TRTLLM-9111][feat] provide the uniform test framework to test all MoE backends (#11128) |
| 650 | de6931bbfd | Gal Hubara-Agam | 2026-02-04 | [None][fix] Fix selective_state_update perf regression for T=1 decode path (#11194) |
| 651 | 04b7db3ab5 | chenfeiz0326 | 2026-02-04 | [TRTLLM-8263][feat] Add Disagg Perf Tests (#10912) |
| 652 | 588db0ed64 | tburt-nv | 2026-02-03 | [None][chore] bump version to 1.3.0rc3 (#11238) |
| 653 | 5d522295e9 | Dmitry Barsukoff | 2026-02-04 | [None][fix] Set continuous_usage_stats default to False to follow OpenAI protocol (#10644) |
| 654 | f9e6045f39 | Taylor Yeonbok Lee | 2026-02-03 | [#11086][feat] Optimize Auto Deploy weight loading by preloading weights to CPU (#11059) |
| 655 | f9c4bdf6cf | Lizhi Zhou | 2026-02-04 | [TRTLLM-8921][feat] implement gen-first disagg_service (#11020) |
| 656 | 8f90330239 | yuanjingx87 | 2026-02-03 | [TRTLLM-10019][infra] Move 6 h100 test stage to aihub platform (#11039) |
| 657 | 710d6ef668 | mpikulski | 2026-02-03 | [https://nvbugs/5739981][fix] unwaive tests using opt-125M (#11100) |
| 658 | 2532eb5adc | Chenjie Luo | 2026-02-03 | [None][fix] Align kv_scales with modelopt HF checkpoint (#10745) |
| 659 | 20946554f6 | xinhe-nv | 2026-02-03 | [None][chore] Add failed cases into waives.txt (#11216) |
| 660 | a56aaa585e | Yiqing Yan | 2026-02-03 | [TRTLLM-10839][infra] Set rerun report stage UNSTABLE and pipeline SUCCESS in post-merge when there are passed rerun tests (#11210) |
| 661 | b7767f682f | xinhe-nv | 2026-02-03 | [None][chore] Add failed cases into waives.txt (#11202) |
| 662 | 03f51bb767 | xinhe-nv | 2026-02-03 | [None][chore] Add failed cases into waives.txt (#11193) |
| 663 | e308eb50f4 | Anish Shanbhag | 2026-02-02 | [TRTLLM-10803][fix] Fix mocking of HuggingFace downloads in `with_mocked_hf_download` (#11200) |
| 664 | 304dc6f3c0 | Taylor Yeonbok Lee | 2026-02-02 | [None][chore] Print memory usage before/after accuracy test in CI (#11155) |
| 665 | 12b4ebd0ad | TensorRT LLM | 2026-02-03 | [None][infra] Check in most recent lock file from nightly pipeline |
| 666 | 061d7879d3 | Abby Wei | 2026-02-03 | [TRTLLM-10307][infra] Add --high-priority in bot help message (#11133) |
| 667 | 13420178fc | Yiqing Yan | 2026-02-03 | [TRTLLM-10561][infra] Fix jaraco-context and wheel vulnerability (#10901) |
| 668 | 897eb0df2b | Venky | 2026-02-02 | [None][doc] Fix GLM4-MoE Eagle support documentation (#11198) |
| 669 | 585fbb2734 | gramnarayan | 2026-02-02 | [#10826][feat] AutoDeploy: Eagle One-Model [2/n]: Prefill-Only Implementation (#11073) |
| 670 | 3ef8a4639b | Izzy Putterman | 2026-02-02 | [None][feat] Nemotron H: Eagle3 support (#11131) |
| 671 | cd7762a2fa | Yanchao Lu | 2026-02-02 | [None][test] Fix an invalid test name (#11195) |
| 672 | f1b85fea4c | Rundong Li | 2026-02-02 | [None][feat] Integrate cuda.tile RMS norm kernels (#9725) |
| 673 | 13b0ab9c0e | Mike Iovine | 2026-01-23 | [None][fix] Fix MTP 1-model sampler (#10369) |
| 674 | d9aef94431 | Mike Iovine | 2026-01-23 | [https://nvbugs/5814914][fix] Fix llama sm120 spec dec (#10765) |
| 675 | fa5c3ead05 | Ivy Zhang | 2026-01-23 | [None][test] Update test list (#10883) |
| 676 | de465efc5f | Yukun He | 2026-01-23 | [https://nvbugs/5814309][fix] Use NCCL as fallback to avoid crash due to insufficient memory (#10928) |
| 677 | d31482686c | Zheyu Fu | 2026-01-22 | [https://nvbugs/5680911][fix] Remove @cache decorator to enhance CI stability for unit tests using single process mode (#10730) |
| 678 | 7e5e5b90b9 | Enwei Zhu | 2026-01-23 | [https://nvbugs/5748600][ci] Update guided decoding waive list (#10904) |
| 679 | dd0a5491ba | Yuxian Qiu | 2026-01-23 | [https://nvbugs/5701445][chore] unwaive tests. (#10913) |
| 680 | 40d6f23dad | Yuxian Qiu | 2026-01-23 | [https://nvbugs/5784543][chore] unwaive test. (#10906) |
| 681 | 68a18f7a3a | Lucas Liebenwein | 2026-01-22 | [https://nvbugs/5814247][fix] AutoDeploy: skip mxfp4_moe test unless on Hopper (#10729) (#10850) |
| 682 | ccdd8461ac | Enwei Zhu | 2026-01-22 | [None][fix] Always reset drafting states for GuidedDecoder (#10899) |
| 683 | fafc22e3d4 | Michal Guzek | 2026-01-21 | [https://nvbugs/5691730][fix] Have LoRa bf16 ckpts work with Llama 3.3-70B-fp8 (#9808) |
| 684 | bc2487bc2c | William Zhang | 2026-01-21 | [https://nvbugs/5826962][fix] Fix PD disaggregation for VLMs that use mrope (#10865) |
| 685 | 4d282bd7c1 | Lizhi Zhou | 2026-01-22 | [https://nvbugs/5821433][fix] fix test_auto_scaling for 2 GPUs (#10866) |
| 686 | 6c2ecad2fe | Zhenhuan Chen | 2026-01-22 | [https://nvbugs/5769425][fix] add syncthreads for tinygemm to resolve intermittent accuracy problem (#10873) |
| 687 | 8fd22ac72d | HuiGao-NV | 2026-01-22 | [https://nvbugs/5740377][fix] Prevent out-of-bounds read (#10868) |
| 688 | 2a5b8800e1 | JunyiXu-nv | 2026-01-22 | [https://nvbugs/5754977][fix] Use free port for serve test (#10878) |
| 689 | 0306c0f12c | Yi Zhang | 2026-02-02 | [TRTLLM-9766][feat] Integration of the KVCacheManager V2 to TRTLLM Runtime (#10659) |
| 690 | d3df3f6feb | Emma Qiao | 2026-02-02 | [None][infra] Waive failed cases and disable a stage on 02/02 (#11177) |
| 691 | 9909dca6fa | Kaiyu Xie | 2026-02-02 | [None] [feat] Add PDL support for moeAlltoAllKernels (#10591) |
| 692 | 77afcbddae | Jin Li | 2026-02-02 | [https://nvbugs/5823284][fix] Unwaive no repro hang issue (#11138) |
| 693 | 3800abe26e | TensorRT LLM | 2026-02-02 | [None][infra] Check in most recent lock file from nightly pipeline |
| 694 | fef0e4b17d | Liao Lanyu | 2026-02-02 | [TRTLLM-10666][chore] Refactor request fetching logic for better separation of concerns (#10988) |
| 695 | b00e8338ec | Lizhi Zhou | 2026-02-02 | [https://nvbugs/5834212][fix] prevent routing ctx and gen requests to the same worker; update doc for unique disagg ID (#11095) |
| 696 | ea49afdf0b | Dmitry Barsukoff | 2026-02-02 | [None][fix] AttributeError with return_perf_metrics on tensorrt backend (#10662) |
| 697 | 1c8f8bed00 | Emma Qiao | 2026-02-01 | [None][infra] Waive failed cases for main on 1/30 (#11142) |
| 698 | 0350922c5f | TensorRT LLM | 2026-02-01 | [None][infra] Check in most recent lock file from nightly pipeline |
| 699 | 2e757e8151 | Yanchao Lu | 2026-02-01 | [None][ci] Waive a flaky test on A10 (#11163) |
| 700 | 278ced972b | shuyixiong | 2026-01-31 | [TRTLLM-9771][feat] Allow overriding quantization configs (#11062) |
| 701 | d1e4527c06 | bhsueh_NV | 2026-01-31 | [https://nvbugs/5804683][infra] unwaive Mistral Large3 test (#10680) |
| 702 | 7910d4d2a9 | Frida Hou | 2026-01-30 | [#8242][feat] Add int4 GPTQ support for AutoDeploy (#8248) |
| 703 | 6bace84167 | Guoming Zhang | 2026-01-31 | [TRTLLM-10398][feat] Enable TRTLLM moe backend for Nemotron Super (#10791) |
| 704 | 531f85dc9b | Balaram Buddharaju | 2026-01-30 | [None][feat] Perfect routing for Deepseek models (#11127) |
| 705 | baf9f7b4dc | TensorRT LLM | 2026-01-31 | [None][infra] Check in most recent lock file from nightly pipeline |
| 706 | 492ed27cdf | Venky | 2026-01-30 | [None][doc] Add Glm4MoeForCausalLM to model support matrix (#11156) |
| 707 | 97ab014bdb | Matt Lefebvre | 2026-01-30 | [TRTINFRA-7548][infra] Update GB200 test configs to use frontend SLURM platforms (#11085) |
| 708 | 5a97374f3c | Karthik | 2026-01-30 | [#9525][feat] add L2 norm pattern matcher and fusion transform (#10767) |
| 709 | 4af47208d8 | nvyocox | 2026-01-31 | [None][feat] Export ONNX for DriveOS LLM (#10117) |
| 710 | f42a6cbae0 | yuanjingx87 | 2026-01-30 | [None][infra] Add source code pulse scan to PLC nightly pipeline (#10961) |
| 711 | 5d7411e131 | dominicshanshan | 2026-01-30 | [https://nvbugs/5853997][chore] Waive test (#11132) |
| 712 | a669a163ff | Yechan Kim | 2026-01-30 | [None][doc] Update Qwen2/3-VL's model on supported_models.md (#10797) |
| 713 | 53cb762ee5 | Yao Yao | 2026-01-30 | [None][feat] New KVCacheManagerV2 APIs for Transceiver (#11003) |
| 714 | 5ff244ce54 | Enwei Zhu | 2026-01-30 | [https://nvbugs/5837281][fix] Fix trtllm-serve guided decoding test (#11101) |
| 715 | 9959a5c78e | Tailing Yuan | 2026-01-30 | [None][fix] Remove `-ccache` from build_wheel.py args (#11064) |
| 716 | f2dd0ee128 | Liao Lanyu | 2026-01-30 | [None][chore] Correct sorting order for attention DP scheduling to prioritize non-relaxed requests (#11106) |
| 717 | 322471cdd7 | Yibin Li | 2026-01-30 | [https://nvbugs/5825514][fix] Add null pointer check to parseNpyHeader (#10944) |
| 718 | 4f0c1b2489 | dongfengy | 2026-01-29 | [TRTLLM-10733][feat] Make TRTLLM MOE the default one for GPTOSS on Blackwell (#11074) |
| 719 | ef268e2062 | Jin Li | 2026-01-30 | [TRTLLM-9904][feat] Changes for future KVCacheV2 MTP support (#11029) |
| 720 | 6506d63466 | JennyLiu | 2026-01-30 | [None][test] Add DGX-Spark VLM gemm3-12b bfp16/fp4/fp8 accuracy and perf cases (#11096) |
| 721 | 29a203aedb | TensorRT LLM | 2026-01-30 | [None][infra] Check in most recent lock file from nightly pipeline |
| 722 | e1e3bb8592 | Yueh-Ting (eop) Chen | 2026-01-30 | [https://nvbugs/5775544][fix] Unwaive test (#11023) |
| 723 | 144b61715f | Necofish | 2026-01-30 | [None][fix] Add missing absolute pe in Qwen3-VL Vision Encoder (#11065) |
| 724 | 54ba056924 | yuanjingx87 | 2026-01-29 | [None][infra] Remove invalid account for blossom CI (#11126) |
| 725 | dbad94715b | Chang Su | 2026-01-29 | [None][feat] Add gRPC server for high-performance external router integration (#11037) |
| 726 | e033929221 | Chenghao Zhang | 2026-01-29 | [None][feat] AutoDeploy: Flashinfer kernels bringup (#10867) |
| 727 | 0ad87895f5 | Mike Iovine | 2026-01-29 | [https://nvbugs/5836592][fix] Fix qwen3 eagle test (#11030) |
| 728 | a4880ffdbb | Lucas Liebenwein | 2026-01-29 | [None][fix] AutoDeploy: remove mem check for a log unit test (#11120) |
| 729 | 4345636b04 | Tailing Yuan | 2026-01-30 | [None][chore] Clean up layer-wise benchmarks code (#11092) |
| 730 | ab7dd34bbe | Harris Nover | 2026-01-29 | [None][chore] Consolidate duplicate kv cache reuse variables. (#10935) |
| 731 | 7d31532850 | Stefan Niebler | 2026-01-29 | [TRTLLM-10312][perf] Improve performance of _write_finish_reasons in TorchSampler (#10459) |
| 732 | 80dd6e70c6 | WeiHaocheng | 2026-01-29 | [TRTLLM-10415][feat] Dump thread stacks for hanging tests before time… (#10708) |
| 733 | c7a86f89de | Balaram Buddharaju | 2026-01-28 | [TRTLLM-10264][feat] Support attention DP + Helix CP (#10477) |
| 734 | 21d475a391 | Zhanrui Sun | 2026-01-29 | [None][infra] Waived flaky tests (#11091) |
| 735 | f6dab8388d | Yi Sun | 2026-01-29 | [https://nvbugs/5813452][fix] Fix "Assertion failed: isLeaf() in kvCacheManager.cpp:465" (#10922) |
| 736 | 91528365a9 | Tailing Yuan | 2026-01-29 | [None][feat] Add performance alignment to layer-wise benchmarks (#11018) |
| 737 | 34a730aaf7 | Enwei Zhu | 2026-01-29 | [None][fix] Fix enable_alltoall passed to CutlassFusedMoE (#11016) |
| 738 | 24ac86c485 | Anish Shanbhag | 2026-01-28 | [https://nvbugs/5761391][fix] Include triton-kernels as a packaged dependency (#10471) |
| 739 | e20f9a9c72 | TensorRT LLM | 2026-01-29 | [None][infra] Check in most recent lock file from nightly pipeline |
| 740 | 6fcbf15fb8 | Yiqing Yan | 2026-01-29 | [None][fix] No need to remove the original waive list (#11060) |
| 741 | f03908cf9e | Frida Hou | 2026-01-28 | [None][fix] fix Qwen2/3 export for AutoDeploy (#11007) |
| 742 | 4e10bf8950 | Ludwig Schneider | 2026-01-28 | [None][fix] nccl symmetric with graceful fallbacks (#11042) |
| 743 | 393c3d259e | Bala Marimuthu | 2026-01-28 | [#10245][feat] AutoDeploy: Add Minimax M2 support (#10525) |
| 744 | 744a955cbb | gramnarayan | 2026-01-28 | [None][chore] AutoDeploy: Eagle One-Model [1/n]: PyTorch impl for Eagle3 Llama checkpoint (#10674) |
| 745 | 0ffa77af51 | Emma Qiao | 2026-01-28 | [None][infra] Waive failed cases for main on 1/28 (#11053) |
| 746 | e70a55bd94 | yingguo-trt | 2026-01-28 | [None][feat] support multi_acc and Lyris GB200 test (#11024) |
| 747 | 29647d9446 | Linda | 2026-01-28 | [None][chore] Removing cpp/tensorrt_llm/pybind (#11026) |
| 748 | 38bcee189c | Grzegorz Kwasniewski | 2026-01-28 | [TRTLLM-10362][feat] Added Mamba and MLA layers to the sharding tests (#10364) |
| 749 | 3e17ee4e38 | yuanjingx87 | 2026-01-28 | [None][infra] Update CI allowList (#11040) |
| 750 | d008494232 | Pengbo Wang | 2026-01-28 | [https://nvbugs/5779536][fix] Cherry-pick #10902: Unwaive DeepSeekR1 nvfp4 pp4 mtp test case (#10902) (#11000) |
| 751 | dc5eda546b | xinhe-nv | 2026-01-28 | [None][fix] unwaive tests (#11047) |
| 752 | a7748ceb57 | TensorRT LLM | 2026-01-28 | [None][infra] Check in most recent lock file from nightly pipeline |
| 753 | 1c2e415b3a | dongfengy | 2026-01-27 | [https://nvbugs/5756804][fix] Re-enable passing test (#10986) |
| 754 | 30348b2753 | Yuan Tong | 2026-01-28 | [None][fix] Proper conditional compilation of sm10x cubins (#10839) |
| 755 | c26a8f764c | Matt Lefebvre | 2026-01-27 | [TRTINFRA-7379][infra] Change SLURM config access to use resolvePlatform (#11006) |
| 756 | 6c1862fb33 | NVShreyas | 2026-01-27 | [TRTLLM-10197][chore] Refactor to setup for RNN cache transceiver (#10957) |
| 757 | f25a2c53bb | Evgueni Petrov | 2026-01-28 | [#10877][fix] restore ipv6 support in serve.py (#10929) |
| 758 | bae2fac834 | Simeng Liu | 2026-01-27 | [https://nvbugs/5721661][chore] Unwaive fixed bug. (#11009) |
| 759 | ff3a494f5c | Lucas Liebenwein | 2026-01-27 | [#10013][feat] AutoDeploy: native cache manager integration (#10635) |
| 760 | 7f8c260601 | Gal Hubara-Agam | 2026-01-27 | [https://nvbugs/5843316][chore] waive overlap_scheduler test (#11025) |
| 761 | 552aa32aa2 | xinhe-nv | 2026-01-27 | [None][chore] Add failed cases into waives.txt (#10993) |
| 762 | b575184fca | Yukun He | 2026-01-27 | [TRTLLM-10308][feat] AutoTuner Cache: reorganize cache file for distributed tuning (#10956) |
| 763 | d6f76d2fae | Chuang Zhu | 2026-01-27 | [TRTLLM-9527][feat] change context params and disagg params (step3) (#10495) |
| 764 | fae4985797 | ZhichenJiang | 2026-01-27 | [TRTLLM-9831][perf] Use TMA.RED to improve effective memory bandwidth (#10987) |
| 765 | 6b251cc7fa | Bo Li | 2026-01-27 | [TRTLLM-9390][chore] Add Fake OPs for One-Sided AlltoAll. (#11002) |
| 766 | 93ae8a14ab | Lizhi Zhou | 2026-01-27 | [#10889][fix] fix pydantic deepcopy bug (#11004) |
| 767 | 069ad30bdb | xinhe-nv | 2026-01-27 | [None][chore] Remove closed bugs (#10982) |
| 768 | ea5d811aec | Yiqing Yan | 2026-01-27 | [None][chore] Bump version to 1.3.0rc2 (#11021) |
| 769 | c761b68481 | Emma Qiao | 2026-01-27 | [None][infra] Waive failed cases for main on 01/27 (#11017) |
| 770 | ca9f70f78c | zhhuang-nv | 2026-01-27 | [https://nvbugs/5612438][fix] Add timeout for SeedOSS test (#8683) |
| 771 | 5553391c5e | Tailing Yuan | 2026-01-27 | [TRTLLM-10560][fix] Fix the time of pause() for overlap scheduler (#10943) |
| 772 | 4a206351bb | Wanli Jiang | 2026-01-27 | [TRTLLM-10453][feat] Update mamba decode kernel to flashinfer (#10757) |
| 773 | da43a28b01 | TensorRT LLM | 2026-01-27 | [None][infra] Check in most recent lock file from nightly pipeline |
| 774 | df8be0c50c | ameynaik-hub | 2026-01-26 | [TRTLLM-10276][feat] Integrate cutedsl argmax kernel (#10476) |
| 775 | ff0dd6076e | sunnyqgg | 2026-01-27 | [TRTLLM-10062][feat] Enable MTP for Nemotron  Super (#10754) |
| 776 | 43b8a5561c | tcherckez-nvidia | 2026-01-26 | [None][chore] update AD model list (#10981) |
| 777 | 00f341be49 | Lucas Liebenwein | 2026-01-26 | [#8982][feat] AutoDeploy attention dp support (#10728) |
| 778 | ce556290c9 | Linda | 2026-01-26 | [None][chore] Removing pybind11 bindings and references (#10550) |
| 779 | ce37e27066 | Pengyun Lin | 2026-01-26 | [#10614][fix] gpt_oss first iteration streaming in trtllm-serve (#10808) |
| 780 | 5d7a5e6800 | Pengbo Wang | 2026-01-26 | [https://nvbugs/5779536][fix] Cherry-pick #10855: Unwaive Llama 3.3 related multi GPU tests (#10942) |
| 781 | e405468230 | Bo Li | 2026-01-26 | [TRTLLM-10048][feat] Fuse the AllGather for expert statistics required by the EPLB. (#10885) |
| 782 | 5efee01da1 | Tian Zheng | 2026-01-26 | [None][feat] Add Skip Softmax MLA kernels for Blackwell and Fix an accuracy bug of NVFP4 KV (#10813) |
| 783 | a3a3ceb17f | Emma Qiao | 2026-01-26 | [None][infra] Waive failed case for main branch on 01/26 (#10994) |
| 784 | d3406cb515 | xinhe-nv | 2026-01-26 | [None][chore] Add failed cases into waives.txt (#10976) |
| 785 | c8f1745a6e | yingguo-trt | 2026-01-26 | [https://nvbugs/5661741][feat] Add 250K-token NVFP4 MoE + PDL regression tests (#10911) |
| 786 | 2d8245d125 | xinhe-nv | 2026-01-26 | [None][chore] Add failed cases into waives.txt (#10974) |
| 787 | d2b5954aea | TensorRT LLM | 2026-01-26 | [None][infra] Check in most recent lock file from nightly pipeline |
| 788 | ffab217974 | Enwei Zhu | 2026-01-26 | [None][fix] Fix CuteDSL MoE unittest (#10983) |
| 789 | 45d7022cc3 | Yanchao Lu | 2026-01-26 | [None][test] Waive failed tests on main 1/25 (#10984) |
| 790 | 72ef732bcf | Enwei Zhu | 2026-01-25 | [TRTLLM-10147][perf] Balanced random MoE workload generator for CuteDSL kernel UT, autotuner and layerwise benchmark (#10279) |
| 791 | fd7fd8c39d | Pengyun Lin | 2026-01-21 | [https://nvbugs/5747938][infra] Unwaive trtllm serve example test (#10820) |
| 792 | c98c286c0f | dominicshanshan | 2026-01-21 | [https://nvbugs/5814203][fix] Fix port 8000 being used issue in stress test. (#10756) |
| 793 | ae58a7ed20 | Yanchao Lu | 2026-01-21 | [None][chore] Revert NVIDIA/TensorRT-LLM#10819 (#10870) |
| 794 | bcd2dc490c | Ivy Zhang | 2026-01-21 | [None][test] Update case for release (#10811) |
| 795 | 18f63dfcec | Yanchao Lu | 2026-01-20 | [None][chore] Reduce tedious logs (#10819) |
| 796 | 44aa6c3b8e | Emma Qiao | 2026-01-20 | [None][infra] Waive failed cases for release branch on 01/20 (#10828) |
| 797 | 0f7ec033f7 | mpikulski | 2026-01-20 | [https://nvbugs/5791242][fix] workaround for flashinfer.sampling.sampling_from_logits (#10713) |
| 798 | 8959c41d8b | Patrice Castonguay | 2026-01-19 | [https://nvbugs/5748664][fix] Increasing disagg acc test timeout (#10764) |
| 799 | 4ebc1b1596 | Ivy Zhang | 2026-01-19 | [None][test] Update test case for release (#10763) |
| 800 | 4df0ca8bd1 | ruodil | 2026-01-19 | [None][test] modify ctx config in 128k8k disagg cases (#10779) |
| 801 | af49fbdf65 | Emma Qiao | 2026-01-19 | [None][infra] Waive failed case for release branch on 01/19 (#10795) |
| 802 | 25bdc30162 | Yukun He | 2026-01-19 | [https://nvbugs/5782112][fix] Cherry-pick #10633: Fix hanging issue for MNNVL Allreduce under PP (#10750) |
| 803 | 2b3bb2e9b0 | Yuxian Qiu | 2026-01-19 | [https://nvbugs/5811697][fix] Fix buffer reuse. (#10716) |
| 804 | 4b833492fb | Emma Qiao | 2026-01-18 | [None][infra] Waive failed cases for release on 10/18 (#10781) |
| 805 | aa410c57bc | Faraz | 2026-01-16 | [TRTLLM-5366][chore] Add dgx-spark beta notes (#10766) |
| 806 | f02948d956 | Mike Iovine | 2026-01-16 | [https://nvbugs/5803813][fix] Fix llama 4 min latency (#10724) |
| 807 | 93e7ae73ea | Patrice Castonguay | 2026-01-16 | [None][doc] 1.2 Release Notes Headers (#10722) |
| 808 | 0c393ebc69 | TensorRT LLM | 2026-01-25 | [None][infra] Check in most recent lock file from nightly pipeline |
| 809 | d548b29a41 | Patrice Castonguay | 2026-01-24 | [None][fix] Bugfix/mtp with async scheduler (#10941) |
| 810 | 6f07fa81d7 | Yao Yao | 2026-01-24 | [TRTLLM-7738][feat] Adding implementation of KVCacheManagerV2 (#10736) |
| 811 | 9fcc93ea7b | Yuxian Qiu | 2026-01-24 | [https://nvbugs/5829097][fix] Re-init TRTLLM sampler to use sample stream in multi-stream cases. (#10918) |
| 812 | 9d65b8bf24 | Emma Qiao | 2026-01-24 | [None][infra] Fix TRT-LLM data scratch mount point for gb10x (#10880) |
| 813 | 78a008d61a | Yanchao Lu | 2026-01-24 | [None][ci] Remove long-running sanity check tests on GH200 (#10924) (#10969) |
| 814 | da967d0bd7 | Kaiyu Xie | 2026-01-24 | [TRTLLM-10334] [feat] Support overlap scheduler for disagg ctx instances (#10755) |
| 815 | 58dc4bea9c | TensorRT LLM | 2026-01-24 | [None][infra] Check in most recent lock file from nightly pipeline |
| 816 | cf88da7eca | jthomson04 | 2026-01-23 | [None][feat] KV Connector Support for MTP (#10932) |
| 817 | 1fbbb1f3cd | Taylor Yeonbok Lee | 2026-01-23 | [None][feat] AutoDeploy: Enhance memory consumption for MoE fusion transform (#10772) |
| 818 | b560598c79 | Jin Li | 2026-01-24 | [https://nvbugs/5707359][fix] Unwaive the test that due to flashinfer… (#10570) |
| 819 | f4b52d3b78 | yuanjingx87 | 2026-01-23 | [None][infra] Regenerate out dated lock file (#10940) |
| 820 | 1d68fab49c | Yihan Wang | 2026-01-24 | [https://nvbugs/5814215][fix] Unwaive test_trtllm_flashinfer_symbol_collision.py::test_flashinfer_fused_moe_matches_torch_moe (#10930) |
| 821 | 54768f3f2c | Yan Chunwei | 2026-01-23 | [None][chore] refine placement group in ray executor (#10235) |
| 822 | 43f2b51e94 | Yihan Wang | 2026-01-23 | [https://nvbugs/5833795][chore] Waive test test_e2e.py::test_ptp_quickstart_advanced[GPT-OSS-120B-gpt_oss/gpt-oss-120b] (#10953) |
| 823 | ae114ec7cf | Emma Qiao | 2026-01-23 | [None][infra] Waive a failed case in pre-merge stage (#10948) |
| 824 | 51c7a06da6 | zackyoray | 2026-01-23 | [None][feat] Upgrade NIXL to v0.9.0 (#10896) |
| 825 | 0f7192c7fe | Stanley Sun | 2026-01-23 | [None][test] Remove unused test list (#10916) |
| 826 | 31d04dfa12 | Leslie Fang | 2026-01-23 | [TRTLLM-9108][feat] Add test configurable moe module multi gpu (#10699) |
| 827 | ea928f62af | yuanjingx87 | 2026-01-22 | [None][infra] Update CI allowlist (#10936) |
| 828 | d793bd973d | Lucas Liebenwein | 2026-01-22 | [https://nvbugs/5688721][fix] unwaive NemotronH accuracy test (#10852) |
| 829 | 2146c23786 | William Zhang | 2026-01-22 | [#9306][refactor] Refactor AutoDeployConfig into LlmArgs (#10613) |
| 830 | d8e6e22060 | Grzegorz Kwasniewski | 2026-01-22 | [https://nvbugs/5819002][fix] fix sharding tests (#10775) |
| 831 | d43be7b65e | Yi Zhang | 2026-01-23 | [None][fix] Avoid Double update for previous batch (#9888) |
| 832 | 944c304bbb | Shi Xiaowei | 2026-01-23 | [TRTLLM-9527][feat] Python transceiver components (step 2) (#10494) |
| 833 | 9adef4eb28 | Shi Xiaowei | 2026-01-23 | [TRTLLM-9527][doc] Add NIXL as a Python attribution (step 4) (#10910) |
| 834 | b3146d095d | Venky | 2026-01-22 | [TRTC-122][feat] Eagle3 Specdec UX improvements (#10124) |
| 835 | 30ffa58b54 | Yan Chunwei | 2026-01-22 | [https://nvbugs/5783876][fix] fix hmac launch (#10434) |
| 836 | a218cf02fd | Bo Deng | 2026-01-22 | [https://nvbugs/5768068][chore] improve disagg acc tests (#10833) |
| 837 | 5e34112b27 | Pengyun Lin | 2026-01-22 | [TRTLLM-10388][feat] Support logprobs for Completions API (#10809) |
| 838 | 9beb971827 | 彭晋韬(jtao peng) | 2026-01-22 | [None][fix] Update RMSNorm custom op plumbing (#10843) |
| 839 | 1dc49b266e | Jiayu Chang | 2026-01-22 | [https://nvbugs/5322131][feat] Multi-LoRA serving with CUDA Graph (#8279) |
| 840 | cdb9ffd0ab | Yihan Wang | 2026-01-22 | [https://nvbugs/5741304][chore] Update flashinfer-python to 0.6.1 (#10872) |
| 841 | 128d4ac5be | tcherckez-nvidia | 2026-01-22 | [None][chore] NVFP4 MoE - Move weights transformation to fusion phase… (#10803) |
| 842 | 0243abee22 | Yiqing Yan | 2026-01-22 | [None][chore] Bump version to 1.3.0rc1 (#10923) |
| 843 | 0b3092e144 | Enwei Zhu | 2026-01-22 | [None][ci] Fix test list llm_spark_func.txt (#10921) |
| 844 | 6e72aff866 | tcherckez-nvidia | 2026-01-22 | [#10838][fix] Add missing dist strategy param. fix typo for ad_logger… (#10892) |
| 845 | 9ce0511d86 | Bo Li | 2026-01-22 | [https://nvbugs/5811159][fix] Unwaive bug 5811159. (#10903) |
| 846 | 9462d90ec7 | Pengbo Wang | 2026-01-22 | [None][feat] Add KV cache cleanup (#7439) |
| 847 | fd2af8d58a | shuyixiong | 2026-01-22 | [TRTLLM-9771][feat] Support partial update weight for fp8 (#10456) |
| 848 | ff0775408d | Wanli Jiang | 2026-01-22 | [None][fix] Fix waived tests for Nemotron-h models (#10758) |
| 849 | be4a431ffd | Enwei Zhu | 2026-01-22 | [TRTLLM-10154][feat] Enable guided decoding with reasoning parsers (#10890) |
| 850 | 895bb94b3d | Taylor Yeonbok Lee | 2026-01-21 | [#8241][feat] Support model_kwargs for pytorch backend (#10351) |
| 851 | 70caa779a4 | Yechan Kim | 2026-01-22 | [None][feat] K-EXAONE MTP support (#10796) |
| 852 | 415739711f | JennyLiu | 2026-01-22 | [None][chore] Add DGX-Spark VLM accuracy and perf spec dec cases (#10804) |
| 853 | f3a41c8d94 | Lizhi Zhou | 2026-01-22 | [TRTLLM-10059][feat] Use global unique id as disagg request id (#10187) |
| 854 | 0434db5bf7 | Daniil | 2026-01-21 | [None][feat] GLM-4.5-Air support (#10653) |
| 855 | bd56b4e1e3 | TensorRT LLM | 2026-01-22 | [None][infra] Check in most recent lock file from nightly pipeline |
| 856 | c2a9e66dff | Yuxian Qiu | 2026-01-22 | [https://nvbugs/5784543][chore] unwaive test. (#10835) |
| 857 | 635cbf01ba | dongxuy04 | 2026-01-21 | [https://nvbugs/5816267][fix] Remove weight tensor holder to release memory earlier (#10876) |
| 858 | 5450485bec | yuanjingx87 | 2026-01-21 | [None][infra] Fix sonarQube job hang by create jenkins homd folder if not exist (#10830) |
| 859 | 8cf8fbbe16 | Guiju Zhang | 2026-01-21 | [TRTLLM-10325][feat] Refactor speculative decoding workers (#10768) |
| 860 | f91ea37a13 | kris1025 | 2026-01-21 | [None][chore] unwaive qwen3 235B accuracy test (#10493) |
| 861 | bf7303c7f1 | Yukun He | 2026-01-21 | [https://nvbugs/5636916][fix] Cherry-pick #10654: Fix accuracy issue of TWO-SHOT AllReduce kernel (#10841) |
| 862 | 165dd360b9 | Emma Qiao | 2026-01-21 | [None][infra] Waive failed cases for main branch on 01/21 (#10882) |
| 863 | 9feebb3a27 | xxi | 2026-01-21 | [None][chore] switch to ConfigurableMoE as the default path (#10792) |
| 864 | a4152c80f6 | Yukun He | 2026-01-21 | [https://nvbugs/5814253][fix] unwaive test_autotuner_distributed_strategy tests (#10793) |
| 865 | 1592dfab6d | HuiGao-NV | 2026-01-21 | [https://nvbugs/5740377][fix] Lock resource to fix potential access to released data (#10827) |
| 866 | d60d6ff6fd | Yukun He | 2026-01-21 | [None][fix] Cherry-pick #10715: Disable short profile for tunable ops with MERGE strategy (#10844) |
| 867 | 87073d1ce4 | Xianjie Qiao | 2026-01-21 | [None][fix] Fix copy start_logs in disagg slurm scripts (#10840) |
| 868 | 9116dfbacd | Yibin Li | 2026-01-20 | [https://nvbugs/5775021] [fix] Replace pickle.load with restricted Unpickler (#10622) |
| 869 | ffd2ed51dd | TensorRT LLM | 2026-01-21 | [None][infra] Check in most recent lock file from nightly pipeline |
| 870 | ccf4d79c6c | Yanchao Lu | 2026-01-21 | [None][chore] Revert NVIDIA/TensorRT-LLM#10847 (#10869) |
| 871 | c381790d15 | shuyixiong | 2026-01-21 | [https://nvbugs/5670458][chore] Unwaive reward model test (#10831) |
| 872 | 2f3b2a3172 | Daniel Stokes | 2026-01-21 | [None][fix] Add a timeout in MNNVL throughput to prevent hangs if one rank crashes (#9532) |
| 873 | 3c39b1faa9 | Yan Chunwei | 2026-01-21 | [https://nvbugs/5759698][fix] unwaive test_base_worker (#10669) |
| 874 | 26c23cf99f | Zheng Duan | 2026-01-21 | [https://nvbugs/5760737][test] only skip mooncake+indexerkcache test (#10266) |
| 875 | 3c8ed19440 | Simeng Liu | 2026-01-20 | [https://nvbugs/5670108][fix] Fix overlap scheduler race condition in… (#10610) |
| 876 | c6163e2b70 | TensorRT LLM | 2026-01-20 | [None][infra] Check in most recent lock file from nightly pipeline |
| 877 | 864b61cadd | Izzy Putterman | 2026-01-20 | [None][feat] Speculative One Model: FlashInfer sampling (#10284) |
| 878 | 66b239a9a9 | Lucas Liebenwein | 2026-01-20 | [None][fix] fix duplicate entry in waives.txt (#10853) |
| 879 | 2db3d7eeba | jthomson04 | 2026-01-20 | [None][chore] Async Transfer Manager (#9891) |
| 880 | e61c942d1f | Gal Hubara-Agam | 2026-01-20 | [#10707][fix] AutoDeploy: Super accuracy test fixes (#10717) |
| 881 | ae8f74b620 | Yanchao Lu | 2026-01-20 | [None][chore] Reduce tedious logs (#10847) |
| 882 | 3a894951e7 | Emma Qiao | 2026-01-20 | [None][infra] Waive failed cases for main branch on 01/20 (#10829) |
| 883 | 338b29d5ae | Bo Deng | 2026-01-20 | [None][infra] trigger multi-gpu tests when install_nixl/ucx.sh is mod… (#10624) |
| 884 | c8a200486d | Yuxian Qiu | 2026-01-20 | [https://nvbugs/5701445][chore] unwaive test. (#10806) |
| 885 | eb326073d8 | Grzegorz Kwasniewski | 2026-01-20 | [TRTLLM-10785][feat] Fix sharding dashboard errors (#10786) |
| 886 | 58311b2345 | Yi Zhang | 2026-01-20 | [None][fix] Remove unused params in attn (#10652) |
| 887 | 47e0ec2527 | xinhe-nv | 2026-01-20 | [None][test] Update sanity test list (#10825) |
| 888 | 99e8cb0999 | Yiqing Yan | 2026-01-20 | [None][fix] Fix vulnerability urllib3 and nbconvert (#10551) |
| 889 | fc467d06c3 | xinhe-nv | 2026-01-20 | [TRTLLM-8638][fix] Add failed cases into waives.txt (#10787) |
| 890 | 4c8468c5d3 | benzh-2025 | 2026-01-20 | [None][fix] default disable gemm+allreduce fusion (#10656) |
| 891 | 26bc16842e | xinhe-nv | 2026-01-20 | [None][chore] Add failed cases into waives.txt (#10776) |
| 892 | 44c5af88dc | TensorRT LLM | 2026-01-20 | [None][infra] Check in most recent lock file from nightly pipeline |
| 893 | f3a985ce27 | Bo Li | 2026-01-20 | [TRTLLM-10296][fix] Fix the potential misaligned access due to vectorized ld/st instructions in NVLinkOneSided A2A. (#10539) |
| 894 | dbb858ae0c | Liao Lanyu | 2026-01-20 | [TRTLLM-10029][scheduler] Re-implement MicroBatchScheduler and CapacityScheduler in Python (#10273) |
| 895 | c6320d924d | Lizhi Zhou | 2026-01-20 | [https://nvbugs/5776445][chore] unwaive test (#10667) |
| 896 | 066fa4cd93 | Zhenhuan Chen | 2026-01-20 | [None][chore] update config.yaml of slurm scripts to align with submit.py change (#10802) |
| 897 | ed95e70150 | Jie Li | 2026-01-19 | [None][chore] Remove trt flow tests in NIM (#10731) |
| 898 | 64ff5cac52 | SamareshSingh | 2026-01-19 | [None][chore] docs: clarify LoRA is not supported with --use_fp8_rowwise in Fp8RowwiseAttention (see #2603) (#10320) |
| 899 | 442d2e8a15 | Shi Xiaowei | 2026-01-19 | [None][test] adjust the dis-agg test timeout threshold (#10800) |
| 900 | cc0bbde745 | Xianjie Qiao | 2026-01-19 | [None][feat] Update disagg slurm scripts (#10712) |
| 901 | 32ab809f36 | Eran Geva | 2026-01-19 | [#10607][chore] Add Nemotron Nano v3 FP8 autodeploy perf test (#10603) |
| 902 | baa250d1d6 | TensorRT LLM | 2026-01-19 | [None][infra] Check in most recent lock file from nightly pipeline |
| 903 | 935c174283 | Emma Qiao | 2026-01-19 | [None][infra] Waive failed cases for main on 01/19 (#10794) |
| 904 | df845a028b | Zhanrui Sun | 2026-01-19 | [TRTLLM-9581][infra] Use /home/scratch.trt_llm_data_ci in computelab (#10616) |
| 905 | 68ab1a47c4 | Yiqing Yan | 2026-01-19 | [None][chore] Add release/1.2 branch into lockfile generation schedule (#10790) |
| 906 | e97af45556 | chenfeiz0326 | 2026-01-19 | [TRTLLM-10300][feat] Upload regression info to artifactory (#10599) |
| 907 | a6a63f5a36 | Lucas Liebenwein | 2026-01-18 | [https://nvbugs/5814247][fix] unwaive AutoDeploy multi-gpu unit tests (#10769) |
| 908 | 4f04532ce7 | Chuang Zhu | 2026-01-19 | [https://nvbugs/5769890][fix] enable system memory to transfer active message in NIXL ucx (#10602) |
| 909 | 9879400479 | Lucas Liebenwein | 2026-01-18 | [#10642][feat] AutoDeploy: optimized canonicalize_graph utilities [1/2] (#10675) |
| 910 | 4d2916d683 | Eran Geva | 2026-01-18 | [#10688][fix] AutoDeploy Fix CUDA graph batch sizes exceeding max_batch_size (#10687) |
| 911 | b64052539d | Lucas Liebenwein | 2026-01-18 | [https://nvbugs/5769712][fix] fix timeout in AutoDeploy llama accuracy test (#10461) |
| 912 | 3aaed62cfc | TensorRT LLM | 2026-01-18 | [None][infra] Check in most recent lock file from nightly pipeline |
| 913 | e1cc8d2337 | yuanjingx87 | 2026-01-18 | [None][infra] Add sonarqube scanning in lockfile generation pipeline (#10700) |
| 914 | a11f0dbd61 | Eran Geva | 2026-01-18 | [#10696][fix] AutoDeploy prevent torch.export from specializing batch dimension when max_batch_size=1 (#10697) |
| 915 | 0af1a0e478 | Yanchao Lu | 2026-01-18 | [None][test] Waive main post-merge test failures 1/18 (#10777) |
| 916 | f8c26409f9 | TensorRT LLM | 2026-01-18 | [None][infra] Check in most recent lock file from nightly pipeline |
| 917 | 0096b50ba0 | Yanchao Lu | 2026-01-18 | [None][infra] Update upgrade related docs for release 1.2 (#10760) (#10773) |
| 918 | 7bf4dd9f63 | Grzegorz Kwasniewski | 2026-01-17 | [TRTLLM-10318][feat] Fixing Nemotron sharding: support for sharding buffers (#10319) |
| 919 | cef67b4f8d | Yuxian Qiu | 2026-01-17 | [None][fix] convert to CUDA tensor before calling _resmooth_kernel. (#10770) |
| 920 | b65560fc32 | Yuxian Qiu | 2026-01-17 | [https://nvbugs/5794313][chore] unwaive tests. (#10660) |
| 921 | 3d16daf696 | Yukun He | 2026-01-17 | [None][fix] Fix tmp dir being deleted too early in unit test. (#10740) |
| 922 | 56073f501a | chenfeiz0326 | 2026-01-17 | [TRTLLM-8263][feat] Add Aggregated Perf Tests (#10598) |
| 923 | 24d7e499b4 | TensorRT LLM | 2026-01-17 | [None][infra] Check in most recent lock file from nightly pipeline |
| 924 | 069ad68d3c | Frida Hou | 2026-01-16 | [None][fix] AutoDeploy: skip mxfp4_moe test unless on Hopper (#10729) |
| 925 | 0b748d5bba | Chenghao Zhang | 2026-01-16 | [None][chore] update flashinfer to 0.6.0 (#10522) |
| 926 | b6acd96616 | Chenghao Zhang | 2026-01-16 | [None][fix] AutoDeploy: Fix the nvfp4 fused_moe (#10727) |
| 927 | 0cfd08745c | Stefan Niebler | 2026-01-16 | [TRTLLM-9735][feat] Add processed logprobs functionality to TorchSampler (#9675) |
| 928 | cfebfbb505 | Tian Zheng | 2026-01-16 | [https://nvbugs/5783509][fix] Fix a hang issue when enabling skip softmax on Blackwell (#10490) |
| 929 | cc43edc8f4 | xinhe-nv | 2026-01-16 | [None][fix] waive tests on sm89 (#10753) |
| 930 | c4db030b88 | Stefan Niebler | 2026-01-16 | [TRTLLM-8425][doc] Update sampling documentation (#10083) |
| 931 | 722978b837 | Wanli Jiang | 2026-01-16 | [TRTLLM-10305][feat] Support customized seq len larger than model config (#10600) |
| 932 | 4f86c5f5ce | Kaiyu Xie | 2026-01-16 | [None] [feat] Support multiple accuracy tasks for slurm scripts (#10500) |
| 933 | 6dfb8d7084 | dongfengy | 2026-01-16 | [None][fix] Fix Piecewise Cuda Graph for GPTOSS (#10631) |
| 934 | 0256c7234f | xinhe-nv | 2026-01-16 | [None][chore] Remove closed bugs (#10586) |
| 935 | b163e66182 | jmydurant | 2026-01-16 | [None][doc] update doc (add minimax model) (#10746) |
| 936 | 03cdf5804f | Necofish | 2026-01-16 | [None][fix] impl fused triton kernel for e8m0 resmooth to reduce memory footprint (#10327) |
| 937 | f001c4946d | Yukun He | 2026-01-16 | [https://nvbugs/5782112][fix] Fix hanging issue for MNNVL Allreduce under PP (#10633) |
| 938 | e2c3373749 | Emma Qiao | 2026-01-16 | [None][infra] Waive failed cases for main branch on 01/16 (#10738) |
| 939 | 7686fbbcbe | Bo Li | 2026-01-16 | [https://nvbugs/5810940][chore] Update waive lists for nvbugs/5810940. (#10737) |
| 940 | 8257b67ea5 | Chuang Zhu | 2026-01-16 | [https://nvbugs/5791936][fix] Add warning for gen-only paused (#10664) |
| 941 | 6541e41c74 | TensorRT LLM | 2026-01-16 | [None][infra] Check in most recent lock file from nightly pipeline |
| 942 | 7b8b9ccbaf | Enwei Zhu | 2026-01-16 | [https://nvbugs/5669671][fix] Support GuidedDecoder with sharded logits (#10698) |
| 943 | 9f741fb254 | Enwei Zhu | 2026-01-16 | [https://nvbugs/5800521][ci] Move test_openai_chat_guided_decoding to H100 stage (to avoid potential OOM) (#10703) |
| 944 | ce561b6a8e | xxi | 2026-01-16 | [TRTLLM-9111][feat] MoE test refactor: Extend MoE quantization test utilities with comprehensive quant algorithm support (#10691) |
| 945 | 7e2cbc0756 | Chuang Zhu | 2026-01-16 | [https://nvbugs/5598674][fix] enable partial reuse in gemma and gpt oss test (#10559) |
| 946 | e3f27e06c7 | heyuhhh | 2026-01-16 | [None][chore] Waive star attention unittests (#10439) |
| 947 | ef838cc852 | Yuxian Qiu | 2026-01-16 | [https://nvbugs/5701445][chore] isolate test. (#10444) |
| 948 | 49c6f73554 | Lucas Liebenwein | 2026-01-15 | [None][bug] AutoDeploy: fix regression in kv cache resize memory estimation (#10726) |
| 949 | 5ad8cf6d5e | Iman Tabrizian | 2026-01-15 | [https://nvbugs/5738168][fix] unwaive test accuracy/test_disaggregated_serving.py::TestDeepSeekV32Exp::test_auto_dtype[False] (#10584) |
| 950 | 0998a7bf20 | Thor Johnsen | 2026-01-15 | [https://nvbugs/5721661][fix] Prevent out-of-bounds read (#9879) |
| 951 | dfac07c045 | heyuhhh | 2026-01-15 | [None][feat] Support to export data in trtllm-eval (#10075) |
| 952 | 43b9db3364 | forrestl | 2026-01-15 | [None][doc] doc updates (#10711) |
| 953 | 93db0d5e18 | Lizhi Zhou | 2026-01-15 | [TRTLLM-9942][feat] new request states and kvcache transceiver APIs in generation-first disagg (#10406) |
| 954 | 3bc17e1aa3 | Jun Yang | 2026-01-15 | [None][doc] doc updates (#10704) |
| 955 | ff277b591e | Lizhi Zhou | 2026-01-15 | [https://nvbugs/5791830][fix] fix pp loop hang caused by i-sending new requests (#10665) |
| 956 | cd55fb4551 | yufeiwu-nv | 2026-01-15 | [None][test] Remove NIM test (#10657) |
| 957 | 683515b1bd | Pengbo Wang | 2026-01-15 | [None][feat] Use XQA JIT impl by default and mitigate perf loss with sliding window (#10335) |
| 958 | 71ccc07d2b | Perkz Zheng | 2026-01-15 | [None][feat] update trtllm-gen to support groupsTokensHeadsQ (#10261) |
| 959 | e12a7119cf | Ludwig Schneider | 2026-01-15 | [https://nvbugs/5741392][fix] [chore] Remove test exemptions from waivers tile (#10517) |
| 960 | f4ace99218 | Yiqing Yan | 2026-01-15 | [None][chore] Bump version to 1.3.0rc0 (#10681) |
| 961 | 22240e43eb | ruodil | 2026-01-15 | [None][test] store per user output and per gpu output metric in csv file (#10658) |
| 962 | 7b3b6f1161 | Emma Qiao | 2026-01-15 | [None][infra] Waive failed tests on main 01/15 (#10683) |
| 963 | faa80e73fd | Anish Shanbhag | 2026-01-14 | [None][feat] Auto download speculative models from HF for pytorch backend, add speculative_model field alias (#10099) |
| 964 | 62050b2381 | Lucas Liebenwein | 2026-01-14 | [None][infra] separate AutoDeploy tests into own stages (#10634) |
| 965 | f7de285a82 | Void | 2026-01-15 | [None][fix] add quantization check for DeepEP LL low precision combine in new moe comm api (#10072) |
| 966 | 482b7b8837 | TensorRT LLM | 2026-01-15 | [None][infra] Check in most recent lock file from nightly pipeline |
| 967 | 15b43e8a14 | Lucas Liebenwein | 2026-01-14 | [https://nvbugs/5777041][fix] fix AutoDeploy ep sharding test (#10460) |
| 968 | 94c7b69048 | Dom Brown | 2026-01-15 | [https://nvbugs/5630196] [fix] Prevent flaky failures in C++ test_e2e.py by using local cached datasets for benchmarking (#10638) |
| 969 | 73d1840c12 | Wanli Jiang | 2026-01-15 | [TRTLLM-10245][feat] Add accuracy tests for super v3 fp8 model (#10482) |
| 970 | 0f2d61b8c6 | dominicshanshan | 2026-01-15 | [https://nvbugs/5766952][fix] Fix AIPerf issue. (#10666) |
| 971 | 5f9fc50233 | bhsueh_NV | 2026-01-15 | [https://nvbugs/5800725][infra] Update waives.txt (#10625) |
| 972 | 211c44b951 | 彭晋韬(jtao peng) | 2026-01-15 | [None][feat] Adding torch ext API for FusedAddRMSNormQuant kernel (#9905) |
| 973 | 968db53194 | TensorRT LLM | 2026-01-14 | [None][infra] Check in most recent lock file from nightly pipeline |
| 974 | c99faaed06 | Tzu-Ling Kan | 2026-01-14 | [#9760][fix] Use RequestError for validation errors to prevent engine shutdown (#9761) |
| 975 | 01083b56bf | Emma Qiao | 2026-01-14 | [TRTLLM-9849][infra] Update dependencies to 25.12 (#9818) |
| 976 | 35c24424f6 | Emma Qiao | 2026-01-14 | [None][infra] Waive failed cases in post-merge on 01/14 (#10668) |
| 977 | b10704428d | HuiGao-NV | 2026-01-14 | [https://nvbugs/5787566][fix] Only keep a limited number of performance statistic data (#10569) |
| 978 | 582dec5bb5 | Bo Li | 2026-01-14 | [https://nvbugs/5774869][infra] Use 2 GPUs to test skip softmax attention on H100. (#10420) |
| 979 | babd5ecacc | shuyixiong | 2026-01-14 | [https://nvbugs/5760740][fix] Enable ray tests (#10272) |
| 980 | 25148d3fee | Kyungmin Lee | 2026-01-14 | [None][feat] Support new Transformers RoPE configuration format (#10636) |
| 981 | e9817461ba | xxi | 2026-01-14 | [None][chore] improve the readability of log for cutlass can only sup… (#10630) |
| 982 | d8862505b9 | xxi | 2026-01-14 | [None][chore] enable EPLB for DEEPGEMM (#10617) |
| 983 | 272688c663 | xinhe-nv | 2026-01-14 | [None][fix] fix L0 issues (#10670) |
| 984 | e7882d5c74 | jmydurant | 2026-01-14 | [None][feat] MiniMax M2 support (#10532) |
| 985 | 052c36ddd2 | mpikulski | 2026-01-14 | [TRTLLM-9522][feat] support image_embeds in OpenAI API (#9715) |
| 986 | 487287a412 | Bo Li | 2026-01-14 | [None][chore] Update test name MNNVL->NVLinkTwoSided. (#9672) |
| 987 | 287f6c2e0f | Zhenhuan Chen | 2026-01-14 | [None][test] add log_samples and output_path for trtllm_eval (#10629) |
| 988 | c4da4fd462 | QI JUN | 2026-01-14 | [https://nvbugs/5637220][ci] unwaive TestQwen3_235B_A22B::test_nvfp4[latency_moe_trtllm_attention_dp] (#9870) |
| 989 | 15281de799 | Yukun He | 2026-01-14 | [None][fix] Reduce host overhead for unified nvfp4 gemm tuning path. (#10503) |
| 990 | 39cefd6125 | Yuxian Qiu | 2026-01-14 | [None][refactor] Unify the usage of MPIDist and TorchDist. (#10380) |
| 991 | f841b43cde | xxi | 2026-01-14 | [None][chore] waive the CI failure (#10655) |
| 992 | 92ae490410 | JennyLiu | 2026-01-14 | [None][test] Spark - Change testlist name and perf yml format (#10626) |
| 993 | 07d9390e9b | xinhe-nv | 2026-01-14 | [None][test] add test into qa test list (#10627) |
| 994 | b65c515314 | tburt-nv | 2026-01-13 | [None][chore] update allowlist 2026-01-13 (#10645) |
| 995 | dd22324675 | TensorRT LLM | 2026-01-14 | [None][infra] Check in most recent lock file from nightly pipeline |
| 996 | 7305c61fc9 | xinhe-nv | 2026-01-14 | [TRTLLM-8638][fix] Add failed cases into waives.txt (#10589) |
| 997 | 795e690bca | Leslie Fang | 2026-01-14 | [https://nvbugs/5753788][chore] Padding empty chunk for configurable moe (#10451) |
| 998 | d3f4fbb742 | Yuxian Qiu | 2026-01-14 | [None][fix] Avoid write-write race for async pp send. (#10488) |
| 999 | 2acd03030a | Yuxian Qiu | 2026-01-14 | [https://nvbugs/5781589][fix] Implement pp skip forward for all spec workers. (#10578) |
| 1000 | bc119f5644 | Leslie Fang | 2026-01-14 | [None][chore] Add test configurable moe module (#10575) |
