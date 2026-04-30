# Commit Section 4

Commits 1501 to 2000 (total: 5612)

| # | Commit Hash | Author | Date | Message |
|---|------------|--------|------|--------|
| 1501 | 95d928f071 | Kanghwan | 2025-12-11 | [None][infra] Add workflow to auto-label 'waiting for feedback' on team comments (#9886) |
| 1502 | fd1270b9ab | Venky | 2025-12-11 | [TRTC-43] [feat] Add config db and docs (#9420) |
| 1503 | 24f92721f2 | Simeng Liu | 2025-12-11 | [https://nvbugs/5597647][ci] Unwaive fixed tests. (#9812) |
| 1504 | 89dabf5aa1 | Erin | 2025-12-11 | [TRTLLM-9736][feat] AsyncLLM and verl integ (#9353) |
| 1505 | 02edb19f43 | JadoTu | 2025-12-12 | [None] [feat] add eos_token_id in generation_config to sampling params (#9514) |
| 1506 | 488d38f88d | xxi | 2025-12-12 | [TRTLLM-8959][feat] ConfigurableMoE support CUTLASS (#9772) |
| 1507 | af2849cc7a | Fanrong Li | 2025-12-11 | [None][doc] Add DeepSeek-V3.2 to the supported models (#9893) |
| 1508 | 04a39a4e2b | Yan Chunwei | 2025-12-11 | [None][chore] enable test_ipc.py (#9865) |
| 1509 | c76b428e2e | Zongfei Jing | 2025-12-11 | [TRTLLM-9685] [feat] Add gather fc1 kernel by cuteDSL (#9618) |
| 1510 | b8a5159fad | ChristinaZ | 2025-12-11 | [None][feat] Enable PDL for indexer topK (#9843) |
| 1511 | d147ad053e | Kanghwan | 2025-12-10 | [#2730][fix] Fix circular import bug in medusa/weight.py (#9866) |
| 1512 | 454e7e59e5 | JunyiXu-nv | 2025-12-11 | [https://nvbugs/5718004][fix] Add warmup for cancellation test (#9860) |
| 1513 | 81222c3670 | Ziyi Xiong | 2025-12-11 | [None] Fix warning when capturing CUDA graph (#9746) |
| 1514 | c1d53ee43d | Bo Deng | 2025-12-11 | [https://nvbugs/5582258][fix] unwaive (#9650) |
| 1515 | 341cb1a12c | fredricz-20070104 | 2025-12-11 | [None][chore] Add GB300 support since it does not support segment (#9731) |
| 1516 | 2c0293c612 | Patrice Castonguay | 2025-12-10 | [https://nvbugs/5601682][fix] Unwaiving disagg test (#9627) |
| 1517 | ece3a8748f | Tian Zheng | 2025-12-10 | [None][doc] Update doc for NVFP4 KV cache (#9475) |
| 1518 | 2f030312a8 | cheshirekow | 2025-12-10 | [TRTLLM-9228][infra] Verify thirdparty C++ process (#9367) |
| 1519 | 1c11cae54d | Yiqing Yan | 2025-12-10 | [None][chore] bump version to 1.2.0rc6 (#9874) |
| 1520 | 072f236002 | Yukun He | 2025-12-10 | [None][fix] Fully resolve the tactic recovery issues in AutoTuner serialized cache (#9835) |
| 1521 | df1adfbb50 | Matt Lefebvre | 2025-12-10 | [TRTINFRA-7328][infra] - Move half B200 tests to lbd (#9853) |
| 1522 | 8cec2da375 | Brian K. Ryu | 2025-12-10 | [None][feat] Port fp4 quantization kernel optimization from FlashInfer (#9854) |
| 1523 | 8fefa2c9d1 | Matt Lefebvre | 2025-12-10 | [None][infra] Fail fast if SLURM entrypoint fails (#9744) |
| 1524 | e34302986d | Perkz Zheng | 2025-12-10 | [https://nvbugs/5727952][fix] PDL bugs with trtllm-gen fmha kernels (#9863) |
| 1525 | 12693a526b | Guoming Zhang | 2025-12-10 | [None][chore] Enable L0 multi-gpus testing for Qwen3-next (#9789) |
| 1526 | 49fe089470 | Zhanrui Sun | 2025-12-10 | [TRTLLM-9811][infra] Update urllib3 version >= 2.6.0 to fix high vulnerability issue (#9823) |
| 1527 | 0e78a4b244 | dominicshanshan | 2025-12-10 | [https://nvbugs/5702791][fix] Unwaive fixed test (#9844) |
| 1528 | 979f37e443 | Yukun He | 2025-12-10 | [None][fix] Fix nvfp4 gemm allowed backends arg passing (#9837) |
| 1529 | 2c46126a93 | QI JUN | 2025-12-10 | [TRTLLM-9794][ci] move some deepseek test cases to gb200 (#9841) |
| 1530 | 9d3c675a0b | Bo Li | 2025-12-10 | [None][chore] Support larger topK for NVLinkOneSided AlltoAll. (#9816) |
| 1531 | 6a39bb983c | TensorRT LLM | 2025-12-10 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1532 | 36c9e7cfe6 | zhanghaotong | 2025-12-10 | [None][chore] Add unittest for otlp tracing (#8716) |
| 1533 | 2d33ae94d5 | dhansen-nvidia | 2025-12-09 | [https://nvbugs/5508301][feat] Move D->H copies to a worker thread whe… (#8463) |
| 1534 | 414448bb37 | Patrice Castonguay | 2025-12-09 | [https://nvbugs/5719561][chore] Unwaive tests for nvbug 5719561 (#9801) |
| 1535 | ff0ef19ee9 | Patrice Castonguay | 2025-12-09 | [https://nvbugs/5688388][chore] Unwaiving fixed disagg test (#9800) |
| 1536 | 5de4e3f621 | Matt Lefebvre | 2025-12-09 | [TRTINFRA-7328][infra] Consume SlurmCluster scratchPath and cleanup mounts (#9600) |
| 1537 | 4da3121363 | Eran Geva | 2025-12-09 | [#8921][chore] AutoDeploy NanoV3 to use SYMM_MEM allreduce strategy (#9797) |
| 1538 | 7d7d05d8db | Patrice Castonguay | 2025-12-09 | [None][chore] Adding flaky auto scaling test to waives (#9851) |
| 1539 | 07c76a5fac | Mike Iovine | 2025-12-09 | [None][feat] Make 2-model spec dec use the 1-model kernels (Hopper) (#8810) |
| 1540 | 3156f2e852 | Dom Brown | 2025-12-09 | [https://nvbugs/5575841] [fix] Nvbug 5575841: Remove additional test waivers for TestMoEFP4 (#9788) |
| 1541 | 75bc386b65 | Emma Qiao | 2025-12-09 | [None][infra] Waive failed cases for main branch on 12/09 (#9839) |
| 1542 | 58c29957d9 | QI JUN | 2025-12-09 | [TRTLLM-9794][ci] move qwen3-next test cases to gb200 (#9827) |
| 1543 | d600b9f851 | Stefan Niebler | 2025-12-09 | [TRTLLM-6756][feat] Update BeamSearch for TorchSampler (#9660) |
| 1544 | 76f49c903b | Robin Kobus | 2025-12-09 | [None][fix] Additional model outputs for pipeline parallelism (#9794) |
| 1545 | 2ddcb45b2a | Yiqing Yan | 2025-12-09 | [None][chore] Generate lock file for release/1.2.0rc4.post1 branch automatically (#9829) |
| 1546 | fbcf03040f | yufeiwu-nv | 2025-12-09 | [None][test] Refactor qa/llm_perf_nim.yml test list (#9700) |
| 1547 | 252769c930 | QI JUN | 2025-12-09 | [TRTLLM-9794][ci] remove duplicated test cases in DGX B200 (#9817) |
| 1548 | 309f92ec09 | Zhanrui Sun | 2025-12-09 | [None][infra] Use artifactory pypi mirror for Cython install (#9774) |
| 1549 | b050804b63 | Shi Xiaowei | 2025-12-09 | [TRTLLM-6537][infra] extend multi-gpu tests related file list (#9614) |
| 1550 | 90890785eb | JunyiXu-nv | 2025-12-09 | [https://nvbugs/5722653][fix] Fix config file used by disagg_client (#9783) |
| 1551 | bafb60c1bc | Balaram Buddharaju | 2025-12-08 | [None][chore] Fix tests failing on pre-merge 12/08 (#9819) |
| 1552 | f2006a1f74 | Bo Li | 2025-12-09 | [https://nvbugs/5726066][infra] Waive timeout disaggregated/test_auto_scaling tests. (#9815) |
| 1553 | c7a2568872 | TensorRT LLM | 2025-12-09 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1554 | f521f6d910 | JunyiXu-nv | 2025-12-09 | [None][fix] Fix unterminated process issue for RemoteOpenAIServer (#9490) |
| 1555 | 4a3a66b124 | Jiagan Cheng | 2025-12-09 | [https://nvbugs/5677746][fix] Use first PP rank's schedule result in other PP ranks to fix PP hang (#9659) |
| 1556 | d6f961d3fe | bhsueh_NV | 2025-12-09 | [None][feat] Add llama4 scaling (#9771) |
| 1557 | 1c4dacb19a | Tri Dao | 2025-12-08 | [None][fix] Fix PDL in TRTLLM MOE for dsv3 (#9799) |
| 1558 | 390391ebf1 | yuanjingx87 | 2025-12-08 | [None][infra] Correct the waived test names due to a merge conflict (#9803) |
| 1559 | 75f5446d67 | Chenghao Zhang | 2025-12-08 | [#9753][feat] AutoDeploy: Implement add rms_norm fusion (#9754) |
| 1560 | da074be037 | Jhao-Ting Chen | 2025-12-08 | [None][fix] Fix #8383 introduced TRTLLM backend python error (#9804) |
| 1561 | 23cf72b0f8 | Eran Geva | 2025-12-08 | [#8921][feat] Added symetric memory AllReduce strategy (#8919) |
| 1562 | f9380581c5 | Thor Johnsen | 2025-12-08 | [https://nvbugs/5508267][fix] Proper handling of inactive canceled requests (#9280) |
| 1563 | faabc1a387 | Yibin Li | 2025-12-08 | [TRTLLM-7967][chore] Add more tests (#9415) |
| 1564 | 0a09465089 | Jhao-Ting Chen | 2025-12-08 | [https://nvbugs/5567586][feat] Ampere xqa swa specdec for GPT-OSS Eagle3-one-model (#8383) |
| 1565 | f6df9eb2a6 | Frank | 2025-12-08 | [TRTLLM-9089][chore] Port prepare_dataset into trtllm-bench (#9250) |
| 1566 | 1c7b7cdd47 | sunnyqgg | 2025-12-08 | [TRTLLM-9506][fix] Fix AR for DeepSeek-R1 2 model path (#9661) |
| 1567 | 98db262a67 | Eran Geva | 2025-12-08 | [None][fix] Switch AutoDeploy's default allreduce strategy to NCCL (#9666) |
| 1568 | 52f78e4000 | Lizhi Zhou | 2025-12-08 | [http://nvbugs/5649010][fix] fix test_auto_scaling.py::test_worker_restart timeout (#9775) |
| 1569 | 96d9b67d65 | fredricz-20070104 | 2025-12-08 | [https://nvbugs/5527655][test] Add test case for RCCA 5527655 (#9511) |
| 1570 | ededeecb0f | fredricz-20070104 | 2025-12-08 | [None][test] Add Kimi k2 WIDEEP perf and accuracy cases (#9686) |
| 1571 | e7395c6607 | Zheng Duan | 2025-12-08 | [None][infra] update mooncake in docker images (#9584) |
| 1572 | 3f55c07223 | xinhe-nv | 2025-12-08 | [None][chore] Remove closed bugs (#9770) |
| 1573 | 448bb1a44f | Guoming Zhang | 2025-12-08 | [TRTLLM-9431][perf] Enable multistream for Linear Attention in Qwen3-… (#9696) |
| 1574 | a422d70be6 | Li Min | 2025-12-08 | [None][chore] Enable tvm_ffi for cute dsl nvfp4_gemm to reduce host overhead. (#9690) |
| 1575 | 2f526583fb | Fanrong Li | 2025-12-08 | [None][chore] Move the rocketkv e2e test to post-merge (#9768) |
| 1576 | 137713a869 | Emma Qiao | 2025-12-08 | [None][infra] Waive failed cases for main on 12/08 (#9773) |
| 1577 | d232709568 | ruodil | 2025-12-08 | [https://nvbugs/5666804][test] only adding sampler config for limited models (#9512) |
| 1578 | 069b05cf3d | Kaiyu Xie | 2025-12-08 | [TRTLLM-9706] [doc] Update wide EP documents (#9724) |
| 1579 | 03f89d7aa4 | TensorRT LLM | 2025-12-08 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1580 | 8b9ab9a701 | Yukun He | 2025-12-08 | [None][fix] Fix two tuning cache miss issues. (#9743) |
| 1581 | 9bfb6179ec | fredricz-20070104 | 2025-12-08 | [https://nvbugs/5422621][test] Add GB 200 WIDEEP test case for RCCA 5422621 (#9506) |
| 1582 | 8e27ce7084 | xxi | 2025-12-08 | [TRTLLM-9603][feat] Enable ConfigurableMoE test in the CI (#9645) |
| 1583 | 4da0e1473c | Zheng Duan | 2025-12-08 | [None][test] add ntp tolerance in time metrics verification (#9741) |
| 1584 | 383178c00a | chenfeiz0326 | 2025-12-08 | [TRTLLM-9000][feat] Add multi-node Perf Tests into CI (#8800) |
| 1585 | 41ce14ab04 | Ludwig Schneider | 2025-12-07 | [None][feat] Enable NCCL_SYMMETRIC as default fallback for AllReduce (#9314) |
| 1586 | d252101a76 | Chenjie Luo | 2025-12-07 | [OMNIML-3036][doc] Re-branding TensorRT-Model-Optimizer as Nvidia Model-Optimizer (#9679) |
| 1587 | f59d64e6c7 | Yanchao Lu | 2025-12-07 | [None][fix] Several minor fixes to CI setting (#9765) |
| 1588 | 7c6c493993 | Emma Qiao | 2025-12-07 | [None][infra] Waive failed cases for main branch on 12/07 (#9769) |
| 1589 | b210f22c7e | JunyiXu-nv | 2025-12-07 | [https://nvbugs/5703953][fix] Preserving ip:port for trtllm-serve before initializing llm (#9646) |
| 1590 | 6dc8877416 | TensorRT LLM | 2025-12-07 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1591 | e4c707845f | Yan Chunwei | 2025-12-07 | [None][fix] enable hmac in RPC (#9745) |
| 1592 | 2645a78f34 | Jonas Li | 2025-12-06 | [TRTLLM-9660][feat] Convert cuteDSL GEMM to opt-in feature (#9682) |
| 1593 | 8d2178d321 | mpikulski | 2025-12-06 | [TRTLLM-9522][chore] implement default `attach_multimodal_embeddings` (#9664) |
| 1594 | 7cd5a67e25 | Enwei Zhu | 2025-12-06 | [TRTLLM-9372][feat] Enable CuteDSL MoE with Large EP (#9592) |
| 1595 | c2f2add6df | xxi | 2025-12-06 | [None][fix] fix a bug: deepseek_fp8_block_scales in TRTLLMGEN-MoE use 2D x_sf instead of 1D (#9658) |
| 1596 | df5b32966d | shuyixiong | 2025-12-06 | [None][fix] Fix triton moe load_weight (#9649) |
| 1597 | 74ed9f0468 | TensorRT LLM | 2025-12-06 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1598 | d4f68195c3 | QI JUN | 2025-12-01 | [TRTLLM-9092][doc] link to modelopt checkpoints in quick start guide (#9571) |
| 1599 | 0406949f32 | QI JUN | 2025-12-01 | [TRTLLM-9093][doc] update hyper links in overview (#9568) |
| 1600 | b7a255d67e | Yan Chunwei | 2025-12-01 | [TRTLLM-9075][doc] refine the slurm examples (#9548) |
| 1601 | 6ebdf1c304 | Yiqing Yan | 2025-11-29 | [None][infra] Updated Linux installation guide (#9485) |
| 1602 | b46e78e263 | Enwei Zhu | 2025-11-27 | [TRTLLM-9157][doc] Guided decoding doc improvement (#9359) |
| 1603 | 0915c4e3a1 | QI JUN | 2025-11-27 | [TRTLLM-9086][doc] Clean up TODOs in documentation (#9292) |
| 1604 | c6dc68a28e | Pengyun Lin | 2025-11-27 | [None][doc] VDR 1.0 trtllm-serve doc enhancement (#9443) |
| 1605 | 3e442922a3 | Yan Chunwei | 2025-11-27 | [TRTLLM-9160][doc] add doc to llm_runtime.py (#9482) |
| 1606 | 6332bf27e6 | jthomson04 | 2025-11-24 | [TRTLLM-9199][docs] KV Connector Docs (#9325) |
| 1607 | 9425f7fe3a | Iman Tabrizian | 2025-11-20 | [https://nvbugs/5601682][fix] Fix cacheTransceiver hang (#9311) |
| 1608 | 31ab367576 | Mike Iovine | 2025-12-05 | [None][chore] Waive flakey disagg tests (#9749) |
| 1609 | d6f95a4363 | Chenghao Zhang | 2025-12-05 | [None][feat] AutoDeploy: Perf optimization for Attention and rmsnorm (#9719) |
| 1610 | c7b5e3ea8f | yuanjingx87 | 2025-12-05 | [None][infra] Update allowed list 20251204 (#9718) |
| 1611 | 299601aebf | jthomson04 | 2025-12-05 | [https://nvbugs/5670672][fix] Fix flaky KV connector tests (#9676) |
| 1612 | eb0b426e5d | Robin Kobus | 2025-12-05 | [None][refactor] Improve request processing function in sampler (#9671) |
| 1613 | faf682b8bc | Robin Kobus | 2025-12-05 | [TRTLLM-7136][feat] Update load_weights method to include mapping parameter in checkpoint loaders (#9583) |
| 1614 | 68253d9d29 | yufeiwu-nv | 2025-12-05 | [https://nvbugs/5518713][test] Refactor core test lists by merging with llm_perf_cluster.yml (#9714) |
| 1615 | e06c582648 | Kaiyu Xie | 2025-12-05 | [None] [tests] Unwaive EPLB tests (#9625) |
| 1616 | a736226abd | TensorRT LLM | 2025-12-05 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1617 | 74df9b180b | gramnarayan | 2025-12-04 | [#9602][feat] AutoDeploy: Support TRTLLM Sampler (#9641) |
| 1618 | cb87c44912 | Kaiyu Xie | 2025-12-05 | [TRTLLM-9562] [doc] Add Deployment Guide for Kimi K2 Thinking on TensorRT LLM - Blackwell (#9711) |
| 1619 | dc766fc126 | Lizhi Zhou | 2025-12-05 | [https://nvbugs/5633340][fix] start disagg workers and servers on free ports (#9694) |
| 1620 | 0d0a16fff4 | Lizhi Zhou | 2025-12-05 | [TRTLLM-8920][feat] decouple disagg service from fastapi (#8714) |
| 1621 | 33224560b8 | Thor Johnsen | 2025-12-04 | [None][doc] Added line about partial reuse (#7846) |
| 1622 | e834f04238 | Yiqing Yan | 2025-12-05 | [TRTLLM-9579][infra] Set mergeWaiveList stage UNSTABLE when there is any issue (#9692) |
| 1623 | 5d6edc3944 | brb-nv | 2025-12-04 | [None][doc] Add feature docs for helix parallelism (#9684) |
| 1624 | 731b2eb4ef | Yiqing Yan | 2025-12-05 | [TRTLLM-5312][infra] Add triton trigger rules (#6440) |
| 1625 | cee7071e27 | pdrake-nv | 2025-12-04 | [None][infra] Add container notices and documentation (#9185) |
| 1626 | 041bb32151 | Aurelien Chartier | 2025-12-04 | [None][fix] Fix TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS for MTP/EAGLE (#9608) |
| 1627 | 530af1a98e | xinhe-nv | 2025-12-04 | [None][chore] Add failed cases into waives.txt (#9662) |
| 1628 | 60cdca3740 | Anthony Chang | 2025-12-04 | [None][fix] Recover TRTLLM MoE Perf for DEP (#9562) |
| 1629 | e5d4305c04 | Jin Li | 2025-12-04 | [https://nvbugs/5467531][fix] Unwaive fused_moe all to all test with … (#9617) |
| 1630 | 8a392af28f | ruodil | 2025-12-04 | [None][test] rename wide ep and disagg metric name in perf test (#9704) |
| 1631 | 398d24232d | zackyoray | 2025-12-04 | [None][feat] Add NIXL-LIBFABRIC support (#9225) |
| 1632 | 05058f5e2a | Yan Chunwei | 2025-12-04 | [None][ci] unwaive tests (#9651) |
| 1633 | f9aa86dbdd | tcherckez-nvidia | 2025-12-04 | [#8733][feat] Add Llama4 MoE handling to AutoDeploy (#9556) |
| 1634 | 6d2daec5d0 | JunyiXu-nv | 2025-12-04 | [TRTLLM-8274][feat] Check if executor is shutdown in /health entrypoint (#9057) |
| 1635 | 4eed648e22 | Tailing Yuan | 2025-12-04 | [None][feat] Add weights initialization and context phase parser to layer-wise benchmarks (#9667) |
| 1636 | 87e0c8a749 | Jin Li | 2025-12-04 | [TRTLLM-7073][feat] Support torch compile for PP for Llama and DeepSeekV3 (#7838) |
| 1637 | 323a82f4d5 | Necofish | 2025-12-04 | [None][fix] fix error when processing batches containing both text and mm data (#8381) |
| 1638 | 47f650ca13 | Yiqing Yan | 2025-12-04 | [TRTLLM-5093][infra] Write env variables to a file in the interactive debug session (#6792) |
| 1639 | 744f0eff1b | mpikulski | 2025-12-04 | [TRTLLM-9522][fix] restore `trtllm-serve mm_embedding_serve` (#9669) |
| 1640 | 94924634e0 | TensorRT LLM | 2025-12-04 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1641 | e31142202e | Yiqing Yan | 2025-12-04 | [TRTLLM-7181][infra] Generate test results when pytest timeout happens (#9396) |
| 1642 | 4485e516a2 | Wanli Jiang | 2025-12-04 | [None][feat] Update Qwen3CodeToolParser to align tool-calling parameters (#9540) |
| 1643 | 098b9ff226 | gramnarayan | 2025-12-03 | [#9147][feat] AutoDeploy: Draft Target Speculative Decoding (#9275) |
| 1644 | a1964bcbbc | Lucas Liebenwein | 2025-12-03 | [#9643][fix] AutoDeploy: fix nano sharding config (#9668) |
| 1645 | d9fba85396 | Wei-Ming Chen | 2025-12-03 | [OMNIML-2932] [feat] nvfp4 awq support (#8698) |
| 1646 | d7bd62b1a0 | Gal Hubara-Agam | 2025-12-03 | [https://nvbugs/5693853][fix] Fix error handling when querying machin… (#9483) |
| 1647 | b5e2b9b51f | Guoming Zhang | 2025-12-04 | [https://nvbugs/5702795][fix] Remove the warning message for aten.log. (#9665) |
| 1648 | 09beaa5933 | Iman Tabrizian | 2025-12-03 | [None][fix] Fix wide ep MoE error (#9642) |
| 1649 | 4e5b10da48 | Michal Guzek | 2025-12-03 | [https://nvbugs/5552132][fix] Enable LoRa for GPT OSS Torch (#8253) |
| 1650 | ae8d8a266a | Patrice Castonguay | 2025-12-03 | [https://nvbugs/5705197][chore] Unwaive timeout disagg tests (#9637) |
| 1651 | e2f82085f1 | Guoming Zhang | 2025-12-03 | [None][doc] Replace the tensorrt icon with torch icon on overview.md (#9644) |
| 1652 | 992781dc7b | Perkz Zheng | 2025-12-03 | [None][feat] update trtllm-gen nvfp4 kernels with better performance (#9510) |
| 1653 | 79e872de31 | Guoming Zhang | 2025-12-03 | [None][test] Update Qwen3-next accuracy testing by setting the cuda … (#9613) |
| 1654 | 743486b2ea | JunyiXu-nv | 2025-12-03 | [TRTLLM-6842][feat] Support Response API for general purpose (#9392) |
| 1655 | 3a748b166b | xinhe-nv | 2025-12-03 | [None][chore] Add failed cases into waives.txt (#9593) |
| 1656 | 1d4fb89235 | Pengyun Lin | 2025-12-03 | [TRTLLM-8241][feat] Aliasing to comply to LlmArgs (#9586) |
| 1657 | 80ff9015ce | fredricz-20070104 | 2025-12-03 | [https://nvbugs/5561153][test] Fix log error for perf test (#9622) |
| 1658 | 43f6ad7813 | brb-nv | 2025-12-02 | [https://nvbugs/5708475][fix] Fix e2e eval accuracy for helix parallelism (#9647) |
| 1659 | 8b5ededc83 | Bo Li | 2025-12-03 | [TRTLLM-9391][chore] Automatically estimate required workspace. (#9535) |
| 1660 | 93871d52b2 | Suyog Gupta | 2025-12-02 | [None][chore] AutoDeploy update cuda stream manager for multi-device (#9575) |
| 1661 | beffbd6002 | JunyiXu-nv | 2025-12-03 | [TRTLLM-9242][doc] Add examples showcasing openai compatible APIs (#9520) |
| 1662 | a08eb81cce | heyuhhh | 2025-12-03 | [None][feat] Add RocketKV usage doc and e2e accuracy test on LongBenchV2 (#9572) |
| 1663 | 097ac32b28 | TensorRT LLM | 2025-12-03 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1664 | 21f2ba74e8 | yufeiwu-nv | 2025-12-03 | [None][test] Remove duplicate test cases (#9623) |
| 1665 | 8c88454fa5 | Yiqing Yan | 2025-12-03 | [TRTLLM-7101][infra] Reuse passed tests (#6894) |
| 1666 | 642dfae73a | Anurag Mukkara | 2025-12-02 | [https://nvbugs/5698434][fix] Use separate weight mapper for draft (#9607) |
| 1667 | a3455f55c7 | Enwei Zhu | 2025-12-03 | [None][chore] Fix trtllm-eval and move GroupedGemmInputsHelper (#9612) |
| 1668 | 3916d032ec | Chang Liu | 2025-12-02 | [None][chore] Remove traceback dump for multimodal input processor (#9634) |
| 1669 | 55c7023c92 | brb-nv | 2025-12-02 | [None][chore] Waive test failing on pre-merge (#9638) |
| 1670 | 7ca38a6c0b | Yu Chi Li | 2025-12-02 | [#9632][feat] Support EXTRA_WHEEL_BUILD_ARGS during wheel build (#9633) |
| 1671 | 0a7a88e74e | Grzegorz Kwasniewski | 2025-12-02 | [TRTLLM-8946][feat] Improved heuristics to detect shardable regions (#9200) |
| 1672 | 3991aa9c72 | Patrice Castonguay | 2025-12-02 | [https://nvbugs/5688388][fix] fix: Reducing num request in disagg test to speed up (#9598) |
| 1673 | a560ba5546 | Neta Zmora | 2025-12-02 | [#9550][feat] AutoDeploy: Add NVFP4 Cutlass MoE kernels  (#9551) |
| 1674 | 227d42e492 | Shi Xiaowei | 2025-12-03 | [https://nvbugs/5651854][fix] Fix dist-serving perf by clearing CPU affinity (#9549) |
| 1675 | e72ce98c0f | Lucas Liebenwein | 2025-12-02 | [#9150][feat] AutoDeploy: reviewer comments for #9150 (#9527) |
| 1676 | 2dd3ebf037 | William Zhang | 2025-12-02 | [#9150][feat] Add code for nano v3 to custom implementation in AD (#9465) |
| 1677 | d5b7f0c8ad | Mike Iovine | 2025-12-02 | [TRTLLM-8980][test] Clean up spec dec tests in test_llm_api_pytorch (#8889) |
| 1678 | 95049eea86 | Thor Johnsen | 2025-12-02 | [https://nvbugs/5627710][fix] Fix synchronization bugs in KvCacheTransferManager that can cause corrupted blocks (#9056) |
| 1679 | b86256eb54 | Yan Chunwei | 2025-12-02 | [TRTLLM-9144][fix] enhance RPC robustness (#8711) |
| 1680 | 21e3dc11d8 | Jin Li | 2025-12-02 | [https://nvbugs/5667774][fix] Refine Piecewise Cuda Graph Condition for DP (#9393) |
| 1681 | 73a543d78f | Chang Liu | 2025-12-02 | [None][fix] Extract GPU count from single-node stage names (#9599) |
| 1682 | be48cdf1d1 | brb-nv | 2025-12-02 | [TRTLLM-9466][test] Evaluate helix parallelism with DSV3 Lite (#9597) |
| 1683 | 1a46bb0d18 | Eran Geva | 2025-12-02 | Lock the gpu clocks in L0 perf tests (#9585) |
| 1684 | 4a8766c11d | Emma Qiao | 2025-12-02 | [None][infra] Remove an invalid test name in waives.txt (#9620) |
| 1685 | f9524bcc07 | yuanjingx87 | 2025-12-02 | [None][infra] Update allowlist 2025/12/01 (#9616) |
| 1686 | 84a1531594 | mpikulski | 2025-12-02 | [TRTLLM-9488][feat] use FlashInfer.sampling by default (#9545) |
| 1687 | 3e4f2388a9 | Emma Qiao | 2025-12-02 | [None][infra] Waive failed cases for main branch (#9615) |
| 1688 | 1a2118b8fe | shuyixiong | 2025-12-02 | [https://nvbugs/5702793][fix] Fix uncontiguous tensor view (#9576) |
| 1689 | ad46d19027 | xinhe-nv | 2025-12-02 | [None][chore] Add failed cases into waives.txt (#9588) |
| 1690 | 4586b5f42f | ruodil | 2025-12-02 | [https://nvbugs/5582091][test] increase warmup times in testing for multi-gpu cases (#9578) |
| 1691 | 5657a00ec0 | Wanli Jiang | 2025-12-02 | [FMDL-1328][feat] Add support for nano-v3 and super-v3 with pytorch backend (#9261) |
| 1692 | 3911d0496e | xinhe-nv | 2025-12-02 | [None][fix] Waive gb200 (#9580) |
| 1693 | 9a6df980cd | JunyiXu-nv | 2025-12-02 | [https://nvbugs/5703953][fix] Use random port for disagg tests (#9582) |
| 1694 | 6fbe87c8b5 | Guoming Zhang | 2025-12-02 | [None][chroe] Polish qwen3-next modeling code. (#8902) |
| 1695 | 96a0e14522 | TensorRT LLM | 2025-12-02 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1696 | 356a52edf5 | Iman Tabrizian | 2025-12-01 | [None][feat] Add support for KVCache reuse for DSv32 (#9383) |
| 1697 | dcf5c86720 | Shijie | 2025-12-02 | [None][feat] Unify nvfp4 gemm backend (#8963) |
| 1698 | d11acee22d | QI JUN | 2025-12-02 | [TRTLLM-9085][doc] fix math formula rendering issues in github (#9605) |
| 1699 | 09c840184c | Yuening Li | 2025-12-02 | [None][fix] Prevent YAML partial kv_cache_config from incorrectly overriding the complete kv_cache_config (#9262) |
| 1700 | c9771ebb99 | Eran Geva | 2025-12-01 | [#9198][feat] Refactor dist ops in AutoDeploy (#9301) |
| 1701 | 0a2104dce9 | Chenghao Zhang | 2025-12-01 | [None][feat] AutoDeploy: Use the router gemm op for nemotron MOE (#9500) |
| 1702 | 639c939a4f | Venky | 2025-12-01 | [TRTC-1943][feat] Env vars override support in LLM API (#9104) |
| 1703 | f61067cbb5 | brb-nv | 2025-12-01 | [None][chore] Defer exposing context parallel configs (#9552) |
| 1704 | f155812eb0 | Stefan Niebler | 2025-12-01 | [TRTLLM-6756][feat] Add Beam Search to TorchSampler (#8509) |
| 1705 | b024040df0 | Emma Qiao | 2025-12-02 | [None][infra] Update the pytest options after MI (#9579) |
| 1706 | c72919980a | Yiqing Yan | 2025-12-01 | [TRTLLM-6768][infra] Fix params for not updating github status (#6747) |
| 1707 | 078d3a576e | Yanchao Lu | 2025-12-01 | [None][ci] Minor change for Slurm scripts (#9561) |
| 1708 | 7127c4407a | Yanchao Lu | 2025-12-01 | [None][test] [None][test] Waive main branch test failures 12/1 (#9566) |
| 1709 | 90345ad3f3 | Enwei Zhu | 2025-12-01 | [None][fix] Skip Allreduce init for Attention DP (#9542) |
| 1710 | 48b1d31895 | Shi Xiaowei | 2025-12-01 | [https://nvbugs/5651854][infra] Enable perf metrics during accuracy testing (#9140) |
| 1711 | 974ad56515 | Martin Marciniszyn Mehringer | 2025-12-01 | [None][chore] reduce the layers of the `devel` docker image (#9077) |
| 1712 | 4107254c82 | alel | 2025-12-01 | [TRTLLM-6222][feat] Several perf opt for cuteDSL nvf4 gemm (#9428) |
| 1713 | 24004535fe | Zhenhuan Chen | 2025-12-01 | [None][chore] refactor disaggregated scripts to use named arguments (#9581) |
| 1714 | 730eb3d859 | Yukun He | 2025-12-01 | [None][fix] Replace hash method with unique_id for cutedsl MoE runners. (#9569) |
| 1715 | bc25fff039 | Neta Zmora | 2025-12-01 | [#9496][fix] AutoDeploy: remove auto-tuner from nvfp4_gemm forward (#9497) |
| 1716 | d69bf9f92a | Fanrong Li | 2025-12-01 | [None][feat] add chat template kwargs support to longbench-v2 (#9544) |
| 1717 | 9d2df04a72 | Gaoji Liu | 2025-12-01 | [None][doc] fix mtp.py typo (#9307) |
| 1718 | a92af27411 | JadoTu | 2025-12-01 | [None][chore] remove qwen3-next accuracy tests (#9534) |
| 1719 | aa3310f64f | Pengbo Wang | 2025-12-01 | [https://nvbugs/5503479][fix] Temporarily lower reference accuracy to stabilize CI (#9398) |
| 1720 | 2e3ac3c48f | Enwei Zhu | 2025-12-01 | [https://nvbugs/5684703][fix] Unwaive disagg guided decoding test (#9466) |
| 1721 | 0b10214f55 | TensorRT LLM | 2025-12-01 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1722 | becd44f9bc | Yuan Tong | 2025-12-01 | [None][fix] Correct virtual memory allocation alignment (#9491) |
| 1723 | 1797e91dfd | Li Min | 2025-12-01 | [TRTLLM-6222][feat] Extend cute_dsl_nvfp4_gemm to sm103. (#9543) |
| 1724 | 34e2fa5c96 | Enwei Zhu | 2025-12-01 | [https://nvbugs/5690172][fix] Fix Qwen3-235B ATP accuracy issue with PDL (#9530) |
| 1725 | 6e470aab72 | heyuhhh | 2025-12-01 | [None] [feat] Optimize the algorithm part of RocketKV (#9333) |
| 1726 | c12e67bb66 | xxi | 2025-12-01 | [TRTLLM-8958][feat] and [TRTLLM-8960]: create ConfigurableMoE and support TRTLLMGenFusedMoE as backend (#9486) |
| 1727 | 694b60d92d | Yanchao Lu | 2025-11-30 | [None][ci] Split H100_PCIe-PyTorch-Post-Merge test stage (#9559) |
| 1728 | 0398875d55 | Yanchao Lu | 2025-11-30 | [None][ci] Split H100_PCIe-PyTorch-Post-Merge test stage (#9558) |
| 1729 | 3f588198dc | JunyiXu-nv | 2025-11-30 | [None][fix] Fix port conflict in disagg tests (#9474) |
| 1730 | c927ccf510 | Emma Qiao | 2025-11-30 | [None][infra] Wiave failed tests for main branch on 11/30 (#9555) |
| 1731 | f03641808b | Yanchao Lu | 2025-11-30 | [None][infra] - Request idle time exemption for OCI jobs (#9528) |
| 1732 | bde69dd1df | TensorRT LLM | 2025-11-30 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1733 | b77f4ffe54 | brb-nv | 2025-11-29 | [TRTLLM-5971][feat] Integrate helix parallelism (#9342) |
| 1734 | 6345074686 | dominicshanshan | 2025-11-29 | [None][chore] Weekly mass integration of release/1.1 -- rebase (#9522) |
| 1735 | ae0124ef84 | TensorRT LLM | 2025-11-29 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1736 | cff54fcae3 | Grzegorz Kwasniewski | 2025-11-28 | [#8948][feat] Support custom sharding config (#9143) |
| 1737 | bc355eadf5 | mpikulski | 2025-11-28 | [TRTLLM-9488][fix] llmapi references (#9547) |
| 1738 | db5b876124 | binghanc | 2025-11-29 | [None][feat] support for more accurate AR calculation (#9323) |
| 1739 | f8dd494536 | Matthias Jouanneaux | 2025-11-28 | [None][perf] Helix: improve all-to-all perf for large CP size (#9494) |
| 1740 | 70efa3ac43 | dominicshanshan | 2025-11-28 | [None][infra] Waive failed case in pre-merge on 11/28 (#9537) |
| 1741 | e5f39ec7cf | mpikulski | 2025-11-28 | [TRTLLM-9488][feat] add 'disable_flashinfer_sampling' config option (#9454) |
| 1742 | 930cdad054 | Zhanrui Sun | 2025-11-28 | [TRTLLM-9541][infra] Use artifactory mirror for download.pytorch.org (#9477) |
| 1743 | 5eae3650c3 | Robin Kobus | 2025-11-28 | [None][fix] Pass checkpoint_format to create_input_processor (#9521) |
| 1744 | 2d7421b314 | Emma Qiao | 2025-11-28 | [None][infra] Waive failed cases for main branch on 11/28 (#9539) |
| 1745 | 7c3bb8534d | Zhenhuan Chen | 2025-11-28 | [None][chore] Revert "[None][fix] change allreduce workspace dtype to torch.int64 t… (#9538) |
| 1746 | 0d3c0c2156 | Kaiyu Xie | 2025-11-28 | [None] [chore] Enhancements and clean up to slurm scripts (#9493) |
| 1747 | 389b73c349 | Chang Liu | 2025-11-27 | [None][fix] Remove FP8 K/V buffer from TRTLLM sparse MLA attention kernel (#9529) |
| 1748 | bf84d9cea1 | Liao Lanyu | 2025-11-28 | [None][chore] add spec_decoding configs in perf benchmark scripts and fix typos (#9533) |
| 1749 | 08755a809d | yufeiwu-nv | 2025-11-28 | [https://nvbugs/5689658][test] Fix gpu lock issue running on cluster (#9441) |
| 1750 | 60c43a200a | Yukun He | 2025-11-28 | [None][fix] Fix on-disk cache and revise logger/statistics for AutoTuner. (#9211) |
| 1751 | c87e81c1d8 | JunyiXu-nv | 2025-11-28 | [https://nvbugs/5685015][fix] Update invalid max_token test (#9435) |
| 1752 | 658d9fc0c5 | Emma Qiao | 2025-11-28 | [TRTLLM-8970][infra] Fix generate report when has isolation test result (#8861) |
| 1753 | 5e52dff6c6 | TensorRT LLM | 2025-11-28 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1754 | 19f3f4e520 | Bo Li | 2025-11-28 | [https://nvbugs/5637037][chore] Update waive lists. (#9386) |
| 1755 | 85b4c92d60 | Kaiyu Xie | 2025-11-28 | [None] [chore] Update to cutlass 4.3 (#8637) |
| 1756 | 2f8bd6fb36 | Lucas Liebenwein | 2025-11-27 | [#9150][feat] AutoDeploy Nemotron-Flash support (#9504) |
| 1757 | c2562fc800 | Enwei Zhu | 2025-11-27 | [https://nvbugs/5687820][fix] Remove self.abort() in DetokenizedGenerationResult (#9449) |
| 1758 | 1c9158fde3 | Yiqing Yan | 2025-11-27 | [TRTLLM-7288][infra] Download merged waive list in slurm script (#8999) |
| 1759 | 4cbfc10b28 | Yueh-Ting (eop) Chen | 2025-11-27 | [https://nvbugs/5674665][chore] Add test coverage for https://nvbugspro.nvidia.com/bug/5674665 (#9518) |
| 1760 | 62b771877c | Bo Li | 2025-11-27 | [TRTLLM-9389][chore] Refactor AlltoallMethodType. (#9388) |
| 1761 | 2d5eadf65f | Fanrong Li | 2025-11-27 | [None][fix] fix TP support for DeepSeek-V3.2 on hopper (#9484) |
| 1762 | 51bf7164d3 | JadoTu | 2025-11-27 | [None][feat] add qwen3-next CI test of accuracy on BF16 and NVFP4 (#9330) |
| 1763 | e47927e847 | Zhenhuan Chen | 2025-11-27 | [None][fix] change allreduce workspace dtype to torch.int64 to avoid overflow (#9479) |
| 1764 | 3ada0bfc65 | yuanjingx87 | 2025-11-27 | [None][infra] Fix Slurm job script (#9508) |
| 1765 | f1ed057b4c | xxi | 2025-11-27 | [cherry-pick][https://nvbugs/5670793][fix] Solve trtllm-serve launch_disaggregated issue (#9346) |
| 1766 | a21be43677 | Emma Qiao | 2025-11-27 | [TRTLLM-9279][infra] Use flexcache for gh200 nodes since they locate in Austin (#9405) |
| 1767 | 8104a78931 | Lizhi Zhou | 2025-11-27 | [None][chore] revert batch_size=1 to prevent timeout and lower accuracy reference by 0.12% as a WAR (#9447) |
| 1768 | 5425d96757 | Liao Lanyu | 2025-11-27 | [TRTLLM-9513][docs] Qwen3 deployment guide (#9488) |
| 1769 | 0442510304 | Emma Qiao | 2025-11-27 | [None][infra] Waive failed case in pre-merge on 11/27 (#9507) |
| 1770 | 1dd55d8507 | Ziyi Xiong | 2025-11-27 | [https://nvbugs/5698581][fix] Init draft tokens for CUDA graph dummy request (#9505) |
| 1771 | 14762e0287 | Jiagan Cheng | 2025-11-27 | [None][fix] Replace PYTORCH_CUDA_ALLOC_CONF with PYTORCH_ALLOC_CONF to fix deprecation warning (#9294) |
| 1772 | 03331bc43d | HuiGao-NV | 2025-11-27 | [https://nvbugs/5547414][fix] enable case after using local cache model (#9473) |
| 1773 | 1b2da426cd | Patrice Castonguay | 2025-11-26 | [https://nvbugs/5680310][fix] Fix ctx only timed out test (#9410) |
| 1774 | 89701a594b | TensorRT LLM | 2025-11-27 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1775 | a67d94963e | QI JUN | 2025-11-27 | [None][chore] update comments in llm_args.py (#9472) |
| 1776 | c6fa042332 | QI JUN | 2025-11-27 | [TRTLLM-9085][doc] fix math formula rendering issues (#9481) |
| 1777 | f2f197360d | Aurelien Chartier | 2025-11-26 | [#9463][feat] Add revision option to trtllm commands (#9498) |
| 1778 | e76e149861 | Shi Xiaowei | 2025-11-27 | [https://nvbugs/5608930][fix] Fix a typo (#9487) |
| 1779 | dbbed1f85a | Zheyu Fu | 2025-11-26 | [None][ci] Waive blackwell test on spec gate. (#9502) |
| 1780 | 18fbda5cdb | Chenghao Zhang | 2025-11-26 | [None][feat] AutoDeploy: Add A_log fusion for Mamba layers (#9422) |
| 1781 | bc7b60e016 | Chenghao Zhang | 2025-11-26 | [None][feat] AutoDeploy: Remove redundant copies in mamba layers (#9461) |
| 1782 | 356f67c1cb | yuanjingx87 | 2025-11-26 | [None][infra] Fail the pipeline when slurm ssh dropped (#9157) |
| 1783 | d7ef8849d2 | yuanjingx87 | 2025-11-26 | [None][infra] Update allowed list 2025.11.25 (#9468) |
| 1784 | ef7ee6a940 | Aurelien Chartier | 2025-11-26 | [None][feat] Add environment variable to force spec-dec number of accepted tokens (#9371) |
| 1785 | b10137fdd5 | Chang Liu | 2025-11-26 | [None][feat] Support MLA chunked prefill for DeepSeek V3.2 model (#9376) |
| 1786 | 1bf2d750a2 | Enwei Zhu | 2025-11-26 | [None][chore] Upgrade CuteDSL to 4.3.0 (#9444) |
| 1787 | b7308a4000 | JunyiXu-nv | 2025-11-26 | [https://nvbugs/5580099][fix] Cherry pick IMA issue fix from release/1.1 (#9032) |
| 1788 | d100599ea7 | Wanli Jiang | 2025-11-26 | [TRTLLM-9264][fix] Add accuracy/unit tests/doc for phi4mm (#9246) |
| 1789 | b04421e5ba | TensorRT LLM | 2025-11-26 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1790 | d8acea1db3 | shuyixiong | 2025-11-26 | [TRTLLM-9293][feat] Enable partial weight loading to support streaming update weights (#9224) |
| 1791 | 5972119e1c | QI JUN | 2025-11-26 | [None][ci] move some slow test cases of DGX-B200 to post merge (#9467) |
| 1792 | 6a64cb4c71 | fredricz-20070104 | 2025-11-26 | [TRTLLM-8936][test] Add disagg and wideep multi-node multi-gpu test cases (#9356) |
| 1793 | 1b9edf62c9 | Yiqing Yan | 2025-11-26 | [None][chore] Bump version to 1.2.0rc5 (#9455) |
| 1794 | 0e9c7f8c07 | Chuang Zhu | 2025-11-26 | [https://nvbugs/5685143][fix] avoid cudaFree overlap with cuda graph (#9438) |
| 1795 | e484bec82f | Suyog Gupta | 2025-11-25 | [None][chore] AutoDeploy add multi stream moe pass to default.yaml (#9430) |
| 1796 | 32f53910ef | Robin Kobus | 2025-11-25 | [TRTLLM-909][feat] Overlap context chunks in pipeline parallel mode (#9308) |
| 1797 | afc52d7b93 | Eran Geva | 2025-11-25 | [https://nvbugs/5647400] [fix] Enlarged the AllReduce workspace size to 64MB. Added AllReduce strategy to AD config. (#9145) |
| 1798 | 899fda9e47 | mpikulski | 2025-11-25 | [TRTLLM-9490][feat] use FlashInfer's top_k_sampling_from_probs (#9457) |
| 1799 | c5f52ab304 | mpikulski | 2025-11-25 | [TRTLLM-8376][feat] top-p optimization (removes redundant softmax) (#9411) |
| 1800 | 8da59103d6 | Fanrong Li | 2025-11-26 | [https://nvbugs/5680905][fix] Relax the MMLU accuracy requirement for DS-v3.2 (#9439) |
| 1801 | 1f43dc8174 | Yan Chunwei | 2025-11-25 | [None][ci] waive a test (#9458) |
| 1802 | cc336c4abd | YueWeng | 2025-11-25 | [TRTLLM-8160][feat] Add draft token tree runtime on CDL (#8586) |
| 1803 | fa61825c74 | Pengyun Lin | 2025-11-25 | [None][feat] Support custom chat template for tool calling (#9297) |
| 1804 | 51ef0379d2 | Tailing Yuan | 2025-11-25 | [None][feat] Add a parser to layer-wise benchmarks (#9440) |
| 1805 | c36f144591 | Fanrong Li | 2025-11-25 | [None][chore] Fix trtllm-eval for PyTorchLLM (#9427) |
| 1806 | 60786574db | Shi Xiaowei | 2025-11-25 | [None][fix] Mitigate test timeout issues (#9445) |
| 1807 | a2d9e6250a | Chao Ni | 2025-11-25 | [https://nvbugs/5667922][fix] Update long context evaluation config (#9426) |
| 1808 | a38d91aae2 | Yueh-Ting (eop) Chen | 2025-11-25 | [https://nvbugs/5537996][fix] Let KV cache manager block initialization be aware whether it is doing a dry run or not (#9093) |
| 1809 | 4742c130db | Anthony Chang | 2025-11-25 | [None][feat] Improve TRTLLM MoE in small hidden size throughput cases (#9377) |
| 1810 | ff02e0f05c | Yanchao Lu | 2025-11-25 | [None][ci] Move more test stages to use OCI machines (#9395) |
| 1811 | 6af01dc664 | Eran Geva | 2025-11-25 | [#8391][chore] test_perf.py to lock clocks read from gpu_configs.yml instead of max freq (#9409) |
| 1812 | 15616e3ee5 | Emma Qiao | 2025-11-25 | [None][infra] Waive failed cases for main branch on 11/25 (#9429) |
| 1813 | e580da4155 | Yukun He | 2025-11-25 | [TRTLLM-7963][feat] Cold L2 cache when doing autotune benchmarking. (#8779) |
| 1814 | a4049fc557 | William Zhang | 2025-11-24 | [#9413][fix] Minor fixes to nemotron H and custom models in AD (#9416) |
| 1815 | efd503751f | Suyog Gupta | 2025-11-24 | [#9271][perf] Enable multi-stream MOE optimization in AutoDeploy (#9322) |
| 1816 | d1c724958d | kris1025 | 2025-11-25 | [None][chore] unwaive ampere kernels test (#9389) |
| 1817 | bf0d1dc6a8 | TensorRT LLM | 2025-11-25 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1818 | 0a9ae2e3e6 | xinhe-nv | 2025-11-25 | [None][chore] Remove closed bugs (#9381) |
| 1819 | 8a0295015f | Yuxian Qiu | 2025-11-25 | [None][chore] Reduce nested nvtx ranges. (#9347) |
| 1820 | 5a99c9734d | Fanrong Li | 2025-11-25 | [TRTLLM-8777][feat] Update DeepGEMM to the latest commit to include optimizations for DeepSeek-v3.2 (#9380) |
| 1821 | 786d308b88 | QI JUN | 2025-11-25 | [https://nvbugs/5685428][fix] fix test_openai_chat_multimodal.py (#9406) |
| 1822 | 1a93583438 | bhsueh_NV | 2025-11-25 | [None][feat] Support Yarn on QwQ-32B model (#9059) |
| 1823 | 1ce483c999 | Yibin Li | 2025-11-24 | [TRTLLM-7967][feat] Adding Starcoder2 PyTorch Backend Support (#8923) |
| 1824 | 336593cac5 | YueWeng | 2025-11-25 | [None][fix] Fix topk outIndices when using vectorized_process (#9404) |
| 1825 | f95edb53e1 | Chuang Zhu | 2025-11-24 | [None][fix] enhance warning in cacheTransBuffer (#9390) |
| 1826 | 2c869f2bda | Emma Qiao | 2025-11-24 | [None][infra] Waive failed cases for main (#9400) |
| 1827 | 6e5384d03c | cheshirekow | 2025-11-23 | [TRTLLM-9299][infra] Add third-party docs for python (#9366) |
| 1828 | 2810be7b3b | cheshirekow | 2025-11-23 | [TRTLLM-9211][infra] Minor fixes to 3rdparty/CMakelists (#9365) |
| 1829 | af72d93fa9 | Emma Qiao | 2025-11-24 | [None][infra] Waive failed cases on main branch (#9384) |
| 1830 | 960851f419 | Yukun He | 2025-11-24 | [None][chore] Remove unnecessary log in the short tuning profile (#9387) |
| 1831 | 39076410a8 | Yukun He | 2025-11-24 | [https://nvbugs/5676748][fix] Fix mismatched nvfp4 gemm sf shape. (#9336) |
| 1832 | c045e359a7 | brb-nv | 2025-11-23 | [https://nvbugs/5637012][fix] Fix helix unit tests (#9369) |
| 1833 | 5a44994d05 | TensorRT LLM | 2025-11-24 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1834 | 34a6d2d28f | QI JUN | 2025-11-24 | [TRTLLM-9302][chore] Move build config from BaseLlmArgs to TrtLlmArgs (#9249) |
| 1835 | c3acf965a6 | Yukun He | 2025-11-24 | [TRTLLM-7963][fix] Several improvements of autotuning quality (#9348) |
| 1836 | fcfec93cad | Bo Li | 2025-11-24 | [TRTLLM-9389][chore] Rename AlltoAll backend names (#9329) |
| 1837 | e1c9aa7d6a | Chenghao Zhang | 2025-11-23 | [None][chore] AutoDeploy: Add the Nemotron MOE to CI (#9328) |
| 1838 | 0582e54b61 | JadoTu | 2025-11-23 | [None][fix] modify qwen3-next sampling stop_tokens (#9331) |
| 1839 | 11a0b276fb | William Zhang | 2025-11-23 | [#9230][feat] Slimmed down implementation of nemotron H (#9235) |
| 1840 | 1ef69ecbb1 | Yan Chunwei | 2025-11-23 | [None][ci] waive two ray tests (#9375) |
| 1841 | a761585d9c | TensorRT LLM | 2025-11-23 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1842 | 268ea9bb8a | dongfengy | 2025-11-21 | [None][test] Add one-model and overlap-scheduling to eagle tests for GPTOSS (#9312) |
| 1843 | 15ceba8705 | TensorRT LLM | 2025-11-22 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1844 | fefa02fa95 | Matt Lefebvre | 2025-11-21 | [TRTINFRA-7326][infra] - Consume SlurmCluster sshPort for clusters with custom SSH port (#9313) |
| 1845 | 3952a61681 | Neta Zmora | 2025-11-22 | [#9388][fix] AutoDeploy: Fix cutlass BF16 MoE kernel invocation (#9339) |
| 1846 | 564989865c | Chenghao Zhang | 2025-11-21 | [TRTLLM-9082][feat] AutoDeploy: Move the moe Align kernel to AOT (#9106) |
| 1847 | eb7792e875 | Izzy Putterman | 2025-11-21 | [None][feat] Eagle: PostNorm and multilayer options (#9233) |
| 1848 | 13fbd4366a | Enwei Zhu | 2025-11-22 | [TRTLLM-9370][feat] Integration of CuteDSL NVFP4 grouped GEMM (Part 2: SwiGLU Fusion and Finalize Fusion) (#9288) |
| 1849 | 9b2abb8d28 | cheshirekow | 2025-11-21 | [TRTLLM-9208][infra] Document the process for C++ deps (#9016) |
| 1850 | 5df907b388 | Ziyi Xiong | 2025-11-21 | [https://nvbugs/5590408][fix] Fallback to greedy sampling in two-model overlap scheduler (#9321) |
| 1851 | f2ebaf288a | Nikita Korobov | 2025-11-21 | [None][feat] TRT-LLM Gen MoE optimize DeepSeek Fp8 activation kernel (#9175) |
| 1852 | 6dd2fcd7b3 | HuiGao-NV | 2025-11-21 | [https://nvbugs/5629833][fix] Don't fill tensors with 0 (#9296) |
| 1853 | cddc7549d1 | mpikulski | 2025-11-21 | [TRTLLM-9191][feat] support out-of-tree models in trtllm-serve (#9269) |
| 1854 | 095b6864a8 | mpikulski | 2025-11-21 | [TRTLLM-8650][fix] beam search request validation (#8433) (#9228) |
| 1855 | 8cd3b496e9 | Yiqing Yan | 2025-11-21 | [None][chore] Bump version to 1.2.0rc4 (#9363) |
| 1856 | 041564188c | Emma Qiao | 2025-11-21 | [None][infra] Waive failed cases in main post-merge on 11/21 (#9360) |
| 1857 | b6483ef3e7 | QI JUN | 2025-11-21 | [None][ci] waive a test case of test_ad_build_small_multi.py (#9355) |
| 1858 | 28e9bf6167 | Ivy Zhang | 2025-11-21 | [None][chore] add periodic junit xml path in conftest (#9337) |
| 1859 | cc0dc7c124 | xxi | 2025-11-21 | [TRTLLM-8957][feat] create communication related classes (#8968) |
| 1860 | 2a27166b59 | Yiqing Yan | 2025-11-21 | [TRTLLM-9183][infra] Add --waives-file in rerun pytest command (#8971) |
| 1861 | 5138ef3227 | Zhanrui Sun | 2025-11-21 | [None][infra] Add fallback when get wheel from build stage is fail (#9290) |
| 1862 | e2a372a3b1 | QI JUN | 2025-11-21 | [None][ci] waive test_llm_context_only_timed_out_kv_cache_exhausted (#9351) |
| 1863 | 39e641872c | TensorRT LLM | 2025-11-21 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1864 | b5863ed1e2 | Yingge He | 2025-11-20 | [TRI-332] [fix] Fix L0_backend_trtllm (#9282) |
| 1865 | 1379cfac3a | cheshirekow | 2025-11-20 | [TRTLLM-9197][infra] Move thirdparty stuff to it's own listfile (#8986) |
| 1866 | b1c9936c36 | Kanghwan | 2025-11-20 | [None][infra] Update goggles_action repository (#9240) |
| 1867 | f8dd52621d | tburt-nv | 2025-11-20 | [None][chore] Upgrade starlette and FastAPI (#9319) |
| 1868 | 69b4e52757 | Mike Iovine | 2025-11-04 | [None][chore] Update linter rules for mass integration |
| 1869 | a3433dd54e | Barry Kang | 2025-11-01 | [https://nvbugs/5325296][fix] Enable relaxed acceptance test on Blackwell (#8709) |
| 1870 | 62e20a5441 | Zhanrui Sun | 2025-10-31 | [None][infra] Remove invaild waived tests which not in release branch (#8841) |
| 1871 | 6185225501 | Jin Li | 2025-10-31 | [https://nvbugs/5488118][fix] Unwaive passed tests (#8758) |
| 1872 | 0c8de1f45d | Dom Brown | 2025-10-30 | [https://nvbugs/5575841] [test] Move test_moe.py to serial tests to improve stability + unwaive FP4 MoE torch unit tests (#8422) |
| 1873 | 05aabfbc1e | xiweny | 2025-10-29 | [https://nvbugs/5601203] [fix]Restrict fp8 blockscale moe case (#8583) |
| 1874 | 8846dac9b4 | Chuang Zhu | 2025-10-29 | [https://nvbugs/5578175][fix] Fix block range index (#8470) |
| 1875 | eca68e4465 | Pengyun Lin | 2025-10-28 | [https://nvbugs/5564465][fix] Overwrite only if default_max_tokens is legal (#8538) |
| 1876 | 3d66e56adb | Eran Geva | 2025-10-28 | [https://nvbugs/5572320][fix] Ported test_ad_trtllm_bench.py from main (#8671) |
| 1877 | 9a79f32f7a | Yukun He | 2025-10-28 | [https://nvbugs/5608489][fix] Fix output unpack issues for Llama3/4 NVFP4 models. (#8679) |
| 1878 | 25c0624750 | Ivy Zhang | 2025-10-24 | [None][test] Clean cache for certain easily hang cases (#8619) |
| 1879 | 36e244f35e | Jie Li | 2025-10-24 | [https://nvbugs/5587456][fix] Remove multimodal test cases using TRT backend (#8611) |
| 1880 | 348668e3ae | Lizhi Zhou | 2025-10-23 | [https://nvbugs/5575902][fix] set max_batch_size=1 to stabilize accuracy test result (#8609) |
| 1881 | 33b0b945c7 | Lizhi Zhou | 2025-10-23 | [https://nvbugs/5582277][fix] rework DisaggPPTerminationHandler to fix hang issue (#8519) |
| 1882 | b5f9fff1c1 | Yan Chunwei | 2025-10-23 | [https://nvbugs/5569754][fix] trtllm-llmapi-launch port conflict  (#8582) |
| 1883 | 81fd9be87d | Pengyun Lin | 2025-10-22 | [https://nvbugs/5575829][fix] Unwaive gpt-oss test (#8576) |
| 1884 | 4ca6fe83d8 | Bo Deng | 2025-10-22 | [https://nvbugs/5565549][fix] unwaive test_disaggregated_spec_dec_bat… (#8500) |
| 1885 | 3454eacd74 | Jin Li | 2025-10-22 | [https://nvbugs/5546510][fix] Move torch.cuda.Stream out of torch com… (#8494) |
| 1886 | af3900a195 | Guoming Zhang | 2025-10-22 | [https://nvbugs/5504095][fix] Unwaive test_user_specify_workspace case. (#8316) |
| 1887 | 9286223288 | Simeng Liu | 2025-10-21 | [https://nvbugs/5515753][ci] Add NCCL_DEBUG=INFO flag to collect more info with CI failure.  (#8440) |
| 1888 | ee6944bfa2 | JunyiXu-nv | 2025-10-22 | [https://nvbugs/5569713][fix] Disable fp8 deep gemm for EXAONE-4.0-32B-FP8 (#8429) |
| 1889 | 0e746fad45 | yufeiwu-nv | 2025-11-20 | [https://nvbugs/5667454][test] Fix Test Case as Chunked Attention not Supported on sm_120 (#9260) |
| 1890 | 04ad9f96fa | Liao Lanyu | 2025-11-20 | [https://nvbugs/5667687][fix] Set correct lm_head_tp_size_upper_bound (#9300) |
| 1891 | 1d6fbbf45d | Neta Zmora | 2025-11-20 | [#9236][feature] Make sharing of activation_type across SW layers more robust (#9238) |
| 1892 | b018b2698d | Emma Qiao | 2025-11-20 | [TRTLLM-9164][infra] Enable checking duplicate items in waives.txt in pre-commit (#9265) |
| 1893 | a39e8c5567 | mpikulski | 2025-11-20 | [TRTLLM-9295][fix] use greedy decoding in test_openai_compatible_json_schema (#9305) |
| 1894 | 5d118e0326 | Yukun He | 2025-11-20 | [None][chore] Revise the description of enable_autotuner. (#9320) |
| 1895 | 1bdd3ba173 | QI JUN | 2025-11-20 | [None][ci] waive test_disagg_server_restart (#9326) |
| 1896 | d5622b2689 | Yechan Kim | 2025-11-20 | [None][fix] Multimodal InputProcessor dummy builder fix (#8916) |
| 1897 | 79a6c9742b | Chang Liu | 2025-11-19 | [None][fix] Use fp32 for indexer weight_proj GEMM (#9243) |
| 1898 | 028fc877a5 | Neta Zmora | 2025-11-20 | [#9096][feature] Auto Deploy: configurable fused MoE backend (#9194) |
| 1899 | cd44f80abd | Chenghao Zhang | 2025-11-19 | [#9316][feat] AutoDeploy: Add the accuracy test for Nemotron MOE models (#9317) |
| 1900 | 3004692949 | TensorRT LLM | 2025-11-20 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1901 | 2128f73d58 | Bo Deng | 2025-11-20 | [TRTLLM-9247][infra] Upgrade NIXL to 0.7.1 (#9055) |
| 1902 | 46dccb5e2d | JunyiXu-nv | 2025-11-20 | [None][chore] Prevent negative `max_tokens` passed into tllm request (#9037) |
| 1903 | b6bced83c0 | Yukun He | 2025-11-20 | [TRTLLM-7963][feat] Use CUDAGraph to improve the tuning accuracy for AutoTuner. (#9089) |
| 1904 | 41e5870a70 | Kanghwan | 2025-11-19 | [#8476][chore] Update license (#8807) |
| 1905 | d4abb86f3e | Fanrong Li | 2025-11-20 | [None][fix] fix EPLB for DeepSeek-V3.2-Exp (#9245) |
| 1906 | f6ec6e2222 | brb-nv | 2025-11-19 | [None][chore] Waive tests timing out on main (#9315) |
| 1907 | 49c45ebef1 | Faraz | 2025-11-19 | [None][fix] change logging for weight loading on unified memory (#9177) |
| 1908 | 1eae941d77 | NVShreyas | 2025-11-19 | [#9237][feat] enable iter stats in autodeploy (#9278) |
| 1909 | a7c0b54ce7 | NVShreyas | 2025-11-19 | [None][feat] add specdec to nemotron nas (#8985) |
| 1910 | 7ab02ad7b5 | Neta Zmora | 2025-11-19 | [None][feature] AutoDeploy: tighter MoE UT thresholds (#9195) |
| 1911 | d8b05894ee | Bo Li | 2025-11-19 | [None][perf] Adjust select_alltoall_method_type. (#8950) |
| 1912 | 46dd9886bb | mpikulski | 2025-11-19 | [https://nvbugs/5661877][fix] fix test regression in TestBatchedSampling::test_samples (#9215) |
| 1913 | 0f77fec932 | xinhe-nv | 2025-11-19 | [None][chore] Add failed cases into waives.txt (#9289) |
| 1914 | ee941ac779 | CarstyYou | 2025-11-19 | [https://nvbugs/5456493][feat] add fp8 dense for sm120 (#9174) |
| 1915 | a79c0dfb43 | nvxuanyuc | 2025-11-18 | [None][fix] Update GLM model accuracy test (#9286) |
| 1916 | 255e4ea9f0 | jiahanc | 2025-11-18 | [None][doc] Update DS-R1 example doc (#9231) |
| 1917 | 67d3eb26af | Emma Qiao | 2025-11-19 | [None][infra] Waive failed cases for main branch on 11/17 (#9266) |
| 1918 | 941a54c66a | ChristinaZ | 2025-11-19 | [None][feat] Update the indexer topK (#9255) |
| 1919 | 286ace22ed | xinhe-nv | 2025-11-19 | [None][chore] Add failed cases into waives.txt (#9242) |
| 1920 | 9135d580bf | TensorRT LLM | 2025-11-19 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1921 | 99ba723e20 | jellysnack | 2025-11-19 | [None][fix] logits device and shape issues in dynamic draft path (#9079) |
| 1922 | 782dfca7e8 | Ivy Zhang | 2025-11-19 | [TRTLLM-9050][test] add llama4 disagg case to cover kv cache overflow error (#9172) |
| 1923 | 7905d6c0da | Grzegorz Kwasniewski | 2025-11-19 | [#9098][feat] Simple sharding latent experts (#9099) |
| 1924 | fbf6c16cd2 | ChristinaZ | 2025-11-19 | [None][fix] Update the default invalid value for deepseek mode of routing (#9222) |
| 1925 | 92f86a50d4 | Grzegorz Kwasniewski | 2025-11-19 | [#9137][feat] Factory sharding as default (#9144) |
| 1926 | 9b0f45298f | Patrice Castonguay | 2025-11-18 | [None][feat] Have ability to cancel disagg request if KV cache resource are exhausted (#9155) |
| 1927 | 35658eab55 | xinhe-nv | 2025-11-19 | [None][chore] Add failed cases into waives.txt (#9193) |
| 1928 | 7c4777a571 | Enwei Zhu | 2025-11-19 | [TRTLLM-9286][feat] Integration of CuteDSL NVFP4 grouped GEMM (#8880) |
| 1929 | c789000a62 | Lizhi Zhou | 2025-11-19 | [https://nvbugs/5649010][fix] increase status-checking interval to avoid instability (#9203) |
| 1930 | 34f845bf69 | Bo Deng | 2025-11-19 | [TRTLLM-9287][infra] Use NIXL backend for accuracy tests (#9247) |
| 1931 | 8d7cda2318 | Ajinkya Rasane | 2025-11-18 | [None][chore] Update the Flux autodeploy example (#8434) |
| 1932 | 7c4344b92e | Ziyi Xiong | 2025-11-19 | [https://nvbugs/5590408][fix] Exclude num of draft tokens from mMaxSeqLenKv (#9210) |
| 1933 | 3ac11a6180 | Eran Geva | 2025-11-18 | [#9152][fix] AutoDeploy fused_allreduce_residual_rmsnorm to support demollm mode (#9197) |
| 1934 | f0b68e4c66 | Chenghao Zhang | 2025-11-18 | [None][feat] AutoDeploy: Perf improvement for small batch size (#9163) |
| 1935 | fe569f0594 | Nikita Korobov | 2025-11-18 | [None][feat] bias for FP4 TRT-LLM Gen MoE (#9220) |
| 1936 | 04fb481da3 | mpikulski | 2025-11-18 | [TRTLLM-9295][fix] restore greedy sampling in _test_openai_chat_guided_decoding (#9178) |
| 1937 | 36d3d8f608 | Gal Hubara-Agam | 2025-11-18 | [None][chore] Print device info in trtllm-bench report (#8584) |
| 1938 | d076aa44d3 | Kaiyu Xie | 2025-11-19 | [None] [tests] Unwaive wide ep related tests (#9204) |
| 1939 | c4e02d7f04 | Zheyu Fu | 2025-11-18 | [TRTLLM-8136][feat] Dynamic draft length in spec decode (stage 1). (#8194) |
| 1940 | 160b361588 | Ivy Zhang | 2025-11-18 | [TRTLLM-8949][test] Add rcca test case for eagle3 consistency check (#9088) |
| 1941 | 9913dc25ae | Robin Kobus | 2025-11-18 | [None][refactor] decoding inputs, part 2 (#5799) |
| 1942 | ca41a71f92 | Ivy Zhang | 2025-11-18 | [TRTLLM-8948][test] Add long bench case (#9165) |
| 1943 | 8e001dd195 | Chang Liu | 2025-11-18 | [None][fix] DeepSeek V3.2  indexer RoPE fix (#9232) |
| 1944 | 07343bb11c | Lizhi Zhou | 2025-11-18 | [None][chore] fix a deepseekv3 error when debug mode is on (#9217) |
| 1945 | 82480346aa | ruodil | 2025-11-18 | [https://nvbugs/5652552][fix] add printing for llm args (#9205) |
| 1946 | 43896af1b1 | Zero Zeng | 2025-11-18 | [None][chore] benchmark refactor (#9207) |
| 1947 | 96cfdd8a72 | Stanley Sun | 2025-11-18 | [None][chore] Change trt-server to trtlllm-server in opentelemetry readme (#9173) |
| 1948 | 5e5300898b | Gal Hubara-Agam | 2025-11-18 | [#8732][feat] Add ReLU2 to TRTLLM Cutlass MoE BF16 kernels (#9191) |
| 1949 | fd9916424f | TensorRT LLM | 2025-11-18 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1950 | fc088e642c | Tri Dao | 2025-11-17 | [None][feat] Support Glm4MoeForCausalLM (#8256) |
| 1951 | c3376fa114 | QI JUN | 2025-11-18 | [None][ci] split speculative test case into several small cases (#9209) |
| 1952 | 6d0a8edbbb | Lucas Liebenwein | 2025-11-17 | [None][chore] local imports for AutoDeploy in serve and bench (#9199) |
| 1953 | e3c9a97075 | zackyoray | 2025-11-18 | [None][feat] Add TRTLLM_NIXL_KVCACHE_BACKEND environment variable for NIXL backend selection (#9075) |
| 1954 | 2d6289b4b4 | TensorRT LLM | 2025-11-17 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1955 | ec36a3af7e | yuanjingx87 | 2025-11-17 | [None][infra] Fix lock file generation script (#9180) |
| 1956 | 470d777744 | Matt Lefebvre | 2025-11-17 | [TRTINFRA-7280][infra] Support enroot/pyxis clusters in multi-node SLURM and enable oci-hsg GB200 in post-merge (#9117) |
| 1957 | df41f220a2 | Robin Kobus | 2025-11-17 | [TRTLLM-8831][feat] Enable early exit with overlap scheduler (#8587) |
| 1958 | 6151a4c9d6 | Mike Iovine | 2025-11-17 | [None][feat] Add simple optimizations for MTP 2-model (#9176) |
| 1959 | 24f5cd7493 | Yiqing Yan | 2025-11-17 | [TRTLLM-8000][infra] Catch error in merge waive list stage (#7289) |
| 1960 | 04be5a704e | Kaiyu Xie | 2025-11-17 | [None] [fix] Fix missing ActivationType issue (#9171) |
| 1961 | 86cfb3ea7e | Anthony Chang | 2025-11-17 | [None][feat] Update TRTLLM MoE cubins; reduce mxfp4 weight padding requirement; tighten TMA bound (#9025) |
| 1962 | 6dc70aa0e5 | Jinyang Yuan | 2025-11-17 | [https://nvbugs/5613089][fix] Fix the rank to access all_rank_chunk_size_list when chunked MoE is used (#8723) |
| 1963 | d16b1a84c5 | Emma Qiao | 2025-11-17 | [None][infra] Waive a failed case in pre-merge stage 11/16 (#9192) |
| 1964 | 7862b15a65 | sunnyqgg | 2025-11-17 | [TRTLLM-8778][feat] Add tree attention support for blackwell arch (#8975) |
| 1965 | e0f69657c7 | Guoming Zhang | 2025-11-17 | [None][fix] Update the attention layers counting for Qwen3-next. (#9072) |
| 1966 | 2854f0cf3d | Emma Qiao | 2025-11-16 | [None][infra] Waive failed tests for main branch 11/15 (#9187) |
| 1967 | 63237494db | brb-nv | 2025-11-16 | [None][chore] Waive failing tests blocking pre-merge (#9189) |
| 1968 | 3cde84581d | JadoTu | 2025-11-15 | [None][fix] Make the sliced nvfp4 output contiguous (#9123) |
| 1969 | 64cd91ae0a | Thor Johnsen | 2025-11-15 | [None][infra] Add trt-llm-kv-cache-manager-devs as code owner for appropriate files (#9182) |
| 1970 | fe69243157 | Erin | 2025-11-14 | [None][chore] Add placement test for ray executor (#9122) |
| 1971 | bdcf837784 | Zhanrui Sun | 2025-11-15 | [TRTLLM-9079][infra] upgrade tritonserver DLFW 25.10 (#8929) |
| 1972 | 83122bfd64 | yuanjingx87 | 2025-11-14 | [None][infra] Update allowlist 2025.11.14 (#9183) |
| 1973 | 73b8783903 | yuanjingx87 | 2025-11-14 | [None][infra] Fix medata.json generated by lock file genreation pipeline (#9179) |
| 1974 | cbabdae57d | TensorRT LLM | 2025-11-14 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1975 | 05b5336ab6 | yuanjingx87 | 2025-11-14 | [None][infra] Lock generation pipeline update (#9084) |
| 1976 | bed4e95e9f | Chang Liu | 2025-11-14 | [https://nvbugs/5629887][fix] Add missing device count guard for DSv32 multiGPU tests (#9159) |
| 1977 | 49b7e6301a | xinhe-nv | 2025-11-14 | [None][chore] Add failed cases into waives.txt (#9156) |
| 1978 | 80bf840e69 | mpikulski | 2025-11-14 | [TRTLLM-9295][fix] unflake test_overlap_scheduler.py::test_overlap_scheduler_consis… (#9146) |
| 1979 | d72321a32e | yuanjingx87 | 2025-11-14 | [None][ci] Waive unittest/_torch/sampler/test_torch_sampler.py::TestBatchedSampling (#9161) |
| 1980 | f6f6e1f25d | Chenghao Zhang | 2025-11-13 | [#9102][feat] AutoDeploy: Support fp8 kv cache (#9107) |
| 1981 | c6cce398f5 | Zero Zeng | 2025-11-14 | [TRTLLM-9053][feat] Support accuracy test and install from wheel (#9038) |
| 1982 | 84483a238a | dongxuy04 | 2025-11-14 | [None][doc] update docs for EPLB (#9166) |
| 1983 | 25bd2e6917 | Fanrong Li | 2025-11-14 | [None][doc] Add DeepSeek-V3.2-Exp document (#9141) |
| 1984 | 8bd779171e | Lizhi Zhou | 2025-11-14 | [https://nvbugs/5631254][fix] avoid torch.compile for multiple times (#9135) |
| 1985 | e90dbaf572 | TensorRT LLM | 2025-11-14 | [None][infra] Check in most recent lock file from nightly pipeline |
| 1986 | d12cb9436d | Suyog Gupta | 2025-11-13 | [None][feat] Autodeploy add triton configs and optimize mamba prefill (#9083) |
| 1987 | 3c950910a0 | QI JUN | 2025-11-14 | [None][ci] waive test_disaggregated.py::test_disaggregated_mixed[TinyLlama-1.1B-Chat-v1.0] (#9162) |
| 1988 | f07e9977c6 | heyuhhh | 2025-11-14 | [None] [feat] Use triton kernels for RocketKV prediction module (#8682) |
| 1989 | cc4c980e03 | Tailing Yuan | 2025-11-14 | [None][feat] Add Qwen3-Next to layer-wise benchmarks (#9065) |
| 1990 | fdb0787e85 | JunyiXu-nv | 2025-11-14 | [None][chore] Support json_schema in response_format (#8934) |
| 1991 | 44d1c75701 | Erin | 2025-11-13 | [TRTLLM-8988][feat] Unify MPI & Ray's req/response handling with RPC Client/Server (#8765) |
| 1992 | 34dc6869f3 | Neta Zmora | 2025-11-14 | [#8732][feat] Update TRTLLM Cutlass MoE kernels with ReLU2 (#9011) |
| 1993 | a370643b26 | dongxuy04 | 2025-11-14 | [None][fix] support topk autotuner input for expert slot per group larger than 32 (#9087) |
| 1994 | daa31d78f4 | Leslie Fang | 2025-11-14 | [https://nvbugs/5652552][fix] Log the llm args for main branch (#9120) |
| 1995 | b51258acdd | Frida Hou | 2025-11-13 | [None][autodeploy] fix weight extraction for graph based quantized checkpoints (#9109) |
| 1996 | e96a3d294d | Frida Hou | 2025-11-13 | [None][autodeploy] minor refactor to rmsnorm transforms (#8657) |
| 1997 | 12f339f3bf | Jinyang Yuan | 2025-11-14 | [None][fix] Fix the aux_stream in Llama4MinLatencyFusedMoE (#9035) |
| 1998 | 9ef7eb70e0 | Iman Tabrizian | 2025-11-13 | [None][fix] Fix KV cache manager test warnings (#9103) |
| 1999 | a7aaf50541 | Ziyi Xiong | 2025-11-13 | [TRTLLM-8084][feat] Enhance the overlap shceduler for two-model spec decoding (#8706) |
| 2000 | 121140cfec | William Zhang | 2025-11-13 | [None][fixes] Add tool call parsing fixes and Qwen3 coder parser (#8817) |
