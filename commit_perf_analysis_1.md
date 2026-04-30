# Performance Optimization Analysis - Part 1

Commits 1 to 29 of 283

---

## 01423ac183 - [None][feat] perf_metrics endpoint functionality improvement (#8005)

- **Date**: 2025-10-02
- **Author**: Yilin Fan
- **Categories**: General Performance

### Optimization Techniques

- Async/stream-based execution
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Speculative decoding
- Attention mechanism optimization

### Applicable Conditions

- Prefill phase
- Decode/generation phase
- Disaggregated serving

### Changed Files

```
.../tensorrt_llm/batch_manager/llmRequest.h        |  48 +++++--
 cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp  |   8 +-
 .../batch_manager/mlaCacheFormatter.cpp            |   8 +-
 .../nanobind/batch_manager/bindings.cpp            |   3 +-
 cpp/tensorrt_llm/pybind/batch_manager/bindings.cpp |   3 +-
 tensorrt_llm/_torch/pyexecutor/py_executor.py      |  25 +++-
 tensorrt_llm/executor/postproc_worker.py           |  24 +++-
 tensorrt_llm/executor/result.py                    |  12 ++
 tensorrt_llm/serve/openai_disagg_server.py         | 139 ++++++++++++++-------
 tensorrt_llm/serve/openai_server.py                |  81 +++++++++---
 tensorrt_llm/serve/responses_utils.py              |  37 ++++++
 .../defs/disaggregated/test_disaggregated.py       | 125 +++++++++++++++---
 tests/unittest/llmapi/apps/openai_server.py        |   3 +-
 13 files changed, 406 insertions(+), 110 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/include/tensorrt_llm/batch_manager/llmRequest.h b/cpp/include/tensorrt_llm/batch_manager/llmRequest.h
index 275bc7572..670dc0df7 100644
--- a/cpp/include/tensorrt_llm/batch_manager/llmRequest.h
+++ b/cpp/include/tensorrt_llm/batch_manager/llmRequest.h
@@ -101,6 +101,7 @@ public:
     using RequestPtr = std::shared_ptr<GenericLlmRequest>;
     using MillisecondsType = std::chrono::milliseconds;
     using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;
+    using Duration = std::chrono::time_point<std::chrono::steady_clock>::duration;
     using CacheSaltIDType = runtime::CacheSaltIDType;
 
     GenericLlmRequest(RequestIdType requestId, SizeType32 maxNewTokens, std::shared_ptr<VecTokens> const& inputTokens,
@@ -1255,7 +1256,7 @@ public:
     {
         if (mPerfMetrics.timingMetrics.firstScheduledTime == executor::RequestPerfMetrics::TimePoint{})
         {
-            mPerfMetrics.timingMetrics.firstScheduledTime = std::chrono::steady_clock::now();
+            mPerfMetrics.timingMetrics.firstScheduledTime = getSteadyClockNow();
         }
     }
 
@@ -1689,22 +1690,22 @@ public:
         mDecodingIter = iter;
     }
 
-    void setKvCacheTransferStart(std::chrono::time_point<std::chrono::steady_clock> const& time)
+    void setKvCacheTransferStart(TimePoint const& time)
     {
-        mPerfMetrics.timingMetrics.kvCacheTransferStart = time;
+        mPerfMetrics.timingMetrics.kvCacheTransferStart = maybeToGlobalSteadyClock(time);
     }
 
-    void setKvCacheTransferEnd(std::chrono::time_point<std::chrono::steady_clock> const& time)
+    void setKvCacheTransferEnd(TimePoint const& time)
     {
-        mPerfMetrics.timingMetrics.kvCacheTransferEnd = time;
+        mPerfMetrics.timingMetrics.kvCacheTransferEnd = maybeToGlobalSteadyClock(time);
     }
 
-    std::chrono::time_point<std::chrono::steady_clock> getKvCacheTransferStart()
+    TimePoint getKvCacheTransferStart()
     {
         return mPerfMetrics.timingMetrics.kvCacheTransferStart;
     }
 
-    std::chrono::time_point<std::chrono::steady_clock> getKvCacheTransferEnd()
+    TimePoint getKvCacheTransferEnd()
     {
         return mPerfMetrics.timingMetrics.kvCacheTransferEnd;
     }
@@ -1788,7 +1789,7 @@ public:
         if (finishReason == executor::FinishReason::kTIMED_OUT)
         {
             TLLM_LOG_DEBUG("Request %ld finished by timeout after %f sec", mRequestId,
-                std::chrono::duration<float>(std::chrono::steady_clock::now() - mStartTime).count());
+                std::chrono::duration<float>(getSteadyClockNow() - mStartTime).count());
         }
         if (finishReason == executor::FinishReason::kCANCELLED)
         {
@@ -1826,10 +1827,9 @@ public:
 
     void updatePerfMetrics(executor::IterationType iter)
     {
-        auto const currentTokenTime = std::chrono::steady_clock::now();
-
         if (!mPerfMetrics.firstIter)
         {
+            auto const currentTokenTime = getSteadyClockNow();
             mPerfMetr
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## 028fc877a5 - [#9096][feature] Auto Deploy: configurable fused MoE backend (#9194)

- **Date**: 2025-11-20
- **Author**: Neta Zmora
- **Categories**: Fusion

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Triton kernel
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../_torch/auto_deploy/config/default.yaml         |  2 +
 .../auto_deploy/custom_ops/fused_moe/triton_moe.py | 10 +++-
 .../auto_deploy/transform/library/fused_moe.py     | 67 +++++++++++++++++-----
 3 files changed, 65 insertions(+), 14 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/config/default.yaml b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
index 2bd932770..55416141e 100644
--- a/tensorrt_llm/_torch/auto_deploy/config/default.yaml
+++ b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
@@ -114,9 +114,11 @@ transforms:
   fuse_moe:
     stage: post_load_fusion
     enabled: true
+    backend: trtllm
   fuse_fp8_moe:
     stage: post_load_fusion
     enabled: true
+    backend: trtllm
   fuse_allreduce_residual_rmsnorm:
     stage: post_load_fusion
   # TODO (lucaslie): add backend selection as part of configurable inference optimizers
diff --git a/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/triton_moe.py b/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/triton_moe.py
index 24eb6d85c..9dcf54439 100644
--- a/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/triton_moe.py
+++ b/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/triton_moe.py
@@ -341,7 +341,7 @@ def get_moe_configs(
     for config_file_path in config_file_paths:
         if os.path.exists(config_file_path):
             with open(config_file_path) as f:
-                ad_logger.info("Using configuration from %s for MoE layer.", config_file_path)
+                ad_logger.info(f"Using configuration from {config_file_path} for MoE layer.")
                 # If a configuration has been found, return it
                 tuned_config = json.load(f)
                 # Delete triton_version from tuned_config
@@ -601,8 +601,16 @@ def triton_fused_moe(
     routing_weights: torch.Tensor,
     w1_stacked_weight: torch.Tensor,
     w2_stacked_weight: torch.Tensor,
+    mlp_style: str = "mlp",
+    act_fn: str = "relu2",
 ) -> torch.Tensor:
     """Triton unquantized MoE with 2-layer MLP and ReLU^2 activation."""
+
+    mlp_style = mlp_style.lower()
+    act_fn = act_fn.lower()
+    assert mlp_style == "mlp", "Triton backend only supports mlp style."
+    assert act_fn == "relu2", "Triton backend only supports relu2 activation."
+
     x_shape = x.shape
     x2d = x.view(-1, x_shape[-1])
 
diff --git a/tensorrt_llm/_torch/auto_deploy/transform/library/fused_moe.py b/tensorrt_llm/_torch/auto_deploy/transform/library/fused_moe.py
index b62fd9f2b..7a21c4961 100644
--- a/tensorrt_llm/_torch/auto_deploy/transform/library/fused_moe.py
+++ b/tensorrt_llm/_torch/auto_deploy/transform/library/fused_moe.py
@@ -1,25 +1,42 @@
 from collections import defaultdict
-from typing import Dict, List, Optional, Tuple
+from typing import Dict, List, Literal, Optional, Tuple, Type
 
 import torch
+from pydantic import Field
 from torch.fx import GraphModule, Node
 
 from ...models.factory import ModelFactory
 from ...shim.interface import CachedSequenceInterface
 from ...utils.cuda_mem_tracker import cuda_memory_tracker
 from ...utils.node_utils import bfs, extract_op_args, identify_regions_between_residuals, is_op
-from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry
+from ..int
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 03b38e9fbf - [TRTLLM-10030][perf] avoid sync in PyTorchModelEngine when using beam search (#11341)

- **Date**: 2026-02-07
- **Author**: mpikulski
- **Categories**: General Performance

### Optimization Techniques

- Async/stream-based execution
- KV cache optimization
- Pinned memory
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/model_engine.py | 3 ++-
 tensorrt_llm/_torch/pyexecutor/sampler.py      | 2 ++
 2 files changed, 4 insertions(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/model_engine.py b/tensorrt_llm/_torch/pyexecutor/model_engine.py
index 37b6fa1e9..e6ff77c99 100644
--- a/tensorrt_llm/_torch/pyexecutor/model_engine.py
+++ b/tensorrt_llm/_torch/pyexecutor/model_engine.py
@@ -2714,7 +2714,8 @@ class PyTorchModelEngine(ModelEngine):
                 #Copy cache indirection to local buffer with offsets changing:  seq_slots[i] -> i
                 # Convert to GPU tensor to avoid implicit sync
                 gen_request_seq_slots_tensor = torch.tensor(
-                    gen_request_seq_slots, dtype=torch.long, device='cuda')
+                    gen_request_seq_slots, dtype=torch.long,
+                    pin_memory=True).to(device='cuda', non_blocking=True)
                 self.cache_indirection_attention[:num_generation_requests].copy_(
                     cache_indirection_buffer[gen_request_seq_slots_tensor])
             if cache_indirection_buffer is not None or is_cuda_graph_during_warmup:
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index ed9aae6cc..31e56ccb0 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -898,6 +898,8 @@ class AsyncWorkerMixin:
 class TorchSampler(Sampler[SampleStateTorch], AsyncWorkerMixin):
     DEFAULT_MAX_TOPK_LOGPROBS = 20
 
+    SampleState = SampleStateTorch
+
     @override
     def get_cache_indirection(self) -> torch.Tensor | None:
         return self.store.cache_indirection

```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. Pinned (page-locked) memory enables faster host-to-device data transfers. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 03cdf5804f - [None][fix] impl fused triton kernel for e8m0 resmooth to reduce memory footprint (#10327)

- **Date**: 2026-01-16
- **Author**: Necofish
- **Categories**: Kernel Optimization, Memory Optimization, Fusion

### Optimization Techniques

- Custom CUDA kernel
- FP8 quantization
- Batching optimization
- Triton kernel

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/quantization/utils/fp8_utils.py | 118 ++++++++++++++++++---------
 1 file changed, 79 insertions(+), 39 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/quantization/utils/fp8_utils.py b/tensorrt_llm/quantization/utils/fp8_utils.py
index 0776e2d4d..e26288b5b 100644
--- a/tensorrt_llm/quantization/utils/fp8_utils.py
+++ b/tensorrt_llm/quantization/utils/fp8_utils.py
@@ -51,45 +51,85 @@ def per_token_cast_to_fp8_e8m0(
             g, m, n), sf
 
 
-def per_block_cast_to_fp8_e8m0(
-        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
-    if x.dim() == 2:
-        m, n = x.shape
-        x_padded = torch.zeros((align(m, 128), align(n, 128)),
-                               dtype=x.dtype,
-                               device=x.device)
-        x_padded[:m, :n] = x
-        x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
-        x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
-        sf = ceil_to_ue8m0(x_amax / 448.0)
-        x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
-        return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
-            x_view.size(0), x_view.size(2))
-    else:
-        g, m, n = x.shape
-        x_padded = torch.zeros((g, align(m, 128), align(n, 128)),
-                               dtype=x.dtype,
-                               device=x.device)
-        x_padded[:, :m, :n] = x
-        x_view = x_padded.view(g, -1, 128, x_padded.size(-1) // 128, 128)
-        x_amax = x_view.abs().float().amax(dim=(2, 4), keepdim=True).clamp(1e-4)
-        sf = ceil_to_ue8m0(x_amax / 448.0)
-        x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
-        return x_scaled.view_as(x_padded)[:, :m, :n].contiguous(), sf.view(
-            x_view.size(0), x_view.size(1), x_view.size(3))
-
-
-def resmooth_to_fp8_e8m0(weight: torch.Tensor,
-                         sf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
-    weight = weight.cuda()
-    sf = sf.cuda()
-    if weight.dim() == 2:
-        x = weight.float() * sf.repeat_interleave(128, dim=0).repeat_interleave(
-            128, dim=1)[:weight.shape[0], :weight.shape[1]]
-    else:
-        x = weight.float() * sf.repeat_interleave(128, dim=1).repeat_interleave(
-            128, dim=2)[:weight.shape[0], :weight.shape[1], :weight.shape[2]]
-    return per_block_cast_to_fp8_e8m0(x)
+@triton.jit
+def _resmooth_kernel(
+    w_ptr,
+    s_ptr,
+    M,
+    K,
+    stride_wb,
+    stride_wm,
+    stride_wk,
+    stride_sb,
+    stride_sm,
+    stride_sk,
+    BLOCK_M: tl.constexpr,
+    BLOCK_K: tl.constexpr,
+):
+    batch_idx = tl.program_id(0)
+    pid_m = tl.program_id(1)
+    pid_k = tl.program_id(2)
+
+    curr_w_ptr = w_ptr + batch_idx * stride_wb
+    curr_s_ptr = s_ptr + batch_idx * stride_sb
+
+    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
+    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
+
+    s_offset = pid_m * stride_sm + pid_k * stride_sk
+    old_scale = tl.load(curr_s_ptr + s_offset)
+
+    w_mask = (rm[:, None] < M) & (rk[None, :] < K)
+    w_offsets = rm[:, None] * stride_wm + rk[None, :] * stride_wk
+    w
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 040103ab56 - [None] [blog] Scaling Expert Parallelism in TensorRT LLM (Part 3: Pushing the Performance Boundary) (#8323)

- **Date**: 2025-10-13
- **Author**: Kaiyu Xie
- **Categories**: Parallelism/Async

### Optimization Techniques

- Torch compilation/JIT optimization
- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Reduce synchronization overhead
- Speculative decoding
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- NVIDIA Hopper (SM90) GPU
- Prefill phase
- Decode/generation phase

### Changed Files

```
.../blogs/media/tech_blog14_MTP_parallel_1.png     | Bin 0 -> 241965 bytes
 .../blogs/media/tech_blog14_MTP_parallel_2.png     | Bin 0 -> 362839 bytes
 .../blogs/media/tech_blog14_alltoall_dataflow.png  | Bin 0 -> 79301 bytes
 .../blogs/media/tech_blog14_overview_after_opt.png | Bin 0 -> 200435 bytes
 .../media/tech_blog14_overview_before_opt.png      | Bin 0 -> 194884 bytes
 docs/source/blogs/media/tech_blog14_pdloff.png     | Bin 0 -> 153624 bytes
 docs/source/blogs/media/tech_blog14_pdlon.png      | Bin 0 -> 172040 bytes
 docs/source/blogs/media/tech_blog14_perf.png       | Bin 0 -> 409669 bytes
 ...ing_Expert_Parallelism_in_TensorRT-LLM_part3.md | 239 +++++++++++++++++++++
 9 files changed, 239 insertions(+)
```

### Diff Preview

```diff
diff --git a/docs/source/blogs/media/tech_blog14_MTP_parallel_1.png b/docs/source/blogs/media/tech_blog14_MTP_parallel_1.png
new file mode 100644
index 000000000..503c8abf1
Binary files /dev/null and b/docs/source/blogs/media/tech_blog14_MTP_parallel_1.png differ
diff --git a/docs/source/blogs/media/tech_blog14_MTP_parallel_2.png b/docs/source/blogs/media/tech_blog14_MTP_parallel_2.png
new file mode 100644
index 000000000..b33c9b9d1
Binary files /dev/null and b/docs/source/blogs/media/tech_blog14_MTP_parallel_2.png differ
diff --git a/docs/source/blogs/media/tech_blog14_alltoall_dataflow.png b/docs/source/blogs/media/tech_blog14_alltoall_dataflow.png
new file mode 100644
index 000000000..183505fe6
Binary files /dev/null and b/docs/source/blogs/media/tech_blog14_alltoall_dataflow.png differ
diff --git a/docs/source/blogs/media/tech_blog14_overview_after_opt.png b/docs/source/blogs/media/tech_blog14_overview_after_opt.png
new file mode 100644
index 000000000..0e9923213
Binary files /dev/null and b/docs/source/blogs/media/tech_blog14_overview_after_opt.png differ
diff --git a/docs/source/blogs/media/tech_blog14_overview_before_opt.png b/docs/source/blogs/media/tech_blog14_overview_before_opt.png
new file mode 100644
index 000000000..aa4665600
Binary files /dev/null and b/docs/source/blogs/media/tech_blog14_overview_before_opt.png differ
diff --git a/docs/source/blogs/media/tech_blog14_pdloff.png b/docs/source/blogs/media/tech_blog14_pdloff.png
new file mode 100644
index 000000000..3899ad153
Binary files /dev/null and b/docs/source/blogs/media/tech_blog14_pdloff.png differ
diff --git a/docs/source/blogs/media/tech_blog14_pdlon.png b/docs/source/blogs/media/tech_blog14_pdlon.png
new file mode 100644
index 000000000..ce5684160
Binary files /dev/null and b/docs/source/blogs/media/tech_blog14_pdlon.png differ
diff --git a/docs/source/blogs/media/tech_blog14_perf.png b/docs/source/blogs/media/tech_blog14_perf.png
new file mode 100644
index 000000000..9c73b60e2
Binary files /dev/null and b/docs/source/blogs/media/tech_blog14_perf.png differ
diff --git a/docs/source/blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.md b/docs/source/blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.md
new file mode 100644
index 000000000..4b80603e2
--- /dev/null
+++ b/docs/source/blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.md
@@ -0,0 +1,239 @@
+# Scaling Expert Parallelism in TensorRT LLM (Part 3: Pushing the Performance Boundary)
+
+This blog post is a continuation of previous posts:
+* [Scaling Expert Parallelism in TensorRT LLM (Part 1: Design and Implementation of Large-scale EP)](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md)
+* [Scaling Expert Parallelism in TensorRT LLM (Part 2: Performance Status and Optimization)](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog8_Scaling_Expert_Par
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## 040f4c70d3 - [None][perf] Accelerate global scale calculations for deepEP fp4 combine (#7126)

- **Date**: 2025-08-27
- **Author**: Void
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- Batching optimization
- Reduce synchronization overhead
- MoE optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU

### Changed Files

```
cpp/tensorrt_llm/kernels/quantization.cu           |  97 +++++++++
 cpp/tensorrt_llm/kernels/quantization.h            |   4 +
 cpp/tensorrt_llm/thop/fp4Quantize.cpp              |  79 ++++++++
 cpp/tensorrt_llm/thop/fp4Quantize.h                |   4 +-
 tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py   |   4 +
 .../_torch/modules/fused_moe/fused_moe_wide_ep.py  |   4 +-
 .../_torch/thop/test_fp4_calculate_global_scale.py | 223 +++++++++++++++++++++
 7 files changed, 412 insertions(+), 3 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/quantization.cu b/cpp/tensorrt_llm/kernels/quantization.cu
index 817b0a57e..78248214c 100644
--- a/cpp/tensorrt_llm/kernels/quantization.cu
+++ b/cpp/tensorrt_llm/kernels/quantization.cu
@@ -302,6 +302,98 @@ void invokeBlockScaleInterleaveReverse(
     block_scale_interleave_reverse_kernel<<<grid, block, 0, stream>>>(b, m, n, SFIn, SFOutput);
 }
 
+template <typename T>
+struct VecTypeImpl
+{
+    using type = T;
+};
+
+template <>
+struct VecTypeImpl<half>
+{
+    using type = half2;
+};
+
+template <>
+struct VecTypeImpl<__nv_bfloat16>
+{
+    using type = __nv_bfloat162;
+};
+
+template <typename T>
+using VecType = typename VecTypeImpl<T>::type;
+
+template <typename T>
+__device__ float getMaxAbs(float4& vec)
+{
+    auto absMaxVec = cuda_abs(reinterpret_cast<VecType<T>*>(&vec)[0]);
+    for (int i = 1; i < 4; ++i)
+    {
+        absMaxVec = cuda_max(absMaxVec, cuda_abs(reinterpret_cast<VecType<T>*>(&vec)[i]));
+    }
+    float absMaxVal;
+    if constexpr (sizeof(T) == 4)
+    {
+        absMaxVal = static_cast<float>(absMaxVec);
+    }
+    else
+    {
+        absMaxVal = static_cast<float>(cuda_max(absMaxVec.x, absMaxVec.y));
+    }
+    tensorrt_llm::common::blockReduceMaxV2<float, 1>(&absMaxVal);
+    return absMaxVal;
+}
+
+template <typename T>
+__global__ void computePerTokenGlobalScaleForFP4QuantizationKernel(
+    int b, int m, int n, T const* input, int const* tokensPerBatch, float* globalScale)
+{
+    static constexpr int ElemsPerVec = 16 / sizeof(T);
+    int batchIdx = blockIdx.x;
+    int realTokensNum = (tokensPerBatch == nullptr) ? m : tokensPerBatch[batchIdx];
+    input += batchIdx * m * n;
+    globalScale += batchIdx * m;
+    for (int tokenIdx = blockIdx.y; tokenIdx < realTokensNum; tokenIdx += gridDim.y)
+    {
+        float perTokenMaxAbsVal = 0.f;
+        for (int vecIdx = threadIdx.x; vecIdx < n / ElemsPerVec; vecIdx += blockDim.x)
+        {
+            float4 vec = reinterpret_cast<float4 const*>(input + tokenIdx * n)[vecIdx];
+            float maxAbsVal = getMaxAbs<T>(vec);
+            perTokenMaxAbsVal = cuda_max(perTokenMaxAbsVal, maxAbsVal);
+        }
+        float globalScaleVal = 448.f * 6.f / perTokenMaxAbsVal;
+        if (threadIdx.x == 0)
+        {
+            globalScale[tokenIdx] = globalScaleVal;
+        }
+    }
+}
+
+template <typename T>
+void computePerTokenGlobalScaleForFP4Quantization(int b, int m, int n, T const* input, int const* tokensPerBatch,
+    float* globalScale, int multiProcessorCount, cudaStream_t stream)
+{
+
+    static constexpr int ElemsPerVec = 16 / sizeof(T);
+    TLLM_CHECK(n % (ElemsPerVec * 32) == 0 and b > 0);
+    dim3 block(std::min(n / ElemsPerVec, 1024));
+    dim3 grid(b, std::max(1, std::min(m, multiProcessorCount / b)));
+
+    cudaLaunchConfig_t config;
+    config.gridDim = grid;
+    config.blockDim = block;
+    config.dynamicSmemBytes = 0;
+    config.stream = stream;
+    cudaLaunchAttribute attrs[1];
+    att
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 0788c5d0d6 - [perf] improve XQA-MLA perf (#5468)

- **Date**: 2025-06-26
- **Author**: Yao Yao
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Batching optimization
- Attention mechanism optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU

### Changed Files

```
cpp/kernels/xqa/barriers.cuh       |  18 ++--
 cpp/kernels/xqa/mha_components.cuh |   4 +-
 cpp/kernels/xqa/mha_sm90.cu        |   2 +-
 cpp/kernels/xqa/mla_sm120.cu       | 207 ++++++++++++++++++-------------------
 cpp/kernels/xqa/mla_sm120.cuh      |   4 +-
 cpp/kernels/xqa/test/test.cpp      |   1 -
 cpp/kernels/xqa/tma.h              |  27 +++--
 7 files changed, 135 insertions(+), 128 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/kernels/xqa/barriers.cuh b/cpp/kernels/xqa/barriers.cuh
index 3c0318be4..d21ba7e86 100644
--- a/cpp/kernels/xqa/barriers.cuh
+++ b/cpp/kernels/xqa/barriers.cuh
@@ -68,7 +68,7 @@ public:
     template <Scope scope = defaultScope, ArriveOrder order = ArriveOrder::RELEASE>
     __device__ inline mha::conditional_t<scope == Scope::CTA, ArrivalToken, void> arrive(uint32_t update = 1)
     {
-        ArrivalToken token;
+        ArrivalToken token{};
 #if __CUDA_ARCH__ >= 900
         if constexpr (scope == Scope::CTA)
         {
@@ -128,9 +128,9 @@ public:
 
     __device__ inline bool isLocal() const
     {
-        uint32_t addrCtaRank;
+        uint32_t addrCtaRank{};
         asm("getctarank.u64 %0, %1;\n" : "=r"(addrCtaRank) : "l"(addr()));
-        uint32_t ctaRank;
+        uint32_t ctaRank{};
         asm("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(ctaRank));
         return addrCtaRank == ctaRank;
     }
@@ -154,7 +154,7 @@ public:
 #if __CUDA_ARCH__ >= 900
         if constexpr (scope == Scope::CTA)
         {
-            ArrivalToken token;
+            ArrivalToken token{};
             asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.b64 %0, [%1], %2;\n"
                          : "=l"(token)
                          : "l"(addr()), "r"(txCount)
@@ -181,7 +181,7 @@ public:
         {
             if constexpr (scope == Scope::CTA)
             {
-                ArrivalToken token;
+                ArrivalToken token{};
                 switch (order)
                 {
                 case ArriveOrder::RELEASE:
@@ -239,7 +239,7 @@ public:
     template <Scope scope = defaultScope>
     __device__ inline bool test_wait(ArrivalToken&& token)
     {
-        uint32_t ready;
+        uint32_t ready{};
         if constexpr (scope == Scope::CGA)
         {
             asm volatile(
@@ -271,7 +271,7 @@ public:
     template <Scope scope = defaultScope>
     __device__ inline bool test_wait_parity(bool parity)
     {
-        uint32_t ready;
+        uint32_t ready{};
         if constexpr (scope == Scope::CGA)
         {
             asm volatile(
@@ -303,7 +303,7 @@ public:
     template <Scope scope = defaultScope>
     __device__ inline bool try_wait(ArrivalToken&& token)
     {
-        uint32_t ready;
+        uint32_t ready{};
         if constexpr (scope == Scope::CGA)
         {
             asm volatile(
@@ -334,7 +334,7 @@ public:
     template <Scope scope = defaultScope>
     __device__ inline bool try_wait_parity(bool parity)
     {
-        uint32_t ready;
+        uint32_t ready{};
         if constexpr (scope == Scope::CGA)
         {
             asm volatile(
diff --git a/cpp/kernels/xqa/mha_components.cuh b/cpp/kernels/xqa/mha_components.cuh
index a2b8619a0..4f4006ff7 100644
--- a/cpp/kernels/xqa/mha_components.cuh
+++ b/cpp/kernels/xqa/mha_components.cuh
@@ -59,7 +59,7 @@ template <uint32_t n>
 __device__ inline QuadRegRowMaxT<n * warp_size> replicateForQuad(Warp const& warp, Vec<float, n> const& src)

```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 09d9878385 - [TRTLLM-9661][chore] Further reduce tuning time for cuteDSL nvFP4 dense gemm. (#10339)

- **Date**: 2026-01-08
- **Author**: Yukun He
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../_torch/custom_ops/cute_dsl_custom_ops.py       | 35 +++++++++++++++-------
 1 file changed, 25 insertions(+), 10 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py b/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
index ae61e2b64..771e7ed7c 100644
--- a/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
+++ b/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
@@ -6,7 +6,7 @@ import torch
 from tensorrt_llm.logger import logger
 
 from ..._utils import get_sm_version
-from ...math_utils import pad_up
+from ...math_utils import ceil_div, pad_up
 from ..autotuner import (AutoTuner, ConstraintSpec, DistributedTuningStrategy,
                          DynamicTensorSpec, OptimizationProfile, TunableRunner,
                          TuningConfig)
@@ -314,6 +314,16 @@ class GatherGroupedGemmInputsHelper(GroupedGemmInputsHelper):
                 num_non_exiting_tiles, global_sf)
 
 
+def get_dense_gemm_approximate_cta_nums(
+        M: int, N: int, tile_mn: Tuple[int, int],
+        cluster_shape_mn: Tuple[int, int]) -> int:
+    tile_m, tile_n = tile_mn
+    cluster_m, cluster_n = cluster_shape_mn
+    clustered_ctas_m = pad_up(ceil_div(M, tile_m), cluster_m)
+    clustered_ctas_n = pad_up(ceil_div(N, tile_n), cluster_n)
+    return clustered_ctas_m * clustered_ctas_n
+
+
 if IS_CUTLASS_DSL_AVAILABLE:
 
     import cutlass
@@ -360,15 +370,6 @@ if IS_CUTLASS_DSL_AVAILABLE:
         def unique_id(self):
             return (self.output_dtype, self.to_userbuffers, self.use_tvm_ffi)
 
-        def __hash__(self):
-            return hash(
-                (self.output_dtype, self.to_userbuffers, self.use_tvm_ffi))
-
-        def __eq__(self, other):
-            if not isinstance(other, self.__class__):
-                return False
-            return self.output_dtype == other.output_dtype and self.to_userbuffers == other.to_userbuffers and self.use_tvm_ffi == other.use_tvm_ffi
-
         def get_valid_tactics(
             self,
             inputs: List[torch.Tensor],
@@ -454,6 +455,7 @@ if IS_CUTLASS_DSL_AVAILABLE:
                 (4, 4),
             ]
             swap_ab_candidates = [True, False]
+            # prune: prefetch is beneficial only when K is large enough
             use_prefetch_candidates = [True, False]
 
             valid_tactics = []
@@ -484,6 +486,19 @@ if IS_CUTLASS_DSL_AVAILABLE:
                         b_major,
                         c_major,
                 ):
+                    # Prefetch pruning to save tuning time
+                    cta_nums = get_dense_gemm_approximate_cta_nums(
+                        m, n, mma_tiler_mn, cluster_shape_mn)
+                    cta_wave_ratio = cta_nums / torch.cuda.get_device_properties(
+                    ).multi_processor_count
+                    if use_prefetch and not any((
+                            # CTA waves ratio between 0.5 and 1.0
+                            0.5 < cta_wave_ratio < 1.0,
+                            # K is large enough
+                            real_k >= 8192,
+                    )):
+                        continue
+
    
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 0a0f93d4a8 - [None][fix] Fix the performance issue of FP8 blockwise grouped GEMM when using attention DP (#8501)

- **Date**: 2025-10-27
- **Author**: Jinyang Yuan
- **Categories**: Quantization Optimization

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- Parallelism optimization
- Batching optimization
- Reduce synchronization overhead
- MoE optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU

### Changed Files

```
.../mixtureOfExpertsBackendBenchmarkFixture.h      |  8 +-
 .../fp8_blockscale_gemm/fp8_blockscale_gemm.cu     | 19 +++--
 .../fp8_blockscale_gemm/fp8_blockscale_gemm.h      |  9 +++
 .../kernels/cutlass_kernels/include/moe_kernels.h  | 88 +++++++++++----------
 .../cutlass_kernels/moe_gemm/moe_kernels.cu        | 81 ++++++++++---------
 .../internal_cutlass_kernels/include/moe_kernels.h | 92 +++++++++++-----------
 .../mixtureOfExperts/mixtureOfExpertsPlugin.cpp    |  4 +-
 cpp/tensorrt_llm/thop/fp8BlockScalingGemm.cpp      |  4 +-
 cpp/tensorrt_llm/thop/moeOp.cpp                    | 28 ++++---
 .../unit_tests/kernels/mixtureOfExpertsTest.cu     |  8 +-
 tensorrt_llm/_torch/custom_ops/torch_custom_ops.py |  1 +
 .../_torch/modules/fused_moe/ops/moe_op_cutlass.py |  1 +
 12 files changed, 192 insertions(+), 151 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h b/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h
index b54aec107..f466a65e8 100644
--- a/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h
+++ b/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h
@@ -994,8 +994,8 @@ public:
                 mUseFinalScale ? mScaleProbs + mScaleProbsSize * mBufferIndex : nullptr,
                 mExpertWeight1 + mExpertWeight1Size * mBufferIndex, mExpertBias1 + mExpertBias1Size * mBufferIndex,
                 ActivationParams(mActType), mExpertWeight2 + mExpertWeight2Size * mBufferIndex,
-                mExpertBias2 + mExpertBias2Size * mBufferIndex, mQuantParams[mBufferIndex], mTotalTokens, mHiddenSize,
-                mHiddenSize, mInterSize, mNumExperts, mK,
+                mExpertBias2 + mExpertBias2Size * mBufferIndex, mQuantParams[mBufferIndex], mTotalTokens, mTotalTokens,
+                mHiddenSize, mHiddenSize, mInterSize, mNumExperts, mK,
                 mWorkspace + mWorkspaceSize * (mBufferIndex % mNumWorkspaceBuffers),
                 mFinalOutput + mFinalOutputSize * (mBufferIndex % mNumInputBuffers),
                 mSourceToExpandedMap + mSourceToExpandedMapSize * mBufferIndex, parallelism_config,
@@ -1007,8 +1007,8 @@ public:
                 mUseFinalScale ? mScaleProbs + mScaleProbsSize * mBufferIndex : nullptr,
                 mExpertWeight1 + mExpertWeight1Size * mBufferIndex, mExpertBias1 + mExpertBias1Size * mBufferIndex,
                 ActivationParams(mActType), mExpertWeight2 + mExpertWeight2Size * mBufferIndex,
-                mExpertBias2 + mExpertBias2Size * mBufferIndex, mQuantParams[mBufferIndex], mTotalTokens, mHiddenSize,
-                mHiddenSize, mInterSize, mNumExperts, mK,
+                mExpertBias2 + mExpertBias2Size * mBufferIndex, mQuantParams[mBufferIndex], mTotalTokens, mTotalTokens,
+                mHiddenSize, mHiddenSize, mInterSize, mNumExperts, mK,
                 mWorkspace + mWorkspaceSize * (mBufferIndex % mNumWorkspaceBuffers),
                 mFinalOutput + mFinalOutputSize * (mBufferIndex % mNumInputBuffers),
                 mSourceToExpandedMap + mSourceToExpandedMapSize * mBufferIndex, parallelism_config,
diff --git a/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu b/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu
index 958760203..c6a22c0f7 100644
--- a/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu
+++ b/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu
@@ -88,8 +88,8 @@ void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::gemm(__nv_fp8
 
 template <typename ElementA, typename ElementB, typename ElementD>
 void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::moeGemm(void* mat_d, void const* mat_a,
-    void const* mat_b, int64_t const* problem_m_offsets,
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 0b81173efa - [TRTLLM-9259][perf] Use torch.compile to fuse copy + layernorm within the LayerNorm module (#9052)

- **Date**: 2025-11-11
- **Author**: Chang Liu
- **Categories**: Fusion

### Optimization Techniques

- General code optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/modules/layer_norm.py | 3 +++
 1 file changed, 3 insertions(+)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/layer_norm.py b/tensorrt_llm/_torch/modules/layer_norm.py
index 8db0af7e9..811067952 100644
--- a/tensorrt_llm/_torch/modules/layer_norm.py
+++ b/tensorrt_llm/_torch/modules/layer_norm.py
@@ -18,6 +18,8 @@ from typing import Optional, Tuple, Union
 import torch
 from torch import nn
 
+from ..utils import maybe_compile
+
 
 class LayerNorm(nn.Module):
     """Layer normalization module with configurable weight and bias parameters.
@@ -65,6 +67,7 @@ class LayerNorm(nn.Module):
                                  persistent=False)
         self.variance_epsilon = eps
 
+    @maybe_compile(dynamic=True)
     def forward(
         self,
         hidden_states: torch.Tensor,

```

### Analysis Summary

This commit applies performance optimization techniques to improve inference efficiency. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 0c19c8e96e - [None][test] Add e2e tests for KV cache connector async loading path (#12053)

- **Date**: 2026-03-12
- **Author**: Iman Tabrizian
- **Categories**: Parallelism/Async, Cache Optimization

### Optimization Techniques

- KV cache optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
.../defs/llmapi/test_llm_api_connector.py          | 69 +++++++++++++++++++++-
 tests/integration/test_lists/test-db/l0_a10.yml    |  1 +
 2 files changed, 69 insertions(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tests/integration/defs/llmapi/test_llm_api_connector.py b/tests/integration/defs/llmapi/test_llm_api_connector.py
index 78fd79840..8a146d0d5 100644
--- a/tests/integration/defs/llmapi/test_llm_api_connector.py
+++ b/tests/integration/defs/llmapi/test_llm_api_connector.py
@@ -1,4 +1,4 @@
-# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 # SPDX-License-Identifier: Apache-2.0
 #
 # Licensed under the Apache License, Version 2.0 (the "License");
@@ -14,6 +14,9 @@
 # limitations under the License.
 
 import math
+import os
+import sys
+import tempfile
 import time
 from unittest.mock import MagicMock, patch
 
@@ -537,3 +540,67 @@ def test_connector_priorities_default(enforce_single_worker,
 
     # Without retention config, priorities should be None
     assert request.priorities is None
+
+
+@pytest.mark.threadleak(enabled=False)
+def test_connector_e2e_persistent_cache(enforce_single_worker):
+    """Test e2e KV cache connector using PersistentKvCacheConnector from examples.
+
+    Runs generation twice with separate LLM instances sharing a disk-based
+    connector cache, verifying that outputs are identical (proving cache
+    save/load works end-to-end).
+    """
+    examples_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..",
+                                "..", "examples", "llm-api")
+    examples_dir = os.path.abspath(examples_dir)
+    sys.path.insert(0, examples_dir)
+
+    cache_dir = tempfile.mkdtemp()
+    os.environ["CONNECTOR_CACHE_FOLDER"] = cache_dir
+
+    try:
+        kv_connector_config = KvCacheConnectorConfig(
+            connector_module="llm_kv_cache_connector",
+            connector_scheduler_class="PersistentKvCacheConnectorLeader",
+            connector_worker_class="PersistentKvCacheConnectorWorker",
+        )
+
+        llm_kwargs = dict(
+            model=f"{llm_models_root()}/Qwen2-0.5B",
+            backend="pytorch",
+            kv_connector_config=kv_connector_config,
+            cuda_graph_config=None,
+            disable_overlap_scheduler=True,
+            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.1),
+        )
+
+        prompt = (
+            "Nvidia Corporation is an American technology company "
+            "headquartered in Santa Clara, California. Founded in 1993 by "
+            "Jensen Huang, Chris Malachowsky, and Curtis Priem, it develops "
+            "graphics processing units (GPUs), system on a chips (SoCs), and "
+            "application programming interfaces (APIs) for data science, "
+            "high-performance computing, and mobile and automotive "
+            "applications. Tell me about the company.")
+
+        sampling_params = SamplingParams(max_tokens=32, ignore_eos=True)
+
+        llm1 = LLM(**llm_kwargs)
+        output1 = llm1.generate([prompt], sampling_params)
+   
```

### Analysis Summary

This commit applies performance optimization techniques to improve inference efficiency. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 0c31502fbc - [None][feat] disable fused gemm for sm121 (#9916)

- **Date**: 2025-12-15
- **Author**: Faraz
- **Categories**: Fusion

### Optimization Techniques

- Operator fusion
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/models/modeling_deepseekv3.py | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_deepseekv3.py b/tensorrt_llm/_torch/models/modeling_deepseekv3.py
index 8df4eae70..605972ab5 100755
--- a/tensorrt_llm/_torch/models/modeling_deepseekv3.py
+++ b/tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -518,7 +518,7 @@ class DeepseekV3Linear(Linear):
                      layer_idx: Optional[int] | None = None):
         num_tokens = input.shape[0]
         if (not self.has_any_quant and 1 <= num_tokens <= 16
-                and get_sm_version() != 120):
+                and get_sm_version() not in [120, 121]):
             output = torch.ops.trtllm.dsv3_fused_a_gemm_op(
                 input, self.weight.t(), bias, None)
         else:

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 0ce36516b9 - [TRTLLM-11257][infra] Unwaive TestDeepSeekR1::test_fp8_blockscale[throughput_mtp] test case (#12059)

- **Date**: 2026-03-12
- **Author**: zhaoyangwang-nvidia
- **Categories**: Throughput/Latency, Quantization Optimization

### Optimization Techniques

- Operator fusion
- FP8 quantization
- KV cache optimization
- Speculative decoding
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tests/integration/test_lists/waives.txt | 1 -
 1 file changed, 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tests/integration/test_lists/waives.txt b/tests/integration/test_lists/waives.txt
index dea39e9fd..26f0ad5cc 100644
--- a/tests/integration/test_lists/waives.txt
+++ b/tests/integration/test_lists/waives.txt
@@ -259,7 +259,6 @@ accuracy/test_llm_api_autodeploy.py::TestNemotronSuperV3::test_accuracy[bf16-4-a
 cpp/test_multi_gpu.py::test_cache_transceiver[8proc-mooncake_kvcache-90] SKIP (https://nvbugs/5838199)
 accuracy/test_llm_api_pytorch.py::TestGPTOSS::test_w4_4gpus[v1_kv_cache-dp4-cutlass-auto] SKIP (https://nvbugs/5838211)
 accuracy/test_llm_api_pytorch.py::TestGPTOSS::test_w4_4gpus[v2_kv_cache-dp4-cutlass-auto] SKIP (https://nvbugs/5838211)
-accuracy/test_llm_api_pytorch.py::TestDeepSeekR1::test_fp8_blockscale[throughput_mtp] SKIP (https://nvbugs/5839028)
 full:A10/unittest/kv_cache_manager_v2_tests/ SKIP (https://nvbugs/5841954)
 unittest/_torch/modules/test_fused_moe.py::test_fused_moe_alltoall_fp4[DeepEP] SKIP (https://nvbugs/5841976)
 unittest/_torch/modeling/test_modeling_nemotron_h.py::test_nemotron_h_cuda_graph_overlap_scheduler SKIP (https://nvbugs/5843316)

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 0dc4b4e699 - [#4403][autodeploy] Refactor: Move more transformations to new inf optimizer, Add quantization_source to factory interface (#6760)

- **Date**: 2025-08-11
- **Author**: Fridah-nv
- **Categories**: Quantization Optimization

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- PyTorch built-in optimized ops
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../_torch/auto_deploy/config/default.yaml         |  36 +-
 tensorrt_llm/_torch/auto_deploy/models/hf.py       |  61 +-
 .../auto_deploy/models/quant_config_reader.py      | 130 ++++
 .../_torch/auto_deploy/transform/interface.py      |  26 +-
 .../auto_deploy/transform/library/attention.py     |  32 +-
 .../auto_deploy/transform/library/build_model.py   |  14 +-
 .../transform/library/cleanup_input_constraints.py |   8 +-
 .../transform/library/cleanup_noop_add.py          |   8 +-
 .../transform/library/cleanup_noop_slice.py        |   8 +-
 .../library/eliminate_redundant_transposes.py      | 124 ++++
 .../auto_deploy/transform/library/export_to_gm.py  |  14 +-
 .../auto_deploy/transform/library/fused_moe.py     | 523 +++++++++++++++
 .../auto_deploy/transform/library/load_weights.py  |  54 ++
 .../auto_deploy/transform/library/quantization.py  | 105 +--
 .../auto_deploy/transform/library/quantize_moe.py  |   8 +-
 .../{transformations => transform}/library/rope.py | 517 ++++++++-------
 .../auto_deploy/transform/library/sharding.py      | 452 +++++++++++++
 .../_torch/auto_deploy/transform/optimizer.py      |  10 +-
 .../transformations/library/__init__.py            |   3 -
 .../library/eliminate_redundant_transposes.py      | 112 ----
 .../auto_deploy/transformations/transform.py       |  86 +--
 .../sharding.py => utils/sharding_utils.py}        | 716 +++++----------------
 .../auto_deploy/_utils_test/_graph_test_helpers.py |   2 +-
 .../transformations/library/test_bmm_sharding.py   |  46 +-
 .../transformations/library/test_ep_sharding.py    |  45 +-
 .../transformations/library/test_tp_sharding.py    |  43 +-
 .../transformations/library/test_moe_fusion.py     |  41 +-
 .../transformations/library/test_quantization.py   |   4 +-
 .../library/test_redundant_transposes.py           |  20 +-
 .../library/test_rope_transformation.py            | 152 +++--
 30 files changed, 2189 insertions(+), 1211 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/config/default.yaml b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
index af6f130ce..f7ad7934a 100644
--- a/tensorrt_llm/_torch/auto_deploy/config/default.yaml
+++ b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
@@ -19,10 +19,6 @@ transforms:
     stage: post_export
   cleanup_input_constraints:
     stage: post_export
-  quantize:
-    stage: pattern_matcher
-  quantize_moe:
-    stage: pattern_matcher
   match_repeat_kv:
     stage: pattern_matcher
   match_eager_attention:
@@ -31,3 +27,35 @@ transforms:
     stage: pattern_matcher
   match_attention_layout:
     stage: pattern_matcher
+  match_moe_pattern:
+    stage: pattern_matcher
+  match_rope_pattern:
+    stage: pattern_matcher
+  match_rope_layout:
+    stage: pattern_matcher
+  eliminate_redundant_transposes:
+    stage: pattern_matcher
+  # TODO (lucaslie): let's move this to perf optimization once TP sharding is improved
+  # see https://github.com/NVIDIA/TensorRT-LLM/pull/3668#discussion_r2052714528
+  optimize_rope:
+    stage: pattern_matcher
+  quantize_from_config:
+    stage: pattern_matcher
+  quantize_from_graph:
+    stage: pattern_matcher
+  quantize_moe:
+    stage: pattern_matcher
+  # TODO: Infer sharding parameters (tp_size, row/column sharding) from the model config.
+  detect_column_row_shard:
+    stage: sharding
+    simple_shard_only: false
+  detect_ep_shard:
+    stage: sharding
+  detect_dp_bmm_shard:
+    stage: sharding
+  # TODO: (hg) need to ensure run_shape_prop after sharding.
+  sharding_transform_executor:
+    stage: sharding
+    run_shape_prop: true
+  load_weights:
+    stage: weight_load
diff --git a/tensorrt_llm/_torch/auto_deploy/models/hf.py b/tensorrt_llm/_torch/auto_deploy/models/hf.py
index fc37c1e55..36d4dc333 100644
--- a/tensorrt_llm/_torch/auto_deploy/models/hf.py
+++ b/tensorrt_llm/_torch/auto_deploy/models/hf.py
@@ -1,6 +1,5 @@
 """Interface to initialize and load HF models."""
 
-import json
 import os
 import types
 from contextlib import contextmanager, nullcontext
@@ -31,6 +30,7 @@ from ..custom_ops.attention_interface import CacheConfig
 from ..utils._config import deep_merge_dicts
 from ..utils.logger import ad_logger
 from .factory import ModelFactory, ModelFactoryRegistry
+from .quant_config_reader import QuantConfigReader, QuantConfigReaderRegistry
 
 
 @contextmanager
@@ -84,9 +84,7 @@ class AutoModelForCausalLMFactory(ModelFactory):
 
     def __init__(self, *args, **kwargs):
         super().__init__(*args, **kwargs)
-
-        self._quant_config: Optional[Dict] = None
-
+        self._quant_config_reader: QuantConfigReader | None = None
         # Ingest defaults for tokenizer and model kwargs
         self.tokenizer_kwargs = deep_merge_dicts(self._tokenizer_defaults, self.tokenizer_kwargs)
         self.model_kwargs = deep_merge_dicts(
@@ -156,9 +154,6 @@ class AutoModelForCausalLMFactory(ModelFactory):
 
     def _build_model(self, device: DeviceLikeType) -> nn.Modul
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 0dcf47f1c2 - [TRTLLM-4717][perf] Set CUDA graph max batch size and padding in throughput benchmark. (#3875)

- **Date**: 2025-05-09
- **Author**: Frank
- **Categories**: Throughput/Latency

### Optimization Techniques

- KV cache optimization
- Batching optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/bench/benchmark/utils/general.py | 2 ++
 1 file changed, 2 insertions(+)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/bench/benchmark/utils/general.py b/tensorrt_llm/bench/benchmark/utils/general.py
index 6cb86f769..c8c70b6f4 100755
--- a/tensorrt_llm/bench/benchmark/utils/general.py
+++ b/tensorrt_llm/bench/benchmark/utils/general.py
@@ -135,8 +135,10 @@ def get_settings(params: dict, dataset_metadata: DatasetMetadata, model: str,
 
     pyt_options = {
         "use_cuda_graph": True,
+        "cuda_graph_padding_enabled": True,
         "enable_overlap_scheduler": True,
         "kv_cache_dtype": kv_cache_dtype,
+        "cuda_graph_max_batch_size": max_batch_size,
     }
     backend = params.get("backend", "pytorch")
 

```

### Analysis Summary

This commit applies performance optimization techniques to improve inference efficiency. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 0fc0cbd1cf - [None][feat] Add flashinfer api for TRTLLMGenFusedMoE (#10453)

- **Date**: 2026-03-13
- **Author**: Song Rong
- **Categories**: Fusion

### Optimization Techniques

- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- PyTorch built-in optimized ops
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU

### Changed Files

```
.../communication/communication_factory.py         |  56 +-
 .../communication/nvlink_two_sided_flashinfer.py   | 226 +++++++
 .../_torch/modules/fused_moe/configurable_moe.py   |   3 +-
 .../modules/fused_moe/fused_moe_trtllm_gen.py      | 186 +++---
 .../_torch/modules/fused_moe/moe_op_backend.py     | 691 +++++++++++++++++++++
 .../unittest/_torch/modules/moe/moe_test_utils.py  |   4 +
 .../_torch/modules/moe/test_moe_backend.py         |   5 +-
 .../unittest/_torch/modules/moe/test_moe_module.py |  24 +-
 tests/unittest/_torch/modules/test_fused_moe.py    |   3 +
 9 files changed, 1071 insertions(+), 127 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fused_moe/communication/communication_factory.py b/tensorrt_llm/_torch/modules/fused_moe/communication/communication_factory.py
index 651306f76..39a1b66fe 100644
--- a/tensorrt_llm/_torch/modules/fused_moe/communication/communication_factory.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/communication/communication_factory.py
@@ -34,6 +34,7 @@ from .deep_ep import DeepEP
 from .deep_ep_low_latency import DeepEPLowLatency
 from .nvlink_one_sided import NVLinkOneSided
 from .nvlink_two_sided import NVLinkTwoSided
+from .nvlink_two_sided_flashinfer import NVLinkTwoSidedFlashinfer
 
 
 class CommunicationFactory:
@@ -55,6 +56,7 @@ class CommunicationFactory:
         expert_size_per_partition: int,
         payload_in_workspace: bool = False,
         alltoall_result_do_sum: bool = True,
+        use_flashinfer: bool = False,
     ) -> Optional[Communication]:
         """
         Create the best communication method for the given configuration
@@ -117,6 +119,7 @@ class CommunicationFactory:
                 expert_size_per_partition,
                 payload_in_workspace,
                 alltoall_result_do_sum,
+                use_flashinfer,
             )
 
         # Auto-selection: Try strategies in priority order using try-catch
@@ -140,14 +143,24 @@ class CommunicationFactory:
             logger.debug(f"NVLinkOneSided not available: {e}")
 
         try:
-            strategy = NVLinkTwoSided(
-                mapping,
-                num_experts,
-                num_slots,
-                top_k,
-                use_low_precision_combine,
-                alltoall_result_do_sum=alltoall_result_do_sum,
-            )
+            if use_flashinfer:
+                strategy = NVLinkTwoSidedFlashinfer(
+                    mapping,
+                    num_experts,
+                    num_slots,
+                    top_k,
+                    use_low_precision_combine,
+                    alltoall_result_do_sum=alltoall_result_do_sum,
+                )
+            else:
+                strategy = NVLinkTwoSided(
+                    mapping,
+                    num_experts,
+                    num_slots,
+                    top_k,
+                    use_low_precision_combine,
+                    alltoall_result_do_sum=alltoall_result_do_sum,
+                )
             logger.info("Selected communication strategy: NVLinkTwoSided")
             return strategy
         except RuntimeError as e:
@@ -203,6 +216,7 @@ class CommunicationFactory:
         expert_size_per_partition: int,
         payload_in_workspace: bool,
         alltoall_result_do_sum: bool,
+        use_flashinfer: bool,
     ) -> Communication:
         """
         Create a specific method (for debugging/testing)
@@ -225,14 +239,24 @@ class CommunicationFactory:
 
         # Create strategy - will raise RuntimeError if platform not supported
         if method in ["NVLINK_TWO_SIDED"]:
-            return NVLinkTwoS
```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 0fee8cd028 - [TRTLLM-7153] [feat] Move stop_criteria to sample_async (#7041)

- **Date**: 2025-09-07
- **Author**: Netanel Haber
- **Categories**: Parallelism/Async

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Pinned memory
- Reduce synchronization overhead
- Speculative decoding

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/llm_request.py   |  19 +-
 tensorrt_llm/_torch/pyexecutor/sampler.py       | 442 ++++++++++++++++++------
 tensorrt_llm/_torch/pyexecutor/sampler_utils.py |  61 ++++
 tensorrt_llm/_torch/speculative/mtp.py          |  61 ++--
 tests/unittest/_torch/test_torch_sampler.py     | 189 ++++++++++
 5 files changed, 633 insertions(+), 139 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/llm_request.py b/tensorrt_llm/_torch/pyexecutor/llm_request.py
index 2bddb6fd5..c6915fbd6 100644
--- a/tensorrt_llm/_torch/pyexecutor/llm_request.py
+++ b/tensorrt_llm/_torch/pyexecutor/llm_request.py
@@ -1,6 +1,8 @@
+from collections.abc import Generator
 from copy import deepcopy
 from dataclasses import dataclass
-from typing import Any, Dict, List, Optional, Union
+from itertools import pairwise
+from typing import Any, Dict, List, Optional, TypeAlias, Union
 
 import torch
 
@@ -424,7 +426,10 @@ class LlmRequest(tensorrt_llm.bindings.internal.batch_manager.LlmRequest):
         self.child_requests.append(py_request)
 
 
-def convert_wordlist(word_list) -> List[List[int]]:
+StopWordList: TypeAlias = list[list[int]]
+
+
+def convert_wordlist(word_list) -> StopWordList:
     """Converts a wordlist from format:
 
     [[word_0 token_0, word_0 token_1, ...], [word_1 token_0, ...], ...]]
@@ -461,6 +466,16 @@ def convert_wordlist(word_list) -> List[List[int]]:
     return [tokens, offsets]
 
 
+def produce_stop_words(
+        py_stop_words_list: StopWordList) -> Generator[list[int], None, None]:
+    """yield stop sequences from the output of `convert_wordlist` above."""
+    stop_words_list, prefix_sum = py_stop_words_list
+    for start, end in pairwise((0, *prefix_sum)):  # first element: prepend 0
+        if end == -1:  # -1 is a sentinel value in convert_wordlist
+            break
+        yield stop_words_list[start:end]
+
+
 def executor_request_to_llm_request(
         req_id: int,
         executor_request: ExecutorRequest,
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index e6d19a9df..9a4d933e0 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -1,12 +1,16 @@
 from abc import ABC, abstractmethod
 from collections.abc import Iterable
 from dataclasses import dataclass
-from typing import List, Literal, Optional
+from functools import cached_property
+from typing import List, Literal, Optional, TypeAlias
 
+import numpy as np
 import torch
 
 from tensorrt_llm._torch.pyexecutor.make_decoding_batch_input_output import \
     MakeDecodingBatchInputOutput
+from tensorrt_llm._torch.pyexecutor.sampler_utils import (
+    BEAM_0, SINGLE_BEAM_WIDTH, handle_stop_single_beam)
 from tensorrt_llm._utils import nvtx_range, torch_dtype_to_binding
 from tensorrt_llm.bindings import (CudaStream, DataType, ModelConfig,
                                    WorldConfig, make_sampling_config)
@@ -355,21 +359,55 @@ def int_tensor(shape: tuple[int, ...], device: str = 'cuda') -> torch.Tensor:
     return torch.empty(shape, dtype=torch.int, device=device)
 
 
+class TorchStore:
+
+    def __init__(self, *, max_draft_len: int, max_num_sequences: int,
+                 max_beam_width: int):
+        self.max_draft_len = max_draft_len
+        self.max_num_sequences = max_num_sequences
+        self.max_beam_width
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Pinned (page-locked) memory enables faster host-to-device data transfers. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 10348f80fd - [None][perf] Add Triton FP8 blockwise quant kernel and autotuner bucket-skip for visual gen (#11854)

- **Date**: 2026-03-06
- **Author**: Chang Liu
- **Categories**: Kernel Optimization, Quantization Optimization

### Optimization Techniques

- Torch compilation/JIT optimization
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Triton kernel
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- Prefill phase
- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/autotuner.py                   |  16 ++-
 tensorrt_llm/_torch/custom_ops/torch_custom_ops.py |  74 ++++++++++--
 tensorrt_llm/_torch/visual_gen/pipeline_loader.py  |   5 +-
 tensorrt_llm/quantization/utils/__init__.py        |   4 +-
 tensorrt_llm/quantization/utils/fp8_quantize.py    | 129 +++++++++++++++++++++
 .../_torch/thop/parallel/test_fp8_quantize.py      | 104 ++++++++++++++++-
 6 files changed, 317 insertions(+), 15 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/autotuner.py b/tensorrt_llm/_torch/autotuner.py
index 1c6684ea3..7e0744092 100644
--- a/tensorrt_llm/_torch/autotuner.py
+++ b/tensorrt_llm/_torch/autotuner.py
@@ -255,12 +255,18 @@ class TunableRunner(ABC):
 
 
 @contextlib.contextmanager
-def autotune(tune_mode: bool = True, cache_path: str = None):
+def autotune(tune_mode: bool = True,
+             cache_path: str = None,
+             skip_dynamic_tuning_buckets: bool = False):
     """Context manager for autotuning with distributed support.
 
     Args:
         tune_mode: Whether to enable tuning mode
         cache_path: Path to save/load cache files
+        skip_dynamic_tuning_buckets: When True, suppress bucket generation in
+            _optimization_profiles() so only actual input shapes from warmup
+            are profiled. Useful for workloads (e.g. diffusion) where the
+            LLM-oriented M-bucket sweep is unnecessary.
     """
     autotuner = AutoTuner.get()
     rank = autotuner.mapping.rank
@@ -277,7 +283,9 @@ def autotune(tune_mode: bool = True, cache_path: str = None):
 
     # record the old tuning mode
     old_mode = autotuner.is_tuning_mode
+    old_skip = autotuner.skip_dynamic_tuning_buckets
     autotuner.is_tuning_mode = tune_required
+    autotuner.skip_dynamic_tuning_buckets = skip_dynamic_tuning_buckets
     autotune_enabled = tune_required and not old_mode
 
     if autotune_enabled:
@@ -287,6 +295,7 @@ def autotune(tune_mode: bool = True, cache_path: str = None):
         yield
     finally:
         autotuner.is_tuning_mode = old_mode
+        autotuner.skip_dynamic_tuning_buckets = old_skip
         if autotune_enabled:
             logger.info("[Autotuner] Autotuning process ends")
 
@@ -726,6 +735,7 @@ class AutoTuner:
         self.stream_delay_micro_secs = stream_delay_micro_secs
         self.profiling_cache = AutoTunerProfilingCache()
         self.is_tuning_mode = False
+        self.skip_dynamic_tuning_buckets = False
 
         # Timing backend: globaltimer kernel vs cuda events.
         # TLLM_PROFILING_TIMER env var overrides auto-detection:
@@ -1291,7 +1301,9 @@ class AutoTuner:
         for spec in tuning_config.dynamic_tensor_specs:
             assert callable(spec.gen_tuning_buckets) or isinstance(spec.gen_tuning_buckets, (list, tuple)), \
                 "The given dynamic dimension must provide a opt value generation function or a list of opt values"
-            if callable(spec.gen_tuning_buckets):
+            if self.skip_dynamic_tuning_buckets:
+                opt_shapes = ()
+            elif callable(spec.gen_tuning_buckets):
                 if tuning_config.tune_max_num_tokens is None:
                     # Use the current input size as the opt value
                     opt_shapes = spec.gen_tuning_buckets(
diff --git a/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py b/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
index 48e7d214a..ee150d1be 100644
--- a/tensorrt_llm/_torch/custom_o
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## 1074aa91b8 - [TRTLLM-11148][perf] _prepare_inputs host time optimization (#11704)

- **Date**: 2026-03-09
- **Author**: Yukun He
- **Categories**: Host-side Optimization

### Optimization Techniques

- KV cache optimization
- Batching optimization
- Pinned memory
- Speculative decoding

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/model_engine.py | 140 ++++++++++++++-----------
 1 file changed, 76 insertions(+), 64 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/model_engine.py b/tensorrt_llm/_torch/pyexecutor/model_engine.py
index 181e2f6b9..1282e6c92 100644
--- a/tensorrt_llm/_torch/pyexecutor/model_engine.py
+++ b/tensorrt_llm/_torch/pyexecutor/model_engine.py
@@ -1565,7 +1565,7 @@ class PyTorchModelEngine(ModelEngine):
                     ] * len(attn_all_rank_num_tokens)
                 else:
                     logger.debug(
-                        f"Not all ranks can run piecewise cuda graph, disable piecewise cuda graph"
+                        "Not all ranks can run piecewise cuda graph, disable piecewise cuda graph"
                     )
                     return total_num_tokens, False, attn_all_rank_num_tokens
             elif num_ctx_requests != 0 and total_num_tokens <= max_captured_num_tokens:
@@ -2243,7 +2243,11 @@ class PyTorchModelEngine(ModelEngine):
         extend_dummy_requests = []
         generation_requests = []
         first_draft_requests = []
+        # Collect generation request IDs during categorization to avoid
+        # a separate iteration over scheduled_requests.generation_requests later.
+        all_gen_request_ids = []
         for request in scheduled_requests.generation_requests:
+            all_gen_request_ids.append(request.py_request_id)
             if get_draft_token_length(
                     request) > 0 or next_draft_tokens_device is not None:
                 if request.is_dummy:
@@ -2423,10 +2427,21 @@ class PyTorchModelEngine(ModelEngine):
             request.py_batch_idx = request.py_seq_slot
 
         helix_is_inactive_rank, helix_position_offsets = [], []
-        for request in generation_requests:
-            request_ids.append(request.py_request_id)
-            beam_width = request.sampling_config.beam_width
-            for beam in range(beam_width):
+        # Cache invariant method result to avoid repeated calls per-request
+        _has_cp_helix = self.mapping.has_cp_helix()
+        _n_gen = len(generation_requests)
+        if _n_gen > 0:
+            # All generation requests have the same beam width
+            beam_width = generation_requests[0].sampling_config.beam_width
+
+            # Pre-extend constant-value lists to avoid per-request append
+            # overhead (saves ~3 append calls per request).
+            draft_lens.extend([0] * (_n_gen * beam_width))
+            sequence_lengths.extend([1] * (_n_gen * beam_width))
+            num_accepted_draft_tokens.extend([0] * (_n_gen * beam_width))
+
+            for request in generation_requests:
+                request_ids.append(request.py_request_id)
                 # the request has no previous tensor:
                 # (1) new_tokens_device is None, which means overlap scheduler is disabled; or
                 # (2) a dummy request; or
@@ -2435,29 +2450,27 @@ class PyTorchModelEngine(ModelEngine):
                     # skip adding input_ids of CUDA graph dummy requests so that new_tokens_device
               
```

### Analysis Summary

Pinned (page-locked) memory enables faster host-to-device data transfers. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 109f27265c - [None][perf] Add MOE support for dynamic cluster shapes and custom epilogue schedules (#6126)

- **Date**: 2025-09-03
- **Author**: Daniel Stokes
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- Parallelism optimization
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- NVIDIA Hopper (SM90) GPU

### Changed Files

```
.../mixtureOfExpertsBackendBenchmarkFixture.h      |  18 +-
 .../mixtureOfExpertsBackendBenchmarkLauncher.cu    |   9 +-
 .../include/cutlass_extensions/gemm_configs.h      | 360 ++++----
 .../kernels/cutlass_kernels/cutlass_heuristic.cpp  | 142 +--
 .../kernels/cutlass_kernels/cutlass_heuristic.h    |  11 +-
 .../launchers/fpA_intB_launcher_sm90.inl           |   2 +-
 .../moe_gemm/launchers/moe_gemm_tma_ws_launcher.h  |  10 +-
 .../launchers/moe_gemm_tma_ws_launcher.inl         | 962 +++++++++++----------
 .../moe_gemm/moe_gemm_template_dispatch.h          |  10 +
 .../moe_gemm/moe_gemm_template_dispatch_tma_ws.h   | 145 +++-
 .../cutlass_kernels/python/generate_kernels.py     |  60 +-
 ...orrt_llm_internal_cutlass_kernels_static.tar.xz |   4 +-
 .../aarch64-linux-gnu/version.txt                  |   4 +-
 ...orrt_llm_internal_cutlass_kernels_static.tar.xz |   4 +-
 .../x86_64-linux-gnu/version.txt                   |   4 +-
 .../unit_tests/kernels/mixtureOfExpertsTest.cu     |  13 +-
 tests/unittest/_torch/thop/parallel/test_moe.py    |   8 +-
 17 files changed, 970 insertions(+), 796 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h b/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h
index 8e8b77469..a6bdada5a 100644
--- a/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h
+++ b/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h
@@ -1074,10 +1074,10 @@ void MixtureOfExpertsBenchmark<TypeTuple_>::runBenchmark(benchmark::State& state
         state.SkipWithMessage("Out of range tactic");
         return;
     }
+    auto tactics1 = mMoERunner.getTactics(MoeGemmId::GEMM_1);
+    auto tactics2 = mMoERunner.getTactics(MoeGemmId::GEMM_2);
     if (LOG_LEVEL >= INFO)
     {
-        auto tactics1 = mMoERunner.getTactics(MoeGemmId::GEMM_1);
-        auto tactics2 = mMoERunner.getTactics(MoeGemmId::GEMM_2);
         std::cout << "Selected tactic #1: " << tactic_idx1 << "/" << tactics1.size() << "\n"
                   << tactics1[tactic_idx1].toString() << std::endl;
         std::cout << "Selected tactic #2: " << tactic_idx2 << "/" << tactics2.size() << "\n"
@@ -1086,6 +1086,20 @@ void MixtureOfExpertsBenchmark<TypeTuple_>::runBenchmark(benchmark::State& state
     state.counters["tactic_idx1"] = tactic_idx1;
     state.counters["tactic_idx2"] = tactic_idx2;
 
+    state.counters["t1_sm_version"] = tactics1[tactic_idx1].sm_version;
+    state.counters["t1_tile_shape"] = tactics1[tactic_idx1].getTileConfigAsInt();
+    state.counters["t1_cluster_shape"] = (int) tactics1[tactic_idx1].cluster_shape;
+    state.counters["t1_dynamic_cluster_shape"] = (int) tactics1[tactic_idx1].dynamic_cluster_shape;
+    state.counters["t1_fallback_cluster_shape"] = (int) tactics1[tactic_idx1].fallback_cluster_shape;
+    state.counters["t1_epilogue_schedule"] = (int) tactics1[tactic_idx1].epilogue_schedule;
+
+    state.counters["t2_sm_version"] = tactics2[tactic_idx2].sm_version;
+    state.counters["t2_tile_shape"] = tactics2[tactic_idx2].getTileConfigAsInt();
+    state.counters["t2_cluster_shape"] = (int) tactics2[tactic_idx2].cluster_shape;
+    state.counters["t2_dynamic_cluster_shape"] = (int) tactics2[tactic_idx2].dynamic_cluster_shape;
+    state.counters["t2_fallback_cluster_shape"] = (int) tactics2[tactic_idx2].fallback_cluster_shape;
+    state.counters["t2_epilogue_schedule"] = (int) tactics2[tactic_idx2].epilogue_schedule;
+
     createGraph(parallelism_config, gemm_to_profile);
 
     {
diff --git a/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkLauncher.cu b/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkLauncher.cu
index 8e18694ad..d76c4239c 100644
--- a/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkLauncher.cu
+++ b/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkLauncher.cu
@@ -160,6 +160,10 @@ void argGenLoadFile(benchmark::internal::Benchmark* benchmark)
      */
 
     std::ifstream file{workloadFile};
+    if (!file.is_open())
+    {
+        throw std::invalid_argument("Failed to open benchmark file: " + std::string(workloadFile));
+    }
     
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 1191555cce - [ci] speedup fused moe tests (#5726)

- **Date**: 2025-07-07
- **Author**: Omer Ullman Argov
- **Categories**: Throughput/Latency, Fusion

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- FP8 quantization
- Integer quantization
- KV cache optimization
- Reduce synchronization overhead
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../unit/multigpu/custom_ops/test_moe_ep.py        | 18 +++--
 tests/unittest/_torch/modules/test_fused_moe.py    | 93 ++++++++++++++--------
 2 files changed, 71 insertions(+), 40 deletions(-)
```

### Diff Preview

```diff
diff --git a/tests/unittest/_torch/auto_deploy/unit/multigpu/custom_ops/test_moe_ep.py b/tests/unittest/_torch/auto_deploy/unit/multigpu/custom_ops/test_moe_ep.py
index 0e8f84a6d..b09ad1b2a 100644
--- a/tests/unittest/_torch/auto_deploy/unit/multigpu/custom_ops/test_moe_ep.py
+++ b/tests/unittest/_torch/auto_deploy/unit/multigpu/custom_ops/test_moe_ep.py
@@ -21,23 +21,25 @@ def _run_moe_ep_test(num_experts: int, topk: int, rank: int, world_size: int):
 
     torch.manual_seed(0)
     torch.cuda.manual_seed(0)
-    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda() * 0.5
+    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device="cuda") * 0.5
 
-    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=torch.float32).cuda()
+    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=torch.float32, device="cuda")
     routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
     final_scales, selected_experts = torch.topk(routing_weights, TOP_K, dim=-1)
     final_scales = final_scales / final_scales.sum(dim=-1, keepdim=True)
     final_scales = final_scales.to(x.dtype)
 
     fused_w3_w1_stacked_weight = torch.empty(
-        (NUM_EXPERTS, INTERMEDIATE_SIZE * 2, HIDDEN_SIZE), dtype=dtype
-    ).cuda()
-    fused_w2_weight = torch.empty((NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype).cuda()
+        (NUM_EXPERTS, INTERMEDIATE_SIZE * 2, HIDDEN_SIZE), dtype=dtype, device="cuda"
+    )
+    fused_w2_weight = torch.empty(
+        (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype, device="cuda"
+    )
     weights = {}
     for expert_id in range(NUM_EXPERTS):
-        w1 = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype).cuda() * 0.5
-        w2 = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype).cuda() * 0.5
-        w3 = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype).cuda() * 0.5
+        w1 = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype, device="cuda") * 0.5
+        w2 = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype, device="cuda") * 0.5
+        w3 = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype, device="cuda") * 0.5
         weights[f"{expert_id}.w1.weight"] = w1
         weights[f"{expert_id}.w2.weight"] = w2
         weights[f"{expert_id}.w3.weight"] = w3
diff --git a/tests/unittest/_torch/modules/test_fused_moe.py b/tests/unittest/_torch/modules/test_fused_moe.py
index 83be18823..367f7300b 100644
--- a/tests/unittest/_torch/modules/test_fused_moe.py
+++ b/tests/unittest/_torch/modules/test_fused_moe.py
@@ -58,17 +58,22 @@ def test_fused_moe(moe_cls, dtype, experts, RoutingMethodCls, mapping=None):
     torch.cuda.set_device(mapping.rank)
     torch.manual_seed(0)
     torch.cuda.manual_seed(0)
-    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
-    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=dtype).cuda()
+    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device="cuda")
+    router_logit
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 11d79aa875 - [https://nvbugs/5832481][test] Add gpt-oss-120b-Eagle3-throughput case on DGX-Spark (#11419)

- **Date**: 2026-02-12
- **Author**: JennyLiu
- **Categories**: Throughput/Latency

### Optimization Techniques

- Vectorized memory access
- Async/stream-based execution
- FP8 quantization
- KV cache optimization
- Speculative decoding
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../integration/defs/perf/pytorch_model_config.py  | 33 +++++++++++++++++++---
 tests/integration/defs/perf/test_perf.py           |  3 +-
 tests/integration/test_lists/qa/llm_spark_perf.yml |  6 ++--
 3 files changed, 34 insertions(+), 8 deletions(-)
```

### Diff Preview

```diff
diff --git a/tests/integration/defs/perf/pytorch_model_config.py b/tests/integration/defs/perf/pytorch_model_config.py
index db2d6a12b..25337faa8 100644
--- a/tests/integration/defs/perf/pytorch_model_config.py
+++ b/tests/integration/defs/perf/pytorch_model_config.py
@@ -320,10 +320,10 @@ def get_model_yaml_config(model_label: str,
                 'num_postprocess_workers': 4
             }
         },
-        # GPT-OSS 120B speculative decoding (Eagle3 draft)
+        # GPT-OSS 120B speculative decoding with Eagle3
         {
             'patterns': [
-                'gpt_oss_120b_fp4-bench-pytorch-streaming-float4-maxbs:1-maxnt:4096-input_output_len:2048,128-reqs:1-con:1',
+                'gpt_oss_120b_eagle3-bench-pytorch',
             ],
             'config': {
                 'enable_attention_dp': False,
@@ -337,9 +337,34 @@ def get_model_yaml_config(model_label: str,
                     'decoding_type':
                     'Eagle',
                     'max_draft_len':
-                    5,
+                    3,
+                    'speculative_model_dir':
+                    f'{llm_models_root()}/gpt_oss/gpt-oss-120b-Eagle3',
+                },
+                'kv_cache_config': {
+                    'enable_block_reuse': False,
+                },
+            }
+        },
+        # GPT-OSS 120B speculative decoding with Eagle3-throughput (https://nvbugspro.nvidia.com/bug/5832481)
+        {
+            'patterns': [
+                'gpt_oss_120b_eagle3_throughput-bench-pytorch',
+            ],
+            'config': {
+                'enable_attention_dp': False,
+                'disable_overlap_scheduler': True,
+                'enable_autotuner': False,
+                'cuda_graph_config': {
+                    'enable_padding': True,
+                },
+                'speculative_config': {
+                    'decoding_type':
+                    'Eagle',
+                    'max_draft_len':
+                    3,
                     'speculative_model_dir':
-                    f"{llm_models_root()}/gpt_oss/gpt-oss-120b-Eagle3",
+                    f'{llm_models_root()}/gpt_oss/gpt-oss-120b-Eagle3-throughput',
                 },
                 'kv_cache_config': {
                     'enable_block_reuse': False,
diff --git a/tests/integration/defs/perf/test_perf.py b/tests/integration/defs/perf/test_perf.py
index 3695cf7e2..120df0f43 100644
--- a/tests/integration/defs/perf/test_perf.py
+++ b/tests/integration/defs/perf/test_perf.py
@@ -173,7 +173,8 @@ MODEL_PATH_DICT = {
     "mistral_small_v3.1_24b": "Mistral-Small-3.1-24B-Instruct-2503",
     "gpt_oss_120b_fp4": "gpt_oss/gpt-oss-120b",
     "gpt_oss_20b_fp4": "gpt_oss/gpt-oss-20b",
-    "gpt_oss_120b_eagle3": "gpt_oss/gpt-oss-120b-Eagle3",
+    "gpt_oss_120b_eagle3": "gpt_oss/gpt-oss-120b",
+    "gpt_oss_120b_eagle3_throughput": "gpt_oss/gpt-oss-120b",
     "nemotron_nano_3_30b_fp8": "Nemotron-Nano-3-30B-A3.5B-FP8-KVFP8-dev",
     "nemot
```

### Analysis Summary

Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 1260e2f33f - feat: Optimize TRTLLM Sampler perf single beam single step (#5550)

- **Date**: 2025-07-07
- **Author**: Daniel Cámpora
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Reduce synchronization overhead
- Speculative decoding

### Applicable Conditions

- Decode/generation phase
- Disaggregated serving

### Changed Files

```
cpp/tensorrt_llm/pybind/batch_manager/bindings.cpp |  98 +++++++++
 .../_torch/pyexecutor/handle_context_logits.py     |  69 -------
 .../_torch/pyexecutor/handle_generation_logits.py  |  37 ----
 tensorrt_llm/_torch/pyexecutor/handle_logits.py    |  66 +++++++
 tensorrt_llm/_torch/pyexecutor/llm_request.py      |   1 +
 .../pyexecutor/make_decoding_batch_input_output.py | 164 ++++-----------
 tensorrt_llm/_torch/pyexecutor/py_executor.py      |   6 -
 tensorrt_llm/_torch/pyexecutor/sampler.py          | 220 +++++++++++++++------
 tensorrt_llm/_torch/pyexecutor/seq_slot_manager.py |   1 +
 9 files changed, 363 insertions(+), 299 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/pybind/batch_manager/bindings.cpp b/cpp/tensorrt_llm/pybind/batch_manager/bindings.cpp
index 71e84d733..f7ba20920 100644
--- a/cpp/tensorrt_llm/pybind/batch_manager/bindings.cpp
+++ b/cpp/tensorrt_llm/pybind/batch_manager/bindings.cpp
@@ -26,6 +26,8 @@
 #include "tensorrt_llm/batch_manager/runtimeBuffers.h"
 #include "tensorrt_llm/batch_manager/sequenceSlotManager.h"
 #include "tensorrt_llm/pybind/common/bindTypes.h"
+#include "tensorrt_llm/runtime/gptDecoderBatched.h"
+#include "tensorrt_llm/runtime/runtimeKernels.h"
 #include "tensorrt_llm/runtime/torch.h"
 #include "tensorrt_llm/runtime/torchView.h"
 
@@ -170,6 +172,7 @@ void initBindings(pybind11::module_& m)
         .def_property_readonly("context_phase_params", &GenLlmReq::getContextPhaseParams)
         .def_property_readonly("is_context_only_request", &GenLlmReq::isContextOnlyRequest)
         .def_property_readonly("is_generation_only_request", &GenLlmReq::isGenerationOnlyRequest)
+        .def_property_readonly("is_generation_complete_state", &GenLlmReq::isGenerationCompleteState)
         .def_property_readonly("is_context_finished", &GenLlmReq::isContextFinished)
         .def_property_readonly("is_disagg_generation_init_state", &GenLlmReq::isDisaggGenerationInitState)
         .def_property_readonly(
@@ -428,6 +431,101 @@ void initBindings(pybind11::module_& m)
                  runtime::TllmRuntime const&>(),
             py::arg("max_beam_width"), py::arg("max_seq_len"), py::arg("buffer_manager"), py::arg("model_config"),
             py::arg("world_config"), py::arg("decoding_config"), py::arg("runtime"));
+
+    m.def(
+        "add_new_tokens_to_requests",
+        [](std::vector<std::shared_ptr<tb::LlmRequest>>& requests,
+            std::vector<tb::LlmRequest::TokenIdType> const& tokens, int beam_idx)
+        {
+            TLLM_CHECK_WITH_INFO(requests.size() == tokens.size(), "Expected the same number of requests and tokens.");
+
+            for (int i = 0; i < requests.size(); ++i)
+            {
+                requests[i]->addNewToken(tokens[i], beam_idx);
+            }
+        },
+        py::arg("requests"), py::arg("tokens"), py::arg("beam_idx"),
+        "Add new tokens to multiple LLM requests. The tokens vector should contain tokens for beam beam_idx of all "
+        "requests in order.");
+
+    m.def(
+        "make_decoding_batch_input",
+        [](std::vector<std::shared_ptr<tb::LlmRequest>>& contextRequests,
+            std::vector<std::shared_ptr<tb::LlmRequest>>& genRequests, tr::ITensor::SharedPtr logits, int beamWidth,
+            std::vector<int> const& numContextLogitsPrefixSum, tb::DecoderInputBuffers const& decoderInputBuffers,
+            runtime::decoder::DecoderState& decoderState, tr::BufferManager const& manager)
+        {
+            std::vector<int> activeSlots;
+            std::vector<int> generationSteps;
+            std::vector<std::vector<tr::ITensor::SharedConstPtr>> logitsVec = {{}};
+
+        
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 12f339f3bf - [None][fix] Fix the aux_stream in Llama4MinLatencyFusedMoE (#9035)

- **Date**: 2025-11-14
- **Author**: Jinyang Yuan
- **Categories**: Throughput/Latency, Fusion

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- Parallelism optimization
- Speculative decoding
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/models/modeling_llama.py             |  3 ++-
 tensorrt_llm/_torch/models/modeling_llama_min_latency.py | 10 ++++++----
 2 files changed, 8 insertions(+), 5 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_llama.py b/tensorrt_llm/_torch/models/modeling_llama.py
index 38d487a7e..ff1db13fd 100644
--- a/tensorrt_llm/_torch/models/modeling_llama.py
+++ b/tensorrt_llm/_torch/models/modeling_llama.py
@@ -41,7 +41,7 @@ from ..modules.linear import Linear, TensorParallelMode
 from ..modules.multi_stream_utils import maybe_execute_in_parallel
 from ..modules.rms_norm import RMSNorm
 from ..speculative import SpecMetadata
-from ..utils import Fp4QuantizedTensor
+from ..utils import AuxStreamType, Fp4QuantizedTensor
 from .modeling_multimodal_utils import fuse_input_embeds
 from .modeling_speculative import SpecDecOneEngineForCausalLM
 from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
@@ -293,6 +293,7 @@ class Llama4MoE(nn.Module):
             weight_loading_mode=MoEWeightLoadingMode.FUSED_GATE_UP_PROJ,
             model_config=model_config,
             apply_router_weight_on_input=True,
+            aux_stream_dict={AuxStreamType.MoeChunkingOverlap: aux_stream},
             layer_idx=layer_idx)
 
         self.router = Linear(
diff --git a/tensorrt_llm/_torch/models/modeling_llama_min_latency.py b/tensorrt_llm/_torch/models/modeling_llama_min_latency.py
index 027eeeace..c4cc71bc0 100644
--- a/tensorrt_llm/_torch/models/modeling_llama_min_latency.py
+++ b/tensorrt_llm/_torch/models/modeling_llama_min_latency.py
@@ -23,7 +23,7 @@ from ..modules.linear import (Linear, TensorParallelMode, WeightMode,
                               WeightsLoadingConfig)
 from ..modules.multi_stream_utils import maybe_execute_in_parallel
 from ..speculative import SpecMetadata
-from ..utils import Fp4QuantizedTensor
+from ..utils import AuxStreamType, Fp4QuantizedTensor
 from .modeling_llama import Llama4Attention, Llama4DecoderLayer, Llama4MoE
 
 # Perf heuristics thresholds.
@@ -438,7 +438,8 @@ class Llama4MinLatencyFusedMoE(CutlassFusedMoE):
         dtype: Optional[torch.dtype] = None,
         reduce_results: bool = False,
         model_config: ModelConfig = ModelConfig(),
-        aux_stream: torch.cuda.Stream = torch.cuda.Stream(),
+        aux_stream_dict: Optional[Dict[AuxStreamType,
+                                       torch.cuda.Stream]] = None,
         weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
         VANILLA,
         apply_router_weight_on_input: bool = False,
@@ -452,7 +453,7 @@ class Llama4MinLatencyFusedMoE(CutlassFusedMoE):
             dtype=dtype,
             reduce_results=reduce_results,
             model_config=model_config,
-            aux_stream=aux_stream,
+            aux_stream_dict=aux_stream_dict,
             weight_loading_mode=weight_loading_mode,
             apply_router_weight_on_input=apply_router_weight_on_input,
         )
@@ -554,6 +555,7 @@ class Llama4MinLatencyMoE(Llama4MoE):
             weight_loading_mode=MoEWeightLoadingMode.FUSED_GATE_UP_PROJ,
             model_config=model_config,
             apply_router_weight_on_input=True,
+
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 147ad69368 - [None][doc] blog: Scaling Expert Parallelism in TensorRT-LLM (Part 2: Performance Status and Optimization) (#6547)

- **Date**: 2025-08-01
- **Author**: Kaiyu Xie
- **Categories**: Parallelism/Async

### Optimization Techniques

- Async/stream-based execution
- Parallelism optimization
- Batching optimization
- Reduce synchronization overhead
- Speculative decoding
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- Decode/generation phase
- Large batch / high concurrency
- Disaggregated serving

### Changed Files

```
.../media/tech_blog8_communication_kernel.png      | Bin 0 -> 13610 bytes
 .../blogs/media/tech_blog8_kernel_breakdown.png    | Bin 0 -> 137738 bytes
 .../blogs/media/tech_blog8_moe_aux_kernels1.png    | Bin 0 -> 82415 bytes
 .../blogs/media/tech_blog8_moe_aux_kernels2.png    | Bin 0 -> 52000 bytes
 .../blogs/media/tech_blog8_perf-1k-1k-dep.png      | Bin 0 -> 132526 bytes
 .../blogs/media/tech_blog8_perf-4k-1k-dep.png      | Bin 0 -> 132786 bytes
 .../blogs/media/tech_blog8_perf-8k-1k-dep.png      | Bin 0 -> 139264 bytes
 .../blogs/media/tech_blog8_perf-8k-1k-e2e-mtp.png  | Bin 0 -> 131852 bytes
 ...4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md |   3 +-
 ...ing_Expert_Parallelism_in_TensorRT-LLM_part2.md | 322 +++++++++++++++++++++
 10 files changed, 324 insertions(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/docs/source/blogs/media/tech_blog8_communication_kernel.png b/docs/source/blogs/media/tech_blog8_communication_kernel.png
new file mode 100644
index 000000000..57f0e53f6
Binary files /dev/null and b/docs/source/blogs/media/tech_blog8_communication_kernel.png differ
diff --git a/docs/source/blogs/media/tech_blog8_kernel_breakdown.png b/docs/source/blogs/media/tech_blog8_kernel_breakdown.png
new file mode 100644
index 000000000..731b074e7
Binary files /dev/null and b/docs/source/blogs/media/tech_blog8_kernel_breakdown.png differ
diff --git a/docs/source/blogs/media/tech_blog8_moe_aux_kernels1.png b/docs/source/blogs/media/tech_blog8_moe_aux_kernels1.png
new file mode 100644
index 000000000..b8464fe3f
Binary files /dev/null and b/docs/source/blogs/media/tech_blog8_moe_aux_kernels1.png differ
diff --git a/docs/source/blogs/media/tech_blog8_moe_aux_kernels2.png b/docs/source/blogs/media/tech_blog8_moe_aux_kernels2.png
new file mode 100644
index 000000000..0911e3634
Binary files /dev/null and b/docs/source/blogs/media/tech_blog8_moe_aux_kernels2.png differ
diff --git a/docs/source/blogs/media/tech_blog8_perf-1k-1k-dep.png b/docs/source/blogs/media/tech_blog8_perf-1k-1k-dep.png
new file mode 100644
index 000000000..b27419720
Binary files /dev/null and b/docs/source/blogs/media/tech_blog8_perf-1k-1k-dep.png differ
diff --git a/docs/source/blogs/media/tech_blog8_perf-4k-1k-dep.png b/docs/source/blogs/media/tech_blog8_perf-4k-1k-dep.png
new file mode 100644
index 000000000..46ed5eced
Binary files /dev/null and b/docs/source/blogs/media/tech_blog8_perf-4k-1k-dep.png differ
diff --git a/docs/source/blogs/media/tech_blog8_perf-8k-1k-dep.png b/docs/source/blogs/media/tech_blog8_perf-8k-1k-dep.png
new file mode 100644
index 000000000..66fb5e13a
Binary files /dev/null and b/docs/source/blogs/media/tech_blog8_perf-8k-1k-dep.png differ
diff --git a/docs/source/blogs/media/tech_blog8_perf-8k-1k-e2e-mtp.png b/docs/source/blogs/media/tech_blog8_perf-8k-1k-e2e-mtp.png
new file mode 100644
index 000000000..6f581ba39
Binary files /dev/null and b/docs/source/blogs/media/tech_blog8_perf-8k-1k-e2e-mtp.png differ
diff --git a/docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md b/docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md
index 5e43b33ac..1fd9cc64c 100644
--- a/docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md
+++ b/docs/source/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.md
@@ -515,7 +515,7 @@ Next, let's use some representative workloads to understand the performance impa
 Clearly in Figure 25, we can see that EPLB brings a clear performance improvement when the EP size increases, for both MoE GroupGEMM and EP communication times.
 
 ## Reproducing steps
-Currently to run through the reproducing steps described in this section, please, use this [feature branch](https://github.com/NVIDIA/TensorRT-LLM/tree/feat/large-ep/tensorrt_llm). It will get me
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 1745102e72 - [TRTLLM-7027][feat] Fuse d2t to logitsBitmaskKernel and fix a race condition in one-model spec (#7481)

- **Date**: 2025-09-04
- **Author**: Enwei Zhu
- **Categories**: Kernel Optimization, Fusion

### Optimization Techniques

- Torch compilation/JIT optimization
- Custom CUDA kernel
- Vectorized memory access
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Pinned memory
- Speculative decoding
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
cpp/tensorrt_llm/kernels/logitsBitmask.cu          | 154 ++++++++++++++++++++-
 cpp/tensorrt_llm/kernels/logitsBitmask.h           |   4 +
 cpp/tensorrt_llm/thop/logitsBitmaskOp.cpp          |  81 ++++++-----
 tensorrt_llm/_torch/attention_backend/interface.py |  20 ++-
 tensorrt_llm/_torch/compilation/utils.py           |   5 +-
 tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py   |   4 -
 tensorrt_llm/_torch/models/modeling_speculative.py |   2 +-
 tensorrt_llm/_torch/pyexecutor/guided_decoder.py   |  72 ++++++----
 tensorrt_llm/_torch/speculative/drafting_loops.py  |  12 +-
 tensorrt_llm/_torch/speculative/eagle3.py          |  20 +--
 tensorrt_llm/_torch/speculative/mtp.py             |  16 +--
 .../_torch/thop/parallel/test_logits_bitmask_op.py |  59 +++++++-
 12 files changed, 336 insertions(+), 113 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/logitsBitmask.cu b/cpp/tensorrt_llm/kernels/logitsBitmask.cu
index 6146656ac..084e660cc 100644
--- a/cpp/tensorrt_llm/kernels/logitsBitmask.cu
+++ b/cpp/tensorrt_llm/kernels/logitsBitmask.cu
@@ -60,6 +60,7 @@ __device__ PackedT packedNegativeInfinity()
     }
     return *reinterpret_cast<PackedT*>(packed);
 }
+} // namespace
 
 template <typename T, typename PackedT, int32_t kBitsPerThread>
 __global__ void __launch_bounds__(kThreadsPerBlock) logitsBitmaskKernel(
@@ -118,7 +119,8 @@ void logitsBitmaskDispatchToBitsPerThread(
     T** logits, uint32_t const** bitmask, int32_t batchSize, int32_t vocabSizePadded, cudaStream_t stream)
 {
     int constexpr kAlignment = sizeof(PackedT) / sizeof(T);
-    int32_t const numBlocksPerRow = ceilDiv(2048 / kThreadsPerBlock * 128, batchSize);
+    static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
+    int32_t const numBlocksPerRow = ceilDiv(2048 / kThreadsPerBlock * smCount, batchSize);
     int32_t const numBitsPerThread = ceilDiv(vocabSizePadded, kThreadsPerBlock * numBlocksPerRow);
     int32_t bitmaskSize = ceilDiv(vocabSizePadded, kBitsPerMaskElement);
 
@@ -145,7 +147,6 @@ void logitsBitmaskDispatchToBitsPerThread(
         logitsBitmaskKernel<T, PackedT, 32><<<grid, block, 0, stream>>>(logits, bitmask, vocabSizePadded, bitmaskSize);
     }
 }
-} // namespace
 
 template <typename T>
 void invokeLogitsBitmask(
@@ -179,5 +180,154 @@ template void invokeLogitsBitmask<half>(
 template void invokeLogitsBitmask<__nv_bfloat16>(
     __nv_bfloat16** logits, uint32_t const** bitmask, int32_t batchSize, int32_t vocabSizePadded, cudaStream_t stream);
 #endif
+
+template <typename T, typename PackedT, int32_t kBitsPerThread>
+__global__ void __launch_bounds__(kThreadsPerBlock) contiguousLogitsBitmaskKernel(T* __restrict__ logits,
+    uint32_t const* __restrict__ bitmask, int32_t const* __restrict__ tokenMask, int32_t const* __restrict__ d2t,
+    int32_t vocabSizePadded, int32_t bitmaskSize)
+{
+    int constexpr kAlignment = sizeof(PackedT) / sizeof(T);
+    uint32_t constexpr kPackedMask = (1 << kAlignment) - 1;
+
+    int const batchIdx = blockIdx.y;
+    if (tokenMask != nullptr && !tokenMask[batchIdx])
+    {
+        return;
+    }
+
+    int const blockOffset = blockIdx.x * kThreadsPerBlock * kBitsPerThread;
+    T* logitsGmemPtr = logits + batchIdx * vocabSizePadded + blockOffset;
+
+    uint32_t const* bitmaskGmemPtr = bitmask + batchIdx * bitmaskSize + blockOffset / kBitsPerMaskElement;
+    int const bitmaskInnerIdx = threadIdx.x % (kBitsPerMaskElement / kAlignment);
+    T logitsReg[kAlignment];
+
+#pragma unroll
+    for (int offset = threadIdx.x * kAlignment; offset < kThreadsPerBlock * kBitsPerThread;
+         offset += kThreadsPerBlock * kAlignment)
+    {
+        if (blockOffset + offset >= vocabSizePadded)
+        {
+            break;
+        }
+
+        uint32_t bitmaskVal = 0;
+        if (d2t == nullptr)
+        {
+       
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Pinned (page-locked) memory enables faster host-to-device data transfers. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 1902d73eb5 - fix: llama4: add an option `apply_router_weight_on_input` for in FusedMoE (#3492)

- **Date**: 2025-04-14
- **Author**: Chang Liu
- **Categories**: Fusion

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Parallelism optimization
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/models/modeling_llama.py |  3 ++-
 tensorrt_llm/_torch/modules/fused_moe.py     | 10 ++++++++++
 2 files changed, 12 insertions(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_llama.py b/tensorrt_llm/_torch/models/modeling_llama.py
index 6fc99754b..dffe91ecb 100644
--- a/tensorrt_llm/_torch/models/modeling_llama.py
+++ b/tensorrt_llm/_torch/models/modeling_llama.py
@@ -119,7 +119,8 @@ class Llama4MoE(nn.Module):
             reduce_results=
             False,  # In both low latency and max-throughput scenarios, FusedMoE needs not to do allreduce inside op.
             weight_loading_mode=MoEWeightLoadingMode.FUSED_GATE_UP_PROJ,
-            model_config=model_config)
+            model_config=model_config,
+            apply_router_weight_on_input=True)
 
         self.shared_expert = GatedMLP(
             hidden_size=hidden_size,
diff --git a/tensorrt_llm/_torch/modules/fused_moe.py b/tensorrt_llm/_torch/modules/fused_moe.py
index f6ef183e5..81ee17ad7 100755
--- a/tensorrt_llm/_torch/modules/fused_moe.py
+++ b/tensorrt_llm/_torch/modules/fused_moe.py
@@ -240,6 +240,7 @@ class FusedMoE(nn.Module):
         aux_stream: torch.cuda.Stream = torch.cuda.Stream(),
         weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
         VANILLA,
+        apply_router_weight_on_input: bool = False,
     ):
         from ..distributed import AllReduce
 
@@ -302,6 +303,9 @@ class FusedMoE(nn.Module):
         if not model_config.skip_create_weights:
             self.create_weights()
 
+        # If True, the router weight will be multiplied on the input rather than at the end of FC2
+        self.apply_router_weight_on_input = apply_router_weight_on_input
+
     def setup_quant_scales(self):
         self.quant_scales = None
         if not self.has_any_quant:
@@ -558,6 +562,12 @@ class FusedMoE(nn.Module):
         assert token_final_scales.dtype == torch.float32
         assert token_selected_experts.dtype == torch.int32
 
+        if self.apply_router_weight_on_input:
+            assert self.routing_method.top_k == 1, "Current walkaround only supports top-1 routing"
+            x = x * token_final_scales.to(x.dtype)
+            # TODO: remove this once we have correct fusedmoe kernel ready
+            token_final_scales = None
+
         x_sf = None
         if self.has_any_quant:
             if self.has_fp8_qdq:

```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 196d94a419 - [TRTLLM-10030][perf] avoid syncs in beam search + other improvements (#11349)

- **Date**: 2026-02-09
- **Author**: mpikulski
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- KV cache optimization
- Batching optimization
- Pinned memory
- Reduce synchronization overhead
- Speculative decoding

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/sampler.py         | 300 ++++++++++++----------
 tests/unittest/_torch/sampler/test_beam_search.py |  22 +-
 2 files changed, 175 insertions(+), 147 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index 31e56ccb0..9fdaf22fb 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -766,6 +766,13 @@ class BeamHistory:
     cum_logprobs: torch.Tensor | None = None
 
 
+BeamHistoryBuilder: TypeAlias = Callable[[], BeamHistory | None]
+"""Builder for BeamHistory.
+
+Used to defer possibly unnecessary host-tensor construction until update_requests().
+"""
+
+
 @dataclass(kw_only=True)
 class SamplingRequestsMetadata:
     req_num_generated_tokens: torch.Tensor
@@ -789,7 +796,7 @@ class SampleStateTensorsHostTorch(SampleStateTensors):
 
 @dataclass(kw_only=True)
 class SampleStateTorch(SampleState[SampleStateTensorsHostTorch, SampleStateTensors]):
-    beam_histories: list[BeamHistory | None] | None = None
+    beam_history_builders: list[BeamHistoryBuilder | None] | None = None
 
 
 class AsyncWorkerMixin:
@@ -1249,9 +1256,6 @@ class TorchSampler(Sampler[SampleStateTorch], AsyncWorkerMixin):
         self,
         token_tensor: torch.Tensor,
         logprobs_tensor: torch.Tensor,
-        sampled_log_probs_indices: torch.Tensor | None,
-        sampled_log_probs_vals: torch.Tensor | None,
-        sampled_log_probs_rank: torch.Tensor | None,
     ) -> list[list[dict[int, Logprob]]]:
         """Convert the logprobs tensor to a list of lists of dictionaries of Logprob objects
 
@@ -1260,9 +1264,6 @@ class TorchSampler(Sampler[SampleStateTorch], AsyncWorkerMixin):
         args:
             token_tensor: torch.Tensor. Shape: beam_width, num_tokens, num_logprobs
             logprobs_tensor: torch.Tensor. Shape: beam_width, num_tokens, num_logprobs
-            sampled_log_probs_indices: torch.Tensor | None. Shape: num_tokens
-            sampled_log_probs_vals: torch.Tensor | None. Shape: num_tokens
-            sampled_log_probs_rank: torch.Tensor | None. Shape: num_tokens
         output:
             list[list[dict[int, Logprob]]]. Shape: (beam_width, num_tokens)
         """
@@ -1274,38 +1275,13 @@ class TorchSampler(Sampler[SampleStateTorch], AsyncWorkerMixin):
         token_log_probs: list[list[dict[int, Logprob]]] = []
         token_list = token_tensor.tolist()
         logprobs_list = logprobs_tensor.tolist()
-        sampled_log_probs_indices_list: list[int] | None = None
-        sampled_log_probs_vals_list: list[float] | None = None
-        sampled_log_probs_rank_list: list[int] | None = None
-        if sampled_log_probs_indices is not None:
-            sampled_log_probs_indices_list = sampled_log_probs_indices.tolist()
-            assert sampled_log_probs_vals is not None, "sampled_log_probs_vals must be provided"
-            assert sampled_log_probs_rank is not None, "sampled_log_probs_rank must be provided"
-            sampled_log_probs_vals_list = sampled_log_probs_vals.tolist()
-            sampled_log_probs_rank_list = sampled_log_probs_rank.tolist()
         for beam_idx in
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Pinned (page-locked) memory enables faster host-to-device data transfers. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 19a0ea363b - [TRTLLM-6743][feat] Optimize and refactor alltoall in WideEP (#6973)

- **Date**: 2025-08-24
- **Author**: dongxuy04
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- PyTorch built-in optimized ops
- Reduce synchronization overhead
- Speculative decoding
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- Prefill phase
- Decode/generation phase
- Disaggregated serving

### Changed Files

```
cpp/tensorrt_llm/kernels/fusedMoeCommKernels.cu    | 1372 +++++++++++++++++++
 cpp/tensorrt_llm/kernels/fusedMoeCommKernels.h     |  562 ++++++++
 cpp/tensorrt_llm/kernels/moeCommKernels.cu         |  804 -----------
 cpp/tensorrt_llm/kernels/moeCommKernels.h          |  268 ----
 cpp/tensorrt_llm/kernels/moeCommKernelsCommon.h    |   47 +
 cpp/tensorrt_llm/kernels/moePrepareKernels.cu      |  459 +------
 cpp/tensorrt_llm/kernels/moePrepareKernels.h       |   60 +-
 cpp/tensorrt_llm/thop/moeCommOp.cpp                |  331 ++---
 cpp/tensorrt_llm/thop/moeLoadBalanceOp.cpp         |    1 -
 cpp/tests/unit_tests/kernels/CMakeLists.txt        |    2 +
 .../unit_tests/kernels/fusedMoeCommKernelTest.cpp  | 1410 ++++++++++++++++++++
 tensorrt_llm/_mnnvl_utils.py                       |  101 +-
 tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py   |   72 +-
 tensorrt_llm/_torch/models/modeling_deepseekv3.py  |    5 +-
 tensorrt_llm/_torch/models/modeling_speculative.py |    4 +
 .../_torch/modules/fused_moe/fused_moe_cutlass.py  |   44 +-
 .../_torch/modules/fused_moe/fused_moe_wide_ep.py  |  135 +-
 tests/integration/test_lists/waives.txt            |    2 -
 tests/unittest/_torch/thop/test_moe_alltoall.py    |  480 +++----
 19 files changed, 3937 insertions(+), 2222 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/fusedMoeCommKernels.cu b/cpp/tensorrt_llm/kernels/fusedMoeCommKernels.cu
new file mode 100644
index 000000000..b04f4280a
--- /dev/null
+++ b/cpp/tensorrt_llm/kernels/fusedMoeCommKernels.cu
@@ -0,0 +1,1372 @@
+/*
+ * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
+ *
+ * Licensed under the Apache License, Version 2.0 (the "License");
+ * you may not use this file except in compliance with the License.
+ * You may obtain a copy of the License at
+ *
+ *     http://www.apache.org/licenses/LICENSE-2.0
+ *
+ * Unless required by applicable law or agreed to in writing, software
+ * distributed under the License is distributed on an "AS IS" BASIS,
+ * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+ * See the License for the specific language governing permissions and
+ * limitations under the License.
+ */
+
+#include "tensorrt_llm/kernels/fusedMoeCommKernels.h"
+
+#include <type_traits>
+
+#include "tensorrt_llm/common/cudaUtils.h"
+#include "tensorrt_llm/common/logger.h"
+
+namespace tensorrt_llm
+{
+namespace kernels
+{
+
+static __device__ __forceinline__ uint32_t __as_ptr_smem(void const* __ptr)
+{
+    // Consider adding debug asserts here.
+    return static_cast<uint32_t>(__cvta_generic_to_shared(__ptr));
+}
+
+static __device__ __forceinline__ uint64_t __as_ptr_gmem(void const* __ptr)
+{
+    // Consider adding debug asserts here.
+    return static_cast<uint64_t>(__cvta_generic_to_global(__ptr));
+}
+
+__device__ __forceinline__ void fence_release_sys()
+{
+    asm volatile("fence.release.sys;" : : : "memory");
+}
+
+__device__ __forceinline__ void mbarrier_init(uint64_t* addr, uint32_t const& count)
+{
+#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
+    asm("mbarrier.init.shared.b64 [%0], %1;" : : "r"(__as_ptr_smem(addr)), "r"(count) : "memory");
+#endif
+}
+
+__device__ __forceinline__ void mbarrier_expect_tx(uint64_t* addr, const uint32_t txCount)
+{
+#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
+    asm("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
+        :
+        : "r"(__as_ptr_smem(addr)), "r"(txCount)
+        : "memory");
+#endif
+}
+
+__device__ __forceinline__ uint64_t mbarrier_arrive(uint64_t* addr)
+{
+#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
+    uint64_t state;
+    asm("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "r"(__as_ptr_smem(addr)) : "memory");
+    return state;
+#else
+    return 0;
+#endif
+}
+
+__device__ __forceinline__ uint64_t mbarrier_arrive_expect_tx(uint64_t* addr, const uint32_t txCount)
+{
+#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
+    uint64_t state;
+    asm("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
+        : "=l"(state)
+        : "r"(__as_ptr_smem(addr)), "r"(txCount)
+        : "memory");
+    return state;
+#else
+    return 0;
+#endif
+}
+
+__device__ __forceinline__ bool mbarrier_try_wait_parity(uint64_t* addr, uint32_t const& phaseParity)
+{
+#i
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

