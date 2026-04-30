# Performance Optimization Analysis - Part 5

Commits 117 to 145 of 283

---

## 6fc6f70a68 - [https://nvbugs/5441729][test] Fix test_modeling_llama_min_latency.py failures (#7478)

- **Date**: 2025-10-13
- **Author**: Po-Han Huang (NVIDIA)
- **Categories**: Throughput/Latency

### Optimization Techniques

- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase
- Disaggregated serving

### Changed Files

```
tensorrt_llm/_torch/models/modeling_llama.py       | 22 +++++++++++++++++-----
 .../modeling/test_modeling_llama_min_latency.py    |  8 +++++---
 2 files changed, 22 insertions(+), 8 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_llama.py b/tensorrt_llm/_torch/models/modeling_llama.py
index 003653597..0829f6bac 100644
--- a/tensorrt_llm/_torch/models/modeling_llama.py
+++ b/tensorrt_llm/_torch/models/modeling_llama.py
@@ -1003,16 +1003,28 @@ class Llama4VisionEncoder(nn.Module):
 
         self.dtype = self.pretrained_config.text_config.torch_dtype
 
-    def load_weights(self):
+    def load_weights(self, weights: Dict):
         module_dict = nn.ModuleDict({
             "vision_model":
             Llama4VisionModel(self.pretrained_config.vision_config),
             "multi_modal_projector":
             Llama4MultiModalProjector(self.pretrained_config),
         })
-        load_sharded_checkpoint(module_dict,
-                                self.pretrained_config._name_or_path,
-                                strict=False)
+
+        # If the named params are present in the weights, load them directly.
+        param_names = [name for name, _ in module_dict.named_parameters()]
+        if all(name in weights for name in param_names):
+            vision_encoder_weights = {
+                name: weights[name]
+                for name in param_names
+            }
+            module_dict.load_state_dict(vision_encoder_weights)
+
+        # Otherwise, load the weights from the checkpoint.
+        else:
+            load_sharded_checkpoint(module_dict,
+                                    self.pretrained_config._name_or_path,
+                                    strict=False)
 
         self.vision_model = module_dict["vision_model"].to(self.device)
         self.mm_projector = module_dict["multi_modal_projector"].to(self.device)
@@ -1295,7 +1307,7 @@ class Llama4ForConditionalGeneration(SpecDecOneEngineForCausalLM[Llama4Model,
 
     def load_weights(self, weights: Dict, weight_mapper: BaseWeightMapper):
         if not DISAGG:
-            self.mm_encoder.load_weights()
+            self.mm_encoder.load_weights(weights)
 
         # Temporarily detach mm_encoder so the TRT-LLM loader doesn't try to load it
         had_mm_encoder = hasattr(self, "mm_encoder")
diff --git a/tests/unittest/_torch/modeling/test_modeling_llama_min_latency.py b/tests/unittest/_torch/modeling/test_modeling_llama_min_latency.py
index 7b3e74a1b..9f96f146b 100644
--- a/tests/unittest/_torch/modeling/test_modeling_llama_min_latency.py
+++ b/tests/unittest/_torch/modeling/test_modeling_llama_min_latency.py
@@ -266,10 +266,12 @@ class TestLlama4MinLatency(unittest.TestCase):
         attention_backend = "TRTLLM"
         metadata_cls = get_attention_backend(attention_backend).Metadata
 
-        if transformers.__version__ >= "4.55.0":
+        if transformers.__version__ >= "4.55.0" \
+            and transformers.__version__ < "4.56.1":
             self.skipTest(
-                "The transformers 4.55.0 has accuracy issues while 4.33.1 works fine. "
-                "https://nvbugspro.nvidia.com/bug/5441729")
+                "The transform
```

### Analysis Summary

Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 6fe89ea00f - [TRTLLM-9819][perf] Reuse alltoall workspace for CuteDSL MoE output (#9840)

- **Date**: 2025-12-19
- **Author**: Enwei Zhu
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
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU

### Changed Files

```
cpp/tensorrt_llm/thop/cuteDslMoeUtilsOp.cpp        | 30 ++++++--
 tensorrt_llm/_torch/compilation/utils.py           |  3 +
 .../_torch/modules/fused_moe/configurable_moe.py   | 26 ++++---
 .../_torch/modules/fused_moe/fused_moe_cute_dsl.py | 82 +++++++++++++---------
 .../_torch/modules/fused_moe/fused_moe_cutlass.py  |  3 +
 .../modules/fused_moe/fused_moe_trtllm_gen.py      |  3 +
 tensorrt_llm/_torch/modules/fused_moe/interface.py |  5 ++
 7 files changed, 98 insertions(+), 54 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/thop/cuteDslMoeUtilsOp.cpp b/cpp/tensorrt_llm/thop/cuteDslMoeUtilsOp.cpp
index 770c1459f..5f17e2372 100644
--- a/cpp/tensorrt_llm/thop/cuteDslMoeUtilsOp.cpp
+++ b/cpp/tensorrt_llm/thop/cuteDslMoeUtilsOp.cpp
@@ -205,24 +205,26 @@ std::tuple<torch::Tensor, torch::optional<torch::Tensor>> moe_permute(torch::Ten
 
 // Unpermute
 
-torch::Tensor moe_unpermute(torch::Tensor const& permuted_input, torch::Tensor const& expanded_idx_to_permuted_idx,
-    torch::Tensor const& topk_scales)
+void moe_unpermute_inplace(torch::Tensor const& permuted_input, torch::Tensor const& output,
+    torch::Tensor const& expanded_idx_to_permuted_idx, torch::Tensor const& topk_scales)
 {
     TORCH_CHECK(permuted_input.dim() == 2, "permuted_input must be 2D.");
     int64_t const max_num_permuted_tokens = permuted_input.size(0);
     int64_t const hidden_size = permuted_input.size(1);
+    TORCH_CHECK(output.dim() == 2, "output must be 2D.");
+    int64_t const num_tokens = output.size(0);
+    TORCH_CHECK(output.size(1) == hidden_size, "output.size(1) must be hidden_size.");
+
     TORCH_CHECK(expanded_idx_to_permuted_idx.dim() == 2, "expanded_idx_to_permuted_idx must be 2D.");
-    int64_t const num_tokens = expanded_idx_to_permuted_idx.size(0);
+    TORCH_CHECK(
+        expanded_idx_to_permuted_idx.size(0) == num_tokens, "expanded_idx_to_permuted_idx.size(0) must be num_tokens.");
     int64_t const top_k = expanded_idx_to_permuted_idx.size(1);
     TORCH_CHECK(topk_scales.dim() == 2, "topk_scales must be 2D.");
     TORCH_CHECK(topk_scales.size(0) == num_tokens, "topk_scales.size(0) must be num_tokens.");
     TORCH_CHECK(topk_scales.size(1) == top_k, "topk_scales.size(1) must be top_k.");
-
     TORCH_CHECK(max_num_permuted_tokens >= num_tokens * top_k,
         "max_num_permuted_tokens must be greater than or equal to num_tokens * top_k.");
 
-    auto output
-        = torch::empty({num_tokens, hidden_size}, torch::dtype(permuted_input.scalar_type()).device(torch::kCUDA));
     auto const& stream = at::cuda::getCurrentCUDAStream(permuted_input.get_device());
 
 #define DISPATCH_MOE_UNPERMUTE(InputType, TopKScaleType)                                                               \
@@ -253,7 +255,19 @@ torch::Tensor moe_unpermute(torch::Tensor const& permuted_input, torch::Tensor c
     }
 
 #undef DISPATCH_MOE_UNPERMUTE
+}
 
+torch::Tensor moe_unpermute(torch::Tensor const& permuted_input, torch::Tensor const& expanded_idx_to_permuted_idx,
+    torch::Tensor const& topk_scales)
+{
+    TORCH_CHECK(permuted_input.dim() == 2, "permuted_input must be 2D.");
+    int64_t const hidden_size = permuted_input.size(1);
+    TORCH_CHECK(expanded_idx_to_permuted_idx.dim() == 2, "expanded_idx_to_permuted_idx must be 2D.");
+    int64_t const num_tokens = expanded_idx_to_permuted_idx.size(0);
+
+    auto output
+        = torch::empty({num_tokens, hidden_size}, torch::dtype(permuted_input.scalar_type()).device(torch::kCUDA));
+    moe_unpermute_inp
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 7081f254cf - [None][perf] Add custom indexer k cache scatter op (#8960)

- **Date**: 2025-11-07
- **Author**: Chang Liu
- **Categories**: Cache Optimization

### Optimization Techniques

- Custom CUDA kernel
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
cpp/tensorrt_llm/kernels/IndexerKCacheScatter.h    |  30 ++++
 cpp/tensorrt_llm/kernels/indexerKCacheScatter.cu   | 152 +++++++++++++++++
 cpp/tensorrt_llm/thop/CMakeLists.txt               |   1 +
 cpp/tensorrt_llm/thop/IndexerKCacheScatterOp.cpp   | 106 ++++++++++++
 .../_torch/attention_backend/sparse/dsa.py         |  20 +--
 .../_torch/attention/sparse/test_dsa_indexer.py    | 188 +++++++++++++++++++--
 6 files changed, 470 insertions(+), 27 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/IndexerKCacheScatter.h b/cpp/tensorrt_llm/kernels/IndexerKCacheScatter.h
new file mode 100644
index 000000000..b0ac689d3
--- /dev/null
+++ b/cpp/tensorrt_llm/kernels/IndexerKCacheScatter.h
@@ -0,0 +1,30 @@
+/*
+ * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
+#pragma once
+
+#include "tensorrt_llm/common/cudaUtils.h"
+
+namespace tensorrt_llm::kernels
+{
+
+void invokeIndexerKCacheScatter(uint8_t const* k_fp8_bytes, uint8_t const* k_scale_bytes, uint8_t* k_cache,
+    int64_t const* slot_mapping_fp8, int64_t const* slot_mapping_scale, int32_t num_tokens, int32_t head_dim,
+    int32_t scale_size, int32_t cache_dim_0, int32_t cache_dim_1, int32_t cache_dim_2, int32_t cache_dim_3,
+    int64_t cache_stride_0, int64_t cache_stride_1, int64_t cache_stride_2, int64_t cache_stride_3,
+    cudaStream_t stream = 0);
+
+}
diff --git a/cpp/tensorrt_llm/kernels/indexerKCacheScatter.cu b/cpp/tensorrt_llm/kernels/indexerKCacheScatter.cu
new file mode 100644
index 000000000..3cb35273a
--- /dev/null
+++ b/cpp/tensorrt_llm/kernels/indexerKCacheScatter.cu
@@ -0,0 +1,152 @@
+/*
+ * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
+#include "IndexerKCacheScatter.h"
+#include "tensorrt_llm/common/assert.h"
+#include "tensorrt_llm/common/cudaUtils.h"
+
+namespace tensorrt_llm::kernels
+{
+
+namespace
+{
+/**
+ * Given a flat element index and tensor shape [d0, d1, d2, d3] with strides [s0, s1, s2, s3],
+ * find the actual memory offset within the given k cache pool using the strides.
+ */
+__device__ __forceinline__ int64_t flatIndexToMemoryOffset(
+    int64_t flat_idx, int32_t d0, int32_t d1, int32_t d2, int32_t d3, int64_t s0, int64_t s1, int64_t s2, int64_t s3)
+{
+    // Unravel from innermost to outermost dimension
+    int32_t i3 = flat_idx % d3;
+    
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## 719e82c429 - [TRTLLM-10030][perf] beam search (remove GPU sync + fix batching + refactor) (#11276)

- **Date**: 2026-02-05
- **Author**: mpikulski
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Pinned memory
- Reduce synchronization overhead
- Speculative decoding
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/sampler.py          | 298 +++++++++++----------
 tensorrt_llm/_torch/pyexecutor/sampling_utils.py   |  20 +-
 .../_torch/pyexecutor/sampling_utils_flashinfer.py | 176 ++++++------
 tests/unittest/_torch/sampler/test_beam_search.py  |   1 -
 .../unittest/_torch/sampler/test_torch_sampler.py  |   6 +-
 5 files changed, 253 insertions(+), 248 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index 876858bfe..ac29abe97 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -21,7 +21,7 @@ from concurrent import futures
 from dataclasses import dataclass
 from functools import cached_property
 from itertools import repeat
-from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, cast
+from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeAlias, TypeVar, cast
 
 import numpy as np
 import torch
@@ -264,42 +264,11 @@ class RequestGroupValue:
     need_processed_logprobs: torch.Tensor
     need_raw_logprobs: torch.Tensor
 
-    def __iter__(self):
-        return iter(
-            (
-                self.indices,
-                self.strategies,
-                self.speculation_needs_probs_indices,
-                self.need_processed_logprobs,
-                self.need_raw_logprobs,
-            )
-        )
-
-    def __len__(self):
-        return 5
-
 
 @dataclass(kw_only=True, frozen=True)
 class RequestGroupValueWithMetadata(RequestGroupValue):
     metadata: StrategyMetadata | None
 
-    @override
-    def __iter__(self):
-        return iter(
-            (
-                self.indices,
-                self.strategies,
-                self.speculation_needs_probs_indices,
-                self.need_processed_logprobs,
-                self.need_raw_logprobs,
-                self.metadata,
-            )
-        )
-
-    @override
-    def __len__(self):
-        return 6
-
 
 class EarlyStopWithMMResult(Sampler):
     """
@@ -307,7 +276,7 @@ class EarlyStopWithMMResult(Sampler):
     """
 
     @override
-    def sample_async(
+    def sample_async(  # type: ignore
         self,
         scheduled_requests: ScheduledRequests,
         model_outputs,
@@ -322,7 +291,7 @@ class EarlyStopWithMMResult(Sampler):
         return SampleStateWithMMResult(scheduled_requests=scheduled_requests, data=data)
 
     @override
-    def update_requests(
+    def update_requests(  # type: ignore
         self,
         state: SampleStateWithMMResult,
         resource_manager: Optional[ResourceManager] = None,
@@ -341,9 +310,9 @@ class EarlyStopWithMMResult(Sampler):
             request.state = LlmRequestState.GENERATION_COMPLETE
             # NOTE: This is a hack: set finish reason manually and set the beam 0
             request.set_finished_reason(FinishReason.LENGTH, 0)
-            if len(mm_embedding) != sum(request.multimodal_lengths):
+            if len(mm_embedding) != sum(request.multimodal_lengths):  # type: ignore
                 raise ValueError(
-                    f"mm_embedding shape mismatch: {len(mm_embedding)} != {sum(request.multimodal_lengths)}"
+                    f"mm_embedding shape mismatch: {len(mm_embedding)} != {sum(request.multimodal_lengths)}"  # type: ignore
                 )
 
             request.py_result.ap
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Pinned (page-locked) memory enables faster host-to-device data transfers. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 72ef732bcf - [TRTLLM-10147][perf] Balanced random MoE workload generator for CuteDSL kernel UT, autotuner and layerwise benchmark (#10279)

- **Date**: 2026-01-25
- **Author**: Enwei Zhu
- **Categories**: Kernel Optimization

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Integer quantization
- KV cache optimization
- Parallelism optimization
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU

### Changed Files

```
.../_torch/custom_ops/cute_dsl_custom_ops.py       | 321 ++++++++++-----------
 .../_torch/modules/fused_moe/fused_moe_cute_dsl.py |   4 +-
 tensorrt_llm/tools/layer_wise_benchmarks/runner.py |  39 ++-
 tests/scripts/cute_dsl_kernels/README.md           |  11 +
 .../cute_dsl_kernels/moe_workload_generator.py     | 176 +++++++++++
 .../_torch/thop/parallel/test_cute_dsl_moe.py      | 103 +++----
 6 files changed, 418 insertions(+), 236 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py b/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
index 771e7ed7c..ff37aa7d9 100644
--- a/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
+++ b/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
@@ -1,4 +1,5 @@
 import itertools
+import math
 from typing import List, Optional, Tuple
 
 import torch
@@ -31,13 +32,19 @@ class GroupedGemmInputsHelper:
     IDX_A = 0
     IDX_SHAPE_INFER = IDX_A  # Default: use a tensor for shape inference
 
-    def __init__(self, num_experts: int, top_k: int, num_local_experts: int,
-                 local_expert_offset: int, tile_size: int):
+    def __init__(self,
+                 num_experts: int,
+                 top_k: int,
+                 num_local_experts: int,
+                 local_expert_offset: int,
+                 tile_size: int,
+                 seed: int = 515):
         self.num_experts = num_experts
         self.top_k = top_k
         self.num_local_experts = num_local_experts
         self.local_expert_offset = local_expert_offset
         self.tile_size = tile_size
+        self.seed = seed
         # Padding values should never be accessed.
         # Intentionally use a large padding value to expose issues early.
         self.pad_val = int(2e9)
@@ -82,118 +89,134 @@ class GroupedGemmInputsHelper:
             self, input_shapes: List[torch.Size]) -> int:
         return self.infer_shape_max_num_tiles(input_shapes) * self.tile_size
 
-    def generate_num_tokens_per_expert(self, num_tokens: int) -> List[int]:
-        average_num_tokens_per_expert = num_tokens * self.top_k / self.num_experts
-        balance = 0
-        num_tokens_per_expert = []
-        for i in range(self.num_local_experts):
-            balance += average_num_tokens_per_expert
-            if balance <= 1e-3:
-                continue
-            curr_num_tokens = int(balance) + 1
-            num_tokens_per_expert.append(curr_num_tokens)
-            balance -= curr_num_tokens
+    def generate_num_tokens_per_expert(self,
+                                       num_tokens: int,
+                                       approx_max_load: bool = False
+                                       ) -> List[int]:
+        ep_size = self.num_experts // self.num_local_experts
+        average_num_tokens_per_rank = num_tokens * self.top_k / ep_size
+
+        if approx_max_load:
+            # https://en.wikipedia.org/wiki/Balls_into_bins_problem
+            # The constant c can be measured empirically, we choose 1.0 for simplicity.
+            c = 1.0
+            extra_num_tokens_on_curr_rank = c * math.sqrt(
+                average_num_tokens_per_rank * math.log(ep_size))
+            num_tokens_on_curr_rank = math.ceil(average_num_tokens_per_rank +
+                                                extra_num_tokens_on_curr_rank)
+        else:
+            num_tokens_on_curr_rank = math.ceil(average_num_tokens_per_rank)
+
+        num_tokens_
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 73b8a95049 - feat: Use inference mode in update_requests to improve perf of TRTLLM Sampler (#5538)

- **Date**: 2025-06-27
- **Author**: Daniel Cámpora
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- Batching optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/sampler.py | 13 ++++++++-----
 1 file changed, 8 insertions(+), 5 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index c96ad3356..560d3a213 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -602,6 +602,7 @@ class TRTLLMSampler(Sampler):
             return req.sampling_config.beam_width
         return 0
 
+    @torch.inference_mode()
     def sample_async(self, scheduled_requests: ScheduledRequests,
                      model_outputs) -> SampleStateTRTLLM:
         batch_size = scheduled_requests.batch_size
@@ -685,6 +686,7 @@ class TRTLLMSampler(Sampler):
                                  host=host,
                                  sampler_event=sampler_event)
 
+    @torch.inference_mode()
     def update_requests(self, state: SampleStateTRTLLM):
         assert isinstance(state, SampleStateTRTLLM)
 
@@ -698,7 +700,6 @@ class TRTLLMSampler(Sampler):
 
         new_tokens_host = state.host.new_tokens
         finished_sum_host = state.host.finished_sum
-        finish_reasons_host = state.host.finish_reasons
         sequence_lengths_host_data = state.host.sequence_lengths
 
         for request in scheduled_requests.all_requests:
@@ -744,10 +745,12 @@ class TRTLLMSampler(Sampler):
                         state.host.cum_log_probs[seq_slot * beam_width +
                                                  beam].item())
 
-                finish_reason = FinishedState(
-                    finish_reasons_host[seq_slot * beam_width +
-                                        beam].item()).to_finish_reason()
-                request.set_finished_reason(finish_reason, beam)
+                finished_state = FinishedState(
+                    state.host.finish_reasons[seq_slot * beam_width +
+                                              beam].item())
+                if finished_state.is_finished:
+                    finish_reason = finished_state.to_finish_reason()
+                    request.set_finished_reason(finish_reason, beam)
 
             if request.py_return_log_probs:
                 request.py_result.append_log_probs([log_probs], cum_log_probs)

```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 73fca4e0bd - [None][feat] Mamba optimization and mixed quantization support for nemotron-h (#11972)

- **Date**: 2026-03-11
- **Author**: Wanli Jiang
- **Categories**: Quantization Optimization

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Triton kernel
- Speculative decoding
- MoE optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
ATTRIBUTIONS-Python.md                             |   2 +-
 examples/llm-api/quickstart_advanced.py            |  16 +++
 requirements.txt                                   |   2 +-
 security_scanning/pyproject.toml                   |   2 +-
 tensorrt_llm/_torch/models/modeling_nemotron_h.py  |  42 +++++++-
 .../_torch/modules/mamba/mamba2_metadata.py        |   4 +-
 tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py  | 112 ++++++++++++++-------
 tensorrt_llm/_torch/pyexecutor/model_loader.py     |  13 ++-
 tensorrt_llm/llmapi/llm_args.py                    |  16 +++
 tensorrt_llm/llmapi/llm_utils.py                   |  10 ++
 tensorrt_llm/models/modeling_utils.py              |  11 ++
 .../api_stability/references/quant_config.yaml     |   6 ++
 12 files changed, 186 insertions(+), 50 deletions(-)
```

### Diff Preview

```diff
diff --git a/ATTRIBUTIONS-Python.md b/ATTRIBUTIONS-Python.md
index b9975173a..fbe7391de 100644
--- a/ATTRIBUTIONS-Python.md
+++ b/ATTRIBUTIONS-Python.md
@@ -5261,7 +5261,7 @@ For more information, please refer to <http://unlicense.org>
   - `Tracker`: https://github.com/tox-dev/py-filelock/issues
 
 
-## flashinfer-python (0.6.4)
+## flashinfer-python (0.6.6)
 
 ### Licenses
 License: `Apache-2.0`
diff --git a/examples/llm-api/quickstart_advanced.py b/examples/llm-api/quickstart_advanced.py
index c6468e986..be6cf5425 100644
--- a/examples/llm-api/quickstart_advanced.py
+++ b/examples/llm-api/quickstart_advanced.py
@@ -106,6 +106,20 @@ def add_llm_args(parser):
                         default='bfloat16',
                         choices=['auto', 'float16', 'bfloat16', 'float32'],
                         help='Data type for Mamba SSM cache.')
+    parser.add_argument(
+        '--mamba_ssm_stochastic_rounding',
+        default=False,
+        action='store_true',
+        help=
+        'Enable stochastic rounding for Mamba SSM state updates (fp16 only, FlashInfer limitation).'
+    )
+    parser.add_argument(
+        '--mamba_ssm_philox_rounds',
+        type=int,
+        default=10,
+        help=
+        'Number of Philox rounds for stochastic rounding PRNG (default: 10). Higher values give better randomness.'
+    )
     parser.add_argument('--log_kv_cache_events',
                         default=False,
                         action='store_true')
@@ -222,6 +236,8 @@ def setup_llm(args, **kwargs):
         tokens_per_block=args.tokens_per_block,
         use_kv_cache_manager_v2=args.use_kv_cache_manager_v2,
         mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
+        mamba_ssm_stochastic_rounding=args.mamba_ssm_stochastic_rounding,
+        mamba_ssm_philox_rounds=args.mamba_ssm_philox_rounds,
         event_buffer_max_size=1024 if args.log_kv_cache_events else 0)
 
     spec_decode_algo = args.spec_decode_algo.upper(
diff --git a/requirements.txt b/requirements.txt
index 178d3a46f..baa857a54 100644
--- a/requirements.txt
+++ b/requirements.txt
@@ -54,7 +54,7 @@ ordered-set
 peft
 patchelf
 einops
-flashinfer-python==0.6.4
+flashinfer-python==0.6.6
 opencv-python-headless
 xgrammar==0.1.32
 llguidance==0.7.29
diff --git a/security_scanning/pyproject.toml b/security_scanning/pyproject.toml
index c2b136642..0b1cdb6ad 100644
--- a/security_scanning/pyproject.toml
+++ b/security_scanning/pyproject.toml
@@ -56,7 +56,7 @@ dependencies = [
     "peft (>=0.18.1,<0.19.0)",
     "patchelf (>=0.17.2.4,<0.18.0.0)",
     "einops (>=0.8.2,<0.9.0)",
-    "flashinfer-python (==0.6.4)",
+    "flashinfer-python (==0.6.6)",
     "xgrammar (==0.1.32)",
     "llguidance (==0.7.29)",
     "jsonschema (>=4.26.0,<5.0.0)",
diff --git a/tensorrt_llm/_torch/models/modeling_nemotron_h.py b/tensorrt_llm/_torch/models/modeling_nemotron_h.py
index 0007efeef..f0353a186 100644
--- a/tensorrt_llm/_torch/models/modeling_nemotron_h.py
+++ b/tensorrt_llm/_torch/mode
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 7588029763 - [None][feat] Async pp send for PPCommTorch. (#9976)

- **Date**: 2025-12-15
- **Author**: Yuxian Qiu
- **Categories**: Parallelism/Async

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- Integer quantization
- KV cache optimization
- Parallelism optimization
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
cpp/tensorrt_llm/thop/ncclCommunicatorOp.cpp    |  2 +
 tensorrt_llm/_torch/device_mesh.py              | 12 ++---
 tensorrt_llm/_torch/distributed/communicator.py | 59 +++++++------------------
 tensorrt_llm/mapping.py                         | 11 ++---
 4 files changed, 30 insertions(+), 54 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/thop/ncclCommunicatorOp.cpp b/cpp/tensorrt_llm/thop/ncclCommunicatorOp.cpp
index 75ae96f36..a45fa955a 100644
--- a/cpp/tensorrt_llm/thop/ncclCommunicatorOp.cpp
+++ b/cpp/tensorrt_llm/thop/ncclCommunicatorOp.cpp
@@ -33,6 +33,7 @@ NcclCommunicatorOp::NcclCommunicatorOp(int64_t worldSize, int64_t rank)
 
 void NcclCommunicatorOp::send(th::Tensor tensor, int64_t toRank) const
 {
+    tensor.record_stream(at::cuda::getCurrentCUDAStream());
     auto ptr = static_cast<std::uint8_t*>(tensor.data_ptr());
     size_t const size = tensor.numel() * th::elementSize(th::typeMetaToScalarType(tensor.dtype()));
     tensorrt_llm::runtime::CudaStream cudaStream{at::cuda::getCurrentCUDAStream().stream(), mRank, false};
@@ -41,6 +42,7 @@ void NcclCommunicatorOp::send(th::Tensor tensor, int64_t toRank) const
 
 void NcclCommunicatorOp::recv(th::Tensor& tensor, int64_t fromRank) const
 {
+    tensor.record_stream(at::cuda::getCurrentCUDAStream());
     auto ptr = static_cast<std::uint8_t*>(tensor.data_ptr());
     size_t const size = tensor.numel() * th::elementSize(th::typeMetaToScalarType(tensor.dtype()));
     tensorrt_llm::runtime::CudaStream cudaStream{at::cuda::getCurrentCUDAStream().stream(), mRank, false};
diff --git a/tensorrt_llm/_torch/device_mesh.py b/tensorrt_llm/_torch/device_mesh.py
index ca8db8338..b5034f8ef 100644
--- a/tensorrt_llm/_torch/device_mesh.py
+++ b/tensorrt_llm/_torch/device_mesh.py
@@ -3,7 +3,7 @@ from typing import TYPE_CHECKING, List
 
 import torch
 import torch.distributed as dist
-from torch.distributed import get_process_group_ranks
+from torch.distributed import ProcessGroup, get_process_group_ranks
 from torch.distributed.device_mesh import init_device_mesh
 
 from tensorrt_llm.logger import logger
@@ -48,27 +48,27 @@ class DeviceMeshTopologyImpl(_MappingBaseForTypeCheck):
     # Access Torch ProcessGroup
     @property
     @require_device_mesh
-    def tp_group_pg(self):
+    def tp_group_pg(self) -> ProcessGroup:
         return self._get_mesh_dim_by_name('tp').get_group()
 
     @property
     @require_device_mesh
-    def pp_group_pg(self):
+    def pp_group_pg(self) -> ProcessGroup:
         return self._get_mesh_dim_by_name('pp').get_group()
 
     @property
     @require_device_mesh
-    def cp_group_pg(self):
+    def cp_group_pg(self) -> ProcessGroup:
         return self._get_mesh_dim_by_name('cp').get_group()
 
     @property
     @require_device_mesh
-    def moe_tp_group_pg(self):
+    def moe_tp_group_pg(self) -> ProcessGroup:
         return self._get_mesh_dim_by_name('moe_tp').get_group()
 
     @property
     @require_device_mesh
-    def moe_ep_group_pg(self):
+    def moe_ep_group_pg(self) -> ProcessGroup:
         return self._get_mesh_dim_by_name('moe_ep').get_group()
 
     # Access rank
diff --git a/tensorrt_llm/_torch/distributed/communicator.py b/tensorrt_llm/_torch/distributed/communicator.py
index 93457691b..18c7e7a63 100644
--- a/tensorrt_llm/_torch/distributed/communicat
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 763bce523b - [None][test] Enable DeepGemm + DeepEPLowLatency MoE test combination (#11876)

- **Date**: 2026-03-03
- **Author**: Iman Tabrizian
- **Categories**: Throughput/Latency

### Optimization Techniques

- Operator fusion
- FP8 quantization
- MoE optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tests/unittest/_torch/modules/moe/moe_test_utils.py | 12 ------------
 1 file changed, 12 deletions(-)
```

### Diff Preview

```diff
diff --git a/tests/unittest/_torch/modules/moe/moe_test_utils.py b/tests/unittest/_torch/modules/moe/moe_test_utils.py
index 6c7f750fe..5db2ddbc0 100644
--- a/tests/unittest/_torch/modules/moe/moe_test_utils.py
+++ b/tests/unittest/_torch/modules/moe/moe_test_utils.py
@@ -408,18 +408,6 @@ def should_skip_deepgemm(
     if backend_type != MoeBackendType.DEEPGEMM:
         return None
 
-    # DeepGemm workspace allocation in set_strides (fused_moe_deepgemm.py) uses a
-    # storage size that is 4x too small when combined with DeepEPLowLatency dispatch.
-    # The workspace is allocated based on assumptions that do not account for the
-    # DeepEPLowLatency output format ([num_local_experts, ep_size * max_tokens, hidden_size]).
-    if comm_method == "DEEPEPLOWLATENCY":
-        return (
-            "[Potential Bug] DeepGemmFusedMoE workspace allocation is incompatible "
-            "with DeepEPLowLatency: set_strides requires storage of "
-            "[num_local_experts * tokens * hidden_size] bytes but the allocated "
-            "workspace is ~4x too small, causing setStorage out of bounds."
-        )
-
     # Issue: DEEPGEMM + FP8_BLOCK_SCALES crashes with CUDA illegal memory access
     # on large expert counts (e.g. e384_k8_h7168_i2048) during post_load_weights().
     # The crash occurs in get_col_major_tma_aligned_packed_tensor (fp8_utils.py)

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 77288d3671 - fix [nvbug5351244]: test_mpi_session submit sync/async (#5608)

- **Date**: 2025-07-01
- **Author**: Yan Chunwei
- **Categories**: Parallelism/Async

### Optimization Techniques

- General code optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tests/unittest/llmapi/_test_remote_mpi_session.sh | 12 ++++++++++++
 tests/unittest/llmapi/test_mpi_session.py         |  1 -
 2 files changed, 12 insertions(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tests/unittest/llmapi/_test_remote_mpi_session.sh b/tests/unittest/llmapi/_test_remote_mpi_session.sh
new file mode 100644
index 000000000..01eff4b27
--- /dev/null
+++ b/tests/unittest/llmapi/_test_remote_mpi_session.sh
@@ -0,0 +1,12 @@
+#!/bin/bash
+set -ex
+
+task=$1
+
+echo "Starting remote MPI session test with task: $task"
+echo "MPI processes: 2"
+
+# Add timeout to prevent infinite hanging
+timeout 60 mpirun -np 2 trtllm-llmapi-launch python3 _run_mpi_comm_task.py --task_type $task
+
+echo "Remote MPI session test completed"
diff --git a/tests/unittest/llmapi/test_mpi_session.py b/tests/unittest/llmapi/test_mpi_session.py
index ae8b0eba7..6d19955ff 100644
--- a/tests/unittest/llmapi/test_mpi_session.py
+++ b/tests/unittest/llmapi/test_mpi_session.py
@@ -54,7 +54,6 @@ def run_client(server_addr, values_to_process):
         return f"Error in client: {str(e)}"
 
 
-@pytest.mark.skip(reason="https://nvbugs/5351244")
 @pytest.mark.parametrize("task_type", ["submit", "submit_sync"])
 def test_remote_mpi_session(task_type: Literal["submit", "submit_sync"]):
     """Test RemoteMpiPoolSessionClient and RemoteMpiPoolSessionServer interaction"""

```

### Analysis Summary

This commit applies performance optimization techniques to improve inference efficiency. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 775c2736d9 - [TRTLLM-9040][perf] Make preprocessing async (#11459)

- **Date**: 2026-02-19
- **Author**: William Zhang
- **Categories**: Parallelism/Async, Host-side Optimization

### Optimization Techniques

- Async/stream-based execution
- KV cache optimization

### Applicable Conditions

- Decode/generation phase
- Disaggregated serving

### Changed Files

```
tensorrt_llm/llmapi/llm.py                         | 143 ++++++++++++++++-----
 tensorrt_llm/serve/openai_server.py                |   9 +-
 tests/unittest/api_stability/references/llm.yaml   |  13 ++
 .../api_stability/references_committed/llm.yaml    |   3 +-
 4 files changed, 131 insertions(+), 37 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/llmapi/llm.py b/tensorrt_llm/llmapi/llm.py
index e95a0d455..7187ec239 100644
--- a/tensorrt_llm/llmapi/llm.py
+++ b/tensorrt_llm/llmapi/llm.py
@@ -7,8 +7,9 @@ import tempfile
 import time
 import weakref
 from collections.abc import Mapping
+from dataclasses import dataclass
 from pathlib import Path
-from typing import Any, List, Literal, Optional, Sequence, Union, cast
+from typing import Any, List, Literal, Optional, Sequence, Tuple, Union, cast
 
 import transformers
 from tqdm import tqdm
@@ -116,6 +117,18 @@ TORCH_LLM_DOCSTRING = TORCH_LLMARGS_EXPLICIT_DOCSTRING + """
 """
 
 
+@dataclass
+class PreprocessedInputs:
+    """Light structure for holding preprocessed inputs.
+
+    Can be passed to `generate_async` to skip preprocessing.
+    """
+
+    prompt_token_ids: List[int]
+    query_token_ids: Optional[List[int]] = None
+    multimodal_params: Optional[MultimodalParams] = None
+
+
 class BaseLLM:
     """
     The base class for all LLM classes.
@@ -361,7 +374,7 @@ class BaseLLM:
     @nvtx_range_debug("LLM.generate_async", color="green", category="LLM")
     def generate_async(
         self,
-        inputs: PromptInputs,
+        inputs: Union[PromptInputs, PreprocessedInputs],
         sampling_params: Optional[SamplingParams] = None,
         lora_request: Optional[LoRARequest] = None,
         prompt_adapter_request: Optional[PromptAdapterRequest] = None,
@@ -377,7 +390,7 @@ class BaseLLM:
         Asynchronous generation accepts single prompt only.
 
         Args:
-            inputs (tensorrt_llm.inputs.data.PromptInputs): The prompt text or token ids; it must be single prompt.
+            inputs (Union[tensorrt_llm.inputs.data.PromptInputs, tensorrt_llm.llmapi.llm.PreprocessedInputs]): The prompt text or token ids, or a `PreprocessedInputs` returned by `preprocess`. If the latter, preprocessing will be skipped by this method.
             sampling_params (tensorrt_llm.sampling_params.SamplingParams, optional): The sampling params for the generation. Defaults to None.
                 A default one will be used if not provided.
             lora_request (tensorrt_llm.executor.request.LoRARequest, optional): LoRA request to use for generation, if any. Defaults to None.
@@ -400,6 +413,7 @@ class BaseLLM:
         ) if self.args.return_perf_metrics else None
 
         sampling_params = self._prepare_sampling_params(sampling_params)
+
         cache_salt_id = get_cache_salt_id(
             cache_salt) if cache_salt is not None else None
         # With pytorch backend, py_executor has logic to handle max_tokens of 1,
@@ -407,11 +421,66 @@ class BaseLLM:
         # TODO: Also support for trt backend
         is_ctx_only = disaggregated_params is not None and disaggregated_params.request_type == "context_only"
         is_gen_only = disaggregated_params is not None and disaggregated_params.request_type == "generation_only"
-        is_mm_disagg = disaggregated_params is not None and disaggregated_params.
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 7a5e0fd300 - [fix] Fix Llama4 min-latency import error (#5209)

- **Date**: 2025-06-15
- **Author**: Yilin Fan
- **Categories**: Throughput/Latency

### Optimization Techniques

- Operator fusion
- FP8 quantization
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/modules/fused_moe/__init__.py | 28 ++++++++++++++++-------
 1 file changed, 20 insertions(+), 8 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fused_moe/__init__.py b/tensorrt_llm/_torch/modules/fused_moe/__init__.py
index 2f741d479..c1c699b55 100644
--- a/tensorrt_llm/_torch/modules/fused_moe/__init__.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/__init__.py
@@ -4,6 +4,7 @@ from .fused_moe_trtllm_gen import TRTLLMGenFusedMoE
 from .fused_moe_vanilla import VanillaMoE
 from .interface import MoE, MoEWeightLoadingMode
 from .moe_load_balancer import MoeLoadBalancer
+from .quantization import FusedMoEQuantScalesFP8
 from .routing import (BaseMoeRoutingMethod, DeepSeekV3MoeRoutingMethod,
                       DefaultMoeRoutingMethod,
                       Llama4RenormalizeMoeRoutingMethod,
@@ -12,12 +13,23 @@ from .routing import (BaseMoeRoutingMethod, DeepSeekV3MoeRoutingMethod,
                       SparseMixerMoeRoutingMethod, StaticMoeRoutingMethod)
 
 __all__ = [
-    "VanillaMoE", "CutlassFusedMoE", "TRTLLMGenFusedMoE",
-    "BaseMoeRoutingMethod", "MoeLoadBalancer",
-    "RenormalizeNaiveMoeRoutingMethod", "Llama4RenormalizeMoeRoutingMethod",
-    "SparseMixerMoeRoutingMethod", "LoadBalancedMoeRoutingMethod",
-    "StaticMoeRoutingMethod", "DefaultMoeRoutingMethod",
-    "DeepSeekV3MoeRoutingMethod", "RoutingMethodType",
-    "RenormalizeMoeRoutingMethod", "MoE", "MoEWeightLoadingMode", "get_moe_cls",
-    "create_moe"
+    "VanillaMoE",
+    "CutlassFusedMoE",
+    "TRTLLMGenFusedMoE",
+    "BaseMoeRoutingMethod",
+    "MoeLoadBalancer",
+    "RenormalizeNaiveMoeRoutingMethod",
+    "Llama4RenormalizeMoeRoutingMethod",
+    "SparseMixerMoeRoutingMethod",
+    "LoadBalancedMoeRoutingMethod",
+    "StaticMoeRoutingMethod",
+    "DefaultMoeRoutingMethod",
+    "DeepSeekV3MoeRoutingMethod",
+    "RoutingMethodType",
+    "RenormalizeMoeRoutingMethod",
+    "MoE",
+    "MoEWeightLoadingMode",
+    "get_moe_cls",
+    "create_moe",
+    "FusedMoEQuantScalesFP8",
 ]

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 7aeac97e4e - [https://nvbugs/5622938][fix] Use async send_requests_to_next_pp. (#9041)

- **Date**: 2025-11-11
- **Author**: Yuxian Qiu
- **Categories**: Parallelism/Async

### Optimization Techniques

- Batching optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/executor_request_queue.py | 15 ++++++---------
 1 file changed, 6 insertions(+), 9 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/executor_request_queue.py b/tensorrt_llm/_torch/pyexecutor/executor_request_queue.py
index 7ba994345..c6a7616ea 100644
--- a/tensorrt_llm/_torch/pyexecutor/executor_request_queue.py
+++ b/tensorrt_llm/_torch/pyexecutor/executor_request_queue.py
@@ -66,6 +66,7 @@ class ExecutorRequestQueue:
         self.start_times = {}
         self.active = True
         self.batch_wait_timeout_ms = batch_wait_timeout_ms
+        self.send_requests_handler = None
 
         # State tracking
         self.num_fetch_requests = 0
@@ -609,15 +610,11 @@ class ExecutorRequestQueue:
 
         if not self.dist.is_last_pp_rank:
             with nvtx_range("send_requests_to_next_pp"):
-                if self._disable_mpi:
-                    isend_payload = self.dist.isend_object(
-                        payloads,
-                        self.dist.next_pp_rank,
-                        tag,
-                    )
-                    isend_payload.wait()
-                else:
-                    self.dist.send_object(payloads, self.dist.next_pp_rank, tag)
+                if self.send_requests_handler is not None:
+                    with nvtx_range("wait_prev_send_requests_handler"):
+                        self.send_requests_handler.wait()
+                self.send_requests_handler = self.dist.isend_object(
+                    payloads, self.dist.next_pp_rank, tag)
 
         return payloads
 

```

### Analysis Summary

This commit applies performance optimization techniques to improve inference efficiency. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 7b210ae9c3 - test: add unit tests for Llama4 min_latency code (#4980)

- **Date**: 2025-06-11
- **Author**: nvpohanh
- **Categories**: Throughput/Latency

### Optimization Techniques

- Torch compilation/JIT optimization
- Custom CUDA kernel
- Operator fusion
- FP8 quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- Prefill phase
- Decode/generation phase

### Changed Files

```
.../_torch/models/modeling_llama_min_latency.py    |   1 +
 .../modeling/test_modeling_llama_min_latency.py    | 443 +++++++++++++++++++++
 2 files changed, 444 insertions(+)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_llama_min_latency.py b/tensorrt_llm/_torch/models/modeling_llama_min_latency.py
index c0e6c1743..4c26e8c88 100644
--- a/tensorrt_llm/_torch/models/modeling_llama_min_latency.py
+++ b/tensorrt_llm/_torch/models/modeling_llama_min_latency.py
@@ -471,6 +471,7 @@ class Llama4MinLatencyFusedMoE(CutlassFusedMoE):
         if num_experts == 128 \
             and hidden_size == 5120 \
             and intermediate_size == 8192 \
+            and model_config.quant_config is not None \
             and model_config.quant_config.quant_mode.has_fp8_qdq() \
             and model_config.mapping.moe_tp_size == 8 \
             and model_config.mapping.moe_ep_size == 1 \
diff --git a/tests/unittest/_torch/modeling/test_modeling_llama_min_latency.py b/tests/unittest/_torch/modeling/test_modeling_llama_min_latency.py
new file mode 100644
index 000000000..4711155d1
--- /dev/null
+++ b/tests/unittest/_torch/modeling/test_modeling_llama_min_latency.py
@@ -0,0 +1,443 @@
+import unittest
+from copy import deepcopy
+from dataclasses import dataclass
+
+import torch
+from parameterized import parameterized
+from transformers import Llama4Config
+from transformers import \
+    Llama4ForConditionalGeneration as HFLlama4ForConditionalGeneration
+from transformers.cache_utils import DynamicCache
+from utils.util import getSMVersion
+
+import tensorrt_llm
+from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
+from tensorrt_llm._torch.metadata import KVCacheParams
+from tensorrt_llm._torch.model_config import ModelConfig
+from tensorrt_llm._torch.models.modeling_llama import \
+    Llama4ForConditionalGeneration
+from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
+from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import \
+    DecodingCUDAGraphRunner
+from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
+from tensorrt_llm.bindings.executor import KvCacheConfig
+from tensorrt_llm.mapping import Mapping
+from tensorrt_llm.models.modeling_utils import QuantConfig
+
+# This is Llama4 Maverick config but with only 2 layers.
+# 2 layers are needed to cover both MLP layer and MoE layer, as well as both
+# RoPE and no-RoPE layers.
+LLAMA_4_MAVERICK_TWO_LAYER_CONFIG = {
+    "architectures": ["Llama4ForConditionalGeneration"],
+    "boi_token_index": 200080,
+    "eoi_token_index": 200081,
+    "image_token_index": 200092,
+    "model_type": "llama4",
+    "text_config": {
+        "_attn_implementation_autoset": True,
+        "attention_bias": False,
+        "attention_chunk_size": 8192,
+        "attention_dropout": 0.0,
+        "bos_token_id": 200000,
+        "eos_token_id": [200001, 200007, 200008],
+        "for_llm_compressor": False,
+        "head_dim": 128,
+        "hidden_act": "silu",
+        "hidden_size": 5120,
+        "initializer_range": 0.02,
+        "interleave_moe_layer_step": 2,
+        "intermediate_size": 8192,
+        "intermediate_size_ml
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## 7c8ba71b49 - [TRTLLM-8832][feat] fully async _select_generated_logits with tests (#8628)

- **Date**: 2025-10-27
- **Author**: mpikulski
- **Categories**: Parallelism/Async

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- Pinned memory
- Reduce synchronization overhead
- Speculative decoding

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/sampler.py          |  10 +-
 tensorrt_llm/_torch/pyexecutor/sampling_utils.py   |  50 ++++--
 tests/integration/test_lists/test-db/l0_a10.yml    |   2 +
 .../_torch/sampler/test_torch_multi_arange.py      | 143 +++++++++++++++++
 .../unittest/_torch/sampler/test_torch_sampler.py  | 177 ++++++++++++++++++++-
 tests/unittest/utils/test_util.py                  |  66 ++++++++
 tests/unittest/utils/util.py                       |  64 ++++++++
 7 files changed, 495 insertions(+), 17 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index 4b5888291..276aa9770 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -1366,6 +1366,7 @@ class TorchSampler(Sampler):
         req_num_generation_steps: torch.Tensor,
         num_context_logits_prefix_sum: list[int],
         generation_requests_total_steps: int,
+        num_logits_to_keep: int,
     ) -> torch.Tensor:
         # raw_logits should contain only the generated logits.
         # If return context logits is requested, select only the generated logits.
@@ -1394,9 +1395,10 @@ class TorchSampler(Sampler):
                 req_num_steps_fictitious_cuda = req_num_generation_steps_cuda[
                     : (len(scheduled_requests.context_requests) + 1)
                 ].clone()
-                req_num_steps_fictitious_cuda[-1] = generation_requests_total_steps
-                next_context_req_offsets_cuda[-1] = (
-                    next_context_req_offsets_cuda[-2] + req_num_steps_fictitious_cuda[-1]
+                req_num_steps_fictitious_cuda[-1].fill_(generation_requests_total_steps)
+                next_context_req_offsets_cuda[-1].copy_(
+                    next_context_req_offsets_cuda[-2] + req_num_steps_fictitious_cuda[-1],
+                    non_blocking=True,
                 )
             else:
                 req_num_steps_fictitious_cuda = req_num_generation_steps_cuda[
@@ -1412,6 +1414,7 @@ class TorchSampler(Sampler):
             indices_to_keep_cuda = torch_multi_arange(
                 starts=(next_context_req_offsets_cuda - req_num_steps_fictitious_cuda),
                 ends=next_context_req_offsets_cuda,
+                output_length=num_logits_to_keep,
             )
 
             raw_logits_cuda = raw_logits_cuda[indices_to_keep_cuda]
@@ -1455,6 +1458,7 @@ class TorchSampler(Sampler):
                 if scheduled_requests.generation_requests
                 else 0
             ),
+            num_logits_to_keep=sum_steps,
         )
 
         # Handle embedding bias
diff --git a/tensorrt_llm/_torch/pyexecutor/sampling_utils.py b/tensorrt_llm/_torch/pyexecutor/sampling_utils.py
index df893c902..0ea3494aa 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampling_utils.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampling_utils.py
@@ -343,11 +343,19 @@ class SimpleGroupedStrategySampler(GroupedStrategySampler[Strategy]):
         )
 
 
+class _AcceptSyncCompute:
+    pass
+
+
+ACCEPT_SYNC_COMPUTE = _AcceptSyncCompute()
+
+
 # Inspired by https://github.com/pytorch/pytorch/issues/80577; note also the
 # suggestion to consider torch.nested.
 def torch_multi_arange(
     ends: torch.Tensor,
     *,
+    output_length: int | _AcceptSyncCompute,
     starts: Optional[torch.Tensor] = None,
     steps: Optional[torch.Tensor] = None,
 ) -> torch.Tensor:
@@ -355,13 +363,23 @@ def torch_multi_arange(
 
     Starts, ends, steps need to share dtype and shape
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Pinned (page-locked) memory enables faster host-to-device data transfers. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 7ceb5e5ab6 - [TRTLLM-9198][perf] Add torch.compile + multi-stream support for k-cache scatter and weight scaling (#8988)

- **Date**: 2025-11-10
- **Author**: Chang Liu
- **Categories**: Parallelism/Async, Cache Optimization

### Optimization Techniques

- Torch compilation/JIT optimization
- Custom CUDA kernel
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Attention mechanism optimization

### Applicable Conditions

- Prefill phase

### Changed Files

```
.../_torch/attention_backend/sparse/dsa.py         | 24 +++++++++++++++-------
 tensorrt_llm/_torch/modules/attention.py           | 13 +-----------
 tensorrt_llm/_torch/utils.py                       | 23 +++++++++++++++++++++
 .../_torch/attention/sparse/test_dsa_indexer.py    |  2 ++
 4 files changed, 43 insertions(+), 19 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/attention_backend/sparse/dsa.py b/tensorrt_llm/_torch/attention_backend/sparse/dsa.py
index c30a0dc47..84a9d63b7 100644
--- a/tensorrt_llm/_torch/attention_backend/sparse/dsa.py
+++ b/tensorrt_llm/_torch/attention_backend/sparse/dsa.py
@@ -17,6 +17,7 @@ from tensorrt_llm._torch.modules.multi_stream_utils import \
     maybe_execute_in_parallel
 from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding
 from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
+from tensorrt_llm._torch.utils import maybe_compile
 from tensorrt_llm._utils import get_size_in_bytes
 from tensorrt_llm.bindings import DataType
 from tensorrt_llm.bindings.executor import KvCacheConfig
@@ -572,6 +573,12 @@ class DSAtrtllmAttentionMetadata(TrtllmAttentionMetadata):
         self.on_update_kv_lens()
 
 
+@maybe_compile(dynamic=True)
+def _scale(weights: torch.Tensor, q_scale: torch.Tensor,
+           s: float) -> torch.Tensor:
+    return weights * q_scale.squeeze(-1) * s
+
+
 class Indexer(nn.Module):
 
     def __init__(self,
@@ -964,9 +971,6 @@ class Indexer(nn.Module):
         if not use_custom_topk:
             topk_indices_buffer[:hidden_states.shape[0]] = -1
 
-        # Store k_fp8 and k_scale into indexer k cache
-        self._update_k_cache(k_fp8, k_scale, metadata)
-
         if has_prefill:
             # Use chunked prefill to reduce memory footprint
             if metadata.indexer_prefill_chunks is not None:
@@ -1121,9 +1125,7 @@ class Indexer(nn.Module):
                      q_scale: torch.Tensor) -> torch.Tensor:
         weights = indexer_weights if indexer_weights is not None else self.weights_proj(
             hidden_states)
-        weights = weights.unsqueeze(-1) * q_scale * self.weight_scale_factor
-        # output weights is guaranteed to be float32 due to type promotion from q_scale (float32)
-        weights = weights.squeeze(-1)
+        weights = _scale(weights, q_scale, self.weight_scale_factor)
         return weights
 
     @torch.inference_mode()
@@ -1192,7 +1194,15 @@ class Indexer(nn.Module):
         q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim)
         q_scale = q_scale.view(-1, self.n_heads, 1)
 
-        weights = self.weight_scale(hidden_states, indexer_weights, q_scale)
+        weights, _ = maybe_execute_in_parallel(
+            lambda: self.weight_scale(hidden_states, indexer_weights, q_scale),
+            lambda: self._update_k_cache(
+                k_fp8, k_scale, metadata),  # store k_fp8 and k_scale in k cache
+            self.ln_events[0],
+            self.ln_events[1],
+            self.aux_stream,
+        )
+
         # Return topk indices buffer for sparse attention [num_tokens, index_topk]
         return self.sparse_attn_indexer(metadata, hidden_states, q_fp8, k_fp8,
                                         k_scale, weights)
diff --git a/tensorrt_llm/_torch/modules/attention.py b/tensorrt_llm/_torch/modules/attention.py
index 834eaaee4..
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## 7d31532850 - [TRTLLM-10312][perf] Improve performance of _write_finish_reasons in TorchSampler (#10459)

- **Date**: 2026-01-29
- **Author**: Stefan Niebler
- **Categories**: General Performance

### Optimization Techniques

- Async/stream-based execution
- KV cache optimization
- Batching optimization
- Pinned memory
- Speculative decoding

### Applicable Conditions

- Decode/generation phase
- Disaggregated serving

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/sampler.py          | 129 ++++++++++++---------
 tensorrt_llm/_torch/speculative/mtp.py             |   9 +-
 .../unittest/_torch/sampler/test_torch_sampler.py  |  24 ++++
 3 files changed, 104 insertions(+), 58 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index 5dfb1382a..876858bfe 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -929,6 +929,12 @@ class TorchSampler(Sampler, AsyncWorkerMixin):
     class Store:
         new_tokens: torch.Tensor
         """Shape: See cpp DecoderState.getAllNewTokens()"""
+        max_lengths_tensor: torch.Tensor
+        """Shape: batch_size
+           Usage: Stores the maximum lengths for each request"""
+        end_ids: torch.Tensor
+        """Shape: batch_size
+           Usage: Stores the end ids for each request"""
         finish_reasons: torch.Tensor
         """Shape: max_tokens, batch_size, beam_width
            Usage: Stores the currently estimated finish_reasons for each request"""
@@ -974,6 +980,8 @@ class TorchSampler(Sampler, AsyncWorkerMixin):
         # Tensors necessary for all sampling methods
         new_tokens = int_tensor(self.NEW_TOKENS_SHAPE)
         finish_reasons = int_tensor(self.NEW_TOKENS_SHAPE)
+        max_lengths_tensor = int_tensor(self.max_num_sequences)
+        end_ids = int_tensor(self.max_num_sequences)
 
         # Only used for logprobs processing or beam search
         sampled_log_probs = torch.empty(self.LOGPROBS_SHAPE, device="cuda", dtype=torch.float32)
@@ -1007,6 +1015,8 @@ class TorchSampler(Sampler, AsyncWorkerMixin):
         return self.Store(
             new_tokens=new_tokens,
             finish_reasons=finish_reasons,
+            max_lengths_tensor=max_lengths_tensor,
+            end_ids=end_ids,
             cache_indirection=cache_indirection,
             cache_indirection_buffer=cache_indirection_buffer,
             cum_log_probs=cum_log_probs,
@@ -1072,6 +1082,9 @@ class TorchSampler(Sampler, AsyncWorkerMixin):
                     FinishReason.CANCELLED,
                 ]  # `in FinishReason` clashes with PyBind11: `TypeError: 'pybind11_type' object is not iterable`
             }
+            self._max_tokens_offset = torch.arange(
+                1, self.max_tokens + 1, device="cuda", dtype=torch.int32
+            ).view(-1, 1, 1)
 
         self._grouped_sampler_cls: Type[GroupedStrategySampler]
         if IS_FLASHINFER_AVAILABLE and not args.disable_flashinfer_sampling:
@@ -1525,14 +1538,47 @@ class TorchSampler(Sampler, AsyncWorkerMixin):
 
         return num_accepted_draft_tokens - 1
 
-    def setup_sampler_step(self, requests: ScheduledRequests):
+    def _is_new_request(self, request: LlmRequest) -> bool:
+        return (
+            not request.is_finished
+            and not request.py_is_draft
+            and (
+                (request.is_context_init_state and request.is_last_context_chunk)
+                or request.is_disagg_generation_transmission_complete
+            )
+        )
+
+    @override
+    def setup_sampler_step(self, scheduled_requests: ScheduledRequests):
         """Setup the sampler step for the requ
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. Pinned (page-locked) memory enables faster host-to-device data transfers. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 7deefb3d2b - [TRTLLM-7192][feat] optimize MLA chunked prefill && support fp8 mla chunked prefill (#7477)

- **Date**: 2025-09-15
- **Author**: jmydurant
- **Categories**: Quantization Optimization

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Pinned memory
- Reduce synchronization overhead
- Speculative decoding
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- NVIDIA Hopper (SM90) GPU
- Prefill phase
- Decode/generation phase

### Changed Files

```
cpp/kernels/fmha_v2/fmha_test.py                   |   3 +-
 cpp/kernels/fmha_v2/setup.py                       | 241 +++++++++---------
 cpp/kernels/fmha_v2/src/fmha/fragment.h            |  92 +++++++
 cpp/kernels/fmha_v2/src/fmha/warpspec/epilogue.h   |   5 +
 .../fmha_v2/src/fused_multihead_attention.cpp      |   5 -
 cpp/tensorrt_llm/common/attentionOp.cpp            |   4 +-
 cpp/tensorrt_llm/common/attentionOp.h              |   5 +-
 .../contextFusedMultiHeadAttention/fmhaRunner.cpp  |  10 +-
 cpp/tensorrt_llm/kernels/mlaChunkedPrefill.cu      |  86 +++----
 cpp/tensorrt_llm/kernels/mlaChunkedPrefill.cuh     |   4 +-
 cpp/tensorrt_llm/nanobind/thop/bindings.cpp        |   3 +-
 cpp/tensorrt_llm/pybind/thop/bindings.cpp          |   3 +-
 cpp/tensorrt_llm/thop/attentionOp.cpp              |  10 +-
 cpp/tensorrt_llm/thop/attentionOp.h                |   3 +-
 cpp/tensorrt_llm/thop/mlaPreprocessOp.cpp          |  42 +--
 .../unit_tests/kernels/mlaChunkedPrefillTest.cu    | 281 ++++++++++++++-------
 tensorrt_llm/_torch/attention_backend/interface.py |   2 +
 tensorrt_llm/_torch/attention_backend/trtllm.py    |  91 +++++--
 tensorrt_llm/_torch/modules/attention.py           |  10 +-
 .../_torch/pyexecutor/py_executor_creator.py       |   4 +-
 .../defs/accuracy/test_llm_api_pytorch.py          |  12 +-
 tests/integration/test_lists/test-db/l0_h100.yml   |   1 +
 22 files changed, 591 insertions(+), 326 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/kernels/fmha_v2/fmha_test.py b/cpp/kernels/fmha_v2/fmha_test.py
index 96b1f0e45..94bebf718 100644
--- a/cpp/kernels/fmha_v2/fmha_test.py
+++ b/cpp/kernels/fmha_v2/fmha_test.py
@@ -178,8 +178,7 @@ def test_trtllm_context_mla_attention_fmha(dtype, s):
         check=True)
 
     # For chunked prefill, we need to enable -save-softmax (dtype: bf16, layout: separate-q-k-v).
-    # Currently fp8 kernel doesn't support saving softmax.
-    if dtype == "-bf16":
+    if dtype in ["-bf16", "-e4m3"]:
         # padding mask
         subprocess.run(
             f"bin/fmha.exe -v 0 -runs 1 -min-s 1024 -s {s} -b 8 -h 8 -d 192 -dv 128 {dtype} "
diff --git a/cpp/kernels/fmha_v2/setup.py b/cpp/kernels/fmha_v2/setup.py
index 2d8a6b416..24a80b8d7 100644
--- a/cpp/kernels/fmha_v2/setup.py
+++ b/cpp/kernels/fmha_v2/setup.py
@@ -3815,124 +3815,126 @@ def enumerate_qgmma_flash_warpspec_kernels(specs,
     combinations = product([False, True], \
         [InputLayout.PACKED_QKV, InputLayout.CONTIGUOUS_Q_KV,
          InputLayout.Q_PAGED_KV, InputLayout.SEPARATE_Q_K_V],
-        [False, True])
-    for (alibi, input_layout, enable_attn_logit_softcapping) in combinations:
+        [False, True], [False, True])
+    for (alibi, input_layout, enable_attn_logit_softcapping,
+         return_softmax) in combinations:
         # alibi and bmm1_tanh_scale shouldn't be used together.
         if alibi and enable_attn_logit_softcapping:
             continue
-        # D <= 64: KV_STEP = 256
-        specs.append(
-            kernel_spec(
-                sm=sm,
-                sm_mma=90,
-                dtype=dtype,
-                seq_len=0,  # support any sequence length
-                head_size=[32, 40, 48, 64],
-                warps_m=4,  #4x1 warpgroups
-                warps_n=1,
-                version=2,
-                interleaved=False,
-                ldgsts_q=
-                False,  # for Hopper kernels, ldgsts = False signals TMA usage.
-                ldgsts_k=False,
-                ldgsts_v=False,
-                share_smem_k_v=False,
-                loop_step=64,
-                q_tile_buffers=1,  # only used by warp specialized kernels
-                has_noloop=0,
-                noloop_step=64,
-                kv_loop_step=256,
-                kv_tile_buffers=4,  # only used by warp specialized kernels
-                unroll_threshold=1,
-                has_scale_max=False,
-                flash_attention=True,
-                warp_specialization=True,
-                alibi=alibi,
-                enable_attn_logit_softcapping=enable_attn_logit_softcapping,
-                return_softmax_stats=
-                False,  # return softmax stats is not supported for fp8 now
-                scheduling_mode=scheduling_mode,
-                input_layout=input_layout,
-                sage_block_sizes=sage_block_sizes,
-                output_dtype=output_dtype))
-
-        # 64 < D <=128: KV_STEP = 128
-        specs.app
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Pinned (page-locked) memory enables faster host-to-device data transfers. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## 7e033c392e - Feat: Add vectorized loading for finalize kernel in MoE Trtllm backend (#5919)

- **Date**: 2025-07-17
- **Author**: ChristinaZ
- **Categories**: Kernel Optimization

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Async/stream-based execution
- FP8 quantization
- Reduce synchronization overhead
- MoE optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU

### Changed Files

```
.../trtllmGenKernels/blockScaleMoe/DevKernel.cu    | 110 ++++++++++++++++++++-
 1 file changed, 106 insertions(+), 4 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.cu b/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.cu
index ac0684ff1..ad5cd15fd 100644
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.cu
+++ b/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.cu
@@ -16,10 +16,21 @@
 
 #include "DevKernel.h"
 
+#include "cutlass/array.h"
+#include "cutlass/numeric_conversion.h"
+#include <cub/cub.cuh>
 #include <cutlass/cutlass.h>
 #include <cutlass/numeric_types.h>
 
-#include <cub/cub.cuh>
+////////////////////////////////////////////////////////////////////////////////////////////////////
+
+// Helper function for array conversion
+template <class T, class U>
+__host__ __device__ constexpr static U arrayConvert(T const& input)
+{
+    cutlass::NumericArrayConverter<typename U::Element, typename T::Element, U::kElements> converter;
+    return converter(input);
+}
 
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 
@@ -518,6 +529,83 @@ __global__ void finalizeKernel(KernelParams params)
     }
 }
 
+constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;
+
+__device__ float4 vectorizedLoadPtx(float4 const* ptr)
+{
+    float4 ret;
+    asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
+                 : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
+                 : "l"(ptr));
+    return ret;
+}
+
+// Final kernel to unpermute and scale
+// This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.
+
+template <typename KernelParams>
+__global__ void finalizeKernelVecLoad(KernelParams params)
+{
+    using Type = typename KernelParams::Type;
+    using TypeExpW = typename KernelParams::TypeExpW;
+
+    int const hiddenDimBits = params.hiddenDim * cutlass::sizeof_bits<Type>::value;
+    assert(hiddenDimBits % 128 == 0);
+
+    // Load 128-bits per thread, according to the smallest data type we read/write
+    constexpr int64_t FINALIZE_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<Type>::value;
+    using InputElem = cutlass::Array<Type, FINALIZE_ELEM_PER_THREAD>;
+    using OutputElem = cutlass::Array<Type, FINALIZE_ELEM_PER_THREAD>;
+    using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
+
+    int64_t const tokenIdx = blockIdx.x;
+    int64_t const startOffset = threadIdx.x;
+    int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
+    int64_t const numElemsInCol = params.hiddenDim / FINALIZE_ELEM_PER_THREAD;
+
+    auto const offset = tokenIdx * params.hiddenDim;
+    Type* outputPtr = params.outPtr + offset;
+    auto* outElemPtr = reinterpret_cast<OutputElem*>(outputPtr);
+    auto const* inElemPtr = reinterpret_cast<InputElem const*>(params.inPtr);
+
+#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
+    // wait on primary kernel when using PDL
+    if constexpr (KernelParams::UsePdl)
+    {
+        cudaGridDependencySynchronize();

```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 7eee9a9d28 - doc: Update doc for Deepseek min latency (#3717)

- **Date**: 2025-04-22
- **Author**: Zongfei Jing
- **Categories**: Throughput/Latency

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Parallelism optimization
- Batching optimization
- Speculative decoding
- MoE optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU
- Decode/generation phase

### Changed Files

```
.../blockScaleMoe/trtllmGenSrc/RoutingKernel.cu          |  7 +++++++
 .../Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md | 16 +++++++++-------
 tensorrt_llm/_torch/models/modeling_deepseekv3.py        |  7 ++++---
 3 files changed, 20 insertions(+), 10 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/trtllmGenSrc/RoutingKernel.cu b/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/trtllmGenSrc/RoutingKernel.cu
index 9409fa840..7199b19eb 100644
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/trtllmGenSrc/RoutingKernel.cu
+++ b/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/trtllmGenSrc/RoutingKernel.cu
@@ -698,6 +698,7 @@ __global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1) __launch_bounds__(Nu
 #else
 __global__ void routingIndicesClusterKernel(KernelParams params)
 {
+    assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
 }
 #endif
 ////////////////////////////////////////////////////////////////////////////////////////////////////
@@ -886,6 +887,8 @@ __global__ void __launch_bounds__(NumThreads) routingIndicesCoopKernel(KernelPar
             params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
         }
     }
+#else
+    assert(false && "routingIndicesCoopKernel is only supported on SM90+ architectures");
 #endif
 }
 
@@ -973,6 +976,8 @@ __global__ void __launch_bounds__(NumThreads) routingIndicesHistogramKernel(Kern
     // Reduce histograms with atomics.
     int32_t const localExpertCount = smemExpertCount[threadIdx.x];
     atomicAdd(&params.mPtrExpertCounts[threadIdx.x], localExpertCount);
+#else
+    assert(false && "routingIndicesHistogramKernel is only supported on SM90+ architectures");
 #endif
 }
 
@@ -1204,6 +1209,8 @@ __global__ void __launch_bounds__(NumThreads) routingIndicesOffsetsKernel(Kernel
     {
         cudaTriggerProgrammaticLaunchCompletion();
     }
+#else
+    assert(false && "routingIndicesOffsetsKernel is only supported on SM90+ architectures");
 #endif
 }
 
diff --git a/docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md b/docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md
index dbdbb9786..7073a6597 100644
--- a/docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md
+++ b/docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md
@@ -110,6 +110,7 @@ cat >./extra-llm-api-config.yml<<EOF
 pytorch_backend_config:
     enable_overlap_scheduler: true
     use_cuda_graph: true
+    moe_backend: TRTLLM
 speculative_config:
     decoding_type: MTP
     num_nextn_predict_layers: 3
@@ -125,7 +126,7 @@ trtllm-bench --model nvidia/DeepSeek-R1-FP4 \
     --concurrency 1 \
     --max_batch_size 1 \
     --tp 8 \
-    --ep 4 \
+    --ep 2 \
     --extra_llm_api_options ./extra-llm-api-config.yml
 ```
 
@@ -147,12 +148,13 @@ The perf can be different when using different datasets and different machines.
 ===========================================================
 = PERFORMANCE OVERVIEW
 ===========================================================
-Request Throughput (req/sec):                     0.1244
-Total Output Throughput (tokens/sec):             254.5535
-Per User Output Throughput (tokens/sec/user):     
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 8039ef45d3 - CI: Performance regression tests update (#3531)

- **Date**: 2025-06-01
- **Author**: amirkl94
- **Categories**: General Performance

### Optimization Techniques

- KV cache optimization
- Batching optimization
- Triton kernel

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU

### Changed Files

```
jenkins/L0_Test.groovy                             |  15 +-
 tests/integration/defs/perf/README.md              |  73 ++++++++--
 tests/integration/defs/perf/base_perf.csv          |   5 +
 tests/integration/defs/perf/base_perf_pytorch.csv  |   4 +
 .../defs/perf/create_perf_comparison_report.py     | 155 +++++++++++++++++++++
 tests/integration/defs/perf/diff_tools.py          |  86 ++++++++++++
 tests/integration/defs/perf/requirements.txt       |   3 +
 tests/integration/defs/perf/sanity_perf_check.py   | 146 ++++++-------------
 tests/integration/defs/perf/utils.py               |  18 ++-
 tests/integration/test_lists/test-db/l0_perf.yml   |  23 ++-
 10 files changed, 402 insertions(+), 126 deletions(-)
```

### Diff Preview

```diff
diff --git a/jenkins/L0_Test.groovy b/jenkins/L0_Test.groovy
index fab60f726..06e6136da 100644
--- a/jenkins/L0_Test.groovy
+++ b/jenkins/L0_Test.groovy
@@ -1230,11 +1230,21 @@ def runLLMTestlistOnPlatformImpl(pipeline, platform, testList, config=VANILLA_CO
         }
 
         if (perfMode) {
+            basePerfFilename = stageName.contains("PyTorch") ? "base_perf_pytorch.csv" : "base_perf.csv"
+            basePerfPath = "${llmSrc}/tests/integration/defs/perf/${basePerfFilename}"
             stage("Check perf result") {
                 sh """
                     python3 ${llmSrc}/tests/integration/defs/perf/sanity_perf_check.py \
                     ${stageName}/perf_script_test_results.csv \
-                    ${llmSrc}/tests/integration/defs/perf/base_perf.csv
+                    ${basePerfPath}
+                """
+            }
+            stage("Create perf report") {
+                sh """
+                    python3 ${llmSrc}/tests/integration/defs/perf/create_perf_comparison_report.py \
+                    --output_path ${stageName}/report.pdf \
+                    --files ${stageName}/perf_script_test_results.csv \
+                    ${basePerfPath}
                 """
             }
         }
@@ -1572,8 +1582,9 @@ def launchTestJobs(pipeline, testFilter, dockerNode=null)
         "H100_PCIe-TensorRT-[Post-Merge]-2": ["h100-cr", "l0_h100", 2, 2],
         "B200_PCIe-Triton-Python-[Post-Merge]-1": ["b100-ts2", "l0_b200", 1, 1],
         "DGX_H100-4_GPUs-TensorRT-[Post-Merge]-1": ["dgx-h100-x4", "l0_dgx_h100", 1, 1, 4],
-        "A100_80GB_PCIE-TensorRT-Perf-1": ["a100-80gb-pcie", "l0_perf", 1, 1],
+        // "A100_80GB_PCIE-TensorRT-Perf-1": ["a100-80gb-pcie", "l0_perf", 1, 1],
         "H100_PCIe-TensorRT-Perf-1": ["h100-cr", "l0_perf", 1, 1],
+        "H100_PCIe-PyTorch-Perf-1": ["h100-cr", "l0_perf", 1, 1],
         "DGX_H200-8_GPUs-PyTorch-[Post-Merge]-1": ["dgx-h200-x8", "l0_dgx_h200", 1, 1, 8],
         "DGX_H200-4_GPUs-PyTorch-[Post-Merge]-1": ["dgx-h200-x4", "l0_dgx_h200", 1, 2, 4],
         "DGX_H200-4_GPUs-PyTorch-[Post-Merge]-2": ["dgx-h200-x4", "l0_dgx_h200", 2, 2, 4],
diff --git a/tests/integration/defs/perf/README.md b/tests/integration/defs/perf/README.md
index 891aa1c08..569063f22 100644
--- a/tests/integration/defs/perf/README.md
+++ b/tests/integration/defs/perf/README.md
@@ -1,11 +1,66 @@
-Sanity Perf Check Introduction
+# Sanity Perf Check Introduction
 
-# Background
-The sanity perf check mechanism is the way of perf regression detection for L0 testing. We create the base_perf.csv which consists of the several models' perf baseline and use the sanity_perf_check.py to detect the perf regression.
-# Usage
-There're four typical scenarios for sanity perf check feature.
+## Background
+"Sanity perf check" is a mechanism to detect performance regressions in the L0 pipeline.
+The tests defined in `l0_perf.yml` are the ones that are required to pass for every PR before merge.
 
-1. The newly added MR 
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 80f261ea36 - [https://nvbugs/5622938][feat] Run sample_async on extra stream. (#10215)

- **Date**: 2026-01-09
- **Author**: Yuxian Qiu
- **Categories**: Parallelism/Async

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- Batching optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/py_executor.py | 24 ++++++++++++++++++++++--
 1 file changed, 22 insertions(+), 2 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/py_executor.py b/tensorrt_llm/_torch/pyexecutor/py_executor.py
index 8f54aca6c..e8ff00754 100644
--- a/tensorrt_llm/_torch/pyexecutor/py_executor.py
+++ b/tensorrt_llm/_torch/pyexecutor/py_executor.py
@@ -250,6 +250,9 @@ class PyExecutor:
         self.send_schedule_handler = None
         self.pp_scheduler_max_retry_count = int(
             os.environ.get("TLLM_PP_SCHEDULER_MAX_RETRY_COUNT", 10))
+        self.sample_stream = torch.cuda.Stream()
+        self.start_sample_event = torch.cuda.Event()
+        self.finish_sample_event = torch.cuda.Event()
 
         # Set of request IDs that are currently in flight across all micro batches.
         # The scheduler will avoid scheduling requests that are already in flight.
@@ -1068,8 +1071,25 @@ class PyExecutor:
                                 guided_decoder_failed_requests = self.guided_decoder.execute(
                                     batch_outputs['logits'])
 
-                            sample_state = self._sample_async(
-                                scheduled_batch, batch_outputs)
+                            if os.environ.get("TRTLLM_PP_MULTI_STREAM_SAMPLE",
+                                              "1") == "1":
+                                # Wait for the previous sample to finish.
+                                self.finish_sample_event.wait()
+                                # Copy the batch outputs as sampler inputs
+                                # to avoid next forward step overwriting them.
+                                batch_outputs_copy = {
+                                    name: tensor.clone()
+                                    for name, tensor in batch_outputs.items()
+                                }
+                                self.start_sample_event.record()
+                                with torch.cuda.stream(self.sample_stream):
+                                    self.start_sample_event.wait()
+                                    sample_state = self._sample_async(
+                                        scheduled_batch, batch_outputs_copy)
+                                    self.finish_sample_event.record()
+                            else:
+                                sample_state = self._sample_async(
+                                    scheduled_batch, batch_outputs)
                             assert sample_state is not None, "Sampling failed"
 
                             # Handle guided decoder errors after _sample_async to avoid state conflicts.

```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 80f9989a1e - [enhanchment] Add beam width to low latency. (#4812)

- **Date**: 2025-06-03
- **Author**: Frank
- **Categories**: Throughput/Latency

### Optimization Techniques

- KV cache optimization
- Batching optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/bench/benchmark/low_latency.py | 12 ++++++++++--
 1 file changed, 10 insertions(+), 2 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/bench/benchmark/low_latency.py b/tensorrt_llm/bench/benchmark/low_latency.py
index 9663da3c1..96868234d 100644
--- a/tensorrt_llm/bench/benchmark/low_latency.py
+++ b/tensorrt_llm/bench/benchmark/low_latency.py
@@ -75,6 +75,12 @@ from tensorrt_llm.sampling_params import SamplingParams
 @optgroup.group("Request Load Control Options",
                 cls=MutuallyExclusiveOptionGroup,
                 help="Limits how requests are loaded.")
+@optgroup.option(
+    "--beam_width",
+    type=int,
+    default=1,
+    help="Number of search beams.",
+)
 @optgroup.option(
     "--concurrency",
     type=int,
@@ -133,6 +139,7 @@ def latency_command(
     checkpoint_path: Path = bench_env.checkpoint_path or bench_env.model
     engine_dir: Path = params.pop("engine_dir")
     concurrency: int = params.pop("concurrency")
+    beam_width: int = params.pop("beam_width")
     warmup: int = params.get("warmup")
     # Engine configuration parsing
     exec_settings, build_cfg = get_settings_from_engine(engine_dir)
@@ -153,7 +160,7 @@ def latency_command(
     exec_settings["settings_config"]["kv_cache_percent"] = kv_cache_percent
     exec_settings["settings_config"]["max_batch_size"] = 1
     exec_settings["settings_config"]["max_num_tokens"] = engine_tokens
-    exec_settings["settings_config"]["beam_width"] = 1
+    exec_settings["settings_config"]["beam_width"] = beam_width
     exec_settings["settings_config"]["chunking"] = False
     exec_settings["settings_config"][
         "scheduler_policy"] = CapacitySchedulerPolicy.GUARANTEED_NO_EVICT
@@ -208,7 +215,8 @@ def latency_command(
         sampling_params = SamplingParams(
             end_id=eos_id,
             pad_id=pad_id,
-            n=1,
+            n=beam_width,
+            use_beam_search=beam_width > 1,
         )
         llm = LLM(**kwargs)
 

```

### Analysis Summary

This commit applies performance optimization techniques to improve inference efficiency. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 812d2ce938 - [#11726][feat] AutoDeploy: Fuse gemms of mixed children (#11793)

- **Date**: 2026-03-02
- **Author**: Taylor Yeonbok Lee
- **Categories**: Fusion

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Batching optimization
- MoE optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../model_registry/configs/qwen3.5_moe_400b.yaml   |   2 +
 .../_torch/auto_deploy/config/default.yaml         |   3 +
 .../_torch/auto_deploy/transform/library/fusion.py | 363 +++++++++++++++++----
 .../transformations/library/test_gemm_fusion.py    | 295 +++++++++++++++++
 4 files changed, 603 insertions(+), 60 deletions(-)
```

### Diff Preview

```diff
diff --git a/examples/auto_deploy/model_registry/configs/qwen3.5_moe_400b.yaml b/examples/auto_deploy/model_registry/configs/qwen3.5_moe_400b.yaml
index 12175d03d..250ee830e 100644
--- a/examples/auto_deploy/model_registry/configs/qwen3.5_moe_400b.yaml
+++ b/examples/auto_deploy/model_registry/configs/qwen3.5_moe_400b.yaml
@@ -16,6 +16,8 @@ model_kwargs:
 transforms:
   export_to_gm:
     num_moe_experts_for_export: 2
+  fuse_gemms_mixed_children:
+    enabled: true
   detect_sharding:
     sharding_dims: ['tp','ep', 'bmm']
     # use only manual config for TP sharding
diff --git a/tensorrt_llm/_torch/auto_deploy/config/default.yaml b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
index cb06fc569..1e814dd64 100644
--- a/tensorrt_llm/_torch/auto_deploy/config/default.yaml
+++ b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
@@ -127,6 +127,9 @@ transforms:
   ############################################################################################
   # RUN POST-LOAD FUSION AND OPTIMIZATIONS
   ############################################################################################
+  fuse_gemms_mixed_children:
+    stage: post_load_fusion
+    enabled: false
   fuse_gemms:
     stage: post_load_fusion
     enabled: false # TODO: https://github.com/NVIDIA/TensorRT-LLM/issues/4674 this is causing OOMs
diff --git a/tensorrt_llm/_torch/auto_deploy/transform/library/fusion.py b/tensorrt_llm/_torch/auto_deploy/transform/library/fusion.py
index c31bbe788..63822eb06 100644
--- a/tensorrt_llm/_torch/auto_deploy/transform/library/fusion.py
+++ b/tensorrt_llm/_torch/auto_deploy/transform/library/fusion.py
@@ -1,7 +1,7 @@
 import operator
 from abc import ABC, abstractmethod
 from collections import defaultdict
-from functools import partial
+from functools import lru_cache, partial
 from itertools import chain
 from typing import Callable, Dict, List, Tuple
 
@@ -14,70 +14,102 @@ from ...shim.interface import CachedSequenceInterface
 from ...utils._graph import delete_all_unused_submodules, eliminate_dead_code
 from ...utils.cuda_mem_tracker import cuda_memory_tracker
 from ...utils.logger import ad_logger
-from ...utils.node_utils import extract_weight_name, is_linear_op, is_op
+from ...utils.node_utils import (
+    WeightBiasInfoCache,
+    extract_weight_name,
+    is_fake_quantized_linear_op,
+    is_linear_op,
+    is_op,
+)
 from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry
 
 
-def _insert_fused_gemm(gm: GraphModule, idx: int, parent_node: Node, linear_nodes: List[Node]):
-    """Fuse GEMMs that have the same input activation.
+def _insert_fused_gemm(
+    gm: GraphModule,
+    idx: int,
+    parent_node: Node,
+    linear_nodes: List[Node],
+    allow_not_contigous: bool = True,
+) -> bool:
+    """Fuse GEMMs sharing the same input activation.
 
-    Below, is a simple example of how the fusion works:
+    Args:
+        allow_not_contigous: If True, split output via torch.narrow (zero-copy view).

```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 81c0764012 - Cherry pick "[NVBUG:5355009] Modify check for fuse_fp4_quant on SM120 (#5724)

- **Date**: 2025-07-04
- **Author**: Faraz
- **Categories**: Fusion, Quantization Optimization

### Optimization Techniques

- Operator fusion
- FP8 quantization
- Integer quantization
- Parallelism optimization
- Speculative decoding
- Attention mechanism optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- Disaggregated serving

### Changed Files

```
cpp/tensorrt_llm/common/attentionOp.cpp | 3 ++-
 tests/integration/test_lists/waives.txt | 4 ----
 2 files changed, 2 insertions(+), 5 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/common/attentionOp.cpp b/cpp/tensorrt_llm/common/attentionOp.cpp
index aba735f82..49a51b798 100644
--- a/cpp/tensorrt_llm/common/attentionOp.cpp
+++ b/cpp/tensorrt_llm/common/attentionOp.cpp
@@ -2423,7 +2423,8 @@ int AttentionOp::initialize() noexcept
 
     // Check requirements for FP4 output.
     TLLM_CHECK_WITH_INFO(!mFuseFp4Quant || mEnableContextFMHA, "Context FMHA must enable if fuse_fp4_quant is enabled");
-    TLLM_CHECK_WITH_INFO(!mFuseFp4Quant || mSM == 100, "fuse_fp4_quant only supports SM100 devices.");
+    TLLM_CHECK_WITH_INFO(
+        !mFuseFp4Quant || mSM == 100 || mSM == 120, "fuse_fp4_quant only supports SM100 or SM120 devices.");
 
     TLLM_CHECK(isRoPE() == (mRotaryEmbeddingDim != 0));
     TLLM_CHECK_WITH_INFO((mSM >= 80) || (mType != nvinfer1::DataType::kBF16),
diff --git a/tests/integration/test_lists/waives.txt b/tests/integration/test_lists/waives.txt
index 6edbd0b68..060451f7f 100644
--- a/tests/integration/test_lists/waives.txt
+++ b/tests/integration/test_lists/waives.txt
@@ -380,14 +380,10 @@ perf/test_perf.py::test_perf[roberta_base-bench-float16-maxbs:32-input_len:128+5
 test_e2e.py::test_openai_multi_chat_example SKIP (https://nvbugs/5236980)
 accuracy/test_llm_api_pytorch.py::TestLlama3_1_8BInstruct::test_bfloat16_4gpus[tp2pp2-attn_backend=TRTLLM-torch_compile=False] SKIP (https://nvbugs/5318143)
 accuracy/test_llm_api_pytorch.py::TestLlama3_1_8BInstruct::test_bfloat16_4gpus[tp2pp2-attn_backend=TRTLLM-torch_compile=True] SKIP (https://nvbugs/5318143)
-test_e2e.py::test_ptp_quickstart_advanced[Llama3.1-70B-NVFP4-nvfp4-quantized/Meta-Llama-3.1-70B] SKIP (https://nvbugs/5323316)
 disaggregated/test_disaggregated.py::test_disaggregated_single_gpu_with_mpirun[TinyLlama-1.1B-Chat-v1.0] SKIP (https://nvbugs/5328160)
 stress_test/stress_test.py::test_run_stress_test[llama-v3-8b-instruct-hf_tp1-stress_time_300s_timeout_450s-MAX_UTILIZATION-pytorch-stress-test] SKIP (https://nvbugs/5328495)
 accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_fp8_block_scales_4gpus[tp4-mtp_nextn=2-fp8kv=True-attention_dp=False-cuda_graph=False-overlap_scheduler=False-torch_compile=False] SKIP (https://nvbugs/5333654)
-test_e2e.py::test_ptp_quickstart_advanced[Llama3.1-8B-NVFP4-nvfp4-quantized/Meta-Llama-3.1-8B] SKIP (https://nvbugs/5333659)
 accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_fp8_block_scales_4gpus[ep4-mtp_nextn=0-fp8kv=True-attention_dp=True-cuda_graph=True-overlap_scheduler=True-torch_compile=False] SKIP (https://nvbugs/5351130)
-test_e2e.py::test_ptp_quickstart_advanced[Mixtral-8x7B-NVFP4-nvfp4-quantized/Mixtral-8x7B-Instruct-v0.1] SKIP (https://nvbugs/5333659)
-test_e2e.py::test_ptp_quickstart_advanced[Nemotron-Super-49B-v1-NVFP4-nvfp4-quantized/Llama-3_3-Nemotron-Super-49B-v1_nvfp4_hf] SKIP (https://nvbugs/5333659)
 accuracy/test_disaggregated_serving.py::TestDeepSeekV3Lite::test_auto_dtype[mtp_nextn=0-overlap_scheduler=True] SKIP (https://nvbugs/5322354)
 accuracy/test_d
```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 81f878c279 - [https://nvbugs/5707392][fix] unwaive test_fused_moe_fp8_blockwise_wide_ep[NotEnabled] (#10428)

- **Date**: 2026-01-08
- **Author**: xxi
- **Categories**: Fusion, Quantization Optimization

### Optimization Techniques

- Operator fusion
- FP8 quantization
- Integer quantization
- MoE optimization

### Applicable Conditions

- Disaggregated serving

### Changed Files

```
tests/integration/test_lists/waives.txt | 1 -
 1 file changed, 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tests/integration/test_lists/waives.txt b/tests/integration/test_lists/waives.txt
index a71d0475c..f211cdb1f 100644
--- a/tests/integration/test_lists/waives.txt
+++ b/tests/integration/test_lists/waives.txt
@@ -370,7 +370,6 @@ accuracy/test_llm_api_pytorch_multimodal.py::TestQwen2_5_VL_7B::test_auto_dtype
 accuracy/test_llm_api_pytorch_multimodal.py::TestLlava_V1_6_Mistral_7B::test_auto_dtype SKIP (https://nvbugs/5707087)
 accuracy/test_llm_api_pytorch_multimodal.py::TestPhi4MMFusedVisionLora::test_auto_dtype SKIP (https://nvbugs/5707087)
 disaggregated/test_disaggregated.py::test_disaggregated_ctxtp2pp2_gentp2pp2[TinyLlama-1.1B-Chat-v1.0] SKIP (https://nvbugs/5705199)
-unittest/_torch/modules/test_fused_moe.py::test_fused_moe_fp8_blockwise_wide_ep[NotEnabled] SKIP (https://nvbugs/5707392)
 accuracy/test_llm_api_pytorch.py::TestLlama3_3NemotronSuper49Bv1::test_auto_dtype_tp2 SKIP (https://nvbugs/5707145)
 accuracy/test_llm_api_pytorch.py::TestLlama3_3NemotronSuper49Bv1::test_fp8_prequantized_tp2 SKIP (https://nvbugs/5707145)
 accuracy/test_llm_api_pytorch.py::TestNemotronH_56B_Base::test_auto_dtype[tp8-cuda_graph=True] SKIP (https://nvbugs/5640697)

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 822cb0115b - [TRTLLM-6286] [perf] Add NoSmem epilogue schedule and dynamic cluster shape for sm10x group gemm (#7757)

- **Date**: 2025-09-21
- **Author**: xiweny
- **Categories**: Memory Optimization

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
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

- NVIDIA Blackwell (SM100) GPU
- NVIDIA Hopper (SM90) GPU

### Changed Files

```
.../mixtureOfExpertsBackendBenchmarkFixture.h      |  38 +++---
 .../gemm/kernel/fused_moe_kernel.cuh               |   4 +-
 .../gemm/kernel/fused_moe_kernel_routine.cuh       |   4 +-
 .../gemm/kernel/fused_moe_kernel_traits.cuh        |   4 +-
 .../kernels/cutlass_kernels/CMakeLists.txt         |   4 +-
 .../kernels/cutlass_kernels/cutlass_heuristic.cpp  |  36 +++---
 .../launchers/fused_moe_gemm_launcher_sm80.inl     |  16 +--
 .../launchers/moe_gemm_tma_ws_launcher.inl         |  35 +++---
 .../moe_gemm/moe_gemm_template_dispatch.h          |  88 +++++++------
 .../moe_gemm/moe_gemm_template_dispatch_tma_ws.h   |  21 ++--
 .../cutlass_kernels/moe_gemm/moe_kernels.cu        |  20 +++
 .../unit_tests/kernels/mixtureOfExpertsTest.cu     | 139 ++++++++++++++++-----
 12 files changed, 256 insertions(+), 153 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h b/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h
index a6bdada5a..b54aec107 100644
--- a/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h
+++ b/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h
@@ -549,6 +549,9 @@ public:
     ActivationType mActType = ActivationType::Relu;
 
     constexpr static int64_t NUM_BUFFERS = 32;
+    int64_t mNumWorkspaceBuffers = NUM_BUFFERS;
+    int64_t mNumInputBuffers = NUM_BUFFERS;
+    int64_t mNumGemmProfilerBuffers = NUM_BUFFERS;
 
     std::array<QuantParams, NUM_BUFFERS> mQuantParams{};
     bool mUseLora = false;
@@ -619,12 +622,12 @@ public:
 
         if (gemm_to_profile == GemmToProfile::LAYER)
         {
-
             mWorkspaceSize = mMoERunner.getWorkspaceSize(mTotalTokens, mHiddenSize, mInterSize, mNumExperts, mK,
                 mActType, parallelism_config, mUseLora, /*use_deepseek_fp8_block_scale=*/false,
                 /*min_latency_mode=*/false, mUsePrequantScale);
 
-            mWorkspace = allocBuffer<char>(mWorkspaceSize * NUM_BUFFERS);
+            mNumWorkspaceBuffers = mWorkspaceSize > 1024 * 1024 * 1024 ? 2 : NUM_BUFFERS;
+            mWorkspace = allocBuffer<char>(mWorkspaceSize * mNumWorkspaceBuffers);
 
             mExpertBias1 = nullptr;
             mExpertBias2 = nullptr;
@@ -690,9 +693,10 @@ public:
             mScaleProbsSize = padSize(mTotalTokens * mK);
             mScaleProbs = allocBuffer<float>(mScaleProbsSize * NUM_BUFFERS);
             mInputTensorSize = padSize(mTotalTokens * mHiddenSize);
-            mInputTensor = allocBuffer<DataType>(mInputTensorSize * NUM_BUFFERS);
+            mNumInputBuffers = mInputTensorSize > 1024 * 1024 * 1024 ? 2 : NUM_BUFFERS;
+            mInputTensor = allocBuffer<DataType>(mInputTensorSize * mNumInputBuffers);
             mFinalOutputSize = padSize(mTotalTokens * mHiddenSize);
-            mFinalOutput = allocBuffer<OutputType>(mFinalOutputSize * NUM_BUFFERS);
+            mFinalOutput = allocBuffer<OutputType>(mFinalOutputSize * mNumInputBuffers);
 
             mSourceToExpandedMapSize = padSize(mTotalTokens * mK);
             mSourceToExpandedMap = allocBuffer<int>(mSourceToExpandedMapSize * NUM_BUFFERS);
@@ -732,10 +736,11 @@ public:
                 = std::max(mGemmProfilerWorkspaceSize, mGemmProfilerBackend.getWorkspaceSize(mTotalTokens));
         }
 
-        int64_t num_gemm_buffers = gemm_to_profile == GemmToProfile::LAYER ? 1 : NUM_BUFFERS;
         mGemmProfilerWorkspaceSize = padSize(mGemmProfilerWorkspaceSize);
+        mNumGemmProfilerBuffers = mGemmProfilerWorkspaceSize > 1024 * 1024 * 1024 ? 2 : NUM_BUFFERS;
+        mNumGemmProfilerBuffers = gemm_to_profile == GemmToProfile::LAYER ? 1 : mNumGemmProfilerBuffers;
         mGemmProfilerWorkspace = mGemmProfilerWorkspaceSize > 0
-            ? allocBuffer<char>(mGemmProfilerWorkspaceSize * num_gemm_buffers)
+            ? allocBuffer<char>(mGemmPr
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 8282d6c1a7 - [fix] Fix llama4 min latency (#5117)

- **Date**: 2025-06-11
- **Author**: liji-nv
- **Categories**: Throughput/Latency

### Optimization Techniques

- Operator fusion
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/models/modeling_llama.py             | 2 +-
 tensorrt_llm/_torch/models/modeling_llama_min_latency.py | 2 +-
 2 files changed, 2 insertions(+), 2 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_llama.py b/tensorrt_llm/_torch/models/modeling_llama.py
index 247785a6f..600808c6b 100644
--- a/tensorrt_llm/_torch/models/modeling_llama.py
+++ b/tensorrt_llm/_torch/models/modeling_llama.py
@@ -301,7 +301,7 @@ class Llama4MoE(nn.Module):
         routed_output = self.experts(
             hidden_states,
             router_logits,
-            cutlass_min_latency_mode=cutlass_min_latency_mode,
+            do_finalize=not cutlass_min_latency_mode,
             all_rank_num_tokens=all_rank_num_tokens,
             use_dp_padding=use_dp_padding,
         )
diff --git a/tensorrt_llm/_torch/models/modeling_llama_min_latency.py b/tensorrt_llm/_torch/models/modeling_llama_min_latency.py
index 4c26e8c88..88a78cfb1 100644
--- a/tensorrt_llm/_torch/models/modeling_llama_min_latency.py
+++ b/tensorrt_llm/_torch/models/modeling_llama_min_latency.py
@@ -515,7 +515,7 @@ class Llama4MinLatencyFusedMoE(CutlassFusedMoE):
 
         return super().forward(x,
                                router_logits,
-                               cutlass_min_latency_mode=False,
+                               do_finalize=True,
                                output_dtype=output_dtype)
 
 

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 83b36ebecd - Fix fused_moe fallback issue. (#3652)

- **Date**: 2025-04-17
- **Author**: Yukun He
- **Categories**: Fusion

### Optimization Techniques

- Operator fusion
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/custom_ops/torch_custom_ops.py | 4 +++-
 1 file changed, 3 insertions(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py b/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
index 421d32bfb..dda4bcc32 100644
--- a/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
+++ b/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
@@ -126,7 +126,9 @@ def fused_moe(
         (2, 0, ((0, ), lambda x: x)),
     ))
 
-    min_latency_tensor = torch.empty(1) if min_latency_mode else torch.empty(0)
+    # TODO: set min_latency_mode always to False due to the error in the moe_kernels
+    min_latency_tensor = torch.empty(0)
+
     # allocate workspace for profiling
     moe_runner = MoERunner(
         x_dtype=input.dtype,

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

