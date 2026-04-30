# Performance Optimization Analysis - Part 6

Commits 146 to 174 of 283

---

## 841608f35e - [None][perf] Use F.rms_norm for per-head QK normalization in visual gen (#11798)

- **Date**: 2026-03-01
- **Author**: Kanghwan
- **Categories**: General Performance

### Optimization Techniques

- Torch compilation/JIT optimization
- Operator fusion
- Async/stream-based execution
- Parallelism optimization
- Batching optimization
- PyTorch built-in optimized ops
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../_torch/visual_gen/models/flux/attention.py     | 32 +++++++++++++++++++---
 .../_torch/visual_gen/modules/attention.py         | 22 ++++++---------
 2 files changed, 36 insertions(+), 18 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/visual_gen/models/flux/attention.py b/tensorrt_llm/_torch/visual_gen/models/flux/attention.py
index 238ba9a88..5c2c6a5b3 100644
--- a/tensorrt_llm/_torch/visual_gen/models/flux/attention.py
+++ b/tensorrt_llm/_torch/visual_gen/models/flux/attention.py
@@ -11,6 +11,7 @@ Key Components:
 from typing import TYPE_CHECKING, Optional, Tuple, Union
 
 import torch
+import torch.nn.functional as F
 
 from tensorrt_llm._torch.modules.linear import Linear, WeightMode, WeightsLoadingConfig
 from tensorrt_llm._torch.modules.rms_norm import RMSNorm
@@ -107,6 +108,30 @@ class FluxJointAttention(Attention):
                 force_dynamic_quantization=self.force_dynamic_quantization,
             )
 
+    def apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
+        """Override: use F.rms_norm for per-head norm
+        - ~1.6× speedup on Flux.2 inputs
+        - Better fusion with torch.compile
+        """
+        q = F.rms_norm(q, (q.shape[-1],), self.norm_q.weight, self.norm_q.variance_epsilon)
+        k = F.rms_norm(k, (k.shape[-1],), self.norm_k.weight, self.norm_k.variance_epsilon)
+        return q, k
+
+    def apply_qk_added_norm(
+        self, enc_q: torch.Tensor, enc_k: torch.Tensor
+    ) -> Tuple[torch.Tensor, torch.Tensor]:
+        """Override: Apply F.rms_norm to text-stream QK,
+        - ~1.6× speedup on Flux.2 inputs
+        - Better fusion with torch.compile
+        """
+        enc_q = F.rms_norm(
+            enc_q, (enc_q.shape[-1],), self.norm_added_q.weight, self.norm_added_q.variance_epsilon
+        )
+        enc_k = F.rms_norm(
+            enc_k, (enc_k.shape[-1],), self.norm_added_k.weight, self.norm_added_k.variance_epsilon
+        )
+        return enc_q, enc_k
+
     def forward(
         self,
         hidden_states: torch.Tensor,
@@ -134,7 +159,7 @@ class FluxJointAttention(Attention):
         key = key.view(batch_size, -1, self.num_attention_heads, self.head_dim)
         value = value.view(batch_size, -1, self.num_attention_heads, self.head_dim)
 
-        # Per-head QK normalization via base (per_head mode operates on 4D)
+        # Per-head QK normalization (F.rms_norm, fusible by torch.compile)
         query, key = self.apply_qk_norm(query, key)
 
         # Text QKV for joint attention (dual-stream blocks)
@@ -147,8 +172,7 @@ class FluxJointAttention(Attention):
             enc_k = enc_k.view(batch_size, -1, self.num_attention_heads, self.head_dim)
             enc_v = enc_v.view(batch_size, -1, self.num_attention_heads, self.head_dim)
 
-            enc_q = self.norm_added_q(enc_q.reshape(-1, enc_q.shape[-1])).view(enc_q.shape)
-            enc_k = self.norm_added_k(enc_k.reshape(-1, enc_k.shape[-1])).view(enc_k.shape)
+            enc_q, enc_k = self.apply_qk_added_norm(enc_q, enc_k)
 
             # Concatenate text + image for joint attention
             query = torch.cat([enc_q, query], dim=1)
@@ -284,7 +308,7 @@ class Flux2Parall
```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 88076eecd0 - [fix] Fix can_use_alltoall in fused_moe_wide_ep.py (#6173)

- **Date**: 2025-07-21
- **Author**: Jinyang Yuan
- **Categories**: Fusion

### Optimization Techniques

- Operator fusion
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/modules/fused_moe/fused_moe_wide_ep.py | 9 ++++-----
 1 file changed, 4 insertions(+), 5 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_wide_ep.py b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_wide_ep.py
index 36de5ddc1..81778c285 100755
--- a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_wide_ep.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_wide_ep.py
@@ -283,16 +283,14 @@ class WideEPMoE(MoE):
         return (num_rows + self.moe_max_num_tokens -
                 1) // self.moe_max_num_tokens
 
-    def can_use_alltoall(self, input, all_rank_num_tokens):
+    def can_use_alltoall(self, all_rank_num_tokens, all_rank_max_num_tokens):
         # Disable alltoall when chunking is used
         if self.calculate_num_chunks(all_rank_num_tokens) > 1:
             return False
 
-        num_tokens = input.shape[0]
-
         # For DeepEPLowLatency, check if tokens exceed the threshold
         if (self.alltoall_method_type == AlltoallMethodType.DeepEPLowLatency
-                and num_tokens > self.deep_ep_max_num_tokens):
+                and all_rank_max_num_tokens > self.deep_ep_max_num_tokens):
             return False
 
         return self.enable_alltoall
@@ -726,7 +724,8 @@ class WideEPMoE(MoE):
 
         # in case of num_rows is larger than max_chunk_size, we need to split the input into multiple chunks
         num_chunks = self.calculate_num_chunks(all_rank_num_tokens)
-        use_all_to_all = self.can_use_alltoall(x, all_rank_num_tokens)
+        use_all_to_all = self.can_use_alltoall(all_rank_num_tokens,
+                                               all_rank_max_num_tokens)
 
         if use_dp_padding:
             all_rank_num_tokens_padded = [all_rank_max_num_tokens

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 89dabf5aa1 - [TRTLLM-9736][feat] AsyncLLM and verl integ (#9353)

- **Date**: 2025-12-11
- **Author**: Erin
- **Categories**: Parallelism/Async

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- KV cache optimization
- Parallelism optimization
- Reduce synchronization overhead

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU
- Decode/generation phase
- Disaggregated serving

### Changed Files

```
tensorrt_llm/__init__.py                           |   3 +-
 tensorrt_llm/_torch/async_llm.py                   | 106 ++++++++++++++
 tensorrt_llm/_torch/pyexecutor/sampler.py          |  45 +++---
 tensorrt_llm/_torch/virtual_memory.py              |   3 +-
 tensorrt_llm/executor/ray_executor.py              | 158 +++++++++++++++++----
 tensorrt_llm/executor/ray_gpu_worker.py            |   4 +-
 tensorrt_llm/llmapi/__init__.py                    |   2 +
 tensorrt_llm/llmapi/llm.py                         |   3 +-
 tensorrt_llm/llmapi/llm_args.py                    |  81 +++++++++++
 tensorrt_llm/llmapi/rlhf_utils.py                  |  16 ++-
 tensorrt_llm/serve/openai_protocol.py              |  10 ++
 tensorrt_llm/serve/openai_server.py                |  41 +++++-
 .../integration/test_lists/test-db/l0_dgx_h100.yml |  17 +++
 tests/integration/test_lists/test-db/l0_h100.yml   |   1 +
 .../ray_orchestrator/multi_gpu/test_executor.py    |  68 +++++++--
 tests/unittest/api_stability/references/llm.yaml   |   4 +
 tests/unittest/llmapi/test_async_llm.py            | 137 ++++++++++++++++++
 17 files changed, 629 insertions(+), 70 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/__init__.py b/tensorrt_llm/__init__.py
index 978cf0796..cea56431b 100644
--- a/tensorrt_llm/__init__.py
+++ b/tensorrt_llm/__init__.py
@@ -84,7 +84,7 @@ from ._utils import (default_gpus_per_node, local_mpi_rank, local_mpi_size,
 from .builder import BuildConfig, Builder, BuilderConfig, build
 from .disaggregated_params import DisaggregatedParams
 from .functional import Tensor, constant
-from .llmapi import LLM, MultimodalEncoder
+from .llmapi import LLM, AsyncLLM, MultimodalEncoder
 from .llmapi.llm_args import LlmArgs, TorchLlmArgs, TrtLlmArgs
 from .logger import logger
 from .mapping import Mapping
@@ -136,6 +136,7 @@ __all__ = [
     'quantization',
     'tools',
     'LLM',
+    'AsyncLLM',
     'MultimodalEncoder',
     'LlmArgs',
     'TorchLlmArgs',
diff --git a/tensorrt_llm/_torch/async_llm.py b/tensorrt_llm/_torch/async_llm.py
new file mode 100644
index 000000000..76c33220d
--- /dev/null
+++ b/tensorrt_llm/_torch/async_llm.py
@@ -0,0 +1,106 @@
+from typing import Any, List, Optional
+
+from ..llmapi.llm import LLM
+from ..llmapi.llm_args import RayPlacementConfig
+
+
+class AsyncLLM(LLM):
+    """AsyncLLM is a subclass of LLM that supports asynchronous setup, release and
+    resume operations that are necessary for RL or agentic scenarios.
+
+    Currently, RL APIs are only supported with Ray orchestrator.
+    """
+
+    def __init__(
+        self,
+        placement_groups: Optional[List[Any]] = None,
+        placement_bundle_indices: Optional[List[List[int]]] = None,
+        per_worker_gpu_share: Optional[float] = None,
+        *args,
+        **kwargs,
+    ):
+        kwargs["orchestrator_type"] = "ray"
+        kwargs["ray_placement_config"] = RayPlacementConfig(
+            defer_workers_init=True,
+            placement_groups=placement_groups,
+            placement_bundle_indices=placement_bundle_indices,
+            per_worker_gpu_share=per_worker_gpu_share,
+        )
+
+        # WAR: RL integration needs to use NCCL AllReduce for TP>1 due to a bug in TRTLLM's AllReduce
+        # which will cause convergence issue when using multiple rollout instances.
+        kwargs["allreduce_strategy"] = "NCCL"
+
+        if "ray_worker_extension_cls" not in kwargs:
+            kwargs["ray_worker_extension_cls"] = "tensorrt_llm.llmapi.rlhf_utils.WorkerExtension"
+
+        super().__init__(*args, **kwargs)
+        self._async_initialized = False
+
+    async def setup_async(self):
+        """Setup the LLM asynchronously."""
+        if not self._async_initialized:
+            await self._executor.init_workers_async()
+            await self._executor.setup_engine_remote_async()
+            self._async_initialized = True
+        return self
+
+    async def release(self, tags: list[str]):
+        """Release the GPU memory used by the LLM asynchronously.
+
+        Args:
+            tags: List of memory tag strings to release (e.g., ["model", "kv_cache"]).
+        """
+        await self.collec
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 8a04c05079 - [None][fix] Only Use Throughput Metrics to Check Regression (#10404)

- **Date**: 2026-01-06
- **Author**: chenfeiz0326
- **Categories**: Throughput/Latency

### Optimization Techniques

- General code optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../integration/defs/perf/open_search_db_utils.py  | 201 +++++++++------------
 tests/integration/defs/perf/test_perf_sanity.py    |   8 +-
 2 files changed, 93 insertions(+), 116 deletions(-)
```

### Diff Preview

```diff
diff --git a/tests/integration/defs/perf/open_search_db_utils.py b/tests/integration/defs/perf/open_search_db_utils.py
index 126a99dae..5fe40eecb 100644
--- a/tests/integration/defs/perf/open_search_db_utils.py
+++ b/tests/integration/defs/perf/open_search_db_utils.py
@@ -22,7 +22,6 @@ import sys
 import time
 from datetime import datetime
 
-import yaml
 from defs.trt_test_alternative import print_error, print_info, print_warning
 
 _project_root = os.path.abspath(
@@ -62,6 +61,14 @@ MINIMIZE_METRICS = [
     "d_p99_e2el",
 ]
 
+# Key metrics that determine regression (throughput metrics only)
+REGRESSION_METRICS = [
+    "d_seq_throughput",
+    "d_token_throughput",
+    "d_total_token_throughput",
+    "d_user_throughput",
+]
+
 # Default threshold values for performance regression detection
 POST_MERGE_THRESHOLD = 0.05
 PRE_MERGE_THRESHOLD = 0.1
@@ -96,7 +103,7 @@ def get_job_info():
     global_vars_str = os.getenv("globalVars", "{}")
     try:
         global_vars = json.loads(global_vars_str)
-    except:
+    except Exception:
         global_vars = {}
 
     # Get job_url and job_id
@@ -272,7 +279,7 @@ def query_history_data(common_values_dict):
         if res is None:
             # No response from database, return None
             print_info(
-                f"Fail to query from {TEST_INFO_PROJECT_NAME}, returned no response"
+                f"Failed to query from {TEST_INFO_PROJECT_NAME}, returned no response"
             )
             return None
         else:
@@ -288,24 +295,25 @@ def query_history_data(common_values_dict):
                 data_dict["_id"] = hit.get("_id", "")
                 if data_dict["_id"] == "":
                     print_info(
-                        f"Fail to query from {TEST_INFO_PROJECT_NAME}, returned data with no _id"
+                        f"Failed to query from {TEST_INFO_PROJECT_NAME}, returned data with no _id"
                     )
                     # Invalid data, return None
                     return None
                 data_list.append(data_dict)
             print_info(
-                f"Successfully query from {TEST_INFO_PROJECT_NAME}, queried {len(data_list)} entries"
+                f"Successfully queried from {TEST_INFO_PROJECT_NAME}, queried {len(data_list)} entries"
             )
             return data_list
     except Exception as e:
         print_info(
-            f"Fail to query from {TEST_INFO_PROJECT_NAME}, returned error: {e}")
+            f"Failed to query from {TEST_INFO_PROJECT_NAME}, returned error: {e}"
+        )
         return None
 
 
 def match(history_data, new_data, match_keys):
     """
-    Check if the server and client config of history data matches the new data
+    Check if the server and client config of history data match the new data
     """
 
     def is_empty(value):
@@ -440,7 +448,7 @@ def get_history_data(new_data_dict, match_keys, common_values_dict):
                         history_data_dict[cmd_idx].append(history_data)
   
```

### Analysis Summary

This commit applies performance optimization techniques to improve inference efficiency. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 8cec2da375 - [None][feat] Port fp4 quantization kernel optimization from FlashInfer (#9854)

- **Date**: 2025-12-10
- **Author**: Brian K. Ryu
- **Categories**: Kernel Optimization, Quantization Optimization

### Optimization Techniques

- Custom CUDA kernel
- FP8 quantization
- Integer quantization
- Batching optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
cpp/tensorrt_llm/kernels/quantization.cuh | 119 +++++++++++++++++++-----------
 1 file changed, 77 insertions(+), 42 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/quantization.cuh b/cpp/tensorrt_llm/kernels/quantization.cuh
index 7aacc0f31..665ec2b42 100644
--- a/cpp/tensorrt_llm/kernels/quantization.cuh
+++ b/cpp/tensorrt_llm/kernels/quantization.cuh
@@ -794,67 +794,102 @@ quantize_with_block_size(
 
     asm volatile("griddepcontrol.wait;");
     // Input tensor batch/row/col loops.
+    // Optimization: Iterate over actual rows first (hot path), then padding rows (cold path)
+    // This improves performance for small batch sizes with swizzled layout
     for (int rowIdx = blockIdx.x; rowIdx < numPaddedRowsForSf; rowIdx += gridDim.x)
     {
-        for (int batchIdx = 0; batchIdx < numbatches; batchIdx++)
+        // Early exit for padding-only blocks: if this block only processes padding rows,
+        // we can skip the batch loop and just zero out the scale factors
+        bool isRowPadding = (rowIdx >= numRows);
+
+        if (isRowPadding)
         {
-            for (int colIdx = threadIdx.x; colIdx < numColThreadsForSf; colIdx += blockDim.x)
+            // Fast path: This row is entirely padding, only zero out scale factors.
+            // Note: Padding rows do NOT exist in the output tensor (which is sized [numRows, K]),
+            // they only exist in the swizzled scale factor layout. Do NOT write to output buffer here.
+            for (int batchIdx = 0; batchIdx < numbatches; batchIdx++)
             {
-                std::optional<int> optionalBatchIdx = batchIdx;
-                std::optional<int> optionalNumRows = numRows;
-
-                // The SF output pointer.
-                auto sf_out = cvt_quant_get_sf_out_offset<uint32_t, CVT_NUM_THREADS_PER_SF>(
-                    optionalBatchIdx, rowIdx, colIdx, optionalNumRows, numPaddedCols / SF_VEC_SIZE, SFout, layout);
-
-                // The input tensor offset.
-                int64_t inOffset = static_cast<int64_t>(batchIdx * numRows + rowIdx) * numColThreads + colIdx;
-                int64_t outOffset = static_cast<int64_t>(batchIdx * numRows + rowIdx) * numPaddedColThreads + colIdx;
-
-                // Set the values to 0 of those are padded columns.
-                if (rowIdx < numRows && colIdx >= numColThreads && colIdx < numPaddedColThreads)
+                for (int colIdx = threadIdx.x; colIdx < numColThreadsForSf; colIdx += blockDim.x)
                 {
-                    // Dispatch the quantization kernel.
-                    if constexpr (quantization_type == BlockScaleQuantizationType::FP16_TO_FP4)
-                    {
-                        reinterpret_cast<uint32_t*>(out)[outOffset] = 0u;
-                    }
-                    else if constexpr (quantization_type == BlockScaleQuantizationType::FP8_TO_FP4
-                        || quantization_type == BlockScaleQuantizationType::FP16_TO_MXFP8)
-                    {
-                        reinterpret_cast<uint64_t*>(out)[outOffset] = 0ull;
-                    }
-                }
+                
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 8e6eead6a5 - refactor: (part1) Add contraints doc for fusedMoe module. (#3882)

- **Date**: 2025-04-29
- **Author**: HuiGao-NV
- **Categories**: Fusion

### Optimization Techniques

- Operator fusion
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- NVIDIA Hopper (SM90) GPU

### Changed Files

```
tensorrt_llm/_torch/modules/fused_moe.py | 77 +++++++++++++++++++++++++++++---
 tensorrt_llm/_torch/utils.py             |  4 ++
 2 files changed, 74 insertions(+), 7 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fused_moe.py b/tensorrt_llm/_torch/modules/fused_moe.py
index 773764f61..d2cfe81cb 100755
--- a/tensorrt_llm/_torch/modules/fused_moe.py
+++ b/tensorrt_llm/_torch/modules/fused_moe.py
@@ -232,6 +232,45 @@ class FusedMoE(nn.Module):
         reduce_results (bool): Whether to reduce the results across devices.
         model_config (ModelConfig): Configuration object for the model.
         enable_alltoall (bool): whether to enable alltoall instead of allgather/reducescatter
+
+    MoE torch custom op:
+        cutlass Backend
+            In min-latency mode:
+            Quant:
+                fp8 block scales (SM90 Hopper only):
+                    FusedMoE Op: dynamic quant + gemm1 + swiglu + gemm2 (return tensor list).
+                fp8 qdq, nvfp4:
+                    FusedMoE Op: gemm1 + swiglu + gemm2 (return tensor list).
+
+            In max-throughput mode:
+            Quant:
+                fp8 block scales (SM90 Hopper only):
+                    FusedMoE Op: dynamic quant + scatter + gemm1 + swiglu + gemm2 + finalizeMoeRoute (return one tensor)
+                p8 qdq, nvfp4:
+                    FusedMoE Op: scatter + gemm1 + swiglu + gemm2 + finalizeMoeRoute (return one tensor)
+
+        trtllm_gen backend:
+            Only support min-latency mode now (SM100 Blackwell only).
+            Quant: fp8 block scales quant and nvfp4 quant
+                FusedMoE Op: routing(topK, etc.) + scatter + gemm1 + swiglu + gemm2 + finalize MoeRoute
+
+    FusedMoE module:
+        cutlass Backend (moe_backend="CUTLASS"):
+            min-latency mode:
+                routing(topK, etc.) + FusedMoE Op
+                equals to: routing(topK, etc.) [+ dynamic quant fp8 qdq | optional dynamic quant nvfp4] + gemm1 + swiglu + gemm2
+
+            max-throughput mode:
+                routing(topK, etc.) [+ dynamic quant for fp8 qdq and nvfp4 ] [+ fp4_allgather] + FusedMoe Op[no allreduce] + reducescatter, with AttentionDP on
+                equals to: dynamic quant + routing(topK, etc.) [+ fp4_allgather] + scatter + gemm1 + swiglu + gemm2 + finalizeMoeRoute [no allreduce] + reducescatter
+
+        trtllm_gen backend (moe_backend="TRTLLM"):
+            min-latency mode (min_latency_mode flag of forward has no effect when trtllm_gen is used):
+                dynamic quant + FusedMoe Op
+                equals to: dynamic quant + routing(topK, etc.) + scatter + gemm1 + swiglu + gemm2 + finalize MoeRoute
+
+    In min-latency mode, setting `reduce_results=False` disables the AllReduce in the FusedMoE module, so any necessary AllReduce operations must be added explicitly in the model definition.
+    AttentionDP should be turned off for min-latency mode.
     """
 
     def __init__(
@@ -332,6 +371,31 @@ class FusedMoE(nn.Module):
 
         # If True, the router weight will be multiplied on the input rather than at the end of FC2
         self.apply_router_weight_on_input = apply_router_weight_on_
```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 90145cf557 - [None][feat] Optimize CUDA graph memory usage for spec decode cases (#6718)

- **Date**: 2025-08-08
- **Author**: Mike Iovine
- **Categories**: Memory Optimization

### Optimization Techniques

- Batching optimization
- Speculative decoding

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/model_engine.py | 7 +++++--
 tensorrt_llm/_torch/speculative/drafter.py     | 9 +++++++--
 2 files changed, 12 insertions(+), 4 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/model_engine.py b/tensorrt_llm/_torch/pyexecutor/model_engine.py
index dba077f38..d39ddc4f2 100644
--- a/tensorrt_llm/_torch/pyexecutor/model_engine.py
+++ b/tensorrt_llm/_torch/pyexecutor/model_engine.py
@@ -726,8 +726,11 @@ class PyTorchModelEngine(ModelEngine):
             # For non-draft model, we also capture the CUDA graph instance for draft length 0,
             # so that when we disable spec decode at runtime, we can still run the captured graph.
             # Note that for one engine mode, we are not able to turn off spec decode at runtime.
-            if not self.is_draft_model and self.max_draft_len > 0 and not self.spec_config.spec_dec_mode.use_one_engine(
-            ):
+            if (not self.is_draft_model and self.max_draft_len > 0
+                    and not self.spec_config.spec_dec_mode.use_one_engine()
+                    # Assume that speculation is always on if the user didn't give us a max_concurrency
+                    # value. This will save on memory.
+                    and self.spec_config.max_concurrency is not None):
                 draft_lengths.append(0)
 
             for bs in cuda_graph_batch_sizes:
diff --git a/tensorrt_llm/_torch/speculative/drafter.py b/tensorrt_llm/_torch/speculative/drafter.py
index 4f2ea0b70..82d816b80 100644
--- a/tensorrt_llm/_torch/speculative/drafter.py
+++ b/tensorrt_llm/_torch/speculative/drafter.py
@@ -1,5 +1,5 @@
 from abc import ABC, abstractmethod
-from typing import List, Optional
+from typing import List, Optional, final
 
 from ..pyexecutor.llm_request import LlmRequest
 from ..pyexecutor.resource_manager import ResourceManager
@@ -26,8 +26,13 @@ class Drafter(ABC):
         """
         raise NotImplementedError
 
+    @final
     def should_use_spec_decode(self, requests: List[LlmRequest]) -> bool:
-        """Check if spec decode should be used for the current iteration."""
+        """
+        You probably don't want to override this. ModelEngine
+        assumes that speculation is always on if max_concurrency
+        is not specified by the user's spec config.
+        """
         if self.max_concurrency is not None:
             return len(requests) <= self.max_concurrency
         return True

```

### Analysis Summary

Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 908463a5f5 - [feat]: improve performance of XQA-MLA for sm120 (#5087)

- **Date**: 2025-06-18
- **Author**: Yao Yao
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- FP8 quantization
- Batching optimization
- Reduce synchronization overhead
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
cpp/kernels/xqa/CMakeLists.txt |  37 +++--
 cpp/kernels/xqa/barriers.cuh   |   2 +-
 cpp/kernels/xqa/mla_sm120.cu   | 363 ++++++++++++++++++++++++-----------------
 cpp/kernels/xqa/test/test.cpp  |  22 ++-
 cpp/kernels/xqa/utils.cuh      |  46 ++++++
 5 files changed, 298 insertions(+), 172 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/kernels/xqa/CMakeLists.txt b/cpp/kernels/xqa/CMakeLists.txt
index 52dbc7842..50a39dc33 100644
--- a/cpp/kernels/xqa/CMakeLists.txt
+++ b/cpp/kernels/xqa/CMakeLists.txt
@@ -84,21 +84,30 @@ add_custom_command(
 add_custom_target(xqa_sources_h DEPENDS ${XQA_SOURCES_H})
 
 if(BUILD_XQA_TESTS)
-  # GoogleTest Preparation - Code block copied from
-  # https://google.github.io/googletest/quickstart-cmake.html
-  include(FetchContent)
-  FetchContent_Declare(
-    googletest
-    GIT_REPOSITORY https://github.com/google/googletest.git
-    GIT_TAG v1.15.2)
-  include(GoogleTest)
+  # Try to find system installed GTest first
+  find_package(GTest QUIET)
+  if(NOT GTest_FOUND)
+    message(STATUS "System GTest not found, fetching from repository")
+    include(FetchContent)
+    FetchContent_Declare(
+      googletest
+      GIT_REPOSITORY https://github.com/google/googletest.git
+      GIT_TAG v1.15.2)
+    FetchContent_MakeAvailable(googletest)
+    include(GoogleTest)
+  endif()
 
-  # Add Eigen via FetchContent
-  FetchContent_Declare(
-    eigen
-    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
-    GIT_TAG 3.4.0)
-  FetchContent_MakeAvailable(googletest eigen)
+  # Try to find system installed Eigen first
+  find_package(Eigen3 3.4 QUIET)
+  if(NOT Eigen3_FOUND)
+    message(STATUS "System Eigen not found, fetching from repository")
+    include(FetchContent)
+    FetchContent_Declare(
+      eigen
+      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
+      GIT_TAG 3.4.0)
+    FetchContent_MakeAvailable(eigen)
+  endif()
 
   enable_testing()
   add_executable(
diff --git a/cpp/kernels/xqa/barriers.cuh b/cpp/kernels/xqa/barriers.cuh
index cc157599e..3c0318be4 100644
--- a/cpp/kernels/xqa/barriers.cuh
+++ b/cpp/kernels/xqa/barriers.cuh
@@ -434,7 +434,7 @@ using CtaBarrier = MBarrier<Scope::CTA>;
 using CgaBarrier = MBarrier<Scope::CGA>;
 
 template <uint32_t nbBars>
-__device__ inline bool toParity(uint32_t i)
+__device__ inline constexpr bool toParity(uint32_t i)
 {
     return i % (nbBars * 2) / nbBars;
 }
diff --git a/cpp/kernels/xqa/mla_sm120.cu b/cpp/kernels/xqa/mla_sm120.cu
index e34c25133..4ae50d5b5 100644
--- a/cpp/kernels/xqa/mla_sm120.cu
+++ b/cpp/kernels/xqa/mla_sm120.cu
@@ -29,6 +29,8 @@
 #include <cuda_runtime.h>
 #endif
 
+#define USE_REG_Q 1
+
 __constant__ constexpr XQAKernelType kernelType = XQAKernelType::kSM120_MLA;
 
 inline constexpr bool allowMultipleInputTokens = true;
@@ -210,13 +212,13 @@ public:
         return ret;
     }
 
-    __device__ inline LdGrain const* const getPtr(uint32_t idxInstM) const
+    __device__ inline LdGrain const* getPtr(uint32_t idxInstM) const
     {
         return checkedVal(basePtr + idxInstM * qmmaShape.m * srcCols, getPtrRef(idxInstM));
     }
 
 private:
-    __device__ inline LdGrain const* const getPtrRef(uint32_t idxInstM) const
+    __device__ inline LdGrain const* getPtrRef(uint32_t idxInstM) const
     {
         return &src.template at<true>(
             b
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 91528365a9 - [None][feat] Add performance alignment to layer-wise benchmarks (#11018)

- **Date**: 2026-01-29
- **Author**: Tailing Yuan
- **Categories**: General Performance

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
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- Long sequence / large context scenarios
- Decode/generation phase
- Disaggregated serving

### Changed Files

```
examples/layer_wise_benchmarks/README.md           | 115 ++++
 .../{template.html => breakdown_template.html}     |   0
 examples/layer_wise_benchmarks/correlation.py      |  96 +++
 .../correlation_template.html                      | 152 +++++
 .../middleware/mpi_env_from_ompi                   |  10 +
 examples/layer_wise_benchmarks/parse.py            | 180 ++----
 examples/layer_wise_benchmarks/parse_e2e.py        | 247 +++++++
 examples/layer_wise_benchmarks/parser_utils.py     | 216 +++++++
 examples/layer_wise_benchmarks/run.py              |  43 +-
 .../sample_performance_alignment.sh                | 144 +++++
 examples/layer_wise_benchmarks/slurm_alloc.sh      |   1 +
 .../layer_wise_benchmarks/slurm_init_containers.sh |   2 +-
 .../_torch/modules/fused_moe/configurable_moe.py   |   4 +
 .../_torch/modules/fused_moe/fused_moe_cutlass.py  |   3 +
 .../_torch/modules/fused_moe/fused_moe_wide_ep.py  |   3 +
 tensorrt_llm/_torch/pyexecutor/py_executor.py      |   8 +
 .../_torch/pyexecutor/py_executor_creator.py       |  10 +
 tensorrt_llm/llmapi/llm_args.py                    |  37 ++
 .../tools/layer_wise_benchmarks/__init__.py        |   5 +-
 .../tools/layer_wise_benchmarks/calibrator.py      | 716 +++++++++++++++++++++
 tensorrt_llm/tools/layer_wise_benchmarks/runner.py |  44 +-
 tests/integration/test_lists/test-db/l0_b200.yml   |   1 +
 tests/unittest/api_stability/references/llm.yaml   |   4 +
 tests/unittest/tools/test_layer_wise_benchmarks.py |  20 +
 24 files changed, 1908 insertions(+), 153 deletions(-)
```

### Diff Preview

```diff
diff --git a/examples/layer_wise_benchmarks/README.md b/examples/layer_wise_benchmarks/README.md
index 426dee02f..7bdc794a9 100644
--- a/examples/layer_wise_benchmarks/README.md
+++ b/examples/layer_wise_benchmarks/README.md
@@ -171,6 +171,121 @@ You will receive three reports, each containing kernel timing statistics grouped
 2. A CSV report at `profiles/report_np4_rank0.csv`
 3. An HTML report at `profiles/report_np4_rank0.html`
 
+## Performance alignment between end-to-end performance and layer-wise benchmarks
+
+An overall example can be found in `sample_performance_alignment.sh`. Here is an abstract of the main steps.
+
+1. Run end-to-end serving in **COLLECT** mode, and capture nsys profiles. This step generates a calibration file.
+
+   Please meet the following requirements.
+
+   1. Add the following fields to `config.yaml`.
+
+      ```yaml
+      layer_wise_benchmarks_config:
+          calibration_mode: COLLECT
+          calibration_file_path: profiles/calibration_data.json
+      ```
+
+   2. Set `TLLM_PROFILE_START_STOP` to a range that can capture some iterations (typically tens of iterations) of GEN phase. Ensure every iteration has the same batch size. Please capture 5 more iterations at beginning, because the first 5 iterations are regarded as warm-ups and will be dropped by the parser by default.
+
+   3. Capture per-rank nsys profiles, and every rank should produce a separate file.
+
+      You need to put `nsys profile` behind `mpirun` or `srun`. To minimize profile overhead and file size, there is no need to capture samples and GPU metrics.
+
+      If you use `trtllm-serve` or `trtllm-bench`, please follow the following command order. If you use `examples/disaggregated/slurm/benchmark/submit.py`, setting `gen_profile_range` is enough.
+
+      ```bash
+      NP=$NP ./mpi_launch.sh middleware/mpi_env_from_ompi \
+      nsys profile \
+          -t cuda,nvtx \
+          --cpuctxsw none --cuda-event-trace false \
+          --cuda-graph-trace node \
+          -c cudaProfilerApi --capture-range-end stop \
+          -o profiles/report_e2e_collect_rank%q{RANK}.nsys-rep \
+          --force-overwrite true \
+      trtllm-llmapi-launch \
+      trtllm-bench \
+          --model ...
+      ```
+
+   4. To be more precise, set the same `TLLM_AUTOTUNER_CACHE_PATH` for all the steps. The autotuner cache file should be generated by Step 1, and be reused by Step 2 and Step 3.
+
+2. If the end-to-end serving uses CUDA Graphs, run Step 1 again in **MARK** mode without CUDA Graphs, and also capture nsys profiles.
+
+   The differences are as follows.
+
+   1. Add the following fields to `config.yaml`.
+
+      ```yaml
+      cuda_graph_config: null
+      layer_wise_benchmarks_config:
+          calibration_mode: MARK
+      ```
+
+   2. Change the paths of profiles. The recommended argument is `-o profiles/report_e2e_mark_rank%q{RANK}.nsys-rep`.
+
+3. Run layer-wise benchmarks with the calibration file obtained by Step 1.
+
+   ```bas
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Pinned (page-locked) memory enables faster host-to-device data transfers. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Most effective for long-context inference workloads.

---

## 93a54457ac - [nvbugs/5274894] fix: Sort requests for functional correctness and performance (adapted from #4608) (#4621)

- **Date**: 2025-05-26
- **Author**: Robin Kobus
- **Categories**: General Performance

### Optimization Techniques

- Async/stream-based execution
- KV cache optimization
- Batching optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
.../batch_manager/microBatchScheduler.cpp          | 11 ++------
 .../batch_manager/trtGptModelInflightBatching.cpp  |  2 --
 .../batch_manager/utils/inflightBatchingUtils.cpp  | 31 +++++++++++++++++-----
 .../batch_manager/utils/inflightBatchingUtils.h    |  8 +++++-
 4 files changed, 33 insertions(+), 19 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp b/cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp
index 169108e3d..6a2dc46d5 100644
--- a/cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp
+++ b/cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp
@@ -16,10 +16,9 @@
  */
 
 #include "tensorrt_llm/batch_manager/microBatchScheduler.h"
+#include "tensorrt_llm/batch_manager/utils/inflightBatchingUtils.h"
 #include "tensorrt_llm/common/nvtxUtils.h"
 
-namespace tle = tensorrt_llm::executor;
-
 namespace tensorrt_llm::batch_manager
 {
 
@@ -310,13 +309,7 @@ std::tuple<RequestVector, RequestVector> MicroBatchScheduler::operator()(Request
         }
     }
 
-    if (!allContextRequestsFit)
-    {
-        // Move context requests that reached the last context chunk to the end of the vector.
-        // This order is required for moveFinishedContextRequestsToGeneration.
-        std::partition(contextRequests.begin(), contextRequests.end(),
-            [](auto const& llmReq) { return !llmReq->isLastContextChunk(); });
-    }
+    utils::sortRequests(contextRequests, generationRequests, !allContextRequestsFit);
 
     TLLM_LOG_DEBUG(
         "batchSize (num ctx/enc requests + num gen requests): %u", contextRequests.size() + generationRequests.size());
diff --git a/cpp/tensorrt_llm/batch_manager/trtGptModelInflightBatching.cpp b/cpp/tensorrt_llm/batch_manager/trtGptModelInflightBatching.cpp
index 2593a7d42..0a1b6f03e 100644
--- a/cpp/tensorrt_llm/batch_manager/trtGptModelInflightBatching.cpp
+++ b/cpp/tensorrt_llm/batch_manager/trtGptModelInflightBatching.cpp
@@ -978,8 +978,6 @@ void TrtGptModelInflightBatching::forwardAsync(RequestList const& activeRequests
                 }
             }
 
-            utils::sortByLoraId(currRequests);
-
             (*mAssignReqSeqSlots)(*mSeqSlotManager, currRequests.contextRequests, currRequests.generationRequests);
 
             if (mKvCacheManager)
diff --git a/cpp/tensorrt_llm/batch_manager/utils/inflightBatchingUtils.cpp b/cpp/tensorrt_llm/batch_manager/utils/inflightBatchingUtils.cpp
index b34bd9cf6..466146f30 100644
--- a/cpp/tensorrt_llm/batch_manager/utils/inflightBatchingUtils.cpp
+++ b/cpp/tensorrt_llm/batch_manager/utils/inflightBatchingUtils.cpp
@@ -39,17 +39,32 @@ TensorPtr collectRequestIds(RequestVector const& contextRequests, RequestVector
     return requestIds;
 }
 
-void sortByLoraId(ScheduledRequests& scheduledRequests)
+void sortRequests(RequestVector& contextRequests, RequestVector& generationRequests, bool chunksPresent)
 {
     TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
 
-    auto sortRequests = [](RequestVector& requests)
+    auto sortByLoraId = [](RequestVector::iterator begin, RequestVector::iterator end)
     {
-        std::sort(requests.begin(), requests.end(),
-            [](auto const& lhs, auto const& rhs) { return lhs->getLoraTaskId() < rhs->getLoraTaskId(); });
+        std::sort(
+            begin, end, [](auto const& lhs, auto const&
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 946ffcd2eb - [None][ci] optimize test cases of dgx b200 (#7948)

- **Date**: 2025-09-24
- **Author**: QI JUN
- **Categories**: General Performance

### Optimization Techniques

- FP8 quantization
- Parallelism optimization
- Speculative decoding
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- Prefill phase
- Disaggregated serving

### Changed Files

```
.../integration/test_lists/test-db/l0_dgx_b200.yml | 26 +++++++++++-----------
 1 file changed, 13 insertions(+), 13 deletions(-)
```

### Diff Preview

```diff
diff --git a/tests/integration/test_lists/test-db/l0_dgx_b200.yml b/tests/integration/test_lists/test-db/l0_dgx_b200.yml
index 496ff4dce..30d753ced 100644
--- a/tests/integration/test_lists/test-db/l0_dgx_b200.yml
+++ b/tests/integration/test_lists/test-db/l0_dgx_b200.yml
@@ -23,8 +23,6 @@ l0_dgx_b200:
   - accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_bfloat16_4gpus[ep4-mtp_nextn=2-attention_dp=True-cuda_graph=True-overlap_scheduler=True-torch_compile=False]
   - accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_bfloat16_4gpus[tp2pp2-mtp_nextn=0-attention_dp=False-cuda_graph=False-overlap_scheduler=False-torch_compile=True]
   - accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_bfloat16_4gpus[tp2pp2-mtp_nextn=2-attention_dp=True-cuda_graph=True-overlap_scheduler=True-torch_compile=False]
-  - accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_bfloat16_4gpus[pp4-mtp_nextn=0-attention_dp=False-cuda_graph=False-overlap_scheduler=False-torch_compile=False]
-  - accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_bfloat16_4gpus[pp4-mtp_nextn=2-attention_dp=True-cuda_graph=True-overlap_scheduler=True-torch_compile=False]
   - accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_nvfp4_4gpus[moe_backend=CUTLASS-mtp_nextn=0-tp4-fp8kv=False-attention_dp=False-cuda_graph=False-overlap_scheduler=False-torch_compile=False]
   - accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_nvfp4_4gpus[moe_backend=CUTLASS-mtp_nextn=0-tp4-fp8kv=True-attention_dp=True-cuda_graph=True-overlap_scheduler=True-torch_compile=False]
   - accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_nvfp4_4gpus[moe_backend=CUTLASS-mtp_nextn=0-tp2pp2-fp8kv=False-attention_dp=False-cuda_graph=False-overlap_scheduler=False-torch_compile=False]
@@ -40,18 +38,7 @@ l0_dgx_b200:
   - accuracy/test_llm_api_pytorch.py::TestQwen3_30B_A3B::test_nvfp4[dep4_latency_moe_trtllm-torch_compile=False]
   - accuracy/test_llm_api_pytorch.py::TestQwen3_30B_A3B::test_nvfp4[dep4_latency_moe_cutlass-torch_compile=False]
   - accuracy/test_llm_api_pytorch.py::TestQwen3_30B_A3B::test_nvfp4[dep4_latency_moe_cutlass-torch_compile=True]
-  - accuracy/test_llm_api_pytorch.py::TestLlama4ScoutInstruct::test_auto_dtype[tp8-cuda_graph=False]
-  - accuracy/test_llm_api_pytorch.py::TestLlama4ScoutInstruct::test_auto_dtype[tp8ep4-cuda_graph=True]
-  - accuracy/test_llm_api_pytorch.py::TestLlama4ScoutInstruct::test_auto_dtype[tp8ep8-cuda_graph=True]
-  - accuracy/test_llm_api_pytorch.py::TestLlama4ScoutInstruct::test_auto_dtype[tp4-cuda_graph=False]
-  - accuracy/test_llm_api_pytorch.py::TestLlama4ScoutInstruct::test_auto_dtype[tp4ep2-cuda_graph=True]
-  - accuracy/test_llm_api_pytorch.py::TestLlama4ScoutInstruct::test_auto_dtype[tp4ep4-cuda_graph=True]
-  - accuracy/test_llm_api_pytorch.py::TestLlama4ScoutInstruct::test_fp8[tp8ep8-cuda_graph=True]
-  - accuracy/test_llm_api_pytorch.py::TestLlama4ScoutInstruct::test_fp8[tp4-cuda_graph=True]
-  - accuracy/test_l
```

### Analysis Summary

Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## 94e6167879 - optimize cudaMemGetInfo for TllmGenFmhaRunner (#3907)

- **Date**: 2025-04-29
- **Author**: zhhuang-nv
- **Categories**: General Performance

### Optimization Techniques

- Async/stream-based execution
- FP8 quantization
- KV cache optimization
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
cpp/tensorrt_llm/common/attentionOp.cpp                       | 5 ++---
 cpp/tensorrt_llm/kernels/fmhaDispatcher.cpp                   | 4 ++--
 cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunner.cpp | 8 ++++++++
 cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunner.h   | 5 +++++
 cpp/tensorrt_llm/kernels/xqaDispatcher.cpp                    | 5 ++---
 5 files changed, 19 insertions(+), 8 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/common/attentionOp.cpp b/cpp/tensorrt_llm/common/attentionOp.cpp
index 58525f0be..a3122aebb 100644
--- a/cpp/tensorrt_llm/common/attentionOp.cpp
+++ b/cpp/tensorrt_llm/common/attentionOp.cpp
@@ -936,6 +936,7 @@ int AttentionOp::mlaGeneration(
 
     if (mUseTllmGen)
     {
+        TLLM_CHECK_WITH_INFO(mTllmGenFMHARunner.get(), "mTllmGenFMHARunner not initialized.");
         TllmGenFmhaRunnerParams tllmRunnerParams;
         memset(&tllmRunnerParams, 0, sizeof(tllmRunnerParams));
 
@@ -991,10 +992,9 @@ int AttentionOp::mlaGeneration(
         tllmRunnerParams.mScaleQ = mQScaling * sqrt((float) (mMLAParams.qk_nope_head_dim + mMLAParams.qk_rope_head_dim))
             / sqrtf((float) (mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim));
 
-        auto const [freeMemory, totalMemory] = tensorrt_llm::common::getDeviceMemoryInfo(false);
         // The kv cache should be based on the maximum headDim of K and V due to paddings.
         int maxHeadDimKv = std::max(tllmRunnerParams.mHeadDimQk, tllmRunnerParams.mHeadDimV);
-        tllmRunnerParams.mNumPagesInMemPool = totalMemory
+        tllmRunnerParams.mNumPagesInMemPool = mTllmGenFMHARunner->getTotalDeviceMemory()
             / (tllmRunnerParams.mNumHeadsKv * tllmRunnerParams.mNumTokensPerPage * maxHeadDimKv * elemSize);
 
         tllmRunnerParams.mMultiProcessorCount = mMultiProcessorCount;
@@ -1010,7 +1010,6 @@ int AttentionOp::mlaGeneration(
                 = reinterpret_cast<float const*>(params.bmm1_scale) + bmm1_scale_offset;
         }
 
-        TLLM_CHECK_WITH_INFO(mTllmGenFMHARunner.get(), "mTllmGenFMHARunner not initialized.");
         mTllmGenFMHARunner->run(tllmRunnerParams);
         sync_check_cuda_error(stream);
     }
diff --git a/cpp/tensorrt_llm/kernels/fmhaDispatcher.cpp b/cpp/tensorrt_llm/kernels/fmhaDispatcher.cpp
index d255ef819..62fdcfe99 100644
--- a/cpp/tensorrt_llm/kernels/fmhaDispatcher.cpp
+++ b/cpp/tensorrt_llm/kernels/fmhaDispatcher.cpp
@@ -126,6 +126,7 @@ void FmhaDispatcher::run(MHARunnerParams runnerParams)
     if (mUseTllmGen)
     {
         TLLM_LOG_DEBUG("Running TRTLLM-GEN context FMHA kernel.");
+        TLLM_CHECK_WITH_INFO(mTllmGenFMHARunner.get(), "mTllmGenFMHARunner not initialized.");
         // Convert from MHAFixedParams + MHARunnerParams to TllmGenFmhaRunnerParams
         void const* kvPoolPtr = nullptr;
         void const* kvPageIdxPtr = nullptr;
@@ -189,10 +190,9 @@ void FmhaDispatcher::run(MHARunnerParams runnerParams)
         tllmRunnerParams.mScaleQ = mFixedParams.qScaling;
         if (mFixedParams.attentionInputLayout == AttentionInputLayout::Q_PAGED_KV)
         {
-            auto const [freeMemory, totalMemory] = tensorrt_llm::common::getDeviceMemoryInfo(false);
             // The kv cache should be based on the maximum headDim of K and V due to paddings.
             int maxHeadDimKv = std::max(tllmRunnerParams.mHeadDimQk, tllmRunnerParams.mHeadDimV);
-            tllmRunnerParams.mNumPagesInMemP
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 9667ea3fff - [#11529][perf] AD host time attention MD optimization for large context (#11624)

- **Date**: 2026-02-25
- **Author**: Eran Geva
- **Categories**: Host-side Optimization

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- KV cache optimization
- Batching optimization
- Pinned memory
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../auto_deploy/custom_ops/attention_interface.py  | 111 ++++++++++++++-------
 1 file changed, 74 insertions(+), 37 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py b/tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py
index 415b434fb..bb2d3a359 100644
--- a/tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py
+++ b/tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py
@@ -28,6 +28,7 @@ import math
 from abc import ABC, abstractmethod
 from typing import Dict, List, Literal, Optional, Protocol, Sequence, Set, Tuple, Type, Union
 
+import numpy as np
 import torch
 from torch._ops import OpOverloadPacket
 from torch.fx import Node
@@ -40,6 +41,29 @@ from ..utils.logger import ad_logger
 
 Constant = Union[int, float, str, None]
 
+# Torch dtype → numpy dtype for fast list-to-tensor conversion.
+# numpy's list→array conversion is ~2-3x faster than torch.tensor(list) for large lists.
+_TORCH_TO_NUMPY_DTYPE: Dict[torch.dtype, np.dtype] = {
+    torch.int: np.int32,
+    torch.int32: np.int32,
+    torch.int64: np.int64,
+    torch.long: np.int64,
+    torch.float: np.float32,
+    torch.float32: np.float32,
+    torch.float64: np.float64,
+    torch.double: np.float64,
+    torch.float16: np.float16,
+    torch.bool: np.bool_,
+}
+
+
+def _list_to_tensor(data: list, dtype: torch.dtype) -> torch.Tensor:
+    """Convert a Python list to a tensor, using numpy for speed."""
+    np_dtype = _TORCH_TO_NUMPY_DTYPE.get(dtype)
+    if np_dtype is not None:
+        return torch.from_numpy(np.array(data, dtype=np_dtype))
+    return torch.tensor(data, dtype=dtype)
+
 
 class PrepareMetadataHostCallable(Protocol):
     def __call__(self, **sequence_info_args: torch.Tensor) -> None: ...
@@ -184,15 +208,15 @@ class InputBuffer:
     def store(
         self,
         name: str,
-        data: List[Number],
+        data: torch.Tensor,
         fill_value: Optional[Number] = None,
     ) -> int:
-        """Store data into the host buffer.
+        """Store a tensor into the pinned host buffer.
 
         Args:
             name: Name of the tensor to store to.
-            data: List of values to store.
-            fill_value: Optional value to fill the entire tensor with before storing.
+            data: 1-D torch.Tensor to store.
+            fill_value: Optional value to fill the entire buffer with before storing.
 
         Returns:
             Number of elements stored.
@@ -203,11 +227,12 @@ class InputBuffer:
         if fill_value is not None:
             host_view.fill_(fill_value)
 
-        length = len(data)
+        length = data.numel()
         assert length <= numel, f"Data too large for buffer '{name}': {length} > {numel}"
-
-        temp_tensor = torch.tensor(data, dtype=dtype)
-        host_view[:length].copy_(temp_tensor)
+        # Use numpy for the memcpy into pinned memory — avoids torch dispatcher overhead
+        dst = host_view[:length].numpy()
+        src = (data if data.dtype == dtype else data.to(dtype)).numpy()
+        np.copyto(dst, src)
 
         self._current_lengths[na
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Pinned (page-locked) memory enables faster host-to-device data transfers. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 97657bfda2 - optimize memset before alltoall communication (#5188)

- **Date**: 2025-06-14
- **Author**: dongxuy04
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- Parallelism optimization
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
cpp/tensorrt_llm/kernels/moeCommKernels.cu | 56 +++++++++++++++++++++++-------
 1 file changed, 44 insertions(+), 12 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/moeCommKernels.cu b/cpp/tensorrt_llm/kernels/moeCommKernels.cu
index eabb5a395..f3228af4f 100644
--- a/cpp/tensorrt_llm/kernels/moeCommKernels.cu
+++ b/cpp/tensorrt_llm/kernels/moeCommKernels.cu
@@ -646,6 +646,48 @@ void computeSendRecvIndices(MoeEpWorldInfo worldInfo, MoeExpertParallelInfo expe
         sendRankLocalIndices, recvRankLocalIndices, backwardRecvRankLocalIndices);
 }
 
+__global__ void moeAllToAllMemsetKernel(MoeEpWorldInfo worldInfo, MoeExpertParallelInfo expertParallelInfo,
+    int maxTokenCountPerRank, int* sendRankCountCumSum, int* recvRankCountCumSum, int* localGatherIndices,
+    int* sendRankLocalIndices, int* recvRankLocalIndices, int* backwardRecvRankLocalIndices)
+{
+    int maxSendRanksPerToken = std::max(worldInfo.epSize, expertParallelInfo.topK);
+    int idx = threadIdx.x + blockIdx.x * blockDim.x;
+    int maxRankRecvTokenCount = maxTokenCountPerRank * worldInfo.epSize;
+    int maxRankSendTokenCount = maxTokenCountPerRank * maxSendRanksPerToken;
+    if (idx < worldInfo.epSize)
+    {
+        sendRankCountCumSum[idx] = 0;
+        recvRankCountCumSum[idx] = 0;
+    }
+    if (idx < maxRankRecvTokenCount)
+    {
+        localGatherIndices[idx] = -1;
+        recvRankLocalIndices[idx] = -1;
+    }
+    if (idx < maxRankSendTokenCount)
+    {
+        sendRankLocalIndices[idx] = -1;
+        backwardRecvRankLocalIndices[idx] = -1;
+    }
+}
+
+void moeAllToAllMemset(MoeEpWorldInfo worldInfo, MoeExpertParallelInfo expertParallelInfo, int maxTokenCountPerRank,
+    int* sendRankCountCumSum, int* recvRankCountCumSum, int* localGatherIndices, int* sendRankLocalIndices,
+    int* recvRankLocalIndices, int* backwardRecvRankLocalIndices, cudaStream_t stream)
+{
+    int maxSendRanksPerToken = std::max(worldInfo.epSize, expertParallelInfo.topK);
+    int maxRankRecvTokenCount = maxTokenCountPerRank * worldInfo.epSize;
+    int maxRankSendTokenCount = maxTokenCountPerRank * maxSendRanksPerToken;
+    int maxEltCount = std::max<int>(maxRankRecvTokenCount, maxRankSendTokenCount);
+    maxEltCount = std::max<int>(maxEltCount, worldInfo.epSize);
+    static constexpr int kBlockSize = 256;
+    int blockCount = (maxEltCount + kBlockSize - 1) / kBlockSize;
+    dim3 grid(blockCount, 1);
+    moeAllToAllMemsetKernel<<<grid, kBlockSize, 0, stream>>>(worldInfo, expertParallelInfo, maxTokenCountPerRank,
+        sendRankCountCumSum, recvRankCountCumSum, localGatherIndices, sendRankLocalIndices, recvRankLocalIndices,
+        backwardRecvRankLocalIndices);
+}
+
 void moeAllToAllPrepareIndices(MoeEpWorldInfo worldInfo, MoeExpertParallelInfo expertParallelInfo,
     int maxTokenCountPerRank, int const* gatheredTargetRankIds, int const* realRankTokenCountCumSum,
     // indices of gatheredTargetRankIds that has the local rank in topK
@@ -663,19 +705,9 @@ void moeAllToAllPrepareIndices(MoeEpWorldInfo worldInfo, MoeExpertParallelInfo e
                                       // rank has maxTokenCountPerRa
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 97b38ac403 - [None] [doc] Update IFB performance guide & GPTOSS deployment guide (#10283)

- **Date**: 2025-12-25
- **Author**: Jatin Gangani
- **Categories**: General Performance

### Optimization Techniques

- Async/stream-based execution
- FP8 quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Triton kernel
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- NVIDIA Hopper (SM90) GPU
- Prefill phase
- Decode/generation phase

### Changed Files

```
.../deployment-guide-for-gpt-oss-on-trtllm.md             | 11 ++++++-----
 docs/source/features/paged-attention-ifb-scheduler.md     | 15 +++++++--------
 examples/configs/curated/gpt-oss-120b-latency.yaml        |  6 +++---
 examples/configs/curated/gpt-oss-120b-throughput.yaml     |  6 +++---
 4 files changed, 19 insertions(+), 19 deletions(-)
```

### Diff Preview

```diff
diff --git a/docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md b/docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md
index adbf521f1..dd11de7a8 100644
--- a/docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md
+++ b/docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md
@@ -26,9 +26,10 @@ There are multiple MOE backends inside TensorRT LLM. Here are the support matrix
 | Device                | Activation Type | MoE Weights Type | MoE Backend | Use Case                       |
 |---------------------- |-----------------|------------------|-------------|--------------------------------|
 | B200/GB200/B300/GB300 | MXFP8           | MXFP4            | TRTLLM      | Low Latency and Max Throughput |
+|         H200          | BF16            | MXFP4            | TRITON      | Low Latency and Max Throughput |
 
 The default moe backend is `CUTLASS`, so for the best possible perf, one must set the `moe_config.backend` explicitly to run the model.
-`CUTLASS` was better for max throughput at first but now we have optimized `TRTLLM` moe to be universally faster.
+For Blackwell, `CUTLASS` was better for max throughput at first but now we have optimized `TRTLLM` moe to be universally faster. For Hopper, Triton is the faster backend.
 
 ## Deployment Steps
 
@@ -139,11 +140,11 @@ These options provide control over TensorRT LLM's behavior and are set within th
 
 #### `max_batch_size`
 
-* **Description:** The maximum number of user requests that can be grouped into a single batch for processing. The actual max batch size that can be achieved depends on total sequence length (input + output).
+* **Description:** The maximum number of user requests that can be grouped into a single batch for processing. The actual max batch size that can be achieved depends on total sequence length (input + output) and GPU memory available for KV cache.
 
 #### `max_num_tokens`
 
-* **Description:** The maximum total number of tokens (across all requests) allowed inside a single scheduled batch.
+* **Description:** The maximum total number of tokens (across all requests) allowed inside a single scheduled batch. All input tokens (prefill phase) per request and 1 output token per decode request count towards this threshold.
 
 #### `max_seq_len`
 
@@ -368,14 +369,14 @@ $$
   * The combined rate at which the system processes both input (prompt) tokens and output (generated) tokens.
 
 $$
-\text{Total TPS} = \frac{\text{Num Input Tokens}+\text{Num Output Tokens}}{T_{last} - T_{first}}
+\text{Total TPS} = \frac{\text{Total input tokens}+\text{Total generated tokens}}{T_{last} - T_{first}}
 $$
 
 #### Tokens Per Second (TPS) or Output Token Throughput
   * how many output tokens the system generates each second.
 
 $$
-\text{TPS} = \frac{\text{Num Output Tokens}}{T_{last} - T_{first}}
+\text{TPS} = \frac{\text{Total generated tokens}}{T_{last} - T_{first}}
 $$
 
 ## Preconfigured Recipes
diff --git a/docs/sourc
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## 97f7e12588 - [fix] Fix perf regression caused by MoE autotuner when using DeepEPLowLatency (#6288)

- **Date**: 2025-07-28
- **Author**: Jinyang Yuan
- **Categories**: Throughput/Latency

### Optimization Techniques

- Operator fusion
- FP8 quantization
- Parallelism optimization
- MoE optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/custom_ops/torch_custom_ops.py | 29 ++++++++++++++++------
 .../_torch/modules/fused_moe/fused_moe_wide_ep.py  | 15 +++++++++++
 tensorrt_llm/_torch/utils.py                       |  4 +--
 3 files changed, 38 insertions(+), 10 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py b/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
index 60ef215fe..e9e0bb913 100644
--- a/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
+++ b/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
@@ -39,7 +39,6 @@ class MoERunner(TunableRunner):
         ep_rank: int,
         cluster_size: int,
         cluster_rank: int,
-        enable_alltoall: bool,
         use_deepseek_fp8_block_scale: bool,
         use_w4a8_group_scaling: bool,
         use_mxfp8_act_scaling: bool,
@@ -55,7 +54,8 @@ class MoERunner(TunableRunner):
         self.ep_rank = ep_rank
         self.cluster_size = cluster_size
         self.cluster_rank = cluster_rank
-        self.enable_alltoall = enable_alltoall
+        # The best tactic is estimated as if alltoall is disabled
+        self.enable_alltoall = False
         self.use_deepseek_fp8_block_scale = use_deepseek_fp8_block_scale
         self.use_w4a8_group_scaling = use_w4a8_group_scaling
         self.use_mxfp8_act_scaling = use_mxfp8_act_scaling
@@ -141,24 +141,37 @@ def fused_moe(
     use_mxfp8_act_scaling: bool = False,
     min_latency_mode: bool = False,
     tune_max_num_tokens: int = 8192,
+    tuner_num_tokens: Optional[int] = None,
+    tuner_top_k: Optional[int] = None,
 ) -> List[torch.Tensor]:
 
     tuner = AutoTuner.get()
     MoERunner.refine_tuning_config(tune_max_num_tokens)
 
+    # Only the non-alltoall case is considered for profiling in the warmup phase.
+    # Therefore, to get the correct tactics during the actual inference, the inputs to the tuner should be the same as when not using alltoall.
+    if enable_alltoall:
+        assert tuner_num_tokens is not None
+        assert tuner_top_k is not None
+        tuner_input = input[:tuner_num_tokens]
+    else:
+        assert tuner_num_tokens is None
+        assert tuner_top_k is None
+        tuner_input = input
+        tuner_top_k = token_selected_experts.size(1)
+
     # allocate workspace for profiling
     moe_runner = MoERunner(
         x_dtype=input.dtype,
         weight_dtype=fc1_expert_weights.dtype,
         output_dtype=output_dtype,
-        top_k=token_selected_experts.size(1),
+        top_k=tuner_top_k,
         tp_size=tp_size,
         tp_rank=tp_rank,
         ep_size=ep_size,
         ep_rank=ep_rank,
         cluster_size=cluster_size,
         cluster_rank=cluster_rank,
-        enable_alltoall=enable_alltoall,
         use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
         use_w4a8_group_scaling=use_w4a8_group_scaling,
         use_mxfp8_act_scaling=use_mxfp8_act_scaling,
@@ -170,8 +183,8 @@ def fused_moe(
         [moe_runner],
         MoERunner.tuning_config,
         [
-            input, fc1_expert_weights, fc1_expert_biases, fc2_expert_weights,
-            fc2_expert_biases
+            tuner_input, fc1_expert_weights, fc1_expert_biases,
+            fc2_expert_weights, fc2_expert_biases
         ],
         gemm_idx=1,
     )

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 985b79ca82 - [TRTLLM-8348][feat] Speed up concat k and copy k_nope in context phase using torch.compile (#8044)

- **Date**: 2025-09-29
- **Author**: Tailing Yuan
- **Categories**: Throughput/Latency

### Optimization Techniques

- Torch compilation/JIT optimization
- Custom CUDA kernel
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/modules/attention.py | 20 +++++++++++++++-----
 1 file changed, 15 insertions(+), 5 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/attention.py b/tensorrt_llm/_torch/modules/attention.py
index 35e5b221f..f736ce8ce 100644
--- a/tensorrt_llm/_torch/modules/attention.py
+++ b/tensorrt_llm/_torch/modules/attention.py
@@ -67,6 +67,16 @@ def extract_extra_attrs(layer_idx: str, attn_type: str):
     return metadata, attn_layer
 
 
+@torch.compile
+def compiled_copy_(dst, src):
+    dst.copy_(src)
+
+
+@torch.compile
+def compiled_cat(tensors, dim):
+    return torch.cat(tensors, dim)
+
+
 @torch.library.custom_op("trtllm::attn_custom_op_inplace",
                          mutates_args=("output", ))
 def attn_custom_op_inplace(
@@ -1063,8 +1073,8 @@ class MLA(nn.Module):
         )
 
         k = torch.empty_like(q).view(-1, self.num_heads, self.qk_head_dim)
-        k[..., :self.qk_nope_head_dim] = k_nope.view(-1, self.num_heads,
-                                                     self.qk_nope_head_dim)
+        compiled_copy_(k[..., :self.qk_nope_head_dim],
+                       k_nope.view(-1, self.num_heads, self.qk_nope_head_dim))
         if self.apply_rotary_emb:
             k[..., self.qk_nope_head_dim:] = k_pe.view(-1, 1,
                                                        self.qk_rope_head_dim)
@@ -1122,7 +1132,7 @@ class MLA(nn.Module):
         full_k_nope = full_k_nope.view(-1, self.num_heads,
                                        self.qk_nope_head_dim)
         full_k_pe = full_k_pe.view(-1, 1, self.qk_rope_head_dim)
-        full_k = torch.cat(
+        full_k = compiled_cat(
             (full_k_nope, full_k_pe.expand(-1, self.num_heads, -1)), dim=-1)
         full_k = full_k.view(-1, self.num_heads * self.qk_head_dim)
 
@@ -1217,7 +1227,7 @@ class MLA(nn.Module):
             chunked_k_nope = chunked_k_nope.view(-1, self.num_heads,
                                                  self.qk_nope_head_dim)
             chunked_k_pe = chunked_k_pe.view(-1, 1, self.qk_rope_head_dim)
-            chunked_k = torch.cat(
+            chunked_k = compiled_cat(
                 (chunked_k_nope, chunked_k_pe.expand(-1, self.num_heads, -1)),
                 dim=-1)
             chunked_k = chunked_k.view(-1, self.num_heads * self.qk_head_dim)
@@ -1275,7 +1285,7 @@ class MLA(nn.Module):
 
         k_nope = k_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
         k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
-        k = torch.cat((k_nope, k_pe.expand(-1, self.num_heads, -1)), dim=-1)
+        k = compiled_cat((k_nope, k_pe.expand(-1, self.num_heads, -1)), dim=-1)
         k = k.view(-1, self.num_heads * self.qk_head_dim)
 
         # copy q_lens to replace kv_lens_runtime

```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 9879400479 - [#10642][feat] AutoDeploy: optimized canonicalize_graph utilities [1/2] (#10675)

- **Date**: 2026-01-18
- **Author**: Lucas Liebenwein
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- KV cache optimization
- Parallelism optimization
- Multi-stream execution
- MoE optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../library/eliminate_redundant_transposes.py      |   4 -
 .../transform/library/fuse_causal_conv.py          |   6 +-
 .../transform/library/fuse_mamba_a_log.py          |   2 +-
 .../auto_deploy/transform/library/fused_moe.py     |  21 +-
 .../_torch/auto_deploy/transform/library/fusion.py |   9 +-
 .../transform/library/multi_stream_moe.py          |   8 +-
 .../auto_deploy/transform/library/sharding.py      |   4 +-
 tensorrt_llm/_torch/auto_deploy/utils/_graph.py    |  90 +++-
 .../unit/multigpu/test_ad_allreduce_strategies.py  |   3 +-
 .../transformations/library/test_ep_sharding.py    |   5 +-
 .../unit/singlegpu/custom_ops/test_multi_stream.py |   5 +-
 .../utils/test_delete_unused_submodules.py         | 478 +++++++++++++++++++++
 12 files changed, 590 insertions(+), 45 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/transform/library/eliminate_redundant_transposes.py b/tensorrt_llm/_torch/auto_deploy/transform/library/eliminate_redundant_transposes.py
index 4f260b2c4..aff4b4f52 100644
--- a/tensorrt_llm/_torch/auto_deploy/transform/library/eliminate_redundant_transposes.py
+++ b/tensorrt_llm/_torch/auto_deploy/transform/library/eliminate_redundant_transposes.py
@@ -110,10 +110,6 @@ class EliminateRedundantTransposes(BaseTransform):
                 original_input.replace_all_uses_with(new_contiguous_node)
                 new_contiguous_node.replace_input_with(new_contiguous_node, original_input)
 
-        # Clean up the graph
-        if nodes_to_eliminate:
-            gm.graph.eliminate_dead_code()
-
         info = TransformInfo(
             skipped=False,
             num_matches=len(nodes_to_eliminate),
diff --git a/tensorrt_llm/_torch/auto_deploy/transform/library/fuse_causal_conv.py b/tensorrt_llm/_torch/auto_deploy/transform/library/fuse_causal_conv.py
index 30f1959bb..bce8476e6 100644
--- a/tensorrt_llm/_torch/auto_deploy/transform/library/fuse_causal_conv.py
+++ b/tensorrt_llm/_torch/auto_deploy/transform/library/fuse_causal_conv.py
@@ -127,12 +127,10 @@ class FuseCausalConvActivation(BaseTransform):
                 graph.erase_node(activation_node)
                 graph.erase_node(conv_node)
 
-        gm.recompile()
-
         info = TransformInfo(
             skipped=False,
             num_matches=len(matches),
-            is_clean=False,
-            has_valid_shapes=False,
+            is_clean=len(matches) == 0,
+            has_valid_shapes=len(matches) == 0,
         )
         return gm, info
diff --git a/tensorrt_llm/_torch/auto_deploy/transform/library/fuse_mamba_a_log.py b/tensorrt_llm/_torch/auto_deploy/transform/library/fuse_mamba_a_log.py
index 977aeeabc..ea0dc3275 100644
--- a/tensorrt_llm/_torch/auto_deploy/transform/library/fuse_mamba_a_log.py
+++ b/tensorrt_llm/_torch/auto_deploy/transform/library/fuse_mamba_a_log.py
@@ -213,5 +213,5 @@ class FuseMambaALog(BaseTransform):
             skipped=False,
             num_matches=num_matches,
             is_clean=num_matches == 0,
-            has_valid_shapes=True,
+            has_valid_shapes=num_matches == 0,
         )
diff --git a/tensorrt_llm/_torch/auto_deploy/transform/library/fused_moe.py b/tensorrt_llm/_torch/auto_deploy/transform/library/fused_moe.py
index 819f80ed6..c7d949f20 100644
--- a/tensorrt_llm/_torch/auto_deploy/transform/library/fused_moe.py
+++ b/tensorrt_llm/_torch/auto_deploy/transform/library/fused_moe.py
@@ -10,6 +10,7 @@ from tensorrt_llm._torch.utils import ActivationType
 
 from ...models.factory import ModelFactory
 from ...shim.interface import CachedSequenceInterface
+from ...utils._graph import delete_all_unused_submodules, eliminate_dead_code
 from ...utils.cuda_mem_tracker import cuda_memory_tracker
 from ...utils.node_utils import bfs, extract_op_args, identify_regions_between_residuals, is_op
 f
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Multi-stream execution enables parallel execution of independent operations on the GPU. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 992781dc7b - [None][feat] update trtllm-gen nvfp4 kernels with better performance (#9510)

- **Date**: 2025-12-03
- **Author**: Perkz Zheng
- **Categories**: Kernel Optimization

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- FP8 quantization
- Integer quantization
- KV cache optimization
- Batching optimization
- Attention mechanism optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- Decode/generation phase

### Changed Files

```
.../batch_manager/kvCacheTransferManager.cpp       |    6 +-
 cpp/tensorrt_llm/kernels/fmhaDispatcher.cpp        |    7 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...eQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...eQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP1VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...1VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...rseP1VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP1VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...1VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...rseP1VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP1VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...1VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...rseP1VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP1VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...1VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...rseP1VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP1VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...1VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...rseP1VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP1VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...1VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...rseP1VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    4 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...eQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...eQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    4 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    4 +-
 .../trtllmGenKernels/fmha/cubin/kernelMetaInfo.h   | 3980 +++++++++++---------
 .../kernels/trtllmGenKernels/fmha/kernelParams.h   |    7 +-
 .../unfusedAttentionKernels_2_template.h           |   34 +-
 cpp/tensorrt_llm/kernels/xqaDispatcher.cpp         |    7 +-
 tensorrt_llm/_torch/pyexecutor/resource_manager.py |   26 +-
 1951 files changed, 6158 insertions(+), 5597 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/batch_manager/kvCacheTransferManager.cpp b/cpp/tensorrt_llm/batch_manager/kvCacheTransferManager.cpp
index 495f6a3ed..e138700e2 100644
--- a/cpp/tensorrt_llm/batch_manager/kvCacheTransferManager.cpp
+++ b/cpp/tensorrt_llm/batch_manager/kvCacheTransferManager.cpp
@@ -114,9 +114,13 @@ void KVCacheTransferManager::copyBlock(BlockPtr const& src, BlockPtr const& dst,
             auto srcPtr = computeBlockPointer(src, pools, poolIdx);
             auto dstPtr = computeBlockPointer(dst, pools, poolIdx);
 
+            // Does it contain block scales?
+            auto containsBlockScales = pools[poolIdx].containsBlockScales;
+
             // If no partial tokens or if the dataType is not supported for partial copy, copy entire block.
+            // Note that nvfp4 kv cache SFs use an interleaved layout, so we need to copy the entire block.
             if (numTokensToCopy <= 0 || srcPtr->getDataType() == nvinfer1::DataType::kINT4
-                || srcPtr->getDataType() == nvinfer1::DataType::kFP4)
+                || srcPtr->getDataType() == nvinfer1::DataType::kFP4 || containsBlockScales)
             {
                 // For partial copy not implemented with these data types,
                 // just do a full copy.
diff --git a/cpp/tensorrt_llm/kernels/fmhaDispatcher.cpp b/cpp/tensorrt_llm/kernels/fmhaDispatcher.cpp
index 11b3e1b0f..b46564d49 100644
--- a/cpp/tensorrt_llm/kernels/fmhaDispatcher.cpp
+++ b/cpp/tensorrt_llm/kernels/fmhaDispatcher.cpp
@@ -189,10 +189,11 @@ void FmhaDispatcher::run(MHARunnerParams runnerParams)
         tllmRunnerParams.attentionSinksPtr = runnerParams.attentionSinksPtr;
         tllmRunnerParams.cumSeqLensQPtr = reinterpret_cast<int const*>(runnerParams.cuQSeqLenPtr);
         tllmRunnerParams.cumSeqLensKvPtr = reinterpret_cast<int const*>(runnerParams.cuKvSeqLenPtr);
+        // Attention scales device pointers (only fp8 kernels need to load scales from the device memory).
         tllmRunnerParams.outputScalePtr = reinterpret_cast<float const*>(runnerParams.scaleBmm2Ptr);
-        // TRTLLM-GEN kernels always use the Log2 scale
-        tllmRunnerParams.scaleSoftmaxLog2Ptr
-            = reinterpret_cast<float const*>(runnerParams.scaleBmm1Ptr + kIdxScaleSoftmaxLog2Ptr);
+        tllmRunnerParams.scaleSoftmaxLog2Ptr = runnerParams.scaleBmm1Ptr
+            ? reinterpret_cast<float const*>(runnerParams.scaleBmm1Ptr + kIdxScaleSoftmaxLog2Ptr)
+            : nullptr;
         tllmRunnerParams.kvPageIdxPtr = reinterpret_cast<int const*>(kvPageIdxPtr);
         tllmRunnerParams.oSfScalePtr = runnerParams.oSfScalePtr;
         tllmRunnerParams.oPtr = runnerParams.outputPtr;
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128PersistentContext_cubin.cpp b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128PersistentContext_cubin.cpp
index 06
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 99b98f1374 - [TRTLLM-7440][fix] Split `fused_input_embed` to separate out host sync (#7280)

- **Date**: 2025-09-06
- **Author**: Chang Liu
- **Categories**: Fusion

### Optimization Techniques

- Torch compilation/JIT optimization
- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Pinned memory
- Speculative decoding
- MoE optimization
- GEMM optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/models/modeling_gemma3vl.py    |   8 +-
 tensorrt_llm/_torch/models/modeling_hyperclovax.py |   3 +-
 tensorrt_llm/_torch/models/modeling_llama.py       |   3 +-
 tensorrt_llm/_torch/models/modeling_llava_next.py  |   2 +-
 tensorrt_llm/_torch/models/modeling_mistral.py     |   5 +
 .../_torch/models/modeling_multimodal_utils.py     |  77 +++++---
 tensorrt_llm/_torch/models/modeling_phi4mm.py      |   1 +
 tensorrt_llm/_torch/models/modeling_qwen2vl.py     |   3 +-
 tensorrt_llm/_torch/models/modeling_vila.py        |   2 +-
 tensorrt_llm/_torch/modules/embedding.py           |  42 +++--
 tensorrt_llm/_torch/pyexecutor/model_engine.py     |  27 +++
 .../_torch/multimodal/test_fuse_input_embeds.py    | 205 +++++++++++++++++++++
 12 files changed, 331 insertions(+), 47 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_gemma3vl.py b/tensorrt_llm/_torch/models/modeling_gemma3vl.py
index 37770d2f0..15e93ad09 100644
--- a/tensorrt_llm/_torch/models/modeling_gemma3vl.py
+++ b/tensorrt_llm/_torch/models/modeling_gemma3vl.py
@@ -263,7 +263,9 @@ class Gemma3VLM(PreTrainedModel):
             embedding_layer=self.llm.model.embed_tokens,
             input_ids=input_ids,
             mm_embeds=mm_embeds,
-            mm_token_ids=self.image_token_ids)
+            mm_token_ids=self.image_token_ids,
+            **kwargs,
+        )
         logits = self.llm.forward(
             attn_metadata=attn_metadata,
             input_ids=input_ids,
@@ -284,3 +286,7 @@ class Gemma3VLM(PreTrainedModel):
                                                attn_metadata=attn_metadata)[-1]
             image_features = self.mm_projector(image_features)
         return image_features
+
+    @property
+    def mm_token_ids(self):
+        return self.image_token_ids
diff --git a/tensorrt_llm/_torch/models/modeling_hyperclovax.py b/tensorrt_llm/_torch/models/modeling_hyperclovax.py
index a05784b9d..975ccdb26 100644
--- a/tensorrt_llm/_torch/models/modeling_hyperclovax.py
+++ b/tensorrt_llm/_torch/models/modeling_hyperclovax.py
@@ -1052,7 +1052,8 @@ class HCXVisionForCausalLM(PreTrainedModel):
                 ]
 
         input_ids, input_embeds = fuse_input_embeds(self.llm.model.embed_tokens,
-                                                    input_ids, mm_embeds)
+                                                    input_ids, mm_embeds,
+                                                    **kwargs)
         output_prob = self.llm.forward(
             attn_metadata=attn_metadata,
             input_ids=input_ids,
diff --git a/tensorrt_llm/_torch/models/modeling_llama.py b/tensorrt_llm/_torch/models/modeling_llama.py
index 1169feb0a..5cb8f1b30 100644
--- a/tensorrt_llm/_torch/models/modeling_llama.py
+++ b/tensorrt_llm/_torch/models/modeling_llama.py
@@ -1280,7 +1280,8 @@ class Llama4ForConditionalGeneration(SpecDecOneEngineForCausalLM[Llama4Model,
                 ]
 
         input_ids, inputs_embeds = fuse_input_embeds(self.model.embed_tokens,
-                                                     input_ids, mm_embeds)
+                                                     input_ids, mm_embeds,
+                                                     **kwargs)
         return super().forward(attn_metadata,
                                input_ids,
                                position_ids,
diff --git a/tensorrt_llm/_torch/models/modeling_llava_next.py b/tensorrt_llm/_torch/models/modeling_llava_next.py
index 9356076dc..7158c23f5 100644
--- a/tensorrt_llm/_torch/models/modeling_llava_next.py
+++ b/tensorrt_llm/_torch/models/modeling_llava_next.py
@@ -474,7 +474,7 @@ class LlavaNextModel(PreTrainedModel):
                     for multimodal_param in multimodal_params
                 ]
         input_ids, inputs_embeds = fuse_input_embed
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Pinned (page-locked) memory enables faster host-to-device data transfers. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 9a070ed709 - [TRTLLM-10421][perf] Add fused cat+fp8_quantize CUDA kernel for DSA indexer (#11899)

- **Date**: 2026-03-10
- **Author**: Kaiyu Xie
- **Categories**: Kernel Optimization, Fusion, Quantization Optimization

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Pinned memory
- Triton kernel
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
cpp/tensorrt_llm/kernels/fusedCatFp8.cu            | 224 ++++++++++++++++
 cpp/tensorrt_llm/kernels/fusedCatFp8.h             |  63 +++++
 cpp/tensorrt_llm/thop/CMakeLists.txt               |   3 +-
 cpp/tensorrt_llm/thop/fusedCatFp8Op.cpp            |  87 ++++++
 .../_torch/attention_backend/sparse/dsa.py         |  14 +-
 tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py   |  10 +
 .../_torch/thop/serial/test_fused_cat_fp8.py       | 298 +++++++++++++++++++++
 7 files changed, 689 insertions(+), 10 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/fusedCatFp8.cu b/cpp/tensorrt_llm/kernels/fusedCatFp8.cu
new file mode 100644
index 000000000..98dacf65f
--- /dev/null
+++ b/cpp/tensorrt_llm/kernels/fusedCatFp8.cu
@@ -0,0 +1,224 @@
+/*
+ * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
+#include "fusedCatFp8.h"
+#include "tensorrt_llm/common/assert.h"
+#include "tensorrt_llm/common/config.h"
+#include "tensorrt_llm/common/cudaUtils.h"
+
+#include <cuda_bf16.h>
+#include <cuda_fp8.h>
+
+#include <cfloat>
+#include <cmath>
+#include <cstdint>
+
+TRTLLM_NAMESPACE_BEGIN
+
+namespace kernels
+{
+
+namespace
+{
+
+// Constants
+constexpr int HEAD_DIM = 128;       // Fixed for DSV3.2 indexer
+constexpr int WARP_SIZE = 32;       // One warp per row
+constexpr int ELEMS_PER_THREAD = 4; // 128 / 32 = 4 elements per thread
+constexpr int ROWS_PER_BLOCK = 8;   // Process 8 rows per block for occupancy
+constexpr float INV_FP8_E4M3_MAX = 1.0f / 448.0f;
+constexpr float MIN_AMAX = 1.0e-12f;
+
+/// Warp-wide max reduction
+__device__ __forceinline__ float warpReduceMax(float val)
+{
+    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
+    {
+        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
+    }
+    return val;
+}
+
+/// Helper union for vectorized BF16 loads (4 BF16 values = 8 bytes).
+union BF16x4
+{
+    int2 vec;
+    __nv_bfloat162 bf16x2[2];
+};
+
+/// Helper union for vectorized FP8 stores (4 FP8 values = 4 bytes).
+union FP8x4
+{
+    uint32_t u32;
+    __nv_fp8_e4m3 fp8[4];
+};
+
+/// Fused kernel: cat + FP8 quantization.
+///
+/// Grid: (ceil(M / ROWS_PER_BLOCK),)
+/// Block: (WARP_SIZE * ROWS_PER_BLOCK,)   i.e., (256,)
+///
+/// Each warp handles one row. Within a warp:
+///   - Thread t handles elements [4t, 4t+1, 4t+2, 4t+3] of the 128-dim row.
+///   - Loads from pe or nope based on element index (vectorized 8-byte loads).
+///   - FP8 quantizes with per-row scale (vectorized 4-byte stores).
+///
+/// Templated on UseUe8m0 to eliminate branch divergence.
+template <bool UseUe8m0>
+__global__ __launch_bounds__(WARP_SIZE* ROWS_PER_BLOCK) void fusedCatFp8Kernel(__nv_fp8_e4m3* __restrict__ fp8_out,
+    float* __restrict__ scale_out, __nv_bfloat16 const* __restrict__ pe, __nv_bfloat16 const* __restrict__ nope,
+    int32_t M, int32_t pe_dim, int32_t nope_dim, int32_t pe_row_stride, int32_t nope_row_stride)
+{
+    int warp_in_block = threadIdx.
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Pinned (page-locked) memory enables faster host-to-device data transfers. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 9a1750c8f9 - [TRTLLM-9493][noop] Refactor fusedMoeCommKernels to enable code sharing (#9922)

- **Date**: 2025-12-14
- **Author**: Balaram Buddharaju
- **Categories**: Kernel Optimization, Fusion

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- Integer quantization
- Parallelism optimization
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU

### Changed Files

```
cpp/tensorrt_llm/kernels/cudaAsyncOps.cuh       | 218 +++++++++++++++++
 cpp/tensorrt_llm/kernels/fusedMoeCommKernels.cu | 307 ++----------------------
 cpp/tensorrt_llm/kernels/fusedMoeCommKernels.h  |  16 +-
 cpp/tensorrt_llm/kernels/ll128Proto.cuh         | 163 +++++++++++++
 cpp/tensorrt_llm/kernels/moeCommKernelsCommon.h |  57 +++++
 5 files changed, 471 insertions(+), 290 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/cudaAsyncOps.cuh b/cpp/tensorrt_llm/kernels/cudaAsyncOps.cuh
new file mode 100644
index 000000000..0e1c87906
--- /dev/null
+++ b/cpp/tensorrt_llm/kernels/cudaAsyncOps.cuh
@@ -0,0 +1,218 @@
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
+#pragma once
+
+#include <cuda_runtime.h>
+#include <stdint.h>
+
+#include "tensorrt_llm/kernels/moeCommKernelsCommon.h"
+
+namespace tensorrt_llm
+{
+namespace kernels
+{
+
+// ============================================================================
+// Address Conversion Utilities
+// ============================================================================
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
+// ============================================================================
+// Memory Fence Operations
+// ============================================================================
+
+__device__ __forceinline__ void fence_release_sys()
+{
+    asm volatile("fence.release.sys;" : : : "memory");
+}
+
+// ============================================================================
+// Memory Barrier Operations (mbarrier)
+// ============================================================================
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
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 9ae705af1b - perf: Add fused q_norm/k_norm/RoPE for Qwen3. (#4482)

- **Date**: 2025-05-23
- **Author**: Bo Li
- **Categories**: Fusion

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- Batching optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu  | 270 +++++++++++++++++++++
 cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.h   |  44 ++++
 cpp/tensorrt_llm/thop/CMakeLists.txt               |   1 +
 cpp/tensorrt_llm/thop/fusedQKNormRopeOp.cpp        |  89 +++++++
 .../_torch/thop/test_fused_qk_norm_rope.py         | 162 +++++++++++++
 5 files changed, 566 insertions(+)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu b/cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu
new file mode 100644
index 000000000..9ce057ee7
--- /dev/null
+++ b/cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu
@@ -0,0 +1,270 @@
+/*
+ * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
+#include "fusedQKNormRopeKernel.h"
+#include "tensorrt_llm/common/cudaUtils.h"
+#include "tensorrt_llm/common/mathUtils.h"
+#include "tensorrt_llm/common/reduceKernelUtils.cuh"
+#include <cmath>
+#include <cuda_bf16.h>
+#include <cuda_fp16.h>
+#include <cuda_fp8.h>
+#include <cuda_runtime.h>
+
+namespace tensorrt_llm::common
+{
+// Specialization for packed_as used in this kernel.
+template <>
+struct packed_as<uint, 1>
+{
+    using type = uint;
+};
+
+template <>
+struct packed_as<uint, 2>
+{
+    using type = uint2;
+};
+
+template <>
+struct packed_as<uint, 4>
+{
+    using type = uint4;
+};
+} // namespace tensorrt_llm::common
+
+namespace tensorrt_llm::kernels
+{
+
+////////////////////////////////////////////////////////////////////////////////////////////////////
+
+// Perform per-head QK Norm and RoPE in a single kernel.
+// head_dim: the dimension of each head
+// interleave: interleave=!is_neox.
+template <int head_dim, bool interleave>
+__global__ void fusedQKNormRopeKernel(
+    __nv_bfloat16* qkv,            // Combined QKV tensor [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
+    int const num_heads_q,         // Number of query heads
+    int const num_heads_k,         // Number of key heads
+    int const num_heads_v,         // Number of value heads
+    float const eps,               // Epsilon for RMS normalization
+    __nv_bfloat16 const* q_weight, // RMSNorm weights for query
+    __nv_bfloat16 const* k_weight, // RMSNorm weights for key
+    float const base,              // Base for RoPE computation
+    int const* position_ids,       // Position IDs for RoPE
+    int const num_tokens           // Number of tokens
+)
+{
+    int const warpsPerBlock = blockDim.x / 32;
+    int const warpId = threadIdx.x / 32;
+    int const laneId = threadIdx.x % 32;
+
+    // Calculate global warp index to determine which head/token this warp processes
+    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;
+
+    // Total number of attention heads (Q and K)
+    int const total_qk_heads = num_heads_q + num_heads_k;
+
+    // Dete
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 9c4432f8a4 - [TRTLLM-7318][feat] MnnvlThroughput AlltoAll implementation. (#7499)

- **Date**: 2025-10-28
- **Author**: Bo Li
- **Categories**: Throughput/Latency

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Triton kernel
- Reduce synchronization overhead
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- NVIDIA Hopper (SM90) GPU
- Decode/generation phase
- Disaggregated serving

### Changed Files

```
ATTRIBUTIONS-CPP-aarch64.md                        |  235 +++
 ATTRIBUTIONS-CPP-x86_64.md                         |  235 +++
 cpp/tensorrt_llm/common/envUtils.cpp               |   39 +
 cpp/tensorrt_llm/common/envUtils.h                 |    9 +
 cpp/tensorrt_llm/common/vec_dtypes.cuh             | 1877 ++++++++++++++++++++
 .../communicationKernels/moeAlltoAllKernels.cu     |  961 ++++++++++
 .../communicationKernels/moeAlltoAllKernels.h      |  178 ++
 cpp/tensorrt_llm/nanobind/thop/bindings.cpp        |    7 +
 cpp/tensorrt_llm/pybind/thop/bindings.cpp          |    7 +
 cpp/tensorrt_llm/thop/CMakeLists.txt               |    1 +
 cpp/tensorrt_llm/thop/moeAlltoAllMeta.h            |   54 +
 cpp/tensorrt_llm/thop/moeAlltoAllOp.cpp            |  595 +++++++
 cpp/tensorrt_llm/thop/moeOp.cpp                    |   42 +-
 jenkins/license_cpp.json                           |    1 +
 tensorrt_llm/_torch/custom_ops/torch_custom_ops.py |  100 +-
 tensorrt_llm/_torch/distributed/__init__.py        |    2 +
 tensorrt_llm/_torch/distributed/moe_alltoall.py    |  228 +++
 .../_torch/modules/fused_moe/fused_moe_cutlass.py  |  151 +-
 .../_torch/modules/fused_moe/ops/moe_op_cutlass.py |   34 +-
 .../defs/accuracy/test_llm_api_pytorch.py          |   49 +-
 tests/unittest/_torch/multi_gpu/test_moe_a2a.py    |  824 +++++++++
 .../_torch/thop/parallel/test_custom_ops.py        |    5 +
 22 files changed, 5501 insertions(+), 133 deletions(-)
```

### Diff Preview

```diff
diff --git a/ATTRIBUTIONS-CPP-aarch64.md b/ATTRIBUTIONS-CPP-aarch64.md
index 311c0c520..9ed9b5338 100755
--- a/ATTRIBUTIONS-CPP-aarch64.md
+++ b/ATTRIBUTIONS-CPP-aarch64.md
@@ -14888,3 +14888,238 @@ Chen, Tianqi
    limitations under the License.
 
 ```
+
+## flashinfer
+
+### License Text
+```
+                                 Apache License
+                           Version 2.0, January 2004
+                        http://www.apache.org/licenses/
+
+   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
+
+   1. Definitions.
+
+      "License" shall mean the terms and conditions for use, reproduction,
+      and distribution as defined by Sections 1 through 9 of this document.
+
+      "Licensor" shall mean the copyright owner or entity authorized by
+      the copyright owner that is granting the License.
+
+      "Legal Entity" shall mean the union of the acting entity and all
+      other entities that control, are controlled by, or are under common
+      control with that entity. For the purposes of this definition,
+      "control" means (i) the power, direct or indirect, to cause the
+      direction or management of such entity, whether by contract or
+      otherwise, or (ii) ownership of fifty percent (50%) or more of the
+      outstanding shares, or (iii) beneficial ownership of such entity.
+
+      "You" (or "Your") shall mean an individual or Legal Entity
+      exercising permissions granted by this License.
+
+      "Source" form shall mean the preferred form for making modifications,
+      including but not limited to software source code, documentation
+      source, and configuration files.
+
+      "Object" form shall mean any form resulting from mechanical
+      transformation or translation of a Source form, including but
+      not limited to compiled object code, generated documentation,
+      and conversions to other media types.
+
+      "Work" shall mean the work of authorship, whether in Source or
+      Object form, made available under the License, as indicated by a
+      copyright notice that is included in or attached to the work
+      (an example is provided in the Appendix below).
+
+      "Derivative Works" shall mean any work, whether in Source or Object
+      form, that is based on (or derived from) the Work and for which the
+      editorial revisions, annotations, elaborations, or other modifications
+      represent, as a whole, an original work of authorship. For the purposes
+      of this License, Derivative Works shall not include works that remain
+      separable from, or merely link (or bind by name) to the interfaces of,
+      the Work and Derivative Works thereof.
+
+      "Contribution" shall mean any work of authorship, including
+      the original version of the Work and any modifications or additions
+      to that Work or Derivative Works thereof, that is intentionally
+      submitted to Licensor for inclusion in the Work by the copyright owner
+      or by an individua
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 9c4b8f66b4 - feat: Integration of Fused QKNorm+RoPE. (#4611)

- **Date**: 2025-05-28
- **Author**: Bo Li
- **Categories**: Fusion

### Optimization Techniques

- Operator fusion
- FP8 quantization
- Batching optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/models/modeling_gemma3.py      |  2 +-
 tensorrt_llm/_torch/models/modeling_llama.py       |  4 +-
 tensorrt_llm/_torch/models/modeling_qwen3.py       | 28 ++++++++-
 tensorrt_llm/_torch/modules/attention.py           | 66 ++++++++++++++--------
 .../_torch/pyexecutor/cuda_graph_runner.py         |  4 +-
 5 files changed, 72 insertions(+), 32 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_gemma3.py b/tensorrt_llm/_torch/models/modeling_gemma3.py
index 1ba1fa029..1f14e2d27 100644
--- a/tensorrt_llm/_torch/models/modeling_gemma3.py
+++ b/tensorrt_llm/_torch/models/modeling_gemma3.py
@@ -53,11 +53,11 @@ class Gemma3Attention(Attention):
             max_position_embeddings=config.max_position_embeddings,
             bias=False,
             pos_embd_params=pos_embd_params,
+            qk_norm_type=QkNormType.pre_rope,
             layer_idx=layer_idx,
             dtype=config.torch_dtype,
             dense_bias=False,
             config=model_config,
-            qk_norm_type=QkNormType.pre_rope,
             q_scaling=q_scaling,
         )
         self.q_norm = RMSNorm(hidden_size=config.head_dim,
diff --git a/tensorrt_llm/_torch/models/modeling_llama.py b/tensorrt_llm/_torch/models/modeling_llama.py
index 79443fd7c..36163e6b1 100644
--- a/tensorrt_llm/_torch/models/modeling_llama.py
+++ b/tensorrt_llm/_torch/models/modeling_llama.py
@@ -73,11 +73,11 @@ class Llama4Attention(Attention):
             max_position_embeddings=config.max_position_embeddings,
             bias=config.attention_bias,
             pos_embd_params=pos_embd_params,
+            qk_norm_type=QkNormType.post_rope
+            if use_qk_norm else QkNormType.none,
             layer_idx=layer_idx,
             dtype=config.torch_dtype,
             config=model_config,
-            qk_norm_type=QkNormType.post_rope
-            if use_qk_norm else QkNormType.none,
             attention_chunk_size=attention_chunk_size,
         )
 
diff --git a/tensorrt_llm/_torch/models/modeling_qwen3.py b/tensorrt_llm/_torch/models/modeling_qwen3.py
index d614100f3..cfd1b2e9f 100644
--- a/tensorrt_llm/_torch/models/modeling_qwen3.py
+++ b/tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -26,8 +26,10 @@ class Qwen3Attention(Attention):
         self,
         model_config: ModelConfig[Qwen3Config],
         layer_idx: Optional[int] = None,
+        fuse_qk_norm_rope: bool = True,
     ):
         config = model_config.pretrained_config
+
         if getattr(config, "rope_scaling", None) is not None:
             pos_embd_params = PositionalEmbeddingParams(
                 type=PositionEmbeddingType.from_string(
@@ -40,20 +42,27 @@ class Qwen3Attention(Attention):
                 rope=RopeParams.from_config(config),
             )
 
+        self.fuse_qk_norm_rope = fuse_qk_norm_rope
+
         super().__init__(
             hidden_size=config.hidden_size,
             num_attention_heads=config.num_attention_heads,
             num_key_value_heads=config.num_key_value_heads,
             max_position_embeddings=config.max_position_embeddings,
             bias=config.attention_bias,
-            pos_embd_params=pos_embd_params,
+            pos_embd_params=pos_embd_params
+            if not self.fuse_qk_norm_rope else None,
+            qk_norm_type=QkNormType.pre_rope,
             layer_idx=layer_idx,
             dtype
```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 9cb5410067 - [https://nvbugs/5454559][fix] handle bias term in fuse_gate_mlp (#7449)

- **Date**: 2025-09-09
- **Author**: Linda
- **Categories**: Fusion

### Optimization Techniques

- Operator fusion

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/models/modeling_utils.py | 5 +++++
 1 file changed, 5 insertions(+)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/models/modeling_utils.py b/tensorrt_llm/models/modeling_utils.py
index dcc375320..7c49855d0 100644
--- a/tensorrt_llm/models/modeling_utils.py
+++ b/tensorrt_llm/models/modeling_utils.py
@@ -1233,6 +1233,11 @@ def fuse_gate_mlp(
                     mlp.gate.activation_scaling_factor.raw_value,
                     mlp.fc.activation_scaling_factor.raw_value,
                 )
+
+                if mlp.bias:
+                    fused_layer.fused_fc.bias.value = np.concatenate(
+                        [mlp.gate.bias.raw_value, mlp.fc.bias.raw_value],
+                        axis=0)
             elif layer_quant_algo is None:
                 fused_layer.fused_fc.weight.value = np.concatenate(
                     [

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 9ee33605bb - [TRTLLM-6019] feat: Remove cutlass min latency code from AutoTuner. (#5394)

- **Date**: 2025-06-26
- **Author**: Yukun He
- **Categories**: Throughput/Latency

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- FP8 quantization
- KV cache optimization
- MoE optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/custom_ops/torch_custom_ops.py | 46 ++++++----------------
 1 file changed, 12 insertions(+), 34 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py b/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
index 9b8b2f059..e94ee6df4 100644
--- a/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
+++ b/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
@@ -22,12 +22,9 @@ def bmm_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
 class MoERunner(TunableRunner):
     # avoid overhead of creating a new runner in forward pass
     runner_dict = dict()
-    # TODO: only profile for min_latency_mode = False due to the error in the moe_kernels
     tuning_config = TuningConfig(dynamic_tensor_specs=(
         DynamicTensorSpec(0, 0, get_last_power_of_2_num_tokens_buckets(8192),
-                          lambda x: min(last_positive_power_of_2(x), 8192)),
-        DynamicTensorSpec(3, 0, (0, ), lambda x: x),
-    ))
+                          lambda x: min(last_positive_power_of_2(x), 8192)), ))
 
     def __init__(
         self,
@@ -44,6 +41,7 @@ class MoERunner(TunableRunner):
         enable_alltoall: bool,
         use_deepseek_fp8_block_scale: bool,
         use_w4a8_group_scaling: bool,
+        min_latency_mode: bool,
     ):
         self.x_dtype = x_dtype
         self.weight_dtype = weight_dtype
@@ -58,7 +56,7 @@ class MoERunner(TunableRunner):
         self.enable_alltoall = enable_alltoall
         self.use_deepseek_fp8_block_scale = use_deepseek_fp8_block_scale
         self.use_w4a8_group_scaling = use_w4a8_group_scaling
-
+        self.min_latency_mode = min_latency_mode
         instance_key = (x_dtype, weight_dtype, output_dtype,
                         use_deepseek_fp8_block_scale, use_w4a8_group_scaling)
 
@@ -74,22 +72,7 @@ class MoERunner(TunableRunner):
         inputs: List[torch.Tensor],
         profile: OptimizationProfile,
     ) -> List[int]:
-        x, _, _, min_latency_mode_tensor = inputs
-        min_latency_mode = min_latency_mode_tensor.size(0) == 1
-        m = x.shape[0]
-
-        # Only profile m <= 128 for min latency mode = True
-        # Profile all valid buckets for min latency mode = False
-        # TODO: min_latency_mode = True will cause the following error:
-        # Cannot profile configuration 4: Cutlass GEMM Tactic
-        # [TensorRT-LLM][ERROR] Assertion failed: Failed to initialize cutlass TMA WS grouped gemm.
-        # Should be fixed in the moe_kernels in the future.
-        invalid = (m > 128 and
-                   min_latency_mode) or (m <= 128 and min_latency_mode and
-                                         (not self.weight_dtype == torch.int64))
-
-        return [] if invalid else list(
-            range(self.fused_moe_runner.get_tactic_num()))
+        return range(self.fused_moe_runner.get_tactic_num())
 
     def forward(
         self,
@@ -98,8 +81,7 @@ class MoERunner(TunableRunner):
         tactic: int = -1,
         do_preparation: bool = False,
     ):
-        x, fc1_expert_weights, fc2_expert_weights, min_latency_mode_tensor = inputs
-        min
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## a030a898d1 - perf: Fuse gemm setup function for SM90/SM100 MOE plugin path (#4146)

- **Date**: 2025-05-21
- **Author**: djns99
- **Categories**: Fusion

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Reduce synchronization overhead
- MoE optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU

### Changed Files

```
...orrt_llm_internal_cutlass_kernels_static.tar.xz |   4 +-
 .../aarch64-linux-gnu/version.txt                  |   4 +-
 .../include/moe_gemm_kernels.h                     |   2 +-
 .../internal_cutlass_kernels/include/moe_kernels.h | 163 ++++++++++++++++-----
 ...orrt_llm_internal_cutlass_kernels_static.tar.xz |   4 +-
 .../x86_64-linux-gnu/version.txt                   |   4 +-
 cpp/tensorrt_llm/thop/moeOp.cpp                    |   7 +-
 .../unit_tests/kernels/mixtureOfExpertsTest.cu     |  71 +++++++--
 tensorrt_llm/logger.py                             |   2 +
 9 files changed, 198 insertions(+), 63 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/aarch64-linux-gnu/tensorrt_llm_internal_cutlass_kernels_static.tar.xz b/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/aarch64-linux-gnu/tensorrt_llm_internal_cutlass_kernels_static.tar.xz
index aa53b594f..93d69e6bf 100644
--- a/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/aarch64-linux-gnu/tensorrt_llm_internal_cutlass_kernels_static.tar.xz
+++ b/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/aarch64-linux-gnu/tensorrt_llm_internal_cutlass_kernels_static.tar.xz
@@ -1,3 +1,3 @@
 version https://git-lfs.github.com/spec/v1
-oid sha256:090be002758e4fb864e16ec25c0db3f8eb562a0033e60a156bbbfd6bce67a5a1
-size 63577888
+oid sha256:626761456288897fb021c78145fa4e56140890be62055db16491353f319cd455
+size 63672012
diff --git a/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/aarch64-linux-gnu/version.txt b/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/aarch64-linux-gnu/version.txt
index ed1277dce..f4e8a3ed2 100644
--- a/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/aarch64-linux-gnu/version.txt
+++ b/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/aarch64-linux-gnu/version.txt
@@ -1,2 +1,2 @@
-aff0f8e617f6ca2f95d121ab9cf0ab17c4e8077cf9e8896bf153d3942a4a50df  libtensorrt_llm_internal_cutlass_kernels_static.a
-commit d61e7684bc095c8ff5ec540363949bd1f491c960
+4a23fc1883a3d35ae8b47c6bac3d8e7e85adc0e5615a9afd46acf8b84c8d62a8  libtensorrt_llm_internal_cutlass_kernels_static.a
+commit 0971a156bd13fe781c1c16659c94bc69e0ca77e1
diff --git a/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/include/moe_gemm_kernels.h b/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/include/moe_gemm_kernels.h
index 7e9f4eea0..ef6cc04b2 100644
--- a/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/include/moe_gemm_kernels.h
+++ b/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/include/moe_gemm_kernels.h
@@ -216,7 +216,7 @@ struct TmaWarpSpecializedGroupedGemmInput
         constexpr static int group_size = 128; // Unused, hard-coded to 128
         bool enabled = false;
         using SFA = __nv_bfloat16;
-        using SFB = __nv_bfloat16;
+        using SFB = __nv_bfloat16; // Unused
         using ProblemShapeInt = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
         using LayoutSFA = typename cutlass::layout::ColumnMajor;
         using LayoutSFB = typename cutlass::layout::ColumnMajor;        // Unused
diff --git a/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/include/moe_kernels.h b/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/include/moe_kernels.h
index 301ec066a..6f732c617 100644
--- a/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/include/moe_kernels.h
+++ b/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/include/moe_kernels.h
@@ -26,9 +26,12 @@
 #include <cuda_fp4.h>
 #endif
 #include <NvInferRuntime.h>
+#include <array>
 #include <cuda_runtime_api.h>
+#include <map>
 #include <optional>
 #include <random>
+#include <utility>
 
 namespace tensorrt_llm::kerne
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## a0d489a8d5 - [TRTLLM-7728][perf] improve batched sampling perf for contiguous batches (#7908)

- **Date**: 2025-09-29
- **Author**: mpikulski
- **Categories**: General Performance

### Optimization Techniques

- Async/stream-based execution
- Pinned memory

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/sampler.py | 15 ++++++++++++---
 1 file changed, 12 insertions(+), 3 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index fcc8d623c..85a3d71c8 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -396,6 +396,7 @@ def _group_requests_by_sampling_strategy(
         requests: Iterable[LlmRequest],
         *,
         pin_memory: bool = False) -> dict[Strategy, torch.Tensor]:
+    # NB: Client code relies on request indices in returned torch.Tensor being sorted.
     strategy_dict: dict[Strategy, list[int]] = defaultdict(list)
     for req_index, req in enumerate(requests):
         strategy_dict[_request_strategy(req)].append(req_index)
@@ -1372,12 +1373,20 @@ class TorchSampler(Sampler):
                     len(speculation_group_indices), dtype=torch.int32)
 
             group_logits_cuda_indices = logits_cuda_indexer[group_req_indices]
-            if group_logits_cuda_indices.numel() != logits_cuda.size(0):
+            # NB: Assuming that group_req_indices are sorted
+            group_req_1st_index, group_req_last_index = group_req_indices[
+                0], group_req_indices[-1]
+            if group_req_last_index - group_req_1st_index + 1 == len(
+                    group_req_indices):
+                # Avoid data movement if indices are contiguous
+                group_logits_cuda = logits_cuda[
+                    req_offsets[group_req_1st_index]:(
+                        req_offsets[group_req_last_index] +
+                        req_num_steps[group_req_last_index])]
+            else:
                 group_logits_cuda_indices_cuda = group_logits_cuda_indices.to(
                     device=logits_cuda.device, non_blocking=True)
                 group_logits_cuda = logits_cuda[group_logits_cuda_indices_cuda]
-            else:
-                group_logits_cuda = logits_cuda
 
             # Indexer for accessing tokens in 'group_logits_cuda' (and 'group_next_tokens_cuda')
             # corresponding to the requests in 'group_req_indices'.

```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. Pinned (page-locked) memory enables faster host-to-device data transfers. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

