# Performance Optimization Analysis - Part 7

Commits 175 to 203 of 283

---

## a1e03af0f4 - [TRTLLM-7346][fix] Improve performance of PyTorchModelEngine._get_lora_params_from_requests (#7033)

- **Date**: 2025-08-25
- **Author**: amitz-nv
- **Categories**: General Performance

### Optimization Techniques

- Batching optimization
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/peft/lora/layer.py             |   2 -
 tensorrt_llm/_torch/pyexecutor/llm_request.py      |   4 +-
 tensorrt_llm/_torch/pyexecutor/model_engine.py     | 102 ++++++++-------------
 .../test_lora_attention_pytorch_flow_vs_trt.py     |   8 --
 4 files changed, 41 insertions(+), 75 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/peft/lora/layer.py b/tensorrt_llm/_torch/peft/lora/layer.py
index fb9846141..2c8bc5e2f 100644
--- a/tensorrt_llm/_torch/peft/lora/layer.py
+++ b/tensorrt_llm/_torch/peft/lora/layer.py
@@ -107,8 +107,6 @@ class LoraLayer(torch.nn.Module):
                 module_idx = int(module_idx)
                 if module_idx in lora_params[layer_idx]:
                     active_lora_module_ids.append(module_idx)
-                    # TODO (dafrimi): needs to pass this is_dora arg
-                    lora_params[layer_idx][module_idx]['is_dora']
                     lora_ranks.append(
                         lora_params[layer_idx][module_idx]['adapter_size'])
                     lora_weight_pointers.append(
diff --git a/tensorrt_llm/_torch/pyexecutor/llm_request.py b/tensorrt_llm/_torch/pyexecutor/llm_request.py
index 96af64fce..fb0670e37 100644
--- a/tensorrt_llm/_torch/pyexecutor/llm_request.py
+++ b/tensorrt_llm/_torch/pyexecutor/llm_request.py
@@ -339,7 +339,9 @@ class LlmRequest(tensorrt_llm.bindings.internal.batch_manager.LlmRequest):
         self.py_decoding_iter = 0
         self.is_attention_dp_dummy = False
         self.is_cuda_graph_dummy = False
-        self.py_lora_task_layer_module_configs = None
+        self.py_lora_task_layer_module_configs: list[
+            tensorrt_llm.bindings.internal.runtime.
+            TaskLayerModuleConfig] | None = None
 
         self.py_return_log_probs = return_log_probs
         self.py_return_context_logits = return_context_logits
diff --git a/tensorrt_llm/_torch/pyexecutor/model_engine.py b/tensorrt_llm/_torch/pyexecutor/model_engine.py
index 1b3fbfbfc..dbc6bb9a4 100644
--- a/tensorrt_llm/_torch/pyexecutor/model_engine.py
+++ b/tensorrt_llm/_torch/pyexecutor/model_engine.py
@@ -2089,7 +2089,6 @@ class PyTorchModelEngine(ModelEngine):
                 module_id: dict
                 {
                     adapter_size: torch tensor: int
-                    is_dora: torch tensor: bool
                     weight_pointers: torch tensor: int64
                 }
             }
@@ -2108,88 +2107,63 @@ class PyTorchModelEngine(ModelEngine):
             for module in request.py_lora_task_layer_module_configs:
                 module_id = module.module_id
                 layer_id = module.layer_id
-                adapter_size = module.adapter_size
-                is_dora = module.scaling_vec_pointer == 0
-                weights_in_pointer = module.weights_in_pointer
-                weights_out_pointer = module.weights_out_pointer
-                scaling_vec_pointer = module.scaling_vec_pointer
-                if weights_in_pointer is None:
-                    weights_in_pointer = 0
-                if weights_out_pointer is None:
-                    weights_out_pointer = 0
-                if scaling_vec_pointer is None:
-                    scaling_vec_pointer = 0
 
                 if layer_id not in lora_params:
                     lora_params[layer_id] = {}
 
```

### Analysis Summary

Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## a20ab5cbdb - [https://nvbugs/5381276][fix] fix warning for fused_a_gemm (#6402)

- **Date**: 2025-08-01
- **Author**: yunruis
- **Categories**: Fusion

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu      | 16 ++--------------
 1 file changed, 2 insertions(+), 14 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu b/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu
index a804eb743..43c21bb1a 100644
--- a/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu
+++ b/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu
@@ -35,7 +35,6 @@ namespace tensorrt_llm::kernels::dsv3MinLatencyKernels
 __device__ void hmma_16_8_16_f32acc_bf16ab(
     float (&d_reg)[4], const bf16_t (&a_reg)[8], const bf16_t (&b_reg)[4], float const (&c_reg)[4])
 {
-#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
     uint32_t a0 = *reinterpret_cast<uint32_t const*>(a_reg + 0);
     uint32_t a1 = *reinterpret_cast<uint32_t const*>(a_reg + 2);
     uint32_t a2 = *reinterpret_cast<uint32_t const*>(a_reg + 4);
@@ -51,7 +50,6 @@ __device__ void hmma_16_8_16_f32acc_bf16ab(
         : "=f"(d_reg[0]), "=f"(d_reg[1]), "=f"(d_reg[2]), "=f"(d_reg[3])
         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d_reg[0]), "f"(d_reg[1]), "f"(d_reg[2]),
         "f"(d_reg[3]));
-#endif
 }
 
 extern "C"
@@ -72,11 +70,9 @@ __device__ void ldgsts_128(void const* gPtr, void* sPtr, uint32_t pred)
 
 __device__ void ldsm_x4(void* smem_ptr, uint32_t* reg_ptr)
 {
-#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
     asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                  : "=r"(reg_ptr[0]), "=r"(reg_ptr[1]), "=r"(reg_ptr[2]), "=r"(reg_ptr[3])
                  : "r"(__nvvm_get_smem_pointer(smem_ptr)));
-#endif
 }
 
 template <class Type>
@@ -90,20 +86,18 @@ __device__ int apply_swizzle_343_on_elem_row_col(int row_idx_, int col_idx_)
     return *reinterpret_cast<int*>(&col_idx);
 }
 
+#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
 __device__ void initialize_barrier(uint64_t* smem_barrier, // 64 bits user-manged barrier in smem
     int thread_count = 1)                                  // Thread count expected to arrive/wait on this barrier
 {
-#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
     uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
     asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_ptr), "r"(thread_count));
-#endif
 }
 
 // Barrier wait
 __device__ void wait_barrier(uint64_t* smem_barrier, // 64 bits user-manged barrier in smem
     int phase_bit)                                   // Current phase bit the barrier waiting to flip
 {
-#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
     uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
     asm volatile(
         "{\n"
@@ -115,12 +109,10 @@ __device__ void wait_barrier(uint64_t* smem_barrier, // 64 bits user-manged barr
         "DONE:\n"
         "}\n" ::"r"(smem_int_ptr),
         "r"(phase_bit));
-#endif
 }
 
 __device__ bool try_wait_barrier(uint64_t* smem_ptr, int phase_bit)
 {
-#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
     uint32_t wait_complete;
     uint32_t smem_int_ptr = __nvvm_get_smem_pointer(
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## a28def9020 - [TRTLLM-9687][feat] Improve are_stop_words performance (#11196)

- **Date**: 2026-03-02
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

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/sampler.py          | 1209 +++++++++++++++-----
 .../unittest/_torch/sampler/test_torch_sampler.py  |  309 ++++-
 .../test_draft_token_tree_verification.py          |    5 +-
 3 files changed, 1182 insertions(+), 341 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index 13e231fa7..197680cf1 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -19,7 +19,6 @@ from collections import defaultdict
 from collections.abc import Iterable
 from concurrent import futures
 from dataclasses import dataclass
-from functools import cached_property
 from itertools import repeat
 from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeAlias, TypeVar, cast
 
@@ -1123,6 +1122,8 @@ class AsyncWorkerMixin:
 
 class TorchSampler(Sampler[SampleStateTorch], AsyncWorkerMixin):
     DEFAULT_MAX_TOPK_LOGPROBS = 20
+    DEFAULT_MAX_STOP_WORD_LENGTH = 20
+    DEFAULT_MAX_STOP_WORDS = 10
 
     SampleState = SampleStateTorch
 
@@ -1136,6 +1137,852 @@ class TorchSampler(Sampler[SampleStateTorch], AsyncWorkerMixin):
     def is_generation_model(self) -> bool:
         return True
 
+    class FinishReasonsHandler:
+        _EMPTY_STOP_WORD_TOKEN_ID: int = -2
+        _PAD_STOP_WORD_TOKEN_ID: int = -1
+
+        @dataclass(kw_only=True)
+        class _FinishReasonsStore:
+            """Auxiliary data structures used for finish reasons handling."""
+
+            # Per-request dynamic data
+            finish_reasons_cuda: torch.Tensor
+            """Shape: [max_tokens, batch_size, beam_width]
+            Usage: Stores the determined finish reasons for all sampled tokens
+            for each request. Some (draft) tokens and corresponding
+            finish reasons might still be discarded."""
+
+            # Per-request static data
+            max_lengths_cuda: torch.Tensor
+            """Shape: [batch_size]
+            Usage: Stores the maximum sequence lengths for each request"""
+            end_ids_cuda: torch.Tensor
+            """Shape: batch_size
+            Usage: Stores the end ids for each request"""
+            stop_words_cuda: torch.Tensor
+            """Shape: [max_num_stop_words, max_stop_word_length, batch_size]
+            Usage: Stores the stop words for each request as a padded tensor."""
+            past_tokens_cuda: torch.Tensor
+            """Shape: [max_stop_word_length,batch_size, beam_width]
+            Usage: Stores the last max_stop_word_length tokens for each beam."""
+            max_stop_word_lengths_host: torch.Tensor
+            """Shape: [batch_size]
+            Usage: Stores the size of the longest stop word for each request."""
+            num_accepted_draft_tokens_host: torch.Tensor
+            """Shape: [batch_size]
+            Usage: Stores the number of accepted tokens for each request."""
+
+        def __init__(
+            self,
+            *,
+            max_stop_word_length: int,
+            max_num_stop_words: int,
+            max_num_sequences: int,
+            max_beam_width: int,
+            max_tokens: int,
+            max_seq_len: int,
+        ):
+            self._update_sizes(
+           
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. Pinned (page-locked) memory enables faster host-to-device data transfers. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## a5768ce316 - [https://nvbugs/5820922][perf] Improve TorchSampler performance by reducing host overhead (#11315)

- **Date**: 2026-02-11
- **Author**: Stefan Niebler
- **Categories**: Host-side Optimization

### Optimization Techniques

- KV cache optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/sampler.py | 17 ++++++++++++-----
 1 file changed, 12 insertions(+), 5 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index 69c97e9e3..509b42431 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -404,14 +404,21 @@ def _request_get_sampling_params(request: LlmRequest) -> UtilsSamplingParams:
     )
 
 
+def _request_sampling_params_cachable(params: UtilsSamplingParams) -> bool:
+    return not params.use_beam_search
+
+
 def _request_strategy(request: LlmRequest, *, vocab_size: int) -> Strategy:
-    """Resolve the sampling strategy for a request.
+    # We try to cache the resolved strategy on the request object, as it's not cheap enough to
+    # resolve it on every iteration.
+    if hasattr(request, "py_sampling_strategy"):
+        return request.py_sampling_strategy
 
-    Note: Callers inside _group_requests_by_strategy_key benefit from store.strategies
-    caching, which ensures this function is called at most once per request per slot.
-    """
     params = _request_get_sampling_params(request)
-    return resolve_sampling_strategy(params, vocab_size=vocab_size)
+    sampling_strategy = resolve_sampling_strategy(params, vocab_size=vocab_size)
+    if _request_sampling_params_cachable(params):
+        request.py_sampling_strategy = sampling_strategy
+    return sampling_strategy
 
 
 class _CachingRequestGrouper(Generic[GenericStrategyKeyType]):

```

### Analysis Summary

This commit applies performance optimization techniques to improve inference efficiency. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## a5a37227d6 - [None][feat] Fused kernels (qknormrope + moe routing) and two-model MTP support for glm4moe (#9852)

- **Date**: 2025-12-13
- **Author**: nvxuanyuc
- **Categories**: Kernel Optimization, Fusion

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Speculative decoding
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- Prefill phase
- Decode/generation phase

### Changed Files

```
cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu  |  53 ++++--
 cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.h   |   1 +
 cpp/tensorrt_llm/thop/fusedQKNormRopeOp.cpp        |  10 +-
 tensorrt_llm/_torch/models/modeling_auto.py        |   4 +-
 tensorrt_llm/_torch/models/modeling_glm.py         | 185 ++++++++++++++++-----
 tensorrt_llm/_torch/models/modeling_speculative.py |  36 +++-
 tensorrt_llm/_torch/modules/qk_norm_attention.py   |   7 +-
 .../defs/accuracy/references/gsm8k.yaml            |   2 +
 .../defs/accuracy/test_llm_api_pytorch.py          |  40 ++++-
 .../test_lists/qa/llm_function_core.txt            |   3 +
 .../thop/parallel/test_fused_qk_norm_rope.py       |  30 ++--
 11 files changed, 289 insertions(+), 82 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu b/cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu
index 73326af8c..a73ea7927 100644
--- a/cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu
+++ b/cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu
@@ -66,6 +66,7 @@ __global__ void fusedQKNormRopeKernel(
     int const num_heads_q,         // Number of query heads
     int const num_heads_k,         // Number of key heads
     int const num_heads_v,         // Number of value heads
+    int const rotary_dim,          // Dimension for RoPE
     float const eps,               // Epsilon for RMS normalization
     __nv_bfloat16 const* q_weight, // RMSNorm weights for query
     __nv_bfloat16 const* k_weight, // RMSNorm weights for key
@@ -184,7 +185,7 @@ __global__ void fusedQKNormRopeKernel(
 
             int dim_idx = laneId * numElemsPerThread + i;
             int half_dim = dim_idx / 2;
-            float freq = powf(base, -2.0f * half_dim / static_cast<float>(head_dim));
+            float freq = powf(base, -2.0f * half_dim / static_cast<float>(rotary_dim));
 
             if (factor != 1.0f)
             {
@@ -212,19 +213,20 @@ __global__ void fusedQKNormRopeKernel(
     {
         // Before data exchange with in warp, we need to sync.
         __syncwarp();
+        int pairOffset = (rotary_dim / 2) / numElemsPerThread;
         // Get the data from the other half of the warp. Fill cos_vals and sin_vals.
         for (int i = 0; i < numElemsPerThread; i++)
         {
-            elements2[i] = __shfl_xor_sync(0xffffffff, elements[i], 16);
-            if (laneId < 16)
+            elements2[i] = __shfl_xor_sync(0xffffffff, elements[i], pairOffset);
+            if (laneId < pairOffset)
             {
                 elements2[i] = -elements2[i];
             }
 
             int dim_idx = laneId * numElemsPerThread + i;
-            dim_idx = (dim_idx * 2) % head_dim;
+            dim_idx = (dim_idx * 2) % rotary_dim;
             int half_dim = dim_idx / 2;
-            float freq = powf(base, -2.0f * half_dim / static_cast<float>(head_dim));
+            float freq = powf(base, -2.0f * half_dim / static_cast<float>(rotary_dim));
 
             if (factor != 1.0f)
             {
@@ -251,9 +253,25 @@ __global__ void fusedQKNormRopeKernel(
         __syncwarp();
     }
 
-    for (int i = 0; i < numElemsPerThread; i++)
+    bool const is_full_rope = (rotary_dim == head_dim);
+    if (is_full_rope)
     {
-        elements[i] = (elements[i] * cos_vals[i] + elements2[i] * sin_vals[i]) * attention_factor;
+        for (int i = 0; i < numElemsPerThread; i++)
+        {
+            elements[i] = (elements[i] * cos_vals[i] + elements2[i] * sin_vals[i]) * attention_factor;
+        }
+    }
+    else
+    {
+        for (int i = 0; i < numElemsPerThread; i++)
+        {
+            int dim_idx = laneId * numElemsPerThread + i;
+
+            if (dim_idx < rotary_dim)
+            {
+                elements[i] = (elements[i] * cos_v
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## a5cfc8368f - [https://nvbugs/5508536][fix] Revert #7041: Move stop_criteria to sample_async (#7041) (#7796)

- **Date**: 2025-09-18
- **Author**: Netanel Haber
- **Categories**: Parallelism/Async

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- FP8 quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Pinned memory
- Reduce synchronization overhead
- Speculative decoding
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase
- Disaggregated serving

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/llm_request.py   |  19 +-
 tensorrt_llm/_torch/pyexecutor/sampler.py       | 442 ++++++------------------
 tensorrt_llm/_torch/pyexecutor/sampler_utils.py |  61 ----
 tensorrt_llm/_torch/speculative/mtp.py          |  61 ++--
 tests/integration/test_lists/waives.txt         |   1 -
 tests/unittest/_torch/test_torch_sampler.py     | 189 ----------
 6 files changed, 139 insertions(+), 634 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/llm_request.py b/tensorrt_llm/_torch/pyexecutor/llm_request.py
index 3d21238ee..f730dfcd9 100644
--- a/tensorrt_llm/_torch/pyexecutor/llm_request.py
+++ b/tensorrt_llm/_torch/pyexecutor/llm_request.py
@@ -1,8 +1,6 @@
-from collections.abc import Generator
 from copy import deepcopy
 from dataclasses import dataclass
-from itertools import pairwise
-from typing import Any, Dict, List, Optional, TypeAlias, Union
+from typing import Any, Dict, List, Optional, Union
 
 import torch
 
@@ -426,10 +424,7 @@ class LlmRequest(tensorrt_llm.bindings.internal.batch_manager.LlmRequest):
         self.child_requests.append(py_request)
 
 
-StopWordList: TypeAlias = list[list[int]]
-
-
-def convert_wordlist(word_list) -> StopWordList:
+def convert_wordlist(word_list) -> List[List[int]]:
     """Converts a wordlist from format:
 
     [[word_0 token_0, word_0 token_1, ...], [word_1 token_0, ...], ...]]
@@ -466,16 +461,6 @@ def convert_wordlist(word_list) -> StopWordList:
     return [tokens, offsets]
 
 
-def produce_stop_words(
-        py_stop_words_list: StopWordList) -> Generator[list[int], None, None]:
-    """yield stop sequences from the output of `convert_wordlist` above."""
-    stop_words_list, prefix_sum = py_stop_words_list
-    for start, end in pairwise((0, *prefix_sum)):  # first element: prepend 0
-        if end == -1:  # -1 is a sentinel value in convert_wordlist
-            break
-        yield stop_words_list[start:end]
-
-
 def executor_request_to_llm_request(
         req_id: int,
         executor_request: ExecutorRequest,
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index 1e2897f9a..d2ee22a21 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -1,16 +1,12 @@
 from abc import ABC, abstractmethod
 from collections.abc import Iterable
 from dataclasses import dataclass
-from functools import cached_property
-from typing import List, Literal, Optional, TypeAlias
+from typing import List, Literal, Optional
 
-import numpy as np
 import torch
 
 from tensorrt_llm._torch.pyexecutor.make_decoding_batch_input_output import \
     MakeDecodingBatchInputOutput
-from tensorrt_llm._torch.pyexecutor.sampler_utils import (
-    BEAM_0, SINGLE_BEAM_WIDTH, handle_stop_single_beam)
 from tensorrt_llm._utils import nvtx_range, torch_dtype_to_binding
 from tensorrt_llm.bindings import (CudaStream, DataType, ModelConfig,
                                    WorldConfig, make_sampling_config)
@@ -359,55 +355,21 @@ def int_tensor(shape: tuple[int, ...], device: str = 'cuda') -> torch.Tensor:
     return torch.empty(shape, dtype=torch.int, device=device)
 
 
-class TorchStore:
-
-    def __init__(self, *, max_draft_len: int, max_num_sequences: int,
-                 max_beam_width: int):
-        self.max_draft_len = max_draft_len
-        self.max_num_sequences = max_num_sequences
-        self.max_beam_width = 
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Pinned (page-locked) memory enables faster host-to-device data transfers. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## a6f2a1e918 - Fix test_fused_moe_w4afp8 (#4393)

- **Date**: 2025-05-16
- **Author**: NVJiangShao
- **Categories**: Fusion, Quantization Optimization

### Optimization Techniques

- Operator fusion
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/modules/fused_moe.py | 26 ++++++--------------------
 1 file changed, 6 insertions(+), 20 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fused_moe.py b/tensorrt_llm/_torch/modules/fused_moe.py
index 4c86ec308..f5d7a15d3 100755
--- a/tensorrt_llm/_torch/modules/fused_moe.py
+++ b/tensorrt_llm/_torch/modules/fused_moe.py
@@ -590,18 +590,12 @@ class FusedMoE(nn.Module):
                                    self.intermediate_size_per_partition // 2)
 
                 fc31_act_scale = nn.Parameter(torch.empty(
-                    self.expert_size_per_partition,
-                    1,
-                    dtype=self.dtype,
-                    device=device),
+                    self.expert_size_per_partition, 1, dtype=self.dtype),
                                               requires_grad=False)
                 self.register_parameter("fc31_act_scale", fc31_act_scale)
 
                 fc2_act_scale = nn.Parameter(torch.empty(
-                    self.expert_size_per_partition,
-                    1,
-                    dtype=self.dtype,
-                    device=device),
+                    self.expert_size_per_partition, 1, dtype=self.dtype),
                                              requires_grad=False)
                 self.register_parameter("fc2_act_scale", fc2_act_scale)
 
@@ -611,8 +605,7 @@ class FusedMoE(nn.Module):
                                 self.hidden_size // (128 * self.interleave[0]),
                                 self.intermediate_size_per_partition * 2 *
                                 self.interleave[0],
-                                dtype=self.dtype,
-                                device=device),
+                                dtype=self.dtype),
                     requires_grad=False)
                 self.register_parameter("fc31_weight_scale", fc31_weight_scale)
 
@@ -622,24 +615,17 @@ class FusedMoE(nn.Module):
                                 self.intermediate_size_per_partition //
                                 (128 * self.interleave[1]),
                                 self.hidden_size * self.interleave[1],
-                                dtype=self.dtype,
-                                device=device),
+                                dtype=self.dtype),
                     requires_grad=False)
                 self.register_parameter("fc2_weight_scale", fc2_weight_scale)
 
                 fc31_alpha = nn.Parameter(torch.empty(
-                    self.expert_size_per_partition,
-                    1,
-                    dtype=torch.float32,
-                    device=device),
+                    self.expert_size_per_partition, 1, dtype=torch.float32),
                                           requires_grad=False)
                 self.register_parameter("fc31_alpha", fc31_alpha)
 
                 fc2_alpha = nn.Parameter(torch.empty(
-                    self.expert_size_per_partition,
-                    1,
-                    dtype=torch.float32,
-                    device=device),
+                    self.expert_size_per_partition, 1, dtype=torch.float32),
       
```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## a891013e3c - [feat] Optimize KV Cache Reuse for MLA (#4869)

- **Date**: 2025-06-13
- **Author**: zhhuang-nv
- **Categories**: Cache Optimization

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- KV cache optimization
- Batching optimization
- Pinned memory
- Reduce synchronization overhead
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
cpp/tensorrt_llm/kernels/mlaKernels.cu            | 318 ++++++++++-----------
 cpp/tensorrt_llm/kernels/mlaKernels.h             |  13 +-
 cpp/tensorrt_llm/thop/mlaPreprocessOp.cpp         | 234 +++++-----------
 cpp/tests/unit_tests/kernels/mlaPreprocessTest.cu | 325 +---------------------
 tensorrt_llm/_torch/attention_backend/trtllm.py   |  71 +----
 tensorrt_llm/_torch/modules/attention.py          |  73 ++---
 6 files changed, 268 insertions(+), 766 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/mlaKernels.cu b/cpp/tensorrt_llm/kernels/mlaKernels.cu
index 1b6cd3dea..ffd0c51ec 100644
--- a/cpp/tensorrt_llm/kernels/mlaKernels.cu
+++ b/cpp/tensorrt_llm/kernels/mlaKernels.cu
@@ -688,7 +688,7 @@ __global__ void loadPagedKVCacheForMLAKernel(T* compressed_kv_ptr, T* k_pe_ptr,
 // k {total_token, h, d}, v {total_token, h, d}, k_pe {total_token, h=1, d_rope}
 // output {b, 2, ceil(max_seq / kv_cache_tokens_per_block), h, kv_cache_tokens_per_block, d}
 template <typename T>
-__global__ void setPagedKVCacheForMLAKernel(T* output, T* const k_ptr, T* const v_ptr, T* const k_pe_ptr,
+__global__ void setPagedKVCacheForMLAKernel(T* output, T const* k_ptr, T const* v_ptr, T const* k_pe_ptr,
     int64_t const* cu_seq_lens, int const max_input_seq_len, int num_heads, int kv_dim, int rope_dim,
     int kv_cache_tokens_per_block, int64_t kv_token_stride)
 {
@@ -718,8 +718,10 @@ __global__ void setPagedKVCacheForMLAKernel(T* output, T* const k_ptr, T* const
             {
                 int ld_kv_global_offset = (global_token_offset + local_token_idx) * kv_token_stride + head_idx * kv_dim;
                 int ld_kv_local_offset = head_dim_vec_idx;
-                auto k_data = (reinterpret_cast<typename KT::VecT*>(k_ptr + ld_kv_global_offset))[ld_kv_local_offset];
-                auto v_data = (reinterpret_cast<typename KT::VecT*>(v_ptr + ld_kv_global_offset))[ld_kv_local_offset];
+                auto k_data
+                    = (reinterpret_cast<typename KT::VecT const*>(k_ptr + ld_kv_global_offset))[ld_kv_local_offset];
+                auto v_data
+                    = (reinterpret_cast<typename KT::VecT const*>(v_ptr + ld_kv_global_offset))[ld_kv_local_offset];
                 // {b, 0, token / kv_cache_tokens_per_block, h, token % kv_cache_tokens_per_block, ...}
                 int st_k_global_offset = batch_idx * 2 * kv_cache_block_num * kv_cache_block_size
                     + local_token_idx / kv_cache_tokens_per_block * kv_cache_block_size
@@ -737,8 +739,8 @@ __global__ void setPagedKVCacheForMLAKernel(T* output, T* const k_ptr, T* const
             {
                 int ld_rope_global_offset = (global_token_offset + local_token_idx) * rope_dim;
                 int ld_rope_local_offset = head_dim_vec_idx - KT::kKVThreadPerHead;
-                auto rope_data
-                    = (reinterpret_cast<typename KT::VecT*>(k_pe_ptr + ld_rope_global_offset))[ld_rope_local_offset];
+                auto rope_data = (reinterpret_cast<typename KT::VecT const*>(
+                    k_pe_ptr + ld_rope_global_offset))[ld_rope_local_offset];
                 // {b, 0, token / kv_cache_tokens_per_block, h, token % kv_cache_tokens_per_block, ...}
                 int st_rope_global_offset = batch_idx * 2 * kv_cache_block_num * kv_cache_block_size
                     + local_token_idx / kv_cache_tokens_per_block * kv_cache_block_size
@@ -756,147 +758,153 @@ __global__ void setPagedKVCacheForMLAKernel(T* outpu
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Pinned (page-locked) memory enables faster host-to-device data transfers. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## a966644a71 - [None][fix] Change Ray submit() to use async RPC (#8636)

- **Date**: 2025-10-27
- **Author**: Erin
- **Categories**: Parallelism/Async

### Optimization Techniques

- Async/stream-based execution

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/executor/ray_executor.py | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/executor/ray_executor.py b/tensorrt_llm/executor/ray_executor.py
index 5d87fdc9b..e0c810d75 100644
--- a/tensorrt_llm/executor/ray_executor.py
+++ b/tensorrt_llm/executor/ray_executor.py
@@ -208,7 +208,7 @@ class RayExecutor(GenerationExecutor):
             self.call_all_ray_workers("enqueue_request",
                                       leader_only=True,
                                       request=request,
-                                      async_call=False,
+                                      async_call=True,
                                       result_wait_queue=result.queue)
 
         return result

```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## ac2ab9ba36 - [AutoDeploy][perf] Further optimize flashinfer backend in AutoDeploy (#4024)

- **Date**: 2025-05-05
- **Author**: Suyog Gupta
- **Categories**: General Performance

### Optimization Techniques

- FP8 quantization
- Integer quantization
- KV cache optimization
- Batching optimization
- Attention mechanism optimization

### Applicable Conditions

- Prefill phase
- Decode/generation phase

### Changed Files

```
.../auto_deploy/custom_ops/flashinfer_attention.py | 34 +++++++---
 .../_torch/auto_deploy/shim/ad_executor.py         |  3 +
 .../custom_ops/test_flashinfer_attention_op.py     | 79 ++++++++++++++++++++--
 3 files changed, 103 insertions(+), 13 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/custom_ops/flashinfer_attention.py b/tensorrt_llm/_torch/auto_deploy/custom_ops/flashinfer_attention.py
index b935dc820..88241c985 100644
--- a/tensorrt_llm/_torch/auto_deploy/custom_ops/flashinfer_attention.py
+++ b/tensorrt_llm/_torch/auto_deploy/custom_ops/flashinfer_attention.py
@@ -186,8 +186,23 @@ def prepare_flashinfer_metadata(
 
     paged_kv_last_page_len = ((offsets + seq_len - 1) % page_size) + 1
 
+    # Compute batch_indices and positions so that they can be reused for kv cache appends
+    # for all the layers
+    batch_indices, positions = flashinfer.get_batch_indices_positions(
+        qo_indptr,
+        flashinfer.get_seq_lens(paged_kv_indptr, paged_kv_last_page_len, page_size),
+        position_ids.numel(),
+    )
+
     # return metadata
-    return (qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len)
+    return (
+        qo_indptr,
+        paged_kv_indptr,
+        paged_kv_indices,
+        paged_kv_last_page_len,
+        batch_indices,
+        positions,
+    )
 
 
 @prepare_flashinfer_metadata.register_fake
@@ -195,11 +210,15 @@ def prepare_flashinfer_metadata_fake(
     input_ids, position_ids, seq_len, input_pos, cache_loc, pages_per_seq, page_size
 ):
     qo_indptr = torch.empty(len(seq_len) + 1, dtype=seq_len.dtype, device=seq_len.device)
+    batch_indices = torch.empty_like(cache_loc)
+    positions = torch.empty_like(cache_loc)
     return (
         qo_indptr,  # qo_indptr
         torch.empty_like(qo_indptr),  # paged_kv_indptr
         torch.empty_like(cache_loc),  # paged_kv_indices
         torch.empty_like(seq_len),  # paged_kv_last_page_len
+        batch_indices,  # batch_indices
+        positions,  # positions
     )
 
 
@@ -214,6 +233,8 @@ def flashinfer_mha_with_cache(
     paged_kv_indptr: torch.Tensor,
     paged_kv_indices: torch.Tensor,
     paged_kv_last_page_len: torch.Tensor,
+    batch_indices: torch.Tensor,
+    positions: torch.Tensor,
     # CACHES
     k_cache: torch.Tensor,
     v_cache: torch.Tensor,
@@ -254,13 +275,6 @@ def flashinfer_mha_with_cache(
         k = (k / k_scale).to(torch.float8_e4m3fn)
         v = (v / v_scale).to(torch.float8_e4m3fn)
 
-    # Append to kv cache
-    batch_indices, positions = flashinfer.get_batch_indices_positions(
-        qo_indptr,
-        flashinfer.get_seq_lens(paged_kv_indptr, paged_kv_last_page_len, pp.page_size),
-        q.shape[0],
-    )
-
     flashinfer.page.append_paged_kv_cache(
         k,
         v,
@@ -296,6 +310,8 @@ def flashinfer_mha_with_cache_fake(
     paged_kv_indptr: torch.Tensor,
     paged_kv_indices: torch.Tensor,
     paged_kv_last_page_len: torch.Tensor,
+    batch_indices: torch.Tensor,
+    positions: torch.Tensor,
     # CACHES
     k_cache: torch.Tensor,
     v_cache: torch.Tensor,
@@ -341,7 +357,7 @@ class FlashInferAttention(AttentionDescriptor):
 
     @classmethod
     def get_prepare_metadata_op(cls) -> Tuple[PrepareMetadataCallable, int]:
-  
```

### Analysis Summary

Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## b10704428d - [https://nvbugs/5787566][fix] Only keep a limited number of performance statistic data (#10569)

- **Date**: 2026-01-14
- **Author**: HuiGao-NV
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- KV cache optimization
- Speculative decoding

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/auto_deploy/llm_args.py         | 6 ++++++
 tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py | 2 ++
 tensorrt_llm/_torch/pyexecutor/py_executor.py       | 4 +++-
 tensorrt_llm/llmapi/llm_args.py                     | 6 ++++++
 tests/unittest/api_stability/references/llm.yaml    | 4 ++++
 5 files changed, 21 insertions(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/llm_args.py b/tensorrt_llm/_torch/auto_deploy/llm_args.py
index aa9f0147c..7c46a48df 100644
--- a/tensorrt_llm/_torch/auto_deploy/llm_args.py
+++ b/tensorrt_llm/_torch/auto_deploy/llm_args.py
@@ -384,6 +384,12 @@ class LlmArgs(AutoDeployConfig, BaseLlmArgs, BaseSettings):
 
     _quant_config: Optional[QuantConfig] = PrivateAttr(default=None)
 
+    max_stats_len: int = Field(
+        default=1000,
+        description="The max number of performance statistic entries.",
+        status="prototype",
+    )
+
     @property
     def quant_config(self) -> QuantConfig:
         if self._quant_config is None:
diff --git a/tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py b/tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py
index 48066cb25..a81e0f3f5 100644
--- a/tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py
+++ b/tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py
@@ -490,10 +490,12 @@ class ADEngine(ModelEngine):
             self.max_beam_width = ad_config.max_beam_width
             self.spec_config = ad_config.speculative_config
             self._disable_overlap_scheduler = ad_config.disable_overlap_scheduler
+            self.llm_args.max_stats_len = ad_config.max_stats_len
         else:
             self.max_beam_width = 1
             self.spec_config = None
             self._disable_overlap_scheduler = False
+            self.llm_args.max_stats_len = 1000
 
         # check for max total draft tokens
         if self.spec_config is not None:
diff --git a/tensorrt_llm/_torch/pyexecutor/py_executor.py b/tensorrt_llm/_torch/pyexecutor/py_executor.py
index 412997336..94fa5020c 100644
--- a/tensorrt_llm/_torch/pyexecutor/py_executor.py
+++ b/tensorrt_llm/_torch/pyexecutor/py_executor.py
@@ -143,7 +143,6 @@ class PyExecutor:
         super(PyExecutor, self).__init__()
         self.device_id = torch.cuda.current_device()
         self.global_rank = dist.rank
-
         # Store the execution stream for model forward operations.
         # This stream is used for proper synchronization with KVCacheTransferManager.
         # execution_stream can be provided by create_py_executor
@@ -181,6 +180,7 @@ class PyExecutor:
         self.max_draft_len = max_draft_len
         self.max_total_draft_tokens = max_total_draft_tokens
         self.llm_args = self.model_engine.llm_args
+        self.max_stats_len = max(self.llm_args.max_stats_len, 1)
         self.max_num_tokens = self.llm_args.max_num_tokens
         self.print_log = self.llm_args.print_iter_log
         self.enable_iter_perf_stats = self.llm_args.enable_iter_perf_stats
@@ -866,6 +866,8 @@ class PyExecutor:
                            req_stats: Optional[List[RequestStats]] = None):
 
         with self.stats_lock:
+            if len(self.stats) > self.max_stats_len:
+                self.stats.pop(0)
             self.stats.append((stats, req_stats))
 
     def _process_iter_stats(
diff --git a/tensorrt_llm/llmapi/llm_args.py b/tensorrt_ll
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## b168adba70 - feat: Add NVFP4 UB pattern optimization pass in torch compile (#3371)

- **Date**: 2025-04-11
- **Author**: liji-nv
- **Categories**: General Performance

### Optimization Techniques

- Torch compilation/JIT optimization
- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- Parallelism optimization
- Reduce synchronization overhead
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- Decode/generation phase

### Changed Files

```
.../kernels/userbuffers/userbuffers.cu             |   6 +-
 cpp/tensorrt_llm/thop/fp4Gemm.cpp                  |  23 +-
 .../_torch/compilation/patterns/ub_allreduce.py    | 156 +++++++++++-
 tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py   |  23 +-
 tensorrt_llm/_torch/custom_ops/torch_custom_ops.py |  19 +-
 tensorrt_llm/_torch/pyexecutor/model_engine.py     |  16 +-
 tensorrt_llm/_torch/utils.py                       |  18 ++
 tensorrt_llm/models/llama/model.py                 |   8 +-
 .../unittest/_torch/multi_gpu/test_user_buffers.py | 283 ++++++++++++++++++++-
 9 files changed, 499 insertions(+), 53 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/userbuffers/userbuffers.cu b/cpp/tensorrt_llm/kernels/userbuffers/userbuffers.cu
index 0c80b3958..16905e1fd 100644
--- a/cpp/tensorrt_llm/kernels/userbuffers/userbuffers.cu
+++ b/cpp/tensorrt_llm/kernels/userbuffers/userbuffers.cu
@@ -568,7 +568,7 @@ __global__ void __launch_bounds__(MAX_THREADS)
 #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
     using PackedVec = PackedVec<DType>;
     cudaTriggerProgrammaticLaunchCompletion();
-    float const sf = 1.f / *scale;
+    float sf = *scale;
     __shared__ float s_variance;
     int hidden_dim = blockDim.x * UNROLL_NLINES * sizeof(int4) / sizeof(DType);
 
@@ -683,7 +683,7 @@ __global__ void __launch_bounds__(MAX_THREADS)
 #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
     using PackedVec = PackedVec<DType>;
     cudaTriggerProgrammaticLaunchCompletion();
-    float const sf = 1.f / *scale;
+    float sf = *scale;
     __shared__ float s_variance;
     int hidden_dim = blockDim.x * UNROLL_NLINES * sizeof(int4) / sizeof(DType);
 
@@ -1914,6 +1914,7 @@ int allreduce2_userbuff_inplace_rmsnorm_quant_fp4_impl(int const handler, size_t
     switch (dataType)
     {
     case nvinfer1::DataType::kHALF:
+    {
         if (kDISABLE_FP32_ACCUMULATION)
         {
             return allreduce2_userbuff_inplace_rmsnorm_quant_fp4<half, true>(handler, offset, out_handler, out_offset,
@@ -1927,6 +1928,7 @@ int allreduce2_userbuff_inplace_rmsnorm_quant_fp4_impl(int const handler, size_t
                 residual_out, comm, stream);
         }
         break;
+    }
 #ifdef ENABLE_BF16
     case nvinfer1::DataType::kBF16:
     {
diff --git a/cpp/tensorrt_llm/thop/fp4Gemm.cpp b/cpp/tensorrt_llm/thop/fp4Gemm.cpp
index f4786f767..cc7f8c1d8 100644
--- a/cpp/tensorrt_llm/thop/fp4Gemm.cpp
+++ b/cpp/tensorrt_llm/thop/fp4Gemm.cpp
@@ -19,6 +19,7 @@
 #include "tensorrt_llm/kernels/internal_cutlass_kernels/include/fp4_gemm.h"
 #include "tensorrt_llm/kernels/quantization.h"
 #include "tensorrt_llm/thop/thUtils.h"
+#include "tensorrt_llm/thop/userbuffersTensor.h"
 
 #include <ATen/cuda/EmptyTensor.h>
 #include <ATen/native/cuda/Resize.h>
@@ -72,7 +73,8 @@ void runGemm(at::Tensor& out, at::Tensor const& mat1, at::Tensor const& mat2, at
 // Only NVFP4 is currently supported
 at::Tensor fp4_bmm_impl(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
     at::Tensor const& mat2Scale, at::Tensor const& globalScale, bool sfUseUE8M0,
-    std::optional<c10::ScalarType> out_dtype, tkc::CutlassGemmConfig const* maybe_config = nullptr)
+    std::optional<c10::ScalarType> out_dtype, bool to_userbuffers = false,
+    tkc::CutlassGemmConfig const* maybe_config = nullptr)
 {
     CHECK_INPUT(mat1, FLOAT4_E2M1X2);
     CHECK_INPUT(mat2, FLOAT4_E2M1X2);
@@ -127,9 +129,16 @@ at::Tensor fp4_bmm_impl(at::Tensor const& mat1, at::Tensor const& mat2, at::Tens
     TORCH_CHECK(out_dtype == torch::kFloat || out_dtype == torch::kHalf || out_dtype == torch::kBFloat16,
      
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## b2095aa074 - [#4674][bugfix] AutoDeploy Fix memory leak in fuse_moe (#7844)

- **Date**: 2025-09-29
- **Author**: Gal Hubara-Agam
- **Categories**: Memory Optimization, Fusion

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- KV cache optimization
- Parallelism optimization
- Reduce synchronization overhead
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../_torch/auto_deploy/config/default.yaml         |  2 +-
 .../auto_deploy/transform/library/fused_moe.py     |  6 +++
 .../transformations/library/test_moe_fusion.py     | 48 ++++++++++++++++++++++
 3 files changed, 55 insertions(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/config/default.yaml b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
index 835c00b52..09629e917 100644
--- a/tensorrt_llm/_torch/auto_deploy/config/default.yaml
+++ b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
@@ -107,7 +107,7 @@ transforms:
     backend: trtllm
   fuse_moe:
     stage: post_load_fusion
-    enabled: false # TODO: https://github.com/NVIDIA/TensorRT-LLM/issues/4674 this is causing OOMs
+    enabled: true
   fuse_allreduce_residual_rmsnorm:
     stage: post_load_fusion
   fuse_collectives:
diff --git a/tensorrt_llm/_torch/auto_deploy/transform/library/fused_moe.py b/tensorrt_llm/_torch/auto_deploy/transform/library/fused_moe.py
index d645fa1e8..88cc53aba 100644
--- a/tensorrt_llm/_torch/auto_deploy/transform/library/fused_moe.py
+++ b/tensorrt_llm/_torch/auto_deploy/transform/library/fused_moe.py
@@ -57,6 +57,12 @@ def _insert_fused_moe_ops(gm: GraphModule) -> int:
         node.replace_all_uses_with(new_node)
         graph.erase_node(node)
 
+        # Delete the unstacked weights immediately to save GPU memory
+        # This will happen automatically after the graph is canonicalized, but for large models we'll run out of memory
+        # during the transformation itself.
+        gm.graph.eliminate_dead_code()
+        gm.delete_all_unused_submodules()
+
     return fused_key_counter
 
 
diff --git a/tests/unittest/_torch/auto_deploy/unit/singlegpu/transformations/library/test_moe_fusion.py b/tests/unittest/_torch/auto_deploy/unit/singlegpu/transformations/library/test_moe_fusion.py
index 2d5cff427..3b55e24f9 100644
--- a/tests/unittest/_torch/auto_deploy/unit/singlegpu/transformations/library/test_moe_fusion.py
+++ b/tests/unittest/_torch/auto_deploy/unit/singlegpu/transformations/library/test_moe_fusion.py
@@ -368,3 +368,51 @@ def test_moe_fusion():
         num_param_nodes_fused < num_param_nodes
     ), f"""number of parameter nodes after fusion {num_param_nodes_fused} <
         number of parameter nodes before fusion {num_param_nodes}"""
+
+
+def test_fuse_moe_cleanup():
+    # Ensure deterministic allocations and a clean slate
+    torch.manual_seed(1234)
+    torch.cuda.manual_seed(1234)
+    torch.cuda.empty_cache()
+
+    device = "cuda"
+    dtype = torch.bfloat16
+
+    # Build model and export to GraphModule (pre-fusion)
+    model = MoEOpModel().to(device=device, dtype=dtype)
+    x = model.get_input(device=device, dtype=dtype)
+    gm = torch_export_to_gm(model, args=(x,), clone=True)
+
+    # Count parameters and measure memory before fusion
+    num_param_nodes_before = len(list(gm.named_parameters()))
+    torch.cuda.synchronize()
+    torch.cuda.empty_cache()
+    mem_before = torch.cuda.memory_allocated()
+
+    # Apply MoE fusion which should stack weights and clean up unstacked params
+    # We need to ensure the cleanup is done as part of the transformation to avoid OOM during the transformation itself.
+    gm_transformed = InferenceOptimizer(
+
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## b4d17d1a4c - [TRTLLM-8991][test] Add Llama 3.3 70B model with different performance config (#8753)

- **Date**: 2025-11-03
- **Author**: yufeiwu-nv
- **Categories**: General Performance

### Optimization Techniques

- Async/stream-based execution
- FP8 quantization
- KV cache optimization
- Parallelism optimization

### Applicable Conditions

- Prefill phase

### Changed Files

```
tensorrt_llm/bench/benchmark/__init__.py           |  2 +-
 .../integration/defs/perf/pytorch_model_config.py  | 36 +++++++++++++++++-----
 tests/integration/defs/perf/test_perf.py           | 34 +++++++++++++++-----
 tests/integration/test_lists/qa/llm_perf_nim.yml   |  2 ++
 4 files changed, 58 insertions(+), 16 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/bench/benchmark/__init__.py b/tensorrt_llm/bench/benchmark/__init__.py
index 2668e5638..aa004afd4 100644
--- a/tensorrt_llm/bench/benchmark/__init__.py
+++ b/tensorrt_llm/bench/benchmark/__init__.py
@@ -105,7 +105,7 @@ def get_llm(runtime_config: RuntimeConfig, kwargs: dict):
     """
     llm_cls = LLM
 
-    if runtime_config.backend != "tensorrt":
+    if runtime_config.backend != None:
         ignore_trt_only_args(kwargs, runtime_config.backend)
 
     if runtime_config.backend == 'pytorch':
diff --git a/tests/integration/defs/perf/pytorch_model_config.py b/tests/integration/defs/perf/pytorch_model_config.py
index 8038ea4ec..28f7f0bbb 100644
--- a/tests/integration/defs/perf/pytorch_model_config.py
+++ b/tests/integration/defs/perf/pytorch_model_config.py
@@ -14,7 +14,7 @@
 # limitations under the License.
 # -*- coding: utf-8 -*-
 """
-Model pytorch yaml config for trtllm-bench perf tests
+Model pytorch/TRT yaml config for trtllm-bench perf tests
 """
 
 
@@ -36,12 +36,18 @@ def get_model_yaml_config(model_label: str,
         Returns:
             dict: yaml config
         """
-    base_config = {
-        'print_iter_log': True,
-        'cuda_graph_config': {
-            'enable_padding': True,
-        },
-    }
+    if 'pytorch' in model_label:
+        # Pytorch backend config
+        base_config = {
+            'print_iter_log': True,
+            'cuda_graph_config': {
+                'enable_padding': True,
+            },
+        }
+    else:
+        # TRT backend config
+        base_config = {}
+
     if 'kv_cache_dtype' in model_label:
         base_config.update({
             'kv_cache_dtype':
@@ -241,6 +247,19 @@ def get_model_yaml_config(model_label: str,
             'config': {
                 'enable_chunked_prefill': True,
             }
+        },
+        # Llama-v3.3 models with xgrammar guided decoding
+        {
+            'patterns': [
+                "llama_v3.3_70b_instruct_fp8-bench-float8-maxbs:512-maxnt:2048-input_output_len:500,2000-reqs:400-con:200-gpus:8-extra"
+            ],
+            'config': {
+                'extended_runtime_perf_knob_config': {
+                    'cuda_graph_cache_size': 1.0,
+                    'cuda_graph_mode': True,
+                },
+                'guided_decoding_backend': 'xgrammar'
+            }
         }
     ]
 
@@ -251,7 +270,8 @@ def get_model_yaml_config(model_label: str,
             patterns = [patterns]
         for pattern in patterns:
             if pattern in model_label.lower():
-                recursive_update(base_config, pattern_config['config'])
+                if pattern_config.get('config'):
+                    recursive_update(base_config, pattern_config['config'])
                 break  # Stop checking other patterns for this config once we find a match
 
     # lora-specific change for pytorch
diff --git a/tests/integration/defs/perf/test_perf.py b/tests/integration/defs/perf/test_perf.py
index 827
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## b4dab23e7b - [TRTLLM-5965] perf: Optimize MoE sort kernels for large-scale EP (#5435)

- **Date**: 2025-06-30
- **Author**: Enwei Zhu
- **Categories**: Kernel Optimization

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../kernels/cutlass_kernels/include/moe_kernels.h  |  91 +--
 .../cutlass_kernels/moe_gemm/moe_kernels.cu        | 811 ++++++++++++---------
 cpp/tensorrt_llm/kernels/moeUtilOp.cu              |  62 +-
 cpp/tensorrt_llm/kernels/moeUtilOp.h               |  30 +-
 cpp/tensorrt_llm/thop/moeUtilOp.cpp                |   6 +-
 .../unit_tests/kernels/mixtureOfExpertsTest.cu     |  43 +-
 6 files changed, 629 insertions(+), 414 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h b/cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h
index abb4911a8..6adf5cbf3 100644
--- a/cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h
+++ b/cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h
@@ -87,32 +87,6 @@ struct LoraParams
 
 namespace cutlass_kernels
 {
-static inline size_t pad_to_multiple_of_16(size_t const& input)
-{
-    static constexpr int ALIGNMENT = 16;
-    return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
-}
-
-class CubKeyValueSorter
-{
-public:
-    CubKeyValueSorter();
-
-    CubKeyValueSorter(int const num_experts_per_node);
-
-    void updateNumExperts(int const num_experts_per_node);
-
-    static size_t getWorkspaceSize(size_t const num_key_value_pairs, int const num_experts_per_node);
-
-    void run(void* workspace, size_t const workspace_size, int const* keys_in, int* keys_out, int const* values_in,
-        int* values_out, size_t const num_key_value_pairs, cudaStream_t stream);
-
-private:
-    static int expertsToBits(int experts);
-    int num_experts_;
-    int num_bits_;
-};
-
 /**
  * \brief Describes what parallelism mode the MoE is using
  *
@@ -397,9 +371,9 @@ public:
         ActivationType fc1_activation_type, void const* fc2_expert_weights, void const* fc2_expert_biases,
         QuantParams quant_params, int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
         int const num_experts, int const experts_per_token, char* workspace_ptr, void* final_output,
-        int* expanded_source_row_to_expanded_dest_row, MOEParallelismConfig parallelism_config,
-        bool const enable_alltoall, bool use_lora, LoraParams& lora_params, bool use_deepseek_fp8_block_scale,
-        bool min_latency_mode, MoeMinLatencyParams& min_latency_params, cudaStream_t stream)
+        int* unpermuted_row_to_permuted_row, MOEParallelismConfig parallelism_config, bool const enable_alltoall,
+        bool use_lora, LoraParams& lora_params, bool use_deepseek_fp8_block_scale, bool min_latency_mode,
+        MoeMinLatencyParams& min_latency_params, cudaStream_t stream)
         = 0;
 
     // Aliases for profiling the gemms
@@ -413,7 +387,7 @@ public:
         int const num_experts_per_node, ActivationType fc1_activation_type, float const** alpha_scale_ptr_array,
         bool bias_is_broadcast, bool use_deepseek_fp8_block_scale, cudaStream_t stream,
         cutlass_extensions::CutlassGemmConfig config, bool min_latency_mode, int* num_active_experts_per,
-        int* active_expert_global_ids, int start_expert)
+        int* active_expert_global_ids)
         = 0;
 
     virtual void gemm2(void const* const input, void* const gemm_output, void* const final_output,
@@ -421,14 +395,14 @@ public:
         void const* const fc2_expert_weights, void const* const fc2_expert_biases, void const* const fc2_int_scales,
         float const* const fc2_fp8_dequant, TmaWarpSpecializedGroupedGemmI
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## b4e9669d2c - [None][chore] Optimize MOE export by tracing with reduced experts and expanding graph (#11504)

- **Date**: 2026-02-13
- **Author**: Suyog Gupta
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- FP8 quantization
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/auto_deploy/export/export.py   | 323 +++++++++++++++++++++
 .../auto_deploy/transform/library/export_to_gm.py  |   8 +
 .../unit/singlegpu/transformations/test_export.py  | 220 ++++++++++++++
 3 files changed, 551 insertions(+)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/export/export.py b/tensorrt_llm/_torch/auto_deploy/export/export.py
index 4265fea9b..b76a72bc3 100644
--- a/tensorrt_llm/_torch/auto_deploy/export/export.py
+++ b/tensorrt_llm/_torch/auto_deploy/export/export.py
@@ -10,6 +10,7 @@ import torch
 import torch.export as te
 import torch.nn as nn
 from torch import fx
+from torch.utils._python_dispatch import TorchDispatchMode
 
 from ..utils._graph import canonicalize_graph, lift_to_meta, load_buffers_and_params, tree_to
 from ..utils.logger import ad_logger
@@ -25,6 +26,309 @@ except ImportError:
     torch_export_context = nullcontext
 
 
+# =====================================================================
+# MOE export optimization: reduce experts for faster tracing, then
+# expand the graph back to include all experts after export.
+# =====================================================================
+
+
+def _infer_target_pattern(target_0: str, target_1: str) -> Tuple[str, str]:
+    """Infer ``(prefix, suffix)`` from two consecutive expert-weight targets.
+
+    Compares two ``get_attr`` targets that differ only in the expert index and
+    returns ``(prefix, suffix)`` such that ``target == prefix + str(idx) + suffix``.
+
+    Example::
+
+        >>> _infer_target_pattern('experts.0.gate.weight', 'experts.1.gate.weight')
+        ('experts.', '.gate.weight')
+    """
+    parts_0 = target_0.split(".")
+    parts_1 = target_1.split(".")
+    if len(parts_0) != len(parts_1):
+        raise ValueError(f"Target structure mismatch: {target_0} vs {target_1}")
+
+    diff_positions = [i for i, (a, b) in enumerate(zip(parts_0, parts_1)) if a != b]
+    if len(diff_positions) != 1:
+        raise ValueError(
+            f"Expected exactly one differing part, found {len(diff_positions)}: "
+            f"{target_0} vs {target_1}"
+        )
+
+    idx = diff_positions[0]
+    prefix = ".".join(parts_0[:idx]) + "." if idx > 0 else ""
+    suffix = "." + ".".join(parts_0[idx + 1 :]) if idx < len(parts_0) - 1 else ""
+    return prefix, suffix
+
+
+def _infer_single_target_pattern(target: str, expert_prefix: str) -> Tuple[str, str]:
+    """Infer ``(prefix, suffix)`` when only one expert target is available.
+
+    Uses the known *expert_prefix* to locate the expert index position.
+
+    Example::
+
+        >>> _infer_single_target_pattern('layer.0.experts.0.w.weight', 'layer.0.experts')
+        ('layer.0.experts.', '.w.weight')
+    """
+    full_prefix = expert_prefix + "."
+    if not target.startswith(full_prefix):
+        raise ValueError(f"Target '{target}' does not start with '{full_prefix}'")
+    remainder = target[len(full_prefix) :]  # e.g. '0.w.weight'
+    _idx_str, _, after_idx = remainder.partition(".")
+    suffix = "." + after_idx if after_idx else ""
+    return full_prefix, suffix
+
+
+def _register_nested_parameter(gm: fx.GraphModule, dotted_name: str, param: nn.Parameter) -> None:
+    """Register a parameter at a nested dotted p
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## b558232ce1 - Refactor CutlassFusedMoE (#5344)

- **Date**: 2025-06-19
- **Author**: hlu1
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
- PyTorch built-in optimized ops
- Speculative decoding
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU
- Decode/generation phase

### Changed Files

```
tensorrt_llm/_mnnvl_utils.py                       |   3 +-
 tensorrt_llm/_torch/models/modeling_deepseekv3.py  | 146 ++--
 tensorrt_llm/_torch/models/modeling_qwen3_moe.py   |   4 +-
 tensorrt_llm/_torch/modules/fused_moe/__init__.py  |  30 +-
 .../_torch/modules/fused_moe/create_moe.py         |  22 +-
 .../_torch/modules/fused_moe/fused_moe_cutlass.py  | 720 +++-------------
 .../_torch/modules/fused_moe/fused_moe_wide_ep.py  | 913 +++++++++++++++++++++
 tests/unittest/_torch/modules/test_fused_moe.py    |  32 +-
 8 files changed, 1115 insertions(+), 755 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_mnnvl_utils.py b/tensorrt_llm/_mnnvl_utils.py
index 4e0b7554e..2c1c5952d 100644
--- a/tensorrt_llm/_mnnvl_utils.py
+++ b/tensorrt_llm/_mnnvl_utils.py
@@ -71,7 +71,8 @@ class MnnvlMemory:
 
     def __del__(self):
         if not sys.is_finalizing():
-            MnnvlMemory.close_mnnvl_memory(self.ptr)
+            if hasattr(self, "ptr"):
+                MnnvlMemory.close_mnnvl_memory(self.ptr)
 
     def as_torch_strided_tensor(self, dtype):
         num_segments = MnnvlMemory.comm.Get_size()
diff --git a/tensorrt_llm/_torch/models/modeling_deepseekv3.py b/tensorrt_llm/_torch/models/modeling_deepseekv3.py
index d0b00450f..5e6014cc7 100644
--- a/tensorrt_llm/_torch/models/modeling_deepseekv3.py
+++ b/tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -53,7 +53,7 @@ from ..modules.attention import MLA
 from ..modules.decoder_layer import DecoderLayer
 from ..modules.embedding import Embedding
 from ..modules.fused_moe import (CutlassFusedMoE, DeepSeekV3MoeRoutingMethod,
-                                 create_moe)
+                                 WideEPMoE, create_moe)
 from ..modules.gated_mlp import GatedMLP
 from ..modules.linear import Linear, TensorParallelMode, WeightsLoadingConfig
 from ..modules.multi_stream_utils import maybe_execute_in_parallel
@@ -511,7 +511,7 @@ class Deepseekv3MoE(nn.Module):
                                           self.mapping,
                                           dim=0,
                                           sizes=all_rank_num_tokens)
-            elif not isinstance(self.experts, CutlassFusedMoE) or (
+            elif not isinstance(self.experts, (CutlassFusedMoE, WideEPMoE)) or (
                     not self.experts.has_fp8_qdq and self.experts.has_nvfp4):
                 # Use padding when not using the cutlass path or when x_sf in self.experts is not None
                 use_dp_padding = True
@@ -721,12 +721,6 @@ class DeepseekV3DecoderLayer(DecoderLayer):
             ) if tp > self.mapping.gpus_per_node else tp  # Avoid costly inter-node TP
         return mlp_tp_size
 
-    def _enable_min_latency_mode(self, num_tokens: int):
-        return (num_tokens <= 128 and self.fusion_config.POST_MOE_FUSION
-                and self.is_nvfp4 and self.model_config.moe_backend == 'CUTLASS'
-                and not self.mapping.is_multi_node()
-                and self.allreduce.mnnvl_allreduce is None)
-
     def forward(
         self,
         position_ids: torch.IntTensor,
@@ -779,116 +773,70 @@ class DeepseekV3DecoderLayer(DecoderLayer):
                 do_finalize=do_finalize,
             )
 
-        cutlass_min_latency_mode = self._enable_min_latency_mode(
-            hidden_states.shape[0])
-
-        if cutlass_min_latency_mode:
-            assert self.fusion_config.PRE_MOE_FUSION and self.fusion_config.POST_MOE_FUSION
-            assert self.model_config.moe_backend == 'CUTLASS'
-
-            hidden_states, hidden_states_act, hidden_states_sf, residual = 
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## b5b83009ff - chore: Reenabling get_stats_async test which seems to have been fixed by recent commit (#3246)

- **Date**: 2025-04-02
- **Author**: pcastonguay
- **Categories**: Parallelism/Async

### Optimization Techniques

- Async/stream-based execution
- Parallelism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tests/unittest/llmapi/test_llm.py | 15 +++++++--------
 1 file changed, 7 insertions(+), 8 deletions(-)
```

### Diff Preview

```diff
diff --git a/tests/unittest/llmapi/test_llm.py b/tests/unittest/llmapi/test_llm.py
index c5c821a57..6e6a2817c 100644
--- a/tests/unittest/llmapi/test_llm.py
+++ b/tests/unittest/llmapi/test_llm.py
@@ -1826,14 +1826,13 @@ def llm_get_stats_async_test_harness(tp_size: int = 1,
     asyncio.run(main())
 
 
-@pytest.mark.parametrize(
-    "return_context_logits, pytorch_backend, use_overlap",
-    [
-        (True, False, False),
-        (False, False, False),
-        (False, True, False),
-        #  (False, True, True), https://nvbugspro.nvidia.com/bug/5163585
-    ])
+@pytest.mark.parametrize("return_context_logits, pytorch_backend, use_overlap",
+                         [
+                             (True, False, False),
+                             (False, False, False),
+                             (False, True, False),
+                             (False, True, True),
+                         ])
 def test_llm_get_stats_async(return_context_logits, pytorch_backend,
                              use_overlap):
     llm_get_stats_async_test_harness(

```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## b622cde5d5 - [None][perf] Fix the tactic sorting in TrtllmGenBatchedGemmRunner::getValidConfigIndices (#7419)

- **Date**: 2025-09-25
- **Author**: Jinyang Yuan
- **Categories**: General Performance

### Optimization Techniques

- Batching optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../trtllmGenKernels/batchedGemm/KernelRunner.cpp  | 92 +++++++++++-----------
 .../trtllmGen_bmm_export/BatchedGemmInterface.h    |  5 +-
 2 files changed, 48 insertions(+), 49 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/KernelRunner.cpp b/cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/KernelRunner.cpp
index ed319323f..5e89f15f2 100644
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/KernelRunner.cpp
+++ b/cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/KernelRunner.cpp
@@ -21,7 +21,8 @@
 #include "tensorrt_llm/common/envUtils.h"
 #include "trtllmGen_bmm_export/BatchedGemmInterface.h"
 #include "trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"
-// DO NOT include logger.h before BatchedGemmInterface.h as it #undef TLLM_LOG_INFO and co.
+// DO NOT include cudaUtils.h and logger.h before BatchedGemmInterface.h as it #undef TLLM_LOG_INFO and co.
+#include "tensorrt_llm/common/cudaUtils.h"
 #include "tensorrt_llm/common/logger.h"
 
 namespace tensorrt_llm
@@ -306,6 +307,8 @@ std::vector<int64_t> TrtllmGenBatchedGemmRunner::getValidConfigIndices(int32_t m
     auto const bmm = BatchedGemmInterface();
     auto const configs = bmm.getBatchedGemmConfigs();
 
+    int32_t multiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
+
     BatchedGemmData gemmData;
     // Dims
     gemmData.mProblemDimensions.mNumBatches = numBatches;
@@ -319,73 +322,68 @@ std::vector<int64_t> TrtllmGenBatchedGemmRunner::getValidConfigIndices(int32_t m
     gemmData.mProblemDimensions.mRank = 0;
     gemmData.mProblemDimensions.mWorldSize = 1;
     gemmData.mProblemDimensions.mMaxNumCtasInTokenDim = maxNumCtasInBatchDim;
-    // Tier 0: K < tileK, prefer higher efficiency.
-    auto cmpTier0 = [&configs, &gemmData](int64_t idx0, int64_t idx1)
+    auto cmpFunc = [&configs, &gemmData, &bmm, &multiProcessorCount](int64_t idx0, int64_t idx1)
     {
         auto const& optionsA = configs[idx0].mOptions;
         auto const& optionsB = configs[idx1].mOptions;
         int32_t sizeK = gemmData.mProblemDimensions.mK;
-        // Both waste computation, prefer higher efficiency.
-        if (sizeK <= optionsA.mTileK && sizeK <= optionsB.mTileK)
-        {
-            double eff_a = (double) sizeK / optionsA.mTileK;
-            double eff_b = (double) sizeK / optionsB.mTileK;
-            return eff_a > eff_b;
-        }
-        // If either can be utilized, sort by tileK.
-        else
+
+        // Tier 0: K < tileK, prefer higher efficiency.
+        if (optionsA.mTileK != optionsB.mTileK)
         {
-            return optionsA.mTileK > optionsB.mTileK;
+            // Both waste computation, prefer higher efficiency.
+            if (sizeK <= optionsA.mTileK && sizeK <= optionsB.mTileK)
+            {
+                double eff_a = (double) sizeK / optionsA.mTileK;
+                double eff_b = (double) sizeK / optionsB.mTileK;
+                return eff_a > eff_b;
+            }
+            // If either can be utilized, sort by tileK.
+            else
+            {
+                return optionsA.mTileK > optionsB.mTileK;
+            }
         }
-    };
-    // Tier 1: When tileK is 
```

### Analysis Summary

GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## b6acd96616 - [None][fix] AutoDeploy: Fix the nvfp4 fused_moe (#10727)

- **Date**: 2026-01-16
- **Author**: Chenghao Zhang
- **Categories**: Fusion

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- MoE optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../_torch/auto_deploy/config/default.yaml         |   2 +-
 .../auto_deploy/custom_ops/fused_moe/trtllm_moe.py |  48 +++--
 .../auto_deploy/models/quant_config_reader.py      |   2 +-
 .../auto_deploy/transform/library/fused_moe.py     |  13 +-
 .../defs/accuracy/test_llm_api_autodeploy.py       |  17 ++
 .../unit/singlegpu/custom_ops/test_trtllm_moe.py   | 210 +++++++++++++++++++++
 6 files changed, 266 insertions(+), 26 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/config/default.yaml b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
index ad590a86f..12ea7c5ed 100644
--- a/tensorrt_llm/_torch/auto_deploy/config/default.yaml
+++ b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
@@ -130,7 +130,7 @@ transforms:
     backend: trtllm
   fuse_nvfp4_moe:
     stage: post_load_fusion
-    enabled: false
+    enabled: true
   fuse_allreduce_residual_rmsnorm:
     stage: post_load_fusion
   # TODO (lucaslie): add backend selection as part of configurable inference optimizers
diff --git a/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/trtllm_moe.py b/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/trtllm_moe.py
index 3a0ab6b4a..bcd903d26 100644
--- a/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/trtllm_moe.py
+++ b/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/trtllm_moe.py
@@ -271,13 +271,9 @@ def trtllm_quant_nvfp4_moe_fused(
     # Convert the inter_size from number of uint8 elements to number of FP4 elements.
     inter_size *= FP4_PER_UINT8
 
-    # Validate shapes and padding requirements as defined by the cutlass kernel.
+    # Validate block scale tensors are 3D (padding requirements handled below)
     assert fc1_weight_blockscale_fp8.ndim == 3, "fc1_weight_blockscale_fp8 must be 3D"
     assert fc2_weight_blockscale_fp8.ndim == 3, "fc2_weight_blockscale_fp8 must be 3D"
-    assert fc1_weight_blockscale_fp8.size(1) % TRTLLM_NVFP4_ROW_SIZE == 0
-    assert fc2_weight_blockscale_fp8.size(1) % TRTLLM_NVFP4_ROW_SIZE == 0
-    assert fc1_weight_blockscale_fp8.size(2) % TRTLLM_NVFP4_COLUMN_SIZE == 0
-    assert fc2_weight_blockscale_fp8.size(2) % TRTLLM_NVFP4_COLUMN_SIZE == 0
 
     _validate_mlp_style_and_act_fn(is_gated_mlp, act_fn)
     act_fn = ActivationType.Swiglu if act_fn == ActivationType.Silu else act_fn
@@ -292,7 +288,7 @@ def trtllm_quant_nvfp4_moe_fused(
         input_blockscale = None
         output_dtype = x.dtype
 
-    # Pad inter_size to be divisible by 128
+    # Pad inter_size to be divisible by TRTLLM_NVFP4_ROW_SIZE
     inter_size_padded = math.ceil(inter_size / TRTLLM_NVFP4_ROW_SIZE) * TRTLLM_NVFP4_ROW_SIZE
     fc1_inter_size_padded = (
         math.ceil(fc1_inter_size / TRTLLM_NVFP4_ROW_SIZE) * TRTLLM_NVFP4_ROW_SIZE
@@ -305,18 +301,30 @@ def trtllm_quant_nvfp4_moe_fused(
         not is_gated_mlp and inter_size_padded != inter_size
     )
     hidden_size_needs_padding = hidden_size % TRTLLM_NVFP4_COLUMN_SIZE != 0
+
+    hidden_blocks_padded = hidden_size_padded // NVFP4_BLOCK_SIZE
+    inter_blocks_padded = inter_size_padded // NVFP4_BLOCK_SIZE
+
     if inter_size_needs_padding or hidden_size_needs_padding:
-        assert False, "See https://github.com/NVIDIA/TensorRT-LLM/issues/10331"
-        # fc1_expert_weights_fp4: [E, I, H] or [E, 2*I, H]
+        # Pad fc1_expert_weights_fp4: [E, I, H/2] or [E, 2*I, H/2]
         fc1_padded = fc1_expert_weights_fp4.new_zeros(
             fc1_expert_weights_fp4.size(0
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## b99c5ce8c1 - Feat/ds r1 min latency opt round3, add router gemm, fused a gemm, PDL  (#4560)

- **Date**: 2025-06-14
- **Author**: yunruis
- **Categories**: Throughput/Latency, Fusion

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
- Speculative decoding
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU
- Decode/generation phase

### Changed Files

```
cpp/tensorrt_llm/kernels/CMakeLists.txt            |   1 +
 .../communicationKernels/allReduceFusionKernels.cu |  30 +-
 .../communicationKernels/allReduceFusionKernels.h  |   1 +
 .../moeAllReduceFusionKernels.cu                   |   5 +-
 .../kernels/dsv3MinLatencyKernels/CMakeLists.txt   |   8 +
 .../dsv3MinLatencyKernels/dsv3FusedAGemm.cu        | 695 +++++++++++++++++++++
 .../kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.h |  31 +
 .../dsv3MinLatencyKernels/dsv3RouterGemm.cu        | 241 +++++++
 .../kernels/dsv3MinLatencyKernels/dsv3RouterGemm.h |  30 +
 cpp/tensorrt_llm/thop/CMakeLists.txt               |   2 +
 cpp/tensorrt_llm/thop/allreduceOp.cpp              |  20 +-
 cpp/tensorrt_llm/thop/cublasScaledMM.h             |  36 ++
 cpp/tensorrt_llm/thop/dsv3FusedAGemmOp.cpp         |  96 +++
 cpp/tensorrt_llm/thop/dsv3RouterGemmOp.cpp         | 117 ++++
 .../compilation/patterns/ar_residual_norm.py       |  17 +-
 .../_torch/compilation/patterns/ub_allreduce.py    |  24 +-
 tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py   |  17 +
 tensorrt_llm/_torch/distributed/ops.py             |   2 +
 tensorrt_llm/_torch/models/modeling_deepseekv3.py  |  82 ++-
 tensorrt_llm/functional.py                         |   4 +-
 .../unittest/_torch/thop/test_dsv3_fused_a_gemm.py |  25 +
 .../unittest/_torch/thop/test_dsv3_router_gemm.py  |  27 +
 22 files changed, 1469 insertions(+), 42 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/CMakeLists.txt b/cpp/tensorrt_llm/kernels/CMakeLists.txt
index d1ecc7f08..c1bafe39f 100644
--- a/cpp/tensorrt_llm/kernels/CMakeLists.txt
+++ b/cpp/tensorrt_llm/kernels/CMakeLists.txt
@@ -83,4 +83,5 @@ add_subdirectory(trtllmGenKernels)
 add_subdirectory(fusedLayernormKernels)
 add_subdirectory(groupRmsNormKernels)
 add_subdirectory(llama4MinLatencyKernels)
+add_subdirectory(dsv3MinLatencyKernels)
 add_subdirectory(causalConv1d)
diff --git a/cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu b/cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu
index 16ffaf34e..2d9415a27 100644
--- a/cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu
+++ b/cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu
@@ -439,7 +439,7 @@ public:
     int tot_access;
 };
 
-template <AllReduceFusionPattern Pattern, typename DType, int NRanks, bool Fp32Acc>
+template <AllReduceFusionPattern Pattern, typename DType, int NRanks, bool Fp32Acc, bool TriggerCompletionAtEnd = true>
 __global__ void allreduce_fusion_kernel_oneshot_lamport(AllReduceFusionParams params)
 {
     IndexHelper<DType> index_helper(params);
@@ -451,8 +451,13 @@ __global__ void allreduce_fusion_kernel_oneshot_lamport(AllReduceFusionParams pa
     int tot_access = index_helper.tot_access;
     float4 clear_vec = get_neg_zero();
     FusedOp<Pattern, DType> fused_op(params, access_id, access_id_in_token);
+
 #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
     cudaGridDependencySynchronize();
+    if constexpr (!TriggerCompletionAtEnd)
+    {
+        cudaTriggerProgrammaticLaunchCompletion();
+    }
 #endif
     LamportComm<NRanks> comm(params.workspace, params.rank);
     int clear_access = comm.clear_size / kElemsPerAccess<DType>;
@@ -503,9 +508,14 @@ __global__ void allreduce_fusion_kernel_oneshot_lamport(AllReduceFusionParams pa
         float4 sum_val = allreduce_sum<DType, NRanks, Fp32Acc>(vals);
         fused_op(sum_val, tidx);
     }
+
     comm.update(params.size * NRanks);
+
 #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
-    cudaTriggerProgrammaticLaunchCompletion();
+    if constexpr (TriggerCompletionAtEnd)
+    {
+        cudaTriggerProgrammaticLaunchCompletion();
+    }
 #endif
 }
 
@@ -591,11 +601,11 @@ int get_sm_count()
     return sm_count;
 }
 
-template <AllReduceFusionPattern Pattern, typename DType, int NRanks, bool Fp32Acc>
+template <AllReduceFusionPattern Pattern, typename DType, int NRanks, bool Fp32Acc, bool TriggerCompletionAtEnd = true>
 void launch_oneshot_lamport(AllReduceFusionParams const& params, cudaLaunchConfig_t& cfg)
 {
-    TLLM_CUDA_CHECK(
-        cudaLaunchKernelEx(&cfg, allreduce_fusion_kernel_oneshot_lamport<Pattern, DType, NRanks, Fp32Acc>, params));
+    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg,
+        allreduce_fusion_kernel_oneshot_lamport<Pattern, DType, NRanks, Fp32Acc, TriggerCompletionAtEnd>, params));
 }
 
 template <AllR
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## b9b1c1368c - feat: Support unfused rope in MLA. (#3610)

- **Date**: 2025-04-17
- **Author**: yuxianq
- **Categories**: Fusion

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- KV cache optimization
- Parallelism optimization
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/modules/attention.py | 83 ++++++++++++++++++++++++--------
 1 file changed, 62 insertions(+), 21 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/attention.py b/tensorrt_llm/_torch/modules/attention.py
index 4c75b0209..833d03bbb 100644
--- a/tensorrt_llm/_torch/modules/attention.py
+++ b/tensorrt_llm/_torch/modules/attention.py
@@ -6,7 +6,8 @@ from torch import nn
 
 from tensorrt_llm.mapping import Mapping
 
-from ..attention_backend import AttentionInputType, AttentionMetadata
+from ..attention_backend import (AttentionInputType, AttentionMetadata,
+                                 TrtllmAttention)
 from ..attention_backend.interface import (PositionalEmbeddingParams,
                                            PredefinedAttentionMask)
 from ..attention_backend.utils import create_attention
@@ -135,7 +136,9 @@ class Attention(nn.Module):
         self.support_fused_qkv = self.attn_backend == "TRTLLM"
         self.support_unfused_qkv = self.attn_backend != "TRTLLM"
         self.rotary_emb = None
-        if not self.enable_rope_fusion and pos_embd_params is not None:
+        self.apply_rotary_emb = (not self.enable_rope_fusion
+                                 and pos_embd_params is not None)
+        if self.apply_rotary_emb:
             self.rotary_emb = RotaryEmbedding(
                 pos_embd_params.rope,
                 head_dim=self.head_dim,
@@ -223,7 +226,7 @@ class Attention(nn.Module):
 
         q, k, v = qkv, None, None
 
-        if self.rotary_emb is not None and position_ids is not None:
+        if self.apply_rotary_emb and position_ids is not None:
             q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
                                 dim=-1)
             q, k = self.rotary_emb(position_ids, [q, k])
@@ -349,8 +352,7 @@ class MLA(nn.Module):
 
             self.q_b_proj = Linear(
                 self.q_lora_rank,
-                tp_size * self.num_heads *
-                (self.qk_nope_head_dim + self.qk_rope_head_dim),
+                tp_size * self.num_heads * self.qk_head_dim,
                 bias=bias,
                 dtype=dtype,
                 mapping=mapping,
@@ -369,14 +371,14 @@ class MLA(nn.Module):
 
             self.q_proj = Linear(
                 self.q_lora_rank,
-                tp_size * self.num_heads *
-                (self.qk_nope_head_dim + self.qk_rope_head_dim),
+                tp_size * self.num_heads * self.qk_head_dim,
                 bias=bias,
                 dtype=dtype,
                 mapping=mapping,
                 tensor_parallel_mode=TensorParallelMode.COLUMN,
                 quant_config=quant_config,
-                skip_create_weights=config.skip_create_weights)
+                skip_create_weights=config.skip_create_weights,
+            )
             self.q_b_proj = self.q_proj
 
         self.kv_a_layernorm = RMSNorm(hidden_size=kv_lora_rank,
@@ -504,6 +506,32 @@ class MLA(nn.Module):
         self.aux_stream = aux_stream
         self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]
 
+        self.enable_rope_fusion = isinstance(self.mha, Tr
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## ba8abeab10 - [OMNIML-2336][feat] add W4A8 NVFP4 FP8 fused moe (#7968)

- **Date**: 2025-09-30
- **Author**: sychen52
- **Categories**: Fusion, Quantization Optimization

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
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- NVIDIA Hopper (SM90) GPU

### Changed Files

```
.../_torch/modules/fused_moe/create_moe.py         |   1 +
 .../modules/fused_moe/fused_moe_trtllm_gen.py      |  49 +++++++-
 tensorrt_llm/_torch/modules/fused_moe/interface.py |   6 +
 .../_torch/modules/fused_moe/quantization.py       |  83 +++++++++++--
 tests/unittest/_torch/modules/test_fused_moe.py    | 135 ++++++++++++++++++++-
 tests/unittest/_torch/thop/parallel/test_moe.py    |  64 ++++++++--
 6 files changed, 312 insertions(+), 26 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fused_moe/create_moe.py b/tensorrt_llm/_torch/modules/fused_moe/create_moe.py
index 74f56ee5d..7382264ed 100644
--- a/tensorrt_llm/_torch/modules/fused_moe/create_moe.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/create_moe.py
@@ -41,6 +41,7 @@ def get_moe_cls(
                 quant_config.quant_mode.has_fp8_block_scales()
                 or quant_config.quant_mode.has_nvfp4()
                 or quant_config.quant_mode.has_w4a16_mxfp4()
+                or quant_config.quant_mode.has_w4a8_nvfp4_fp8()
                 or quant_config.quant_mode.has_w4a8_mxfp4_fp8()
                 or quant_config.quant_mode.has_w4a8_mxfp4_mxfp8()):
             return TRTLLMGenFusedMoE
diff --git a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py
index 09caf8ed4..3123783af 100644
--- a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py
@@ -15,6 +15,7 @@ from .quantization import (DeepSeekFP8BlockScalesFusedMoEMethod,
                            NVFP4TRTLLMGenFusedMoEMethod,
                            W4A8MXFP4FP8TRTLLMGenFusedMoEMethod,
                            W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod,
+                           W4A8NVFP4FP8TRTLLMGenFusedMoEMethod,
                            W4A16MXFP4TRTLLMGenFusedMoEMethod)
 from .routing import BaseMoeRoutingMethod, DeepSeekV3MoeRoutingMethod
 
@@ -111,7 +112,7 @@ class TRTLLMGenFusedMoE(MoE):
 
     def _check_configs(self):
         assert self.has_deepseek_fp8_block_scales \
-            or self.has_nvfp4 or self.has_w4a16_mxfp4 \
+            or self.has_nvfp4 or self.has_w4a16_mxfp4 or self.has_w4a8_nvfp4_fp8 \
             or self.has_w4a8_mxfp4_fp8 or self.has_w4a8_mxfp4_mxfp8, "TRTLLMGenFusedMoE only supports fp8_block_scaling, nvfp4, w4a16_mxfp4, w4a8_mxfp4_fp8 and w4a8_mxfp4_mxfp8 dtypes."
 
         if self.bias or self.swiglu_alpha is not None or self.swiglu_beta is not None or self.swiglu_limit is not None:
@@ -125,6 +126,8 @@ class TRTLLMGenFusedMoE(MoE):
                 return NVFP4TRTLLMGenFusedMoEMethod()
             elif self.quant_config.layer_quant_mode.has_w4a16_mxfp4():
                 return W4A16MXFP4TRTLLMGenFusedMoEMethod()
+            elif self.quant_config.layer_quant_mode.has_w4a8_nvfp4_fp8():
+                return W4A8NVFP4FP8TRTLLMGenFusedMoEMethod()
             elif self.quant_config.layer_quant_mode.has_w4a8_mxfp4_fp8():
                 return W4A8MXFP4FP8TRTLLMGenFusedMoEMethod()
             elif self.quant_config.layer_quant_mode.has_w4a8_mxfp4_mxfp8():
@@ -147,8 +150,8 @@ class TRTLLMGenFusedMoE(MoE):
         self._weights_created = True
         self._check_configs()
 
-        # TODO: FIX this.
-        if (self.has_w4a16_mxfp4 or self.has_w4a8_mxfp4_fp8
+        if (self.has_w4a16_mxfp4 or self.has_w4a8_nvfp4_fp8
+                or self.has_w4a8_mxfp4_fp8
     
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## be2065755c - [None][fix] Enforce minimum NVSHMEM_QP_DEPTH of 128 for DeepEP low latency (#12100)

- **Date**: 2026-03-11
- **Author**: Iman Tabrizian
- **Categories**: Throughput/Latency

### Optimization Techniques

- Operator fusion
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../_torch/modules/fused_moe/communication/deep_ep_low_latency.py     | 4 ++--
 1 file changed, 2 insertions(+), 2 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fused_moe/communication/deep_ep_low_latency.py b/tensorrt_llm/_torch/modules/fused_moe/communication/deep_ep_low_latency.py
index bfaadd5c5..be733da57 100644
--- a/tensorrt_llm/_torch/modules/fused_moe/communication/deep_ep_low_latency.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/communication/deep_ep_low_latency.py
@@ -1,4 +1,4 @@
-# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 # SPDX-License-Identifier: Apache-2.0
 #
 # Licensed under the Apache License, Version 2.0 (the "License");
@@ -92,7 +92,7 @@ class DeepEPLowLatency(Communication):
 
         # Set nvshmem queue pair depth larger than the number of on-flight WRs
         # (ref: https://github.com/deepseek-ai/DeepEP/issues/427)
-        os.environ["NVSHMEM_QP_DEPTH"] = str(2 * (self.deep_ep_max_num_tokens + 1))
+        os.environ["NVSHMEM_QP_DEPTH"] = str(max(128, 2 * (self.deep_ep_max_num_tokens + 1)))
 
         self.deep_ep_buffer = buffer_pool.get_low_latency_buffer(mapping)
         self.deep_ep_buffer.reserve(self.deep_ep_max_num_tokens, hidden_size, num_slots)

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## bf1b958f1a - [TRTLLM-7319][perf] Fuse slicing into MoE. (#6728)

- **Date**: 2025-08-26
- **Author**: Bo Li
- **Categories**: Fusion

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- Parallelism optimization
- PyTorch built-in optimized ops
- Reduce synchronization overhead
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU

### Changed Files

```
.../mixtureOfExpertsBackendBenchmarkFixture.h      |  16 +--
 .../epilogue/fusion/sm90_visitor_scatter.hpp       |  52 +++++---
 .../cutlass_kernels/include/moe_gemm_kernels.h     |   1 +
 .../kernels/cutlass_kernels/include/moe_kernels.h  |  80 ++++++------
 .../cutlass_kernels/include/moe_util_kernels.h     |   5 +-
 .../launchers/moe_gemm_tma_ws_launcher.inl         |   2 +-
 .../moe_gemm_tma_warp_specialized_input.cu         |   1 +
 .../cutlass_kernels/moe_gemm/moe_kernels.cu        | 143 ++++++++++++---------
 .../internal_cutlass_kernels/include/moe_kernels.h |   7 +-
 .../mixtureOfExperts/mixtureOfExpertsPlugin.cpp    |   4 +-
 cpp/tensorrt_llm/thop/moeOp.cpp                    |  35 +++--
 cpp/tensorrt_llm/thop/moeUtilOp.cpp                |  23 ++--
 .../unit_tests/kernels/mixtureOfExpertsTest.cu     |  55 ++++++--
 tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py   |   5 +-
 tensorrt_llm/_torch/custom_ops/torch_custom_ops.py |   9 ++
 .../_torch/modules/fused_moe/fused_moe_cute_dsl.py |   3 +-
 .../_torch/modules/fused_moe/fused_moe_cutlass.py  |  29 ++---
 .../_torch/modules/fused_moe/fused_moe_deepgemm.py |   3 +-
 18 files changed, 282 insertions(+), 191 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h b/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h
index 36cbe7654..8e8b77469 100644
--- a/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h
+++ b/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h
@@ -707,13 +707,13 @@ public:
 
 #ifdef USING_OSS_CUTLASS_MOE_GEMM
         mGemmProfilerBackend.init(mMoERunner, GemmProfilerBackend::GemmToProfile::Undefined, typeToDtypeID<DataType>(),
-            typeToDtypeID<WeightType>(), typeToDtypeID<OutputType>(), mNumExperts, mK, mHiddenSize, mInterSize,
-            mGroupSize, mActType, mUseBias, mUseLora, /*min_latency_mode=*/false,
+            typeToDtypeID<WeightType>(), typeToDtypeID<OutputType>(), mNumExperts, mK, mHiddenSize, mHiddenSize,
+            mInterSize, mGroupSize, mActType, mUseBias, mUseLora, /*min_latency_mode=*/false,
             /*need_weights=*/false, parallelism_config, /*enable_alltoall=*/false);
 #else
         mGemmProfilerBackend.init(mMoERunner, GemmProfilerBackend::GemmToProfile::Undefined, typeToDtypeID<DataType>(),
-            typeToDtypeID<WeightType>(), typeToDtypeID<OutputType>(), mNumExperts, mK, mHiddenSize, mInterSize,
-            mGroupSize, mActType, mUseBias, mUseLora, /*min_latency_mode=*/false,
+            typeToDtypeID<WeightType>(), typeToDtypeID<OutputType>(), mNumExperts, mK, mHiddenSize, mHiddenSize,
+            mInterSize, mGroupSize, mActType, mUseBias, mUseLora, /*min_latency_mode=*/false,
             /*need_weights=*/false, parallelism_config);
 #endif
 
@@ -989,7 +989,7 @@ public:
                 mExpertWeight1 + mExpertWeight1Size * mBufferIndex, mExpertBias1 + mExpertBias1Size * mBufferIndex,
                 ActivationParams(mActType), mExpertWeight2 + mExpertWeight2Size * mBufferIndex,
                 mExpertBias2 + mExpertBias2Size * mBufferIndex, mQuantParams[mBufferIndex], mTotalTokens, mHiddenSize,
-                mInterSize, mNumExperts, mK, mWorkspace + mWorkspaceSize * mBufferIndex,
+                mHiddenSize, mInterSize, mNumExperts, mK, mWorkspace + mWorkspaceSize * mBufferIndex,
                 mFinalOutput + mFinalOutputSize * mBufferIndex,
                 mSourceToExpandedMap + mSourceToExpandedMapSize * mBufferIndex, parallelism_config,
                 /*enable_alltoall=*/false, mUseLora, mLoraParams[mBufferIndex],
@@ -1001,10 +1001,10 @@ public:
                 mExpertWeight1 + mExpertWeight1Size * mBufferIndex, mExpertBias1 + mExpertBias1Size * mBufferIndex,
                 ActivationParams(mActType), mExpertWeight2 + mExpertWeight2Size * mBufferIndex,
                 mExpertBias2 + mExpertBias2Size * mBufferIndex, mQuantParams[mBufferIndex], mTotalTokens, mHiddenSize,
-                mInterSize, mNumExperts, mK, mWorkspace + mWorkspaceSize * mBufferIndex,
+                mHiddenSize, mInterSize, mNumExperts, mK, mWorkspace + mWorkspaceSize * mBufferIndex,
                 mFinalOutput + mFinalOutputSiz
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## bf5b2a2e0a - test: amend regex match for perf throughput (#4186)

- **Date**: 2025-05-09
- **Author**: ruodil
- **Categories**: Throughput/Latency

### Optimization Techniques

- FP8 quantization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tests/integration/defs/perf/test_perf.py                      | 2 +-
 tests/integration/test_lists/qa/trt_llm_release_perf_test.yml | 6 +++---
 2 files changed, 4 insertions(+), 4 deletions(-)
```

### Diff Preview

```diff
diff --git a/tests/integration/defs/perf/test_perf.py b/tests/integration/defs/perf/test_perf.py
index 8542d167f..cfe692c43 100644
--- a/tests/integration/defs/perf/test_perf.py
+++ b/tests/integration/defs/perf/test_perf.py
@@ -184,7 +184,7 @@ BENCH_PERF_METRIC_LOG_QUERIES = {
     PerfMetricType.INFERENCE_TIME:
     re.compile(r"Total Latency \(ms\):\s+([\d\.]+)"),
     PerfMetricType.TOKEN_THROUGHPUT:
-    re.compile(r"GPU Output Throughput \(tokens\/sec\/gpu\):\s+([\d\.]+)"),
+    re.compile(r"GPU Output Throughput \(tps\/gpu\):\s+([\d\.]+)"),
     PerfMetricType.SEQ_THROUGHPUT:
     re.compile(r"Request Throughput \(req\/sec\):\s+([\d\.]+)"),
     PerfMetricType.FIRST_TOKEN_TIME:
diff --git a/tests/integration/test_lists/qa/trt_llm_release_perf_test.yml b/tests/integration/test_lists/qa/trt_llm_release_perf_test.yml
index 06c35bf41..f978bd224 100644
--- a/tests/integration/test_lists/qa/trt_llm_release_perf_test.yml
+++ b/tests/integration/test_lists/qa/trt_llm_release_perf_test.yml
@@ -42,7 +42,7 @@ trt_llm_release_perf_test:
   - perf/test_perf.py::test_perf[llama_v3.1_8b-bench-bfloat16-input_output_len:512,32]
   - perf/test_perf.py::test_perf[llama_v3.1_8b_instruct_fp8-bench-pytorch-float8-input_output_len:128,128-cons:8]
   - perf/test_perf.py::test_perf[qwen2_7b_instruct-bench-float16-input_output_len:128,128]
-  - perf/test_perf.py::test_perf[starcoder2_3b-bench-float16-input_output_len:512,200]
+  - perf/test_perf.py::test_perf[starcoder2_3b-bench-pytorch-float16-input_output_len:512,200]
   - perf/test_perf.py::test_perf[mistral_7b_v0.1-bench-float16-input_output_len:128,128]
 
   # E2E ENC-DEC
@@ -101,7 +101,7 @@ trt_llm_release_perf_test:
   - perf/test_perf.py::test_perf[llama_v3.1_70b-bench-bfloat16-input_output_len:512,32-gpus:4]
   - perf/test_perf.py::test_perf[qwen_14b_chat-bench-float16-input_output_len:128,128-gpus:4]
   - perf/test_perf.py::test_perf[qwen_14b_chat-bench-float16-input_output_len:512,32-gpus:4]
-  - perf/test_perf.py::test_perf[starcoder_15b-bench-float16-input_output_len:512,200-gpus:4]
+  - perf/test_perf.py::test_perf[starcoder_15b-bench-pytorch-float16-input_output_len:512,200-gpus:4]
 
 - condition:
     ranges:
@@ -198,7 +198,7 @@ trt_llm_release_perf_test:
   - perf/test_perf.py::test_perf[llama_v3.1_70b-bench-bfloat16-input_output_len:512,200-quant:fp8-tp:4]
   - perf/test_perf.py::test_perf[llama_v3.3_70b_instruct_fp8-bench-pytorch-float8-input_output_len:128,128-tp:4]
   - perf/test_perf.py::test_perf[mixtral_8x22b_v0.1-bench-float16-input_output_len:512,512-quant:fp8-tp:4]
-  - perf/test_perf.py::test_perf[starcoder_15b-bench-float16-input_output_len:512,512-quant:fp8-tp:4]
+  - perf/test_perf.py::test_perf[starcoder_15b-bench-pytorch-float16-input_output_len:512,512-quant:fp8-tp:4]
 
 - condition:
     terms:

```

### Analysis Summary

Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## c0cf5a3706 - [None][feat] Optimize 6KD fp8 blockscale gemm (#11502)

- **Date**: 2026-03-13
- **Author**: CarstyYou
- **Categories**: Quantization Optimization

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- NVIDIA Hopper (SM90) GPU

### Changed Files

```
.../fp8_blockscale_gemm/fp8_blockscale_gemm.cu     |  17 +
 .../fp8_blockscale_gemm_kernel.cuh                 | 217 ++++--
 .../sm120_blockwise_gemm/sm120_fp8_gemm_1d1d.cuh   | 589 +++++++++------
 .../sm120_fp8_moe_gemm_1d1d.cuh                    | 800 +++++++++++++++++++++
 .../sm120_blockwise_gemm/sm120_utils.cuh           | 182 ++++-
 cpp/tensorrt_llm/thop/fp8BlockScalingGemm.cpp      |  46 +-
 .../_torch/modules/fused_moe/fused_moe_cutlass.py  |  14 +-
 .../_torch/modules/fused_moe/quantization.py       |   7 +-
 tests/unittest/_torch/helpers.py                   |  26 +-
 .../unittest/_torch/modules/moe/quantize_utils.py  |   4 +
 .../unittest/_torch/modules/moe/test_moe_module.py |  12 +-
 tests/unittest/_torch/modules/test_fused_moe.py    |  15 +-
 .../thop/parallel/test_fp8_block_scale_gemm.py     |  90 ++-
 13 files changed, 1680 insertions(+), 339 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu b/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu
index e8552e21f..cb59a97b6 100644
--- a/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu
+++ b/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu
@@ -136,6 +136,23 @@ void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::moeGemm(void*
         }
     }
 
+    int arch = tensorrt_llm::common::getSMVersion();
+    if (arch == 120)
+    {
+        if constexpr (std::is_same_v<ElementA, __nv_bfloat16> && std::is_same_v<ElementB, __nv_fp8_e4m3>)
+        {
+            fp8_grouped_gemm_run(reinterpret_cast<__nv_bfloat16 const*>(mat_a), fp8_mat_a, per_token_per_128c_scales,
+                nullptr, fp8_mat_b, per_block_scales, reinterpret_cast<__nv_bfloat16*>(mat_d), problem_m_offsets,
+                num_problems, expected_m, max_shape_m_4_align_, max_shape_m_32_align_padded_, shape_n, shape_k, stream,
+                internal_quantize_a, internal_quantize_b);
+        }
+        else
+        {
+            TLLM_THROW("sm120 fp8 blockscale moe gemm only supports ElementA=bfloat16, ElementB=fp8_e4m3.");
+        }
+        return;
+    }
+
 #ifdef COMPILE_HOPPER_TMA_GEMMS
     if constexpr (std::is_same_v<ElementA, __nv_bfloat16> && std::is_same_v<ElementB, __nv_bfloat16>)
     {
diff --git a/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh b/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh
index e3dbcbae9..cd3ed32b8 100644
--- a/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh
+++ b/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh
@@ -31,6 +31,7 @@
 #include "fp8_blockscale_mma_utils.cuh"
 #include "fp8_blockscale_tma_utils.cuh"
 #include "sm120_blockwise_gemm/sm120_fp8_gemm_1d1d.cuh"
+#include "sm120_blockwise_gemm/sm120_fp8_moe_gemm_1d1d.cuh"
 #include "tensorrt_llm/common/config.h"
 #include "tensorrt_llm/common/cudaTypeUtils.cuh"
 #include "tensorrt_llm/common/cudaUtils.h"
@@ -713,8 +714,11 @@ void gemm_dispatch_sm89(void* mat_a, void* mat_b, void* mat_d, float* scales_a,
     TLLM_CHECK_WITH_INFO(result == cudaSuccess, "sm89 gemm kernel runtime error: %s", cudaGetErrorString(result));
 }
 
-void gemm_dispatch_sm120(void* mat_a, void* mat_b, void* mat_d, float* scales_a, float* scales_b, uint32_t shape_m,
-    uint32_t shape_n, uint32_t shape_k, cudaStream_t stream, int num_device_sms = kNumDeviceSMs)
+template <int TileM, int TileN, int NumStages>
+void launch_sm120_gemm_kernel(__nv_fp8_e4m3* mat_a, int64_t ld_a, int64_t stride_a, __nv_fp8_e4m3* mat_b, int64_t ld_b,
+    int64_t stride_b, __nv_bfloat16* mat_d, int64_t ld_d, int64_t stride_d, float* scales_a, int64_t stride_scales_a,
+    float* scales_b, int64_t st
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## c12e67bb66 - [TRTLLM-8958][feat] and [TRTLLM-8960]: create ConfigurableMoE and support TRTLLMGenFusedMoE as backend (#9486)

- **Date**: 2025-12-01
- **Author**: xxi
- **Categories**: Fusion

### Optimization Techniques

- Torch compilation/JIT optimization
- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- PyTorch built-in optimized ops
- Speculative decoding
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- NVIDIA Hopper (SM90) GPU
- Prefill phase
- Decode/generation phase
- Large batch / high concurrency

### Changed Files

```
tensorrt_llm/_torch/model_config.py                |    5 +
 tensorrt_llm/_torch/models/modeling_deepseekv3.py  |   17 +-
 tensorrt_llm/_torch/models/modeling_gpt_oss.py     |   12 +
 tensorrt_llm/_torch/models/modeling_hunyuan_moe.py |   16 +-
 tensorrt_llm/_torch/models/modeling_utils.py       |   22 +
 .../modules/fused_moe/communication/__init__.py    |   12 +-
 .../communication/allgather_reducescatter.py       |    7 +
 .../_torch/modules/fused_moe/communication/base.py |   25 +
 .../communication/communication_factory.py         |  205 ++--
 .../modules/fused_moe/communication/deep_ep.py     |   38 +-
 .../fused_moe/communication/deep_ep_low_latency.py |    4 +-
 .../{mnnvl_throughput.py => nvlink_one_sided.py}   |  144 +--
 .../{mnnvl_latency.py => nvlink_two_sided.py}      |   40 +-
 .../_torch/modules/fused_moe/configurable_moe.py   | 1098 ++++++++++++++++++++
 .../_torch/modules/fused_moe/create_moe.py         |  187 +++-
 .../_torch/modules/fused_moe/fused_moe_cutlass.py  |   73 +-
 .../_torch/modules/fused_moe/fused_moe_deepgemm.py |   25 +-
 .../modules/fused_moe/fused_moe_trtllm_gen.py      |  690 ++++++------
 .../_torch/modules/fused_moe/fused_moe_vanilla.py  |    6 +-
 .../_torch/modules/fused_moe/fused_moe_wide_ep.py  |    9 +-
 tensorrt_llm/_torch/modules/fused_moe/interface.py |   93 +-
 .../_torch/modeling/test_modeling_nemotron_h.py    |    4 +-
 tests/unittest/_torch/modules/test_fused_moe.py    |   76 +-
 23 files changed, 2210 insertions(+), 598 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/model_config.py b/tensorrt_llm/_torch/model_config.py
index b7e42fc09..232683a27 100644
--- a/tensorrt_llm/_torch/model_config.py
+++ b/tensorrt_llm/_torch/model_config.py
@@ -165,6 +165,11 @@ class ModelConfig(Generic[TConfig]):
             self.allreduce_strategy = get_all_reduce_strategy(
                 self.allreduce_strategy)
 
+        # Set default moe_max_num_tokens if not specified
+        # The maximum number of tokens in MoE are multiplied by DP size when attention DP is enabled
+        if self.moe_max_num_tokens is None:
+            self.moe_max_num_tokens = self.max_num_tokens * self.mapping.dp_size
+
     @property
     def torch_dtype(self) -> torch.dtype:
         """Get the torch dtype of the model."""
diff --git a/tensorrt_llm/_torch/models/modeling_deepseekv3.py b/tensorrt_llm/_torch/models/modeling_deepseekv3.py
index 419dc8d70..e81662b0e 100755
--- a/tensorrt_llm/_torch/models/modeling_deepseekv3.py
+++ b/tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -55,7 +55,7 @@ from ..model_config import ModelConfig
 from ..modules.attention import MLA
 from ..modules.decoder_layer import DecoderLayer
 from ..modules.embedding import Embedding
-from ..modules.fused_moe import (DeepSeekV3MoeRoutingMethod,
+from ..modules.fused_moe import (DeepSeekV3MoeRoutingMethod, MoE,
                                  MoEWeightLoadingMode, create_moe)
 from ..modules.fused_moe.fused_moe_wide_ep import WideEPMoE
 from ..modules.gated_mlp import GatedMLP
@@ -391,6 +391,21 @@ class DeepseekV3WeightLoader:
                         "gate_proj": "w1",
                     })
                     module.load_weights(weights=[module_weights])
+                elif names[-1] == "backend" and isinstance(module, MoE):
+                    # Special case: ConfigurableMoE.backend (TRTLLMGenFusedMoE)
+                    # Currently saved MoE weights don't include 'backend' in their names.
+                    # After MoE refactoring, ConfigurableMoE now has a backend submodule,
+                    # and weights loading is done in the backend, so module name includes '.backend'.
+                    # We need to use parent module name (without .backend) to match saved weight names.
+                    # After MoE refactoring is fully complete, all paths will follow this branch.
+                    parent_name = '.'.join(names[:-1])
+                    module_weights = filter_weights(parent_name, weights)
+                    module_weights = rename_moe_weight(module_weights, {
+                        "down_proj": "w2",
+                        "up_proj": "w3",
+                        "gate_proj": "w1",
+                    })
+                    module.load_weights(weights=[module_weights])
                 elif names[-1] == "self_attn":
                     continue
                 elif names[-1] == "next_layer_layernorm":
diff --git a/tensorrt_llm/_torch/models/modeling_gpt_oss.py b/tensorrt_llm/_torch/models/mo
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## c2d3c6cdba - [https://nvbugs/5884735][fix] fix deepeplowlatency with DeepGEMM (#11700)

- **Date**: 2026-02-26
- **Author**: Leslie Fang
- **Categories**: Throughput/Latency

### Optimization Techniques

- Operator fusion
- Parallelism optimization
- MoE optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py   | 8 +++++++-
 tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py | 7 ++++++-
 2 files changed, 13 insertions(+), 2 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py b/tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py
index 51ef184c2..cd0cb71fb 100644
--- a/tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py
@@ -495,7 +495,13 @@ class ConfigurableMoE(MoE):
             # When using communication, dispatch will create tensors with shape:
             # [ep_size * max_tokens_per_rank, ...] due to padding for balanced distribution
             # So we need to allocate workspace based on this size
-            num_rows = self.mapping.moe_ep_size * max(all_rank_num_tokens)
+            if isinstance(self.comm, DeepEPLowLatency):
+                # deeptplowlatency dispatch outputs shape is
+                # [#local_experts * moe_ep_size * max_tokens_per_rank, hidden size]
+                # local_experts = self.num_slots / moe_ep_size
+                num_rows = self.num_slots * max(all_rank_num_tokens)
+            else:
+                num_rows = self.mapping.moe_ep_size * max(all_rank_num_tokens)
 
         workspaces = self.backend.get_workspaces([num_rows])
         return workspaces[0]
diff --git a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py
index f1e88cf74..b051c9348 100644
--- a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py
@@ -740,6 +740,11 @@ class DeepGemmFusedMoE(CutlassFusedMoE):
                                    expert_first_token_offset_tensor,
                                    token_to_expert_map)
 
+        topk = self.routing_method.top_k
+        if token_selected_experts is not None:
+            # For the deepgemmlowlatency, the topk has been viewed into 1
+            topk = token_selected_experts.shape[-1]
+
         final_hidden_states = torch.ops.trtllm.moe_finalize_scale_op(
             permuted_data_tensor,
             None,  # biases
@@ -752,7 +757,7 @@ class DeepGemmFusedMoE(CutlassFusedMoE):
             x.shape[0],  # num_rows
             x.shape[1],  # (possibly padded) hidden_size
             self.unpadded_hidden_size,  # original hidden size
-            self.routing_method.top_k,
+            topk,
             self.expert_size_per_partition,  # num_experts_per_node
             self.tp_size,
             self.tp_rank,

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

