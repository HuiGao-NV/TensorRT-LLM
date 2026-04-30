# Performance Optimization Analysis - Part 10

Commits 262 to 283 of 283

---

## f0b68e4c66 - [None][feat] AutoDeploy: Perf improvement for small batch size (#9163)

- **Date**: 2025-11-18
- **Author**: Chenghao Zhang
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- FP8 quantization
- Integer quantization
- Batching optimization
- Triton kernel
- MoE optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../_torch/auto_deploy/custom_ops/quant.py         |  8 +++++-
 .../auto_deploy/models/patches/nemotron_h.py       | 29 ++++++++++++++++++++++
 .../auto_deploy/transform/library/fused_moe.py     |  8 ++----
 3 files changed, 38 insertions(+), 7 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/custom_ops/quant.py b/tensorrt_llm/_torch/auto_deploy/custom_ops/quant.py
index d219abd59..90ea04db8 100644
--- a/tensorrt_llm/_torch/auto_deploy/custom_ops/quant.py
+++ b/tensorrt_llm/_torch/auto_deploy/custom_ops/quant.py
@@ -104,9 +104,15 @@ def trtllm_quant_fp8_linear(
     assert input_scale is not None
     input_fp8, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(input, input_scale)
 
+    enable_cuda_core = False
+    if torch.cuda.is_available():
+        capability = torch.cuda.get_device_capability(0)
+        enable_cuda_core = capability == (8, 9) or capability == (12, 0)
     # Use TensorRT-LLM FP8 scaled matrix multiply
     # Choose between CUDA core (for small M) and cuBLAS (for large M) implementations
-    if input_fp8.shape[0] <= 8:  # NOTE: this kernel work with n % 2 == 0 as well??
+    if (
+        input_fp8.shape[0] <= 8 and enable_cuda_core
+    ):  # NOTE: this kernel work with n % 2 == 0 as well??
         # Use CUDA core for small M dimension (better for small batch sizes)
         output = torch.ops.trtllm.cuda_scaled_mm(
             input_fp8,
diff --git a/tensorrt_llm/_torch/auto_deploy/models/patches/nemotron_h.py b/tensorrt_llm/_torch/auto_deploy/models/patches/nemotron_h.py
index 8518681a7..8b7c370fb 100644
--- a/tensorrt_llm/_torch/auto_deploy/models/patches/nemotron_h.py
+++ b/tensorrt_llm/_torch/auto_deploy/models/patches/nemotron_h.py
@@ -88,6 +88,34 @@ def _nemotron_h_block_forward(
         return hidden_states
 
 
+def _nemotron_h_topk_router_forward(self, hidden_states):
+    """
+    Forward pass for NemotronHTopkRouter using the optimized noaux_tc_op kernel.
+
+    This replaces the original forward method which used pure PyTorch operations
+    with a fused CUDA kernel that performs:
+    1. Sigmoid activation of logits
+    2. Group-based expert selection
+    3. Top-k selection within selected groups
+    4. Normalized weight computation
+    """
+    hidden_states = hidden_states.view(-1, self.config.hidden_size)
+    router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
+
+    # Use the fused noaux_tc_op kernel which applies sigmoid internally
+    # and performs group-based top-k selection with normalization
+    topk_weights, topk_indices = torch.ops.trtllm.noaux_tc_op(
+        router_logits,
+        self.e_score_correction_bias,
+        self.n_group,
+        self.topk_group,
+        self.top_k,
+        self.routed_scaling_factor,
+    )
+
+    return topk_indices, topk_weights
+
+
 # Note: we assume experts have no bias for now
 def _nemotron_h_moe_forward(self, hidden_states: torch.Tensor):
     """
@@ -138,6 +166,7 @@ CUSTOM_MODULE_PATCHES: Dict[str, List[Tuple[str, Callable]]] = {
     ],
     "NemotronHBlock": [("forward", _nemotron_h_block_forward)],
     "NemotronHMOE": [("forward", _nemotron_h_moe_forward)],
+    "NemotronHTopkRouter": [("forward", _nemotron_h_topk_router_forward)],
 }

```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## f2aee0db03 - [TRTLLM-9854][feat] Optimize the host overhead of _sample_async (#9935)

- **Date**: 2025-12-15
- **Author**: Ziyi Xiong
- **Categories**: Parallelism/Async, Host-side Optimization

### Optimization Techniques

- Async/stream-based execution
- Batching optimization
- Speculative decoding

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/sampler.py          | 76 ++++++++++++++++++++++
 .../unittest/_torch/sampler/test_torch_sampler.py  |  9 ++-
 2 files changed, 84 insertions(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index 83826eaad..c358c9eef 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -968,6 +968,23 @@ class TorchSampler(Sampler, AsyncWorkerMixin):
     def _use_beam_search(self) -> bool:
         return self.max_beam_width > 1
 
+    def _can_use_fast_greedy_path(self, requests: list[LlmRequest]) -> bool:
+        """
+        Check if we can use the fast argmax path for greedy sampling.
+        """
+
+        # Check if all requests use greedy sampling and don't require features
+        # that the fast path skips
+        for req in requests:
+            # vocab_size doesn't affect greediness check
+            if _request_strategy(req, vocab_size=2**31) != GREEDY:
+                return False
+
+            # Fast path skips logprobs handling
+            if req.py_return_log_probs:
+                return False
+        return True
+
     @staticmethod
     def _meet_max_token_stop_criteria(
         request: LlmRequest, max_seq_len: int, beam_idx: int = DEFAULT_BEAM_IDX
@@ -1882,6 +1899,34 @@ class TorchSampler(Sampler, AsyncWorkerMixin):
             d2t = model_outputs["d2t"][tokens]
             tokens += d2t
 
+    @staticmethod
+    @nvtx_range("fast_greedy_sample_kernel")
+    def _fast_greedy_sample_kernel(
+        logits_cuda: torch.Tensor,
+        new_tokens_cuda: torch.Tensor,
+        batch_dest_indices: torch.Tensor,
+        max_beam_width: int,
+        d2t: torch.Tensor | None,
+    ) -> None:
+        """Applies fast greedy sampling to the logits.
+
+        Performs argmax, applies d2t translation if present, and scatters
+        tokens into the output buffer. All operations are in-place.
+        """
+        # Simple argmax for greedy sampling
+        next_tokens = torch.argmax(logits_cuda, dim=-1).to(dtype=new_tokens_cuda.dtype)
+
+        # Apply draft-to-target token translation if present (for Eagle3)
+        if d2t is not None:
+            next_tokens += d2t[next_tokens]
+
+        # Scatter tokens into output buffer
+        batch_dest_indices_expanded = batch_dest_indices.unsqueeze(1).expand(-1, max_beam_width)
+        next_tokens_expanded = next_tokens.unsqueeze(1).expand(-1, max_beam_width)
+        new_tokens_cuda.view(-1, *new_tokens_cuda.shape[2:]).scatter_(
+            0, batch_dest_indices_expanded, next_tokens_expanded
+        )
+
     @staticmethod
     def _apply_embedding_bias(
         logits: torch.Tensor,
@@ -2372,6 +2417,7 @@ class TorchSampler(Sampler, AsyncWorkerMixin):
             if (r.py_stop_words_list is not None and len(r.py_stop_words_list[0]) > 0)
         ]
 
+    @nvtx_range("_write_finish_reasons")
     def _write_finish_reasons(
         self,
         requests: list[LlmRequest],
@@ -2637,6 +2683,36 @@ class TorchSampler(Sampler, AsyncWorkerMixin):
             sampling_requests_metadata.req_num_beams,
         )
 
+        # 
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## f2ebaf288a - [None][feat] TRT-LLM Gen MoE optimize DeepSeek Fp8 activation kernel (#9175)

- **Date**: 2025-11-21
- **Author**: Nikita Korobov
- **Categories**: Kernel Optimization, Quantization Optimization

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Async/stream-based execution
- FP8 quantization
- Parallelism optimization
- Reduce synchronization overhead
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../trtllmGenKernels/blockScaleMoe/DevKernel.cu    | 308 ++++++++++++++++++---
 .../trtllmGenKernels/blockScaleMoe/DevKernel.h     |  42 ++-
 2 files changed, 306 insertions(+), 44 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.cu b/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.cu
index 1b71483fd..cef2588b5 100644
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.cu
+++ b/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.cu
@@ -100,14 +100,136 @@ __global__ void activationKernel(KernelParams params)
 
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 
+struct Float4Max
+{
+    __device__ __forceinline__ float4 operator()(float4 const& a, float4 const& b) const
+    {
+        float4 result;
+        result.x = fmaxf(a.x, b.x);
+        result.y = fmaxf(a.y, b.y);
+        result.z = fmaxf(a.z, b.z);
+        result.w = fmaxf(a.w, b.w);
+        return result;
+    }
+};
+
+struct Float2Max
+{
+    __device__ __forceinline__ float2 operator()(float2 const& a, float2 const& b) const
+    {
+        float2 result;
+        result.x = fmaxf(a.x, b.x);
+        result.y = fmaxf(a.y, b.y);
+        return result;
+    }
+};
+
+////////////////////////////////////////////////////////////////////////////////////////////////////
+
+template <typename VecType, int size>
+__device__ __forceinline__ VecType packedTypeFromArray(float data[size])
+{
+    return {};
+}
+
+template <>
+__device__ __forceinline__ float4 packedTypeFromArray<float4, 4>(float data[4])
+{
+    float4 result;
+    result.x = data[0];
+    result.y = data[1];
+    result.z = data[2];
+    result.w = data[3];
+    return result;
+}
+
+template <>
+__device__ __forceinline__ float2 packedTypeFromArray<float2, 2>(float data[2])
+{
+    float2 result;
+    result.x = data[0];
+    result.y = data[1];
+    return result;
+}
+
+template <>
+__device__ __forceinline__ float packedTypeFromArray<float, 1>(float data[1])
+{
+    return data[0];
+}
+
+////////////////////////////////////////////////////////////////////////////////////////////////////
+
+template <typename PackedType, int size>
+__device__ __forceinline__ cutlass::Array<float, size> arrayFromPackedType(PackedType data)
+{
+    return cutlass::Array<float, size>{};
+}
+
+template <>
+__device__ __forceinline__ cutlass::Array<float, 4> arrayFromPackedType<float4, 4>(float4 data)
+{
+    return cutlass::Array<float, 4>{data.x, data.y, data.z, data.w};
+}
+
+template <>
+__device__ __forceinline__ cutlass::Array<float, 2> arrayFromPackedType<float2, 2>(float2 data)
+{
+    return cutlass::Array<float, 2>{data.x, data.y};
+}
+
+template <>
+__device__ __forceinline__ cutlass::Array<float, 1> arrayFromPackedType<float, 1>(float data)
+{
+    return cutlass::Array<float, 1>{data};
+}
+
+////////////////////////////////////////////////////////////////////////////////////////////////////
+
+template <int NUM_TOKENS_PER_CTA>
+struct KernelTraits;
+
+template <>
+struct KernelTraits<4>
+{
+    using MaxOp = Float4Max;
+    using PackedType = float4;
+};
+
+template <>
+struct Ke
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## f39e1a8603 - [https://nvbugs/5846489][perf] Apply TE's FP8 per-tensor quantization (#11057)

- **Date**: 2026-02-24
- **Author**: Min Yu
- **Categories**: Quantization Optimization

### Optimization Techniques

- Torch compilation/JIT optimization
- Custom CUDA kernel
- FP8 quantization
- Integer quantization
- KV cache optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/custom_ops/torch_custom_ops.py | 139 +++++++++++++++++++++
 .../trt/quantization/test_fp8_quantization.py      |  42 +++++++
 2 files changed, 181 insertions(+)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py b/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
index 7db85b4b3..f88d13ddb 100644
--- a/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
+++ b/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
@@ -2023,3 +2023,142 @@ def _(
 ) -> torch.Tensor:
     return act_fp4.new_empty((act_fp4.size(0), weight_fp4.size(0)),
                              dtype=output_dtype)
+
+
+class QuantizeE4M3PerTensorRunner(TunableRunner):
+    """
+    Runner for FP8 E4M3 per-tensor quantization with auto-tuning between backends.
+
+    Supports two backends:
+    - "trtllm": TensorRT-LLM's native implementation
+    - "te": Transformer Engine's implementation
+    """
+
+    tuning_config = TuningConfig(
+        dynamic_tensor_specs=(DynamicTensorSpec(
+            0, -2, get_last_power_of_2_num_tokens_buckets,
+            last_positive_power_of_2), ),
+        tune_max_num_tokens=8192,
+    )
+
+    # Lazy init for TE to avoid import errors if not installed
+    _te_available = None
+    _te_quantizer = None
+
+    def __init__(self):
+        super().__init__()
+
+    @classmethod
+    def _check_te_available(cls):
+        """Check if Transformer Engine is available (cached)."""
+        if cls._te_available is None:
+            try:
+                import transformer_engine_torch as tex
+                from transformer_engine.pytorch.tensor.float8_tensor import \
+                    Float8CurrentScalingQuantizer
+                cls._te_available = True
+                # Initialize quantizer once
+                cls._te_quantizer = Float8CurrentScalingQuantizer(
+                    fp8_dtype=tex.DType.kFloat8E4M3, device="cuda")
+            except ImportError:
+                cls._te_available = False
+                logger.warning(
+                    "Transformer Engine not available. Only TRTLLM backend will be used for FP8 quantization."
+                )
+        return cls._te_available
+
+    def get_valid_tactics(self, inputs: List[torch.Tensor],
+                          profile: OptimizationProfile, **kwargs) -> List[int]:
+        """Return list of available backend indices."""
+        tactics = ["trtllm"]
+
+        if self._check_te_available():
+            tactics.append("te")
+
+        return tactics
+
+    def forward(self,
+                inputs: List[torch.Tensor],
+                tactic: str = "trtllm") -> Tuple[torch.Tensor, torch.Tensor]:
+        """
+        Forward pass with backend selection.
+
+        Args:
+            inputs: [input_tensor]
+            tactic: "trtllm" or "te"
+
+        Returns:
+            (quantized_tensor, scale)
+        """
+        input_tensor = inputs[0]
+
+        # Call the appropriate backend
+        if tactic == "te":
+            return self._quantize_te(input_tensor)
+        else:
+            return self._quantize_trtllm(input_tensor)
+
+    def _quantize_trtllm(
+            self, input: torch.Tensor) -> Tuple
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## f3a985ce27 - [TRTLLM-10296][fix] Fix the potential misaligned access due to vectorized ld/st instructions in NVLinkOneSided A2A. (#10539)

- **Date**: 2026-01-20
- **Author**: Bo Li
- **Categories**: Kernel Optimization

### Optimization Techniques

- Vectorized memory access
- Operator fusion
- Integer quantization
- KV cache optimization
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
cpp/tensorrt_llm/thop/moeAlltoAllOp.cpp            | 59 +++++++++++++---------
 tensorrt_llm/_torch/distributed/moe_alltoall.py    | 32 +++++++-----
 .../fused_moe/communication/nvlink_one_sided.py    | 39 +++++++-------
 3 files changed, 74 insertions(+), 56 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/thop/moeAlltoAllOp.cpp b/cpp/tensorrt_llm/thop/moeAlltoAllOp.cpp
index af6d7cb37..29ad780d4 100644
--- a/cpp/tensorrt_llm/thop/moeAlltoAllOp.cpp
+++ b/cpp/tensorrt_llm/thop/moeAlltoAllOp.cpp
@@ -33,6 +33,8 @@ namespace torch_ext
 namespace moe_comm
 {
 
+static constexpr size_t CACHELINE_ALIGNMENT = 128;
+
 // TODO: Is Alignment necessary?
 // Helper function to align offset to specified byte boundary
 inline size_t alignOffset(size_t offset, size_t alignment)
@@ -46,7 +48,6 @@ MoeA2ADataOffsets calculateOffsets(int epSize, int maxNumTokens)
     // TODO: Use lambdas to encapsulate offset and alignment for each entry, which is less error prone and easier to
     // read.
     constexpr size_t SIZEOF_INT32 = 4;
-    constexpr size_t CACHELINE_ALIGNMENT = 128;
 
     MoeA2ADataOffsets offsets;
     size_t offset = 0;
@@ -203,12 +204,18 @@ std::tuple<std::vector<torch::Tensor>, int64_t> moeA2ADispatchOp(torch::Tensor c
         TORCH_CHECK(payload.is_contiguous(), "All payloads must be contiguous");
     }
 
-    // Calculate buffer sizes for all payloads
-    // Each payload buffer needs space for data from ALL ranks: epSize * maxTokensPerRank * elementsPerToken
-    int64_t totalBytesNeeded = 0;
-    std::vector<int64_t> payloadByteSizes;
+    // Record the cacheline aligned start offset for each payload's recv buffer.
+    // 1. We assume the base workspace ptr of each rank is aligned (checked in this OP)
+    // 2. offsets[PAYLOAD_DATA_OFFSET_INDEX] is aligned (ensured in calculateOffsets)
+    // 3. We align the currentOffset during update.
+    // In this way, it is guaranteed that the recv buffer is (over-)aligned, sufficient for 128bit vectorized ld/st.
+
     std::vector<int> payloadElementSizes;
     std::vector<int> payloadElementsPerToken;
+    std::vector<size_t> payloadRecvBufferOffsets;
+
+    // Start offset for the first payload
+    size_t currentOffset = static_cast<size_t>(offsets[PAYLOAD_DATA_OFFSET_INDEX]);
     for (auto const& payload : inputPayloads)
     {
         CHECK_CONTIGUOUS(payload);
@@ -216,16 +223,24 @@ std::tuple<std::vector<torch::Tensor>, int64_t> moeA2ADispatchOp(torch::Tensor c
         TORCH_CHECK(payload.dim() == 2, "payload must be a 2D tensor");
         TORCH_CHECK(
             payload.size(0) == localNumTokens, "payload must have the same first dimension as tokenSelectedExperts");
+        // Unlike recv buffer for payloads, payload itself is not allocated by us and we cannot control its alignment.
+        // We only make sure the payload start offset is 16-byte aligned, while the actual vectorized ld/st width is
+        // dynamically determined based on bytes per token of this payload.
+        TORCH_CHECK(reinterpret_cast<uintptr_t>(payload.data_ptr()) % 16 == 0, "payload must be 16-byte aligned");
 
         int elementsPerToken = static_cast<int>(payload.size(1));
         int elementSize = static_cast<int>(payload.dtype().itemsize());
         // Each payload buffe
```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## f3d784c6f6 - [#10345][perf] Enable multi-stream MOE for super. Also adds multi-stream MLA attn (#11520)

- **Date**: 2026-02-15
- **Author**: Suyog Gupta
- **Categories**: Parallelism/Async

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
- PyTorch built-in optimized ops
- Multi-stream execution
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- Prefill phase

### Changed Files

```
.../model_registry/configs/glm-4.7-flash.yaml      |  14 +
 examples/auto_deploy/super_v3.yaml                 |   2 +-
 .../_torch/auto_deploy/config/default.yaml         |  21 +
 .../auto_deploy/custom_ops/linear/__init__.py      |   2 +
 .../_torch/auto_deploy/custom_ops/linear/swiglu.py | 311 +++++++++++
 tensorrt_llm/_torch/auto_deploy/export/export.py   |  41 +-
 .../models/custom/modeling_glm4_moe_lite.py        |   2 +-
 .../auto_deploy/transform/library/fuse_swiglu.py   | 618 +++++++++++++++++++++
 .../transform/library/fused_add_rms_norm.py        | 166 ++++--
 .../transform/library/multi_stream_attn.py         | 215 +++++++
 .../transform/library/multi_stream_moe.py          | 465 ++++++----------
 tensorrt_llm/_torch/auto_deploy/utils/_graph.py    |  94 ++++
 .../_torch/auto_deploy/utils/multi_stream_utils.py | 242 ++++++++
 .../defs/accuracy/test_llm_api_autodeploy.py       |  14 +-
 .../singlegpu/custom_ops/rope/test_triton_rope.py  |   3 +
 .../unit/singlegpu/custom_ops/test_multi_stream.py | 134 -----
 .../singlegpu/custom_ops/test_multi_stream_attn.py | 230 ++++++++
 .../singlegpu/custom_ops/test_multi_stream_moe.py  | 502 +++++++++++++++++
 .../transformations/library/test_fuse_swiglu.py    | 188 +++++++
 .../library/test_fused_add_rms_norm.py             | 300 ++++++++--
 .../transformations/library/test_nvfp4_swiglu.py   | 378 +++++++++++++
 .../unit/singlegpu/transformations/test_export.py  | 182 +++---
 .../utils/test_create_derived_custom_op.py         | 163 ++++++
 23 files changed, 3673 insertions(+), 614 deletions(-)
```

### Diff Preview

```diff
diff --git a/examples/auto_deploy/model_registry/configs/glm-4.7-flash.yaml b/examples/auto_deploy/model_registry/configs/glm-4.7-flash.yaml
index 9d8e16234..79090f279 100644
--- a/examples/auto_deploy/model_registry/configs/glm-4.7-flash.yaml
+++ b/examples/auto_deploy/model_registry/configs/glm-4.7-flash.yaml
@@ -4,5 +4,19 @@ max_seq_len: 4096
 enable_chunked_prefill: true
 cuda_graph_batch_sizes: [1, 2, 4, 8, 16, 32, 64]
 transforms:
+  match_swiglu_pattern:
+    enabled: true
+  match_nvfp4_swiglu_pattern:
+    enabled: true
   fuse_nvfp4_moe:
     allow_different_input_scales: true
+  fuse_nvfp4_swiglu:
+    enabled: true
+  fuse_swiglu:
+    enabled: true
+  multi_stream_moe:
+    stage: compile
+    enabled: true
+  multi_stream_mla_attn:
+    stage: compile
+    enabled: true
diff --git a/examples/auto_deploy/super_v3.yaml b/examples/auto_deploy/super_v3.yaml
index 13b536a63..d0968245e 100644
--- a/examples/auto_deploy/super_v3.yaml
+++ b/examples/auto_deploy/super_v3.yaml
@@ -37,7 +37,7 @@ transforms:
         "fc2_latent_proj": "gather"
   multi_stream_moe:
     stage: compile
-    enabled: false
+    enabled: true
   gather_logits_before_lm_head:
     # TODO: fix https://github.com/NVIDIA/TensorRT-LLM/issues/9878 to enable by default
     enabled: true
diff --git a/tensorrt_llm/_torch/auto_deploy/config/default.yaml b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
index 84aca711e..7ff030f10 100644
--- a/tensorrt_llm/_torch/auto_deploy/config/default.yaml
+++ b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
@@ -50,6 +50,7 @@ transforms:
     expected_layout: bsnd
   match_rmsnorm_pattern:
     stage: pattern_matcher
+    run_shape_prop: true
   match_l2norm_pattern:
     stage: pattern_matcher
   ############################################################################################
@@ -75,6 +76,18 @@ transforms:
     stage: pattern_matcher
   quantize_nvfp4_from_graph:
     stage: pattern_matcher
+  # SwiGLU pattern matching must run AFTER quantization transforms. For pre-quantized
+  # checkpoints (e.g., NVFP4), quantization converts torch_linear_simple ops to quantized
+  # ops first, and then match_nvfp4_swiglu_pattern captures the NVFP4 SwiGLU pattern.
+  # For non-quantized models, quantization transforms are no-ops, so match_swiglu_pattern
+  # proceeds normally.
+  match_swiglu_pattern:
+    stage: pattern_matcher
+    enabled: false
+  match_nvfp4_swiglu_pattern:
+    stage: pattern_matcher
+    requires_shape_prop: true
+    enabled: false
   quantize_fp8_moe:
     stage: pattern_matcher
   quantize_nvfp4_moe:
@@ -126,6 +139,8 @@ transforms:
   fuse_nvfp4_linear:
     stage: post_load_fusion
     backend: trtllm
+  fuse_nvfp4_swiglu:
+    stage: post_load_fusion
   fuse_moe:
     stage: post_load_fusion
     expect_mem_change: true
@@ -149,6 +164,9 @@ transforms:
   fuse_l2norm:
     stage: post_load_fusion
     backend: fla
+  fuse_swiglu:
+    stage: post_load_fusion
+    enabled: false
   fuse_add_rms_norm:
 
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Multi-stream execution enables parallel execution of independent operations on the GPU. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## f512ddaeef - [None][feat] add skip condition in AutoDeploy's triton fused moe kernel (#8632)

- **Date**: 2025-10-24
- **Author**: Suyog Gupta
- **Categories**: Kernel Optimization, Fusion

### Optimization Techniques

- Operator fusion
- Triton kernel
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/triton_moe.py | 6 ++++++
 1 file changed, 6 insertions(+)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/triton_moe.py b/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/triton_moe.py
index 6fd04f509..625e588a1 100644
--- a/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/triton_moe.py
+++ b/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/triton_moe.py
@@ -41,6 +41,7 @@ def fused_mlp_moe_kernel(
     topk_weights_ptr,
     sorted_token_ids_ptr,
     expert_ids_ptr,
+    num_tokens_post_padded_ptr,
     # Matrix dimensions
     N,
     K,
@@ -84,6 +85,10 @@ def fused_mlp_moe_kernel(
     pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
     pid_n = (pid % num_pid_in_group) // group_size_m
 
+    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
+    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
+        return
+
     offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
     # Bounds check: EM might not be a multiple of BLOCK_SIZE_M
     # so offs_token_id can exceed EM-1. Load with mask to avoid out-of-bounds.
@@ -270,6 +275,7 @@ def _invoke_kernel(
         topk_weights if topk_weights is not None else C,
         sorted_token_ids,
         expert_ids,
+        num_tokens_post_padded,
         B.size(1),
         B.size(2),
         EM,

```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## f5b6d453aa - doc： DS r1 min latency blog (#4386)

- **Date**: 2025-05-16
- **Author**: Kefeng-Duan
- **Categories**: Throughput/Latency

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Speculative decoding
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
docs/source/blogs/media/tech_blog1_fuse_a_gemm.png | Bin 0 -> 32914 bytes
 .../blogs/media/tech_blog1_model_details.png       | Bin 0 -> 157196 bytes
 .../blogs/media/tech_blog1_model_overview.png      | Bin 0 -> 52407 bytes
 docs/source/blogs/media/tech_blog1_router_gemm.png | Bin 0 -> 32940 bytes
 .../media/tech_blog1_sparse_exp_as_a_gemm.png      | Bin 0 -> 122299 bytes
 ..._DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.md | 266 +++++++++++++++++++++
 docs/source/index.rst                              |   1 +
 7 files changed, 267 insertions(+)
```

### Diff Preview

```diff
diff --git a/docs/source/blogs/media/tech_blog1_fuse_a_gemm.png b/docs/source/blogs/media/tech_blog1_fuse_a_gemm.png
new file mode 100644
index 000000000..0547d017e
Binary files /dev/null and b/docs/source/blogs/media/tech_blog1_fuse_a_gemm.png differ
diff --git a/docs/source/blogs/media/tech_blog1_model_details.png b/docs/source/blogs/media/tech_blog1_model_details.png
new file mode 100644
index 000000000..1a5907791
Binary files /dev/null and b/docs/source/blogs/media/tech_blog1_model_details.png differ
diff --git a/docs/source/blogs/media/tech_blog1_model_overview.png b/docs/source/blogs/media/tech_blog1_model_overview.png
new file mode 100644
index 000000000..87b20335c
Binary files /dev/null and b/docs/source/blogs/media/tech_blog1_model_overview.png differ
diff --git a/docs/source/blogs/media/tech_blog1_router_gemm.png b/docs/source/blogs/media/tech_blog1_router_gemm.png
new file mode 100644
index 000000000..48c7184fc
Binary files /dev/null and b/docs/source/blogs/media/tech_blog1_router_gemm.png differ
diff --git a/docs/source/blogs/media/tech_blog1_sparse_exp_as_a_gemm.png b/docs/source/blogs/media/tech_blog1_sparse_exp_as_a_gemm.png
new file mode 100644
index 000000000..25f6bf396
Binary files /dev/null and b/docs/source/blogs/media/tech_blog1_sparse_exp_as_a_gemm.png differ
diff --git a/docs/source/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.md b/docs/source/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.md
new file mode 100644
index 000000000..6a19c021e
--- /dev/null
+++ b/docs/source/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.md
@@ -0,0 +1,266 @@
+# Pushing Latency Boundaries: Optimizing DeepSeek-R1 Performance on NVIDIA B200 GPUs
+by NVIDIA TensorRT-LLM team
+## Table of Contents
+
+- [Background](#background)
+- [Implementation Configuration](#implementation-configuration)
+  - [Workload Profile](#workload-profile)
+  - [Model Architecture](#model-architecture)
+  - [Precision Strategy](#precision-strategy)
+  - [Parallelism Strategy](#parallelism-strategy)
+  - [Everything in One Diagram](#everything-in-one-diagram)
+- [Key Optimizations](#key-optimizations)
+  - [System Level optimizations](#system-level-optimizations)
+    - [CUDA Graph & Programmatic Dependent Launch](#cuda-graph--programmatic-dependent-launch)
+    - [MTP](#mtp)
+      - [Autoregressive MTP Layers](#autoregressive-mtp-layers)
+      - [Relax Acceptance Verification](#relax-acceptance-verification)
+    - [Multi-streams](#multi-streams)
+    - [Sparse Experts as GEMMs](#sparse-experts-as-gemms-only-works-when-moe_backendcutlass)
+    - [Re-balanced the sparse experts](#re-balanced-the-sparse-experts)
+      - [Mixed ETP](#mixed-etp)
+      - [Smart Router](#smart-router)
+  - [Kernel Level optimizations](#kernel-level-optimizations)
+    - [Attention Kernel](#attention-kernel)
+    - [Group
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## f6654f26a4 - [#5255][autodeploy] Update FuseAllreduceResidualRMSNorm to use pattern matcher utility; remove fuse_collective (#7545)

- **Date**: 2025-10-05
- **Author**: Frida Hou
- **Categories**: Fusion

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
.../_torch/auto_deploy/config/default.yaml         |   6 +-
 .../_torch/auto_deploy/transform/interface.py      |  10 +-
 .../auto_deploy/transform/library/attention.py     |   3 -
 .../auto_deploy/transform/library/collectives.py   | 245 +++++++--------------
 .../auto_deploy/transform/library/fused_moe.py     |   6 +-
 .../_torch/auto_deploy/transform/library/fusion.py |  11 +-
 .../auto_deploy/transform/library/quantization.py  |   2 +-
 .../_torch/auto_deploy/transformations/_graph.py   |  31 +++
 .../_torch/auto_deploy/utils/sharding_utils.py     |   8 +-
 .../test_allreduce_residual_rmsnorm_fusion.py      |  31 ++-
 .../library/test_collective_fusion.py              | 108 ---------
 .../library/test_attention_matcher.py              |   1 +
 .../library/test_attention_matcher_hf.py           |   1 +
 13 files changed, 161 insertions(+), 302 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/config/default.yaml b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
index 37f314c00..1684fe08c 100644
--- a/tensorrt_llm/_torch/auto_deploy/config/default.yaml
+++ b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
@@ -32,6 +32,7 @@ transforms:
     stage: pattern_matcher
   match_repeat_kv:
     stage: pattern_matcher
+    run_shape_prop: true
   match_eager_attention:
     stage: pattern_matcher
   match_grouped_attention:
@@ -111,13 +112,14 @@ transforms:
     enabled: true
   fuse_allreduce_residual_rmsnorm:
     stage: post_load_fusion
-  fuse_collectives:
-    stage: post_load_fusion
+  # TODO (lucaslie): add backend selection as part of configurable inference optimizers
+  # check if we can fuse rmsnorm
   fuse_rmsnorm:
     # TODO (lucaslie): add backend selection as part of configurable inference optimizers
     # check if we can fuse rmsnorm
     stage: post_load_fusion
     backend: flashinfer
+    requires_shape_prop: true
   ############################################################################################
   # SWITCH TO CACHED+FLATTENED ATTENTION + INITIALIZE CACHES
   ############################################################################################
diff --git a/tensorrt_llm/_torch/auto_deploy/transform/interface.py b/tensorrt_llm/_torch/auto_deploy/transform/interface.py
index 3af556cc4..a0895b61d 100644
--- a/tensorrt_llm/_torch/auto_deploy/transform/interface.py
+++ b/tensorrt_llm/_torch/auto_deploy/transform/interface.py
@@ -5,6 +5,7 @@ This module defines the base classes and interfaces for all transforms.
 
 import time
 from abc import ABC, abstractmethod
+from contextlib import nullcontext
 from enum import Enum
 from functools import total_ordering, wraps
 from typing import Any, Callable, Dict, Mapping, Tuple, Type, Union, final
@@ -19,6 +20,7 @@ from ..transformations._graph import (
     canonicalize_graph,
     lift_to_meta,
     named_graphmodules,
+    placeholders_on_meta,
     run_shape_prop,
 )
 from ..utils.logger import ad_logger
@@ -416,11 +418,13 @@ class BaseTransform(ABC):
         is_clean = info.is_clean
         has_valid_shapes = is_clean and info.has_valid_shapes
 
+        use_meta = isinstance(gm, GraphModule) and placeholders_on_meta(gm)
+
         # check if run cleanup depending on the config and info
         if self.config.requires_shape_prop and not has_valid_shapes:
             self._log_info("running pre-cleanup with shape_prop")
             canonicalize_graph(gm)
-            with lift_to_meta(gm):
+            with lift_to_meta(gm) if use_meta else nullcontext():
                 run_shape_prop(gm)
             is_clean = True
             has_valid_shapes = True
@@ -444,11 +448,13 @@ class BaseTransform(ABC):
         if not self.config.run_graph_cleanup:
             return info
 
+        use_meta = isinstance(gm, GraphModule) and placeholders_on_meta(gm)
+
         # check if run cleanup depending on the config a
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## f670a036df - [Qwen3] chore: fix bug of fused_moe on tp > 1 (#4093)

- **Date**: 2025-05-07
- **Author**: bhsueh_NV
- **Categories**: Fusion

### Optimization Techniques

- Operator fusion
- Parallelism optimization
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/modules/fused_moe.py | 5 +----
 1 file changed, 1 insertion(+), 4 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fused_moe.py b/tensorrt_llm/_torch/modules/fused_moe.py
index fe69d899b..f2728fbf9 100755
--- a/tensorrt_llm/_torch/modules/fused_moe.py
+++ b/tensorrt_llm/_torch/modules/fused_moe.py
@@ -821,10 +821,7 @@ class FusedMoE(nn.Module):
             final_hidden_states = final_hidden_states[0]
 
         if not self.enable_alltoall:
-            if self.reduce_results and self.parallel_size > 1:
-                return self.all_reduce(final_hidden_states)
-            else:
-                return final_hidden_states
+            return final_hidden_states
         else:
             return self.alltoall_combine(final_hidden_states, alltoall_info,
                                          token_count)

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## f77aca9f2c - [TRTLLM-7385][feat] Optimize Qwen2/2.5-VL performance (#7250)

- **Date**: 2025-09-22
- **Author**: Yechan Kim
- **Categories**: General Performance

### Optimization Techniques

- Torch compilation/JIT optimization
- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Pinned memory
- Speculative decoding
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- Decode/generation phase
- Disaggregated serving

### Changed Files

```
examples/llm-api/quickstart_multimodal.py          |   8 +
 tensorrt_llm/_torch/attention_backend/interface.py |   3 +
 tensorrt_llm/_torch/models/checkpoints/__init__.py |   3 +-
 .../models/checkpoints/hf/qwen2vl_weight_mapper.py |  22 +
 tensorrt_llm/_torch/models/modeling_gpt_oss.py     |   4 +-
 tensorrt_llm/_torch/models/modeling_hunyuan_moe.py |   2 -
 tensorrt_llm/_torch/models/modeling_llama.py       |  20 +-
 .../_torch/models/modeling_llama_min_latency.py    |   5 +-
 tensorrt_llm/_torch/models/modeling_qwen.py        |   3 +-
 tensorrt_llm/_torch/models/modeling_qwen2vl.py     | 738 ++++++++++++++++-----
 tensorrt_llm/_torch/models/modeling_qwen3.py       |   6 +-
 tensorrt_llm/_torch/modules/attention.py           |  20 +-
 tensorrt_llm/_torch/modules/rotary_embedding.py    |  75 ++-
 .../_torch/pyexecutor/cuda_graph_runner.py         |  49 +-
 tensorrt_llm/_torch/pyexecutor/model_engine.py     | 117 ++--
 tensorrt_llm/_torch/pyexecutor/resource_manager.py |  17 +-
 tensorrt_llm/inputs/multimodal.py                  |  89 ++-
 tensorrt_llm/serve/scripts/benchmark_dataset.py    |   3 -
 tests/integration/test_lists/test-db/l0_l40s.yml   |   7 +-
 .../_torch/modeling/test_modeling_qwen2_5vl.py     | 532 +++++++++++++++
 .../_torch/modules/test_rotary_embedding.py        | 160 +++++
 .../_torch/multimodal/test_share_multiparams.py    |  19 +
 22 files changed, 1615 insertions(+), 287 deletions(-)
```

### Diff Preview

```diff
diff --git a/examples/llm-api/quickstart_multimodal.py b/examples/llm-api/quickstart_multimodal.py
index 062fdaa43..526bcf3ab 100644
--- a/examples/llm-api/quickstart_multimodal.py
+++ b/examples/llm-api/quickstart_multimodal.py
@@ -300,6 +300,14 @@ def main():
         prompt = args.prompt[i]
         generated_text = output.outputs[0].text
         print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")
+        if args.return_context_logits:
+            print(f"[{i}] Context logits: {output.context_logits}")
+        if args.return_generation_logits:
+            print(
+                f"[{i}] Generation logits: {output.outputs[0].generation_logits}"
+            )
+        if args.logprobs:
+            print(f"[{i}] Logprobs: {output.outputs[0].logprobs}")
 
 
 if __name__ == "__main__":
diff --git a/tensorrt_llm/_torch/attention_backend/interface.py b/tensorrt_llm/_torch/attention_backend/interface.py
index e960dcf6b..cdbe7c8c9 100644
--- a/tensorrt_llm/_torch/attention_backend/interface.py
+++ b/tensorrt_llm/_torch/attention_backend/interface.py
@@ -511,6 +511,9 @@ class PositionalEmbeddingParams:
     rope: Optional[RopeParams] = None
     is_neox: bool = True
 
+    # mRoPE params (currently, Qwen2/2.5-VL uses it)
+    mrope_section: Optional[List[int]] = None
+
     def __post_init__(self) -> None:
         if self.type.is_deferred():
             assert self.embedder is not None, f"{self.type} requires a not-none external embedder"
diff --git a/tensorrt_llm/_torch/models/checkpoints/__init__.py b/tensorrt_llm/_torch/models/checkpoints/__init__.py
index 58789f364..6718e5749 100644
--- a/tensorrt_llm/_torch/models/checkpoints/__init__.py
+++ b/tensorrt_llm/_torch/models/checkpoints/__init__.py
@@ -6,6 +6,7 @@ from .hf.llama4_weight_mapper import Llama4HfWeightMapper
 from .hf.mixtral_weight_mapper import MixtralHfWeightMapper
 from .hf.nemotron_h_weight_mapper import NemotronHHfWeightMapper
 from .hf.qwen2_moe_weight_mapper import Qwen2MoeHfWeightMapper
+from .hf.qwen2vl_weight_mapper import Qwen2VLHfWeightMapper
 from .hf.qwen3_moe_weight_mapper import Qwen3MoeHfWeightMapper
 from .hf.weight_loader import HfWeightLoader
 from .hf.weight_mapper import HfWeightMapper
@@ -14,5 +15,5 @@ __all__ = [
     "HfConfigLoader", "HfWeightLoader", "HfWeightMapper",
     "BaseCheckpointLoader", "HfCheckpointLoader", "NemotronHHfWeightMapper",
     "Gemma3HfWeightMapper", "MixtralHfWeightMapper", "Llama4HfWeightMapper",
-    "Qwen2MoeHfWeightMapper", "Qwen3MoeHfWeightMapper"
+    "Qwen2MoeHfWeightMapper", "Qwen3MoeHfWeightMapper", "Qwen2VLHfWeightMapper"
 ]
diff --git a/tensorrt_llm/_torch/models/checkpoints/hf/qwen2vl_weight_mapper.py b/tensorrt_llm/_torch/models/checkpoints/hf/qwen2vl_weight_mapper.py
new file mode 100644
index 000000000..f7fe951ec
--- /dev/null
+++ b/tensorrt_llm/_torch/models/checkpoints/hf/qwen2vl_weight_mapper.py
@@ -0,0 +1,22 @@
+from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
+    HfWeigh
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Pinned (page-locked) memory enables faster host-to-device data transfers. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## f7c597ec40 - [None][perf] Make finalize fusion part of the tactic selection logic (#6915)

- **Date**: 2025-08-22
- **Author**: Daniel Stokes
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
- Batching optimization
- MoE optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- NVIDIA Hopper (SM90) GPU

### Changed Files

```
.../mixtureOfExpertsBackendBenchmarkFixture.h      |  21 +-
 .../mixtureOfExpertsBackendBenchmarkLauncher.cu    | 250 ++++-----------------
 .../include/cutlass_extensions/gemm_configs.h      |  14 +-
 .../cutlass_kernels/include/moe_gemm_kernels.h     |  17 +-
 .../kernels/cutlass_kernels/include/moe_kernels.h  |  36 +--
 .../cutlass_kernels/int8_gemm/int8_gemm_template.h |   1 -
 .../moe_gemm/moe_gemm_template_dispatch.h          |  37 ++-
 .../cutlass_kernels/moe_gemm/moe_kernels.cu        |  55 +++--
 ...orrt_llm_internal_cutlass_kernels_static.tar.xz |   4 +-
 .../aarch64-linux-gnu/version.txt                  |   4 +-
 ...orrt_llm_internal_cutlass_kernels_static.tar.xz |   4 +-
 .../x86_64-linux-gnu/version.txt                   |   4 +-
 .../mixtureOfExperts/mixtureOfExpertsPlugin.cpp    |   6 +-
 .../mixtureOfExperts/mixtureOfExpertsPlugin.h      |   1 +
 cpp/tensorrt_llm/thop/moeOp.cpp                    |  29 ++-
 .../unit_tests/kernels/mixtureOfExpertsTest.cu     |  56 +++--
 tensorrt_llm/_torch/autotuner.py                   |   3 +-
 tensorrt_llm/_torch/custom_ops/torch_custom_ops.py |  44 ++--
 .../_torch/custom_ops/trtllm_gen_custom_ops.py     |  35 +--
 tests/unittest/_torch/misc/test_autotuner.py       |  13 +-
 20 files changed, 263 insertions(+), 371 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h b/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h
index 2559ae548..36cbe7654 100644
--- a/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h
+++ b/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h
@@ -833,7 +833,7 @@ public:
     // Runs for 3 iterations or 1 second and picks the best option
     int pickBestTactic(MOEParallelismConfig parallelism_config, GemmToProfile gemm_to_profile)
     {
-        auto tactics = mMoERunner.getTactics();
+        auto tactics = mMoERunner.getTactics(static_cast<MoeGemmId>(gemm_to_profile));
         ::nvtx3::scoped_range nvtx(tensorrt_llm::common::nvtx::nextColor(),
             "Tactic Profiling GEMM " + std::to_string(static_cast<int>(gemm_to_profile)));
         // We save space by reusing the same workspace buffer for all tactics when doing full layer profiling. So we
@@ -925,12 +925,14 @@ public:
     std::pair<int, int> setTactic(
         int tactic_idx1, int tactic_idx2, MOEParallelismConfig parallelism_config, GemmToProfile gemm_to_profile)
     {
-        auto tactics = mMoERunner.getTactics();
+        auto tactics1 = mMoERunner.getTactics(MoeGemmId::GEMM_1);
+        auto tactics2 = mMoERunner.getTactics(MoeGemmId::GEMM_2);
         std::vector<std::pair<std::reference_wrapper<int>, GemmToProfile>> tactics_to_profile{
             {tactic_idx1, GemmToProfile::GEMM_1}, {tactic_idx2, GemmToProfile::GEMM_2}};
         for (auto& combo : tactics_to_profile)
         {
             auto& t = combo.first.get();
+            auto& tactics = combo.second == GemmToProfile::GEMM_1 ? tactics1 : tactics2;
             if (combo.second != gemm_to_profile && gemm_to_profile != GemmToProfile::LAYER)
             {
                 t = 0; // Unneeded tactic, set to 0
@@ -947,7 +949,7 @@ public:
             }
         }
 
-        mMoERunner.setTactic(tactics[tactic_idx1], tactics[tactic_idx2]);
+        mMoERunner.setTactic(tactics1[tactic_idx1], tactics2[tactic_idx2]);
         mBestTacticGemm1 = tactic_idx1;
         mBestTacticGemm2 = tactic_idx2;
         return {tactic_idx1, tactic_idx2};
@@ -965,7 +967,7 @@ public:
             auto expert_weights_size
                 = gemm_to_profile == GemmToProfile::GEMM_1 ? mExpertWeight1Size : mExpertWeight2Size;
 
-            auto tactics = mMoERunner.getTactics()[tactic_idx];
+            auto tactics = mMoERunner.getTactics(static_cast<MoeGemmId>(gemm_to_profile))[tactic_idx];
             if (static_cast<int>(gemm_to_profile) != static_cast<int>(mGemmProfilerBackend.mGemmToProfile))
             {
                 throw std::runtime_error("Configuration mismatch between mGemmProfilerBackend and runMoEPermute");
@@ -1074,11 +1076,12 @@ void MixtureOfExpertsBenchmark<TypeTuple_>::runBenchmark(benchmark::State& state
     }
     if (LOG_LEVEL >= INFO)
     {
-        auto tactics = mMoERunner.getTactics();
-        std::cout << "Selected tactic #1: " 
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## f7e245668b - [TRTLLM-9680][perf] Optimize TRTLLMSampler log_probs performance (Core fix has been merged via #9353) (#9655)

- **Date**: 2025-12-17
- **Author**: Yuan Tong
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- Batching optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/sampler.py | 64 ++++++++++++++++---------------
 1 file changed, 34 insertions(+), 30 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index 0aefd2a43..b9bbb7cbf 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -3088,13 +3088,8 @@ class TRTLLMSampler(Sampler, AsyncWorkerMixin):
     @nvtx_range("update_requests_single_beam_single_step")
     def update_requests_single_beam_single_step(self, state: SampleStateTRTLLM):
         """Specialization of update_requests for single beam and single step"""
-        new_tokens_host = state.host.new_tokens.flatten().tolist()
         sequence_lengths_host_data = state.host.sequence_lengths.flatten().tolist()
         finish_reasons = state.host.finish_reasons.flatten().tolist()
-        log_probs_host_tensor = state.host.log_probs
-        cum_log_probs_host = (
-            state.host.cum_log_probs.tolist() if state.host.cum_log_probs is not None else None
-        )
 
         reqs = [
             r for r in state.scheduled_requests.context_requests if not r.is_context_init_state
@@ -3104,44 +3099,53 @@ class TRTLLMSampler(Sampler, AsyncWorkerMixin):
             if not r.is_generation_complete_state
         ]
 
-        reqs_with_new_tokens = [
-            r for r in reqs if (sequence_lengths_host_data[r.py_seq_slot] > r.get_num_tokens(0))
-        ]
+        # NB: To ensure good performance, we must
+        #  1. Avoid accessing torch.Tensor object inside the for-each-request loops
+        #  2. Convert only necessary data to Python list
 
         # Add new tokens
-        new_tokens = [new_tokens_host[r.py_seq_slot] for r in reqs_with_new_tokens]
+        reqs_with_new_tokens = []
+        seq_slots = []
+        seq_slots_need_log_probs = []
+        for request in reqs:
+            if sequence_lengths_host_data[request.py_seq_slot] <= request.get_num_tokens(0):
+                continue
+
+            reqs_with_new_tokens.append(request)
+            seq_slots.append(request.py_seq_slot)
+
+            if request.py_return_log_probs:
+                seq_slots_need_log_probs.append(request.py_seq_slot)
+
+        # [maxTokensPerStep, batchSize, maxBeamWidth]
+        new_tokens = state.host.new_tokens[0, seq_slots, 0].tolist()
         add_new_tokens_to_requests(reqs_with_new_tokens, new_tokens, 0)
 
         # Log probs
-        if log_probs_host_tensor is not None:
-            # Log probs
-            seq_slots = []
-            seq_lens = []
-            for request in reqs_with_new_tokens:
-                if request.py_return_log_probs:
-                    seq_slot = request.py_seq_slot
-                    seq_slots.append(seq_slot)
-                    seq_lens.append(sequence_lengths_host_data[seq_slot] - 1)
-
-            log_probs_host = log_probs_host_tensor[seq_slots, 0, seq_lens].tolist()
-            idx = 0
-            for request in reqs_with_new_tokens:
+        if state.host.log_probs is not None:
+            # [batchSize, maxBeamWidth]
+            se
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## f8a4cc0629 - perf: Add total token throughput metric. (#3212)

- **Date**: 2025-04-04
- **Author**: Frank
- **Categories**: Throughput/Latency

### Optimization Techniques

- General code optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/bench/dataclasses/reporting.py  | 9 +++++++++
 tensorrt_llm/bench/dataclasses/statistics.py | 5 +++++
 2 files changed, 14 insertions(+)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/bench/dataclasses/reporting.py b/tensorrt_llm/bench/dataclasses/reporting.py
index 373d9c47b..ad4bf9ed9 100755
--- a/tensorrt_llm/bench/dataclasses/reporting.py
+++ b/tensorrt_llm/bench/dataclasses/reporting.py
@@ -194,6 +194,12 @@ class ReportUtility:
         """Output throughput in tokens per second."""
         return self.convert_rate_to_s(self.statistics.output_throughput_tok_ns)
 
+    @property
+    def total_token_throughput_tok_s(self) -> float:
+        """Total token throughput in tokens per second."""
+        return self.convert_rate_to_s(
+            self.statistics.total_token_throughput_tok_ns)
+
     @property
     def per_user_generation_token_throughput_s(self) -> float:
         """Output throughput per user in tokens per second."""
@@ -314,6 +320,8 @@ class ReportUtility:
             "system_output_throughput_tok_s":
             self.output_throughput_tok_s,
             # Output throughput per user (average per request output throughput)
+            "system_total_throughput_tok_s":
+            self.total_token_throughput_tok_s,
             "output_throughput_per_user_tok_s":
             self.per_user_output_throughput_tok_s,
             # Output throughput per GPU (total throughput / world size)
@@ -477,6 +485,7 @@ class ReportUtility:
             f"Total Output Throughput (tokens/sec):             {perf['system_output_throughput_tok_s']:.4f}\n"
             f"Per User Output Throughput (tokens/sec/user):     {perf['output_throughput_per_user_tok_s']:.4f}\n"
             f"Per GPU Output Throughput (tokens/sec/gpu):       {perf['output_throughput_per_gpu_tok_s']:.4f}\n"
+            f"Total Token Throughput (tokens/sec):              {perf['system_total_throughput_tok_s']:.4f}\n"
             f"Total Latency (ms):                               {perf['total_latency_ms']:.4f}\n"
             f"Average request latency (ms):                     {perf['avg_request_latency_ms']:.4f}\n"
         )
diff --git a/tensorrt_llm/bench/dataclasses/statistics.py b/tensorrt_llm/bench/dataclasses/statistics.py
index 0aa88d555..84a1ac089 100644
--- a/tensorrt_llm/bench/dataclasses/statistics.py
+++ b/tensorrt_llm/bench/dataclasses/statistics.py
@@ -183,6 +183,11 @@ class BenchmarkStatistics(BaseModel):
     def output_throughput_tok_ns(self) -> float:
         return float(self.total_output_tokens) / self.total_latency_ns
 
+    @computed_field
+    def total_token_throughput_tok_ns(self) -> float:
+        return float(self.total_input_tokens +
+                     self.total_output_tokens) / self.total_latency_ns
+
     @computed_field
     def output_throughput_tok_ns_per_user(self) -> float:
         return self.output_throughput_percentiles.average

```

### Analysis Summary

This commit applies performance optimization techniques to improve inference efficiency. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## f8dd494536 - [None][perf] Helix: improve all-to-all perf for large CP size (#9494)

- **Date**: 2025-11-28
- **Author**: Matthias Jouanneaux
- **Categories**: General Performance

### Optimization Techniques

- Async/stream-based execution
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
cpp/tensorrt_llm/thop/alltoallOp.cpp     |  4 ++++
 tensorrt_llm/_torch/distributed/ops.py   |  1 +
 tensorrt_llm/_torch/modules/attention.py | 32 ++++++++++++++------------------
 3 files changed, 19 insertions(+), 18 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/thop/alltoallOp.cpp b/cpp/tensorrt_llm/thop/alltoallOp.cpp
index dc7d21939..fdc691575 100644
--- a/cpp/tensorrt_llm/thop/alltoallOp.cpp
+++ b/cpp/tensorrt_llm/thop/alltoallOp.cpp
@@ -64,6 +64,10 @@ public:
         // note: ensures that input_list size > 0
         TLLM_CHECK_WITH_INFO(static_cast<int>(input_list.size()) == num_ranks * num_lists_,
             "input_list size should be equal to group size * num_lists");
+        for (auto const& input : input_list)
+        {
+            TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
+        }
         std::vector<torch::Tensor> output_list(static_cast<size_t>(num_lists_));
         auto stream = at::cuda::getCurrentCUDAStream(input_list[0].get_device());
         ncclGroupStart();
diff --git a/tensorrt_llm/_torch/distributed/ops.py b/tensorrt_llm/_torch/distributed/ops.py
index 4d7853794..552f6b899 100644
--- a/tensorrt_llm/_torch/distributed/ops.py
+++ b/tensorrt_llm/_torch/distributed/ops.py
@@ -336,6 +336,7 @@ def alltoall_helix(
         inputs (List[Tensor]): The input tensors.
             Its length must be a multiple of the group size,
             and all tensors in a group must have the same shape.
+            All input tensors must be contiguous.
         group (List[int]): The group of ranks to participate in the all-to-all.
     Returns:
         The output tensors.
diff --git a/tensorrt_llm/_torch/modules/attention.py b/tensorrt_llm/_torch/modules/attention.py
index 3472c064f..8aa0a46b9 100644
--- a/tensorrt_llm/_torch/modules/attention.py
+++ b/tensorrt_llm/_torch/modules/attention.py
@@ -1072,24 +1072,20 @@ class MLA(nn.Module):
             # similar to the post-processing of ring attention
             kv_lora_rank = partial_o.shape[-1] // self.num_heads_tp
             assert self.kv_lora_rank == kv_lora_rank
-            chunks_o = [
-                t.contiguous() for t in torch.split(partial_o,
-                                                    partial_o.shape[-1] //
-                                                    self.mapping.cp_size,
-                                                    dim=-1)
-            ]
-            chunks_stats = [
-                t.contiguous() for t in torch.split(softmax_stats,
-                                                    softmax_stats.shape[1] //
-                                                    self.mapping.cp_size,
-                                                    dim=1)
-            ]
-            gathered_o, gathered_stats = alltoall_helix(
-                chunks_o + chunks_stats,
-                self.mapping.cp_group,
-            )
-            return torch.ops.trtllm.helix_post_process(gathered_o,
-                                                       gathered_stats, 1.0)
+            # transpose the tensors to make the split across cp_size contiguous
+            # for both tensors, we need to split across the second dimension
+            chunks = []
+          
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## f98fa0cf8b - [None][feat] Optimize kv cache transfer TEP (#7613)

- **Date**: 2025-09-26
- **Author**: Chuang Zhu
- **Categories**: Cache Optimization

### Optimization Techniques

- Async/stream-based execution
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase
- Disaggregated serving

### Changed Files

```
cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp  |  7 +-
 .../batch_manager/cacheTransBuffer.cpp             |  4 +-
 cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp | 88 ++++++++++++++++------
 .../batch_manager/mlaCacheFormatter.cpp            | 22 ++++--
 cpp/tensorrt_llm/common/envUtils.cpp               |  8 +-
 .../batch_manager/cacheTransBufferTest.cpp         | 10 +--
 .../unit_tests/multi_gpu/cacheTransceiverTest.cpp  | 21 +++++-
 docs/source/features/disagg-serving.md             |  3 +-
 .../legacy/advanced/disaggregated-service.md       |  3 +-
 9 files changed, 113 insertions(+), 53 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp b/cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp
index 306cd6418..168ea8969 100644
--- a/cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp
+++ b/cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp
@@ -90,9 +90,9 @@ bool CacheFormatter::needSendCache(
             = selfConfig.getParallelConfig().mTensorParallelism / selfConfig.getParallelConfig().mDPsize;
         selfTpRankInDpGroup = selfTpRank % selfTPNumInDPGroup;
     }
+    int destDPRank = destConfig.getParallelConfig().mEnableAttentionDP ? destConfig.getParallelConfig().mDPrank : 0;
 
-    // only TP rank % dupHeadFactor == 0 need to send cache.
-    return selfTpRankInDpGroup % targetInfo.mDupHeadFactor == 0;
+    return (destDPRank % targetInfo.mDupHeadFactor) == (selfTpRankInDpGroup % targetInfo.mDupHeadFactor);
 }
 
 void checkAlternateWindow(BaseKVCacheManager* cacheManager, BaseCacheFormatter::CacheState const& selfConfig,
@@ -140,11 +140,12 @@ std::vector<size_t> CacheFormatter::pickRecvConnections(
         return ret;
     }
     TLLM_CHECK(numConnections == targetInfo.mIRanks.size());
+    int selfDPRank = selfConfig.getParallelConfig().mEnableAttentionDP ? selfConfig.getParallelConfig().mDPrank : 0;
 
     std::vector<size_t> ret;
     for (int i = 0; i < targetInfo.mDomainTPSize; i++)
     {
-        if (i % targetInfo.mPeerDupHeadFactor == 0)
+        if ((i % targetInfo.mPeerDupHeadFactor) == (selfDPRank % targetInfo.mPeerDupHeadFactor))
         {
             for (int j = 0; j < targetInfo.mDomainPPSize; j++)
             {
diff --git a/cpp/tensorrt_llm/batch_manager/cacheTransBuffer.cpp b/cpp/tensorrt_llm/batch_manager/cacheTransBuffer.cpp
index 33986426f..424f02826 100644
--- a/cpp/tensorrt_llm/batch_manager/cacheTransBuffer.cpp
+++ b/cpp/tensorrt_llm/batch_manager/cacheTransBuffer.cpp
@@ -219,7 +219,7 @@ CacheTransBufferManager::CacheTransBufferManager(
         = maxNumTokens.has_value() ? bufferSizeFromMaxNumToken : common::getEnvMemSizeForKVCacheTransferBuffer();
     mOnlyUseDynamicBuffer = mTransferBufferSize == 0;
     mRecvBufferCount = common::getEnvRequestKVCacheConcurrent() ? common::getEnvKVCacheRecvBufferCount() : 1;
-    mSendBufferCount = common::getEnvParallelCacheSend() ? common::getEnvKVCacheSendMaxConcurrenceNum() : 1;
+    mSendBufferCount = common::getEnvKVCacheSendMaxConcurrenceNum();
     mUseFabricMemory = !(common::getEnvKVCacheTransferUseSyncBuffer() || common::getEnvKVCacheTransferUseAsyncBuffer())
         && FabricMemory::supportFbaricMemory();
     if (mUseFabricMemory)
@@ -269,7 +269,7 @@ size_t CacheTransBufferManager::preAllocBufferSize(
         TransferBufferSize = FabricMemory::getAlignedSize(TransferBufferSize);
     }
     size_t RecvBufferCount = common::getEnvRequestKVCacheConcurrent() ? common::getEnvKVCacheRecvBufferCount() : 1;
-    size_t SendBufferCount = common::getEnvParallelCacheSend() ? common::getEnvKVCacheSendMaxConcurrenceNum() : 1;
+    size_t SendBufferCo
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## f9a455651b - perf: Use tokenizers API to optimize incremental detokenization perf (#5574)

- **Date**: 2025-07-01
- **Author**: Kaiyu Xie
- **Categories**: General Performance

### Optimization Techniques

- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Speculative decoding
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- Decode/generation phase

### Changed Files

```
tensorrt_llm/executor/result.py                    |  3 --
 tensorrt_llm/inputs/registry.py                    | 19 +++++++------
 tensorrt_llm/llmapi/tokenizer.py                   | 33 ++++++++++++++++++++++
 .../defs/accuracy/test_llm_api_pytorch.py          | 10 +++++--
 tests/integration/test_lists/test-db/l0_b200.yml   |  3 +-
 tests/unittest/_torch/test_trtllm_sampler.py       |  6 ++--
 6 files changed, 57 insertions(+), 17 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/executor/result.py b/tensorrt_llm/executor/result.py
index 67c1d3d12..abd1a8649 100644
--- a/tensorrt_llm/executor/result.py
+++ b/tensorrt_llm/executor/result.py
@@ -360,9 +360,6 @@ class DetokenizedGenerationResultBase(GenerationResultBase):
         self.tokenizer = tokenizer
         self._streaming = streaming
 
-    @nvtx_range_debug("handle_response",
-                      color="red",
-                      category="DetokenizedGenerationResultBase")
     def _handle_response(self, response: "GenerationExecutor.Response"):
         GenerationResultBase._handle_response(self, response)
 
diff --git a/tensorrt_llm/inputs/registry.py b/tensorrt_llm/inputs/registry.py
index 5a841f121..010eb674a 100644
--- a/tensorrt_llm/inputs/registry.py
+++ b/tensorrt_llm/inputs/registry.py
@@ -3,6 +3,7 @@ from typing import (Any, Callable, Dict, List, Optional, Protocol, Tuple, Type,
 
 from torch import nn
 
+from .._utils import nvtx_range_debug
 from ..logger import logger
 from ..sampling_params import SamplingParams
 from .data import TextPrompt
@@ -61,16 +62,18 @@ class DefaultInputProcessor(InputProcessor):
             kwargs = dict(truncation=True,
                           max_length=sampling_params.truncate_prompt_tokens)
 
-        token_ids = self.tokenizer.encode(
-            inputs["prompt"],
-            add_special_tokens=sampling_params.add_special_tokens,
-            **kwargs)
-
-        if "query" in inputs:
-            query_token_ids = self.tokenizer.encode(
-                inputs["query"],
+        with nvtx_range_debug("tokenize prompt"):
+            token_ids = self.tokenizer.encode(
+                inputs["prompt"],
                 add_special_tokens=sampling_params.add_special_tokens,
                 **kwargs)
+
+        if "query" in inputs:
+            with nvtx_range_debug("tokenize query"):
+                query_token_ids = self.tokenizer.encode(
+                    inputs["query"],
+                    add_special_tokens=sampling_params.add_special_tokens,
+                    **kwargs)
             return token_ids, {"query_token_ids": query_token_ids}
 
         return token_ids, None
diff --git a/tensorrt_llm/llmapi/tokenizer.py b/tensorrt_llm/llmapi/tokenizer.py
index 76c6e1110..994333852 100644
--- a/tensorrt_llm/llmapi/tokenizer.py
+++ b/tensorrt_llm/llmapi/tokenizer.py
@@ -1,9 +1,13 @@
+import os
 from pathlib import Path
 from typing import Any, Dict, List, Optional, Tuple, Union
 
+from tokenizers.decoders import DecodeStream
 from transformers import (AutoTokenizer, PreTrainedTokenizerBase,
                           PreTrainedTokenizerFast)
 
+from .._utils import nvtx_range_debug
+
 
 class TokenizerBase(PreTrainedTokenizerBase):
     ''' This is a protocol for the tokenizer. Users can implement their own tokenizer by inheriting this class.  '''
@@ -16,6 +20,9 @@ class TransformersTokenizer(TokenizerBase):
     def __init__(self, tokenizer):
         self.tokenizer = tokeniz
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## f9e6045f39 - [#11086][feat] Optimize Auto Deploy weight loading by preloading weights to CPU (#11059)

- **Date**: 2026-02-03
- **Author**: Taylor Yeonbok Lee
- **Categories**: General Performance

### Optimization Techniques

- KV cache optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/auto_deploy/models/factory.py  |  12 ++-
 tensorrt_llm/_torch/auto_deploy/models/hf.py       | 115 +++++++++++++++++++--
 .../auto_deploy/transform/library/load_weights.py  |  10 +-
 .../_torch/auto_deploy/transform/optimizer.py      |   5 +
 4 files changed, 128 insertions(+), 14 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/models/factory.py b/tensorrt_llm/_torch/auto_deploy/models/factory.py
index ecfc11889..d04c35e5f 100644
--- a/tensorrt_llm/_torch/auto_deploy/models/factory.py
+++ b/tensorrt_llm/_torch/auto_deploy/models/factory.py
@@ -263,7 +263,9 @@ class ModelFactory(ABC):
         """
         return model_name_or_path
 
-    def load_or_random_init(self, model: nn.Module, device: DeviceLikeType):
+    def load_or_random_init(
+        self, model: nn.Module, device: DeviceLikeType, disable_preload: bool = False
+    ):
         """Load the checkpoint into the model or randomly initialize the model.
 
         Args:
@@ -271,6 +273,7 @@ class ModelFactory(ABC):
                 the same model that is built above but it needs to have a state dict compatible with
                 the model built above.
             device: The device to load the model on.
+            disable_preload: If True, disable preloading weights to CPU before moving to device.
             load_factoy_model: If True, will load weights for the factory model in addition to main
                 gm. This is useful for the transformers model.
 
@@ -303,7 +306,7 @@ class ModelFactory(ABC):
 
         if not self.skip_loading_weights:
             self.prefetch_checkpoint(force=True)
-            self._load_checkpoint(model, device)
+            self._load_checkpoint(model, device, disable_preload=disable_preload)
 
     @staticmethod
     def _to_maybe_random(model: nn.Module, device: DeviceLikeType):
@@ -323,7 +326,9 @@ class ModelFactory(ABC):
         )
 
     @abstractmethod
-    def _load_checkpoint(self, model: nn.Module, device: DeviceLikeType):
+    def _load_checkpoint(
+        self, model: nn.Module, device: DeviceLikeType, disable_preload: bool = False
+    ):
         """Load the checkpoint into the model.
 
         Args:
@@ -331,6 +336,7 @@ class ModelFactory(ABC):
                 the same model that is built above but it needs to have a state dict compatible with
                 the model built above.
             device: The device to load the model on.
+            disable_preload: If True, disable preloading weights to CPU before moving to device.
         """
 
     def get_example_inputs(self) -> Dict[str, torch.Tensor]:
diff --git a/tensorrt_llm/_torch/auto_deploy/models/hf.py b/tensorrt_llm/_torch/auto_deploy/models/hf.py
index fad26fa6e..4bd525df8 100644
--- a/tensorrt_llm/_torch/auto_deploy/models/hf.py
+++ b/tensorrt_llm/_torch/auto_deploy/models/hf.py
@@ -1,5 +1,6 @@
 """Interface to initialize and load HF models."""
 
+import json
 import os
 import re
 import types
@@ -7,6 +8,7 @@ from abc import abstractmethod
 from contextlib import contextmanager, nullcontext
 from typing import Any, Dict, List, Optional, Tuple, Type, Union
 
+import safetensors.torch
 import torch
 import torch.nn as nn
 from accelerate import init_empty_weights, load_checkpoint_in_model
@@ -419,7 +421,9 @@ class AutoModelForCausalLMFactory(AutoM
```

### Analysis Summary

This commit applies performance optimization techniques to improve inference efficiency. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## fac47e2826 - [https://nvbugs/5510879][fix] Fix pytorch & TRT-python flows fused LoRA adapter modules weight split with TP>1 (#8063)

- **Date**: 2025-10-12
- **Author**: amitz-nv
- **Categories**: Fusion

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Speculative decoding
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/resource_manager.py |  29 +++--
 tensorrt_llm/_utils.py                             |   6 +-
 tensorrt_llm/executor/base_worker.py               |   9 +-
 tensorrt_llm/lora_helper.py                        |   1 +
 tensorrt_llm/lora_manager.py                       | 125 +++++++++++++--------
 tensorrt_llm/runtime/enc_dec_model_runner.py       |  12 +-
 tensorrt_llm/runtime/generation.py                 |  38 ++++++-
 tensorrt_llm/runtime/model_runner.py               |   8 +-
 tensorrt_llm/runtime/model_runner_cpp.py           |  12 +-
 tests/unittest/llmapi/lora_test_utils.py           |  56 ++++++++-
 .../unittest/llmapi/test_llm_multi_gpu_pytorch.py  |  11 +-
 11 files changed, 229 insertions(+), 78 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/resource_manager.py b/tensorrt_llm/_torch/pyexecutor/resource_manager.py
index 6e4f4a984..bc2804584 100644
--- a/tensorrt_llm/_torch/pyexecutor/resource_manager.py
+++ b/tensorrt_llm/_torch/pyexecutor/resource_manager.py
@@ -13,6 +13,7 @@ from tensorrt_llm._utils import mpi_disabled
 from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
 from tensorrt_llm.lora_helper import LoraConfig
 from tensorrt_llm.lora_manager import LoraManager, LoraModelConfig
+from tensorrt_llm.runtime import ModelConfig as ModelConfigPython
 from tensorrt_llm.sampling_params import SamplingParams
 
 from ..._utils import binding_to_str_dtype, get_size_in_bytes, nvtx_range
@@ -32,7 +33,7 @@ BufferManagerCpp = tensorrt_llm.bindings.internal.runtime.BufferManager
 KVCacheManagerCpp = tensorrt_llm.bindings.internal.batch_manager.KVCacheManager
 KvCacheConfigCpp = tensorrt_llm.bindings.executor.KvCacheConfig
 CacheTypeCpp = tensorrt_llm.bindings.internal.batch_manager.CacheType
-ModelConfig = tensorrt_llm.bindings.ModelConfig
+ModelConfigCpp = tensorrt_llm.bindings.ModelConfig
 DataType = tensorrt_llm.bindings.DataType
 KVCacheEventManagerCpp = tensorrt_llm.bindings.internal.batch_manager.KVCacheEventManager
 RequestList = list[LlmRequest]
@@ -160,7 +161,7 @@ class KVCacheManager(BaseResourceManager):
         spec_config: Optional["DecodingBaseConfig"] = None,
         layer_mask: Optional[List[bool]] = None,
         max_num_tokens: int = 8192,
-        model_config: Optional[ModelConfig] = None,
+        model_config: Optional[ModelConfigCpp] = None,
         max_beam_width: int = 1,
         is_draft: bool = False,
         kv_connector_manager: Optional[KvCacheConnectorManager] = None,
@@ -371,7 +372,7 @@ class KVCacheManager(BaseResourceManager):
 
     @classmethod
     def from_model_config(cls,
-                          model_config: ModelConfig,
+                          model_config: ModelConfigCpp,
                           kv_cache_config: KvCacheConfigCpp,
                           mapping: Mapping,
                           kv_cache_type: CacheTypeCpp = CacheTypeCpp.SELF,
@@ -772,7 +773,7 @@ class KVCacheManager(BaseResourceManager):
         window_size_to_layers: Dict[int, List[int]],
         max_attention_window_vec: List[int],
         kv_cache_config: KvCacheConfigCpp,
-        model_config: ModelConfig,
+        model_config: ModelConfigCpp,
         pool_memory_bytes: int,
         kv_factor: int,
         dtype: DataType,
@@ -887,7 +888,7 @@ class KVCacheManager(BaseResourceManager):
     def calculate_max_num_blocks_from_cpp(
             self,
             kv_cache_config: KvCacheConfigCpp,
-            model_config: ModelConfig,
+            model_config: ModelConfigCpp,
             extra_cost_memory: int = 0) -> dict[int, tuple[int, int]]:
         """
         This function is a wrapper of KVCacheManagerCpp.calculate_max_num_blocks.
@@ -1133,7 +1134,7 @@ class PeftCacheManager(BaseResour
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## fae4985797 - [TRTLLM-9831][perf] Use TMA.RED to improve effective memory bandwidth (#10987)

- **Date**: 2026-01-27
- **Author**: ZhichenJiang
- **Categories**: Memory Optimization

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- Async/stream-based execution
- MoE optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU

### Changed Files

```
.../_torch/custom_ops/cute_dsl_custom_ops.py       |   1 +
 ...aled_contiguous_grouped_gemm_finalize_fusion.py | 208 ++++++++++++++++-----
 .../_torch/cute_dsl_kernels/blackwell/utils.py     |  51 +++++
 3 files changed, 211 insertions(+), 49 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py b/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
index ff37aa7d9..a75b9aedd 100644
--- a/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
+++ b/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
@@ -1305,6 +1305,7 @@ if IS_CUTLASS_DSL_AVAILABLE:
                     sf_vec_size=self.scaling_vector_size,
                     mma_tiler_mn=mma_tiler_mn,
                     cluster_shape_mn=cluster_shape_mn,
+                    use_blkred=True,
                     raster_along_m=raster_along_m,
                 )
                 # Compute max active clusters on current device
diff --git a/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_grouped_gemm_finalize_fusion.py b/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_grouped_gemm_finalize_fusion.py
index bc2856acb..50d36beff 100644
--- a/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_grouped_gemm_finalize_fusion.py
+++ b/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_grouped_gemm_finalize_fusion.py
@@ -40,6 +40,9 @@ from cutlass.cute.nvgpu import cpasync, tcgen05
 from .utils import (
     TRTLLM_ENABLE_PDL,
     atomic_add_func,
+    blk_reduce_bf16,
+    blk_reduce_fp16,
+    blk_reduce_fp32,
     griddepcontrol_launch_dependents,
     griddepcontrol_wait,
     is_power_of_2,
@@ -341,6 +344,7 @@ class Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel:
         sf_vec_size: int,
         mma_tiler_mn: Tuple[int, int],
         cluster_shape_mn: Tuple[int, int],
+        use_blkred: bool = False,
         raster_along_m: bool = False,
     ):
         """Initializes the configuration for a Blackwell blockscaled dense GEMM kernel.
@@ -371,6 +375,9 @@ class Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel:
 
         self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
 
+        # Block reduce configuration
+        self.use_blkred = use_blkred
+
         self.occupancy = 1
         self.epilog_warp_id = (0, 1, 2, 3)
         self.mma_warp_id = 4
@@ -528,12 +535,12 @@ class Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel:
             self.a_dtype,
             self.b_dtype,
             self.out_dtype,
-            self.gemm_output_layout,
-            self.epi_tile,
+            self.cta_tile_shape_mnk,
             self.sf_dtype,
             self.sf_vec_size,
             self.num_smem_capacity,
             self.occupancy,
+            self.use_blkred,
         )
 
         # Compute A/B/C/Scale shared memory layout
@@ -562,12 +569,16 @@ class Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel:
             self.num_ab_stage,
         )
 
-        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
-            self.out_dtype,
-            self.gemm_output_layout,
-            self.epi_tile,
-            self.num_c_stage,
+        swizzled_pad = 1
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## fcda1a1442 - [None][fix] disable async pp send for ray cases. (#9959)

- **Date**: 2025-12-14
- **Author**: Yuxian Qiu
- **Categories**: Parallelism/Async

### Optimization Techniques

- Operator fusion
- Async/stream-based execution
- Parallelism optimization
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
jenkins/L0_MergeRequest.groovy                  | 1 +
 tensorrt_llm/_torch/distributed/communicator.py | 7 +++++++
 2 files changed, 8 insertions(+)
```

### Diff Preview

```diff
diff --git a/jenkins/L0_MergeRequest.groovy b/jenkins/L0_MergeRequest.groovy
index deaa59fc2..60f48063e 100644
--- a/jenkins/L0_MergeRequest.groovy
+++ b/jenkins/L0_MergeRequest.groovy
@@ -712,6 +712,7 @@ def getMultiGpuFileChanged(pipeline, testFilter, globalVars)
         "tensorrt_llm/_torch/compilation/patterns/ub_allreduce.py",
         "tensorrt_llm/_torch/custom_ops/torch_custom_ops.py",
         "tensorrt_llm/_torch/custom_ops/userbuffers_custom_ops.py",
+        "tensorrt_llm/_torch/distributed/",
         "tensorrt_llm/_torch/models/modeling_llama.py",
         "tensorrt_llm/_torch/models/modeling_qwen3_next.py",
         "tensorrt_llm/_torch/modules/fused_moe/",
diff --git a/tensorrt_llm/_torch/distributed/communicator.py b/tensorrt_llm/_torch/distributed/communicator.py
index 5e4968f29..93457691b 100644
--- a/tensorrt_llm/_torch/distributed/communicator.py
+++ b/tensorrt_llm/_torch/distributed/communicator.py
@@ -856,6 +856,13 @@ class PPCommTorch(PPCommBase):
     def direct_send(self, tensor: torch.Tensor, dest: int):
         self.pg.send([tensor], self._global_to_local_rank(dest), tag=0).wait()
 
+    # TODO: support async pp send for PPCommTorch
+    def send(self, tensor: torch.Tensor, dest: Optional[int] = None):
+        if dest is None:
+            dest = self.mapping.next_pp_rank()
+
+        self.pg.send([tensor], self._global_to_local_rank(dest), tag=0).wait()
+
     def recv(self, tensor: torch.Tensor, src: Optional[int] = None):
         if src is None:
             src = self.mapping.prev_pp_rank()

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## ff82aef99b - Fix the issues related to fused moe path. (#3435)

- **Date**: 2025-04-11
- **Author**: Yukun He
- **Categories**: Fusion

### Optimization Techniques

- Operator fusion
- FP8 quantization
- Parallelism optimization
- MoE optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU

### Changed Files

```
cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp | 5 -----
 tensorrt_llm/_torch/modules/fused_moe.py                       | 7 ++++++-
 2 files changed, 6 insertions(+), 6 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp b/cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp
index d950c1dd0..e6b778c10 100644
--- a/cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp
+++ b/cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp
@@ -374,11 +374,6 @@ std::vector<CutlassGemmConfig> get_candidate_configs_sm100(CutlassGemmConfig::Ca
                         }
                     }
 
-                    if (cluster_n == 1 && cluster_m == 1 && ((config & CutlassGemmConfig::FP8_ONLY) != 0))
-                    {
-                        base.push_back(CutlassTileConfigSM100::CtaShape128x8x256B);
-                    }
-
                     std::vector onesm{CutlassTileConfigSM100::CtaShape64x64x128B,
                         CutlassTileConfigSM100::CtaShape64x128x128B, CutlassTileConfigSM100::CtaShape64x256x128B,
                         CutlassTileConfigSM100::CtaShape128x64x128B};
diff --git a/tensorrt_llm/_torch/modules/fused_moe.py b/tensorrt_llm/_torch/modules/fused_moe.py
index 5991e89cd..d5311523c 100755
--- a/tensorrt_llm/_torch/modules/fused_moe.py
+++ b/tensorrt_llm/_torch/modules/fused_moe.py
@@ -624,11 +624,16 @@ class FusedMoE(nn.Module):
         if min_latency_mode:
             assert not self.reduce_results
             return final_hidden_states
+        else:
+            # Custom op requires all inputs are in the same type.
+            # Only in min_latency_mode, the output is a list of tensors.
+            # Otherwise, the output should be unpacked as a single tensor.
+            final_hidden_states = final_hidden_states[0]
 
         if self.reduce_results and self.parallel_size > 1:
             return self.all_reduce(final_hidden_states)
         else:
-            return final_hidden_states[0]
+            return final_hidden_states
 
     def forward(
         self,

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

