# Performance Optimization Analysis - Part 2

Commits 30 to 58 of 283

---

## 1bab9000a6 - perf: Optimize swizzle_sf, unswizzle_sf, reswizzle_sf (#5318)

- **Date**: 2025-06-26
- **Author**: Bo Li
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
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
- Disaggregated serving

### Changed Files

```
cpp/tensorrt_llm/kernels/quantization.cu       |  65 +++++--
 cpp/tensorrt_llm/kernels/quantization.h        |   5 +-
 cpp/tensorrt_llm/thop/fp4Op.cpp                |  95 ++++++++---
 tensorrt_llm/__init__.py                       |   2 +
 tensorrt_llm/_torch/utils.py                   | 123 ++++++++------
 tensorrt_llm/math_utils.py                     |  20 +++
 tensorrt_llm/quantization/utils/fp4_utils.py   |   9 +-
 tests/unittest/_torch/thop/test_fp4_swizzle.py | 226 +++++++++++++++++++++++++
 8 files changed, 444 insertions(+), 101 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/quantization.cu b/cpp/tensorrt_llm/kernels/quantization.cu
index 772208a98..e78a6c9b3 100644
--- a/cpp/tensorrt_llm/kernels/quantization.cu
+++ b/cpp/tensorrt_llm/kernels/quantization.cu
@@ -226,18 +226,22 @@ void invokeBatchedFP4Quantization(int b, int m, int n, T const* input, float con
     }
 }
 
-__global__ void nvfp4_block_scale_interleave_kernel(
-    int numbatches, int numRows, int numCols, uint8_t const* SFIn, uint8_t* SFOutput)
+__global__ void nvfp4_block_scale_interleave_kernel(int numBatches, int numRows, int numRowsPadded, int numCols,
+    int numColsPadded, uint8_t const* SFIn, uint8_t* SFOutput)
 {
     constexpr int SF_VEC_SIZE = 16;
-    for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x)
+    for (int rowIdx = blockIdx.x; rowIdx < numRowsPadded; rowIdx += gridDim.x)
     {
-        for (int batchIdx = 0; batchIdx < numbatches; batchIdx++)
+        for (int batchIdx = 0; batchIdx < numBatches; batchIdx++)
         {
-            for (int colIdx = threadIdx.x; colIdx < numCols; colIdx += blockDim.x)
+            for (int colIdx = threadIdx.x; colIdx < numColsPadded; colIdx += blockDim.x)
             {
-                int64_t inOffset = batchIdx * numRows * numCols + rowIdx * numCols + colIdx;
-                auto sf = SFIn[inOffset];
+                uint8_t sf = 0;
+                if (rowIdx < numRows && colIdx < numCols)
+                {
+                    int64_t inOffset = batchIdx * numRows * numCols + rowIdx * numCols + colIdx;
+                    sf = SFIn[inOffset];
+                }
 
                 std::optional<int> batchIdxOpt = batchIdx;
                 std::optional<int> numRowsOpt = numRows;
@@ -246,16 +250,55 @@ __global__ void nvfp4_block_scale_interleave_kernel(
                 // int const numSfTilesK = (numCols + 4 - 1) / 4;
                 // int const tileOffset = ((mi / 128) * numSfTilesK + ki / 4) * 512;
                 // int const dstIdx = tileOffset + (mi % 32) * 16 + ((mi % 128) / 32) * 4 + ki % 4;
-                auto dstIdx
-                    = get_sf_out_offset_128x4<SF_VEC_SIZE>(batchIdxOpt, rowIdx, colIdx, numRowsOpt, numCols * 16);
+                auto dstIdx = get_sf_out_offset_128x4<SF_VEC_SIZE>(
+                    batchIdxOpt, rowIdx, colIdx, numRowsOpt, numCols * SF_VEC_SIZE);
                 SFOutput[dstIdx] = sf;
             }
         }
     }
 }
 
+__global__ void nvfp4_block_scale_interleave_reverse_kernel(
+    int numBatches, int numRows, int numCols, uint8_t const* SFIn, uint8_t* SFOutput)
+{
+    constexpr int SF_VEC_SIZE = 16;
+    for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x)
+    {
+        for (int batchIdx = 0; batchIdx < numBatches; batchIdx++)
+        {
+            for (int colIdx = threadIdx.x; colIdx < numCols; colIdx += blockDim.x)
+            {
+                std::optional<int> batchIdxOpt = batchIdx;
+                std::optional<int> numRowsOpt = numRows;
+
+    
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 1c69aad850 - [TRTLLM-10309] [feat] Optimize qk rope/nope concat for DSA (#10571)

- **Date**: 2026-01-09
- **Author**: Kaiyu Xie
- **Categories**: General Performance

### Optimization Techniques

- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/attention_backend/sparse/dsa.py |  4 ++--
 tensorrt_llm/_torch/modules/attention.py            | 13 ++-----------
 tensorrt_llm/_torch/utils.py                        | 10 ++++++++++
 3 files changed, 14 insertions(+), 13 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/attention_backend/sparse/dsa.py b/tensorrt_llm/_torch/attention_backend/sparse/dsa.py
index a827ff7b9..9ae60d0c1 100644
--- a/tensorrt_llm/_torch/attention_backend/sparse/dsa.py
+++ b/tensorrt_llm/_torch/attention_backend/sparse/dsa.py
@@ -17,7 +17,7 @@ from tensorrt_llm._torch.modules.multi_stream_utils import \
     maybe_execute_in_parallel
 from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding
 from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
-from tensorrt_llm._torch.utils import maybe_compile
+from tensorrt_llm._torch.utils import maybe_compile, maybe_compiled_cat
 from tensorrt_llm._utils import get_size_in_bytes, get_sm_version
 from tensorrt_llm.bindings import DataType
 from tensorrt_llm.bindings.executor import KvCacheConfig
@@ -1541,7 +1541,7 @@ class Indexer(nn.Module):
 
     def _prep_q_or_k(self, qk_pe: torch.Tensor, qk_nope: torch.Tensor):
         """Concatenate, rotate, and FP8 quantize for Q or K"""
-        q_or_k = torch.cat([qk_pe, qk_nope], dim=-1)
+        q_or_k = maybe_compiled_cat([qk_pe, qk_nope], dim=-1)
         q_or_k = rotate_activation(q_or_k)
         q_or_k = q_or_k.view(-1, self.head_dim)
         q_or_k = fp8_utils.fp8_quantize_1x128_sf_transpose(
diff --git a/tensorrt_llm/_torch/modules/attention.py b/tensorrt_llm/_torch/modules/attention.py
index 47e793e48..69ae31371 100644
--- a/tensorrt_llm/_torch/modules/attention.py
+++ b/tensorrt_llm/_torch/modules/attention.py
@@ -25,7 +25,8 @@ from ..distributed import AllReduceParams, HelixAllToAllNative, alltoall_helix
 from ..model_config import ModelConfig
 from ..peft.lora.layer import LoraLayer, LoraModuleType
 from ..utils import (Fp4QuantizedTensor, get_model_extra_attrs,
-                     is_torch_compiling, maybe_compile)
+                     is_torch_compiling, maybe_compiled_cat,
+                     maybe_compiled_copy_)
 from .linear import Linear, TensorParallelMode, WeightMode, WeightsLoadingConfig
 from .multi_stream_utils import maybe_execute_in_parallel
 from .rms_norm import RMSNorm
@@ -78,16 +79,6 @@ def extract_extra_attrs(layer_idx: str, attn_type: str):
     return metadata, attn_layer
 
 
-@maybe_compile
-def maybe_compiled_copy_(dst, src):
-    dst.copy_(src)
-
-
-@maybe_compile
-def maybe_compiled_cat(tensors, dim):
-    return torch.cat(tensors, dim)
-
-
 def create_attn_outputs_impl(q: torch.Tensor, attention_mask: str,
                              layer_idx: str) -> List[torch.Tensor]:
     metadata, attn_layer = extract_extra_attrs(layer_idx, "attn")
diff --git a/tensorrt_llm/_torch/utils.py b/tensorrt_llm/_torch/utils.py
index 1c3c02ca3..55276832f 100644
--- a/tensorrt_llm/_torch/utils.py
+++ b/tensorrt_llm/_torch/utils.py
@@ -404,3 +404,13 @@ def split(x: torch.Tensor,
 
 def relu2(x: torch.Tensor) -> torch.Tensor:
     return torch.square(F.relu(x))
+
+
+@maybe_compile
+def maybe_compiled_copy_(dst, src):
+    dst.copy_(src)
+
+
+@maybe_compile
+def 
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 1d3b98b920 - perf: Optimize quantization kernels used in DeepSeek on Hopper (#3466)

- **Date**: 2025-04-15
- **Author**: jiahanc
- **Categories**: Kernel Optimization, Quantization Optimization

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- Batching optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../fp8_blockscale_gemm_kernel.cuh                 | 91 +++++++++++++++-------
 1 file changed, 61 insertions(+), 30 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh b/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh
index 4747fb25c..9234883db 100644
--- a/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh
+++ b/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh
@@ -979,7 +979,7 @@ __global__ void scale_1x128_kernel(
     size_t scales_along_dim_x = div_up(dim_x, 128);
     size_t scales_along_dim_y = div_up(dim_y, 1);
     size_t stride_scale_dim_y = div_up(dim_y, 4) * 4;
-
+    using Input2Type = typename std::conditional<std::is_same<InputType, half>::value, half2, __nv_bfloat162>::type;
     for (size_t warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
          warp_idx < scales_along_dim_x * scales_along_dim_y; warp_idx += gridDim.x * blockDim.x / 32)
     {
@@ -988,21 +988,34 @@ __global__ void scale_1x128_kernel(
 
         InputType const* input_line = input + (size_t) scales_idx_y * dim_x + scales_idx_x * 128;
         InputType input_amax = InputType(0);
-        int lane_id = threadIdx.x % 32;
-        InputType input_frag[4] = {0};
+        // Each thread reads 2 elements from input_line
+        int lane_id = threadIdx.x % 32 * 2;
 
-        for (int i = 0; i < 4; i++)
+        Input2Type input_frag2[2] = {Input2Type(0, 0), Input2Type(0, 0)};
+#pragma unroll
+        for (int i = 0; i < 2; i++)
         {
-            if (scales_idx_x * 128 + i * 32 + lane_id >= dim_x)
+            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x)
             {
                 break;
             }
             else
             {
-                input_frag[i] = input_line[lane_id];
-                input_amax = InputType(std::max(float(input_amax), std::fabs(float(input_frag[i]))));
+                input_frag2[i] = *((Input2Type*) (input_line) + lane_id / 2);
+            }
+            input_line += 64;
+        }
+#pragma unroll
+        for (int i = 0; i < 2; i++)
+        {
+            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x)
+            {
+                break;
+            }
+            else
+            {
+                input_amax = InputType(__hmax(input_amax, __hmax(__habs(input_frag2[i].x), __habs(input_frag2[i].y))));
             }
-            input_line += 32;
         }
 
         InputType amax = find_max_elem_in_warp(input_amax);
@@ -1014,18 +1027,21 @@ __global__ void scale_1x128_kernel(
         }
 
         OutputType* output_line = output + (size_t) scales_idx_y * dim_x + scales_idx_x * 128;
-        for (int i = 0; i < 4; i++)
+#pragma unroll
+        for (int i = 0; i < 2; i++)
         {
-            if (scales_idx_x * 128 + i * 32 + lane_id >= dim_x)
+            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x)
             {
                 break;
             }
             else
             {
-                ScaleType value
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 1d68fab49c - [https://nvbugs/5814215][fix] Unwaive test_trtllm_flashinfer_symbol_collision.py::test_flashinfer_fused_moe_matches_torch_moe (#10930)

- **Date**: 2026-01-24
- **Author**: Yihan Wang
- **Categories**: Fusion

### Optimization Techniques

- Operator fusion
- FP8 quantization
- Parallelism optimization
- Speculative decoding
- MoE optimization
- Attention mechanism optimization

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
index 4a1355742..0a7c3bd82 100644
--- a/tests/integration/test_lists/waives.txt
+++ b/tests/integration/test_lists/waives.txt
@@ -343,7 +343,6 @@ accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_nvfp4_4gpus[moe_backe
 accuracy/test_llm_api_pytorch.py::TestDeepSeekR1::test_nvfp4_multi_gpus[latency] SKIP (https://nvbugs/5814309)
 accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_bfloat16_4gpus[tp4-mtp_nextn=2-attention_dp=False-cuda_graph=False-overlap_scheduler=False-torch_compile=True] SKIP (https://nvbugs/5800646)
 stress_test/stress_test.py::test_run_stress_test[llama-v3-8b-instruct-hf_tp1-stress_time_300s_timeout_450s-MAX_UTILIZATION-pytorch-stress-test] SKIP (https://nvbugs/5814203)
-unittest/_torch/attention/test_trtllm_flashinfer_symbol_collision.py::test_flashinfer_fused_moe_matches_torch_moe SKIP (https://nvbugs/5814215)
 full:sm89/accuracy/test_llm_api_pytorch_multimodal.py::TestNVILA_8B::test_auto_dtype SKIP (https://nvbugs/5814504)
 accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_nvfp4_4gpus[moe_backend=CUTLASS-mtp_nextn=0-tp2pp2-fp8kv=False-attention_dp=False-cuda_graph=False-overlap_scheduler=False-torch_compile=True] SKIP (https://nvbugs/5819005)
 unittest/llmapi/test_mpi_session.py::test_llmapi_launch_multiple_tasks SKIP (https://nvbugs/5819014)

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 1e317c98c6 - [feat]: Allow for a settable end-of-sequence/padding token in max throughput benchmark. (#3776)

- **Date**: 2025-04-30
- **Author**: Frank
- **Categories**: Throughput/Latency

### Optimization Techniques

- Parallelism optimization
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/bench/benchmark/throughput.py | 41 ++++++++++++++++++++----------
 1 file changed, 27 insertions(+), 14 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/bench/benchmark/throughput.py b/tensorrt_llm/bench/benchmark/throughput.py
index 90ea2d6d0..c71661d25 100755
--- a/tensorrt_llm/bench/benchmark/throughput.py
+++ b/tensorrt_llm/bench/benchmark/throughput.py
@@ -94,6 +94,14 @@ from tensorrt_llm.sampling_params import SamplingParams
     required=False,
     help="Pass in a dataset file for parsing instead of stdin.",
 )
+@optgroup.option(
+    "--eos_id",
+    type=int,
+    default=-1,
+    required=False,
+    help=
+    "Set the end-of-sequence token for the benchmark. Set to -1 to disable EOS.",
+)
 @optgroup.option(
     "--modality",
     type=click.Choice(["image", "video"]),
@@ -122,6 +130,22 @@ from tensorrt_llm.sampling_params import SamplingParams
     default=2,
     help="Number of requests warm up benchmark.",
 )
+@optgroup.option(
+    "--target_input_len",
+    default=None,
+    type=click.IntRange(min=1),
+    help="Target (average) input length for tuning heuristics.",
+)
+@optgroup.option(
+    "--target_output_len",
+    default=None,
+    type=click.IntRange(min=1),
+    help="Target (average) sequence length for tuning heuristics.",
+)
+@optgroup.group(
+    "World Configuration",
+    help="Options for configuring the backend multi-GPU world.",
+)
 @optgroup.option(
     "--tp",
     type=int,
@@ -146,18 +170,6 @@ from tensorrt_llm.sampling_params import SamplingParams
     default=None,
     help="expert cluster parallelism size",
 )
-@optgroup.option(
-    "--target_input_len",
-    default=None,
-    type=click.IntRange(min=1),
-    help="Target (average) input length for tuning heuristics.",
-)
-@optgroup.option(
-    "--target_output_len",
-    default=None,
-    type=click.IntRange(min=1),
-    help="Target (average) sequence length for tuning heuristics.",
-)
 @optgroup.group("Request Load Control Options",
                 cls=MutuallyExclusiveOptionGroup,
                 help="Limits how requests are loaded.")
@@ -218,6 +230,7 @@ def throughput_command(
     # Parameters from CLI
     # Model, experiment, and engine params
     dataset_path: Path = params.pop("dataset")
+    eos_id: int = params.pop("eos_id")
     warmup: int = params.get("warmup")
     num_requests: int = params.pop("num_requests")
     max_seq_len: int = params.pop("max_seq_len")
@@ -329,8 +342,8 @@ def throughput_command(
         else:
             llm = LLM(**kwargs)
 
-        sampling_params = SamplingParams(end_id=-1,
-                                         pad_id=-1,
+        sampling_params = SamplingParams(end_id=eos_id,
+                                         pad_id=eos_id,
                                          beam_width=beam_width)
 
         # Perform warmup if requested.

```

### Analysis Summary

MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 1e5e71aa42 - Mtp optimizations round1 (#5689)

- **Date**: 2025-07-25
- **Author**: ameynaik-hub
- **Categories**: General Performance

### Optimization Techniques

- Torch compilation/JIT optimization
- Custom CUDA kernel
- Async/stream-based execution
- Parallelism optimization
- Batching optimization
- Speculative decoding
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/models/modeling_deepseekv3.py  |  85 ++++++++---
 tensorrt_llm/_torch/models/modeling_speculative.py |   1 +
 tensorrt_llm/_torch/speculative/mtp.py             | 155 ++++++++++++++++-----
 tensorrt_llm/_torch/speculative/utils.py           |   6 +-
 4 files changed, 191 insertions(+), 56 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_deepseekv3.py b/tensorrt_llm/_torch/models/modeling_deepseekv3.py
index c8523deea..9d0e16518 100644
--- a/tensorrt_llm/_torch/models/modeling_deepseekv3.py
+++ b/tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -131,11 +131,21 @@ class DeepseekV3MTPHead(nn.Module):
     def __init__(self, model_config: ModelConfig[PretrainedConfig]):
         super().__init__()
         config = model_config.pretrained_config
+        self.model_config = model_config
 
         self.norm = RMSNorm(hidden_size=config.hidden_size,
                             eps=config.rms_norm_eps,
                             dtype=config.torch_dtype)
 
+    @torch.compile(options={"max-autotune": True})
+    def get_last_token_states(self, hidden_states, attn_metadata):
+        last_tokens = torch.cumsum(
+            attn_metadata.seq_lens_cuda,
+            dim=0,
+            dtype=torch.long,
+        ) - 1
+        return hidden_states[last_tokens]
+
     def forward(self,
                 hidden_states: torch.Tensor,
                 lm_head: Linear,
@@ -143,16 +153,16 @@ class DeepseekV3MTPHead(nn.Module):
                 return_context_logits: bool = False) -> torch.Tensor:
         if not return_context_logits:
             if attn_metadata is not None:
-                last_tokens = torch.cumsum(
-                    attn_metadata.seq_lens_cuda,
-                    dim=0,
-                    dtype=torch.long,
-                ) - 1
-                hidden_states = hidden_states[last_tokens]
+                hidden_states = self.get_last_token_states(
+                    hidden_states, attn_metadata)
             else:
                 hidden_states = hidden_states[-1].unsqueeze(0)
 
+        if not (self.model_config.mapping.enable_attention_dp):
+            lm_head.gather_output = False
         logits = lm_head(hidden_states)
+        if not (self.model_config.mapping.enable_attention_dp):
+            lm_head.gather_output = True
         return logits
 
 
@@ -903,6 +913,12 @@ class DeepseekV3MTP(DeepseekV3DecoderLayer):
         self.num_shared_experts = config.n_shared_experts
         self.top_k = config.num_experts_per_tok
 
+        self.aux_stream = aux_stream_dict[AuxStreamType.MoeShared]
+        self.event_dict = {
+            key: torch.cuda.Event()
+            for key in [EventType.Main, EventType.MoeShared]
+        }
+
         self.enorm = RMSNorm(hidden_size=config.hidden_size,
                              eps=config.rms_norm_eps,
                              dtype=config.torch_dtype)
@@ -910,15 +926,27 @@ class DeepseekV3MTP(DeepseekV3DecoderLayer):
         self.hnorm = RMSNorm(hidden_size=config.hidden_size,
                              eps=config.rms_norm_eps,
                              dtype=config.torch_dtype)
-
-        self.eh_proj = Linear(
-            config.hidden_size * 2,
-            config.hidden_size,
-            bias=False,
-            dtype=config.torch_dtype,
-  
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 1f292ff2a0 - [https://jirasw.nvidia.com/browse/TRTLLM-4645] support mutliCtasKvMode for high-throughput MLA kernels (#5426)

- **Date**: 2025-06-25
- **Author**: Perkz Zheng
- **Categories**: Kernel Optimization, Throughput/Latency

### Optimization Techniques

- FP8 quantization
- KV cache optimization
- Batching optimization
- Speculative decoding
- Attention mechanism optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- Decode/generation phase

### Changed Files

```
...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Persistent2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128Static2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Persistent2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128Static2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
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
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Persistent2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128Static2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Persistent2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128Static2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Persistent2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128Static2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Persistent2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128Static2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Persistent2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128Static2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Persistent2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128Static2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Persistent2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128Static2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Persistent2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128Static2CtaKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...Kv64PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...SizeKv64StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticKeepsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeQ128TileSizeKv128PersistentContext_cubin.cpp |    3 -
 ...ileSizeQ128TileSizeKv128StaticContext_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...v128PersistentSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...izeKv128StaticSwapsMmaAbForGeneration_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
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
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
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
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
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
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...eQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
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
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
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
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
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
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
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
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
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
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
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
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...eQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 +
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
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
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
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
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 +
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
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
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...eQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 .../trtllmGenKernels/fmha/cubin/kernelMetaInfo.h   | 8762 +++++++++-----------
 .../kernels/trtllmGenKernels/fmha/fmhaKernels.h    |   48 +-
 2922 files changed, 7493 insertions(+), 10077 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QBfloat16KvBfloat16AccFp32OBfloat16HQk128HV128LayoutPackedQkvMaskCausalVarSeqLenTileSizeQ128TileSizeKv128PersistentContext_cubin.cpp b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QBfloat16KvBfloat16AccFp32OBfloat16HQk128HV128LayoutPackedQkvMaskCausalVarSeqLenTileSizeQ128TileSizeKv128PersistentContext_cubin.cpp
deleted file mode 100644
index 988f4801b..000000000
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QBfloat16KvBfloat16AccFp32OBfloat16HQk128HV128LayoutPackedQkvMaskCausalVarSeqLenTileSizeQ128TileSizeKv128PersistentContext_cubin.cpp
+++ /dev/null
@@ -1,3 +0,0 @@
-version https://git-lfs.github.com/spec/v1
-oid sha256:0a3b0b0d4aa1414c48c4cc39cc33855343b204bc4bc1dc65ea545e934f9659cd
-size 1340171
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QBfloat16KvBfloat16AccFp32OBfloat16HQk128HV128LayoutPackedQkvMaskCausalVarSeqLenTileSizeQ128TileSizeKv128StaticContext_cubin.cpp b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QBfloat16KvBfloat16AccFp32OBfloat16HQk128HV128LayoutPackedQkvMaskCausalVarSeqLenTileSizeQ128TileSizeKv128StaticContext_cubin.cpp
deleted file mode 100644
index 46ece625e..000000000
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QBfloat16KvBfloat16AccFp32OBfloat16HQk128HV128LayoutPackedQkvMaskCausalVarSeqLenTileSizeQ128TileSizeKv128StaticContext_cubin.cpp
+++ /dev/null
@@ -1,3 +0,0 @@
-version https://git-lfs.github.com/spec/v1
-oid sha256:7dcefb2233db742e41ef20641e5a82dbace28be552a59a3dd29da7c2cd627f9a
-size 1209973
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QBfloat16KvBfloat16AccFp32OBfloat16HQk128HV128LayoutPackedQkvMaskDenseVarSeqLenTileSizeQ128TileSizeKv128PersistentContext_cubin.cpp b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QBfloat16KvBfloat16AccFp32OBfloat16HQk128HV128LayoutPackedQkvMaskDenseVarSeqLenTileSizeQ128TileSizeKv128PersistentContext_cubin.cpp
deleted file mode 100644
index 72a2d61fc..000000000
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QBfloat16KvBfloat16AccFp32OBfloat16HQk128HV128LayoutPackedQkvMaskDenseVarSeqLenTileSizeQ128TileSizeKv128PersistentContext_cubin.cpp
+++ /dev/null
@@ -1,3 +0,0 @@
-version https://git-lfs.github.com/spec/v1
-oid sha256:2ad04f87a1bdf10813d04b4369e6a6a534cbc49d798e95decde3450db928688c
-size 1336765
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QBfloat16KvBfloat16AccFp32OBfloat16HQk128HV128LayoutPackedQkvMaskDenseVarSeqLenTileSizeQ128TileSizeKv128StaticContext_cubin.cpp b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QBfloat16KvBfloat16AccFp32OBfloat16HQk128HV128LayoutPackedQkvMaskDenseVarSeqLenTileSizeQ128TileSizeKv128StaticContext_cubin.cpp
deleted file mode 100644
index a299ae547..000000000
--- a/cpp/tensorrt_llm
```

### Analysis Summary

Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 1fef88e95d - [None][chore] Improve sampler performance by replacing torch.where with masked_fill_ (#11949)

- **Date**: 2026-03-10
- **Author**: Stefan Niebler
- **Categories**: General Performance

### Optimization Techniques

- Async/stream-based execution
- Batching optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/sampler.py | 26 ++++++++++++++------------
 1 file changed, 14 insertions(+), 12 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index 001bf4084..e6c17a51d 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -1792,7 +1792,8 @@ class TorchSampler(Sampler[SampleStateTorch], AsyncWorkerMixin):
                     if not single_token_stop_words_only
                     else self._are_stop_words_single_token
                 )
-                batched_finish_reasons[:, stop_word_indices] = torch.where(
+                batched_finish_reasons_stop_words = batched_finish_reasons[:, stop_word_indices]
+                _ = batched_finish_reasons_stop_words.masked_fill_(
                     stop_words_func(
                         stop_seq_slots,
                         stop_tokens,
@@ -1801,18 +1802,17 @@ class TorchSampler(Sampler[SampleStateTorch], AsyncWorkerMixin):
                         else num_accepted_tokens,
                     ),
                     FinishReason.STOP_WORDS.value,
-                    batched_finish_reasons[:, stop_word_indices],
                 )
+                batched_finish_reasons[:, stop_word_indices] = batched_finish_reasons_stop_words
 
-            batched_finish_reasons = torch.where(
+            _ = batched_finish_reasons.masked_fill_(
                 self._are_max_length(seq_lens, store.max_lengths_cuda[seq_slots]),
                 FinishReason.LENGTH.value,
-                batched_finish_reasons,
             )
-            batched_finish_reasons = torch.where(
+
+            _ = batched_finish_reasons.masked_fill_(
                 self._are_end_id(store.end_ids_cuda[seq_slots], tokens),
                 FinishReason.END_ID.value,
-                batched_finish_reasons,
             )
 
             finish_reasons[:, seq_slots] = batched_finish_reasons
@@ -1916,7 +1916,7 @@ class TorchSampler(Sampler[SampleStateTorch], AsyncWorkerMixin):
             # Fill in the new tokens at the end of the past tokens buffer
             full_tokens[-self._max_tokens :] = tokens
             # short words are padded with _PAD_STOP_WORD_TOKEN_ID, so we need to mask them
-            mask = stop_words != self._PAD_STOP_WORD_TOKEN_ID
+            mask = stop_words == self._PAD_STOP_WORD_TOKEN_ID
             matches = torch.empty(
                 (
                     self._max_tokens,
@@ -1941,15 +1941,15 @@ class TorchSampler(Sampler[SampleStateTorch], AsyncWorkerMixin):
             stop_words_for_match = stop_words.unsqueeze(0)
             _ = torch.eq(full_tokens_for_match, stop_words_for_match, out=matches)
             # Mask the padding tokens
-            matches_after_mask = torch.where(
-                mask.unsqueeze(0).expand(self._max_tokens, -1, -1, -1, -1), matches, True
+            _ = matches.masked_fill_(
+                mask.unsqueeze(0).expand(self._max_tokens, -1, -1, -1, -1), True
             )
             # Update the past tokens storage for the next iteration
  
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 211c44b951 - [None][feat] Adding torch ext API for FusedAddRMSNormQuant kernel (#9905)

- **Date**: 2026-01-15
- **Author**: 彭晋韬(jtao peng)
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
- Reduce synchronization overhead
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- NVIDIA Hopper (SM90) GPU
- Decode/generation phase

### Changed Files

```
.../low_latency_layernorm.cuh                      |  33 ++--
 .../kernels/fusedLayernormKernels/ws_layernorm.cuh |  71 +++++---
 cpp/tensorrt_llm/thop/CMakeLists.txt               |   1 +
 cpp/tensorrt_llm/thop/fusedAddRMSNormQuant.cpp     | 200 +++++++++++++++++++++
 tensorrt_llm/_torch/custom_ops/torch_custom_ops.py |  53 ++++++
 tensorrt_llm/_torch/models/modeling_llama.py       |  46 +++--
 tensorrt_llm/_torch/modules/rms_norm.py            | 111 +++++++++++-
 tensorrt_llm/mapping.py                            |  33 ++++
 .../defs/accuracy/test_llm_api_pytorch.py          |  14 ++
 9 files changed, 512 insertions(+), 50 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/fusedLayernormKernels/low_latency_layernorm.cuh b/cpp/tensorrt_llm/kernels/fusedLayernormKernels/low_latency_layernorm.cuh
index 9545d919c..6a925c551 100644
--- a/cpp/tensorrt_llm/kernels/fusedLayernormKernels/low_latency_layernorm.cuh
+++ b/cpp/tensorrt_llm/kernels/fusedLayernormKernels/low_latency_layernorm.cuh
@@ -115,8 +115,6 @@ struct LowLatencyLayerNorm
 
         uint32_t work_id = blockIdx.x;
 
-        FusedOperator fused_operator(param);
-
         constexpr auto PACKED_PER_N_BLOCK = Traits::N_BLOCK / N_THREADS / Traits::PACKED_ELEMS_PER_COMPUTE;
 
         typename Traits::AccumulatorType data[PACKED_PER_N_BLOCK][Traits::PACKED_ELEMS_PER_COMPUTE];
@@ -139,7 +137,7 @@ struct LowLatencyLayerNorm
             for (int i = 0; i < PACKED_PER_N_BLOCK; i++)
             {
                 auto offset = (thread_id + i * N_THREADS) * Traits::PACKED_ELEMS_PER_COMPUTE;
-                if (offset <= sz)
+                if (offset < sz)
                 {
                     data[i] = *reinterpret_cast<PackedType const*>(&g_data[offset]);
                 }
@@ -155,6 +153,14 @@ struct LowLatencyLayerNorm
 
         static_assert(Traits::OUTPUT_SCALE != SCALE_TYPE::VECTOR);
 
+#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDACC_VER_MAJOR__ >= 12))
+        if constexpr (arch::is_major_v<9> || arch::is_major_v<10>)
+        {
+            cudaGridDependencySynchronize();
+        }
+#endif
+        FusedOperator fused_operator(param);
+
         if constexpr (Traits::BIAS == SCALE_TYPE::VECTOR)
         {
             load_to_register(param.bias, r_bias, param.n);
@@ -175,13 +181,6 @@ struct LowLatencyLayerNorm
             load_to_register(param.beta, r_beta, param.n);
         }
 
-#if (defined(__CUDA_ARCH__) && (__CUDACC_VER_MAJOR__ >= 12))
-        if constexpr (arch::is_major_v<9> || arch::is_major_v<10>)
-        {
-            cudaGridDependencySynchronize();
-            cudaTriggerProgrammaticLaunchCompletion();
-        }
-#endif
         load_to_register(&param.input[work_id * param.n], data, param.n);
 
         if constexpr (Traits::RESIDUAL)
@@ -259,12 +258,12 @@ struct LowLatencyLayerNorm
         if constexpr (!Traits::RMS_NORM)
         {
             mean = var_and_mean[1] / param.n;
-            variance = rsqrtf(
-                var_and_mean[0] / param.n - var_and_mean[1] * var_and_mean[1] + (Traits::AccumulatorType)(1e-5));
+            variance = rsqrtf(var_and_mean[0] / param.n - var_and_mean[1] * var_and_mean[1]
+                + (Traits::AccumulatorType)(param.layernorm_eps));
         }
         else
         {
-            variance = rsqrtf(var_and_mean[0] / param.n + (Traits::AccumulatorType)(1e-5));
+            variance = rsqrtf(var_and_mean[0] / param.n + (Traits::AccumulatorType)(param.layernorm_eps));
         }
 
         for (int i = 0; i < PACKED_PER_N_BLOCK; i++)
@@ -333,6 +332,14 @@ struct LowLatencyLayerNorm
     {
         __shared__ Shared shared;
     
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 215fb20567 - chore : split GptExecutor tests out of gpt tests to reduce single test time (#3412)

- **Date**: 2025-04-10
- **Author**: peaceh-nv
- **Categories**: General Performance

### Optimization Techniques

- FP8 quantization
- Speculative decoding
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tests/integration/defs/cpp_common.py            | 9 +++++++++
 tests/integration/defs/test_cpp.py              | 3 ++-
 tests/integration/test_lists/test-db/l0_a30.yml | 1 +
 3 files changed, 12 insertions(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tests/integration/defs/cpp_common.py b/tests/integration/defs/cpp_common.py
index a0e1bc1a0..a9c480c07 100755
--- a/tests/integration/defs/cpp_common.py
+++ b/tests/integration/defs/cpp_common.py
@@ -20,6 +20,7 @@ default_test_timeout = 3600
 
 include_test_map = {
     "gpt": ("Gpt[^j]", ),
+    "gpt_executor": ("GptExecutor", ),
     "gptj": ("Gptj", ),
     "llama": ("Llama", ),
     "chatglm": ("ChatGlm", ),
@@ -40,6 +41,7 @@ include_test_map = {
 
 def generate_excluded_model_tests() -> Generator[str, None, None]:
     yield "Gpt[^j]"
+    yield "GptExecutor"
     yield "Gptj"
     yield "Llama"
     yield "ChatGlm"
@@ -619,6 +621,10 @@ def prepare_model_tests(model_name: str,
             beams_arg = ['--beams', '1,2']
         model_name = 'enc_dec'
 
+    # share the same script for gpt and gpt_executor
+    if model_name == 'gpt_executor':
+        model_name = 'gpt'
+
     build_engines = [
         python_exe,
         str(scripts_dir / f"build_{model_name}_engines.py")
@@ -710,6 +716,9 @@ def run_single_gpu_tests(build_dir: _pl.Path,
 
     excluded_tests = ["FP8"] if not run_fp8 else []
 
+    if "gpt" in test_list and "gpt_executor" not in test_list:
+        excluded_tests.append("GptExecutor")
+
     ctest = ["ctest", "--output-on-failure", "--output-junit", resultFileName]
 
     if included_tests:
diff --git a/tests/integration/defs/test_cpp.py b/tests/integration/defs/test_cpp.py
index a591d8baa..f2eddcdfb 100644
--- a/tests/integration/defs/test_cpp.py
+++ b/tests/integration/defs/test_cpp.py
@@ -335,7 +335,8 @@ def test_unit_tests(build_google_tests, build_dir, lora_setup):
                          indirect=True)
 @pytest.mark.parametrize("model", [
     "bart", "chatglm", "eagle", "encoder", "enc_dec_language_adapter", "gpt",
-    "llama", "mamba", "medusa", "recurrentgemma", "redrafter", "t5"
+    "gpt_executor", "llama", "mamba", "medusa", "recurrentgemma", "redrafter",
+    "t5"
 ])
 @pytest.mark.parametrize("run_fp8", [False, True], ids=["", "fp8"])
 def test_model(build_google_tests, model, prepare_model, run_model_tests,
diff --git a/tests/integration/test_lists/test-db/l0_a30.yml b/tests/integration/test_lists/test-db/l0_a30.yml
index 6765e9ccd..f0f3fcbfe 100644
--- a/tests/integration/test_lists/test-db/l0_a30.yml
+++ b/tests/integration/test_lists/test-db/l0_a30.yml
@@ -40,6 +40,7 @@ l0_a30:
   # ------------- CPP tests ---------------
   - test_cpp.py::test_unit_tests[80]
   - test_cpp.py::test_model[gpt-80]
+  - test_cpp.py::test_model[gpt_executor-80]
   - test_cpp.py::test_benchmarks[gpt-80]
 - condition:
     ranges:

```

### Analysis Summary

Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 21a696b671 - [None][feat] Optimize the q3n decode kernel with IO read (#11344)

- **Date**: 2026-03-12
- **Author**: JadoTu
- **Categories**: Kernel Optimization

### Optimization Techniques

- Operator fusion
- Triton kernel

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/modules/fla/fused_sigmoid_gating_recurrent.py | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fla/fused_sigmoid_gating_recurrent.py b/tensorrt_llm/_torch/modules/fla/fused_sigmoid_gating_recurrent.py
index 87902a68f..2d3b1987c 100644
--- a/tensorrt_llm/_torch/modules/fla/fused_sigmoid_gating_recurrent.py
+++ b/tensorrt_llm/_torch/modules/fla/fused_sigmoid_gating_recurrent.py
@@ -177,7 +177,7 @@ def fused_sigmoid_gating_delta_rule_update(
     B, T, H, K, V = *k.shape, v.shape[-1]
     HV = v.shape[2]
     N = B if cu_seqlens is None else len(cu_seqlens) - 1
-    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
+    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
     NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
     assert NK == 1, "NK > 1 is not supported yet"
     num_stages = 3

```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 21a93fbf9d - [TRTLLM-9992][perf] Enable PDL for CuteDSL kernels and overlap MoeOutputMemset (#10043)

- **Date**: 2025-12-20
- **Author**: Enwei Zhu
- **Categories**: Kernel Optimization

### Optimization Techniques

- Torch compilation/JIT optimization
- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- Async/stream-based execution
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Speculative decoding
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- Decode/generation phase

### Changed Files

```
.../_torch/custom_ops/cute_dsl_custom_ops.py       |  26 ----
 ...contiguous_gather_grouped_gemm_swiglu_fusion.py |  49 ++-----
 .../blockscaled_contiguous_grouped_gemm.py         |  12 +-
 ...aled_contiguous_grouped_gemm_finalize_fusion.py |  84 +++--------
 ...scaled_contiguous_grouped_gemm_swiglu_fusion.py |  49 ++-----
 .../blackwell/dense_blockscaled_gemm_persistent.py |   8 +-
 .../_torch/cute_dsl_kernels/blackwell/utils.py     | 154 +++++++++++++++++++++
 tensorrt_llm/_torch/models/modeling_deepseekv3.py  |   3 +-
 tensorrt_llm/_torch/models/modeling_glm.py         |   3 +-
 tensorrt_llm/_torch/models/modeling_qwen3_moe.py   |   1 +
 tensorrt_llm/_torch/models/modeling_speculative.py |   4 +-
 .../_torch/modules/fused_moe/configurable_moe.py   |   1 -
 .../_torch/modules/fused_moe/fused_moe_cute_dsl.py |  47 +++++--
 tensorrt_llm/_torch/utils.py                       |   1 +
 14 files changed, 259 insertions(+), 183 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py b/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
index 2405d3e5f..fbde925a2 100644
--- a/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
+++ b/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
@@ -2178,32 +2178,6 @@ if IS_CUTLASS_DSL_AVAILABLE:
                                    device=input_scale.device)
         return output, output_scale
 
-    class FusedMoEInputsHelper:
-
-        def __init__(self, num_experts: int, top_k: int, num_local_experts: int,
-                     local_expert_offset: int):
-            self.num_experts = num_experts
-            self.top_k = top_k
-            self.num_local_experts = num_local_experts
-            self.local_expert_offset = local_expert_offset
-
-        def infer_shape_num_tokens(self, input_shapes: List[torch.Size]) -> int:
-            return input_shapes[0][0]
-
-        def inputs_pre_hook(self,
-                            inputs: List[torch.Tensor]) -> List[torch.Tensor]:
-            x, x_sf, token_selected_experts, token_final_scales, *others = inputs
-            num_tokens = token_selected_experts.size(0)
-            new_token_final_scales, new_token_selected_experts = torch.randn(
-                num_tokens,
-                self.num_experts,
-                device=token_selected_experts.device).topk(self.top_k, dim=-1)
-            new_token_selected_experts = new_token_selected_experts.to(
-                token_selected_experts.dtype)
-            new_token_final_scales = new_token_final_scales.softmax(dim=-1).to(
-                token_final_scales.dtype)
-            return x, x_sf, new_token_selected_experts, new_token_final_scales, *others
-
     class Sm100BlockScaledFusedMoERunner(TunableRunner):
         tuning_config_cache = dict()
 
diff --git a/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py b/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py
index 3540f9155..5dd84c57e 100644
--- a/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py
+++ b/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py
@@ -35,44 +35,18 @@ import cutlass.pipeline as pipeline
 import cutlass.utils as utils
 import cutlass.utils.blackwell_helpers as sm100_utils
 import cutlass.utils.blockscaled_layout as blockscaled_utils
-from cutlass._mlir.dialects import math, nvvm
+from cutlass._mlir.dialects import math
 from cutlass.cute.nvgpu import cpasync, tcgen05
-from cutlass.cute.typing import Float32
-from cutlass.cutlass_dsl import T, dsl_user_op
 
 from .custom_pipeline import PipelineCpAsyncUmma
-from .utils import is_power_of_2
-
-
-@dsl_user_op
-def fmin(
-    a: Union[float, Float32], b: Union[float, Float32], *, nan=False, loc=None, ip=None
-) -> Float32:
-    return Float32(
-        nvvm.fm
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 225d3a9001 - [None][perf] TRTLLM MoE maps to lower tuning buckets when ep>1 (#9998)

- **Date**: 2026-01-06
- **Author**: Anthony Chang
- **Categories**: General Performance

### Optimization Techniques

- FP8 quantization
- KV cache optimization
- MoE optimization
- GEMM optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/autotuner.py                   |  65 ++++++--
 .../_torch/custom_ops/trtllm_gen_custom_ops.py     | 183 ++++++++++++---------
 tests/unittest/_torch/misc/test_autotuner.py       |  78 ++++++++-
 3 files changed, 232 insertions(+), 94 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/autotuner.py b/tensorrt_llm/_torch/autotuner.py
index 01193e68b..1f1827068 100644
--- a/tensorrt_llm/_torch/autotuner.py
+++ b/tensorrt_llm/_torch/autotuner.py
@@ -51,7 +51,7 @@ class DynamicTensorSpec:
         input_idx: The index of the input tensor.
         dim_idx: The index of the dimension to tune.
         gen_tuning_buckets: A tuple of values to try or a function generating values.
-        map_to_tuning_buckets: A function to map dimensions to valid values during inference.
+        map_to_tuning_buckets: A function to map dimensions to tuning buckets during inference.
     """
     input_idx: int
     dim_idx: int
@@ -83,7 +83,7 @@ class TuningConfig:
             should be tuned to optimize performance. Each spec defines:
             - Which input tensor dimension is dynamic
             - How to generate tuning values
-            - How to map dimensions to valid values during inference
+            - How to map dimensions to tuning values during inference
 
             Example:
                 >>> config = TuningConfig(
@@ -390,6 +390,7 @@ class AutoTunerProfilingCache:
         runners: List[TunableRunner],
         input_shapes: Tuple[torch.Size],
         tuning_config: TuningConfig,
+        apply_map_to_tuning_buckets: bool = True,
     ) -> Tuple[bool, int, int, Dict[str, Any], OptimizationProfile]:
         """Search for cached profiling results matching the current configuration.
 
@@ -397,6 +398,8 @@ class AutoTunerProfilingCache:
             custom_op (str): The name of the custom operation to be tuned
             runners (List[TunableRunner]): List of candidate implementations to profile
             profile (OptimizationProfile): Optimization profile
+            apply_map_to_tuning_buckets: If True, apply map_to_tuning_buckets for runtime cache lookups.
+                If False, use raw bucket values for tuning cache storage.
 
         Returns:
             A tuple containing:
@@ -404,8 +407,9 @@ class AutoTunerProfilingCache:
             runner_id is the index in the current runners list
         """
         for idx, r in enumerate(runners):
-            if (cache_key := self.get_cache_key(custom_op, r, input_shapes,
-                                                tuning_config)) in self.cache:
+            if (cache_key := self.get_cache_key(
+                    custom_op, r, input_shapes, tuning_config,
+                    apply_map_to_tuning_buckets)) in self.cache:
                 # Return the current index in runners list, not the cached runner_id
                 cached_runner_id, tactic, min_time = self.cache[cache_key]
                 return True, idx, tactic, min_time
@@ -418,6 +422,7 @@ class AutoTunerProfilingCache:
         runner: TunableRunner,
         input_shapes: Tuple[torch.Size],
         tuning_config: TuningConfig,
+        apply_map_to_tuning_buckets: bool = True,
     ) -> Tuple:
         return (
             custom_op,
@@ -428,6 +433,7 @@ class A
```

### Analysis Summary

Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 22b45ff9c7 - [TRTLLM-7758][feat] Phi4-mm image modality inference optimization (#7918)

- **Date**: 2025-09-25
- **Author**: Wanli Jiang
- **Categories**: General Performance

### Optimization Techniques

- Torch compilation/JIT optimization
- Custom CUDA kernel
- Operator fusion
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- PyTorch built-in optimized ops
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase
- Disaggregated serving

### Changed Files

```
tensorrt_llm/_torch/models/modeling_phi4mm.py | 712 +++++++++++++++++++++-----
 1 file changed, 580 insertions(+), 132 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_phi4mm.py b/tensorrt_llm/_torch/models/modeling_phi4mm.py
index d5cafc16c..129c43bc6 100644
--- a/tensorrt_llm/_torch/models/modeling_phi4mm.py
+++ b/tensorrt_llm/_torch/models/modeling_phi4mm.py
@@ -1,21 +1,34 @@
 # Plan for phi4-mm model support.
 # (done) step 1: support legacy inference pipeline for phi4-mm model.
 # (done) step 2: refactor the inference pipeline to use AGGREGATE mode (https://github.com/NVIDIA/TensorRT-LLM/pull/5522).
-# (todo) step 3: optimization
+# (done) step 3: optimization phi4-mm image modality inference.
+# (todo) step 4: misc tasks:
+#   * optimize audio modality.
 #   * use TRTLLM-attention to replace original pytorch attention in vision/audio encoders.
 #   * use data parallel to accelerate inference.
 
 import copy
+import enum
 import importlib
+import math
 import os
 import sys
 from pathlib import Path
-from typing import List, Optional, Tuple
+from types import MethodType
+from typing import Dict, List, Optional, Tuple, Union
 
 import torch
+import torchvision
 import transformers
+from einops import rearrange
 from PIL import Image
+from torchvision.transforms.functional import get_image_size, pad, resize
+from transformers.image_processing_utils import BatchFeature
+from transformers.image_utils import (ImageInput, is_pil_image,
+                                      make_list_of_images, valid_images)
+from transformers.utils import TensorType
 
+from tensorrt_llm._utils import nvtx_range
 from tensorrt_llm.inputs.multimodal import MultimodalParams
 
 from ...executor.request import LoRARequest
@@ -34,6 +47,7 @@ from .modeling_multimodal_utils import (find_input_mm_embeds, fuse_input_embeds,
 from .modeling_utils import register_auto_model
 
 # Special token ids from the original Phi-4-multimodal-instruct implementation
+# Hardcoded in https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/processing_phi4mm.py#L44.
 _IMAGE_SPECIAL_TOKEN_ID = 200010  # '<|endoftext10|>' from HF `modeling_phi4mm.py`
 _AUDIO_SPECIAL_TOKEN_ID = 200011  # '<|endoftext11|>' from HF `modeling_phi4mm.py`
 _PAD_TOKEN_ID = 199999  # '<|endoftext|>' from HF `special_tokens_map.json`
@@ -42,7 +56,12 @@ _COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE = [-9999,
 _COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE = [float('-inf'), -10000
                                             ]  # from HF `modeling_phi4mm.py`
 
-# Below classes will be loaded from HuggingFace codes, rather than using transformers version,
+# SigLip input config.
+# Hardcoded in https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/processing_phi4mm.py#L195.
+_BASE_RESOLUTION = 448
+_MASK_RESOLUTION = _BASE_RESOLUTION // 14
+
+# Below classes will be loaded from HuggingFace code, rather than using transformers version,
 # since transformers version is not compatible with checkpoints and configs from `microsoft/Phi-4-multimodal-instruct`.
 Phi4MMAudioEmbedding = None
 Phi4MMImageEmbedding = None
@@ -50,6 +
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 22c1748b80 - [TRTLLM-8816][feat] add optimized trtllm-gen attention kernels on sm103 (#9081)

- **Date**: 2025-11-13
- **Author**: Perkz Zheng
- **Categories**: Kernel Optimization

### Optimization Techniques

- FP8 quantization
- KV cache optimization
- Attention mechanism optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU

### Changed Files

```
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
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...ausalP64VarSeqQ128Kv128StaticContext_cubin.cpp} |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...eQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...eQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
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
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
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
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...eQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
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
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
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
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
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
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...eQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
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
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
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
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
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
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
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
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
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
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
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
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
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
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
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
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...enseVarSeqQ128Kv128PersistentContext_cubin.cpp} |    4 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...eQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...eQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 +
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 +
 .../trtllmGenKernels/fmha/cubin/kernelMetaInfo.h   | 5495 ++++++++++++--------
 2211 files changed, 7704 insertions(+), 5567 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128PersistentContext_cubin.cpp b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128PersistentContext_cubin.cpp
index 6c5370e24..066477f2b 100644
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128PersistentContext_cubin.cpp
+++ b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128PersistentContext_cubin.cpp
@@ -1,3 +1,3 @@
 version https://git-lfs.github.com/spec/v1
-oid sha256:5d102f6cbad85254651f267f36559243fcf009bdb2f9a96b374906ffd8f90217
-size 664593
+oid sha256:a74c90bed8cdfc61d4d30985f0a037b948a845387af20641313e17b1892c830b
+size 612398
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128StaticContext_cubin.cpp b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128StaticContext_cubin.cpp
index 6685a17e9..dd8d82a3f 100644
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128StaticContext_cubin.cpp
+++ b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128StaticContext_cubin.cpp
@@ -1,3 +1,3 @@
 version https://git-lfs.github.com/spec/v1
-oid sha256:d3b4b62ab07c9f9196472401b0f58043435ef6616c2d47724484ffa84068d44b
-size 572774
+oid sha256:47b29936b5167f32d44959bdbbdb8943a3b4152c0f42860fbcafbd27fbe930d4
+size 547072
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP64VarSeqQ128Kv128PersistentContext_cubin.cpp b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP64VarSeqQ128Kv128PersistentContext_cubin.cpp
index a9a737b29..b9e8644ac 100644
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP64VarSeqQ128Kv128PersistentContext_cubin.cpp
+++ b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP64VarSeqQ128Kv128PersistentContext_cubin.cpp
@@ -1,3 +1,3 @@
 version https://git-lfs.github.com/spec/v1
-oid sha256:2f49a6f8b4480f0e7874d2132ffdde83a450694c540d4092ff85cb56bf526267
-size 648807
+oid sha256:64a5d40ff29adb68f36625bd0f0fba00b347fc0e20c7acf901eea8b97919a1bf
+size 601346
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP64VarSeqQ128Kv128StaticContext_cubin.cpp b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP64VarSeqQ128Kv128StaticContext_cubin.cpp
index ceaedb765..5af
```

### Analysis Summary

Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 255779a91d - Chore: fuse _merge_requests method into _fetch_new_requests method (#4689)

- **Date**: 2025-05-29
- **Author**: QI JUN
- **Categories**: Fusion

### Optimization Techniques

- Custom CUDA kernel
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Speculative decoding
- Attention mechanism optimization

### Applicable Conditions

- Disaggregated serving

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/py_executor.py | 78 ++++++++++++---------------
 1 file changed, 33 insertions(+), 45 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/py_executor.py b/tensorrt_llm/_torch/pyexecutor/py_executor.py
index 727088c6d..4c4fc628c 100644
--- a/tensorrt_llm/_torch/pyexecutor/py_executor.py
+++ b/tensorrt_llm/_torch/pyexecutor/py_executor.py
@@ -683,19 +683,16 @@ class PyExecutor:
 
     def _executor_loop_pp(self):
         torch.cuda.set_device(self.device_id)
-        got_finish_signal = False
         microbatch_id = 0
         with self._profiler() as profile_step:
             iter_start_time = time.time()
             iter_stats = None
-            while not got_finish_signal or len(self.active_requests) > 0:
+            while not self.is_shutdown or len(self.active_requests) > 0:
                 profile_step()
                 if self.enable_iter_perf_stats:
                     iter_start_time = time.time()
                 new_requests = self._fetch_new_requests()
-                got_finish_signal = self._merge_requests(
-                    new_requests) or got_finish_signal
-                if got_finish_signal and len(self.active_requests) == 0:
+                if self.is_shutdown and len(self.active_requests) == 0:
                     break
 
                 if self.enable_iter_perf_stats:
@@ -703,8 +700,7 @@ class PyExecutor:
                         len(new_requests),
                         self.new_active_requests_queue_latency_ms)
 
-                if not got_finish_signal:
-                    self._pad_attention_dp_dummy_request()
+                self._pad_attention_dp_dummy_request()
 
                 scheduled_batch, _, _ = self._schedule()
 
@@ -839,7 +835,6 @@ class PyExecutor:
 
     def _executor_loop(self):
         torch.cuda.set_device(self.device_id)
-        got_finish_signal = False
         is_ngram = hasattr(
             self.model_engine, "spec_config"
         ) and self.model_engine.spec_config is not None and self.model_engine.spec_config.spec_dec_mode.is_ngram(
@@ -847,14 +842,12 @@ class PyExecutor:
         with self._profiler() as profile_step:
             iter_start_time = time.time()
             iter_stats = None
-            while not got_finish_signal or len(self.active_requests) > 0:
+            while not self.is_shutdown or len(self.active_requests) > 0:
                 profile_step()
                 if self.enable_iter_perf_stats:
                     iter_start_time = time.time()
                 new_requests = self._fetch_new_requests()
-                got_finish_signal = self._merge_requests(
-                    new_requests) or got_finish_signal
-                if got_finish_signal and len(self.active_requests) == 0:
+                if self.is_shutdown and len(self.active_requests) == 0:
                     break
 
                 if self.kv_cache_transceiver:
@@ -865,8 +858,7 @@ class PyExecutor:
                         len(new_requests),
                         self.new_active_requests_queue_latency_ms)
 
-                if not got_finish_signal:
-                
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 25cd4f215e - [PERF] Move calculation Qwen2-VL's rotary_cos_sin to LLM worker process (#6004)

- **Date**: 2025-07-31
- **Author**: Vadim Gimpelson
- **Categories**: General Performance

### Optimization Techniques

- Operator fusion
- Batching optimization
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/models/modeling_qwen2vl.py | 111 +++++++++++++++----------
 1 file changed, 67 insertions(+), 44 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_qwen2vl.py b/tensorrt_llm/_torch/models/modeling_qwen2vl.py
index 3371bb6fc..03f15c37b 100644
--- a/tensorrt_llm/_torch/models/modeling_qwen2vl.py
+++ b/tensorrt_llm/_torch/models/modeling_qwen2vl.py
@@ -10,6 +10,7 @@ from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
 
 from tensorrt_llm.inputs.multimodal import MultimodalParams
 
+from ..._utils import nvtx_range_debug
 from ...functional import RopeEmbeddingUtils, RotaryScalingType
 from ...inputs import (ExtraProcessedInputs, InputProcessor, TextPrompt,
                        register_input_processor)
@@ -41,7 +42,6 @@ class Qwen2VLInputProcessorBase(InputProcessor):
             trust_remote_code=trust_remote_code)
 
         self.tllm_multimodal_token_id = self.model_config.vocab_size + 1
-        self._post_init_()
 
     @classmethod
     def get_rope_index(
@@ -217,22 +217,6 @@ class Qwen2VLInputProcessorBase(InputProcessor):
             mrope_position_deltas, device=input_ids.device).unsqueeze(1)
         return position_ids, mrope_position_deltas
 
-    def _post_init_(self):
-        _, rotary_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
-            num_pos=self.model_config.max_position_embeddings,
-            dim=int(self.model_config.hidden_size /
-                    self.model_config.num_attention_heads),
-            theta=float(self.model_config.rope_theta),
-            scale_type=RotaryScalingType.mrope)
-        self.rotary_cos_sin = torch.from_numpy(rotary_cos_sin)
-        self.rotary_cos_sin = self.rotary_cos_sin.reshape(
-            self.model_config.max_position_embeddings,
-            int(self.model_config.hidden_size /
-                self.model_config.num_attention_heads / 2), 2)
-
-        self.cos_ori = self.rotary_cos_sin[:, :, 0]
-        self.sin_ori = self.rotary_cos_sin[:, :, 1]
-
     def get_num_tokens_per_image(
         self,
         *,
@@ -304,30 +288,8 @@ class Qwen2VLInputProcessorBase(InputProcessor):
             self.model_config, input_ids, image_grid_thw, video_grid_thw,
             attention_mask, second_per_grid_ts)
 
-        mrope_position_ids = mrope_position_ids.transpose(1, 0)
-        mrope_position_ids_padding = torch.zeros(
-            mrope_position_ids.shape[:-1] +
-            (self.model_config.max_position_embeddings, ),
-            dtype=torch.int32,
-            device=input_ids.device)
-        mrope_position_ids_padding[:, :, :mrope_position_ids.
-                                   shape[-1]] = mrope_position_ids
-        cos = self.cos_ori[mrope_position_ids_padding]
-        sin = self.sin_ori[mrope_position_ids_padding]
-
-        mrope_section = [16, 24, 24]
-        cos = torch.cat([
-            m[:, i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))
-        ],
-                        dim=-1).unsqueeze(-1)
-        sin = torch.cat([
-            m[:, i % 3] for i, m in enumerate(sin.split(
```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 264d38e6c5 - [TRTLLM-9175][test] ensure sampling is async (#9076)

- **Date**: 2025-11-12
- **Author**: mpikulski
- **Categories**: Parallelism/Async

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- Integer quantization
- KV cache optimization
- Batching optimization
- Reduce synchronization overhead
- Speculative decoding

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../unittest/_torch/sampler/test_torch_sampler.py  | 1075 ++++++++++++--------
 tests/unittest/utils/util.py                       |   35 +-
 2 files changed, 677 insertions(+), 433 deletions(-)
```

### Diff Preview

```diff
diff --git a/tests/unittest/_torch/sampler/test_torch_sampler.py b/tests/unittest/_torch/sampler/test_torch_sampler.py
index 8072e638e..75a17044b 100644
--- a/tests/unittest/_torch/sampler/test_torch_sampler.py
+++ b/tests/unittest/_torch/sampler/test_torch_sampler.py
@@ -12,11 +12,21 @@
 # See the License for the specific language governing permissions and
 # limitations under the License.
 
-from contextlib import contextmanager
+from contextlib import contextmanager, nullcontext
 from dataclasses import dataclass
 from itertools import product
-from random import shuffle as shuffle_inplace
-from typing import Callable, Final, Generator, Optional, Type, Union, cast
+from typing import (
+    Callable,
+    ContextManager,
+    Final,
+    Generator,
+    Optional,
+    Protocol,
+    Type,
+    TypeVar,
+    Union,
+    cast,
+)
 
 import flashinfer.sampling
 import numpy as np
@@ -344,6 +354,52 @@ class TestStrategySelection:
         assert torch_sampler.should_provide_draft_probs(request) == (not is_greedy)
 
 
+class UutProvider(Protocol):
+    def __call__(self, is_warmup: bool) -> ContextManager[Callable[[], None]]: ...
+
+
+def _run_test_with_warmup(
+    uut_provider: UutProvider,
+    warmup_sizes_bytes: tuple[int] = (4 * 2**30,),
+    max_sync_s: Optional[float] = None,
+):
+    """Run UUT including setup and warmup.
+
+    This is mainly used to check that the UUT does not CUDA device sync. Thus,
+    given that PyTorch's caching memory allocator can device sync when it runs
+    out of cached GPU memory segments, the warmup allocates some GPU memory.
+
+    The warmup also runs the test once. This avoids issues with things like lazy loading
+    of device code. The UUT provider can use the 'is_warmup' argument to adapt its
+    behavior to the warmup and final test runs.
+
+    If max_sync_s is provided, this helper checks that the UUT does not device sync,
+    assuming that the sync (CPU) part of the code takes no longer than max_sync_s
+    seconds to complete.
+
+    It is the user's responsibility to ensure that the amount of submitted work
+    does not exceed the CUDA driver/device queue capacity, which would make
+    the execution appear synchronous.
+    """
+    with torch.cuda.Stream():
+        with uut_provider(is_warmup=True) as uut:
+            bufs = []
+            for warmup_size in warmup_sizes_bytes:
+                bufs.append(
+                    torch.ones(warmup_size, device=torch.cuda.current_device(), dtype=torch.int8)
+                )
+            del bufs
+            uut()
+
+        with uut_provider(is_warmup=False) as uut:
+            with (
+                assert_no_cuda_sync(sync_timeout_s=max_sync_s)
+                if max_sync_s is not None
+                else nullcontext()
+            ):
+                uut()
+
+
 @force_ampere
 @pytest.mark.parametrize(
     "draft_len, with_ctx, with_gen",
@@ -363,7 +419,7 @@ def test_select_generated_logits(draft_len: int, with_ctx: bool, with_gen
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 2967d299fb - [TRTLLM-10271][test] Add Spark QA functional and performance cases (#10564)

- **Date**: 2026-01-13
- **Author**: JennyLiu
- **Categories**: General Performance

### Optimization Techniques

- Vectorized memory access
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Speculative decoding
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- NVIDIA Hopper (SM90) GPU

### Changed Files

```
tests/integration/defs/perf/test_perf.py           | 30 ++++++++++-
 tests/integration/defs/test_e2e.py                 | 53 +++++++++++++++++--
 .../integration/test_lists/qa/llm_digits_core.txt  | 39 ++++++++++++++
 .../integration/test_lists/qa/llm_digits_func.txt  | 59 ++++++++++++++--------
 .../integration/test_lists/qa/llm_digits_perf.txt  | 28 ----------
 .../integration/test_lists/qa/llm_digits_perf.yml  | 47 +++++++++++++++++
 .../integration/test_lists/qa/llm_perf_sanity.yml  |  2 +-
 7 files changed, 205 insertions(+), 53 deletions(-)
```

### Diff Preview

```diff
diff --git a/tests/integration/defs/perf/test_perf.py b/tests/integration/defs/perf/test_perf.py
index 90931b2da..a0db92426 100644
--- a/tests/integration/defs/perf/test_perf.py
+++ b/tests/integration/defs/perf/test_perf.py
@@ -69,6 +69,8 @@ MODEL_PATH_DICT = {
     "nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1",
     "llama_v3.3_nemotron_super_49b_fp8":
     "nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1-FP8",
+    "llama_v3.3_nemotron_super_49b_v1.5_fp8":
+    "nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1_5-FP8",
     "llama_v3.1_nemotron_ultra_253b":
     "nemotron-nas/Llama-3_1-Nemotron-Ultra-253B-v1",
     "llama_v3.1_nemotron_ultra_253b_fp8":
@@ -90,11 +92,16 @@ MODEL_PATH_DICT = {
     "modelopt-hf-model-hub/Mixtral-8x7B-Instruct-v0.1-fp4",
     "mistral_nemo_12b_base": "Mistral-Nemo-Base-2407",
     "deepseek_r1_distill_qwen_32b": "DeepSeek-R1/DeepSeek-R1-Distill-Qwen-32B",
+    "deepseek_r1_distill_llama_70b":
+    "DeepSeek-R1/DeepSeek-R1-Distill-Llama-70B/",
     "mixtral_8x22b_v0.1": "Mixtral-8x22B-v0.1",
     "mistral_7b_v0.1": "mistral-7b-v0.1",
     "ministral_8b": "Ministral-8B-Instruct-2410",
     "ministral_8b_fp8": "Ministral-8B-Instruct-2410-FP8",
     "gemma_3_1b_it": "gemma/gemma-3-1b-it",
+    "gemma_3_27b_it": "gemma/gemma-3-27b-it",
+    "gemma_3_27b_it_fp8": "gemma/gemma-3-27b-it-fp8",
+    "gemma_3_27b_it_fp4": "gemma/gemma-3-27b-it-FP4",
     "deepseek_r1_fp8": "DeepSeek-R1/DeepSeek-R1",
     "deepseek_r1_nvfp4": "DeepSeek-R1/DeepSeek-R1-FP4",
     "deepseek_r1_0528_fp8": "DeepSeek-R1/DeepSeek-R1-0528/",
@@ -106,8 +113,21 @@ MODEL_PATH_DICT = {
     "qwen_14b_chat": "Qwen-14B-Chat",
     "qwen3_0.6b": "Qwen3/Qwen3-0.6B",
     "qwen3_4b_eagle3": "Qwen3/Qwen3-4B",
+    "qwen3_8b": "Qwen3/Qwen3-8B",
+    "qwen3_8b_fp8": "Qwen3/nvidia-Qwen3-8B-FP8",
+    "qwen3_8b_fp4": "Qwen3/nvidia-Qwen3-8B-NVFP4",
+    "qwen3_14b": "Qwen3/Qwen3-14B",
+    "qwen3_14b_fp8": "Qwen3/nvidia-Qwen3-14B-FP8",
+    "qwen3_14b_fp4": "Qwen3/nvidia-Qwen3-14B-NVFP4",
+    "qwen3_30b_a3b": "Qwen3/Qwen3-30B-A3B",
+    "qwen3_30b_a3b_fp4": "Qwen3/saved_models_Qwen3-30B-A3B_nvfp4_hf",
+    "qwen3_32b": "Qwen3/Qwen3-32B",
+    "qwen3_32b_fp4": "Qwen3/nvidia-Qwen3-32B-NVFP4",
     "qwen3_235b_a22b_fp8": "Qwen3/saved_models_Qwen3-235B-A22B_fp8_hf",
     "qwen3_235b_a22b_fp4": "Qwen3/saved_models_Qwen3-235B-A22B_nvfp4_hf",
+    "qwen2_5_vl_7b_instruct": "multimodals/Qwen2.5-VL-7B-Instruct",
+    "qwen2_5_vl_7b_instruct_fp8": "multimodals/Qwen2.5-VL-7B-Instruct-FP8",
+    "qwen2_5_vl_7b_instruct_fp4": "multimodals/Qwen2.5-VL-7B-Instruct-FP4",
     "starcoder2_3b": "starcoder2-3b",
     "starcoder2_7b": "starcoder2-7b",
     "starcoder2_15b": "starcoder2-15b",
@@ -126,9 +146,14 @@ MODEL_PATH_DICT = {
     "gpt_20b": "gpt-neox-20b",
     "gpt_350m_moe": "gpt2-medium",
     "phi_4_mini_instruct": "Phi-4-mini-instruct",
+    "phi_4_reasoning_plus": "Phi-4-reasoning-plus",
+    "phi_4_reasoning_plus_fp8": "nvidia-Phi-4-reasoning-plus-FP8",
+    "phi_4_reasoning_p
```

### Analysis Summary

Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 29e63d3bc2 - [https://nvbugs/5532248][fix] Fix fused_moe OOM (#7931)

- **Date**: 2025-09-24
- **Author**: HuiGao-NV
- **Categories**: Memory Optimization, Fusion

### Optimization Techniques

- Async/stream-based execution
- FP8 quantization
- Integer quantization
- Parallelism optimization
- Speculative decoding
- MoE optimization
- GEMM optimization

### Applicable Conditions

- Prefill phase

### Changed Files

```
cpp/tensorrt_llm/thop/moeOp.cpp         | 10 +++++++---
 tests/integration/test_lists/waives.txt |  2 --
 2 files changed, 7 insertions(+), 5 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/thop/moeOp.cpp b/cpp/tensorrt_llm/thop/moeOp.cpp
index 5c4c53fe2..9b9731a07 100644
--- a/cpp/tensorrt_llm/thop/moeOp.cpp
+++ b/cpp/tensorrt_llm/thop/moeOp.cpp
@@ -390,12 +390,14 @@ public:
 
         auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
 
-        std::vector<int64_t> output_shape = {num_rows, unpadded_hidden_size_val};
-        auto output = torch::empty(output_shape, input.options().dtype(mOutputDtype));
-
         WorkspaceInfo const& workspace_info = getWorkspaceInfo(num_rows, hidden_size, inter_size, num_experts_total,
             static_cast<int>(experts_per_token), base_activation_type, parallelism_config, min_latency_mode, stream);
 
+        // output is smaller than workspace. Create output after workspace to avoid output_shape occupied a little
+        // piece of memory which makes a big partition of memory segment can't be used by workspace.
+        std::vector<int64_t> output_shape = {num_rows, unpadded_hidden_size_val};
+        auto output = torch::empty(output_shape, input.options().dtype(mOutputDtype));
+
         auto const quant_params = getQuantParams(num_experts_on_rank, hidden_size, inter_size, quant_scales);
         kernels::MoeMinLatencyParams min_latency_params{};
 
@@ -787,6 +789,8 @@ private:
                 TLLM_LOG_DEBUG("MoE workspace size is not enough, increase the size from %ld bytes to %ld bytes",
                     workspace_info.workspace.numel(), total_workspace_size);
             }
+            // Release memory first to avoid OOM.
+            workspace_info = WorkspaceInfo();
             workspace_info.workspace = torch::empty({static_cast<long>(total_workspace_size)},
                 torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
         }
diff --git a/tests/integration/test_lists/waives.txt b/tests/integration/test_lists/waives.txt
index cda3c151b..a7cdb1d4a 100644
--- a/tests/integration/test_lists/waives.txt
+++ b/tests/integration/test_lists/waives.txt
@@ -335,8 +335,6 @@ accuracy/test_llm_api_pytorch.py::TestDeepSeekR1::test_fp8_blockscale[throughput
 accuracy/test_llm_api_pytorch.py::TestDeepSeekR1::test_fp8_blockscale_chunked_prefill[latency] SKIP (https://nvbugs/5481198)
 accuracy/test_llm_api_pytorch.py::TestDeepSeekR1::test_fp8_blockscale_chunked_prefill[throughput] SKIP (https://nvbugs/5481198)
 accuracy/test_llm_api_pytorch.py::TestDeepSeekR1::test_nvfp4_multi_gpus[latency_trtllmgen] SKIP (https://nvbugs/5516845)
-accuracy/test_llm_api_pytorch.py::TestDeepSeekR1::test_nvfp4_multi_gpus[throughput_mtp] SKIP (https://nvbugs/5532248)
-accuracy/test_llm_api_pytorch.py::TestDeepSeekR1::test_nvfp4_multi_gpus[throughput] SKIP (https://nvbugs/5532248)
 accuracy/test_cli_flow.py::TestLlama3_1_8B::test_fp8_rowwise_tp4[disable_gemm_allreduce_plugin] SKIP (https://nvbugs/5532023)
 accuracy/test_cli_flow.py::TestMixtral8x7B::test_fp8_tp2pp2 SKIP (https://nvbugs/5532023)
 accuracy/test_llm_api.py::TestLlama3_1_8BIns
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## 2b58dba0f6 - [https://nvbugs/5524714][fix] Fix TP sharding of fused-QKV weight scales in W4A16 AWQ (#8432)

- **Date**: 2025-10-19
- **Author**: danielafrimi
- **Categories**: Fusion, Quantization Optimization

### Optimization Techniques

- Vectorized memory access
- Operator fusion
- Integer quantization
- Parallelism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/modules/linear.py | 5 +++--
 1 file changed, 3 insertions(+), 2 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/linear.py b/tensorrt_llm/_torch/modules/linear.py
index 75a8a472b..7aac183d3 100644
--- a/tensorrt_llm/_torch/modules/linear.py
+++ b/tensorrt_llm/_torch/modules/linear.py
@@ -1393,7 +1393,7 @@ class W4A16_AWQ_LinearMethod(LinearMethodBase):
         group_size = module.quant_config.group_size
         if in_features % group_size != 0:
             raise ValueError(
-                f"in_features ({self.in_features}) must be divisible by group_size ({group_size}) "
+                f"in_features ({in_features}) must be divisible by group_size ({group_size}) "
                 f"for INT4 per-group quantization scale dimensions.")
 
         module.weight_scale = Parameter(torch.empty(
@@ -1492,7 +1492,8 @@ class W4A16_AWQ_LinearMethod(LinearMethodBase):
 
         copy_weight(module.weight, fused_weight)
 
-        weight_scales = self.load_weight_scales(weights)
+        weight_scales = self.load_weight_scales(weights, module.tp_size,
+                                                module.tp_rank, module.tp_mode)
 
         # Create concatenated weight scale tensor
         cat_weight_scale = torch.cat(weight_scales, dim=0).T.contiguous()

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 2bc2acda4f - [https://nvbugs/5708901][perf] reduce logprobs=0 overhead in TorchSampler (#11983)

- **Date**: 2026-03-10
- **Author**: mpikulski
- **Categories**: General Performance

### Optimization Techniques

- Torch compilation/JIT optimization
- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- KV cache optimization
- Batching optimization
- PyTorch built-in optimized ops
- Speculative decoding

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/llm_request.py      |   2 +
 tensorrt_llm/_torch/pyexecutor/sampler.py          | 104 ++++++++++++++-------
 tensorrt_llm/_torch/pyexecutor/sampling_utils.py   |  71 ++++++++++++++
 .../_torch/sampler/test_logits_logprobs.py         |  44 +++++----
 4 files changed, 169 insertions(+), 52 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/llm_request.py b/tensorrt_llm/_torch/pyexecutor/llm_request.py
index 452092a84..7b9f8ce53 100644
--- a/tensorrt_llm/_torch/pyexecutor/llm_request.py
+++ b/tensorrt_llm/_torch/pyexecutor/llm_request.py
@@ -234,6 +234,8 @@ class LogProbStorage:
             if cum_log_probs is not None:
                 self.cum_log_probs[beam_idx] = cum_log_probs[beam_idx]
             else:
+                # FIXME: This relies on the ordering of LogProb's in the dictionary. TorchSampler ensures
+                #        that the sampled logprob is in the first position.
                 self.cum_log_probs[beam_idx] += sum(
                     next(iter(prob.values())).logprob for prob in probs)
 
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index e6c17a51d..a723e66fc 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -24,7 +24,6 @@ from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeAlias
 
 import numpy as np
 import torch
-import torch.nn.functional as F
 
 from tensorrt_llm._torch.pyexecutor.make_decoding_batch_input_output import (
     MakeDecodingBatchInputOutput,
@@ -78,6 +77,7 @@ from .sampling_utils import (
     Strategy,
     StrategyMetadata,
     UtilsSamplingParams,
+    _Fusions,
     get_rejected_indices,
     resolve_sampling_strategy,
     sample,
@@ -416,8 +416,8 @@ def _request_sampling_params_cachable(params: UtilsSamplingParams) -> bool:
 def _request_strategy(request: LlmRequest, *, vocab_size: int) -> Strategy:
     # We try to cache the resolved strategy on the request object, as it's not cheap enough to
     # resolve it on every iteration.
-    if hasattr(request, "py_sampling_strategy"):
-        return request.py_sampling_strategy
+    if (cached_sampling_strategy := getattr(request, "py_sampling_strategy", None)) is not None:
+        return cast(Strategy, cached_sampling_strategy)
 
     params = _request_get_sampling_params(request)
     sampling_strategy = resolve_sampling_strategy(params, vocab_size=vocab_size)
@@ -2189,7 +2189,6 @@ class TorchSampler(Sampler[SampleStateTorch], AsyncWorkerMixin):
         """
         Check if we can use the fast argmax path for greedy sampling.
         """
-
         # Check if all requests use greedy sampling and don't require features
         # that the fast path skips
         for req in requests:
@@ -2379,28 +2378,45 @@ class TorchSampler(Sampler[SampleStateTorch], AsyncWorkerMixin):
         sampled_log_probs_vals_list = logprobs_state_list.sampled_vals[req_seq_slot]
         sampled_log_probs_rank_list = logprobs_state_list.sampled_rank[req_seq_slot]
 
-        token_log_probs: list[list[dict[int, Logprob]]] = []
-        for beam_idx in range(beam_width):
-            beam_token_log_probs: list[dict[int, Logprob]] = []
-            for step_idx, (topk_token, topk_logprob) in enumerate(
-                zip(token
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 2db3d7eeba - [None][chore] Async Transfer Manager (#9891)

- **Date**: 2026-01-20
- **Author**: jthomson04
- **Categories**: Parallelism/Async

### Optimization Techniques

- Async/stream-based execution
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Speculative decoding

### Applicable Conditions

- Decode/generation phase
- Disaggregated serving

### Changed Files

```
.../tensorrt_llm/batch_manager/cacheTransceiver.h  |  16 +-
 .../batch_manager/cacheTransceiver.cpp             |  15 +-
 .../batch_manager/trtGptModelInflightBatching.cpp  |   6 +-
 .../nanobind/batch_manager/cacheTransceiver.cpp    |  24 +-
 .../pybind/batch_manager/cacheTransceiver.cpp      |  27 +-
 tensorrt_llm/_torch/pyexecutor/py_executor.py      | 365 ++++++++++++---------
 tests/integration/test_lists/test-db/l0_a10.yml    |   1 +
 .../_torch/executor/test_async_transfer_manager.py | 182 ++++++++++
 8 files changed, 470 insertions(+), 166 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/include/tensorrt_llm/batch_manager/cacheTransceiver.h b/cpp/include/tensorrt_llm/batch_manager/cacheTransceiver.h
index de68e9805..4da26f72d 100644
--- a/cpp/include/tensorrt_llm/batch_manager/cacheTransceiver.h
+++ b/cpp/include/tensorrt_llm/batch_manager/cacheTransceiver.h
@@ -190,6 +190,14 @@ public:
         std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig = std::nullopt);
 };
 
+struct RequestStatuses
+{
+    /// Requests that have completed their transfer successfully.
+    std::unordered_set<LlmRequest::RequestIdType> completedRequestIds;
+    /// Requests that have encountered an error during their transfer.
+    std::unordered_set<LlmRequest::RequestIdType> errorRequestIds;
+};
+
 class BaseCacheTransceiver
 {
 public:
@@ -202,7 +210,10 @@ public:
     virtual void requestAndReceiveSync(LlmRequest* llmRequest) = 0;
     virtual void requestAndReceiveAsync(LlmRequest* llmRequest) = 0;
 
-    virtual void checkContextTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) = 0;
+    /// Check all requests transferring context, and return the requests that have completed or encountered an error.
+    virtual RequestStatuses checkContextTransferStatus(
+        std::optional<int> const& atLeastRequestNum = std::nullopt, bool markComplete = false)
+        = 0;
 
     virtual void checkGenTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) = 0;
 
@@ -243,7 +254,8 @@ public:
     void requestAndReceiveSync(LlmRequest* llmRequest) override;
     void requestAndReceiveAsync(LlmRequest* llmRequest) override;
 
-    void checkContextTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) override;
+    RequestStatuses checkContextTransferStatus(
+        std::optional<int> const& atLeastRequestNum = std::nullopt, bool markComplete = false) override;
 
     void checkGenTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) override;
 
diff --git a/cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp b/cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp
index 7e4c26bfd..2170370d5 100644
--- a/cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp
+++ b/cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp
@@ -427,7 +427,8 @@ void updateKVCacheTransferBW(std::shared_ptr<CacheTransceiverComm> const& mComm,
     }
 }
 
-void CacheTransceiver::checkContextTransferStatus(std::optional<int> const& atLeastRequestNum)
+RequestStatuses CacheTransceiver::checkContextTransferStatus(
+    std::optional<int> const& atLeastRequestNum, bool markComplete)
 {
     bool blockAll = !atLeastRequestNum.has_value();
     std::optional<int> senderFutureTimeoutMs = std::nullopt;
@@ -486,6 +487,8 @@ void CacheTransceiver::checkContextTransferStatus(std::optional<int> const& atLe
         toCompleteIdSet.insert(request->mRequestId);
     }
 
+    RequestStatuses requestsStatus{};
+
     // Complete all the requests in toCompleteIdSet
     for (auto it = mSenderFu
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 2ea4077993 - [Model load] Fix llama min-latency model load (#5883)

- **Date**: 2025-07-14
- **Author**: Rashid Kaleem
- **Categories**: Throughput/Latency

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- FP8 quantization
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/models/modeling_llama.py       |  3 +++
 .../_torch/models/modeling_llama_min_latency.py    |  3 +++
 tensorrt_llm/_torch/models/modeling_utils.py       | 30 +++++++++++++++++++---
 3 files changed, 33 insertions(+), 3 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_llama.py b/tensorrt_llm/_torch/models/modeling_llama.py
index fc3febe83..1c17eeb5a 100644
--- a/tensorrt_llm/_torch/models/modeling_llama.py
+++ b/tensorrt_llm/_torch/models/modeling_llama.py
@@ -624,6 +624,7 @@ class Llama4Model(DecoderModel):
         self.num_hidden_layers = config.num_hidden_layers
         self.aux_stream = torch.cuda.Stream()
         self.mapping = model_config.mapping
+        self.preload_weight_modules = []
 
         if self.model_config.mapping.enable_attention_dp:
             self.embed_tokens = Embedding(
@@ -646,6 +647,7 @@ class Llama4Model(DecoderModel):
         if model_config.enable_min_latency:
             from .modeling_llama_min_latency import Llama4MinLatencyDecoderLayer
             DecoderLayerClass = Llama4MinLatencyDecoderLayer
+            self.preload_weight_modules = ["gate_up_proj"]
 
         self.layers = nn.ModuleList([
             DecoderLayerClass(
@@ -878,6 +880,7 @@ class Llama4ForConditionalGeneration(SpecDecOneEngineForCausalLM[Llama4Model,
         model_config.pretrained_config = model_config.pretrained_config.text_config
         model_config.pretrained_config.architectures = architectures
         super().__init__(Llama4Model(model_config), model_config)
+        self.preload_weight_modules = self.model.preload_weight_modules
 
     def forward(
         self,
diff --git a/tensorrt_llm/_torch/models/modeling_llama_min_latency.py b/tensorrt_llm/_torch/models/modeling_llama_min_latency.py
index 88a78cfb1..72a5b4843 100644
--- a/tensorrt_llm/_torch/models/modeling_llama_min_latency.py
+++ b/tensorrt_llm/_torch/models/modeling_llama_min_latency.py
@@ -98,6 +98,9 @@ class Llama4MinLatencyLinear(Linear):
         # After loading weights, calculate the combined scale (input_scale * weight_scale) for special kernels and
         # trtllm-gen kernels.
         if self.has_fp8_qdq:
+            if self.weight_scale.device != self.input_scale.device:
+                self.weight_scale = torch.nn.Parameter(
+                    self.weight_scale.to(self.input_scale.device))
             self.combined_scale = self.input_scale * self.weight_scale
 
             # If this is gate_up_proj + swiglu and trtllm-gen kernels will be used, we need to reorder the weights
diff --git a/tensorrt_llm/_torch/models/modeling_utils.py b/tensorrt_llm/_torch/models/modeling_utils.py
index a8ce31bf2..1dac009f5 100755
--- a/tensorrt_llm/_torch/models/modeling_utils.py
+++ b/tensorrt_llm/_torch/models/modeling_utils.py
@@ -525,7 +525,11 @@ class DecoderModelForCausalLM(nn.Module,
         )
 
     def load_weights(self, weights: Dict, skip_modules: List[str] = []):
-        _load_weights_impl(self, weights, skip_modules)
+        preload_weight_modules = getattr(self, "preload_weight_modules", None)
+        _load_weights_impl(self,
+                           weights,
+                           skip_modules,
+                           preload_weight_modules=preload_
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 2f3b2a3172 - [None][fix] Add a timeout in MNNVL throughput to prevent hangs if one rank crashes (#9532)

- **Date**: 2026-01-21
- **Author**: Daniel Stokes
- **Categories**: Throughput/Latency

### Optimization Techniques

- Custom CUDA kernel
- MoE optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU

### Changed Files

```
.../communicationKernels/moeAlltoAllKernels.cu     | 35 ++++++++++++++++++++--
 1 file changed, 32 insertions(+), 3 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu b/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu
index f32ce4c7d..303255e6a 100644
--- a/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu
+++ b/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu
@@ -32,6 +32,10 @@ namespace kernels::moe_comm
 #define ENABLE_DEBUG_PRINT 0
 #define DISABLE_SYNC_FOR_PROFILING 0
 
+#ifndef DISABLE_TIMEOUT
+#define DISABLE_TIMEOUT 0
+#endif
+
 // Macros for concise launch-time specialization
 #define SWITCH_BOOL(flag, NAME, ...)                                                                                   \
     if (flag)                                                                                                          \
@@ -141,6 +145,13 @@ namespace kernels::moe_comm
         __VA_ARGS__                                                                                                    \
     }
 
+#if DISABLE_TIMEOUT
+#define check_timeout(s) false
+#else
+// 300 * 2000 MHz - should be high enough on any GPU but will prevent a hang
+#define check_timeout(s) ((clock64() - (s)) > (300ll * 2000ll * 1000ll * 1000ll))
+#endif
+
 // ============================================================================
 // Helper Functions for Expert-to-Rank Mapping
 // ============================================================================
@@ -515,6 +526,7 @@ __global__ void moeA2ADispatchKernel(int32_t const* token_selected_experts, // [
             for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize)
             {
                 bool flag_set = false;
+                auto s = clock64();
                 do
                 {
                     uint32_t* flag_ptr = &ptrs.completion_flags[rank_id][peer_rank];
@@ -528,7 +540,15 @@ __global__ void moeA2ADispatchKernel(int32_t const* token_selected_experts, // [
                         rank_id, peer_rank, flag_value, expected_value, flag_ptr);
 #endif
                     flag_set = flag_value == expected_value;
-                } while (!flag_set);
+                } while (!flag_set && !check_timeout(s));
+
+                if (__builtin_expect(!flag_set, 0))
+                {
+                    printf("dispatch: ---Rank %d timed out waiting for completion flag from rank %d\n", rank_id,
+                        peer_rank);
+                    asm volatile("trap;");
+                    return;
+                }
             }
 #endif
         }
@@ -1038,6 +1058,7 @@ __global__ void moeA2ACombineKernel(
         for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize)
         {
             bool flag_set = false;
+            auto s = clock64();
             do
             {
                 uint32_t* flag_ptr = &ptrs.completion_flags[rank_id][peer_rank];
@@ -1046,12 +1067,20 @@ __global__ void moeA2ACombineKernel(
                 asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 2f725eae08 - [https://nvbugs/5775256] [fix] Reopen fp8_dsl_fused_moe ut. (#11779)

- **Date**: 2026-03-02
- **Author**: Li Min
- **Categories**: Fusion, Quantization Optimization

### Optimization Techniques

- Operator fusion
- FP8 quantization
- Triton kernel
- Speculative decoding
- MoE optimization
- Attention mechanism optimization

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
index 4be2b0d7e..cafd433f8 100644
--- a/tests/integration/test_lists/waives.txt
+++ b/tests/integration/test_lists/waives.txt
@@ -213,7 +213,6 @@ full:sm89/accuracy/test_disaggregated_serving.py::TestLlama3_1_8BInstruct::test_
 accuracy/test_llm_api_pytorch.py::TestLlama3_3_70BInstruct::test_fp8_eagle3_tp8[eagle3_one_model=False-torch_compile=True] SKIP (https://nvbugs/5775326)
 triton_server/test_triton.py::test_llava_onevision[llava_onevision] SKIP (https://nvbugs/5775205)
 triton_server/test_triton.py::test_gpt_ib_lad[gpt-ib-lad] SKIP (https://nvbugs/5775223)
-unittest/_torch/modules/test_fused_moe.py::test_fused_moe_fp8_blockwise_cute_dsl_multi_gpu[MoEWeightLoadingMode.FUSED_GATE_UP_PROJ-DefaultMoeRoutingMethod-1] SKIP (https://nvbugs/5775256)
 unittest/_torch/attention/test_flashinfer_star_attn.py::TestStarAttention::test_flashinfer_star_attention[num_layers:2-num_heads:32-num_kv_heads:8-head_dim:64-anchor_size:64-block_size:64-dtype:torch.float16] SKIP (https://nvbugs/5781389)
 unittest/_torch/ray_orchestrator/multi_gpu/test_ops.py::test_reducescatter_pg_op[var_len:True-seqlen:16-hidden:128] SKIP (https://nvbugs/5781383)
 cpp/test_e2e.py::test_model[-mamba-86] SKIP (https://nvbugs/5781665)

```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 31cdbdfd72 - [https://nvbugs/5808500][chore] Move DeepEPLowLatency tests to machines that support IBGDA with GPU handles (#11178)

- **Date**: 2026-02-12
- **Author**: Tailing Yuan
- **Categories**: Throughput/Latency

### Optimization Techniques

- Operator fusion
- Async/stream-based execution
- FP8 quantization
- KV cache optimization
- MoE optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU
- Disaggregated serving

### Changed Files

```
tests/integration/test_lists/test-db/l0_dgx_b200.yml | 1 +
 tests/integration/test_lists/test-db/l0_dgx_h100.yml | 1 -
 tests/integration/test_lists/waives.txt              | 1 -
 3 files changed, 1 insertion(+), 2 deletions(-)
```

### Diff Preview

```diff
diff --git a/tests/integration/test_lists/test-db/l0_dgx_b200.yml b/tests/integration/test_lists/test-db/l0_dgx_b200.yml
index ed6cc0ac4..3c8eb1b6b 100644
--- a/tests/integration/test_lists/test-db/l0_dgx_b200.yml
+++ b/tests/integration/test_lists/test-db/l0_dgx_b200.yml
@@ -16,6 +16,7 @@ l0_dgx_b200:
       orchestrator: mpi
   tests:
   - unittest/_torch/misc/test_autotuner.py::test_autotuner_distributed_strategy
+  - unittest/_torch/modules/test_fused_moe.py::test_fused_moe_alltoall[DeepEPLowLatency]
   - unittest/_torch/modules/test_fused_moe.py::test_fused_moe_alltoall_fp4[DeepEPLowLatency]
   - unittest/_torch/modules/test_fused_moe.py::test_fused_moe_alltoall_fp4[NVLinkTwoSided]
   - accuracy/test_llm_api_pytorch.py::TestNemotronV3Super::test_auto_dtype_4gpus[4-4-False-True-True]
diff --git a/tests/integration/test_lists/test-db/l0_dgx_h100.yml b/tests/integration/test_lists/test-db/l0_dgx_h100.yml
index aa9d560d7..d58b93bfe 100644
--- a/tests/integration/test_lists/test-db/l0_dgx_h100.yml
+++ b/tests/integration/test_lists/test-db/l0_dgx_h100.yml
@@ -159,7 +159,6 @@ l0_dgx_h100:
   - unittest/_torch/multi_gpu_modeling/test_deepseek.py::test_deepseek_streaming[tp1-bf16-trtllm-deepseekv3_lite]
   - unittest/_torch/multi_gpu_modeling/test_deepseek.py::test_deepseek_streaming[tp4-bf16-trtllm-deepseekv3_lite]
   - unittest/_torch/modules/test_fused_moe.py::test_fused_moe_alltoall[DeepEP]
-  - unittest/_torch/modules/test_fused_moe.py::test_fused_moe_alltoall[DeepEPLowLatency]
   - unittest/_torch/modules/test_fused_moe.py::test_fused_moe_alltoall[NVLinkTwoSided]
   - unittest/_torch/modules/test_fused_moe.py::test_fused_moe_w4afp8[MoEWeightLoadingMode.VANILLA-dtype0]
   - unittest/_torch/modules/test_fused_moe.py::test_fused_moe_w4afp8[MoEWeightLoadingMode.VANILLA-dtype1]
diff --git a/tests/integration/test_lists/waives.txt b/tests/integration/test_lists/waives.txt
index 59bbfb2e0..bc7c425a3 100644
--- a/tests/integration/test_lists/waives.txt
+++ b/tests/integration/test_lists/waives.txt
@@ -194,7 +194,6 @@ test_e2e.py::test_trtllm_bench_pytorch_backend_sanity[meta-llama/Llama-3.1-8B-ll
 accuracy/test_disaggregated_serving.py::TestLlama4ScoutInstruct::test_auto_dtype[False] SKIP (https://nvbugs/5629792)
 llmapi/test_llm_examples.py::test_llmapi_example_multilora SKIP (https://nvbugs/5636857)
 accuracy/test_llm_api_pytorch.py::TestLlama3_1_8BInstruct::test_bfloat16_4gpus[tp4-attn_backend=TRTLLM-torch_compile=False] SKIP (https://nvbugs/5616182)
-unittest/_torch/modules/test_fused_moe.py::test_fused_moe_alltoall[DeepEPLowLatency] SKIP (https://nvbugs/5808500)
 unittest/_torch/auto_deploy/unit/multigpu/test_ad_build_small_multi.py::test_build_ad[meta-llama/Meta-Llama-3.1-8B-Instruct-llm_extra_args0-2] SKIP (https://nvbugs/5680755)
 full:H100_PCIe/unittest/llmapi/test_llm_pytorch.py::test_llama_7b_multi_lora_evict_and_reload_lora_gpu_cache SKIP (https://nvbugs/5682551)
 test_e2e.py::test_openai_completions_example[trt] SKIP (https://nvbugs/57014
```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 320195dc0d - [Architecture] Refactor FusedMoE (#4790)

- **Date**: 2025-06-02
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
- Batching optimization
- PyTorch built-in optimized ops
- Speculative decoding
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- NVIDIA Hopper (SM90) GPU
- Decode/generation phase

### Changed Files

```
.../_torch/auto_deploy/custom_ops/fused_moe.py     |    2 +-
 tensorrt_llm/_torch/models/modeling_deepseekv3.py  |   43 +-
 tensorrt_llm/_torch/models/modeling_llama.py       |    6 +-
 .../_torch/models/modeling_llama_min_latency.py    |    4 +-
 tensorrt_llm/_torch/models/modeling_mixtral.py     |    4 +-
 tensorrt_llm/_torch/models/modeling_qwen3_moe.py   |   13 +-
 tensorrt_llm/_torch/models/modeling_qwen_moe.py    |    6 +-
 tensorrt_llm/_torch/models/modeling_utils.py       |    4 +-
 tensorrt_llm/_torch/modules/fused_moe.py           | 2513 --------------------
 tensorrt_llm/_torch/modules/fused_moe/__init__.py  |   22 +
 .../_torch/modules/fused_moe/create_moe.py         |  119 +
 .../_torch/modules/fused_moe/fused_moe_cutlass.py  |  641 +++++
 .../modules/fused_moe/fused_moe_trtllm_gen.py      |  217 ++
 .../_torch/modules/fused_moe/fused_moe_vanilla.py  |  540 +++++
 tensorrt_llm/_torch/modules/fused_moe/interface.py |  124 +
 .../modules/{ => fused_moe}/moe_load_balancer.py   |    0
 .../_torch/modules/fused_moe/quantization.py       | 1168 +++++++++
 tensorrt_llm/_torch/modules/fused_moe/routing.py   |  264 ++
 tensorrt_llm/_torch/modules/linear.py              |    4 +-
 .../unit/singlegpu/custom_ops/test_ad_moe_op.py    |    2 +-
 .../_torch/modeling/test_modeling_deepseek.py      |   26 +-
 tests/unittest/_torch/modules/test_fused_moe.py    |   51 +-
 .../_torch/modules/test_moe_host_sharer.py         |    3 +-
 .../_torch/modules/test_moe_load_balancer.py       |    2 +-
 24 files changed, 3203 insertions(+), 2575 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe.py b/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe.py
index 8336b8179..18d392b5f 100644
--- a/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe.py
+++ b/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe.py
@@ -3,7 +3,7 @@ from typing import List
 import torch
 import torch.nn.functional as F
 
-from ...modules.fused_moe import FusedMoE  # noqa: F401
+from ...modules.fused_moe import MoE  # noqa: F401
 
 
 @torch.library.custom_op("moe::torch_moe", mutates_args=())
diff --git a/tensorrt_llm/_torch/models/modeling_deepseekv3.py b/tensorrt_llm/_torch/models/modeling_deepseekv3.py
index 49d717c08..4e88d1a10 100644
--- a/tensorrt_llm/_torch/models/modeling_deepseekv3.py
+++ b/tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -41,6 +41,7 @@ from transformers import PretrainedConfig
 from tensorrt_llm._mnnvl_utils import MnnvlMemory
 from tensorrt_llm.functional import PositionEmbeddingType
 from tensorrt_llm.llmapi.utils import enable_llm_debug
+from tensorrt_llm.models.modeling_utils import QuantConfig
 
 from ..attention_backend import AttentionMetadata
 from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
@@ -51,10 +52,10 @@ from ..models.modeling_utils import ModelConfig
 from ..modules.attention import MLA
 from ..modules.decoder_layer import DecoderLayer
 from ..modules.embedding import Embedding
-from ..modules.fused_moe import DeepSeekV3MoeRoutingMethod, FusedMoE
+from ..modules.fused_moe import (CutlassFusedMoE, DeepSeekV3MoeRoutingMethod,
+                                 MoeLoadBalancer, create_moe)
 from ..modules.gated_mlp import GatedMLP
 from ..modules.linear import Linear
-from ..modules.moe_load_balancer import MoeLoadBalancer
 from ..modules.multi_stream_utils import maybe_execute_in_parallel
 from ..modules.rms_norm import RMSNorm
 from ..speculative import MTPEagleWorker, MTPSpecMetadata, MTPWorker
@@ -342,6 +343,7 @@ class Deepseekv3MoE(nn.Module):
                  aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
                  dtype: Optional[torch.dtype] = None,
                  model_config: ModelConfig = ModelConfig(),
+                 override_quant_config: Optional[QuantConfig] = None,
                  moe_load_balancer: Optional[MoeLoadBalancer] = None,
                  layer_idx: Optional[int] = None):
         from ..distributed import AllReduce
@@ -365,7 +367,7 @@ class Deepseekv3MoE(nn.Module):
             fuse_routing_kernel=True,
             apply_routing=False,
             moe_backend=model_config.moe_backend)
-        self.experts = FusedMoE(
+        self.experts = create_moe(
             num_experts=num_experts,
             routing_method=self.gate.routing_method,
             hidden_size=hidden_size,
@@ -374,6 +376,7 @@ class Deepseekv3MoE(nn.Module):
             reduce_results=
             False,  # In both low‑latency and attention‑DP modes, FusedMoE skips the in‑op all‑reduce.
    
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 326a201473 - [https://nvbugs/5508536][fix] Take Over (#8627): Reintroduce: Move stop_criteria to sample_async (#7041) (#8794)

- **Date**: 2025-11-07
- **Author**: Stefan Niebler
- **Categories**: Parallelism/Async

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- KV cache optimization
- Batching optimization
- Pinned memory
- PyTorch built-in optimized ops
- Reduce synchronization overhead
- Speculative decoding

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/sampler.py          | 342 ++++++++++++++++++---
 tensorrt_llm/_torch/speculative/mtp.py             |  48 +--
 .../unittest/_torch/sampler/test_torch_sampler.py  | 192 ++++++++++++
 .../test_draft_token_tree_verification.py          |   7 +-
 4 files changed, 521 insertions(+), 68 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index 276aa9770..0ca3a27bd 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -18,9 +18,11 @@ from abc import ABC, abstractmethod
 from collections import defaultdict
 from collections.abc import Iterable
 from dataclasses import dataclass
+from functools import cached_property
 from itertools import repeat
 from typing import Any, Callable, List, Optional, TypeVar, cast
 
+import numpy as np
 import torch
 import torch.nn.functional as F
 
@@ -563,9 +565,28 @@ class _UnpackedStepIndexer(_StridedStepIndexTranslator):
             raise ValueError(f"Invalid dim_order: {dim_order}")
 
 
+BEAM = 0
+MAX_BEAM_WIDTH = BEAM + 1
+
+FinishReasonsList = list[list[int]]
+
+
+@dataclass(kw_only=True)
+class SampleStateTensorsHostTorch(SampleStateTensors):
+    finish_reasons: torch.Tensor
+
+    def finish_reasons_list(self) -> FinishReasonsList:
+        """`(num_seq_slots, num_steps)`"""
+        return self.finish_reasons[:, :, BEAM].T.tolist()
+
+
+@dataclass(kw_only=True)
+class SampleStateTorch(SampleState):
+    host: SampleStateTensorsHostTorch
+
+
 class TorchSampler(Sampler):
-    BEAM = 0
-    MAX_BEAM_WIDTH = BEAM + 1
+    SampleState = SampleStateTorch
 
     @override
     def is_generation_model(self) -> bool:
@@ -575,9 +596,10 @@ class TorchSampler(Sampler):
     class Store:
         new_tokens: torch.Tensor
         """Shape: See cpp DecoderState.getAllNewTokens()"""
+        finish_reasons: torch.Tensor
 
-    def create_store(self) -> Store:
-        return self.Store(new_tokens=int_tensor(self.NEW_TOKENS_SHAPE))
+        def __post_init__(self):
+            assert self.new_tokens.shape == self.finish_reasons.shape
 
     @dataclass(frozen=True, kw_only=True)
     class Args:
@@ -590,17 +612,35 @@ class TorchSampler(Sampler):
     def __init__(self, args: Args):
         self.max_seq_len = args.max_seq_len
         self.max_tokens = args.max_total_draft_tokens + 1
-        assert args.max_beam_width == self.MAX_BEAM_WIDTH, (
-            "TorchSampler only supports beam_width = 1"
-        )
+        assert args.max_beam_width == MAX_BEAM_WIDTH, "TorchSampler only supports beam_width = 1"
         self.max_num_sequences = args.max_num_sequences
 
-        self.NEW_TOKENS_SHAPE = (self.max_tokens, self.max_num_sequences, self.MAX_BEAM_WIDTH)
         # AutoDeploy build creates the sampler in inference mode,
         # which would disallow in-place mutating of new_tokens.
         # So, we temporarily exit inference mode.
         with torch.inference_mode(False):
-            self.store = self.create_store()
+            self.store = self.Store(
+                new_tokens=int_tensor((self.max_tokens, self.max_num_sequences, MAX_BEAM_WIDTH)),
+                finish_reasons=int_tensor(
+                    (self.max_tokens, self.max_num_sequences, MAX_BEAM_WIDTH)
+                ),
+   
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Pinned (page-locked) memory enables faster host-to-device data transfers. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

