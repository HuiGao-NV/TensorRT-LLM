# Performance Optimization Analysis - Part 4

Commits 88 to 116 of 283

---

## 4e55b83101 - [None][perf] Add more optimization options for MOE CuteDSL finalized kernel (#10042)

- **Date**: 2025-12-18
- **Author**: ZhichenJiang
- **Categories**: Kernel Optimization

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
- Reduce synchronization overhead
- MoE optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- Decode/generation phase

### Changed Files

```
.../_torch/custom_ops/cute_dsl_custom_ops.py       |   33 +-
 ...aled_contiguous_grouped_gemm_finalize_fusion.py |  573 ++++++----
 ...aled_contiguous_grouped_gemm_finalize_fusion.py | 1091 ++++++++++++++++++++
 .../_torch/thop/parallel/test_cute_dsl_moe.py      |    2 +-
 4 files changed, 1505 insertions(+), 194 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py b/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
index 70aec7156..2405d3e5f 100644
--- a/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
+++ b/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
@@ -1151,6 +1151,11 @@ if IS_CUTLASS_DSL_AVAILABLE:
                     f"{self.__class__.kernel_class.__name__} supports SM 100 (B200) and SM 103 (B300) only, but got SM {sm_version}"
                 )
 
+            if self.tile_size not in [128, 256]:
+                raise ValueError(
+                    f"{self.__class__.kernel_class.__name__} supports tile size (mma tile M dimension) 128 and 256 only, but got {self.tile_size}"
+                )
+
         def unique_id(self):
             return (
                 self.num_experts,
@@ -1173,19 +1178,21 @@ if IS_CUTLASS_DSL_AVAILABLE:
             l, n = b.size(0), b.size(1)
 
             # TODO: Add full shmoo
-            mma_tiler_mn_candidates = [(128, 128), (128, 256)]
-            cluster_shape_mn_candidates = [(1, 1), (1, 2)]
+            mma_tiler_mn_candidates = [(self.tile_size, 128),
+                                       (self.tile_size, 256)]
+            cluster_shape_mn_candidates = [(self.tile_size // 128, 1),
+                                           (self.tile_size // 128, 2)]
+            raster_along_m_candidates = [True, False]
 
             valid_tactics = []
-            for mma_tiler_mn, cluster_shape_mn in itertools.product(
-                    mma_tiler_mn_candidates, cluster_shape_mn_candidates):
+            for mma_tiler_mn, cluster_shape_mn, raster_along_m in itertools.product(
+                    mma_tiler_mn_candidates, cluster_shape_mn_candidates,
+                    raster_along_m_candidates):
                 if self.__class__.kernel_class.can_implement(
                         ab_dtype=cutlass.Float4E2M1FN,
                         sf_dtype=cutlass.Float8E4M3FN,
                         sf_vec_size=self.scaling_vector_size,
-                        acc_dtype=cutlass.Float32,
                         out_dtype=cutlass.BFloat16,
-                        use_2cta_instrs=False,
                         mma_tiler_mn=mma_tiler_mn,
                         cluster_shape_mn=cluster_shape_mn,
                         m=m,
@@ -1194,10 +1201,10 @@ if IS_CUTLASS_DSL_AVAILABLE:
                         l=l,
                         a_major="k",
                         b_major="k",
-                        c_major="n",
-                        m_aligned=self.tile_size,
+                        out_major="n",
                 ):
-                    valid_tactics.append((mma_tiler_mn, cluster_shape_mn))
+                    valid_tactics.append(
+                        (mma_tiler_mn, cluster_shape_mn, raster_along_m))
 
             return valid_tactics
 
@@ -1311,19 +1318,19 @@ if IS_CUTLASS_DSL_AVAILABLE:
             stream = cuda.CUstream(torch_stream.cuda_stream)
 
             i
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 4f6d4da035 - [None][perf] Fix TPOT when `min_tokens` set (#9862)

- **Date**: 2025-12-11
- **Author**: jthomson04
- **Categories**: General Performance

### Optimization Techniques

- Async/stream-based execution

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/sampler.py | 7 ++++++-
 1 file changed, 6 insertions(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index 40d1450e4..83826eaad 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -2230,9 +2230,14 @@ class TorchSampler(Sampler, AsyncWorkerMixin):
                     for beam_idx in range(num_beams[index]):
                         for step in range(num_steps[index]):
                             if r.get_num_tokens(beam_idx) + step < r.py_min_length[0]:
+                                # NOTE(jthomson04): We can NOT just assign logits[...] = float("-inf").
+                                # This introduces a pageable HtoD transfer, which wreaks havoc on TPOT (up to ~20%)
+                                # Instead, we create a little tensor on device, then assign to that.
+                                # This way, we avoid the pageable transfer.
+                                neg_inf_tensor = torch.full((), float("-inf"), device=logits.device)
                                 logits[
                                     current_offset + num_steps[index] * beam_idx + step, r.py_end_id
-                                ] = float("-inf")
+                                ] = neg_inf_tensor
                             else:
                                 # early exit
                                 break

```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 52110e8ca7 - [#11529][perf] Replace Python-traced FP8 quantization with optimized CUDA op in AD MoE (#11626)

- **Date**: 2026-02-23
- **Author**: Eran Geva
- **Categories**: Quantization Optimization

### Optimization Techniques

- Torch compilation/JIT optimization
- Operator fusion
- FP8 quantization
- Integer quantization
- Triton kernel
- MoE optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../auto_deploy/custom_ops/fused_moe/trtllm_moe.py       | 16 ++++------------
 1 file changed, 4 insertions(+), 12 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/trtllm_moe.py b/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/trtllm_moe.py
index 47fa356c2..cc67beae4 100644
--- a/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/trtllm_moe.py
+++ b/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/trtllm_moe.py
@@ -291,16 +291,6 @@ def trtllm_moe_fused_fake(
     return torch.empty_like(x)
 
 
-# NOTE(suyogg): If compile ever fails because of this, just write a triton kernel
-# for this function and use it as a custom op.
-@torch.compile
-def _quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
-    """Quantize tensor to FP8 with clamping (matches torch_quant_fp8_linear)."""
-    FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
-    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
-    return (x / scale).clamp(FP8_MIN, FP8_MAX).to(torch.float8_e4m3fn)
-
-
 def _validate_mlp_style_and_act_fn(is_gated_mlp: bool, act_fn: int) -> None:
     assert (is_gated_mlp and act_fn in [ActivationType.Silu, ActivationType.Swiglu]) or (
         not is_gated_mlp and act_fn == ActivationType.Relu2
@@ -363,8 +353,10 @@ def trtllm_quant_fp8_moe_fused(
     x_shape = x.shape
     x2d = x.view(-1, x_shape[-1])
 
-    # Quantize the input using precomputed max scale
-    x_q_fp8 = _quantize_fp8(x2d, fc1_act_scale)
+    # Quantize the input using precomputed max scale.
+    # Use the optimized CUDA kernel (same as PT backend's static_quantize path) instead of
+    # Python-traced ops that compile to a slower Triton kernel.
+    x_q_fp8, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x2d, fc1_act_scale)
 
     # Prepare quant_scales for TensorRT-LLM (Cutlass) FP8 format:
     # [fc1_dequant_scale, fc2_act_scale_reciprocal, fc2_dequant_scale, gemm1_input_dequant_scale]

```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 5339d367ce - [perf] Reduce the workspace size of FP4 activation scales for MoE (#4303)

- **Date**: 2025-05-30
- **Author**: Jinyang Yuan
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- Parallelism optimization
- Batching optimization
- PyTorch built-in optimized ops
- Reduce synchronization overhead
- MoE optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU

### Changed Files

```
cpp/include/tensorrt_llm/deep_gemm/fp8_gemm.cuh    |  11 +-
 cpp/include/tensorrt_llm/deep_gemm/scheduler.cuh   |  19 +--
 .../mixtureOfExpertsBackendBenchmarkFixture.h      |  11 +-
 .../fp8_blockscale_gemm/fp8_blockscale_gemm.cu     |  22 +--
 .../fp8_blockscale_gemm_kernel.cuh                 | 154 ++++-----------------
 ...orrt_llm_internal_cutlass_kernels_static.tar.xz |   4 +-
 .../aarch64-linux-gnu/version.txt                  |   4 +-
 .../internal_cutlass_kernels/include/moe_kernels.h |  11 +-
 ...orrt_llm_internal_cutlass_kernels_static.tar.xz |   4 +-
 .../x86_64-linux-gnu/version.txt                   |   4 +-
 .../mixtureOfExperts/mixtureOfExpertsPlugin.cpp    |   8 +-
 cpp/tensorrt_llm/thop/moeOp.cpp                    |  17 ++-
 .../unit_tests/kernels/mixtureOfExpertsTest.cu     |  11 +-
 tensorrt_llm/_torch/custom_ops/torch_custom_ops.py |  20 ++-
 tensorrt_llm/_torch/modules/fused_moe.py           |   1 +
 tests/unittest/_torch/thop/deep_gemm_tests.py      |  28 ++--
 16 files changed, 124 insertions(+), 205 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/include/tensorrt_llm/deep_gemm/fp8_gemm.cuh b/cpp/include/tensorrt_llm/deep_gemm/fp8_gemm.cuh
index ceff2fd5b..63e185f7e 100644
--- a/cpp/include/tensorrt_llm/deep_gemm/fp8_gemm.cuh
+++ b/cpp/include/tensorrt_llm/deep_gemm/fp8_gemm.cuh
@@ -250,8 +250,8 @@ template <typename LayoutIndexType>
 void runGemm(cudaKernel_t kernel, void* mat_a, int ld_a, void* mat_b, int ld_b, void* mat_d, int ld_d, float* scales_a,
     float* scales_b, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, uint32_t block_m, uint32_t block_n,
     uint32_t block_k, uint32_t num_groups, uint32_t num_tma_multicast, GemmType gemm_type,
-    LayoutIndexType* problem_m_offsets, LayoutIndexType* problem_m_padded_offsets, cudaStream_t stream, int num_sms,
-    uint32_t smem_size, uint32_t max_shape_m_padded)
+    LayoutIndexType* problem_m_offsets, cudaStream_t stream, int num_sms, uint32_t smem_size,
+    uint32_t max_shape_m_padded)
 {
     auto tma_a_desc = make_2d_tma_a_desc(
         reinterpret_cast<__nv_fp8_e4m3*>(mat_a), shape_m, shape_k, block_m, block_k, num_groups, gemm_type);
@@ -281,7 +281,6 @@ void runGemm(cudaKernel_t kernel, void* mat_a, int ld_a, void* mat_b, int ld_b,
 
     GroupedWithOffsetSchedulerInput input;
     input.problem_m_offsets = problem_m_offsets;
-    input.problem_m_padded_offsets = problem_m_padded_offsets;
 
     // Launch
     auto status = cudaLaunchKernelEx(&config, kernel, reinterpret_cast<__nv_bfloat16*>(mat_d), scales_b, input,
@@ -293,9 +292,8 @@ template <typename LayoutIndexType>
 void runGemmSwapAB(cudaKernel_t kernel, void* mat_a /* weight*/, int ld_a, void* mat_b /* act*/, int ld_b, void* mat_d,
     int ld_d, float* scales_a /* weight scales*/, float* scales_b /* act scales*/, uint32_t shape_m, uint32_t shape_n,
     uint32_t shape_k, uint32_t block_m, uint32_t block_n, uint32_t block_k, uint32_t num_groups,
-    uint32_t num_tma_multicast, GemmType gemm_type, LayoutIndexType* problem_n_offsets,
-    LayoutIndexType* problem_n_padded_offsets, cudaStream_t stream, int num_sms, uint32_t smem_size,
-    uint32_t max_shape_n_padded)
+    uint32_t num_tma_multicast, GemmType gemm_type, LayoutIndexType* problem_n_offsets, cudaStream_t stream,
+    int num_sms, uint32_t smem_size, uint32_t max_shape_n_padded)
 {
     // Create tensor mappings using swapAB version functions, note the parameter order
     auto tma_a_desc = make_2d_tma_a_desc_swapAB(
@@ -327,7 +325,6 @@ void runGemmSwapAB(cudaKernel_t kernel, void* mat_a /* weight*/, int ld_a, void*
     // Update input structure to use N dimension offsets
     GroupedWithOffsetSchedulerInputSwapAB input;
     input.problem_n_offsets = problem_n_offsets; // Now offsets are for N dimension
-    input.problem_n_padded_4_offsets = problem_n_padded_offsets;
 
     auto status = cudaLaunchKernelEx(&config, kernel, reinterpret_cast<__nv_bfloat16*>(mat_d), scales_a, input,
         tma_a_desc, tma_b_desc, tma_scales_b_desc, tma_d_desc);
diff --git a/cpp/include/tensorrt_llm/dee
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 53491ffdb1 - [#9023][feat] reduce AD graph optimization time for non-participating passes (#9024)

- **Date**: 2025-11-12
- **Author**: Neta Zmora
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
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
.../_torch/auto_deploy/transform/interface.py        |  4 ++--
 .../auto_deploy/transform/library/attention.py       | 20 ++++++++++----------
 .../transform/library/cleanup_input_constraints.py   |  7 ++++++-
 .../transform/library/cleanup_noop_add.py            |  7 ++++++-
 .../transform/library/cleanup_noop_slice.py          |  7 ++++++-
 .../auto_deploy/transform/library/collectives.py     |  5 ++++-
 .../library/eliminate_redundant_transposes.py        |  4 ++--
 .../auto_deploy/transform/library/fuse_quant.py      |  8 ++++----
 .../auto_deploy/transform/library/fused_moe.py       | 14 ++++++++++----
 .../_torch/auto_deploy/transform/library/fusion.py   | 10 ++++++++--
 .../auto_deploy/transform/library/mxfp4_moe.py       |  8 ++++----
 .../auto_deploy/transform/library/quantization.py    |  7 ++++---
 .../_torch/auto_deploy/transform/library/rms_norm.py |  8 +++++---
 .../_torch/auto_deploy/transform/library/rope.py     | 16 +++++++++++-----
 .../_torch/auto_deploy/transform/library/sharding.py |  5 ++++-
 15 files changed, 86 insertions(+), 44 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/transform/interface.py b/tensorrt_llm/_torch/auto_deploy/transform/interface.py
index 2a51b36d4..cbe3c5661 100644
--- a/tensorrt_llm/_torch/auto_deploy/transform/interface.py
+++ b/tensorrt_llm/_torch/auto_deploy/transform/interface.py
@@ -410,14 +410,14 @@ class BaseTransform(ABC):
             return self._apply_to_full_model(mod, cm, factory, shared_config)
 
         # just run it on first graph module we are encountering for now...
-        info = TransformInfo()
+        info = None
         for k, graph_sub in named_graphmodules(mod):
             graph_sub, info_apply = self._apply(graph_sub, cm, factory, shared_config)
             if k == "":
                 mod = graph_sub
             else:
                 mod.set_submodule(k, graph_sub)
-            info = info & info_apply
+            info = info & info_apply if info is not None else info_apply
         return mod, info
 
     @final
diff --git a/tensorrt_llm/_torch/auto_deploy/transform/library/attention.py b/tensorrt_llm/_torch/auto_deploy/transform/library/attention.py
index 1e69bfaf5..91729d497 100644
--- a/tensorrt_llm/_torch/auto_deploy/transform/library/attention.py
+++ b/tensorrt_llm/_torch/auto_deploy/transform/library/attention.py
@@ -304,8 +304,8 @@ class MatchRepeatKV(BaseTransform):
         info = TransformInfo(
             skipped=False,
             num_matches=num_kv_patterns,
-            is_clean=False,
-            has_valid_shapes=False,
+            is_clean=num_kv_patterns == 0,
+            has_valid_shapes=num_kv_patterns == 0,
         )
 
         return gm, info
@@ -333,8 +333,8 @@ class MatchEagerAttention(BaseTransform):
         info = TransformInfo(
             skipped=False,
             num_matches=num_eager_patterns,
-            is_clean=False,
-            has_valid_shapes=False,
+            is_clean=num_eager_patterns == 0,
+            has_valid_shapes=num_eager_patterns == 0,
         )
 
         return gm, info
@@ -647,8 +647,8 @@ class MatchSDPAToTorchAttention(BaseTransform):
         info = TransformInfo(
             skipped=False,
             num_matches=num_patterns,
-            is_clean=False,
-            has_valid_shapes=False,
+            is_clean=num_patterns == 0,
+            has_valid_shapes=num_patterns == 0,
         )
         return gm, info
 
@@ -685,8 +685,8 @@ class MatchRepeatKVWithTorchAttention(BaseTransform):
         info = TransformInfo(
             skipped=False,
             num_matches=num_patterns,
-            is_clean=False,
-            has_valid_shapes=False,
+            is_clean=num_patterns == 0,
+            has_valid_shapes=num_patterns == 0,
         )
         return gm, info
 
@@ -870,7 +870,7 @@ class MatchAttentionLayout(BaseTransform):
         info = TransformInfo(
             skipped=False,
             num_matches=num_matches,
-            is_clean=False,
-            has_valid_shapes=False,
+            is_clean=num_matches == 0,
+      
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 53adb3cb4e - test: waive flaky test_kv_cache_event_async_api (#3062)

- **Date**: 2025-03-25
- **Author**: Yuan Tong
- **Categories**: Parallelism/Async, Cache Optimization

### Optimization Techniques

- Async/stream-based execution
- KV cache optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tests/unittest/llmapi/test_llm_kv_cache_events.py | 2 ++
 1 file changed, 2 insertions(+)
```

### Diff Preview

```diff
diff --git a/tests/unittest/llmapi/test_llm_kv_cache_events.py b/tests/unittest/llmapi/test_llm_kv_cache_events.py
index 81d5728fd..85a7feac7 100644
--- a/tests/unittest/llmapi/test_llm_kv_cache_events.py
+++ b/tests/unittest/llmapi/test_llm_kv_cache_events.py
@@ -1,6 +1,7 @@
 import asyncio
 import time
 
+import pytest
 from test_llm import get_model_path
 
 import tensorrt_llm
@@ -119,6 +120,7 @@ def test_expected_kv_cache_events():
                 assert event[0]["data"]["type"] == "stored"
 
 
+@pytest.mark.skip("https://nvbugs/5150466: flaky fail")
 def test_kv_cache_event_async_api():
     llm = create_llm()
     sampling_params = SamplingParams(max_tokens=6, temperature=0.01)

```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 55fed1873c - [None][chore] AutoDeploy: cleanup old inference optimizer configs (#8039)

- **Date**: 2025-10-17
- **Author**: h-guo18
- **Categories**: General Performance

### Optimization Techniques

- Torch compilation/JIT optimization
- Custom CUDA kernel
- Operator fusion
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Triton kernel
- PyTorch built-in optimized ops
- Reduce synchronization overhead
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
.../advanced/benchmarking_with_trtllm_bench.md     |  32 ++--
 .../auto_deploy/advanced/expert_configurations.md  |  44 ++---
 .../features/auto_deploy/advanced/workflow.md      |   2 -
 docs/source/features/auto_deploy/support_matrix.md |   1 +
 .../advanced/benchmarking_with_trtllm_bench.md     |  32 ++--
 .../auto_deploy/advanced/expert_configurations.md  |  44 ++---
 .../advanced/serving_with_trtllm_serve.md          |  42 +++--
 docs/source/torch/auto_deploy/advanced/workflow.md |  17 +-
 examples/auto_deploy/.vscode/launch.json           |   4 +-
 examples/auto_deploy/README.md                     |  28 +--
 examples/auto_deploy/build_and_run_ad.py           |   2 +-
 .../_torch/auto_deploy/config/default.yaml         |  19 +-
 .../_torch/auto_deploy/config/transformers.yaml    |   3 +-
 tensorrt_llm/_torch/auto_deploy/export/export.py   |   7 +-
 tensorrt_llm/_torch/auto_deploy/llm_args.py        | 127 ++++++++------
 .../_torch/auto_deploy/shim/ad_executor.py         |   4 +-
 tensorrt_llm/_torch/auto_deploy/shim/interface.py  |   5 +
 .../_torch/auto_deploy/transform/interface.py      |   4 +-
 .../auto_deploy/transform/library/attention.py     |  16 +-
 .../auto_deploy/transform/library/build_model.py   |   2 +-
 .../auto_deploy/transform/library/compile_model.py |   6 +-
 .../auto_deploy/transform/library/kvcache.py       |  13 +-
 .../auto_deploy/transform/library/load_weights.py  |  16 +-
 .../library/visualization.py                       |  56 ++++--
 .../_torch/auto_deploy/transform/optimizer.py      |   4 +
 .../_torch/auto_deploy/transformations/__init__.py |   1 -
 .../transformations/library/__init__.py            |   6 -
 .../auto_deploy/transformations/transform.py       | 117 -------------
 tensorrt_llm/_torch/auto_deploy/utils/_config.py   |  19 +-
 .../{transformations => utils}/_graph.py           |   4 +-
 tensorrt_llm/bench/dataclasses/configuration.py    |   1 -
 .../defs/accuracy/test_llm_api_autodeploy.py       |  26 ++-
 tests/integration/defs/perf/test_perf.py           |  10 +-
 .../auto_deploy/_utils_test/_model_test_utils.py   |  14 --
 .../unit/multigpu/test_ad_build_small_multi.py     |  32 ++--
 .../unit/singlegpu/models/test_hybrid_patches.py   |  16 +-
 .../unit/singlegpu/models/test_llama4_vlm_patch.py |   2 +-
 .../unit/singlegpu/models/test_mistral3_patches.py |   2 +-
 .../auto_deploy/unit/singlegpu/shim/test_engine.py |   6 +-
 .../unit/singlegpu/shim/test_llm_config.py         |  78 ++++++---
 .../unit/singlegpu/test_ad_build_small_single.py   | 194 +++++++++++++++------
 .../unit/singlegpu/test_ad_trtllm_bench.py         |   8 +-
 .../library/test_attention_matcher.py              |  29 +--
 .../library/test_attention_matcher_hf.py           |  21 +--
 .../transformations/library/test_kv_cache.py       |   2 +-
 .../unit/singlegpu/utils/test_config.py            |  17 --
 46 files changed, 559 insertions(+), 576 deletions(-)
```

### Diff Preview

```diff
diff --git a/docs/source/features/auto_deploy/advanced/benchmarking_with_trtllm_bench.md b/docs/source/features/auto_deploy/advanced/benchmarking_with_trtllm_bench.md
index 1bbf2d1bd..d5e0cde8f 100644
--- a/docs/source/features/auto_deploy/advanced/benchmarking_with_trtllm_bench.md
+++ b/docs/source/features/auto_deploy/advanced/benchmarking_with_trtllm_bench.md
@@ -40,29 +40,31 @@ trtllm-bench \
 #### Basic Performance Configuration (`autodeploy_config.yaml`)
 
 ```yaml
-# Compilation backend
-compile_backend: torch-opt
-
-# Runtime engine
+# runtime engine
 runtime: trtllm
 
-# Model loading
+# model loading
 skip_loading_weights: false
 
-# Fraction of free memory to use for kv-caches
-free_mem_ratio: 0.8
-
-# CUDA Graph optimization
-cuda_graph_batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256]
-
-# Attention backend
-attn_backend: flashinfer
-
 # Sequence configuration
 max_batch_size: 256
+
+# transform options
+transforms:
+  insert_cached_attention:
+    # attention backend
+    backend: flashinfer
+  resize_kv_cache:
+    # fraction of free memory to use for kv-caches
+    free_mem_ratio: 0.8
+  compile_model:
+    # compilation backend
+    backend: torch-opt
+    # CUDA Graph optimization
+    cuda_graph_batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256]
 ```
 
-Enable multi-GPU execution by specifying `--tp n`, where `n` is the number of GPUs
+Enable multi-GPU execution by specifying `--tp n`, where `n` is the number of GPUs.
 
 ## Configuration Options Reference
 
diff --git a/docs/source/features/auto_deploy/advanced/expert_configurations.md b/docs/source/features/auto_deploy/advanced/expert_configurations.md
index 60cfd197a..471bd2139 100644
--- a/docs/source/features/auto_deploy/advanced/expert_configurations.md
+++ b/docs/source/features/auto_deploy/advanced/expert_configurations.md
@@ -63,15 +63,15 @@ args:
     num_hidden_layers: 12
     hidden_size: 1024
   world_size: 4
-  compile_backend: torch-compile
-  attn_backend: triton
   max_seq_len: 2048
   max_batch_size: 16
   transforms:
-    sharding:
-      strategy: auto
-    quantization:
-      enabled: false
+    detect_sharding:
+      support_partial_config: true
+    insert_cached_attention:
+      backend: triton
+    compile_model:
+      backend: torch-compile
 
 prompt:
   batch_size: 8
@@ -79,13 +79,6 @@ prompt:
     max_tokens: 150
     temperature: 0.8
     top_k: 50
-
-benchmark:
-  enabled: true
-  num: 20
-  bs: 4
-  isl: 1024
-  osl: 256
 ```
 
 Create an additional override file (e.g., `production.yaml`):
@@ -94,11 +87,10 @@ Create an additional override file (e.g., `production.yaml`):
 # production.yaml
 args:
   world_size: 8
-  compile_backend: torch-opt
   max_batch_size: 32
-
-benchmark:
-  enabled: false
+  transforms:
+    compile_model:
+      backend: torch-opt
 ```
 
 Then use these configurations:
@@ -107,18 +99,18 @@ Then use these configurations:
 # Using single YAML config
 python build_and_run_ad.py \
   --model "meta-llama/Meta-Llama-3.1-8B-Instruct
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 572551b586 - [None][perf] Autotune TRT-LLM Gen MoE when using CUDA graphs (#7285)

- **Date**: 2025-09-03
- **Author**: Jinyang Yuan
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Batching optimization
- Reduce synchronization overhead
- Speculative decoding

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/model_engine.py | 6 ++++++
 1 file changed, 6 insertions(+)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/model_engine.py b/tensorrt_llm/_torch/pyexecutor/model_engine.py
index 8e3656928..c9ef51853 100644
--- a/tensorrt_llm/_torch/pyexecutor/model_engine.py
+++ b/tensorrt_llm/_torch/pyexecutor/model_engine.py
@@ -825,6 +825,12 @@ class PyTorchModelEngine(ModelEngine):
                         f"Run generation only CUDA graph warmup for batch size={bs}, draft_len={draft_len}"
                     )
                     self.enable_spec_decode = draft_len > 0 or self.is_draft_model
+                    if self.pytorch_backend_config.enable_autotuner:
+                        with self.no_cuda_graph(), autotune():
+                            self.forward(batch,
+                                         new_tensors_device=None,
+                                         resource_manager=resource_manager)
+                        torch.cuda.synchronize()
                     self.forward(batch,
                                  new_tensors_device=None,
                                  resource_manager=resource_manager)

```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 5a99c9734d - [TRTLLM-8777][feat] Update DeepGEMM to the latest commit to include optimizations for DeepSeek-v3.2 (#9380)

- **Date**: 2025-11-25
- **Author**: Fanrong Li
- **Categories**: General Performance

### Optimization Techniques

- FP8 quantization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU

### Changed Files

```
3rdparty/CMakeLists.txt                                    |  2 +-
 tests/unittest/_torch/attention/sparse/test_dsa_indexer.py | 10 +++++-----
 2 files changed, 6 insertions(+), 6 deletions(-)
```

### Diff Preview

```diff
diff --git a/3rdparty/CMakeLists.txt b/3rdparty/CMakeLists.txt
index bf044537d..5bd3a6ee9 100644
--- a/3rdparty/CMakeLists.txt
+++ b/3rdparty/CMakeLists.txt
@@ -39,7 +39,7 @@ FetchContent_Declare(
 FetchContent_Declare(
   deepgemm
   GIT_REPOSITORY https://github.com/ruoqianguo/DeepGEMM
-  GIT_TAG 9fa5965e265e27995f539e0dd73a06351a8a9eaf
+  GIT_TAG 6cb8161516302550785d9af924d2778afef1f3f6 # swapab_sm100 branch
   GIT_SUBMODULES_RECURSE
   ON
   SOURCE_SUBDIR
diff --git a/tests/unittest/_torch/attention/sparse/test_dsa_indexer.py b/tests/unittest/_torch/attention/sparse/test_dsa_indexer.py
index bf74c3108..c569d37a3 100644
--- a/tests/unittest/_torch/attention/sparse/test_dsa_indexer.py
+++ b/tests/unittest/_torch/attention/sparse/test_dsa_indexer.py
@@ -308,9 +308,9 @@ def test_deepgemm_fp8_mqa_logits_basic():
     """
     torch.manual_seed(0)
 
-    num_heads, head_dim = 32, 128
-    seq_len = 512
-    seq_len_kv = 1024
+    num_heads, head_dim = 64, 128
+    seq_len = 2048
+    seq_len_kv = 4096
     #[seq_len, num_heads, head_dim]
     q = torch.randn(
         seq_len,
@@ -335,8 +335,8 @@ def test_deepgemm_fp8_mqa_logits_basic():
     )
     # ks[i] -> ke[i] for each q[i]
     ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
-    ke = torch.arange(seq_len, dtype=torch.int, device="cuda") + (
-        seq_len_kv - seq_len) + 1  # +1 for exclusive end
+    ke = torch.arange(seq_len, dtype=torch.int,
+                      device="cuda") + (seq_len_kv - seq_len)
 
     # Convert to FP8
     q_fp8 = q.to(torch.float8_e4m3fn)

```

### Analysis Summary

Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 5ddeaf9990 - [None][perf] Vectorize quantize_fp8_blockwise with CUDA kernel (#11724)

- **Date**: 2026-02-27
- **Author**: Kanghwan
- **Categories**: Kernel Optimization, Quantization Optimization

### Optimization Techniques

- Operator fusion
- FP8 quantization
- Integer quantization
- PyTorch built-in optimized ops

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/visual_gen/quantization/ops.py | 59 +++++++++++-----------
 1 file changed, 29 insertions(+), 30 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/visual_gen/quantization/ops.py b/tensorrt_llm/_torch/visual_gen/quantization/ops.py
index 57e19bcbd..ce376f01d 100644
--- a/tensorrt_llm/_torch/visual_gen/quantization/ops.py
+++ b/tensorrt_llm/_torch/visual_gen/quantization/ops.py
@@ -1,3 +1,6 @@
+# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+
 """
 Quantization operations for diffusion models.
 
@@ -59,43 +62,39 @@ def quantize_fp8_blockwise(
         - This uses 128x128 block scaling compatible with Linear module's FP8_BLOCK_SCALES
     """
     out_features, in_features = weight.shape
-    weight_fp32 = weight.float()
-
-    # Calculate number of blocks
     num_blocks_out = (out_features + block_size - 1) // block_size
     num_blocks_in = (in_features + block_size - 1) // block_size
 
-    # Initialize outputs
-    qweight = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
-    block_scales = torch.empty(
-        (num_blocks_out, num_blocks_in), dtype=torch.float32, device=weight.device
+    # Pad to multiple of block_size
+    pad_out = num_blocks_out * block_size - out_features
+    pad_in = num_blocks_in * block_size - in_features
+    if pad_out > 0 or pad_in > 0:
+        weight_padded = torch.nn.functional.pad(weight, (0, pad_in, 0, pad_out))
+    else:
+        weight_padded = weight
+
+    # Reshape so each block becomes a row:
+    # (out, in) -> (nb_out, bs, nb_in, bs) -> (nb_out, nb_in, bs, bs) -> (nb_out*nb_in, bs*bs)
+    rows_per_block = (
+        weight_padded.reshape(num_blocks_out, block_size, num_blocks_in, block_size)
+        .permute(0, 2, 1, 3)
+        .reshape(num_blocks_out * num_blocks_in, block_size * block_size)
     )
 
-    # Quantize each block
-    for i in range(num_blocks_out):
-        row_start = i * block_size
-        row_end = min((i + 1) * block_size, out_features)
-
-        for j in range(num_blocks_in):
-            col_start = j * block_size
-            col_end = min((j + 1) * block_size, in_features)
+    # Single CUDA kernel: per-row FP8 quantization
+    # quantize_e4m3_activation uses PER_TOKEN mode: one scale per row
+    qrows, scales = torch.ops.tensorrt_llm.quantize_e4m3_activation(rows_per_block.contiguous())
 
-            # Extract block
-            block = weight_fp32[row_start:row_end, col_start:col_end]
-
-            # Compute block scale
-            max_val = block.abs().max()
-            scale = (
-                max_val / FP8_E4M3_MAX if max_val > 0 else torch.tensor(1.0, device=weight.device)
-            )
-
-            # Quantize block
-            inv_scale = scale.reciprocal() if scale > 0 else torch.tensor(1.0, device=weight.device)
-            qblock = (block * inv_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
+    # Reshape back: (nb_out*nb_in, bs*bs) -> (nb_out, nb_in, bs, bs) -> (out_padded, in_padded)
+    qweight_padded = (
+        qrows.reshape(num_blocks_
```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 5e272eef81 - feat : reduce trt engine build time in testing (#3014)

- **Date**: 2025-03-26
- **Author**: peaceh-nv
- **Categories**: General Performance

### Optimization Techniques

- Integer quantization
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tests/unittest/attention/test_gpt_attention.py | 4 +++-
 1 file changed, 3 insertions(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tests/unittest/attention/test_gpt_attention.py b/tests/unittest/attention/test_gpt_attention.py
index d3e9a823a..579d14cbe 100644
--- a/tests/unittest/attention/test_gpt_attention.py
+++ b/tests/unittest/attention/test_gpt_attention.py
@@ -754,7 +754,9 @@ class TestFunctional(unittest.TestCase):
                 precision=dtype,
                 int8=int8_trt_flag,
                 quant_mode=quant_mode)
-
+            # Reuce the TRT engine build time by setting the max allowed number of tactics in builder tactic profiling.
+            if builder_config.trt_builder_config.max_num_tactics == -1:
+                builder_config.trt_builder_config.max_num_tactics = 30
             if session is None:
                 engine = builder.build_engine(net, builder_config)
                 session = tensorrt_llm.runtime.Session.from_serialized_engine(

```

### Analysis Summary

Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 5ef65872a3 - [None][fix] type annotations in fuse_input_embeds (#8976)

- **Date**: 2025-11-07
- **Author**: mpikulski
- **Categories**: Fusion

### Optimization Techniques

- Operator fusion
- PyTorch built-in optimized ops

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/models/modeling_multimodal_utils.py | 6 +++---
 1 file changed, 3 insertions(+), 3 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_multimodal_utils.py b/tensorrt_llm/_torch/models/modeling_multimodal_utils.py
index acd68706f..2aa33405a 100644
--- a/tensorrt_llm/_torch/models/modeling_multimodal_utils.py
+++ b/tensorrt_llm/_torch/models/modeling_multimodal_utils.py
@@ -17,7 +17,7 @@
 # and s2wrapper: https://github.com/bfshi/scaling_on_scales
 
 import math
-from typing import Any, Dict, List, Optional, Tuple
+from typing import Any, Dict, List, Optional, Tuple, cast
 
 import torch
 import torch.nn.functional as F
@@ -291,7 +291,7 @@ def fuse_input_embeds(
     text_token_indices: Optional[torch.IntTensor] = None,
     mm_token_indices: Optional[torch.IntTensor] = None,
     **kwargs,
-) -> Tuple[Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
+) -> Tuple[Optional[torch.IntTensor], Optional[torch.FloatTensor]]:
     """
     Fuse text and multimodal embeddings. input_ids is [text_total_length + mm_total_length] and mm_embed is [mm_total_length, hidden_dim]. We just need to fuse them into [text_total_length + mm_total_length, hidden_dim] by slice-and-assign to the corresponding entries.
 
@@ -337,7 +337,7 @@ def fuse_input_embeds(
     input_embeds[mm_token_indices, :] = mm_embed.to(dtype=input_embeds.dtype,
                                                     device=input_embeds.device)
 
-    return None, input_embeds
+    return None, cast(torch.FloatTensor, input_embeds)
 
 
 #region VILA utils

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 5f737b8dbe - [None][perf] Use fp8 quant kernel in DS3.2 indexer module (#8701)

- **Date**: 2025-10-28
- **Author**: Chang Liu
- **Categories**: Kernel Optimization, Quantization Optimization

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
.../fp8_blockscale_gemm/fp8_blockscale_gemm.cu     |  6 +-
 .../fp8_blockscale_gemm/fp8_blockscale_gemm.h      | 12 +++-
 .../fp8_blockscale_gemm_kernel.cuh                 | 42 +++++++++---
 cpp/tensorrt_llm/thop/fp8Quantize.cpp              |  6 +-
 .../_torch/attention_backend/sparse/dsa.py         | 35 +++++-----
 tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py   |  2 +-
 tensorrt_llm/quantization/utils/fp8_utils.py       | 13 ++++
 .../_torch/attention/sparse/test_dsa_indexer.py    | 48 ++++----------
 .../_torch/thop/parallel/test_fp8_quantize.py      | 77 ++++++++++++++++++++++
 9 files changed, 173 insertions(+), 68 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu b/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu
index c6a22c0f7..d234ef8b7 100644
--- a/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu
+++ b/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu
@@ -185,10 +185,10 @@ void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::strideBatchGe
 }
 
 template <typename ElementA, typename ElementB, typename ElementD>
-void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::fp8CS1x128(
-    __nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x, int shape_y, cudaStream_t stream)
+void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::fp8CS1x128(__nv_fp8_e4m3* mat_quant, float* scales,
+    __nv_bfloat16 const* mat, int shape_x, int shape_y, cudaStream_t stream, bool use_ue8m0)
 {
-    fp8_1x128_cs(mat_quant, scales, mat, shape_x, shape_y, stream);
+    fp8_1x128_cs(mat_quant, scales, mat, shape_x, shape_y, stream, use_ue8m0);
 }
 
 template <typename ElementA, typename ElementB, typename ElementD>
diff --git a/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h b/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h
index 2ca0f3fbc..29a954ac1 100644
--- a/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h
+++ b/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h
@@ -55,9 +55,13 @@ public:
         int shape_k, cudaStream_t stream, float* scales_a, int stride_scales_a, float* scales_b)
         = 0;
 
+    // Backward compatibility to support old signature
     virtual void fp8CS1x128(__nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x, int shape_y,
         cudaStream_t stream)
         = 0;
+    virtual void fp8CS1x128(__nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x, int shape_y,
+        cudaStream_t stream, bool use_ue8m0)
+        = 0;
     virtual void fp8CS1x128Reshape(__nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x,
         int shape_h, int shape_y, int stride_x, cudaStream_t stream)
         = 0;
@@ -113,7 +117,13 @@ public:
         cudaStream_t stream, float* scales_a, int stride_scales_a, float* scales_b) override;
 
     void fp8CS1x128(__nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x, int shape_y,
-        cudaStream_t stream) override;
+        cudaStream_t stream) override
+    {
+        fp8CS1x128(mat_quant, scales, mat, shape_x, shape_y, stream, false);
+    }
+
+    void fp8CS1x128(__nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x, int shape_y,
+        cudaStream_t stream, bool use_ue8m0) override;
     void fp8CS1x128Reshape(__nv_fp8_e4m3* mat_quant, float* scales,
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## 607bf4c395 - Doc: Add llama4 Maverick eagle3 and max-throughput and low_latency benchmark guide (#5810)

- **Date**: 2025-07-08
- **Author**: jiahanc
- **Categories**: Throughput/Latency

### Optimization Techniques

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
- NVIDIA Hopper (SM90) GPU
- Large batch / high concurrency

### Changed Files

```
.../tech_blog/blog6_Llama4_maverick_eagle_guide.md | 148 +++++++++++++++++++++
 examples/models/core/llama4/README.md              | 143 ++++++++++++++++++++
 2 files changed, 291 insertions(+)
```

### Diff Preview

```diff
diff --git a/docs/source/blogs/tech_blog/blog6_Llama4_maverick_eagle_guide.md b/docs/source/blogs/tech_blog/blog6_Llama4_maverick_eagle_guide.md
new file mode 100644
index 000000000..ac3131fe7
--- /dev/null
+++ b/docs/source/blogs/tech_blog/blog6_Llama4_maverick_eagle_guide.md
@@ -0,0 +1,148 @@
+# How to launch Llama4 Maverick + Eagle3 TensorRT-LLM server
+
+Artificial Analysis has benchmarked the Llama4 Maverick with Eagle3 enabled TensorRT-LLM server running at over [1000 tokens per second per user on 8xB200 GPUs](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/). This implementation leverages NVIDIA's TensorRT-LLM combined with speculative decoding using the Eagle3 model to further boost performance.
+
+In the guide below, we will walk you through how to launch your own high-performance Llama4 Maverick with Eagle3 enabled TensorRT-LLM server, from build to deployment.  (Note that your specific performance numbers may vary—speculative decoding speedups depend upon the dataset!)
+
+## Prerequisites
+
+- 8x NVIDIA B200 GPUs in a single node (we have a forthcoming guide for getting great performance on H100)
+- CUDA Toolkit 12.8 or later
+- Docker with NVIDIA Container Toolkit installed
+- Fast SSD storage for model weights
+- Access to Llama4 Maverick and Eagle3 model checkpoints
+- A love of speed
+
+## Download Artifacts
+
+* [NVIDIA Llama 4 Maverick 17B 128E Instruct FP8](https://huggingface.co/nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8)
+* [NVIDIA Llama 4 Maverick 17B 128E Eagle3 BF16](https://huggingface.co/nvidia/Llama-4-Maverick-17B-128E-Eagle3)
+
+In [Step 4: Start the TensorRT-LLM server](#step-4-start-the-tensorrt-llm-server), `/path/to/maverick` and `/path/to/eagle` refer to the download paths of the above respective models.
+
+## Launching the server
+
+### Step 1: Clone the repository
+
+```
+git clone https://github.com/NVIDIA/TensorRT-LLM.git
+cd TensorRT-LLM
+git submodule update --init --recursive
+git lfs pull
+```
+
+The last command, `git lfs pull`, ensures all large files stored with Git LFS are properly downloaded. If `git lfs` is not installed, please install following [Install Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)
+
+### Step 2: Prepare the TensorRT-LLM release Docker image
+
+
+#### Option 1. Use weekly release NGC docker image
+TensorRT-LLM provides weekly release [docker image](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release)
+
+#### Option 2. Build TensorRT-LLM Docker image (Alternative way)
+If you want to compile a specific TensorRT-LLM commit, you can build the docker image by checking out the specific branch or commit and running a make command. This may take 15-30 minutes depending on your system.
+
+```
+make -C docker release_build
+```
+
+### Step 3: (Optional) Tag and push the Docker image to your registry
+
+If you want to use this i
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 6151a4c9d6 - [None][feat] Add simple optimizations for MTP 2-model (#9176)

- **Date**: 2025-11-17
- **Author**: Mike Iovine
- **Categories**: General Performance

### Optimization Techniques

- Torch compilation/JIT optimization
- Parallelism optimization
- Speculative decoding
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/speculative/drafting_loops.py | 1 +
 tensorrt_llm/_torch/speculative/interface.py      | 4 ----
 2 files changed, 1 insertion(+), 4 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/speculative/drafting_loops.py b/tensorrt_llm/_torch/speculative/drafting_loops.py
index 886f0111e..25bcac643 100644
--- a/tensorrt_llm/_torch/speculative/drafting_loops.py
+++ b/tensorrt_llm/_torch/speculative/drafting_loops.py
@@ -59,6 +59,7 @@ def save_metadata_state(attn_metadata: AttentionMetadata,
             spec_metadata.eagle3_resource_manager.is_first_draft = True
 
 
+@torch.compile(options={'max-autotune': True})
 def prepare_for_generation(attn_metadata: AttentionMetadata,
                            spec_metadata: SpecMetadata,
                            position_ids: torch.Tensor) -> torch.Tensor:
diff --git a/tensorrt_llm/_torch/speculative/interface.py b/tensorrt_llm/_torch/speculative/interface.py
index 41be42a54..0102ba916 100644
--- a/tensorrt_llm/_torch/speculative/interface.py
+++ b/tensorrt_llm/_torch/speculative/interface.py
@@ -67,10 +67,6 @@ class SpeculativeDecodingMode(IntEnum):
         ) or self.is_ngram()
 
     def support_overlap_scheduler(self):
-        # TODO: fix accuracy issue
-        if self.is_mtp_eagle():
-            return False
-
         return self.is_mtp_one_model() or self.is_eagle3_one_model(
         ) or self.has_draft_model()
 

```

### Analysis Summary

Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 6157f30b06 - [#11318][infra] AutoDeploy: Add fused rope kernel - triton_rope_on_interleaved_qk_inputs (#11327)

- **Date**: 2026-02-17
- **Author**: Bala Marimuthu
- **Categories**: Kernel Optimization, Fusion

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Triton kernel
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
.../_torch/auto_deploy/custom_ops/README.md        | 129 +++++++++++++-
 .../auto_deploy/custom_ops/rope/triton_rope.py     | 109 +++++++++++-
 .../custom_ops/rope/triton_rope_kernel.py          | 185 +++++++++++++++++++++
 .../_torch/auto_deploy/transform/library/rope.py   | 120 ++++++++++++-
 .../singlegpu/custom_ops/rope/test_triton_rope.py  | 117 ++++++++++++-
 .../library/test_rope_transformation.py            |  61 +++++++
 6 files changed, 717 insertions(+), 4 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/custom_ops/README.md b/tensorrt_llm/_torch/auto_deploy/custom_ops/README.md
index addc6cc22..5263bc79d 100644
--- a/tensorrt_llm/_torch/auto_deploy/custom_ops/README.md
+++ b/tensorrt_llm/_torch/auto_deploy/custom_ops/README.md
@@ -4,4 +4,131 @@ All AutoDeploy custom operators follow the following naming convention:
 
 `torch.ops.auto_deploy.<kernel_backend>_<op_category>_<op_name>`
 
-The table below lists the operators ordered by their backend.
+The table below lists the operators grouped by category.
+
+### Available Custom Operators
+
+#### Attention
+
+| Operator Name | Description |
+|--------------|-------------|
+| `torch.ops.auto_deploy.torch_attention` | Grouped SDPA implementation with `bsnd` and `bnsd` layout supported |
+| `torch.ops.auto_deploy.torch_attention_sdpa` | Standard scaled dot-product attention (SDPA) implementation |
+| `torch.ops.auto_deploy.torch_attention_repeat_kv` | KV repetition for grouped-query attention |
+| `torch.ops.auto_deploy.torch_cached_attention_with_cache` | PyTorch backend attention with KV cache management |
+| `torch.ops.auto_deploy.flashinfer_attention_mha_with_cache` | FlashInfer multi-head attention with KV cache support |
+| `torch.ops.auto_deploy.flashinfer_attention_prepare_metadata` | FlashInfer attention metadata preparation |
+| `torch.ops.auto_deploy.triton_attention_flattened_mha_with_cache` | Triton flattened MHA with cache |
+| `torch.ops.auto_deploy.torch_onnx_attention_plugin` | Fused attention with RoPE placeholder for ONNX export |
+| `torch.ops.auto_deploy.torch_onnx_gather_nd` | N-dimensional gather operation for ONNX export |
+
+#### MLA (Multi-head Latent Attention)
+
+| Operator Name | Description |
+|--------------|-------------|
+| `torch.ops.auto_deploy.torch_mla` | Multi-head Latent Attention (MLA) implementation |
+| `torch.ops.auto_deploy.torch_cached_mla_with_cache` | PyTorch backend cached MLA with KV cache |
+| `torch.ops.auto_deploy.flashinfer_mla_with_cache` | FlashInfer MLA with cache |
+| `torch.ops.auto_deploy.flashinfer_mla_prepare_metadata` | FlashInfer MLA metadata preparation |
+
+#### RoPE (Rotary Position Embedding)
+
+| Operator Name | Description |
+|--------------|-------------|
+| `torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin` | RoPE with explicit cosine/sine |
+| `torch.ops.auto_deploy.torch_rope_with_complex_freqs` | RoPE with complex frequencies |
+| `torch.ops.auto_deploy.torch_rope_with_qk_interleaving` | RoPE with QK interleaving |
+| `torch.ops.auto_deploy.triton_rope_with_input_pos` | Triton RoPE with input positions |
+| `torch.ops.auto_deploy.triton_rope_on_flattened_inputs` | Triton RoPE on flattened inputs |
+| `torch.ops.auto_deploy.triton_rope_on_interleaved_qk_inputs` | Triton fused RoPE on interleaved QK inputs (position lookup + de-interleave + RoPE) |
+| `torch.ops.auto_deploy.flashinfer_rope` | FlashInfer RoPE implementation |
+
+#### Linear
+
+| Operator Name | Description |
+|----
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 61c5a53642 - [#5403][perf] Conditionally enable SWAP AB for speculative decoding (#5404)

- **Date**: 2025-07-01
- **Author**: 杨凯旋
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- FP8 quantization
- Integer quantization
- KV cache optimization
- Batching optimization
- Speculative decoding
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU

### Changed Files

```
cpp/kernels/xqa/defines.h     |   6 ++
 cpp/kernels/xqa/mha.cu        |   1 -
 cpp/kernels/xqa/mha_sm90.cu   | 125 ++++++++++++++++++++++++++++++++++++------
 cpp/kernels/xqa/test/test.cpp |   6 +-
 4 files changed, 119 insertions(+), 19 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/kernels/xqa/defines.h b/cpp/kernels/xqa/defines.h
index 935bc85eb..5a983fcc6 100644
--- a/cpp/kernels/xqa/defines.h
+++ b/cpp/kernels/xqa/defines.h
@@ -67,6 +67,12 @@ using MaskType = uint32_t;
 #endif
 #endif
 
+// Enables SWAP AB optimization for speculative decoding when using a small, fixed Q_SEQ_LEN.
+// NOTE: Requires a uniform input sequence length for the entire batch.
+#ifdef SPEC_Q_SEQ_LEN
+static_assert(SPEC_DEC, "SPEC_Q_SEQ_LEN should only be used when SPEC_DEC is enabled.");
+#endif
+
 // 0: half/bf16 based on INPUT_FP16; 1: int8_t; 2: __nv_fp8_e4m3
 #ifndef CACHE_ELEM_ENUM
 #define CACHE_ELEM_ENUM 2
diff --git a/cpp/kernels/xqa/mha.cu b/cpp/kernels/xqa/mha.cu
index 6d398339b..77a3ca12e 100644
--- a/cpp/kernels/xqa/mha.cu
+++ b/cpp/kernels/xqa/mha.cu
@@ -1427,7 +1427,6 @@ CUBIN_EXPORT __global__
     uint32_t const idxSubSeqInSeq = allowMultiBlockMode ? blockIdx.x : 0;
     assert(!isMultiBlock || (semaphores != nullptr && scratch != nullptr));
 
-    static_assert(inputSeqLen == 1);
     // gridDim: x - K/V sequence-dim split; y - number of K or V heads per token; z - number of requests
     assert(gridDim.z == batchSize && gridDim.y == nbKHeads);
     extern __shared__ char smemByteBuf[];
diff --git a/cpp/kernels/xqa/mha_sm90.cu b/cpp/kernels/xqa/mha_sm90.cu
index 1d40662da..9c76c0633 100644
--- a/cpp/kernels/xqa/mha_sm90.cu
+++ b/cpp/kernels/xqa/mha_sm90.cu
@@ -22,12 +22,6 @@
 #include "specDec.h"
 #endif
 
-#define SWAP_AB (!SPEC_DEC)
-
-#define IS_SUPPORTED_F16_CASE (CACHE_ELEM_ENUM == 0 && !SPEC_DEC && SWAP_AB && !USE_INPUT_KV && !LOW_PREC_OUTPUT)
-
-inline constexpr bool swapAB = SWAP_AB;
-
 #ifndef GENERATE_CUBIN
 #include "hostUtils.h"
 #include "tensorMap.h"
@@ -41,6 +35,19 @@ inline constexpr bool swapAB = SWAP_AB;
 
 #define DBG_PRINT 0
 
+#ifdef SPEC_Q_SEQ_LEN
+static_assert(SPEC_DEC, "SPEC_Q_SEQ_LEN is only supported for SPEC_DEC");
+constexpr uint32_t specDecQLen = SPEC_Q_SEQ_LEN;
+static_assert(specDecQLen * headGrpSize <= 32, "SPEC_Q_SEQ_LEN macro value is too large");
+#define SWAP_AB 1
+#else
+#define SWAP_AB (!SPEC_DEC)
+#endif
+
+#define IS_SUPPORTED_F16_CASE (CACHE_ELEM_ENUM == 0 && !SPEC_DEC && SWAP_AB && !USE_INPUT_KV && !LOW_PREC_OUTPUT)
+
+inline constexpr bool swapAB = SWAP_AB;
+
 #pragma region Config
 
 static_assert(
@@ -53,10 +60,26 @@ constexpr uint32_t gmmaWarpGrpSize = warp_size * gmmaWarpsPerGrp;
 constexpr uint32_t gemm0NbGmmaGrps = 1;
 constexpr uint32_t gemm0NbThrds = gmmaWarpGrpSize * gemm0NbGmmaGrps;
 constexpr uint32_t gemm0NbWarps = gmmaWarpsPerGrp * gemm0NbGmmaGrps;
-#if SPEC_DEC
+#if SPEC_DEC && !SWAP_AB
 inline constexpr uint32_t ctaNbQHeads = Q_HEADS_PER_CTA;
 inline constexpr uint32_t inputTokensPerCta = exactDiv(ctaNbQHeads, headGrpSize);
 constexpr uint32_t ctaNbValidQHeads = ctaNbQHeads;
+#elif SPEC_DEC && SWAP_AB
+inline constexpr uint32_t inputTokensPerCta = specDecQLen;
+inline constexpr uint32_t ctaNbValidQHeads = headGrpSize * inputTokensPerCta;
+inline constexpr
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 6313c9799c - [https://nvbugs/5488582][fix] Cherry-pick 7495: Avoid unexpected Triton recompilation in DG fused_moe (#7708)

- **Date**: 2025-09-17
- **Author**: Yukun He
- **Categories**: Kernel Optimization, Fusion

### Optimization Techniques

- Operator fusion
- MoE optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py | 6 +++---
 1 file changed, 3 insertions(+), 3 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py
index 71493b261..4c9bd9bb7 100644
--- a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py
@@ -220,7 +220,7 @@ def _preprocess_after_permute_kernel(
     expert_offsets_ptr,
     masked_m_ptr,
     token_map_ptr,
-    TOTAL_TOKENS: tl.constexpr,
+    total_tokens,
     NUM_EXPERTS: tl.constexpr,
     BLOCK_SIZE_M: tl.constexpr,
 ):
@@ -228,7 +228,7 @@ def _preprocess_after_permute_kernel(
     pid_y = tl.program_id(1)
     if pid_y == 0:
         token_offsets = pid_x * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
-        token_mask = token_offsets < TOTAL_TOKENS
+        token_mask = token_offsets < total_tokens
         # get expert_id for each token in the block
         expert_ids = tl.full((BLOCK_SIZE_M, ), NUM_EXPERTS - 1, dtype=tl.int32)
         found_mask = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.int1)
@@ -287,7 +287,7 @@ def preprocess_after_permute(expert_first_token_offset_tensor,
         expert_first_token_offset_tensor,
         masked_m,
         token_to_expert_map,
-        TOTAL_TOKENS=total_tokens,
+        total_tokens,
         NUM_EXPERTS=num_experts,
         BLOCK_SIZE_M=BLOCK_SIZE_M,
     )

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 64db7d27f6 - [feat] Optimizations on weight-only batched gemv kernel (#5420)

- **Date**: 2025-06-30
- **Author**: Cheng Hang
- **Categories**: Kernel Optimization

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Async/stream-based execution
- Integer quantization
- Batching optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../kernels/weightOnlyBatchedGemv/kernel.h         | 27 ++++++--
 .../kernels/weightOnlyBatchedGemv/utility.h        | 75 ++++++++++++++--------
 .../kernels/weightOnly/weightOnlyKernelTest.cpp    | 27 ++++----
 .../unittest/trt/quantization/test_quant_layer.py  |  2 +-
 .../quantization/test_weight_only_quant_matmul.py  |  2 +-
 5 files changed, 83 insertions(+), 50 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/kernel.h b/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/kernel.h
index 7df9305d9..de4a960e1 100644
--- a/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/kernel.h
+++ b/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/kernel.h
@@ -63,6 +63,10 @@ __global__ void kernel(TypeA* act, TypeA* act_scale, uint8_t* weight, TypeA* sca
         = (tid * StepK / (Details::kInterleave * Details::LayoutDetails::kTileSize)) * Details::LayoutDetails::kTileSize
         + ((tid * StepK) % Details::LayoutDetails::kTileSize);
 
+    bool constexpr scale_zero_ldg128 = Details::kInterleave == 1 && CtaN == 8;
+
+    using AccessTypeScaleZero = std::conditional_t<scale_zero_ldg128, AccessTypeA, TypeA>;
+
     GMemIterator<Mandatory, AccessTypeA, CtaM, Details::kAccessNumA, TypeA> act_iterator(
         act, offset_m * origin_k + real_offset_k, CtaK / Details::kInterleave, origin_k);
     GMemIterator<EnableActScale, AccessTypeA, 1, Details::kAccessNumA, TypeA> act_scale_iterator(
@@ -70,10 +74,10 @@ __global__ void kernel(TypeA* act, TypeA* act_scale, uint8_t* weight, TypeA* sca
     GMemIterator<Mandatory, AccessTypeW, CtaN, Details::kAccessNumW, uint8_t> weight_iterator(weight,
         (interleaved_offset_n * interleaved_k + tid * StepK) / Details::kElemsPerByteW, CtaK / Details::kElemsPerByteW,
         interleaved_k / Details::kElemsPerByteW);
-    GMemIterator<Mandatory, TypeA, CtaN, 1, TypeA> scales_iterator(scales,
+    GMemIterator<Mandatory, AccessTypeScaleZero, CtaN, 1, TypeA> scales_iterator(scales,
         (GroupSize != 0 ? real_offset_k / GroupSize * n : 0) + real_offset_n,
         (GroupSize != 0 ? CtaK / Details::kInterleave / GroupSize * n : 0), Details::kInterleave);
-    GMemIterator<EnableZero, TypeA, CtaN, 1, TypeA> zeros_iterator(zeros,
+    GMemIterator<EnableZero, AccessTypeScaleZero, CtaN, 1, TypeA> zeros_iterator(zeros,
         (GroupSize != 0 ? real_offset_k / GroupSize * n : 0) + real_offset_n,
         (GroupSize != 0 ? CtaK / Details::kInterleave / GroupSize * n : 0), Details::kInterleave);
 
@@ -92,11 +96,19 @@ __global__ void kernel(TypeA* act, TypeA* act_scale, uint8_t* weight, TypeA* sca
         TypeA vec_scale[CtaN], vec_zero[CtaN];
         TypeA tile_a[StepK], tile_w[StepK], tile_w_pack2[CtaN * StepK];
         uint8_t tile_w_quantized[StepK / Details::kElemsPerByteW];
-#pragma unroll
-        for (int i = 0; i < CtaN; ++i)
+        if constexpr (scale_zero_ldg128)
         {
-            scales_iterator.load(vec_scale + i, iter, i);
-            zeros_iterator.load(vec_zero + i, iter, i);
+            scales_iterator.load(vec_scale, iter);
+            zeros_iterator.load(vec_zero, iter);
+        }
+        else
+        {
+#pragma unroll
+            for (int i = 0; i < CtaN; ++i)
+            {
+                scales_iterator.load(vec_scale + i, iter, i);
+                zeros_iterator.load(vec_zero + i, iter, i);
+            }
         }
         act_scale_iterato
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 655d0f48d0 - [https://nvbugs/5455140][fix] unwaive DSR1-fp4 throughput_tp8 (#7022)

- **Date**: 2025-08-19
- **Author**: Fanrong Li
- **Categories**: Throughput/Latency

### Optimization Techniques

- FP8 quantization
- Parallelism optimization
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
index a16195d8a..4117a833e 100644
--- a/tests/integration/test_lists/waives.txt
+++ b/tests/integration/test_lists/waives.txt
@@ -286,7 +286,6 @@ accuracy/test_llm_api_pytorch.py::TestQwen3_30B_A3B::test_nvfp4[tep4_latency_moe
 accuracy/test_llm_api_pytorch.py::TestQwen3_30B_A3B::test_nvfp4[tep4_latency_moe_cutlass-torch_compile=True] SKIP (https://nvbugs/5403818)
 accuracy/test_llm_api_pytorch.py::TestQwen3_30B_A3B::test_nvfp4[tep4_latency_moe_trtllm-torch_compile=False] SKIP (https://nvbugs/5403818)
 accuracy/test_llm_api_pytorch.py::TestQwen3_30B_A3B::test_nvfp4[tep4_latency_moe_trtllm-torch_compile=True] SKIP (https://nvbugs/5403818)
-accuracy/test_llm_api_pytorch.py::TestDeepSeekR1::test_nvfp4_multi_gpus[throughput_tp8] SKIP (https://nvbugs/5442827,https://nvbugs/5445466)
 test_e2e.py::test_ptp_quickstart_advanced[Llama3.1-70B-FP8-llama-3.1-model/Llama-3.1-70B-Instruct-FP8] SKIP (https://nvbugs/5453992)
 accuracy/test_llm_api_pytorch.py::TestQwen3_235B_A22B::test_nvfp4[latency_moe_cutlass] SKIP (https://nvbugs/5454898)
 accuracy/test_llm_api_pytorch.py::TestQwen3_235B_A22B::test_nvfp4[latency_moe_trtllm] SKIP (https://nvbugs/5454898)

```

### Analysis Summary

Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 6711ad9cf3 - [TRTLLM-5589] feat: Minor optimizations for tunable FP8 batched GEMM op. (#5139)

- **Date**: 2025-06-18
- **Author**: Yukun He
- **Categories**: Quantization Optimization

### Optimization Techniques

- FP8 quantization
- KV cache optimization
- Batching optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU

### Changed Files

```
cpp/tensorrt_llm/thop/fp8BatchedGemmTrtllmGen.cpp  | 26 ++++-----
 tensorrt_llm/_torch/custom_ops/torch_custom_ops.py | 68 +++++++++-------------
 2 files changed, 40 insertions(+), 54 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/thop/fp8BatchedGemmTrtllmGen.cpp b/cpp/tensorrt_llm/thop/fp8BatchedGemmTrtllmGen.cpp
index 8e5b70c9d..3631b8177 100644
--- a/cpp/tensorrt_llm/thop/fp8BatchedGemmTrtllmGen.cpp
+++ b/cpp/tensorrt_llm/thop/fp8BatchedGemmTrtllmGen.cpp
@@ -224,7 +224,19 @@ public:
         std::optional<at::Tensor> const& dDqSfsA, std::optional<at::Tensor> const& dDqSfsB,
         std::optional<at::Tensor> const& scaleC, int64_t configIndex)
     {
-
+        // If configIndex is not provided, use the default valid config index
+        if (configIndex == -1)
+        {
+            int64_t b = mat1.size(0);
+            int64_t m = mat1.size(1);
+            int64_t n = mat2.size(1);
+            int64_t k = mat1.size(2);
+            int32_t const numTokens = 0;
+            int32_t const maxNumCtasInBatchDim = 0;
+            std::vector<int32_t> const batchedTokens(b, m);
+            configIndex
+                = mRunner->getDefaultValidConfigIndex(m, n, k, batchedTokens, numTokens, b, maxNumCtasInBatchDim);
+        }
         return fp8_batched_gemm_sm100(mat1, mat2, mTileSize, mUseDeepSeekFp8, mLowLatencyKernel, mEpilogueTileM,
             dDqSfsA, dDqSfsB, scaleC, mOutDtypeArg, *mRunner, configIndex);
     }
@@ -240,17 +252,6 @@ public:
         return mRunner->getValidConfigIndices(m, n, k, batchedTokens, numTokens, numBatches, maxNumCtasInBatchDim);
     }
 
-    int64_t getDefaultValidConfigIndex(int64_t numBatches, int64_t m, int64_t n, int64_t k) const
-    {
-        // numTokens and maxNumCtasInBatchDim are not used for static batching
-        int32_t const numTokens = 0;
-        int32_t const maxNumCtasInBatchDim = 0;
-
-        std::vector<int32_t> const batchedTokens(numBatches, m);
-
-        return mRunner->getDefaultValidConfigIndex(m, n, k, batchedTokens, numTokens, numBatches, maxNumCtasInBatchDim);
-    }
-
 private:
     using RunnerType = tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner;
     using RunnerOptionsType = tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions;
@@ -271,6 +272,5 @@ TORCH_LIBRARY_FRAGMENT(trtllm, m)
     m.class_<torch_ext::FP8BatchedGemmRunner>("FP8BatchedGemmRunner")
         .def(torch::init<at::ScalarType, bool, bool, int64_t, int64_t>())
         .def("get_valid_configs", &torch_ext::FP8BatchedGemmRunner::getValidConfigs)
-        .def("get_default_valid_config", &torch_ext::FP8BatchedGemmRunner::getDefaultValidConfigIndex)
         .def("run_batched_gemm", &torch_ext::FP8BatchedGemmRunner::runBatchedGemm);
 }
diff --git a/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py b/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
index d6c2a1e22..6e1fd681f 100644
--- a/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
+++ b/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
@@ -290,7 +290,6 @@ class FP4GemmRunner(TunableRunner):
         self,
         inputs: List[torch.Tensor],
         tactic: int = -1,
-        do_preparation: bool = False,
     ) -> torch.Tensor:
         
```

### Analysis Summary

Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 68687a9f56 - [WAR][nvbug/5321947] Add an async sleep to unblock event loop. (#5342)

- **Date**: 2025-06-19
- **Author**: Frank
- **Categories**: Parallelism/Async

### Optimization Techniques

- Async/stream-based execution

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/bench/benchmark/utils/asynchronous.py | 4 ++++
 1 file changed, 4 insertions(+)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/bench/benchmark/utils/asynchronous.py b/tensorrt_llm/bench/benchmark/utils/asynchronous.py
index 4151cbc36..af6ae18ab 100644
--- a/tensorrt_llm/bench/benchmark/utils/asynchronous.py
+++ b/tensorrt_llm/bench/benchmark/utils/asynchronous.py
@@ -125,6 +125,10 @@ class LlmManager:
             while not self._stop.is_set():
                 async for stats in self.llm.get_stats_async(2):
                     await socket.send_json(stats)
+                # NOTE: This is a WAR to force this loop to relinquish control
+                # that was preventing other async tasks from holding the event
+                # loop. If we don't
+                await asyncio.sleep(0)
 
             # Wrap up by sending any remaining statistics data
             logger.debug("Iteration log worker wrapping up...")

```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 695d7a0bdd - [TRTLLM-9939][perf] Short-sequence MHA optimization for DSA MLA prefill (#11677)

- **Date**: 2026-03-03
- **Author**: Kaiyu Xie
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- KV cache optimization
- Parallelism optimization
- Batching optimization
- PyTorch built-in optimized ops
- Attention mechanism optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU
- Short sequence scenarios
- Prefill phase
- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/modules/attention.py           | 182 ++++--
 .../_torch/attention/sparse/test_short_seq_mha.py  | 665 +++++++++++++++++++++
 2 files changed, 802 insertions(+), 45 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/attention.py b/tensorrt_llm/_torch/modules/attention.py
index a92f69aba..5afd1d369 100644
--- a/tensorrt_llm/_torch/modules/attention.py
+++ b/tensorrt_llm/_torch/modules/attention.py
@@ -1,5 +1,6 @@
 import functools
 import math
+import os
 import weakref
 from typing import List, Optional, Union, cast
 
@@ -1125,29 +1126,6 @@ class MLA(nn.Module):
         mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
         q_scaling = 1.0 / (mscale * mscale)
 
-        if not self.is_dsa:
-            self.mha = create_attention(
-                config.attn_backend,
-                self.layer_idx,
-                self.num_heads_tp,
-                head_dim=self.qk_head_dim,
-                num_kv_heads=self.num_key_value_heads_tp,
-                pos_embd_params=pos_embd_params,
-                quant_config=quant_config,
-                q_scaling=q_scaling,
-                is_mla_enable=True,
-                q_lora_rank=self.q_lora_rank,
-                kv_lora_rank=self.kv_lora_rank,
-                qk_nope_head_dim=self.qk_nope_head_dim,
-                qk_rope_head_dim=self.qk_rope_head_dim,
-                v_head_dim=self.v_head_dim,
-                predicted_tokens_per_seq=self.predicted_tokens_per_seq,
-                skip_create_weights_in_init=config.skip_create_weights_in_init,
-                sparse_attention_config=config.sparse_attention_config,
-            )
-        else:
-            self.mha = None
-
         self.mqa = create_attention(
             config.attn_backend,
             self.layer_idx,
@@ -1186,6 +1164,48 @@ class MLA(nn.Module):
                 is_neox=pos_embd_params.is_neox,
             )
 
+        # Short-sequence MHA optimization for DSA models:
+        # For short prefill sequences, use MHA (kv_b_proj expansion + standard
+        # attention) instead of the absorption path, which has overhead from
+        # extra BMMs and larger head_dim (kv_lora_rank + qk_rope_head_dim).
+        # Only active when rope_fusion is True (DSA with TrtllmAttention).
+        _threshold_str = os.environ.get('TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD',
+                                        '0')
+        try:
+            self.short_seq_mha_threshold = int(_threshold_str)
+        except ValueError as err:
+            raise ValueError(
+                f"TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD must be an integer, "
+                f"got '{_threshold_str}'") from err
+
+        # MHA attention backend: used by non-DSA (standard MLA) and optionally
+        # by DSA for the short-seq path (dense attention, no sparse config).
+        _short_seq_mha = (self.is_dsa and self.short_seq_mha_threshold > 0
+                          and not self.apply_rotary_emb)
+        if not self.is_dsa or _short_seq_mha:
+            self.mha = create_attention(
+                config.attn_backend,
+                self.layer_idx,
+                self.num_heads_tp,
+                head_dim=sel
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Most effective for short-sequence, latency-sensitive workloads.

---

## 696f754ef4 - [None][fix] avoid implicit cudaStreamSynchronize in sample_async. (#10120)

- **Date**: 2025-12-23
- **Author**: Yuxian Qiu
- **Categories**: Parallelism/Async

### Optimization Techniques

- Async/stream-based execution
- Pinned memory
- Speculative decoding

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/sampler.py          | 39 +++++++++++-----------
 .../unittest/_torch/sampler/test_torch_sampler.py  |  4 +--
 2 files changed, 22 insertions(+), 21 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index 62a43a50b..c0bb0acf7 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -2427,11 +2427,15 @@ class TorchSampler(Sampler, AsyncWorkerMixin):
         ]
 
     def _request_indices_with_stop_words(self, requests: list[LlmRequest]) -> torch.Tensor:
-        return [
-            ridx
-            for ridx, r in enumerate(requests)
-            if (r.py_stop_words_list is not None and len(r.py_stop_words_list[0]) > 0)
-        ]
+        return torch.tensor(
+            [
+                ridx
+                for ridx, r in enumerate(requests)
+                if (r.py_stop_words_list is not None and len(r.py_stop_words_list[0]) > 0)
+            ],
+            dtype=torch.int32,
+            pin_memory=True,
+        ).to(device="cuda", non_blocking=True)
 
     @nvtx_range("_write_finish_reasons")
     def _write_finish_reasons(
@@ -2635,12 +2639,11 @@ class TorchSampler(Sampler, AsyncWorkerMixin):
 
         padded_tokens = self._padded_old_tokens(requests, tokens, predecessor_beams)
 
-        def request_stop_words(request: LlmRequest, new_tokens: torch.Tensor):
+        for request_idx, request in enumerate(requests):
             swl, ends = request.py_stop_words_list
             if -1 in ends:
                 ends = ends[: ends.index(-1)]
             lens = np.diff(ends, prepend=0)
-            lens_device = torch.tensor(list(lens), pin_memory=True).to("cuda", non_blocking=True)
             max_len = np.max(lens)
 
             words = torch.zeros(len(lens), max_len, dtype=torch.int32, pin_memory=True)
@@ -2649,20 +2652,18 @@ class TorchSampler(Sampler, AsyncWorkerMixin):
             words_device = words.to("cuda", non_blocking=True)
 
             draft_token_length = get_draft_token_length(request)
-            for step_idx in range(draft_token_length + 1):
-                size_per_step = new_tokens.size(0) - draft_token_length + step_idx
-                for word, L in zip(words_device, lens_device):
-                    truncated_seq = new_tokens[size_per_step - L : size_per_step]
-                    if torch.equal(truncated_seq, word[:L]):
-                        # We don't care about subsequent steps because we already found a stop word match
-                        return step_idx
-            return None
 
-        for request_idx, request in enumerate(requests):
             for beam_idx in range(self.max_beam_width):
-                step = request_stop_words(request, padded_tokens[request_idx, beam_idx])
-                if step is not None:
-                    per_step[step, request_idx, beam_idx] = True
+                new_tokens = padded_tokens[request_idx, beam_idx]
+                for step_idx in range(draft_token_length + 1):
+                    size_per_step = new_tokens.size(0) - draft_token_length + step_idx
+                    matches = []
+       
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. Pinned (page-locked) memory enables faster host-to-device data transfers. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 6b3242654e - fix: Fix broken vanilla moe since FusedMoE refactor. (#4897)

- **Date**: 2025-06-05
- **Author**: Yuxian Qiu
- **Categories**: Fusion

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- Parallelism optimization
- Reduce synchronization overhead
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU

### Changed Files

```
examples/pytorch/quickstart_advanced.py            |  2 +-
 tensorrt_llm/_torch/models/modeling_deepseekv3.py  | 14 +++++++------
 tensorrt_llm/_torch/models/modeling_llama.py       | 12 ++++++-----
 .../_torch/modules/fused_moe/create_moe.py         |  9 --------
 .../_torch/modules/fused_moe/fused_moe_vanilla.py  | 24 ++++++++++++----------
 tests/unittest/_torch/modules/test_fused_moe.py    |  3 +--
 6 files changed, 30 insertions(+), 34 deletions(-)
```

### Diff Preview

```diff
diff --git a/examples/pytorch/quickstart_advanced.py b/examples/pytorch/quickstart_advanced.py
index 7381f777d..59a6cbd86 100644
--- a/examples/pytorch/quickstart_advanced.py
+++ b/examples/pytorch/quickstart_advanced.py
@@ -50,7 +50,7 @@ def add_llm_args(parser):
     parser.add_argument('--moe_backend',
                         type=str,
                         default='CUTLASS',
-                        choices=['CUTLASS', 'TRTLLM'])
+                        choices=['CUTLASS', 'TRTLLM', 'VANILLA'])
     parser.add_argument('--enable_attention_dp',
                         default=False,
                         action='store_true')
diff --git a/tensorrt_llm/_torch/models/modeling_deepseekv3.py b/tensorrt_llm/_torch/models/modeling_deepseekv3.py
index 4e88d1a10..8c18575f1 100644
--- a/tensorrt_llm/_torch/models/modeling_deepseekv3.py
+++ b/tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -486,12 +486,14 @@ class Deepseekv3MoE(nn.Module):
 
         router_logits = self.gate(hidden_states)
 
-        routed_output = self.experts(hidden_states_fp4 or hidden_states,
-                                     router_logits,
-                                     cutlass_min_latency_mode,
-                                     output_dtype=hidden_states.dtype,
-                                     all_rank_num_tokens=all_rank_num_tokens,
-                                     use_dp_padding=use_dp_padding)
+        routed_output = self.experts(
+            hidden_states_fp4 or hidden_states,
+            router_logits,
+            cutlass_min_latency_mode=cutlass_min_latency_mode,
+            output_dtype=hidden_states.dtype,
+            all_rank_num_tokens=all_rank_num_tokens,
+            use_dp_padding=use_dp_padding,
+        )
 
         return routed_output
 
diff --git a/tensorrt_llm/_torch/models/modeling_llama.py b/tensorrt_llm/_torch/models/modeling_llama.py
index 6e67e35d6..a92e83a7e 100644
--- a/tensorrt_llm/_torch/models/modeling_llama.py
+++ b/tensorrt_llm/_torch/models/modeling_llama.py
@@ -286,11 +286,13 @@ class Llama4MoE(nn.Module):
                 (0, 0, 0,
                  max_num_token_across_dp_ranks - hidden_states.shape[0]))
         router_logits = self.router(hidden_states)
-        routed_output = self.experts(hidden_states,
-                                     router_logits,
-                                     cutlass_min_latency_mode,
-                                     all_rank_num_tokens=all_rank_num_tokens,
-                                     use_dp_padding=use_dp_padding)
+        routed_output = self.experts(
+            hidden_states,
+            router_logits,
+            cutlass_min_latency_mode=cutlass_min_latency_mode,
+            all_rank_num_tokens=all_rank_num_tokens,
+            use_dp_padding=use_dp_padding,
+        )
         return routed_output
 
     def forward(
diff --git a/tensorrt_llm/_torch/modules/fused_moe/create_moe.py b/tensorrt_llm/_torch/modules/fused_moe/create_moe.py
index
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 6da95f29a9 - [None][feat] Add support for fused gate_up_proj scales for FP8 blockwise (#6496)

- **Date**: 2025-08-05
- **Author**: Aurelien Chartier
- **Categories**: Fusion, Quantization Optimization

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- FP8 quantization
- Parallelism optimization
- MoE optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU

### Changed Files

```
.../_torch/modules/fused_moe/quantization.py       | 63 +++++++++++++---------
 tests/unittest/_torch/modules/test_fused_moe.py    | 33 ++++++++++--
 2 files changed, 67 insertions(+), 29 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fused_moe/quantization.py b/tensorrt_llm/_torch/modules/fused_moe/quantization.py
index 18e9c7cc9..249aadc04 100644
--- a/tensorrt_llm/_torch/modules/fused_moe/quantization.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/quantization.py
@@ -528,31 +528,44 @@ class DeepSeekFP8BlockScalesFusedMoEMethod(FusedMoEMethodBase):
             load_expert_ids: List[int], dst_w3_w1_weight_scale: torch.Tensor,
             dst_w2_weight_scale: torch.Tensor, device):
         for local_slot_id, expert_id in enumerate(load_expert_ids):
-            w3_scale = load_weight_shard(
-                weights[f"{expert_id}.w3.weight_scale_inv"],
-                module.tp_size,
-                module.tp_rank,
-                TensorParallelMode.COLUMN,
-                device=device)
-            dst_w3_w1_weight_scale[local_slot_id][:dst_w3_w1_weight_scale.
-                                                  shape[-2] //
-                                                  2].copy_(w3_scale)
-            w1_scale = load_weight_shard(
-                weights[f"{expert_id}.w1.weight_scale_inv"],
-                module.tp_size,
-                module.tp_rank,
-                TensorParallelMode.COLUMN,
-                device=device)
-            dst_w3_w1_weight_scale[local_slot_id][dst_w3_w1_weight_scale.
-                                                  shape[-2] //
-                                                  2:].copy_(w1_scale)
-            w2_scale = load_weight_shard(
-                weights[f"{expert_id}.w2.weight_scale_inv"],
-                module.tp_size,
-                module.tp_rank,
-                TensorParallelMode.ROW,
-                device=device)
-            dst_w2_weight_scale[local_slot_id].copy_(w2_scale)
+            if module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
+                w3_scale = weights['gate_up_proj_weight_scale'][
+                    expert_id].transpose(0, 1).contiguous()
+                w1_scale = None
+                w2_scale = weights['down_proj_weight_scale'][
+                    expert_id].transpose(0, 1).contiguous()
+            elif module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
+                w3_scale = weights[f"{expert_id}.w3.weight_scale_inv"]
+                w1_scale = weights[f"{expert_id}.w1.weight_scale_inv"]
+                w2_scale = weights[f"{expert_id}.w2.weight_scale_inv"]
+            else:
+                raise NotImplementedError(
+                    f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
+                )
+
+            w3_w1_scale_shard = load_weight_shard(w3_scale,
+                                                  module.tp_size,
+                                                  module.tp_rank,
+                                                  TensorParallelMode.COLUMN,
+                                                  device=device)
+
+            if w1_scale is not
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 6e1aee6fd6 - [fix] Performance Optimization for MNNVL TwoShot Kernel (#5934)

- **Date**: 2025-07-16
- **Author**: Shiyu Li
- **Categories**: Kernel Optimization

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- Async/stream-based execution
- Integer quantization
- Parallelism optimization
- Reduce synchronization overhead
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
.../mnnvlTwoShotAllreduceKernels.cu                | 148 +++++++++++++--------
 cpp/tensorrt_llm/runtime/mcastDeviceMemory.cpp     |  33 +++--
 cpp/tensorrt_llm/runtime/mcastDeviceMemory.h       |  16 ++-
 tensorrt_llm/_torch/models/modeling_deepseekv3.py  |  12 +-
 4 files changed, 130 insertions(+), 79 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/communicationKernels/mnnvlTwoShotAllreduceKernels.cu b/cpp/tensorrt_llm/kernels/communicationKernels/mnnvlTwoShotAllreduceKernels.cu
index f2e87e39d..6f85317ae 100644
--- a/cpp/tensorrt_llm/kernels/communicationKernels/mnnvlTwoShotAllreduceKernels.cu
+++ b/cpp/tensorrt_llm/kernels/communicationKernels/mnnvlTwoShotAllreduceKernels.cu
@@ -61,6 +61,31 @@ inline __device__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float val)
     return __float2bfloat16(val);
 }
 
+__device__ float4 loadfloat4(void const* ptr)
+{
+
+    float return_value[4];
+
+    asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
+                 : "=f"(return_value[0]), "=f"(return_value[1]), "=f"(return_value[2]), "=f"(return_value[3])
+                 : "l"(ptr));
+
+    return *(float4*) return_value;
+}
+
+__device__ __inline__ float2 loadfloat2(void const* ptr)
+{
+
+    float return_value[2];
+
+    asm volatile("ld.volatile.global.v2.f32 {%0, %1}, [%2];\n"
+                 : "=f"(return_value[0]), "=f"(return_value[1])
+                 : "l"(ptr)
+                 : "memory");
+
+    return *(float2*) return_value;
+}
+
 template <int WORLD_SIZE, typename T>
 __global__ void twoshot_allreduce_kernel(T* output_ptr, T* shard_ptr, T** input_ptrs, T* mcast_ptr, int num_tokens,
     int buffer_M, int token_dim, int rank, uint32_t* buffer_flags, bool wait_for_results)
@@ -74,20 +99,13 @@ __global__ void twoshot_allreduce_kernel(T* output_ptr, T* shard_ptr, T** input_
     cudaGridDependencySynchronize();
 #endif
 
+    // [input_ptr, clear_ptr, buffer_size, access_counter]
+    uint4 flag = reinterpret_cast<uint4*>(buffer_flags)[0];
+    // Each buffer is M * N and we have 2 buffers in each group, one for reduce-scatter and one for allgather
+    uint32_t buffer_group_size = flag.z << 1;
+    uint32_t input_offset = flag.x * buffer_group_size;
+    uint32_t clear_offset = flag.y * buffer_group_size;
     uint32_t* offset_access_ptr = &buffer_flags[3];
-    // Buffer size is M * N, and we need two buffers for reduce-scatter and allgather
-    uint32_t buffer_size = (buffer_flags[2] << 1);
-    uint32_t input_offset = buffer_flags[0] * buffer_size;
-    uint32_t clear_offset = buffer_flags[1] * buffer_size;
-
-    if (wait_for_results)
-    {
-        __syncthreads();
-        if (threadIdx.x == 0)
-        {
-            atomicAdd(offset_access_ptr, 1);
-        }
-    }
 
     if (elt < token_dim)
     {
@@ -101,17 +119,16 @@ __global__ void twoshot_allreduce_kernel(T* output_ptr, T* shard_ptr, T** input_
 
         // Reduce and broadcast
 
-        int global_token = token * WORLD_SIZE + rank;
-        if (global_token < num_tokens)
+        if ((token % WORLD_SIZE) == rank)
         {
-
+            int local_token = token / WORLD_SIZE;
             float accum = 0.f;
 
             T values[WORLD_SIZE];
 
             for (int r = 0; r < WORLD_SIZE; r++)
             {
-                input_ptrs[rank][clear_offset + tok
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 6e470aab72 - [None] [feat] Optimize the algorithm part of RocketKV (#9333)

- **Date**: 2025-12-01
- **Author**: heyuhhh
- **Categories**: General Performance

### Optimization Techniques

- Torch compilation/JIT optimization
- Custom CUDA kernel
- Async/stream-based execution
- FP8 quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Triton kernel
- Reduce synchronization overhead
- Attention mechanism optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- Long sequence / large context scenarios
- Prefill phase
- Decode/generation phase

### Changed Files

```
cpp/tensorrt_llm/thop/IndexerTopKOp.cpp            |   2 -
 examples/llm-api/llm_sparse_attention.py           |  16 +-
 examples/longbench/eval_longbench_v1.py            |  24 +-
 examples/longbench/eval_longbench_v2.py            |  14 +-
 .../_torch/attention_backend/sparse/kernel.py      | 283 ++++++++++-----------
 .../_torch/attention_backend/sparse/rocket.py      | 112 +++++---
 tensorrt_llm/llmapi/llm_args.py                    |  17 +-
 .../_torch/attention/sparse/test_rocketkv.py       |  32 ++-
 .../_torch/attention/sparse/test_triton_bmm.py     |  40 ++-
 .../_torch/thop/parallel/test_indexer_topk.py      |   8 +-
 10 files changed, 321 insertions(+), 227 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/thop/IndexerTopKOp.cpp b/cpp/tensorrt_llm/thop/IndexerTopKOp.cpp
index 471ee19be..8a5003238 100644
--- a/cpp/tensorrt_llm/thop/IndexerTopKOp.cpp
+++ b/cpp/tensorrt_llm/thop/IndexerTopKOp.cpp
@@ -57,7 +57,6 @@ void indexer_topk_decode(
     TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
 
     TORCH_CHECK(next_n > 0, "next_n must be greater than 0");
-    TORCH_CHECK(index_topk == 2048, "index_topk must be 2048 for now");
 
     int32_t num_rows = static_cast<int32_t>(numRows64);
     int32_t num_columns = static_cast<int32_t>(numColumns64);
@@ -95,7 +94,6 @@ void indexer_topk_prefill(th::Tensor const& logits, th::Tensor const& row_starts
 
     TORCH_CHECK(indices.dim() == 2, "indices must be a 2D Tensor");
     TORCH_CHECK(logits.dim() == 2, "logits must be a 2D Tensor");
-    TORCH_CHECK(index_topk == 2048, "index_topk must be 2048 for now");
 
     auto const inputSize = logits.sizes();
     auto const numRows64 = inputSize[0];
diff --git a/examples/llm-api/llm_sparse_attention.py b/examples/llm-api/llm_sparse_attention.py
index 2739ecaa5..3ebe4dcb6 100644
--- a/examples/llm-api/llm_sparse_attention.py
+++ b/examples/llm-api/llm_sparse_attention.py
@@ -44,6 +44,7 @@ def parse_arguments():
         type=str,
         default="tests/unittest/_torch/multi_gpu/test_star_attention_input.jsonl"
     )
+
     # Build config
     parser.add_argument('--algo',
                         type=str,
@@ -53,6 +54,8 @@ def parse_arguments():
                         type=str,
                         default='TRTLLM',
                         choices=['VANILLA', 'TRTLLM'])
+
+    # RocketKV config
     parser.add_argument('--window_size',
                         type=int,
                         default=32,
@@ -65,6 +68,14 @@ def parse_arguments():
                         type=int,
                         default=2048,
                         help="The prompt budget for RocketKV.")
+    parser.add_argument('--topk',
+                        type=int,
+                        default=64,
+                        help='Top-k for RocketKV')
+    parser.add_argument('--kt_cache_dtype',
+                        type=str,
+                        default='float8_e5m2',
+                        choices=['bfloat16', 'float8_e5m2'])
     parser.add_argument('--index_max_chunk_size',
                         type=int,
                         default=32768,
@@ -106,6 +117,7 @@ def parse_arguments():
     # KV cache
     parser.add_argument('--kv_cache_dtype', type=str, default='auto')
     parser.add_argument("--kv_cache_fraction", type=float, default=0.7)
+    parser.add_argument('--tokens_per_block', type=int, default=32)
     parser.add_argument('--num_samples', type=int, default=10)
 
     # Runtime
@@ -139,8 +151,8 @@ def run_llm(args, sparse_attention_config):
         enable_block_reuse=
         False,  # sparse attention does not support kv cache reuse now
         free_gpu_memory_fraction=args.kv_cache_f
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Most effective for long-context inference workloads.

---

## 6f3acc0614 - [https://nvbugs/5892646][perf] Long-sequence token-parallel optimization for DSA indexer prefill (#11871)

- **Date**: 2026-03-10
- **Author**: nvxuanyuc
- **Categories**: Parallelism/Async

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- FP8 quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Speculative decoding
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- Short sequence scenarios
- Prefill phase

### Changed Files

```
.../_torch/attention_backend/sparse/dsa.py         | 62 +++++++++++++++++-----
 tensorrt_llm/_torch/model_config.py                |  5 +-
 tensorrt_llm/llmapi/llm_args.py                    |  6 +++
 .../defs/accuracy/test_llm_api_pytorch.py          | 19 +++++--
 .../test_lists/qa/llm_function_core.txt            |  1 +
 5 files changed, 75 insertions(+), 18 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/attention_backend/sparse/dsa.py b/tensorrt_llm/_torch/attention_backend/sparse/dsa.py
index b38e6ad7d..1cef604ed 100644
--- a/tensorrt_llm/_torch/attention_backend/sparse/dsa.py
+++ b/tensorrt_llm/_torch/attention_backend/sparse/dsa.py
@@ -11,6 +11,7 @@ from tensorrt_llm._torch.attention_backend.interface import (
     MLAParams, PositionalEmbeddingParams)
 from tensorrt_llm._torch.attention_backend.trtllm import (
     TrtllmAttention, TrtllmAttentionMetadata)
+from tensorrt_llm._torch.distributed.ops import allgather
 from tensorrt_llm._torch.modules.layer_norm import LayerNorm
 from tensorrt_llm._torch.modules.linear import Linear
 from tensorrt_llm._torch.modules.multi_stream_utils import \
@@ -1368,40 +1369,77 @@ class Indexer(nn.Module):
         if has_prefill and not metadata.skip_indexer_for_ctx_reqs:
             # Use chunked prefill to reduce memory footprint
             if metadata.indexer_prefill_chunks is not None:
+
+                # Default to 8192 if sparse_attention_config is not available (e.g., in unit tests)
+                q_split_threshold = metadata.sparse_attention_config.q_split_threshold if metadata.sparse_attention_config is not None else 8192
+                q_split_eligible = q_split_threshold >= 0 and metadata.mapping is not None and not metadata.mapping.enable_attention_dp and metadata.mapping.tp_size > 1
+
+                if q_split_eligible:
+                    tp_rank = metadata.mapping.tp_rank
+                    tp_size = metadata.mapping.tp_size
+
                 for chunk in metadata.indexer_prefill_chunks:
                     # Gather K from cache for this chunk (dual to _update_k_cache)
                     chunk_k_fp8, chunk_k_scale = self._gather_k_cache_for_chunk(
                         metadata, chunk)
+
+                    chunk_num_token = chunk.token_end - chunk.token_start
+                    apply_q_split = q_split_eligible and chunk_num_token >= q_split_threshold
+                    if apply_q_split:
+                        chunk_q_start = chunk_num_token * tp_rank // tp_size
+                        chunk_q_end = chunk_num_token * (tp_rank + 1) // tp_size
+                    else:
+                        chunk_q_start = 0
+                        chunk_q_end = chunk_num_token
+
+                    global_q_start = chunk.token_start + chunk_q_start
+                    global_q_end = chunk.token_start + chunk_q_end
+
                     logits = fp8_mqa_logits(
-                        q_fp8[chunk.token_start:chunk.token_end, ...],
+                        q_fp8[global_q_start:global_q_end, ...],
                         (chunk_k_fp8, chunk_k_scale),
-                        weights[chunk.token_start:chunk.token_end, ...],
-                        chunk.cu_seqlen_ks,
-                        chunk.cu_seqlen_ke,
+                        weights[global_q_start:global_q_end, ...],
+                        chunk.cu_seqlen_ks[chunk_q_start:chunk_q_e
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Most effective for short-sequence, latency-sensitive workloads.

---

