# Performance Optimization Analysis - Part 3

Commits 59 to 87 of 283

---

## 32dfdfba30 - feat: fuse w4a8 moe pre-quant scale on Hopper (#5613)

- **Date**: 2025-07-02
- **Author**: Xiaowei Wang
- **Categories**: Fusion, Quantization Optimization

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- MoE optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU

### Changed Files

```
.../cutlass_kernels/include/moe_util_kernels.h     |  5 +-
 .../cutlass_kernels/moe_gemm/moe_kernels.cu        | 82 +++++++++++++++++-----
 tests/unittest/_torch/modules/test_fused_moe.py    |  1 -
 3 files changed, 68 insertions(+), 20 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h b/cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h
index 819520c0d..6b346d730 100644
--- a/cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h
+++ b/cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h
@@ -52,13 +52,14 @@ void threeStepBuildExpertMapsSortFirstToken(int const* token_selected_experts, i
     int64_t const num_tokens, int64_t const num_experts_per_node, int64_t const num_experts_per_token,
     int const start_expert_id, cudaStream_t stream);
 
-template <class InputActivationsType, class ExpandedActivationsType>
+template <class InputActivationsType, class ExpandedActivationsType, bool PRE_QUANT_AWQ = false>
 void expandInputRowsKernelLauncher(InputActivationsType const* unpermuted_input,
     ExpandedActivationsType* permuted_output, float const* unpermuted_scales, float* permuted_scales,
     int const* permuted_row_to_unpermuted_row, int64_t const num_rows, int64_t const cols, int const k,
     int const num_experts_per_node, float const* fc1_act_global_scale, bool use_per_expert_act_scale,
     int64_t* expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
-    TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, cudaStream_t stream);
+    TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, cudaStream_t stream,
+    void const* prequant_scales = nullptr);
 
 template <class OutputType, class GemmOutputType, class ScaleBiasType>
 void finalizeMoeRoutingKernelLauncher(GemmOutputType const* expanded_permuted_rows,
diff --git a/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu b/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu
index daedfee0f..1610546e2 100644
--- a/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu
+++ b/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu
@@ -1422,13 +1422,14 @@ __host__ __device__ constexpr static U arrayConvert(T const& input)
 
 constexpr static int EXPAND_THREADS_PER_BLOCK = 256;
 
-template <class InputActivationsType, class ExpandedActivationsType>
+template <class InputActivationsType, class ExpandedActivationsType, bool PRE_QUANT_AWQ>
 __global__ void expandInputRowsKernel(InputActivationsType const* unpermuted_input,
     ExpandedActivationsType* permuted_output, float const* unpermuted_scales, float* permuted_scales,
     int const* permuted_row_to_unpermuted_row, int64_t const num_rows, int64_t const cols, int64_t const k,
     float const* fc1_act_global_scale, bool use_per_expert_act_scale, int64_t const* expert_first_token_offset,
     TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
-    TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, int64_t const num_experts_per_node)
+    TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, int64_t const num_experts_per_node,
+    InputActivationsType const* prequa
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 336593cac5 - [None][fix] Fix topk outIndices when using vectorized_process (#9404)

- **Date**: 2025-11-25
- **Author**: YueWeng
- **Categories**: Kernel Optimization

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access

### Applicable Conditions

- General LLM inference

### Changed Files

```
cpp/tensorrt_llm/kernels/indexerTopK.cu | 10 +++++++++-
 1 file changed, 9 insertions(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/indexerTopK.cu b/cpp/tensorrt_llm/kernels/indexerTopK.cu
index 4b2beb895..361748a38 100644
--- a/cpp/tensorrt_llm/kernels/indexerTopK.cu
+++ b/cpp/tensorrt_llm/kernels/indexerTopK.cu
@@ -570,7 +570,15 @@ static __device__ void topKPerRowJob(int const* indices, float const* logits, in
         }
         else
         {
-            outIndices[i] = smemOutput[i] - rowStart;
+            if (stride1 == 1)
+            {
+                // stride1 == 1 will use vectorized_process, which indexes already skip the rowStart.
+                outIndices[i] = smemOutput[i];
+            }
+            else
+            {
+                outIndices[i] = smemOutput[i] - rowStart;
+            }
         }
     }
 }

```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 34a730aaf7 - [None][fix] Fix enable_alltoall passed to CutlassFusedMoE (#11016)

- **Date**: 2026-01-29
- **Author**: Enwei Zhu
- **Categories**: Fusion

### Optimization Techniques

- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- Parallelism optimization
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
examples/layer_wise_benchmarks/run.py              |  6 +++-
 .../_torch/modules/fused_moe/configurable_moe.py   | 33 ++++------------------
 .../_torch/modules/fused_moe/fused_moe_cutlass.py  |  7 ++++-
 3 files changed, 16 insertions(+), 30 deletions(-)
```

### Diff Preview

```diff
diff --git a/examples/layer_wise_benchmarks/run.py b/examples/layer_wise_benchmarks/run.py
index 1c33881ca..a560cff95 100644
--- a/examples/layer_wise_benchmarks/run.py
+++ b/examples/layer_wise_benchmarks/run.py
@@ -206,7 +206,11 @@ if args.run_type == "GEN":
         max(1, 20480 // ctx_seq_len_q),
     )
     ctx_attn_workspace = torch.empty((0,), device="cuda", dtype=torch.int8)
-    with mock.patch.dict(os.environ, {"TRTLLM_FORCE_ALLTOALL_METHOD": "NotEnabled"}, clear=False):
+    with mock.patch.dict(
+        os.environ,
+        {"TRTLLM_FORCE_ALLTOALL_METHOD": "NotEnabled", "TRTLLM_FORCE_COMM_METHOD": "ALLGATHER"},
+        clear=False,
+    ):
         ctx_runner = Runner(
             args.model,
             mapping,
diff --git a/tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py b/tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py
index ce251234e..f1de1b752 100644
--- a/tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py
@@ -313,7 +313,7 @@ class ConfigurableMoE(MoE):
         2. Validates if current AllToAll strategy can be used for given workload
         3. Falls back to AllGather if current strategy cannot be used (logs info message)
 
-        After calling this method, use _is_using_alltoall() to check which method is active.
+        After calling this method, use enable_alltoall to check which method is active.
 
         Args:
             all_rank_num_tokens: Token counts per rank
@@ -348,23 +348,6 @@ class ConfigurableMoE(MoE):
             # Switch to AllGather (always works)
             self.comm = AllGatherReduceScatter(mapping=self.mapping)
 
-    def _is_using_alltoall(self) -> bool:
-        """
-        Check if current communication strategy uses alltoall
-
-        Returns:
-            True: Strategy uses alltoall (NVLINK, DeepEP, etc.)
-            False: Strategy uses allgather (AllGatherReduceScatter or None)
-
-        Note: Can be called anytime. If comm is None, returns False (no alltoall).
-              Typically called after determine_communication_method() to get accurate result.
-        """
-        if self.comm is None:
-            return False  # No strategy means no alltoall
-
-        # AllGather uses allgather, all others use alltoall
-        return not isinstance(self.comm, AllGatherReduceScatter)
-
     def _create_comm_strategy_auto(self) -> Communication:
         """
         Auto-create the best communication strategy based on hardware and configuration
@@ -810,11 +793,7 @@ class ConfigurableMoE(MoE):
 
         Same as original implementation - chunking logic is backend-agnostic
 
-        Note: use_all_to_all is determined internally via _is_using_alltoall()
-
         """
-        # Determine if using alltoall
-        use_all_to_all = self._is_using_alltoall()
         # ========== Chunk preparation ==========
         if self.use_dp:
             # When using DP: need all ranks' token counts for redu
```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 361132b98a - [https://nvbugs/5885070][fix] fix deepeplowlatency with cutedsl moe backend (#11769)

- **Date**: 2026-03-02
- **Author**: Leslie Fang
- **Categories**: Throughput/Latency

### Optimization Techniques

- Operator fusion
- Parallelism optimization
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py | 6 +++++-
 1 file changed, 5 insertions(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py
index 3f2d74144..9812c4ef8 100644
--- a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py
@@ -697,6 +697,10 @@ class CuteDslFusedMoE(CutlassFusedMoE):
             b_sf=self.quant_scales[1],
             offset_array=expert_first_token_offset,
         )
+        top_k = self.routing_method.top_k
+        if token_selected_experts is not None:
+            top_k = token_selected_experts.shape[-1]
+
         x = torch.ops.trtllm.moe_finalize_scale_op(
             x,
             None,  # biases
@@ -709,7 +713,7 @@ class CuteDslFusedMoE(CutlassFusedMoE):
             token_final_scales.size(0),  # num_rows
             self.hidden_size,  # (possibly padded) hidden_size
             self.unpadded_hidden_size,  # original hidden size
-            self.routing_method.top_k,
+            top_k,
             self.expert_size_per_partition,  # num_experts_per_node
             self.tp_size,
             self.tp_rank,

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 37a1bd810f - [https://nvbugs/5481385][fix] Fix max_seq_len in cuda graph warmup and intermediate_size in fused_moe_deepgemm (#7345)

- **Date**: 2025-08-29
- **Author**: Fanrong Li
- **Categories**: Fusion

### Optimization Techniques

- Operator fusion
- FP8 quantization
- KV cache optimization
- Batching optimization
- Speculative decoding
- MoE optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py | 8 ++++----
 tensorrt_llm/_torch/pyexecutor/model_engine.py              | 8 ++++++++
 2 files changed, 12 insertions(+), 4 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py
index 4b32f7f47..392fff091 100644
--- a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py
@@ -411,7 +411,7 @@ class DeepGemmFusedMoE(CutlassFusedMoE):
 
     def get_workspace(self, m_max: int, group_size: int):
         hidden_size = self.hidden_size
-        intermediate_size = self.intermediate_size
+        intermediate_size = self.intermediate_size_per_partition
         num_experts = self.expert_size_per_partition
 
         # create workspace
@@ -564,7 +564,7 @@ class DeepGemmFusedMoE(CutlassFusedMoE):
         # grouped gemm 1
         h1 = set_strides(workspace["workspace_1"],
                          self.expert_size_per_partition, m_max,
-                         self.intermediate_size * 2)
+                         self.intermediate_size_per_partition * 2)
 
         deepgemm_fp8_group_blockwise_gemm(
             d=h1,
@@ -579,9 +579,9 @@ class DeepGemmFusedMoE(CutlassFusedMoE):
         # activation and quantization
         act_input_fp8 = set_strides(workspace["workspace_0"],
                                     self.expert_size_per_partition, m_max,
-                                    self.intermediate_size)
+                                    self.intermediate_size_per_partition)
 
-        scale_k = fp8_utils.ceil_div(self.intermediate_size, 128)
+        scale_k = fp8_utils.ceil_div(self.intermediate_size_per_partition, 128)
         scale_k_padded = fp8_utils.align(scale_k, 4)
         act_input_sf = set_strides(workspace["workspace_sf"],
                                    self.expert_size_per_partition,
diff --git a/tensorrt_llm/_torch/pyexecutor/model_engine.py b/tensorrt_llm/_torch/pyexecutor/model_engine.py
index 2920f5b69..1e359728a 100644
--- a/tensorrt_llm/_torch/pyexecutor/model_engine.py
+++ b/tensorrt_llm/_torch/pyexecutor/model_engine.py
@@ -583,7 +583,15 @@ class PyTorchModelEngine(ModelEngine):
 
                 # Add one dummy request with the maximum possible sequence length.
                 # The sequence length is limited by both the max_seq_len and the number of available blocks.
+                # Also, the sequence length is limited by the max_position_embeddings.
                 token_num = max(1, min(available_tokens, self.max_seq_len - 1))
+                model_config = self.model.model_config.pretrained_config
+                max_position_embeddings = getattr(model_config,
+                                                  'max_position_embeddings',
+                                                  None)
+                if max_position_embeddings is not None:
+                    token_num = min(token_num,
+                                    max_position_embeddings - draft_len)
                 max_seq_len_request = kv_cache_manager.add_dummy_requests(
                     request_id
```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 390a7fd6b1 - [None][feat] Qwen3.5 perf optimizations (#11581)

- **Date**: 2026-03-13
- **Author**: Suyog Gupta
- **Categories**: General Performance

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
- Triton kernel
- PyTorch built-in optimized ops
- Reduce synchronization overhead
- Multi-stream execution
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
.../model_registry/configs/qwen3.5_moe_35b.yaml    |  45 +-
 .../model_registry/configs/qwen3.5_moe_400b.yaml   |  42 +-
 .../_torch/auto_deploy/config/default.yaml         |  18 +-
 .../custom_ops/attention/flashinfer_attention.py   |   2 +-
 .../auto_deploy/custom_ops/attention_interface.py  |  15 +
 .../custom_ops/fla/fla_backend_delta.py            |  13 +-
 .../custom_ops/fla/fla_backend_gated_delta.py      | 138 +++--
 .../auto_deploy/custom_ops/fla/fla_gated_delta.py  |  77 ++-
 .../auto_deploy/custom_ops/fla/gdn_gating.py       | 186 ++++++
 .../custom_ops/fla/torch_backend_gated_delta.py    | 212 ++++---
 .../custom_ops/fused_moe/benchmark_routing.py      | 195 ++++++
 .../custom_ops/fused_moe/triton_routing.py         | 214 +++++++
 .../_torch/auto_deploy/custom_ops/linear/swiglu.py | 137 ++++
 .../custom_ops/mamba/flashinfer_backend_mamba.py   |   3 +
 .../custom_ops/mamba/mamba_backend_common.py       |  12 +-
 .../custom_ops/mamba/triton_backend_mamba.py       |   3 +
 .../custom_ops/normalization/rms_norm.py           |   2 +-
 .../custom_ops/quantization/torch_quant.py         |  17 +-
 .../models/custom/modeling_qwen3_5_moe.py          |  48 +-
 .../auto_deploy/models/patches/qwen3_next.py       |  30 +-
 .../transform/library/fuse_gdn_gating.py           |  86 +++
 .../auto_deploy/transform/library/fuse_swiglu.py   | 268 ++++++++
 .../auto_deploy/transform/library/moe_routing.py   | 225 +++++++
 .../transform/library/multi_stream_gemm.py         | 348 +++++++++++
 .../transform/library/multi_stream_moe.py          |   2 +-
 .../auto_deploy/transform/library/rms_norm.py      |  88 +++
 .../auto_deploy/transform/library/sharding.py      |  42 +-
 .../defs/accuracy/test_llm_api_autodeploy.py       | 178 +++---
 .../auto_deploy/_utils_test/_model_test_utils.py   |   4 +-
 .../transformations/library/test_tp_sharding.py    |  29 +-
 .../fla/test_fla_cached_gated_delta_rule.py        | 196 +++---
 .../custom_ops/fla/test_fused_gdn_gating.py        |  90 +++
 .../fla/test_torch_cached_gated_delta_rule.py      | 153 +++--
 .../mamba/test_flashinfer_mamba_cached_op.py       |   3 +
 .../mamba/test_triton_mamba_cached_op.py           |   4 +
 .../singlegpu/custom_ops/test_multi_stream_gemm.py | 690 +++++++++++++++++++++
 .../singlegpu/models/test_qwen3_5_moe.py           |   2 +-
 .../models/test_qwen3_next_gdn_patches.py          |  71 ++-
 .../singlegpu/smoke/test_ad_build_small_single.py  |  13 +-
 .../library/test_finegrained_fp8_swiglu.py         | 397 ++++++++++++
 .../library/test_fuse_gdn_gating.py                | 112 ++++
 .../library/test_gated_delta_rule_cache.py         |  60 +-
 .../library/test_torch_gated_delta_rule_cache.py   |  60 +-
 43 files changed, 3911 insertions(+), 619 deletions(-)
```

### Diff Preview

```diff
diff --git a/examples/auto_deploy/model_registry/configs/qwen3.5_moe_35b.yaml b/examples/auto_deploy/model_registry/configs/qwen3.5_moe_35b.yaml
index 1a86f4626..0870b07a3 100644
--- a/examples/auto_deploy/model_registry/configs/qwen3.5_moe_35b.yaml
+++ b/examples/auto_deploy/model_registry/configs/qwen3.5_moe_35b.yaml
@@ -1,38 +1,31 @@
 runtime: trtllm
 compile_backend: torch-cudagraph
-max_seq_len: 4096
+attn_backend: trtllm
+max_seq_len: 8192
 max_num_tokens: 4096
 max_batch_size: 512
-world_size: 2
+world_size: 4
+cuda_graph_batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
 enable_chunked_prefill: true
 model_factory: AutoModelForCausalLM
 kv_cache_config:
   enable_block_reuse: false
-  free_gpu_memory_fraction: 0.95
-  tokens_per_block: 64
+  free_gpu_memory_fraction: 0.8
+  tokens_per_block: 32
 model_kwargs:
   torch_dtype: bfloat16
-  # text_config:
-  #   num_hidden_layers: 6
-  # vision_config:
-  #   depth: 2
 transforms:
+  export_to_gm:
+    num_moe_experts_for_export: 2
+  fuse_gemms_mixed_children:
+    enabled: true
   detect_sharding:
-    sharding_dims: ['tp','ep', 'bmm']
-    # use only manual config for TP sharding
-    sharding_source: ['manual']
-    manual_config:
-      tp_plan:
-        # GDN layer
-        "in_proj_qkv": "delta"
-        # attention layer
-        "q_proj": "colwise"
-        "k_proj": "colwise"
-        "v_proj": "colwise"
-        "o_proj": "rowwise"
-        # replicating shared experts (keep them commented out)
-        # "shared_expert_gate_proj": "colwise"
-        # "shared_expert_up_proj": "colwise"
-        # "shared_expert_down_proj": "rowwise"
-        # gating layer should be replicated as well
-        # "gate": "gather"
+    allreduce_strategy: SYMM_MEM
+  multi_stream_moe:
+    stage: compile
+    enabled: true
+  multi_stream_gemm:
+    stage: compile
+    enabled: true
+  gather_logits_before_lm_head:
+    enabled: true
diff --git a/examples/auto_deploy/model_registry/configs/qwen3.5_moe_400b.yaml b/examples/auto_deploy/model_registry/configs/qwen3.5_moe_400b.yaml
index 250ee830e..49ced4dbc 100644
--- a/examples/auto_deploy/model_registry/configs/qwen3.5_moe_400b.yaml
+++ b/examples/auto_deploy/model_registry/configs/qwen3.5_moe_400b.yaml
@@ -1,16 +1,17 @@
 runtime: trtllm
 compile_backend: torch-cudagraph
-max_seq_len: 2048
-max_num_tokens: 2048
-max_batch_size: 512
-cuda_graph_batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
+attn_backend: trtllm
+max_seq_len: 262144
+max_num_tokens: 8192
+max_batch_size: 32
+cuda_graph_batch_sizes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
 world_size: 8
 enable_chunked_prefill: true
 model_factory: AutoModelForCausalLM
 kv_cache_config:
-  enable_block_reuse: false
-  free_gpu_memory_fraction: 0.95
-  tokens_per_block: 64
+  enable_block_reuse: true
+  free_gpu_memory_fraction: 0.8
+  tokens_per_block: 32
 model_kwargs:
   torch_dtype: bfloat16
 transforms:
@@ -19,
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Multi-stream execution enables parallel execution of independent operations on the GPU. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## 3991aa9c72 - [https://nvbugs/5688388][fix] fix: Reducing num request in disagg test to speed up (#9598)

- **Date**: 2025-12-02
- **Author**: Patrice Castonguay
- **Categories**: Throughput/Latency

### Optimization Techniques

- General code optimization

### Applicable Conditions

- Disaggregated serving

### Changed Files

```
tests/integration/defs/disaggregated/test_disaggregated_single_gpu.py | 4 ++--
 1 file changed, 2 insertions(+), 2 deletions(-)
```

### Diff Preview

```diff
diff --git a/tests/integration/defs/disaggregated/test_disaggregated_single_gpu.py b/tests/integration/defs/disaggregated/test_disaggregated_single_gpu.py
index 570e499fb..d6b63d3ab 100644
--- a/tests/integration/defs/disaggregated/test_disaggregated_single_gpu.py
+++ b/tests/integration/defs/disaggregated/test_disaggregated_single_gpu.py
@@ -351,8 +351,8 @@ def test_disaggregated_llama_context_capacity(model, enable_cuda_graph,
             max_tokens = 25
 
             requests = []
-            # Send 256 requests to make sure the context worker is saturated
-            for _ in range(256):
+            # Send 32 requests to make sure the context worker is saturated
+            for _ in range(32):
                 requests.append(
                     (prompt, SamplingParams(max_tokens=1, ignore_eos=True),
                      DisaggregatedParams(request_type="context_only")))

```

### Analysis Summary

This commit applies performance optimization techniques to improve inference efficiency. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 3ac11a6180 - [#9152][fix] AutoDeploy fused_allreduce_residual_rmsnorm to support demollm mode (#9197)

- **Date**: 2025-11-18
- **Author**: Eran Geva
- **Categories**: Fusion

### Optimization Techniques

- Operator fusion
- Parallelism optimization
- PyTorch built-in optimized ops

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../_torch/auto_deploy/distributed/trtllm.py       | 45 +++++++++++++++++-----
 1 file changed, 36 insertions(+), 9 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/distributed/trtllm.py b/tensorrt_llm/_torch/auto_deploy/distributed/trtllm.py
index 7d19c9cd6..434cc1693 100644
--- a/tensorrt_llm/_torch/auto_deploy/distributed/trtllm.py
+++ b/tensorrt_llm/_torch/auto_deploy/distributed/trtllm.py
@@ -40,15 +40,42 @@ try:
     def fused_allreduce_residual_rmsnorm(
         tensor: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor, eps: float
     ) -> tuple[torch.Tensor, torch.Tensor]:
-        """Fusing allreduce, residual (add), and hf_rms_norm together."""
-        all_reduce_params = AllReduceParams(
-            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
-            bias=None,
-            residual=residual,
-            norm_weight=norm_weight,
-            eps=eps,
-        )
-        return trtllm_allreduce(tensor, ReduceOp.SUM, all_reduce_params=all_reduce_params)
+        """Fusing allreduce, residual (add), and hf_rms_norm together.
+
+        When TRT-LLM ops are available (MPI mode), uses the fused kernel.
+        Otherwise, falls back to separate operations using torch distributed.
+        """
+        # Only use TRT-LLM fused op when running with MPI
+        if is_trtllm_op_available():
+            all_reduce_params = AllReduceParams(
+                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
+                bias=None,
+                residual=residual,
+                norm_weight=norm_weight,
+                eps=eps,
+            )
+            return trtllm_allreduce(tensor, ReduceOp.SUM, all_reduce_params=all_reduce_params)
+        else:
+            # Fallback: unfused implementation using torch distributed
+            # This is used in demollm mode without MPI
+            from .common import all_reduce as torch_all_reduce
+
+            # 1. All-reduce the tensor
+            tensor_reduced = tensor.clone()
+            torch_all_reduce(tensor_reduced, op=ReduceOp.SUM)
+
+            # 2. Add residual
+            tensor_with_residual = tensor_reduced + residual
+
+            # 3. Apply RMSNorm using PyTorch's built-in function
+            norm_out = torch.nn.functional.rms_norm(
+                tensor_with_residual,
+                normalized_shape=(tensor_with_residual.size(-1),),
+                weight=norm_weight,
+                eps=eps,
+            )
+
+            return norm_out, tensor_with_residual
 
     @fused_allreduce_residual_rmsnorm.register_fake
     def fused_allreduce_residual_rmsnorm_fake(

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 3b7120d60e - DeepSeek R1 throughut optimization tech blog for Blackwell GPUs (#4791)

- **Date**: 2025-05-30
- **Author**: Tao Li @ NVIDIA
- **Categories**: General Performance

### Optimization Techniques

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
.gitattributes                                     |   1 +
 docs/source/blogs/media/tech_blog3_mla_absorb.png  |   3 +
 ...pSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md | 174 +++++++++++++++++++++
 3 files changed, 178 insertions(+)
```

### Diff Preview

```diff
diff --git a/.gitattributes b/.gitattributes
index e0030fb1e..de0b56a73 100644
--- a/.gitattributes
+++ b/.gitattributes
@@ -5,3 +5,4 @@
 *.xz filter=lfs diff=lfs merge=lfs -text
 triton_backend/tools/gpt/input_data.json filter=lfs diff=lfs merge=lfs -text
 *cubin.cpp filter=lfs diff=lfs merge=lfs -text
+docs/source/blogs/media/tech_blog3_mla_absorb.png filter=lfs diff=lfs merge=lfs -text
diff --git a/docs/source/blogs/media/tech_blog3_mla_absorb.png b/docs/source/blogs/media/tech_blog3_mla_absorb.png
new file mode 100644
index 000000000..1395badf7
--- /dev/null
+++ b/docs/source/blogs/media/tech_blog3_mla_absorb.png
@@ -0,0 +1,3 @@
+version https://git-lfs.github.com/spec/v1
+oid sha256:998eb460cb9ceced195ff2231d07278fefd0dc816ac1982311d7cc63384beb4a
+size 560643
diff --git a/docs/source/blogs/tech_blog/blog3_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md b/docs/source/blogs/tech_blog/blog3_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md
new file mode 100644
index 000000000..28f380e8e
--- /dev/null
+++ b/docs/source/blogs/tech_blog/blog3_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md
@@ -0,0 +1,174 @@
+# Optimizing DeepSeek R1 Throughput on NVIDIA Blackwell GPUs: A Deep Dive for Developers
+
+By NVIDIA TensorRT-LLM team
+## Table of Contents
+  - [Introduction](#introduction)
+  - [Precision strategy](#precision-strategy)
+  - [Parallel strategy](#parallel-strategy)
+    - [Weights absorb and MQA](#weights-absorb-and-mqa)
+    - [Data Parallel for Attention module (ADP)](#data-parallel-for-attention-module-adp)
+    - [Expert parallel for MOE (EP)](#expert-parallel-for-moe-ep)
+  - [MLA Layers Optimizations](#mla-layers-optimizations)
+  - [MoE Layers Optimizations](#moe-layers-optimizations)
+  - [Runtime Optimizations](#runtime-optimizations)
+  - [How to reproduce](#how-to-reproduce)
+  - [Future Works](#future-works)
+  - [Acknowledgment](#acknowledgment)
+
+## Introduction
+The open source DeepSeek R1 model's innovative architecture including the multi-head latent attention (MLA) and large sparse Mixture-of-Experts (MoE) significantly improved the inference efficiency of the LLM models. However, harnessing the full potential of such an innovative structure requires equally important hardware/software co-optimization. This post delves into the optimization strategies for DeepSeek R1 throughput oriented scenarios (TPS/GPU), developed by NVIDIA within TensorRT-LLM on NVIDIA's Blackwell B200 GPUs. We will explore the rationale behind each enhancement. [The other min-latency optimization blog](./blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.md) explained in detail how TensorRT-LLM optimizes the R1 performance to achieve the best of the TPS/USER.
+
+These optimizations have significantly boosted DeepSeek R1 throughput on Blackwell. Performance increased from approximately 2000 TPS/GPU in February to 4600 TPS/GPU on ISL/OSL 1K/2K dataset. The optimizations are ge
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## 3c5aec19c2 - [#5048][enhance] AutoDeploy: Optimize prepare_inputs (#6634)

- **Date**: 2025-08-10
- **Author**: Gal Hubara-Agam
- **Categories**: Host-side Optimization

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- KV cache optimization
- Batching optimization
- Pinned memory
- Speculative decoding
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
.../compile/backends/torch_cudagraph.py            |   2 +-
 .../auto_deploy/custom_ops/attention_interface.py  | 205 ++++++++++++++++-----
 .../_torch/auto_deploy/shim/ad_executor.py         |  35 ++--
 .../transformations/library/test_kv_cache.py       |   2 +-
 4 files changed, 175 insertions(+), 69 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/compile/backends/torch_cudagraph.py b/tensorrt_llm/_torch/auto_deploy/compile/backends/torch_cudagraph.py
index 0b309ae2b..c2081e00d 100644
--- a/tensorrt_llm/_torch/auto_deploy/compile/backends/torch_cudagraph.py
+++ b/tensorrt_llm/_torch/auto_deploy/compile/backends/torch_cudagraph.py
@@ -162,7 +162,7 @@ class CapturedGraph(nn.Module):
 
         # copy inputs to input buffers
         for i, input_tensor in enumerate(args_batched):
-            self._input_buffers[i][: input_tensor.shape[0]] = input_tensor
+            self._input_buffers[i][: input_tensor.shape[0]].copy_(input_tensor, non_blocking=True)
 
         # run forward pass via graph
         self.graphs[combined_shape].replay()
diff --git a/tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py b/tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py
index 13c91652b..d486d93b8 100644
--- a/tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py
+++ b/tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py
@@ -18,6 +18,8 @@ from torch._ops import OpOverloadPacket
 from torch.export import Dim
 from torch.fx import Node
 
+from tensorrt_llm._utils import nvtx_range
+
 
 @dataclass
 class CacheConfig:
@@ -87,11 +89,13 @@ class SequenceInfo:
     # Similarly, if a batch is composed of generate-only requests,
     # then the maximum number of sequences possible in the batch is min (max_batch_size, max_num_tokens).
     max_num_tokens: Optional[int] = None
+    # device is the device on which the sequence info is stored.
+    device: str = "cuda"
 
     ## [UPDATE WITH CARE] TENSOR FIELDS THAT WILL BE PASSED TO PREPARE_METADATA OP #################
     # input_ids MUST ALWAYS BE THE FIRST FIELD
-    input_ids: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 1, dtype=torch.int))
-    position_ids: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 1, dtype=torch.long))
+    input_ids: torch.Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.int))
+    position_ids: torch.Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.long))
 
     seq_len: torch.Tensor = field(default_factory=lambda: torch.ones(1, dtype=torch.int))
     input_pos: torch.Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.int))
@@ -110,24 +114,44 @@ class SequenceInfo:
         # NOTE (lucaslie): WAR to address issue when using flashinfer attention with
         # (max_batch_size, max_seq_len) input in trtllm runtime.
         # see https://github.com/NVIDIA/TensorRT-LLM/issues/4504
-        max_seq_len_adjusted = self.max_seq_len + 1
+        self.max_seq_len_adjusted = self.max_seq_len + 1
 
         if self.max_num_tokens is None or self.max_num_tokens < 1:
-            self.max_num_tokens = self.max_batch_size * max_seq_len_adjusted
+            self.max_num_tokens = self.max_batch_size * self.max_seq_len_adjusted
         # if the provided max_num_tokens is less th
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Pinned (page-locked) memory enables faster host-to-device data transfers. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 3d0e38e074 - [None][perf] AutoDeploy optimize _get_unique_value (#8822)

- **Date**: 2025-10-31
- **Author**: Suyog Gupta
- **Categories**: General Performance

### Optimization Techniques

- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../_torch/auto_deploy/custom_ops/attention_interface.py    | 13 +++++++++----
 1 file changed, 9 insertions(+), 4 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py b/tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py
index 775ef628d..02f7001cf 100644
--- a/tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py
+++ b/tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py
@@ -576,17 +576,22 @@ class SequenceInfo:
             else:
                 self._extra_args[name] = None
 
+    @nvtx_range("ad_get_unique_value")
     def _get_unique_value(self, occupied: Set[int], max_val: int) -> int:
         """Get un unoccupied value from the range indicated by max_val.
 
         In addition, this function performs a sanity check to ensure that no value in the occupied
         set is out of bounds.
         """
-        full_range = set(range(max_val))
-        free_values = full_range - occupied
-        out_of_range = occupied - full_range
+        # Validate without materializing the full range set
+        out_of_range = [v for v in occupied if v < 0 or v >= max_val]
         assert not out_of_range, f"Out of range values: {out_of_range}"
-        return free_values.pop() if free_values else 0
+
+        # Return the smallest free value; fall back to 0 if none
+        for candidate in range(max_val):
+            if candidate not in occupied:
+                return candidate
+        return 0
 
     @nvtx_range("ad_nest_sequences")
     def nest_sequences(

```

### Analysis Summary

Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 3dbb087292 - [TRTLLM-5188] fix: [AutoDeploy] update output shape of prepare_fused_mha_metadata_fake (#4199)

- **Date**: 2025-05-12
- **Author**: Fridah-nv
- **Categories**: Fusion

### Optimization Techniques

- Operator fusion
- KV cache optimization
- Triton kernel
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../_torch/auto_deploy/custom_ops/triton_attention.py         | 11 +++++++----
 1 file changed, 7 insertions(+), 4 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/custom_ops/triton_attention.py b/tensorrt_llm/_torch/auto_deploy/custom_ops/triton_attention.py
index 1773e16f7..817068dd6 100644
--- a/tensorrt_llm/_torch/auto_deploy/custom_ops/triton_attention.py
+++ b/tensorrt_llm/_torch/auto_deploy/custom_ops/triton_attention.py
@@ -272,15 +272,18 @@ def prepare_fused_mha_metadata(
     )
 
 
+# TODO: Move the truncation of inputs out of this custom op
+# SequenceInfo._get_sanitized_num_sequences could break in fake mode
 @prepare_fused_mha_metadata.register_fake
 def prepare_fused_mha_metadata_fake(
     input_ids, position_ids, seq_len, input_pos, cache_loc, pages_per_seq, page_size
 ):
+    num_seq = SequenceInfo._get_sanitized_num_sequences(input_ids, seq_len)
     return (
-        torch.empty_like(seq_len),
-        torch.empty_like(input_pos),
-        torch.empty_like(cache_loc),
-        torch.empty_like(seq_len),
+        torch.empty_like(seq_len[:num_seq]),
+        torch.empty_like(input_pos[:num_seq]),
+        torch.empty_like(cache_loc[:num_seq]),
+        torch.empty_like(seq_len[:num_seq]),
     )
 
 

```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 3ddc9d2b48 - [https://nvbugs/5729697][fix] MNNVL Allreduce: use CUDA runtime instead of Macro to get SM version. (#10062)

- **Date**: 2025-12-23
- **Author**: Shiyu Li
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- Async/stream-based execution
- Parallelism optimization
- Reduce synchronization overhead
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../communicationKernels/mnnvlAllreduceKernels.cu  | 124 ++++++++++-----------
 .../_torch/multi_gpu/test_mnnvl_allreduce.py       |   2 +-
 2 files changed, 63 insertions(+), 63 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/communicationKernels/mnnvlAllreduceKernels.cu b/cpp/tensorrt_llm/kernels/communicationKernels/mnnvlAllreduceKernels.cu
index 47d4cf373..8bb09476b 100644
--- a/cpp/tensorrt_llm/kernels/communicationKernels/mnnvlAllreduceKernels.cu
+++ b/cpp/tensorrt_llm/kernels/communicationKernels/mnnvlAllreduceKernels.cu
@@ -230,59 +230,62 @@ inline __device__ __host__ T divUp(T m, T n)
 // Return (block_size, cluster_size, loads_per_thread)
 std::tuple<int, int, int> adjustGridConfig(int numTokens, int dim, int eltsPerThread)
 {
-    // Start with preferred block_size and cluster_size
-#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
-    int clusterSize = 8;
-#else
-    int clusterSize = 1;
-#endif
+    static int SM = tensorrt_llm::common::getSMVersion();
+
+    int clusterSize = SM >= 90 ? 8 : 1;
     int blockSize = 128;
     // ========================== Adjust the grid configuration ==========================
     int threadsNeeded = divUp(dim, eltsPerThread);
     int loadsPerThread = 1;
 
     blockSize = divUp(threadsNeeded, clusterSize);
-#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
-    while (threadsNeeded % clusterSize != 0 && clusterSize > 1)
-    {
-        clusterSize /= 2;
-    }
-    blockSize = divUp(threadsNeeded, clusterSize);
-    while (blockSize < 128 && clusterSize >= 2)
-    {
-        blockSize *= 2;
-        clusterSize /= 2;
-    }
-    int smCount = getMultiProcessorCount();
-    while (numTokens * clusterSize > smCount && clusterSize > 1 && blockSize <= 512)
+    if (clusterSize > 1)
     {
-        blockSize *= 2;
-        clusterSize /= 2;
+        while (threadsNeeded % clusterSize != 0 && clusterSize > 1)
+        {
+            clusterSize /= 2;
+        }
+        blockSize = divUp(threadsNeeded, clusterSize);
+        while (blockSize < 128 && clusterSize >= 2)
+        {
+            blockSize *= 2;
+            clusterSize /= 2;
+        }
+        int smCount = getMultiProcessorCount();
+        while (numTokens * clusterSize > smCount && clusterSize > 1 && blockSize <= 512)
+        {
+            blockSize *= 2;
+            clusterSize /= 2;
+        }
     }
-#endif
 
     // Trying to scale up use multiple loads or CGA
     while (blockSize > 1024)
     {
-#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
-        if (clusterSize < 8)
+        // Scale up with CGA if supported
+        if (SM >= 90)
         {
-            clusterSize = clusterSize << 1;
-        }
-        else
-        {
-            break;
-        }
-#else
-        if (loadsPerThread < 8)
-        {
-            loadsPerThread += 1;
+            if (clusterSize < 8)
+            {
+                clusterSize = clusterSize << 1;
+            }
+            else
+            {
+                break;
+            }
         }
         else
         {
-            break;
+
+            if (loadsPerThread < 8)
+            {
+                loadsPerThread += 1;
+            }
+            
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 3fd5fafb58 - [https://nvbugs/5911143][fix] add async worker to MTP/Eagle3 sampler,… (#11573)

- **Date**: 2026-02-26
- **Author**: dhansen-nvidia
- **Categories**: Parallelism/Async

### Optimization Techniques

- Torch compilation/JIT optimization
- Custom CUDA kernel
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Pinned memory
- Triton kernel
- PyTorch built-in optimized ops
- Speculative decoding
- Attention mechanism optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU
- Prefill phase
- Decode/generation phase
- Disaggregated serving

### Changed Files

```
.pre-commit-config.yaml                            |   5 +
 scripts/check_pinned_memory_usage.py               |  78 ++++++++++++++
 tensorrt_llm/_torch/attention_backend/interface.py |   5 +-
 .../_torch/attention_backend/sparse/dsa.py         |  32 +++---
 .../_torch/attention_backend/sparse/rocket.py      |   6 +-
 tensorrt_llm/_torch/attention_backend/trtllm.py    |  37 +++----
 .../custom_ops/attention/trtllm_attention.py       |  18 ++--
 .../auto_deploy/custom_ops/attention_interface.py  |   8 +-
 tensorrt_llm/_torch/models/modeling_clip.py        |   4 +-
 tensorrt_llm/_torch/models/modeling_qwen2vl.py     |   6 +-
 tensorrt_llm/_torch/models/modeling_qwen3vl.py     |   4 +-
 tensorrt_llm/_torch/models/modeling_radio.py       |   5 +-
 tensorrt_llm/_torch/models/modeling_siglip.py      |   4 +-
 .../_torch/modules/mamba/mamba2_metadata.py        |   5 +-
 .../_torch/peft/lora/cuda_graph_lora_params.py     |  20 ++--
 tensorrt_llm/_torch/pyexecutor/guided_decoder.py   |  10 +-
 tensorrt_llm/_torch/pyexecutor/llm_request.py      |   5 +-
 .../_torch/pyexecutor/mamba_cache_manager.py       |  11 +-
 tensorrt_llm/_torch/pyexecutor/model_engine.py     | 120 ++++++++++++---------
 tensorrt_llm/_torch/pyexecutor/py_executor.py      |   1 +
 tensorrt_llm/_torch/pyexecutor/resource_manager.py |  30 +++---
 tensorrt_llm/_torch/pyexecutor/sampler.py          |  74 ++++++++-----
 tensorrt_llm/_torch/pyexecutor/sampling_utils.py   |   7 +-
 .../_torch/pyexecutor/sampling_utils_flashinfer.py |   3 +-
 tensorrt_llm/_torch/speculative/eagle3.py          |  11 +-
 tensorrt_llm/_torch/speculative/interface.py       |  14 ++-
 tensorrt_llm/_torch/speculative/model_drafter.py   |   6 +-
 tensorrt_llm/_torch/speculative/mtp.py             |  32 +++---
 .../_torch/speculative/spec_tree_manager.py        |   8 +-
 tensorrt_llm/_utils.py                             |  50 +++++++--
 tensorrt_llm/inputs/multimodal.py                  |   5 +-
 tensorrt_llm/runtime/model_runner_cpp.py           |  10 +-
 tensorrt_llm/runtime/multimodal_model_runner.py    |  11 +-
 .../tools/layer_wise_benchmarks/calibrator.py      |   3 +-
 .../multimodal/multimodal_encoders/1/model.py      |  44 ++++----
 35 files changed, 441 insertions(+), 251 deletions(-)
```

### Diff Preview

```diff
diff --git a/.pre-commit-config.yaml b/.pre-commit-config.yaml
index e5932f6d3..07920ca99 100644
--- a/.pre-commit-config.yaml
+++ b/.pre-commit-config.yaml
@@ -1473,3 +1473,8 @@ repos:
         entry: ./scripts/dco_check.py
         language: script
         stages: [commit-msg]
+    -   id: pinned memory policy
+        name: Disallow raw pinned-memory APIs in runtime code
+        entry: ./scripts/check_pinned_memory_usage.py
+        language: script
+        files: ^(tensorrt_llm|triton_backend)/.*\.py$
diff --git a/scripts/check_pinned_memory_usage.py b/scripts/check_pinned_memory_usage.py
new file mode 100755
index 000000000..ef31f885e
--- /dev/null
+++ b/scripts/check_pinned_memory_usage.py
@@ -0,0 +1,78 @@
+#!/usr/bin/env python3
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+
+import ast
+import pathlib
+import sys
+
+
+class PinnedMemoryUsageChecker(ast.NodeVisitor):
+    def __init__(self, *, allow_direct_pin_memory: bool) -> None:
+        self.allow_direct_pin_memory = allow_direct_pin_memory
+        self.violations: list[tuple[int, str]] = []
+
+    def visit_Call(self, node: ast.Call) -> None:
+        if isinstance(node.func, ast.Attribute) and node.func.attr == "pin_memory":
+            if not self.allow_direct_pin_memory:
+                self.violations.append(
+                    (
+                        node.lineno,
+                        "Use `maybe_pin_memory(tensor)` instead of direct `.pin_memory()`.",
+                    )
+                )
+
+        for keyword in node.keywords:
+            if (
+                keyword.arg == "pin_memory"
+                and isinstance(keyword.value, ast.Constant)
+                and keyword.value.value is True
+            ):
+                self.violations.append(
+                    (
+                        node.lineno,
+                        "Use `pin_memory=prefer_pinned()` instead of `pin_memory=True`.",
+                    )
+                )
+
+        self.generic_visit(node)
+
+
+def _check_file(path: pathlib.Path) -> list[tuple[int, str]]:
+    try:
+        source = path.read_text(encoding="utf-8")
+    except OSError as exc:
+        return [(0, f"Failed to read file: {exc}")]
+
+    try:
+        tree = ast.parse(source, filename=str(path))
+    except SyntaxError as exc:
+        return [(exc.lineno or 0, f"Failed to parse file: {exc.msg}")]
+
+    allow_direct_pin_memory = path.as_posix().endswith("tensorrt_llm/_utils.py")
+    checker = PinnedMemoryUsageChecker(allow_direct_pin_memory=allow_direct_pin_memory)
+    checker.visit(tree)
+    return checker.violations
+
+
+def main(argv: list[str]) -> int:
+    if len(argv) <= 1:
+        return 0
+
+    has_violations = False
+    for file_arg in argv[1:]:
+        path = pathlib.Path(file_arg)
+        violations = _check_file(path)
+        for lineno, message in violations:
+            has_violations = Tr
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Pinned (page-locked) memory enables faster host-to-device data transfers. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## 3ff4f503ad - [None][opt] ADP schedule balance optimization (#6061)

- **Date**: 2025-08-06
- **Author**: yunruis
- **Categories**: General Performance

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
- NVIDIA Hopper (SM90) GPU
- Decode/generation phase
- Disaggregated serving

### Changed Files

```
examples/llm-api/quickstart_advanced.py            | 23 ++++++--
 .../_torch/auto_deploy/shim/ad_executor.py         |  3 +
 tensorrt_llm/_torch/pyexecutor/config.py           |  4 ++
 tensorrt_llm/_torch/pyexecutor/py_executor.py      | 65 +++++++++++++++++++++-
 tensorrt_llm/llmapi/__init__.py                    | 20 ++++---
 tensorrt_llm/llmapi/llm_args.py                    | 57 ++++++++++++++++++-
 tests/integration/defs/test_e2e.py                 | 39 +++++++++++++
 .../integration/test_lists/test-db/l0_dgx_h100.yml |  1 +
 tests/unittest/api_stability/references/llm.yaml   |  4 ++
 9 files changed, 200 insertions(+), 16 deletions(-)
```

### Diff Preview

```diff
diff --git a/examples/llm-api/quickstart_advanced.py b/examples/llm-api/quickstart_advanced.py
index 9e9287812..13740f3d3 100644
--- a/examples/llm-api/quickstart_advanced.py
+++ b/examples/llm-api/quickstart_advanced.py
@@ -1,10 +1,11 @@
 import argparse
 
 from tensorrt_llm import LLM, SamplingParams
-from tensorrt_llm.llmapi import (AutoDecodingConfig, CudaGraphConfig,
-                                 DraftTargetDecodingConfig, EagleDecodingConfig,
-                                 KvCacheConfig, MoeConfig, MTPDecodingConfig,
-                                 NGramDecodingConfig, TorchCompileConfig)
+from tensorrt_llm.llmapi import (AttentionDpConfig, AutoDecodingConfig,
+                                 CudaGraphConfig, DraftTargetDecodingConfig,
+                                 EagleDecodingConfig, KvCacheConfig, MoeConfig,
+                                 MTPDecodingConfig, NGramDecodingConfig,
+                                 TorchCompileConfig)
 
 example_prompts = [
     "Hello, my name is",
@@ -57,6 +58,13 @@ def add_llm_args(parser):
     parser.add_argument('--enable_attention_dp',
                         default=False,
                         action='store_true')
+    parser.add_argument('--attention_dp_enable_balance',
+                        default=False,
+                        action='store_true')
+    parser.add_argument('--attention_dp_time_out_iters', type=int, default=0)
+    parser.add_argument('--attention_dp_batching_wait_iters',
+                        type=int,
+                        default=0)
     parser.add_argument('--enable_trtllm_sampler',
                         default=False,
                         action='store_true')
@@ -196,6 +204,12 @@ def setup_llm(args, **kwargs):
         enable_padding=args.cuda_graph_padding_enabled,
     ) if args.use_cuda_graph else None
 
+    attention_dp_config = AttentionDpConfig(
+        enable_balance=args.attention_dp_enable_balance,
+        timeout_iters=args.attention_dp_time_out_iters,
+        batching_wait_iters=args.attention_dp_batching_wait_iters,
+    )
+
     llm = LLM(
         model=args.model_dir,
         backend='pytorch',
@@ -218,6 +232,7 @@ def setup_llm(args, **kwargs):
         max_batch_size=args.max_batch_size,
         max_num_tokens=args.max_num_tokens,
         enable_attention_dp=args.enable_attention_dp,
+        attention_dp_config=attention_dp_config,
         tensor_parallel_size=args.tp_size,
         pipeline_parallel_size=args.pp_size,
         moe_expert_parallel_size=args.moe_ep_size,
diff --git a/tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py b/tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py
index 7f759d679..ff0fb204f 100644
--- a/tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py
+++ b/tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py
@@ -132,6 +132,9 @@ class ADEngine(ModelEngine):
         self.pytorch_backend_config.enable_iter_perf_stats = False
         self.pytorch_backend_config.enable_iter_req_stats = False
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 4018806742 - feat: large-scale EP(part 3 - refactor: FusedMoe for redundant expert) (#4495)

- **Date**: 2025-05-21
- **Author**: dongxuy04
- **Categories**: Fusion

### Optimization Techniques

- Operator fusion
- FP8 quantization
- Parallelism optimization
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/modules/fused_moe.py | 106 +++++++++++++++++++------------
 1 file changed, 64 insertions(+), 42 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fused_moe.py b/tensorrt_llm/_torch/modules/fused_moe.py
index 5b2b89d89..8e170d1ba 100755
--- a/tensorrt_llm/_torch/modules/fused_moe.py
+++ b/tensorrt_llm/_torch/modules/fused_moe.py
@@ -272,6 +272,10 @@ class FusedMoE(nn.Module):
 
     In min-latency mode, setting `reduce_results=False` disables the AllReduce in the FusedMoE module, so any necessary AllReduce operations must be added explicitly in the model definition.
     AttentionDP should be turned off for min-latency mode.
+
+    When we have redundant expert, we have more weight slots than `num_experts`, in that case, we separate the concepts of expert and slot.
+    Expert is the concept from model's perspective while slot is the concept from model engine's perspective.
+    There should be at lease `num_experts` slots in the model engine. More than that is OK, in that case, some experts may have multiple replicas.
     """
 
     def __init__(
@@ -326,11 +330,25 @@ class FusedMoE(nn.Module):
 
         self.intermediate_size_per_partition = intermediate_size // self.tp_size
 
-        self.expert_size_per_partition = num_experts // self.ep_size
-        self.expert_start = self.ep_rank * self.expert_size_per_partition
-        self.expert_end = min(
-            self.expert_start + self.expert_size_per_partition,
-            self.num_experts)
+        # self.expert_slots_per_partition will be replaced with real slots_per_partition to enable redundant expert slots
+        self.expert_slots_per_partition = num_experts // self.ep_size
+        assert self.expert_slots_per_partition * self.ep_size >= num_experts, "total slots should be at lease num_experts"
+        if self.smart_router:
+            assert self.expert_slots_per_partition == num_experts // self.ep_size,\
+                "Smart router should not have redundant slots"
+        self.num_slots = self.expert_slots_per_partition * self.ep_size
+        # Here the meaning of expert_size_per_partition is the number of expert slots that each rank has.
+        self.expert_size_per_partition = self.expert_slots_per_partition
+        self.slot_start = self.ep_rank * self.expert_size_per_partition
+        self.slot_end = self.slot_start + self.expert_size_per_partition
+
+        self.initial_global_assignments = [
+            (ep_rank * self.num_experts // self.ep_size + local_slot_id) %
+            self.num_experts for ep_rank in range(self.ep_size)
+            for local_slot_id in range(self.expert_slots_per_partition)
+        ]
+        self.initial_local_expert_ids = self.initial_global_assignments[
+            self.slot_start:self.slot_end]
 
         max_num_tokens = model_config.max_num_tokens
         # The maximum number of tokens in MoE are multiplied by DP size when attention DP is enabled
@@ -354,7 +372,7 @@ class FusedMoE(nn.Module):
         # around 16k tokens per expert, which is well into the compute bound domain.
         self.tune_max_num_tokens = min(
            
```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 411fa9ff87 - [TRTLLM-10030][perf] pin host memory and batch sampler setup in beam search (#11390)

- **Date**: 2026-02-10
- **Author**: mpikulski
- **Categories**: Memory Optimization

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- KV cache optimization
- Pinned memory
- Speculative decoding

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/sampler.py | 101 +++++++++++++++++++++---------
 1 file changed, 73 insertions(+), 28 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index da33c6cb3..27e2b99b5 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -1521,12 +1521,10 @@ class TorchSampler(Sampler[SampleStateTorch], AsyncWorkerMixin):
         Args:
             requests: list[LlmRequest]. The requests to setup the sampler step for
         """
-        if self._use_beam_search:
-            self._prepare_beam_search(scheduled_requests.all_requests())
-
         seq_slots: list[int] = []
         max_lens: list[int] = []
         end_ids: list[int] = []
+        prompt_lens: list[int] = []
         for request in scheduled_requests.context_requests:
             if self._is_new_request(request):
                 assert request.py_seq_slot is not None
@@ -1535,47 +1533,89 @@ class TorchSampler(Sampler[SampleStateTorch], AsyncWorkerMixin):
                     min(self.max_seq_len, request.orig_prompt_len + request.py_max_new_tokens)
                 )
                 end_ids.append(request.py_end_id if request.py_end_id is not None else -1)
+
+                if self._use_beam_search:
+                    if request.py_return_log_probs and request.py_num_logprobs > 1:
+                        raise ValueError("Beam search does not support multiple logprobs")
+                    prompt_lens.append(request.py_prompt_len)
+
         if len(seq_slots) > 0:
             full_list = [seq_slots, max_lens, end_ids]
+            if self._use_beam_search:
+                full_list.append(prompt_lens)
             # perform only a single copy
-            full_list_tensor = torch.tensor(full_list, device="cpu", dtype=torch.int32).to(
-                device="cuda", non_blocking=True
-            )
+            full_list_tensor = torch.tensor(
+                full_list, device="cpu", dtype=torch.int32, pin_memory=True
+            ).to(device="cuda", non_blocking=True)
             seq_slots_tensor = full_list_tensor[0]
             max_lens_tensor = full_list_tensor[1]
             end_ids_tensor = full_list_tensor[2]
             self.store.max_lengths_tensor[seq_slots_tensor] = max_lens_tensor
             self.store.end_ids[seq_slots_tensor] = end_ids_tensor
 
+            if self._use_beam_search:
+                prompt_lens_tensor = full_list_tensor[3]
+                self._prepare_beam_search(
+                    seq_slots=seq_slots_tensor,
+                    prompt_lens=prompt_lens_tensor,
+                )
+
     def _prepare_beam_search(
         self,
-        requests: list[LlmRequest],
+        seq_slots: torch.Tensor,
+        prompt_lens: torch.Tensor,
     ):
         """Prepare the beam search buffers for the requests
 
         If the last context chunk is being processed,
         initialize/reset the buffers for the request
         """
-        for request in requests:
-            if self._is_new_request(request):
-                if request.py_r
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Pinned (page-locked) memory enables faster host-to-device data transfers. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 421eb9e39c - [None][feat] Optimize NemotronH model with elementwise and nvfp4 fusion (#11273)

- **Date**: 2026-02-12
- **Author**: Wanli Jiang
- **Categories**: Fusion

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
.../kernels/causalConv1d/causalConv1d.cu           |  52 +++-
 cpp/tensorrt_llm/kernels/fusedActivationQuant.cu   | 187 ++++++++++++
 cpp/tensorrt_llm/kernels/fusedActivationQuant.h    |  33 ++
 .../fusedLayernormKernels/layernorm_param.h        |   1 +
 .../low_latency_layernorm.cuh                      |   4 +-
 .../kernels/fusedLayernormKernels/ws_layernorm.cuh |  12 +-
 .../kernels/fusedLayernormKernels/ws_layernorm.h   |   2 +-
 .../ws_layernorm_fp4_traits.cu                     |  49 ++-
 cpp/tensorrt_llm/thop/CMakeLists.txt               |   1 +
 cpp/tensorrt_llm/thop/fusedActivationQuant.cpp     |  94 ++++++
 cpp/tensorrt_llm/thop/fusedAddRMSNormQuant.cpp     |  25 +-
 tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py   |  24 +-
 tensorrt_llm/_torch/models/modeling_nemotron_h.py  | 142 ++++++---
 .../_torch/modules/mamba/fuse_elementwise_ops.py   | 176 +++++++++++
 .../_torch/modules/mamba/mamba2_metadata.py        | 143 ++++++---
 tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py  |  46 +--
 .../_torch/modules/mamba/ssd_chunk_scan.py         | 131 +++++++-
 .../_torch/modules/mamba/ssd_chunk_state.py        | 126 +++++---
 tensorrt_llm/_torch/modules/mlp.py                 |  68 ++++-
 tensorrt_llm/_torch/modules/rms_norm.py            |  27 +-
 .../_torch/pyexecutor/mamba_cache_manager.py       |   4 +-
 tensorrt_llm/tools/layer_wise_benchmarks/runner.py |   3 +-
 .../_torch/modules/mamba/test_causal_conv1d.py     | 246 +++++++++++++++
 .../modules/mamba/test_fuse_elementwise_ops.py     | 113 +++++++
 .../_torch/modules/mamba/test_mamba2_metadata.py   | 133 ++++++++
 .../_torch/modules/test_fused_activation_quant.py  | 223 ++++++++++++++
 .../modules/test_fused_add_rms_norm_quant.py       | 336 +++++++++++++++++++++
 27 files changed, 2195 insertions(+), 206 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/causalConv1d/causalConv1d.cu b/cpp/tensorrt_llm/kernels/causalConv1d/causalConv1d.cu
index 8ec6bbbf8..a5f22858a 100644
--- a/cpp/tensorrt_llm/kernels/causalConv1d/causalConv1d.cu
+++ b/cpp/tensorrt_llm/kernels/causalConv1d/causalConv1d.cu
@@ -43,6 +43,8 @@ struct Causal_conv1d_fwd_kernel_traits
     static_assert(kWidth <= kNElts);
     static constexpr bool kIsVecLoad = kIsVecLoad_;
     using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
+    static_assert(kNThreads_ % 32 == 0, "kNThreads must be a multiple of 32 for warp shuffle");
+    static_assert(sizeof(vec_t) == 16, "vec_t must be 16 bytes for warp shuffle optimization");
     using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNElts, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
     using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, 1, cub::BLOCK_LOAD_DIRECT>;
     using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNElts, cub::BLOCK_STORE_WARP_TRANSPOSE>;
@@ -123,7 +125,7 @@ __global__ __launch_bounds__(Ktraits::kNThreads) void causal_conv1d_fwd_kernel(C
 #pragma unroll
     for (int i = 0; i < kWidth; ++i)
     {
-        weight_vals[i] = float(weight[i * params.weight_width_stride]);
+        weight_vals[i] = float(__ldg(&weight[i * params.weight_width_stride]));
     }
 
     constexpr int kChunkSize = kNThreads * kNElts;
@@ -144,20 +146,41 @@ __global__ __launch_bounds__(Ktraits::kNThreads) void causal_conv1d_fwd_kernel(C
                 x, *reinterpret_cast<input_t(*)[kNElts]>(&x_vals_load[kNElts]), seqlen - chunk * kChunkSize);
         }
         x += kChunkSize;
+
+        int const lane_id = tidx & 31;
+        vec_t high_val = reinterpret_cast<vec_t*>(x_vals_load)[1];
+
         __syncthreads();
         // Thread kNThreads - 1 don't write yet, so that thread 0 can read
         // the last elements of the previous chunk.
         if (tidx < kNThreads - 1)
         {
-            smem_exchange[tidx] = reinterpret_cast<vec_t*>(x_vals_load)[1];
+            smem_exchange[tidx] = high_val;
         }
         __syncthreads();
-        reinterpret_cast<vec_t*>(x_vals_load)[0] = smem_exchange[tidx > 0 ? tidx - 1 : kNThreads - 1];
+
+        // Get neighbor data: use warp shuffle for most threads, shared memory for warp boundaries
+        vec_t neighbor;
+        uint32_t* high_val_p = reinterpret_cast<uint32_t*>(&high_val);
+        uint32_t* nbr_p = reinterpret_cast<uint32_t*>(&neighbor);
+        nbr_p[0] = __shfl_up_sync(0xFFFFFFFF, high_val_p[0], 1);
+        nbr_p[1] = __shfl_up_sync(0xFFFFFFFF, high_val_p[1], 1);
+        nbr_p[2] = __shfl_up_sync(0xFFFFFFFF, high_val_p[2], 1);
+        nbr_p[3] = __shfl_up_sync(0xFFFFFFFF, high_val_p[3], 1);
+
+        // Lane 0 must use shared memory to handle the cross-warp boundary.
+        // thread 0 uses the last element of the previous chunk.
+        if (lane_id == 0)
+        {
+            neighbor = smem_exchange[tidx > 0 ? tidx - 1 : kNThreads - 1];
+        }
+        reinterpret_
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## 448bb1a44f - [TRTLLM-9431][perf] Enable multistream for Linear Attention in Qwen3-… (#9696)

- **Date**: 2025-12-08
- **Author**: Guoming Zhang
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- Parallelism optimization
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/models/modeling_qwen3_next.py | 33 +++++++++++++++++------
 1 file changed, 25 insertions(+), 8 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_qwen3_next.py b/tensorrt_llm/_torch/models/modeling_qwen3_next.py
index e8b2021fb..8061be539 100644
--- a/tensorrt_llm/_torch/models/modeling_qwen3_next.py
+++ b/tensorrt_llm/_torch/models/modeling_qwen3_next.py
@@ -647,11 +647,10 @@ def fused_gdn_gating(
 
 class Qwen3NextGatedDeltaNet(nn.Module):
 
-    def __init__(
-        self,
-        model_config: ModelConfig[Qwen3NextConfig],
-        layer_idx: Optional[int] = None,
-    ):
+    def __init__(self,
+                 model_config: ModelConfig[Qwen3NextConfig],
+                 aux_stream: torch.cuda.Stream,
+                 layer_idx: Optional[int] = None):
         super().__init__()
         config = model_config.pretrained_config
         self.model_config = model_config
@@ -778,6 +777,12 @@ class Qwen3NextGatedDeltaNet(nn.Module):
             force_dynamic_quantization=model_config.force_dynamic_quantization,
             use_cute_dsl_blockscaling_mm=False)
 
+        self.event_dict = {
+            key: torch.cuda.Event()
+            for key in [EventType.Main, EventType.Attention]
+        }
+        self.aux_stream = aux_stream
+
     def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
         """
         Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
@@ -1032,8 +1037,19 @@ class Qwen3NextGatedDeltaNet(nn.Module):
             ssm_states[state_indices_p] = 0
             # conv_states[state_indices_p] = 0 # not necessary
 
-        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
-        projected_states_ba = self.in_proj_ba(hidden_states)
+        def _compute_projected_states_qkvz():
+            return self.in_proj_qkvz(hidden_states)
+
+        def _compute_projected_states_ba():
+            return self.in_proj_ba(hidden_states)
+
+        projected_states_qkvz, projected_states_ba = maybe_execute_in_parallel(
+            _compute_projected_states_qkvz,
+            _compute_projected_states_ba,
+            self.event_dict[EventType.Main],
+            self.event_dict[EventType.Attention],
+            self.aux_stream,
+        )
 
         # Use fused kernel when possible to avoid elementwise ops
         if self.num_v_heads // self.num_k_heads in [1, 2,
@@ -1098,7 +1114,8 @@ class Qwen3NextLinearDecoderLayer(nn.Module):
         super().__init__()
         self.model_config = model_config
         config = model_config.pretrained_config
-        self.linear_attn = Qwen3NextGatedDeltaNet(model_config, layer_idx)
+        self.linear_attn = Qwen3NextGatedDeltaNet(model_config, aux_stream,
+                                                  layer_idx)
 
         self.mapping = model_config.mapping
         self.enable_attention_dp = self.mapping.enable_attention_dp

```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 4632a8642d - [None][doc] blog: Optimizing DeepSeek-V3.2 on NVIDIA Blackwell GPUs (#10565)

- **Date**: 2026-01-09
- **Author**: Fanrong Li
- **Categories**: General Performance

### Optimization Techniques

- Torch compilation/JIT optimization
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
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- Prefill phase
- Disaggregated serving

### Changed Files

```
.../blogs/media/tech_blog15_ds32_wide_ep.png       | Bin 0 -> 225411 bytes
 .../blogs/media/tech_blog15_dsa_architecture.png   | Bin 0 -> 524742 bytes
 .../blogs/media/tech_blog15_indexer_topk.png       | Bin 0 -> 336605 bytes
 .../blogs/media/tech_blog15_radix_select_topk.png  | Bin 0 -> 611882 bytes
 ...mizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md | 423 +++++++++++++++++++++
 5 files changed, 423 insertions(+)
```

### Diff Preview

```diff
diff --git a/docs/source/blogs/media/tech_blog15_ds32_wide_ep.png b/docs/source/blogs/media/tech_blog15_ds32_wide_ep.png
new file mode 100644
index 000000000..f8f6f4592
Binary files /dev/null and b/docs/source/blogs/media/tech_blog15_ds32_wide_ep.png differ
diff --git a/docs/source/blogs/media/tech_blog15_dsa_architecture.png b/docs/source/blogs/media/tech_blog15_dsa_architecture.png
new file mode 100644
index 000000000..cf906c137
Binary files /dev/null and b/docs/source/blogs/media/tech_blog15_dsa_architecture.png differ
diff --git a/docs/source/blogs/media/tech_blog15_indexer_topk.png b/docs/source/blogs/media/tech_blog15_indexer_topk.png
new file mode 100644
index 000000000..0883baa6d
Binary files /dev/null and b/docs/source/blogs/media/tech_blog15_indexer_topk.png differ
diff --git a/docs/source/blogs/media/tech_blog15_radix_select_topk.png b/docs/source/blogs/media/tech_blog15_radix_select_topk.png
new file mode 100644
index 000000000..69927b4d5
Binary files /dev/null and b/docs/source/blogs/media/tech_blog15_radix_select_topk.png differ
diff --git a/docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md b/docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md
new file mode 100644
index 000000000..0d95d5d6e
--- /dev/null
+++ b/docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md
@@ -0,0 +1,423 @@
+# Optimizing DeepSeek-V3.2 on NVIDIA Blackwell GPUs
+By NVIDIA TensorRT LLM team
+
+## Table of Contents
+- [Optimizing DeepSeek-V3.2 on NVIDIA Blackwell GPUs](#optimizing-deepseek-v32-on-nvidia-blackwell-gpus)
+    - [Table of Contents](#table-of-contents)
+    - [Introduction](#introduction)
+    - [DeepSeek Sparse Attention (DSA)](#deepseek-sparse-attention-dsa)
+    - [Precision Strategy](#precision-strategy)
+    - [Parallel Strategy](#parallel-strategy)
+    - [Key Features](#key-features)
+        - [MTP](#mtp)
+        - [Disaggregated Serving](#disaggregated-serving)
+        - [Chunked Prefill and KV Cache Reuse](#chunked-prefill-and-kv-cache-reuse)
+        - [Wide Expert Parallelism (Wide-EP)](#wide-expert-parallelism-wide-ep)
+		- [Chat Template and Tool Parser](#chat-template-and-tool-parser)
+    - [Key Optimizations](#key-optimizations)
+        - [Kernel Optimizations](#kernel-optimizations)
+            - [Sparse MLA Kernel](#sparse-mla-kernel)
+            - [Indexer Top-K Kernel](#indexer-top-k-kernel)
+            - [DeepGEMM MQA Kernel](#deepgemm-mqa-kernel)
+            - [Kernel Fusion](#kernel-fusion)
+        - [System Optimizations](#system-optimizations)
+            - [Multi-steams](#multi-steams)
+            - [A Fast Path for Short Sequences](#a-fast-path-for-short-sequences)
+    - [How to Reproduce](#how-to-reproduce)
+        - [Accuracy Evaluation](#accuracy-evaluation)
+        - [Benchmark on B200](#benchmark-on-b200)
+            - [Min-latency](#min-latency)
+            - [Max-throughput](#max-throughput)
+     
```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## 46e4af5688 - [TRTLLM-9831][perf] Enable 2CTA with autotune for CuteDSL MoE and Grouped GEMM optimizations (#10201)

- **Date**: 2025-12-25
- **Author**: ZhichenJiang
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
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU

### Changed Files

```
tensorrt_llm/_torch/autotuner.py                   |  31 +-
 .../_torch/custom_ops/cute_dsl_custom_ops.py       | 317 ++------
 ...contiguous_gather_grouped_gemm_swiglu_fusion.py |  35 +-
 .../blockscaled_contiguous_grouped_gemm.py         | 251 ++++---
 ...aled_contiguous_grouped_gemm_finalize_fusion.py |  47 +-
 ...scaled_contiguous_grouped_gemm_swiglu_fusion.py |  50 +-
 .../_torch/modules/fused_moe/fused_moe_cute_dsl.py | 206 ++++-
 tests/integration/test_lists/test-db/l0_b200.yml   |   4 +-
 .../integration/test_lists/test-db/l0_dgx_b200.yml |   2 +-
 tests/integration/test_lists/test-db/l0_gb202.yml  |   4 +-
 .../test_lists/test-db/l0_rtx_pro_6000.yml         |   4 +-
 .../run_blockscaled_contiguous_grouped_gemm.py     | 831 +++++++++++++++++++++
 ...aled_contiguous_grouped_gemm_finalize_fusion.py |   4 +-
 tests/unittest/_torch/modules/test_fused_moe.py    |  22 +-
 .../_torch/thop/parallel/test_cute_dsl_moe.py      |   2 +-
 15 files changed, 1308 insertions(+), 502 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/autotuner.py b/tensorrt_llm/_torch/autotuner.py
index afe583ad1..33ef41af8 100644
--- a/tensorrt_llm/_torch/autotuner.py
+++ b/tensorrt_llm/_torch/autotuner.py
@@ -639,10 +639,13 @@ class AutoTuner:
         - Current replay state (which config and call index)
         """
 
+        runner_tactic_comb_checkers: List[Callable] = []
+
         def __init__(self, autotuner):
             # State for captured contexts
             self._captured_contexts: List[Dict[str, Any]] = []
-            self._configurations = None
+            self._context_tactics_lists: Optional[List[List[Tuple[int,
+                                                                  Any]]]] = None
             # State for replay mode
             self._replay_runner_tactic_list: Optional[List[Tuple[int,
                                                                  int]]] = None
@@ -654,10 +657,13 @@ class AutoTuner:
             For single context: yields (runner, tactic)
             For multiple contexts: yields ((runner_ctx0, tactic_ctx0), (runner_ctx1, tactic_ctx1), ...)
             """
-            if self._configurations is None:
-                self._configurations = self._generate_configurations()
+            if self._context_tactics_lists is None:
+                self._context_tactics_lists = self._generate_context_tactics_lists(
+                )
 
-            for config in self._configurations:
+            # Generate cartesian product from context and tactics where all_configrations[i][ctx] = (runner, tactic)
+            # Such that each element in all_configrations is a replay of multiple contexts of all possible replays
+            for config in itertools.product(*self._context_tactics_lists):
                 # config is a tuple of (runner_idx, tactic) for each context
                 # Convert to (runner, tactic) format for user
                 runner_tactic_pairs = []
@@ -666,9 +672,14 @@ class AutoTuner:
                     runner = runners[runner_idx]
                     runner_tactic_pairs.append((runner, tactic))
 
+                if not all(
+                        checker(runner_tactic_pairs) for checker in
+                        self.__class__.runner_tactic_comb_checkers):
+                    continue
+
                 yield tuple(runner_tactic_pairs)
 
-        def _generate_configurations(self):
+        def _generate_context_tactics_lists(self):
             """Generate all valid tactic combinations."""
             if not self._captured_contexts:
                 raise RuntimeError(
@@ -694,15 +705,17 @@ class AutoTuner:
                         tactics_lists.append((runner_idx, tactic))
                 context_tactics_lists.append(tactics_lists)
 
-            # Generate cartesian product from context and tactics where all_configrations[i][ctx] = (runner, tactic)
-            # Such that each element in all_configrations is a replay of multiple contexts of all possible replays
-            a
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 4742c130db - [None][feat] Improve TRTLLM MoE in small hidden size throughput cases (#9377)

- **Date**: 2025-11-25
- **Author**: Anthony Chang
- **Categories**: Throughput/Latency

### Optimization Techniques

- Vectorized memory access
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- Parallelism optimization
- Batching optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU

### Changed Files

```
.../batchedGemm/trtllmGen_bmm_export/.clang-format |  78 --
 .../trtllmGen_bmm_export/BatchedGemmEnums.h        |   2 +-
 .../trtllmGen_bmm_export/BatchedGemmInterface.h    |   5 +-
 .../trtllmGen_bmm_export/BatchedGemmOptions.h      |   6 +-
 .../trtllmGen_bmm_export/GemmGatedActOptions.h     |  19 +-
 .../batchedGemm/trtllmGen_bmm_export/GemmOptions.h |  26 +-
 .../trtllmGen_bmm_export/KernelMetaInfo.h          | 900 ++++++++++-----------
 .../trtllmGen_bmm_export/KernelParams.h            |  28 +-
 .../trtllmGen_bmm_export/KernelTraits.h            |  10 +-
 .../trtllmGen_bmm_export/TmaDescriptor.h           |   8 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...p_tmOv_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ut_schedS_biasM_bN_tmaOpt_clmp_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ut_schedS_biasM_bN_tmaOpt_clmp_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100a_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm103a_cubin.cpp |   4 +-
 ...x2x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...schedS_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x1x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x2x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...schedS_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x1x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x2x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...schedS_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x2x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...schedS_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x1x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...schedS_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x1x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...schedS_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x1x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x1x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x3_eW8_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x3_eW8_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x2x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...schedS_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x2x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...schedS_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x1x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...schedS_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x1x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...schedS_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x2x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...schedS_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x1x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x2x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...schedS_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x1x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...lA_dsFp8_schedS_bN_tmaOpt_clmp_sm100f_cubin.cpp |   4 +-
 ...x2x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...schedS_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...lA_dsFp8_schedS_bN_tmaOpt_clmp_sm100f_cubin.cpp |   4 +-
 ...x2x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...schedS_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x1x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...schedS_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...x1x2x3_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...schedS_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...transOut_schedS_bN_tmaOpt_clmp_sm100f_cubin.cpp |   4 +-
 ...transOut_schedS_bN_tmaOpt_clmp_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...p_tmOv_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._biasM_bN_tmaOpt_clmp_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...lmp_geGlu_lbW8_lsfbW4_dynBatch_sm100f_cubin.cpp |   4 +-
 ...mp_swiGlu_lbW8_lsfbW4_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ut_schedS_biasM_bN_tmaOpt_clmp_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ut_schedS_biasM_bN_tmaOpt_clmp_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...tma_tmaOpt_clmp_geGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100a_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm103a_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW4_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW4_dynBatch_sm100f_cubin.cpp |   4 +-
 ...aOpt_clmp_swiGlu_lbW8_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW4_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW4_dynBatch_sm100f_cubin.cpp |   4 +-
 ...aOpt_clmp_swiGlu_lbW8_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW2_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW2_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW2_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW2_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...aOpt_clmp_swiGlu_lbW8_dynBatch_sm100f_cubin.cpp |   4 +-
 ...aOpt_clmp_swiGlu_lbW8_dynBatch_sm100f_cubin.cpp |   4 +-
 ...aOpt_clmp_swiGlu_lbW8_dynBatch_sm100f_cubin.cpp |   4 +-
 ...aOpt_clmp_swiGlu_lbW8_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW2_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW2_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW2_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW2_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW4_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW4_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW4_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW4_dynBatch_sm100f_cubin.cpp |   4 +-
 ...aOpt_clmp_swiGlu_lbW8_dynBatch_sm100f_cubin.cpp |   4 +-
 ...aOpt_clmp_swiGlu_lbW8_dynBatch_sm100f_cubin.cpp |   4 +-
 ...lA_dsFp8_schedS_bN_tmaOpt_clmp_sm100f_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW2_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW2_dynBatch_sm100f_cubin.cpp |   4 +-
 ...lA_dsFp8_schedS_bN_tmaOpt_clmp_sm100f_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW2_dynBatch_sm100f_cubin.cpp |   4 +-
 ..._tma_tmaOpt_clmp_lbW2_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...transOut_schedS_bN_tmaOpt_clmp_sm100f_cubin.cpp |   4 +-
 ...transOut_schedS_bN_tmaOpt_clmp_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ts_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ut_schedS_biasM_bN_tmaOpt_clmp_sm100f_cubin.cpp |   4 +-
 ...ut_schedS_biasM_bN_tmaOpt_clmp_sm100f_cubin.cpp |   4 +-
 ...lA_dsFp8_schedS_bN_tmaOpt_clmp_sm100f_cubin.cpp |   4 +-
 ...lA_dsFp8_schedS_bN_tmaOpt_clmp_sm100f_cubin.cpp |   4 +-
 ...transOut_schedS_bN_tmaOpt_clmp_sm100f_cubin.cpp |   4 +-
 ...transOut_schedS_bN_tmaOpt_clmp_sm100f_cubin.cpp |   4 +-
 ...mp_swiGlu_lbW8_lsfbW4_dynBatch_sm100f_cubin.cpp |   4 +-
 ...mp_swiGlu_lbW8_lsfbW4_dynBatch_sm100f_cubin.cpp |   4 +-
 ...mp_swiGlu_lbW8_lsfbW4_dynBatch_sm100f_cubin.cpp |   4 +-
 ...mp_swiGlu_lbW8_lsfbW4_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...mp_swiGlu_lbW8_lsfbW4_dynBatch_sm100f_cubin.cpp |   4 +-
 ...mp_swiGlu_lbW8_lsfbW4_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...mp_swiGlu_lbW1_lsfbW1_dynBatch_sm100f_cubin.cpp |   4 +-
 ...mp_swiGlu_lbW1_lsfbW1_dynBatch_sm100f_cubin.cpp |   4 +-
 ...mp_swiGlu_lbW1_lsfbW1_dynBatch_sm100f_cubin.cpp |   4 +-
 ...mp_swiGlu_lbW1_lsfbW1_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 ...ma_tmaOpt_clmp_swiGlu_dynBatch_sm100f_cubin.cpp |   4 +-
 .../trtllm/gen/CudaKernelLauncher.h                |   2 +-
 .../trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h    |  24 +-
 461 files changed, 1431 insertions(+), 1473 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/.clang-format b/cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/.clang-format
deleted file mode 100644
index 6b89120c4..000000000
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/.clang-format
+++ /dev/null
@@ -1,78 +0,0 @@
----
-AccessModifierOffset: -4
-AlignAfterOpenBracket: DontAlign
-AlignConsecutiveAssignments: None
-AlignConsecutiveDeclarations: None
-AlignOperands:   false
-AlignTrailingComments: true
-AllowAllParametersOfDeclarationOnNextLine: true
-AllowShortBlocksOnASingleLine: Empty
-AllowShortCaseLabelsOnASingleLine: true
-AllowShortFunctionsOnASingleLine: Empty
-AllowShortIfStatementsOnASingleLine: false
-AllowShortLoopsOnASingleLine: false
-AlwaysBreakAfterDefinitionReturnType: None
-AlwaysBreakAfterReturnType: None
-AlwaysBreakBeforeMultilineStrings: true
-AlwaysBreakTemplateDeclarations: Yes
-BasedOnStyle: None
-BinPackArguments: true
-BinPackParameters: true
-BreakBeforeBinaryOperators: All
-BreakBeforeBraces: Allman
-BreakBeforeTernaryOperators: true
-BreakConstructorInitializersBeforeComma: true
-ColumnLimit:     120
-CommentPragmas:  '^ IWYU pragma:'
-ConstructorInitializerAllOnOneLineOrOnePerLine: false
-ConstructorInitializerIndentWidth: 4
-ContinuationIndentWidth: 4
-Cpp11BracedListStyle: true
-DerivePointerAlignment: false
-DisableFormat:   false
-ExperimentalAutoDetectBinPacking: false
-ForEachMacros:   [ foreach, Q_FOREACH, BOOST_FOREACH ]
-IncludeBlocks: Preserve
-IncludeCategories:
-  - Regex:           '^"(llvm|llvm-c|clang|clang-c)/'
-    Priority:        2
-  - Regex:           '^(<|"(gtest|isl|json)/)'
-    Priority:        3
-  - Regex:           '.*'
-    Priority:        1
-IndentCaseLabels: false
-IndentWidth:     4
-IndentWrappedFunctionNames: false
-KeepEmptyLinesAtTheStartOfBlocks: true
-Language: Cpp
-MacroBlockBegin: ''
-MacroBlockEnd:   ''
-MaxEmptyLinesToKeep: 1
-NamespaceIndentation: None
-ObjCBlockIndentWidth: 4
-ObjCSpaceAfterProperty: true
-ObjCSpaceBeforeProtocolList: true
-PenaltyBreakBeforeFirstCallParameter: 19
-PenaltyBreakComment: 300
-PenaltyBreakFirstLessLess: 120
-PenaltyBreakString: 1000
-PenaltyExcessCharacter: 1000000
-PenaltyReturnTypeOnItsOwnLine: 60
-PointerAlignment: Left
-QualifierAlignment: Right
-ReflowComments:  true
-SeparateDefinitionBlocks: Always
-SortIncludes:    false
-SpaceAfterCStyleCast: true
-SpaceBeforeAssignmentOperators: true
-SpaceBeforeParens: ControlStatements
-SpaceInEmptyParentheses: false
-SpacesBeforeTrailingComments: 1
-SpacesInAngles:  false
-SpacesInCStyleCastParentheses: false
-SpacesInContainerLiterals: true
-SpacesInParentheses: false
-SpacesInSquareBrackets: false
-Standard:        c++14
-TabWidth:        4
-UseTab:          Never
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/BatchedGemmEnums.h b/cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_expo
```

### Analysis Summary

Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 497a07021d - [None][update] optimized sparse mla kernels && fix unspecified cuda launch (#8866)

- **Date**: 2025-11-03
- **Author**: Perkz Zheng
- **Categories**: Kernel Optimization

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- FP8 quantization
- KV cache optimization
- Batching optimization
- Speculative decoding
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
 ...SparseQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...SparseQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...seQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...SparseQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
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
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
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
 ...SparseQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...SparseQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...seQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...SparseQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
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
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
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
 .../trtllmGenKernels/fmha/cubin/kernelMetaInfo.h   | 2945 ++++++++++----------
 .../kernels/trtllmGenKernels/fmha/fmhaKernels.h    |    4 +-
 .../kernels/trtllmGenKernels/fmha/fmhaReduction.cu |   27 +-
 .../kernels/trtllmGenKernels/fmha/kernelParams.h   |   16 +-
 1464 files changed, 4418 insertions(+), 4398 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128PersistentContext_cubin.cpp b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128PersistentContext_cubin.cpp
index f43976f4e..cd3b96381 100644
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128PersistentContext_cubin.cpp
+++ b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128PersistentContext_cubin.cpp
@@ -1,3 +1,3 @@
 version https://git-lfs.github.com/spec/v1
-oid sha256:65b4bfd91a2e48211d5613a3f2baa36862cd8c2880204fe2b9322847ce8d1c9b
-size 674929
+oid sha256:25e363695ec56d2b74b48a67d2ad3a214c664a33894e2f14428521aef1f74028
+size 664199
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128StaticContext_cubin.cpp b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128StaticContext_cubin.cpp
index 65a241f8d..a0e0c4d46 100644
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128StaticContext_cubin.cpp
+++ b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128StaticContext_cubin.cpp
@@ -1,3 +1,3 @@
 version https://git-lfs.github.com/spec/v1
-oid sha256:eab155475ab87dab2a9b8dcf8caac39eee33529fbd53cd031cd6abb700b30ebb
-size 603608
+oid sha256:452b9fbe68a5606a6551f450d0ad259b9529d08dd1a0e968768a53ba9d1a5522
+size 572380
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP64VarSeqQ128Kv128PersistentContext_cubin.cpp b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP64VarSeqQ128Kv128PersistentContext_cubin.cpp
index 740872d98..6b8c9e711 100644
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP64VarSeqQ128Kv128PersistentContext_cubin.cpp
+++ b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP64VarSeqQ128Kv128PersistentContext_cubin.cpp
@@ -1,3 +1,3 @@
 version https://git-lfs.github.com/spec/v1
-oid sha256:e061bc572b0c04d7496a197ed49891b905ebdb750781ff65d559bca9c4c15597
-size 657563
+oid sha256:86891a3f60bc637ff3c92e4398dc6f5f2d4aa58848d466bf5694fe8d1e778f42
+size 648411
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP64VarSeqQ128Kv128StaticContext_cubin.cpp b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP64VarSeqQ128Kv128StaticContext_cubin.cpp
index 425467ab4..82f
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 498b25cb60 - [TRTLLM-11259][perf] Parallel VAE harness and implementation for WAN (#11875)

- **Date**: 2026-03-06
- **Author**: NVShreyas
- **Categories**: Parallelism/Async

### Optimization Techniques

- Torch compilation/JIT optimization
- Custom CUDA kernel
- Operator fusion
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
examples/visual_gen/visual_gen_wan_i2v.py          |   2 +
 examples/visual_gen/visual_gen_wan_t2v.py          |   2 +
 tensorrt_llm/_torch/visual_gen/config.py           |   2 +-
 .../_torch/visual_gen/models/wan/__init__.py       |   8 +-
 .../_torch/visual_gen/models/wan/parallel_vae.py   | 151 +++++++++++++
 .../_torch/visual_gen/models/wan/pipeline_wan.py   |   7 +-
 .../visual_gen/models/wan/pipeline_wan_i2v.py      |   8 +-
 .../_torch/visual_gen/modules/vae/__init__.py      |  12 +
 .../_torch/visual_gen/modules/vae/attention.py     |  47 ++++
 tensorrt_llm/_torch/visual_gen/modules/vae/conv.py | 241 +++++++++++++++++++++
 tensorrt_llm/_torch/visual_gen/modules/vae/norm.py |  59 +++++
 .../modules/vae/parallel_vae_interface.py          | 113 ++++++++++
 tensorrt_llm/_torch/visual_gen/parallelism.py      |  32 ++-
 tensorrt_llm/_torch/visual_gen/pipeline.py         |  50 ++++-
 tensorrt_llm/_torch/visual_gen/pipeline_loader.py  |   3 +
 tensorrt_llm/_torch/visual_gen/utils.py            |   4 +
 .../multi_gpu/test_parallel_attention.py           | 121 +++++++++++
 .../visual_gen/multi_gpu/test_parallel_conv.py     | 199 +++++++++++++++++
 .../multi_gpu/test_parallel_group_norm.py          | 142 ++++++++++++
 .../visual_gen/multi_gpu/test_parallel_vae.py      | 209 ++++++++++++++++++
 20 files changed, 1406 insertions(+), 6 deletions(-)
```

### Diff Preview

```diff
diff --git a/examples/visual_gen/visual_gen_wan_i2v.py b/examples/visual_gen/visual_gen_wan_i2v.py
index b2ed3e7bf..050215b51 100644
--- a/examples/visual_gen/visual_gen_wan_i2v.py
+++ b/examples/visual_gen/visual_gen_wan_i2v.py
@@ -133,6 +133,7 @@ def parse_args():
         default=1,
         help="Ulysses (sequence) parallel size within each CFG group.",
     )
+    parser.add_argument("--disable_parallel_vae", action="store_true", help="Disable parallel VAE")
 
     # CUDA graph
     parser.add_argument(
@@ -187,6 +188,7 @@ def main():
         "parallel": {
             "dit_cfg_size": args.cfg_size,
             "dit_ulysses_size": args.ulysses_size,
+            "enable_parallel_vae": not args.disable_parallel_vae,
         },
         "torch_compile": {
             "enable_torch_compile": not args.disable_torch_compile,
diff --git a/examples/visual_gen/visual_gen_wan_t2v.py b/examples/visual_gen/visual_gen_wan_t2v.py
index 83ac956f3..29c1da66d 100755
--- a/examples/visual_gen/visual_gen_wan_t2v.py
+++ b/examples/visual_gen/visual_gen_wan_t2v.py
@@ -133,6 +133,7 @@ def parse_args():
         "Example: ulysses_size=2 on 4 GPUs with cfg_size=2 -> "
         "2 CFG groups × 2 Ulysses ranks = 4 GPUs total.",
     )
+    parser.add_argument("--disable_parallel_vae", action="store_true", help="Disable parallel VAE")
 
     # CUDA graph
     parser.add_argument(
@@ -196,6 +197,7 @@ def main():
         "parallel": {
             "dit_cfg_size": args.cfg_size,
             "dit_ulysses_size": args.ulysses_size,
+            "enable_parallel_vae": not args.disable_parallel_vae,
         },
         "torch_compile": {
             "enable_torch_compile": not args.disable_torch_compile,
diff --git a/tensorrt_llm/_torch/visual_gen/config.py b/tensorrt_llm/_torch/visual_gen/config.py
index bb076cc89..3111957cb 100644
--- a/tensorrt_llm/_torch/visual_gen/config.py
+++ b/tensorrt_llm/_torch/visual_gen/config.py
@@ -85,7 +85,7 @@ class ParallelConfig(BaseModel):
            GPU 4-7: CFG group 1 (negative), Ulysses parallel
     """
 
-    disable_parallel_vae: bool = False
+    enable_parallel_vae: bool = True
     parallel_vae_split_dim: Literal["width", "height"] = "width"
 
     # DiT Parallelism
diff --git a/tensorrt_llm/_torch/visual_gen/models/wan/__init__.py b/tensorrt_llm/_torch/visual_gen/models/wan/__init__.py
index f17774080..ca6386f12 100644
--- a/tensorrt_llm/_torch/visual_gen/models/wan/__init__.py
+++ b/tensorrt_llm/_torch/visual_gen/models/wan/__init__.py
@@ -1,5 +1,11 @@
+from .parallel_vae import WanParallelVAEAdapter
 from .pipeline_wan import WanPipeline
 from .pipeline_wan_i2v import WanImageToVideoPipeline
 from .transformer_wan import WanTransformer3DModel
 
-__all__ = ["WanPipeline", "WanImageToVideoPipeline", "WanTransformer3DModel"]
+__all__ = [
+    "WanPipeline",
+    "WanImageToVideoPipeline",
+    "WanTransformer3DModel",
+    "WanParallelVAEAdapter",
+]
diff --git a/tensorrt_llm/_torch/visual_gen/models/wan/parallel_vae.py 
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 49f2f1f8eb - Expose new tech blog about DSR1 throughput optimization to the main R… (#4803)

- **Date**: 2025-05-30
- **Author**: juney-nvidia
- **Categories**: Throughput/Latency

### Optimization Techniques

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
- Decode/generation phase

### Changed Files

```
README.md                                                      |  3 +++
 ...timizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md | 10 +++++-----
 2 files changed, 8 insertions(+), 5 deletions(-)
```

### Diff Preview

```diff
diff --git a/README.md b/README.md
index ff7e07180..22c54b6fa 100644
--- a/README.md
+++ b/README.md
@@ -18,6 +18,9 @@ TensorRT-LLM
 <div align="left">
 
 ## Tech Blogs
+* [05/30] Optimizing DeepSeek R1 Throughput on NVIDIA Blackwell GPUs: A Deep Dive for Developers
+✨ [➡️ link](./docs/source/blogs/tech_blog/blog3_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md)
+
 * [05/23] DeepSeek R1 MTP Implementation and Optimization
 ✨ [➡️ link](./docs/source/blogs/tech_blog/blog2_DeepSeek_R1_MTP_Implementation_and_Optimization.md)
 
diff --git a/docs/source/blogs/tech_blog/blog3_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md b/docs/source/blogs/tech_blog/blog3_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md
index 28f380e8e..15b418f9b 100644
--- a/docs/source/blogs/tech_blog/blog3_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md
+++ b/docs/source/blogs/tech_blog/blog3_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md
@@ -7,7 +7,7 @@ By NVIDIA TensorRT-LLM team
   - [Parallel strategy](#parallel-strategy)
     - [Weights absorb and MQA](#weights-absorb-and-mqa)
     - [Data Parallel for Attention module (ADP)](#data-parallel-for-attention-module-adp)
-    - [Expert parallel for MOE (EP)](#expert-parallel-for-moe-ep)
+    - [Expert parallel for MoE (EP)](#expert-parallel-for-moe-ep)
   - [MLA Layers Optimizations](#mla-layers-optimizations)
   - [MoE Layers Optimizations](#moe-layers-optimizations)
   - [Runtime Optimizations](#runtime-optimizations)
@@ -31,15 +31,15 @@ The checkpoint used in this blog is hosted in [nvidia/DeepSeek-R1-FP4](https://h
 
 | Precision | GPQA Diamond | MATH-500
 | :-- | :-- | :-- |
-| TRTLLM FP8 | 0.697	| 0.954 |
-| TRTLLM FP4 | 0.705	| 0.96 |
+| TensorRT-LLM FP8 | 0.697	| 0.954 |
+| TensorRT-LLM FP4 | 0.705	| 0.96 |
 
 ** Note there are some run-to-run variance for these evaluations, so FP4 data is slight higher here. We think FP4 has comparable accuracy with FP8 on these datasets.
 
 The MoE layers inside this checkpoint have been quantized into FP4. Quantizing the MoE layer weights into FP4 has the following benefits:
 
 * Fully utilize the 5th generation Tensor Core FLOPS of the NVIDIA Blackwell GPUs
-* Reduce the memory load needs of the weights by almost half for MoE. Since the MOE parts are still memory bound for the decoding phase for the scenario, and 97% of the weights in the DeepSeek R1 model are from MOE layers.
+* Reduce the memory load needs of the weights by almost half for MoE. Since the MoE parts are still memory bound for the decoding phase for the scenario, and 97% of the weights in the DeepSeek R1 model are from MoE layers.
 * Reduce the memory footprint of the model weights, thus freeing more GPU memories for KV cache and then increasing the max concurrency. [The original FP8 model checkpoint of the DeepSeek R1 model](https://huggingface.co/deepseek-ai/DeepSeek-R1) is about 640GB, while the NVIDIA provided [DeepSeek R1 FP4 quantized model](htt
```

### Analysis Summary

Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 4ae46b6714 - fix: [nvbugs/5324229] Fix broken WInt4AFP8FusedMoEMethod since FusedMoE refactor. (#4930)

- **Date**: 2025-06-13
- **Author**: Yuxian Qiu
- **Categories**: Fusion, Quantization Optimization

### Optimization Techniques

- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tests/unittest/_torch/modules/test_moe_load_balancer.py | 2 --
 1 file changed, 2 deletions(-)
```

### Diff Preview

```diff
diff --git a/tests/unittest/_torch/modules/test_moe_load_balancer.py b/tests/unittest/_torch/modules/test_moe_load_balancer.py
index 16a60e768..a49c1a4ea 100644
--- a/tests/unittest/_torch/modules/test_moe_load_balancer.py
+++ b/tests/unittest/_torch/modules/test_moe_load_balancer.py
@@ -1,7 +1,6 @@
 import unittest
 from unittest.mock import MagicMock, patch
 
-import pytest
 import torch
 from mpi4py import MPI
 
@@ -179,7 +178,6 @@ class TestMoeLoadBalancer(unittest.TestCase):
         # Verify the global state is cleaned up
         self.assertIsNone(get_moe_load_balancer())
 
-    @pytest.mark.skip(reason="https://nvbugs/5324229")
     @patch('tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer')
     def test_single_layer_moe_load_balancer_methods(self,
                                                     mock_load_balancer_impl):

```

### Analysis Summary

MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 4b82b8b4c7 - [TRTLLM-5330] perf: Optimize MoE supplementary kernels for large-scale EP (#5215)

- **Date**: 2025-06-17
- **Author**: Enwei Zhu
- **Categories**: Kernel Optimization

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- Parallelism optimization
- Reduce synchronization overhead
- MoE optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU

### Changed Files

```
.../mixtureOfExpertsBackendBenchmarkFixtureOss.h   |   4 +-
 .../kernels/cutlass_kernels/include/moe_kernels.h  |  42 +-
 .../cutlass_kernels/moe_gemm/moe_kernels.cu        | 442 +++++++++++++--------
 .../mixtureOfExperts/mixtureOfExpertsPlugin.cpp    |   7 +-
 cpp/tensorrt_llm/thop/moeOp.cpp                    |  15 +-
 .../unit_tests/kernels/mixtureOfExpertsOssTest.cu  |  65 ++-
 .../unit_tests/kernels/mixtureOfExpertsTest.cu     |   1 -
 .../_torch/auto_deploy/custom_ops/fused_moe.py     |   1 +
 tensorrt_llm/_torch/custom_ops/torch_custom_ops.py |   6 +
 .../_torch/modules/fused_moe/fused_moe_cutlass.py  |   1 +
 10 files changed, 365 insertions(+), 219 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixtureOss.h b/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixtureOss.h
index df4e47b07..8c15da199 100644
--- a/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixtureOss.h
+++ b/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixtureOss.h
@@ -642,7 +642,7 @@ public:
         GemmProfilerBackend profiler;
         profiler.init(mMoERunner, gemm_to_profile, typeToDtypeID<DataType>(), typeToDtypeID<WeightType>(),
             typeToDtypeID<OutputType>(), mNumExperts, mK, mHiddenSize, mInterSize, mGroupSize, mActType, mUseBias,
-            mUseLora, /*min_latency_mode=*/false, /*need_weights=*/true, parallelism_config);
+            mUseLora, /*min_latency_mode=*/false, /*need_weights=*/true, parallelism_config, /*enable_alltoall=*/false);
         auto workspace_size = profiler.getWorkspaceSize(mTotalTokens);
         auto workspace = bufferManager->gpu(workspace_size);
 
@@ -753,7 +753,7 @@ public:
         mMoERunner.runMoe(mInputTensor, nullptr, mSelectedExperts, mUseFinalScale ? mScaleProbs : nullptr,
             mExpertWeight1, mExpertBias1, mActType, mExpertWeight2, mExpertBias2, mQuantParams, mTotalTokens,
             mHiddenSize, mInterSize, mNumExperts, mK, mWorkspace, mFinalOutput, mSourceToExpandedMap,
-            parallelism_config, mUseLora, mLoraParams,
+            parallelism_config, /*enable_alltoall=*/false, mUseLora, mLoraParams,
             /*use_fp8_block_scaling=*/false, /*min_latency_mode=*/false, min_latency_params, stream);
     }
 
diff --git a/cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h b/cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h
index f321a905e..7796c57ca 100644
--- a/cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h
+++ b/cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h
@@ -348,9 +348,9 @@ public:
         ActivationType fc1_activation_type, void const* fc2_expert_weights, void const* fc2_expert_biases,
         QuantParams quant_params, int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
         int const num_experts, int const experts_per_token, char* workspace_ptr, void* final_output,
-        int* expanded_source_row_to_expanded_dest_row, MOEParallelismConfig parallelism_config, bool use_lora,
-        LoraParams& lora_params, bool use_fp8_block_scaling, bool min_latency_mode,
-        MoeMinLatencyParams& min_latency_params, cudaStream_t stream)
+        int* expanded_source_row_to_expanded_dest_row, MOEParallelismConfig parallelism_config,
+        bool const enable_alltoall, bool use_lora, LoraParams& lora_params, bool use_fp8_block_scaling,
+        bool min_latency_mode, MoeMinLatencyParams& min_latency_params, cudaStream_t stream)
         = 0;
 
     // Aliases for profiling the gemms
@@ -378,8 +378,8 @@ public:
         int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node,
         int64_t const ex
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## 4c15db0bfa - [https://nvbugs/5732958][bug] Fix TestLlama4MinLatency::test_llama_allclose_to_hf failure (#10191)

- **Date**: 2026-03-09
- **Author**: Po-Han Huang (NVIDIA)
- **Categories**: Throughput/Latency

### Optimization Techniques

- Operator fusion
- Integer quantization
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/models/modeling_llama.py       | 32 ++++++++++++++++++++++
 .../_torch/modeling/test_modeling_llama.py         |  1 +
 .../modeling/test_modeling_llama_min_latency.py    |  6 ++--
 3 files changed, 35 insertions(+), 4 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_llama.py b/tensorrt_llm/_torch/models/modeling_llama.py
index 54193a32c..743e0b8ef 100644
--- a/tensorrt_llm/_torch/models/modeling_llama.py
+++ b/tensorrt_llm/_torch/models/modeling_llama.py
@@ -449,6 +449,10 @@ class Llama4DecoderLayer(DecoderLayer):
         self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                        eps=config.rms_norm_eps,
                                        dtype=config.torch_dtype)
+        # When post_load_weights() chains layernorms across layers,
+        # this flag is set to True to skip the input layernorm in
+        # forward() since it is handled by the previous layer.
+        self.skip_input_layernorm = False
 
         self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                 eps=config.rms_norm_eps,
@@ -493,6 +497,8 @@ class Llama4DecoderLayer(DecoderLayer):
 
         if residual is None:
             residual = hidden_states
+
+        if not self.skip_input_layernorm:
             hidden_states = self.input_layernorm(hidden_states)
 
         # Self Attention
@@ -668,6 +674,10 @@ class LlamaDecoderLayer(DecoderLayer):
             quantize_type="nvfp4"
             if not self.disable_nvfp4_layernorm_fusion and self.is_nvfp4
             and not (differ_pp_stage_with_previous_layer) else None)
+        # When post_load_weights() chains layernorms across layers,
+        # this flag is set to True to skip the input layernorm in
+        # forward() since it is handled by the previous layer.
+        self.skip_input_layernorm = False
 
         self.post_attention_layernorm = RMSNorm(
             hidden_size=config.hidden_size,
@@ -765,6 +775,8 @@ class LlamaDecoderLayer(DecoderLayer):
     ) -> Union[torch.Tensor, Fp4QuantizedTensor]:
         if residual is None:
             residual = hidden_states
+
+        if not self.skip_input_layernorm:
             hidden_states = self.input_layernorm(hidden_states)
 
         hidden_states = self.self_attn(
@@ -936,6 +948,10 @@ class Llama4Model(DecoderModel):
         self.norm = RMSNorm(hidden_size=config.hidden_size,
                             eps=config.rms_norm_eps,
                             dtype=config.torch_dtype)
+        # When post_load_weights() chains the final norm into the
+        # last decoder layer, this flag is set to True to skip
+        # applying it again in forward().
+        self.skip_norm = False
 
     def forward(
         self,
@@ -969,6 +985,10 @@ class Llama4Model(DecoderModel):
                 lora_params=lora_params,
             )
 
+        # If self.norm is not handled by the last layer, apply it here.
+        if not self.skip_norm:
+            hidden_states = self.norm(hidden_states)
+
         return hidden_states
 
 
@@ -1033,6 +1053,10 @@ class LlamaModel(DecoderModel):
         self.norm = RMSNorm(hidden_size=config.hidden_size,
                            
```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## 4dc7bc525f - [None][fix] Refine tests/unittest/_torch/flashinfer/test_trtllm_flashinfer_symbol_collision.py to reduce jit-compile time (#11890)

- **Date**: 2026-03-06
- **Author**: Yihan Wang
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Parallelism optimization
- Speculative decoding
- MoE optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU

### Changed Files

```
tests/integration/test_lists/test-db/l0_b200.yml   |   2 +
 tests/integration/test_lists/test-db/l0_h100.yml   |   1 -
 .../test_trtllm_flashinfer_symbol_collision.py     | 126 ++++++++++-----------
 3 files changed, 63 insertions(+), 66 deletions(-)
```

### Diff Preview

```diff
diff --git a/tests/integration/test_lists/test-db/l0_b200.yml b/tests/integration/test_lists/test-db/l0_b200.yml
index f32f0e07d..10bca53b2 100644
--- a/tests/integration/test_lists/test-db/l0_b200.yml
+++ b/tests/integration/test_lists/test-db/l0_b200.yml
@@ -103,6 +103,8 @@ l0_b200:
   - unittest/_torch/modules/moe/test_moe_module.py::test_configurable_moe_single_gpu -k "TRTLLM"
   - unittest/_torch/modules/moe/test_moe_module.py::test_configurable_moe_single_gpu -k "CUTEDSL"
   - unittest/_torch/modules/moe/test_moe_module.py::test_configurable_moe_single_gpu -k "DEEPGEMM"
+  # ------------- MoE: FlashInfer & TRTLLM symbol collision tests ---------------
+  - unittest/_torch/flashinfer/test_trtllm_flashinfer_symbol_collision.py
   # --- MoE end
   - unittest/_torch/multimodal
   - unittest/_torch/sampler
diff --git a/tests/integration/test_lists/test-db/l0_h100.yml b/tests/integration/test_lists/test-db/l0_h100.yml
index 6631322f8..c8e3ff7e7 100644
--- a/tests/integration/test_lists/test-db/l0_h100.yml
+++ b/tests/integration/test_lists/test-db/l0_h100.yml
@@ -46,7 +46,6 @@ l0_h100:
   - unittest/_torch/speculative -k "not eagle3"
   - unittest/_torch/thop/parallel
   - unittest/_torch/thop/serial
-  - unittest/_torch/flashinfer/test_trtllm_flashinfer_symbol_collision.py
   # Only key models in H100: llama/mixtral/nemotron/deepseek
   - unittest/_torch/modeling -k "modeling_llama"
   - unittest/_torch/modeling -k "modeling_mixtral"
diff --git a/tests/unittest/_torch/flashinfer/test_trtllm_flashinfer_symbol_collision.py b/tests/unittest/_torch/flashinfer/test_trtllm_flashinfer_symbol_collision.py
index 6e3a6415b..5c75d25d4 100644
--- a/tests/unittest/_torch/flashinfer/test_trtllm_flashinfer_symbol_collision.py
+++ b/tests/unittest/_torch/flashinfer/test_trtllm_flashinfer_symbol_collision.py
@@ -1,84 +1,80 @@
-"""Unit tests for FlashInfer fused MOE custom op."""
+# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+# http://www.apache.org/licenses/LICENSE-2.0
+#
+# Unless required by applicable law or agreed to in writing, software
+# distributed under the License is distributed on an "AS IS" BASIS,
+# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+# See the License for the specific language governing permissions and
+# limitations under the License.
+"""
+Unit tests verifying no symbol collision between TensorRT-LLM and FlashInfer.
+
+FlashInfer copies several TensorRT-LLM CUTLASS MOE kernel source files
+(under nv_internal/tensorrt_llm/) and JIT-compiles them. Without the
+inline-namespace fix (TRTLLM_ABI_NAMESPACE _v1), the resulting .so exports
+symbols with identical mangled names as libth_common.so (loaded with
+RTLD_GLOBAL), causing heap corr
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

