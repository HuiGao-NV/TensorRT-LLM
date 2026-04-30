# Performance Optimization Analysis - Part 9

Commits 233 to 261 of 283

---

## d913955952 - [TRTLLM-6898][feat] make fused_moe_cute_dsl work on blackwell (#6616)

- **Date**: 2025-08-08
- **Author**: Li Min
- **Categories**: Fusion

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
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

### Changed Files

```
.../_torch/modules/fused_moe/fused_moe_deepgemm.py |  15 +-
 .../_torch/modules/fused_moe/quantization.py       |  81 +++++----
 tensorrt_llm/_torch/modules/gated_mlp.py           |   9 +-
 tensorrt_llm/_torch/modules/linear.py              |  30 ++--
 tensorrt_llm/evaluate/lm_eval.py                   |   3 +-
 .../defs/accuracy/test_llm_api_pytorch.py          |   4 +-
 tests/unittest/_torch/modules/test_fused_moe.py    | 190 +++++++++++++++++++--
 7 files changed, 262 insertions(+), 70 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py
index bcdf8d441..e659801e0 100644
--- a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py
@@ -13,7 +13,8 @@ from ...distributed import allgather
 from ...model_config import ModelConfig
 from ...utils import Fp4QuantizedTensor
 from .fused_moe_cutlass import CutlassFusedMoE
-from .quantization import MoEWeightLoadingMode
+from .quantization import (DeepSeekFP8BlockScalesFusedMoEMethodDeepGemm,
+                           MoEWeightLoadingMode, UnquantizedFusedMoEMethod)
 from .routing import BaseMoeRoutingMethod
 
 
@@ -340,6 +341,18 @@ class DeepGemmFusedMoE(CutlassFusedMoE):
             layer_idx=layer_idx,
         )
 
+    def _get_quant_method(self):
+        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
+                exclude_kv_cache=True):
+            if self.quant_config.layer_quant_mode.has_fp8_block_scales():
+                return DeepSeekFP8BlockScalesFusedMoEMethodDeepGemm()
+            else:
+                raise ValueError(
+                    f"Unsupported quantization mode: {self.quant_config.quant_mode}"
+                )
+        else:
+            return UnquantizedFusedMoEMethod()
+
     @nvtx_range("[DG] forward")
     def forward_chunk(
         self,
diff --git a/tensorrt_llm/_torch/modules/fused_moe/quantization.py b/tensorrt_llm/_torch/modules/fused_moe/quantization.py
index 1510fac47..ca373c2ed 100644
--- a/tensorrt_llm/_torch/modules/fused_moe/quantization.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/quantization.py
@@ -629,45 +629,8 @@ class DeepSeekFP8BlockScalesFusedMoEMethod(FusedMoEMethodBase):
 
     def load_weights(self, module: torch.nn.Module, weights: List[Dict],
                      weight_loading_mode: MoEWeightLoadingMode):
-
-        if get_sm_version() == 100:
-            expert_ids = set(module.initial_local_expert_ids)
-            if self.need_load_shared_weights(module):
-                expert_ids.update(
-                    module.layer_load_balancer.get_load_expert_ids())
-            for name in list(weights.keys()):
-                if name.endswith("weight_scale_inv"):
-                    if int(name.split(".")[0]) not in expert_ids:
-                        continue
-                    weight_name = name.replace("weight_scale_inv", "weight")
-                    logger.debug(f"Resmoothing {weight_name}")
-                    weight = weights[weight_name][:]
-                    scale = weights[name][:]
-                    weights[weight_name], weights[name] = resmooth_to_fp8_e8m0(
-                        weight, scale)
         super().load_weights(module, weights, weight_loading_mode)
 
-        if get_sm_version() == 100:
-            transfromed_w3_w1_scale = transform_sf_into_required_layout(
-                module.quant_s
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## da6cb541a2 - [None][feat] Optimize MLA kernels with separate reduction kernels (#7597)

- **Date**: 2025-09-09
- **Author**: Perkz Zheng
- **Categories**: Kernel Optimization

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Reduce synchronization overhead
- Speculative decoding
- Attention mechanism optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- Decode/generation phase

### Changed Files

```
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
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
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
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
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
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
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
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
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
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
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
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
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
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
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
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    3 -
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
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
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
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
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
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
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
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
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
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
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
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
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
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
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
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
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
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
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
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
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
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
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
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    3 -
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    3 -
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    3 -
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    3 -
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    3 -
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...eQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    2 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    2 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...eQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    2 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 +
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    3 -
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...eqQ64Kv128Persistent2CtaKeepsAbForGen_cubin.cpp |    2 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...VarSeqQ64Kv128Static2CtaKeepsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...32VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...nseP32VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...vCgaVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...sKvCgaVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...mSepVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 +
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    3 -
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...CtasKvVarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ16Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ16Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...VarSeqQ64Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ64Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...64VarSeqQ8Kv64PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...nseP64VarSeqQ8Kv64StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...QkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...DenseVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...dQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...ausalVarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...kedCausalVarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P32VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...CgaVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...sKvVarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    4 +-
 ...arSeqQ128Kv128PersistentKeepsAbForGen_cubin.cpp |    2 +-
 ...P64VarSeqQ128Kv128StaticKeepsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...seP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...vDenseP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...eP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...seP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP32VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP32VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP32VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...2VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP32VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...vCgaVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...KvCgaVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...asKvVarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...tasKvVarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    4 +-
 ...alP64VarSeqQ128Kv128PersistentContext_cubin.cpp |    2 +-
 ...CausalP64VarSeqQ128Kv128StaticContext_cubin.cpp |    2 +-
 ...VarSeqQ16Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...lP64VarSeqQ16Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 ...4VarSeqQ8Kv128PersistentSwapsAbForGen_cubin.cpp |    2 +-
 ...alP64VarSeqQ8Kv128StaticSwapsAbForGen_cubin.cpp |    2 +-
 .../trtllmGenKernels/fmha/cubin/kernelMetaInfo.h   | 2954 ++++++++++----------
 .../kernels/trtllmGenKernels/fmha/fmhaKernels.h    |   12 +-
 .../kernels/trtllmGenKernels/fmha/fmhaReduction.cu |  374 +++
 .../kernels/trtllmGenKernels/fmha/fmhaReduction.h  |   36 +
 .../trtllmGenKernels/fmha/fmhaRunnerParams.h       |    9 +-
 .../kernels/trtllmGenKernels/fmha/kernelUtils.h    |  173 ++
 .../defs/accuracy/references/cnn_dailymail.yaml    |   12 +
 .../defs/accuracy/test_llm_api_pytorch.py          |   16 +-
 .../integration/test_lists/test-db/l0_dgx_b200.yml |    1 +
 2737 files changed, 4086 insertions(+), 7265 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QkvBfloat16OBfloat16H128PackedQkvCausalVarSeqQ128Kv128PersistentContext_cubin.cpp b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QkvBfloat16OBfloat16H128PackedQkvCausalVarSeqQ128Kv128PersistentContext_cubin.cpp
deleted file mode 100644
index 39322cf8d..000000000
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QkvBfloat16OBfloat16H128PackedQkvCausalVarSeqQ128Kv128PersistentContext_cubin.cpp
+++ /dev/null
@@ -1,3 +0,0 @@
-version https://git-lfs.github.com/spec/v1
-oid sha256:2965edaf6cc339fa943d7761c55f2b1ef670bb16359aaa9fd5ab6f7107bcb099
-size 802287
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QkvBfloat16OBfloat16H128PackedQkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QkvBfloat16OBfloat16H128PackedQkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp
deleted file mode 100644
index 8bdb5d176..000000000
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QkvBfloat16OBfloat16H128PackedQkvCausalVarSeqQ128Kv128StaticContext_cubin.cpp
+++ /dev/null
@@ -1,3 +0,0 @@
-version https://git-lfs.github.com/spec/v1
-oid sha256:8e2f7f42b15d8e57c0916c95bdb1ea392d23864f9c7ed4a34f800ea0c7605b9b
-size 715155
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QkvBfloat16OBfloat16H128PackedQkvDenseVarSeqQ128Kv128PersistentContext_cubin.cpp b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QkvBfloat16OBfloat16H128PackedQkvDenseVarSeqQ128Kv128PersistentContext_cubin.cpp
deleted file mode 100644
index 0aa797a36..000000000
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QkvBfloat16OBfloat16H128PackedQkvDenseVarSeqQ128Kv128PersistentContext_cubin.cpp
+++ /dev/null
@@ -1,3 +0,0 @@
-version https://git-lfs.github.com/spec/v1
-oid sha256:bf63599cd1a80e9d8c8a1f0acf1f4e17a77cfd9a8f2d5387553c3f0a292042e9
-size 804209
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QkvBfloat16OBfloat16H128PackedQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QkvBfloat16OBfloat16H128PackedQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp
deleted file mode 100644
index 05f67f931..000000000
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QkvBfloat16OBfloat16H128PackedQkvDenseVarSeqQ128Kv128StaticContext_cubin.cpp
+++ /dev/null
@@ -1,3 +0,0 @@
-version https://git-lfs.github.com/spec/v1
-oid sha256:df223168d786a4aac8157a1ad45a49607d59979642af46721c9b726f4f8caea2
-size 715005
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QkvBfloat16OBfloat16H128PackedQkvSlidingOrChunkedCausalVarSeqQ128Kv128PersistentContext_cubin.cpp b/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/FmhaSm100Kernel_QkvBfloat16OBfloat1
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## dbad94715b - [None][feat] Add gRPC server for high-performance external router integration (#11037)

- **Date**: 2026-01-29
- **Author**: Chang Su
- **Categories**: General Performance

### Optimization Techniques

- Async/stream-based execution
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Triton kernel
- Attention mechanism optimization

### Applicable Conditions

- Prefill phase
- Decode/generation phase
- Disaggregated serving

### Changed Files

```
setup.py                                  |  44 +++
 tensorrt_llm/commands/serve.py            | 155 ++++++++-
 tensorrt_llm/grpc/__init__.py             | 121 +++++++
 tensorrt_llm/grpc/compile_protos.py       | 167 +++++++++
 tensorrt_llm/grpc/grpc_request_manager.py | 420 +++++++++++++++++++++++
 tensorrt_llm/grpc/grpc_servicer.py        | 545 ++++++++++++++++++++++++++++++
 tensorrt_llm/grpc/trtllm_service.proto    | 511 ++++++++++++++++++++++++++++
 tests/unittest/llmapi/test_grpc.py        | 315 +++++++++++++++++
 8 files changed, 2273 insertions(+), 5 deletions(-)
```

### Diff Preview

```diff
diff --git a/setup.py b/setup.py
index 795d66899..29f1f532a 100644
--- a/setup.py
+++ b/setup.py
@@ -18,9 +18,52 @@ from pathlib import Path
 from typing import List
 
 from setuptools import find_packages, setup
+from setuptools.command.build_py import build_py
 from setuptools.dist import Distribution
 
 
+class BuildPyWithProtoCompile(build_py):
+    """Custom build_py command that compiles protobuf files before building."""
+
+    def run(self):
+        self.compile_grpc_protos()
+        super().run()
+
+    def compile_grpc_protos(self):
+        """Compile gRPC protobuf files if the proto file exists."""
+        grpc_dir = Path(__file__).parent / "tensorrt_llm" / "grpc"
+        proto_file = grpc_dir / "trtllm_service.proto"
+        compile_script = grpc_dir / "compile_protos.py"
+
+        if not proto_file.exists():
+            return
+
+        # Check if pb2 files need to be generated
+        pb2_file = grpc_dir / "trtllm_service_pb2.py"
+        pb2_grpc_file = grpc_dir / "trtllm_service_pb2_grpc.py"
+
+        # Regenerate if pb2 files don't exist or are older than proto file
+        needs_compile = (not pb2_file.exists() or not pb2_grpc_file.exists() or
+                         pb2_file.stat().st_mtime < proto_file.stat().st_mtime)
+
+        if needs_compile and compile_script.exists():
+            import subprocess
+            import sys
+
+            print("Compiling gRPC protobuf files...")
+            try:
+                subprocess.run(
+                    [sys.executable, str(compile_script)],
+                    check=True,
+                    cwd=str(grpc_dir.parent.parent),
+                )
+                print("gRPC protobuf compilation successful")
+            except subprocess.CalledProcessError as e:
+                print(f"Warning: Failed to compile gRPC protos: {e}")
+            except Exception as e:
+                print(f"Warning: gRPC proto compilation skipped: {e}")
+
+
 def parse_requirements(filename: os.PathLike):
     with open(filename) as f:
         requirements = f.read().splitlines()
@@ -374,6 +417,7 @@ packages += find_packages(include=["triton_kernels", "triton_kernels.*"])
 setup(
     name='tensorrt_llm',
     version=get_version(),
+    cmdclass={'build_py': BuildPyWithProtoCompile},
     description=
     ('TensorRT LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and supports '
      'state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs.'
diff --git a/tensorrt_llm/commands/serve.py b/tensorrt_llm/commands/serve.py
index 02bfbf5f0..76cbde964 100644
--- a/tensorrt_llm/commands/serve.py
+++ b/tensorrt_llm/commands/serve.py
@@ -233,6 +233,124 @@ def launch_server(
         asyncio.run(server(host, port, sockets=[s]))
 
 
+def launch_grpc_server(host: str, port: int, llm_args: dict):
+    """
+    Launch a gRPC server for TensorRT-LLM.
+
+    This provides a high-performance gRPC interface designed for external ro
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## dee6644ed9 - feat(scaffolding): add streaming scaffolding_llm.generate_async support (#5345)

- **Date**: 2025-07-08
- **Author**: Zhenhuan Chen
- **Categories**: Parallelism/Async

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- Parallelism optimization
- Batching optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
.../stream_generation_controller.py                |   3 +-
 .../contrib/Dynasor/scaffolding_dynasor_run.py     |  31 ++-
 examples/scaffolding/run_basic_generation.py       |  31 ++-
 examples/scaffolding/token_budget_majority_vote.py |  13 +-
 tensorrt_llm/scaffolding/__init__.py               |   2 +-
 .../contrib/AsyncGeneration/stream_generation.py   |   2 +-
 .../contrib/Dynasor/dynasor_controller.py          |  24 ++-
 tensorrt_llm/scaffolding/contrib/__init__.py       |  21 --
 tensorrt_llm/scaffolding/controller.py             |  18 +-
 tensorrt_llm/scaffolding/result.py                 |  88 +++++++++
 tensorrt_llm/scaffolding/scaffolding_llm.py        | 215 ++++++++++-----------
 tensorrt_llm/scaffolding/task.py                   |  65 +++++--
 tensorrt_llm/scaffolding/worker.py                 |  20 +-
 13 files changed, 322 insertions(+), 211 deletions(-)
```

### Diff Preview

```diff
diff --git a/examples/scaffolding/contrib/AsyncGeneration/stream_generation_controller.py b/examples/scaffolding/contrib/AsyncGeneration/stream_generation_controller.py
index 87884bd08..05f12ab3c 100644
--- a/examples/scaffolding/contrib/AsyncGeneration/stream_generation_controller.py
+++ b/examples/scaffolding/contrib/AsyncGeneration/stream_generation_controller.py
@@ -3,7 +3,8 @@ from enum import Enum
 from typing import List
 
 from tensorrt_llm.scaffolding import Controller, GenerationTask, Task
-from tensorrt_llm.scaffolding.contrib import StreamGenerationTask
+from tensorrt_llm.scaffolding.contrib.AsyncGeneration import \
+    StreamGenerationTask
 
 
 class NativeStreamGenerationController(Controller):
diff --git a/examples/scaffolding/contrib/Dynasor/scaffolding_dynasor_run.py b/examples/scaffolding/contrib/Dynasor/scaffolding_dynasor_run.py
index 8bf431223..5abd5e86d 100644
--- a/examples/scaffolding/contrib/Dynasor/scaffolding_dynasor_run.py
+++ b/examples/scaffolding/contrib/Dynasor/scaffolding_dynasor_run.py
@@ -1,8 +1,9 @@
 import argparse
+import asyncio
 
 from tensorrt_llm.scaffolding import (MajorityVoteController, ScaffoldingLlm,
                                       TRTLLMWorker)
-from tensorrt_llm.scaffolding.contrib import DynasorGenerationController
+from tensorrt_llm.scaffolding.contrib.Dynasor import DynasorGenerationController
 
 
 def parse_arguments():
@@ -16,13 +17,16 @@ def parse_arguments():
     parser.add_argument("--max_num_tokens", type=int, default=7000)
     parser.add_argument("--majority_vote", action='store_true')
     parser.add_argument('--sample_num', type=int, default=3)
+    parser.add_argument('--streaming', action='store_true')
     args = parser.parse_args()
     return args
 
 
 def test(prompts, proposer_worker, args):
     dynasor_generation_controller = DynasorGenerationController(
-        generation_dir=args.model_dir, max_tokens=args.max_num_tokens)
+        generation_dir=args.model_dir,
+        max_tokens=args.max_num_tokens,
+        streaming=args.streaming)
 
     # If majority voting is requested, wrap the controller in MajorityVoteController
     if args.majority_vote:
@@ -47,9 +51,26 @@ def test(prompts, proposer_worker, args):
             },
         )
 
-    results = llm.generate(prompts)
-    for result in results:
-        print(result.output.output_str)
+    if args.streaming:
+
+        async def task(prompt: str):
+            i = 0
+            async for result in llm.generate_async(prompt):
+                i += 1
+                print(">>>", i, result)
+                async for output in result.cur_output:
+                    print(">>>", i, len(output.outputs[0].token_ids), "\n",
+                          output.outputs[0].text)
+            print(f">>> final output {len(result.outputs[0].token_ids)}\n",
+                  result.outputs[0].text)
+
+        # Need to provide LLM's event loop to get results in the middle of the whole process.
+        asyncio.run_coro
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## df0b976b99 - [https://nvbugs/5785206][infra] Waive TestQwen3_30B_A3B::test_fp8[latency-torch_compile=False]. (#10441)

- **Date**: 2026-01-06
- **Author**: Bo Li
- **Categories**: Throughput/Latency, Quantization Optimization

### Optimization Techniques

- FP8 quantization
- Speculative decoding

### Applicable Conditions

- Disaggregated serving

### Changed Files

```
tests/integration/test_lists/waives.txt | 1 +
 1 file changed, 1 insertion(+)
```

### Diff Preview

```diff
diff --git a/tests/integration/test_lists/waives.txt b/tests/integration/test_lists/waives.txt
index 6e6820e71..49e547bcf 100644
--- a/tests/integration/test_lists/waives.txt
+++ b/tests/integration/test_lists/waives.txt
@@ -507,3 +507,4 @@ accuracy/test_llm_api_pytorch.py::TestLlama3_1_8BInstruct::test_guided_decoding_
 disaggregated/test_auto_scaling.py::test_worker_restart[etcd-round_robin] SKIP (https://nvbugs/5776445)
 accuracy/test_llm_api_pytorch.py::TestGPTOSS::test_eagle3_vswa_reuse_4gpus[one_model] SKIP (https://nvbugs/5756028)
 accuracy/test_llm_api_pytorch.py::TestGPTOSS::test_eagle3_vswa_reuse_4gpus[two_model] SKIP (https://nvbugs/5756028)
+accuracy/test_llm_api_pytorch.py::TestQwen3_30B_A3B::test_fp8[latency-torch_compile=False] SKIP (https://nvbugs/5785206)

```

### Analysis Summary

Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## df3484ddfa - [#11529][perf] AD NemotronH topk router to use the model default dtype (#11623)

- **Date**: 2026-02-23
- **Author**: Eran Geva
- **Categories**: General Performance

### Optimization Techniques

- MoE optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../_torch/auto_deploy/models/custom/modeling_nemotron_h.py      | 9 ++++++---
 1 file changed, 6 insertions(+), 3 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/models/custom/modeling_nemotron_h.py b/tensorrt_llm/_torch/auto_deploy/models/custom/modeling_nemotron_h.py
index b71a0da2f..f87af7e93 100644
--- a/tensorrt_llm/_torch/auto_deploy/models/custom/modeling_nemotron_h.py
+++ b/tensorrt_llm/_torch/auto_deploy/models/custom/modeling_nemotron_h.py
@@ -346,9 +346,12 @@ class NemotronHTopkRouter(nn.Module):
         self.topk_group = config.topk_group
         self.norm_topk_prob = config.norm_topk_prob
 
-        self.weight = nn.Parameter(
-            torch.empty((self.n_routed_experts, config.hidden_size), dtype=torch.float32)
-        )
+        # Do NOT set dtype=torch.float32 here. When loaded via HF from_pretrained(dtype="auto"),
+        # an explicit float32 dtype bypasses the torch.set_default_dtype(bfloat16) context,
+        # keeping the weight in float32 and forcing F.linear with float32 input → slow TF32 GEMM.
+        # Without explicit dtype, the weight inherits the model's BF16 default, enabling the
+        # faster dsv3_router_gemm_op path (BF16 weight → float32 logits via nvjet kernels).
+        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
         self.register_buffer(
             "e_score_correction_bias", torch.zeros(self.n_routed_experts, dtype=torch.float32)
         )

```

### Analysis Summary

GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## e0253ee805 - [None][perf] Disable Swap AB when num tokens exceeds N dimension (#7104)

- **Date**: 2025-08-29
- **Author**: Daniel Stokes
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
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- NVIDIA Hopper (SM90) GPU

### Changed Files

```
.../epilogue/fusion/sm90_visitor_scatter.hpp       | 198 +++++
 .../include/cutlass_extensions/gemm_configs.h      |   9 +-
 .../cutlass_kernels/include/moe_gemm_kernels.h     |  56 +-
 .../kernels/cutlass_kernels/include/moe_kernels.h  |   7 +-
 .../moe_gemm/launchers/moe_gemm_tma_ws_launcher.h  |   2 +-
 .../launchers/moe_gemm_tma_ws_launcher.inl         | 856 +++++++++++----------
 .../moe_gemm_tma_ws_mixed_input_launcher.inl       |  37 +-
 .../moe_gemm/moe_gemm_template_dispatch.h          |  18 +
 .../moe_gemm/moe_gemm_template_dispatch_tma_ws.h   |  14 +-
 .../moe_gemm_tma_warp_specialized_input.cu         |  60 +-
 .../cutlass_kernels/moe_gemm/moe_kernels.cu        | 320 +++-----
 .../cutlass_kernels/python/generate_kernels.py     |  75 +-
 ...orrt_llm_internal_cutlass_kernels_static.tar.xz |   4 +-
 .../aarch64-linux-gnu/version.txt                  |   4 +-
 ...orrt_llm_internal_cutlass_kernels_static.tar.xz |   4 +-
 .../x86_64-linux-gnu/version.txt                   |   4 +-
 .../unit_tests/kernels/mixtureOfExpertsTest.cu     |   9 +-
 17 files changed, 941 insertions(+), 736 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_scatter.hpp b/cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_scatter.hpp
index 5ade62c4b..874dbf0dd 100644
--- a/cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_scatter.hpp
+++ b/cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_scatter.hpp
@@ -373,6 +373,22 @@ struct ScaledAccPerRowBias
   static constexpr bool IsPerRowBiasSupported = true;
 };
 
+template<
+  class ElementOutput_,
+  class ElementCompute_,
+  class ElementBias_ = ElementOutput_,
+  class ElementScalar_ = ElementCompute_,
+  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
+  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
+>
+struct ScaledAccPerColBias
+    : ScaledAcc<ElementOutput_, ElementCompute_, ElementScalar_, RoundStyle_>
+{
+  using ElementBias = ElementBias_;
+  static constexpr int AlignmentBias = AlignmentBias_;
+  static constexpr bool IsPerColBiasSupported = true;
+};
+
 template<
   class GmemLayoutTagOut,
   class ElementOutput,
@@ -393,6 +409,26 @@ struct ScaledAccPerRowBiasPerColScaleScatter
   static constexpr bool IsAuxOutSupported = true;
 };
 
+template<
+  class GmemLayoutTagOut,
+  class ElementOutput,
+  class ElementCompute,
+  class ElementBias = ElementOutput,
+  class ElementScale = ElementCompute,
+  class ElementScalar = ElementCompute,
+  int AlignmentBias = 128 / cute::sizeof_bits_v<ElementBias>,
+  int AlignmentOutput = 128 / cute::sizeof_bits_v<ElementOutput>,
+  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
+>
+struct ScaledAccPerColBiasPerRowScaleScatter
+    : ScaledAccPerColBias<ElementOutput, ElementCompute, ElementBias, ElementScalar, AlignmentBias, RoundStyle>
+{
+  using ElementAux = ElementOutput;
+  using GmemLayoutTagAux = GmemLayoutTagOut;
+  static constexpr int AlignmentAux = AlignmentOutput;
+  static constexpr bool IsAuxOutSupported = true;
+};
+
 // D = alpha * acc + per-row bias
 template<
   class CtaTileShapeMNK,
@@ -410,6 +446,22 @@ using Sm90ScaledAccPerRowBiasPtrArray =
     Sm90ColBroadcast<0, CtaTileShapeMNK, ElementBias *, ElementCompute, Stride<_1,_0,int64_t>, AlignmentBias> // bias
   >;
 
+template<
+  class CtaTileShapeMNK,
+  class ElementOutput,
+  class ElementCompute,
+  class ElementBias = ElementOutput,
+  class ElementScalar = ElementCompute,
+  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
+  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
+>
+using Sm90ScaledAccPerColBiasPtrArray =
+  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // alpha * acc + bias
+    Sm90ScalarBroadcastPtrArray<ElementScalar, Stride<_0,_0,int64_t>>, // alpha
+    Sm90AccFetch, // acc
+    Sm90RowBroadcast<0, CtaTileShapeMNK, ElementBias *, ElementCompute, Stride<_0,_1,int64_t>, AlignmentBia
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## e12868bc00 - [None][fix] Remove and fuse some element-wise ops in the ds-r1-fp8 model (#7238)

- **Date**: 2025-08-27
- **Author**: Fanrong Li
- **Categories**: Fusion, Quantization Optimization

### Optimization Techniques

- Torch compilation/JIT optimization
- Operator fusion
- FP8 quantization
- Integer quantization
- Batching optimization
- Triton kernel
- PyTorch built-in optimized ops
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/models/modeling_deepseekv3.py  | 10 ++-
 tensorrt_llm/_torch/modules/attention.py           | 28 +------
 .../_torch/modules/fused_moe/fused_moe_deepgemm.py | 92 +++++++++++++++++-----
 tensorrt_llm/quantization/utils/fp8_utils.py       |  2 +-
 4 files changed, 82 insertions(+), 50 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_deepseekv3.py b/tensorrt_llm/_torch/models/modeling_deepseekv3.py
index cf4040424..5fdcb43be 100644
--- a/tensorrt_llm/_torch/models/modeling_deepseekv3.py
+++ b/tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -279,10 +279,16 @@ class Deepseekv3RoutingImpl():
         self.routed_scaling_factor = routed_scaling_factor
         self.is_fused = is_fused
 
-    def noaux_tc(self, logits, e_score_correction_bias):
-        n_group = self.n_group
+    @torch.compile(options={"max-autotune": True})
+    def get_scores(self, logits, e_score_correction_bias):
         scores = F.sigmoid(logits)
         scores_with_bias = scores + e_score_correction_bias
+        return scores, scores_with_bias
+
+    def noaux_tc(self, logits, e_score_correction_bias):
+        n_group = self.n_group
+        scores, scores_with_bias = self.get_scores(logits,
+                                                   e_score_correction_bias)
         scores_shape = list(scores_with_bias.shape)
 
         if enable_llm_debug():
diff --git a/tensorrt_llm/_torch/modules/attention.py b/tensorrt_llm/_torch/modules/attention.py
index f4f0d39dc..9b89ffc19 100644
--- a/tensorrt_llm/_torch/modules/attention.py
+++ b/tensorrt_llm/_torch/modules/attention.py
@@ -568,33 +568,7 @@ def fp8_block_scaling_bmm_out(
         torch.ops.trtllm.fp8_block_scaling_bmm_out(mat1_fp8, mat2_fp8,
                                                    mat1_scale, mat2_scale, out)
     elif sm_version == 100:
-        output = torch.bmm(mat1.transpose(0, 1), mat2_dequant.transpose(1, 2))
-        out.copy_(output)
-
-        # low_latency = True
-        # use_deep_seek_fp8 = True
-        # tile_size = 8
-        # epilogue_tile_m = 64 if use_deep_seek_fp8 else 128
-        # m_size = mat1.shape[0]
-        # if m_size % tile_size != 0:
-        #     tiled_shape = ((m_size + tile_size - 1) // tile_size) * tile_size
-        #     mat1 = torch.nn.functional.pad(
-        #         mat1, (0, 0, 0, 0, 0, tiled_shape - m_size), "constant", 0)
-
-        # mat1_fp8, mat1_scale = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
-        #     mat1)
-        # output, output_sf = torch.ops.trtllm.fp8_batched_gemm_trtllmgen(
-        #     mat1_fp8,
-        #     mat2_fp8,
-        #     tile_size=tile_size,
-        #     epilogue_tile_m=epilogue_tile_m,
-        #     use_deep_seek_fp8=use_deep_seek_fp8,
-        #     low_latency=low_latency,
-        #     dq_sfs_a=mat1_scale.reshape(mat1.shape[-1] // 128, -1),
-        #     dq_sfs_b=mat2_scale,
-        #     out_dtype=out.dtype,
-        # )
-        # out.copy_(output[:, :m_size])
+        torch.bmm(mat1.transpose(0, 1), mat2_dequant.transpose(1, 2), out=out)
     else:
         raise NotImplementedError(f"SM{sm_version} is not supported")
 
diff --git a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py
index 79af3cb7f..4b32f
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## e134a52e07 - Perf: reduce DeepEPLowLatency memory and time (#5712)

- **Date**: 2025-07-04
- **Author**: Tailing Yuan
- **Categories**: Memory Optimization, Throughput/Latency

### Optimization Techniques

- Operator fusion
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../_torch/modules/fused_moe/fused_moe_wide_ep.py  | 39 ++++++++++++++--------
 1 file changed, 26 insertions(+), 13 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_wide_ep.py b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_wide_ep.py
index 86f351ee6..dbb0f03f4 100755
--- a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_wide_ep.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_wide_ep.py
@@ -416,30 +416,28 @@ class WideEPMoE(MoE):
                     deep_ep_topk_weights = token_final_scales
                     x, recv_expert_count, deep_ep_handle = \
                         self.deep_ep_buffer.low_latency_dispatch(x, deep_ep_topk_idx, self.deep_ep_max_num_tokens, self.num_slots)
-                    # x shape: [#local experts, #max recv tokens, hidden_size]
+                    # x shape: [#local experts, EP size * deep_ep_max_num_tokens, hidden_size]
                     # recv_expert_count shape: [#local experts]
 
                     # Adapter between `torch.ops.trtllm.fused_moe` and DeepEP
                     # TODO: remove the adapter by changing `torch.ops.trtllm.fused_moe` API
+                    x = x[:, :self.mapping.moe_ep_size *
+                          all_rank_max_num_tokens]
                     mask = torch.arange(
                         x.shape[1], dtype=torch.int32, device=x.device).expand(
                             x.shape[0],
                             x.shape[1]) < recv_expert_count.unsqueeze(1)
-                    token_selected_slots = torch.full(
-                        (x.shape[0], x.shape[1], self.routing_method.top_k),
-                        self.num_slots,
-                        dtype=torch.int32,
-                        device=x.device)
-                    token_selected_slots[:, :, 0] = torch.where(
+                    token_selected_slots = torch.where(
                         mask,
                         torch.arange(
                             x.shape[0] * self.mapping.moe_ep_rank,
                             x.shape[0] * (self.mapping.moe_ep_rank + 1),
                             dtype=torch.int32,
                             device=x.device).unsqueeze(1), self.num_slots)
-                    x = x.view(x.shape[0] * x.shape[1], x.shape[2])
+                    x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
+                    # Cheat the fused_moe API with fake top_k=1
                     token_selected_slots = token_selected_slots.view(
-                        x.shape[0], self.routing_method.top_k)
+                        x.shape[0], 1)
                     token_final_scales = torch.ones_like(
                         token_selected_slots, dtype=token_final_scales.dtype)
             else:
@@ -640,12 +638,27 @@ class WideEPMoE(MoE):
                 final_hidden_states = self.deep_ep_buffer.combine(
                     final_hidden_states, deep_ep_handle)
             elif self.alltoall_method_type == AlltoallMethodType.DeepEPLowLatency:
-                final_hidden_states = self.deep_ep_buffer.low_latency_combine(
-                    final_hidden_states.view(

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## e18dacc931 - [#4403][refactor] Move fusion, kvcache, and compile to modular inference optimizer (#7057)

- **Date**: 2025-08-21
- **Author**: Fridah-nv
- **Categories**: Fusion, Cache Optimization

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- FP8 quantization
- Integer quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Triton kernel
- PyTorch built-in optimized ops
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../_torch/auto_deploy/config/default.yaml         |  49 +-
 .../_torch/auto_deploy/transform/interface.py      |   6 +-
 .../auto_deploy/transform/library/collectives.py   | 204 ++++++++
 .../auto_deploy/transform/library/compile_model.py |  65 +++
 .../library/fusion.py                              |  64 +--
 .../auto_deploy/transform/library/kvcache.py       | 299 ++++++++++++
 .../auto_deploy/transform/library/rms_norm.py      | 148 ++++++
 .../transformations/library/__init__.py            |   6 -
 .../transformations/library/collectives.py         | 167 -------
 .../transformations/library/fused_moe.py           | 511 ---------------------
 .../auto_deploy/transformations/library/kvcache.py | 193 --------
 .../transformations/library/rms_norm.py            | 113 -----
 .../auto_deploy/transformations/transform.py       | 128 ++----
 .../test_allreduce_residual_rmsnorm_fusion.py      |  21 +-
 .../library/test_collective_fusion.py              |  19 +-
 .../transformations/library/test_fuse_rmsnorm.py   |  30 +-
 .../transformations/library/test_gemm_fusion.py    |  19 +-
 .../transformations/library/test_kv_cache.py       | 117 +++--
 18 files changed, 969 insertions(+), 1190 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/config/default.yaml b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
index f7ad7934a..041d51e73 100644
--- a/tensorrt_llm/_torch/auto_deploy/config/default.yaml
+++ b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
@@ -19,6 +19,11 @@ transforms:
     stage: post_export
   cleanup_input_constraints:
     stage: post_export
+  ############################################################################################
+  # RUN PATTERN MATCHER TRANSFORMATIONS TO STANDARDIZE GRAPH REPRESENTATION
+  ############################################################################################
+  match_moe_pattern:
+    stage: pattern_matcher
   match_repeat_kv:
     stage: pattern_matcher
   match_eager_attention:
@@ -27,12 +32,13 @@ transforms:
     stage: pattern_matcher
   match_attention_layout:
     stage: pattern_matcher
-  match_moe_pattern:
-    stage: pattern_matcher
   match_rope_pattern:
     stage: pattern_matcher
   match_rope_layout:
     stage: pattern_matcher
+  ############################################################################################
+  # RUN TRANSFORMATIONS ON STANDARDIZED GRAPH REPRESENTATION
+  ############################################################################################
   eliminate_redundant_transposes:
     stage: pattern_matcher
   # TODO (lucaslie): let's move this to perf optimization once TP sharding is improved
@@ -57,5 +63,44 @@ transforms:
   sharding_transform_executor:
     stage: sharding
     run_shape_prop: true
+  ############################################################################################
+  # MOVE MODEL AND LOAD WEIGHTS
+  ############################################################################################
   load_weights:
     stage: weight_load
+  ############################################################################################
+  # RUN POST-LOAD FUSION AND OPTIMIZATIONS
+  ############################################################################################
+  # TODO: https://github.com/NVIDIA/TensorRT-LLM/issues/4674 this is causing OOMs
+  # fuse_moe:
+  #   stage: post_load_fusion
+  # fuse_gemms:
+  #   stage: post_load_fusion
+  fuse_allreduce_residual_rmsnorm:
+    stage: post_load_fusion
+  fuse_collectives:
+    stage: post_load_fusion
+  # TODO (lucaslie): add backend selection as part of configurable inference optimizers
+  # check if we can fuse rmsnorm
+  fuse_rmsnorm:
+    stage: post_load_fusion
+    backend: flashinfer
+  ############################################################################################
+  # SWITCH TO CACHED+FLATTENED ATTENTION + INITIALIZE CACHES
+  ############################################################################################
+  update_in_out_nodes:
+    stage: cache_init
+  insert_cached_attention:
+    stage: cache_init
+  insert_cached_mla_attention:
+    stage: cache_init
+    attn_backend: MultiHeadLatentAttention
+  initia
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## e405468230 - [TRTLLM-10048][feat] Fuse the AllGather for expert statistics required by the EPLB. (#10885)

- **Date**: 2026-01-26
- **Author**: Bo Li
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
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../communicationKernels/moeAlltoAllKernels.cu     |  76 +++++++----
 .../communicationKernels/moeAlltoAllKernels.h      |  18 ++-
 cpp/tensorrt_llm/thop/moeAlltoAllMeta.h            |   6 +-
 cpp/tensorrt_llm/thop/moeAlltoAllOp.cpp            |  94 +++++++++++--
 tensorrt_llm/_torch/distributed/moe_alltoall.py    |  88 +++++++++---
 .../_torch/modules/fused_moe/communication/base.py |   6 +
 .../communication/communication_factory.py         |   5 +
 .../fused_moe/communication/nvlink_one_sided.py    | 117 +++++++++++-----
 .../_torch/modules/fused_moe/configurable_moe.py   |  31 ++++-
 .../_torch/modules/fused_moe/fused_moe_cutlass.py  |  52 ++++++--
 .../modules/fused_moe/fused_moe_trtllm_gen.py      |  54 ++++++--
 tests/unittest/_torch/multi_gpu/test_moe_a2a.py    | 147 +++++++++++++--------
 12 files changed, 508 insertions(+), 186 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu b/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu
index 303255e6a..da1aed6a3 100644
--- a/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu
+++ b/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu
@@ -361,19 +361,15 @@ __global__ void moeA2APrepareDispatchKernel(
 }
 
 // ============================================================================
-// Generic Dispatch Kernel Implementation
-// One warp per token design:
-// - Each CTA has 256 threads = 8 warps
-// - Each warp independently processes one token and all its payloads
-// - Better GPU utilization and reduced synchronization overhead
+// Dispatch Kernels
 // ============================================================================
 
-template <typename ThreadingPolicy, int TOP_K>
+template <typename ThreadingPolicy, int TOP_K, bool ENABLE_EPLB>
 __global__ void moeA2ADispatchKernel(int32_t const* token_selected_experts, // [local_num_tokens, TOP_K]
     const DispatchKernelPointers ptrs,                                      // Struct containing all kernel pointers
     int num_payloads,                                                       // Number of payloads
     int max_tokens_per_rank,                                                // Maximum tokens per rank
-    int local_num_tokens, int rank_id, int ep_size, int num_experts_per_rank)
+    int local_num_tokens, int rank_id, int ep_size, int num_experts, int eplb_stats_num_experts)
 {
 
     int thread_idx = ThreadingPolicy::offset();
@@ -411,6 +407,7 @@ __global__ void moeA2ADispatchKernel(int32_t const* token_selected_experts, // [
         }
 
         uint64_t already_copied = 0;
+        int num_experts_per_rank = num_experts / ep_size;
         for (int k = 0; k < TOP_K; k++)
         {
             int expert_id = token_selected_experts[local_token_idx * TOP_K + k];
@@ -501,6 +498,21 @@ __global__ void moeA2ADispatchKernel(int32_t const* token_selected_experts, // [
                 ptrs.recv_counters[target_rank][rank_id] = send_count;
             }
 
+            if constexpr (ENABLE_EPLB)
+            {
+                // Write local stats into peer buffers before the release fence below.
+#pragma unroll 1
+                for (int target_rank = 0; target_rank < ep_size; ++target_rank)
+                {
+                    int* target_stats = ptrs.eplb_gathered_stats[target_rank];
+                    for (int expert_id = lane_id; expert_id < eplb_stats_num_experts; expert_id += warpSize)
+                    {
+                        int stat_val = ptrs.eplb_local_stats[expert_id];
+                        target_stats[rank_id * eplb_stats_num_experts + expert_id] = stat_val;
+                    }
+                }
+            }
+
 #if !DISABLE_SYNC_FOR_PROFILING
             uint32_t expected_value = *ptrs.flag_val;
 
@@ -588,6 +600,7 @@ void moe_a2a_dispatch_launch(MoeA2ADi
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## e4bf29bc66 - [None][feat] Integrate MnnvlThroughput into TRTLLM MoE. (#8728)

- **Date**: 2025-11-04
- **Author**: Bo Li
- **Categories**: Throughput/Latency

### Optimization Techniques

- Custom CUDA kernel
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

### Changed Files

```
.../communicationKernels/moeAlltoAllKernels.cu     |   6 +-
 .../communicationKernels/moeAlltoAllKernels.h      |  66 +--
 cpp/tensorrt_llm/nanobind/thop/bindings.cpp        |   2 +-
 cpp/tensorrt_llm/pybind/thop/bindings.cpp          |   2 +-
 cpp/tensorrt_llm/thop/moeAlltoAllMeta.h            |  33 +-
 cpp/tensorrt_llm/thop/moeAlltoAllOp.cpp            | 493 ++++++++++-----------
 cpp/tensorrt_llm/thop/mxFp4BlockScaleMoe.cpp       |  31 +-
 .../_torch/custom_ops/trtllm_gen_custom_ops.py     |  12 +-
 tensorrt_llm/_torch/distributed/moe_alltoall.py    | 181 ++++----
 .../_torch/modules/fused_moe/fused_moe_cutlass.py  |  43 +-
 .../modules/fused_moe/fused_moe_trtllm_gen.py      | 205 ++++++---
 tests/unittest/_torch/multi_gpu/test_moe_a2a.py    | 310 ++++++-------
 12 files changed, 736 insertions(+), 648 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu b/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu
index b31f6bb74..144aadbc7 100644
--- a/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu
+++ b/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu
@@ -23,7 +23,7 @@
 #include <cstdint>
 #include <type_traits>
 
-namespace tensorrt_llm::kernels::moe_a2a
+namespace tensorrt_llm::kernels::mnnvl_throughput
 {
 
 #define ENABLE_DEBUG_PRINT 0
@@ -506,7 +506,7 @@ void moe_a2a_dispatch_launch(MoeA2ADispatchParams const& params)
     TLLM_CHECK(params.num_payloads > 0 && params.num_payloads <= kMaxPayloads);
 
     // Prepare kernel pointers struct
-    DispatchKernelPointers kernel_ptrs = {}; // Zero-initialize
+    DispatchKernelPointers kernel_ptrs = {};
 
     // Fill source data pointers and payload sizes
     for (int i = 0; i < params.num_payloads; i++)
@@ -958,4 +958,4 @@ void moe_a2a_sanitize_expert_ids_launch(int32_t* expert_ids, int32_t const* recv
         expert_ids, recv_counters, ep_size, max_tokens_per_rank, top_k, invalid_id);
 }
 
-} // namespace tensorrt_llm::kernels::moe_a2a
+} // namespace tensorrt_llm::kernels::mnnvl_throughput
diff --git a/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h b/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h
index ad8fae07b..27b6f926d 100644
--- a/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h
+++ b/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h
@@ -19,7 +19,7 @@
 #include <cuda_bf16.h>
 #include <cuda_fp16.h>
 
-namespace tensorrt_llm::kernels::moe_a2a
+namespace tensorrt_llm::kernels::mnnvl_throughput
 {
 
 // Configuration constants
@@ -91,7 +91,7 @@ struct MoeA2ADispatchParams
 
     // Token configuration
     int local_num_tokens;    // Number of tokens on this rank
-    int max_tokens_per_rank; // Maximum tokens per rank for pre-allocation
+    int max_tokens_per_rank; // Maximum tokens per rank for pre-allocation TODO: Rename to runtime_max_tokens_per_rank
     int top_k;               // Number of experts per token
 
     // Expert routing information
@@ -101,23 +101,22 @@ struct MoeA2ADispatchParams
     int num_payloads;                         // Number of different payload types
     PayloadDescriptor payloads[kMaxPayloads]; // Array of payload descriptors
 
-    // Receive buffers and synchronization
-    void* recv_buffers[kMaxRanks][kMaxPayloads]; // Per-rank receive buffers for each payload
+    // Local aux data
+    uint32_t* flag_val;       // The value of the flag for this round (stored on the local rank)
+    int* local_token_counter; // Atomic counter for completed tokens on this rank
+    int* send_counters;       // [ep_size] atomic counters - tracks tokens sent to each target rank
+    int* topk_target_ranks; // Top-K compact routing info per local token (size: [local_num_tokens, top_k]), target rank
+                   
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## e57d83c5dc - [TRTLLM-8768][chore] Fuse QK down_proj with indexer K + weight_proj for FP4 ckpt (#8771)

- **Date**: 2025-11-05
- **Author**: Chang Liu
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
- Speculative decoding
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- Prefill phase
- Decode/generation phase

### Changed Files

```
.../_torch/attention_backend/sparse/dsa.py         | 86 ++++++++++++--------
 tensorrt_llm/_torch/models/modeling_deepseekv3.py  | 92 ++++++++++++++++++++--
 tensorrt_llm/_torch/modules/attention.py           | 40 ++++++----
 .../attention/sparse/test_sparse_mla_forward.py    | 62 +++------------
 4 files changed, 175 insertions(+), 105 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/attention_backend/sparse/dsa.py b/tensorrt_llm/_torch/attention_backend/sparse/dsa.py
index 080976fa1..8fa57e201 100644
--- a/tensorrt_llm/_torch/attention_backend/sparse/dsa.py
+++ b/tensorrt_llm/_torch/attention_backend/sparse/dsa.py
@@ -629,6 +629,7 @@ class Indexer(nn.Module):
         self.scale_fmt = "ue8m0"
         self.aux_stream = aux_stream
         self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]
+        self.weight_scale_factor = self.softmax_scale * self.n_heads**-0.5
 
     @staticmethod
     def prepare_one_prefill_chunk(
@@ -1105,65 +1106,86 @@ class Indexer(nn.Module):
 
         return topk_indices_buffer
 
+    def weight_scale(self, hidden_states: torch.Tensor,
+                     indexer_weights: Optional[torch.Tensor],
+                     q_scale: torch.Tensor) -> torch.Tensor:
+        weights = indexer_weights if indexer_weights is not None else self.weights_proj(
+            hidden_states)
+        weights = weights.unsqueeze(-1) * q_scale * self.weight_scale_factor
+        # output weights is guaranteed to be float32 due to type promotion from q_scale (float32)
+        weights = weights.squeeze(-1)
+        return weights
+
     @torch.inference_mode()
     def forward(self, qr: torch.Tensor, hidden_states: torch.Tensor,
                 metadata: DSAtrtllmAttentionMetadata,
-                position_ids: torch.Tensor):
+                position_ids: torch.Tensor, indexer_k: Optional[torch.Tensor],
+                indexer_weights: Optional[torch.Tensor]):
         quant_block_size = metadata.kv_cache_manager.quant_block_size
         assert quant_block_size == 128, "Only support quant_block_size = 128 for now"
 
+        if indexer_k is not None:
+            q, k = maybe_execute_in_parallel(
+                lambda: self.wq_b(
+                    qr),  # TODO: fuse wq_b and move this outside of the indexer
+                lambda: self.k_norm(indexer_k),
+                self.ln_events[0],
+                self.ln_events[1],
+                self.aux_stream,
+            )
+        else:
+            q, k = maybe_execute_in_parallel(
+                lambda: self.wq_b(qr),
+                lambda: self.k_norm(self.wk(hidden_states)),
+                self.ln_events[0],
+                self.ln_events[1],
+                self.aux_stream,
+            )
+
+        # q/k rope + possible fast_hadamard_transform
+        q = q.view(-1, self.n_heads, self.head_dim)
+
         q, k = maybe_execute_in_parallel(
-            lambda: self.wq_b(qr),
-            lambda: self.wk(hidden_states),
+            lambda: torch.split(
+                q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1),
+            lambda: torch.split(
+                k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1),
             self.ln_events[0],
             self.ln_events[1],
             self.aux_stream,
         )
-        q = q.view(-1, self.n_heads, self.head_dim)
-        q_
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## e5d4305c04 - [https://nvbugs/5467531][fix] Unwaive fused_moe all to all test with … (#9617)

- **Date**: 2025-12-04
- **Author**: Jin Li
- **Categories**: Fusion

### Optimization Techniques

- Operator fusion
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/modules/fused_moe/fused_moe_wide_ep.py | 3 ++-
 tests/unittest/_torch/modules/test_fused_moe.py            | 4 ----
 2 files changed, 2 insertions(+), 5 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_wide_ep.py b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_wide_ep.py
index 62fee4785..8fab75d77 100755
--- a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_wide_ep.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_wide_ep.py
@@ -966,6 +966,7 @@ class WideEPMoE(MoE):
         output_dtype: Optional[torch.dtype] = None,
         all_rank_num_tokens: Optional[List[int]] = None,
         use_dp_padding: Optional[bool] = None,
+        alltoall_result_do_sum: bool = True,
         **kwargs,
     ) -> Union[torch.Tensor, List[torch.Tensor]]:
         moe_output = super().forward_fake(
@@ -976,7 +977,7 @@ class WideEPMoE(MoE):
             all_rank_num_tokens=all_rank_num_tokens,
             use_dp_padding=use_dp_padding,
             **kwargs)
-        if self.alltoall_method_type == AlltoallMethodType.MNNVL:
+        if self.alltoall_method_type == AlltoallMethodType.NVLinkTwoSided and not alltoall_result_do_sum:
             shape = moe_output.shape
             top_k = self.routing_method.experts_per_token
             new_shape = [shape[0], top_k, shape[1]]
diff --git a/tests/unittest/_torch/modules/test_fused_moe.py b/tests/unittest/_torch/modules/test_fused_moe.py
index 14acccf45..cea8ff57e 100644
--- a/tests/unittest/_torch/modules/test_fused_moe.py
+++ b/tests/unittest/_torch/modules/test_fused_moe.py
@@ -335,10 +335,6 @@ def test_fused_moe_alltoall(alltoall_method_type):
 ],
                          ids=lambda s: s.name)
 def test_fused_moe_alltoall_fp4(alltoall_method_type):
-
-    if alltoall_method_type == AlltoallMethodType.DeepEPLowLatency:
-        pytest.skip("Skipped due to https://nvbugs/5467531")
-
     world_size = 4
     dtype = torch.bfloat16
     HIDDEN_SIZE = 4096

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## e67f4da9b5 - [Perf]: Add residual, norm for nemotron_nas models (#6455)

- **Date**: 2025-07-30
- **Author**: NVShreyas
- **Categories**: General Performance

### Optimization Techniques

- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
.../_torch/models/modeling_nemotron_nas.py         | 54 +++++++++++++++++++---
 1 file changed, 47 insertions(+), 7 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_nemotron_nas.py b/tensorrt_llm/_torch/models/modeling_nemotron_nas.py
index 3ab1cdb37..cbd5ff4a9 100644
--- a/tensorrt_llm/_torch/models/modeling_nemotron_nas.py
+++ b/tensorrt_llm/_torch/models/modeling_nemotron_nas.py
@@ -149,12 +149,17 @@ class NemotronNASDecoderLayer(DecoderLayer):
         position_ids: torch.IntTensor,
         hidden_states: torch.Tensor,
         attn_metadata: AttentionMetadata,
+        residual: Optional[torch.Tensor] = None,
         **kwargs,
     ) -> torch.Tensor:
         if not self.block_config.attention.no_op:
             # Self Attention
-            residual = hidden_states
-            hidden_states = self.input_layernorm(hidden_states)
+            if residual is None:
+                residual = hidden_states
+                hidden_states = self.input_layernorm(hidden_states)
+            else:
+                hidden_states, residual = self.input_layernorm(
+                    hidden_states, residual)
 
             hidden_states = self.self_attn(
                 position_ids=position_ids,
@@ -162,16 +167,18 @@ class NemotronNASDecoderLayer(DecoderLayer):
                 attn_metadata=attn_metadata,
                 **kwargs,
             )
-            hidden_states = residual + hidden_states
 
         if not self.block_config.ffn.no_op:
             # Fully Connected
-            residual = hidden_states
-            hidden_states = self.post_attention_layernorm(hidden_states)
+            if residual is None:
+                residual = hidden_states
+                hidden_states = self.post_attention_layernorm(hidden_states)
+            else:
+                hidden_states, residual = self.post_attention_layernorm(
+                    hidden_states, residual)
             hidden_states = self.mlp(hidden_states, **kwargs)
-            hidden_states = residual + hidden_states
 
-        return hidden_states
+        return hidden_states, residual
 
 
 class NemotronNASModel(DecoderModel):
@@ -225,6 +232,39 @@ class NemotronNASModel(DecoderModel):
                             eps=config.rms_norm_eps,
                             dtype=config.torch_dtype)
 
+    def forward(
+        self,
+        attn_metadata: AttentionMetadata,
+        input_ids: Optional[torch.IntTensor] = None,
+        position_ids: Optional[torch.IntTensor] = None,
+        inputs_embeds: Optional[torch.FloatTensor] = None,
+        lora_params: Optional[dict] = None,
+        **kwargs,
+    ) -> torch.Tensor:
+
+        if (input_ids is None) ^ (inputs_embeds is not None):
+            raise ValueError(
+                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
+            )
+
+        if inputs_embeds is None:
+            inputs_embeds = self.embed_tokens(input_ids)
+
+        hidden_states = inputs_embeds
+        residual = None
+
+        for decoder_layer in self.layers:
+            hidden_states, residual 
```

### Analysis Summary

Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## e6b482ef47 - fix: change the seq_lens sync copy to an async one (#3786)

- **Date**: 2025-04-29
- **Author**: Fanrong Li
- **Categories**: Parallelism/Async

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
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
tensorrt_llm/_torch/attention_backend/interface.py |  2 +-
 tensorrt_llm/_torch/pyexecutor/model_engine.py     | 38 ++++++++++++----------
 tensorrt_llm/_torch/speculative/interface.py       |  3 ++
 3 files changed, 24 insertions(+), 19 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/attention_backend/interface.py b/tensorrt_llm/_torch/attention_backend/interface.py
index 7807d99f2..d0dc6e5e1 100644
--- a/tensorrt_llm/_torch/attention_backend/interface.py
+++ b/tensorrt_llm/_torch/attention_backend/interface.py
@@ -167,7 +167,7 @@ class AttentionMetadata:
                 # This copy is safe because the batch size is guaranteed to not
                 # change in the CUDA graph case. The seqlens can change if we
                 # are doing spec decode.
-                self._seq_lens_cuda.copy_(self._seq_lens)
+                self._seq_lens_cuda.copy_(self._seq_lens, non_blocking=True)
             else:
                 self._seq_lens_cuda = self._seq_lens.cuda(non_blocking=True)
 
diff --git a/tensorrt_llm/_torch/pyexecutor/model_engine.py b/tensorrt_llm/_torch/pyexecutor/model_engine.py
index 076c1343e..6aea374cd 100644
--- a/tensorrt_llm/_torch/pyexecutor/model_engine.py
+++ b/tensorrt_llm/_torch/pyexecutor/model_engine.py
@@ -244,6 +244,7 @@ class PyTorchModelEngine(ModelEngine):
         self.dist = dist
         self.pytorch_backend_config = pytorch_backend_config
         self.spec_config = spec_config
+        self.is_spec_decode = spec_config is not None
         # We keep a reference to the last used spec metadata to
         # accommodate certain target/draft model use cases. See
         # py_executor.py for how this is used.
@@ -342,7 +343,7 @@ class PyTorchModelEngine(ModelEngine):
         self.position_ids_cuda = torch.empty((self.max_num_tokens, ),
                                              dtype=torch.int,
                                              device='cuda')
-        if self.spec_config is not None:
+        if self.is_spec_decode:
             self.spec_metadata = None
             self.spec_config.update_from_model_config(self.model.config)
             max_num_draft_tokens = self.spec_config.max_draft_tokens * batch_size
@@ -393,7 +394,7 @@ class PyTorchModelEngine(ModelEngine):
         def get_cuda_graph_warmup_request(batch_size):
             available_blocks = kv_cache_manager.get_num_free_blocks()
 
-            max_num_draft_tokens = self.spec_config.max_draft_tokens if self.spec_config is not None else 0
+            max_num_draft_tokens = self.spec_config.max_draft_tokens if self.is_spec_decode else 0
 
             if available_blocks >= batch_size:
                 result = ScheduledRequests()
@@ -449,7 +450,7 @@ class PyTorchModelEngine(ModelEngine):
                     num_tokens_per_request / kv_cache_manager.tokens_per_block):
                 # Should only need (at most) one more page per request.
                 is_gen = num_tokens_per_request == 1
-                max_num_draft_tokens = self.spec_config.max_draft_tokens if self.spec_config is not None and is_gen else 0
+                max_num_draft_tokens = self.spec_config.max_draft_tokens if self.is_spec_decode and is_gen else 0
 
                 requests = kv_cache_manager.add_dummy_r
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Pinned (page-locked) memory enables faster host-to-device data transfers. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## e8ad899f93 - [None][feat] TRT-LLM Gen MoE finalize kernel optimization (#11501)

- **Date**: 2026-03-01
- **Author**: Nikita Korobov
- **Categories**: Kernel Optimization

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Async/stream-based execution
- FP8 quantization
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../trtllmGenKernels/blockScaleMoe/DevKernel.cu    | 13 +--
 .../trtllmGenKernels/blockScaleMoe/DevKernel.h     | 99 +++++++++++++++++++---
 2 files changed, 93 insertions(+), 19 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.cu b/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.cu
index cef2588b5..15e642e5b 100644
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.cu
+++ b/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.cu
@@ -782,11 +782,12 @@ __global__ void finalizeKernelVecLoad(KernelParams params)
     using OutputElem = cutlass::Array<Type, FINALIZE_ELEM_PER_THREAD>;
     using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
 
+    int64_t const hiddenBlockIdx = blockIdx.y;
     int64_t const tokenIdx = blockIdx.x;
-    int64_t const startOffset = threadIdx.x;
+    int64_t const startOffset = threadIdx.x + hiddenBlockIdx * params.hiddenDimPerBlock / FINALIZE_ELEM_PER_THREAD;
     int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
     int64_t const numElemsInPaddedCol = params.hiddenDimPadded / FINALIZE_ELEM_PER_THREAD;
-    int64_t const numElemsInCol = params.hiddenDim / FINALIZE_ELEM_PER_THREAD;
+    int64_t const numElemsInColPerBlock = (hiddenBlockIdx + 1) * params.hiddenDimPerBlock / FINALIZE_ELEM_PER_THREAD;
 
     auto const offset = tokenIdx * params.hiddenDim;
     Type* outputPtr = params.outPtr + offset;
@@ -801,7 +802,7 @@ __global__ void finalizeKernelVecLoad(KernelParams params)
     }
 #endif
 
-    for (int elemIndex = startOffset; elemIndex < numElemsInCol; elemIndex += stride)
+    for (int elemIndex = startOffset; elemIndex < numElemsInColPerBlock; elemIndex += stride)
     {
         ComputeElem threadOutput;
         threadOutput.fill(0);
@@ -916,7 +917,7 @@ void run(Data const& data, void* stream)
         int const numBlocksY = std::min(8192, data.numTokens);
         dim3 numBlocks(numBlocksX, numBlocksY);
 
-        LAUNCH_EXPW(data, finalizeDeepSeekKernel, numBlocks, numThreads, 0, stream);
+        LAUNCH_EXPW(data, finalizeDeepSeekKernel, false, numBlocks, numThreads, 0, stream);
     }
     else
     {
@@ -933,11 +934,11 @@ void run(Data const& data, void* stream)
             // This limitation is intended to ensure that when the number of waves is greater than 1, we choose to use
             // the kernel with vectorized loading.
             dim3 numBlocks(numBlocksX, numBlocksY);
-            LAUNCH_EXPW(data, finalizeKernel, numBlocks, numThreads, 0, stream);
+            LAUNCH_EXPW(data, finalizeKernel, false, numBlocks, numThreads, 0, stream);
         }
         else
         {
-            LAUNCH_EXPW(data, finalizeKernelVecLoad, /*numBlocks=*/data.numTokens,
+            LAUNCH_EXPW(data, finalizeKernelVecLoad, true, /*numBlocks=*/data.numTokens,
                 /*numThreads=*/FINALIZE_THREADS_PER_BLOCK, 0, stream);
         }
     }
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.h b/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.h
index 7edc3d195..8da081b09 100644
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScal
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## ea3739ee62 - Fix: fuse message not aligned on different processes (#3067)

- **Date**: 2025-03-26
- **Author**: Kaiyu Xie
- **Categories**: Fusion

### Optimization Techniques

- Operator fusion
- Async/stream-based execution
- Batching optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/executor/postproc_worker.py | 4 +++-
 tensorrt_llm/executor/worker.py          | 5 +++--
 2 files changed, 6 insertions(+), 3 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/executor/postproc_worker.py b/tensorrt_llm/executor/postproc_worker.py
index 38fe93e6d..b3347a09c 100644
--- a/tensorrt_llm/executor/postproc_worker.py
+++ b/tensorrt_llm/executor/postproc_worker.py
@@ -158,7 +158,9 @@ class PostprocWorker:
 
         async def handle_single_input(inp: PostprocWorker.Input,
                                       batch: List[PostprocWorker.Output]):
-            assert isinstance(inp, PostprocWorker.Input)
+            assert isinstance(
+                inp, PostprocWorker.Input
+            ), f"Expect PostprocWorker.Input, got {type(inp)}."
             client_id = inp.rsp.client_id
             is_final = inp.rsp.result.is_final if isinstance(
                 inp.rsp, tllm.Response) else True
diff --git a/tensorrt_llm/executor/worker.py b/tensorrt_llm/executor/worker.py
index 92a3fd9d3..2504f6e5c 100644
--- a/tensorrt_llm/executor/worker.py
+++ b/tensorrt_llm/executor/worker.py
@@ -552,7 +552,7 @@ def worker_main(
             # processes, each one is a PAIR zmq socket
             result_queues = [
                 FusedIpcQueue(is_server=True,
-                              fuse_message=True,
+                              fuse_message=not BATCH_RESP_IN_AWAIT,
                               name=f"postprocess_{i}_feedin_queue")
                 for i in range(postproc_worker_config.num_postprocess_workers)
             ]
@@ -803,7 +803,8 @@ class AwaitResponseHelper:
 
         if postproc_batches:
             for wid, batch in enumerate(postproc_batches):
-                self.worker.postproc_queues[wid].put(batch)
+                if batch:
+                    self.worker.postproc_queues[wid].put(batch)
 
         if rsp_batch:
             self.worker.result_queue.put(rsp_batch)

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## ead89a0e40 - [None][perf] Improve the performance of online EPLB on Hopper by better overlapping (#6624)

- **Date**: 2025-08-12
- **Author**: Jinyang Yuan
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- Parallelism optimization
- Triton kernel
- PyTorch built-in optimized ops
- Speculative decoding
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/models/modeling_deepseekv3.py  |   5 +-
 tensorrt_llm/_torch/models/modeling_mixtral.py     |   3 +-
 tensorrt_llm/_torch/models/modeling_qwen3_moe.py   |   3 +-
 tensorrt_llm/_torch/models/modeling_qwen_moe.py    |   3 +-
 .../_torch/modules/fused_moe/create_moe.py         |  14 +-
 .../_torch/modules/fused_moe/fused_moe_cute_dsl.py |  11 +-
 .../_torch/modules/fused_moe/fused_moe_cutlass.py  |  14 +-
 .../_torch/modules/fused_moe/fused_moe_deepgemm.py |  11 +-
 .../_torch/modules/fused_moe/fused_moe_triton.py   |   1 -
 .../modules/fused_moe/fused_moe_trtllm_gen.py      |   1 -
 .../_torch/modules/fused_moe/fused_moe_wide_ep.py  | 116 ++++----
 tensorrt_llm/_torch/modules/fused_moe/interface.py |   1 -
 .../_torch/modules/fused_moe/moe_load_balancer.py  | 304 +++++++++++++--------
 tensorrt_llm/_torch/utils.py                       |   7 +-
 .../_torch/modules/test_moe_load_balancer.py       |  31 ++-
 15 files changed, 299 insertions(+), 226 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_deepseekv3.py b/tensorrt_llm/_torch/models/modeling_deepseekv3.py
index 07045489e..a5a61d9a7 100644
--- a/tensorrt_llm/_torch/models/modeling_deepseekv3.py
+++ b/tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -453,7 +453,7 @@ class Deepseekv3MoE(nn.Module):
             False,  # In both low‑latency and attention‑DP modes, FusedMoE skips the in‑op all‑reduce.
             model_config=model_config,
             override_quant_config=override_quant_config,
-            aux_stream=aux_stream_dict[AuxStreamType.MoeChunkingOverlap],
+            aux_stream_dict=aux_stream_dict,
             layer_idx=layer_idx)
 
         self.mapping = model_config.mapping
@@ -1049,11 +1049,12 @@ class DeepseekV3Model(DecoderModel):
         config = model_config.pretrained_config
         self.vocab_size = config.vocab_size
         self.num_hidden_layers = config.num_hidden_layers
-        aux_stream_list = [torch.cuda.Stream() for _ in range(2)]
+        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
         self.aux_stream_dict = {
             AuxStreamType.Attention: aux_stream_list[0],
             AuxStreamType.MoeShared: aux_stream_list[0],
             AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
+            AuxStreamType.MoeBalancer: aux_stream_list[2],
         }
 
         self.embed_tokens = Embedding(
diff --git a/tensorrt_llm/_torch/models/modeling_mixtral.py b/tensorrt_llm/_torch/models/modeling_mixtral.py
index a1e60f3f1..21dcc2006 100644
--- a/tensorrt_llm/_torch/models/modeling_mixtral.py
+++ b/tensorrt_llm/_torch/models/modeling_mixtral.py
@@ -15,6 +15,7 @@ from ..modules.embedding import Embedding
 from ..modules.fused_moe import RenormalizeMoeRoutingMethod, create_moe
 from ..modules.linear import Linear
 from ..modules.rms_norm import RMSNorm
+from ..utils import AuxStreamType
 from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                              register_auto_model)
 
@@ -49,7 +50,7 @@ class MixtralMoE(nn.Module):
             routing_method=RenormalizeMoeRoutingMethod(top_k=self.top_k),
             hidden_size=self.hidden_dim,
             intermediate_size=self.ffn_dim,
-            aux_stream=aux_stream,
+            aux_stream_dict={AuxStreamType.MoeChunkingOverlap: aux_stream},
             dtype=config.torch_dtype,
             reduce_results=reduce_results,
             model_config=model_config,
diff --git a/tensorrt_llm/_torch/models/modeling_qwen3_moe.py b/tensorrt_llm/_torch/models/modeling_qwen3_moe.py
index eeefecb42..bd2ccfae0 100644
--- a/tensorrt_llm/_torch/models/modeling_qwen3_moe.py
+++ b/tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -22,6 +22,7 @@ from ..modules.fused_moe import (BaseMoeRoutingMethod,
 from ..modules.linear import TensorParallelMode
 from ..modules.rms_norm import RMSNorm
 from ..speculative import SpecMetadata
+from ..utils import AuxStreamType
 from .modeling_qwen3 import Qwen3Attention
 from 
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## eb2d51a429 - [fix] Fix llama4 min-latency mode (#4810)

- **Date**: 2025-06-01
- **Author**: Yilin Fan
- **Categories**: Throughput/Latency

### Optimization Techniques

- Operator fusion
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/models/modeling_llama_min_latency.py | 7 +++----
 1 file changed, 3 insertions(+), 4 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_llama_min_latency.py b/tensorrt_llm/_torch/models/modeling_llama_min_latency.py
index 32f6ae4a5..db5a501b5 100644
--- a/tensorrt_llm/_torch/models/modeling_llama_min_latency.py
+++ b/tensorrt_llm/_torch/models/modeling_llama_min_latency.py
@@ -127,7 +127,6 @@ class Llama4MinLatencyLinear(Linear):
     def apply_linear(
         self,
         input,
-        weight,
         bias,
         lora_params: Optional[dict] | None = None,
         layer_idx: Optional[int] | None = None,
@@ -232,12 +231,12 @@ class Llama4MinLatencyLinear(Linear):
         # If special gemm+swiglu kernel is not used and enable_fused_gemm_swiglu is True, we need to apply swiglu
         # manually.
         if self.enable_fused_gemm_swiglu:
-            intermediate = super().apply_linear(input, weight, bias,
-                                                lora_params, layer_idx)
+            intermediate = super().apply_linear(input, bias, lora_params,
+                                                layer_idx)
             return swiglu(intermediate)
 
         # Otherwise, call the default apply_linear method.
-        return super().apply_linear(input, weight, bias, lora_params, layer_idx)
+        return super().apply_linear(input, bias, lora_params, layer_idx)
 
     # Set the position_ids for the next call to apply_linear.
     def set_position_ids(self, position_ids: Optional[torch.LongTensor] = None):

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## ec6b1821c7 - [fix] Fix W4A8 weight loading error in WInt4AFP8FusedMoEMethod (#5026)

- **Date**: 2025-06-10
- **Author**: Xiaowei Wang
- **Categories**: Fusion, Quantization Optimization

### Optimization Techniques

- Vectorized memory access
- Operator fusion
- FP8 quantization
- Integer quantization
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../_torch/modules/fused_moe/quantization.py       | 25 +++++++++++-----------
 1 file changed, 13 insertions(+), 12 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fused_moe/quantization.py b/tensorrt_llm/_torch/modules/fused_moe/quantization.py
index 2785cd52a..b821b693b 100644
--- a/tensorrt_llm/_torch/modules/fused_moe/quantization.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/quantization.py
@@ -550,13 +550,13 @@ class WInt4AFP8FusedMoEMethod(FusedMoEMethodBase):
                            module.intermediate_size_per_partition // 2)
 
         fc31_act_scale = nn.Parameter(torch.empty(1,
-                                                  self.hidden_size,
-                                                  dtype=self.dtype),
+                                                  module.hidden_size,
+                                                  dtype=module.dtype),
                                       requires_grad=False)
         module.register_parameter("fc31_act_scale", fc31_act_scale)
 
         fc2_act_scale = nn.Parameter(torch.empty(
-            1, self.intermediate_size_per_partition, 1, dtype=self.dtype),
+            1, module.intermediate_size_per_partition, 1, dtype=module.dtype),
                                      requires_grad=False)
         module.register_parameter("fc2_act_scale", fc2_act_scale)
 
@@ -701,11 +701,11 @@ class WInt4AFP8FusedMoEMethod(FusedMoEMethodBase):
         all_w3_w1_input_scales_max = torch.max(
             torch.stack(all_w3_input_scales),
             torch.stack(all_w1_input_scales)).max()
-        self.fc31_act_scale.data.copy_(
-            torch.ones_like(self.fc31_act_scale) *
+        module.fc31_act_scale.data.copy_(
+            torch.ones_like(module.fc31_act_scale) *
             (1 / all_w3_w1_input_scales_max))
-        self.fc31_alpha.data.copy_((torch.ones_like(self.fc31_alpha) *
-                                    all_w3_w1_input_scales_max).float())
+        module.fc31_alpha.data.copy_((torch.ones_like(module.fc31_alpha) *
+                                      all_w3_w1_input_scales_max).float())
 
         all_w3_scales = [
             load_weight_shard(weights[f"{expert_id}.w3.weight_scale_inv"],
@@ -744,11 +744,12 @@ class WInt4AFP8FusedMoEMethod(FusedMoEMethodBase):
             for expert_id in module.initial_local_expert_ids
         ]
         all_w2_input_scales_max = torch.stack(all_w2_input_scales).to(
-            self.dtype).max()
-        self.fc2_act_scale.data.copy_(
-            torch.ones_like(self.fc2_act_scale) * (1 / all_w2_input_scales_max))
-        self.fc2_alpha.data.copy_(
-            (torch.ones_like(self.fc2_alpha) * all_w2_input_scales_max).float())
+            module.dtype).max()
+        module.fc2_act_scale.data.copy_(
+            torch.ones_like(module.fc2_act_scale) *
+            (1 / all_w2_input_scales_max))
+        module.fc2_alpha.data.copy_((torch.ones_like(module.fc2_alpha) *
+                                     all_w2_input_scales_max).float())
 
         all_w2_scales = [
             load_weight_shard(weights[f"{expert_id}.w2.weight_scale_inv"],
```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## ec9cf715a2 - [None][feat] AutoDeploy: Perf improvement for mamba layers (#8991)

- **Date**: 2025-11-11
- **Author**: Chenghao Zhang
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- KV cache optimization
- Batching optimization
- Triton kernel
- PyTorch built-in optimized ops
- Attention mechanism optimization

### Applicable Conditions

- Prefill phase
- Decode/generation phase

### Changed Files

```
.../_torch/auto_deploy/config/default.yaml         |   2 +
 .../custom_ops/mamba/cuda_backend_causal_conv.py   |  19 ++--
 .../custom_ops/mamba/torch_backend_causal_conv.py  |   3 +-
 .../custom_ops/mamba/triton_backend_mamba.py       |  13 ++-
 .../transform/library/fuse_causal_conv.py          | 120 +++++++++++++++++++++
 .../custom_ops/test_cuda_causal_conv_cached_op.py  |   1 +
 6 files changed, 143 insertions(+), 15 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/config/default.yaml b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
index b9d001c3d..fabde3af9 100644
--- a/tensorrt_llm/_torch/auto_deploy/config/default.yaml
+++ b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
@@ -165,6 +165,8 @@ transforms:
   ############################################################################################
   # COMPILE MODEL
   ############################################################################################
+  fuse_causal_conv_activation:
+    stage: compile
   compile_model:
     stage: compile
     run_per_gm: false
diff --git a/tensorrt_llm/_torch/auto_deploy/custom_ops/mamba/cuda_backend_causal_conv.py b/tensorrt_llm/_torch/auto_deploy/custom_ops/mamba/cuda_backend_causal_conv.py
index 014f8cc7e..080337584 100644
--- a/tensorrt_llm/_torch/auto_deploy/custom_ops/mamba/cuda_backend_causal_conv.py
+++ b/tensorrt_llm/_torch/auto_deploy/custom_ops/mamba/cuda_backend_causal_conv.py
@@ -112,6 +112,7 @@ def _cuda_cached_causal_conv1d(
     dilation: int,
     groups: int,
     padding_mode: str,
+    activation: Optional[str],
 ) -> torch.Tensor:
     """Flattened cached causal conv that respects slot-indexed state caches (CUDA backend).
 
@@ -175,7 +176,7 @@ def _cuda_cached_causal_conv1d(
             cache_indices=cache_indices,
             has_initial_state=has_initial_state,
             conv_states=conv_state_cache,
-            activation=None,
+            activation=activation,
             pad_slot_id=PAD_SLOT_ID,
         )  # (dim, total_prefill_tokens)
 
@@ -185,16 +186,16 @@ def _cuda_cached_causal_conv1d(
 
     # DECODE: batch update for single-token sequences
     if num_decode > 0:
-        # Use true start offsets for decode tokens (tail after prefills)
-        decode_idx = seq_start[num_prefill:].to(torch.long)
-        x_decode = inp_flat.index_select(0, decode_idx)  # [num_decode, C_in]
+        x_decode = inp_flat[
+            total_prefill_tokens : total_prefill_tokens + num_decode
+        ]  # [num_decode, C_in]
 
         y_dec = causal_conv1d_update(
             x_decode,  # [batch, dim]
             conv_state_cache,
             w2d,
             bias,
-            activation=None,
+            activation=activation,
             cache_seqlens=None,
             conv_state_indices=slot_idx[num_prefill:].to(torch.int32),
             pad_slot_id=PAD_SLOT_ID,
@@ -202,7 +203,9 @@ def _cuda_cached_causal_conv1d(
 
         if y_dec.dim() == 3:
             y_dec = y_dec.squeeze(-1)
-        y_flat.index_copy_(0, decode_idx, y_dec.to(y_flat.dtype))
+        y_flat[total_prefill_tokens : total_prefill_tokens + num_decode].copy_(
+            y_dec.to(y_flat.dtype)
+        )
 
     # Custom op must not return an alias of any input; return a fresh tensor
     return y.contiguous().clone()
@@ -227,6 +230,7 @@ def _cuda_cached_causal_conv1d_fake(
     dilation: int,
     groups: int,
     padding_mode: str,
+    activation: Op
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## ed297d7c2e - [None][chore] Optimize perf for the RPC executor and add some profile utilities to llm-api (#8415)

- **Date**: 2025-11-04
- **Author**: Yan Chunwei
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- Parallelism optimization
- Batching optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/py_executor.py |  41 ---
 tensorrt_llm/_utils.py                        |  80 ++++++
 tensorrt_llm/bench/benchmark/utils/general.py |   3 +-
 tensorrt_llm/executor/executor.py             |   3 +
 tensorrt_llm/executor/rpc/rpc_client.py       | 349 +++++++++++++++-----------
 tensorrt_llm/executor/rpc/rpc_server.py       |   5 +-
 tensorrt_llm/executor/rpc_proxy.py            |  16 +-
 tensorrt_llm/executor/rpc_worker.py           |  21 +-
 8 files changed, 325 insertions(+), 193 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/py_executor.py b/tensorrt_llm/_torch/pyexecutor/py_executor.py
index f3b444a97..248599835 100644
--- a/tensorrt_llm/_torch/pyexecutor/py_executor.py
+++ b/tensorrt_llm/_torch/pyexecutor/py_executor.py
@@ -1,13 +1,11 @@
 import dataclasses
 import datetime
 import functools
-import gc
 import os
 import pickle  # nosec B403
 import threading
 import time
 import traceback
-import weakref
 from contextlib import contextmanager
 from typing import Dict, Iterable, List, Optional, Tuple, Union
 
@@ -59,10 +57,6 @@ from .scheduler import RequestScheduler, ScheduledRequests
 # Format: "start1-stop1,start2-stop2,..." or single iterations "iter1,iter2,..."
 PROFILE_START_STOP_ENV_VAR_NAME = "TLLM_PROFILE_START_STOP"
 
-# Environment variable to enable garbage collection profiling.
-# Set to "1" to enable recording of garbage collection events during profiling.
-PROFILE_RECORD_GC_ENV_VAR_NAME = "TLLM_PROFILE_RECORD_GC"
-
 # Environment variable to enable PyTorch profiler tracing.
 # Set to a path to save detailed tracing of PyTorch operations.
 PROFILE_TRACE_ENV_VAR_NAME = "TLLM_TORCH_PROFILE_TRACE"
@@ -97,40 +91,6 @@ def _load_iteration_indexes(env_var: str):
     return frozenset(starts), frozenset(stops)
 
 
-class _GCNvtxHandle:
-    pass
-
-
-def _gc_nvtx_watcher():
-    enabled = os.environ.get(PROFILE_RECORD_GC_ENV_VAR_NAME, None)
-    if not enabled:
-        return None
-
-    range_id: Optional[int] = None
-
-    def gc_callback(phase, _):
-        nonlocal range_id
-        if phase == "start":
-            assert range_id is None, "Unexpected state in GC callback: another GC while last GC not finished?"
-            range_id = torch.cuda.nvtx.range_start("Python GC")
-        elif phase == "stop":
-            assert range_id is not None, "Unexpected state in GC callback: no active GC but got GC finished?"
-            torch.cuda.nvtx.range_end(range_id)
-            range_id = None
-
-    gc.callbacks.append(gc_callback)
-
-    def gc_cleanup(callback):
-        try:
-            gc.callbacks.remove(callback)
-        except ValueError:
-            pass
-
-    handle = _GCNvtxHandle()
-    weakref.finalize(handle, gc_cleanup, gc_callback)
-    return handle
-
-
 @dataclasses.dataclass
 class BatchState:
     sample_state: SampleState
@@ -178,7 +138,6 @@ class PyExecutor:
         # profile config
         self.profile_start_iters, self.profile_stop_iters = _load_iteration_indexes(
             PROFILE_START_STOP_ENV_VAR_NAME)
-        self.gc_nvtx_watcher_handle = _gc_nvtx_watcher()
 
         # related modules
         self.resource_manager = resource_manager
diff --git a/tensorrt_llm/_utils.py b/tensorrt_llm/_utils.py
index a56608912..a3bf1024e 100644
--- a/tensorrt_llm/_utils.py
+++ b/tensorrt_llm/_utils.py
@@ -918,6 +918,18 @@ def nvtx_range_debug(msg: str,
         return _null_context_manager()
 
 
+def nvtx_mark_debug(msg: str,
+                    color: str = "grey",
+                    do
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## edab7532dd - feat/add latency support for trtllm bench (#3730)

- **Date**: 2025-07-15
- **Author**: danielafrimi
- **Categories**: Throughput/Latency

### Optimization Techniques

- Async/stream-based execution
- FP8 quantization
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Speculative decoding
- MoE optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/bench/benchmark/low_latency.py   | 177 +++++++++++++++++++++-----
 tensorrt_llm/bench/benchmark/throughput.py    |   9 +-
 tensorrt_llm/bench/benchmark/utils/general.py |   2 +
 tests/integration/defs/test_e2e.py            |   8 +-
 4 files changed, 153 insertions(+), 43 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/bench/benchmark/low_latency.py b/tensorrt_llm/bench/benchmark/low_latency.py
index 490ac62f4..cacb7a2ad 100644
--- a/tensorrt_llm/bench/benchmark/low_latency.py
+++ b/tensorrt_llm/bench/benchmark/low_latency.py
@@ -9,11 +9,14 @@ import click
 import yaml
 from click_option_group import (MutuallyExclusiveOptionGroup, OptionGroup,
                                 optgroup)
+from huggingface_hub import snapshot_download
 
+from tensorrt_llm import LLM as PyTorchLLM
 from tensorrt_llm._tensorrt_engine import LLM
 from tensorrt_llm.bench.benchmark.utils.asynchronous import async_benchmark
 from tensorrt_llm.bench.benchmark.utils.general import generate_warmup_dataset
 from tensorrt_llm.bench.benchmark.utils.processes import IterationWriter
+from tensorrt_llm.bench.build.build import get_model_config
 from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
 from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment
 from tensorrt_llm.bench.dataclasses.reporting import ReportUtility
@@ -21,10 +24,11 @@ from tensorrt_llm.llmapi import CapacitySchedulerPolicy
 from tensorrt_llm.models.modeling_utils import SpeculativeDecodingMode
 
 # isort: off
-from tensorrt_llm.bench.benchmark.utils.general import get_settings_from_engine
+from tensorrt_llm.bench.benchmark.utils.general import get_settings_from_engine, get_settings, ALL_SUPPORTED_BACKENDS
 # isort: on
 from tensorrt_llm.bench.utils.data import (create_dataset_from_stream,
-                                           initialize_tokenizer)
+                                           initialize_tokenizer,
+                                           update_metadata_for_multimodal)
 from tensorrt_llm.logger import logger
 from tensorrt_llm.sampling_params import SamplingParams
 
@@ -38,15 +42,25 @@ from tensorrt_llm.sampling_params import SamplingParams
                     readable=True,
                     path_type=Path,
                     resolve_path=True),
-    required=True,
+    default=None,
     help="Path to a serialized TRT-LLM engine.",
 )
+@optgroup.option("--backend",
+                 type=click.Choice(ALL_SUPPORTED_BACKENDS),
+                 default="pytorch",
+                 help="The backend to use when running benchmarking.")
 @optgroup.option(
     "--kv_cache_free_gpu_mem_fraction",
     type=float,
     default=.90,
     help="The percentage of memory to use for KV Cache after model load.",
 )
+@optgroup.option(
+    "--max_seq_len",
+    type=int,
+    default=None,
+    help="Maximum sequence length.",
+)
 @optgroup.group(
     "Engine Input Configuration",
     help="Input configuration for driving the engine.",
@@ -60,6 +74,20 @@ from tensorrt_llm.sampling_params import SamplingParams
     default=None,
     help="Pass in a dataset file for parsing instead of stdin.",
 )
+@optgroup.option(
+    "--modality",
+    type=click.Choice(["image", "video"]),
+    default=None,
+    help="Modality of the multimodal requests.",
+)
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## ee471df07c - [None][chore] optimize kv cache transfer for context TEP and  gen DEP (#6657)

- **Date**: 2025-08-07
- **Author**: Chuang Zhu
- **Categories**: Cache Optimization

### Optimization Techniques

- KV cache optimization
- Parallelism optimization
- Batching optimization
- Attention mechanism optimization

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp  |  7 ++++---
 .../batch_manager/mlaCacheFormatter.cpp            | 22 +++++++++++++++-------
 cpp/tests/batch_manager/cacheTransceiverTest.cpp   |  8 +++++---
 3 files changed, 24 insertions(+), 13 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp b/cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp
index 2edfd5f77..d95ca1b41 100644
--- a/cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp
+++ b/cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp
@@ -75,7 +75,6 @@ BlockRange getBlockRangeForReceiving(BaseKVCacheManager* cacheManager, LlmReques
 bool CacheFormatter::needSendCache(
     CacheState const& selfConfig, CacheState const& destConfig, runtime::SizeType32 selfIdx)
 {
-    // int selfTpRank = selfIdx % selfConfig.getParallelConfig().mTensorParallelism;
     auto targetInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);
     if (targetInfo.mDupHeadFactor <= 1)
     {
@@ -90,8 +89,9 @@ bool CacheFormatter::needSendCache(
             = selfConfig.getParallelConfig().mTensorParallelism / selfConfig.getParallelConfig().mDPsize;
         selfTpRankInDpGroup = selfTpRank % selfTPNumInDPGroup;
     }
+    int destDPRank = destConfig.getParallelConfig().mEnableAttentionDP ? destConfig.getParallelConfig().mDPrank : 0;
 
-    return selfTpRankInDpGroup % targetInfo.mDupHeadFactor == 0;
+    return (destDPRank % targetInfo.mDupHeadFactor) == (selfTpRankInDpGroup % targetInfo.mDupHeadFactor);
 }
 
 void checkAlternateWindow(BaseKVCacheManager* cacheManager, BaseCacheFormatter::CacheState const& selfConfig,
@@ -128,11 +128,12 @@ std::vector<size_t> CacheFormatter::pickRecvConnections(
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
diff --git a/cpp/tensorrt_llm/batch_manager/mlaCacheFormatter.cpp b/cpp/tensorrt_llm/batch_manager/mlaCacheFormatter.cpp
index 810edd6f4..824a31129 100644
--- a/cpp/tensorrt_llm/batch_manager/mlaCacheFormatter.cpp
+++ b/cpp/tensorrt_llm/batch_manager/mlaCacheFormatter.cpp
@@ -45,10 +45,12 @@ std::vector<size_t> MLACacheFormatter::pickRecvConnections(
     auto targetInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);
     TLLM_CHECK(numConnections == targetInfo.mIRanks.size());
     std::vector<size_t> ret;
-    // targetInfo , mRanks [tpranks, dpranks]
+    // targetInfo , mRanks [tpranks, ppranks]
+    int dpRank = selfConfig.getParallelConfig().mEnableAttentionDP ? selfConfig.getParallelConfig().mDPrank : 0;
+
     for (int i = 0; i < targetInfo.mDomainPPSize; i++)
     {
-        ret.push_back(i);
+        ret.push_back(i + (dpRank % (targetInfo.mDomainTPSize)) * targetInfo.mDomainPPSize);
     }
     return ret;
 }
@@ -58,19 +60,24 @@ bool MLACacheFormatter::needSendCache(
 {

```

### Analysis Summary

Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## eeb555e37b - chore: memoize weight shuffle index to speed up weight preproc in moe_backend=TRTLLM (#4826)

- **Date**: 2025-06-06
- **Author**: Anthony Chang
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
- Speculative decoding
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- Disaggregated serving

### Changed Files

```
.../trtllmGenKernels/blockScaleMoe/runner.h        |   2 +-
 .../_torch/modules/fused_moe/quantization.py       | 135 ++++++++++++++-------
 .../defs/accuracy/test_llm_api_pytorch.py          |  12 +-
 .../test_lists/qa/examples_test_list.txt           |   1 +
 .../integration/test_lists/qa/llm_sanity_test.txt  |   1 +
 tests/integration/test_lists/test-db/l0_b200.yml   |   1 +
 tests/unittest/_torch/thop/test_moe.py             |   2 +-
 7 files changed, 99 insertions(+), 55 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h b/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h
index 9c343d56c..a7ac7e0c5 100644
--- a/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h
+++ b/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h
@@ -35,7 +35,7 @@ namespace Routing
 {
 
 // The type of method in top-K routing, for use in torch custom op
-// Please keep this in sync with the counterpart defined in tensorrt_llm/_torch/modules/fused_moe.py
+// Please keep this in sync with the counterpart defined in tensorrt_llm/_torch/modules/fused_moe/routing.py
 enum class RoutingMethodType : int64_t
 {
     // Default: Softmax -> TopK
diff --git a/tensorrt_llm/_torch/modules/fused_moe/quantization.py b/tensorrt_llm/_torch/modules/fused_moe/quantization.py
index e457457e5..a4030b20f 100644
--- a/tensorrt_llm/_torch/modules/fused_moe/quantization.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/quantization.py
@@ -1,6 +1,5 @@
-import threading
 from abc import ABC, abstractmethod
-from typing import Dict, List, NamedTuple
+from typing import Dict, List, NamedTuple, Union
 
 import torch
 from torch import nn
@@ -8,8 +7,7 @@ from torch import nn
 from tensorrt_llm._utils import get_sm_version
 from tensorrt_llm.quantization.utils.fp4_utils import (
     float4_sf_dtype, get_reorder_rows_for_gated_act_gemm_row_indices,
-    get_shuffle_matrix_a_row_indices, get_shuffle_matrix_sf_a_row_indices,
-    shuffle_matrix_a, shuffle_matrix_sf_a)
+    get_shuffle_matrix_a_row_indices, get_shuffle_matrix_sf_a_row_indices)
 
 from ..linear import TensorParallelMode, load_weight_shard
 from .interface import MoEWeightLoadingMode
@@ -80,12 +78,8 @@ class FusedMoEMethodBase(ABC):
 
     def load_weights(self, module: torch.nn.Module, weights: List[Dict],
                      weight_loading_mode: MoEWeightLoadingMode):
-        # Use multi-threading to load expert weights in parallel.
-        # Even though CPython has global interpreter lock (GIL),
-        # it's still faster to load weights in parallel because it can utilize
-        # CPU memory bandwidth better.
-        threads = []
-
+        # Multithread weight load is superseded by prefetch_files() in model_engine.py
+        # Also, threading adds overhead in order to protect shuffle index cache with critical section.
         for local_slot_id, expert_id in enumerate(
                 module.initial_local_expert_ids):
             # expert_idx is the local slot index of current rank
@@ -106,21 +100,11 @@ class FusedMoEMethodBase(ABC):
                     f"Unknown weight loading mode in MoE: {weight_loading_mode}"
                 )
 
-            thread = threading.Thread(
-                target=self.load_expert_w3_w1_weight,
-                args=(module, w1_weight, w3_weight,
-                      module.w3_w1_weight.data[expert_idx]))
-            thread.start()
-            threads.append(thread)
-
-            thread = thr
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## efd503751f - [#9271][perf] Enable multi-stream MOE optimization in AutoDeploy (#9322)

- **Date**: 2025-11-24
- **Author**: Suyog Gupta
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
- Triton kernel
- PyTorch built-in optimized ops
- Multi-stream execution
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- Prefill phase
- Decode/generation phase

### Changed Files

```
examples/auto_deploy/nano_v3.yaml                  |   3 +
 .../auto_deploy/custom_ops/fused_moe/trtllm_moe.py |  27 +--
 .../custom_ops/mamba/cuda_backend_causal_conv.py   |   4 +-
 .../custom_ops/mamba/triton_backend_mamba.py       |  13 +-
 .../_torch/auto_deploy/custom_ops/multi_stream.py  | 235 +++++++++++++++++++++
 .../_torch/auto_deploy/custom_ops/rms_norm.py      |   9 +-
 .../auto_deploy/models/patches/nemotron_h.py       |   8 +-
 .../auto_deploy/transform/library/fused_moe.py     |  90 ++++++--
 .../transform/library/multi_stream_moe.py          |  80 +++++++
 .../defs/accuracy/test_llm_api_autodeploy.py       |   1 +
 .../auto_deploy/_utils_test/_model_test_utils.py   |   6 +
 .../unit/singlegpu/custom_ops/test_multi_stream.py | 132 ++++++++++++
 .../custom_ops/test_triton_mamba_cached_op.py      |   1 -
 .../unit/singlegpu/custom_ops/test_trtllm_moe.py   |   7 +
 .../unit/singlegpu/test_ad_build_small_single.py   |   8 +
 15 files changed, 567 insertions(+), 57 deletions(-)
```

### Diff Preview

```diff
diff --git a/examples/auto_deploy/nano_v3.yaml b/examples/auto_deploy/nano_v3.yaml
index 411037cc1..9d9acf6ef 100644
--- a/examples/auto_deploy/nano_v3.yaml
+++ b/examples/auto_deploy/nano_v3.yaml
@@ -15,6 +15,9 @@ transforms:
   detect_sharding:
     sharding_source: ['factory', 'heuristic']
     sharding_dims: ['ep', 'bmm']
+  multi_stream_moe:
+    stage: compile
+    enabled: true
   # tunable mamba cache dtype
   # --> use float32 for accuracy and default (null) for speed
   insert_cached_ssm_attention:
diff --git a/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/trtllm_moe.py b/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/trtllm_moe.py
index 62e7b36dd..8b130d987 100644
--- a/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/trtllm_moe.py
+++ b/tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/trtllm_moe.py
@@ -67,7 +67,9 @@ def trtllm_moe_fused_fake(
     return torch.empty_like(x)
 
 
-# Todo: refactor this repeating code block
+# NOTE(suyogg): If compile ever fails because of this, just write a triton kernel
+# for this function and use it as a custom op.
+@torch.compile
 def _quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
     """Quantize tensor to FP8 with clamping (matches torch_quant_fp8_linear)."""
     FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
@@ -107,6 +109,9 @@ def trtllm_quant_fp8_moe_fused(
     w1_weight_scale: torch.Tensor,  # [E] stacked weight scales
     w2_weight_scale: torch.Tensor,  # [E] stacked weight scales
     w3_weight_scale: torch.Tensor,  # [E] or unused
+    gemm1_dequant: torch.Tensor,  # [E]
+    gemm2_act_quant: torch.Tensor,  # [E]
+    gemm2_dequant: torch.Tensor,  # [E]
     mlp_style: str = "gated_mlp",
     act_fn: str = "silu",
 ) -> torch.Tensor:
@@ -125,6 +130,9 @@ def trtllm_quant_fp8_moe_fused(
         w1_weight_scale: Weight scales for w1 [E]
         w2_weight_scale: Weight scales for w2 [E]
         w3_weight_scale: Weight scales for w3 [E]
+        gemm1_dequant: Precomputed gemm1 dequant scale [E]
+        gemm2_act_quant: Precomputed gemm2 act quant scale [1]
+        gemm2_dequant: Precomputed gemm2 dequant scale [E]
         mlp_style: "gated_mlp" or "mlp"
         act_fn: "silu" for gated_mlp, "relu2" for mlp
 
@@ -144,28 +152,20 @@ def trtllm_quant_fp8_moe_fused(
     x_q_fp8 = _quantize_fp8(x2d, w1_input_scale[0])
 
     # Scales are stored in float32
-    w1_weight_scale = w1_weight_scale.to(torch.float32)
-    w2_weight_scale = w2_weight_scale.to(torch.float32)
-    w1_input_scale = w1_input_scale.to(torch.float32)[0]
-    w2_input_scale = w2_input_scale.to(torch.float32)[0]
+    w1_input_scale = w1_input_scale[0]
 
     # Prepare quant_scales for TensorRT-LLM FP8 format:
     # [gemm1_dequant_scale, gemm2_act_quant_scale, gemm2_dequant_scale, gemm1_input_dequant_scale]
     # For gated MLP:
+    # These are precomputed in `fused_moe` transform
     # - gemm1_dequant_scale: w1_weight_scale * w1_input_scale (combined for w1 and w3)
     # - g
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Multi-stream execution enables parallel execution of independent operations on the GPU. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## f02948d956 - [https://nvbugs/5803813][fix] Fix llama 4 min latency (#10724)

- **Date**: 2026-01-16
- **Author**: Mike Iovine
- **Categories**: Throughput/Latency

### Optimization Techniques

- Operator fusion
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/models/modeling_llama_min_latency.py | 11 +++--------
 1 file changed, 3 insertions(+), 8 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_llama_min_latency.py b/tensorrt_llm/_torch/models/modeling_llama_min_latency.py
index 9af2d99f0..ae3d1601f 100644
--- a/tensorrt_llm/_torch/models/modeling_llama_min_latency.py
+++ b/tensorrt_llm/_torch/models/modeling_llama_min_latency.py
@@ -88,9 +88,10 @@ class Llama4MinLatencyLinear(Linear):
         self.enable_trtllm_gen = enable_trtllm_gen
         self.position_ids = None
 
-    def load_weights(self, weights: List[Dict]):
+    def load_weights(self, weights: List[Dict], allow_partial_loading: bool):
 
-        super().load_weights(weights)
+        super().load_weights(weights,
+                             allow_partial_loading=allow_partial_loading)
 
         # After loading weights, calculate the combined scale (input_scale * weight_scale) for special kernels and
         # trtllm-gen kernels.
@@ -384,11 +385,6 @@ class Llama4MinLatencyAttention(Llama4Attention):
                 and self.floor_scale == 8192.0 \
                 and self.attn_scale == 0.1
 
-            qkv_shard_indices_mapping = {
-                "q": (0, self.q_size),
-                "k": (self.q_size, self.kv_size),
-                "v": (self.q_size + self.kv_size, self.kv_size),
-            }
             # When min-latency QKV gemm is enabled, override qkv_proj.
             self.qkv_proj = Llama4MinLatencyLinear(
                 self.hidden_size,
@@ -405,7 +401,6 @@ class Llama4MinLatencyAttention(Llama4Attention):
                 enable_fused_gemm_attn_scaling=self.
                 enable_fused_gemm_attn_scaling,
                 enable_trtllm_gen=True,
-                fused_weight_shard_indices_mapping=qkv_shard_indices_mapping,
             )
 
     def _forward_nope(

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## f03053b4dd - [None][fix] update accelerate dependency to 1.7+ for AutoDeploy (#7077)

- **Date**: 2025-08-20
- **Author**: Fridah-nv
- **Categories**: General Performance

### Optimization Techniques

- General code optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
requirements.txt | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/requirements.txt b/requirements.txt
index e2582f503..a7821f15d 100644
--- a/requirements.txt
+++ b/requirements.txt
@@ -1,6 +1,6 @@
 --extra-index-url https://download.pytorch.org/whl/cu128
 -c constraints.txt
-accelerate>=0.25.0
+accelerate>=1.7.0
 build
 colored
 cuda-python>=12,<13

```

### Analysis Summary

This commit applies performance optimization techniques to improve inference efficiency. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

