# Performance Optimization Analysis - Part 8

Commits 204 to 232 of 283

---

## c3729dbd7d - infra: [TRTLLM-5873] Use build stage wheels to speed up docker release image build (#4939)

- **Date**: 2025-07-30
- **Author**: Zhanrui Sun
- **Categories**: Throughput/Latency

### Optimization Techniques

- KV cache optimization

### Applicable Conditions

- Disaggregated serving

### Changed Files

```
docker/Dockerfile.multi           |   3 +-
 docker/Makefile                   |   2 +
 jenkins/Build.groovy              |   1 +
 jenkins/BuildDockerImage.groovy   |  41 ++++++++++++--
 scripts/get_wheel_from_package.py | 112 ++++++++++++++++++++++++++++++++++++++
 5 files changed, 152 insertions(+), 7 deletions(-)
```

### Diff Preview

```diff
diff --git a/docker/Dockerfile.multi b/docker/Dockerfile.multi
index 95aa670a0..0d156c7a7 100644
--- a/docker/Dockerfile.multi
+++ b/docker/Dockerfile.multi
@@ -128,8 +128,9 @@ ENV CCACHE_DIR=/root/.cache/ccache
 # Build the TRT-LLM wheel
 ARG GITHUB_MIRROR=""
 ARG BUILD_WHEEL_ARGS="--clean --benchmarks"
+ARG BUILD_WHEEL_SCRIPT="scripts/build_wheel.py"
 RUN --mount=type=cache,target=/root/.cache/pip --mount=type=cache,target=${CCACHE_DIR} \
-    GITHUB_MIRROR=$GITHUB_MIRROR python3 scripts/build_wheel.py ${BUILD_WHEEL_ARGS}
+    GITHUB_MIRROR=$GITHUB_MIRROR python3 ${BUILD_WHEEL_SCRIPT} ${BUILD_WHEEL_ARGS}
 
 FROM ${DEVEL_IMAGE} AS release
 
diff --git a/docker/Makefile b/docker/Makefile
index 2b5022b1e..dde0e461c 100644
--- a/docker/Makefile
+++ b/docker/Makefile
@@ -39,6 +39,7 @@ PLATFORM           ?= $(shell uname -m | grep -q 'aarch64' && echo "arm64" || ec
 CUDA_ARCHS         ?= $(if $(filter arm64,$(PLATFORM)),'90-real;100-real;120-real',)
 BUILD_WHEEL_OPTS   ?=
 BUILD_WHEEL_ARGS   ?= $(shell grep '^ARG BUILD_WHEEL_ARGS=' Dockerfile.multi | grep -o '=.*' | tr -d '="')$(if $(CUDA_ARCHS), --cuda_architectures $(CUDA_ARCHS))$(if $(BUILD_WHEEL_OPTS), $(BUILD_WHEEL_OPTS))
+BUILD_WHEEL_SCRIPT ?=
 TORCH_INSTALL_TYPE ?= skip
 CUDA_VERSION       ?=
 CUDNN_VERSION      ?=
@@ -80,6 +81,7 @@ endef
 		$(if $(BASE_IMAGE), --build-arg BASE_IMAGE=$(BASE_IMAGE)) \
 		$(if $(BASE_TAG), --build-arg BASE_TAG=$(BASE_TAG)) \
 		$(if $(BUILD_WHEEL_ARGS), --build-arg BUILD_WHEEL_ARGS="$(BUILD_WHEEL_ARGS)") \
+		$(if $(BUILD_WHEEL_SCRIPT), --build-arg BUILD_WHEEL_SCRIPT="$(BUILD_WHEEL_SCRIPT)") \
 		$(if $(TORCH_INSTALL_TYPE), --build-arg TORCH_INSTALL_TYPE="$(TORCH_INSTALL_TYPE)") \
 		$(if $(CUDA_VERSION), --build-arg CUDA_VER="$(CUDA_VERSION)") \
 		$(if $(CUDNN_VERSION), --build-arg CUDNN_VER="$(CUDNN_VERSION)") \
diff --git a/jenkins/Build.groovy b/jenkins/Build.groovy
index 77e12ee51..5dae931b6 100644
--- a/jenkins/Build.groovy
+++ b/jenkins/Build.groovy
@@ -460,6 +460,7 @@ def runLLMBuild(pipeline, buildFlags, tarName, is_linux_x86_64)
     sh "mkdir -p TensorRT-LLM/benchmarks/cpp"
     sh "cp ${LLM_ROOT}/cpp/build/benchmarks/bertBenchmark TensorRT-LLM/benchmarks/cpp"
     sh "cp ${LLM_ROOT}/cpp/build/benchmarks/gptManagerBenchmark TensorRT-LLM/benchmarks/cpp"
+    sh "cp ${LLM_ROOT}/cpp/build/benchmarks/disaggServerBenchmark TensorRT-LLM/benchmarks/cpp"
     sh "cp ${LLM_ROOT}/cpp/build/tensorrt_llm/libtensorrt_llm.so TensorRT-LLM/benchmarks/cpp"
     sh "cp ${LLM_ROOT}/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so TensorRT-LLM/benchmarks/cpp"
 
diff --git a/jenkins/BuildDockerImage.groovy b/jenkins/BuildDockerImage.groovy
index 88ab26503..b09a91352 100644
--- a/jenkins/BuildDockerImage.groovy
+++ b/jenkins/BuildDockerImage.groovy
@@ -27,6 +27,9 @@ LLM_SHORT_COMMIT = env.gitlabCommit ? env.gitlabCommit.substring(0, 7) : "undefi
 LLM_DEFAULT_TAG = env.defaultTag ?: "${LLM_SHORT_COMMIT}-${LLM_BRANCH_TAG}-${BUILD_NUMBER}"
 
 RUN_SANITY_CHECK
```

### Analysis Summary

This commit applies performance optimization techniques to improve inference efficiency. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## c37531c3f7 - [TRTLLM-10669][fix] Fix Eagle3 draft model weight loading for throughput checkpoint (#11010)

- **Date**: 2026-01-28
- **Author**: Guiju Zhang
- **Categories**: Throughput/Latency

### Optimization Techniques

- Speculative decoding

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/models/modeling_speculative.py | 19 ++++++++++++++++++-
 1 file changed, 18 insertions(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_speculative.py b/tensorrt_llm/_torch/models/modeling_speculative.py
index 1849f1acf..a22e196a9 100755
--- a/tensorrt_llm/_torch/models/modeling_speculative.py
+++ b/tensorrt_llm/_torch/models/modeling_speculative.py
@@ -482,9 +482,26 @@ class Eagle3ForCausalLM(DecoderModelForCausalLM[Eagle3DraftModel,
         )
 
     def load_weights(self, weights: Dict, weight_mapper: BaseWeightMapper):
+        # Remap weight names: some Eagle3 checkpoints use "layers.X.*" naming convention
+        # while the model expects "midlayer.*" naming. Handle both formats.
+        import re
+        remapped_weights = {}
+        # Access num_layers from the inner draft model (self.model is Eagle3DraftModel)
+        num_layers = self.model.num_layers
+        for k, v in weights.items():
+            new_k = k
+            # For single-layer models: "layers.0.*" -> "midlayer.*"
+            # For multi-layer models: "layers.X.*" -> "midlayer.X.*"
+            if num_layers == 1:
+                # Single layer: layers.0.foo -> midlayer.foo
+                new_k = re.sub(r'^layers\.0\.', 'midlayer.', new_k)
+            else:
+                # Multi-layer: layers.X.foo -> midlayer.X.foo
+                new_k = re.sub(r'^layers\.(\d+)\.', r'midlayer.\1.', new_k)
+            remapped_weights[new_k] = v
 
         new_weights = {}
-        for k, v in weights.items():
+        for k, v in remapped_weights.items():
             if 'lm_head' not in k:
                 new_k = "model." + k
             else:

```

### Analysis Summary

Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## c39bbb2d1a - [TRTLLM-11090][perf] Improve fp8 (per-tensor) quant kernel by vectorized load/store (#11662)

- **Date**: 2026-02-25
- **Author**: Chang Liu
- **Categories**: Kernel Optimization, Quantization Optimization

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- Reduce synchronization overhead

### Applicable Conditions

- General LLM inference

### Changed Files

```
cpp/tensorrt_llm/common/cudaFp8Utils.cu | 109 +++++++++++++++++++++++++++++++-
 1 file changed, 108 insertions(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/common/cudaFp8Utils.cu b/cpp/tensorrt_llm/common/cudaFp8Utils.cu
index b5ffde593..9cd022773 100644
--- a/cpp/tensorrt_llm/common/cudaFp8Utils.cu
+++ b/cpp/tensorrt_llm/common/cudaFp8Utils.cu
@@ -1,5 +1,5 @@
 /*
- * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
+ * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
  *
  * Licensed under the Apache License, Version 2.0 (the "License");
  * you may not use this file except in compliance with the License.
@@ -20,6 +20,7 @@
 #include "tensorrt_llm/common/envUtils.h"
 #include "tensorrt_llm/common/reduceKernelUtils.cuh"
 #include <algorithm>
+#include <cstdint>
 #include <cstdio>
 #include <cuda_fp16.h>
 #include <limits>
@@ -32,6 +33,16 @@ namespace common
 #ifdef ENABLE_FP8
 
 constexpr int CTA_SIZE = 256;
+constexpr int kVecSize = 8;
+constexpr int kPairsPerVec = kVecSize / 2;
+constexpr int64_t kMaxGridDim = 65536;
+
+inline int64_t scaleMatrixVecGridSize(int64_t numel)
+{
+    int64_t vecElements = numel / kVecSize;
+    int64_t blocks = (vecElements + CTA_SIZE - 1) / CTA_SIZE;
+    return std::max(int64_t(1), std::min(blocks, kMaxGridDim));
+}
 
 template <bool QUANTIZE>
 __inline__ __device__ float scale(float a, float b)
@@ -39,6 +50,57 @@ __inline__ __device__ float scale(float a, float b)
     return QUANTIZE ? a / b : a * b;
 }
 
+// Vectorized PER_TENSOR kernel for bf16 input → fp8 output.
+// Each thread processes 8 bf16 values per iteration via 128-bit loads / 64-bit stores.
+// T_S may be float or __nv_bfloat16; the single scale element is converted to float once.
+template <bool QUANTIZE, typename T_S>
+__global__ void scaleMatrixPerTensorVec(
+    __nv_fp8_e4m3* output, T_S const* input_scale, __nv_bfloat16 const* input, int64_t numel)
+{
+#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
+    cudaGridDependencySynchronize();
+#endif
+
+    float const s = static_cast<float>(input_scale[0]);
+    float const factor = QUANTIZE ? (1.0f / s) : s;
+    int64_t const vecElements = numel / kVecSize;
+    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
+
+    for (int64_t vi = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x; vi < vecElements; vi += stride)
+    {
+        int64_t const base = vi * kVecSize;
+
+        // 128-bit load: 8 × bf16
+        float4 raw = *reinterpret_cast<float4 const*>(input + base);
+        __nv_bfloat162 const* pairs = reinterpret_cast<__nv_bfloat162 const*>(&raw);
+
+        // Convert 4 × bf16-pair → 4 × float2 → 4 × fp8-pair → pack into 2 × uint32
+        __nv_fp8x2_storage_t fp8_pairs[kPairsPerVec];
+#pragma unroll
+        for (int p = 0; p < kPairsPerVec; ++p)
+        {
+            float2 f2 = __bfloat1622float2(pairs[p]);
+            f2.x *= factor;
+            f2.y *= factor;
+            fp8_pairs[p] = __nv_cvt_float2_to_fp8x2(f2, __NV_SATFINITE, __NV_E4M3);
+        }
+
+        // 64-bit store: 8 × fp8
+        *reinterpret_cast<uint2*>(outp
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## c4da4fd462 - [https://nvbugs/5637220][ci] unwaive TestQwen3_235B_A22B::test_nvfp4[latency_moe_trtllm_attention_dp] (#9870)

- **Date**: 2026-01-14
- **Author**: QI JUN
- **Categories**: Throughput/Latency

### Optimization Techniques

- FP8 quantization
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
index a8c3c5097..8e763d000 100644
--- a/tests/integration/test_lists/waives.txt
+++ b/tests/integration/test_lists/waives.txt
@@ -213,7 +213,6 @@ accuracy/test_cli_flow.py::TestMinitron4BBase::test_fp8 SKIP (https://nvbugs/560
 examples/test_gpt.py::test_llm_minitron_fp8_with_pseudo_loras[4b] SKIP (https://nvbugs/5606233)
 test_e2e.py::test_trtllm_bench_pytorch_backend_sanity[meta-llama/Llama-3.1-8B-llama-3.1-8b-hf-nvfp4-False-False] SKIP (https://nvbugs/5629791)
 accuracy/test_disaggregated_serving.py::TestLlama4ScoutInstruct::test_auto_dtype[False] SKIP (https://nvbugs/5629792)
-accuracy/test_llm_api_pytorch.py::TestQwen3_235B_A22B::test_nvfp4[latency_moe_trtllm_attention_dp] SKIP (https://nvbugs/5637220)
 llmapi/test_llm_examples.py::test_llmapi_example_multilora SKIP (https://nvbugs/5636857)
 accuracy/test_llm_api_pytorch.py::TestLlama3_1_8BInstruct::test_bfloat16_4gpus[tp4-attn_backend=TRTLLM-torch_compile=False] SKIP (https://nvbugs/5616182)
 examples/test_phi.py::test_llm_phi_quantization_1gpu[Phi-3-small-128k-instruct-fp8-bfloat16] SKIP (https://nvbugs/5465143)

```

### Analysis Summary

Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## c53bc19f5e - [infra] Make test_chunked_prefill faster (#5248)

- **Date**: 2025-06-16
- **Author**: Mike Iovine
- **Categories**: Throughput/Latency

### Optimization Techniques

- General code optimization

### Applicable Conditions

- Prefill phase

### Changed Files

```
tests/integration/defs/accuracy/test_llm_api_pytorch.py | 9 +++++----
 1 file changed, 5 insertions(+), 4 deletions(-)
```

### Diff Preview

```diff
diff --git a/tests/integration/defs/accuracy/test_llm_api_pytorch.py b/tests/integration/defs/accuracy/test_llm_api_pytorch.py
index 687bc280a..8df20ae0e 100644
--- a/tests/integration/defs/accuracy/test_llm_api_pytorch.py
+++ b/tests/integration/defs/accuracy/test_llm_api_pytorch.py
@@ -61,16 +61,17 @@ class TestLlama3_1_8BInstruct(LlmapiAccuracyTestHarness):
     @pytest.mark.skip_less_device_memory(32000)
     @parametrize_with_ids("attn_backend", ["TRTLLM", "FLASHINFER"])
     def test_chunked_prefill(self, attn_backend):
-        pytorch_config = dict(attn_backend=attn_backend, )
+        pytorch_config = dict(
+            attn_backend=attn_backend,
+            # https://nvbugspro.nvidia.com/bug/5345391
+            disable_overlap_scheduler=True)
         llm = LLM(self.MODEL_PATH,
                   enable_chunked_prefill=True,
-                  max_num_tokens=64,
+                  max_num_tokens=512,
                   **pytorch_config)
         with llm:
             task = MMLU(self.MODEL_NAME)
             task.evaluate(llm)
-            task = GSM8K(self.MODEL_NAME)
-            task.evaluate(llm)
 
     @pytest.mark.skip_less_device_memory(32000)
     @parametrize_with_ids(

```

### Analysis Summary

This commit applies performance optimization techniques to improve inference efficiency. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## c59aa8bec5 - [TRTLLM-9962][feat] Some optimizations for two-model spec dec (#10208)

- **Date**: 2025-12-28
- **Author**: Ziyi Xiong
- **Categories**: General Performance

### Optimization Techniques

- Torch compilation/JIT optimization
- Operator fusion
- Async/stream-based execution
- Batching optimization
- Pinned memory
- Speculative decoding
- Attention mechanism optimization

### Applicable Conditions

- Prefill phase
- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/model_engine.py    | 12 ++---
 tensorrt_llm/_torch/pyexecutor/py_executor.py     |  1 -
 tensorrt_llm/_torch/speculative/drafting_loops.py | 40 +++++++-------
 tensorrt_llm/_torch/speculative/model_drafter.py  | 63 +++++++++++++++++------
 4 files changed, 74 insertions(+), 42 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/model_engine.py b/tensorrt_llm/_torch/pyexecutor/model_engine.py
index 7b5b2cf0b..99ecac85e 100644
--- a/tensorrt_llm/_torch/pyexecutor/model_engine.py
+++ b/tensorrt_llm/_torch/pyexecutor/model_engine.py
@@ -895,8 +895,6 @@ class PyTorchModelEngine(ModelEngine):
             return None
 
         num_extra_decoding_steps = self._get_num_extra_decoding_steps()
-        if num_extra_decoding_steps > 0:
-            return None  # Disable autotuning for fused drafting loops for now.
 
         if num_gen_requests > self.batch_size:
             return None
@@ -909,7 +907,10 @@ class PyTorchModelEngine(ModelEngine):
         ctx_requests = []
         gen_requests = []
 
-        max_seq_len = self.max_seq_len - 1
+        # For drafting loops, reduce max_seq_len to leave room for extra decoding steps
+        max_seq_len = self.max_seq_len - 1 - num_extra_decoding_steps
+        if max_seq_len < 1:
+            return None  # Not enough sequence length for drafting loop
         num_full_seqs = 0
         num_left_over_tokens = 0
 
@@ -954,7 +955,8 @@ class PyTorchModelEngine(ModelEngine):
                 token_nums=ctx_token_nums,
                 is_gen=False,
                 max_num_draft_tokens=self.runtime_draft_len,
-                use_mrope=self.use_mrope)
+                use_mrope=self.use_mrope,
+                num_extra_decoding_steps=num_extra_decoding_steps)
 
             if spec_resource_manager is not None:
                 spec_resource_manager.add_dummy_requests(
@@ -1546,7 +1548,6 @@ class PyTorchModelEngine(ModelEngine):
 
         return lora_params
 
-    @torch.compile(options={"max-autotune": True})
     def _update_draft_input_tensors(self,
                                     num_accepted_tokens_device: torch.Tensor,
                                     new_tokens_device: torch.Tensor,
@@ -1671,7 +1672,6 @@ class PyTorchModelEngine(ModelEngine):
 
         return inputs, self.gather_ids_cuda[:num_generation_tokens]
 
-    @torch.compile(options={"max-autotune": True})
     def _update_target_input_tensors(
             self, num_accepted_tokens_device: torch.Tensor,
             new_tokens_device: torch.Tensor,
diff --git a/tensorrt_llm/_torch/pyexecutor/py_executor.py b/tensorrt_llm/_torch/pyexecutor/py_executor.py
index 3af32ebe4..bf42281ab 100644
--- a/tensorrt_llm/_torch/pyexecutor/py_executor.py
+++ b/tensorrt_llm/_torch/pyexecutor/py_executor.py
@@ -1708,7 +1708,6 @@ class PyExecutor:
                 self.iter_counter += 1
 
     @nvtx_range("_accept_draft_tokens")
-    @torch.compile(options={"max-autotune": True})
     def _accept_draft_tokens(
         self, scheduled_batch: ScheduledRequests,
         target_outputs: SampleStateTensors,
diff --git a/tensorrt_llm/_torch/speculative/drafting_loops.py b/tensorrt_llm/_torch/speculative/drafting_loops.py
index 159cd9d52..8c828c986 100644
--- a/tensorrt_llm/_torch/speculative/drafting_loops.py
+++ b/tensorrt_llm/_torch/
```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Pinned (page-locked) memory enables faster host-to-device data transfers. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## c5f52ab304 - [TRTLLM-8376][feat] top-p optimization (removes redundant softmax) (#9411)

- **Date**: 2025-11-25
- **Author**: mpikulski
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- Integer quantization
- Batching optimization
- Reduce synchronization overhead

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/sampling_utils.py | 48 +++++++++++++++++-------
 tests/unittest/utils/util.py                     |  5 +++
 2 files changed, 39 insertions(+), 14 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/sampling_utils.py b/tensorrt_llm/_torch/pyexecutor/sampling_utils.py
index 35e64afe4..954280612 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampling_utils.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampling_utils.py
@@ -171,27 +171,47 @@ def top_k_top_p_sampling_batch(
         sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
 
         # compute cumulative probability distribution of each sample
-        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
+        probs_sorted = torch.softmax(sorted_logits, dim=-1)
+        cumulative_probs = torch.cumsum(probs_sorted, dim=-1)
 
         # get the location of top_p
-        # NB: Currently selecting the smallest index with cumulative_probs > top_p.
+        # NB: Currently selecting the smallest index with cumulative_probs >= top_p.
         #     Thus, top_p -> 0 resembles greedy; agreement requires torch.sort(..., stable=True).
-        sorted_indices_to_remove = cumulative_probs > top_p
-        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
-        sorted_indices_to_remove[:, 0] = 0
-
-        # set the logits to -inf for token indices outside top_p
-        indices_to_remove = sorted_indices_to_remove.scatter(
-            1, sorted_indices, sorted_indices_to_remove
+        mask_to_remove = cumulative_probs >= top_p  # at least one 'True' per row
+        last_index_to_keep = torch.searchsorted(
+            mask_to_remove.to(torch.int8, non_blocking=True),
+            torch.ones((1,), dtype=torch.int8, device=mask_to_remove.device).expand(
+                (mask_to_remove.size(0), 1)
+            ),
+            right=False,
+            out_int32=True,
+        )
+        mask_to_remove.scatter_(
+            1,
+            last_index_to_keep,
+            torch.zeros((1,), dtype=torch.bool, device=mask_to_remove.device).expand_as(
+                last_index_to_keep
+            ),
         )
-        logits = logits.masked_fill(indices_to_remove, float("-inf"))
 
-    # compute probability distribution
-    softmax = torch.softmax(logits, dim=-1)
+        # mask not selected probs
+        probs_sorted.masked_fill_(mask_to_remove, 0.0)
+        probs = torch.empty_like(probs_sorted)
+        probs.scatter_(1, sorted_indices, probs_sorted)
+        probs /= cumulative_probs[  # renormalize probs
+            torch.arange(
+                cumulative_probs.size(0), dtype=torch.int32, device=cumulative_probs.device
+            ),  # needed for advanced indexing
+            last_index_to_keep.squeeze(-1),
+        ].unsqueeze(-1)
+        del logits  # do not use, inconsistent with probs
+    else:
+        # compute probability distribution
+        probs = torch.softmax(logits, dim=-1)
 
     # sample from the distribution and generate result of [batch_size, 1]
-    next_tokens = torch.multinomial(softmax, num_samples=1, generator=generator).squeeze(-1)
-    retu
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## c678774c99 - feat: Apply the new torch-flow compatible AutoTuner to both Fused MoE and NVFP4 Linear operators. (#3151)

- **Date**: 2025-04-08
- **Author**: Yukun He
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
- Reduce synchronization overhead
- MoE optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU
- NVIDIA Hopper (SM90) GPU
- Decode/generation phase

### Changed Files

```
cpp/tensorrt_llm/thop/fp4Gemm.cpp                  | 544 ++-------------------
 cpp/tensorrt_llm/thop/moeOp.cpp                    | 466 ++++--------------
 .../_torch/auto_deploy/custom_ops/fused_moe.py     |   4 +-
 .../_torch/auto_deploy/custom_ops/quant.py         |  29 +-
 tensorrt_llm/_torch/autotuner.py                   | 211 ++++----
 tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py   |  20 -
 tensorrt_llm/_torch/custom_ops/torch_custom_ops.py | 300 ++++++++++++
 tensorrt_llm/_torch/modules/fused_moe.py           |  43 +-
 tensorrt_llm/_torch/modules/linear.py              |  32 +-
 tensorrt_llm/_torch/utils.py                       |  10 +-
 tests/unittest/_torch/test_autotuner.py            |  71 ++-
 tests/unittest/_torch/test_fp4_bmm_quantize.py     |  48 --
 tests/unittest/_torch/test_fp4_linear.py           |   5 +-
 tests/unittest/_torch/test_fused_moe.py            |  50 +-
 tests/unittest/_torch/thop/test_moe_op.py          | 263 ----------
 15 files changed, 655 insertions(+), 1441 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/thop/fp4Gemm.cpp b/cpp/tensorrt_llm/thop/fp4Gemm.cpp
index b2d985ba1..f4786f767 100644
--- a/cpp/tensorrt_llm/thop/fp4Gemm.cpp
+++ b/cpp/tensorrt_llm/thop/fp4Gemm.cpp
@@ -64,10 +64,11 @@ void runGemm(at::Tensor& out, at::Tensor const& mat1, at::Tensor const& mat2, at
 
 // mat1: [B, M, K / 2], FLOAT4_E2M1X2
 // mat2: [B, N, K / 2], FLOAT4_E2M1X2
-// out:  [B, M, N], fp16/bf16/fp32
-// mat1Scale: B * ceil(M / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
-// mat2Scale: B * ceil(N / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
+// out: [B, M, N], fp16/bf16/fp32
+// mat1Scale: ceil(M / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
+// mat2Scale: ceil(N / 128) * 128 * ceil(K / sfVecSize / 4) * 4, SF_DTYPE (UE4M3 or UE8M0)
 // globalScale: [1], 1 / (((448 * 6) / mat1.abs().max()) * ((448 * 6) / mat2.abs().max()))
+// B = 1 for GEMM op as a special case
 // Only NVFP4 is currently supported
 at::Tensor fp4_bmm_impl(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
     at::Tensor const& mat2Scale, at::Tensor const& globalScale, bool sfUseUE8M0,
@@ -83,82 +84,33 @@ at::Tensor fp4_bmm_impl(at::Tensor const& mat1, at::Tensor const& mat2, at::Tens
 
     TORCH_CHECK(!sfUseUE8M0, "use UE8M0 for FP4 Block Scale Factors is not supported yet");
 
-    TORCH_CHECK(mat1.dim() == 3, "mat1 must be a batch of matrices");
-    TORCH_CHECK(mat2.dim() == 3, "mat2 must be a batch of matrices");
-    TORCH_CHECK(mat1.sizes()[0] == mat2.sizes()[0], "mat1 and mat2 must have the same number of batches");
-    TORCH_CHECK(mat1.sizes()[2] == mat2.sizes()[2], "mat1 and mat2 shapes cannot be multiplied (", mat1.sizes()[1], "x",
-        mat1.sizes()[2], " and ", mat2.sizes()[1], "x", mat2.sizes()[2], ")");
-
-    auto const m = mat1.sizes()[1];
-    auto const n = mat2.sizes()[1];
-    auto const k = mat1.sizes()[2] * 2;
-
-    auto const batch_count = mat1.sizes()[0];
-
-    auto config = maybe_config ? *maybe_config : getDefaultGemmConfig(m, n, k);
-
-    constexpr int alignment = 32;
-    TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ", alignment, ", but got mat1 shape: (",
-        mat1.sizes()[0], "x", mat1.sizes()[1], "), k: ", k, ".");
-    TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment, ", but got mat2 shape: (",
-        mat2.sizes()[0], "x", mat2.sizes()[1], ").");
-
-    if (!out_dtype)
+    int64_t m, n, k, b;
+    if (mat1.dim() == 2)
     {
-        out_dtype = torch::kHalf;
+        TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
+        TORCH_CHECK(mat1.sizes()[1] == mat2.sizes()[1], "mat1 and mat2 shapes cannot be multiplied (", mat1.sizes()[0],
+            "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");
+        m = mat1.sizes()[0];
+        n = mat2.sizes()[0];
+        k = mat1.sizes()[1] * 2;
+        b = 1;
     }
-    TORCH_CHECK(out_dtype == torch::k
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## c7548ad72c - perf: Add optimizations for deepseek in min latency mode (#3093)

- **Date**: 2025-04-02
- **Author**: Zongfei Jing
- **Categories**: Throughput/Latency

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
.../mixtureOfExpertsBackendBenchmarkFixture.h      |   7 +-
 cpp/tensorrt_llm/kernels/allReduceFusionKernels.cu | 284 ++++++++++++
 cpp/tensorrt_llm/kernels/allReduceFusionKernels.h  |  28 ++
 cpp/tensorrt_llm/kernels/customAllReduceKernels.h  |   2 +
 ...btensorrt_llm_internal_cutlass_kernels_static.a |   4 +-
 ...llm_internal_cutlass_kernels_static.pre_cxx11.a |   4 +-
 .../aarch64-linux-gnu/version.txt                  |   6 +-
 .../internal_cutlass_kernels/include/moe_kernels.h |  79 +++-
 ...btensorrt_llm_internal_cutlass_kernels_static.a |   4 +-
 ...llm_internal_cutlass_kernels_static.pre_cxx11.a |   4 +-
 .../x86_64-linux-gnu/version.txt                   |   6 +-
 .../mixtureOfExperts/mixtureOfExpertsPlugin.cpp    |  12 +-
 .../thop/deepseekAllreduceFusionOp.cpp             |  92 +++-
 cpp/tensorrt_llm/thop/moeOp.cpp                    | 137 +++++-
 .../kernels/allReduce/allReduceFusionTest.cu       | 478 ++++++++++++++++++++-
 .../unit_tests/kernels/mixtureOfExpertsTest.cu     |  10 +-
 tensorrt_llm/_torch/models/modeling_deepseekv3.py  | 131 ++++--
 tensorrt_llm/_torch/modules/fused_moe.py           | 105 +++--
 tensorrt_llm/_torch/modules/gated_mlp.py           |   8 +-
 tensorrt_llm/functional.py                         |   2 +
 .../_torch/multi_gpu/test_deepseek_allreduce.py    | 241 +++++++++--
 tests/unittest/_torch/thop/test_moe_op.py          |  10 +-
 22 files changed, 1488 insertions(+), 166 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h b/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h
index b6b58ae9a..3d01cc15b 100644
--- a/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h
+++ b/cpp/micro_benchmarks/mixtureOfExpertsBackendBenchmarkFixture.h
@@ -497,7 +497,7 @@ public:
         auto const gated_inter = mInterSize * mGatedMultiplier;
 
         size_t workspace_size = mMoERunner.getWorkspaceSize(mTotalTokens, mHiddenSize, mInterSize, mNumExperts, mK,
-            mActType, {}, mUseLora, /*use_fp8_block_scaling=*/false, mUsePrequantScale);
+            mActType, {}, mUseLora, /*use_fp8_block_scaling=*/false, /*min_latency_mode=*/false, mUsePrequantScale);
 
         mWorkspace = allocBuffer<char>(workspace_size);
         size_t const expert_matrix_size = mNumExperts * mHiddenSize * mInterSize;
@@ -625,7 +625,7 @@ public:
         GemmProfilerBackend profiler;
         profiler.init(mMoERunner, gemm_to_profile, typeToDtypeID<DataType>(), typeToDtypeID<WeightType>(),
             typeToDtypeID<OutputType>(), mNumExperts, mK, mHiddenSize, mInterSize, mGroupSize, mActType, mUseBias,
-            mUseLora, parallelism_config);
+            mUseLora, /*min_latency_mode=*/false, parallelism_config);
         auto workspace_size = profiler.getWorkspaceSize(mTotalTokens);
         auto workspace = bufferManager->gpu(workspace_size);
 
@@ -729,11 +729,12 @@ public:
     void runMoEPermute(MOEParallelismConfig parallelism_config)
     {
         auto stream = streamPtr->get();
+        MoeMinLatencyParams min_latency_params;
         mMoERunner.runMoe(mInputTensor, nullptr, mSelectedExperts, mUseFinalScale ? mScaleProbs : nullptr,
             mExpertWeight1, mExpertBias1, mActType, mExpertWeight2, mExpertBias2, mQuantParams, mTotalTokens,
             mHiddenSize, mInterSize, mNumExperts, mK, mWorkspace, mFinalOutput, mSourceToExpandedMap,
             parallelism_config, mUseLora, mLoraParams,
-            /*use_fp8_block_scaling=*/false, stream);
+            /*use_fp8_block_scaling=*/false, /*min_latency_mode=*/false, min_latency_params, stream);
     }
 
     void runBenchmark(benchmark::State& state);
diff --git a/cpp/tensorrt_llm/kernels/allReduceFusionKernels.cu b/cpp/tensorrt_llm/kernels/allReduceFusionKernels.cu
index afdc8bc91..07b0da895 100644
--- a/cpp/tensorrt_llm/kernels/allReduceFusionKernels.cu
+++ b/cpp/tensorrt_llm/kernels/allReduceFusionKernels.cu
@@ -741,4 +741,288 @@ void** Workspace::get_workspace()
 {
     return reinterpret_cast<void**>(m_workspace);
 }
+
+/////////////////////////////////////////////////////////////////
+//                  * MoE Reduction Fusion *                   //
+/////////////////////////////////////////////////////////////////
+
+template <typename DType, int NRanks, bool ResidualOut, bool NormOut, bool QuantOut>
+__global__ void moereduce_allreduce_fusion_kernel_oneshot_lamport(MoeReductionAllReduceFusionParams params)
+{
+#if (defined(_
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## c7d8cc1f34 - [None][perf] Use UE8M0 FP8 quant kernel for DeepGemm blockwise GEMM (#11607)

- **Date**: 2026-02-23
- **Author**: Chang Liu
- **Categories**: Kernel Optimization, Quantization Optimization

### Optimization Techniques

- Operator fusion
- FP8 quantization
- Integer quantization
- Parallelism optimization
- Triton kernel
- GEMM optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/custom_ops/torch_custom_ops.py | 5 +++--
 1 file changed, 3 insertions(+), 2 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py b/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
index 482e22d5a..7db85b4b3 100644
--- a/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
+++ b/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
@@ -6,7 +6,6 @@ import torch
 import triton  # type: ignore[import]
 
 import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
-import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils
 from tensorrt_llm import deep_gemm
 from tensorrt_llm._utils import get_sm_version
 from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy
@@ -1486,7 +1485,9 @@ class fp8SwapABGemmRunner(TunableRunner):
         tactic: int = -1,
     ) -> torch.Tensor:
         input, weight, weight_scale = inputs
-        a, a_sf = fp8_utils.per_token_quant_and_transform(input)
+        a, a_sf = torch.ops.trtllm.fp8_quantize_1x128(input, use_ue8m0=True)
+        a_sf = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(
+            a_sf.transpose(0, 1))
         output = torch.empty(
             (input.size(0), weight.size(0)),
             device=input.device,

```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## c8b9998acb - [TRTLLM-8637][feat] Optimize the routing kernel for DeepseekV3 (MoE CUTLASS backend); Add support for KimiK2 and Qwen-next (MoE TRTLLM backend) (#7761)

- **Date**: 2025-10-20
- **Author**: ChristinaZ
- **Categories**: Kernel Optimization

### Optimization Techniques

- Custom CUDA kernel
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

- General LLM inference

### Changed Files

```
cpp/tensorrt_llm/kernels/moeTopKFuncs.cuh          |  88 ++-
 cpp/tensorrt_llm/kernels/noAuxTcKernels.cu         | 832 ++++++---------------
 cpp/tensorrt_llm/kernels/noAuxTcKernels.h          |   8 +-
 .../trtllmGenKernels/blockScaleMoe/DevKernel.h     |  65 +-
 .../blockScaleMoe/RoutingDeepSeek.cu               | 199 +++--
 .../blockScaleMoe/RoutingKernel.cuh                |  53 +-
 .../trtllmGenKernels/blockScaleMoe/RoutingKernel.h |  23 +-
 .../blockScaleMoe/RoutingKernelTopK.cuh            |  65 +-
 .../blockScaleMoe/RoutingLlama4.cu                 |  68 +-
 .../blockScaleMoe/RoutingRenormalize.cu            | 125 ++--
 cpp/tensorrt_llm/thop/fp4BlockScaleMoe.cpp         |   6 +-
 cpp/tensorrt_llm/thop/fp8BlockScaleMoe.cpp         |   2 +-
 cpp/tensorrt_llm/thop/mxFp4BlockScaleMoe.cpp       |   4 +-
 cpp/tensorrt_llm/thop/noAuxTcOp.cpp                | 117 ++-
 .../kernels/routing/routingDeepSeekTest.cpp        |  96 ++-
 .../kernels/routing/routingRenormalizeTest.cpp     |  37 +-
 tensorrt_llm/_torch/models/modeling_deepseekv3.py  |  27 +-
 tests/unittest/_torch/thop/parallel/test_moe.py    |  49 +-
 .../unittest/_torch/thop/parallel/test_noaux_tc.py |   3 +-
 19 files changed, 1013 insertions(+), 854 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/moeTopKFuncs.cuh b/cpp/tensorrt_llm/kernels/moeTopKFuncs.cuh
index 933b599db..665086c7d 100644
--- a/cpp/tensorrt_llm/kernels/moeTopKFuncs.cuh
+++ b/cpp/tensorrt_llm/kernels/moeTopKFuncs.cuh
@@ -51,7 +51,7 @@ struct TopKRedType
     static __host__ __device__ inline TypeCmp makeCmpVal(T val, int32_t idx = 0)
     {
         auto valueBits = cub::Traits<T>::TwiddleIn(reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(val));
-        TypeCmp compactTmp = reinterpret_cast<TypeCmp&>(valueBits);
+        TypeCmp compactTmp = valueBits;
         compactTmp = (compactTmp << kMoveBits) | (0xFFFF & (kMaxIdx - idx));
         // Use 65535 minus idx to give higher priority to elements with smaller indices.
         return compactTmp;
@@ -162,9 +162,28 @@ struct Sort<4, RedType>
     }
 };
 
+template <int K, typename Type>
+__forceinline__ __device__ void reduceTopK(cg::thread_block_tile<kWARP_SIZE> const& warp, Type (&out)[K],
+    int32_t (&outIdx)[K], Type value, int32_t idx, Type const minValue, int actualK = K)
+{
+    static_assert(K > 0, "Top K must have K > 0");
+    static_assert(K < kWARP_SIZE, "Top K must have K < kWARP_SIZE");
+    using RedType = TopKRedType<Type>;
+    RedType topK{value, idx};
+    typename RedType::TypeCmp packedMax{};
+#pragma unroll
+    for (int kk = 0; kk < actualK; ++kk) //@todo: check if actualK is correct
+    {
+        topK = kk > 0 && packedMax == topK.compValIdx ? RedType{minValue, idx} : topK;
+        // get the next largest value
+        packedMax = topK.reduce(warp);
+        RedType::unpack(out[kk], outIdx[kk], packedMax);
+    }
+};
+
 template <int K, typename Type, int N, bool IsSorted = false>
-__device__ void reduceTopK(cg::thread_block_tile<kWARP_SIZE> const& warp, Type (&out)[K], int32_t (&outIdx)[K],
-    Type (&value)[N], int32_t (&idx)[N], Type minValue)
+__device__ void reduceTopKFunc(cg::thread_block_tile<kWARP_SIZE> const& warp, Type (&out)[K], int32_t (&outIdx)[K],
+    Type (&value)[N], int32_t (&idx)[N], Type minValue, int actualK = K)
 {
     static_assert(K > 0, "Top K must have K > 0");
     static_assert(K < kWARP_SIZE, "Top K must have K < kWARP_SIZE");
@@ -184,7 +203,7 @@ __device__ void reduceTopK(cg::thread_block_tile<kWARP_SIZE> const& warp, Type (
     }
     typename RedType::TypeCmp packedMax{};
 #pragma unroll
-    for (int kk = 0; kk < K; ++kk)
+    for (int kk = 0; kk < actualK; ++kk)
     {
         bool update = kk > 0 && packedMax == topK[0].compValIdx;
 #pragma unroll
@@ -198,6 +217,67 @@ __device__ void reduceTopK(cg::thread_block_tile<kWARP_SIZE> const& warp, Type (
     }
 };
 
+template <int K, typename Type, int N>
+__forceinline__ __device__ void reduceTopK(cg::thread_block_tile<kWARP_SIZE> const& warp, Type (&out)[K],
+    int32_t (&outIdx)[K], Type (&value)[N], int32_t (&idx)[N], Type const minValue, int actualK = K)
+{
+    static_assert(K > 0, "Top K must have K > 0");
+    static_assert(K < kWARP_SIZE, "Top K must have 
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## cb1d8d130f - [TRTLLM-10791][feat] TorchSampler general host time optimization (#11141)

- **Date**: 2026-02-13
- **Author**: Yukun He
- **Categories**: Host-side Optimization

### Optimization Techniques

- Vectorized memory access
- Async/stream-based execution
- KV cache optimization
- Batching optimization
- Pinned memory
- Speculative decoding

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/pyexecutor/llm_request.py      |   8 +-
 tensorrt_llm/_torch/pyexecutor/sampler.py          | 296 ++++++++++++++++-----
 .../unittest/_torch/sampler/test_torch_sampler.py  |  24 +-
 3 files changed, 253 insertions(+), 75 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/pyexecutor/llm_request.py b/tensorrt_llm/_torch/pyexecutor/llm_request.py
index 092fde5a4..da39ea59f 100644
--- a/tensorrt_llm/_torch/pyexecutor/llm_request.py
+++ b/tensorrt_llm/_torch/pyexecutor/llm_request.py
@@ -1,14 +1,10 @@
 from copy import copy, deepcopy
 from dataclasses import dataclass, field
-from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
+from typing import Any, Dict, List, Optional, Union
 
 import torch
 
 import tensorrt_llm.bindings
-
-if TYPE_CHECKING:
-    from tensorrt_llm._torch.pyexecutor.sampler import Strategy
-
 from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
 from tensorrt_llm.bindings import executor as tllm_executor
 from tensorrt_llm.executor.result import TokenLogprobs
@@ -676,8 +672,6 @@ class LlmRequest(tensorrt_llm.bindings.internal.batch_manager.LlmRequest):
             additional_outputs=additional_outputs)
         self.child_requests = []
 
-        self._py_sampling_strategy: "Strategy | None" = None
-
         self._py_embedding_bias_1d: Optional[torch.Tensor] = None
         if hasattr(self, 'embedding_bias') and self.embedding_bias is not None:
             # Pre-squeeze to 1D if needed (remove batch dimension)
diff --git a/tensorrt_llm/_torch/pyexecutor/sampler.py b/tensorrt_llm/_torch/pyexecutor/sampler.py
index db6210da8..b6841f086 100644
--- a/tensorrt_llm/_torch/pyexecutor/sampler.py
+++ b/tensorrt_llm/_torch/pyexecutor/sampler.py
@@ -15,7 +15,7 @@
 import enum
 import sys
 from abc import ABC, abstractmethod
-from collections import defaultdict, namedtuple
+from collections import defaultdict
 from collections.abc import Iterable
 from concurrent import futures
 from dataclasses import dataclass
@@ -369,10 +369,6 @@ def _get_max_beam_width(request: LlmRequest) -> int:
     return max_beam_width
 
 
-def _request_sampling_params_cachable(params: UtilsSamplingParams) -> bool:
-    return not params.use_beam_search
-
-
 def _request_get_sampling_params(request: LlmRequest) -> UtilsSamplingParams:
     sampling_config = request.sampling_config
     temperature = _unwrap_singleton(cast(Optional[list[float]], sampling_config.temperature))
@@ -393,16 +389,13 @@ def _request_get_sampling_params(request: LlmRequest) -> UtilsSamplingParams:
 
 
 def _request_strategy(request: LlmRequest, *, vocab_size: int) -> Strategy:
-    # We try to cache the resolved strategy on the request object, as it's not cheap enough to
-    # resolve it on every iteration.
-    if request._py_sampling_strategy is not None:
-        return request._py_sampling_strategy
+    """Resolve the sampling strategy for a request.
 
+    Note: Callers inside _group_requests_by_strategy_key benefit from store.strategies
+    caching, which ensures this function is called at most once per request per slot.
+    """
     params = _request_get_sampling_params(request)
-    sampling_strategy = resolve_sampling_strategy(params, vocab_size=vocab_size)
-    if _request_sampling
```

### Analysis Summary

Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Pinned (page-locked) memory enables faster host-to-device data transfers. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## cc16289dfe - [None][feat] Optimize by fuse nvfp4_quant to layernorm_gated for mamba2_mixer (#11473)

- **Date**: 2026-03-07
- **Author**: Wanli Jiang
- **Categories**: Fusion, Quantization Optimization

### Optimization Techniques

- Custom CUDA kernel
- Vectorized memory access
- Operator fusion
- Async/stream-based execution
- FP8 quantization
- Integer quantization
- Parallelism optimization
- Batching optimization
- Triton kernel
- Reduce synchronization overhead
- MoE optimization
- Attention mechanism optimization
- GEMM optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU

### Changed Files

```
cpp/tensorrt_llm/CMakeLists.txt                    |   1 +
 cpp/tensorrt_llm/kernels/CMakeLists.txt            |   2 +
 .../kernels/fusedGatedRMSNormQuant/CMakeLists.txt  |  40 ++
 .../fusedGatedRMSNormQuant.cu                      | 763 +++++++++++++++++++++
 .../fusedGatedRMSNormQuant.cuh                     |  83 +++
 cpp/tensorrt_llm/thop/CMakeLists.txt               |   1 +
 cpp/tensorrt_llm/thop/fusedGatedRMSNormQuant.cpp   | 168 +++++
 tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py   |  18 +
 .../_torch/modules/mamba/layernorm_gated.py        |  38 +-
 tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py  |  31 +-
 .../_torch/modules/mamba/ssd_chunk_scan.py         |  14 +-
 tensorrt_llm/_torch/modules/rms_norm.py            |   2 +-
 .../_torch/modules/mamba/test_layernorm_gated.py   | 340 +++++++++
 13 files changed, 1486 insertions(+), 15 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/CMakeLists.txt b/cpp/tensorrt_llm/CMakeLists.txt
index 77d48d54d..2813da8c3 100644
--- a/cpp/tensorrt_llm/CMakeLists.txt
+++ b/cpp/tensorrt_llm/CMakeLists.txt
@@ -189,6 +189,7 @@ set(TRTLLM_LINK_LIBS
     trtllm_gen_batched_gemm
     selective_scan_src
     ws_layernorm_src
+    fusedGatedRMSNormQuant_src
     fpA_intB_gemm_src
     # moe_gemm_src
     fb_gemm_src
diff --git a/cpp/tensorrt_llm/kernels/CMakeLists.txt b/cpp/tensorrt_llm/kernels/CMakeLists.txt
index 7fde9b03d..30b2f8846 100644
--- a/cpp/tensorrt_llm/kernels/CMakeLists.txt
+++ b/cpp/tensorrt_llm/kernels/CMakeLists.txt
@@ -28,6 +28,7 @@ add_subdirectory(groupRmsNormKernels)
 add_subdirectory(llama4MinLatencyKernels)
 add_subdirectory(dsv3MinLatencyKernels)
 add_subdirectory(causalConv1d)
+add_subdirectory(fusedGatedRMSNormQuant)
 
 file(GLOB_RECURSE SRC_CPP *.cpp)
 file(GLOB_RECURSE SRC_CU *.cu)
@@ -51,6 +52,7 @@ list(FILTER SRC_CU EXCLUDE REGEX "selectiveScan/.*")
 list(FILTER SRC_CPP EXCLUDE REGEX "userbuffers/.*")
 list(FILTER SRC_CU EXCLUDE REGEX "userbuffers/.*")
 list(FILTER SRC_CU EXCLUDE REGEX "fusedLayernormKernels/.*")
+list(FILTER SRC_CU EXCLUDE REGEX "fusedGatedRMSNormQuant/.*")
 
 if(NOT ENABLE_MULTI_DEVICE)
   list(FILTER SRC_CU EXCLUDE REGEX "customAllReduceKernels*.*cu$")
diff --git a/cpp/tensorrt_llm/kernels/fusedGatedRMSNormQuant/CMakeLists.txt b/cpp/tensorrt_llm/kernels/fusedGatedRMSNormQuant/CMakeLists.txt
new file mode 100644
index 000000000..1bc425f8a
--- /dev/null
+++ b/cpp/tensorrt_llm/kernels/fusedGatedRMSNormQuant/CMakeLists.txt
@@ -0,0 +1,40 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
+# All rights reserved. SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License"); you may not
+# use this file except in compliance with the License. You may obtain a copy of
+# the License at
+#
+# http://www.apache.org/licenses/LICENSE-2.0
+#
+# Unless required by applicable law or agreed to in writing, software
+# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
+# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
+# License for the specific language governing permissions and limitations under
+# the License.
+
+file(GLOB_RECURSE SRC_CU *.cu)
+add_library(fusedGatedRMSNormQuant_src STATIC ${SRC_CU})
+
+if("100" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
+  # for blackwell
+  set(FUSED_GATED_RMSNORM_NVCC_FLAGS)
+  list(APPEND FUSED_GATED_RMSNORM_NVCC_FLAGS --extra-device-vectorization)
+  list(APPEND FUSED_GATED_RMSNORM_NVCC_FLAGS
+       --ptxas-options=--warn-on-local-memory-usage,--warn-on-spills)
+
+  target_compile_options(
+    fusedGatedRMSNormQuant_src
+    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${FUSED_GATED_RMSNORM_NVCC_FLAGS}>)
+endif()
+
+if(NOT WIN32)
+  target_compile_options(
+    fusedGatedRMSNormQuant_src
+    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-psabi>)
+endif()
+
+set_property(TARGET fusedG
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Vectorized load/store operations improve memory bandwidth utilization by processing multiple elements per memory transaction. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## cc989ea49f - perf: Optimise MOE prologue to use fused setup function (#3790)

- **Date**: 2025-04-30
- **Author**: djns99
- **Categories**: Fusion

### Optimization Techniques

- General code optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../tensorrt_llm_internal_cutlass_kernels_static.tar.xz               | 4 ++--
 .../kernels/internal_cutlass_kernels/aarch64-linux-gnu/version.txt    | 4 ++--
 .../tensorrt_llm_internal_cutlass_kernels_static.tar.xz               | 4 ++--
 .../kernels/internal_cutlass_kernels/x86_64-linux-gnu/version.txt     | 4 ++--
 4 files changed, 8 insertions(+), 8 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/aarch64-linux-gnu/tensorrt_llm_internal_cutlass_kernels_static.tar.xz b/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/aarch64-linux-gnu/tensorrt_llm_internal_cutlass_kernels_static.tar.xz
index fbf0ae0f0..b1263959c 100644
--- a/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/aarch64-linux-gnu/tensorrt_llm_internal_cutlass_kernels_static.tar.xz
+++ b/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/aarch64-linux-gnu/tensorrt_llm_internal_cutlass_kernels_static.tar.xz
@@ -1,3 +1,3 @@
 version https://git-lfs.github.com/spec/v1
-oid sha256:3eed242cb94e3392440b7f955c45df9a898985aeb310a17be01d58c5f3e86d7f
-size 46415808
+oid sha256:cbf058e66f8b7683bc602ff489b922440d0fbff08eeffd6d7bb4dbd031257c30
+size 47927952
diff --git a/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/aarch64-linux-gnu/version.txt b/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/aarch64-linux-gnu/version.txt
index b9b50e4e9..3e2967a55 100644
--- a/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/aarch64-linux-gnu/version.txt
+++ b/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/aarch64-linux-gnu/version.txt
@@ -1,2 +1,2 @@
-935953fc376653cc2f28d5bde5270910d8cbce3df27a60916d48c0677843b782  libtensorrt_llm_internal_cutlass_kernels_static.a
-commit 2e9a657ce59d0b29618eef31114b9a10dd1756e0
+55b2914da188e320fc1692f9e20e5e373748bf7798bf35783784e13de19057f9  libtensorrt_llm_internal_cutlass_kernels_static.a
+commit 639d3fbebcd5bed54e0086824747b43e8b4aa8d6
diff --git a/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/x86_64-linux-gnu/tensorrt_llm_internal_cutlass_kernels_static.tar.xz b/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/x86_64-linux-gnu/tensorrt_llm_internal_cutlass_kernels_static.tar.xz
index cd69f3601..5c2e154c8 100644
--- a/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/x86_64-linux-gnu/tensorrt_llm_internal_cutlass_kernels_static.tar.xz
+++ b/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/x86_64-linux-gnu/tensorrt_llm_internal_cutlass_kernels_static.tar.xz
@@ -1,3 +1,3 @@
 version https://git-lfs.github.com/spec/v1
-oid sha256:f2f239ca8e6fc6b29af71a2b58c1dc9a1cbdaea09df8eb57d36100f2a5a5644d
-size 46050764
+oid sha256:d22f863a9ec02dce97dc47e8e3f4c1bb2f8faf97252499f742286fa439f79b6e
+size 47541200
diff --git a/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/x86_64-linux-gnu/version.txt b/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/x86_64-linux-gnu/version.txt
index d789c1807..6d2ecf644 100644
--- a/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/x86_64-linux-gnu/version.txt
+++ b/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/x86_64-linux-gnu/version.txt
@@ -1,2 +1,2 @@
-2d3e677e590f6ad5ad1c414e8ae92e5f207b733b382752f9e4708e0971fc3c1a  libtensorrt_llm_internal_cutlass_kernels_static.a
-commit 2e9a657ce59d0b29618eef31114b9a10dd1756e0
+017677b6508406ad084e822a97a3217004c3153b8839cbafed829e7f022009b3  libtensorrt_llm_internal_cutlass_kernels_static.a
+commit 639d3fbebcd5bed54e008
```

### Analysis Summary

This commit applies performance optimization techniques to improve inference efficiency. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## ccc64da287 - [TRTLLM-9847][fix] WAR fix hanging fused allreduce. (#10087)

- **Date**: 2025-12-23
- **Author**: Grzegorz Kwasniewski
- **Categories**: Fusion

### Optimization Techniques

- Parallelism optimization
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/auto_deploy/config/default.yaml           | 1 +
 tensorrt_llm/_torch/auto_deploy/transform/library/sharding.py | 8 +++++++-
 2 files changed, 8 insertions(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/config/default.yaml b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
index 362483291..68c508622 100644
--- a/tensorrt_llm/_torch/auto_deploy/config/default.yaml
+++ b/tensorrt_llm/_torch/auto_deploy/config/default.yaml
@@ -81,6 +81,7 @@ transforms:
     sharding_source: ['manual', 'factory', 'heuristic']
     support_partial_config: true
     sharding_dims: ['tp', 'ep', 'bmm']
+    shard_all_unprocessed: true
     allreduce_strategy: 'NCCL'
     dist_backend: auto
     requires_shape_prop: true
diff --git a/tensorrt_llm/_torch/auto_deploy/transform/library/sharding.py b/tensorrt_llm/_torch/auto_deploy/transform/library/sharding.py
index bae85f3a2..044cf7002 100644
--- a/tensorrt_llm/_torch/auto_deploy/transform/library/sharding.py
+++ b/tensorrt_llm/_torch/auto_deploy/transform/library/sharding.py
@@ -140,6 +140,11 @@ class ShardingTransformConfig(TransformConfig):
     sharding_dims: List[ShardingDim] = Field(
         default_factory=lambda: [ShardingDim.TP, ShardingDim.EP, ShardingDim.BMM]
     )
+    shard_all_unprocessed: bool = Field(
+        default=False,
+        description="When True, apply simple shard (column split + all_gather) to "
+        "'leftover' linear nodes that are not part of any layer subgraph.",
+    )
     allreduce_strategy: AllReduceStrategy = Field(
         default=AllReduceStrategy.AUTO,
         description="AllReduce strategy for distributed operations. "
@@ -2608,7 +2613,8 @@ def detect_column_row_shard(
                 num_attention_shards += 1
 
     # simple shard remaining linear nodes
-    num_simple_shards += _process_simple_shard(unprocessed_linear_nodes, transform_container)
+    if config.shard_all_unprocessed:
+        num_simple_shards += _process_simple_shard(unprocessed_linear_nodes, transform_container)
     num_column_row_shards += num_ssm_shards
     num_shards = num_simple_shards + num_column_row_shards
     ad_logger.info(

```

### Analysis Summary

Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## cd4e639536 - [None][feat] Async pp send. (#9952)

- **Date**: 2025-12-13
- **Author**: Yuxian Qiu
- **Categories**: Parallelism/Async

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- KV cache optimization
- Parallelism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/distributed/communicator.py | 57 ++++++++++++++++++++-----
 1 file changed, 47 insertions(+), 10 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/distributed/communicator.py b/tensorrt_llm/_torch/distributed/communicator.py
index 67790b240..5e4968f29 100644
--- a/tensorrt_llm/_torch/distributed/communicator.py
+++ b/tensorrt_llm/_torch/distributed/communicator.py
@@ -16,6 +16,7 @@ try:
 except Exception:
     MPI = None  # deferred; functions will error if used when ENABLE_MULTI_DEVICE is True
 
+from tensorrt_llm._torch.hostfunc import hostfunc
 from tensorrt_llm._utils import (mpi_allgather, mpi_barrier, mpi_comm,
                                  mpi_disabled, mpi_isend, mpi_isend_object,
                                  mpi_recv, mpi_recv_object, mpi_send,
@@ -782,18 +783,57 @@ class TorchDist(Distributed):
             return ret[0]
 
 
-class PPCommNCCL:
+class PPCommBase:
 
     def __init__(self, global_mapping: Mapping):
         self.mapping = global_mapping
+        self.tensor_ready_event = torch.cuda.Event()
+        self.send_stream = torch.cuda.Stream()
+        self.tensor_cache = {}
+
+    def _cache_tensor(self, tensor: torch.Tensor):
+        cache_id = id(tensor)
+        self.tensor_cache[cache_id] = tensor
+
+    @hostfunc
+    def _release_tensor(self, tensor: torch.Tensor):
+        cache_id = id(tensor)
+        del self.tensor_cache[cache_id]
+
+    @abstractmethod
+    def direct_send(self, tensor: torch.Tensor, dest: int):
+        raise NotImplementedError("direct_send is not implemented")
+
+    def send(self, tensor: torch.Tensor, dest: Optional[int] = None):
+        if dest is None:
+            dest = self.mapping.next_pp_rank()
+
+        # NCCL send kernel in send_stream cannot be captured,
+        # so we send in the current stream instead in CUDA graph cases.
+        if torch.cuda.is_current_stream_capturing():
+            self.direct_send(tensor, dest)
+            return
+
+        self.tensor_ready_event.record()
+        with torch.cuda.stream(self.send_stream):
+            self.tensor_ready_event.wait()
+            # tensor may be released before NCCL send finished,
+            # so we cache it first and release it after send finished.
+            self._cache_tensor(tensor)
+            self.direct_send(tensor, dest)
+            self._release_tensor(tensor)
+
+
+class PPCommNCCL(PPCommBase):
+
+    def __init__(self, global_mapping: Mapping):
+        super().__init__(global_mapping)
         self.nccl_comm = torch.classes.trtllm.NcclCommunicatorOp(
             self.mapping.world_size,
             self.mapping.rank,
         )
 
-    def send(self, tensor: torch.Tensor, dest: Optional[int] = None):
-        if dest is None:
-            dest = self.mapping.next_pp_rank()
+    def direct_send(self, tensor: torch.Tensor, dest: int):
         self.nccl_comm.send(tensor, dest)
 
     def recv(self, tensor: torch.Tensor, src: Optional[int] = None):
@@ -802,10 +842,10 @@ class PPCommNCCL:
         self.nccl_comm.recv(tensor, src)
 
 
-class PPCommTorch:
+class PPCommTorch(PPCommBase):
 
     def __init__(self,
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## cea5dd1e38 - [TRTLLM-5835][feat] Optimized Mamba2Mixer prefill (#5128)

- **Date**: 2025-06-16
- **Author**: tomeras91
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- KV cache optimization
- Parallelism optimization
- Batching optimization
- Attention mechanism optimization

### Applicable Conditions

- Prefill phase
- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/models/modeling_nemotron_h.py  |  18 +-
 .../_torch/modules/mamba/mamba2_metadata.py        |  47 ++++
 tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py  | 270 +++++++++------------
 tensorrt_llm/_torch/pyexecutor/resource_manager.py |   4 +-
 4 files changed, 183 insertions(+), 156 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_nemotron_h.py b/tensorrt_llm/_torch/models/modeling_nemotron_h.py
index c3c037033..0e9f9a033 100644
--- a/tensorrt_llm/_torch/models/modeling_nemotron_h.py
+++ b/tensorrt_llm/_torch/models/modeling_nemotron_h.py
@@ -20,6 +20,8 @@ from torch import nn
 from torch.nn import functional as F
 from transformers import AutoConfig, PretrainedConfig
 
+from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
+
 from ..attention_backend import AttentionMetadata
 from ..model_config import ModelConfig
 from ..modules.attention import Attention
@@ -71,6 +73,7 @@ class MLPLayer(MLP):
         self,
         hidden_states: torch.Tensor,
         attn_metadata: AttentionMetadata,
+        **kwargs,
     ) -> torch.Tensor:
         return super().forward(hidden_states)
 
@@ -99,6 +102,7 @@ class TransformerLayer(Attention):
         self,
         hidden_states: torch.Tensor,
         attn_metadata: AttentionMetadata,
+        **kwargs,
     ) -> torch.Tensor:
         return super().forward(position_ids=None,
                                hidden_states=hidden_states,
@@ -153,12 +157,13 @@ class NemotronHLayer(DecoderLayer):
         position_ids: torch.IntTensor,
         hidden_states: torch.Tensor,
         attn_metadata: AttentionMetadata,
+        **kwargs,
     ) -> torch.Tensor:
 
         residual = hidden_states
 
         hidden_states = self.norm(hidden_states)
-        hidden_states = self.mixer(hidden_states, attn_metadata)
+        hidden_states = self.mixer(hidden_states, attn_metadata, **kwargs)
         hidden_states = torch.add(hidden_states, residual)
 
         return hidden_states
@@ -190,6 +195,8 @@ class NemotronHModel(DecoderModel):
             dtype=config.torch_dtype,
         )
 
+        self.mamba_metadata: Optional[Mamba2Metadata] = None
+
     def forward(
         self,
         attn_metadata: AttentionMetadata,
@@ -203,13 +210,20 @@ class NemotronHModel(DecoderModel):
                 "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
             )
 
+        if self.mamba_metadata is None or self.mamba_metadata.max_batch_size != attn_metadata.max_num_requests:
+            self.mamba_metadata = Mamba2Metadata(attn_metadata.max_num_requests)
+        self.mamba_metadata.prepare(attn_metadata)
+
         if inputs_embeds is None:
             inputs_embeds = self.embed_tokens(input_ids)
 
         hidden_states = inputs_embeds
 
         for layer in self.layers:
-            hidden_states = layer(position_ids, hidden_states, attn_metadata)
+            hidden_states = layer(position_ids,
+                                  hidden_states,
+                                  attn_metadata,
+                                  mamba_metadata=self.mamba_metadata)
 
         hidden_states = self.norm_f(hidden_states)
 
diff --git a/tensorrt_llm/_torch/modules/mamba/mamba2_metadata.py b/tensorrt_llm/_torch/modules/mamb
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## d12cb9436d - [None][feat] Autodeploy add triton configs and optimize mamba prefill (#9083)

- **Date**: 2025-11-13
- **Author**: Suyog Gupta
- **Categories**: Kernel Optimization

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- FP8 quantization
- KV cache optimization
- Batching optimization
- Triton kernel
- PyTorch built-in optimized ops
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- NVIDIA Hopper (SM90) GPU
- Prefill phase
- Decode/generation phase

### Changed Files

```
LICENSE                                            |   4 +
 setup.py                                           |   1 +
 .../compile/backends/torch_cudagraph.py            |   2 +-
 .../auto_deploy/custom_ops/attention_interface.py  |  12 +-
 .../auto_deploy/custom_ops/flashinfer_attention.py |   3 +-
 ...8,N=1856,device_name=NVIDIA_H100_80GB_HBM3.json | 147 +++++++++++++++
 .../E=128,N=1856,device_name=NVIDIA_L40S.json      | 147 +++++++++++++++
 .../auto_deploy/custom_ops/fused_moe/triton_moe.py |  96 +++++++++-
 .../custom_ops/mamba/cuda_backend_causal_conv.py   |   9 +-
 .../custom_ops/mamba/torch_backend_causal_conv.py  |   1 +
 .../custom_ops/mamba/torch_backend_mamba.py        |   3 +-
 .../custom_ops/mamba/triton_backend_mamba.py       | 207 ++++++++++++++++-----
 tensorrt_llm/_torch/auto_deploy/custom_ops/mla.py  |   3 +-
 .../custom_ops/torch_backend_attention.py          |   1 +
 .../auto_deploy/custom_ops/triton_attention.py     |   3 +-
 tensorrt_llm/_torch/auto_deploy/models/factory.py  |   5 +
 tensorrt_llm/_torch/auto_deploy/models/hf.py       |   7 +
 .../_torch/auto_deploy/shim/ad_executor.py         |   4 +-
 .../custom_ops/test_cuda_causal_conv_cached_op.py  |  10 +-
 .../custom_ops/test_torch_attention_op.py          |   2 +-
 .../custom_ops/test_torch_causal_conv_cached_op.py |   9 +-
 .../custom_ops/test_torch_mamba_cached_op.py       |  10 +-
 .../custom_ops/test_triton_mamba_cached_op.py      |  19 +-
 23 files changed, 615 insertions(+), 90 deletions(-)
```

### Diff Preview

```diff
diff --git a/LICENSE b/LICENSE
index d64569567..7582da94b 100644
--- a/LICENSE
+++ b/LICENSE
@@ -1,3 +1,7 @@
+Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+
+Portions of this project are under the following copyright:
+- Copyright contributors to the vLLM project
 
                                  Apache License
                            Version 2.0, January 2004
diff --git a/setup.py b/setup.py
index 05af3eb2c..5c61029aa 100644
--- a/setup.py
+++ b/setup.py
@@ -134,6 +134,7 @@ package_data += [
     "_torch/auto_deploy/config/*.yaml",
     # Include CUDA source for fused MoE align extension so runtime JIT can find it in wheels
     '_torch/auto_deploy/custom_ops/fused_moe/moe_align_kernel.cu',
+    '_torch/auto_deploy/custom_ops/fused_moe/triton_fused_moe_configs/*'
 ]
 
 
diff --git a/tensorrt_llm/_torch/auto_deploy/compile/backends/torch_cudagraph.py b/tensorrt_llm/_torch/auto_deploy/compile/backends/torch_cudagraph.py
index 1fb094e7e..4a98593c6 100644
--- a/tensorrt_llm/_torch/auto_deploy/compile/backends/torch_cudagraph.py
+++ b/tensorrt_llm/_torch/auto_deploy/compile/backends/torch_cudagraph.py
@@ -175,7 +175,7 @@ class CapturedGraph(nn.Module):
 
         # retrieve output from buffer, cut to batch size, and unflatten
         bs = args_batched[0].shape[0]
-        out_flat = [o_b[:bs].detach().clone() for o_b in self._out_buffer_flat]
+        out_flat = [o_b[:bs] for o_b in self._out_buffer_flat]
         return self._out_spec.unflatten(out_flat)
 
 
diff --git a/tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py b/tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py
index 0add719af..b34e08178 100644
--- a/tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py
+++ b/tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py
@@ -116,6 +116,7 @@ class SequenceInfo:
         page_size: int = 0,
         max_num_tokens: Optional[int] = None,
         vocab_size_padded: Optional[int] = None,
+        chunk_size: Optional[int] = None,
     ):
         """Initialize the SequenceInfo object.
 
@@ -142,7 +143,10 @@ class SequenceInfo:
         self.max_batch_size = max_batch_size
         self.page_size = page_size if page_size > 0 else max_seq_len
         self.vocab_size_padded = vocab_size_padded
-
+        self.chunk_size = chunk_size
+        # Chunk size is an input to a custom op, so we need to set a default value if it is not provided.
+        if self.chunk_size is None:
+            self.chunk_size = 128
         # NOTE (lucaslie): WAR to address issue when using flashinfer attention with
         # (max_batch_size, max_seq_len) input in trtllm runtime.
         # see https://github.com/NVIDIA/TensorRT-LLM/issues/4504
@@ -193,7 +197,7 @@ class SequenceInfo:
             "input_pos": torch.empty(self.max_batch_size, dtype=torch.int),
             "cache_loc": torch.empty(max_num_cache_loc_assignments, dtype=torch.int),
             "pages_per_seq": torch.empty(s
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Primarily benefits the prefill (prompt processing) phase.

---

## d1ba3b8620 - [TRTLLM-11093][feat] add 5D A2A for fused ulysses (#11787)

- **Date**: 2026-03-06
- **Author**: NVShreyas
- **Categories**: Fusion

### Optimization Techniques

- Custom CUDA kernel
- Operator fusion
- KV cache optimization
- Parallelism optimization
- Batching optimization
- PyTorch built-in optimized ops
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/distributed/__init__.py        |   5 +-
 tensorrt_llm/_torch/distributed/ops.py             |  74 ++++++++
 .../visual_gen/attention_backend/parallel.py       | 102 ++++++-----
 .../_torch/visual_gen/attention_backend/trtllm.py  |   9 +-
 .../visual_gen/multi_gpu/test_flux_ulysses.py      |  81 +++++++-
 .../visual_gen/multi_gpu/test_ulysses_attention.py | 204 ++++++++++++++++++++-
 6 files changed, 426 insertions(+), 49 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/distributed/__init__.py b/tensorrt_llm/_torch/distributed/__init__.py
index 2dafa88bf..5e18d0d7b 100644
--- a/tensorrt_llm/_torch/distributed/__init__.py
+++ b/tensorrt_llm/_torch/distributed/__init__.py
@@ -4,11 +4,12 @@ from .communicator import Distributed, MPIDist, TorchDist
 from .moe_alltoall import MoeAlltoAll
 from .ops import (AllReduce, AllReduceParams, AllReduceStrategy,
                   HelixAllToAllNative, MoEAllReduce, MoEAllReduceParams,
-                  all_to_all_4d, allgather, alltoall_helix, cp_allgather,
-                  reducescatter, userbuffers_allreduce_finalize)
+                  all_to_all_4d, all_to_all_5d, allgather, alltoall_helix,
+                  cp_allgather, reducescatter, userbuffers_allreduce_finalize)
 
 __all__ = [
     "all_to_all_4d",
+    "all_to_all_5d",
     "allgather",
     "alltoall_helix",
     "cp_allgather",
diff --git a/tensorrt_llm/_torch/distributed/ops.py b/tensorrt_llm/_torch/distributed/ops.py
index 525a825a3..a26975eb1 100644
--- a/tensorrt_llm/_torch/distributed/ops.py
+++ b/tensorrt_llm/_torch/distributed/ops.py
@@ -1082,3 +1082,77 @@ def all_to_all_4d(
             output = output_reshaped.view(batch, seq, heads, head_dim)
 
     return output
+
+
+def all_to_all_5d(
+    input: torch.Tensor,
+    scatter_dim: int,
+    gather_dim: int,
+    process_group: Optional[torch.distributed.ProcessGroup] = None,
+) -> torch.Tensor:
+    """
+    All-to-all for 5D tensors with a fused QKV dimension.
+
+    Operates on [B, S, 3, H, D] tensors where dim 2 is the QKV count.
+    Used for Ulysses sequence parallelism with fused QKV to reduce the
+    number of all-to-all collectives from 3 (one per Q/K/V) to 1.
+
+    Supported scatter/gather combinations:
+    - scatter_dim=3 (heads), gather_dim=1 (seq): [B, S/P, 3, H, D] -> [B, S, 3, H/P, D]
+    - scatter_dim=1 (seq), gather_dim=3 (heads): [B, S, 3, H/P, D] -> [B, S/P, 3, H, D]
+    """
+    if not mpi_disabled():
+        raise NotImplementedError(
+            "all_to_all_5d currently only supports PyTorch distributed mode.")
+
+    world_size = torch.distributed.get_world_size(group=process_group)
+    if world_size == 1:
+        return input
+
+    assert input.dim() == 5, f"Expected 5D tensor, got {input.dim()}D"
+    assert scatter_dim in [1, 3] and gather_dim in [1, 3]
+    assert scatter_dim != gather_dim
+
+    batch, seq, qkv_count, heads, head_dim = input.shape
+    assert input.shape[scatter_dim] % world_size == 0, \
+        f"Dim {scatter_dim} size {input.shape[scatter_dim]} not divisible by world_size {world_size}"
+
+    if scatter_dim == 3 and gather_dim == 1:
+        # [B, S/P, 3, H, D] -> [B, S, 3, H/P, D]
+        sharded_heads = heads // world_size
+        inp = input.reshape(batch, seq, qkv_count, world_size, sharded_heads,
+                            head_dim)
+        inp = inp.permute(3, 0, 1, 2, 4,
+                          5).contiguous()  # [P, B, S/P, 3, H/P, D]
+
+        o
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## d252101a76 - [OMNIML-3036][doc] Re-branding TensorRT-Model-Optimizer as Nvidia Model-Optimizer (#9679)

- **Date**: 2025-12-07
- **Author**: Chenjie Luo
- **Categories**: General Performance

### Optimization Techniques

- Torch compilation/JIT optimization
- Operator fusion
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
- Decode/generation phase
- Disaggregated serving

### Changed Files

```
ATTRIBUTIONS-Python.md                                     |  4 ++--
 README.md                                                  |  4 ++--
 ...g14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.md |  2 +-
 ...timizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.md |  2 +-
 ...zing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md |  2 +-
 docs/source/developer-guide/perf-benchmarking.md           |  4 ++--
 docs/source/developer-guide/perf-overview.md               |  2 +-
 docs/source/features/auto_deploy/support_matrix.md         |  2 +-
 docs/source/features/quantization.md                       |  8 ++++----
 docs/source/legacy/performance/perf-benchmarking.md        |  2 +-
 docs/source/torch/auto_deploy/support_matrix.md            |  2 +-
 docs/source/torch/features/quantization.md                 |  6 +++---
 examples/auto_deploy/README.md                             |  8 ++++----
 examples/disaggregated/README.md                           |  2 +-
 examples/llm-api/_tensorrt_engine/llm_medusa_decoding.py   |  4 ++--
 examples/llm-api/_tensorrt_engine/quickstart_example.py    |  2 +-
 examples/llm-api/llm_inference.py                          |  2 +-
 examples/llm-api/quickstart_example.py                     |  2 +-
 examples/medusa/README.md                                  |  2 +-
 examples/models/core/deepseek_v3/README.md                 |  6 +++---
 examples/models/core/exaone/README.md                      | 10 +++++-----
 examples/models/core/llama/README.md                       |  2 +-
 examples/models/core/llama4/README.md                      |  6 +++---
 examples/models/core/qwen/README.md                        | 14 +++++++-------
 examples/quantization/README.md                            |  2 +-
 security_scanning/examples/models/core/mllama/poetry.lock  |  2 +-
 security_scanning/poetry.lock                              |  2 +-
 27 files changed, 53 insertions(+), 53 deletions(-)
```

### Diff Preview

```diff
diff --git a/ATTRIBUTIONS-Python.md b/ATTRIBUTIONS-Python.md
index f7360a7e9..4e350512a 100644
--- a/ATTRIBUTIONS-Python.md
+++ b/ATTRIBUTIONS-Python.md
@@ -25486,7 +25486,7 @@ limitations under the License.
 ```
 
 ### URLs
-  - `Homepage`: https://github.com/NVIDIA/TensorRT-Model-Optimizer
+  - `Homepage`: https://github.com/NVIDIA/Model-Optimizer
 
 
 ## nvidia-modelopt-core (0.33.1)
@@ -25513,7 +25513,7 @@ limitations under the License.
 ```
 
 ### URLs
-  - `Homepage`: https://github.com/NVIDIA/TensorRT-Model-Optimizer
+  - `Homepage`: https://github.com/NVIDIA/Model-Optimizer
 
 
 ## nvidia-nccl-cu12 (2.27.3)
diff --git a/README.md b/README.md
index f09c61783..208767b03 100644
--- a/README.md
+++ b/README.md
@@ -164,7 +164,7 @@ state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs.<
 [➡️ link](https://www.bentoml.com/blog/tuning-tensor-rt-llm-for-optimal-serving-with-bentoml)
 
 
-* [2024/08/20] 🏎️SDXL with #TensorRT Model Optimizer ⏱️⚡ 🏁 cache diffusion 🏁 quantization aware training 🏁 QLoRA 🏁 #Python 3.12
+* [2024/08/20] 🏎️SDXL with #Model Optimizer ⏱️⚡ 🏁 cache diffusion 🏁 quantization aware training 🏁 QLoRA 🏁 #Python 3.12
 [➡️ link](https://developer.nvidia.com/blog/nvidia-tensorrt-model-optimizer-v0-15-boosts-inference-performance-and-expands-model-support/)
 
 * [2024/08/13] 🐍 DIY Code Completion with #Mamba ⚡ #TensorRT #LLM for speed 🤖 NIM for ease ☁️ deploy anywhere
@@ -209,7 +209,7 @@ Technical Deep Dive for serious coders ✅+99% compression ✅1 set of weights 
 * [2024/05/21] ✨@modal_labs has the codes for serverless @AIatMeta Llama 3 on #TensorRT #LLM ✨👀 📚 Marvelous Modal Manual:
 Serverless TensorRT LLM (LLaMA 3 8B) | Modal Docs [➡️ link](https://modal.com/docs/examples/trtllm_llama)
 
-* [2024/05/08] NVIDIA TensorRT Model Optimizer -- the newest member of the #TensorRT ecosystem is a library of post-training and training-in-the-loop model optimization techniques ✅quantization ✅sparsity ✅QAT [➡️ blog](https://developer.nvidia.com/blog/accelerate-generative-ai-inference-performance-with-nvidia-tensorrt-model-optimizer-now-publicly-available/)
+* [2024/05/08] NVIDIA Model Optimizer -- the newest member of the #TensorRT ecosystem is a library of post-training and training-in-the-loop model optimization techniques ✅quantization ✅sparsity ✅QAT [➡️ blog](https://developer.nvidia.com/blog/accelerate-generative-ai-inference-performance-with-nvidia-tensorrt-model-optimizer-now-publicly-available/)
 
 * [2024/05/07] 🦙🦙🦙 24,000 tokens per second 🛫Meta Llama 3 takes off with #TensorRT #LLM 📚[➡️ link](https://blogs.nvidia.com/blog/meta-llama3-inference-acceleration/)
 
diff --git a/docs/source/blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.md b/docs/source/blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.md
index 4b80603e2..800c406bd 100644
--- a/docs/source/blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.md
+++ b/docs/source/blogs/tech_blog/blog14_S
```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## d2a04abb95 - [fix] Fixes to parameter usage and low latency configuration. (#6343)

- **Date**: 2025-07-28
- **Author**: Frank
- **Categories**: Throughput/Latency

### Optimization Techniques

- Async/stream-based execution
- KV cache optimization
- Parallelism optimization
- Batching optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/bench/benchmark/low_latency.py | 14 +++++++++++
 tensorrt_llm/bench/benchmark/throughput.py  | 38 ++++++++++++++---------------
 2 files changed, 33 insertions(+), 19 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/bench/benchmark/low_latency.py b/tensorrt_llm/bench/benchmark/low_latency.py
index fd701700a..af86fb2b1 100644
--- a/tensorrt_llm/bench/benchmark/low_latency.py
+++ b/tensorrt_llm/bench/benchmark/low_latency.py
@@ -13,6 +13,7 @@ from huggingface_hub import snapshot_download
 
 from tensorrt_llm import LLM as PyTorchLLM
 from tensorrt_llm._tensorrt_engine import LLM
+from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM
 from tensorrt_llm.bench.benchmark.utils.asynchronous import async_benchmark
 from tensorrt_llm.bench.benchmark.utils.general import generate_warmup_dataset
 from tensorrt_llm.bench.benchmark.utils.processes import IterationWriter
@@ -298,7 +299,20 @@ def latency_command(
             kwargs["pytorch_backend_config"].enable_iter_perf_stats = True
 
         if runtime_config.backend == 'pytorch':
+            if kwargs.pop("extended_runtime_perf_knob_config", None):
+                logger.warning(
+                    "Ignore extended_runtime_perf_knob_config for pytorch backend."
+                )
             llm = PyTorchLLM(**kwargs)
+        elif runtime_config.backend == "_autodeploy":
+            if kwargs.pop("extended_runtime_perf_knob_config", None):
+                logger.warning(
+                    "Ignore extended_runtime_perf_knob_config for _autodeploy backend."
+                )
+            kwargs["world_size"] = kwargs.pop("tensor_parallel_size", None)
+            kwargs.pop("pipeline_parallel_size", None)
+
+            llm = AutoDeployLLM(**kwargs)
         else:
             llm = LLM(**kwargs)
 
diff --git a/tensorrt_llm/bench/benchmark/throughput.py b/tensorrt_llm/bench/benchmark/throughput.py
index 27f845ee5..b1b30125d 100755
--- a/tensorrt_llm/bench/benchmark/throughput.py
+++ b/tensorrt_llm/bench/benchmark/throughput.py
@@ -255,25 +255,25 @@ def throughput_command(
     logger.info("Preparing to run throughput benchmark...")
     # Parameters from CLI
     # Model, experiment, and engine params
-    dataset_path: Path = params.pop("dataset")
-    eos_id: int = params.pop("eos_id")
+    dataset_path: Path = params.get("dataset")
+    eos_id: int = params.get("eos_id")
     warmup: int = params.get("warmup")
-    num_requests: int = params.pop("num_requests")
-    max_seq_len: int = params.pop("max_seq_len")
+    num_requests: int = params.get("num_requests")
+    max_seq_len: int = params.get("max_seq_len")
     model: str = bench_env.model
     checkpoint_path: Path = bench_env.checkpoint_path or bench_env.model
-    engine_dir: Path = params.pop("engine_dir")
-    concurrency: int = params.pop("concurrency")
+    engine_dir: Path = params.get("engine_dir")
+    concurrency: int = params.get("concurrency")
     backend: str = params.get("backend")
-    modality: str = params.pop("modality")
-    max_input_len: int = params.pop("max_input_len")
+    modality: str = params.get("modality")
+    max_input_len: int = params.get("max_input_len")
     model_type = get_
```

### Analysis Summary

Async execution overlaps computation with data transfer, reducing idle time on the GPU. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## d3f4fbb742 - [None][fix] Avoid write-write race for async pp send. (#10488)

- **Date**: 2026-01-14
- **Author**: Yuxian Qiu
- **Categories**: Parallelism/Async

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution

### Applicable Conditions

- General LLM inference

### Changed Files

```
tensorrt_llm/_torch/distributed/communicator.py | 8 ++++++--
 1 file changed, 6 insertions(+), 2 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/distributed/communicator.py b/tensorrt_llm/_torch/distributed/communicator.py
index 20401b5a2..d6e54a641 100644
--- a/tensorrt_llm/_torch/distributed/communicator.py
+++ b/tensorrt_llm/_torch/distributed/communicator.py
@@ -839,9 +839,13 @@ class PPCommNCCL:
             self.nccl_comm.send(tensor, dest)
             return
 
-        self.tensor_ready_event.record()
+        # If the tensor is allocated from non-default memory pool
+        # like userbuffers, its underlying memory may be reused
+        # before the send operation is completed.
+        # We clone the tensor to avoid write-write conflicts.
+        tensor = tensor.clone()
+        self.send_stream.wait_stream(torch.cuda.current_stream())
         with torch.cuda.stream(self.send_stream):
-            self.tensor_ready_event.wait()
             self.nccl_comm.send(tensor, dest)
 
     def recv(self, tensor: torch.Tensor, src: Optional[int] = None):

```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## d548b29a41 - [None][fix] Bugfix/mtp with async scheduler (#10941)

- **Date**: 2026-01-24
- **Author**: Patrice Castonguay
- **Categories**: Parallelism/Async

### Optimization Techniques

- KV cache optimization
- Batching optimization
- Speculative decoding

### Applicable Conditions

- General LLM inference

### Changed Files

```
cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp | 12 ++++++++++++
 1 file changed, 12 insertions(+)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp b/cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp
index 4138e4c60..f20019a0d 100644
--- a/cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp
+++ b/cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp
@@ -2915,6 +2915,18 @@ void KVCacheManager::removeToken(RequestIdType requestId)
 
 void KVCacheManager::rewindKVCache(RequestIdType requestId, SizeType32 rewindLengths)
 {
+    // Check if the sequence still exists before rewinding
+    // In overlap mode with MTP, the request may have been terminated and removed
+    // from mSequences before rewindKVCache is called
+    {
+        std::scoped_lock lck(mSequencesMtx);
+        if (mSequences.find(requestId) == mSequences.end())
+        {
+            TLLM_LOG_DEBUG("Request %lu has already been removed from KV cache manager, skipping rewind", requestId);
+            return;
+        }
+    }
+
     for (SizeType32 si = 0; si < rewindLengths; ++si)
     {
         removeToken(requestId);

```

### Analysis Summary

Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## d643aef73c - [Perf] Improve Llama4 performance for small max_seqlen cases (#6306)

- **Date**: 2025-08-08
- **Author**: Yilin Fan
- **Categories**: General Performance

### Optimization Techniques

- Attention mechanism optimization

### Applicable Conditions

- NVIDIA Blackwell (SM100) GPU

### Changed Files

```
tensorrt_llm/_torch/models/modeling_llama.py | 5 +++++
 1 file changed, 5 insertions(+)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/models/modeling_llama.py b/tensorrt_llm/_torch/models/modeling_llama.py
index 6ec655796..380f82a50 100644
--- a/tensorrt_llm/_torch/models/modeling_llama.py
+++ b/tensorrt_llm/_torch/models/modeling_llama.py
@@ -74,6 +74,11 @@ class Llama4Attention(Attention):
         elif get_sm_version() <= 90 and model_config.spec_config is not None:
             # pre-Blackwell spec-dec kernel does not support
             attention_chunk_size = None
+        else:
+            # Disable chunked attention when max_seq_len is smaller than attention_chunk_size
+            # TODO: Remove this after all attention kernels in TRTLLM backend support chunked attention
+            if attention_chunk_size and model_config.max_seq_len and model_config.max_seq_len < attention_chunk_size:
+                attention_chunk_size = None
 
         super().__init__(
             hidden_size=config.hidden_size,

```

### Analysis Summary

Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## d6e49542bd - [https://nvbugs/5848377][fix] fix deepeplowlatency with trtllm moe backend running fp8 DS_R1 (#11266)

- **Date**: 2026-02-10
- **Author**: Leslie Fang
- **Categories**: Throughput/Latency, Quantization Optimization

### Optimization Techniques

- Operator fusion
- Parallelism optimization
- MoE optimization
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../_torch/modules/fused_moe/communication/deep_ep_low_latency.py    | 4 ++++
 tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py            | 5 ++++-
 tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py        | 4 ++++
 3 files changed, 12 insertions(+), 1 deletion(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/modules/fused_moe/communication/deep_ep_low_latency.py b/tensorrt_llm/_torch/modules/fused_moe/communication/deep_ep_low_latency.py
index d7e96a656..656f4957f 100644
--- a/tensorrt_llm/_torch/modules/fused_moe/communication/deep_ep_low_latency.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/communication/deep_ep_low_latency.py
@@ -283,6 +283,10 @@ class DeepEPLowLatency(Communication):
             self.expert_size_per_partition, num_tokens_per_expert, self.hidden_size
         )
 
+        if deep_ep_topk_weights.dtype != torch.float32:
+            # Deep ep low latency combine requires for fp32 weights
+            deep_ep_topk_weights = deep_ep_topk_weights.to(torch.float32)
+
         if self.use_low_precision_combine:
             if self._has_nvfp4():
                 precision = "nvfp4"
diff --git a/tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py b/tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py
index 1c47aa5af..b99de2086 100644
--- a/tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py
@@ -756,7 +756,10 @@ class ConfigurableMoE(MoE):
             if self.enable_dummy_allreduce:
                 self.dummy_allreduce()
             # Use unified combine interface (reads dispatch state from strategy)
-            final_hidden_states = self.comm.combine(final_hidden_states)
+            all_rank_max_num_tokens = max(all_rank_num_tokens)
+            final_hidden_states = self.comm.combine(
+                final_hidden_states, all_rank_max_num_tokens=all_rank_max_num_tokens
+            )
         else:
             # For non-comm case, It should be attention TP or single rank.
             # only check if allreduce is needed
diff --git a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py
index 637c402f4..ea259cc16 100644
--- a/tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py
+++ b/tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py
@@ -531,6 +531,10 @@ class TRTLLMGenFusedMoE(MoE):
 
         routing_bias = routing_bias if router_logits is not None else None
 
+        if token_selected_experts is not None:
+            # for cases like deepep low latency where fake top_k=1 might be used
+            top_k = token_selected_experts.shape[-1]
+
         # Ensure x_sf is 2D before flattening
         if x_sf is not None:
             assert len(

```

### Analysis Summary

Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## d6f95a4363 - [None][feat] AutoDeploy: Perf optimization for Attention and rmsnorm (#9719)

- **Date**: 2025-12-05
- **Author**: Chenghao Zhang
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- FP8 quantization
- KV cache optimization
- Triton kernel
- Attention mechanism optimization

### Applicable Conditions

- General LLM inference

### Changed Files

```
.../auto_deploy/custom_ops/flashinfer_attention.py    | 19 ++++++++++++-------
 .../_torch/auto_deploy/custom_ops/rms_norm.py         |  3 ++-
 2 files changed, 14 insertions(+), 8 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/auto_deploy/custom_ops/flashinfer_attention.py b/tensorrt_llm/_torch/auto_deploy/custom_ops/flashinfer_attention.py
index f621539c0..4a806dc1c 100644
--- a/tensorrt_llm/_torch/auto_deploy/custom_ops/flashinfer_attention.py
+++ b/tensorrt_llm/_torch/auto_deploy/custom_ops/flashinfer_attention.py
@@ -7,6 +7,7 @@ from torch._ops import OpOverloadPacket
 from torch._subclasses import FakeTensor
 from torch.fx import Node
 
+from ...flashinfer_utils import get_env_enable_pdl
 from ..utils.cuda_graph import cuda_graph_state
 from ..utils.logger import ad_logger
 from ..utils.node_utils import extract_op_args
@@ -256,9 +257,9 @@ def flashinfer_mha_with_cache(
     q_shape_og = q.shape
     b, s = q_shape_og[:2]
 
-    q = q.contiguous().view(b * s, -1, head_dim)
-    k = k.contiguous().view(b * s, -1, head_dim)
-    v = v.contiguous().view(b * s, -1, head_dim)
+    q = q.reshape(b * s, -1, head_dim)
+    k = k.reshape(b * s, -1, head_dim)
+    v = v.reshape(b * s, -1, head_dim)
 
     n_heads = q.shape[1]
     n_kv_heads = k.shape[1]
@@ -275,11 +276,12 @@ def flashinfer_mha_with_cache(
         sm_scale=scale,
     )
 
-    # Assuming k_scale = v_scale = 1.0, we just have to cast k and v to fp8 before appending to kv cache
+    # Assuming k_scale = v_scale = 1.0
     k_scale, v_scale = 1.0, 1.0
+    # k = (k / k_scale).to(torch.float8_e4m3fn) if k_scale != 1.0, same for v
     if k_cache.dtype == torch.float8_e4m3fn:
-        k = (k / k_scale).to(torch.float8_e4m3fn)
-        v = (v / v_scale).to(torch.float8_e4m3fn)
+        k = k.to(torch.float8_e4m3fn)
+        v = v.to(torch.float8_e4m3fn)
 
     flashinfer.page.append_paged_kv_cache(
         k,
@@ -300,7 +302,10 @@ def flashinfer_mha_with_cache(
         paged_kv_last_page_len,
         pp,
     )
-    y = wrapper.run(q, (k_cache, v_cache), k_scale=k_scale, v_scale=v_scale)
+
+    y = wrapper.run(
+        q, (k_cache, v_cache), k_scale=k_scale, v_scale=v_scale, enable_pdl=get_env_enable_pdl()
+    )
 
     return y.view(q_shape_og)  # [b,s,n*h_d] or [b,s, n, h_d]
 
diff --git a/tensorrt_llm/_torch/auto_deploy/custom_ops/rms_norm.py b/tensorrt_llm/_torch/auto_deploy/custom_ops/rms_norm.py
index bc6819f24..426521745 100644
--- a/tensorrt_llm/_torch/auto_deploy/custom_ops/rms_norm.py
+++ b/tensorrt_llm/_torch/auto_deploy/custom_ops/rms_norm.py
@@ -3,6 +3,7 @@
 import flashinfer
 import torch
 
+from ...flashinfer_utils import get_env_enable_pdl
 from ...modules.mamba.layernorm_gated import _layer_norm_fwd
 from .triton_kernels.rms_norm import rms_norm
 
@@ -21,7 +22,7 @@ def flashinfer_rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float) ->
     """
     # Flashinfer rmsnorm expects a 2D input
     input_flat = input.reshape(-1, input.shape[-1])
-    rmsnorm_flat = flashinfer.norm.rmsnorm(input_flat, weight, eps)
+    rmsnorm_flat = flashinfer.norm.rmsnorm(input_flat, weight, eps, enable_pdl=get_env_enable_pdl())
     return rmsnorm_flat.reshape(input.shape
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## d7087015f1 - [TRTLLM-8271][fix] Fix CDL overlap scheduling performance (#7971)

- **Date**: 2025-09-26
- **Author**: Mike Iovine
- **Categories**: General Performance

### Optimization Techniques

- Custom CUDA kernel
- Async/stream-based execution
- Batching optimization
- Reduce synchronization overhead
- Speculative decoding

### Applicable Conditions

- Decode/generation phase

### Changed Files

```
tensorrt_llm/_torch/speculative/model_drafter.py | 22 ++++++++++++++++++++--
 1 file changed, 20 insertions(+), 2 deletions(-)
```

### Diff Preview

```diff
diff --git a/tensorrt_llm/_torch/speculative/model_drafter.py b/tensorrt_llm/_torch/speculative/model_drafter.py
index f6082ac32..0a1a58d85 100644
--- a/tensorrt_llm/_torch/speculative/model_drafter.py
+++ b/tensorrt_llm/_torch/speculative/model_drafter.py
@@ -525,7 +525,8 @@ class ModelDrafter(Drafter):
         return draft_batch, req_id_to_old_request
 
     def process_static_draft_outputs(
-            self, outputs: Any, draft_batch: ScheduledRequests,
+            self, outputs: torch.Tensor | SampleState,
+            draft_batch: ScheduledRequests,
             req_id_to_old_request: Dict[int, LlmRequest]) -> None:
         """
         Process outputs from static draft loop, update target requests, and clean up resources.
@@ -535,7 +536,13 @@ class ModelDrafter(Drafter):
             draft_batch: The draft batch that was processed
             req_id_to_old_request: Mapping from draft request ID to original request
         """
-        outputs_host = outputs.cpu()
+        if isinstance(outputs, torch.Tensor):
+            # For non-overlap scheduler path.
+            outputs_host = outputs.cpu()
+        else:
+            outputs_host = outputs.host.new_tokens
+            outputs.sampler_event.synchronize()
+
         for token_idx in range(self.max_draft_tokens):
             for req_idx, req in enumerate(draft_batch.all_requests()):
                 target_model_req = req_id_to_old_request[req.py_request_id]
@@ -703,6 +710,17 @@ class ModelDrafter(Drafter):
                 draft_length=self.max_draft_tokens,
                 draft_batch=draft_batch,
                 req_id_to_old_request=req_id_to_old_request)
+
+            new_tokens_host = outputs.to(device='cpu', non_blocking=True)
+            sampler_event = torch.cuda.Event()
+            sampler_event.record()
+
+            outputs = SampleState(
+                scheduled_requests=draft_batch,
+                device=SampleStateTensors(new_tokens=outputs),
+                host=SampleStateTensors(new_tokens=new_tokens_host),
+                sampler_event=sampler_event)
+
             return target_inputs, outputs, draft_batch
 
         # Handle guided decoder and sampling for non-static loop

```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Reducing synchronization points improves pipeline throughput by allowing more concurrent execution. Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Primarily benefits the decode (token generation) phase.

---

## d7c51c953b - test: add INTEGRATION_TEST env var to speed up integration test (#3618)

- **Date**: 2025-05-08
- **Author**: Ivy Zhang
- **Categories**: Throughput/Latency

### Optimization Techniques

- KV cache optimization
- Batching optimization
- Speculative decoding

### Applicable Conditions

- General LLM inference

### Changed Files

```
examples/summarize.py                            | 10 ++++++++++
 tests/integration/defs/accuracy/accuracy_core.py | 20 ++++++++++++++------
 2 files changed, 24 insertions(+), 6 deletions(-)
```

### Diff Preview

```diff
diff --git a/examples/summarize.py b/examples/summarize.py
index b9e4292e9..474dac2d8 100644
--- a/examples/summarize.py
+++ b/examples/summarize.py
@@ -42,6 +42,16 @@ from prompt_lookup.run_dtm_pld import run_dtm_pld
 
 
 def main(args):
+    is_integration_test = os.getenv('INTEGRATION_TEST', '0') == '1'
+    if is_integration_test:
+        logger.info(
+            "Running in integration test mode - will only run one batch and skip accuracy checks"
+        )
+        logger.info(
+            "Setting max_ite=1 and check_accuracy=False for integration test")
+        args.max_ite = 1
+        args.check_accuracy = False
+
     runtime_rank = tensorrt_llm.mpi_rank()
     logger.set_level(args.log_level)
 
diff --git a/tests/integration/defs/accuracy/accuracy_core.py b/tests/integration/defs/accuracy/accuracy_core.py
index 0c7e26394..a7415dc9f 100644
--- a/tests/integration/defs/accuracy/accuracy_core.py
+++ b/tests/integration/defs/accuracy/accuracy_core.py
@@ -151,13 +151,21 @@ class AccuracyTask:
             raise ValueError(
                 f"Not recognized speculative_config: {llm.args.speculative_config}."
             )
+        is_integration_test = os.getenv('INTEGRATION_TEST', '0') == '1'
 
-        num_samples, threshold = self.get_num_samples_and_threshold(
-            dtype=llm.args.dtype,
-            quant_algo=llm.args.quant_config.quant_algo,
-            kv_cache_quant_algo=llm.args.quant_config.kv_cache_quant_algo,
-            spec_dec_algo=spec_dec_algo,
-            extra_acc_spec=extra_acc_spec)
+        if is_integration_test:
+            num_samples = 1
+            logger.info(
+                "Running in INTEGRATION_TEST mode: using only 1 sample and skipping accuracy verification"
+            )
+            threshold = 0
+        else:
+            num_samples, threshold = self.get_num_samples_and_threshold(
+                dtype=llm.args.dtype,
+                quant_algo=llm.args.quant_config.quant_algo,
+                kv_cache_quant_algo=llm.args.quant_config.kv_cache_quant_algo,
+                spec_dec_algo=spec_dec_algo,
+                extra_acc_spec=extra_acc_spec)
 
         sampling_params = SamplingParams(
             max_tokens=self.MAX_OUTPUT_LEN,

```

### Analysis Summary

Speculative decoding reduces generation latency by predicting multiple tokens in parallel. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

## d8b05894ee - [None][perf] Adjust select_alltoall_method_type. (#8950)

- **Date**: 2025-11-19
- **Author**: Bo Li
- **Categories**: General Performance

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
.../communicationKernels/moeAlltoAllKernels.cu     |  6 +++
 cpp/tensorrt_llm/thop/moeAlltoAllOp.cpp            |  2 +-
 cpp/tensorrt_llm/thop/mxFp4BlockScaleMoe.cpp       |  2 +-
 .../_torch/modules/fused_moe/fused_moe_cutlass.py  | 58 ++++++++++------------
 .../_torch/modules/fused_moe/fused_moe_deepgemm.py |  5 ++
 .../modules/fused_moe/fused_moe_trtllm_gen.py      | 33 +++++++-----
 .../unit/singlegpu/test_ad_trtllm_serve.py         |  4 +-
 7 files changed, 62 insertions(+), 48 deletions(-)
```

### Diff Preview

```diff
diff --git a/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu b/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu
index 144aadbc7..9bb85a878 100644
--- a/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu
+++ b/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu
@@ -51,6 +51,12 @@ namespace tensorrt_llm::kernels::mnnvl_throughput
         __VA_ARGS__;                                                                                                   \
         break;                                                                                                         \
     }                                                                                                                  \
+    case 6:                                                                                                            \
+    {                                                                                                                  \
+        constexpr int TOP_K = 6;                                                                                       \
+        __VA_ARGS__;                                                                                                   \
+        break;                                                                                                         \
+    }                                                                                                                  \
     case 4:                                                                                                            \
     {                                                                                                                  \
         constexpr int TOP_K = 4;                                                                                       \
diff --git a/cpp/tensorrt_llm/thop/moeAlltoAllOp.cpp b/cpp/tensorrt_llm/thop/moeAlltoAllOp.cpp
index d6e5b7465..8095d0ebd 100644
--- a/cpp/tensorrt_llm/thop/moeAlltoAllOp.cpp
+++ b/cpp/tensorrt_llm/thop/moeAlltoAllOp.cpp
@@ -31,7 +31,7 @@ namespace torch_ext
 namespace mnnvl_throughput
 {
 
-// TODO: Is Alignment necessary?obu guo
+// TODO: Is Alignment necessary?
 // Helper function to align offset to specified byte boundary
 inline size_t alignOffset(size_t offset, size_t alignment)
 {
diff --git a/cpp/tensorrt_llm/thop/mxFp4BlockScaleMoe.cpp b/cpp/tensorrt_llm/thop/mxFp4BlockScaleMoe.cpp
index ea7131c3f..2fdc8573c 100644
--- a/cpp/tensorrt_llm/thop/mxFp4BlockScaleMoe.cpp
+++ b/cpp/tensorrt_llm/thop/mxFp4BlockScaleMoe.cpp
@@ -554,7 +554,7 @@ public:
             topk_group, intermediate_size, valid_hidden_size, valid_intermediate_size, local_expert_offset,
             local_num_experts, routed_scaling_factor, tileN, routing_method_type, mDtypeAct, *mRunners[tileN], config,
             topk_weights, topk_ids,
-            /*output=*/torch::nullopt); // TODO: Support user-provided output
+            /*out_tensor=*
```

### Analysis Summary

This commit introduces or optimizes a GPU kernel implementation, which can significantly improve compute-bound operations. Operator fusion reduces kernel launch overhead and intermediate memory allocations by combining multiple operations into a single kernel. Async execution overlaps computation with data transfer, reducing idle time on the GPU. Quantization reduces memory footprint and can leverage specialized hardware (e.g., Tensor Cores) for faster computation. Attention optimization is critical for LLM inference as attention computation is often the bottleneck, especially for long sequences. GEMM optimization improves the performance of matrix multiplications which dominate LLM computation. MoE optimization improves routing and expert execution efficiency for Mixture-of-Experts models. 
**Application Scenarios**:  Applicable to general LLM inference workloads.

---

