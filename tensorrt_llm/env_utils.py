# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Iterator, Optional


@dataclass
class _EnvEntry:
    """Metadata record for one environment variable.

    Attributes:
        default: The default value supplied the first time ``get()`` was
            called for this key, or the canonical default pre-registered in
            ``__init__``.  ``None`` if no default is applicable.
        content: Cached value from ``os.environ``.  ``None`` means the value
            has not yet been successfully fetched (key absent or not yet
            accessed).
    """

    default: Any = None
    content: Optional[str] = None


class _TRTLLMENVClass(MutableMapping):
    """Centralized environment variable manager for TensorRT-LLM.

    An internal registry (``_registry``) maps every known environment-variable
    name to an ``_EnvEntry(default, content)`` pair:

    * ``default`` – canonical default registered at startup (or the value
      supplied on the first ``get()`` call).
    * ``content`` – actual value fetched from ``os.environ``; ``None`` until
      the key is first successfully read.

    **Get-type interfaces** (``__getitem__`` and ``get``) return
    ``entry.content`` directly when it is not ``None``, bypassing
    ``os.environ``.  On a cache miss ``os.environ`` is consulted and the
    result is stored in ``content``.

    ``get()`` raises ``NameError`` if the requested key has not been
    pre-registered and has never been written through this object.  This
    provides a compile-time-like safety net against typos in env-var names.

    **Write interfaces** (``__setitem__``, ``update``) propagate to both the
    registry and ``os.environ``.  A write also adds a new registry entry for
    previously unknown keys, so ``env_overrides`` applied at runtime are
    handled transparently.

    **Delete** (``__delitem__``) removes the key from ``os.environ`` and
    resets ``content`` to ``None``, invalidating the cached value while
    preserving the registered default.

    The object satisfies the full ``MutableMapping`` protocol and can be
    passed directly as the ``env=`` argument to ``subprocess.Popen``.
    """

    # ------------------------------------------------------------------
    # Pre-registered environment variables with canonical defaults
    # ------------------------------------------------------------------
    #
    # Format: "ENV_VAR_NAME": _EnvEntry(default=<value>)
    #   - Use _EnvEntry(default=None) when no meaningful default exists
    #     (write-only vars, or vars only checked with "is None" / truthiness).
    #   - Use the string/int/bool that appears at the call-site otherwise.
    #
    _KNOWN_VARS: dict[str, "_EnvEntry"] = {
        # ---- Logging / severity ----------------------------------------
        "TLLM_LOG_LEVEL": _EnvEntry(default=None),

        # ---- MPI / process model ----------------------------------------
        "TLLM_DISABLE_MPI": _EnvEntry(default=None),
        "OMPI_MCA_coll_ucc_enable": _EnvEntry(default=None),
        "OMPI_COMM_WORLD_RANK": _EnvEntry(default=None),
        "OMPI_COMM_WORLD_LOCAL_RANK": _EnvEntry(default=None),
        "OMPI_COMM_WORLD_SIZE": _EnvEntry(default=None),

        # ---- Library initialisation ------------------------------------
        "TRT_LLM_NO_LIB_INIT": _EnvEntry(default="0"),
        "TRTLLM_PRINT_STACKS_PERIOD": _EnvEntry(default="-1"),
        "IS_BUILDING": _EnvEntry(default=None),

        # ---- NVTX / profiling ------------------------------------------
        "TLLM_LLMAPI_ENABLE_NVTX": _EnvEntry(default="0"),
        "TLLM_NVTX_DEBUG": _EnvEntry(default="0"),
        "TLLM_PROFILE_RECORD_GC": _EnvEntry(default=None),
        "TLLM_TORCH_PROFILE_TRACE": _EnvEntry(default=None),
        "TLLM_PROFILE_START_STOP": _EnvEntry(default=None),
        "TLLM_TRACE_EXECUTOR_LOOP": _EnvEntry(default=None),
        "TLLM_TRACE_MODEL_FORWARD": _EnvEntry(default=None),
        "NSYS_PROFILING_SESSION_ID": _EnvEntry(default=None),

        # ---- Prometheus ------------------------------------------------
        "PROMETHEUS_MULTIPROC_DIR": _EnvEntry(default=None),

        # ---- Tokenizer / detokenization --------------------------------
        "TLLM_INCREMENTAL_DETOKENIZATION_BACKEND": _EnvEntry(default="HF"),
        "TLLM_STREAM_INTERVAL_THRESHOLD": _EnvEntry(default="24"),
        "TOKENIZERS_PARALLELISM": _EnvEntry(default="false"),

        # ---- Sampling ---------------------------------------------------
        "TLLM_ALLOW_N_GREEDY_DECODING": _EnvEntry(default=None),
        "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS": _EnvEntry(default="0"),

        # ---- Build / compilation ---------------------------------------
        "BUILDER_FORCE_NUM_PROFILES": _EnvEntry(default=None),
        "DISABLE_TORCH_DEVICE_SET": _EnvEntry(default=False),
        "XLA_PYTHON_CLIENT_PREALLOCATE": _EnvEntry(default=None),

        # ---- LLM API ---------------------------------------------------
        "TLLM_LLM_ENABLE_DEBUG": _EnvEntry(default="0"),
        "TLLM_LLMAPI_ENABLE_DEBUG": _EnvEntry(default="0"),
        "TLLM_LLM_ENABLE_TRACER": _EnvEntry(default="0"),
        "TLLM_WORKER_USE_SINGLE_PROCESS": _EnvEntry(default="0"),
        "TLLM_LLMAPI_BUILD_CACHE": _EnvEntry(default=None),
        "TLLM_LLMAPI_BUILD_CACHE_ROOT": _EnvEntry(
            default="/tmp/.cache/tensorrt_llm/llmapi/"),

        # ---- IPC / ZMQ -------------------------------------------------
        "TLLM_LLMAPI_ZMQ_PAIR": _EnvEntry(default="0"),
        "TLLM_LLMAPI_ZMQ_DEBUG": _EnvEntry(default="0"),
        "TLLM_SPAWN_PROXY_PROCESS": _EnvEntry(default=None),
        "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR": _EnvEntry(default=None),
        "TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY": _EnvEntry(default=None),
        "TLLM_NUMA_AWARE_WORKER_AFFINITY": _EnvEntry(default=None),

        # ---- Disaggregated serving -------------------------------------
        "TLLM_DISAGG_INSTANCE_IDX": _EnvEntry(default=None),
        "TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT": _EnvEntry(default=None),
        "TRTLLM_DISAGG_BENCHMARK_GEN_ONLY": _EnvEntry(default=None),
        "TRTLLM_SERVER_DISABLE_GC": _EnvEntry(default="0"),
        "TRTLLM_DISAGG_SERVER_DISABLE_GC": _EnvEntry(default="1"),
        "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP": _EnvEntry(default=None),
        "TRTLLM_KVCACHE_TIME_OUTPUT_PATH": _EnvEntry(default=""),

        # ---- Ray executor -----------------------------------------------
        "RAY_LOCAL_WORLD_SIZE": _EnvEntry(default=None),
        "TLLM_RAY_FORCE_LOCAL_CLUSTER": _EnvEntry(default="0"),
        "TRTLLM_RAY_PER_WORKER_GPUS": _EnvEntry(default="1.0"),
        "TRTLLM_RAY_BUNDLE_INDICES": _EnvEntry(default=None),
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": _EnvEntry(default=None),
        "RAY_DEDUP_LOGS": _EnvEntry(default=None),

        # ---- Distributed / MPI world -----------------------------------
        "MASTER_ADDR": _EnvEntry(default=None),
        "MASTER_PORT": _EnvEntry(default=None),
        "RANK": _EnvEntry(default=None),
        "WORLD_SIZE": _EnvEntry(default=None),
        "LOCAL_RANK": _EnvEntry(default=None),
        "SLURM_PROCID": _EnvEntry(default=None),
        "tllm_mpi_size": _EnvEntry(default=None),

        # ---- CUDA visibility -------------------------------------------
        "CUDA_VISIBLE_DEVICES": _EnvEntry(default=None),
        "TORCH_CUDA_ARCH_LIST": _EnvEntry(default=None),

        # ---- Worker -------------------------------------------------------
        "TRTLLM_WORKER_PRINT_STACKS_PERIOD": _EnvEntry(default="-1"),
        "TRTLLM_WORKER_DISABLE_GC": _EnvEntry(default="0"),

        # ---- Executor loop / pipeline ----------------------------------
        "TLLM_BENCHMARK_REQ_QUEUES_SIZE": _EnvEntry(default=0),
        "TLLM_PP_SCHEDULER_MAX_RETRY_COUNT": _EnvEntry(default=10),
        "TRTLLM_PP_MULTI_STREAM_SAMPLE": _EnvEntry(default="1"),
        "TRTLLM_PP_REQ_SEND_ASYNC": _EnvEntry(default="0"),

        # ---- Model loading / weight conversion -------------------------
        "TLLM_OVERRIDE_LAYER_NUM": _EnvEntry(default="0"),
        "TRTLLM_DISABLE_UNIFIED_CONVERTER": _EnvEntry(default=None),
        "TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL": _EnvEntry(default="False"),
        "OVERRIDE_QUANT_ALGO": _EnvEntry(default=None),

        # ---- Determinism / workspace -----------------------------------
        "FORCE_DETERMINISTIC": _EnvEntry(default="0"),
        "FORCE_ALL_REDUCE_DETERMINISTIC": _EnvEntry(default="0"),
        "FORCE_ALLREDUCE_KERNEL_WORKSPACE_SIZE": _EnvEntry(default="1000000000"),
        "TRTLLM_ALLREDUCE_FUSION_WORKSPACE_SIZE": _EnvEntry(default=None),
        "TLLM_DISABLE_ALLREDUCE_AUTOTUNE": _EnvEntry(default="0"),
        "DISABLE_LAMPORT_REDUCE_NORM_FUSION": _EnvEntry(default=None),

        # ---- AllReduce / AllToAll / MoE comms -------------------------
        "TRTLLM_FORCE_COMM_METHOD": _EnvEntry(default=None),
        "TRTLLM_FORCE_ALLTOALL_METHOD": _EnvEntry(default=None),
        "TRTLLM_CAN_USE_DEEP_EP": _EnvEntry(default="0"),
        "TRTLLM_MOE_POST_QUANT_ALLTOALLV": _EnvEntry(default="1"),
        "TRTLLM_DEEP_EP_TOKEN_LIMIT": _EnvEntry(default=None),
        "TRTLLM_MOE_A2A_WORKSPACE_MB": _EnvEntry(default=None),
        "TRTLLM_DEEP_EP_DISABLE_P2P_FOR_LOW_LATENCY_MODE": _EnvEntry(default="0"),
        "NVSHMEM_QP_DEPTH": _EnvEntry(default=None),
        "TRTLLM_ENABLE_DUMMY_ALLREDUCE": _EnvEntry(default="0"),

        # ---- MNNVL -------------------------------------------------
        "TRTLLM_FORCE_MNNVL_AR": _EnvEntry(default="0"),
        "TLLM_TEST_MNNVL": _EnvEntry(default="0"),

        # ---- Autotuner -------------------------------------------------
        "TLLM_AUTOTUNER_CACHE_PATH": _EnvEntry(default=None),
        "TLLM_AUTOTUNER_LOG_LEVEL_DEBUG_TO_INFO": _EnvEntry(default="0"),
        "TLLM_AUTOTUNER_DISABLE_SHORT_PROFILE": _EnvEntry(default="0"),

        # ---- FlashInfer / PDL ------------------------------------------
        "TRTLLM_ENABLE_PDL": _EnvEntry(default="1"),

        # ---- Attention backend -----------------------------------------
        "TRTLLM_PRINT_SKIP_SOFTMAX_STAT": _EnvEntry(default="0"),
        "TRTLLM_ENABLE_ATTENTION_NVFP4_OUTPUT": _EnvEntry(default="1"),

        # ---- Model fusion flags ----------------------------------------
        "TRTLLM_LLAMA_EAGER_FUSION_DISABLED": _EnvEntry(default="0"),
        "TRTLLM_DEEPSEEK_EAGER_FUSION_DISABLED": _EnvEntry(default="0"),
        "TRTLLM_QWEN3_EAGER_FUSION_DISABLED": _EnvEntry(default="0"),
        "TRTLLM_EXAONE_EAGER_FUSION_ENABLED": _EnvEntry(default="0"),
        "TRTLLM_GLM_EAGER_FUSION_DISABLED": _EnvEntry(default="0"),
        "TRTLLM_DISABLE_NVFP4_LAYERNORM_FUSION": _EnvEntry(default="1"),
        "TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED": _EnvEntry(default="1"),
        "ENABLE_PERFECT_ROUTER": _EnvEntry(default="0"),
        "ENABLE_CONFIGURABLE_MOE": _EnvEntry(default="0"),
        "LM_HEAD_TP_SIZE": _EnvEntry(default=None),

        # ---- Multimodal ------------------------------------------------
        "TLLM_MULTIMODAL_DISAGGREGATED": _EnvEntry(default="0"),
        "TLLM_MULTIMODAL_ENCODER_TORCH_COMPILE": _EnvEntry(default="0"),
        "TLLM_VIDEO_PRUNING_RATIO": _EnvEntry(default="0"),

        # ---- FLA (fused linear attention) -------------------------------
        "FLA_COMPILER_MODE": _EnvEntry(default=None),
        "FLA_CI_ENV": _EnvEntry(default=None),
        "FLA_USE_CUDA_GRAPH": _EnvEntry(default="0"),
        "FLA_USE_FAST_OPS": _EnvEntry(default="0"),
        "FLA_CACHE_RESULTS": _EnvEntry(default="1"),
        "GDN_RECOMPUTE_SUPPRESS_LEVEL": _EnvEntry(default="0"),

        # ---- Triton kernel tuning --------------------------------------
        "TRITON_ROOT": _EnvEntry(default=None),
        "TRITON_MOE_MXFP4_NUM_WARPS": _EnvEntry(default=4),

        # ---- Expert statistic / load balancer --------------------------
        "EXPERT_STATISTIC_ITER_RANGE": _EnvEntry(default=None),
        "EXPERT_STATISTIC_PATH": _EnvEntry(default="expert_statistic"),
        "TRTLLM_EPLB_SHM_NAME": _EnvEntry(default="moe_shared"),

        # ---- KV cache / scales -----------------------------------------
        "TRTLLM_LOAD_KV_SCALES": _EnvEntry(default="0"),
        "TLLM_ALLOW_LONG_MAX_MODEL_LEN": _EnvEntry(default="0"),

        # ---- Serve / OpenAI API ----------------------------------------
        "TRTLLM_RESPONSES_API_DISABLE_STORE": _EnvEntry(default=""),
        "TRTLLM_XGUIDANCE_LENIENT": _EnvEntry(default=None),
        "DISABLE_HARMONY_ADAPTER": _EnvEntry(default="0"),
        "OPENAI_API_KEY": _EnvEntry(default=None),

        # ---- Tracing / observability -----------------------------------
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": _EnvEntry(default="grpc"),

        # ---- Benchmark / reporting -------------------------------------
        "SAVE_TO_PYTORCH_BENCHMARK_FORMAT": _EnvEntry(default=False),
        "PYTORCH_ALLOC_CONF": _EnvEntry(default=""),

        # ---- XGrammar cache --------------------------------------------
        "XGRAMMAR_CACHE_LIMIT_GB": _EnvEntry(default="1"),
    }

    def __init__(self) -> None:
        # {env_name: _EnvEntry}  — fresh copies from _KNOWN_VARS so each
        # instance gets its own mutable entries (content starts as None).
        self._registry: dict[str, _EnvEntry] = {
            name: _EnvEntry(default=entry.default)
            for name, entry in self._KNOWN_VARS.items()
        }

    # ------------------------------------------------------------------
    # Get-type interfaces
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> str:
        entry = self._registry.get(key)
        if entry is not None and entry.content is not None:
            return entry.content
        # Cache miss: fetch from os.environ (raises KeyError if absent)
        value: str = os.environ[key]
        if entry is None:
            self._registry[key] = _EnvEntry(content=value)
        else:
            entry.content = value
        return value

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        """Return cached content when available; otherwise query os.environ.

        Raises:
            NameError: If *key* is not in the pre-registered registry.  This
                acts as a safety net against unknown or misspelled env-var
                names.  Keys added at runtime via ``__setitem__`` or
                ``update`` are also accepted.
        """
        entry = self._registry.get(key)
        if entry is None:
            raise NameError(
                f"Environment variable {key!r} is not registered in "
                f"_TRTLLMENVClass.  Add it to _KNOWN_VARS in env_utils.py."
            )
        if entry.content is not None:
            return entry.content
        
        # default is not used now to follow the action of os.environ.get to return
        # the given default value.
        # if entry.default is not None:
        #     return entry.default

        # Cache miss: fetch from os.environ only when content is None
        value = os.environ.get(key, default)
        entry.content = value
        return value if value is not None else default

    # ------------------------------------------------------------------
    # Write interfaces
    # ------------------------------------------------------------------

    def __setitem__(self, key: str, value: str) -> None:
        os.environ[key] = value
        entry = self._registry.get(key)
        if entry is None:
            self._registry[key] = _EnvEntry(content=value)
        else:
            entry.content = value

    def __delitem__(self, key: str) -> None:
        del os.environ[key]  # raises KeyError if absent
        entry = self._registry.get(key)
        if entry is not None:
            entry.content = None  # invalidate cache; preserve registered default

    # ------------------------------------------------------------------
    # Membership and iteration — always authoritative from os.environ
    # ------------------------------------------------------------------

    def __contains__(self, key: object) -> bool:
        return key in os.environ

    def __iter__(self) -> Iterator[str]:
        return iter(os.environ)

    def __len__(self) -> int:
        return len(os.environ)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({dict(os.environ)!r})"


TRTLLMENV = _TRTLLMENVClass()
