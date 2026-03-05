# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
import inspect

import triton

from tensorrt_llm.env_utils import TRTLLMENV

FLA_CACHE_RESULTS = TRTLLMENV.get("FLA_CACHE_RESULTS", "1") == "1"


supports_autotune_cache = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if supports_autotune_cache else {}
