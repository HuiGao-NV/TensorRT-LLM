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
"""Performance benchmark: os.environ API vs _TRTLLMENVClass cached access.

Each method queries the *same* environment variable 100 times.  The
minimum wall-clock time across several repetitions is reported.

Two distinct access paths inside _TRTLLMENVClass.get() are benchmarked:

  Path B – warm cache (entry.content is not None)
      Returned immediately without calling os.environ.
      Content is set on the first call that receives a non-None result.

      Two warm-cache sub-scenarios are shown:
        B1 – key present in os.environ (content = real value)
        B2 – key absent from os.environ but a default is provided
             (os.environ.get(key, default) returns the default, which
              is then stored in entry.content for all future calls)

  Path C – cold (entry.content is None)
      os.environ is consulted and the result is stored in entry.content.
      This happens on the first call, or whenever no non-None value has
      been cached yet.

The benchmark compares Path B against the three standard OS accessors:
os.getenv(), os.environ.get(), and os.environ["key"].
"""
import importlib.util
import os
import timeit
from pathlib import Path
from typing import Callable

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: load env_utils without triggering tensorrt_llm/__init__.py
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "tensorrt_llm.env_utils",
    _REPO_ROOT / "tensorrt_llm" / "env_utils.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_TRTLLMENVClass = _mod._TRTLLMENVClass

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------
_QUERIES = 100   # queries per timing run ("100 times" per the requirement)
_REPEATS = 7     # independent timing repetitions; minimum is reported

# Registered key with a non-None canonical default ("0").
# Used for Path B2: key absent from os.environ, warm via explicit default.
_KEY_WITH_DEFAULT = "FORCE_DETERMINISTIC"
_KEY_WITH_DEFAULT_VAL = "0"

# Registered key with canonical default=None.
# Used for Path B1 (key in os.environ) and Path C (cold).
_KEY_NO_DEFAULT = "TLLM_DISABLE_MPI"
_KEY_NO_DEFAULT_VAL = "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _min_time(fn: Callable, n: int = _QUERIES, repeats: int = _REPEATS) -> float:
    """Return the minimum total time (seconds) for *n* calls across *repeats* runs."""
    return min(timeit.repeat(fn, number=n, repeat=repeats))


def _us(seconds: float) -> str:
    return f"{seconds * 1_000_000:.2f} µs"


def _print_table(title: str, results: dict[str, float]) -> None:
    sep = "-" * 62
    print(f"\n{sep}")
    print(f"  {title}")
    print(f"  (100 queries, minimum of {_REPEATS} runs)")
    print(sep)
    fastest = min(results.values())
    for label, t in results.items():
        ratio = t / fastest
        bar = "█" * min(int(ratio * 20), 60)
        print(f"  {label:<38s}  {_us(t):>12s}  {ratio:5.2f}x  {bar}")
    print(sep)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fresh_env():
    """Yield a fresh _TRTLLMENVClass instance isolated from the global singleton."""
    return _TRTLLMENVClass()


@pytest.fixture(autouse=True)
def _clean_os_environ():
    """Save and restore the two test keys around each test."""
    saved = {
        _KEY_WITH_DEFAULT: os.environ.pop(_KEY_WITH_DEFAULT, None),
        _KEY_NO_DEFAULT: os.environ.pop(_KEY_NO_DEFAULT, None),
    }
    yield
    for key, val in saved.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


# ===========================================================================
# Path B2: warm cache — key absent, default provided
# ===========================================================================

def test_perf_path_b2_warm_cache_absent_key(capsys, fresh_env):
    """Path B2 — warm cache when the key is absent from os.environ.

    On the first get(key, default) call:
      os.environ.get(key, default) returns the default value;
      that value is stored in entry.content.
    All subsequent calls return entry.content directly without hitting
    os.environ.
    """
    env = fresh_env
    assert _KEY_WITH_DEFAULT not in os.environ

    # Warm up: call with an explicit default so entry.content gets populated.
    # os.environ.get(_KEY_WITH_DEFAULT, "0") == "0" → entry.content = "0".
    env.get(_KEY_WITH_DEFAULT, _KEY_WITH_DEFAULT_VAL)
    assert env._registry[_KEY_WITH_DEFAULT].content == _KEY_WITH_DEFAULT_VAL

    t_os_getenv  = _min_time(lambda: os.getenv(_KEY_WITH_DEFAULT,
                                               _KEY_WITH_DEFAULT_VAL))
    t_os_get     = _min_time(lambda: os.environ.get(_KEY_WITH_DEFAULT,
                                                    _KEY_WITH_DEFAULT_VAL))
    t_os_bracket = _min_time(lambda: os.environ.get(_KEY_WITH_DEFAULT))
    t_trtllm     = _min_time(lambda: env.get(_KEY_WITH_DEFAULT,
                                             _KEY_WITH_DEFAULT_VAL))

    results = {
        "os.getenv(key, default)": t_os_getenv,
        "os.environ.get(key, default)": t_os_get,
        "os.environ.get(key)  [no default]": t_os_bracket,
        "TRTLLMENV.get(key)   [Path B2: warm]": t_trtllm,
    }

    with capsys.disabled():
        _print_table(
            "Path B2 — warm cache, absent key (os.environ not called)", results)

    assert t_trtllm < t_os_getenv, (
        f"Expected TRTLLMENV Path-B2 ({_us(t_trtllm)}) < "
        f"os.getenv ({_us(t_os_getenv)})")
    assert t_trtllm < t_os_get, (
        f"Expected TRTLLMENV Path-B2 ({_us(t_trtllm)}) < "
        f"os.environ.get ({_us(t_os_get)})")


# ===========================================================================
# Path B1: warm cache — key present in os.environ
# ===========================================================================

def test_perf_path_b1_warm_cache_present_key(capsys, fresh_env):
    """Path B1 — warm cache when the key is present in os.environ.

    The first get() call fetches from os.environ and caches the result.
    All subsequent calls return the cached value in O(1) without any OS call.
    """
    env = fresh_env
    os.environ[_KEY_NO_DEFAULT] = _KEY_NO_DEFAULT_VAL

    # Warm up: first call stores value in entry.content.
    result = env.get(_KEY_NO_DEFAULT)
    assert result == _KEY_NO_DEFAULT_VAL
    assert env._registry[_KEY_NO_DEFAULT].content == _KEY_NO_DEFAULT_VAL

    # All 100 queries below will hit Path B1 (entry.content is not None).
    t_os_getenv  = _min_time(lambda: os.getenv(_KEY_NO_DEFAULT))
    t_os_get     = _min_time(lambda: os.environ.get(_KEY_NO_DEFAULT))
    t_os_bracket = _min_time(lambda: os.environ[_KEY_NO_DEFAULT])
    t_trtllm     = _min_time(lambda: env.get(_KEY_NO_DEFAULT))

    results = {
        "os.getenv(key)": t_os_getenv,
        "os.environ.get(key)": t_os_get,
        'os.environ["key"]': t_os_bracket,
        "TRTLLMENV.get(key)   [Path B1: warm]": t_trtllm,
    }

    with capsys.disabled():
        _print_table("Path B1 — warm cache, present key (os.environ not called)",
                     results)

    assert t_trtllm < t_os_getenv, (
        f"Expected TRTLLMENV Path-B1 ({_us(t_trtllm)}) < "
        f"os.getenv ({_us(t_os_getenv)})")
    assert t_trtllm < t_os_get, (
        f"Expected TRTLLMENV Path-B1 ({_us(t_trtllm)}) < "
        f"os.environ.get ({_us(t_os_get)})")


# ===========================================================================
# Path C: cold first lookup
# ===========================================================================

def test_perf_path_c_cold_first_lookup(capsys, fresh_env):
    """Path C — first call hits os.environ; subsequent calls use the cache.

    This test is informational: it shows how long the very first get() takes
    compared to os.environ methods (it must call os.environ, so it cannot be
    faster).  After the warm-up, re-timing confirms the cache kicks in.
    """
    env = fresh_env
    os.environ[_KEY_NO_DEFAULT] = _KEY_NO_DEFAULT_VAL

    # Cold timing: every call clears the cache first.
    def _cold_query():
        env._registry[_KEY_NO_DEFAULT].content = None
        return env.get(_KEY_NO_DEFAULT)

    t_os_getenv  = _min_time(lambda: os.getenv(_KEY_NO_DEFAULT))
    t_os_get     = _min_time(lambda: os.environ.get(_KEY_NO_DEFAULT))
    t_os_bracket = _min_time(lambda: os.environ[_KEY_NO_DEFAULT])
    t_cold       = _min_time(_cold_query)

    # Warm the cache once, then measure hot access.
    env.get(_KEY_NO_DEFAULT)
    t_warm = _min_time(lambda: env.get(_KEY_NO_DEFAULT))

    results = {
        "os.getenv(key)": t_os_getenv,
        "os.environ.get(key)": t_os_get,
        'os.environ["key"]': t_os_bracket,
        "TRTLLMENV.get(key)   [Path C: cold]": t_cold,
        "TRTLLMENV.get(key)   [Path B: warm]": t_warm,
    }

    with capsys.disabled():
        _print_table("Path C — cold vs warm TRTLLMENV.get()", results)

    # Cold path must call os.environ so it is expected to be slower; no assertion.
    # Warm path should be faster than os.getenv (highest-overhead OS accessor).
    assert t_warm < t_os_getenv, (
        f"Expected warm TRTLLMENV ({_us(t_warm)}) < "
        f"os.getenv ({_us(t_os_getenv)})")


# ===========================================================================
# Summary: side-by-side comparison of both paths vs all OS accessors
# ===========================================================================

def test_perf_summary(capsys, fresh_env):
    """Print a single consolidated comparison table for all access patterns."""
    env = fresh_env

    # Set up os.environ
    os.environ[_KEY_NO_DEFAULT] = _KEY_NO_DEFAULT_VAL
    assert _KEY_WITH_DEFAULT not in os.environ

    # Warm caches
    # B2: key absent — must provide the explicit default so entry.content is set.
    env.get(_KEY_WITH_DEFAULT, _KEY_WITH_DEFAULT_VAL)
    assert env._registry[_KEY_WITH_DEFAULT].content == _KEY_WITH_DEFAULT_VAL
    # B1: key present — first call populates entry.content from os.environ.
    env.get(_KEY_NO_DEFAULT)
    assert env._registry[_KEY_NO_DEFAULT].content == _KEY_NO_DEFAULT_VAL

    # OS accessors (use the key that is in os.environ for a fair comparison)
    t_os_getenv  = _min_time(lambda: os.getenv(_KEY_NO_DEFAULT))
    t_os_get     = _min_time(lambda: os.environ.get(_KEY_NO_DEFAULT))
    t_os_bracket = _min_time(lambda: os.environ[_KEY_NO_DEFAULT])

    # TRTLLMENV warm paths
    t_path_b2 = _min_time(lambda: env.get(_KEY_WITH_DEFAULT,
                                          _KEY_WITH_DEFAULT_VAL))
    t_path_b1 = _min_time(lambda: env.get(_KEY_NO_DEFAULT))

    # Cold path (reset per call)
    def _cold():
        env._registry[_KEY_NO_DEFAULT].content = None
        return env.get(_KEY_NO_DEFAULT)

    t_path_c = _min_time(_cold)
    # Restore after cold measurements
    env.get(_KEY_NO_DEFAULT)

    results = {
        "os.getenv(key)": t_os_getenv,
        "os.environ.get(key)": t_os_get,
        'os.environ["key"]': t_os_bracket,
        "TRTLLMENV.get  [Path B2: warm, absent key]": t_path_b2,
        "TRTLLMENV.get  [Path B1: warm, present key]": t_path_b1,
        "TRTLLMENV.get  [Path C: cold]": t_path_c,
    }

    with capsys.disabled():
        _print_table("Summary — all access patterns, 100 queries each", results)

    # Both warm paths must beat os.getenv (highest-overhead OS accessor).
    assert t_path_b2 < t_os_getenv, (
        f"Path B2 ({_us(t_path_b2)}) should be faster than "
        f"os.getenv ({_us(t_os_getenv)})")
    assert t_path_b1 < t_os_getenv, (
        f"Path B1 ({_us(t_path_b1)}) should be faster than "
        f"os.getenv ({_us(t_os_getenv)})")
