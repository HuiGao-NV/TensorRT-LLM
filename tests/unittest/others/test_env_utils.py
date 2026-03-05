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
"""Unit tests for tensorrt_llm.env_utils._TRTLLMENVClass.

The module is loaded directly via importlib so the test is runnable without
the full TensorRT-LLM package (which requires mpi4py, tensorrt, etc.).
"""
import importlib.util
import os
import subprocess
import sys
from collections.abc import MutableMapping
from pathlib import Path

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
_EnvEntry = _mod._EnvEntry
TRTLLMENV = _mod.TRTLLMENV

# All test keys are prefixed to avoid colliding with real environment variables.
_PFX = "_TRTLLMENV_TEST_"

# A registered key with a non-None canonical default ("0").
# get() returns entry.default immediately — os.environ is never consulted.
_REG_KEY = "FORCE_DETERMINISTIC"
_REG_DEFAULT = "0"

# A registered key whose canonical default IS None.
# get() falls through to os.environ on the first call, then caches the result.
_REG_KEY_NO_DEFAULT = "TLLM_DISABLE_MPI"

# An unregistered key that should never appear in _KNOWN_VARS.
_UNREG_KEY = _PFX + "UNREGISTERED_KEY"


@pytest.fixture(autouse=True)
def _cleanup_test_keys():
    """Isolate each test from the host environment.

    * Removes all ``_PFX``-prefixed keys from os.environ and _registry.
    * Saves and removes ``_REG_KEY`` from os.environ so tests that need it
      absent can rely on that, then restores the original value afterwards.
    * Resets all cached content in _registry so each test starts with a
      clean cache.
    """
    # Save and clear both test keys so every test starts from a known state.
    _saved = {
        _REG_KEY: os.environ.pop(_REG_KEY, None),
        _REG_KEY_NO_DEFAULT: os.environ.pop(_REG_KEY_NO_DEFAULT, None),
    }

    def _clean():
        for key in [k for k in list(os.environ) if k.startswith(_PFX)]:
            del os.environ[key]
        os.environ.pop(_REG_KEY, None)
        os.environ.pop(_REG_KEY_NO_DEFAULT, None)
        for key in [k for k in list(TRTLLMENV._registry) if k.startswith(_PFX)]:
            del TRTLLMENV._registry[key]
        # Reset each entry to its canonical state: content=None, default restored
        # from _KNOWN_VARS (get() writes back `entry.default`, so without this
        # reset one test would pollute the next).
        for name, entry in TRTLLMENV._registry.items():
            entry.content = None
            if name in _TRTLLMENVClass._KNOWN_VARS:
                entry.default = _TRTLLMENVClass._KNOWN_VARS[name].default
            else:
                entry.default = None

    _clean()
    yield
    _clean()

    # Restore host values.
    for key, val in _saved.items():
        if val is not None:
            os.environ[key] = val


# ===========================================================================
# __init__ — pre-populated registry
# ===========================================================================


def test_known_vars_pre_registered():
    """Every key in _KNOWN_VARS is in _registry after construction."""
    fresh = _TRTLLMENVClass()
    for name in _TRTLLMENVClass._KNOWN_VARS:
        assert name in fresh._registry, f"{name!r} missing from registry"


def test_known_vars_canonical_default_stored():
    """Pre-registered entries carry the canonical default from _KNOWN_VARS."""
    fresh = _TRTLLMENVClass()
    for name, entry in _TRTLLMENVClass._KNOWN_VARS.items():
        assert fresh._registry[name].default == entry.default, (
            f"{name!r}: expected default {entry.default!r}, "
            f"got {fresh._registry[name].default!r}"
        )


def test_known_vars_content_initially_none():
    """All pre-registered entries start with content=None (not yet fetched)."""
    fresh = _TRTLLMENVClass()
    for name in _TRTLLMENVClass._KNOWN_VARS:
        assert fresh._registry[name].content is None, (
            f"{name!r} content should be None at construction"
        )


def test_known_vars_values_are_env_entry_instances():
    """Every value in _KNOWN_VARS is an _EnvEntry instance."""
    for name, entry in _TRTLLMENVClass._KNOWN_VARS.items():
        assert isinstance(entry, _EnvEntry), (
            f"{name!r}: expected _EnvEntry, got {type(entry).__name__}"
        )


def test_init_creates_independent_copies():
    """__init__ copies _KNOWN_VARS entries so instances do not share state."""
    inst_a = _TRTLLMENVClass()
    inst_b = _TRTLLMENVClass()
    for name in _TRTLLMENVClass._KNOWN_VARS:
        assert inst_a._registry[name] is not inst_b._registry[name], (
            f"{name!r}: two instances share the same _EnvEntry object"
        )
        assert inst_a._registry[name] is not _TRTLLMENVClass._KNOWN_VARS[name], (
            f"{name!r}: registry entry is the same object as _KNOWN_VARS template"
        )


def test_no_duplicate_keys_in_known_vars():
    """_KNOWN_VARS contains no duplicate keys (Python dicts silently drop them)."""
    seen = []
    # Re-parse the source to detect duplicates that dict comprehension would hide.
    import ast
    src = (_REPO_ROOT / "tensorrt_llm" / "env_utils.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "_TRTLLMENVClass"):
            for item in ast.walk(node):
                if (isinstance(item, ast.Assign)
                        and any(
                            getattr(t, 'id', '') == '_KNOWN_VARS'
                            for t in item.targets)):
                    for elt in item.value.keys:
                        seen.append(ast.literal_eval(elt))
    assert len(seen) == len(set(seen)), (
        f"Duplicate keys found in _KNOWN_VARS: "
        f"{[k for k in seen if seen.count(k) > 1]}"
    )


def test_specific_defaults_match_code():
    """Spot-check a few canonical defaults against their call-site values."""
    expected = {
        "FORCE_DETERMINISTIC": "0",
        "TLLM_DISABLE_MPI": None,
        "TRTLLM_ENABLE_PDL": "1",
        "TLLM_LLMAPI_BUILD_CACHE_ROOT": "/tmp/.cache/tensorrt_llm/llmapi/",
        "EXPERT_STATISTIC_PATH": "expert_statistic",
        "XGRAMMAR_CACHE_LIMIT_GB": "1",
        "TRTLLM_EPLB_SHM_NAME": "moe_shared",
        "TRITON_MOE_MXFP4_NUM_WARPS": 4,
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "grpc",
    }
    for name, exp_default in expected.items():
        # Check both the class-level template and a fresh instance.
        assert _TRTLLMENVClass._KNOWN_VARS[name].default == exp_default, (
            f"{name!r} _KNOWN_VARS: expected {exp_default!r}, "
            f"got {_TRTLLMENVClass._KNOWN_VARS[name].default!r}"
        )
        assert TRTLLMENV._registry[name].default == exp_default, (
            f"{name!r} registry: expected {exp_default!r}, "
            f"got {TRTLLMENV._registry[name].default!r}"
        )


# ===========================================================================
# get() — NameError for unregistered keys
# ===========================================================================


def test_get_unregistered_key_raises_nameerror():
    """get() raises NameError for a key not in the registry."""
    assert _UNREG_KEY not in TRTLLMENV._registry
    with pytest.raises(NameError, match=_UNREG_KEY):
        TRTLLMENV.get(_UNREG_KEY)


def test_get_unregistered_key_with_default_still_raises():
    """get(key, default) also raises NameError for unregistered keys."""
    with pytest.raises(NameError):
        TRTLLMENV.get(_UNREG_KEY, "fallback")


def test_get_registered_key_does_not_raise():
    """get() does not raise for a pre-registered key."""
    TRTLLMENV.get(_REG_KEY)  # must not raise


def test_get_key_registered_via_setitem_does_not_raise():
    """A key added at runtime via __setitem__ is accepted by get()."""
    new_key = _PFX + "DYNAMIC"
    TRTLLMENV[new_key] = "val"
    TRTLLMENV.get(new_key)  # must not raise


def test_get_key_registered_via_update_does_not_raise():
    """A key added via update() is accepted by get()."""
    new_key = _PFX + "VIA_UPDATE"
    TRTLLMENV.update({new_key: "v"})
    TRTLLMENV.get(new_key)  # must not raise


def test_nameerror_message_mentions_key_and_hint():
    """NameError message includes the key name and guidance."""
    with pytest.raises(NameError) as exc_info:
        TRTLLMENV.get(_UNREG_KEY)
    msg = str(exc_info.value)
    assert _UNREG_KEY in msg
    assert "_KNOWN_VARS" in msg


# ===========================================================================
# get() — cache-first: os.environ only called when content is None
# ===========================================================================


def test_get_registered_absent_key_returns_default():
    """get(registered_key, default) returns default when key is absent from os.environ."""
    assert _REG_KEY not in os.environ
    result = TRTLLMENV.get(_REG_KEY, _REG_DEFAULT)
    assert result == _REG_DEFAULT


def test_get_os_environ_takes_priority_over_registered_default():
    """get() always consults os.environ on a cache miss.

    The registered default in _KNOWN_VARS is metadata only — it is NOT used
    as a short-circuit return value.  When the key exists in os.environ, that
    value is returned and cached regardless of what entry.default holds.
    """
    # _REG_KEY has default="0" in _KNOWN_VARS, but os.environ says "1".
    os.environ[_REG_KEY] = "1"
    result = TRTLLMENV.get(_REG_KEY)
    assert result == "1"                                    # os.environ wins
    assert TRTLLMENV._registry[_REG_KEY].content == "1"    # cached from os.environ


def test_get_registered_present_key_returns_value():
    """get() returns the os.environ value for any registered key on cache miss."""
    os.environ[_REG_KEY_NO_DEFAULT] = "1"
    assert TRTLLMENV.get(_REG_KEY_NO_DEFAULT) == "1"


def test_get_caches_content_after_first_call():
    """After the first get() the os.environ value is stored in entry.content."""
    os.environ[_REG_KEY_NO_DEFAULT] = "cached_val"
    TRTLLMENV.get(_REG_KEY_NO_DEFAULT)
    assert TRTLLMENV._registry[_REG_KEY_NO_DEFAULT].content == "cached_val"


def test_get_uses_cache_skips_os_environ():
    """Once content is cached, get() returns it; direct os.environ changes ignored."""
    os.environ[_REG_KEY_NO_DEFAULT] = "original"
    assert TRTLLMENV.get(_REG_KEY_NO_DEFAULT) == "original"   # populates cache
    os.environ[_REG_KEY_NO_DEFAULT] = "changed"               # bypasses TRTLLMENV
    assert TRTLLMENV.get(_REG_KEY_NO_DEFAULT) == "original"   # cache hit


def test_get_missing_key_stores_default_as_content():
    """When the key is absent, os.environ.get(key, default) returns the default.
    get() stores that returned value in entry.content (the default itself).
    """
    assert _REG_KEY_NO_DEFAULT not in os.environ
    TRTLLMENV.get(_REG_KEY_NO_DEFAULT, "dflt")
    # os.environ.get(key, "dflt") == "dflt" → entry.content is set to "dflt".
    assert TRTLLMENV._registry[_REG_KEY_NO_DEFAULT].content == "dflt"


def test_get_missing_key_returns_call_site_default():
    """When default=None and key absent from os.environ, call-site default is returned."""
    assert _REG_KEY_NO_DEFAULT not in os.environ
    assert TRTLLMENV.get(_REG_KEY_NO_DEFAULT, "my_default") == "my_default"


def test_get_preserves_canonical_default_on_repeat_calls():
    """Repeated get() calls do not overwrite the canonical default in the registry."""
    canonical = TRTLLMENV._registry[_REG_KEY].default
    TRTLLMENV.get(_REG_KEY, "call_site_default")
    assert TRTLLMENV._registry[_REG_KEY].default == canonical


# ===========================================================================
# __getitem__ / [] read access
# ===========================================================================


def test_getitem_existing_key():
    """TRTLLMENV[key] returns the value present in os.environ."""
    os.environ[_REG_KEY] = "hello"
    assert TRTLLMENV[_REG_KEY] == "hello"


def test_getitem_returns_str():
    """The returned value is a str."""
    os.environ[_REG_KEY] = "42"
    assert TRTLLMENV[_REG_KEY] == "42"
    assert isinstance(TRTLLMENV[_REG_KEY], str)


def test_getitem_missing_key_raises_keyerror():
    """Accessing a missing registered key via [] raises KeyError."""
    assert _REG_KEY not in os.environ
    with pytest.raises(KeyError):
        _ = TRTLLMENV[_REG_KEY]


def test_getitem_caches_on_first_access():
    """After the first [] access the value is stored in _registry[key].content."""
    os.environ[_REG_KEY] = "cached"
    _ = TRTLLMENV[_REG_KEY]
    assert TRTLLMENV._registry[_REG_KEY].content == "cached"


def test_getitem_returns_cached_content():
    """Once cached, [] returns the registry content even if os.environ changed."""
    os.environ[_REG_KEY] = "first"
    assert TRTLLMENV[_REG_KEY] == "first"       # caches "first"
    os.environ[_REG_KEY] = "second"             # direct write, bypasses cache
    assert TRTLLMENV[_REG_KEY] == "first"       # cache hit
    TRTLLMENV[_REG_KEY] = "second"              # proper write through TRTLLMENV
    assert TRTLLMENV[_REG_KEY] == "second"      # cache updated


# ===========================================================================
# __setitem__ / [] write access
# ===========================================================================


def test_setitem_new_key_readable_via_bracket():
    """A new key set through TRTLLMENV is readable back through TRTLLMENV[]."""
    key = _PFX + "SET_NEW"
    TRTLLMENV[key] = "value1"
    assert TRTLLMENV[key] == "value1"


def test_setitem_propagates_to_os_environ():
    """Writes through TRTLLMENV are immediately visible in os.environ."""
    TRTLLMENV[_REG_KEY] = "propagated"
    assert os.environ.get(_REG_KEY) == "propagated"


def test_setitem_overwrites_existing():
    """TRTLLMENV[key] = v replaces any pre-existing value."""
    os.environ[_REG_KEY] = "old"
    TRTLLMENV[_REG_KEY] = "new"
    assert TRTLLMENV[_REG_KEY] == "new"
    assert os.environ[_REG_KEY] == "new"


def test_setitem_empty_string():
    """Setting a key to an empty string is valid and round-trips correctly."""
    TRTLLMENV[_REG_KEY] = ""
    assert TRTLLMENV[_REG_KEY] == ""
    assert os.environ[_REG_KEY] == ""


def test_setitem_updates_registry_content():
    """__setitem__ stores the new value in _registry[key].content."""
    TRTLLMENV[_REG_KEY] = "reg_val"
    assert TRTLLMENV._registry[_REG_KEY].content == "reg_val"


def test_setitem_updates_existing_registry_entry_in_place():
    """__setitem__ on an already-registered key updates the same entry object."""
    TRTLLMENV[_REG_KEY] = "first"
    entry_before = TRTLLMENV._registry[_REG_KEY]
    TRTLLMENV[_REG_KEY] = "second"
    assert TRTLLMENV._registry[_REG_KEY] is entry_before
    assert entry_before.content == "second"


# ===========================================================================
# __delitem__
# ===========================================================================


def test_delitem_removes_from_os_environ():
    """del TRTLLMENV[key] removes the key from os.environ."""
    os.environ[_REG_KEY] = "to_delete"
    del TRTLLMENV[_REG_KEY]
    assert _REG_KEY not in os.environ


def test_delitem_key_absent_from_trtllmenv_after_del():
    """del TRTLLMENV[key] makes the key absent from TRTLLMENV."""
    TRTLLMENV[_REG_KEY] = "val"
    del TRTLLMENV[_REG_KEY]
    assert _REG_KEY not in TRTLLMENV


def test_delitem_invalidates_cache():
    """del TRTLLMENV[key] resets _registry[key].content to None."""
    TRTLLMENV[_REG_KEY] = "cached"
    assert TRTLLMENV._registry[_REG_KEY].content == "cached"
    del TRTLLMENV[_REG_KEY]
    assert TRTLLMENV._registry[_REG_KEY].content is None


def test_delitem_preserves_canonical_default():
    """del TRTLLMENV[key] does not erase the registered default."""
    canonical = TRTLLMENV._registry[_REG_KEY].default
    TRTLLMENV[_REG_KEY] = "tmp"
    del TRTLLMENV[_REG_KEY]
    assert TRTLLMENV._registry[_REG_KEY].default == canonical


def test_delitem_missing_key_raises_keyerror():
    """Deleting a missing key raises KeyError."""
    assert _REG_KEY not in os.environ
    with pytest.raises(KeyError):
        del TRTLLMENV[_REG_KEY]


# ===========================================================================
# __contains__
# ===========================================================================


def test_contains_existing_key():
    os.environ[_REG_KEY] = "yes"
    assert _REG_KEY in TRTLLMENV


def test_contains_missing_key():
    assert _REG_KEY not in os.environ
    assert _REG_KEY not in TRTLLMENV


# ===========================================================================
# update()
# ===========================================================================


def test_update_from_dict():
    """update() applies all key-value pairs to both os.environ and registry."""
    k1, k2 = _PFX + "UPD1", _PFX + "UPD2"
    TRTLLMENV.update({k1: "a", k2: "b"})
    assert os.environ[k1] == "a"
    assert os.environ[k2] == "b"
    assert TRTLLMENV._registry[k1].content == "a"
    assert TRTLLMENV._registry[k2].content == "b"


def test_update_overwrites_existing_values():
    TRTLLMENV[_REG_KEY] = "old"
    TRTLLMENV.update({_REG_KEY: "new"})
    assert TRTLLMENV[_REG_KEY] == "new"


# ===========================================================================
# __iter__ and __len__
# ===========================================================================


def test_len_matches_os_environ():
    assert len(TRTLLMENV) == len(os.environ)


def test_len_increases_after_set():
    before = len(TRTLLMENV)
    TRTLLMENV[_PFX + "LEN_NEW"] = "1"
    assert len(TRTLLMENV) == before + 1


def test_iter_yields_all_keys():
    assert set(TRTLLMENV) == set(os.environ)


def test_iter_reflects_new_key():
    key = _PFX + "ITER_NEW"
    TRTLLMENV[key] = "1"
    assert key in set(TRTLLMENV)


# ===========================================================================
# Registry structure
# ===========================================================================


def test_registry_entry_is_env_entry_type():
    """Each _registry value is an _EnvEntry instance."""
    assert isinstance(TRTLLMENV._registry[_REG_KEY], _EnvEntry)


def test_registry_entry_has_default_and_content_fields():
    """_EnvEntry exposes .default and .content attributes."""
    entry = _EnvEntry(default="d", content="c")
    assert entry.default == "d"
    assert entry.content == "c"


# ===========================================================================
# Write-through to os.environ
# ===========================================================================


def test_write_through_to_os_environ():
    """Writes via TRTLLMENV are visible through os.environ."""
    TRTLLMENV[_REG_KEY] = "written"
    assert os.environ[_REG_KEY] == "written"
    TRTLLMENV[_REG_KEY] = "updated"
    assert os.environ[_REG_KEY] == "updated"


# ===========================================================================
# MutableMapping contract and singleton
# ===========================================================================


def test_is_mutable_mapping_instance():
    assert isinstance(TRTLLMENV, MutableMapping)


def test_singleton_identity():
    """The module-level TRTLLMENV object is always the same instance."""
    assert TRTLLMENV is _mod.TRTLLMENV


def test_repr_contains_class_name():
    assert "_TRTLLMENVClass" in repr(TRTLLMENV)


# ===========================================================================
# subprocess compatibility
# ===========================================================================


def test_passable_as_subprocess_env():
    """TRTLLMENV works as the env= argument to subprocess.run/Popen."""
    key = _PFX + "SUBPROC"
    TRTLLMENV[key] = "subprocess_test"
    result = subprocess.run(
        [
            sys.executable, "-c",
            f"import os; print(os.environ.get('{key}', ''))"
        ],
        capture_output=True,
        text=True,
        env=TRTLLMENV,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "subprocess_test"
