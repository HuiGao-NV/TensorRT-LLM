# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""pytest plugin that kills the GPU processes a test session leaves behind.

The session gets a unique launch token in ``os.environ``, which every process the
tests start inherits, however they start it. When the session ends, whatever
still holds the GPU and carries the token is killed. It is a safety net for tests
that leak MPI workers or servers; it cannot help when the session itself is killed
with ``os._exit()``, which ``run_with_gpu_orphan_sweep.py`` covers from outside.

Set ``TRTLLM_TEST_GPU_ORPHAN_SWEEP=0`` to turn it off.
"""

import os

import pytest

_DISABLE_ENV = "TRTLLM_TEST_GPU_ORPHAN_SWEEP"


class GpuOrphanSweep:
    """Tag the session with a launch token and sweep its GPU processes at the end."""

    def __init__(self, harness):
        """``harness`` is the ``trt_test_alternative`` module."""
        self._harness = harness
        self._token = None
        self._had_token_env = False
        self._previous_token_env = None

    def start(self):
        """Put a launch token in ``os.environ`` for the rest of the session."""
        if os.environ.get(_DISABLE_ENV) == "0":
            return
        kwargs, token = self._harness.tag_launch({})
        if token is None:  # no sweep on this platform
            return
        env_name = self._harness.LAUNCH_TOKEN_ENV
        self._token = token
        self._had_token_env = env_name in os.environ
        self._previous_token_env = os.environ.get(env_name)
        os.environ[env_name] = kwargs["env"][env_name]

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        """Kill what still holds the GPU, then restore the environment."""
        if self._token is None:
            return
        env_name = self._harness.LAUNCH_TOKEN_ENV
        try:
            self._harness.kill_orphaned_gpu_processes(self._token,
                                                      include_descendants=True)
        finally:
            if self._had_token_env:
                os.environ[env_name] = self._previous_token_env
            else:
                os.environ.pop(env_name, None)
            self._token = None
