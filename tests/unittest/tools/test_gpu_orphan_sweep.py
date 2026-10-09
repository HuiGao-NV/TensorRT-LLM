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
"""Unit tests for the session-wide GPU orphan sweep.

Covers ``run_with_gpu_orphan_sweep.py``, which CI puts in front of each stage's
pytest, and the ``GpuOrphanSweep`` pytest plugin. The harness that finds and kills
the processes is replaced by a fake, so no GPU is needed.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

__extra_import_path__ = ["~/tests/integration/defs/utils"]
import gpu_orphan_sweep_plugin
import run_with_gpu_orphan_sweep as wrapper

pytestmark = pytest.mark.cpu_only

TOKEN_ENV = "TRTLLM_TEST_LAUNCH_TOKEN"
DISABLE_ENV = "TRTLLM_TEST_GPU_ORPHAN_SWEEP"
WRAPPER_SCRIPT = Path(wrapper.__file__)


class FakeHarness:
    """Stands in for ``trt_test_alternative``; records the sweeps."""

    LAUNCH_TOKEN_ENV = TOKEN_ENV

    def __init__(self, token="tok"):
        self.token = token
        self.sweeps = []

    def tag_launch(self, kwargs):
        if self.token is None:
            return kwargs, None
        outer = os.environ.get(TOKEN_ENV)
        value = f"{outer}:{self.token}" if outer else self.token
        return {**kwargs, "env": {**os.environ, TOKEN_ENV: value}}, self.token

    def kill_orphaned_gpu_processes(self, token, include_descendants=False):
        self.sweeps.append((token, include_descendants))


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    monkeypatch.delenv(DISABLE_ENV, raising=False)


def _python(code):
    return [sys.executable, "-c", code]


@pytest.mark.parametrize("returncode, status", [(0, 0), (3, 3), (-9, 137), (-15, 143)])
def test_exit_status_maps_signal_deaths_like_a_shell(returncode, status):
    assert wrapper._exit_status(returncode) == status


@pytest.mark.parametrize("code", [0, 3])
def test_wrapper_returns_exit_status_and_sweeps_afterwards(monkeypatch, code):
    harness = FakeHarness()
    monkeypatch.setattr(wrapper, "_load_harness", lambda: harness)

    status = wrapper.main(["--"] + _python(f"raise SystemExit({code})"))

    assert status == code
    assert harness.sweeps == [("tok", True)]


def test_wrapper_gives_the_command_the_launch_token(monkeypatch, tmp_path):
    monkeypatch.setattr(wrapper, "_load_harness", FakeHarness)
    out = tmp_path / "token"
    code = f"import os; open({str(out)!r}, 'w').write(os.environ[{TOKEN_ENV!r}])"

    assert wrapper.main(["--"] + _python(code)) == 0

    assert out.read_text() == "tok"


def test_wrapper_without_command_is_a_usage_error(monkeypatch, capsys):
    monkeypatch.setattr(wrapper, "_load_harness", FakeHarness)

    assert wrapper.main(["--"]) == 2
    assert "usage" in capsys.readouterr().err


def test_disabled_wrapper_runs_the_command_unchanged():
    env = {**os.environ, DISABLE_ENV: "0"}

    result = subprocess.run(
        [sys.executable, str(WRAPPER_SCRIPT), "--"] + _python("raise SystemExit(5)"),
        env=env,
        timeout=60,
    )

    assert result.returncode == 5


@pytest.mark.skipif(sys.platform != "linux", reason="the sweep is Linux-only")
def test_wrapper_script_gives_the_command_a_token():
    env = {k: v for k, v in os.environ.items() if k not in (TOKEN_ENV, DISABLE_ENV)}

    out = subprocess.check_output(
        [sys.executable, str(WRAPPER_SCRIPT), "--"]
        + _python(f"import os; print(os.environ.get({TOKEN_ENV!r}, ''))"),
        env=env,
        text=True,
    )

    assert out.strip()


def test_plugin_tags_the_session_and_sweeps_with_descendants(monkeypatch):
    harness = FakeHarness()
    plugin = gpu_orphan_sweep_plugin.GpuOrphanSweep(harness)

    plugin.start()
    assert os.environ[TOKEN_ENV] == "tok"
    plugin.pytest_sessionfinish(SimpleNamespace(), 0)

    assert harness.sweeps == [("tok", True)]
    assert TOKEN_ENV not in os.environ


def test_plugin_keeps_an_enclosing_token_and_restores_it(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "outer")
    harness = FakeHarness()
    plugin = gpu_orphan_sweep_plugin.GpuOrphanSweep(harness)

    plugin.start()
    assert os.environ[TOKEN_ENV] == "outer:tok"
    plugin.pytest_sessionfinish(SimpleNamespace(), 0)

    assert os.environ[TOKEN_ENV] == "outer"


def test_plugin_restores_the_environment_when_the_sweep_fails():
    harness = FakeHarness()

    def boom(*args, **kwargs):
        raise RuntimeError("sweep failed")

    harness.kill_orphaned_gpu_processes = boom
    plugin = gpu_orphan_sweep_plugin.GpuOrphanSweep(harness)
    plugin.start()

    with pytest.raises(RuntimeError):
        plugin.pytest_sessionfinish(SimpleNamespace(), 0)

    assert TOKEN_ENV not in os.environ


def test_plugin_does_nothing_when_disabled(monkeypatch):
    monkeypatch.setenv(DISABLE_ENV, "0")
    harness = FakeHarness()
    plugin = gpu_orphan_sweep_plugin.GpuOrphanSweep(harness)

    plugin.start()
    plugin.pytest_sessionfinish(SimpleNamespace(), 0)

    assert TOKEN_ENV not in os.environ
    assert harness.sweeps == []


def test_plugin_does_nothing_where_there_is_no_sweep():
    harness = FakeHarness(token=None)
    plugin = gpu_orphan_sweep_plugin.GpuOrphanSweep(harness)

    plugin.start()
    plugin.pytest_sessionfinish(SimpleNamespace(), 0)

    assert TOKEN_ENV not in os.environ
    assert harness.sweeps == []
