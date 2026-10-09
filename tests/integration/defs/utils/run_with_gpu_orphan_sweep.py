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
"""Run a command and kill the GPU processes it leaves behind.

Usage: run_with_gpu_orphan_sweep.py -- <command> [args...]

pytest-timeout with ``--timeout-method=thread`` ends the test process with
``os._exit()``, so nothing inside that process can clean up its MPI workers; they
keep their GPU memory and make later tests fail with out-of-memory errors. This
wrapper outlives the command. It gives the command a unique launch token through the
environment, which every descendant inherits, and after the command exits it kills
the GPU processes of the current user that still carry the token.

The command's exit status is returned unchanged. The wrapper runs the command
without a sweep when the sweep cannot work (not Linux, psutil missing) or when
``TRTLLM_TEST_GPU_ORPHAN_SWEEP=0``.
"""

import os
import signal
import subprocess
import sys

_DISABLE_ENV = "TRTLLM_TEST_GPU_ORPHAN_SWEEP"
_FORWARDED_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


def _load_harness():
    """Return the ``trt_test_alternative`` module, or None if unusable."""
    if os.environ.get(_DISABLE_ENV) == "0" or not sys.platform.startswith("linux"):
        return None
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import trt_test_alternative
    except ImportError:
        return None
    return trt_test_alternative


def _exit_status(returncode: int) -> int:
    """Map a Popen return code to a shell exit status."""
    return 128 - returncode if returncode < 0 else returncode


def main(argv) -> int:
    """Run the command in ``argv`` and sweep afterwards; return its exit status."""
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    if not argv:
        print(f"usage: {sys.argv[0]} -- <command> [args...]", file=sys.stderr)
        return 2

    harness = _load_harness()
    if harness is None:
        os.execvp(argv[0], argv)

    kwargs, token = harness.tag_launch({})
    child = subprocess.Popen(argv, env=kwargs["env"])
    for sig in _FORWARDED_SIGNALS:
        signal.signal(sig, lambda signum, _frame: child.send_signal(signum))
    returncode = child.wait()

    # The command is gone, so nothing of this launch should still hold the GPU.
    harness.kill_orphaned_gpu_processes(token, include_descendants=True)
    return _exit_status(returncode)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
