#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Run lm_eval accuracy tasks against a running disaggregated server.

Reads configuration from environment variables set by submit.py / slurm_launch_disagg.sh:

  ACCURACY_CONFIG_JSON   JSON string with nested accuracy config (tasks dict format).
  DISAGG_SERVER_HOST     Hostname/IP of the disaggregated server.
  DISAGG_SERVER_PORT     Port of the disaggregated server (default: 8333).
  LLM_MODELS_ROOT        Root directory for model weights.
  MODEL_DIR_NAME         Model directory name relative to LLM_MODELS_ROOT.
  jobWorkspace           Work directory for logs and output (default: current dir).

Accuracy config JSON format (examples/disaggregated/slurm/benchmark/config.yaml style):
  {
    "enable_accuracy_test": true,
    "tasks": {
      "gsm8k": {
        "model": "local-completions",
        "model_args_extra": "num_concurrent=512,...",
        "extra_kwargs": {
          "trust_remote_code": true,
          "include_path": "/path/to/lm_eval_configs"   # optional, for custom tasks
        }
      }
    }
  }
"""

import json
import os
import subprocess
import sys

END_POINT_MAP = {
    "local-completions": "v1/completions",
    "local-chat-completions": "v1/chat/completions",
}


def main():
    # --- Read env vars ---
    acc_json = os.environ.get("ACCURACY_CONFIG_JSON", "")
    if not acc_json:
        print("ACCURACY_CONFIG_JSON is not set, nothing to do.")
        sys.exit(0)

    try:
        acc_cfg = json.loads(acc_json)
    except json.JSONDecodeError as e:
        print(f"Failed to parse ACCURACY_CONFIG_JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if not acc_cfg.get("enable_accuracy_test"):
        print("enable_accuracy_test is false, skipping accuracy evaluation.")
        sys.exit(0)

    host = os.environ.get("DISAGG_SERVER_HOST", "")
    if not host:
        print("DISAGG_SERVER_HOST is not set", file=sys.stderr)
        sys.exit(1)

    port = os.environ.get("DISAGG_SERVER_PORT", "8333")
    llm_models_root = os.environ.get("LLM_MODELS_ROOT", "")
    model_dir_name = os.environ.get("MODEL_DIR_NAME", "")
    work_dir = os.environ.get("jobWorkspace", ".")

    model_path = (
        os.path.join(llm_models_root, model_dir_name)
        if model_dir_name
        else llm_models_root
    )

    tasks = acc_cfg.get("tasks", {})
    if not tasks:
        print("No accuracy tasks configured, skipping.", file=sys.stderr)
        sys.exit(0)

    failures = []

    for task_name, task_cfg in tasks.items():
        model = task_cfg.get("model", "local-completions")
        model_args_extra = task_cfg.get("model_args_extra", "")
        extra_kwargs = task_cfg.get("extra_kwargs", {})

        endpoint = END_POINT_MAP.get(model, "v1/completions")
        model_args = (
            f"model={model_path},"
            f"base_url=http://{host}:{port}/{endpoint}"
        )
        if model_args_extra:
            model_args += f",{model_args_extra}"

        cmd = [
            "lm_eval",
            "--model", model,
            "--tasks", task_name,
            "--model_args", model_args,
            "--log_samples",
            "--output_path", os.path.join(work_dir, f"accuracy_eval_{task_name}"),
        ]

        # Append extra_kwargs as CLI flags (mirrors examples/disaggregated/slurm/benchmark/submit.py)
        for key, value in extra_kwargs.items():
            if key == "include_path":
                cmd.append(f"--include_path={value}")
            elif isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.append(f"--{key}={value}")

        log_file = os.path.join(work_dir, f"7_accuracy_eval_{task_name}.log")
        print(f"Running accuracy test: task={task_name}, model={model}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Log: {log_file}")

        with open(log_file, "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

        if result.returncode != 0:
            failures.append(task_name)
            print(f"  FAILED (exit code {result.returncode})")
        else:
            print(f"  PASSED")

    if failures:
        print(f"\nAccuracy tests FAILED for tasks: {failures}", file=sys.stderr)
        sys.exit(1)

    print("\nAll accuracy tests completed successfully.")


if __name__ == "__main__":
    main()
