# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Measure peak host memory of CPU-staged vs direct-to-device loads on Spyre.

Every load runs in a fresh process, sampled from outside through
``/proc/<pid>/status`` (see ``tests/spyre/_load_memory_helpers.py``). Columns:

* ``base anon``  anonymous memory after imports, before the load
* ``peak anon``  highest anonymous memory during the load (cannot be reclaimed)
* ``load anon``  ``peak anon`` minus ``base anon``: what the load itself added
* ``peak file``  highest file-backed memory (checkpoint mmap; reclaimable)
* ``peak total`` highest anon + file + shmem in a single sample
* ``off spyre``  parameters not on the device afterwards (should be 0)

By default each model is measured in its checkpoint dtype and in the other
16-bit dtype, staged and direct: four loads per model.

Usage (on a Spyre pod)::

    python scripts/measure_direct_load_memory.py \\
        --model-path ibm-granite/granite-3.3-8b-instruct

    # one dtype, and the direct path with Transformers' read-ahead left on
    # (the behaviour before the direct block turned it off)
    python scripts/measure_direct_load_memory.py \\
        --model-path ibm-granite/granite-3.3-8b-instruct --dtype float16 \\
        --keep-read-ahead

    # several models, results also as JSON
    python scripts/measure_direct_load_memory.py \\
        --model-path ibm-granite/granite-3.3-8b-instruct \\
        --model-path ibm-granite/granite-3.3-2b-instruct --json mem.json
"""

import argparse
import dataclasses
import json
import os
import sys

# Ensure the project root (parent of scripts/) is on sys.path so that the
# measurement helpers under tests/ resolve.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.spyre._load_memory_helpers import (  # noqa: E402
    cast_dtype_name,
    checkpoint_dtype_name,
    format_table,
    measure_load,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-path",
        action="append",
        required=True,
        help="HF model id or local path; repeat for several models",
    )
    parser.add_argument(
        "--dtype",
        action="append",
        choices=("float16", "bfloat16"),
        help="load dtype; repeat for several (default: checkpoint dtype and its cast)",
    )
    parser.add_argument(
        "--path",
        action="append",
        choices=("staged", "direct"),
        help="load path; repeat for both (default: both)",
    )
    parser.add_argument(
        "--keep-read-ahead",
        action="store_true",
        help="also run each direct load with Transformers' read-ahead left on",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json", help="also write every measurement to this file")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    trust_remote_code = True if args.trust_remote_code else None
    paths = args.path or ["staged", "direct"]

    rows = []
    for model_path in args.model_path:
        dtypes = args.dtype
        if not dtypes:
            checkpoint = checkpoint_dtype_name(model_path, trust_remote_code)
            cast = cast_dtype_name(checkpoint)
            if cast is None:
                sys.exit(
                    f"{model_path}: checkpoint dtype is {checkpoint}; pass --dtype"
                )
            dtypes = [checkpoint, cast]

        runs = [(dtype, path, False) for dtype in dtypes for path in paths]
        if args.keep_read_ahead and "direct" in paths:
            runs += [(dtype, "direct", True) for dtype in dtypes]

        for dtype, path, keep_read_ahead in runs:
            print(f"loading {model_path} {dtype} {path}...", flush=True)
            rows.append(
                measure_load(
                    model_path,
                    direct=path == "direct",
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                    keep_read_ahead=keep_read_ahead,
                )
            )
            print(format_table(rows[-1:]).splitlines()[-1], flush=True)

    print()
    print(format_table(rows))
    if args.json:
        with open(args.json, "w") as out:
            json.dump([dataclasses.asdict(row) for row in rows], out, indent=2)
        print(f"\nwrote {len(rows)} measurements to {args.json}")


if __name__ == "__main__":
    main()
