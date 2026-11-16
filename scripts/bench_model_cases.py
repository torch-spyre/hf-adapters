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

"""Print the case names (comma separated) of the benchmarks that cover the given model paths.

    bench_model_cases.py MODEL_PATH...

A model with no declared benchmark is named on stderr and left out. Used by `make bench-perf`.
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_bench_definitions as gen  # noqa: E402


def cases_for(models: list[str]) -> tuple[list[str], list[str]]:
    perf = yaml.safe_load(gen.PERF_YAML.read_text())
    names = (yaml.safe_load(gen.BENCH_YAML.read_text()).get("benchmark_names")) or {}
    declared = set(gen._models(perf, "nightly") + gen._models(perf, "weekly"))
    cases, missing = [], []
    for path in dict.fromkeys(models):
        if path not in declared:
            missing.append(path)
            continue
        name = gen._name(path, names)
        cases += [gen._case(name, c, m) for c in gen.CONFIGS for m in gen.MODES]
    return cases, missing


def main(argv: list[str] | None = None) -> int:
    cases, missing = cases_for(list(sys.argv[1:] if argv is None else argv))
    for path in missing:
        print(f"bench: {path} has no benchmark definition", file=sys.stderr)
    print(",".join(cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
