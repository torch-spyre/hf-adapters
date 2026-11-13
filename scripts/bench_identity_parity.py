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

"""Print, per benchmark case, the id the existing model-benchmark rows use and the id
the engine derives from the definition.

The existing id hashes (component, name, no tags) plus six discriminators; a model row
sets record_type, config_name and run_mode and leaves the other three empty. The engine hashes every flattened case/system attribute, so the
two differ until the engine can pin a legacy discriminator set.

    python scripts/bench_identity_parity.py [--known FILE]

FILE is tab-separated ``name  config_name  run_mode  benchmark_id`` of ids to compare
with. Needs the spyre-bench package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spyre_bench import engine

CATALOG = Path(__file__).resolve().parent.parent / "benchmarks"
LEGACY_KEYS = (
    "record_type",
    "config_name",
    "input_shapes",
    "run_mode",
    "kernel_name",
    "is_total",
)


def legacy_id(component: str, name: str, config: str, mode: str) -> str:
    disc = {"record_type": "model", "config_name": config, "run_mode": mode}
    return engine.benchmark_id(component, name, [], disc, LEGACY_KEYS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--known", help="TSV of ids to compare with")
    args = ap.parse_args(argv)
    cat = engine.Catalog([str(CATALOG)], [], [])
    if cat.errors:
        print("\n".join(cat.errors), file=sys.stderr)
        return 2
    known: dict[tuple[str, str, str], str] = {}
    if args.known:
        for line in Path(args.known).read_text().splitlines():
            name, config, mode, bid = line.split("\t")
            known[(name.lower(), config, mode)] = bid
    print("name\tconfig_name\trun_mode\tlegacy_id\tengine_id\tknown")
    for case in cat.cases:
        ident = cat.identity(case)
        disc = ident["disc"]
        config, mode = disc["case.layer"], disc["case.flags.run_mode"]
        old = legacy_id(ident["component"], ident["name"], config, mode)
        new = cat.ids(case)["benchmark_id"]
        status = "-"
        if known:
            want = known.get((ident["name"], config, mode))
            status = "no-row" if want is None else ("match" if want == old else "DIFF")
        print(f"{ident['name']}\t{config}\t{mode}\t{old}\t{new}\t{status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
