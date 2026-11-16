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

"""Run command of a model case: this case's rows of the model's perf report, in the layout the ingest reads.

The report comes from the repo's own perf entry point (`make perf-tests`, or BENCH_SUITE_CMD with {suite_dir} for the
target directory; the engine hands a command only the variables named in BENCH_PASS_ENV) and covers every
config and mode of the model, so it runs once per --suite-dir. Reports of an earlier `perf-tests` run are reused as
is: those already in --suite-dir, or in BENCH_PERF_RESULTS (a side-by-side run reads the legacy results and
measures nothing twice). Each case keeps only the testcases of its benchmark name, config and mode, with the name
the ingest derives (`granite_` ops fold into `granite`, every other op takes its `model` tag), so the ingest writes
exactly one row per case and the cases cannot double-count, whether the report is per model or combined.

    run_model_case.py --model PATH --name NAME --config CONFIG --mode MODE --suite-dir DIR --raw-dir DIR
        [--repeat-dir DIR]

With --repeat-dir the case's wall-clock time is also written as `metrics.json` (`kernel.total_duration`, seconds),
the engine's own sample for a native `bench` bundle; the XML stays the legacy-native input of the unchanged ingest.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DONE = ".suite-done"
PERF_NAME = re.compile(
    r"^perf_(?P<op>.+?)_(?:wall_clock|cpu|spyre|kernel|memory_transfer|runtime|compiler?|mem_size)_(?:ms|MB)(?:_.+)?$"
)


def slug(model: str) -> str:
    return model.replace("/", "_")


def tags(testcase) -> dict[str, str]:
    """The `tag` properties (`key__value`) of a testcase, as the ingest reads them."""
    out = {}
    for prop in testcase.iter("property"):
        value = prop.get("value", "").strip()
        if prop.get("name", "").strip() == "tag" and "__" in value:
            key, _, val = value.partition("__")
            out[key] = val
    return out


def ingest_name(testcase, t: dict[str, str]) -> str:
    """The benchmark name the ingest writes for a model testcase ('' when it is not a model row)."""
    op = t.get("op") or (PERF_NAME.match(testcase.get("name", "")) or [None, ""])[1]
    if op.startswith("granite_"):
        return "granite"
    return t.get("model", "").strip().lower()


def reports(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.xml"))


def suite_reports(suite_dir: Path, model: str) -> list[Path]:
    reuse = os.environ.get("BENCH_PERF_RESULTS")
    found = reports(Path(reuse) if reuse else suite_dir)
    if reuse:
        if found:
            return found
        raise SystemExit(f"BENCH_PERF_RESULTS={reuse} holds no report XML")
    done = suite_dir / DONE
    if found and done.is_file():
        return found
    if found:
        raise SystemExit(
            f"{suite_dir} holds reports of a suite run that did not finish; remove it and rerun"
        )
    suite_dir.mkdir(parents=True, exist_ok=True)
    custom = os.environ.get("BENCH_SUITE_CMD", "").replace(
        "{suite_dir}", str(suite_dir)
    )
    cmd = shlex.split(custom) or [
        "make", "-C", str(ROOT), "perf-tests", f"MODEL_PATH={model}", f"RESULTS_DIR={suite_dir}", "BENCH_MODE=off",
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    found = reports(suite_dir)
    if not found:
        raise SystemExit(
            f"the perf suite wrote no report XML for {model} in {suite_dir}"
        )
    done.write_text("")
    return found


def filtered(report: Path, name: str, config: str, mode: str):
    tree = ET.parse(report)
    kept = 0
    for suite in tree.getroot().iter("testsuite"):
        for tc in list(suite.findall("testcase")):
            t = tags(tc)
            if (
                t.get("config") == config
                and t.get("mode") == mode
                and ingest_name(tc, t) == name.lower()
            ):
                kept += 1
            else:
                suite.remove(tc)
    return tree, kept


def wall_clock_seconds(raw_dir: Path) -> float | None:
    """The case's wall-clock row of the benchmark report (its `time` is in milliseconds), in seconds."""
    for report in sorted(raw_dir.glob("*.xml")):
        for tc in ET.parse(report).getroot().iter("testcase"):
            if PERF_NAME.match(tc.get("name", "")) and "_wall_clock_ms" in tc.get(
                "name", ""
            ):
                return float(tc.get("time", 0)) / 1000.0
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="the model path of perf.yaml")
    ap.add_argument(
        "--name",
        required=True,
        help="the benchmark name the ingest derives for the model",
    )
    ap.add_argument(
        "--config", required=True, help="full_model or one_block, as the report tags it"
    )
    ap.add_argument(
        "--mode", required=True, help="prefill_only, decode_only or generate"
    )
    ap.add_argument("--suite-dir", required=True, type=Path)
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--repeat-dir", type=Path)
    args = ap.parse_args(argv)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for report in suite_reports(args.suite_dir / slug(args.model), args.model):
        tree, kept = filtered(report, args.name, args.config, args.mode)
        if kept:
            tree.write(
                args.raw_dir / report.name, encoding="utf-8", xml_declaration=True
            )
            total += kept
    if not total:
        print(
            f"no testcase for {args.name} config={args.config} mode={args.mode} in the perf report",
            file=sys.stderr,
        )
        return 3
    seconds = wall_clock_seconds(args.raw_dir) if args.repeat_dir else None
    if seconds is not None:
        args.repeat_dir.mkdir(parents=True, exist_ok=True)
        (args.repeat_dir / "metrics.json").write_text(
            json.dumps({"samples": {"kernel.total_duration": [seconds]}})
        )
    print(f"{args.name} {args.config} {args.mode}: kept {total} testcases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
