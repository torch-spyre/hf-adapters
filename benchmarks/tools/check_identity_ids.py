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

"""Conformance check: the `identity` blocks of benchmarks/ derive the ids the ingest writes.

For every case the stored id from `spyre-bench ids` must equal the id `spyre_clickhouse_ingest` writes for the
matching row of a synthetic perf-suite report (parsed by the ingest's own parsers): the model-level row of a
benchmark report, and for the kernel breakdown a raw-named kernel, two compiled kernels (ranked) and a Total. The
ids are pinned in the golden table, so an edit that would start a new series fails here. The quality cases
(profile generic) are derived from the row `run_eval_case.py` records, through the ingest's generic bundle code.

    python benchmarks/tools/check_identity_ids.py [--update]

Needs pyyaml, regex, lxml, clickhouse-connect and the spyre-bench package (with the ingest); exits 2 naming the
missing module (0 with a notice when BENCH_OPTIONAL=1). Kernel names and the op names of the synthetic reports are
fixtures; a benchmark named `granite` is fed the `granite_` op spelling the ingest folds into that name.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "benchmarks"
GOLDEN = CATALOG / "ingest-ids.golden.tsv"
HEADER = "component\tname\trow\tconfig_name\trun_mode\tkernel_name\tbenchmark_id"
COMPONENT = "hf-adapters"
RAW_KERNEL = "aten::copy_"
COMPILED = (
    ("spyre_kernel_v1_fused_add_aaaaaaaaaaaaaaaa#2", 0.003),
    ("spyre_kernel_v1_fused_add_bbbbbbbbbbbbbbbb#2", 0.27),
)
SUITE = '<testsuites><testsuite name="spyre-perf-suite" timestamp="2026-01-01T00:00:00Z">{}</testsuite></testsuites>'


def shared_catalog():
    """The engine's own catalog directory, which holds the Environments this one refers to.

    Without it every Environment reference dangles and `ids` reports one error per case, so it is
    passed as --central rather than letting the real cause hide behind 164 identical lines.
    """
    out = subprocess.run(
        [sys.executable, "-m", "spyre_bench", "catalogs", "--path", "llm"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() if out.returncode == 0 else ""


def engine(*args, catalog=CATALOG):
    central = shared_catalog()
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "spyre_bench",
            *args,
            "--catalog",
            str(catalog),
            *(["--central", central] if central else []),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode:
        hint = (
            ""
            if central
            else "\nnote: this engine ships no shared catalog (spyre_bench/catalogs/llm), so the "
            "Environments this catalog names cannot resolve; install a newer spyre-bench"
        )
        raise RuntimeError(
            f"spyre-bench {args[0]} failed:\n{out.stdout}{out.stderr}{hint}"
        )
    return json.loads(out.stdout)


def declared(catalog=CATALOG):
    """Per case: the identity inputs and the engine's stored id."""
    stored = {r["case"]: r["benchmark_id"] for r in engine("ids", catalog=catalog)}
    inputs = {
        r["case"]: r["inputs"] for r in engine("ids", "--inputs", catalog=catalog)
    }
    return [(case, inputs[case], stored[case]) for case in stored]


def tag_props(**tags):
    props = "".join(f'<property name="tag" value="{k}__{v}"/>' for k, v in tags.items())
    return f"<properties>{props}</properties>"


PHASE = {"prefill_only": "prefill", "decode_only": "decode", "generate": "full"}


def op_name(name, bare, config, mode):
    """The op spelling the ingest turns into this benchmark name: `granite_` ops fold into `granite`, every
    other model op takes its `model` tag (the ingest's tests/test_model_identity.py spellings).
    """
    if name == "granite":
        return f"granite_bs1_pl512_{PHASE[mode]}_{config}"
    return f"{bare}_{PHASE[mode]}-mode_{config}"


def report_xml(entries):
    """Benchmark report: one wall-clock testcase per (benchmark, config, mode), tagged as the model rows are."""
    cases = []
    for name, bare, config, mode in entries:
        op = op_name(name, bare, config, mode)
        props = tag_props(op=op, model=bare, config=config, mode=mode)
        cases.append(
            f'<testcase name="perf_{op}_wall_clock_ms" time="1.0">{props}</testcase>'
        )
    return '<?xml version="1.0"?>' + SUITE.format("".join(cases))


def kernel_xml(entries):
    """Kernel breakdown: raw kernel, two compiled kernels of one stem, a Total, per (benchmark, config, mode)."""
    cases = []
    for name, bare, config, mode in entries:
        op = op_name(name, bare, config, mode)
        for kernel, ms in ((RAW_KERNEL, 0.5), *COMPILED, ("Total", 1.0)):
            props = tag_props(
                op=op,
                model=bare,
                config=config,
                mode=mode,
                kernel=kernel,
                metric="spyre_kernel_ms",
            )
            cases.append(
                f'<testcase classname="kernel_benchmark" name="k" time="{ms}">{props}</testcase>'
            )
    return '<?xml version="1.0"?>' + SUITE.format("".join(cases))


def load_ingest():
    from spyre_clickhouse_ingest import results
    from spyre_clickhouse_ingest.identity import BenchmarkId

    return results, BenchmarkId


def parse(results, text):
    with tempfile.TemporaryDirectory() as d:
        xml = Path(d) / "report.xml"
        xml.write_text(text)
        if "kernel_benchmark" in text:
            _, records = results.parse_kernel_xml(
                xml, "bench-conformance", "", "x86_64"
            )
        else:
            _, records = results.parse_benchmark_xml(
                xml, "bench-conformance", "", "x86_64"
            )
    return results._bench_entries(records)


def lint(catalog=CATALOG):
    """Problems with the declared cost and identity shape: est_hours is the only cost hint."""
    import yaml

    problems = []
    for path in sorted(Path(catalog).rglob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text()):
            if not doc:
                continue
            spec = doc.get("spec") or {}
            ann = (doc.get("metadata") or {}).get("annotations") or {}
            if "schedule" in spec or any(k.startswith("schedule.") for k in ann):
                problems.append(
                    f"{path.name}: cadence belongs to the invoker, not the definitions"
                )
            if doc.get("kind") == "Benchmark":
                ident, name = spec.get("identity") or {}, doc["metadata"]["name"]
                profile = "perf-kernel" if spec.get("type") == "perf" else "generic"
                if ident.get("mode") != "ingest" or ident.get("profile") != profile:
                    problems.append(
                        f"{path.name}: {name} needs identity mode ingest, profile {profile}"
                    )
                if "est_hours" not in spec:
                    problems.append(f"{path.name}: {name} has no est_hours")
    return problems


def asset_problems(catalog=CATALOG):
    """Vendored question files whose sha256 differs from the Asset that names them."""
    import hashlib

    import yaml

    problems = []
    for doc in yaml.safe_load_all((Path(catalog) / "assets-eval.yaml").read_text()):
        spec = (doc or {}).get("spec") or {}
        src = spec.get("source") or {}
        for f in (spec.get("integrity") or {}).get("files") or []:
            path = ROOT / src.get("path", "") / f["path"]
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != f["sha256"]
            ):
                problems.append(
                    f"{doc['metadata']['name']}: {path.relative_to(ROOT)} differs from its declared sha256"
                )
    return problems


def bare_models(catalog=CATALOG):
    """{benchmark name: bare model name} as the perf suite names it (last path segment, case kept)."""
    import yaml

    out = {}
    for path in sorted((Path(catalog) / "hf-models").glob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text()):
            if doc and doc.get("kind") == "Benchmark":
                spec = doc["spec"]
                out[spec["identity"]["name"]] = spec["workload"]["model"].rsplit(
                    "/", 1
                )[-1]
    return out


def table(results, bid, cases, bare):
    """(rows, mismatches): golden rows and every case whose definition disagrees with the ingest parser."""
    keys = results._V2_BENCH_ID_KEYS
    seen = {}
    generic = [c for c in cases if c[1]["profile"] == "generic"]
    cases = [c for c in cases if c[1]["profile"] != "generic"]
    for _, inp, _ in cases:
        d = inp["disc"]
        seen[(inp["name"], bare[inp["name"]], d["config_name"], d["run_mode"])] = None
    entries = list(seen)
    written = {
        (
            e["name"].lower(),
            e["disc"]["config_name"],
            e["disc"]["run_mode"],
        ): bid.derive(COMPONENT, e["name"], e["tags"], e["disc"], e["disc_keys"])
        for e in parse(results, report_xml(entries))
    }
    kernels = {}
    for e in bid.rank_kernels(COMPONENT, parse(results, kernel_xml(entries))):
        d = e["disc"]
        kernels.setdefault(
            (e["name"].lower(), d["config_name"], d["run_mode"]), []
        ).append(
            (
                d["kernel_name"],
                bid.derive(COMPONENT, e["name"], e["tags"], d, e["disc_keys"]),
            )
        )
    rows, bad = [], []
    for case, inp, stored in cases:
        name, d = inp["name"], inp["disc"]
        key = (name, d["config_name"], d["run_mode"])
        if inp["component"] != COMPONENT or list(inp["disc_keys"]) != list(keys):
            bad.append(
                f"{case}: component/keys {inp['component']}/{inp['disc_keys']} differ from the ingest's"
            )
        if written.get(key) != stored:
            bad.append(
                f"{case}: definition derives {stored}, the ingest writes {written.get(key)}"
            )
        rows.append(
            (COMPONENT, name, "report", key[1], key[2], "", written.get(key) or "")
        )
        for kernel, kid in kernels.get(key, []):
            kind = "total" if kernel == "Total" else "kernel"
            rows.append((COMPONENT, name, kind, key[1], key[2], kernel, kid))
    quality_rows(bid, generic, rows, bad)
    return rows, bad


def quality_rows(bid, cases, rows, bad):
    """Quality cases: the id of the row the recorder writes, through the ingest's generic bundle code."""
    from spyre_clickhouse_ingest import bundle

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_eval_case

    for case, inp, stored in cases:
        d = inp["disc"]
        _, doc = run_eval_case.build(
            inp["name"], d["task"], d["model"], 0, int(d["num_questions"]), ""
        )
        entry = bundle.generic_entries(doc["benchmarks"])[0]
        written = bid.derive(
            COMPONENT, entry["name"], entry["tags"], entry["disc"], entry["disc_keys"]
        )
        if written != stored:
            bad.append(
                f"{case}: definition derives {stored}, the recorder writes {written}"
            )
        rows.append((COMPONENT, inp["name"], "quality", "", "", "", written))


def render(rows):
    return "\n".join([HEADER, *("\t".join(r) for r in rows)]) + "\n"


def check(catalog=CATALOG, golden=GOLDEN, update=False):
    """Return (exit_code, messages)."""
    results, BenchmarkId = load_ingest()
    try:
        cases = declared(catalog)
    except RuntimeError as e:
        return 1, [str(e)]
    msgs = lint(catalog) + asset_problems(catalog)
    rows, bad = table(results, BenchmarkId, cases, bare_models(catalog))
    msgs += bad
    text = render(rows)
    if update and not msgs:
        Path(golden).write_text(text)
        return 0, [f"wrote {golden}: {len(rows)} ids"]
    old = {}
    if Path(golden).exists():
        old = {
            tuple(ln.split("\t")[:6]): ln.split("\t")[6]
            for ln in Path(golden).read_text().splitlines()[1:]
        }
    for r in rows:
        if old.get(r[:6]) != r[6]:
            msgs.append(
                f"{r[1]} {r[3]} {r[4]} {r[2]} {r[5]}: derived {r[6]}, golden {old.get(r[:6])}"
            )
    if len(old) != len(rows):
        msgs.append(f"golden has {len(old)} rows, the catalog derives {len(rows)}")
    if msgs:
        return 1, msgs
    return 0, [
        f"PASS identity conformance: {len(cases)} cases, {len(rows)} ids match {Path(golden).name}"
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--update",
        action="store_true",
        help="rewrite the golden table (an identity change)",
    )
    args = ap.parse_args(argv)
    try:
        code, msgs = check(update=args.update)
    except ImportError as e:
        print(
            f"notice: identity conformance skipped, missing module: {e.name}",
            file=sys.stderr,
        )
        return 0 if os.environ.get("BENCH_OPTIONAL") == "1" else 2
    for m in msgs:
        print(m, file=sys.stderr if code else sys.stdout)
    if code:
        print(
            "identity or definition problems: review; an intended id change reruns with --update",
            file=sys.stderr,
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
