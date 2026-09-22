#!/usr/bin/env python3
"""
parse_module_test_logs.py
--------------------------
Parse GitHub Actions log files for 'Spyre model-module tests' jobs and emit a
JSON structure mirroring the torch-spyre nightly model_ops schema.

Output file: <log_dir>/../module_tests_<run_id>.json   (inside hf-adapters root)

JSON schema per file:
{
  "run_id": "YYYYMMDD-<github_run_id>",
  "suite_name": "<model> Spyre Module Tests",
  "model_name": "<yaml stem>",
  "yaml_file":  "<yaml filename>",
  "source_log": "<log filename>",
  "github_run_id": "...", "github_sha": "...", "github_ref": "...",
  "runner_name": "...", "runner_node": "...", "test_date": "YYYY-MM-DD",
  "summary": {
    "total_tests": N,
    "spyre_enabled_count": N,      // XPASS, no cpu fallback
    "not_implemented_count": N,    // XFAIL, no cpu fallback
    "cpu_fallback_count": N,       // any test that emitted FallbackWarning
    "spyre_failed_count": N        // FAILED (unexpected)
  },
  "operations": {
    "spyre_enabled":   [...],
    "not_implemented": [...],
    "cpu_fallback":    [...],
    "spyre_failed":    [...]
  }
}

Each test record:
{
  "module":           "GraniteAttention",
  "test_label":       "test_with_cpu",
  "test_name":        "test_with_cpu_GraniteAttention_spyre_bfloat16",
  "test_file":        "test_modules_custom.py",
  "classification":   "not_implemented",
  "status":           "XFAIL",
  "duration_s":       0.0,
  "dtype":            "torch.bfloat16",
  "has_cpu_fallback": false,
  "cpu_fallback_ops": [],
  "inputs": {
    "prefill": {
      "args": { "arg_0": {"shape":[1,128,4096], "stride":[524288,4096,1], "dtype":"torch.bfloat16"} },
      "kwargs": {
        "hidden_states":      {"shape":[1,128,4096], "stride":[524288,4096,1], "dtype":"torch.bfloat16"},
        "position_ids":       {"shape":[1,128],      "stride":[128,1],        "dtype":"torch.int64"},
        "position_embeddings":[
          {"shape":[1,128,128], "stride":[16384,128,1], "dtype":"torch.bfloat16"},
          {"shape":[1,128,128], "stride":[16384,128,1], "dtype":"torch.bfloat16"}
        ]
      }
    },
    "decode": { ... }
  }
}

OOT test_forward inputs (shard 0):
  The OOT wrapper (test_modules__oot_wrapper.py) never emits [module-input] lines
  because PyTorch's upstream test harness does not call _log_forward_inputs().
  However the YAML config (tests/configs/module_tests/<model>.yaml) contains the
  complete forward_inputs for every module under
    test_suite_config.files[0].tests[0].edits.modules.include[*].forward_inputs
  Each forward_inputs entry corresponds to a variant (index 0 = prefill, 1 = decode).
  We load the YAML at parse time and back-fill inputs on every test_forward_* record.
"""

import ast
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml as _yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# ── regex ─────────────────────────────────────────────────────────
RE_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z ")
RE_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})T")
RE_ENV_KV = re.compile(r"^(\w+)=(.+)$")

# Test-start: path at the BEGINNING of the stripped line
# Matches: test_modules*.py::ClassXXX::test_<name>_spyre_<dtype>
#   followed by either " <- ..." (verbose -s mode) or end-of-line
RE_TEST_START = re.compile(
    r"^(test_modules(?:_custom)?(?:__oot_wrapper)?\.py)"
    r"::\w+::"
    r"(test_\w+)"  # full test name
    r"(?:\s+<-[^\[{]*|$)"  # optional " <- path/to/file.py" before any JSON
)

# [module-input] JSON blob – may appear mid-line after the test-start path
RE_MOD_INPUT = re.compile(r"\[module-input\]\s+(\{.+\})")

# Inline result: XPASS/XFAIL appearing anywhere on the line (with optional [Ns])
RE_INLINE_RESULT = re.compile(r"\b(XPASS|XFAIL|FAILED|PASSED)\s+\[?([\d.]+)s\]?")

# Standalone result: the ENTIRE stripped line is just the outcome ± duration
RE_STANDALONE = re.compile(r"^(XPASS|XFAIL|FAILED|PASSED)(?:\s+\[?([\d.]+)s\]?)?$")

# FallbackWarning op extraction
RE_FB_ATEN = re.compile(r"FallbackWarning:\s*(aten\.\S+)")
RE_FB_MSG = re.compile(
    r"FallbackWarning:\s*(.+?)(?:\s+is falling back to cpu|falling back to cpu)?$"
)

# Section headers that mark the END of live test output for the current shard
RE_SUMMARY_HDR = re.compile(
    r"^=+\s*(?:short test summary info|warnings summary|XPASSES)\s*=+"
)
# New pytest session starting – resets the summary-section gate
RE_SESSION_START = re.compile(r"=+\s*test session starts\s*=+")


def strip(line: str) -> str:
    return RE_TS.sub("", line).rstrip()


def _tensor(d: dict) -> dict:
    return {
        "shape": d.get("shape", []),
        "stride": d.get("stride", []),
        "dtype": d.get("dtype", ""),
    }


def build_inputs(raw: list[dict]) -> dict[str, dict]:
    """
    Convert a list of raw [module-input] dicts into:
      { "prefill": {"args": {...}, "kwargs": {...}},
        "decode":  {"args": {...}, "kwargs": {...}} }
    Shapes, strides and dtypes are preserved exactly as logged.
    """
    out: dict[str, dict] = {}
    for inp in raw:
        label = "prefill" if inp.get("variant", 0) == 0 else "decode"
        entry: dict[str, Any] = {"args": {}, "kwargs": {}}
        for arg in inp.get("args", []):
            entry["args"][f"arg_{arg.get('index', 0)}"] = _tensor(arg)
        for kname, kval in inp.get("kwargs", {}).items():
            if isinstance(kval, list):
                entry["kwargs"][kname] = [
                    _tensor(x) for x in kval if isinstance(x, dict)
                ]
            elif isinstance(kval, dict):
                entry["kwargs"][kname] = _tensor(kval)
        out[label] = entry
    return out


# ── YAML forward_inputs loader (OOT / test_forward inputs) ────────────────────


def _tensor_from_yaml(t: dict) -> dict:
    """Convert a YAML tensor spec to the same shape/stride/dtype dict as logged inputs.
    YAML uses 'null' stride (not yet allocated), so stride is omitted/empty for these.
    """
    return {
        "shape": t.get("shape", []),
        "stride": t.get("stride") or [],  # null in YAML → empty list
        "dtype": t.get("dtype", ""),
    }


def _yaml_kwarg(val: Any) -> Any:
    """Convert a YAML kwarg value to tensor-summary form.
    Handles: tensor spec dict, tensor_list (list of tensor specs), cache (skip), others.
    """
    if not isinstance(val, dict):
        return None
    if "tensor" in val:
        return _tensor_from_yaml(val["tensor"])
    if "tensor_list" in val:
        return [_tensor_from_yaml(t) for t in val["tensor_list"]]
    # cache / config_path / other complex types — not a plain tensor, skip
    return None


def _build_inputs_from_yaml_forward_input(fi: dict) -> dict[str, Any]:
    """Convert one forward_inputs entry from the YAML into args/kwargs input form."""
    entry: dict[str, Any] = {"args": {}, "kwargs": {}}

    for idx, arg_spec in enumerate(fi.get("args", []) or []):
        if not isinstance(arg_spec, dict):
            continue
        t = arg_spec.get("tensor")
        if t:
            entry["args"][f"arg_{idx}"] = _tensor_from_yaml(t)

    for kname, kval in (fi.get("kwargs") or {}).items():
        converted = _yaml_kwarg(kval)
        if converted is not None:
            entry["kwargs"][kname] = converted

    return entry


def load_yaml_forward_inputs(yaml_path: str) -> dict[str, dict[str, Any]]:
    """
    Load tests/configs/module_tests/<model>.yaml and return a dict:
      { "<base_module_name>": { "prefill": {...}, "decode": {...} }, ... }

    base_module_name is the class-name prefix before the first '_' suffix that
    the YAML uses as a disambiguator (e.g. 'GraniteAttention_layer0' → 'GraniteAttention').

    forward_inputs[0] → prefill (variant 0)
    forward_inputs[1] → decode  (variant 1)
    """
    if not _YAML_AVAILABLE:
        return {}
    if not os.path.isfile(yaml_path):
        return {}

    try:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            cfg = _yaml.safe_load(fh)
    except Exception:
        return {}

    result: dict[str, dict[str, Any]] = {}

    # Traverse test_suite_config.files[*].tests[*].edits.modules.include[*]
    for file_entry in (cfg or {}).get("test_suite_config", {}).get("files", []):
        for test_entry in file_entry.get("tests", []) or []:
            include = (test_entry.get("edits") or {}).get("modules", {}).get(
                "include"
            ) or []
            for mod_spec in include:
                full_name = mod_spec.get("name", "")
                # Derive the bare class name: 'GraniteAttention_layer0' → 'GraniteAttention'
                # Strategy: strip any trailing _<suffix> that looks like a disambiguator
                # (lower-case word, hex hash, or 'layer<N>').  The module_path gives the
                # canonical class name directly.
                module_path = mod_spec.get("module_path", "")
                base_name = module_path.rsplit(".", 1)[-1] if module_path else full_name

                forward_inputs = mod_spec.get("forward_inputs") or []
                inputs: dict[str, Any] = {}
                for variant_idx, fi in enumerate(forward_inputs):
                    variant_label = "prefill" if variant_idx == 0 else "decode"
                    inputs[variant_label] = _build_inputs_from_yaml_forward_input(fi)

                # Store under the canonical class name; don't overwrite if already set
                # (YAML anchor *id001 means the same include list is referenced by both
                # the oot-wrapper file and the custom file — identical data, no conflict).
                if base_name and base_name not in result:
                    result[base_name] = inputs

    return result


def classify(status: str, has_fb: bool) -> str:
    if status == "FAILED":
        return "spyre_failed"
    if has_fb:
        return "cpu_fallback"
    return "spyre_enabled" if status == "XPASS" else "not_implemented"


def parse_log_file(log_path: str, yaml_config_dir: str | None = None) -> dict | None:
    """Parse one GHA log file.

    yaml_config_dir: directory containing tests/configs/module_tests/*.yaml files
                     (i.e. the hf-adapters repo root).  When provided the OOT
                     test_forward tests are back-filled with forward_inputs from
                     the YAML so they carry the same prefill/decode shape data as
                     the custom-test entries.  Defaults to the parent of the
                     directory that contains this script.
    """
    fname = Path(log_path).name
    m = re.search(r"\((.+?\.yaml)\)", fname)
    if not m:
        return None
    yaml_file = m.group(1)
    model_name = yaml_file.replace(".yaml", "")
    log_num = (
        re.match(r"^(\d+)_", fname) or type("", (), {"group": lambda *_: "0"})()
    ).group(1)

    # ── Load YAML forward_inputs for OOT test_forward back-fill ──────────────
    if yaml_config_dir is None:
        # Default: <script_dir>/tests/configs/module_tests/<model>.yaml
        yaml_config_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_cfg_path = os.path.join(
        yaml_config_dir, "tests", "configs", "module_tests", yaml_file
    )
    yaml_fwd_inputs: dict[str, dict] = load_yaml_forward_inputs(yaml_cfg_path)

    # metadata
    env: dict[str, str] = {}
    test_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    date_set = False

    # parser state
    in_summary = False
    cur_test: str | None = None
    cur_file: str = ""
    cur_inputs: list[dict] = []
    cur_fb: list[str] = []
    completed: list[dict] = []
    seen: set[str] = set()

    def flush(status: str, dur: float) -> None:
        nonlocal cur_test, cur_inputs, cur_fb
        if cur_test is None:
            return
        tn = cur_test
        has_fb = bool(cur_fb)
        cl = classify(status, has_fb)
        tm = re.match(
            r"(test_forward|test_eager_vs_compile|test_layout_stride|test_with_cpu|test_\w+?)_(\w+?)_spyre_(\w+)$",
            tn,
        )
        label, module, dtype = (
            (tm.group(1), tm.group(2), f"torch.{tm.group(3)}") if tm else (tn, tn, "")
        )
        completed.append(
            {
                "module": module,
                "test_label": label,
                "test_name": tn,
                "test_file": cur_file,
                "classification": cl,
                "status": status,
                "duration_s": dur,
                "dtype": dtype,
                "has_cpu_fallback": has_fb,
                "cpu_fallback_ops": list(dict.fromkeys(cur_fb)),
                "inputs": build_inputs(cur_inputs),
            }
        )
        seen.add(tn)
        cur_test = None
        cur_inputs = []
        cur_fb = []

    def handle_module_input(line: str) -> None:
        """Extract all [module-input] Python-dict blobs from a line.
        The blobs use single-quoted Python repr style, not JSON — use ast.literal_eval.
        """
        for blob in RE_MOD_INPUT.finditer(line):
            try:
                cur_inputs.append(ast.literal_eval(blob.group(1)))
            except (ValueError, SyntaxError):
                pass

    # ── main scan ─────────────────────────────────────────────────
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            ln = strip(raw)

            if not date_set:
                dm = RE_DATE.match(raw)
                if dm:
                    test_date = dm.group(1)
                    date_set = True

            ekv = RE_ENV_KV.match(ln)
            if ekv:
                env[ekv.group(1)] = ekv.group(2)

            # ── session / summary boundary ────────────────────────
            if RE_SESSION_START.search(ln):
                in_summary = False
                continue
            if RE_SUMMARY_HDR.search(ln):
                in_summary = True
                continue
            if in_summary:
                continue

            # ── FallbackWarning ───────────────────────────────────
            if "FallbackWarning" in ln:
                fa = RE_FB_ATEN.search(ln)
                if fa:
                    cur_fb.append(fa.group(1))
                else:
                    fb = RE_FB_MSG.search(ln)
                    if fb:
                        cur_fb.append(fb.group(1).strip())
                continue

            # ── standalone result (only outcome on line) ──────────
            sr = RE_STANDALONE.match(ln)
            if sr and cur_test is not None:
                flush(sr.group(1), float(sr.group(2) or 0))
                continue

            # ── test-start line ───────────────────────────────────
            ts = RE_TEST_START.match(ln)
            if ts:
                tf = ts.group(1)  # file name
                full_test = ts.group(
                    2
                )  # e.g. test_with_cpu_GraniteAttention_spyre_bfloat16

                if full_test in seen:
                    # Already completed – line may still carry inline [module-input] for a different test;
                    # just ignore the whole line.
                    continue

                # Close any previous open test
                if cur_test is not None and cur_test != full_test:
                    flush("XFAIL", 0.0)

                cur_test = full_test
                cur_file = (
                    "test_modules_custom.py" if "custom" in tf else "test_modules.py"
                )

                # The SAME line may carry [module-input] blobs after " <- path.py"
                handle_module_input(ln)

                # The SAME line may also carry an inline result
                ir = RE_INLINE_RESULT.search(ln)
                if ir:
                    flush(ir.group(1), float(ir.group(2) or 0))
                continue

            # ── [module-input] on a standalone line ───────────────
            if "[module-input]" in ln and cur_test is not None:
                handle_module_input(ln)
                continue

    # Final dangling test
    if cur_test and cur_test not in seen:
        flush("XFAIL", 0.0)

    if not completed:
        return None

    # ── Back-fill YAML forward_inputs for test_forward (OOT shard 0) ─────────
    # test_forward tests come from the OOT wrapper which never emits [module-input]
    # lines.  Their shapes/dtypes ARE defined in the YAML config under forward_inputs.
    # We match by module class name extracted from the test name.
    if yaml_fwd_inputs:
        for rec in completed:
            if rec["test_label"] != "test_forward":
                continue
            if rec["inputs"]:  # already has inputs (shouldn't happen, but guard)
                continue
            module_cls = rec["module"]  # e.g. "GraniteAttention"
            if module_cls in yaml_fwd_inputs:
                rec["inputs"] = yaml_fwd_inputs[module_cls]

    # ── bucket ────────────────────────────────────────────────────
    spyre_enabled: list[dict] = []
    not_implemented: list[dict] = []
    cpu_fallback: list[dict] = []
    spyre_failed: list[dict] = []
    for rec in completed:
        cl = rec["classification"]
        if cl == "spyre_enabled":
            spyre_enabled.append(rec)
        elif cl == "not_implemented":
            not_implemented.append(rec)
        elif cl == "cpu_fallback":
            cpu_fallback.append(rec)
        else:
            spyre_failed.append(rec)

    gri = env.get("GITHUB_RUN_ID", "")
    run_id = f"{test_date.replace('-','')}-{gri or log_num}"

    return {
        "run_id": run_id,
        "suite_name": f"{model_name} Spyre Module Tests",
        "model_name": model_name,
        "yaml_file": yaml_file,
        "source_log": fname,
        "github_run_id": gri,
        "github_sha": env.get("GITHUB_SHA", ""),
        "github_ref": env.get("GITHUB_HEAD_REF") or env.get("GITHUB_REF", ""),
        "runner_name": env.get("RUNNER_NAME", ""),
        "runner_node": env.get("GHA_RUNNER_POD_NODE_NAME", ""),
        "test_date": test_date,
        "summary": {
            "total_tests": len(completed),
            "spyre_enabled_count": len(spyre_enabled),
            "not_implemented_count": len(not_implemented),
            "cpu_fallback_count": len(cpu_fallback),
            "spyre_failed_count": len(spyre_failed),
        },
        "operations": {
            "spyre_enabled": spyre_enabled,
            "not_implemented": not_implemented,
            "cpu_fallback": cpu_fallback,
            "spyre_failed": spyre_failed,
        },
    }


def main(log_dir: str, output_path: str) -> None:
    # Accept both the original GHA naming ("*_run-tests _ Spyre model-module tests*.txt")
    # and the sanitised names produced by the push-test-results-to-clickhouse workflow
    # ("*_Spyre_model-module_tests*.txt" / "*_Spyre_model*module*tests*.txt").
    patterns = [
        os.path.join(log_dir, "*_run-tests _ Spyre model-module tests*.txt"),
        os.path.join(log_dir, "*_Spyre_model*module*tests*.txt"),
        os.path.join(log_dir, "*_Spyre*model-module*tests*.txt"),
    ]
    log_files = sorted({f for p in patterns for f in glob.glob(p)})
    if not log_files:
        print(f"No model-module test logs found in: {log_dir}", file=sys.stderr)
        sys.exit(1)

    results = []
    for lf in log_files:
        print(f"Parsing: {os.path.basename(lf)}")
        res = parse_log_file(lf)
        if res:
            results.append(res)
            s = res["summary"]
            print(
                f"  → {res['model_name']:46s}  total={s['total_tests']:3d}  "
                f"xpass(spyre)={s['spyre_enabled_count']:3d}  "
                f"xfail(not-impl)={s['not_implemented_count']:3d}  "
                f"cpu_fallback={s['cpu_fallback_count']:3d}  "
                f"failed={s['spyre_failed_count']:3d}"
            )
        else:
            print("  → (skipped – no test results)")

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {len(results)} records → {output_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Parse Spyre model-module test GHA logs → JSON"
    )
    ap.add_argument(
        "log_dir",
        nargs="?",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "logs_88094510086"
        ),
        help="Directory containing GHA log .txt files",
    )
    ap.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON file (default: <hf-adapters>/module_tests_<run>.json)",
    )
    args = ap.parse_args()

    log_dir = os.path.abspath(args.log_dir)
    if not os.path.isdir(log_dir):
        print(f"ERROR: not a directory: {log_dir}", file=sys.stderr)
        sys.exit(1)

    dir_name = os.path.basename(log_dir)
    m = re.search(r"(\d+)", dir_name)
    run_suffix = m.group(1) if m else "unknown"

    # Default output is inside hf-adapters (parent of the log dir)
    hf_adapters_root = os.path.dirname(log_dir)
    output = args.output or os.path.join(
        hf_adapters_root, f"module_tests_{run_suffix}.json"
    )

    main(log_dir, output)
