"""Compare the weekly scan on a fixed CSV ranking with pretrained/random weights.

Run on a Spyre host with the repository root on PYTHONPATH. Results, input lists,
logs and a JSON comparison are retained under --output-dir. Oversized entries
are reported as skips, using the same ceiling as the weekly workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

from tests.spyre.weekly_generation.failure_categories import MAX_NUMBER_PARAMS
from tests.spyre.weekly_generation.model_prefilter import prefilter_models

ROOT = Path(__file__).resolve().parents[3]


def prepare_list(source: Path, limit: int, output: Path) -> list[dict[str, Any]]:
    """Freeze the first ranked rows, normalize CSV flags, and apply the size cap."""
    with source.open(newline="") as handle:
        rows: list[dict[str, Any]] = list(csv.DictReader(handle))[:limit]
    for row in rows:
        row["curated"] = False
        for key in ("is_moe", "is_gated", "is_custom_code", "is_multimodal"):
            row[key] = str(row.get(key, "")).lower() == "true"
    partition = prefilter_models(rows, MAX_NUMBER_PARAMS)
    if output.exists() and json.loads(output.read_text()) != partition.keep:
        raise ValueError(
            f"{output} does not match this catalog; choose a new output directory"
        )
    output.write_text(json.dumps(partition.keep, indent=2) + "\n")
    skipped = [
        {"model_name": item.row["model_id"], "failure_category": item.failure_category}
        for item in partition.skipped
    ]
    output.with_suffix(".skipped.json").write_text(json.dumps(skipped, indent=2) + "\n")
    return rows


def summarize(
    output_dir: Path, modes: tuple[str, ...] = ("embedding", "generative")
) -> dict[str, Any]:
    """Compare verdicts, retaining skips and missing runs rather than treating them as passes."""
    comparisons: list[dict[str, Any]] = []
    fields = ("verified_on_cpu", "verified_on_spyre", "failure_category")
    for mode in modes:
        results: dict[str, dict[str, dict[str, str]]] = {}
        for weights in ("pretrained", "random"):
            path = output_dir / f"{weights}-{mode}.csv"
            with path.open(newline="") as handle:
                results[weights] = {
                    row["model_name"]: row for row in csv.DictReader(handle)
                }
        rows: list[dict[str, Any]] = json.loads(
            (output_dir / f"{mode}.json").read_text()
        )
        for row in rows:
            name = str(row["model_id"])
            pretrained = results["pretrained"].get(name, {})
            random = results["random"].get(name, {})
            comparisons.append(
                {
                    "mode": mode,
                    "model_name": name,
                    "completed_pair": bool(pretrained and random),
                    "same_verdict": bool(pretrained and random)
                    and all(pretrained.get(key) == random.get(key) for key in fields),
                    "pretrained": {
                        key: pretrained.get(key) for key in (*fields, "error")
                    },
                    "random": {key: random.get(key) for key in (*fields, "error")},
                }
            )
        for skipped in json.loads((output_dir / f"{mode}.skipped.json").read_text()):
            comparisons.append({"mode": mode, **skipped, "skipped_both": True})
    report: dict[str, Any] = {
        "compared": sum(row.get("completed_pair", False) for row in comparisons),
        "pending": sum(
            "completed_pair" in row and not row["completed_pair"] for row in comparisons
        ),
        "same_verdict": sum(row.get("same_verdict", False) for row in comparisons),
        "skipped_both": sum(row.get("skipped_both", False) for row in comparisons),
        "models": comparisons,
    }
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--mode", choices=("embedding", "generative", "all"), default="all"
    )
    parser.add_argument(
        "--weights", choices=("pretrained", "random", "both"), default="both"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--snapshot-date", type=date.fromisoformat, default=date.today()
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    modes: tuple[str, ...] = (
        ("embedding", "generative") if args.mode == "all" else (args.mode,)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        for mode in modes:
            model_list: Path = args.output_dir / f"{mode}.json"
            prepare_list(
                ROOT / "resources" / f"top_{mode}_models_DEPRECATED.csv",
                args.top_k,
                model_list,
            )
            weight_modes = (
                ("pretrained", "random") if args.weights == "both" else (args.weights,)
            )
            for weights in weight_modes:
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "tests.spyre.weekly_generation.weekly_test",
                    "--mode",
                    mode,
                    "--model-list-file",
                    str(model_list),
                    "--snapshot-date",
                    str(args.snapshot_date),
                    "--models-per-process",
                    "1",
                    "--write-to-csv",
                    str(args.output_dir / f"{weights}.csv"),
                    "--seed",
                    str(args.seed),
                ]
                if weights == "random":
                    command.append("--random-weights")
                print(f"Running {mode} with {weights} weights", flush=True)
                with (args.output_dir / f"{weights}-{mode}.log").open("w") as log:
                    subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        cwd=ROOT,
                    )
    if all(
        (args.output_dir / f"{weights}-{mode}.csv").exists()
        for weights in ("pretrained", "random")
        for mode in modes
    ):
        report = summarize(args.output_dir, modes)
        print(
            f"Matching verdicts: {report['same_verdict']}/{report['compared']}; "
            f"pending pairs: {report['pending']}; "
            f"skipped in both: {report['skipped_both']}"
        )


if __name__ == "__main__":
    main()
