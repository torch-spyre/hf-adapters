#!/usr/bin/env python3
"""Render one model's Spyre prefill report and setup timings as Markdown."""

import argparse
import math
import re
from pathlib import Path

_BLOCKS = ("full_model", "one_block")
_METRICS = (
    ("wall_clock_ms.mean_ms", "End-to-end wall clock"),
    ("spyre_ms.mean_ms", "Spyre device"),
    ("kernel_ms.mean_ms", "Kernel execution"),
    ("memory_transfer_ms.mean_ms", "Memory transfer"),
    ("runtime_launch_ms.mean_ms", "Runtime launch"),
    ("compile_ms.mean_ms", "Compilation during measurement"),
)
_MODEL_NAMES = {
    "ibm-granite/granite-4.1-8b": "Granite 4.1 8B",
    "mistralai/Ministral-3-14B-Instruct-2512": "Ministral 3 14B Instruct",
    "google/gemma-4-26B-A4B-it": "Gemma 4 26B-A4B",
}
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
Timings = dict[str, dict[str, float]]


def _milliseconds(value: str, scale: float = 1.0) -> float:
    result = float(value) * scale
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid timing value: {value}")
    return result


def parse_report(report: str) -> Timings:
    """Read the torch-spyre mean column from the two prefill report sections."""
    timings: Timings = {block: {} for block in _BLOCKS}
    keys = {key for key, _ in _METRICS}
    block: str | None = None
    for raw_line in _ANSI.sub("", report).splitlines():
        line = raw_line.strip()
        if line.startswith("Granite Config:"):
            match = re.search(r"_(full_model|one_block)(?:_|$)", line)
            block = match.group(1) if match else None
            continue
        fields = line.split()
        if block is not None and fields and fields[0] in keys:
            key = fields[0]
            if len(fields) < 2 or key in timings[block]:
                raise ValueError(f"Missing or duplicate report value: {block}/{key}")
            timings[block][key] = _milliseconds(fields[1])
    return timings


def parse_setup_timings(benchmark_log: str) -> Timings:
    """Convert loading seconds to ms and total only explicit warmup iterations."""
    timings: Timings = {block: {} for block in _BLOCKS}
    warmups: dict[str, list[float]] = {block: [] for block in _BLOCKS}
    expected_warmups: dict[str, int] = {}
    block: str | None = None
    in_warmup = False
    for raw_line in _ANSI.sub("", benchmark_log).splitlines():
        line = raw_line.strip()
        if line.startswith("Block:"):
            label = line.split(":", 1)[1].strip()
            block = "full_model" if label == "full_blocks" else label
            if block not in _BLOCKS:
                block = None
            in_warmup = False
            continue
        if block is None:
            continue
        load = re.fullmatch(r"Load time:\s*(\S+)s", line)
        start = re.fullmatch(r"Running (\d+) warmup iterations\.\.\.", line)
        timing = re.fullmatch(
            r"\[SPYRE-PERF-SUITE\] E2E timing information:\s*(\S+?)\s*ms", line
        )
        if load:
            if "load_ms" in timings[block]:
                raise ValueError(f"Duplicate load time: {block}")
            timings[block]["load_ms"] = _milliseconds(load.group(1), 1000)
        elif start:
            expected_warmups[block] = int(start.group(1))
            in_warmup = True
        elif line == "Warmup complete" or re.fullmatch(
            r"Running \d+ benchmark iterations\.\.\.", line
        ):
            in_warmup = False
        elif in_warmup and timing:
            warmups[block].append(_milliseconds(timing.group(1)))

    for block, expected in expected_warmups.items():
        if len(warmups[block]) != expected:
            raise ValueError(f"Incomplete warmup timings for {block}")
        timings[block]["warmup_ms"] = sum(warmups[block])
    return timings


def render_summary(model: str, report: str, benchmark_log: str) -> str:
    timings = parse_report(report)
    setup = parse_setup_timings(benchmark_log)
    rows = (
        *_METRICS,
        ("load_ms", "Model loading — excluded"),
        ("warmup_ms", "Warmup — excluded"),
    )
    for block in _BLOCKS:
        timings[block].update(setup[block])
        missing = [key for key, _ in rows if key not in timings[block]]
        if missing:
            raise ValueError(f"Missing timings for {block}: {', '.join(missing)}")

    lines = [
        f"### {_MODEL_NAMES.get(model, model)}",
        "",
        "All values are in **ms**. Benchmark timings are means over measured "
        "iterations; model loading and total warmup time are excluded.",
        "",
        "| Measure | Full model | One block |",
        "|---|---:|---:|",
    ]
    for key, label in rows:
        values = [f"{timings[block][key]:,.3f}" for block in _BLOCKS]
        if key == "load_ms":
            values = [value.rstrip("0").rstrip(".") for value in values]
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--summary-file", type=Path)
    args = parser.parse_args()
    try:
        summary = render_summary(
            args.model,
            args.report.read_text(encoding="utf-8"),
            args.log.read_text(encoding="utf-8"),
        )
        if args.summary_file is not None:
            with args.summary_file.open("a", encoding="utf-8") as output:
                output.write(summary + "\n")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(summary)


if __name__ == "__main__":
    main()
