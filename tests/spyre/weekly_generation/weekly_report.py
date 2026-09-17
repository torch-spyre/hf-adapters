"""Week-over-week textual comparison report for the Spyre weekly scan.

Queries ClickHouse for two consecutive snapshot_dates and prints a structured
plain-text report covering the six analysis areas defined in issue #484:

  1. Coverage (model count changes)
  2. verified_on_spyre delta
  3. failure_category distribution shift
  4. Error pattern analysis
  5. Adapter / config_class coverage changes
  6. Size / family breakdown

Usage::

    # Compare the two most recent snapshots automatically
    python tests/spyre/weekly_generation/weekly_report.py --mode generative
    python tests/spyre/weekly_generation/weekly_report.py --mode embedding

    # Compare specific dates
    python tests/spyre/weekly_generation/weekly_report.py \\
        --mode generative --prev 2025-06-21 --curr 2025-06-28

Credentials are read from .env (CLICKHOUSE_HOST / CLICKHOUSE_PASS etc.) exactly
as the rest of the weekly pipeline does — see clickhouse_db.py.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from tests.spyre.weekly_generation.clickhouse_db import get_client
from tests.spyre.weekly_generation.failure_categories import (
    FAILURE_CATEGORY_HARDWARE_EXCEPTION,
    FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER,
    FAILURE_CATEGORY_WORKER_CRASHED,
    FAILURE_CATEGORY_WORKER_TIMEOUT,
)
from tests.spyre.weekly_generation.model_type import ModelType
from tests.spyre.weekly_generation.table_schema import (
    DATABASE,
    EMBEDDING_TABLE_NAME,
    GENERATIVE_TABLE_NAME,
)

# ---------------------------------------------------------------------------
# Path bootstrap (mirrors weekly_test.py)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (
    _REPO_ROOT / "tests" / "spyre",
    _REPO_ROOT / "tests",
    _REPO_ROOT,
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# # Categories that are deterministic pre-filter verdicts (never reflect runtime).
# _PREFILTER_CATEGORIES: frozenset[str] = frozenset(
#     {
#         FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER,
#         FAILURE_CATEGORY_MODEL_TOO_LARGE,
#         FAILURE_CATEGORY_MOE,
#     }
# )

# Categories that reflect infrastructure problems, not model quality.
_INFRA_CATEGORIES: frozenset[str] = frozenset(
    {
        FAILURE_CATEGORY_HARDWARE_EXCEPTION,
        FAILURE_CATEGORY_WORKER_CRASHED,
        FAILURE_CATEGORY_WORKER_TIMEOUT,
    }
)

# When clustering error messages, show up to this many distinct examples.
_MAX_ERROR_EXAMPLES: int = 5

# When listing changed models, show up to this many per bucket.
_MAX_MODELS_SHOWN: int = 8


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def _fetch_snapshot(
    client: Any, table: str, snapshot_date: date
) -> list[dict[str, Any]]:
    """Return all rows for *snapshot_date* as a list of dicts."""
    cols: list[str] = [
        "model_name",
        "config_class",
        "adapter_name",
        "added_date",
        "verified_on_cpu",
        "verified_on_spyre",
        "num_downloads",
        "family",
        "architecture",
        "parameters_number",
        "failure_category",
        "error",
        "curated",
    ]
    result = client.query(
        f"SELECT {', '.join(cols)} "
        "FROM {db:Identifier}.{tbl:Identifier} "
        "WHERE snapshot_date = {d:Date} "
        "ORDER BY num_downloads DESC",
        parameters={"db": DATABASE, "tbl": table, "d": snapshot_date},
    )
    return [dict(zip(cols, row)) for row in result.result_rows]


def _two_latest_dates(client: Any, table: str) -> tuple[date, date]:
    """Return the two most recent distinct snapshot_dates from *table*.

    Raises ValueError when fewer than two snapshots exist.
    """
    result = client.query(
        "SELECT DISTINCT snapshot_date "
        "FROM {db:Identifier}.{tbl:Identifier} "
        "ORDER BY snapshot_date DESC "
        "LIMIT 2",
        parameters={"db": DATABASE, "tbl": table},
    )
    dates = [row[0] for row in result.result_rows]
    if len(dates) < 2:
        raise ValueError(
            f"Table {DATABASE}.{table} has fewer than 2 distinct snapshot_dates "
            f"(found: {dates}). Pass --prev and --curr explicitly."
        )
    return dates[1], dates[0]  # older, newer


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _delta(curr: int, prev: int) -> str:
    """Format an integer delta as '+N', '-N', or '0'."""
    d = curr - prev
    if d > 0:
        return f"+{d}"
    if d < 0:
        return str(d)
    return "0"


def _is_infra_failure(row: dict[str, Any]) -> bool:
    """True when the row's verdict is an infrastructure failure, not a model one.

    A ``verified_on_spyre = False`` caused by ``hardware_exception`` /
    ``worker_crashed`` / ``worker_timeout`` says nothing about the model, so it
    must not be counted as a regression (nor its later disappearance as a
    recovery).
    """
    return (row.get("failure_category") or "") in _INFRA_CATEGORIES


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator * 100:.1f}%"


def _trunc(models: list[str], limit: int = _MAX_MODELS_SHOWN) -> str:
    shown = models[:limit]
    suffix = f"  … and {len(models) - limit} more" if len(models) > limit else ""
    return "\n".join(f"    • {m}" for m in shown) + suffix


def _format_models_with_downloads(
    models: list[tuple[str, int]], limit: int = _MAX_MODELS_SHOWN
) -> str:
    """Render ``(model_name, num_downloads)`` pairs as an aligned two-column list.

    Model names are left-padded to the widest shown name so the download counts
    (right-aligned, thousands-separated) line up in a monospace terminal.
    """
    shown = models[:limit]
    name_width: int = max(len(name) for name, _ in shown)
    dl_width: int = max(len(f"{dl:,}") for _, dl in shown)
    body: str = "\n".join(
        f"    • {name:<{name_width}}  {dl:>{dl_width},} downloads" for name, dl in shown
    )
    suffix = f"\n  … and {len(models) - limit} more" if len(models) > limit else ""
    return body + suffix


def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def _section_coverage(
    prev_rows: list[dict[str, Any]],
    curr_rows: list[dict[str, Any]],
    prev_date: date,
    curr_date: date,
) -> str:
    prev_models = {r["model_name"] for r in prev_rows}
    curr_models = {r["model_name"] for r in curr_rows}

    added = sorted(curr_models - prev_models)

    # New models, most-downloaded first, annotated with their download count.
    curr_by_name: dict[str, dict[str, Any]] = {r["model_name"]: r for r in curr_rows}
    added_by_downloads: list[tuple[str, int]] = sorted(
        ((m, curr_by_name[m].get("num_downloads") or 0) for m in added),
        key=lambda pair: pair[1],
        reverse=True,
    )

    prev_curated = sum(1 for r in prev_rows if r["curated"])
    curr_curated = sum(1 for r in curr_rows if r["curated"])

    lines = [
        _hr("═"),
        "1. COVERAGE — MODEL COUNT",
        _hr(),
        f"  Total models   : {len(prev_models):>6}  →  {len(curr_models):>6}  ({_delta(len(curr_models), len(prev_models))})",
        f"  Curated models : {prev_curated:>6}  →  {curr_curated:>6}  ({_delta(curr_curated, prev_curated)})",
        "",
        f"  New models this week ({len(added)}):",
    ]
    if added_by_downloads:
        lines.append(_format_models_with_downloads(added_by_downloads))

    return "\n".join(lines)


def _section_verified_on_spyre(
    prev_rows: list[dict[str, Any]],
    curr_rows: list[dict[str, Any]],
) -> str:
    prev_by = {r["model_name"]: r for r in prev_rows}
    curr_by = {r["model_name"]: r for r in curr_rows}
    common = set(prev_by) & set(curr_by)

    prev_pass = sum(1 for r in prev_rows if r["verified_on_spyre"])
    curr_pass = sum(1 for r in curr_rows if r["verified_on_spyre"])

    # Exclude infrastructure-failures rows from the denominator (they never reach Spyre)
    prev_eligible = [
        r for r in prev_rows if r.get("failure_category") not in _INFRA_CATEGORIES
    ]
    curr_eligible = [
        r for r in curr_rows if r.get("failure_category") not in _INFRA_CATEGORIES
    ]

    # A PASS→FAIL only counts as a regression if the current FAIL is a real
    # model failure, not infra noise. Likewise a FAIL→PASS is only a genuine
    # recovery if the previous FAIL was a real model failure.
    regressed = sorted(
        m
        for m in common
        if prev_by[m]["verified_on_spyre"]
        and not curr_by[m]["verified_on_spyre"]
        and not _is_infra_failure(curr_by[m])
    )
    recovered = sorted(
        m
        for m in common
        if not prev_by[m]["verified_on_spyre"]
        and curr_by[m]["verified_on_spyre"]
        and not _is_infra_failure(prev_by[m])
    )
    # cpu_ok_spyre_fail = [
    #     r for r in curr_rows if r["verified_on_cpu"] and not r["verified_on_spyre"]
    # ]
    # Infrastructure-noise models (their verdict is not about the model)
    infra_noise = [
        r for r in curr_rows if (r.get("failure_category") or "") in _INFRA_CATEGORIES
    ]

    lines = [
        "",
        _hr("═"),
        "2. VERIFIED_ON_SPYRE DELTA",
        _hr(),
        f"  Absolute count  : {prev_pass:>6}  →  {curr_pass:>6}  ({_delta(curr_pass, prev_pass)})",
        "  Pass rate (excl. infrastructure-failures):",
        f"    prev  {_pct(prev_pass, len(prev_eligible))}  ({prev_pass}/{len(prev_eligible)})",
        f"    curr  {_pct(curr_pass, len(curr_eligible))}  ({curr_pass}/{len(curr_eligible)})",
        "",
        f"  Infrastructure noise (hardware_exception / worker_crashed / worker_timeout): {len(infra_noise)}",
        "  ⚠ If non-zero, curr verified_on_spyre count is understated for those models.",
        "",
        f"  Regressions PASS→FAIL ({len(regressed)}):",
    ]
    if regressed:
        lines.append(_trunc(regressed))
        # Annotate with failure_category
        lines.append("  Failure categories for regressed models:")
        cats = Counter(
            (curr_by[m].get("failure_category") or "unknown") for m in regressed
        )
        for cat, n in cats.most_common():
            lines.append(f"    {cat:<40}  {n:>4}")
    else:
        lines.append("    (none)")

    lines += [
        "",
        f"  Recoveries FAIL→PASS ({len(recovered)}):",
    ]
    if recovered:
        lines.append(_trunc(recovered))
    else:
        lines.append("    (none)")

    # lines += [
    #     "",
    #     f"  CPU-OK but Spyre-FAIL (currently): {len(cpu_ok_spyre_fail)}",
    # ]
    # if cpu_ok_spyre_fail:
    #     cats = Counter(
    #         (r.get("failure_category") or "unknown") for r in cpu_ok_spyre_fail
    #     )
    #     for cat, n in cats.most_common():
    #         lines.append(f"    {cat:<40}  {n:>4}")

    return "\n".join(lines)


def _section_failure_categories(
    prev_rows: list[dict[str, Any]],
    curr_rows: list[dict[str, Any]],
) -> str:
    # Only rows with a failure_category; passing rows (NULL) are not failures.
    prev_cats = Counter(
        r["failure_category"] for r in prev_rows if r.get("failure_category")
    )
    curr_cats = Counter(
        r["failure_category"] for r in curr_rows if r.get("failure_category")
    )
    all_cats = sorted(set(prev_cats) | set(curr_cats))

    lines = [
        "",
        _hr("═"),
        "3. FAILURE_CATEGORY DISTRIBUTION",
        _hr(),
        f"  {'category':<42}  {'prev':>6}  {'curr':>6}  {'delta':>6}",
        f"  {_hr('-', 42)}  {'------':>6}  {'------':>6}  {'------':>6}",
    ]

    infra_prev = infra_curr = 0

    for cat in all_cats:
        p = prev_cats.get(cat, 0)
        c = curr_cats.get(cat, 0)
        tag = ""
        if cat in _INFRA_CATEGORIES:
            tag = " [infra]"
            infra_prev += p
            infra_curr += c
        lines.append(f"  {cat + tag:<42}  {p:>6}  {c:>6}  {_delta(c, p):>6}")

    return "\n".join(lines)


def _section_error_patterns(
    prev_rows: list[dict[str, Any]],
    curr_rows: list[dict[str, Any]],
) -> str:
    prev_by = {r["model_name"]: r for r in prev_rows}
    curr_by = {r["model_name"]: r for r in curr_rows}
    common = set(prev_by) & set(curr_by)

    # Only newly failing models (not pre-existing failures)
    new_failures = [
        curr_by[m]
        for m in common
        if prev_by[m]["verified_on_spyre"]
        and not curr_by[m]["verified_on_spyre"]
        and curr_by[m].get("error")
    ]
    # All current failures with an error string
    all_failures_with_error = [r for r in curr_rows if r.get("error")]

    def _top_errors(
        rows: list[dict[str, Any]], limit: int = _MAX_ERROR_EXAMPLES
    ) -> list[str]:
        counts: Counter[str] = Counter()
        for r in rows:
            err = (r.get("error") or "").strip()
            # Truncate long error strings for readability
            counts[err[:120]] += 1
        out = []
        for err, n in counts.most_common(limit):
            out.append(f"    [{n:>4}×]  {err}")
        return out

    lines = [
        "",
        _hr("═"),
        "4. ERROR PATTERN ANALYSIS",
        _hr(),
        f"  Newly failing models with an error string: {len(new_failures)}",
    ]
    if new_failures:
        lines.append(f"  Top {_MAX_ERROR_EXAMPLES} error patterns (new regressions):")
        lines += _top_errors(new_failures)

        # Family / architecture breakdown
        fam_counts: Counter[str] = Counter(
            (r.get("family") or "(unknown)") for r in new_failures
        )
        lines += [
            "",
            "  Families affected by new regressions:",
        ]
        for fam, n in fam_counts.most_common():
            lines.append(f"    {fam:<40}  {n:>4}")

    lines += [
        "",
        f"  All current failures with error string: {len(all_failures_with_error)}",
        f"  Top {_MAX_ERROR_EXAMPLES} recurring error patterns (current week, all failures):",
    ]
    lines += _top_errors(all_failures_with_error)

    return "\n".join(lines)


def _section_adapter_coverage(
    prev_rows: list[dict[str, Any]],
    curr_rows: list[dict[str, Any]],
) -> str:
    def _adapters(rows: list[dict[str, Any]]) -> set[str]:
        return {r["adapter_name"] for r in rows if r.get("adapter_name")}

    def _covered_classes(rows: list[dict[str, Any]]) -> set[str]:
        return {r["config_class"] for r in rows if r.get("adapter_name")}

    def _uncovered_classes(rows: list[dict[str, Any]]) -> set[str]:
        return {
            r["config_class"]
            for r in rows
            if (r.get("failure_category") or "")
            == FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER
        }

    prev_adapters = _adapters(prev_rows)
    curr_adapters = _adapters(curr_rows)
    new_adapters = sorted(curr_adapters - prev_adapters)
    lost_adapters = sorted(prev_adapters - curr_adapters)

    prev_covered = _covered_classes(prev_rows)
    curr_covered = _covered_classes(curr_rows)
    newly_covered = sorted(curr_covered - prev_covered)
    lost_covered = sorted(prev_covered - curr_covered)

    prev_uncovered = _uncovered_classes(prev_rows)
    curr_uncovered = _uncovered_classes(curr_rows)
    newly_uncovered = sorted(curr_uncovered - prev_uncovered)

    # added_date per adapter (issue §5), taken from the current snapshot rows.
    added_date_by_adapter: dict[str, date | None] = {}
    for r in curr_rows:
        name: str | None = r.get("adapter_name")
        if name and name not in added_date_by_adapter:
            added_date_by_adapter[name] = r.get("added_date")

    lines = [
        "",
        _hr("═"),
        "5. ADAPTER / CONFIG_CLASS COVERAGE",
        _hr(),
        f"  Distinct adapters   : {len(prev_adapters):>5}  →  {len(curr_adapters):>5}  ({_delta(len(curr_adapters), len(prev_adapters))})",
        f"  Covered config_classes : {len(prev_covered):>5}  →  {len(curr_covered):>5}  ({_delta(len(curr_covered), len(prev_covered))})",
        "",
        f"  New adapters ({len(new_adapters)}):",
    ]
    if new_adapters:
        for adapter in new_adapters[:_MAX_MODELS_SHOWN]:
            added_on = added_date_by_adapter.get(adapter)
            when = f"  (added {added_on})" if added_on else ""
            lines.append(f"    • {adapter}{when}")
        if len(new_adapters) > _MAX_MODELS_SHOWN:
            lines.append(f"  … and {len(new_adapters) - _MAX_MODELS_SHOWN} more")
    else:
        lines.append("    (none)")
    lines += [
        "",
        f"  Lost adapters ({len(lost_adapters)}):",
    ]
    lines.append(_trunc(lost_adapters) if lost_adapters else "    (none)")
    lines += [
        "",
        f"  config_classes newly covered ({len(newly_covered)}):",
    ]
    lines.append(_trunc(newly_covered) if newly_covered else "    (none)")
    lines += [
        "",
        f"  config_classes that lost coverage ({len(lost_covered)}):",
    ]
    lines.append(_trunc(lost_covered) if lost_covered else "    (none)")
    lines += [
        "",
        f"  config_classes newly not-implemented-adapter ({len(newly_uncovered)}):",
    ]
    lines.append(_trunc(newly_uncovered) if newly_uncovered else "    (none)")

    return "\n".join(lines)


def _section_family_breakdown(
    prev_rows: list[dict[str, Any]],
    curr_rows: list[dict[str, Any]],
) -> str:
    def _family_pass_rate(rows: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
        """Return {family: (pass_count, total_count)} excluding infra-failure rows.

        Only infrastructure failures (hardware_exception / worker_crashed /
        worker_timeout) are excluded — their verdict is not about the model.
        Pre-filter verdicts (not-implemented-adapter / model_too_large / moe) are
        kept in the denominator: a model unsupported on Spyre counts as a failure.
        """
        result: dict[str, list[int]] = {}
        for r in rows:
            if (r.get("failure_category") or "") in _INFRA_CATEGORIES:
                continue
            fam = r.get("family") or "(unknown)"
            if fam not in result:
                result[fam] = [0, 0]
            result[fam][1] += 1
            if r["verified_on_spyre"]:
                result[fam][0] += 1
        return {k: (v[0], v[1]) for k, v in result.items()}

    prev_fam = _family_pass_rate(prev_rows)
    curr_fam = _family_pass_rate(curr_rows)
    all_fams = sorted(set(prev_fam) | set(curr_fam))

    # Find families with a notable pass-rate change
    changed: list[tuple[str, float]] = []
    for fam in all_fams:
        p_pass, p_tot = prev_fam.get(fam, (0, 0))
        c_pass, c_tot = curr_fam.get(fam, (0, 0))
        if p_tot == 0 or c_tot == 0:
            continue
        delta_pct = (c_pass / c_tot - p_pass / p_tot) * 100
        if abs(delta_pct) >= 5.0:
            changed.append((fam, delta_pct))
    changed.sort(key=lambda x: x[1])

    lines = [
        "",
        _hr("═"),
        "6. SIZE / FAMILY BREAKDOWN",
        _hr(),
        f"  {'family':<30}  {'prev pass':>9}  {'curr pass':>9}  {'delta pp':>8}",
        f"  {_hr('-', 30)}  {'---------':>9}  {'---------':>9}  {'--------':>8}",
    ]
    for fam in all_fams:
        p_pass, p_tot = prev_fam.get(fam, (0, 0))
        c_pass, c_tot = curr_fam.get(fam, (0, 0))
        prev_r = _pct(p_pass, p_tot)
        curr_r = _pct(c_pass, c_tot)
        if p_tot > 0 and c_tot > 0:
            delta_pp = (c_pass / c_tot - p_pass / p_tot) * 100
            delta_str = f"{delta_pp:+.1f}pp"
        else:
            delta_str = "n/a"
        lines.append(f"  {fam:<30}  {prev_r:>9}  {curr_r:>9}  {delta_str:>8}")

    lines += [
        "",
        "  Families with ≥5pp pass-rate change:",
    ]
    if changed:
        for fam, delta_pct in changed:
            lines.append(f"    {fam:<30}  {delta_pct:+.1f}pp")
    else:
        lines.append("    (none)")

    # Parameter-size of regressions
    prev_by = {r["model_name"]: r for r in prev_rows}
    curr_by = {r["model_name"]: r for r in curr_rows}
    common = set(prev_by) & set(curr_by)
    regressed = [
        curr_by[m]
        for m in common
        if prev_by[m]["verified_on_spyre"]
        and not curr_by[m]["verified_on_spyre"]
        and not _is_infra_failure(curr_by[m])
    ]
    if regressed:
        sizes: list[int] = sorted(r.get("parameters_number") or 0 for r in regressed)
        median: int = int(statistics.median(sizes))
        lines += [
            "",
            "  Regressed models — parameter sizes:",
            f"    count={len(sizes)}, min={min(sizes):,}, median={median:,}, max={max(sizes):,}",
        ]

        # Finer than family: architecture breakdown of the regressions (issue §6).
        arch_counts: Counter[str] = Counter(
            (r.get("architecture") or "(unknown)") for r in regressed
        )
        lines.append("  Regressed models — architecture breakdown:")
        for arch, n in arch_counts.most_common():
            lines.append(f"    {arch:<40}  {n:>4}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["generative", "embedding"],
        required=True,
        help="Which table to analyse.",
    )
    parser.add_argument(
        "--prev",
        type=date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help="Previous snapshot date. Defaults to the second-most-recent in the table.",
    )
    parser.add_argument(
        "--curr",
        type=date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help="Current snapshot date. Defaults to the most-recent in the table.",
    )
    args = parser.parse_args(argv)
    if (args.prev is None) != (args.curr is None):
        parser.error("--prev and --curr must be provided together or not at all.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    model_type = ModelType(args.mode)
    table = (
        GENERATIVE_TABLE_NAME
        if model_type is ModelType.GENERATIVE
        else EMBEDDING_TABLE_NAME
    )

    client = get_client()

    if args.prev is None:
        prev_date, curr_date = _two_latest_dates(client, table)
    else:
        prev_date, curr_date = args.prev, args.curr

    prev_rows = _fetch_snapshot(client, table, prev_date)
    curr_rows = _fetch_snapshot(client, table, curr_date)

    header = "\n".join(
        [
            _hr("═"),
            f"WEEKLY SPYRE SCAN — {model_type.value.upper()} REPORT",
            f"  Previous snapshot : {prev_date}",
            f"  Current snapshot  : {curr_date}",
        ]
    )

    report = "\n".join(
        [
            header,
            _section_coverage(prev_rows, curr_rows, prev_date, curr_date),
            _section_verified_on_spyre(prev_rows, curr_rows),
            _section_failure_categories(prev_rows, curr_rows),
            _section_error_patterns(prev_rows, curr_rows),
            _section_adapter_coverage(prev_rows, curr_rows),
            _section_family_breakdown(prev_rows, curr_rows),
            _hr("═"),
        ]
    )
    print(report)


if __name__ == "__main__":
    main()
