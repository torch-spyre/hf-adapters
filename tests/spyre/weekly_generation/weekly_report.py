"""Week-over-week textual comparison report for the Spyre weekly scan.

Queries ClickHouse for two consecutive snapshot_dates and prints a structured
plain-text report covering the six analysis areas defined in issue #484:

  1. Coverage (model count changes)
  2. Adapter / config_class coverage changes
  3. verified_on_spyre delta
  4. failure_category distribution shift
  5. Error pattern analysis
  6. Size / family breakdown

Usage::

    # Compare the two most recent snapshots automatically
    python tests/spyre/weekly_generation/weekly_report.py --mode generative
    python tests/spyre/weekly_generation/weekly_report.py --mode embedding

    # Compare specific dates
    python tests/spyre/weekly_generation/weekly_report.py \\
        --mode generative --prev 2025-06-21 --curr 2025-06-28

    # Choose how section 5 groups error messages (default: normalized)
    python tests/spyre/weekly_generation/weekly_report.py \\
        --mode generative --cluster-method exact

Credentials are read from .env (CLICKHOUSE_HOST / CLICKHOUSE_PASS etc.) exactly
as the rest of the weekly pipeline does — see clickhouse_db.py.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any

from tests.spyre.weekly_generation.clickhouse_db import get_client
from tests.spyre.weekly_generation.failure_categories import (
    FAILURE_CATEGORY_CPU_GENERATE_FAILED,
    FAILURE_CATEGORY_CPU_LOAD_FAILED,
    FAILURE_CATEGORY_HARDWARE_EXCEPTION,
    FAILURE_CATEGORY_MISFORMED_HF_FAILED,
    FAILURE_CATEGORY_MODEL_TOO_LARGE,
    FAILURE_CATEGORY_MOE,
    FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER,
    FAILURE_CATEGORY_QUANTIZED_MODEL,
    FAILURE_CATEGORY_TEST_EXECUTION_EXCEPTION,
    FAILURE_CATEGORY_UNSUPPORTED_CHECKPOINT,
    FAILURE_CATEGORY_VERIFICATION_FAILED,
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

# Section 4 rolls the raw failure_category values up into four meaningful groups
# so the distribution reads as "what kind of failure" rather than a flat list.
# Every category string defined in failure_categories.py must belong to exactly
# one group below (a stray/unknown value falls into "other" at render time).
#
#   unrelated  — nothing to do with Spyre: the model failed on CPU, is quantized,
#                is a malformed/unsupported checkpoint. Not our verdict.
#   expected   — Spyre cannot run this model *by design* and we know it up front:
#                no adapter implemented, too large for the device, MoE. A
#                deterministic pre-filter verdict, not a bug.
#   unexpected — Spyre *should* have run this model but it failed at test time:
#                the adapter compiled/ran wrong (test_execution_exception) or the
#                output was incorrect (verification_failed). These are the ones
#                worth chasing — a real Spyre/adapter defect.
#   infra      — the run itself broke (device unreachable, worker crash/timeout);
#                the verdict is about the harness, not the model. Mirrors
#                _INFRA_CATEGORIES.
_GROUP_UNRELATED: str = "unrelated model issue"
_GROUP_EXPECTED: str = "expected Spyre limitation"
_GROUP_UNEXPECTED: str = "unexpected Spyre failure"
_GROUP_INFRA: str = "infrastructure"
_GROUP_OTHER: str = "other / uncategorised"

# Insertion order here is the display order of the groups in section 4.
_CATEGORY_GROUPS: dict[str, frozenset[str]] = {
    _GROUP_UNEXPECTED: frozenset(
        {
            FAILURE_CATEGORY_TEST_EXECUTION_EXCEPTION,
            FAILURE_CATEGORY_VERIFICATION_FAILED,
        }
    ),
    _GROUP_EXPECTED: frozenset(
        {
            FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER,
            FAILURE_CATEGORY_MODEL_TOO_LARGE,
            FAILURE_CATEGORY_MOE,
            FAILURE_CATEGORY_QUANTIZED_MODEL,
        }
    ),
    _GROUP_UNRELATED: frozenset(
        {
            FAILURE_CATEGORY_CPU_LOAD_FAILED,
            FAILURE_CATEGORY_CPU_GENERATE_FAILED,
            FAILURE_CATEGORY_MISFORMED_HF_FAILED,
            FAILURE_CATEGORY_UNSUPPORTED_CHECKPOINT,
        }
    ),
    _GROUP_INFRA: _INFRA_CATEGORIES,
}


def _group_for_category(category: str) -> str:
    """Return the display group a raw ``failure_category`` belongs to.

    Falls back to ``_GROUP_OTHER`` for any category not mapped in
    ``_CATEGORY_GROUPS`` (e.g. a newly added string not yet grouped here), so an
    unknown value is surfaced rather than silently dropped.
    """
    for group, members in _CATEGORY_GROUPS.items():
        if category in members:
            return group
    return _GROUP_OTHER


# When clustering error messages, show up to this many distinct examples.
_MAX_ERROR_EXAMPLES: int = 20

# When listing changed models, show up to this many per bucket.
_MAX_MODELS_SHOWN: int = 5


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


# ---------------------------------------------------------------------------
# Error clustering
# ---------------------------------------------------------------------------
#
# Section 5 groups the failing models' error messages into clusters and reports
# the biggest ones. Two grouping methods are offered, selectable with
# ``--cluster-method``:
#
#   exact       Group by the raw first line, truncated, with only bracketed
#               spans ([...]) collapsed. Two errors cluster when their text is
#               identical apart from a tensor shape / index list — so the
#               "cannot reshape tensor … into shape [1, 0, -1, 64]" family lands
#               on one row — but everything else (versions, repo ids, quoted
#               names) stays literal, keeping this method faithful and far
#               coarser-grained than the normalized one.
#
#   normalized  Group by a signature that erases the model-specific token
#               classes (numbers, tensor shapes, versions, repo ids, quoted
#               literals) from the first line, so errors describing the *same*
#               fault collapse onto one row while distinct faults stay apart.
#               Validated against a full snapshot (1234 generative + 600
#               embedding non-infra errors on 2026-09-05): it reduced ~160 raw
#               distinct first-lines to ~80 clusters whose top 20 cover >90% of
#               all failures, with no rule written for any specific message.
#
# Both look only at the *first* line: the lines beneath it are a traceback /
# source snippet whose file paths and line numbers vary by install and would
# defeat either grouping.


class ClusterMethod(StrEnum):
    """How section 5 groups error messages. See the block comment above."""

    EXACT = "exact"
    NORMALIZED = "normalized"


# The default when ``--cluster-method`` is not given: the semantic grouping,
# which is what makes the section readable on a real snapshot.
_DEFAULT_CLUSTER_METHOD: ClusterMethod = ClusterMethod.NORMALIZED

# "exact" method: how many characters of the (bracket-collapsed) first line to
# keep. Long enough to distinguish faults, short enough that a runaway tail does
# not dominate. Collapsing brackets before truncating also equalizes the cutoff
# point across a fault family whose only difference was the length of a shape
# list, so the truncated tail no longer splits the cluster.
_EXACT_TRUNC_LEN: int = 120

# The normalized substitutions are ordered — URL before repo-id before the
# quote/number rules — because an earlier rule consumes text a later rule would
# otherwise mis-handle (e.g. a URL contains an ``org/model`` that the repo rule
# should not see, and a version like ``1.2b`` must be protected before the bare
# integer rule reaches it).

# Ordered (pattern, replacement) pairs applied left to right to one first line.
_NORMALIZE_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    # URLs first — they embed slashes and digits the later rules would mangle.
    (re.compile(r"https?://\S+"), "<url>"),
    # Hex addresses: 0x7f3c… -> 0xADDR
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),
    # HuggingFace repo ids: exactly one slash, no spaces (org/model-name.v2).
    # Runs before the version/number rules so digits inside a repo name (Carbon-3B,
    # GENERator-1.2b) do not first get rewritten and split the cluster.
    (re.compile(r"\b[\w.\-]+/[\w.\-]+\b"), "<repo>"),
    # Quoted literals (config classes, token names, dtypes, module names). Length
    # bounded and non-greedy so an apostrophe inside a word (``Can't``) cannot
    # swallow the rest of the line.
    (re.compile(r"'[^']{0,80}?'"), "'…'"),
    (re.compile(r'"[^"]{0,80}?"'), '"…"'),
    (re.compile(r"`[^`]{0,80}?`"), "`…`"),
    # Version strings: 0.15.0, 0.46.1 -> VER (before the bare-integer rule).
    (re.compile(r"\b\d+(?:\.\d+)+\b"), "VER"),
)

# Innermost-out collapse of bracketed / parenthesised spans (tensor shapes,
# argument tuples). Applied repeatedly so a nested ``([1, 2], [3])`` reduces
# fully. Kept separate from _NORMALIZE_SUBS because it needs the loop.
_BRACKET_RE: re.Pattern[str] = re.compile(r"\[[^\[\]]*\]")
_PAREN_RE: re.Pattern[str] = re.compile(r"\([^()]*\)")

# Safety net only: clustering is done by the normalization above, not by this
# cap. The prefix-length sweep on real data was essentially flat (79 clusters at
# full length vs 77 at 90 chars), so this just guards against a pathological
# runaway tail; it is deliberately generous.
_SIGNATURE_PREFIX_LEN: int = 100


def _first_nonempty_line(text: str) -> str:
    for raw in text.splitlines():
        if raw.strip():
            return raw.strip()
    return ""


def _exact_signature(error: str) -> str:
    """Signature for the ``exact`` method: first line, only ``[...]`` collapsed.

    The one normalization applied is collapsing bracketed spans (tensor shapes,
    index lists) to ``[…]``, so a fault whose message differs only in a shape —
    ``reshape … into shape [1, 0, -1, 64]`` vs ``[-1, 0]`` — clusters together.
    Everything else stays literal, so this remains far coarser-grained than the
    normalized method. Doubles as the displayed label.
    """
    sig: str = _first_nonempty_line(error)
    for _ in range(4):
        collapsed: str = _BRACKET_RE.sub("[…]", sig)
        if collapsed == sig:
            break
        sig = collapsed
    return sig[:_EXACT_TRUNC_LEN]


def _normalized_signature(error: str) -> str:
    """Signature for the ``normalized`` method: first line, token classes erased.

    Erases every model-specific token class (URLs, repo ids, quoted literals,
    versions, bracketed shapes, bare numbers) from the first line so two errors
    describing the same fault reduce to the same string. The result is also the
    human-readable label shown for the cluster — the placeholders keep it
    legible — so this doubles as both grouping key and display text.
    """
    sig: str = _first_nonempty_line(error)
    for pattern, replacement in _NORMALIZE_SUBS:
        sig = pattern.sub(replacement, sig)
    for _ in range(4):
        collapsed: str = _PAREN_RE.sub("(…)", _BRACKET_RE.sub("[…]", sig))
        if collapsed == sig:
            break
        sig = collapsed
    sig = re.sub(r"\b\d+\b", "N", sig)
    sig = re.sub(r"\s+", " ", sig).strip()
    return sig[:_SIGNATURE_PREFIX_LEN]


def _signature_for(error: str, method: ClusterMethod) -> str:
    """Compute the clustering signature of *error* under the chosen *method*."""
    if method is ClusterMethod.EXACT:
        return _exact_signature(error)
    return _normalized_signature(error)


def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def _section_coverage(
    prev_rows: list[dict[str, Any]],
    curr_rows: list[dict[str, Any]],
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
    delta_str = (
        f"[{_delta(len(curr_models), len(prev_models))} ({_delta(curr_curated, prev_curated)})]"
        if (_delta(len(curr_models), len(prev_models)) != "0")
        else ""
    )

    lines = [
        _hr("═"),
        "1. COVERAGE — MODEL COUNT",
        _hr(),
        f"  Total models (Curated models)   : {len(prev_models):>6} ({prev_curated})  →  {len(curr_models):>6} ({curr_curated})  {delta_str}",
        "\n",
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
    # recovery if the previous FAIL was a real model failure. Both lists are
    # ordered by num_downloads (most-downloaded first) so the models that matter
    # most surface at the top; the current snapshot's count is used as it is the
    # freshest, tie-broken by name for stable output.
    def _by_downloads_desc(name: str) -> tuple[int, str]:
        return (-(curr_by[name].get("num_downloads") or 0), name)

    regressed = sorted(
        (
            m
            for m in common
            if prev_by[m]["verified_on_spyre"]
            and not curr_by[m]["verified_on_spyre"]
            and not _is_infra_failure(curr_by[m])
        ),
        key=_by_downloads_desc,
    )
    recovered = sorted(
        (
            m
            for m in common
            if not prev_by[m]["verified_on_spyre"]
            and curr_by[m]["verified_on_spyre"]
            and not _is_infra_failure(prev_by[m])
        ),
        key=_by_downloads_desc,
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
        "3. VERIFIED_ON_SPYRE DELTA",
        _hr(),
        f"  Absolute count  : {prev_pass:>6}  →  {curr_pass:>6}  ({_delta(curr_pass, prev_pass)})",
        "  Pass rate (excl. infrastructure-failures):",
        f"    prev  {_pct(prev_pass, len(prev_eligible))}  ({prev_pass}/{len(prev_eligible)})",
        f"    curr  {_pct(curr_pass, len(curr_eligible))}  ({curr_pass}/{len(curr_eligible)})",
        "",
        f"  Infrastructure noise (hardware_exception / worker_crashed / worker_timeout): {len(infra_noise)}",
        "  ⚠ If non-zero, curr verified_on_spyre count is understated for those models.",
        "\n"
        f"  Regressions PASS→FAIL (ignoring infrastructure noise) - {len(regressed)}:",
    ]
    if regressed:
        lines.append(
            _format_models_with_downloads(
                [(m, curr_by[m].get("num_downloads") or 0) for m in regressed]
            )
        )
    else:
        lines.append("    (none)")

    if recovered:
        lines += [
            "",
            f"  Recoveries FAIL→PASS (ignoring infrastructure noise) - {len(recovered)}:",
        ]
        lines.append(
            _format_models_with_downloads(
                [(m, curr_by[m].get("num_downloads") or 0) for m in recovered]
            )
        )

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
    all_cats: set[str] = set(prev_cats) | set(curr_cats)

    # Bucket every category into its meaningful group (see _CATEGORY_GROUPS).
    by_group: dict[str, list[str]] = {}
    for cat in all_cats:
        by_group.setdefault(_group_for_category(cat), []).append(cat)

    lines = [
        "",
        _hr("═"),
        "4. FAILURE_CATEGORY DISTRIBUTION",
        _hr(),
        f"  {'category':<42}  {'prev':>6}  {'curr':>6}  {'delta':>6}",
        f"  {_hr('-', 42)}  {'------':>6}  {'------':>6}  {'------':>6}",
    ]

    # Render one labelled block per group, groups ordered as _CATEGORY_GROUPS
    # (unexpected first — those are the ones worth chasing), then any leftover
    # "other" group last. Within a group, categories sort by descending delta,
    # tie-broken by name. Each block leads with a subtotal row so the group
    # magnitudes are comparable at a glance.
    group_order: list[str] = list(_CATEGORY_GROUPS) + [_GROUP_OTHER]
    total_prev = total_curr = 0

    for group in group_order:
        cats = by_group.get(group)
        if not cats:
            continue
        cats.sort(
            key=lambda cat: (-(curr_cats.get(cat, 0) - prev_cats.get(cat, 0)), cat)
        )
        g_prev: int = sum(prev_cats.get(cat, 0) for cat in cats)
        g_curr: int = sum(curr_cats.get(cat, 0) for cat in cats)
        total_prev += g_prev
        total_curr += g_curr

        lines.append("")
        lines.append(
            f"  {group.upper():<42}  {g_prev:>6}  {g_curr:>6}  {_delta(g_curr, g_prev):>6}"
        )
        for cat in cats:
            p = prev_cats.get(cat, 0)
            c = curr_cats.get(cat, 0)
            lines.append(f"    {cat:<40}  {p:>6}  {c:>6}  {_delta(c, p):>6}")

    lines.append(f"  {_hr('-', 42)}  {'------':>6}  {'------':>6}  {'------':>6}")
    lines.append(
        f"  {'TOTAL':<42}  {total_prev:>6}  {total_curr:>6}  {_delta(total_curr, total_prev):>6}"
    )

    return "\n".join(lines)


def _section_error_patterns(
    prev_rows: list[dict[str, Any]],
    curr_rows: list[dict[str, Any]],
    method: ClusterMethod = _DEFAULT_CLUSTER_METHOD,
) -> str:
    # All current failures with an error string, excluding infra failures
    # (their error is about the infrastructure, not the model).
    all_failures_with_error: list[dict[str, Any]] = [
        r for r in curr_rows if r.get("error") and not _is_infra_failure(r)
    ]

    def _top_errors(
        rows: list[dict[str, Any]], limit: int = _MAX_ERROR_EXAMPLES
    ) -> list[str]:
        # Group by the signature for the chosen method (see _signature_for). The
        # signature doubles as the readable label. `raw_variants` counts the
        # distinct raw first-lines each cluster absorbed — always 1 under the
        # exact method (identical text is the grouping condition), so the
        # "(+N variants)" marker is only ever shown for the normalized method,
        # where it flags a genuine family versus a lone recurring string.
        counts: Counter[str] = Counter()
        raw_variants: dict[str, set[str]] = {}
        for r in rows:
            err: str = (r.get("error") or "").strip()
            if not err:
                continue
            sig: str = _signature_for(err, method)
            counts[sig] += 1
            raw_variants.setdefault(sig, set()).add(_first_nonempty_line(err))
        out: list[str] = []
        for sig, n in counts.most_common(limit):
            extra: int = len(raw_variants[sig]) - 1
            suffix: str = (
                f"  (+{extra} variant{'s' if extra > 1 else ''})" if extra else ""
            )
            out.append(f"    [{n:>5}×]  {sig}{suffix}")
        return out

    total_errors: int = len(all_failures_with_error)
    clusters: set[str] = {
        _signature_for(r["error"] or "", method) for r in all_failures_with_error
    }
    grouping_note: str = (
        "near-identical messages grouped"
        if method is ClusterMethod.NORMALIZED
        else "identical messages grouped"
    )

    lines: list[str] = [
        "",
        _hr("═"),
        "5. ERROR PATTERN ANALYSIS",
        _hr(),
        f"  Clustering method: {method.value}",
        f"  Failures with an error message (excl. infra): {total_errors}",
        f"  Distinct error clusters: {len(clusters)}",
        "",
        f"  Top {_MAX_ERROR_EXAMPLES} recurring error clusters ({grouping_note}):",
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

    # added_date per adapter (issue §5), taken from the current snapshot rows.
    added_date_by_adapter: dict[str, date | None] = {}
    for r in curr_rows:
        name: str | None = r.get("adapter_name")
        if name and name not in added_date_by_adapter:
            added_date_by_adapter[name] = r.get("added_date")

    lines = [
        "",
        _hr("═"),
        "2. ADAPTER COVERAGE",
        _hr(),
        f"  Distinct adapters   : {len(prev_adapters):>5}  →  {len(curr_adapters):>5}  ({_delta(len(curr_adapters), len(prev_adapters))})",
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
    if lost_adapters:
        lines += [
            "",
            f"  Lost adapters ({len(lost_adapters)}):",
        ]
        lines.append(_trunc(lost_adapters) if lost_adapters else "    (none)")

    return "\n".join(lines)


def _section_family_breakdown(
    prev_rows: list[dict[str, Any]],
    curr_rows: list[dict[str, Any]],
) -> str:
    def _adapter_pass_rate(rows: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
        """Return {adapter_name: (pass_count, total_count)} excluding infra rows.

        Only infrastructure failures (hardware_exception / worker_crashed /
        worker_timeout) are excluded — their verdict is not about the model.
        Pre-filter verdicts (not-implemented-adapter / model_too_large / moe) are
        kept in the denominator: a model unsupported on Spyre counts as a failure.
        Rows without an adapter bucket under ``(no adapter)``.
        """
        result: dict[str, list[int]] = {}
        for r in rows:
            if (r.get("failure_category") or "") in _INFRA_CATEGORIES:
                continue
            adapter = r.get("adapter_name") or "(no adapter)"
            if adapter not in result:
                result[adapter] = [0, 0]
            result[adapter][1] += 1
            if r["verified_on_spyre"]:
                result[adapter][0] += 1
        return {k: (v[0], v[1]) for k, v in result.items()}

    prev_adapter = _adapter_pass_rate(prev_rows)
    curr_adapter = _adapter_pass_rate(curr_rows)
    all_adapters = sorted(set(prev_adapter) | set(curr_adapter))

    # Find adapters with a notable pass-rate change
    changed: list[tuple[str, float]] = []
    for adapter in all_adapters:
        p_pass, p_tot = prev_adapter.get(adapter, (0, 0))
        c_pass, c_tot = curr_adapter.get(adapter, (0, 0))
        if p_tot == 0 or c_tot == 0:
            continue
        delta_pct = (c_pass / c_tot - p_pass / p_tot) * 100
        if abs(delta_pct) >= 5.0:
            changed.append((adapter, delta_pct))
    changed.sort(key=lambda x: x[1])

    lines = [
        "",
        _hr("═"),
        "6. ADAPTER BREAKDOWN",
        _hr(),
        "  Adapters with ≥5pp pass-rate change:",
    ]
    if changed:
        for adapter, delta_pct in changed:
            lines.append(f"    {adapter:<30}  {delta_pct:+.1f}pp")
    else:
        lines.append("    (none)")

    # # Parameter-size of regressions
    # prev_by = {r["model_name"]: r for r in prev_rows}
    # curr_by = {r["model_name"]: r for r in curr_rows}
    # common = set(prev_by) & set(curr_by)
    # regressed = [
    #     curr_by[m]
    #     for m in common
    #     if prev_by[m]["verified_on_spyre"]
    #     and not curr_by[m]["verified_on_spyre"]
    #     and not _is_infra_failure(curr_by[m])
    # ]
    # if regressed:
    #     sizes: list[int] = sorted(r.get("parameters_number") or 0 for r in regressed)
    #     median: int = int(statistics.median(sizes))
    #     lines += [
    #         "",
    #         "  Regressed models — parameter sizes:",
    #         f"    count={len(sizes)}, min={min(sizes):,}, median={median:,}, max={max(sizes):,}",
    #     ]
    #
    #     # Finer than family: architecture breakdown of the regressions (issue §6).
    #     arch_counts: Counter[str] = Counter(
    #         (r.get("architecture") or "(unknown)") for r in regressed
    #     )
    #     lines.append("  Regressed models — architecture breakdown:")
    #     for arch, n in arch_counts.most_common():
    #         lines.append(f"    {arch:<40}  {n:>4}")

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
    parser.add_argument(
        "--cluster-method",
        choices=[m.value for m in ClusterMethod],
        default=_DEFAULT_CLUSTER_METHOD.value,
        help=(
            "How section 5 groups error messages. "
            "'normalized' (default) erases model-specific tokens so the same "
            "fault clusters together; 'exact' groups only byte-identical "
            "first lines."
        ),
    )
    args = parser.parse_args(argv)
    if (args.prev is None) != (args.curr is None):
        parser.error("--prev and --curr must be provided together or not at all.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    model_type = ModelType(args.mode)
    cluster_method = ClusterMethod(args.cluster_method)
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
            _section_coverage(prev_rows, curr_rows),
            _section_adapter_coverage(prev_rows, curr_rows),
            _section_verified_on_spyre(prev_rows, curr_rows),
            _section_failure_categories(prev_rows, curr_rows),
            _section_error_patterns(prev_rows, curr_rows, cluster_method),
            _section_family_breakdown(prev_rows, curr_rows),
            _hr("═"),
        ]
    )
    print(report)


if __name__ == "__main__":
    main()
