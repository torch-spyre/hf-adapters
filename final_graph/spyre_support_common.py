#!/usr/bin/env python3
"""Shared code for the Spyre support-trend charts.

Both ``plot_spyre_support.py`` (absolute counts) and
``plot_spyre_support_percent.py`` (pass rate) read the same two
``*_model_spyre_support.csv`` exports (schema:
``tests/spyre/weekly_generation/table_schema.py``), aggregate per
``snapshot_date``, and draw a two-panel figure: a top panel that differs
between the two variants, and an identical bottom panel showing the number of
distinct adapters per snapshot.

This module holds everything the two variants share — CSV parsing, dummy-data
fallback, the merge onto a common date axis, the reusable drawing primitives,
and the figure scaffolding (``render``) — leaving each variant to supply only
its top-panel drawing via a :class:`TopPanel` callback bundle.

The CSV has no header row and carries two extra trailing columns (a UUID and an
ingest timestamp) beyond the 15 schema columns; only ``model_name`` (col 0),
``adapter_name`` (col 2), ``snapshot_date`` (col 4) and ``verified_on_spyre``
(col 7) are read, so the extras are harmless.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

# Column indices within a CSV row (see table_schema.TABLE_COLUMNS).
COL_MODEL: int = 0
COL_ADAPTER: int = 2
COL_SNAPSHOT: int = 4
COL_VERIFIED_SPYRE: int = 7

# --- palette (light-mode blue + orange) ---
GEN_VER: str = "#c24e1f"  # generative, verified    (dark orange)
GEN_NOT: str = "#f6cdb6"  # generative, not verified (light orange)
GEN_LINE: str = "#8f3912"  # generative verified line + labels
EMB_VER: str = "#1c5cab"  # embedding, verified     (dark blue)
EMB_NOT: str = "#bcd6f6"  # embedding, not verified (light blue)
EMB_LINE: str = "#13406f"  # embedding verified line + labels

INK: str = "#0b0b0b"
SECONDARY: str = "#52514e"
MUTED: str = "#898781"
GRID: str = "#e1e0d9"
BASELINE: str = "#c3c2b7"
SURFACE: str = "#fcfcfb"

# Shared figure geometry.
BAR_WIDTH: float = 0.34
OFFSET: float = 0.19  # each column sits ±OFFSET from the date center


# A record is ``(tested, verified)`` for one table on one date, or ``None`` when
# that table had no snapshot on that date.
Record = tuple[int, int] | None


# --------------------------------------------------------------------------- #
# Data: CSV parsing, dummy fallback, merge
# --------------------------------------------------------------------------- #
def _truthy(value: str) -> bool:
    """Parse a ClickHouse-exported boolean cell (``true``/``false``/1/0)."""
    return value.strip().strip('"').lower() in ("true", "1", "t")


def aggregate(path: Path) -> dict[str, tuple[int, int]]:
    """Count verified / not-verified models per snapshot date.

    Returns ``{snapshot_date: (verified, not_verified)}``. Deduped by model
    within each date (a model counts once; a True wins over a stray False row)
    so duplicate rows from the ReplacingMergeTree source cannot inflate counts.
    """
    verified: dict[str, set[str]] = defaultdict(set)
    not_verified: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            date: str = row[COL_SNAPSHOT].strip().strip('"')
            model: str = row[COL_MODEL]
            if _truthy(row[COL_VERIFIED_SPYRE]):
                verified[date].add(model)
                not_verified[date].discard(model)
            elif model not in verified[date]:
                not_verified[date].add(model)
    dates: set[str] = set(verified) | set(not_verified)
    return {d: (len(verified[d]), len(not_verified[d])) for d in dates}


def adapters_per_date(path: Path) -> dict[str, int]:
    """Count distinct adapters (``adapter_name``) per snapshot date.

    Returns ``{snapshot_date: n_adapters}``. Counts unique adapter names, so
    the many models a single adapter covers collapse to one.
    """
    adapters: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            date: str = row[COL_SNAPSHOT].strip().strip('"')
            adapters[date].add(row[COL_ADAPTER].strip().strip('"'))
    return {d: len(names) for d, names in adapters.items()}


def dummy_adapters(dates: list[str]) -> dict[str, int]:
    """Synthesize a plausible adapter-count ramp (1 -> ~25) over ``dates``."""
    n: int = len(dates)
    return {
        date: (1 if n <= 1 else round(1 + 24 * i / (n - 1)))
        for i, date in enumerate(sorted(dates))
    }


def dummy_aggregate(scale: int) -> dict[str, tuple[int, int]]:
    """Synthesize a plausible ``{snapshot_date: (verified, not_verified)}``.

    Used as a fallback when a table's CSV path is not provided or missing, so
    the script still renders a representative chart (a cold start ramping to a
    plateau) rather than failing. ``scale`` is the roughly-constant tested
    population, so the two tables can be given different sizes.
    """
    # (snapshot_date, fraction of ``scale`` that is verified_on_spyre)
    schedule: list[tuple[str, float]] = [
        ("2026-04-18", 0.00),
        ("2026-05-16", 0.03),
        ("2026-05-30", 0.41),
        ("2026-06-13", 0.77),
        ("2026-06-20", 0.77),
        ("2026-08-01", 0.77),
        ("2026-08-08", 0.77),
        ("2026-08-15", 0.88),
        ("2026-08-29", 0.90),
        ("2026-09-12", 0.90),
        ("2026-09-26", 0.90),
    ]
    out: dict[str, tuple[int, int]] = {}
    for date, verified_fraction in schedule:
        verified: int = round(scale * verified_fraction)
        out[date] = (verified, scale - verified)
    return out


def resolve_table(
    path: Path | None,
    dummy_scale: int,
) -> tuple[dict[str, tuple[int, int]], dict[str, int]]:
    """Aggregate a table from its CSV, or fall back to dummy data.

    Returns ``(model_counts, adapter_counts)`` where ``model_counts`` is
    ``{date: (verified, not_verified)}`` and ``adapter_counts`` is
    ``{date: n_adapters}``. Falls back to dummy data when ``path`` is ``None``
    (flag not provided) or points at a file that does not exist.
    """
    if path is None or not path.exists():
        reason: str = "no path provided" if path is None else f"file not found: {path}"
        print(f"[dummy] {reason} — generating dummy data (scale={dummy_scale:,})")
        models: dict[str, tuple[int, int]] = dummy_aggregate(dummy_scale)
        return models, dummy_adapters(list(models))
    return aggregate(path), adapters_per_date(path)


def _short(date: str) -> str:
    """``2026-09-26`` -> ``26-09-26`` for compact x-axis ticks."""
    year, month, day = date.split("-")
    return f"{year[2:]}-{month}-{day}"


def build_rows(
    gen: dict[str, tuple[int, int]],
    emb: dict[str, tuple[int, int]],
    gen_adapters: dict[str, int],
    emb_adapters: dict[str, int],
) -> list[dict[str, object]]:
    """Merge the two tables onto the union of their snapshot dates.

    Each row is ``{"date", "gen": (tested, verified) | None, "emb": ...,
    "gen_adapters": int | None, "emb_adapters": int | None}``; a table absent
    on a date yields ``None`` so its column and line break rather than
    interpolating across a snapshot that was never measured.
    """
    all_dates: list[str] = sorted(set(gen) | set(emb))
    rows: list[dict[str, object]] = []
    for date in all_dates:
        row: dict[str, object] = {"date": _short(date)}
        if date in gen:
            v, nv = gen[date]
            row["gen"] = (v + nv, v)
        else:
            row["gen"] = None
        if date in emb:
            v, nv = emb[date]
            row["emb"] = (v + nv, v)
        else:
            row["emb"] = None
        row["gen_adapters"] = gen_adapters.get(date)
        row["emb_adapters"] = emb_adapters.get(date)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Drawing primitives
# --------------------------------------------------------------------------- #
def _label(ax: plt.Axes, x: float, y: float, text: str, color: str, dy: float) -> None:
    """Place one bold, surface-haloed value label above a point."""
    ax.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(0, dy),
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=color,
        zorder=7,
        path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)],
    )


def draw_line(
    ax: plt.Axes,
    xpos: np.ndarray,
    values: list[float | None],
    color: str,
    *,
    marker_size: float = 42,
) -> list[tuple[float, float]]:
    """Draw a line broken across ``None`` gaps; return the present points.

    The returned ``(x, value)`` list (gaps dropped) lets the caller place value
    labels wherever it likes — on every point, or only the endpoint.
    """
    present: list[tuple[float, float]] = []
    seg_x: list[float] = []
    seg_y: list[float] = []

    def flush() -> None:
        if seg_x:
            ax.plot(
                seg_x,
                seg_y,
                color=color,
                lw=2,
                zorder=5,
                solid_capstyle="round",
                solid_joinstyle="round",
            )
            ax.scatter(
                seg_x,
                seg_y,
                s=marker_size,
                color=color,
                zorder=6,
                edgecolors=SURFACE,
                linewidths=2,
            )

    for x, value in zip(xpos, values):
        if value is None:
            flush()
            seg_x, seg_y = [], []
        else:
            seg_x.append(float(x))
            seg_y.append(value)
            present.append((float(x), value))
    flush()
    return present


def draw_line_all_labels(
    ax: plt.Axes,
    xpos: np.ndarray,
    values: list[float | None],
    color: str,
    fmt: Callable[[float], str],
) -> None:
    """Draw a gap-aware line and label every present point via ``fmt``."""
    for x, value in draw_line(ax, xpos, values, color):
        _label(ax, x, value, fmt(value), color, dy=9)


def draw_adapter_line(
    ax: plt.Axes,
    xpos: np.ndarray,
    series: list[int | None],
    color: str,
) -> None:
    """Draw one table's adapter-count line, labeling only the last point.

    The lower panel's small 0..max scale makes per-point labels crowd, so only
    the endpoint is labeled, showing the current count.
    """
    values: list[float | None] = [None if v is None else float(v) for v in series]
    present: list[tuple[float, float]] = draw_line(
        ax, xpos, values, color, marker_size=30
    )
    if present:
        last_x, last_y = present[-1]
        _label(ax, last_x, last_y, f"{int(last_y)}", color, dy=8)


def style_axis(
    ax: plt.Axes,
    ylabel: str,
    yticks: np.ndarray,
    yticklabels: list[str],
    ylim_top: float,
) -> None:
    """Apply the shared recessive-grid / no-spine styling to one panel."""
    ax.set_ylim(0, ylim_top)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=SECONDARY, fontsize=10)
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)


# --------------------------------------------------------------------------- #
# Figure scaffolding
# --------------------------------------------------------------------------- #
@dataclass
class TopPanel:
    """Everything a variant must supply for the (differing) top panel.

    ``draw`` receives the top ``Axes`` and the per-table series lists and is
    responsible for all bars/lines/labels and the panel's y-axis (via
    :func:`style_axis`). The strings and legend handles position the shared
    title, legend and footer around it.
    """

    draw: Callable[[plt.Axes, np.ndarray, list[Record], list[Record]], None]
    title: str
    footer: str
    legend_handles: list[object]
    legend_ncol: int


def render(rows: list[dict[str, object]], out_path: Path, top: TopPanel) -> None:
    """Draw the two-panel figure and save it to ``out_path``.

    The top panel is delegated to ``top.draw``; the bottom panel (distinct
    adapters per snapshot) and all the surrounding chrome are shared.
    """
    dates: list[str] = [str(r["date"]) for r in rows]
    gen_series: list[Record] = [r["gen"] for r in rows]  # type: ignore[misc]
    emb_series: list[Record] = [r["emb"] for r in rows]  # type: ignore[misc]
    gen_adapters: list[int | None] = [r["gen_adapters"] for r in rows]  # type: ignore[misc]
    emb_adapters: list[int | None] = [r["emb_adapters"] for r in rows]  # type: ignore[misc]

    n: int = len(dates)
    x: np.ndarray = np.arange(n, dtype=float)

    fig, (ax, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(13.5, 7.2),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax_bot.set_facecolor(SURFACE)

    # --- top panel: variant-specific ---
    top.draw(ax, x, gen_series, emb_series)

    # --- bottom panel: adapter counts (shared) ---
    draw_adapter_line(ax_bot, x - OFFSET, gen_adapters, GEN_LINE)
    draw_adapter_line(ax_bot, x + OFFSET, emb_adapters, EMB_LINE)

    adapter_values: list[int] = [
        a for a in gen_adapters + emb_adapters if a is not None
    ]
    adapter_top: int = (
        int(np.ceil(max(adapter_values) / 5.0) * 5) if adapter_values else 5
    )
    style_axis(
        ax_bot,
        ylabel="adapters",
        yticks=np.linspace(0, adapter_top, 3),
        yticklabels=[f"{int(v)}" for v in np.linspace(0, adapter_top, 3)],
        ylim_top=adapter_top * 1.15,
    )
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(dates, color=MUTED, fontsize=9)

    # --- shared chrome ---
    fig.suptitle(
        top.title,
        x=0.012,
        y=0.985,
        ha="left",
        color=INK,
        fontsize=13.5,
        fontweight="bold",
    )
    ax.legend(
        handles=top.legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=top.legend_ncol,
        frameon=False,
        fontsize=8.5,
        labelcolor=SECONDARY,
        handlelength=1.4,
        columnspacing=1.8,
    )
    fig.text(0.5, 0.004, top.footer, ha="center", color=MUTED, fontsize=7.5)

    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.055, right=0.985)
    fig.savefig(out_path, facecolor=SURFACE)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser(description: str, default_output: str) -> argparse.ArgumentParser:
    """Build the shared ``--generative`` / ``--embedding`` / ``--output`` CLI."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--generative",
        type=Path,
        default=None,
        help="path to the generative-table CSV export; "
        "if omitted or missing, dummy data is generated",
    )
    parser.add_argument(
        "--embedding",
        type=Path,
        default=None,
        help="path to the embedding-table CSV export; "
        "if omitted or missing, dummy data is generated",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(default_output),
        help="output PNG path",
    )
    return parser


def load_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    """Resolve both tables (CSV or dummy) and merge them onto a common axis."""
    gen, gen_adapters = resolve_table(args.generative, dummy_scale=8100)
    emb, emb_adapters = resolve_table(args.embedding, dummy_scale=9900)
    return build_rows(gen, emb, gen_adapters, emb_adapters)
