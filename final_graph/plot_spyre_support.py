#!/usr/bin/env python3
"""Render the Spyre support-trend chart (generative vs embedding) to a PNG.

Reads the two ``*_model_spyre_support.csv`` exports (same schema as
``tests/spyre/weekly_generation/table_schema.py``), aggregates per
``snapshot_date`` — counting, per table, how many models are
``verified_on_spyre`` True vs False — and draws paired stacked columns with a
verified-count line and bold value labels.

Colors: orange = generative, blue = embedding; dark shade = verified, light =
not verified. Column height = total models tested that snapshot.

The CSV has no header row and carries two extra trailing columns (a UUID and an
ingest timestamp) beyond the 15 schema columns; only ``model_name`` (col 0),
``snapshot_date`` (col 4) and ``verified_on_spyre`` (col 7) are read, so the
extras are harmless.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Column indices within a CSV row (see table_schema.TABLE_COLUMNS).
COL_MODEL: int = 0
COL_SNAPSHOT: int = 4
COL_VERIFIED_SPYRE: int = 7

# --- palette (light-mode blue + orange) ---
GEN_VER: str = "#c24e1f"  # generative, verified   (dark orange)
GEN_NOT: str = "#f6cdb6"  # generative, not verified (light orange)
GEN_LINE: str = "#8f3912"  # generative verified-count line + labels
EMB_VER: str = "#1c5cab"  # embedding, verified     (dark blue)
EMB_NOT: str = "#bcd6f6"  # embedding, not verified  (light blue)
EMB_LINE: str = "#13406f"  # embedding verified-count line + labels

INK: str = "#0b0b0b"
SECONDARY: str = "#52514e"
MUTED: str = "#898781"
GRID: str = "#e1e0d9"
BASELINE: str = "#c3c2b7"
SURFACE: str = "#fcfcfb"


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


def resolve_table(path: Path | None, dummy_scale: int) -> dict[str, tuple[int, int]]:
    """Aggregate a table from its CSV, or fall back to dummy data.

    Returns dummy data when ``path`` is ``None`` (flag not provided) or points
    at a file that does not exist; otherwise reads and aggregates the CSV.
    """
    if path is None or not path.exists():
        reason: str = "no path provided" if path is None else f"file not found: {path}"
        print(f"[dummy] {reason} — generating dummy data (scale={dummy_scale:,})")
        return dummy_aggregate(dummy_scale)
    return aggregate(path)


def _short(date: str) -> str:
    """``2026-09-26`` -> ``26-09-26`` for compact x-axis ticks."""
    year, month, day = date.split("-")
    return f"{year[2:]}-{month}-{day}"


def build_rows(
    gen: dict[str, tuple[int, int]],
    emb: dict[str, tuple[int, int]],
) -> list[dict[str, object]]:
    """Merge the two tables onto the union of their snapshot dates.

    Each row is ``{"date", "gen": (tested, verified) | None, "emb": ...}``;
    a table absent on a date yields ``None`` so its column and line break
    rather than interpolating across a snapshot that was never measured.
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
        rows.append(row)
    return rows


def _draw_bars(
    ax: plt.Axes,
    xpos: np.ndarray,
    series: list[tuple[int, int] | None],
    color_verified: str,
    color_not: str,
    bar_width: float,
) -> None:
    """Draw one table's stacked columns (verified bottom, not-verified top)."""
    for x, rec in zip(xpos, series):
        if rec is None:
            continue
        tested, verified = rec
        ax.bar(x, verified, bar_width, color=color_verified, zorder=3)
        ax.bar(
            x, tested - verified, bar_width, bottom=verified, color=color_not, zorder=3
        )


def _draw_line_and_labels(
    ax: plt.Axes,
    xpos: np.ndarray,
    series: list[tuple[int, int] | None],
    color: str,
) -> None:
    """Draw the verified-count line (broken across gaps) and bold labels."""
    seg_x: list[float] = []
    seg_y: list[int] = []

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
                s=42,
                color=color,
                zorder=6,
                edgecolors=SURFACE,
                linewidths=2,
            )

    for x, rec in zip(xpos, series):
        if rec is None:
            flush()
            seg_x, seg_y = [], []
        else:
            seg_x.append(float(x))
            seg_y.append(rec[1])
    flush()

    for x, rec in zip(xpos, series):
        if rec is None:
            continue
        ax.annotate(
            f"{rec[1]:,}",
            (x, rec[1]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=color,
            zorder=7,
            path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)],
        )


def render(rows: list[dict[str, object]], out_path: Path) -> None:
    """Draw the full chart for the merged rows and save it to ``out_path``."""
    dates: list[str] = [str(r["date"]) for r in rows]
    gen_series: list[tuple[int, int] | None] = [r["gen"] for r in rows]  # type: ignore[misc]
    emb_series: list[tuple[int, int] | None] = [r["emb"] for r in rows]  # type: ignore[misc]

    n: int = len(dates)
    x: np.ndarray = np.arange(n, dtype=float)
    bar_width: float = 0.34
    offset: float = 0.19  # each column sits ±offset from the date center

    fig, ax = plt.subplots(figsize=(13.5, 5.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    _draw_bars(ax, x - offset, gen_series, GEN_VER, GEN_NOT, bar_width)
    _draw_bars(ax, x + offset, emb_series, EMB_VER, EMB_NOT, bar_width)
    _draw_line_and_labels(ax, x - offset, gen_series, GEN_LINE)
    _draw_line_and_labels(ax, x + offset, emb_series, EMB_LINE)

    tested_values: list[int] = [
        rec[0] for rec in gen_series + emb_series if rec is not None
    ]
    nice_top: int = int(np.ceil(max(tested_values) / 1000.0) * 1000)
    ax.set_ylim(0, nice_top * 1.07)
    ax.set_yticks(np.linspace(0, nice_top, 6))
    ax.set_yticklabels(
        [f"{int(v):,}" for v in np.linspace(0, nice_top, 6)], color=MUTED, fontsize=9
    )
    ax.set_xticks(x)
    ax.set_xticklabels(dates, color=MUTED, fontsize=9)
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)

    fig.suptitle(
        "Spyre support by snapshot date — generative vs embedding",
        x=0.012,
        y=0.985,
        ha="left",
        color=INK,
        fontsize=13.5,
        fontweight="bold",
    )

    # One row per table: Generative on top, Embedding below. matplotlib fills a
    # legend column-major (down each column), so to get row-grouped output the
    # handles are interleaved (gen, emb, gen, emb, ...) with ncol=3.
    gen_row: list[object] = [
        Patch(facecolor=GEN_VER, label="Generative — verified"),
        Patch(facecolor=GEN_NOT, label="Generative — not verified"),
        Line2D(
            [0],
            [0],
            color=GEN_LINE,
            lw=2,
            marker="o",
            mfc=GEN_LINE,
            mec=SURFACE,
            label="Generative verified count",
        ),
    ]
    emb_row: list[object] = [
        Patch(facecolor=EMB_VER, label="Embedding — verified"),
        Patch(facecolor=EMB_NOT, label="Embedding — not verified"),
        Line2D(
            [0],
            [0],
            color=EMB_LINE,
            lw=2,
            marker="o",
            mfc=EMB_LINE,
            mec=SURFACE,
            label="Embedding verified count",
        ),
    ]
    legend_elems: list[object] = [
        handle for pair in zip(gen_row, emb_row) for handle in pair
    ]
    ax.legend(
        handles=legend_elems,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        labelcolor=SECONDARY,
        handlelength=1.4,
        columnspacing=1.8,
    )

    fig.text(
        0.5,
        0.004,
        "Verified + not-verified = total tested (column height).",
        ha="center",
        color=MUTED,
        fontsize=7.5,
    )

    fig.subplots_adjust(top=0.82, bottom=0.10, left=0.055, right=0.985)
    fig.savefig(out_path, facecolor=SURFACE)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=Path("spyre-support-final.png"),
        help="output PNG path",
    )
    args = parser.parse_args()

    # Fall back to dummy data per-table when its path is not provided or missing.
    # Distinct scales give the two tables visibly different populations.
    gen: dict[str, tuple[int, int]] = resolve_table(args.generative, dummy_scale=8100)
    emb: dict[str, tuple[int, int]] = resolve_table(args.embedding, dummy_scale=9900)
    rows: list[dict[str, object]] = build_rows(gen, emb)
    render(rows, args.output)


if __name__ == "__main__":
    main()
