#!/usr/bin/env python3
"""Render the Spyre support-trend chart (generative vs embedding) to a PNG.

Absolute-count variant. For each table, each snapshot is a stacked column —
verified (dark) over not-verified (light) — whose height is the total models
tested; a verified-count line with bold value labels rides on top. The lower
panel shows the number of distinct adapters per snapshot.

Colors: orange = generative, blue = embedding; dark shade = verified, light =
not verified. All shared logic lives in :mod:`spyre_support_common`; this file
only defines the absolute-count top panel.
"""

from __future__ import annotations

import numpy as np
import spyre_support_common as common
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from spyre_support_common import (
    BAR_WIDTH,
    EMB_LINE,
    EMB_NOT,
    EMB_VER,
    GEN_LINE,
    GEN_NOT,
    GEN_VER,
    OFFSET,
    SURFACE,
    Record,
    TopPanel,
    draw_line_all_labels,
    style_axis,
)


def _draw_stacked_bars(
    ax: Axes,
    xpos: np.ndarray,
    series: list[Record],
    color_verified: str,
    color_not: str,
) -> None:
    """Draw one table's stacked columns (verified bottom, not-verified top)."""
    for x, rec in zip(xpos, series):
        if rec is None:
            continue
        tested, verified = rec
        ax.bar(x, verified, BAR_WIDTH, color=color_verified, zorder=3)
        ax.bar(
            x, tested - verified, BAR_WIDTH, bottom=verified, color=color_not, zorder=3
        )


def _draw_top(
    ax: Axes,
    x: np.ndarray,
    gen_series: list[Record],
    emb_series: list[Record],
) -> None:
    """Top panel: stacked bars + verified-count lines on a ~10k scale."""
    gen_verified: list[float | None] = [None if r is None else r[1] for r in gen_series]
    emb_verified: list[float | None] = [None if r is None else r[1] for r in emb_series]

    _draw_stacked_bars(ax, x - OFFSET, gen_series, GEN_VER, GEN_NOT)
    _draw_stacked_bars(ax, x + OFFSET, emb_series, EMB_VER, EMB_NOT)
    draw_line_all_labels(
        ax, x - OFFSET, gen_verified, GEN_LINE, lambda v: f"{int(v):,}"
    )
    draw_line_all_labels(
        ax, x + OFFSET, emb_verified, EMB_LINE, lambda v: f"{int(v):,}"
    )

    tested_values: list[int] = [
        rec[0] for rec in gen_series + emb_series if rec is not None
    ]
    nice_top: int = (
        int(np.ceil(max(tested_values) / 1000.0) * 1000) if tested_values else 1000
    )
    style_axis(
        ax,
        ylabel="models tested",
        yticks=np.linspace(0, nice_top, 6),
        yticklabels=[f"{int(v):,}" for v in np.linspace(0, nice_top, 6)],
        ylim_top=nice_top * 1.07,
    )


def _top_panel() -> TopPanel:
    """Bundle the absolute-count top panel: drawing, text and legend."""
    # One row per table (Generative, Embedding). matplotlib fills a legend
    # column-major, so the handles are interleaved (gen, emb, ...) with ncol=3.
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
    return TopPanel(
        draw=_draw_top,
        title="Spyre support by snapshot date — generative vs embedding",
        footer=(
            "Top: verified + not-verified = total tested (column height). "
            "Bottom: distinct adapters per snapshot (same colors as the "
            "verified-count lines)."
        ),
        legend_handles=[h for pair in zip(gen_row, emb_row) for h in pair],
        legend_ncol=3,
    )


def main() -> None:
    parser = common.build_arg_parser(__doc__, default_output="spyre-support-final.png")
    args = parser.parse_args()
    rows: list[dict[str, object]] = common.load_rows(args)
    common.render(rows, args.output, _top_panel())


if __name__ == "__main__":
    main()
