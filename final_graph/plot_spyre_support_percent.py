#!/usr/bin/env python3
"""Render the Spyre pass-rate chart as PERCENTAGES (generative vs embedding).

Percentage variant of :mod:`plot_spyre_support`. For each table, each snapshot
is a single bar showing the PASS RATE — verified models as a percentage of the
total tested that snapshot — with a pass-rate line and bold ``NN%`` labels on
top. The lower panel still shows the absolute number of distinct adapters per
snapshot (counts, not percentages).

Differences from the absolute-count variant:
  1. No "not verified" bar — only the verified share is drawn.
  2. The y-axis and all labels are percentages (verified / total tested * 100).
  3. The default output PNG is named accordingly.

All shared logic lives in :mod:`spyre_support_common`; this file only defines
the percentage top panel.
"""

from __future__ import annotations

import numpy as np
import spyre_support_common as common
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from spyre_support_common import (
    BAR_WIDTH,
    EMB_LINE,
    EMB_VER,
    GEN_LINE,
    GEN_VER,
    OFFSET,
    SURFACE,
    Record,
    TopPanel,
    draw_line_all_labels,
    style_axis,
)


def _verified_percent(rec: Record) -> float | None:
    """``(tested, verified)`` -> verified share in percent, or ``None``.

    Returns ``None`` when the record is absent or the tested count is zero
    (so a missing/empty snapshot breaks the bar and line rather than dividing
    by zero or implying a real 0%).
    """
    if rec is None:
        return None
    tested, verified = rec
    if tested == 0:
        return None
    return verified / tested * 100.0


def _draw_percent_bars(
    ax: Axes,
    xpos: np.ndarray,
    percents: list[float | None],
    color: str,
) -> None:
    """Draw one table's pass-rate bars (a single bar per snapshot)."""
    for x, pct in zip(xpos, percents):
        if pct is None:
            continue
        ax.bar(x, pct, BAR_WIDTH, color=color, zorder=3)


def _draw_top(
    ax: Axes,
    x: np.ndarray,
    gen_series: list[Record],
    emb_series: list[Record],
) -> None:
    """Top panel: pass-rate bars + pass-rate lines on a 0..100% scale."""
    gen_pct: list[float | None] = [_verified_percent(rec) for rec in gen_series]
    emb_pct: list[float | None] = [_verified_percent(rec) for rec in emb_series]

    _draw_percent_bars(ax, x - OFFSET, gen_pct, GEN_VER)
    _draw_percent_bars(ax, x + OFFSET, emb_pct, EMB_VER)
    draw_line_all_labels(ax, x - OFFSET, gen_pct, GEN_LINE, lambda v: f"{v:.0f}%")
    draw_line_all_labels(ax, x + OFFSET, emb_pct, EMB_LINE, lambda v: f"{v:.0f}%")

    style_axis(
        ax,
        ylabel="pass rate",
        yticks=np.linspace(0, 100, 6),
        yticklabels=[f"{int(v)}%" for v in np.linspace(0, 100, 6)],
        ylim_top=108,  # headroom above 100% for the top labels
    )


def _top_panel() -> TopPanel:
    """Bundle the percentage top panel: drawing, text and legend.

    The bar and line encode the same pass-rate value (the bar top meets the
    line), so the legend shows only the two lines to avoid a redundant swatch.
    """
    gen_row: list[object] = [
        Line2D(
            [0],
            [0],
            color=GEN_LINE,
            lw=2,
            marker="o",
            mfc=GEN_LINE,
            mec=SURFACE,
            label="Generative pass rate",
        ),
    ]
    emb_row: list[object] = [
        Line2D(
            [0],
            [0],
            color=EMB_LINE,
            lw=2,
            marker="o",
            mfc=EMB_LINE,
            mec=SURFACE,
            label="Embedding pass rate",
        ),
    ]
    return TopPanel(
        draw=_draw_top,
        title="Spyre pass rate by snapshot date — generative vs embedding",
        footer=(
            "Top: verified models as a percent of total tested that snapshot. "
            "Bottom: distinct adapters per snapshot (same colors as the "
            "pass-rate lines)."
        ),
        legend_handles=[h for pair in zip(gen_row, emb_row) for h in pair],
        legend_ncol=2,
    )


def main() -> None:
    parser = common.build_arg_parser(
        __doc__, default_output="spyre-support-percent.png"
    )
    args = parser.parse_args()
    rows: list[dict[str, object]] = common.load_rows(args)
    common.render(rows, args.output, _top_panel())


if __name__ == "__main__":
    main()
