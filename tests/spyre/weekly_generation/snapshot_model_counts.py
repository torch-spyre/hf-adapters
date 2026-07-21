"""Per-snapshot_date *cumulative* counts of models with an adapter and passing Spyre verification.

Usage::

    python tests/spyre/weekly_generation/snapshot_model_counts.py <csv_file>

Arguments:

* ``csv_file``  Path to an enriched CSV produced by ``add_past_rows.py``.
"""

from __future__ import annotations

import argparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv_file",
        metavar="csv_file",
        help="Path to the enriched CSV file to analyse.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    df = pd.read_csv(args.csv_file)

    # Boolean masks
    has_adapter = df["adapter_name"].notna() & (df["adapter_name"].str.strip() != "")
    passes_spyre = df["verified_on_spyre"].astype(str).str.strip().str.lower() == "true"

    df = df.assign(has_adapter=has_adapter, passes_spyre=passes_spyre)

    # Cumulative distinct models: for each snapshot_date, collect the set of
    # model_names seen in ALL rows up to and including that date.
    sorted_dates = sorted(df["snapshot_date"].unique())
    seen_models: set[str] = set()
    seen_adapter_models: set[str] = set()
    seen_adapter_names: set[str] = set()
    seen_spyre: set[str] = set()
    rows = []
    for d in sorted_dates:
        slice_ = df[df["snapshot_date"] == d]
        seen_models.update(slice_["model_name"].dropna().unique())
        seen_adapter_models.update(
            slice_.loc[slice_["has_adapter"], "model_name"].dropna().unique()
        )
        seen_adapter_names.update(
            slice_.loc[slice_["has_adapter"], "adapter_name"].dropna().unique()
        )
        seen_spyre.update(
            slice_.loc[slice_["passes_spyre"], "model_name"].dropna().unique()
        )
        rows.append(
            {
                "snapshot_date": d,
                "cumulative_models": len(seen_models),
                "cumulative_with_adapter": len(seen_adapter_models),
                "cumulative_adapter_names": len(seen_adapter_names),
                "cumulative_verified_on_spyre": len(seen_spyre),
            }
        )
    summary = pd.DataFrame(rows)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)

    print(f"\nSource: {args.csv_file}")
    print(
        f"Total rows: {len(df):,}  |  Distinct snapshot_dates: {summary['snapshot_date'].nunique()}\n"
    )
    print(
        summary[
            [
                "snapshot_date",
                "cumulative_models",
                "cumulative_with_adapter",
                "cumulative_adapter_names",
                "cumulative_verified_on_spyre",
            ]
        ].to_string(index=False)
    )

    # _plot_1(summary)
    # _plot_2(summary)
    _plot_3(summary, args.csv_file)


# def _plot_1(summary: pd.DataFrame) -> None:
#     """Render a two-panel matplotlib figure and show it.
#
#     Top panel  — step-line chart: cumulative models (reference), models with an
#                  adapter, and models verified on Spyre on the left Y-axis;
#                  distinct adapter names on the right Y-axis.
#     Bottom panel — horizontal funnel bar at the final snapshot date.
#     """
#     summary = summary.copy()
#     summary["snapshot_date"] = pd.to_datetime(summary["snapshot_date"])
#     dates = summary["snapshot_date"]
#
#     final = summary.iloc[-1]
#     total = int(final["cumulative_models"])
#     n_adapter = int(final["cumulative_with_adapter"])
#     n_spyre = int(final["cumulative_verified_on_spyre"])
#     n_adapter_names = int(final["cumulative_adapter_names"])
#
#     fig, (ax_top, ax_bot) = plt.subplots(
#         2,
#         1,
#         figsize=(11, 8),
#         gridspec_kw={"height_ratios": [3, 1]},
#     )
#     fig.suptitle(f"Spyre {mode} adapter coverage over time", fontsize=14, fontweight="bold")
#
#     # ── Top panel: step lines ────────────────────────────────────────────────
#     COLOR_TOTAL = "#aaaaaa"
#     COLOR_ADAPTER = "#3b82d4"
#     COLOR_SPYRE = "#22a06b"
#     COLOR_NAMES = "#e07b39"
#
#     ax_top.step(
#         dates,
#         summary["cumulative_models"],
#         where="post",
#         color=COLOR_TOTAL,
#         linewidth=1.5,
#         linestyle="--",
#         label=f"Total distinct models ({total:,})",
#     )
#     ax_top.step(
#         dates,
#         summary["cumulative_with_adapter"],
#         where="post",
#         color=COLOR_ADAPTER,
#         linewidth=2,
#         label=f"Models with an adapter ({n_adapter:,})",
#     )
#     ax_top.step(
#         dates,
#         summary["cumulative_verified_on_spyre"],
#         where="post",
#         color=COLOR_SPYRE,
#         linewidth=2,
#         label=f"Verified on Spyre ({n_spyre:,})",
#     )
#
#     ax_top.set_ylabel("Cumulative distinct models", color="#1f2328")
#     ax_top.tick_params(axis="y")
#     ax_top.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
#     ax_top.set_ylim(bottom=0)
#
#     # Right Y-axis: adapter names (different scale)
#     ax_r = ax_top.twinx()
#     ax_r.step(
#         dates,
#         summary["cumulative_adapter_names"],
#         where="post",
#         color=COLOR_NAMES,
#         linewidth=2,
#         linestyle=":",
#         label=f"Distinct adapter names ({n_adapter_names})",
#     )
#     ax_r.set_ylabel("Cumulative distinct adapter names", color=COLOR_NAMES)
#     ax_r.tick_params(axis="y", labelcolor=COLOR_NAMES)
#     ax_r.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
#     ax_r.set_ylim(bottom=0)
#
#     # Combine legends from both axes
#     lines_l, labels_l = ax_top.get_legend_handles_labels()
#     lines_r, labels_r = ax_r.get_legend_handles_labels()
#     ax_top.legend(
#         lines_l + lines_r,
#         labels_l + labels_r,
#         loc="upper left",
#         fontsize=9,
#         framealpha=0.85,
#     )
#
#     ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
#     ax_top.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
#     fig.autofmt_xdate(rotation=30, ha="right")
#     ax_top.set_title(
#         f"Cumulative coverage  ·  final snapshot: {final['snapshot_date'].strftime('%Y-%m-%d')}",
#         fontsize=10,
#         color="#57606a",
#     )
#     ax_top.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
#
#     # ── Bottom panel: funnel bar ─────────────────────────────────────────────
#     labels = ["Total models", "With adapter", "Verified on Spyre"]
#     values = [total, n_adapter, n_spyre]
#     colors = [COLOR_TOTAL, COLOR_ADAPTER, COLOR_SPYRE]
#     y_pos = [2, 1, 0]
#
#     bars = ax_bot.barh(y_pos, values, color=colors, height=0.5)
#     ax_bot.set_yticks(y_pos)
#     ax_bot.set_yticklabels(labels)
#     ax_bot.set_xlabel("Distinct model count")
#     ax_bot.set_title(
#         f"Funnel at final snapshot ({final['snapshot_date'].strftime('%Y-%m-%d')})",
#         fontsize=10,
#         color="#57606a",
#     )
#     ax_bot.set_xlim(0, total * 1.15)
#     ax_bot.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
#     ax_bot.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
#
#     for bar, val, total_val in zip(bars, values, [total, total, total]):
#         pct = val / total_val * 100
#         ax_bot.text(
#             val + total * 0.01,
#             bar.get_y() + bar.get_height() / 2,
#             f"{val:,}  ({pct:.1f}%)",
#             va="center",
#             fontsize=9,
#         )
#
#     plt.tight_layout()
#     plt.show()
#
#
# def _plot_2(summary: pd.DataFrame) -> None:
#     """Two-panel figure.
#
#     Top panel  — grouped bar chart: for each snapshot_date, two side-by-side
#                  bars showing % of total models with an adapter and % verified
#                  on Spyre. Dates are shown on the X axis.
#     Bottom panel — horizontal funnel bar at the final snapshot date (same as
#                    _plot_1).
#     """
#     summary = summary.copy()
#     summary["snapshot_date"] = pd.to_datetime(summary["snapshot_date"])
#
#     total = summary["cumulative_models"]
#     pct_adapter = summary["cumulative_with_adapter"] / total * 100
#     pct_spyre   = summary["cumulative_verified_on_spyre"] / total * 100
#
#     dates = summary["snapshot_date"]
#     n = len(dates)
#     # Numeric positions for the groups
#     x = range(n)
#     bar_w = 0.35
#
#     COLOR_ADAPTER = "#3b82d4"
#     COLOR_SPYRE   = "#22a06b"
#     COLOR_TOTAL   = "#aaaaaa"
#
#     final = summary.iloc[-1]
#     total_final    = int(final["cumulative_models"])
#     n_adapter      = int(final["cumulative_with_adapter"])
#     n_spyre_final  = int(final["cumulative_verified_on_spyre"])
#
#     fig, (ax_top, ax_bot) = plt.subplots(
#         2, 1,
#         figsize=(13, 8),
#         gridspec_kw={"height_ratios": [3, 1]},
#     )
#     fig.suptitle("Spyre adapter coverage over time", fontsize=14, fontweight="bold")
#
#     # ── Top panel: grouped bars ──────────────────────────────────────────────
#     bars_a = ax_top.bar(
#         [i - bar_w / 2 for i in x], pct_adapter,
#         width=bar_w, color=COLOR_ADAPTER, alpha=0.85,
#         label="% models with adapter",
#     )
#     bars_s = ax_top.bar(
#         [i + bar_w / 2 for i in x], pct_spyre,
#         width=bar_w, color=COLOR_SPYRE, alpha=0.85,
#         label="% models verified on Spyre",
#     )
#
#     ax_top.set_ylabel("% of total distinct models")
#     ax_top.set_ylim(0, max(pct_adapter.max(), pct_spyre.max()) * 1.18)
#     ax_top.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
#     ax_top.set_xticks(list(x))
#     ax_top.set_xticklabels(
#         [d.strftime("%Y-%m-%d") for d in dates],
#         rotation=40, ha="right", fontsize=8,
#     )
#     ax_top.set_title(
#         f"Cumulative coverage as % of total models ({total_final:,})  ·  "
#         f"final snapshot: {final['snapshot_date'].strftime('%Y-%m-%d')}",
#         fontsize=10, color="#57606a",
#     )
#     ax_top.legend(fontsize=9, framealpha=0.85)
#     ax_top.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
#
#     # Annotate each bar with its percentage value
#     for bar in bars_a:
#         h = bar.get_height()
#         if h > 0:
#             ax_top.text(
#                 bar.get_x() + bar.get_width() / 2, h + 0.3,
#                 f"{h:.1f}%", ha="center", va="bottom", fontsize=6.5,
#                 color=COLOR_ADAPTER,
#             )
#     for bar in bars_s:
#         h = bar.get_height()
#         if h > 0:
#             ax_top.text(
#                 bar.get_x() + bar.get_width() / 2, h + 0.3,
#                 f"{h:.1f}%", ha="center", va="bottom", fontsize=6.5,
#                 color=COLOR_SPYRE,
#             )
#
#     # ── Bottom panel: horizontal funnel bar ──────────────────────────────────
#     labels = ["Total models", "With adapter", "Verified on Spyre"]
#     values = [total_final, n_adapter, n_spyre_final]
#     colors = [COLOR_TOTAL, COLOR_ADAPTER, COLOR_SPYRE]
#
#     bars_h = ax_bot.barh([2, 1, 0], values, color=colors, height=0.5)
#     ax_bot.set_yticks([2, 1, 0])
#     ax_bot.set_yticklabels(labels)
#     ax_bot.set_xlabel("Distinct model count")
#     ax_bot.set_title(
#         f"Funnel at final snapshot ({final['snapshot_date'].strftime('%Y-%m-%d')})",
#         fontsize=10, color="#57606a",
#     )
#     ax_bot.set_xlim(0, total_final * 1.15)
#     ax_bot.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
#     ax_bot.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
#
#     for bar, val in zip(bars_h, values):
#         pct = val / total_final * 100
#         ax_bot.text(
#             val + total_final * 0.01, bar.get_y() + bar.get_height() / 2,
#             f"{val:,}  ({pct:.1f}%)",
#             va="center", fontsize=9,
#         )
#
#     plt.tight_layout()
#     plt.show()
#


def _plot_3(summary: pd.DataFrame, filename: str) -> None:
    """Three-panel figure combining _plot_1 and _plot_2.

    Top panel    — step-line chart (same as _plot_1): cumulative model counts
                   on the left Y-axis; distinct adapter names on the right.
    Middle panel — grouped bar chart (same as _plot_2): % models with adapter
                   and % verified on Spyre per snapshot_date.
    Bottom panel — horizontal funnel bar at the final snapshot date.
    """
    summary = summary.copy()
    summary["snapshot_date"] = pd.to_datetime(summary["snapshot_date"])
    dates = summary["snapshot_date"]
    n = len(dates)
    x = range(n)
    bar_w = 0.35

    COLOR_TOTAL = "#aaaaaa"
    COLOR_ADAPTER = "#3b82d4"
    COLOR_SPYRE = "#22a06b"
    COLOR_NAMES = "#e07b39"

    final = summary.iloc[-1]
    total_final = int(final["cumulative_models"])
    n_adapter = int(final["cumulative_with_adapter"])
    n_spyre_final = int(final["cumulative_verified_on_spyre"])
    n_names_final = int(final["cumulative_adapter_names"])

    total = summary["cumulative_models"]
    pct_adapter = summary["cumulative_with_adapter"] / total * 100
    pct_spyre = summary["cumulative_verified_on_spyre"] / total * 100

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3,
        1,
        figsize=(13, 12),
        gridspec_kw={"height_ratios": [3, 3, 1]},
    )
    mode: str = "embedding" if "embedding" in filename else "generative"
    fig.suptitle(
        f"Spyre {mode} adapter coverage over time", fontsize=14, fontweight="bold"
    )

    # ── Top panel: step lines (from _plot_1) ────────────────────────────────
    ax_top.step(
        dates,
        summary["cumulative_models"],
        where="post",
        color=COLOR_TOTAL,
        linewidth=1.5,
        linestyle="--",
        label=f"Total models ({total_final:,})",
    )
    ax_top.step(
        dates,
        summary["cumulative_with_adapter"],
        where="post",
        color=COLOR_ADAPTER,
        linewidth=2,
        label=f"Models with an adapter ({n_adapter:,})",
    )
    ax_top.step(
        dates,
        summary["cumulative_verified_on_spyre"],
        where="post",
        color=COLOR_SPYRE,
        linewidth=2,
        label=f"Green test on Spyre ({n_spyre_final:,})",
    )

    ax_top.set_ylabel("Cumulative distinct models")
    ax_top.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_top.set_ylim(bottom=0)

    ax_r = ax_top.twinx()
    ax_r.step(
        dates,
        summary["cumulative_adapter_names"],
        where="post",
        color=COLOR_NAMES,
        linewidth=2,
        linestyle=":",
        label=f"Distinct adapter ({n_names_final})",
    )
    ax_r.set_ylabel("Cumulative distinct adapter", color=COLOR_NAMES)
    ax_r.tick_params(axis="y", labelcolor=COLOR_NAMES)
    ax_r.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_r.set_ylim(bottom=0)

    lines_l, labels_l = ax_top.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax_top.legend(
        lines_l + lines_r,
        labels_l + labels_r,
        loc="upper left",
        fontsize=9,
        framealpha=0.85,
    )
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_top.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax_top.set_title(
        f"Final snapshot: {final['snapshot_date'].strftime('%Y-%m-%d')}",
        fontsize=10,
        color="#57606a",
    )
    ax_top.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

    # ── Middle panel: grouped bars (from _plot_2) ────────────────────────────
    bars_a = ax_mid.bar(
        [i - bar_w / 2 for i in x],
        pct_adapter,
        width=bar_w,
        color=COLOR_ADAPTER,
        alpha=0.85,
        label="% models with adapter",
    )
    bars_s = ax_mid.bar(
        [i + bar_w / 2 for i in x],
        pct_spyre,
        width=bar_w,
        color=COLOR_SPYRE,
        alpha=0.85,
        label="% models - Green test on Spyre",
    )

    ax_mid.set_ylabel("% of total distinct models")
    ax_mid.set_ylim(0, max(pct_adapter.max(), pct_spyre.max()) * 1.18)
    ax_mid.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax_mid.set_xticks(list(x))
    ax_mid.set_xticklabels(
        [d.strftime("%Y-%m-%d") for d in dates],
        rotation=40,
        ha="right",
        fontsize=8,
    )
    ax_mid.set_title(
        "Cumulative coverage as % of total models", fontsize=10, color="#57606a"
    )
    ax_mid.legend(fontsize=9, framealpha=0.85)
    ax_mid.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

    for bar in bars_a:
        h = bar.get_height()
        if h > 0:
            ax_mid.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.3,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=COLOR_ADAPTER,
            )
    for bar in bars_s:
        h = bar.get_height()
        if h > 0:
            ax_mid.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.3,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=COLOR_SPYRE,
            )

    # ── Bottom panel: horizontal funnel bar ──────────────────────────────────
    labels = ["Total models", "With adapter", "Green test on Spyre"]
    values = [total_final, n_adapter, n_spyre_final]
    colors = [COLOR_TOTAL, COLOR_ADAPTER, COLOR_SPYRE]

    bars_h = ax_bot.barh([2, 1, 0], values, color=colors, height=0.5)
    ax_bot.set_yticks([2, 1, 0])
    ax_bot.set_yticklabels(labels)
    ax_bot.set_xlabel("Distinct model count")
    ax_bot.set_title(
        f"Funnel at final snapshot ({final['snapshot_date'].strftime('%Y-%m-%d')})",
        fontsize=10,
        color="#57606a",
    )
    ax_bot.set_xlim(0, total_final * 1.15)
    ax_bot.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_bot.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)

    for bar, val in zip(bars_h, values):
        pct = val / total_final * 100
        ax_bot.text(
            val + total_final * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}  ({pct:.1f}%)",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    output_file_name = filename.replace(".csv", ".png")
    plt.savefig(fname=output_file_name)
    plt.show()


if __name__ == "__main__":
    main()
