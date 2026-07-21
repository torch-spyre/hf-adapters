"""Per-snapshot_date *cumulative* counts of models with an adapter and passing Spyre verification.

Usage::

    python tests/spyre/weekly_generation/snapshot_model_counts.py <csv_file>

Arguments:

* ``csv_file``  Path to an enriched CSV produced by ``add_past_rows.py``.
"""

from __future__ import annotations

import argparse

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


if __name__ == "__main__":
    main()
