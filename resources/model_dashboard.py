"""Dashboard for browsing top HuggingFace generative models and Spyre adapter coverage."""

import os

import pandas as pd
from nicegui import ui

CSV_PATH = os.path.join(os.path.dirname(__file__), "top_generative_models.csv")

COVERED_TYPES = {
    "granite",
    "granite4_vision",
    "granitemoehybrid",
    "llama",
    "mistral",
    "olmo",
    "olmo2",
    "phi3",
    "qwen2",
    "qwen3",
    "smollm3",
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["model_type"] = df["model_type"].fillna("")
    df["parameters"] = pd.to_numeric(df["parameters"], errors="coerce")
    df["downloads"] = (
        pd.to_numeric(df["downloads"], errors="coerce").fillna(0).astype(int)
    )
    df["covered"] = df["model_type"].isin(COVERED_TYPES)
    return df


df = load_data()

all_types = sorted(df[df["model_type"] != ""]["model_type"].unique().tolist())
distinct_types = set(all_types)
covered_in_csv = distinct_types & COVERED_TYPES
type_coverage_pct = (
    len(covered_in_csv) / len(distinct_types) * 100 if distinct_types else 0
)

covered_models = df[df["covered"]].shape[0]
total_with_type = df[df["model_type"] != ""].shape[0]
model_coverage_pct = covered_models / total_with_type * 100 if total_with_type else 0


def format_params(val):
    if pd.isna(val):
        return ""
    if val >= 1e9:
        return f"{val / 1e9:.1f}B"
    if val >= 1e6:
        return f"{val / 1e6:.0f}M"
    return f"{val:.0f}"


def format_downloads(val):
    if val >= 1e6:
        return f"{val / 1e6:.1f}M"
    if val >= 1e3:
        return f"{val / 1e3:.0f}K"
    return str(val)


def build_table_rows(filtered: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in filtered.iterrows():
        rows.append(
            {
                "rank": int(r["rank"]),
                "model_id": r["model_id"],
                "downloads": format_downloads(r["downloads"]),
                "downloads_raw": int(r["downloads"]),
                "model_type": r["model_type"] or "(unknown)",
                "architectures": (
                    r["architectures"] if pd.notna(r["architectures"]) else ""
                ),
                "parameters": format_params(r["parameters"]),
                "covered": "Yes" if r["covered"] else "No",
            }
        )
    return rows


columns = [
    {
        "name": "rank",
        "label": "Rank",
        "field": "rank",
        "sortable": True,
        "align": "left",
    },
    {
        "name": "model_id",
        "label": "Model",
        "field": "model_id",
        "sortable": True,
        "align": "left",
    },
    {
        "name": "downloads",
        "label": "Downloads",
        "field": "downloads",
        "sortable": True,
        "align": "right",
    },
    {
        "name": "model_type",
        "label": "Type",
        "field": "model_type",
        "sortable": True,
        "align": "left",
    },
    {
        "name": "parameters",
        "label": "Params",
        "field": "parameters",
        "sortable": True,
        "align": "right",
    },
    {
        "name": "architectures",
        "label": "Architecture",
        "field": "architectures",
        "sortable": True,
        "align": "left",
    },
    {
        "name": "covered",
        "label": "Covered",
        "field": "covered",
        "sortable": True,
        "align": "center",
    },
]


def apply_filters():
    filtered = df.copy()

    search = search_input.value.strip().lower() if search_input.value else ""
    if search:
        filtered = filtered[
            filtered["model_id"].str.lower().str.contains(search, na=False)
        ]

    selected_types = type_select.value
    if selected_types:
        filtered = filtered[filtered["model_type"].isin(selected_types)]

    cov = coverage_toggle.value
    if cov == "Covered":
        filtered = filtered[filtered["covered"]]
    elif cov == "Uncovered":
        filtered = filtered[~filtered["covered"] & (filtered["model_type"] != "")]

    table.rows = build_table_rows(filtered)
    results_label.set_text(f"{len(filtered)} models shown")


ui.dark_mode(True)

with ui.header().classes("items-center justify-between"):
    ui.label("HF Generative Models — Spyre Adapter Coverage").classes(
        "text-h5 font-bold"
    )

with ui.row().classes("w-full justify-center gap-4 px-4 mt-4"):
    with ui.card().classes("p-4"):
        ui.label("Model Type Coverage").classes("text-subtitle2 text-grey")
        ui.label(
            f"{len(covered_in_csv)} / {len(distinct_types)} types ({type_coverage_pct:.0f}%)"
        ).classes("text-h6 font-bold text-green")
    with ui.card().classes("p-4"):
        ui.label("Model Coverage (by count)").classes("text-subtitle2 text-grey")
        ui.label(
            f"{covered_models} / {total_with_type} models ({model_coverage_pct:.0f}%)"
        ).classes("text-h6 font-bold text-blue")
    with ui.card().classes("p-4"):
        ui.label("Total in Dataset").classes("text-subtitle2 text-grey")
        ui.label(f"{len(df)} models").classes("text-h6 font-bold")

with ui.card().classes("w-full mx-4 mt-4 p-4"):
    ui.label("Filters").classes("text-subtitle1 font-bold mb-2")
    with ui.row().classes("w-full items-end gap-4 flex-wrap"):
        search_input = ui.input(
            label="Search model name", placeholder="e.g. llama, qwen"
        ).classes("min-w-[200px]")
        search_input.on("keydown.enter", lambda: apply_filters())
        search_input.on("update:model-value", lambda: apply_filters())

        type_select = (
            ui.select(
                options=all_types,
                label="Filter by model_type",
                multiple=True,
                clearable=True,
            )
            .classes("min-w-[250px]")
            .props("use-chips")
        )
        type_select.on("update:model-value", lambda: apply_filters())

        coverage_toggle = ui.toggle(["All", "Covered", "Uncovered"], value="All").props(
            "rounded"
        )
        coverage_toggle.on("update:model-value", lambda: apply_filters())

        ui.button("Reset", on_click=lambda: reset_filters(), icon="refresh").props(
            "flat"
        )

    results_label = ui.label(f"{len(df)} models shown").classes("mt-2 text-caption")


def reset_filters():
    search_input.set_value("")
    type_select.set_value([])
    coverage_toggle.set_value("All")
    apply_filters()


with ui.card().classes("w-full mx-4 mt-4 p-2"):
    table = ui.table(
        columns=columns,
        rows=build_table_rows(df),
        row_key="rank",
        pagination={"rowsPerPage": 25, "sortBy": "rank"},
    ).classes("w-full")

    table.add_slot(
        "body-cell-covered",
        r"""
        <q-td :props="props">
            <q-badge :color="props.value === 'Yes' ? 'green' : 'grey'" :label="props.value" />
        </q-td>
        """,
    )
    table.add_slot(
        "body-cell-model_id",
        r"""
        <q-td :props="props">
            <span class="font-mono text-xs">{{ props.value }}</span>
        </q-td>
        """,
    )

with ui.expansion("Coverage by Model Type", icon="analytics").classes(
    "w-full mx-4 mt-4"
):
    type_counts = (
        df[df["model_type"] != ""]
        .groupby("model_type")
        .agg(
            count=("model_id", "size"),
            total_downloads=("downloads", "sum"),
        )
        .sort_values("total_downloads", ascending=False)
        .reset_index()
    )

    cov_columns = [
        {
            "name": "model_type",
            "label": "Model Type",
            "field": "model_type",
            "sortable": True,
            "align": "left",
        },
        {
            "name": "count",
            "label": "Models",
            "field": "count",
            "sortable": True,
            "align": "right",
        },
        {
            "name": "total_downloads",
            "label": "Total Downloads",
            "field": "total_downloads",
            "sortable": True,
            "align": "right",
        },
        {
            "name": "status",
            "label": "Adapter",
            "field": "status",
            "sortable": True,
            "align": "center",
        },
    ]
    cov_rows = []
    for _, r in type_counts.iterrows():
        cov_rows.append(
            {
                "model_type": r["model_type"],
                "count": int(r["count"]),
                "total_downloads": format_downloads(int(r["total_downloads"])),
                "status": (
                    "Supported" if r["model_type"] in COVERED_TYPES else "Not yet"
                ),
            }
        )

    cov_table = ui.table(
        columns=cov_columns,
        rows=cov_rows,
        row_key="model_type",
        pagination={"rowsPerPage": 50},
    ).classes("w-full")
    cov_table.add_slot(
        "body-cell-status",
        r"""
        <q-td :props="props">
            <q-badge :color="props.value === 'Supported' ? 'green' : 'orange'" :label="props.value" />
        </q-td>
        """,
    )

ui.run(title="HF Models — Spyre Coverage", port=8080)
