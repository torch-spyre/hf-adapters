#!/usr/bin/env python3
"""
Interactive UI for viewing and filtering top generative models data.
Displays CSV data with elegant filtering and coverage statistics.
"""

import csv
import os
from typing import Any, Dict, List, Set

from nicegui import ui


class ModelDataViewer:
    """Handles loading and filtering of model data."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.all_data: List[Dict[str, Any]] = []
        self.filtered_data: List[Dict[str, Any]] = []
        self.columns: List[str] = []
        self.filters: Dict[str, List[str]] = {}
        self.unique_values: Dict[str, Set[str]] = {}

    def load_data(self) -> bool:
        """Load data from CSV file."""
        if not os.path.exists(self.csv_path):
            return False

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.columns = list(reader.fieldnames or [])
            self.all_data = list(reader)
            self.filtered_data = self.all_data.copy()

        # Extract unique values for each column
        self._extract_unique_values()
        return True

    def _extract_unique_values(self):
        """Extract unique values for each column for filter dropdowns."""
        for column in self.columns:
            values = set()
            for row in self.all_data:
                value = row.get(column, "")
                if value:  # Only add non-empty values
                    values.add(str(value))
            self.unique_values[column] = set(sorted(values))

    def apply_filters(self):
        """Apply current filters to data."""
        self.filtered_data = self.all_data.copy()

        for column, selected_values in self.filters.items():
            if selected_values:  # If any values are selected for this column
                self.filtered_data = [
                    row
                    for row in self.filtered_data
                    if str(row.get(column, "")) in selected_values
                ]

    def get_coverage_stats(self) -> Dict[str, Any]:
        """Calculate coverage statistics for supported models."""
        total = len(self.filtered_data)
        if total == 0:
            return {"total": 0, "supported": 0, "percentage": 0.0}

        supported = sum(
            1
            for row in self.filtered_data
            if row.get("is_supported", "").lower() == "true"
        )

        return {
            "total": total,
            "supported": supported,
            "unsupported": total - supported,
            "percentage": (supported / total * 100) if total > 0 else 0.0,
        }

    def get_model_type_stats(self) -> Dict[str, int]:
        """Get statistics by model type."""
        stats = {}
        for row in self.filtered_data:
            model_type = row.get("model_type", "Unknown")
            if not model_type:
                model_type = "Unknown"
            stats[model_type] = stats.get(model_type, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))


# Global viewer instance
viewer = ModelDataViewer("top_generative_models.csv")

# Global UI state
ui_state = {"content_container": None, "stats_cards": {}, "filter_selects": {}}


def create_stats_card(stats: Dict[str, Any]):
    """Create a statistics card showing coverage info."""
    with ui.card().classes("w-full mb-4"):
        ui.label("📊 Coverage Statistics").classes("text-2xl font-bold mb-2")

        with ui.row().classes("w-full gap-4"):
            # Total models card
            with ui.card().classes("flex-1 bg-blue-100"):
                ui.label("Total Models").classes("text-sm text-gray-600")
                ui.label(str(stats["total"])).classes(
                    "text-3xl font-bold text-blue-600"
                )

            # Supported models card
            with ui.card().classes("flex-1 bg-green-100"):
                ui.label("Supported").classes("text-sm text-gray-600")
                ui.label(str(stats["supported"])).classes(
                    "text-3xl font-bold text-green-600"
                )

            # Unsupported models card
            with ui.card().classes("flex-1 bg-red-100"):
                ui.label("Unsupported").classes("text-sm text-gray-600")
                ui.label(str(stats["unsupported"])).classes(
                    "text-3xl font-bold text-red-600"
                )

            # Coverage percentage card
            with ui.card().classes("flex-1 bg-purple-100"):
                ui.label("Coverage %").classes("text-sm text-gray-600")
                ui.label(f"{stats['percentage']:.1f}%").classes(
                    "text-3xl font-bold text-purple-600"
                )


def create_model_type_chart(type_stats: Dict[str, int]):
    """Create a chart showing model type distribution."""
    if not type_stats:
        return

    with ui.card().classes("w-full mb-4"):
        ui.label("📈 Model Type Distribution (Top 10)").classes(
            "text-xl font-bold mb-2"
        )

        # Take top 10 model types
        top_types = dict(list(type_stats.items())[:10])

        with ui.row().classes("w-full gap-2 flex-wrap"):
            max_count = max(top_types.values()) if top_types else 1
            for model_type, count in top_types.items():
                with ui.card().classes("flex-1 min-w-[200px]"):
                    ui.label(model_type or "Unknown").classes("text-sm font-semibold")
                    with ui.row().classes("items-center gap-2 w-full"):
                        ui.linear_progress(value=count / max_count).classes("flex-1")
                        ui.label(str(count)).classes("text-sm font-bold")


def create_data_table(data: List[Dict[str, Any]], columns: List[str]):
    """Create an interactive data table."""
    if not data:
        ui.label("No data to display").classes("text-gray-500 text-center p-4")
        return

    # Define column configurations
    column_configs = [
        {
            "name": "rank",
            "label": "Rank",
            "field": "rank",
            "sortable": True,
            "align": "left",
        },
        {
            "name": "model_id",
            "label": "Model ID",
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
            "name": "likes",
            "label": "Likes",
            "field": "likes",
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
            "name": "architectures",
            "label": "Architectures",
            "field": "architectures",
            "sortable": True,
            "align": "left",
        },
        {
            "name": "parameters (str)",
            "label": "Params",
            "field": "parameters (str)",
            "sortable": True,
            "align": "right",
        },
        {
            "name": "library",
            "label": "Library",
            "field": "library",
            "sortable": True,
            "align": "left",
        },
        {
            "name": "is_gated",
            "label": "Gated",
            "field": "is_gated",
            "sortable": True,
            "align": "center",
        },
        {
            "name": "is_moe",
            "label": "MoE",
            "field": "is_moe",
            "sortable": True,
            "align": "center",
        },
        {
            "name": "config_class",
            "label": "Config Class",
            "field": "config_class",
            "sortable": True,
            "align": "left",
        },
        {
            "name": "is_supported",
            "label": "Supported",
            "field": "is_supported",
            "sortable": True,
            "align": "center",
        },
        {
            "name": "Year",
            "label": "Year",
            "field": "Year",
            "sortable": True,
            "align": "center",
        },
    ]

    # Format data for table
    rows = []
    for row in data:
        formatted_row = {}
        for col in columns:
            value = row.get(col, "")
            # Format large numbers
            if col == "downloads" and value:
                try:
                    formatted_row[col] = f"{int(value):,}"
                except ValueError:
                    formatted_row[col] = value
            elif col == "is_supported":
                # Add color coding for supported status
                formatted_row[col] = "✅" if value.lower() == "true" else "❌"
            elif col == "is_moe":
                formatted_row[col] = "✅" if value.lower() == "true" else "❌"
            elif col == "is_gated":
                formatted_row[col] = "🔒" if value.lower() == "true" else ""
            else:
                formatted_row[col] = value
        rows.append(formatted_row)

    table = ui.table(
        columns=column_configs,
        rows=rows,
        row_key="rank",
        pagination={"rowsPerPage": 20, "sortBy": "rank", "descending": False},
    ).classes("w-full")

    table.add_slot(
        "body-cell-is_supported",
        """
        <q-td :props="props">
            <q-badge :color="props.value === '✅' ? 'green' : 'red'">
                {{ props.value }}
            </q-badge>
        </q-td>
    """,
    )


def refresh_display():
    """Refresh the entire display with current filters."""
    viewer.apply_filters()

    # Clear and rebuild the content
    if ui_state["content_container"] is not None:
        ui_state["content_container"].clear()

        with ui_state["content_container"]:
            # Statistics
            stats = viewer.get_coverage_stats()
            create_stats_card(stats)

            # Model type distribution
            type_stats = viewer.get_model_type_stats()
            create_model_type_chart(type_stats)

            # Data table
            with ui.card().classes("w-full"):
                ui.label(
                    f"📋 Models Table ({len(viewer.filtered_data)} models)"
                ).classes("text-xl font-bold mb-2")
                create_data_table(viewer.filtered_data, viewer.columns)


def create_filter_panel():
    """Create the filter panel."""
    with ui.card().classes("w-full mb-4"):
        ui.label("🔍 Filters").classes("text-xl font-bold mb-2")
        ui.label("Select values to filter (multiple selection allowed)").classes(
            "text-sm text-gray-600 mb-2"
        )

        with ui.row().classes("w-full gap-2 flex-wrap"):
            # Key filters with multi-select
            filter_fields = [
                ("model_type", "Model Type"),
                ("architectures", "Architecture"),
                ("config_class", "Config Class"),
                ("library", "Library"),
                ("is_moe", "MoE"),
                ("is_gated", "Gated"),
                ("Year", "Year"),
            ]

            default_filters: Dict[str, List[str]] = {
                "is_gated": ["False"],
                "is_moe": ["False"],
            }

            for field, label in filter_fields:
                options = list(viewer.unique_values.get(field, []))
                if options:
                    default = [
                        v for v in default_filters.get(field, []) if v in options
                    ]
                    if default:
                        viewer.filters[field] = default
                    ui.select(
                        label=label,
                        options=options,
                        value=default or None,
                        multiple=True,
                        clearable=True,
                        on_change=lambda e, f=field: update_filter(f, e.value),
                    ).classes("flex-1 min-w-[200px]").props("use-chips")

        with ui.row().classes("gap-2 mt-2"):
            ui.button("Clear All Filters", on_click=clear_filters).props(
                "color=secondary"
            )


def update_filter(field: str, value: List[str]):
    """Update a specific filter and refresh the display immediately."""
    viewer.filters[field] = value if value else []
    refresh_display()


def clear_filters():
    """Clear all filters and refresh."""
    viewer.filters.clear()
    refresh_display()


# Main UI
@ui.page("/")
def main_page():
    """Main page of the application."""
    # Header
    with ui.header().classes(
        "items-center justify-between bg-gradient-to-r from-blue-600 to-purple-600"
    ):
        ui.label("🤗 HuggingFace Model Viewer").classes("text-2xl font-bold text-white")
        ui.label("Top Generative Models Analysis").classes(
            "text-sm text-white opacity-80"
        )

    # Check if data is loaded
    if not viewer.load_data():
        with ui.column().classes("items-center justify-center h-screen"):
            ui.label("⚠️ CSV file not found!").classes(
                "text-2xl font-bold text-red-600"
            )
            ui.label(f"Expected file: {viewer.csv_path}").classes("text-gray-600")
            ui.label("Please run fetch_top_generative_models.py first.").classes(
                "text-gray-600 mt-2"
            )
        return

    # Main content
    with ui.column().classes("w-full max-w-[1600px] mx-auto p-4"):
        # Filter panel
        create_filter_panel()

        # Content container (will be refreshed)
        ui_state["content_container"] = ui.column().classes("w-full")

        # Initial display
        refresh_display()


def main():
    """Run the application."""
    ui.run(
        title="HuggingFace Model Viewer",
        favicon="🤗",
        dark=False,
        reload=False,
        port=8080,
        show=True,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()

# Made with Bob
