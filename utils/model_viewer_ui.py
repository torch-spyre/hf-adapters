#!/usr/bin/env python3
"""
Interactive UI for viewing and filtering top generative models data.
Displays CSV data with elegant filtering and coverage statistics.
"""

import ast
import asyncio
import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from hf_model_catalog import RESOURCES_DIR
from nicegui import ui

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTERS_DIR = PROJECT_ROOT / "hf_adapters"
AUTO_SPYRE_PATH = ADAPTERS_DIR / "auto_spyre_model.py"


def _parse_config_to_module_map() -> Dict[str, str]:
    """Parse CONFIG_TO_ADAPTER_MODULE_MAPPING
    from auto_spyre_model.py without importing it.
    Returns {config_class_name: adapter_module_name}, e.g. {"Qwen3Config": "hf_qwen3"}.
    """
    try:
        tree = ast.parse(AUTO_SPYRE_PATH.read_text())
    except (OSError, SyntaxError):
        return {}
    for node in ast.walk(tree):
        # Handle annotated assignment (e.g., var: type = value)
        if isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "CONFIG_TO_ADAPTER_MODULE_MAPPING"
                and isinstance(node.value, ast.Dict)
            ):
                result = {}
                for k, v in zip(node.value.keys, node.value.values):
                    if isinstance(k, ast.Name) and isinstance(v, ast.Name):
                        result[k.id] = v.id
                return result
        # Handle regular assignment (e.g., var = value)
        elif isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "CONFIG_TO_ADAPTER_MODULE_MAPPING" in targets and isinstance(
                node.value, ast.Dict
            ):
                result = {}
                for k, v in zip(node.value.keys, node.value.values):
                    if isinstance(k, ast.Name) and isinstance(v, ast.Name):
                        result[k.id] = v.id
                return result
    return {}


CONFIG_CLASS_TO_MODULE = _parse_config_to_module_map()


_UNSET = object()


@dataclass
class FilterState:
    """Immutable filter state to prevent race conditions."""

    filters: Dict[str, List[str]] = field(default_factory=dict)
    params_min: Optional[float] = None
    params_max: Optional[float] = 20e9

    def copy(self) -> "FilterState":
        """Create a deep copy of the filter state."""
        return FilterState(
            filters={k: v.copy() for k, v in self.filters.items()},
            params_min=self.params_min,
            params_max=self.params_max,
        )


class ModelDataViewer:
    """Handles loading and filtering of model data."""

    def __init__(self, csv_path: str | Path):
        self.csv_path = csv_path
        self.all_data: List[Dict[str, Any]] = []
        self.columns: List[str] = []
        self.unique_values: Dict[str, List[str]] = {}

        # Use immutable filter state with thread-safe access
        self._filter_state = FilterState()
        self._state_lock = Lock()

        # Debouncing for rapid filter changes
        self._refresh_task: Optional[asyncio.Task] = None
        self._refresh_delay = 0.3  # 300ms debounce

    @property
    def filters(self) -> Dict[str, List[str]]:
        """Thread-safe access to filters."""
        with self._state_lock:
            return {k: v.copy() for k, v in self._filter_state.filters.items()}

    @property
    def params_min(self) -> Optional[float]:
        """Thread-safe access to params_min."""
        with self._state_lock:
            return self._filter_state.params_min

    @property
    def params_max(self) -> Optional[float]:
        """Thread-safe access to params_max."""
        with self._state_lock:
            return self._filter_state.params_max

    def update_filter_state(
        self,
        filters: Optional[Dict[str, List[str]]] = None,
        params_min: Any = _UNSET,
        params_max: Any = _UNSET,
        _clear: bool = False,
    ) -> FilterState:
        """Thread-safe update of filter state. Returns new state.

        params_min / params_max use the _UNSET sentinel so that explicitly
        passing None (e.g. when the user clears the Min/Max number input)
        actually clears the bound — using None as the "not provided" marker
        would silently drop clear operations.
        """
        with self._state_lock:
            if _clear:
                self._filter_state = FilterState()
            else:
                new_state = self._filter_state.copy()

                if filters is not None:
                    new_state.filters = {k: v.copy() for k, v in filters.items()}

                if params_min is not _UNSET:
                    new_state.params_min = params_min

                if params_max is not _UNSET:
                    new_state.params_max = params_max

                self._filter_state = new_state

            return self._filter_state.copy()

    def clear_filters(self) -> FilterState:
        """Clear all filters. Returns new state."""
        return self.update_filter_state(_clear=True)

    def load_data(self) -> bool:
        """Load data from CSV file."""
        if not os.path.exists(self.csv_path):
            return False

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.columns = list(reader.fieldnames or [])
            self.all_data = list(reader)

        # Extract unique values for each column
        self._extract_unique_values()
        return True

    def _extract_unique_values(self):
        """Extract unique values for each column for filter dropdowns.

        Stored as a case-insensitively sorted list so the select dropdowns
        display options alphabetically.
        """
        for column in self.columns:
            values = set()
            for row in self.all_data:
                value = row.get(column, "")
                if value:  # Only add non-empty values
                    values.add(str(value))
            self.unique_values[column] = sorted(values, key=str.casefold)

    def apply_filters(self) -> List[Dict[str, Any]]:
        """Apply current filters to data. Returns NEW filtered list (immutable)."""
        # Get a snapshot of current filter state
        with self._state_lock:
            current_state = self._filter_state.copy()

        # Work on a copy to avoid modifying shared state
        filtered = self.all_data.copy()

        # Apply column filters
        for column, selected_values in current_state.filters.items():
            if selected_values:  # If any values are selected for this column
                filtered = [
                    row
                    for row in filtered
                    if str(row.get(column, "")) in selected_values
                ]

        # Apply parameter range filter
        if current_state.params_min is not None or current_state.params_max is not None:
            lo = (
                current_state.params_min
                if current_state.params_min is not None
                else float("-inf")
            )
            hi = (
                current_state.params_max
                if current_state.params_max is not None
                else float("inf")
            )
            kept = []
            for row in filtered:
                raw = row.get("parameters", "")
                try:
                    n = float(raw)
                except (TypeError, ValueError):
                    continue
                if lo <= n <= hi:
                    kept.append(row)
            filtered = kept

        return filtered  # Return new list instead of modifying self.filtered_data

    def get_coverage_stats(self, filtered_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate coverage statistics for filtered data."""
        total = len(filtered_data)
        if total == 0:
            return {
                "total": 0,
                "supported": 0,
                "unsupported": 0,
                "percentage": 0.0,
            }

        supported = sum(
            1 for row in filtered_data if row.get("is_supported", "").lower() == "true"
        )

        return {
            "total": total,
            "supported": supported,
            "unsupported": total - supported,
            "percentage": (supported / total * 100) if total > 0 else 0.0,
        }

    def get_model_type_stats(
        self, filtered_data: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Get statistics by model type from filtered data."""
        stats = {}
        for row in filtered_data:
            model_type = row.get("model_type", "Unknown")
            if not model_type:
                model_type = "Unknown"
            stats[model_type] = stats.get(model_type, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))


@dataclass(frozen=True)
class ViewMode:
    """Visual + data configuration for a viewer mode."""

    key: str
    label: str
    icon: str
    csv_path: Path
    header_gradient: str
    accent: str  # tailwind color name, e.g. "blue", "teal"
    extra_filters: tuple = ()  # extra (field, label) tuples appended to filter panel
    extra_columns: tuple = ()  # extra column configs appended to table


GENERATIVE_MODE = ViewMode(
    key="generative",
    label="Generative Models",
    icon="🤗",
    csv_path=RESOURCES_DIR / "top_generative_models.csv",
    header_gradient="bg-gradient-to-r from-blue-600 to-purple-600",
    accent="blue",
)

EMBEDDING_MODE = ViewMode(
    key="embedding",
    label="Embedding Models",
    icon="🧬",
    csv_path=RESOURCES_DIR / "top_embedding_models.csv",
    header_gradient="bg-gradient-to-r from-teal-600 to-emerald-600",
    accent="teal",
    extra_filters=(("is_multimodal", "Multimodal"),),
    extra_columns=(
        {
            "name": "is_multimodal",
            "label": "Multimodal",
            "field": "is_multimodal",
            "sortable": True,
            "align": "center",
        },
    ),
)

MODES: Dict[str, ViewMode] = {
    GENERATIVE_MODE.key: GENERATIVE_MODE,
    EMBEDDING_MODE.key: EMBEDDING_MODE,
}


def create_stats_card(stats: Dict[str, Any], mode: ViewMode):
    """Create a statistics card showing coverage info."""
    accent = mode.accent
    with ui.card().classes("w-full mb-4"):
        ui.label(f"📊 {mode.label} — Coverage Statistics").classes(
            "text-2xl font-bold mb-2"
        )

        with ui.row().classes("w-full gap-4"):
            # Total models card — uses the mode accent color
            with ui.card().classes(f"flex-1 bg-{accent}-100"):
                ui.label("Total Models").classes("text-sm text-gray-600")
                ui.label(str(stats["total"])).classes(
                    f"text-3xl font-bold text-{accent}-600"
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


def create_data_table(data: List[Dict[str, Any]], columns: List[str], mode: ViewMode):
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
            ":format": "v => v == null ? '' : v.toLocaleString()",
        },
        {
            "name": "likes",
            "label": "Likes",
            "field": "likes",
            "sortable": True,
            "align": "right",
            ":format": "v => v == null ? '' : v.toLocaleString()",
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
    column_configs.extend(mode.extra_columns)

    # Format data for table
    rows = []
    for row in data:
        formatted_row = {}
        for col in columns:
            value = row.get(col, "")
            if col in ("downloads", "likes"):
                try:
                    formatted_row[col] = int(value) if value not in ("", None) else None
                except (TypeError, ValueError):
                    formatted_row[col] = None
            elif col == "is_supported":
                # Add color coding for supported status
                formatted_row[col] = "✅" if value.lower() == "true" else "❌"
            elif col == "is_moe":
                formatted_row[col] = "✅" if value.lower() == "true" else "❌"
            elif col == "is_multimodal":
                formatted_row[col] = "✅" if value.lower() == "true" else "❌"
            elif col == "is_gated":
                formatted_row[col] = "🔒" if value.lower() == "true" else "🆓"
            else:
                formatted_row[col] = value
        # Attach adapter module name when the config_class is supported,
        # so the table slot can render it as a link.
        formatted_row["config_class_module"] = CONFIG_CLASS_TO_MODULE.get(
            row.get("config_class", ""), ""
        )
        rows.append(formatted_row)

    with ui.element("div").classes("w-full overflow-x-auto"):
        table = (
            ui.table(
                columns=column_configs,
                rows=rows,
                row_key="rank",
                pagination={
                    "rowsPerPage": 20,
                    "sortBy": "downloads",
                    "descending": True,
                },
            )
            .classes("min-w-[1600px]")
            .props("dense")
        )

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

    table.add_slot(
        "body-cell-config_class",
        """
        <q-td :props="props">
            <a v-if="props.row.config_class_module"
               :href="'/adapter/' + props.row.config_class_module"
               target="_blank"
               class="text-blue-600 underline">
                {{ props.value }}
            </a>
            <span v-else>{{ props.value }}</span>
        </q-td>
    """,
    )


async def refresh_display_async(
    viewer: ModelDataViewer, content_container, mode: ViewMode
) -> None:
    """Async refresh with proper state handling."""
    if content_container is None:
        return

    # Get filtered data (immutable operation)
    filtered_data = viewer.apply_filters()

    # Clear and rebuild UI
    content_container.clear()

    with content_container:
        # Statistics
        stats = viewer.get_coverage_stats(filtered_data)
        create_stats_card(stats, mode)

        # Model type distribution
        # type_stats = viewer.get_model_type_stats(filtered_data)
        # create_model_type_chart(type_stats)

        # Data table
        with ui.card().classes("w-full"):
            ui.label(f"📋 {mode.label} Table ({len(filtered_data)} models)").classes(
                "text-xl font-bold mb-2"
            )
            create_data_table(filtered_data, viewer.columns, mode)


def create_filter_panel_lazy(
    viewer: ModelDataViewer, get_content_container, mode: ViewMode
) -> None:
    """Create the filter panel with debounced refresh.

    Args:
        viewer: The ModelDataViewer instance
        get_content_container: A callable that returns the content container
        mode: The active ViewMode (controls extra filters and labels)
    """

    async def debounced_refresh():
        """Debounced refresh to prevent race conditions."""
        # Cancel any pending refresh
        if viewer._refresh_task and not viewer._refresh_task.done():
            viewer._refresh_task.cancel()
            try:
                await viewer._refresh_task
            except asyncio.CancelledError:
                pass

        # Schedule new refresh after delay
        async def delayed_refresh():
            await asyncio.sleep(viewer._refresh_delay)
            container = get_content_container()
            if container is not None:
                await refresh_display_async(viewer, container, mode)

        viewer._refresh_task = asyncio.create_task(delayed_refresh())

    def update_filter(field: str, value: List[str]) -> None:
        new_filters = viewer.filters  # Get current filters
        new_filters[field] = value if value else []
        viewer.update_filter_state(filters=new_filters)
        asyncio.create_task(debounced_refresh())

    def update_params_range(min_b: Any, max_b: Any) -> None:
        params_min: Any = _UNSET
        params_max: Any = _UNSET

        if min_b is not _UNSET:
            params_min = float(min_b) * 1e9 if min_b not in (None, "") else None
        if max_b is not _UNSET:
            params_max = float(max_b) * 1e9 if max_b not in (None, "") else None

        viewer.update_filter_state(params_min=params_min, params_max=params_max)
        asyncio.create_task(debounced_refresh())

    def clear_filters() -> None:
        viewer.clear_filters()
        asyncio.create_task(debounced_refresh())

    with ui.card().classes(f"w-full mb-4 border-l-4 border-{mode.accent}-500"):
        ui.label(f"🔍 Filters — {mode.label}").classes(
            f"text-xl font-bold mb-2 text-{mode.accent}-700"
        )
        ui.label("Select values to filter (multiple selection allowed)").classes(
            "text-sm text-gray-600 mb-2"
        )

        with ui.element("div").classes(
            "w-full grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2"
        ):
            # Key filters with multi-select
            filter_fields = [
                ("model_type", "Model Type"),
                ("architectures", "Architecture"),
                ("config_class", "Config Class"),
                ("library", "Library"),
                ("is_supported", "Supported"),
                ("is_moe", "MoE"),
                ("is_gated", "Gated"),
                ("Year", "Year"),
                *mode.extra_filters,
            ]

            default_filters: Dict[str, List[str]] = {
                "is_gated": ["False"],
                "is_moe": ["False"],
            }

            # Accumulate defaults and push them to the viewer's filter state
            # in one shot. Assigning to `viewer.filters[...]` does NOT work —
            # the `filters` property returns a copy, so per-field assignments
            # mutate a throwaway dict and the initial refresh would run with
            # no filters applied (e.g. MoE=True models would still appear).
            initial_filters: Dict[str, List[str]] = {}

            for field, label in filter_fields:
                options = list(viewer.unique_values.get(field, []))
                if options:
                    default = [
                        v for v in default_filters.get(field, []) if v in options
                    ]
                    if default:
                        initial_filters[field] = default

                    # Boolean-only fields (True/False) get a single-select with
                    # a "no selection means show all" semantics — multi-select
                    # with both values picked is equivalent to no filter, which
                    # is confusing.
                    is_boolean = set(options) <= {"True", "False"}

                    if is_boolean:
                        ui.select(
                            label=label,
                            options=options,
                            value=(default[0] if default else None),
                            multiple=False,
                            clearable=True,
                            on_change=lambda e, f=field: update_filter(
                                f, [e.value] if e.value else []
                            ),
                        ).classes("w-full")
                        continue

                    def _on_change(e, f=field, sel=None):
                        update_filter(f, e.value)
                        # Clear the typed search text after each selection so the
                        # input doesn't keep stale fragments like "openb"
                        # alongside the selected chip "openba". This calls the
                        # underlying Quasar QSelect.updateInputValue() method.
                        if sel is not None:
                            sel.run_method("updateInputValue", "", True)

                    select = (
                        ui.select(
                            label=label,
                            options=options,
                            value=default or None,
                            multiple=True,
                            clearable=True,
                            with_input=True,
                        )
                        .classes("w-full")
                        .props("use-chips")
                    )
                    select.on_value_change(
                        lambda e, f=field, sel=select: _on_change(e, f, sel)
                    )

            # Numeric range filter on parameters (in billions, B)
            with ui.row().classes("items-center gap-2 w-full"):
                ui.label("Params (B):").classes("text-sm font-semibold")
                ui.number(
                    label="Min",
                    value=None,
                    min=0,
                    step=0.1,
                    format="%.2f",
                    on_change=lambda e: update_params_range(e.value, _UNSET),
                ).classes("flex-1").props("clearable")
                ui.number(
                    label="Max",
                    value=20,
                    min=0,
                    step=0.1,
                    format="%.2f",
                    on_change=lambda e: update_params_range(_UNSET, e.value),
                ).classes("flex-1").props("clearable")

        # Commit accumulated default filter selections to the shared filter
        # state so the initial render honors them (the per-widget default=
        # only seeds the UI control's displayed value).
        if initial_filters:
            viewer.update_filter_state(filters=initial_filters)

        with ui.row().classes("gap-2 mt-2"):
            ui.button("Clear All Filters", on_click=clear_filters).props(
                "color=secondary"
            )


# Main UI
@ui.page("/adapter/{module_name}")
def adapter_source_page(module_name: str):
    """Display the source of a Spyre adapter module."""
    if module_name not in CONFIG_CLASS_TO_MODULE.values():
        ui.label(f"Unknown adapter module: {module_name}").classes("text-red-600 p-4")
        return
    path = ADAPTERS_DIR / f"{module_name}.py"
    if not path.exists():
        ui.label(f"File not found: {path}").classes("text-red-600 p-4")
        return
    source = path.read_text()
    ui.label(f"📄 hf_adapters/{module_name}.py").classes("text-2xl font-bold p-4")
    ui.code(source, language="python").classes("w-full")


@ui.page("/")
def main_page(mode: str = "generative"):
    """Main page of the application.

    The `mode` query parameter selects which catalog to display:
    `/?mode=generative` (default) or `/?mode=embedding`.
    """
    view_mode = MODES.get(mode, GENERATIVE_MODE)

    # Per-session state: each browser connection gets its own viewer so
    # filters / params range / filtered_data are not shared across users.
    viewer = ModelDataViewer(view_mode.csv_path)

    # Header — gradient + accent change with the mode for clear visual distinction.
    with ui.header().classes(
        f"items-center justify-between {view_mode.header_gradient}"
    ):
        with ui.row().classes("items-center gap-3"):
            ui.label(f"{view_mode.icon} HuggingFace Model Viewer").classes(
                "text-2xl font-bold text-white"
            )
            ui.label(view_mode.label).classes("text-sm text-white opacity-80")

    # Prominent mode selector — large segmented control below the header.
    # The active segment is filled with the mode's accent color; the inactive one
    # is a clearly-clickable outlined button. Sits in its own bar so users can't
    # miss it.
    with ui.row().classes(
        "w-full items-center justify-center gap-3 py-3 bg-gray-100 border-b shadow-sm"
    ):
        ui.label("").classes("text-base font-semibold text-gray-700")
        for m in (GENERATIVE_MODE, EMBEDDING_MODE):
            is_active = m.key == view_mode.key
            if is_active:
                ui.button(
                    f"{m.icon}  {m.label}",
                    on_click=lambda _, k=m.key: ui.navigate.to(f"/?mode={k}"),
                ).classes(
                    f"bg-{m.accent}-600 text-white font-bold "
                    f"text-base px-6 py-2 rounded-full shadow-lg "
                    f"ring-2 ring-{m.accent}-300 ring-offset-2"
                ).props(
                    "no-caps unelevated"
                )
            else:
                ui.button(
                    f"{m.icon}  {m.label}",
                    on_click=lambda _, k=m.key: ui.navigate.to(f"/?mode={k}"),
                ).classes(
                    f"bg-white text-{m.accent}-700 font-semibold "
                    f"text-base px-6 py-2 rounded-full "
                    f"border-2 border-{m.accent}-400 hover:bg-{m.accent}-50"
                ).props(
                    "no-caps flat"
                )

    # Check if data is loaded
    if not viewer.load_data():
        with ui.column().classes("items-center justify-center h-screen"):
            ui.label("⚠️ CSV file not found!").classes(
                "text-2xl font-bold text-red-600"
            )
            ui.label(f"Expected file: {viewer.csv_path}").classes("text-gray-600")
            ui.label(f"Please populate {view_mode.csv_path.name} first.").classes(
                "text-gray-600 mt-2"
            )
        return

    # Main content
    with ui.column().classes("w-full max-w-[1600px] mx-auto p-4"):
        # Use a list to hold the content container reference (late binding)
        content_ref: List[Any] = [None]

        # Filter panel — renders first (at top of page).
        create_filter_panel_lazy(viewer, lambda: content_ref[0], view_mode)

        # Content container (rendered below the filter panel).
        # The accent border reinforces which mode is active.
        content_ref[0] = ui.column().classes(
            f"w-full border-l-4 border-{view_mode.accent}-400 pl-2"
        )

        # Initial display
        asyncio.create_task(refresh_display_async(viewer, content_ref[0], view_mode))


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
