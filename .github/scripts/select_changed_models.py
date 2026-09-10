#!/usr/bin/env python3
"""
Restrict the PR test matrix to the adapters actually touched by a diff.

Reads a list of changed file paths (one per line) and decides:
  - Which adapter(s) changed, if the diff is confined to hf_adapters/hf_*.py
    files -- restrict the Spyre matrix to those adapters' registered models
    (PLUS every adapter that imports from one of them, transitively --
    e.g. hf_granite.py also pulls in hf_granite_swa.py, hf_granite_vision.py,
    and hf_granitemoehybrid.py, all of which import from it) via the existing
    `models` input to generate_test_matrix.py / _test_matrix.yaml.
  - Whether the whole hardware suite can be skipped -- via the existing
    `skip_tests` input -- when every changed file is documentation or
    confined to the weekly model-discovery scan (tests/spyre/weekly_generation/,
    which no adapter test suite exercises).

The adapter-to-adapter dependency graph is built by statically parsing (via
``ast`` -- never executing) each hf_adapters/hf_*.py file's import
statements, so this needs no torch/transformers install.

Conservative by design: doc-only and weekly-only files are filtered out of
consideration first; whatever remains must be entirely single, non-shared
adapter files for a restriction to apply. Genuinely shared/foundational
hf_adapters/ infrastructure -- hf_common.py, auto_spyre_model.py,
st_backend.py, _dspark_common.py, __init__.py -- always forces the full
matrix rather than trying to trace its blast radius: some of that coupling
(e.g. every adapter's `from hf_adapters import ...` implicitly requires
__init__.py to execute first, a Python package-import guarantee that isn't
visible in any *importer's* own source) can't be certified by statically
parsing one file at a time. Anything outside hf_adapters/ entirely (test
infra, CI config, the model registry itself, ...) also falls back to the
full matrix.

Usage:
    python select_changed_models.py changed_files.txt >> "$GITHUB_OUTPUT"

Prints, in GITHUB_OUTPUT format:
    models=<space-separated model paths, or empty for "no restriction">
    skip_tests=<true|false>
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Add the project root to the Python path so we can import from tests/
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import tests.model_registry as registry  # noqa: E402

ADAPTER_DIR_NAME = "hf_adapters"
ADAPTER_DIR = project_root / ADAPTER_DIR_NAME

# Same convention tests/test_adapter_coverage.py uses: adapter files are
# hf_adapters/hf_*.py, excluding hf_common.py (a shared utilities module).
# Everything else directly under hf_adapters/ is foundational/shared
# infrastructure and is deliberately excluded from the dependency graph below
# -- see the module docstring for why those files always force a full run.
_ADAPTER_MODULE_NAMES: frozenset[str] = frozenset(
    p.stem for p in ADAPTER_DIR.glob("hf_*.py") if p.name != "hf_common.py"
)

DOC_SUFFIXES = (".md",)
DOC_PREFIXES = ("docs/",)

# Changes confined to these paths only exercise the weekly model-discovery
# scan; no adapter test suite (smoke/token-compare/embed-compare/vlm/...)
# runs this code, so it's safe to skip them.
WEEKLY_ONLY_PREFIXES = (
    "tests/spyre/weekly_generation/",
    "tests/test_weekly_prefilter.py",
    ".github/scripts/generate_weekly_shards.py",
    ".github/scripts/ingest_xml_hf_adapters.py",
)

ALL_MODEL_REGISTRIES = (
    registry.CAUSAL_LM_MODELS,
    registry.EMBEDDING_MODELS,
    registry.QUESTION_ANSWERING_MODELS,
    registry.MASKED_LM_MODELS,
    registry.VISION_MODELS,
    registry.RERANKER_MODELS,
)


def _adapter_imports(module_name: str) -> set[str]:
    """Other adapter modules that *module_name* imports from, directly.

    Only edges between two real adapter files (both in
    ``_ADAPTER_MODULE_NAMES``) are tracked -- imports of hf_common.py,
    __init__.py, auto_spyre_model.py, etc. are deliberately not represented
    here; those always force the full matrix through a separate check in
    ``select()`` instead of being folded into this graph.
    """
    path = ADAPTER_DIR / f"{module_name}.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    deps: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "hf_adapters":
                # from hf_adapters import hf_granite, ...
                deps.update(
                    alias.name
                    for alias in node.names
                    if alias.name in _ADAPTER_MODULE_NAMES
                )
            elif node.module.startswith("hf_adapters."):
                # from hf_adapters.hf_granite import ...
                submodule = node.module.split(".")[1]
                if submodule in _ADAPTER_MODULE_NAMES:
                    deps.add(submodule)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                # import hf_adapters.hf_granite [as x]
                parts = alias.name.split(".")
                if (
                    len(parts) == 2
                    and parts[0] == "hf_adapters"
                    and parts[1] in _ADAPTER_MODULE_NAMES
                ):
                    deps.add(parts[1])
    deps.discard(module_name)
    return deps


def _build_adapter_dependents_graph() -> dict[str, set[str]]:
    """dependents[X] = adapter modules that import X, directly."""
    dependents: dict[str, set[str]] = {name: set() for name in _ADAPTER_MODULE_NAMES}
    for module_name in _ADAPTER_MODULE_NAMES:
        for dep in _adapter_imports(module_name):
            dependents[dep].add(module_name)
    return dependents


def _affected_adapter_modules(changed: set[str]) -> set[str]:
    """*changed* plus every adapter module that (transitively) imports one of them."""
    dependents = _build_adapter_dependents_graph()
    affected = set(changed)
    frontier = list(changed)
    while frontier:
        current = frontier.pop()
        for dep in dependents.get(current, ()):
            if dep not in affected:
                affected.add(dep)
                frontier.append(dep)
    return affected


def _is_doc_only(path: str) -> bool:
    return path.endswith(DOC_SUFFIXES) or path.startswith(DOC_PREFIXES)


def _is_weekly_only(path: str) -> bool:
    return path.startswith(WEEKLY_ONLY_PREFIXES)


def _single_adapter_module(path: str) -> str | None:
    """Return the adapter module name if *path* is a single, non-shared adapter file."""
    prefix = f"{ADAPTER_DIR_NAME}/"
    if not path.startswith(prefix):
        return None
    rest = path[len(prefix) :]
    if "/" in rest or not rest.endswith(".py"):
        return None
    module_name = rest[: -len(".py")]
    if module_name not in _ADAPTER_MODULE_NAMES:
        return None
    return module_name


def _paths_for_modules(module_names: set[str]) -> list[str]:
    paths: set[str] = set()
    for models in ALL_MODEL_REGISTRIES:
        for info in models.values():
            if info["adapter"].removesuffix(".py") in module_names:
                paths.add(info["path"])
    return sorted(paths)


def select(changed_files: list[str]) -> tuple[list[str], bool]:
    """Return (models, skip_tests) for the given changed file list.

    Empty models means "no restriction" (run everything -- generate-matrix's
    default). skip_tests=True means the whole Spyre suite can be skipped.
    """
    if not changed_files:
        # No diff info (e.g. can't determine changed files) -- run everything.
        return [], False

    testable = [
        f for f in changed_files if not _is_doc_only(f) and not _is_weekly_only(f)
    ]
    if not testable:
        # Every changed file is doc-only and/or weekly-scan-only.
        return [], True

    changed_modules: set[str] = set()
    for f in testable:
        module_name = _single_adapter_module(f)
        if module_name is None:
            # Something outside the "single adapter file" allowlist changed
            # (shared adapter infrastructure, tests infra, CI config, ...) --
            # don't guess, run the full matrix.
            return [], False
        changed_modules.add(module_name)

    affected = _affected_adapter_modules(changed_modules)
    models = _paths_for_modules(affected)
    if not models:
        # An adapter file changed but nothing in the registry references it
        # (it, or its dependents) yet -- don't emit a restriction that would
        # collapse the matrix to nothing; run everything instead.
        return [], False

    return models, False


def main() -> None:
    changed_files_path = Path(sys.argv[1])
    changed_files = [
        line.strip()
        for line in changed_files_path.read_text().splitlines()
        if line.strip()
    ]

    models, skip_tests = select(changed_files)

    print(f"models={' '.join(models)}")
    print(f"skip_tests={'true' if skip_tests else 'false'}")


if __name__ == "__main__":
    main()
