#!/usr/bin/env python3
"""End-to-end SLOC report for the ``hf_adapters/`` Spyre adapter package.

Does four things in order:

1. Compute the plain SLOC of the **whole folder** (every ``.py``), with blank
   lines, comment-only lines, and docstrings excluded.
2. Compute, for each ``hf_*.py`` adapter **except** ``hf_common.py``, its
   *total* SLOC = its own SLOC + the SLOC of every package symbol it imports,
   resolved transitively and deduplicated within that one file.
3. Save the per-file ``total`` (and ``own`` / ``imported`` breakdown) to a JSON
   file.
4. Render five candidate PNG visualizations of ``(file, total)`` into an output
   directory.

Counting rules (identical to the earlier analysis)
---------------------------------------------------
* Blank and comment-only lines are dropped via :mod:`tokenize`.
* Docstrings (a bare string literal that is the first statement of a module,
  class, function, or async function) are dropped via :mod:`ast`.
* "imported" follows **module-level** ``from hf_adapters.X import sym`` /
  ``from X import sym`` statements only (not function-body imports or
  ``import ... as`` alias attribute access), so it is a slight under-count,
  never an over-count. Shared helpers are counted once *per importing file*.

Usage
-----
    python3 sloc_report.py [--pkg hf_adapters] [--out-dir /tmp] \
                           [--json /tmp/sloc_data.json] [--no-charts]
"""

from __future__ import annotations

import argparse
import ast
import glob
import io
import json
import os
import tokenize

PKG_DEFAULT: str = "hf_adapters"  # package name used to resolve intra-package imports


# ===========================================================================
# Line-classification helpers
# ===========================================================================
def code_line_set(path: str) -> set[int]:
    """1-indexed lines of *path* carrying a real code token (no blanks/comments)."""
    with open(path, "rb") as f:
        src: bytes = f.read()
    lines: set[int] = set()
    skip: tuple[int, ...] = (
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
    )
    for tok in tokenize.tokenize(io.BytesIO(src).readline):
        if tok.type in skip:
            continue
        for ln in range(tok.start[0], tok.end[0] + 1):
            lines.add(ln)
    return lines


def _node_docstring_lines(node: ast.AST) -> set[int]:
    """Lines occupied by *node*'s own docstring (first-statement string), if any."""
    lines: set[int] = set()
    body = getattr(node, "body", None)
    if body:
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            for ln in range(first.lineno, first.end_lineno + 1):
                lines.add(ln)
    return lines


def docstring_line_set(path: str) -> set[int]:
    """Every line occupied by a true docstring in *path*."""
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    doc_nodes = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, doc_nodes):
            lines |= _node_docstring_lines(node)
    return lines


def own_sloc(path: str) -> int:
    """SLOC of *path* alone: code lines minus docstrings."""
    return len(code_line_set(path) - docstring_line_set(path))


# ===========================================================================
# Per-symbol index (for the transitive-import count)
# ===========================================================================
class _SymRefs(ast.NodeVisitor):
    """Collect bare ``Name`` loads referenced inside a node (its dependencies)."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802 (ast API)
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)
        self.generic_visit(node)


SymInfo = dict[str, object]  # {"sloc": int, "refs": set[str]}


def _resolve_module(
    mod_str: str | None, modules: dict[str, str], pkg: str
) -> str | None:
    """Map a dotted import string to a local package module name, or ``None``."""
    if mod_str is None:
        return None
    parts: list[str] = mod_str.split(".")
    cand: str = parts[1] if parts[0] == pkg and len(parts) >= 2 else parts[-1]
    return cand if cand in modules else None


def build_index(
    modules: dict[str, str],
    pkg: str,
) -> tuple[
    dict[str, dict[str, SymInfo]],
    dict[str, dict[str, tuple[str, str]]],
]:
    """Index every module's top-level symbols and intra-package imports.

    Returns ``(module_index, module_imports)`` where
    ``module_index[mod][symbol]`` -> ``{"sloc": int, "refs": set[str]}`` and
    ``module_imports[mod][local_name]`` -> ``(target_module, target_symbol)``.
    """
    module_index: dict[str, dict[str, SymInfo]] = {}
    module_imports: dict[str, dict[str, tuple[str, str]]] = {}

    for mod, path in modules.items():
        with open(path, "r", encoding="utf-8") as f:
            src: str = f.read()
        tree = ast.parse(src, filename=path)
        code_lines: set[int] = code_line_set(path)

        syms: dict[str, SymInfo] = {}
        imap: dict[str, tuple[str, str]] = {}

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                span: set[int] = set(range(node.lineno, node.end_lineno + 1))
                doc: set[int] = set()
                for sub in ast.walk(node):
                    if isinstance(
                        sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    ):
                        doc |= _node_docstring_lines(sub)
                sloc: int = len((span & code_lines) - doc)

                refs = _SymRefs()
                refs.visit(node)
                refs.names.discard(node.name)
                syms[node.name] = {"sloc": sloc, "refs": refs.names}

            elif isinstance(node, ast.ImportFrom):
                target_mod: str | None = _resolve_module(node.module, modules, pkg)
                if target_mod is not None:
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        local: str = alias.asname or alias.name
                        imap[local] = (target_mod, alias.name)

        module_index[mod] = syms
        module_imports[mod] = imap

    return module_index, module_imports


def _resolve_name(
    cur_mod: str,
    name: str,
    module_index: dict[str, dict[str, SymInfo]],
    module_imports: dict[str, dict[str, tuple[str, str]]],
) -> tuple[str, str] | None:
    """Resolve a name used in *cur_mod* to the ``(module, symbol)`` it defines."""
    if name in module_index[cur_mod]:
        return (cur_mod, name)
    if name in module_imports[cur_mod]:
        tmod, tsym = module_imports[cur_mod][name]
        if tsym in module_index.get(tmod, {}):
            return (tmod, tsym)
        if tsym in module_imports.get(tmod, {}):
            return module_imports[tmod][tsym]
    return None


def transitive_imported_sloc(
    file_mod: str,
    module_index: dict[str, dict[str, SymInfo]],
    module_imports: dict[str, dict[str, tuple[str, str]]],
) -> tuple[int, int]:
    """SLOC reachable from *file_mod*'s package imports, transitively, deduped.

    Returns ``(total_imported_sloc, num_distinct_symbols)``; excludes
    *file_mod*'s own top-level symbols.
    """
    seen: set[tuple[str, str]] = set()
    stack: list[tuple[str, str]] = []
    for _local, (tmod, tsym) in module_imports[file_mod].items():
        if tsym in module_index.get(tmod, {}):
            stack.append((tmod, tsym))
        elif tsym in module_imports.get(tmod, {}):
            stack.append(module_imports[tmod][tsym])

    total: int = 0
    while stack:
        mod, sym = stack.pop()
        if (mod, sym) in seen:
            continue
        info: SymInfo | None = module_index.get(mod, {}).get(sym)
        if info is None:
            continue
        seen.add((mod, sym))
        total += int(info["sloc"])
        for ref in info["refs"]:  # type: ignore[union-attr]
            resolved = _resolve_name(mod, ref, module_index, module_imports)
            if resolved is not None and resolved not in seen:
                stack.append(resolved)

    return total, len(seen)


# ===========================================================================
# Steps 1-3: compute totals and save JSON
# ===========================================================================
def discover_modules(pkg_dir: str) -> dict[str, str]:
    """Map ``module_name -> path`` for every ``.py`` in *pkg_dir* except __init__."""
    return {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in sorted(glob.glob(os.path.join(pkg_dir, "*.py")))
        if os.path.basename(p) != "__init__.py"
    }


def whole_folder_sloc(pkg_dir: str) -> tuple[int, list[tuple[str, int]]]:
    """Step 1: plain SLOC of EVERY ``.py`` in the folder (incl. __init__, non-hf_)."""
    rows: list[tuple[str, int]] = []
    for p in sorted(glob.glob(os.path.join(pkg_dir, "*.py"))):
        rows.append((os.path.basename(p), own_sloc(p)))
    return sum(s for _, s in rows), rows


def per_adapter_totals(pkg_dir: str, pkg: str) -> list[dict]:
    """Step 2: per ``hf_*.py`` (excluding hf_common.py) own + transitive-imported SLOC."""
    modules: dict[str, str] = discover_modules(pkg_dir)
    module_index, module_imports = build_index(modules, pkg)

    hf_files: list[str] = sorted(
        m for m in modules if m.startswith("hf_") and m != "hf_common"
    )
    rows: list[dict] = []
    for mod in hf_files:
        own: int = own_sloc(modules[mod])
        imported, n_sym = transitive_imported_sloc(mod, module_index, module_imports)
        rows.append(
            {
                "file": f"{mod}.py",
                "own": own,
                "imported": imported,
                "total": own + imported,
                "n_imported_symbols": n_sym,
            }
        )
    rows.sort(key=lambda r: r["total"], reverse=True)
    return rows


def save_json(rows: list[dict], json_path: str) -> None:
    """Step 3: write the per-file rows to *json_path*."""
    os.makedirs(os.path.dirname(os.path.abspath(json_path)) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


# ===========================================================================
# Step 4: render the five PNG visualizations
# ===========================================================================
def render_charts(rows: list[dict], out_dir: str) -> list[str]:
    """Render the five candidate charts into *out_dir*; return the file paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    # Palette (dataviz skill reference, light mode).
    surface, ink, ink2, muted, grid = (
        "#fcfcfb",
        "#0b0b0b",
        "#52514e",
        "#898781",
        "#e1e0d9",
    )
    blue_steps: list[str] = [
        "#cde2fb",
        "#b7d3f6",
        "#9ec5f4",
        "#86b6ef",
        "#6da7ec",
        "#5598e7",
        "#3987e5",
        "#2a78d6",
        "#256abf",
        "#1c5cab",
        "#184f95",
        "#104281",
        "#0d366b",
    ]
    blue_cm = LinearSegmentedColormap.from_list("brand_blue", blue_steps)
    accent, deemph = "#2a78d6", "#d8d7d1"

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "figure.facecolor": surface,
            "axes.facecolor": surface,
            "savefig.facecolor": surface,
            "text.color": ink,
            "axes.edgecolor": grid,
            "axes.labelcolor": ink2,
            "xtick.color": muted,
            "ytick.color": ink2,
        }
    )

    rows = sorted(rows, key=lambda r: r["total"], reverse=True)
    files: list[str] = [r["file"] for r in rows]
    totals: list[int] = [r["total"] for r in rows]
    tmax: int = max(totals)
    n: int = len(files)
    xlabel: str = "total SLOC (own + transitively imported)"
    saved: list[str] = []

    def color(v: int) -> tuple:
        return blue_cm(0.25 + 0.75 * v / tmax)

    def style_bar_axes(ax) -> None:
        ax.set_xlim(0, tmax * 1.08)
        ax.xaxis.grid(True, color=grid, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(length=0)

    # ---- Option 1: ranked bars, all files, sequential blue ----------------
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.barh(range(n), totals, color=[color(v) for v in totals], height=0.72, zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels(files, fontsize=8.5)
    ax.invert_yaxis()
    for i, v in enumerate(totals):
        ax.text(
            v + tmax * 0.008,
            i,
            f"{v:,}",
            va="center",
            ha="left",
            fontsize=7.8,
            color=ink2,
        )
    style_bar_axes(ax)
    ax.set_title(
        "Adapter size incl. imported package code (SLOC)",
        fontsize=14,
        fontweight="bold",
        loc="left",
        pad=14,
        color=ink,
    )
    ax.set_xlabel(xlabel, fontsize=9)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "opt1_ranked_bars.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved.append(p1)

    # ---- Option 2: emphasis, top 8 accent, rest gray ----------------------
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.barh(
        range(n),
        totals,
        color=[accent if i < 8 else deemph for i in range(n)],
        height=0.72,
        zorder=3,
    )
    ax.set_yticks(range(n))
    ax.set_yticklabels(files, fontsize=8.5)
    for tick, i in zip(ax.get_yticklabels(), range(n)):
        tick.set_color(ink if i < 8 else muted)
    ax.invert_yaxis()
    for i, v in enumerate(totals):
        ax.text(
            v + tmax * 0.008,
            i,
            f"{v:,}",
            va="center",
            ha="left",
            fontsize=7.8,
            color=ink2 if i < 8 else muted,
        )
    style_bar_axes(ax)
    ax.set_title(
        "The 8 heaviest adapters carry most of the weight",
        fontsize=14,
        fontweight="bold",
        loc="left",
        pad=14,
        color=ink,
    )
    ax.set_xlabel(xlabel, fontsize=9)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "opt2_emphasis_top8.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved.append(p2)

    # ---- Option 3: top 15 only, spacious ----------------------------------
    top: int = min(15, n)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.barh(
        range(top),
        totals[:top],
        color=[color(v) for v in totals[:top]],
        height=0.68,
        zorder=3,
    )
    ax.set_yticks(range(top))
    ax.set_yticklabels(files[:top], fontsize=10)
    ax.invert_yaxis()
    for i, v in enumerate(totals[:top]):
        ax.text(
            v + tmax * 0.008,
            i,
            f"{v:,}",
            va="center",
            ha="left",
            fontsize=9,
            color=ink2,
        )
    ax.set_xlim(0, tmax * 1.1)
    ax.xaxis.grid(True, color=grid, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(length=0)
    ax.set_title(
        f"Top {top} adapters by total SLOC",
        fontsize=15,
        fontweight="bold",
        loc="left",
        pad=14,
        color=ink,
    )
    ax.set_xlabel(xlabel, fontsize=10)
    fig.text(
        0.012,
        0.015,
        f"+ {n - top} more files below {totals[top - 1]:,} SLOC",
        fontsize=8.5,
        color=muted,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p3 = os.path.join(out_dir, "opt3_top15.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved.append(p3)

    # ---- Option 4: treemap, area proportional to total --------------------
    def squarify(values: list[float], x: float, y: float, w: float, h: float) -> list:
        scaled: list[float] = [v * (w * h) / sum(values) for v in values]
        rects: list = []
        cx, cy, cw, ch = x, y, w, h

        def worst(row: list[float], length: float) -> float:
            s: float = sum(row)
            return max(
                max((length**2) * rr / (s**2), (s**2) / ((length**2) * rr))
                for rr in row
            )

        i: int = 0
        while i < len(scaled):
            row: list[float] = [scaled[i]]
            j: int = i + 1
            length: float = min(cw, ch)
            while j < len(scaled):
                cand: list[float] = row + [scaled[j]]
                if worst(cand, length) <= worst(row, length):
                    row = cand
                    j += 1
                else:
                    break
            s: float = sum(row)
            if cw >= ch:
                rw: float = s / ch
                oy: float = cy
                for r in row:
                    rh: float = r / rw if rw else 0.0
                    rects.append((cx, oy, rw, rh))
                    oy += rh
                cx += rw
                cw -= rw
            else:
                rh = s / cw
                ox: float = cx
                for r in row:
                    rww: float = r / rh if rh else 0.0
                    rects.append((ox, cy, rww, rh))
                    ox += rww
                cy += rh
                ch -= rh
            i = j
        return rects

    fig, ax = plt.subplots(figsize=(13, 9))
    for (rx, ry, rw, rh), v, name in zip(
        squarify(totals, 0, 0, 100, 100), totals, files
    ):
        ax.add_patch(
            Rectangle(
                (rx, ry), rw, rh, facecolor=color(v), edgecolor=surface, linewidth=2
            )
        )
        if rw * rh > 55:
            short: str = name.replace("hf_", "").replace(".py", "")
            fs: float = max(6.5, min(13, (rw * rh) ** 0.5 * 0.9))
            txt: str = f"{short}\n{v:,}" if rw * rh > 120 else short
            ax.text(
                rx + rw / 2,
                ry + rh / 2,
                txt,
                ha="center",
                va="center",
                fontsize=fs,
                color="white" if v / tmax > 0.55 else ink,
                linespacing=1.1,
            )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(
        "Adapter footprint - tile area scales with total SLOC",
        fontsize=15,
        fontweight="bold",
        loc="left",
        color=ink,
    )
    fig.tight_layout()
    p4 = os.path.join(out_dir, "opt4_treemap.png")
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    saved.append(p4)

    # ---- Option 5: lollipop, all files ------------------------------------
    fig, ax = plt.subplots(figsize=(10, 13))
    for i, v in enumerate(totals):
        ax.plot([0, v], [i, i], color=grid, lw=1.4, zorder=2)
        ax.plot(v, i, "o", color=color(v), markersize=8, zorder=3)
        ax.text(
            v + tmax * 0.01,
            i,
            f"{v:,}",
            va="center",
            ha="left",
            fontsize=7.8,
            color=ink2,
        )
    ax.set_yticks(range(n))
    ax.set_yticklabels(files, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, tmax * 1.1)
    ax.xaxis.grid(True, color=grid, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(length=0)
    ax.set_title(
        "Adapter size - lollipop (total SLOC)",
        fontsize=14,
        fontweight="bold",
        loc="left",
        pad=14,
        color=ink,
    )
    ax.set_xlabel(xlabel, fontsize=9)
    fig.tight_layout()
    p5 = os.path.join(out_dir, "opt5_lollipop.png")
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    saved.append(p5)

    return saved


# ===========================================================================
# Driver
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pkg", default=PKG_DEFAULT, help="package directory (default: hf_adapters)"
    )
    parser.add_argument(
        "--out-dir", default="/tmp", help="directory for the PNG charts (default: /tmp)"
    )
    parser.add_argument(
        "--json",
        default="/tmp/sloc_data.json",
        help="path for the per-file JSON (default: /tmp/sloc_data.json)",
    )
    parser.add_argument(
        "--no-charts", action="store_true", help="skip PNG rendering (steps 1-3 only)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.pkg):
        raise SystemExit(f"package directory not found: {args.pkg!r}")

    # Step 1 — whole-folder SLOC.
    folder_total, folder_rows = whole_folder_sloc(args.pkg)
    print(f"[1] Whole-folder SLOC ({args.pkg}/, all .py): {folder_total:,}")
    for name, s in sorted(folder_rows, key=lambda x: -x[1])[:6]:
        print(f"      {s:6,}  {name}")
    print(f"      ... {len(folder_rows)} files total")

    # Step 2 — per-adapter totals (hf_*.py, excluding hf_common.py).
    rows = per_adapter_totals(args.pkg, args.pkg)
    print(f"\n[2] Per-adapter totals: {len(rows)} files (hf_common.py excluded)")
    print(f"      heaviest: {rows[0]['file']} = {rows[0]['total']:,}")
    print(f"      lightest: {rows[-1]['file']} = {rows[-1]['total']:,}")
    print(f"      sum of totals: {sum(r['total'] for r in rows):,}")

    # Step 3 — save JSON.
    save_json(rows, args.json)
    print(f"\n[3] Saved per-file JSON -> {args.json}")

    # Step 4 — render charts.
    if args.no_charts:
        print("\n[4] Charts skipped (--no-charts).")
        return
    paths = render_charts(rows, args.out_dir)
    print(f"\n[4] Rendered {len(paths)} charts:")
    for p in paths:
        print(f"      {p}")


if __name__ == "__main__":
    main()
