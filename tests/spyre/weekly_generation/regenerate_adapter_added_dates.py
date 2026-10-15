"""Regenerate ``resources/adapter_added_dates.json`` from full git history.

``added_date`` is a *stable per-adapter attribute*: the ISO date an adapter's
``hf_adapters/hf_*.py`` file was first added to git. The weekly scan stamps it
onto every row it writes to ClickHouse, so no two rows sharing an
``adapter_name`` should ever disagree on it (issue #372).

Deriving that date from ``git log`` at scan time is unsafe: the weekly workflow
runs inside a ``git clone --depth=1`` checkout (see
``.github/actions/build-hf-adapters``), and on a shallow clone
``git log --diff-filter=A --follow`` cannot see a file's real add-commit — it
reports the shallow-root (tip) commit's date instead, the same wrong date for
every adapter in that run. Those poisoned dates then coexist in the table with
the correct dates written from full-history checkouts.

The fix is to stop deriving the date at scan time and read it from a committed
JSON map instead (``_get_adapter_dates`` in ``weekly_test.py``). This script is
how that map is (re)generated — run it from a **full** checkout whenever an
adapter is added or renamed, then commit the updated JSON:

    python tests/spyre/weekly_generation/regenerate_adapter_added_dates.py

``tests/test_adapter_added_dates.py`` asserts the committed file matches what
this script would produce on a full checkout, so a stale file fails CI (the
adapter-coverage job runs on a full checkout) rather than silently drifting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# tests/spyre/weekly_generation/ -> repo root is three parents up.
_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
_ADAPTER_DIR: Path = _REPO_ROOT / "hf_adapters"
# Committed source of truth read by weekly_test._get_adapter_dates(). Lives in
# resources/ with the other checked-in scan data. The path is computed locally
# rather than imported as RESOURCES_DIR from utils.hf_model_catalog on purpose:
# this script and its test must run with only the stdlib + git (the
# adapter-coverage CI job installs just pytest), and that module pulls in
# transformers/huggingface_hub at import time.
DATES_JSON: Path = _REPO_ROOT / "resources" / "adapter_added_dates.json"


def _git_add_date(rel_path: str) -> str | None:
    """ISO add-date (``YYYY-MM-DD``) of *rel_path* per git, or ``None``.

    Uses the same ``--diff-filter=A --follow`` query the scan historically used,
    so dates match the correct rows already in the table. Requires a full
    checkout: on a shallow clone the add-commit is not present and git reports
    the shallow-root date instead — this script must never be run there (see the
    module docstring), and the accompanying test guards against a stale result.
    """
    out = subprocess.run(
        [
            "git",
            "log",
            "--diff-filter=A",
            "--follow",
            "--format=%aI",
            "-1",
            "--",
            rel_path,
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    lines: list[str] = out.stdout.strip().splitlines()
    # --follow can emit several lines across renames; the file's own add-commit
    # is the last (oldest) one, matching the historical scan behaviour.
    return lines[-1][:10] if lines else None


def compute_added_dates() -> dict[str, str | None]:
    """Map every ``hf_adapters/hf_*.py`` stem to its git add-date.

    Keys are sorted so the emitted JSON has a stable, review-friendly order.
    """
    dates: dict[str, str | None] = {}
    for f in sorted(_ADAPTER_DIR.glob("hf_*.py")):
        dates[f.stem] = _git_add_date(str(f.relative_to(_REPO_ROOT)))
    return dates


def render_json(dates: dict[str, str | None]) -> str:
    """Serialize *dates* exactly as the committed file stores it.

    A trailing newline keeps the file POSIX-clean and diff-friendly; the test
    compares against this same rendering, so the two never disagree on format.
    """
    return json.dumps(dates, indent=2, ensure_ascii=False) + "\n"


def _is_shallow() -> bool:
    """True if this repo is a shallow clone (git history truncated)."""
    out = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() == "true"


def _is_committed(rel_path: str) -> bool:
    """True if *rel_path* exists in ``HEAD`` (already committed, not brand-new).

    A brand-new adapter staged in the current commit is not yet in ``HEAD`` and
    has no git add-date until that commit lands — so it is exempt from the hook's
    presence check (see the module docstring's two-step flow).
    """
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"HEAD:{rel_path}"],
            cwd=_REPO_ROOT,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def _load_committed_map() -> dict[str, str | None]:
    """The JSON as currently on disk, or ``{}`` if missing/unparseable."""
    try:
        return json.loads(DATES_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def check_hook() -> tuple[bool, str]:
    """Pre-commit check. Returns ``(ok, message)``.

    Enforces exactly what a commit can guarantee, and no more:

    * Every adapter file **already in HEAD** must have a JSON entry. This needs
      no git history, so it holds on a shallow clone too. A brand-new adapter
      staged in this very commit is exempt — git cannot date it until the commit
      exists, and we never fabricate a date (the follow-up commit regenerates).
    * On a **full** checkout, every entry git can date must match git exactly, so
      a stale or hand-edited date is caught. Skipped on a shallow clone, where
      git is unreliable — the very condition the JSON exists to survive.
    """
    committed: dict[str, str | None] = _load_committed_map()
    on_disk: list[Path] = sorted(_ADAPTER_DIR.glob("hf_*.py"))

    # Presence: already-committed adapters must be in the map.
    missing: list[str] = [
        f.stem
        for f in on_disk
        if _is_committed(str(f.relative_to(_REPO_ROOT))) and f.stem not in committed
    ]
    if missing:
        return False, (
            f"{DATES_JSON.name} is missing entries for already-committed "
            f"adapter(s): {', '.join(sorted(missing))}.\n"
            f"Regenerate it from a full checkout:\n"
            f"    python {Path(__file__).relative_to(_REPO_ROOT)}\n"
            f"    git add {DATES_JSON.relative_to(_REPO_ROOT)}"
        )

    if _is_shallow():
        return True, (
            f"{DATES_JSON.name}: present for all committed adapters "
            "(shallow clone — dates not verified)."
        )

    # Full checkout: every date git can produce must match the stored value.
    # Files git cannot date here are brand-new/uncommitted — exempt, as above.
    git_dates: dict[str, str | None] = compute_added_dates()
    mismatched: list[str] = [
        stem
        for stem, gd in git_dates.items()
        if gd is not None and committed.get(stem) != gd
    ]
    if mismatched:
        return False, (
            f"{DATES_JSON.name} is out of date with git for: "
            f"{', '.join(sorted(mismatched))}.\n"
            f"Regenerate it:\n"
            f"    python {Path(__file__).relative_to(_REPO_ROOT)}\n"
            f"    git add {DATES_JSON.relative_to(_REPO_ROOT)}"
        )
    return True, f"{DATES_JSON.name} is up to date."


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help=(
            "Do not write. Exit 0 if the committed JSON already equals what a "
            "full-checkout regeneration would produce, non-zero otherwise. Used "
            "by tests/test_adapter_added_dates.py's full-checkout equality test."
        ),
    )
    mode.add_argument(
        "--check-hook",
        action="store_true",
        help=(
            "Do not write. The pre-commit check (see check_hook): committed "
            "adapters must have an entry; on a full checkout stored dates must "
            "match git; brand-new staged adapters and shallow clones are handled "
            "leniently. Used by scripts/check_adapter_added_dates.sh."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.check_hook:
        ok, message = check_hook()
        print(message, file=sys.stderr if not ok else sys.stdout)
        return 0 if ok else 1

    dates: dict[str, str | None] = compute_added_dates()
    rendered: str = render_json(dates)

    if args.check:
        current: str = (
            DATES_JSON.read_text(encoding="utf-8") if DATES_JSON.is_file() else ""
        )
        if current == rendered:
            print(f"{DATES_JSON.name} is up to date ({len(dates)} adapters).")
            return 0
        print(
            f"{DATES_JSON.name} is stale. Regenerate it from a full checkout:\n"
            f"    python {Path(__file__).relative_to(_REPO_ROOT)}",
            file=sys.stderr,
        )
        return 1

    missing: list[str] = [name for name, d in dates.items() if d is None]
    if missing:
        # A None on a full checkout means git could not date the file at all —
        # almost certainly a shallow clone. Refuse to write a partial map that
        # would poison rows with nulls.
        print(
            "Refusing to write: no git add-date for "
            f"{len(missing)} adapter(s): {', '.join(missing)}. "
            "Run this from a full (non-shallow) checkout.",
            file=sys.stderr,
        )
        return 1

    DATES_JSON.write_text(rendered, encoding="utf-8")
    print(f"Wrote {DATES_JSON.relative_to(_REPO_ROOT)} ({len(dates)} adapters).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
