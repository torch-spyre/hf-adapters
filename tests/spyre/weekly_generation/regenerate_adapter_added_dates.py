"""Maintain the authoritative ``resources/adapter_added_dates.json`` (issue #372).

Existing dates are preserved. New adapters receive today's UTC date, or the
explicit ``--added-date YYYY-MM-DD``. Add the adapter and its metadata in the
same commit. For a rename, move its existing JSON entry to the new module name
before running this updater; removed adapters are pruned.

The initial map was seeded from full git history. Git is never used to update
or validate dates: squash merges rewrite introducing commits and their dates,
so comparing the map with branch or main history would make it unstable.

``--check`` and ``--check-hook`` validate completeness and ISO dates without
writing. Every mode works in full, shallow, and non-git checkouts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
_ADAPTER_DIR: Path = _REPO_ROOT / "hf_adapters"
DATES_JSON: Path = _REPO_ROOT / "resources" / "adapter_added_dates.json"
_REGEN_SCRIPT: str = "tests/spyre/weekly_generation/regenerate_adapter_added_dates.py"


def _parse_date(value: str) -> date:
    """Require a real date in canonical YYYY-MM-DD form."""
    parsed: date = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError(f"expected YYYY-MM-DD, got {value!r}")
    return parsed


def _load_committed_map() -> dict[str, str]:
    """Load existing authoritative dates; reject corruption instead of replacing it."""
    if not DATES_JSON.exists():
        return {}
    loaded: object = json.loads(DATES_JSON.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{DATES_JSON.name} must be a JSON object")
    dates: dict[str, str] = {}
    for stem, value in loaded.items():
        try:
            if not isinstance(value, str):
                raise ValueError("expected a date string")
            _parse_date(value)
        except ValueError as error:
            raise ValueError(
                f"{DATES_JSON.name} has invalid dates for {stem}: {error}"
            ) from error
        dates[stem] = value
    return dates


def compute_added_dates(added_date: date | None = None) -> dict[str, str]:
    """Preserve stored dates, fill new entries, and prune removed adapter names."""
    existing: dict[str, str] = _load_committed_map()
    new_date: str = (added_date or datetime.now(timezone.utc).date()).isoformat()
    return {
        adapter_path.stem: existing.get(adapter_path.stem, new_date)
        for adapter_path in sorted(_ADAPTER_DIR.glob("hf_*.py"))
    }


def render_json(dates: dict[str, str]) -> str:
    """Serialize dates in a deterministic order with a trailing newline."""
    return json.dumps(dates, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def check_hook() -> tuple[bool, str]:
    """Validate all adapters, including new ones, without consulting git history."""
    try:
        if not DATES_JSON.is_file():
            return False, f"{DATES_JSON.name} is missing. Run python {_REGEN_SCRIPT}."
        committed: dict[str, str] = _load_committed_map()
    except (OSError, ValueError) as error:
        return False, str(error)

    on_disk: set[str] = {path.stem for path in _ADAPTER_DIR.glob("hf_*.py")}
    missing: list[str] = sorted(on_disk - committed.keys())
    if missing:
        return False, (
            f"{DATES_JSON.name} is missing entries for adapter(s): {', '.join(missing)}.\n"
            f"Run python {_REGEN_SCRIPT}, then stage the JSON with the adapter."
        )
    stale: list[str] = sorted(committed.keys() - on_disk)
    if stale:
        return False, (
            f"{DATES_JSON.name} lists removed adapter(s): {', '.join(stale)}.\n"
            f"Run python {_REGEN_SCRIPT}. For a rename, move the existing date to the new name first."
        )
    return (
        True,
        f"{DATES_JSON.name} is complete with valid ISO dates ({len(committed)} adapters).",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Validate metadata without writing or consulting git.",
    )
    mode.add_argument(
        "--check-hook",
        action="store_true",
        help="Alias of --check for the pre-commit hook.",
    )
    parser.add_argument(
        "--added-date",
        type=_parse_date,
        help="Date for new entries only, in YYYY-MM-DD form (default: today in UTC).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate or update metadata while preserving every existing date."""
    args = _parse_args(argv)
    if args.check or args.check_hook:
        ok, message = check_hook()
        print(message, file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 1
    try:
        dates: dict[str, str] = compute_added_dates(args.added_date)
        DATES_JSON.write_text(render_json(dates), encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"Cannot update adapter dates: {error}", file=sys.stderr)
        return 1
    print(
        f"Updated {DATES_JSON.name} ({len(dates)} adapters; existing dates preserved)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
