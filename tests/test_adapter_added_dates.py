"""Guard the committed ``resources/adapter_added_dates.json`` (issue #372).

``added_date`` is a stable per-adapter attribute the weekly scan stamps onto
every ClickHouse row. It used to be derived from ``git log`` at scan time, which
is wrong on the ``--depth=1`` clone the workflow runs in — the shallow root's
date gets reported for every adapter. The scan now reads the committed JSON map
instead (``weekly_test._get_adapter_dates``), so these tests keep that map
correct:

* Every ``hf_adapters/hf_*.py`` on disk must appear in the map with a valid ISO
  date — an adapter added without regenerating the map fails here rather than
  silently writing ``null`` dates. This runs on any checkout, shallow included.
* On a full checkout, the map must equal what ``regenerate_adapter_added_dates``
  produces from git, so a stale or hand-edited date is caught. Skipped on a
  shallow clone, where git cannot see the real add-commits (the same limitation
  the JSON exists to work around) — the completeness check above still runs.

Run with ``pytest --noconftest`` (see ``test_weekly_prefilter``'s header): no
torch, no database driver, only ``pytest`` and a git checkout.
"""

from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import pytest

from tests.spyre.weekly_generation import regenerate_adapter_added_dates as regen
from tests.spyre.weekly_generation.regenerate_adapter_added_dates import (
    DATES_JSON,
    check_hook,
    compute_added_dates,
    render_json,
)

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_ADAPTER_DIR: Path = _REPO_ROOT / "hf_adapters"


def _is_shallow_clone() -> bool:
    """True if this repo is a shallow clone (git history is truncated).

    ``regenerate_adapter_added_dates`` cannot recover real add-dates here, so the
    git-equality check is skipped; the completeness check does not need history.
    """
    out = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() == "true"


def _load_committed_map() -> dict[str, str | None]:
    return json.loads(DATES_JSON.read_text(encoding="utf-8"))


def test_dates_file_exists_and_parses() -> None:
    """The committed map exists and is a JSON object."""
    assert DATES_JSON.is_file(), f"{DATES_JSON} is missing"
    assert isinstance(_load_committed_map(), dict)


def test_every_adapter_has_a_valid_date() -> None:
    """Every hf_*.py on disk maps to a parseable ISO date — no gaps, no nulls.

    This is the check that fires when someone adds an adapter without running
    the regenerator: the new stem is absent (or ``null``) and the scan would
    otherwise write ``added_date=NULL`` for every model using it.
    """
    committed: dict[str, str | None] = _load_committed_map()
    on_disk: set[str] = {f.stem for f in _ADAPTER_DIR.glob("hf_*.py")}

    missing: set[str] = on_disk - committed.keys()
    assert not missing, (
        f"adapters on disk absent from {DATES_JSON.name}: {sorted(missing)}. "
        "Regenerate it: python tests/spyre/weekly_generation/"
        "regenerate_adapter_added_dates.py"
    )

    for stem in sorted(on_disk):
        value = committed[stem]
        assert isinstance(value, str) and value, (
            f"{stem} has no add-date in {DATES_JSON.name} (got {value!r}); "
            "regenerate from a full checkout."
        )
        # Raises ValueError if not a real YYYY-MM-DD date.
        date.fromisoformat(value)


def test_no_stale_entries() -> None:
    """The map has no entry for an adapter file that no longer exists."""
    committed: dict[str, str | None] = _load_committed_map()
    on_disk: set[str] = {f.stem for f in _ADAPTER_DIR.glob("hf_*.py")}
    stale: set[str] = committed.keys() - on_disk
    assert not stale, (
        f"{DATES_JSON.name} lists removed adapter(s): {sorted(stale)}. "
        "Regenerate it after deleting an adapter."
    )


def test_matches_git_on_full_checkout() -> None:
    """On a full checkout the committed map must equal what git yields.

    Skipped on a shallow clone — the exact condition the JSON exists to survive.
    In CI this is enforced by any job that checks out full history; a developer
    regenerating locally sees a failure here the moment a date drifts.
    """
    if _is_shallow_clone():
        pytest.skip("shallow clone: git cannot see real add-commits (see issue #372)")

    expected: str = render_json(compute_added_dates())
    actual: str = DATES_JSON.read_text(encoding="utf-8")
    assert actual == expected, (
        f"{DATES_JSON.name} is out of date. Regenerate it:\n"
        "    python tests/spyre/weekly_generation/regenerate_adapter_added_dates.py"
    )


# --------------------------------------------------------------------------- #
# check_hook(): the pre-commit gate. Tested by monkeypatching the git-touching
# helpers so each branch is exercised deterministically, on any clone depth.
# --------------------------------------------------------------------------- #


def test_check_hook_passes_on_current_tree() -> None:
    """The committed repo state satisfies the hook (a self-consistency check)."""
    ok, message = check_hook()
    assert ok, message


def test_check_hook_fails_when_committed_adapter_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A committed adapter with no JSON entry fails — the 'forgot to regen' case."""
    monkeypatch.setattr(regen, "_load_committed_map", lambda: {})  # entry absent
    monkeypatch.setattr(regen, "_is_committed", lambda rel_path: True)  # all committed
    ok, message = check_hook()
    assert not ok
    assert "missing entries for already-committed" in message


def test_check_hook_exempts_brand_new_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A brand-new staged adapter (not in HEAD, no entry) is allowed through.

    This is the two-step flow: the introducing commit cannot carry a date git
    does not have yet, so the hook must not block it.
    """
    monkeypatch.setattr(regen, "_load_committed_map", lambda: {})
    monkeypatch.setattr(regen, "_is_committed", lambda rel_path: False)  # all brand-new
    # Skip the full-checkout date comparison — irrelevant when nothing is committed.
    monkeypatch.setattr(regen, "_is_shallow", lambda: True)
    ok, message = check_hook()
    assert ok, message


def test_check_hook_fails_on_wrong_date_full_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a full checkout, a stored date that disagrees with git fails the hook."""
    if _is_shallow_clone():
        pytest.skip("shallow clone: git date comparison is skipped by design")

    real: dict[str, str | None] = _load_committed_map()
    poisoned: dict[str, str | None] = dict(real)
    # Flip one known adapter's date to the poisoned tip-date from issue #372.
    poisoned["hf_bert"] = "2026-08-14"
    monkeypatch.setattr(regen, "_load_committed_map", lambda: poisoned)
    ok, message = check_hook()
    assert not ok
    assert "out of date with git" in message
    assert "hf_bert" in message


def test_check_hook_skips_date_check_on_shallow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a shallow clone the hook passes as long as committed adapters have entries.

    Even a wrong date value passes here — the shallow clone cannot verify dates,
    which is the whole reason the committed JSON exists (issue #372).
    """
    poisoned: dict[str, str | None] = dict(_load_committed_map())
    poisoned["hf_bert"] = "2026-08-14"
    monkeypatch.setattr(regen, "_load_committed_map", lambda: poisoned)
    monkeypatch.setattr(regen, "_is_shallow", lambda: True)
    ok, message = check_hook()
    assert ok, message
