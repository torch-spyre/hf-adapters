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
import os
import subprocess
from datetime import date
from pathlib import Path

import pytest

from tests.spyre.weekly_generation import regenerate_adapter_added_dates as regen
from tests.spyre.weekly_generation.adapter_added_dates import load_added_dates
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


def _git(repo: Path, *args: str, timestamp: str = "2026-05-21T12:00:00+00:00") -> str:
    """Run git in a disposable fixture with deterministic author/commit dates."""
    env: dict[str, str] = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="Adapter date test",
        GIT_AUTHOR_EMAIL="adapter-date-test@example.com",
        GIT_COMMITTER_NAME="Adapter date test",
        GIT_COMMITTER_EMAIL="adapter-date-test@example.com",
        GIT_AUTHOR_DATE=timestamp,
        GIT_COMMITTER_DATE=timestamp,
    )
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def dated_repo(tmp_path: Path) -> Path:
    """A real repo whose tip date reproduces the poisoned date from issue #372."""
    repo: Path = tmp_path / "full"
    repo.mkdir()
    _git(repo, "init")
    (repo / "hf_adapters").mkdir()
    (repo / "resources").mkdir()
    (repo / "hf_adapters/hf_bert.py").write_text("# Original adapter\n")
    (repo / "resources/adapter_added_dates.json").write_text(
        render_json({"hf_bert": "2026-05-21"})
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Add BERT adapter and metadata")
    (repo / "unrelated.txt").write_text("An unrelated later change\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Later tip", timestamp="2026-08-14T12:00:00+00:00")
    return repo


@pytest.fixture
def shallow_repo(dated_repo: Path, tmp_path: Path) -> Path:
    repo: Path = tmp_path / "shallow"
    _git(tmp_path, "clone", "--depth=1", "--no-local", str(dated_repo), str(repo))
    assert _git(repo, "rev-parse", "--is-shallow-repository") == "true"
    return repo


def _use_repo(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the regenerator at a disposable repository."""
    dates_json: Path = repo / "resources/adapter_added_dates.json"
    monkeypatch.setattr(regen, "_REPO_ROOT", repo)
    monkeypatch.setattr(regen, "_ADAPTER_DIR", repo / "hf_adapters")
    monkeypatch.setattr(regen, "DATES_JSON", dates_json)
    return dates_json


def test_shallow_scan_uses_real_committed_date(shallow_repo: Path) -> None:
    """The depth-one git query is wrong, but the scan's metadata reader is right."""
    poisoned: str = _git(
        shallow_repo,
        "log",
        "--diff-filter=A",
        "--follow",
        "--format=%aI",
        "-1",
        "--",
        "hf_adapters/hf_bert.py",
    )
    assert poisoned[:10] == "2026-08-14"
    assert load_added_dates(
        shallow_repo / "hf_adapters",
        shallow_repo / "resources/adapter_added_dates.json",
    ) == {"hf_bert": "2026-05-21"}


@pytest.mark.parametrize("argv", [[], ["--check"]])
def test_regenerator_refuses_shallow_history_without_writing(
    shallow_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    dates_json: Path = _use_repo(shallow_repo, monkeypatch)
    before: bytes = dates_json.read_bytes()
    assert regen.main(argv) == 1
    assert "shallow checkout" in capsys.readouterr().err
    assert dates_json.read_bytes() == before
    with pytest.raises(ValueError, match="shallow checkout"):
        compute_added_dates()


def test_hook_accepts_complete_metadata_on_real_shallow_clone(
    shallow_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_repo(shallow_repo, monkeypatch)
    ok, message = check_hook()
    assert ok, message
    assert "shallow clone" in message


def test_scan_reader_never_calls_git(
    dated_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_subprocess(*args: object, **kwargs: object) -> None:
        raise AssertionError("The scan must read metadata without calling git")

    monkeypatch.setattr(subprocess, "run", no_subprocess)
    (dated_repo / "hf_adapters/hf_new.py").write_text("# New adapter\n")
    assert load_added_dates(
        dated_repo / "hf_adapters", dated_repo / "resources/adapter_added_dates.json"
    ) == {"hf_bert": "2026-05-21", "hf_new": None}


@pytest.mark.parametrize("contents", ["[", "[]", "null", '"wrong shape"'])
def test_invalid_json_stays_unknown(
    dated_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: str,
) -> None:
    dates_json: Path = _use_repo(dated_repo, monkeypatch)
    dates_json.write_text(contents)
    assert load_added_dates(dated_repo / "hf_adapters", dates_json) == {"hf_bert": None}
    assert not check_hook()[0]


@pytest.mark.parametrize("value", [None, 123, [], "2026-02-30", "20260521", "wrong"])
def test_invalid_date_rejected_even_on_shallow_clone(
    shallow_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    dates_json: Path = _use_repo(shallow_repo, monkeypatch)
    dates_json.write_text(json.dumps({"hf_bert": value}))
    assert load_added_dates(shallow_repo / "hf_adapters", dates_json) == {
        "hf_bert": None
    }
    ok, message = check_hook()
    assert not ok
    assert "invalid dates" in message


def test_hook_rejects_removed_adapter_on_shallow_clone(
    shallow_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates_json: Path = _use_repo(shallow_repo, monkeypatch)
    dates_json.write_text(
        render_json({"hf_bert": "2026-05-21", "hf_removed": "2026-05-21"})
    )
    ok, message = check_hook()
    assert not ok
    assert "removed adapter(s): hf_removed" in message


def test_new_adapter_two_commit_workflow(
    dated_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates_json: Path = _use_repo(dated_repo, monkeypatch)
    before: bytes = dates_json.read_bytes()
    (dated_repo / "hf_adapters/hf_new.py").write_text("# New adapter\n")
    _git(dated_repo, "add", ".")
    ok, message = check_hook()
    assert ok, message  # The introducing commit is exempt.
    assert regen.main([]) == 1  # No date exists until that commit lands.
    assert dates_json.read_bytes() == before
    _git(
        dated_repo,
        "commit",
        "-m",
        "Add new adapter",
        timestamp="2026-09-24T12:00:00+00:00",
    )
    assert not check_hook()[0]  # The follow-up must regenerate the map.
    assert regen.main([]) == 0
    assert json.loads(dates_json.read_text()) == {
        "hf_bert": "2026-05-21",
        "hf_new": "2026-09-24",
    }
    ok, message = check_hook()
    assert ok, message


def test_renamed_adapter_preserves_original_date(
    dated_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates_json: Path = _use_repo(dated_repo, monkeypatch)
    _git(dated_repo, "mv", "hf_adapters/hf_bert.py", "hf_adapters/hf_renamed.py")
    _git(
        dated_repo,
        "commit",
        "-m",
        "Rename adapter",
        timestamp="2026-09-24T12:00:00+00:00",
    )
    assert regen.main([]) == 0
    assert json.loads(dates_json.read_text()) == {"hf_renamed": "2026-05-21"}


def test_git_error_does_not_write_metadata(
    dated_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates_json: Path = _use_repo(dated_repo, monkeypatch)
    before: bytes = dates_json.read_bytes()

    def no_git(*args: object, **kwargs: object) -> None:
        raise OSError("git is unavailable")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert regen.main([]) == 1
    assert dates_json.read_bytes() == before
