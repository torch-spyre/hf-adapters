"""Validate authoritative adapter dates, including shallow and squash merges.

Run with pytest --noconftest. Metadata checks require only stdlib + pytest;
regression fixtures use git to reproduce history changes without Spyre or a DB.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import Mock

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


def _load_committed_map() -> dict[str, object]:
    loaded: object = json.loads(DATES_JSON.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_dates_file_exists_and_parses() -> None:
    assert DATES_JSON.is_file(), f"{DATES_JSON} is missing"
    assert isinstance(_load_committed_map(), dict)


def test_every_adapter_has_a_valid_date() -> None:
    """Every adapter, including new files, must have a canonical ISO date."""
    committed: dict[str, object] = _load_committed_map()
    on_disk: set[str] = {path.stem for path in _ADAPTER_DIR.glob("hf_*.py")}
    missing: set[str] = on_disk - committed.keys()
    assert not missing, f"Adapters absent from {DATES_JSON.name}: {sorted(missing)}"
    for stem in sorted(on_disk):
        value: object = committed[stem]
        assert isinstance(value, str), f"Invalid date for {stem}: {value!r}"
        assert date.fromisoformat(value).isoformat() == value


def test_no_stale_entries() -> None:
    committed: dict[str, object] = _load_committed_map()
    on_disk: set[str] = {path.stem for path in _ADAPTER_DIR.glob("hf_*.py")}
    stale: set[str] = committed.keys() - on_disk
    assert not stale, f"Removed adapters in {DATES_JSON.name}: {sorted(stale)}"


def test_check_hook_passes_on_current_tree() -> None:
    ok, message = check_hook()
    assert ok, message


@pytest.fixture
def metadata_tree(tmp_path: Path) -> Path:
    """Metadata validation and updates also work outside a git checkout."""
    repo: Path = tmp_path / "full"
    repo.mkdir()
    (repo / "hf_adapters").mkdir()
    (repo / "resources").mkdir()
    (repo / "hf_adapters/hf_bert.py").write_text("# Original adapter\n")
    (repo / "resources/adapter_added_dates.json").write_text(
        render_json({"hf_bert": "2026-05-21"})
    )
    return repo


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
def dated_repo(metadata_tree: Path) -> Path:
    """A real repo whose later tip reproduces the poisoned date from issue #372."""
    _git(metadata_tree, "init", "--initial-branch=main")
    _git(metadata_tree, "add", ".")
    _git(metadata_tree, "commit", "-m", "Add BERT adapter and metadata")
    (metadata_tree / "unrelated.txt").write_text("An unrelated later change\n")
    _git(metadata_tree, "add", ".")
    _git(
        metadata_tree,
        "commit",
        "-m",
        "Later tip",
        timestamp="2026-08-14T12:00:00+00:00",
    )
    return metadata_tree


@pytest.fixture
def shallow_repo(dated_repo: Path, tmp_path: Path) -> Path:
    repo: Path = tmp_path / "shallow"
    _git(tmp_path, "clone", "--depth=1", "--no-local", str(dated_repo), str(repo))
    assert _git(repo, "rev-parse", "--is-shallow-repository") == "true"
    return repo


def _use_repo(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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


def test_shallow_checks_and_update_preserve_existing_dates(
    shallow_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates_json: Path = _use_repo(shallow_repo, monkeypatch)
    before: bytes = dates_json.read_bytes()
    assert regen.main(["--check"]) == 0
    assert regen.main(["--check-hook"]) == 0
    assert regen.main([]) == 0
    assert dates_json.read_bytes() == before
    (shallow_repo / "hf_adapters/hf_new.py").write_text("# New adapter\n")
    assert not check_hook()[0]
    assert regen.main(["--added-date", "2026-09-24"]) == 0
    assert json.loads(dates_json.read_text()) == {
        "hf_bert": "2026-05-21",
        "hf_new": "2026-09-24",
    }
    assert check_hook()[0]


def test_reader_checks_and_updater_never_call_git(
    metadata_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_subprocess(*args: object, **kwargs: object) -> None:
        raise AssertionError("Adapter metadata must not depend on git")

    dates_json: Path = _use_repo(metadata_tree, monkeypatch)
    monkeypatch.setattr(subprocess, "run", no_subprocess)
    assert load_added_dates(metadata_tree / "hf_adapters", dates_json) == {
        "hf_bert": "2026-05-21"
    }
    assert check_hook()[0]
    assert regen.main(["--check"]) == 0
    assert regen.main([]) == 0
    (metadata_tree / "hf_adapters/hf_new.py").write_text("# New adapter\n")
    assert load_added_dates(metadata_tree / "hf_adapters", dates_json)["hf_new"] is None
    assert regen.main(["--added-date", "2026-09-24"]) == 0
    assert check_hook()[0]


@pytest.mark.parametrize("contents", ["[", "[]", "null", '"wrong shape"'])
def test_invalid_json_stays_unknown_and_is_never_overwritten(
    metadata_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: str,
) -> None:
    dates_json: Path = _use_repo(metadata_tree, monkeypatch)
    dates_json.write_text(contents)
    before: bytes = dates_json.read_bytes()
    assert load_added_dates(metadata_tree / "hf_adapters", dates_json) == {
        "hf_bert": None
    }
    assert not check_hook()[0]
    assert regen.main([]) == 1
    assert dates_json.read_bytes() == before


@pytest.mark.parametrize("value", [None, 123, [], "2026-02-30", "20260521", "wrong"])
def test_invalid_date_is_rejected_and_never_overwritten(
    metadata_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    dates_json: Path = _use_repo(metadata_tree, monkeypatch)
    dates_json.write_text(json.dumps({"hf_bert": value}))
    before: bytes = dates_json.read_bytes()
    assert load_added_dates(metadata_tree / "hf_adapters", dates_json) == {
        "hf_bert": None
    }
    ok, message = check_hook()
    assert not ok
    assert "invalid dates" in message
    assert regen.main([]) == 1
    assert dates_json.read_bytes() == before


def test_removed_adapter_is_pruned_without_redating_others(
    metadata_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates_json: Path = _use_repo(metadata_tree, monkeypatch)
    dates_json.write_text(
        render_json({"hf_bert": "2026-05-21", "hf_removed": "2026-05-21"})
    )
    ok, message = check_hook()
    assert not ok
    assert "removed adapter(s): hf_removed" in message
    assert regen.main([]) == 0
    assert json.loads(dates_json.read_text()) == {"hf_bert": "2026-05-21"}


def test_missing_metadata_can_be_created_with_an_explicit_date(
    metadata_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates_json: Path = _use_repo(metadata_tree, monkeypatch)
    dates_json.unlink()
    assert not check_hook()[0]
    assert regen.main(["--added-date", "2026-05-21"]) == 0
    assert json.loads(dates_json.read_text()) == {"hf_bert": "2026-05-21"}


@pytest.mark.parametrize("argv", [["--check"], ["--check-hook"]])
def test_check_does_not_fill_a_missing_date(
    metadata_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    dates_json: Path = _use_repo(metadata_tree, monkeypatch)
    before: bytes = dates_json.read_bytes()
    (metadata_tree / "hf_adapters/hf_new.py").write_text("# New adapter\n")
    assert regen.main(argv) == 1
    assert dates_json.read_bytes() == before


def test_default_new_date_uses_utc_and_preserves_existing_dates(
    metadata_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_repo(metadata_tree, monkeypatch)
    (metadata_tree / "hf_adapters/hf_new.py").write_text("# New adapter\n")
    clock: Mock = Mock()
    clock.now.return_value = datetime(2026, 9, 24, 0, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(regen, "datetime", clock)
    assert compute_added_dates() == {"hf_bert": "2026-05-21", "hf_new": "2026-09-24"}
    clock.now.assert_called_once_with(timezone.utc)


def test_new_adapter_and_metadata_land_in_one_commit(
    dated_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_repo(dated_repo, monkeypatch)
    (dated_repo / "hf_adapters/hf_new.py").write_text("# New adapter\n")
    assert not check_hook()[0]  # New files are no longer exempt.
    assert regen.main(["--added-date", "2026-09-24"]) == 0
    assert check_hook()[0]
    _git(dated_repo, "add", ".")
    _git(
        dated_repo,
        "commit",
        "-m",
        "Add adapter with its date",
        timestamp="2026-09-24T12:00:00+00:00",
    )
    assert check_hook()[0]


def test_dates_survive_merge_queue_and_later_squash_merge(
    dated_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both synthetic queue history and main rewrite the introducing commit date."""
    dates_json: Path = _use_repo(dated_repo, monkeypatch)
    _git(dated_repo, "switch", "-c", "feature")
    (dated_repo / "hf_adapters/hf_new.py").write_text("# New adapter\n")
    assert regen.main(["--added-date", "2026-09-24"]) == 0
    _git(dated_repo, "add", ".")
    _git(
        dated_repo, "commit", "-m", "Add adapter", timestamp="2026-09-24T23:30:00-07:00"
    )
    before: bytes = dates_json.read_bytes()

    for branch, merged_on in [("merge-queue", "2026-09-27"), ("main", "2026-09-28")]:
        _git(dated_repo, "switch", "main")
        if branch != "main":
            _git(dated_repo, "switch", "-c", branch)
        _git(dated_repo, "merge", "--squash", "feature")
        _git(
            dated_repo,
            "commit",
            "-m",
            "Squash adapter PR",
            timestamp=f"{merged_on}T12:00:00+00:00",
        )
        actual_git_date: str = _git(
            dated_repo,
            "log",
            "--diff-filter=A",
            "--follow",
            "--format=%aI",
            "-1",
            "--",
            "hf_adapters/hf_new.py",
        )[:10]
        assert actual_git_date == merged_on
        assert actual_git_date != "2026-09-24"
        assert check_hook()[0]
        assert regen.main(["--check"]) == 0
        assert regen.main([]) == 0
        assert dates_json.read_bytes() == before
        assert (
            load_added_dates(dated_repo / "hf_adapters", dates_json)["hf_new"]
            == "2026-09-24"
        )

    # A subsequent PR can change an adapter without being blocked by rewritten history.
    _git(dated_repo, "switch", "-c", "next-pr")
    (dated_repo / "hf_adapters/hf_bert.py").write_text("# Updated adapter\n")
    assert check_hook()[0]
    assert regen.main(["--added-date", "2026-10-01"]) == 0
    assert dates_json.read_bytes() == before


def test_renamed_adapter_keeps_its_authoritative_date(
    dated_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates_json: Path = _use_repo(dated_repo, monkeypatch)
    _git(dated_repo, "mv", "hf_adapters/hf_bert.py", "hf_adapters/hf_renamed.py")
    dates_json.write_text(render_json({"hf_renamed": "2026-05-21"}))
    assert regen.main([]) == 0
    _git(dated_repo, "add", ".")
    _git(
        dated_repo,
        "commit",
        "-m",
        "Rename adapter with metadata",
        timestamp="2026-09-24T12:00:00+00:00",
    )
    assert json.loads(dates_json.read_text()) == {"hf_renamed": "2026-05-21"}
    assert check_hook()[0]
