"""Unit tests for the weekly-scan weight cleanup (#738).

Lives at the ``tests/`` root and runs with ``pytest --noconftest`` like
``test_weekly_prefilter.py``, so only pytest itself is needed. The tests build
a miniature HF hub cache in ``tmp_path`` — shared-blob store, repo-local etag
symlinks, snapshot symlinks, ``.refs`` refcounts — and assert that deleting
one repo's weights never destroys a blob another repo still references.

The failure these guard against is the 1 Nov CI breakage: the scan cleaned up
a mirror repo, unlinked the shared blob its weights deduplicated into, and
offline CI found configs but no ``model.safetensors`` for a registry model.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
from huggingface_hub import constants as hf_constants

from tests.spyre.weekly_generation.weekly_test import _delete_repo_weights

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MANAGE_CACHE_PATH = _REPO_ROOT / ".github" / "cache_config" / "manage_cache.py"


def _load_manage_cache():
    spec = importlib.util.spec_from_file_location("manage_cache", _MANAGE_CACHE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def fake_hub(tmp_path, monkeypatch):
    """Build a tiny HF hub cache and point HF_HUB_CACHE at it."""
    hub = tmp_path / "hub"
    (hub / "blobs").mkdir(parents=True)
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(hub))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(hub))

    def shared_blob(sha: str, size: int) -> Path:
        shard_dir = hub / "blobs" / sha[:2]
        shard_dir.mkdir(exist_ok=True)
        target = shard_dir / sha
        target.write_bytes(b"x" * size)
        return target

    def add_repo(repo_id: str, files: dict[str, tuple[str, int]]) -> Path:
        """files maps snapshot name -> (shared sha | "local", size)."""
        repo_dir = hub / ("models--" + repo_id.replace("/", "--"))
        (repo_dir / "blobs").mkdir(parents=True)
        snap_dir = repo_dir / "snapshots" / "deadbeef"
        snap_dir.mkdir(parents=True)
        for name, (sha, size) in files.items():
            if sha == "local":
                entry = repo_dir / "blobs" / f"etag-{name}"
                entry.write_bytes(b"y" * size)
            else:
                target = shared_blob(sha, size)
                entry = repo_dir / "blobs" / f"etag-{name}"
                entry.symlink_to(os.path.relpath(target, entry.parent))
            (snap_dir / name).symlink_to(os.path.relpath(entry, snap_dir))
        return repo_dir

    def write_refs(sha: str, entries: list[str]) -> None:
        target = hub / "blobs" / sha[:2] / sha
        target.with_name(target.name + ".refs").write_text("\n".join(entries) + "\n")

    return type(
        "FakeHub",
        (),
        {"hub": hub, "shared_blob": staticmethod(shared_blob),
         "add_repo": staticmethod(add_repo), "write_refs": staticmethod(write_refs)},
    )()


def test_shared_blob_survives_while_another_repo_references_it(fake_hub):
    """Deleting the mirror must not break the CI model sharing its blob."""
    sha = "aa" * 32
    ci = fake_hub.add_repo("Qwen/Qwen3-0.6B", {"model.safetensors": (sha, 100)})
    mirror = fake_hub.add_repo("PrimeIntellect/Qwen3-0.6B", {"model.safetensors": (sha, 100)})
    fake_hub.write_refs(
        sha,
        ["models--Qwen--Qwen3-0.6B/blobs/etag-model.safetensors",
         "models--PrimeIntellect--Qwen3-0.6B/blobs/etag-model.safetensors"],
    )

    freed = _delete_repo_weights(["PrimeIntellect/Qwen3-0.6B"])

    target = fake_hub.hub / "blobs" / "aa" / sha
    assert target.is_file(), "shared blob was unlinked while the CI repo still references it"
    assert (ci / "snapshots" / "deadbeef" / "model.safetensors").exists()
    assert not mirror.exists() or not any(mirror.rglob("*.safetensors"))
    assert freed == 0, "no bytes should be freed while the blob stays"


def test_shared_blob_removed_once_no_repo_references_it(fake_hub):
    sha = "ab" * 32
    fake_hub.add_repo("a/gone", {"model.safetensors": (sha, 512)})
    fake_hub.add_repo("a/also-gone", {"model.safetensors": (sha, 512)})
    fake_hub.write_refs(
        sha,
        ["models--a--gone/blobs/etag-model.safetensors",
         "models--a--also-gone/blobs/etag-model.safetensors"],
    )

    _delete_repo_weights(["a/gone"])
    target = fake_hub.hub / "blobs" / "ab" / sha
    assert target.is_file(), "blob removed while one referencer is left"

    freed = _delete_repo_weights(["a/also-gone"])
    assert not target.exists(), "blob kept after its last referencer was deleted"
    assert freed == 512


def test_shared_blob_without_refs_is_kept(fake_hub):
    """No .refs means the refcount is unknown: keep the blob, drop only links."""
    sha = "cd" * 32
    repo = fake_hub.add_repo("a/scan", {"model.safetensors": (sha, 256)})

    _delete_repo_weights(["a/scan"])

    assert (fake_hub.hub / "blobs" / "cd" / sha).is_file()
    assert not list((repo / "snapshots").rglob("*.safetensors"))


def test_old_layout_real_blob_file_is_freed(fake_hub):
    repo = fake_hub.add_repo("a/old", {"pytorch_model.bin": ("local", 128)})

    freed = _delete_repo_weights(["a/old"])

    assert freed == 128
    assert not list(repo.rglob("*.bin"))


def test_rebuild_shared_blob_refs_recovers_unlisted_referencers(fake_hub):
    """A repo cached before the migration is absent from .refs; the rebuild
    must add it so a refs-based delete cannot orphan it (#738 root cause)."""
    sha = "ef" * 32
    fake_hub.add_repo("Qwen/Qwen3-0.6B", {"model.safetensors": (sha, 64)})
    fake_hub.add_repo("some/mirror", {"model.safetensors": (sha, 64)})
    fake_hub.write_refs(sha, ["models--some--mirror/blobs/etag-model.safetensors"])

    manage_cache = _load_manage_cache()
    written = manage_cache.rebuild_shared_blob_refs(fake_hub.hub)

    assert written >= 1
    refs = (fake_hub.hub / "blobs" / "ef" / (sha + ".refs")).read_text().split()
    assert "models--Qwen--Qwen3-0.6B/blobs/etag-model.safetensors" in refs
    assert "models--some--mirror/blobs/etag-model.safetensors" in refs

    # End to end: deleting the mirror must leave the CI model intact.
    _delete_repo_weights(["some/mirror"])
    assert (fake_hub.hub / "blobs" / "ef" / sha).is_file()
