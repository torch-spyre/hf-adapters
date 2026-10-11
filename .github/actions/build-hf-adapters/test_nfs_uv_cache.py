# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from torch-spyre/spyre-inference PR #1193
# (.github/scripts/test_nfs_uv_cache.py); the tests after
# test_key_is_path_independent cover the hf-adapters changes (see
# nfs_uv_cache.py). Run: python3 -m pytest .github/actions/build-hf-adapters

"""Logic tests for the NFS-backed uv cache helper: key, scope rules, atomic
save, cross-scope isolation, and GC. The tar roundtrip runs on real files.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import nfs_uv_cache as nuc  # noqa: E402


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def lockfiles(tmp_path):
    a = _write(tmp_path / "uv.lock", "lock-a\n")
    b = _write(tmp_path / "spyre-rpms.lock", "rpm-b\n")
    return [a, b]


def test_key_is_deterministic(lockfiles):
    assert nuc.compute_key(lockfiles) == nuc.compute_key(lockfiles)
    assert len(nuc.compute_key(lockfiles)) == nuc.KEY_LEN


def test_key_tracks_content_and_order(tmp_path):
    a = _write(tmp_path / "a.lock", "one")
    b = _write(tmp_path / "b.lock", "two")
    base = nuc.compute_key([a, b])
    _write(tmp_path / "a.lock", "one-changed")
    assert nuc.compute_key([a, b]) != base
    _write(tmp_path / "a.lock", "one")
    assert nuc.compute_key([b, a]) != base


def test_key_salt_busts_everything(lockfiles):
    base = nuc.compute_key(lockfiles)
    assert nuc.compute_key(lockfiles, salt="nfs-uv-cache-v2") != base


@pytest.mark.parametrize(
    "event,ref,pr,expected",
    [
        ("push", "refs/heads/main", "", "main"),
        ("push", "refs/heads/feature", "", None),
        ("pull_request", "refs/pull/7/merge", "7", "pr-7"),
        ("pull_request", "refs/pull/7/merge", "", None),
        ("merge_group", "refs/heads/gh-readonly-queue/main/x", "", None),
        ("workflow_dispatch", "refs/heads/main", "", None),
        ("schedule", "refs/heads/main", "", None),
    ],
)
def test_write_scope(event, ref, pr, expected):
    assert nuc.write_scope(event, ref, pr) == expected


def test_read_scopes_pr_prefers_own_then_main():
    assert nuc.read_scopes("pull_request", "refs/pull/9/merge", "9") == ["pr-9", "main"]


@pytest.mark.parametrize(
    "event,ref,pr",
    [
        ("push", "refs/heads/main", ""),
        ("merge_group", "refs/heads/gh-readonly-queue/main/x", ""),
        ("workflow_dispatch", "refs/heads/main", ""),
        ("pull_request", "refs/pull/9/merge", ""),  # no number -> main only
    ],
)
def test_read_scopes_non_pr_is_main_only(event, ref, pr):
    assert nuc.read_scopes(event, ref, pr) == ["main"]


def _args(cmd, nfs, local, keyfiles, arch="x86_64", **extra):
    argv = [cmd, "--nfs-root", str(nfs), "--local-dir", str(local), "--arch", arch]
    for kf in keyfiles:
        argv += ["--key-file", kf]
    for k, v in extra.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv


def _populate(cache_dir):
    (cache_dir / "wheels").mkdir(parents=True, exist_ok=True)
    (cache_dir / "wheels" / "pkg.whl").write_bytes(b"PK\x03\x04 fake wheel")
    (cache_dir / "marker.txt").write_text("built")


def _push_main():
    return {"event_name": "push", "ref": "refs/heads/main"}


def _pr(n):
    return {"event_name": "pull_request", "ref": f"refs/pull/{n}/merge", "pr_number": n}


def test_roundtrip_ship_wipe_restore_main(tmp_path, lockfiles):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    _populate(local)

    assert nuc.main(_args("save", nfs, local, lockfiles, **_push_main())) == 0
    key = nuc.compute_key(lockfiles)
    assert (nfs / "main" / "x86_64" / f"{key}.tar").is_file()

    shutil.rmtree(local)
    gh_env = tmp_path / "env"
    gh_out = tmp_path / "out"
    assert (
        nuc.main(
            _args(
                "restore",
                nfs,
                local,
                lockfiles,
                github_env=gh_env,
                github_output=gh_out,
                **_push_main(),
            )
        )
        == 0
    )
    assert (local / "wheels" / "pkg.whl").read_bytes() == b"PK\x03\x04 fake wheel"
    assert f"UV_CACHE_DIR={local}" in gh_env.read_text()
    assert "cache-hit=true" in gh_out.read_text()
    assert f"cache-key={key}" in gh_out.read_text()


def test_pr_writes_isolated_scope_main_cannot_read(tmp_path):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    pr_lock = [_write(tmp_path / "uv.lock", "pr-only-lock")]
    _populate(local)
    assert nuc.main(_args("save", nfs, local, pr_lock, **_pr("42"))) == 0
    key = nuc.compute_key(pr_lock)
    assert (nfs / "pr-42" / "x86_64" / f"{key}.tar").is_file()
    assert not (nfs / "main").exists()

    shutil.rmtree(local)
    gh_out = tmp_path / "out"
    assert (
        nuc.main(
            _args("restore", nfs, local, pr_lock, github_output=gh_out, **_push_main())
        )
        == 0
    )
    assert "cache-hit=false" in gh_out.read_text()
    assert not (local / "marker.txt").exists()


def test_pr_reads_own_scope_as_hit(tmp_path):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    pr_lock = [_write(tmp_path / "uv.lock", "pr-lock")]
    _populate(local)
    assert nuc.main(_args("save", nfs, local, pr_lock, **_pr("42"))) == 0

    shutil.rmtree(local)
    gh_out = tmp_path / "out"
    assert (
        nuc.main(
            _args("restore", nfs, local, pr_lock, github_output=gh_out, **_pr("42"))
        )
        == 0
    )
    assert "cache-hit=true" in gh_out.read_text()
    assert (local / "marker.txt").exists()


def test_pr_falls_back_to_main_scope(tmp_path):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    lock = [_write(tmp_path / "uv.lock", "shared")]
    _populate(local)
    assert nuc.main(_args("save", nfs, local, lock, **_push_main())) == 0

    shutil.rmtree(local)
    gh_out = tmp_path / "out"
    assert (
        nuc.main(_args("restore", nfs, local, lock, github_output=gh_out, **_pr("7")))
        == 0
    )
    assert "cache-hit=true" in gh_out.read_text()  # falls back to main
    assert (local / "marker.txt").exists()

    _populate(local)
    assert nuc.main(_args("save", nfs, local, lock, **_pr("7"))) == 0
    assert not (nfs / "pr-7").exists()


def test_save_skipped_in_readonly_context(tmp_path, lockfiles):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    _populate(local)
    assert (
        nuc.main(
            _args(
                "save",
                nfs,
                local,
                lockfiles,
                event_name="merge_group",
                ref="refs/heads/gh-readonly-queue/main/x",
            )
        )
        == 0
    )
    assert not (nfs / "main").exists()


def test_save_idempotent_when_key_present(tmp_path, lockfiles):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    _populate(local)
    assert nuc.main(_args("save", nfs, local, lockfiles, **_push_main())) == 0
    key = nuc.compute_key(lockfiles)
    tar = nfs / "main" / "x86_64" / f"{key}.tar"
    first = tar.stat().st_mtime_ns
    (local / "marker.txt").write_text("DIFFERENT")
    assert nuc.main(_args("save", nfs, local, lockfiles, **_push_main())) == 0
    assert tar.stat().st_mtime_ns == first


def test_corrupt_tar_restores_cold(tmp_path, lockfiles):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    key = nuc.compute_key(lockfiles)
    (nfs / "main" / "x86_64").mkdir(parents=True)
    (nfs / "main" / "x86_64" / f"{key}.tar").write_bytes(b"not a tar at all")
    (local / "stale").mkdir(parents=True)
    gh_out = tmp_path / "out"
    assert (
        nuc.main(
            _args(
                "restore", nfs, local, lockfiles, github_output=gh_out, **_push_main()
            )
        )
        == 0
    )
    assert not (local / "stale").exists()
    assert "cache-hit=false" in gh_out.read_text()


def test_gc_keeps_newest_and_protects_target(tmp_path):
    arch_dir = tmp_path / "main" / "x86_64"
    arch_dir.mkdir(parents=True)
    tars = []
    for i in range(5):
        t = arch_dir / f"k{i}.tar"
        t.write_bytes(b"x")
        os.utime(t, (time.time() + i, time.time() + i))
        tars.append(t)
    nuc._gc(arch_dir, keep=2, protect=tars[0])
    survivors = {p.name for p in arch_dir.iterdir()}
    assert survivors == {"k4.tar", "k3.tar", "k0.tar"}


def test_sweep_evicts_stale_pr_scopes_only(tmp_path):
    nfs = tmp_path / "nfs"
    old = nfs / "pr-1" / "x86_64"
    fresh = nfs / "pr-2" / "x86_64"
    main = nfs / "main" / "x86_64"
    for d in (old, fresh, main):
        d.mkdir(parents=True)
        (d / "k.tar").write_bytes(b"x")
    stale = time.time() - 30 * 86400
    os.utime(old / "k.tar", (stale, stale))
    os.utime(main / "k.tar", (stale, stale))  # main is never swept by ttl
    nuc._sweep_pr_scopes(nfs, ttl_days=14)
    assert not (old / "k.tar").exists()
    assert (fresh / "k.tar").exists()
    assert (main / "k.tar").exists()


def test_partial_restore_from_newest_tar(tmp_path):
    """A new key with no exact tar extracts the scope's newest tar: warm, not a hit."""
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    _populate(local)
    assert (
        nuc.main(
            _args(
                "save", nfs, local, [_write(tmp_path / "uv.lock", "v1")], **_push_main()
            )
        )
        == 0
    )

    shutil.rmtree(local)
    gh_out = tmp_path / "out"
    assert (
        nuc.main(
            _args(
                "restore",
                nfs,
                local,
                [_write(tmp_path / "uv.lock", "v2")],
                github_output=gh_out,
                **_push_main(),
            )
        )
        == 0
    )
    text = gh_out.read_text()
    assert "cache-hit=false" in text
    assert "restored-from=main/" in text
    assert (local / "marker.txt").exists()


@pytest.mark.parametrize("compress", ["gzip", "zstd"])
def test_compression_roundtrip(tmp_path, lockfiles, compress):
    if compress == "zstd" and shutil.which("zstd") is None:
        pytest.skip("zstd binary not installed")
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    _populate(local)
    assert (
        nuc.main(
            _args("save", nfs, local, lockfiles, compress=compress, **_push_main())
        )
        == 0
    )
    ext = nuc.COMPRESS[compress][1]
    assert (nfs / "main" / "x86_64" / f"{nuc.compute_key(lockfiles)}{ext}").is_file()

    shutil.rmtree(local)
    gh_out = tmp_path / "out"
    assert (
        nuc.main(
            _args(
                "restore", nfs, local, lockfiles, github_output=gh_out, **_push_main()
            )
        )
        == 0
    )
    assert "cache-hit=true" in gh_out.read_text()
    assert (local / "marker.txt").read_text() == "built"


def test_key_is_path_independent(tmp_path):
    """Byte-identical lockfiles at different absolute paths must hash the same."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ka = _write(tmp_path / "a" / "uv.lock", "same-bytes")
    kb = _write(tmp_path / "b" / "uv.lock", "same-bytes")
    assert nuc.compute_key([ka]) == nuc.compute_key([kb])


# --- hf-adapters additions -------------------------------------------------


def _populate_uv_like(cache_dir):
    """A uv-shaped cache: content-keyed parts plus local-source builds."""
    keep = {
        "archive-v0/abc/pkg/__init__.py": "registry wheel contents",
        "wheels-v6/pypi/pkg/pkg-1.0-py3-none-any.http": "http entry",
        "simple-v25/pypi/pkg.rkyv": "index page",
        "sdists-v9/git/1234/abcd/oot.whl": "git build, keyed by commit",
        "git-v1/db/1234/HEAD": "git db",
    }
    drop = {
        "sdists-v9/path/5678/rev/torch_spyre-0.6.0-cp312.whl": "local build",
        "sdists-v9/editable/9abc/rev/hf_adapters.whl": "editable build",
        "wheels-v6/path/def0/torch_spyre.http": "local wheel file",
        "builds-v0/.tmpXYZ/bin/python": "build env",
    }
    for rel, text in {**keep, **drop}.items():
        (cache_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (cache_dir / rel).write_text(text)
    return keep, drop


def test_save_excludes_local_source_builds(tmp_path, lockfiles):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    keep, drop = _populate_uv_like(local)
    assert nuc.main(_args("save", nfs, local, lockfiles, **_push_main())) == 0

    shutil.rmtree(local)
    gh_out = tmp_path / "out"
    assert (
        nuc.main(
            _args(
                "restore", nfs, local, lockfiles, github_output=gh_out, **_push_main()
            )
        )
        == 0
    )
    assert "status=hit" in gh_out.read_text()
    for rel, text in keep.items():
        assert (local / rel).read_text() == text
    for rel in drop:
        assert not (local / rel).exists(), rel
    # The bucket dirs themselves are gone, not just their files.
    assert not (local / "sdists-v9" / "path").exists()
    assert not (local / "builds-v0").exists()
    assert (local / "sdists-v9" / "git").is_dir()


def test_save_extra_exclude(tmp_path, lockfiles):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    _populate_uv_like(local)
    argv = _args("save", nfs, local, lockfiles, **_push_main())
    assert nuc.main(argv + ["--exclude", "./git-v*"]) == 0
    shutil.rmtree(local)
    assert nuc.main(_args("restore", nfs, local, lockfiles, **_push_main())) == 0
    assert not (local / "git-v1").exists()
    assert (local / "archive-v0").is_dir()


def test_no_partial_restores_cold_on_key_miss(tmp_path):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    _populate(local)
    old = [_write(tmp_path / "uv.lock", "v1")]
    assert nuc.main(_args("save", nfs, local, old, **_push_main())) == 0

    shutil.rmtree(local)
    gh_out = tmp_path / "out"
    new = [_write(tmp_path / "uv.lock", "v2")]
    argv = _args("restore", nfs, local, new, github_output=gh_out, **_push_main())
    assert nuc.main(argv + ["--no-partial"]) == 0
    text = gh_out.read_text()
    assert "status=miss" in text
    assert "restored-from=\n" in text
    assert local.is_dir() and not any(local.iterdir())


def test_no_partial_still_takes_exact_hit(tmp_path, lockfiles):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    _populate(local)
    assert nuc.main(_args("save", nfs, local, lockfiles, **_pr("5"))) == 0
    shutil.rmtree(local)
    gh_out = tmp_path / "out"
    argv = _args("restore", nfs, local, lockfiles, github_output=gh_out, **_pr("5"))
    assert nuc.main(argv + ["--no-partial"]) == 0
    assert "status=hit" in gh_out.read_text()
    assert (local / "marker.txt").exists()


def test_status_output_partial_and_error(tmp_path, lockfiles):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    _populate(local)
    (tmp_path / "old").mkdir()
    old_lock = [_write(tmp_path / "old" / "uv.lock", "old")]
    assert nuc.main(_args("save", nfs, local, old_lock, **_push_main())) == 0
    gh_out = tmp_path / "out"
    assert (
        nuc.main(
            _args(
                "restore", nfs, local, lockfiles, github_output=gh_out, **_push_main()
            )
        )
        == 0
    )
    assert "status=partial" in gh_out.read_text()

    # Corrupt the only (newest) tar: restore wipes and reports an error.
    for tar in (nfs / "main" / "x86_64").iterdir():
        tar.write_bytes(b"garbage")
    gh_out2 = tmp_path / "out2"
    assert (
        nuc.main(
            _args(
                "restore", nfs, local, lockfiles, github_output=gh_out2, **_push_main()
            )
        )
        == 0
    )
    assert "status=error" in gh_out2.read_text()
    assert not any(local.iterdir())


def test_unreadable_nfs_root_is_a_cold_miss_not_a_failure(tmp_path, lockfiles):
    blocker = tmp_path / "file"
    blocker.write_text("not a dir")
    local = tmp_path / "uvcache"
    gh_out = tmp_path / "out"
    gh_env = tmp_path / "env"
    argv = _args(
        "restore",
        blocker / "nfs",
        local,
        lockfiles,
        github_output=gh_out,
        github_env=gh_env,
        **_push_main(),
    )
    assert nuc.main(argv) == 0
    assert "status=miss" in gh_out.read_text()
    assert f"UV_CACHE_DIR={local}" in gh_env.read_text()
    # Save to an unwritable root warns and still exits 0.
    _populate(local)
    assert (
        nuc.main(_args("save", blocker / "nfs", local, lockfiles, **_push_main())) == 0
    )


def test_pr_number_from_event_payload(tmp_path):
    event = tmp_path / "event.json"
    event.write_text('{"pull_request": {"number": 656}}')
    assert nuc.pr_number_from_event(str(event)) == "656"
    event.write_text('{"ref": "refs/heads/main"}')
    assert nuc.pr_number_from_event(str(event)) == ""
    event.write_text("not json")
    assert nuc.pr_number_from_event(str(event)) == ""
    assert nuc.pr_number_from_event(str(tmp_path / "missing.json")) == ""
    assert nuc.pr_number_from_event("") == ""


def test_save_scope_from_event_payload(tmp_path, lockfiles):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    _populate(local)
    event = tmp_path / "event.json"
    event.write_text('{"pull_request": {"number": 77}}')
    argv = _args(
        "save",
        nfs,
        local,
        lockfiles,
        event_name="pull_request",
        ref="refs/pull/77/merge",
        event_path=event,
    )
    assert nuc.main(argv) == 0
    key = nuc.compute_key(lockfiles)
    assert (nfs / "pr-77" / "x86_64" / f"{key}.tar").is_file()
    assert not (nfs / "main").exists()


def test_pr_cannot_write_main_even_when_main_is_empty(tmp_path, lockfiles):
    nfs = tmp_path / "nfs"
    local = tmp_path / "uvcache"
    _populate(local)
    for event, ref, pr in [
        ("pull_request", "refs/heads/main", "3"),
        ("pull_request", "refs/pull/3/merge", ""),
        ("workflow_dispatch", "refs/heads/main", ""),
        ("schedule", "refs/heads/main", ""),
        ("push", "refs/tags/v1", ""),
    ]:
        argv = _args("save", nfs, local, lockfiles, event_name=event, ref=ref)
        assert nuc.main(argv + (["--pr-number", pr] if pr else [])) == 0
    assert not (nfs / "main").exists()
    assert [p.name for p in nfs.iterdir()] == ["pr-3"]


def test_concurrent_savers_leave_one_complete_tar(tmp_path, lockfiles):
    """Eight processes save one new key at once: one tar, no temp tars left."""
    import subprocess

    nfs = tmp_path / "nfs"
    procs = []
    for i in range(8):
        local = tmp_path / f"uvcache{i}"
        _populate(local)
        (local / "big.bin").write_bytes(os.urandom(4 << 20))
        argv = _args("save", nfs, local, lockfiles, **_push_main())
        script = pathlib.Path(nuc.__file__)
        procs.append(subprocess.Popen([sys.executable, "-I", str(script), *argv]))
    assert [p.wait(timeout=120) for p in procs] == [0] * 8
    arch_dir = nfs / "main" / "x86_64"
    names = sorted(p.name for p in arch_dir.iterdir())
    assert names == [f"{nuc.compute_key(lockfiles)}.tar"]
    out = tmp_path / "restored"
    assert nuc.main(_args("restore", nfs, out, lockfiles, **_push_main())) == 0
    assert (out / "big.bin").stat().st_size == 4 << 20


def test_sweep_stale_tmp(tmp_path):
    arch_dir = tmp_path / "main" / "x86_64"
    arch_dir.mkdir(parents=True)
    old = arch_dir / ".k.1.a.tmp"
    new = arch_dir / ".k.2.b.tmp"
    tar = arch_dir / "k.tar"
    for p in (old, new, tar):
        p.write_bytes(b"x")
    stale = time.time() - 2 * nuc.STALE_TMP_SECONDS
    os.utime(old, (stale, stale))
    os.utime(tar, (stale, stale))
    nuc._sweep_stale_tmp(arch_dir)
    assert not old.exists()
    assert new.exists()
    assert tar.exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
