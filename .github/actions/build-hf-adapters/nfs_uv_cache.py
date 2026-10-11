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
# (.github/scripts/nfs_uv_cache.py, by joerunde). Changes for hf-adapters:
#   * save leaves out the parts of uv's cache built from local sources
#     (EXCLUDE_DEFAULT: the torch-spyre checkout and the hf-adapters project),
#     so a snapshot can never hand a job a stale torch-spyre build;
#   * restore --no-partial, for the one job that writes (no newest-tar
#     fallback, so a snapshot holds only what its own uv.lock needs);
#   * the PR number is read from GITHUB_EVENT_PATH when --pr-number is unset
#     (the node shim in #1193 did that; hf-adapters calls the helper directly);
#   * restore writes a `status` output (hit|partial|miss|error);
#   * save cleans up its temp tar on SIGTERM (step timeout) and removes
#     abandoned temp tars older than a day.

"""Ship the uv cache to and from the shared NFS mount, actions/cache style.

uv only ever sees a plain local cache dir, so `uv sync` never takes a lock on
NFS -- pointing UV_CACHE_DIR straight at NFS serializes every pod on uv's
per-path build locks, which NFS hands between waiters only every ~20-30s. NFS is
a dumb shelf of one tar per key at <nfs-root>/<scope>/<arch>/<key>.tar.

Scopes mirror GitHub's cache service so a PR cannot poison main: push-to-main
writes the "main" scope every job reads; PR #N writes an isolated "pr-<N>" it
alone reads (falling back to main). N is GitHub-assigned, so a fork cannot forge
it to reach main or a sibling. A job writes only when the key is absent from
every scope it reads, so an ordinary PR writes nothing; the atomic rename keeps
concurrent writers of one key safe. A flat mount has no server-side ACL, so this
is the honest-path topology, not an enforced boundary -- the gate on an
untrusted fork is whatever approves fork PRs onto these runners. Restore and
save are best-effort: on failure uv starts cold, never a red job.
"""

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# Bump to invalidate every cached tar at once after a format change.
KEY_SALT = "nfs-uv-cache-v1"
KEY_LEN = 16

# Default is plain `.tar`: the cache is mostly already-compressed wheels, and it
# needs no compressor binary on the runner.
TAR_SUFFIXES = (".tar.zst", ".tar.gz", ".tar")

COMPRESS = {
    "none": (None, ".tar"),
    "zstd": ("zstd -1 -T0", ".tar.zst"),
    "gzip": ("gzip", ".tar.gz"),
}

# Parts of uv's cache that are never saved (tar --exclude patterns, anchored at
# the cache root). uv keys what it built from a local directory by that path
# and treats it as fresh while pyproject.toml/setup.py/setup.cfg keep their
# mtimes -- which a `git checkout` of another commit leaves alone unless it
# changes them. So a saved build (or metadata) of /home/senuser/torch-spyre
# could pass for a newer commit's. Everything else in the cache is keyed by
# content (registry wheels by hash, git builds by commit) and safe to share.
#   sdists-v*/path      builds + metadata of local source trees (torch-spyre)
#   sdists-v*/editable  editable builds of the project (hf-adapters)
#   wheels-v*/path      wheels installed from a local file (the run's wheel)
#   builds-v*           temporary build environments
EXCLUDE_DEFAULT = (
    "./sdists-v*/path",
    "./sdists-v*/editable",
    "./wheels-v*/path",
    "./builds-v*",
)

# A temp tar left behind (a killed save) is removed by a later save after this.
STALE_TMP_SECONDS = 86400


def _log(msg):
    print(msg, flush=True)


def _warn(msg):
    print(f"::warning::{msg}", flush=True)


def compute_key(key_files, salt=KEY_SALT):
    """Content-address the cache on the lockfiles' basenames and bytes.

    Only the basename, never the absolute path, feeds the digest -- runners with
    different GITHUB_WORKSPACE must agree on the key for byte-identical lockfiles,
    or a PR never hits the shared main scope.
    """
    digest = hashlib.sha256(salt.encode())
    for path in key_files:
        digest.update(b"\0")
        digest.update(os.fsencode(os.path.basename(path)))
        digest.update(b"\0")
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:KEY_LEN]


MAIN_SCOPE = "main"


def read_scopes(event_name, ref, pr_number):
    """Scopes this job may read, highest priority first (a PR falls back to main)."""
    if event_name == "pull_request" and pr_number:
        return [f"pr-{pr_number}", MAIN_SCOPE]
    return [MAIN_SCOPE]


def write_scope(event_name, ref, pr_number):
    """The one scope this job may write, or None in a read-only context."""
    if event_name == "push" and ref == "refs/heads/main":
        return MAIN_SCOPE
    if event_name == "pull_request" and pr_number:
        return f"pr-{pr_number}"
    return None


def pr_number_from_event(event_path):
    """The PR number of a pull_request event payload ('' if there is none).

    GitHub assigns it, so a fork cannot forge it to reach main or a sibling.
    """
    if not event_path:
        return ""
    try:
        with open(event_path, encoding="utf-8") as handle:
            event = json.load(handle)
        number = (event.get("pull_request") or {}).get("number")
    except (OSError, ValueError, AttributeError):
        return ""
    return str(number) if isinstance(number, int) else ""


def _reset_dir(path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _is_empty(path):
    return not any(path.iterdir())


def _scope_dir(nfs_root, scope, arch):
    return Path(nfs_root) / scope / arch


def _find_exact(arch_dir, key):
    for suffix in TAR_SUFFIXES:
        candidate = arch_dir / f"{key}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _newest_tar(arch_dir):
    if not arch_dir.is_dir():
        return None
    tars = [
        p for p in arch_dir.iterdir() if p.is_file() and p.name.endswith(TAR_SUFFIXES)
    ]
    if not tars:
        return None
    return max(tars, key=lambda p: p.stat().st_mtime)


def _extract(tar_path, dest):
    # tar preserves uv's internal hardlinks and avoids a per-file metadata storm
    # over NFS; GNU tar auto-detects compression on extract.
    subprocess.run(
        ["tar", "-xf", str(tar_path), "-C", str(dest)],
        check=True,
    )


def _create_tar(src, tar_path, compress_prog, excludes=()):
    cmd = ["tar"]
    if compress_prog:
        cmd += ["--use-compress-program", compress_prog]
    if excludes:
        # --anchored: patterns match from the archive root ("./..."), not any
        # path component; GNU tar's exclude wildcards also match "/".
        cmd += ["--anchored"] + [f"--exclude={pattern}" for pattern in excludes]
    cmd += ["-cf", str(tar_path), "-C", str(src), "."]
    subprocess.run(cmd, check=True)


def _write_kv(path, pairs):
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        for key, value in pairs.items():
            handle.write(f"{key}={value}\n")


def _pr_number(args):
    return args.pr_number or pr_number_from_event(args.event_path)


def cmd_restore(args):
    pr_number = _pr_number(args)
    scopes = read_scopes(args.event_name, args.ref, pr_number)
    local = Path(args.local_dir)

    key = ""
    hit = False
    restored_from = ""
    errored = False
    try:
        key = compute_key(args.key_file)
        _reset_dir(local)
        # Prefer an exact-key tar in any readable scope (priority order); else
        # the newest tar there, so a lockfile change still starts warm.
        chosen = None
        for scope in scopes:
            exact = _find_exact(_scope_dir(args.nfs_root, scope, args.arch), key)
            if exact is not None:
                chosen, hit = (scope, exact), True
                break
        if chosen is None and not args.no_partial:
            for scope in scopes:
                cand = _newest_tar(_scope_dir(args.nfs_root, scope, args.arch))
                if cand is not None:
                    chosen = (scope, cand)
                    break
        if chosen is not None:
            scope, tar = chosen
            try:
                _extract(tar, local)
                restored_from = f"{scope}/{args.arch}/{tar.name}"
            except (subprocess.CalledProcessError, OSError) as exc:
                # A half-extracted cache is worse than none: wipe and go cold.
                _warn(f"uv cache restore failed ({type(exc).__name__}); starting cold")
                _reset_dir(local)
                restored_from = ""
                hit = False
                errored = True
    except OSError as exc:
        _warn(f"uv cache restore errored ({type(exc).__name__}); starting cold")
        errored = True

    # Always hand uv a usable local cache dir, hit or miss.
    local.mkdir(parents=True, exist_ok=True)
    if hit:
        status = "hit"
    elif restored_from:
        status = "partial"
    else:
        status = "error" if errored else "miss"
    _write_kv(args.github_env, {"UV_CACHE_DIR": str(local)})
    _write_kv(
        args.github_output,
        {
            "cache-hit": "true" if hit else "false",
            "cache-key": key,
            "restored-from": restored_from,
            "status": status,
        },
    )

    scope_list = "+".join(scopes)
    if hit:
        _log(f"✅ uv cache HIT {restored_from} -> {local}")
    elif restored_from:
        _log(
            f"♻️ uv cache partial: restored {restored_from} (key {key}); uv fills the diff"
        )
    else:
        _log(
            f"🧊 uv cache MISS key {key} in [{scope_list}] ({args.arch}); starting cold"
        )
    return 0


def _gc(arch_dir, keep, protect):
    # A skip-save on a fallback-scope hit reaches here with the (unwritten,
    # nonexistent) write-scope dir, so missing is a no-op, not a warning.
    if keep <= 0 or not arch_dir.is_dir():
        return
    tars = sorted(
        (
            p
            for p in arch_dir.iterdir()
            if p.is_file() and p.name.endswith(TAR_SUFFIXES)
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in tars[keep:]:
        if stale == protect:
            continue
        try:
            stale.unlink()
            _log(f"🧹 evicted old cache {stale.name}")
        except OSError:
            pass


def _sweep_stale_tmp(arch_dir, max_age=STALE_TMP_SECONDS):
    """Remove temp tars a killed save left behind (never a live writer's)."""
    if not arch_dir.is_dir():
        return
    cutoff = time.time() - max_age
    for tmp in arch_dir.glob(".*.tmp"):
        try:
            if tmp.is_file() and tmp.stat().st_mtime < cutoff:
                tmp.unlink()
                _log(f"🧹 removed abandoned temp tar {tmp.name}")
        except OSError:
            pass


def _sweep_pr_scopes(nfs_root, ttl_days):
    """Evict PR-scope tars older than ttl_days; main is bounded by _gc instead."""
    if ttl_days <= 0:
        return
    cutoff = time.time() - ttl_days * 86400
    root = Path(nfs_root)
    if not root.is_dir():
        return
    for scope_dir in root.glob("pr-*"):
        if not scope_dir.is_dir():
            continue
        for tar in scope_dir.rglob("*"):
            try:
                if (
                    tar.is_file()
                    and tar.name.endswith(TAR_SUFFIXES)
                    and tar.stat().st_mtime < cutoff
                ):
                    tar.unlink()
                    _log(f"🧹 evicted stale PR cache {tar.relative_to(root)}")
            except OSError:
                pass


def cmd_save(args):
    # Sweep stale PR scopes on every invocation (even read-only contexts), so a
    # merged PR's cache is reclaimed by whatever job runs next.
    with contextlib.suppress(OSError):
        _sweep_pr_scopes(args.nfs_root, args.pr_ttl_days)

    pr_number = _pr_number(args)
    scope = write_scope(args.event_name, args.ref, pr_number)
    if scope is None:
        _log(
            f"read-only context (event={args.event_name} ref={args.ref}); "
            "skipping uv cache save"
        )
        return 0

    prog, ext = COMPRESS[args.compress]
    local = Path(args.local_dir)
    excludes = EXCLUDE_DEFAULT + tuple(args.exclude or ())

    try:
        key = compute_key(args.key_file)
        arch_dir = _scope_dir(args.nfs_root, scope, args.arch)
        target = arch_dir / f"{key}{ext}"
        # Skip if the key is already readable here, so an ordinary PR (key in
        # main) writes nothing; only a new key populates its own scope.
        for readable in read_scopes(args.event_name, args.ref, pr_number):
            found = _find_exact(_scope_dir(args.nfs_root, readable, args.arch), key)
            if found is not None:
                _log(
                    f"uv cache key {key} already in {readable}/{args.arch}; skipping save"
                )
                _gc(arch_dir, args.keep, target)
                return 0
        if not local.is_dir() or _is_empty(local):
            _log(f"no local uv cache at {local}; nothing to save")
            return 0

        arch_dir.mkdir(parents=True, exist_ok=True)
        _sweep_stale_tmp(arch_dir)
        # Temp lives in the target dir so the rename is a same-filesystem,
        # atomic rename(2) -- readers never see a half-written tar.
        tmp = arch_dir / f".{key}.{os.getpid()}.{random.randrange(1 << 30):x}.tmp"
        try:
            _create_tar(local, tmp, prog, excludes)
            # Re-check under the gun: a concurrent writer may have won while we
            # tarred. Equal content, so skipping ours (not overwriting) is fine.
            if _find_exact(arch_dir, key) is not None:
                _log(
                    f"uv cache {scope}/{args.arch} key {key} appeared during save; "
                    "discarding ours"
                )
            else:
                os.replace(tmp, target)
                size = target.stat().st_size / (1024 * 1024)
                _log(
                    f"✅ saved uv cache {scope}/{args.arch}/{target.name} ({size:.0f}MB)"
                )
        finally:
            if tmp.exists():
                tmp.unlink()
        _gc(arch_dir, args.keep, target)
    except (subprocess.CalledProcessError, OSError) as exc:
        # Never fail CI on a cache write (NFS full, perms, race): warn and move on.
        _warn(f"uv cache save failed ({type(exc).__name__}: {exc}); continuing")
    return 0


def _exit_on_sigterm(signum, frame):
    # A step timeout sends SIGTERM: unwind through the `finally` above so the
    # temp tar is removed, instead of dying with it on the mount.
    raise SystemExit(128 + signum)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument(
            "--nfs-root",
            required=True,
            help="e.g. $STORAGE_1_DIR/.cache/uv-snapshots",
        )
        p.add_argument(
            "--local-dir", required=True, help="local UV_CACHE_DIR to populate/ship"
        )
        # Matches `uname -m` (x86_64, ppc64le, s390x); wheels are arch-specific.
        p.add_argument("--arch", default=platform.machine())
        p.add_argument("--key-file", action="append", required=True, dest="key_file")
        p.add_argument("--event-name", default=os.getenv("GITHUB_EVENT_NAME", ""))
        p.add_argument("--ref", default=os.getenv("GITHUB_REF", ""))
        p.add_argument("--pr-number", default="", help="default: from --event-path")
        p.add_argument(
            "--event-path",
            default=os.getenv("GITHUB_EVENT_PATH", ""),
            help="event payload to read the PR number from",
        )

    r = sub.add_parser("restore")
    add_common(r)
    r.add_argument("--github-env", default=os.getenv("GITHUB_ENV", ""))
    r.add_argument("--github-output", default=os.getenv("GITHUB_OUTPUT", ""))
    r.add_argument(
        "--no-partial",
        action="store_true",
        help="exact key only: no newest-tar fallback (for the job that saves)",
    )
    r.set_defaults(func=cmd_restore)

    s = sub.add_parser("save")
    add_common(s)
    s.add_argument("--compress", choices=sorted(COMPRESS), default="none")
    s.add_argument("--keep", type=int, default=5, help="tars to retain per scope/arch")
    s.add_argument(
        "--pr-ttl-days",
        type=int,
        default=14,
        help="evict PR-scope tars older than this",
    )
    s.add_argument(
        "--exclude",
        action="append",
        help="extra tar --exclude pattern (anchored, e.g. ./git-v*), on top of "
        "EXCLUDE_DEFAULT",
    )
    s.set_defaults(func=cmd_save)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    sys.exit(main())
