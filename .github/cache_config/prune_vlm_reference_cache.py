# Copyright 2026 The Torch-Spyre Authors.
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

"""Prune completed VLM CPU references using only filesystem metadata."""

import argparse
import re
import stat
import sys
import time
from pathlib import Path

_ENTRY_NAME = re.compile(r"vlm-reference-[0-9a-f]{64}\.pt")


def prune_cache(cache_dir: Path, *, cutoff: float, delete: bool = False) -> int:
    """Prune files last accessed and modified before cutoff; return an exit code."""
    try:
        entries = sorted(cache_dir.iterdir())
    except FileNotFoundError:
        print(f"VLM reference cache does not exist: {cache_dir}")
        return 0
    except OSError as exc:
        print(f"Could not list VLM reference cache {cache_dir}: {exc}", file=sys.stderr)
        return 1

    count = total_bytes = errors = 0
    action = "Deleted" if delete else "Would delete"
    for path in entries:
        if _ENTRY_NAME.fullmatch(path.name) is None:
            continue
        try:
            metadata = path.lstat()
            # Do not follow symlinks or descend into directories. Temporary
            # publications have a leading dot and do not match _ENTRY_NAME.
            if not stat.S_ISREG(metadata.st_mode):
                continue
            if max(metadata.st_atime, metadata.st_mtime) >= cutoff:
                continue
            if delete:
                path.unlink()
        except FileNotFoundError:
            # Another maintenance job may have removed the entry already.
            continue
        except OSError as exc:
            print(f"Could not prune VLM reference {path}: {exc}", file=sys.stderr)
            errors += 1
            continue
        count += 1
        total_bytes += metadata.st_size
        print(f"{action} {path} ({metadata.st_size} bytes)")

    print(f"{action} {count} VLM references ({total_bytes} bytes); {errors} errors.")
    return 1 if errors else 0


def _positive_days(value: str) -> int:
    days = int(value)
    if days <= 0:
        raise argparse.ArgumentTypeError("max-age-days must be a positive integer")
    return days


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--max-age-days", type=_positive_days, default=30)
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete expired references; the default only previews deletions.",
    )
    args = parser.parse_args(argv)
    return prune_cache(
        args.cache_dir.expanduser(),
        cutoff=time.time() - args.max_age_days * 24 * 60 * 60,
        delete=args.delete,
    )


if __name__ == "__main__":
    sys.exit(main())
