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

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prune_vlm_reference_cache as cleanup

_CUTOFF = 1_700_000_000


class PruneVLMReferenceCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache_dir = self.root / "references"
        self.cache_dir.mkdir()

    def entry(self, key, *, accessed=_CUTOFF - 1, modified=_CUTOFF - 1):
        path = self.cache_dir / f"vlm-reference-{key:064x}.pt"
        path.write_bytes(b"cached reference")
        os.utime(path, (accessed, modified))
        return path

    def prune(self, *, delete=False):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            result = cleanup.prune_cache(self.cache_dir, cutoff=_CUTOFF, delete=delete)
        return result, output.getvalue(), errors.getvalue()

    def test_preview_keeps_expired_files(self):
        expired = self.entry(1)
        result, output, errors = self.prune()
        self.assertEqual(result, 0)
        self.assertTrue(expired.exists())
        self.assertIn("Would delete 1 VLM references (16 bytes)", output)
        self.assertEqual(errors, "")

    def test_delete_keeps_recently_read_or_written_and_boundary_entries(self):
        expired = self.entry(1)
        recent_read = self.entry(2, accessed=_CUTOFF + 100)
        recent_write = self.entry(3, modified=_CUTOFF + 100)
        boundary = self.entry(4, accessed=_CUTOFF, modified=_CUTOFF)
        result, output, errors = self.prune(delete=True)
        self.assertEqual(result, 0)
        self.assertFalse(expired.exists())
        self.assertTrue(recent_read.exists())
        self.assertTrue(recent_write.exists())
        self.assertTrue(boundary.exists())
        self.assertIn("Deleted 1 VLM references (16 bytes)", output)
        self.assertEqual(errors, "")

    def test_unrelated_files_temporary_files_directories_and_symlinks_are_kept(self):
        keep = [
            self.cache_dir / "model.pt",
            self.cache_dir / "vlm-reference-old.pt",
            self.cache_dir / f".vlm-reference-{5:064x}.pt.in-progress",
        ]
        for path in keep:
            path.write_bytes(b"unrelated")
            os.utime(path, (_CUTOFF - 1, _CUTOFF - 1))
        directory = self.cache_dir / f"vlm-reference-{6:064x}.pt"
        directory.mkdir()
        nested = directory / f"vlm-reference-{7:064x}.pt"
        nested.write_bytes(b"nested")
        target = self.root / "outside.pt"
        target.write_bytes(b"outside")
        os.utime(target, (_CUTOFF - 1, _CUTOFF - 1))
        symlink = self.cache_dir / f"vlm-reference-{8:064x}.pt"
        symlink.symlink_to(target)

        result, _, _ = self.prune(delete=True)

        self.assertEqual(result, 0)
        for path in [*keep, nested, target]:
            self.assertTrue(path.exists())
        self.assertTrue(symlink.is_symlink())

    def test_missing_cache_is_a_noop(self):
        self.cache_dir.rmdir()
        result, _, errors = self.prune(delete=True)
        self.assertEqual(result, 0)
        self.assertFalse(self.cache_dir.exists())
        self.assertEqual(errors, "")

    def test_concurrent_removal_is_ignored(self):
        disappearing = self.entry(1)
        other = self.entry(2)
        original_unlink = Path.unlink

        def unlink(path, *args, **kwargs):
            original_unlink(path, *args, **kwargs)
            if path == disappearing:
                raise FileNotFoundError(path)

        with patch.object(Path, "unlink", unlink):
            result, _, errors = self.prune(delete=True)
        self.assertEqual(result, 0)
        self.assertFalse(other.exists())
        self.assertEqual(errors, "")

    def test_permission_failure_is_reported_and_other_entries_are_pruned(self):
        blocked = self.entry(1)
        other = self.entry(2)
        original_unlink = Path.unlink

        def unlink(path, *args, **kwargs):
            if path == blocked:
                raise PermissionError("read-only cache entry")
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", unlink):
            result, output, errors = self.prune(delete=True)
        self.assertEqual(result, 1)
        self.assertTrue(blocked.exists())
        self.assertFalse(other.exists())
        self.assertIn("Deleted 1 VLM references", output)
        self.assertIn("read-only cache entry", errors)

    def test_unreadable_cache_is_reported(self):
        with patch.object(Path, "iterdir", side_effect=PermissionError("unreadable")):
            result, _, errors = self.prune(delete=True)
        self.assertEqual(result, 1)
        self.assertIn("Could not list VLM reference cache", errors)

    def test_cli_defaults_to_preview_and_honors_retention_and_delete(self):
        expired = self.entry(1, accessed=0, modified=0)
        args = ["--cache-dir", str(self.cache_dir), "--max-age-days", "7"]
        with patch.object(cleanup.time, "time", return_value=8 * 24 * 60 * 60):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cleanup.main(args), 0)
                self.assertTrue(expired.exists())
                self.assertEqual(cleanup.main([*args, "--delete"]), 0)
        self.assertFalse(expired.exists())

    def test_cli_rejects_invalid_retention(self):
        expired = self.entry(1)
        for days in ["0", "-1", "nan", "invalid"]:
            with self.subTest(days=days), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    cleanup.main(
                        [
                            "--cache-dir",
                            str(self.cache_dir),
                            "--max-age-days",
                            days,
                            "--delete",
                        ]
                    )
                self.assertEqual(raised.exception.code, 2)
        self.assertTrue(expired.exists())


if __name__ == "__main__":
    unittest.main()
