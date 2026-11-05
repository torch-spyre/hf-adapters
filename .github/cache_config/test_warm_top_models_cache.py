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
"""Offline tests for the cache warmer's review-driven guards (PR #748)."""

import argparse
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import warm_top_models_cache as warm
from hf_snapshot_worker import describe, http_status, is_transient
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
)
from huggingface_hub.hf_api import ModelInfo

_REQUEST: httpx.Request = httpx.Request("GET", "https://huggingface.co/api/models/a/b")


def _http_error(status: int) -> HfHubHTTPError:
    return HfHubHTTPError(
        f"{status}", response=httpx.Response(status, request=_REQUEST)
    )


def _wrapped(cause: BaseException) -> LocalEntryNotFoundError:
    """What snapshot_download raises when the metadata request fails."""
    try:
        raise LocalEntryNotFoundError("cannot find snapshot") from cause
    except LocalEntryNotFoundError as e:
        return e


class CachedSnapshotTests(unittest.TestCase):
    """Comment 1: a snapshot is cached only when nothing it needs is missing."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.snapshot: Path = Path(temporary.name)

    def _write(self, name: str, content: str = "x") -> None:
        path: Path = self.snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _index(self, *shards: str) -> None:
        weight_map: dict[str, str] = {f"w{i}": s for i, s in enumerate(shards)}
        self._write(
            "model.safetensors.index.json", json.dumps({"weight_map": weight_map})
        )

    def test_one_of_two_indexed_shards_is_incomplete(self) -> None:
        shards: tuple[str, str] = (
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        )
        self._index(*shards)
        self._write(shards[0])
        missing: list[str] = warm.missing_files(self.snapshot, None, [])
        self.assertEqual(missing, [shards[1]])

    def test_all_indexed_shards_present_is_complete(self) -> None:
        self._index("a.safetensors", "b.safetensors")
        self._write("a.safetensors")
        self._write("b.safetensors")
        self.assertEqual(warm.missing_files(self.snapshot, None, []), [])

    def test_ignored_bin_shards_are_not_required(self) -> None:
        self._write(
            "pytorch_model.bin.index.json",
            json.dumps({"weight_map": {"w": "pytorch_model-00001.bin"}}),
        )
        self._write("model.safetensors")
        self.assertEqual(warm.missing_files(self.snapshot, None, ["*.bin"]), [])

    def test_catalog_files_must_all_be_present(self) -> None:
        self._write("model.safetensors")
        info: ModelInfo = ModelInfo(
            id="a/b",
            siblings=[
                {"rfilename": "model.safetensors"},
                {"rfilename": "tokenizer.json"},
            ],
        )
        self.assertEqual(
            warm.missing_files(self.snapshot, info, []), ["tokenizer.json"]
        )

    def test_dangling_symlink_counts_as_missing(self) -> None:
        (self.snapshot / "model.safetensors").symlink_to(self.snapshot / "gone")
        self.assertEqual(
            warm.missing_files(self.snapshot, None, []), ["<no weight files>"]
        )

    def test_config_only_snapshot_is_not_cached(self) -> None:
        self._write("config.json", "{}")
        self.assertEqual(
            warm.missing_files(self.snapshot, None, []), ["<no weight files>"]
        )

    def test_corrupt_index_is_not_cached(self) -> None:
        self._write("model.safetensors")
        self._write("model.safetensors.index.json", "{not json")
        self.assertTrue(warm.missing_files(self.snapshot, None, []))


class DownloadSizeTests(unittest.TestCase):
    """Comment 2: the disk guard uses real sizes, and fp32 is priced at 4 bytes."""

    def test_reviewer_example_fp32_20b_is_priced_at_80_gb(self) -> None:
        info: ModelInfo = ModelInfo(
            id="a/b",
            safetensors={"parameters": {"F32": 20_000_000_000}, "total": 1},
        )
        size, method = warm.download_size(None, "a/b", [], info)
        self.assertEqual((size, method), (80_000_000_000, "safetensors"))
        # 550 GiB free, 500 GiB floor: the old 2-bytes estimate let this through.
        free: int = 550 * 1024**3
        self.assertLess(free - size, 500 * 1024**3)

    def test_exact_sizes_win_when_the_hub_answers(self) -> None:
        api = MagicMock()
        api.model_info.return_value = MagicMock(
            siblings=[
                MagicMock(
                    rfilename="model.safetensors", size=None, lfs=MagicMock(size=7)
                ),
                MagicMock(rfilename="x.onnx", size=None, lfs=MagicMock(size=99)),
            ]
        )
        self.assertEqual(warm.download_size(api, "a/b", ["*.onnx"], None), (7, "exact"))

    def test_parameter_count_alone_is_not_trusted(self) -> None:
        api = MagicMock()
        api.model_info.side_effect = RuntimeError("down")
        self.assertEqual(warm.download_size(api, "a/b", [], None), (None, None))


class DeadlineTests(unittest.TestCase):
    """Comment 3: no download, retry wait or cool-down outlasts the budget."""

    def test_running_download_is_killed_at_the_deadline(self) -> None:
        timeouts: list[float] = []

        def slow(repo_id: str, ignore: list[str], timeout: float) -> dict[str, object]:
            timeouts.append(timeout)
            raise subprocess.TimeoutExpired("worker", timeout)

        with patch.object(warm, "_run_worker", slow):
            outcome = warm.download_with_retries("a/b", [], 4, time.monotonic() + 30)
        self.assertTrue(outcome.out_of_time)
        self.assertIsNone(outcome.path)
        self.assertLessEqual(timeouts[0], 30)

    def test_retry_wait_past_the_deadline_is_not_started(self) -> None:
        transient: dict[str, object] = {
            "ok": False,
            "error": "503",
            "status": 503,
            "transient": True,
            "disk_full": False,
        }
        with (
            patch.object(warm, "_run_worker", return_value=transient) as run,
            patch.object(warm.time, "sleep") as sleep,
        ):
            outcome = warm.download_with_retries("a/b", [], 4, time.monotonic() + 30)
        self.assertTrue(outcome.out_of_time)
        self.assertEqual(run.call_count, 1)
        sleep.assert_not_called()

    def test_expired_budget_starts_no_attempt(self) -> None:
        with patch.object(warm, "_run_worker") as run:
            outcome = warm.download_with_retries("a/b", [], 4, time.monotonic() - 1)
        self.assertTrue(outcome.out_of_time)
        run.assert_not_called()

    def test_real_worker_is_killed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            hanging: Path = Path(directory) / "hang.py"
            hanging.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
            started: float = time.monotonic()
            with (
                patch.object(warm, "_WORKER", hanging),
                self.assertRaises(subprocess.TimeoutExpired),
            ):
                warm._run_worker("a/b", [], 0.5)
            self.assertLess(time.monotonic() - started, 10)

    def test_breaker_returns_its_cooldown_instead_of_sleeping(self) -> None:
        breaker = warm._Breaker(failures_before_cooldown=1, cooldown_seconds=1800)
        with patch.object(warm.time, "sleep") as sleep:
            self.assertEqual(breaker.record_failure(), 1800)
        sleep.assert_not_called()


class RateLimitTests(unittest.TestCase):
    """Comment 4: a 429 wrapped in LocalEntryNotFoundError still waits 10 minutes."""

    def test_wrapped_429_is_found_in_the_cause_chain(self) -> None:
        error: LocalEntryNotFoundError = _wrapped(_http_error(429))
        self.assertEqual(http_status(error), 429)
        self.assertTrue(is_transient(error))
        self.assertEqual(describe(error)["status"], 429)

    def test_wrapped_429_waits_the_rate_limit_pause(self) -> None:
        limited: dict[str, object] = describe(_wrapped(_http_error(429)))
        ok: dict[str, object] = {"ok": True, "path": "/tmp/snap"}
        with (
            patch.object(warm, "_run_worker", side_effect=[limited, ok]),
            patch.object(warm.time, "sleep") as sleep,
        ):
            outcome = warm.download_with_retries(
                "a/b", [], 4, time.monotonic() + 7200, rate_limit_pause=600
            )
        self.assertEqual(outcome.path, Path("/tmp/snap"))
        sleep.assert_called_once_with(600)

    def test_wrapped_permanent_errors_are_not_retried(self) -> None:
        self.assertFalse(is_transient(_wrapped(_http_error(404))))
        gated = GatedRepoError("gated", response=httpx.Response(403, request=_REQUEST))
        self.assertFalse(is_transient(_wrapped(gated)))

    def test_unreachable_hub_is_transient(self) -> None:
        self.assertTrue(is_transient(_wrapped(httpx.ConnectError("no route"))))
        self.assertTrue(is_transient(LocalEntryNotFoundError("offline")))


_SETTINGS: dict[str, int | bool] = {
    "top_k_downloads": 5,
    "top_k_trending": 3,
    "max_params": 60_000_000_000,
    "adapters_only": True,
}


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "top_k_downloads": 5,
        "top_k_trending": 3,
        "max_params": 60_000_000_000,
        "all_models": False,
        "catalog_workers": 4,
        "model_list_file": None,
        **overrides,
    }
    return argparse.Namespace(**values)


class SavedModelListTests(unittest.TestCase):
    """Option 1: the estimator's list is saved once and reused, not refetched."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path: Path = Path(temporary.name) / "list.json"
        info: ModelInfo = ModelInfo(
            id="org/model",
            siblings=[{"rfilename": "model.safetensors"}, {"rfilename": "x.bin"}],
            safetensors={"parameters": {"F32": 10, "BF16": 4}, "total": 14},
        )
        self.rows: list[dict[str, object]] = [
            {
                "model_id": "org/model",
                "mode": "embedding",
                "curated": False,
                "downloads": 9,
                "parameters": 14,
                "model_info": info,
            },
            {"model_id": "org/bare", "mode": "generative", "parameters": None},
        ]

    def test_round_trip_keeps_what_sizing_and_cache_checks_read(self) -> None:
        warm.save_model_list(self.path, self.rows, _SETTINGS)
        loaded: list[dict[str, object]] = warm.load_model_list(self.path, _SETTINGS)
        self.assertEqual([r["model_id"] for r in loaded], ["org/model", "org/bare"])
        first: object = loaded[0]["model_info"]
        assert isinstance(first, ModelInfo)
        original: object = self.rows[0]["model_info"]
        assert isinstance(original, ModelInfo)
        self.assertEqual(
            warm.safetensors_bytes(first), warm.safetensors_bytes(original)
        )
        self.assertEqual(
            warm.ignore_patterns_for(first), warm.ignore_patterns_for(original)
        )
        self.assertEqual(warm.selected_files(first, []), ["model.safetensors", "x.bin"])
        self.assertEqual(loaded[0]["downloads"], 9)
        second: object = loaded[1]["model_info"]
        assert isinstance(second, ModelInfo)
        self.assertIsNone(warm.safetensors_bytes(second))

    def test_list_built_with_other_flags_is_refused(self) -> None:
        warm.save_model_list(self.path, self.rows, _SETTINGS)
        with self.assertRaisesRegex(ValueError, "pass the same flags"):
            warm.load_model_list(self.path, {**_SETTINGS, "top_k_downloads": 6})
        with self.assertRaisesRegex(ValueError, "pass the same flags"):
            warm.load_model_list(self.path, {**_SETTINGS, "adapters_only": False})

    def test_unknown_format_is_refused(self) -> None:
        self.path.write_text(json.dumps({"format_version": 99}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "format_version"):
            warm.load_model_list(self.path, _SETTINGS)

    def test_saved_list_is_used_without_fetching(self) -> None:
        warm.save_model_list(self.path, self.rows, _SETTINGS)
        with patch.object(warm, "build_model_list") as fetch:
            rows = warm.get_model_list(_args(model_list_file=self.path))
        fetch.assert_not_called()
        self.assertEqual(len(rows), 2)


class CatalogWorkersTests(unittest.TestCase):
    """Option 2: the list fetch runs --catalog-workers requests at a time."""

    def test_fetch_passes_the_worker_count_to_the_catalog(self) -> None:
        seen: list[str | None] = []

        def fake_build(*_: object, **__: object) -> list[dict[str, object]]:
            seen.append(os.environ.get("HF_CATALOG_MAX_WORKERS"))
            return []

        with (
            patch.dict(os.environ, {}, clear=False),
            patch.object(warm, "build_model_list", fake_build),
        ):
            os.environ.pop("HF_CATALOG_MAX_WORKERS", None)
            warm.get_model_list(_args(catalog_workers=3))
        self.assertEqual(seen, ["3"])

    def test_zero_workers_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            warm.get_model_list(_args(catalog_workers=0))

    def test_catalog_reads_the_variable_and_keeps_the_scan_default(self) -> None:
        from utils.hf_model_catalog import (
            DEFAULT_CATALOG_MAX_WORKERS,
            catalog_max_workers,
        )

        self.assertEqual(DEFAULT_CATALOG_MAX_WORKERS, 8)
        with patch.dict(os.environ, {"HF_CATALOG_MAX_WORKERS": "4"}):
            self.assertEqual(catalog_max_workers(), 4)
        with patch.dict(os.environ, {"HF_CATALOG_MAX_WORKERS": ""}):
            self.assertEqual(catalog_max_workers(), 8)
        for bad in ("0", "many"):
            with patch.dict(os.environ, {"HF_CATALOG_MAX_WORKERS": bad}):
                with self.assertRaises(ValueError):
                    catalog_max_workers()


def _catalog_row(model_id: str, downloads: int, parameters: int = 1) -> dict:
    return {"model_id": model_id, "downloads": downloads, "parameters": parameters}


class WeeklySelectionTests(unittest.TestCase):
    """The list is the weekly scan's: by-name + downloads + trending, per type."""

    def setUp(self) -> None:
        self.calls: list[tuple[str, int, str]] = []
        ranked: dict[tuple[str, str], list[dict]] = {
            ("generative", "downloads"): [
                _catalog_row("g/dl", 50),
                _catalog_row("g/both", 40),
                _catalog_row("g/huge", 30, parameters=70_000_000_000),
            ],
            ("generative", "trending_score"): [
                _catalog_row("g/both", 40),
                _catalog_row("g/trend", 5),
                _catalog_row("g/curated", 1),
            ],
            ("embedding", "downloads"): [_catalog_row("e/dl", 90)],
            ("embedding", "trending_score"): [_catalog_row("e/trend", 2)],
        }

        def fetcher(mode: str):  # type: ignore[no-untyped-def]
            def fetch(limit: int, ranking: str) -> list[dict]:
                self.calls.append((mode, limit, ranking))
                return [dict(r) for r in ranked[(mode, ranking)]]

            return fetch

        def metadata(model_ids: list[str]) -> list[dict]:
            return [_catalog_row(m, 0) for m in model_ids]

        from tests.spyre.weekly_generation import model_fetcher
        from tests.spyre.weekly_generation.model_type import ModelType

        by_name: dict[ModelType, list[str]] = {
            ModelType.GENERATIVE: ["g/curated", "g/registry"],
            ModelType.EMBEDDING: ["e/registry"],
        }
        for patcher in (
            patch.dict(
                model_fetcher.all_fetchers,
                {t: fetcher(str(t)) for t in ModelType},
            ),
            patch.dict(
                model_fetcher.all_metadata_fetchers,
                {t: metadata for t in ModelType},
            ),
            patch.object(model_fetcher, "by_name_ids", lambda t: by_name[t]),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _ids(self, **kwargs: object) -> list[str]:
        rows = warm.build_model_list(
            5, 3, 60_000_000_000, adapters_only=False, **kwargs
        )
        return [str(r["model_id"]) for r in rows]

    def test_by_name_first_then_ranked_by_downloads_deduplicated(self) -> None:
        self.assertEqual(
            self._ids(),
            [
                "g/curated",
                "g/registry",
                "e/registry",
                "e/dl",
                "g/dl",
                "g/both",
                "g/trend",
                "e/trend",
            ],
        )

    def test_each_type_fetches_both_rankings_with_their_own_top_k(self) -> None:
        self._ids()
        self.assertCountEqual(
            self.calls,
            [
                ("generative", 5, "downloads"),
                ("generative", 3, "trending_score"),
                ("embedding", 5, "downloads"),
                ("embedding", 3, "trending_score"),
            ],
        )

    def test_by_name_rows_are_curated_and_keep_their_mode(self) -> None:
        rows = warm.build_model_list(5, 3, 60_000_000_000, adapters_only=False)
        flags = {r["model_id"]: (r["mode"], r["curated"]) for r in rows}
        self.assertEqual(flags["g/curated"], ("generative", True))
        self.assertEqual(flags["e/registry"], ("embedding", True))
        self.assertEqual(flags["g/trend"], ("generative", False))

    def test_zero_top_k_skips_that_ranking(self) -> None:
        warm.build_model_list(5, 0, 60_000_000_000, adapters_only=False)
        self.assertNotIn("trending_score", [ranking for *_, ranking in self.calls])

    def test_models_without_an_adapter_are_dropped_by_default(self) -> None:
        def verdict(model_id: str) -> str | None:
            return (
                None if model_id.endswith(("dl", "registry")) else "SpyreNoAdapterError"
            )

        with patch.object(warm, "_adapter_verdict", verdict):
            rows = warm.build_model_list(5, 3, 60_000_000_000, workers=2)
        self.assertEqual(
            [r["model_id"] for r in rows], ["g/registry", "e/registry", "e/dl", "g/dl"]
        )


if __name__ == "__main__":
    unittest.main()
