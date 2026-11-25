"""Offline coverage of vision discovery, routing, and weekly backend verdicts.

Run with pytest --noconftest; no Spyre runtime or database is required.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml
from huggingface_hub.hf_api import ModelInfo

from tests.model_registry import (
    WEEKLY_SCAN_VISION_PATHS,
    WEEKLY_SCAN_VISION_TOWER_PATHS,
)
from tests.spyre.weekly_generation import model_fetcher, weekly_sub_process
from tests.spyre.weekly_generation.model_type import ModelType, scan_model_types
from tests.spyre.weekly_generation.sink.capability_write import capability_results
from utils import fetch_top_vision_models as vision
from utils.hf_model_catalog import Ranking


@pytest.mark.parametrize("ranking", ["downloads", "trending_score"])
def test_vision_query_uses_vlm_tag_and_requested_ranking(ranking: Ranking) -> None:
    calls: list[dict[str, Any]] = []
    model = ModelInfo(id="org/new-vlm")

    def list_models(**kwargs: Any) -> list[ModelInfo]:
        calls.append(kwargs)
        return [model]

    api: Any = types.SimpleNamespace(list_models=list_models)
    assert vision._create_fetch(ranking)(api, 10) == [model]
    assert calls[0]["pipeline_tag"] == "image-text-to-text"
    assert calls[0]["sort"] == ranking
    assert calls[0]["limit"] > 10


def test_unknown_vlm_architecture_is_not_filtered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vision, "keep_generative", lambda model, token: (True, ""))
    model = ModelInfo(
        id="org/new-vlm", library_name="transformers", config={"model_type": "new_vlm"}
    )
    assert vision.keep(model, False) == (True, "")
    model.library_name = "mlx"
    assert not vision.keep(model, False)[0]


def test_vision_selection_includes_tower_role_and_nonrepresentative_checkpoints() -> (
    None
):
    assert scan_model_types("vision") == [ModelType.VISION, ModelType.VISION_TOWER]
    assert scan_model_types("embedding") == [ModelType.EMBEDDING]
    assert "mistralai/Ministral-3-14B-Instruct-2512" in WEEKLY_SCAN_VISION_PATHS
    assert "google/gemma-4-26B-A4B-it" in WEEKLY_SCAN_VISION_PATHS
    assert "ibm-granite/granite-vision-4.1-4b" in WEEKLY_SCAN_VISION_TOWER_PATHS
    assert set(WEEKLY_SCAN_VISION_TOWER_PATHS) <= set(WEEKLY_SCAN_VISION_PATHS)


def test_towers_do_not_query_vlm_rankings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_fetcher, "all_fetchers", {})
    assert model_fetcher.fetch(ModelType.VISION_TOWER, 10) == []


def test_early_failures_keep_tower_and_vlm_capability_identities_distinct() -> None:
    rows = capability_results(
        [
            {"model_name": "org/shared", "model_type": mode.value, "adapter_name": ""}
            for mode in (ModelType.VISION, ModelType.VISION_TOWER)
        ]
    )
    assert {row["name"] for row in rows} == {"unknown_vision", "unknown_vision_tower"}


def test_vision_shards_merge_rankings_and_preserve_both_roles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = Path(__file__).parents[1] / ".github/scripts/generate_weekly_shards.py"
    spec = importlib.util.spec_from_file_location("weekly_vision_shards", path)
    assert spec is not None and spec.loader is not None
    sharder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sharder)
    calls: list[tuple[int, Ranking]] = []

    def metadata(model_ids: list[str]) -> list[dict[str, Any]]:
        return [{"model_id": model_id, "parameters": 100} for model_id in model_ids]

    def ranked(limit: int, ranking: Ranking) -> list[dict[str, Any]]:
        calls.append((limit, ranking))
        return metadata(["org/shared", f"org/{ranking}"])

    monkeypatch.setitem(model_fetcher.all_fetchers, ModelType.VISION, ranked)
    for mode in (ModelType.VISION, ModelType.VISION_TOWER):
        monkeypatch.setitem(model_fetcher.all_curated_loaders, mode, lambda: [])
        monkeypatch.setitem(model_fetcher.all_registry_paths, mode, ["org/shared"])
        monkeypatch.setitem(model_fetcher.all_metadata_fetchers, mode, metadata)
    matrix: list[dict[str, Any]] = sharder.generate_shards(
        top_k_downloads=10,
        top_k_trending=10,
        max_params=1000,
        shard_size_generative=250,
        shard_size_embedding=500,
        x1_max_params=500,
        x2_max_params=800,
        x2_shard_size=100,
        x4_shard_size=50,
        output_dir=tmp_path / "shards",
        model_types=scan_model_types("vision"),
        snapshot_date=date(2026, 11, 25),
        write_to_csv=tmp_path / "verdicts.csv",
    )
    assert calls == [(10, "downloads"), (10, "trending_score")]
    rows = {
        item["mode"]: json.loads((tmp_path / "shards" / item["shard_file"]).read_text())
        for item in matrix
    }
    assert [r["model_id"] for r in rows[ModelType.VISION]] == [
        "org/shared",
        "org/downloads",
        "org/trending_score",
    ]
    assert rows[ModelType.VISION][0]["curated"] is True
    assert [r["model_id"] for r in rows[ModelType.VISION_TOWER]] == ["org/shared"]


@pytest.mark.parametrize(
    ("reference_ok", "error", "category"),
    [
        (False, ValueError("processor failed"), "cpu_generate_failed"),
        (True, AssertionError("logits differ"), "verification_failed"),
        (True, RuntimeError("compile failed"), "test_execution_exception"),
        (True, RuntimeError("Replace card"), "hardware_exception"),
        (True, None, None),
    ],
)
@pytest.mark.parametrize("mode", [ModelType.VISION, ModelType.VISION_TOWER])
def test_backend_verdicts_preserve_reference_success(
    monkeypatch: pytest.MonkeyPatch,
    reference_ok: bool,
    error: Exception | None,
    category: str | None,
    mode: ModelType,
) -> None:
    def evaluate(
        model_path: str, *, on_reference: Callable[[], None], **kwargs: Any
    ) -> None:
        if reference_ok:
            on_reference()
        if error is not None:
            raise error

    name = (
        "tests.spyre.test_vlm_e2e_spyre"
        if mode == ModelType.VISION
        else "tests.spyre.weekly_generation.vision_tower"
    )
    module = types.ModuleType(name)
    setattr(
        module,
        "run_vlm_compare" if mode == ModelType.VISION else "run_tower_compare",
        evaluate,
    )
    monkeypatch.setitem(sys.modules, name, module)
    metrics = weekly_sub_process.eval_model("org/model", module, mode)
    assert metrics["load"] is reference_ok
    assert metrics["correct"] is (error is None)
    assert metrics["failure_category"] == category


def test_vlm_adapter_resolution_uses_multimodal_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping: dict[str, str] = {"vision_config": "vision_adapter"}
    auto = types.ModuleType("hf_adapters.auto_spyre_model")
    setattr(auto, "IMAGE_TEXT_TO_TEXT_CONFIG_TO_ADAPTER_MODULE_MAPPING", mapping)
    conftest = types.ModuleType("tests.conftest")
    calls: list[dict[str, Any]] = []

    def resolve(model_path: str, **kwargs: Any) -> types.ModuleType:
        calls.append(kwargs)
        return auto

    setattr(conftest, "resolve_adapter_module_for_test", resolve)
    monkeypatch.setitem(sys.modules, "hf_adapters.auto_spyre_model", auto)
    monkeypatch.setitem(sys.modules, "tests.conftest", conftest)
    assert (
        weekly_sub_process.resolve_weekly_adapter("org/shared", ModelType.VISION)
        is auto
    )
    assert calls == [{"mapping": mapping, "trust_remote_code": False}]


def test_ranked_fetch_expands_until_postfilter_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_sizes: list[int] = []
    checked: list[str] = []

    def fetch(api: Any, limit: int) -> list[ModelInfo]:
        query_sizes.append(limit)
        return [ModelInfo(id=f"org/{index}") for index in range(int(limit * 3.5))]

    def keep(model: ModelInfo, token: str | bool) -> tuple[bool, str]:
        checked.append(model.id)
        return model.id in {"org/0", "org/8"}, "filtered"

    def catalog(
        *,
        fetch_fn: Callable[[int], list[ModelInfo]],
        filter_fn: Callable[[ModelInfo], tuple[bool, str]],
        limit: int,
        **kwargs: Any,
    ) -> list[dict[str, object]]:
        return [
            {"model_id": model.id} for model in fetch_fn(limit) if filter_fn(model)[0]
        ][:limit]

    monkeypatch.setattr(vision, "build_catalog", catalog)
    result = vision._fetch_vision_models(fetch, keep, 2, fill_ranked=True)
    assert result == [{"model_id": "org/0"}, {"model_id": "org/8"}]
    assert query_sizes == [2, 4]
    assert len(checked) == len(set(checked))


def test_workflow_exposes_vision_and_scheduled_ranking_budget() -> None:
    path = Path(__file__).parents[1] / ".github/workflows/push-to-clickhouse.yaml"
    workflow: dict[str, Any] = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    options = workflow["on"]["workflow_dispatch"]["inputs"]["model_type"]["options"]
    assert {"vision", "vision_tower"} <= set(options)
    command = next(
        step["run"]
        for step in workflow["jobs"]["generate-matrix"]["steps"]
        if step.get("name") == "Generate shards"
    )
    assert (
        "--top-k-vision-downloads" in command and "--top-k-vision-trending" in command
    )
