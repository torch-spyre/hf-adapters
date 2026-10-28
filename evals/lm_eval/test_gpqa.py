"""Offline regressions for deterministic GPQA preprocessing and provenance."""

import copy
import random

import pytest
from datasets import Dataset, DatasetDict, load_from_disk

from evals.lm_eval import gpqa
from evals.lm_eval import run_benchmarks as runner


@pytest.fixture
def documents():
    return [
        {
            "Question": f"Question {i}",
            "Incorrect Answer 1": " wrong  one ",
            "Incorrect Answer 2": "wrong [title] two",
            "Incorrect Answer 3": None,
            "Correct Answer": f" right {i} ",
        }
        for i in range(24)
    ]


def test_cold_warm_cache_and_seed_identity(tmp_path, documents, monkeypatch):
    path = tmp_path / "dataset"
    Dataset.from_list(documents).save_to_disk(path)
    dataset = load_from_disk(path)
    # Populate a cache using the buggy upstream processor first.
    from lm_eval.tasks.gpqa.zeroshot.utils import process_docs as upstream

    upstream(dataset)
    cold = gpqa.process_docs(dataset, seed=1234)
    assert cold.cache_files
    other_seed = gpqa.process_docs(dataset, seed=5678)
    other_version = gpqa.process_docs(dataset, seed=1234, version="future_v2")
    assert len({d._fingerprint for d in (cold, other_seed, other_version)}) == 3
    assert cold.to_list() != other_seed.to_list()

    # Keep function hashing intact, but fail if a warm map executes any rows.
    def fail(*args, **kwargs):
        pytest.fail("warm cache should avoid executing the shuffle")

    with monkeypatch.context() as patch:
        patch.setattr(random.Random, "shuffle", fail)
        warm = gpqa.process_docs(load_from_disk(path), seed=1234)
    assert warm._fingerprint == cold._fingerprint
    assert warm.to_list() == cold.to_list()
    # A separate cold dataset yields identical document contents as well.
    assert (
        gpqa.process_docs(Dataset.from_list(documents), seed=1234).to_list()
        == cold.to_list()
    )


def test_independent_of_global_rng_and_document_order(documents):
    state = random.getstate()
    try:
        random.seed(1)
        before = random.getstate()
        first = gpqa.process_docs(Dataset.from_list(documents), seed=1234)
        assert random.getstate() == before
        random.seed(999)
        for _ in range(100):
            random.random()
        second = gpqa.process_docs(Dataset.from_list(documents[::-1]), seed=1234)
        assert first.to_list() == second.to_list()[::-1]
        for doc in first:
            choices = [doc[f"choice{i}"] for i in range(1, 5)]
            assert sorted(choices) == sorted(
                gpqa.preprocess(doc[k]) for k in gpqa.ANSWER_FIELDS
            )
            assert choices[ord(doc["answer"][1]) - ord("A")] == gpqa.preprocess(
                doc["Correct Answer"]
            )
    finally:
        random.setstate(state)


@pytest.mark.parametrize("grouped", [False, True])
def test_installed_before_real_task_construction(documents, monkeypatch, grouped):
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._factory import TaskFactory
    from lm_eval.tasks._index import TaskIndex

    def download(self, *args, **kwargs):
        self.dataset = DatasetDict(train=Dataset.from_list(documents))

    monkeypatch.setattr(ConfigurableTask, "download", download)
    manager = TaskManager()
    untouched = TaskManager()
    gpqa.install_gpqa_preprocessing(manager, 1234)
    names = sorted(gpqa.TASKS)
    # The runner accepts registered group IDs, not inline config overrides.
    manager._index["gpqa_test"] = TaskIndex.entry_from_config(
        {"group": "gpqa_test", "task": names}
    )
    requested = ["gpqa_test"] if grouped else names

    # The upstream process_docs uses global random.shuffle, ours must not.
    def fail(*args, **kwargs):
        pytest.fail("upstream random preprocessing ran")

    monkeypatch.setattr(random, "shuffle", fail)
    loaded = manager.load(requested)
    assert set(loaded["tasks"]) == gpqa.TASKS
    assert type(untouched._factory) is TaskFactory
    for name, task in loaded["tasks"].items():
        assert (
            task.task_docs.to_list()
            == gpqa.process_docs(Dataset.from_list(documents), seed=1234).to_list()
        )
        cfg = task.dump_config()
        assert cfg["metadata"][gpqa.METADATA_KEY] == {
            "version": gpqa.CHOICE_ORDER_VERSION,
            "seed": 1234,
        }
        assert cfg["metadata"]["version"] == 2.2
        assert cfg["doc_to_choice"] == ["(A)", "(B)", "(C)", "(D)"]
        # Unmodified manager still resolves the original upstream function.
        original = untouched._factory._load_full_config(untouched._index[name])
        assert gpqa.METADATA_KEY not in original["metadata"]
        assert original["process_docs"] is not task.config.process_docs


def test_preprocessing_provenance_rejects_legacy_reference():
    task = "gpqa_diamond_zeroshot"
    raw = {
        "results": {task: {"acc,none": 0.5}},
        "samples": {
            task: [
                {
                    "doc_id": 0,
                    "doc_hash": "d",
                    "prompt_hash": "p",
                    "target_hash": "t",
                    "filter": "none",
                    "metrics": ["acc"],
                    "acc": 0.5,
                }
            ]
        },
        "versions": {task: 2.2},
        "config": {"model_sha": "abc", "model_dtype": "bfloat16"},
        "configs": {
            task: {
                "metadata": {
                    gpqa.METADATA_KEY: {
                        "version": gpqa.CHOICE_ORDER_VERSION,
                        "seed": 1234,
                    }
                }
            }
        },
    }
    packages = {p: "test" for p in runner.PROTOCOL_PACKAGES}
    current = runner.summarize_result(raw, {"seed": 1234, "num_fewshot": 0}, packages)
    reference = copy.deepcopy(current)
    assert runner.compare(current, reference)["comparison"] == "compatible"
    del reference["protocol"]["preprocessing"]
    comparison = runner.compare(current, reference)
    assert comparison["comparison"] == "incompatible_reference"
    assert comparison["differences"] == ["preprocessing"]
    # No preprocessing metadata: no new protocol key for unrelated tasks.
    raw["configs"] = {"other_task": {"metadata": {"version": 1}}}
    assert (
        runner.summarize_result(raw, {"seed": 1234, "num_fewshot": 0}, packages)
        == reference
    )
