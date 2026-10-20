"""Runner-local workaround for lm-eval 0.4.13's cached, global-RNG GPQA shuffle."""

import hashlib
import json
import random
from functools import partial

CHOICE_ORDER_VERSION = "sha256_per_document_v1"
METADATA_KEY = "gpqa_choice_order"
TASKS = frozenset(
    f"gpqa_{subset}_zeroshot" for subset in ("main", "diamond", "extended")
)
ANSWER_FIELDS = (
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
    "Correct Answer",
)


def preprocess(text):
    """Preserve the upstream zero-shot answer cleaning exactly."""
    if text is None:
        return " "
    return text.strip().replace(" [title]", ". ").replace("  ", " ")


def process_doc(doc, *, seed, version):
    # Hash original fields, not cleaned answers or dataset position. This makes
    # choices independent of traversal order, model loading, and other tasks.
    identity = [version, seed, doc["Question"], *[doc[k] for k in ANSWER_FIELDS]]
    hashed = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False).encode("utf-8")
    ).digest()
    choices = [preprocess(doc[k]) for k in ANSWER_FIELDS]
    random.Random(int.from_bytes(hashed, "big")).shuffle(choices)
    correct = choices.index(preprocess(doc["Correct Answer"]))
    return {
        **{f"choice{i + 1}": choice for i, choice in enumerate(choices)},
        "answer": f"({chr(65 + correct)})",
    }


def process_docs(dataset, *, seed, version=CHOICE_ORDER_VERSION):
    # Explicit kwargs participate in datasets' cache fingerprint. Neither a
    # warm upstream cache nor a cache from another suite seed can be reused.
    return dataset.map(process_doc, fn_kwargs={"seed": seed, "version": version})


def install_gpqa_preprocessing(manager, seed):
    """Override configs before task construction, for this manager only.

    The private factory hook is specific to the pinned lm-eval 0.4.13. Using it
    also covers tasks expanded from groups/tags, without patching the library
    globally or triggering dataset preprocessing twice.
    """
    from lm_eval.tasks._factory import TaskFactory

    class GPQATaskFactory(TaskFactory):
        def _load_full_config(self, entry, overrides=None):
            cfg = super()._load_full_config(entry, overrides)
            task = cfg.get("task")
            if isinstance(task, str) and task in TASKS:
                if (
                    cfg.get("dataset_path") != "Idavidrein/gpqa"
                    or cfg.get("output_type") != "multiple_choice"
                ):
                    raise ValueError(f"Unexpected GPQA task definition: {task}")
                cfg["process_docs"] = partial(process_docs, seed=seed)
                cfg["metadata"][METADATA_KEY] = {
                    "version": CHOICE_ORDER_VERSION,
                    "seed": seed,
                }
            return cfg

    manager._factory = GPQATaskFactory(meta=manager.metadata)
