# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU accuracy test for the sentence-transformers ``backend="spyre"`` hook.

For each registered checkpoint, ``test_st_backend[<key>]`` loads the model
twice:

  1. Stock ``SentenceTransformer`` on CPU (reference).
  2. ``SentenceTransformer(..., backend="spyre")`` via ``hf_adapters.st_backend``,
     with compiled blocks unwrapped.

Asserts per-sentence cosine similarity stays above ``COS_SIM_THRESHOLD``.

DEVICE='cpu' patching of ``hf_common`` happens once in ``tests/conftest.py``;
this file is plain pytest. The ``hf_adapters.st_backend`` import is deferred
to inside the test (its side-effect registers the "spyre" backend with
sentence-transformers) so module collection succeeds on hosts that lack
``sentence_transformers``.
"""

import gc
from collections.abc import Iterator
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from transformers import PretrainedConfig

from hf_adapters import auto_spyre_model
from tests.cpu.conftest import _unwrap_compiled_blocks, cosine_per_row
from tests.model_registry import EMBED_PATHS

pytestmark = pytest.mark.model_harness("embedding")

# sentence_transformers is an optional dependency; skip the whole module if
# it's missing. The hf_adapters.st_backend import is deferred to inside the
# test so collection doesn't fail on CPU-only hosts that lack ST.
pytest.importorskip("sentence_transformers")

COS_SIM_THRESHOLD: float = 0.999

TEST_SENTENCES: list[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Spyre accelerates transformer inference at low latency.",
    "Embeddings are dense vector representations of text.",
    "Paris is the capital of France.",
]


@pytest.mark.parametrize(
    "target_device", ["cpu", torch.device("cpu"), "spyre", "spyre:1"]
)
@pytest.mark.parametrize("checkpoint", list(auto_spyre_model.MODEL_DTYPE_POLICIES))
def test_model_dtype_policy(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: str,
    target_device: str | torch.device,
) -> None:
    """Honor every declared policy, consulting config only for unspecified dtypes."""
    policy: auto_spyre_model.ModelDTypePolicy = auto_spyre_model.MODEL_DTYPE_POLICIES[
        checkpoint
    ]
    device_type: str = (
        target_device.type
        if isinstance(target_device, torch.device)
        else target_device.split(":", 1)[0]
    )
    declared_dtype: torch.dtype | None = (
        policy.cpu_dtype or policy.dtype if device_type == "cpu" else policy.dtype
    )
    config_lookups: list[str] = []

    def config_lookup(
        model_name_or_path: str, trust_remote_code: bool | None = None
    ) -> PretrainedConfig:
        assert declared_dtype is None, "An explicit policy must not need config lookup"
        config_lookups.append(model_name_or_path)
        return PretrainedConfig(dtype=torch.bfloat16)

    monkeypatch.setattr(
        auto_spyre_model,
        "_autoconfig_with_subfolder_fallback",
        config_lookup,
    )
    expected_dtype: torch.dtype = declared_dtype or torch.bfloat16
    if device_type == "spyre" and expected_dtype == torch.float32:
        expected_dtype = torch.float16
    assert (
        auto_spyre_model.dtype_for_model_path(checkpoint, target_device=target_device)
        == expected_dtype
    )
    assert config_lookups == ([checkpoint] if declared_dtype is None else [])


@pytest.mark.parametrize("device_index", [0, 1])
@pytest.mark.parametrize("backend", ["spyre", "torch"])
def test_st_device_with_cpu_first_parameter(
    monkeypatch: pytest.MonkeyPatch, device_index: int, backend: str
) -> None:
    """CPU-pinned embeddings must not hide the Spyre transformer blocks."""
    from sentence_transformers import SentenceTransformer

    import hf_adapters.st_backend  # noqa: F401

    model: SentenceTransformer = SentenceTransformer(
        modules=[torch.nn.Linear(2, 2)], backend=backend, device="cpu"
    )
    # Only simulate device metadata: no Spyre tensors or global backend rename
    # are needed to exercise the real SentenceTransformer.device property.
    spyre_device = cast(torch.device, SimpleNamespace(type="spyre", index=device_index))
    cpu_parameter = torch.nn.Parameter(torch.ones(1))
    reported_parameters: list[torch.nn.Parameter] = [
        cpu_parameter,
        cast(torch.nn.Parameter, SimpleNamespace(device=spyre_device)),
    ]

    def parameters(recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        return iter(reported_parameters)

    monkeypatch.setattr(model, "parameters", parameters)
    if backend == "spyre":
        assert model.device is spyre_device
        assert model.device is spyre_device  # repeated encode() device lookup
    else:
        assert model.device == torch.device("cpu")

    # Explicitly moving all parameters to CPU must remain observable: do not
    # hard-code/cache the construction-time Spyre device in the getter.
    reported_parameters[:] = [cpu_parameter]
    assert model.device == torch.device("cpu")


@pytest.mark.parametrize("backend", ["spyre", "torch"])
@pytest.mark.parametrize("with_parameters", [False, True])
def test_st_device_cpu_fallback(backend: str, with_parameters: bool) -> None:
    """Keep stock behavior for CPU tests and parameterless modules."""
    from sentence_transformers import SentenceTransformer

    import hf_adapters.st_backend  # noqa: F401

    module: torch.nn.Module = (
        torch.nn.Linear(2, 2) if with_parameters else torch.nn.Identity()
    )
    model: SentenceTransformer = SentenceTransformer(
        modules=[module], backend=backend, device="cpu"
    )
    assert model.device == torch.device("cpu")


@pytest.mark.parametrize("model_path", EMBED_PATHS, ids=EMBED_PATHS)
def test_st_backend(model_path: str) -> None:
    from sentence_transformers import SentenceTransformer

    import hf_adapters.st_backend  # noqa: F401  (registers "spyre" backend with ST)

    # Reference: stock ST on CPU
    dtype: torch.dtype = auto_spyre_model.dtype_for_model_path(model_path, "cpu")
    ref_model = SentenceTransformer(
        model_path, device="cpu", model_kwargs={"dtype": dtype}
    )
    ref_embeddings = ref_model.encode(TEST_SENTENCES, convert_to_tensor=True)
    del ref_model
    gc.collect()

    # Spyre backend (DEVICE patched to cpu by conftest)
    spyre_model = SentenceTransformer(
        model_path, backend="spyre", device="cpu", model_kwargs={"dtype": dtype}
    )
    _unwrap_compiled_blocks(spyre_model)
    spyre_embeddings = spyre_model.encode(TEST_SENTENCES, convert_to_tensor=True)
    del spyre_model
    gc.collect()

    cos_sims = cosine_per_row(ref_embeddings, spyre_embeddings)
    min_sim = cos_sims.min().item()
    assert min_sim >= COS_SIM_THRESHOLD, (
        f"min cosine {min_sim:.6f} < threshold {COS_SIM_THRESHOLD}; "
        f"per-sentence: {cos_sims.tolist()}"
    )


@pytest.mark.parametrize("has_tokens", [False, True])
def test_embedding_comparison_output_capabilities(has_tokens: bool) -> None:
    from tests.spyre.test_e2e_embed_compare_spyre import _compare_results, _print_table

    features: dict[str, torch.Tensor] = {"sentence_embedding": torch.tensor([1.0, 0.0])}
    if has_tokens:
        features.update(
            token_embeddings=torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
            attention_mask=torch.ones(2),
        )
    rows = _compare_results(
        [features], [features], "test/embedder", supports_token_embeddings=has_tokens
    )
    assert rows[0]["match"]
    assert (rows[0]["min_cos"] is not None) == has_tokens
    _print_table(rows)


def test_embedding_comparison_rejects_missing_token_outputs() -> None:
    from tests.spyre.test_e2e_embed_compare_spyre import _compare_results

    features: dict[str, torch.Tensor] = {"sentence_embedding": torch.tensor([1.0, 0.0])}
    with pytest.raises(AssertionError, match="expected token embeddings"):
        _compare_results([features], [features], "test/token-embedder")


def test_projected_embedding_comparison_checks_accuracy() -> None:
    from tests.spyre.test_e2e_embed_compare_spyre import _compare_results

    rows = _compare_results(
        [{"sentence_embedding": torch.tensor([1.0, 0.0])}],
        [{"sentence_embedding": torch.tensor([0.0, 1.0])}],
        "test/projected-embedder",
        supports_token_embeddings=False,
    )
    assert not rows[0]["match"]
