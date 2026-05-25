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

import pytest
import torch.nn.functional as F
from conftest import EMBEDDING_MODELS

# sentence_transformers is an optional dependency; skip the whole module if
# it's missing. The hf_adapters.st_backend import is deferred to inside the
# test so collection doesn't fail on CPU-only hosts that lack ST.
pytest.importorskip("sentence_transformers")

COS_SIM_THRESHOLD = 0.999

TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Spyre accelerates transformer inference at low latency.",
    "Embeddings are dense vector representations of text.",
    "Paris is the capital of France.",
]


@pytest.mark.parametrize(
    "model_key", list(EMBEDDING_MODELS.keys()), ids=list(EMBEDDING_MODELS.keys())
)
def test_st_backend(model_key, unwrap_compiled_blocks):
    from sentence_transformers import SentenceTransformer

    import hf_adapters.st_backend  # noqa: F401  (registers "spyre" backend with ST)

    cfg = EMBEDDING_MODELS[model_key]

    # Reference: stock ST on CPU
    ref_model = SentenceTransformer(cfg["path"], device="cpu")
    ref_embeddings = ref_model.encode(TEST_SENTENCES, convert_to_tensor=True)
    del ref_model
    gc.collect()

    # Spyre backend (DEVICE patched to cpu by conftest)
    spyre_model = SentenceTransformer(cfg["path"], backend="spyre", device="cpu")
    unwrap_compiled_blocks(spyre_model._first_module().model)
    spyre_embeddings = spyre_model.encode(TEST_SENTENCES, convert_to_tensor=True)
    del spyre_model
    gc.collect()

    ref_norm = F.normalize(ref_embeddings.float(), dim=-1)
    spyre_norm = F.normalize(spyre_embeddings.float(), dim=-1)
    cos_sims = (ref_norm * spyre_norm).sum(dim=-1)

    min_sim = cos_sims.min().item()
    assert min_sim >= COS_SIM_THRESHOLD, (
        f"min cosine {min_sim:.6f} < threshold {COS_SIM_THRESHOLD}; "
        f"per-sentence: {cos_sims.tolist()}"
    )
