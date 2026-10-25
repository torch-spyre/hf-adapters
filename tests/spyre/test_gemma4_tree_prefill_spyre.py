# Copyright 2026 The Torch-Spyre Authors.
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

"""The tree verifier's prefill must give upstream ``generate``'s first token.

Usage::

    pytest -s -vvv tests/spyre/test_gemma4_tree_prefill_spyre.py
"""

import pytest
import torch
import torch_spyre  # noqa: F401
from transformers import AutoTokenizer

from hf_adapters import AutoSpyreModelForCausalLM
from hf_adapters._gemma4_tree import TreeVerifier
from hf_adapters.hf_common import encode_prompts
from tests.model_registry import CAUSAL_LM_MODELS, XPRESS_PATHS

pytestmark = pytest.mark.requires_spyre

# The Gemma 4 MoE targets of the registered XPress drafters.
TARGET_PATHS = sorted(
    {i["target"] for i in CAUSAL_LM_MODELS.values() if i["path"] in XPRESS_PATHS}
)
PROMPT = (
    "Explain, step by step, how a sliding-window KV cache works in a decoder. " * 80
)


@pytest.mark.parametrize("target_path", TARGET_PATHS)
def test_tree_prefill_first_token_matches_generate(target_path):
    tok = AutoTokenizer.from_pretrained(target_path)
    ids = encode_prompts(tok, [PROMPT])["input_ids"]
    assert ids.shape[1] > 1024  # past the sliding window
    model = AutoSpyreModelForCausalLM.from_pretrained(target_path, dtype=torch.float16)
    verifier = TreeVerifier(model, 16)
    # The verifier's chunk geometry: a one-shot 1472-row prefill gives
    # non-finite logits on stock torch-spyre f1e0145.
    out = model.generate(
        ids,
        max_new_tokens=1,
        do_sample=False,
        output_logits=True,
        return_dict_in_generate=True,
        prefill_chunk_size=verifier.prefill_chunk,
    )
    want = out.logits[0][0].float()
    assert torch.isfinite(want).all()
    got = (
        verifier.prefill(ids, max_new_tokens=32)[0, -1, : want.shape[-1]]
        .to("cpu")
        .float()
    )
    assert torch.isfinite(got).all()
    assert int(got.argmax()) == int(want.argmax())
    cos = torch.nn.functional.cosine_similarity(got, want, dim=0)
    assert cos > 0.99, float(cos)
