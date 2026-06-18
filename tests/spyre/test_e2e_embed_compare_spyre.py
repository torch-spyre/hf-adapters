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

"""E2E embedding accuracy: HF stock forward (CPU) vs adapter forward (Spyre).

Encoder counterpart of test_e2e_token_compare_spyre.py. For each registered
embedding model, runs a single prefill on both CPU (stock HF AutoModel) and
Spyre (adapter via prefill_encoder / prefill_embed) over a batch with mixed
prompt lengths, then asserts:
  - No NaN in either side's hidden states.
  - Per-row min cosine over real (unmasked) tokens >= COSINE_THRESHOLD.

Mixed lengths exercise the right-padded additive bidirectional mask, the
post-LN clone workaround, and the pad-to-BLOCK_SIZE / crop-back logic.

Usage (on Spyre pod):
    pytest -s -vvv tests/spyre/test_e2e_embed_compare_spyre.py
    pytest -s -vvv tests/spyre/test_e2e_embed_compare_spyre.py -k bge_base
"""

import importlib

import pytest
import torch
import torch.nn.functional as F
from _helpers import torch_dtype_for
from model_registry import EMBEDDING_MODELS

PROMPTS = [
    "Hi.",
    "The capital of France is Paris.",
    "Sentence embeddings are useful for retrieval, clustering, "
    "and semantic search across large document collections.",
]

# Default per-row min-cosine threshold. Per-model overrides go in COSINE_OVERRIDES
# when current behavior is known to be looser (none today).
COSINE_THRESHOLD = 0.99
COSINE_OVERRIDES: dict = {}


def _mean_pool(hidden, mask):
    m = mask.unsqueeze(-1).float()
    summed = (hidden.float() * m).sum(dim=1)
    counts = m.sum(dim=1).clamp(min=1)
    return summed / counts


@pytest.mark.parametrize("model_key", list(EMBEDDING_MODELS.keys()))
def test_embed_compare(model_key):
    """CPU stock AutoModel vs Spyre adapter: per-token cosine over real tokens."""
    from transformers import AutoModel, AutoTokenizer

    from hf_adapters.hf_common import (
        _move_to_spyre_with_layout,
        _untie_embedding_and_lm_head,
        prefill_embed,
        prefill_encoder,
    )

    info = EMBEDDING_MODELS[model_key]
    threshold = COSINE_OVERRIDES.get(model_key, COSINE_THRESHOLD)
    print(f"\n  {info['name']}: {info['path']}")
    print(f"  min-cosine threshold: {threshold}")

    adapter_module_name = info["adapter"].replace(".py", "")
    adapter = importlib.import_module(f"hf_adapters.{adapter_module_name}")

    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    dtype = torch_dtype_for(info)
    model = AutoModel.from_pretrained(info["path"], torch_dtype=dtype, device_map="cpu")
    model.eval()
    model.requires_grad_(False)

    encoded = tokenizer(
        PROMPTS,
        return_tensors="pt",
        padding=True,
        padding_side="right",
        truncation=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    lengths = attention_mask.sum(dim=1).tolist()
    print(
        f"  Inputs: {len(PROMPTS)} prompts, padded to {input_ids.shape[1]} tokens"
        f" (real lengths: {lengths})"
    )

    # CPU reference BEFORE prepare_for_spyre (kept for ordering discipline).
    print("  Running HF reference on CPU ...")
    with torch.no_grad():
        out = model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
    hf_hidden = out.last_hidden_state

    print("  Preparing adapter and moving to Spyre ...")
    _untie_embedding_and_lm_head(model)
    adapter.prepare_for_spyre(model)
    _move_to_spyre_with_layout(model, dtype)

    print("  Running adapter on Spyre ...")
    with torch.no_grad():
        if getattr(adapter, "_is_encoder_only", False):
            h_dev, _ = prefill_encoder(
                adapter._run_backbone_forward, model, input_ids, attention_mask
            )
        else:
            h_dev, _ = prefill_embed(
                adapter._run_backbone_forward, model, input_ids, attention_mask
            )
    ad_hidden = h_dev.to("cpu")

    assert hf_hidden.shape == ad_hidden.shape, (
        f"{model_key}: shape mismatch HF {tuple(hf_hidden.shape)} "
        f"vs adapter {tuple(ad_hidden.shape)}"
    )

    h32 = hf_hidden.float()
    a32 = ad_hidden.float()
    per_tok_cos = F.cosine_similarity(h32, a32, dim=-1)

    pooled_h = _mean_pool(hf_hidden, attention_mask)
    pooled_a = _mean_pool(ad_hidden, attention_mask)
    pooled_cos = F.cosine_similarity(pooled_h, pooled_a, dim=-1)

    failures = []
    for b in range(hf_hidden.shape[0]):
        m = attention_mask[b].bool()
        n_real = int(m.sum().item())
        cos_real = per_tok_cos[b][m]
        min_cos = cos_real.min().item()
        mean_cos = cos_real.mean().item()
        hf_nan = bool(hf_hidden[b].isnan().any().item())
        sp_nan = bool(ad_hidden[b].isnan().any().item())
        print(
            f"  row={b} real_len={n_real} mean_cos={mean_cos:.6f} "
            f"min_cos={min_cos:.6f} pooled_cos={pooled_cos[b].item():.6f}"
        )
        if hf_nan:
            failures.append(f"row {b}: HF hidden contains NaN")
        if sp_nan:
            failures.append(f"row {b}: Spyre hidden contains NaN")
        if min_cos < threshold:
            failures.append(f"row {b}: min_cos {min_cos:.6f} < threshold {threshold}")

    if failures:
        pytest.fail(f"{model_key}: " + "; ".join(failures))
