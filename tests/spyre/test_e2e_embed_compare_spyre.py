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

"""
E2E embedding accuracy: HF stock forward (CPU) vs adapter forward (Spyre).

Encoder-only counterpart of test_e2e_token_compare_spyre.py.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_e2e_embed_compare_spyre.py
    pytest -s -vvv tests/spyre/test_e2e_embed_compare_spyre.py -k bge_base
"""

import types
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_registry import EMBED_PATHS

from hf_adapters.auto_spyre_model import (
    resolve_adapter_module,
    torch_dtype_for_model_path,
)
from hf_adapters.hf_common import (
    move_model_to_spyre,
    prefill_embed,
    prefill_encoder,
)
from tests.conftest import load_ref_model

PROMPTS = [
    "Hi.",
    "The capital of France is Paris.",
    "Sentence embeddings are useful for retrieval, clustering, "
    "and semantic search across large document collections.",
]

COSINE_THRESHOLD = 0.99


def _hf_reference_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run stock HF encoder forward on CPU; return last_hidden_state."""
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
    return out.last_hidden_state


def _adapter_forward(
    adapter: types.ModuleType,
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run adapter prefill on Spyre; return last_hidden_state on CPU."""
    with torch.no_grad():
        if getattr(adapter, "_is_encoder_only", False):
            h_dev, _ = prefill_encoder(
                adapter._run_backbone_forward,
                model,
                input_ids,
                attention_mask,
            )
        else:
            h_dev, _ = prefill_embed(
                adapter._run_backbone_forward,
                model,
                input_ids,
                attention_mask,
            )
    return h_dev.to("cpu")


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    summed = (hidden.float() * m).sum(dim=1)
    counts = m.sum(dim=1).clamp(min=1)
    return summed / counts


def _compare_results(
    hf_hidden: torch.Tensor,
    ad_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    model_name: str,
) -> list[dict[str, Any]]:
    """Compare per-sequence: per-token cosine, max diff, pooled cosine, NaN."""
    assert hf_hidden.shape == ad_hidden.shape, (
        f"shape mismatch: hf {tuple(hf_hidden.shape)} vs adapter "
        f"{tuple(ad_hidden.shape)}"
    )

    h32 = hf_hidden.float()
    a32 = ad_hidden.float()

    per_tok_cos = F.cosine_similarity(h32, a32, dim=-1)
    abs_diff = (h32 - a32).abs()

    pooled_h = _mean_pool(hf_hidden, attention_mask)
    pooled_a = _mean_pool(ad_hidden, attention_mask)
    pooled_cos = F.cosine_similarity(pooled_h, pooled_a, dim=-1)

    rows = []
    bsz = hf_hidden.shape[0]
    for b in range(bsz):
        m = attention_mask[b].bool()
        n_real = int(m.sum().item())
        cos_real = per_tok_cos[b][m]
        diff_real = abs_diff[b][m]
        rows.append(
            {
                "model": model_name,
                "row": b,
                "n_real": n_real,
                "mean_cos": cos_real.mean().item(),
                "min_cos": cos_real.min().item(),
                "pooled_cos": pooled_cos[b].item(),
                "max_diff": diff_real.max().item(),
                "mean_diff": diff_real.mean().item(),
                "hf_nan": bool(hf_hidden[b].isnan().any().item()),
                "spyre_nan": bool(ad_hidden[b].isnan().any().item()),
                "match": cos_real.min().item() >= COSINE_THRESHOLD,
            }
        )
    return rows


def _run_model_test(model_path: str) -> list[dict[str, Any]]:
    """Full comparison for one encoder model."""
    from transformers import AutoModel, AutoTokenizer

    adapter = resolve_adapter_module(model_path)

    print(f"\n{'=' * 70}")
    print(f"  {model_path}")
    print(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_ref_model(
        model_path=model_path,
        adapter_mod=adapter,
        auto_model_cls=AutoModel,
    )

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

    print("  Running HF reference on CPU ...")
    hf_hidden = _hf_reference_forward(model, input_ids, attention_mask)

    dtype = torch_dtype_for_model_path(model_path=model_path)
    move_model_to_spyre(model=model, module=adapter, dtype=dtype)

    print("  Running adapter on Spyre ...")
    ad_hidden = _adapter_forward(adapter, model, input_ids, attention_mask)

    return _compare_results(hf_hidden, ad_hidden, attention_mask, model_path)


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Markdown comparison table — one line per prompt row."""
    print("\n## E2E Embedding Comparison: HF (CPU) vs Adapter (Spyre)\n")
    print(
        "| Model | Row | Real Len | Mean Cos | Min Cos | Pooled Cos "
        "| Max Diff | Mean Diff | HF NaN | Spyre NaN | Match |"
    )
    print(
        "|-------|-----|----------|----------|---------|------------"
        "|----------|-----------|--------|-----------|-------|"
    )
    for r in rows:
        match = "OK" if r["match"] else "FAIL"
        hn = "Yes" if r["hf_nan"] else "No"
        sn = "Yes" if r["spyre_nan"] else "No"
        print(
            f"| {r['model']} | {r['row']} | {r['n_real']} "
            f"| {r['mean_cos']:.6f} | {r['min_cos']:.6f} | {r['pooled_cos']:.6f} "
            f"| {r['max_diff']:.4f} | {r['mean_diff']:.6f} "
            f"| {hn} | {sn} | {match} |"
        )


def embed_compare_spyre(
    model_path: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _run_model_test(model_path)
    mismatches = [r for r in rows if not r["match"]]
    return mismatches, rows


@pytest.mark.parametrize("model_path", EMBED_PATHS, ids=EMBED_PATHS)
def test_e2e_embed_compare_spyre(model_path: str) -> None:
    mismatches, rows = embed_compare_spyre(model_path)
    _print_table(rows)
    n_match = sum(1 for r in rows if r["match"])
    print(f"\nPer-row min-cosine >= {COSINE_THRESHOLD}: {n_match}/{len(rows)} rows")
    assert not mismatches, mismatches
