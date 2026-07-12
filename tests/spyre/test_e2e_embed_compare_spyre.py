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
E2E embedding accuracy: stock SentenceTransformers (CPU) vs ``backend="spyre"``.

Both sides go through the standard ``sentence_transformers`` API. The CPU side
uses the stock loader; the Spyre side uses the ``"spyre"`` backend registered by
``hf_adapters.st_backend``, which loads/prepares the backbone via
``AutoSpyreModel`` and runs prefill on device. All downstream ST modules
(``Pooling``, ``Normalize``, ``model.encode()``) run unchanged on both sides.

We encode with ``output_value=None`` so ST hands back the full per-sequence
feature dict (``token_embeddings``, ``attention_mask``, ``sentence_embedding``).
This lets us report both the final sentence-embedding cosine and the detailed
per-token metrics (mean/min token cosine, pooled cosine) from the raw
``last_hidden_state`` — the same diagnostics the original raw-backbone test had.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_e2e_embed_compare_spyre.py
    pytest -s -vvv tests/spyre/test_e2e_embed_compare_spyre.py -k bge_base
"""

from typing import Any

import pytest
import torch
import torch.nn.functional as F
from conftest import get_dtype_for_cpu
from model_registry import EMBED_PATHS

# Registers the "spyre" backend with sentence_transformers on import.
import hf_adapters.st_backend  # noqa: F401
from hf_adapters.auto_spyre_model import torch_dtype_for_model_path

PROMPTS = [
    "Hi.",
    "The capital of France is Paris.",
    "Sentence embeddings are useful for retrieval, clustering, "
    "and semantic search across large document collections.",
]

COSINE_THRESHOLD = 0.99


def _encode(model, prompts: list[str]) -> list[dict[str, torch.Tensor]]:
    """Encode prompts, returning ST's full per-sequence feature dicts.

    ``output_value=None`` yields a list (one dict per prompt) carrying
    ``token_embeddings`` (padded ``[seq, hidden]``), ``attention_mask``, and the
    final ``sentence_embedding``.
    """
    return model.encode(prompts, output_value=None, convert_to_numpy=False)


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    summed = (hidden.float() * m).sum(dim=0)
    counts = m.sum(dim=0).clamp(min=1)
    return summed / counts


def _compare_results(
    cpu_out: list[dict[str, torch.Tensor]],
    spyre_out: list[dict[str, torch.Tensor]],
    model_name: str,
) -> list[dict[str, Any]]:
    """Compare per-prompt: token-level cosine/diff, pooled cosine, sentence cosine."""
    assert len(cpu_out) == len(
        spyre_out
    ), f"row count mismatch: cpu {len(cpu_out)} vs spyre {len(spyre_out)}"

    rows = []
    for b, (c, s) in enumerate(zip(cpu_out, spyre_out)):
        c_tok = c["token_embeddings"].float().cpu()
        s_tok = s["token_embeddings"].float().cpu()
        assert c_tok.shape == s_tok.shape, (
            f"row {b} token shape mismatch: cpu {tuple(c_tok.shape)} vs spyre "
            f"{tuple(s_tok.shape)}"
        )

        mask = c["attention_mask"].bool().cpu()
        n_real = int(mask.sum().item())

        per_tok_cos = F.cosine_similarity(c_tok, s_tok, dim=-1)[mask]
        abs_diff = (c_tok - s_tok).abs()[mask]

        pooled_c = _mean_pool(c_tok, c["attention_mask"].cpu())
        pooled_s = _mean_pool(s_tok, s["attention_mask"].cpu())
        pooled_cos = F.cosine_similarity(pooled_c, pooled_s, dim=0).item()

        sent_c = c["sentence_embedding"].float().cpu()
        sent_s = s["sentence_embedding"].float().cpu()
        sent_cos = F.cosine_similarity(sent_c, sent_s, dim=0).item()

        rows.append(
            {
                "model": model_name,
                "row": b,
                "n_real": n_real,
                "mean_cos": per_tok_cos.mean().item(),
                "min_cos": per_tok_cos.min().item(),
                "pooled_cos": pooled_cos,
                "sent_cos": sent_cos,
                "max_diff": abs_diff.max().item(),
                "mean_diff": abs_diff.mean().item(),
                "cpu_nan": bool(c_tok.isnan().any().item()),
                "spyre_nan": bool(s_tok.isnan().any().item()),
                "match": per_tok_cos.min().item() >= COSINE_THRESHOLD,
            }
        )
    return rows


def _run_model_test(model_path: str) -> list[dict[str, Any]]:
    """Full comparison for one embedding model via the ST API."""
    from sentence_transformers import SentenceTransformer

    print(f"\n{'=' * 70}")
    print(f"  {model_path}")
    print(f"{'=' * 70}")

    print("  Loading stock SentenceTransformer on CPU ...")
    cpu_dtype = get_dtype_for_cpu(model_path)
    cpu_model = SentenceTransformer(
        model_path,
        device="cpu",
        model_kwargs={"torch_dtype": cpu_dtype},
    )

    print("  Encoding on CPU ...")
    cpu_out = _encode(cpu_model, PROMPTS)

    # Match the Spyre-safe dtype the backend uses so the comparison is fair.
    dtype = torch_dtype_for_model_path(model_path=model_path)
    print(f"  Loading SentenceTransformer with backend='spyre' (dtype={dtype}) ...")
    spyre_model = SentenceTransformer(
        model_path,
        backend="spyre",
        model_kwargs={"torch_dtype": dtype},
    )

    print("  Encoding on Spyre ...")
    spyre_out = _encode(spyre_model, PROMPTS)

    return _compare_results(cpu_out, spyre_out, model_path)


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Markdown comparison table — one line per prompt row."""
    print("\n## E2E Embedding Comparison: ST (CPU) vs backend='spyre'\n")
    print(
        "| Model | Row | Real Len | Mean Cos | Min Cos | Pooled Cos | Sent Cos "
        "| Max Diff | Mean Diff | CPU NaN | Spyre NaN | Match |"
    )
    print(
        "|-------|-----|----------|----------|---------|------------|----------"
        "|----------|-----------|---------|-----------|-------|"
    )
    for r in rows:
        match = "OK" if r["match"] else "FAIL"
        cn = "Yes" if r["cpu_nan"] else "No"
        sn = "Yes" if r["spyre_nan"] else "No"
        print(
            f"| {r['model']} | {r['row']} | {r['n_real']} "
            f"| {r['mean_cos']:.6f} | {r['min_cos']:.6f} | {r['pooled_cos']:.6f} "
            f"| {r['sent_cos']:.6f} | {r['max_diff']:.4f} | {r['mean_diff']:.6f} "
            f"| {cn} | {sn} | {match} |"
        )


@pytest.mark.parametrize("model_path", EMBED_PATHS, ids=EMBED_PATHS)
def test_e2e_embed_compare_spyre(model_path: str) -> None:
    rows = _run_model_test(model_path)
    _print_table(rows)
    n_match = sum(1 for r in rows if r["match"])
    print(f"\nPer-row min-cosine >= {COSINE_THRESHOLD}: {n_match}/{len(rows)} rows")
    mismatches = [r for r in rows if not r["match"]]
    assert not mismatches, mismatches
