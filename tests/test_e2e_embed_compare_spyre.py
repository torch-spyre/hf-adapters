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

Encoder-only counterpart of test_e2e_token_compare_spyre.py. For each
registered BERT-family model, runs a single prefill on both CPU (stock HF
AutoModel) and Spyre (adapter via ``prefill_encoder``) over a batch with
mixed prompt lengths, then compares the last_hidden_state (per-token) and
the mean-pooled sentence embedding (per-sequence).

Mixed lengths exercise:
- The right-padded additive bidirectional mask (§2 of the bringup doc).
- The post-LN clone workaround in ``encoder_backbone_forward``.
- ``prefill_encoder``'s pad-to-BLOCK_SIZE + crop-back logic.

Usage (on Spyre pod):
    python3 tests/test_e2e_embed_compare_spyre.py [bge-base|minilm]
"""

import importlib
import sys
import traceback

import torch
import torch.nn.functional as F

from hf_adapters.hf_common import (
    _move_to_spyre_with_layout,
    _untie_embedding_and_lm_head,
    prefill_encoder,
)

DEVICE = "spyre"

MODEL_REGISTRY = {
    "bge-base": {
        "name": "BGE-base-en-v1.5",
        "path": "BAAI/bge-base-en-v1.5",
        "adapter": "hf_adapters.hf_bert",
    },
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "path": "sentence-transformers/all-MiniLM-L6-v2",
        "adapter": "hf_adapters.hf_bert",
    },
}

# Mixed-length prompts: short / medium / long. Forces a non-trivial padding
# pattern in the right-padded mask so that a broken bidirectional SDPA path
# would surface as NaN or per-token cosine drop.
PROMPTS = [
    "Hi.",
    "The capital of France is Paris.",
    "Sentence embeddings are useful for retrieval, clustering, "
    "and semantic search across large document collections.",
]

# Per-token cosine threshold over real (unmasked) positions. Matches the
# decoder e2e tolerance: fp16 + multi-layer drift, but well above what a
# broken kernel produces.
COSINE_THRESHOLD = 0.99


# ---------------------------------------------------------------------------
# HF reference: stock AutoModel.forward on CPU
# ---------------------------------------------------------------------------


def hf_reference_forward(model, input_ids, attention_mask):
    """Run stock HF encoder forward on CPU; return last_hidden_state."""
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
    return out.last_hidden_state  # [B, L, H]


# ---------------------------------------------------------------------------
# Adapter forward on Spyre
# ---------------------------------------------------------------------------


def adapter_forward(adapter, model, input_ids, attention_mask):
    """Run adapter prefill on Spyre; return last_hidden_state on CPU."""
    with torch.no_grad():
        h_dev, _ = prefill_encoder(
            adapter._run_backbone_forward,
            model,
            input_ids,
            attention_mask,
        )
    return h_dev.to("cpu")  # [B, L, H]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _mean_pool(hidden, mask):
    """Mean-pool hidden states over real tokens (sentence-transformers default)."""
    m = mask.unsqueeze(-1).float()
    summed = (hidden.float() * m).sum(dim=1)
    counts = m.sum(dim=1).clamp(min=1)
    return summed / counts  # [B, H]


def compare_results(hf_hidden, ad_hidden, attention_mask, model_name):
    """Compare per-sequence: per-token cosine, max diff, pooled cosine, NaN."""
    assert hf_hidden.shape == ad_hidden.shape, (
        f"shape mismatch: hf {tuple(hf_hidden.shape)} vs adapter "
        f"{tuple(ad_hidden.shape)}"
    )

    h32 = hf_hidden.float()
    a32 = ad_hidden.float()

    per_tok_cos = F.cosine_similarity(h32, a32, dim=-1)  # [B, L]
    abs_diff = (h32 - a32).abs()

    pooled_h = _mean_pool(hf_hidden, attention_mask)
    pooled_a = _mean_pool(ad_hidden, attention_mask)
    pooled_cos = F.cosine_similarity(pooled_h, pooled_a, dim=-1)  # [B]

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


def run_model_test(model_key):
    """Full comparison for one encoder model."""
    from transformers import AutoModel, AutoTokenizer

    info = MODEL_REGISTRY[model_key]
    adapter = importlib.import_module(info["adapter"])

    print(f"\n{'='*70}")
    print(f"  {info['name']}: {info['path']}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(info["path"])
    model = AutoModel.from_pretrained(
        info["path"],
        torch_dtype=torch.float16,
        device_map="cpu",
    )
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

    # --- HF reference on CPU (BEFORE prepare_for_spyre) ---
    # Encoder adapters do not patch globally today, but we keep the same
    # ordering discipline as the decoder e2e test.
    print("  Running HF reference on CPU ...")
    hf_hidden = hf_reference_forward(model, input_ids, attention_mask)

    # --- Adapter on Spyre ---
    print("  Preparing adapter ...")
    _untie_embedding_and_lm_head(model)  # no-op for encoders, mirrors decoder path
    adapter.prepare_for_spyre(model)
    print("  Moving model to Spyre ...")
    _move_to_spyre_with_layout(model, torch.float16)
    print("  Running adapter on Spyre ...")
    ad_hidden = adapter_forward(adapter, model, input_ids, attention_mask)

    rows = compare_results(hf_hidden, ad_hidden, attention_mask, info["name"])
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_table(all_rows):
    """Print markdown comparison table."""
    print("\n## E2E Embedding Comparison: HF (CPU) vs Adapter (Spyre)\n")
    print(
        "| Model | Row | Real Len | Mean Cos | Min Cos | Pooled Cos "
        "| Max Diff | Mean Diff | HF NaN | Spyre NaN | Match |"
    )
    print(
        "|-------|-----|----------|----------|---------|------------"
        "|----------|-----------|--------|-----------|-------|"
    )
    for r in all_rows:
        match = "OK" if r["match"] else "FAIL"
        hn = "Yes" if r["hf_nan"] else "No"
        sn = "Yes" if r["spyre_nan"] else "No"
        print(
            f"| {r['model']} | {r['row']} | {r['n_real']} "
            f"| {r['mean_cos']:.6f} | {r['min_cos']:.6f} | {r['pooled_cos']:.6f} "
            f"| {r['max_diff']:.4f} | {r['mean_diff']:.6f} "
            f"| {hn} | {sn} | {match} |"
        )


if __name__ == "__main__":
    which = sys.argv[1:] if len(sys.argv) > 1 else list(MODEL_REGISTRY.keys())

    all_rows = []
    for key in which:
        if key not in MODEL_REGISTRY:
            print(f"Unknown: {key}. Options: {list(MODEL_REGISTRY.keys())}")
            continue
        try:
            rows = run_model_test(key)
            all_rows.extend(rows)
        except Exception:
            print(f"\n!!! {MODEL_REGISTRY[key]['name']} FAILED:")
            traceback.print_exc()

    if all_rows:
        print_table(all_rows)
        n_match = sum(1 for r in all_rows if r["match"])
        print(
            f"\nPer-row min-cosine >= {COSINE_THRESHOLD}: "
            f"{n_match}/{len(all_rows)} rows"
        )
        sys.exit(0 if n_match == len(all_rows) else 1)
    sys.exit(1)
