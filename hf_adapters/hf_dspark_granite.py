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
HuggingFace Transformers adapter for the Granite **DSpark drafter** on Spyre.

A speculative-decoding *drafter* (``GraniteDSparkModel``); see ``_dspark_common``
for the shared block-propose machinery and public ``_run_draft_block`` surface.

Granite specifics vs. the shared draft base (the same scalar multipliers as the
Granite *target* adapter, ``hf_granite``):

- ``attention_multiplier`` is the SDPA scale (not ``head_dim**-0.5``);
- ``embedding_multiplier`` scales the noise embedding (applied in
  ``_dspark_common.embed_noise_block`` via ``model.embedding_multiplier``);
- ``residual_multiplier`` scales each residual branch (applied per block via the
  layer's ``residual_multiplier``);
- ``logits_scaling`` divides the base logits — applied by the drafter's own
  ``compute_logits`` downstream, not here.

No per-head q/k RMSNorm (``use_qk_norm=False``).

Usage: see ``hf_dspark_qwen3``.
"""

from hf_adapters._dspark_common import prepare_dspark_common, run_draft_block

_run_draft_block = run_draft_block  # reuse the common runner
CTX_PAD = 56


def prepare_for_spyre(model):
    """Apply Spyre adaptations to the Granite DSpark drafter in-place."""
    from transformers.models.granite.modeling_granite import GraniteRMSNorm

    block_size = int(model.block_size)
    kv_pad = ((CTX_PAD + block_size + 31) // 32) * 32
    # Granite attention scale is the configured multiplier, not head_dim**-0.5.
    model._spyre_attn_scaling = float(model.config.attention_multiplier)
    prepare_dspark_common(
        model, GraniteRMSNorm, ctx_pad=CTX_PAD, kv_pad=kv_pad, use_qk_norm=False
    )
