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
HuggingFace Transformers adapter for the Qwen3 **DSpark drafter** on Spyre.

This is a speculative-decoding *drafter* (``Qwen3DSparkModel``), not a target
causal-LM: it proposes a whole block of tokens in one shot from the target
model's intermediate hidden states. See ``_dspark_common`` for the shared
machinery and why the public forward is ``_run_draft_block`` (block-propose,
the shared ``run_draft_block``) rather than the token-by-token
``_run_forward``/``generate`` surface.

Qwen3 specifics vs. the shared draft base: per-head q/k RMSNorm before RoPE
(``use_qk_norm=True``); no family scalar multipliers.

Usage::

    from hf_adapters import resolve_adapter_module
    from hf_adapters.hf_common import load_model_common, move_model_to_spyre

    module = resolve_adapter_module("/path/to/dspark_qwen3_drafter")   # -> this module
    model = load_model_common("/path/to/dspark_qwen3_drafter", module, torch.float16)
    move_model_to_spyre(model, module, torch.float16)
    block_hidden = module._run_draft_block(
        model, draft_input_ids, target_hidden_states, selected_freqs, ctx_valid_len)
"""

from hf_adapters._dspark_common import prepare_dspark_common, run_draft_block

_run_draft_block = run_draft_block  # reuse the common runner
# Fixed context / kv widths (stick-aligned). CTX_PAD=56 keeps kv_pad =
# round32(CTX_PAD + block_size=7) = 64 — the proven-compilable attention width.
CTX_PAD = 56


def prepare_for_spyre(model):
    """Apply Spyre adaptations to the Qwen3 DSpark drafter in-place."""
    from deepspec.modeling.dspark.qwen3.modeling import (  # type: ignore[import-not-found]
        Qwen3RMSNorm,
    )

    block_size = int(model.block_size)
    kv_pad = ((CTX_PAD + block_size + 31) // 32) * 32
    prepare_dspark_common(
        model, Qwen3RMSNorm, ctx_pad=CTX_PAD, kv_pad=kv_pad, use_qk_norm=True
    )
