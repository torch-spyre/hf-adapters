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
CPU accuracy test for the embedding path.

Covers both decoder-backbone embedders (Qwen3-Embedding, etc.)
and encoder-only models (e.g., BERT-family).

For each registered embedder backbone, ``test_auto_loader[<path>]`` compares
stock ``AutoModel`` with ``AutoSpyreModel`` for two input profiles:

- A short, naturally padded batch.
- A padded 65-token batch that crosses the 64-token block boundary.

The adapter forward dispatches to ``prefill_encoder`` for encoder-only
adapters and ``prefill_embed`` for decoder-backbone embedders. Both models are
loaded only once per checkpoint, so adding the boundary profile has little
cost compared with parametrizing the whole test by profile.

The DEVICE='cpu' patching of ``hf_common`` happens once in
``tests/conftest.py``; this file is plain pytest.
"""

import gc
import sys
import types

import pytest
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from tests.conftest import (
    load_ref_model,
    resolve_adapter_module_for_test,
)
from tests.cpu.conftest import _unwrap_compiled_blocks, encode_padded, min_cosine
from tests.model_registry import EMBED_PATHS, REMOTE_CODE_PATHS

pytestmark = pytest.mark.model_harness("embedding")

SHORT_PROMPTS: list[str] = [
    "The capital of France is Paris.",
    "Sentence embeddings are useful.",
]
BOUNDARY_SEQUENCE_LENGTH: int = 65
LONG_PROMPTS: list[str] = [
    "Embedding models map text to vectors. " * 80,
    "Short input.",
]
COS_THRESHOLD: float = 0.999


def _make_input_profiles(
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Create generic padded inputs for ordinary and block-boundary coverage."""
    short_input_ids, short_attention_mask = encode_padded(tokenizer, SHORT_PROMPTS)
    boundary_batch = tokenizer(
        LONG_PROMPTS,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=BOUNDARY_SEQUENCE_LENGTH,
    )
    boundary_input_ids: torch.Tensor = boundary_batch["input_ids"]
    boundary_attention_mask: torch.Tensor = boundary_batch["attention_mask"]
    return {
        "short_padded": (short_input_ids, short_attention_mask),
        "block_boundary": (boundary_input_ids, boundary_attention_mask),
    }


def _run_prefill(
    adapter_mod: types.ModuleType,
    hf_common_mod: types.ModuleType,
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Dispatch to prefill_encoder or prefill_embed based on adapter type."""
    if getattr(adapter_mod, "_is_encoder_only", False):
        return hf_common_mod.prefill_encoder(
            adapter_mod._run_backbone_forward, model, input_ids, attention_mask
        )
    else:
        return hf_common_mod.prefill_embed(
            adapter_mod._run_backbone_forward, model, input_ids, attention_mask
        )


@pytest.mark.parametrize("model_path", EMBED_PATHS, ids=EMBED_PATHS)
def test_auto_loader(model_path: str, trust_remote_code: bool | None) -> None:
    auto_spyre_model = sys.modules["hf_adapters.auto_spyre_model"]
    hf_common_mod = sys.modules["hf_adapters.hf_common"]
    if trust_remote_code is None:
        trust_remote_code = model_path in REMOTE_CODE_PATHS
    adapter_module = resolve_adapter_module_for_test(
        model_path, trust_remote_code=trust_remote_code
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    input_profiles = _make_input_profiles(tokenizer)

    # HF reference (loaded fresh, before the auto-loader path).
    ref_model = load_ref_model(
        model_path=model_path,
        adapter_mod=adapter_module,
        auto_model_cls=AutoModel,
        trust_remote_code=trust_remote_code,
    )
    reference_hidden: dict[str, torch.Tensor] = {}
    for profile_name, (input_ids, attention_mask) in input_profiles.items():
        with torch.no_grad():
            reference_hidden[profile_name] = ref_model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).last_hidden_state
    del ref_model
    gc.collect()

    # Auto-loader path
    model = auto_spyre_model.AutoSpyreModel.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    _unwrap_compiled_blocks(model)
    for profile_name, (input_ids, attention_mask) in input_profiles.items():
        with torch.no_grad():
            adapter_hidden = _run_prefill(
                adapter_module, hf_common_mod, model, input_ids, attention_mask
            )
        ref_hidden = reference_hidden[profile_name]
        assert adapter_hidden.shape == ref_hidden.shape, (
            f"{profile_name}: shape mismatch: adapter {adapter_hidden.shape} "
            f"vs ref {ref_hidden.shape}"
        )
        min_cos = min_cosine(adapter_hidden, ref_hidden, attention_mask)
        assert min_cos >= COS_THRESHOLD, (
            f"{profile_name}: min per-token cosine {min_cos:.6f} "
            f"< threshold {COS_THRESHOLD}"
        )
    del model
    gc.collect()
