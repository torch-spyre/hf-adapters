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

"""sentence-transformers ``backend="spyre"`` support.

Importing this module registers a ``"spyre"`` backend with
``sentence_transformers`` by monkey-patching
``sentence_transformers.base.modules.transformer.Transformer._load_model``.
After import, the standard ST API works unchanged::

    import hf_adapters.st_backend
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", backend="spyre")
    embeddings = model.encode(["hello", "world"])

The patch intercepts calls where ``backend == "spyre"``, loads and prepares
the backbone via ``AutoSpyreModel.from_pretrained``, then overrides
``model.forward`` to call ``prefill_embed`` and return a
``BaseModelOutput(last_hidden_state=...)`` — exactly the shape that ST's
``Transformer.forward`` expects.

All downstream ST modules (``Pooling``, ``Normalize``, ``model.encode()``,
``model.similarity()``) run entirely unchanged.
"""

import types

import torch
from transformers.modeling_outputs import BaseModelOutput

from hf_adapters.auto_spyre_model import AutoSpyreModel, resolve_adapter_module
from hf_adapters.hf_common import prefill_embed, prefill_encoder


def _spyre_load_model(
    self,
    model_name_or_path,
    transformer_task,
    config,
    backend,
    is_peft_model,
    **model_kwargs,
):
    """Replacement for ``Transformer._load_model`` that handles ``backend="spyre"``."""
    if backend != "spyre":
        return _original_load_model(
            self,
            model_name_or_path,
            transformer_task,
            config,
            backend,
            is_peft_model,
            **model_kwargs,
        )

    dtype = model_kwargs.pop("torch_dtype", torch.float16)
    model = AutoSpyreModel.from_pretrained(model_name_or_path, dtype=dtype)

    adapter_module = resolve_adapter_module(model_name_or_path)
    run_backbone_forward = adapter_module._run_backbone_forward

    # Route to the right prefill driver. Encoder-only adapters set
    # _is_encoder_only = True; decoder adapters leave it absent.
    is_encoder_only = getattr(adapter_module, "_is_encoder_only", False)

    if is_encoder_only:

        def _spyre_forward(self, input_ids=None, attention_mask=None, **kwargs):
            token_type_ids = kwargs.get("token_type_ids", None)
            h, _ = prefill_encoder(
                run_backbone_forward,
                self,
                input_ids,
                attention_mask,
                token_type_ids=token_type_ids,
            )
            return BaseModelOutput(last_hidden_state=h)

    else:

        def _spyre_forward(self, input_ids=None, attention_mask=None, **kwargs):
            h, _ = prefill_embed(run_backbone_forward, self, input_ids, attention_mask)
            return BaseModelOutput(last_hidden_state=h)

    model.forward = types.MethodType(_spyre_forward, model)

    return model


def register():
    """Monkey-patch ``Transformer._load_model`` to add ``backend="spyre"`` support.

    Idempotent: calling ``register()`` more than once is safe.
    """
    from sentence_transformers.base.modules.transformer import Transformer

    global _original_load_model

    if getattr(Transformer, "_spyre_patched", False):
        return

    _original_load_model = Transformer._load_model
    Transformer._load_model = _spyre_load_model
    Transformer._spyre_patched = True


# Register immediately on import.
_original_load_model = None
register()
