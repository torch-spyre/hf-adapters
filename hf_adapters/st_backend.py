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
``sentence_transformers`` by monkey-patching two methods:
``sentence_transformers.base.modules.transformer.Transformer._load_model`` and
``sentence_transformers.SentenceTransformer.__init__``. After import, the
standard ST API works unchanged::

    import hf_adapters.st_backend
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", backend="spyre")
    embeddings = model.encode(["hello", "world"])

``_load_model`` intercepts calls where ``backend == "spyre"``, loads and prepares
the backbone via ``AutoSpyreModel.from_pretrained``, then overrides
``model.forward`` to call ``prefill_embed`` and return a
``BaseModelOutput(last_hidden_state=...)`` — exactly the shape that ST's
``Transformer.forward`` expects.

The backbone runs on Spyre; ST's Pooling and Normalize run on the host (they are
dynamic-shaped pointwise ops that do not compile on Spyre). Two pieces arrange
that with an honest ``model.device`` (Spyre):

- The ``__init__`` patch defaults the device to ``"spyre"`` for
  ``backend="spyre"`` (unless the caller passed one), so ST's ``self.to(device)``
  keeps the prepared backbone on Spyre instead of moving it to CPU.
- ``_load_model`` wraps the ``Transformer`` module's ``forward`` to move the
  output ``features`` dict back to CPU before the downstream pooling modules read
  it, and moves the (Spyre-placed) inputs to CPU for the prefill drivers.

All downstream ST modules (``Pooling``, ``Normalize``, ``model.encode()``,
``model.similarity()``) run entirely unchanged.
"""

import types

import torch
from transformers.modeling_outputs import BaseModelOutput

from hf_adapters.auto_spyre_model import AutoSpyreModel, resolve_adapter_module
from hf_adapters.hf_common import prefill_embed, prefill_encoder


def _to_cpu(t):
    """Move a tensor to CPU if it is one; pass through ``None``/non-tensors."""
    return t.to("cpu") if isinstance(t, torch.Tensor) else t


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
            # ST places features on ``model.device`` (Spyre); the prefill drivers
            # build their CPU-side auxiliary tensors from these inputs (e.g.
            # ``attention_mask.sum`` for lengths) before moving to device, so the
            # inputs must be on CPU.
            h = prefill_encoder(
                run_backbone_forward,
                self,
                _to_cpu(input_ids),
                _to_cpu(attention_mask),
                token_type_ids=_to_cpu(token_type_ids),
            )
            return BaseModelOutput(last_hidden_state=h)

    else:

        def _spyre_forward(self, input_ids=None, attention_mask=None, **kwargs):
            # See the encoder branch: prefill drivers require CPU inputs.
            h = prefill_embed(
                run_backbone_forward, self, _to_cpu(input_ids), _to_cpu(attention_mask)
            )
            return BaseModelOutput(last_hidden_state=h)

    model.forward = types.MethodType(_spyre_forward, model)

    # Run ST's Pooling and Normalize (both dynamic-shaped pointwise ops that do
    # not compile on Spyre) on the host. ``self`` is the ST ``Transformer``
    # module: its ``forward`` runs the backbone on device, then hands the
    # ``features`` dict to the downstream Pooling/Normalize modules. Wrap it so
    # every tensor in the returned dict is on CPU — the backbone still runs on
    # Spyre, but ``token_embeddings``, ``attention_mask``, etc. are moved back to
    # the host before pooling reads them.
    _original_transformer_forward = self.forward

    def _cpu_pooling_forward(features, **kwargs):
        features = _original_transformer_forward(features, **kwargs)
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                features[key] = value.to("cpu")
        return features

    self.forward = _cpu_pooling_forward

    return model


def _spyre_init(self, *args, **kwargs):
    """``SentenceTransformer.__init__`` wrapper that defaults the device to Spyre.

    ``BaseModel.__init__`` resolves an unset ``device`` via ``get_device_name()``
    (CPU here) and finishes with ``self.to(device)``. For ``backend="spyre"`` that
    would move our prepared, device-resident backbone back to CPU. Thus, we
    default the device to ``"spyre".
    """
    if kwargs.get("backend") == "spyre" and kwargs.get("device") is None:
        kwargs["device"] = "spyre"
    return _original_init(self, *args, **kwargs)


def register():
    """Monkey-patch ST to add ``backend="spyre"`` support.

    Patches ``Transformer._load_model`` (to load/prepare the Spyre backbone) and
    ``SentenceTransformer.__init__`` (to default the device to Spyre for
    ``backend="spyre"``). Idempotent: calling ``register()`` more than once is safe.
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.base.modules.transformer import Transformer

    global _original_load_model, _original_init

    if getattr(Transformer, "_spyre_patched", False):
        return

    _original_load_model = Transformer._load_model
    Transformer._load_model = _spyre_load_model

    _original_init = SentenceTransformer.__init__
    SentenceTransformer.__init__ = _spyre_init

    Transformer._spyre_patched = True


# Register immediately on import.
_original_load_model = None
_original_init = None
register()
