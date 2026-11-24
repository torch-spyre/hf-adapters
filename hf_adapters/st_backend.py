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
``sentence_transformers`` by patching model loading, CPU handoff, and device
placement. After import, the standard ST API works unchanged::

    import hf_adapters.st_backend
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", backend="spyre")
    embeddings = model.encode(["hello", "world"])

``_load_model`` intercepts calls where ``backend == "spyre"``, loads and prepares
the backbone via ``AutoSpyreModel.from_pretrained``, then overrides
``model.forward`` to call ``prefill_embed`` and return a
``BaseModelOutput(last_hidden_state=...)`` — exactly the shape that ST's
``Transformer.forward`` expects.

Dual encoders exposing ``get_text_features`` and ``get_image_features`` keep
their native HF forwards and pooling. ST dispatches directly to these feature
methods, whose Spyre adapters already return CPU embeddings.

Multimodal encoders such as EmbeddingGemma 2 opt into retaining their native
forward too, with the same CPU handoff to ST pooling as text-only encoders.

The backbone runs on Spyre; ST's Pooling and Normalize run on the host (they are
dynamic-shaped pointwise ops that do not compile on Spyre). Two pieces arrange
that with an honest ``model.device`` (Spyre):

- The ``__init__`` patch defaults the device to ``"spyre"`` for
  ``backend="spyre"`` (unless the caller passed one), so ST's ``self.to(device)``
  keeps the prepared backbone on Spyre instead of moving it to CPU.
- ``_load_model`` wraps the ``Transformer`` module's ``forward`` to move the
  output ``features`` dict back to CPU before the downstream pooling modules read
  it, and moves the (Spyre-placed) inputs to CPU for the prefill drivers.

Adapters declaring a resident compute parameter also opt into preserving their
mixed CPU/Spyre placement during inference. ST's uniform-device check must not
copy their CPU helpers to Spyre before each encode.
"""

import types
from collections.abc import Callable
from typing import Any

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from hf_adapters.auto_spyre_model import AutoSpyreModel, resolve_adapter_module
from hf_adapters.hf_common import cpu_submodules_on_host, prefill_embed, prefill_encoder


def _to_cpu(t):
    """Move a tensor to CPU if it is one; pass through ``None``/non-tensors."""
    return t.to("cpu") if isinstance(t, torch.Tensor) else t


def _spyre_load_model(
    self: Any,
    model_name_or_path: str,
    transformer_task: str,
    config: PretrainedConfig,
    backend: str,
    is_peft_model: bool,
    **model_kwargs: Any,
) -> PreTrainedModel:
    """Replacement for ``Transformer._load_model`` that handles ``backend="spyre"``."""
    if backend != "spyre":
        assert _original_load_model is not None
        original_model: PreTrainedModel = _original_load_model(
            self,
            model_name_or_path,
            transformer_task,
            config,
            backend,
            is_peft_model,
            **model_kwargs,
        )
        return original_model

    dtype = model_kwargs.pop("dtype", None)
    if dtype is None:
        dtype = model_kwargs.pop("torch_dtype", None)

    # Remote-code checkpoints ship custom modeling code that both the backbone
    # load and the adapter-module resolution must opt into; ST callers pass it
    # via ``model_kwargs`` (e.g. ``SentenceTransformer(path, backend="spyre",
    # model_kwargs={"trust_remote_code": True})``).
    trust_remote_code = model_kwargs.pop("trust_remote_code", None)

    model = AutoSpyreModel.from_pretrained(
        model_name_or_path, dtype=dtype, trust_remote_code=trust_remote_code
    )
    # ST detects native dual-encoder feature methods. These adapters already
    # retain HF pooling/projection and return CPU features; replacing forward
    # with a token-level prefill driver would break the public joint API.
    if callable(getattr(model, "get_text_features", None)) and callable(
        getattr(model, "get_image_features", None)
    ):
        return model
    adapter_module = resolve_adapter_module(
        model_name_or_path, trust_remote_code=trust_remote_code
    )

    preserve_forward: bool = getattr(adapter_module, "_preserve_native_forward", False)
    run_backbone_forward: Callable[..., torch.Tensor] | None = (
        None if preserve_forward else adapter_module._run_backbone_forward
    )

    # Route to the right prefill driver. Encoder-only adapters set
    # _is_encoder_only = True; decoder adapters leave it absent.
    is_encoder_only = getattr(adapter_module, "_is_encoder_only", False)

    if is_encoder_only:

        def _spyre_forward(
            self: PreTrainedModel,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> BaseModelOutput:
            assert run_backbone_forward is not None
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

        def _spyre_forward(
            self: PreTrainedModel,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> BaseModelOutput:
            assert run_backbone_forward is not None
            # See the encoder branch: prefill drivers require CPU inputs.
            h = prefill_embed(
                run_backbone_forward, self, _to_cpu(input_ids), _to_cpu(attention_mask)
            )
            return BaseModelOutput(last_hidden_state=h)

    if not preserve_forward:
        model.forward = types.MethodType(_spyre_forward, model)

    # Run ST's Pooling and Normalize (both dynamic-shaped pointwise ops that do
    # not compile on Spyre) on the host. ``self`` is the ST ``Transformer``
    # module: its ``forward`` runs the backbone on device, then hands the
    # ``features`` dict to the downstream Pooling/Normalize modules. Wrap it so
    # every tensor in the returned dict is on CPU — the backbone still runs on
    # Spyre, but ``token_embeddings``, ``attention_mask``, etc. are moved back to
    # the host before pooling reads them.
    _original_transformer_forward = self.forward

    def _cpu_pooling_forward(features: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        features = _original_transformer_forward(features, **kwargs)
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                features[key] = value.to("cpu")
        return features

    self.forward = _cpu_pooling_forward

    return model


def _spyre_init(self: Any, *args: Any, **kwargs: Any) -> None:
    """``SentenceTransformer.__init__`` wrapper that defaults the device to Spyre.

    ``BaseModel.__init__`` resolves an unset ``device`` via ``get_device_name()``
    (CPU here) and finishes with ``self.to(device)``. For ``backend="spyre"`` that
    would move our prepared, device-resident backbone back to CPU. Thus, we
    default the device to ``"spyre".
    """
    self._spyre_backend = kwargs.get("backend") == "spyre"
    if self._spyre_backend and kwargs.get("device") is None:
        kwargs["device"] = "spyre"
    assert _original_init is not None
    _original_init(self, *args, **kwargs)
    if _spyre_compute_device(self) is not None:
        _restore_cpu_modules(self)


def _restore_cpu_submodule(submod: torch.nn.Module) -> None:
    """Move a sub-module back to CPU and restore any integer buffers corrupted by Spyre.

    Moving a module to Spyre casts all buffers to the model's float dtype —
    including Long index tensors like ``position_ids``. Moving back to CPU with
    ``.to("cpu")`` restores the device but not the dtype. Walk every registered
    buffer and re-cast known integer buffers (position_ids, token_type_ids) back
    to long so downstream ``nn.Embedding`` lookups receive the correct dtype.
    """
    submod.to("cpu")
    for name, buf in submod.named_buffers():
        if buf is not None and "ids" in name and buf.dtype != torch.long:
            # Re-cast in-place via register_buffer to preserve the attribute reference.
            # Split the path so we target the right sub-module.
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                owner: torch.nn.Module = submod.get_submodule(parts[0])
                owner.register_buffer(parts[1], buf.to(torch.long))
            else:
                submod.register_buffer(name, buf.to(torch.long))


def _restore_cpu_modules(self: Any) -> None:
    """Pin ST heads and adapter-declared helper modules to the host."""
    preserve_buffers: bool = _spyre_compute_device(self) is not None
    for module in list(self.children())[1:]:
        module.to("cpu")
    # ST Transformer and CLIP wrappers use different backbone attributes.
    backbone_wrapper: torch.nn.Module | None = next(iter(self.children()), None)
    if backbone_wrapper is not None:
        hf_model: torch.nn.Module | None = getattr(
            backbone_wrapper, "auto_model", None
        ) or getattr(backbone_wrapper, "model", None)
        if hf_model is not None:
            for submod_name in getattr(hf_model, "_spyre_cpu_submodules", []):
                try:
                    submod: torch.nn.Module = hf_model.get_submodule(submod_name)
                    if preserve_buffers:
                        submod.to("cpu")
                    else:
                        _restore_cpu_submodule(submod)
                except AttributeError:
                    pass


def _spyre_forward(self: Any, input: dict[str, Any], **kwargs: Any) -> Any:
    """Keep host helpers usable after explicit moves or ST's default placement.

    For adapters preserving mixed placement these CPU moves are no-ops during
    normal inference. Other adapters retain their existing CPU restoration.
    """
    if getattr(self, "_spyre_backend", False):
        _restore_cpu_modules(self)
    assert _original_forward is not None
    return _original_forward(self, input, **kwargs)


def register() -> None:
    """Monkey-patch ST to add ``backend="spyre"`` support.

    Patches ``Transformer._load_model`` (to load/prepare the Spyre backbone) and
    ``SentenceTransformer.__init__`` (to default the device to Spyre for
    ``backend="spyre"``). Idempotent: calling ``register()`` more than once is safe.
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.base.modules.transformer import Transformer

    global _original_load_model, _original_init, _original_forward, _original_device
    global _original_resolve_inference_device
    global _original_to

    if getattr(Transformer, "_spyre_patched", False):
        return

    _original_load_model = Transformer._load_model
    Transformer._load_model = _spyre_load_model

    _original_init = SentenceTransformer.__init__
    SentenceTransformer.__init__ = _spyre_init

    _original_forward = SentenceTransformer.forward
    SentenceTransformer.forward = _spyre_forward

    _original_device = SentenceTransformer.device
    SentenceTransformer.device = property(_spyre_device)

    # Only SentenceTransformers >= 6.2 has this hook. Older releases still work
    # for every adapter except those opting into mixed CPU/Spyre placement,
    # which _spyre_compute_device rejects with a clear error.
    resolve_inference_device: Callable[..., torch.device] | None = getattr(
        SentenceTransformer, "_resolve_inference_device", None
    )
    if resolve_inference_device is not None:
        _original_resolve_inference_device = resolve_inference_device
        SentenceTransformer._resolve_inference_device = _spyre_resolve_inference_device

    _original_to = SentenceTransformer.to
    SentenceTransformer.to = _spyre_to

    Transformer._spyre_patched = True


def _spyre_compute_device(self: Any) -> torch.device | None:
    """Get the resident device for adapters opting into mixed placement."""
    if getattr(self, "_spyre_backend", False):
        hf_model: PreTrainedModel | None = self.transformers_model
        parameter_name: str | None = getattr(
            hf_model, "_spyre_device_parameter_name", None
        )
        if hf_model is not None and parameter_name is not None:
            if _original_resolve_inference_device is None:
                raise RuntimeError(
                    f"{type(hf_model).__name__} keeps some modules on CPU and needs "
                    "sentence-transformers>=6.2.0 for its Spyre placement; "
                    "upgrade sentence-transformers."
                )
            device: torch.device = hf_model.get_parameter(parameter_name).device
            return device
    return None


def _spyre_device(self: Any) -> torch.device:
    """Report the compute device only for adapters that explicitly opt in."""
    device: torch.device | None = _spyre_compute_device(self)
    if device is not None:
        return device
    assert _original_device is not None and _original_device.fget is not None
    original_device: torch.device = _original_device.fget(self)
    return original_device


def _spyre_to(self: Any, *args: Any, **kwargs: Any) -> Any:
    """Keep opted-in CPU helpers off the accelerator during ST initialization.

    Also protect ST's host pooling/projection modules. Weight dtype requests
    still apply to those helpers, while their buffers retain their HF values.
    """
    assert _original_to is not None
    if _spyre_compute_device(self) is None:
        return _original_to(self, *args, **kwargs)
    hf_model: PreTrainedModel = self.transformers_model
    backbone_path: str = next(
        name for name, module in self.named_modules() if module is hf_model
    )
    cpu_paths: tuple[str, ...] = tuple(getattr(hf_model, "_spyre_cpu_submodules", ()))
    paths: tuple[str, ...] = tuple(
        f"{backbone_path}.{name}" for name in cpu_paths
    ) + tuple(name for name, _ in list(self.named_children())[1:])
    dtype: torch.dtype | None = torch._C._nn._parse_to(*args, **kwargs)[1]
    with cpu_submodules_on_host(self, dtype, submodule_paths=paths):
        return _original_to(self, *args, **kwargs)


def _spyre_resolve_inference_device(
    self: Any, device: int | str | torch.device | None
) -> torch.device:
    """Resolve input placement without moving a deliberately split model."""
    resident_device: torch.device | None = _spyre_compute_device(self)
    if resident_device is None:
        assert _original_resolve_inference_device is not None
        resolved: torch.device = _original_resolve_inference_device(self, device)
        return resolved
    if device is not None:
        # Do not allocate a dummy tensor to canonicalize the device: zero-length
        # tensors are unsafe on Spyre. An unindexed device means the resident one.
        requested = torch.device(device)
        if requested.type != resident_device.type or (
            requested.index is not None
            and requested.type != "cpu"
            and requested.index != resident_device.index
        ):
            raise ValueError(
                f"encode(device={device!r}) cannot relocate this Spyre adapter; "
                f"its compute weights are on {resident_device}. "
                "Use the resident device or omit device."
            )
    return resident_device


# Register immediately on import.
_original_load_model = None
_original_init = None
_original_forward = None
_original_device: property | None = None
_original_resolve_inference_device: Callable[..., torch.device] | None = None
_original_to: Callable[..., Any] | None = None
register()
