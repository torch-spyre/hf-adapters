# Copyright 2026 The Torch-Spyre Authors.
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

"""Opt-in direct-to-device checkpoint loading for Spyre.

The default path stages the whole model on the host and then transfers it:
``load_model_common(device_map="cpu")`` followed by ``model.to("spyre")``
(see ``_move_to_spyre_with_layout``). Peak host RSS is therefore the size of
the full checkpoint, and every weight is written twice.

This module keeps Transformers' checkpoint slices lazy on the host and
transfers each parameter the moment it reaches its *final* name and target
module. Peak host residency becomes one parameter at a time, plus whatever the
adapter declares in ``SPYRE_DIRECT_HOST_MODULES`` -- today the LM head, which
``pad_lm_head`` has to pad on the host. For a model whose head is a large
fraction of its weights the saving is correspondingly smaller, so measure
rather than assume.

Hook point
----------
``transformers.core_model_loading.set_param_for_module`` is the single funnel
where a realized tensor meets the module it belongs to::

    module_path, _, param_name = target_name.rpartition(".")
    module_obj = model.get_submodule(module_path) if module_path else model

Choosing the layout here rather than in ``safe_open`` matters: by this point
Transformers has already applied key renames and the weight converters
(``MergeModulelist`` expert stacking, ``Concatenate``, ``Chunk``,
``Transpose``) and has cast to the load dtype in ``_materialize_copy``. A
layout chosen from the checkpoint key, before those run, is a layout for a
name and shape that no longer exist.

Deciding per *parameter* rather than per *load* also means a module that must
stay on the host is handled by skipping its parameters, instead of the whole
feature switching itself off the way a mixed ``device_map`` would. Adapters
that keep ``_spyre_cpu_submodules`` on the host (BERT, DistilBERT, MPNet,
CLIP) are a separate matter: they have not opted in at all, and when they do,
those submodules are what ``SPYRE_DIRECT_HOST_MODULES`` should name.

A parameter *tied* to a host-resident one is skipped too. A tied output
projection has no checkpoint key, so only its source is ever offered to this
funnel; placing that source would place the head with it and the declared host
module would not hold. See ``_tied_to_host``.

Relationship to tensor parallel
-------------------------------
The TP path already loads shard-by-shard with the right layouts via the styles
in ``spyre_tensor_parallel.py``, so it gains nothing here. This module stays
out of its way: ``set_param_for_module`` receives a non-``None``
``distributed_operation`` under TP and the wrapper hands those straight to
Transformers. ``load_model_common`` additionally refuses to combine
``direct_to_device=True`` with ``tp_plan``.

Status
------
Per-adapter opt-in via ``SPYRE_SUPPORTS_DIRECT_LOAD``. An adapter may only opt
in once its ``prepare_for_spyre`` is known to work on a model whose weights are
already on the device: the prep step is documented as operating on the CPU
model, and RoPE caches in particular must be built on the host (see
``prebuild_rope_cache``). ``SPYRE_DIRECT_HOST_MODULES`` is the escape hatch for
modules that must stay on the host through prep; they are transferred later by
the ordinary ``model.to(device)``.
"""

import threading
from collections.abc import Sequence
from contextlib import contextmanager

import torch
import torch.nn as nn

__all__ = [
    "spyre_direct_placement",
    "supports_direct_load",
]

# Swapping a module-level function is process-global, so concurrent loads in
# one process must not each install and restore their own copy. The lock guards
# install/restore and the depth count makes the last one out restore exactly
# once, so an inner block leaving never strands an outer one.
_patch_lock = threading.Lock()
_patch_depth = 0
_original_set_param_for_module = None

# The installed wrapper is shared, but its *settings* are not: each thread
# loading a model has its own device and dtype (rank-local under torchrun).
# Keeping them in a thread-local stack means a concurrent load cannot inherit
# another thread's rank, and a thread that is not inside a direct-load block
# sees an empty stack and is passed straight through to Transformers.
_active = threading.local()


def _active_settings():
    """Settings for the direct-load block this thread is inside, if any."""
    stack = getattr(_active, "stack", None)
    return stack[-1] if stack else None


def supports_direct_load(adapter_module) -> bool:
    """True when ``adapter_module`` has opted in to direct-to-device loading."""
    return bool(getattr(adapter_module, "SPYRE_SUPPORTS_DIRECT_LOAD", False))


def _host_module_prefixes(adapter_module) -> tuple[str, ...]:
    """Dotted module paths whose parameters must stay on the host during load."""
    declared = getattr(adapter_module, "SPYRE_DIRECT_HOST_MODULES", ())
    if isinstance(declared, str):
        return (declared,)
    return tuple(str(prefix) for prefix in declared)


def _stays_on_host(target_name: str, host_prefixes: Sequence[str]) -> bool:
    """True when ``target_name`` belongs to a module kept on the host.

    Matches on dotted-path boundaries so ``"cls"`` does not also claim
    ``"classifier.weight"``.
    """
    for prefix in host_prefixes:
        if target_name == prefix or target_name.startswith(f"{prefix}."):
            return True
    return False


def _lookup_param(model, dotted_name):
    """Return the parameter or buffer at ``dotted_name``, or ``None`` if absent."""
    module_path, _, attr = dotted_name.rpartition(".")
    try:
        owner = model.get_submodule(module_path) if module_path else model
    except AttributeError:
        return None
    return getattr(owner, attr, None)


def _tying_is_active(model) -> bool:
    """True when this model ties its output projection to its input embedding."""
    config = getattr(model, "config", None)
    if config is None:
        return False
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    return bool(getattr(config, "tie_word_embeddings", False))


def _tied_to_host(model, target_name, host_prefixes: Sequence[str]) -> bool:
    """True when placing ``target_name`` would also place a host-resident weight.

    A tied output projection has no checkpoint key of its own: Transformers
    binds ``lm_head.weight`` to ``model.embed_tokens.weight`` once loading is
    finished, so only the embedding is ever offered to this funnel. Placing it
    therefore places the head too, and ``_stays_on_host("lm_head.weight")`` is
    never consulted -- the declared host module silently fails to hold.

    That matters beyond the layout: ``untie_embedding_and_lm_head`` then runs
    ``head.weight.detach().to("cpu").clone()`` on a device tensor, which is a
    readback, and for fp16 a lossy one (DLFLOAT16 on the device).

    Note the tie is **not** observable by identity here. Tying runs after the
    weights land, so while this funnel is live the tied parameter is still the
    meta placeholder and is not yet the same object as its source. The three
    conditions below are what can be seen at placement time:

    1. the model declares the pair in ``_tied_weights_keys`` -- which it does
       whether or not tying is switched on, so this alone is not enough;
    2. ``tie_word_embeddings`` is set on the (text) config;
    3. the tied parameter is still on ``meta``, i.e. nothing is going to fill
       it from the checkpoint independently.

    Condition 3 also keeps the answer right if a checkpoint does carry its own
    ``lm_head.weight``: that key arrives at this funnel on its own and is
    skipped by the ordinary prefix check instead.
    """
    tied_keys = getattr(model, "_tied_weights_keys", None)
    if not isinstance(tied_keys, dict) or not _tying_is_active(model):
        # Some families declare a list of name patterns rather than pairs;
        # there is no source to resolve, so leave those to the prefix check.
        return False
    for tied_name, source_name in tied_keys.items():
        if source_name != target_name:
            continue
        if not _stays_on_host(tied_name, host_prefixes):
            continue
        tied_param = _lookup_param(model, tied_name)
        if tied_param is not None and tied_param.device.type == "meta":
            return True
    return False


def _placement_copy(module, param_name, value, *, device, dtype):
    """Transfer one parameter to Spyre, layout chosen from its target module.

    Mirrors the decision ``torch_spyre.model_utils._transfer_module`` makes for
    ``model.to("spyre")`` -- 2D ``nn.Linear`` weights get ``dim_order=[1, 0]``,
    2D ``nn.Embedding`` tables get the gather-optimal indirect-access layout,
    everything else the default layout -- and reuses the same DMA helpers the
    TP styles call, so a weight lands identically whichever path loaded it.

    ``dtype`` is applied only to floating-point tensors. Integer and boolean
    buffers (token-type ids, causal masks, position ids) must keep their own
    dtype; handing them ``target_dtype=torch.float16`` would silently reinterpret
    them.
    """
    from hf_adapters.spyre_tensor_parallel import (
        _copy_default,
        _copy_embedding,
        _copy_linear,
    )

    target_dtype = dtype if value.is_floating_point() else None

    if param_name == "weight":
        if isinstance(module, nn.Linear):
            return _copy_linear(value, device=device, dtype=target_dtype)
        if isinstance(module, nn.Embedding):
            return _copy_embedding(value, device=device, dtype=target_dtype)
    return _copy_default(value, device=device, dtype=target_dtype)


def _resolve_target_module(model, target_name):
    """Return the ``nn.Module`` owning ``target_name``, or ``None`` if absent.

    Resolution mirrors ``set_param_for_module``. A missing submodule is not an
    error here: the wrapper simply declines to place that tensor and lets
    Transformers handle it as it would without this patch.
    """
    module_path, _, param_name = target_name.rpartition(".")
    if not module_path:
        return model, param_name
    try:
        return model.get_submodule(module_path), param_name
    except AttributeError:
        return None, param_name


def _should_place(
    target_name, param_value, distributed_operation, hf_quantizer, host_prefixes
):
    """Decide whether this wrapper owns the placement of one tensor.

    Declines, leaving Transformers' own behaviour untouched, when:

    * a ``distributed_operation`` is attached, i.e. a TP style owns it;
    * a quantizer owns it, since its packed representation is not ours to
      re-lay-out;
    * the value is not a plain tensor (``DTensor`` and friends);
    * it is not on the host already -- re-placing it would mean a device round
      trip, which for fp16 is also lossy (DLFLOAT16 on the device);
    * the adapter asked for that module to stay on the host.
    """
    if distributed_operation is not None or hf_quantizer is not None:
        return False
    if type(param_value) is not torch.Tensor and not isinstance(
        param_value, nn.Parameter
    ):
        return False
    if param_value.device.type != "cpu":
        return False
    return not _stays_on_host(target_name, host_prefixes)


@contextmanager
def spyre_direct_placement(*, device, dtype, host_modules=()):
    """Place each loaded parameter straight onto Spyre as it is materialized.

    Args:
        device: Full Spyre device **including its index** (``hf_common.DEVICE``,
            i.e. ``"spyre:<LOCAL_RANK>"``). The index is carried through to the
            DMA helpers rather than collapsed to the device type, so a rank
            loads onto its own card.
        dtype: Weight dtype for floating-point tensors. Transformers has
            already cast to the load dtype by this point, so this normally
            agrees with what arrives; it is passed through to keep a single
            source of truth for the device dtype.
        host_modules: Dotted module paths to leave on the host (adapters that
            keep submodules on CPU, and anything whose prep must run on the
            host). The ordinary ``model.to(device)`` transfers them afterwards.

    Only active inside the ``with`` block, so a process that never loads a
    model pays nothing -- in particular ``transformers`` is imported here and
    not at ``import torch`` time.

    Safe to nest and to enter from several threads at once: the installed
    wrapper is shared, but the device, dtype and host modules it uses are
    per-thread, and a thread outside any direct-load block is unaffected.
    """
    global _patch_depth, _original_set_param_for_module

    from transformers import core_model_loading

    settings = (torch.device(device), dtype, tuple(host_modules))

    with _patch_lock:
        if _patch_depth == 0:
            _original_set_param_for_module = core_model_loading.set_param_for_module
            core_model_loading.set_param_for_module = _spyre_set_param_for_module
        _patch_depth += 1

    stack = getattr(_active, "stack", None)
    if stack is None:
        stack = _active.stack = []
    stack.append(settings)

    try:
        yield
    finally:
        stack.pop()
        with _patch_lock:
            _patch_depth -= 1
            if _patch_depth == 0:
                core_model_loading.set_param_for_module = _original_set_param_for_module
                _original_set_param_for_module = None


def _spyre_set_param_for_module(
    model,
    target_name,
    param_value,
    loading_info,
    distributed_operation,
    hf_quantizer,
    *args,
    **kwargs,
):
    """Transfer the tensor to Spyre, then hand it to Transformers to install.

    Transformers keeps every other responsibility: ``set_param_for_module``
    still resolves the module, wraps parameters and records loading info. All
    this does is decide the tensor's device and layout first.
    """
    settings = _active_settings()
    placed = param_value
    if settings is not None:
        device, dtype, host_prefixes = settings
        if _should_place(
            target_name,
            param_value,
            distributed_operation,
            hf_quantizer,
            host_prefixes,
        ) and not _tied_to_host(model, target_name, host_prefixes):
            module, param_name = _resolve_target_module(model, target_name)
            if module is not None:
                placed = _placement_copy(
                    module, param_name, param_value, device=device, dtype=dtype
                )
    return _original_set_param_for_module(
        model,
        target_name,
        placed,
        loading_info,
        distributed_operation,
        hf_quantizer,
        *args,
        **kwargs,
    )


_spyre_set_param_for_module._spyre_direct_placement = True  # type: ignore[attr-defined]
