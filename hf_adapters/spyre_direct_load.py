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

The default path is ``load_model_common(device_map="cpu")`` followed by
``model.to("spyre")`` (see ``_move_to_spyre_with_layout``). For a
dtype-converting load that stages a full host copy of every weight; when
the load dtype matches the checkpoint, Transformers already mmap's and
that copy is not made.

This module keeps Transformers' checkpoint slices lazy on the host and
transfers each parameter the moment it reaches its *final* name and target
module. On Spyre the transfers run on torch-spyre's upload threads, as
``model.to``'s do, while loading continues, and at most
``max_in_flight_bytes`` (1 GiB by default) of host tensors wait for them.
Peak *anonymous* host memory is then that, plus whatever the adapter holds
back (``SPYRE_DIRECT_HOST_MODULES`` and ``spyre_direct_stays_on_host``) --
today the LM head, which ``pad_lm_head`` has to pad on the host, and
attention projections ``pad_attention_heads`` rebuilds when the head dim is
not stick-aligned.

That saving is for *dtype-converting* loads. When the load dtype matches the
checkpoint, Transformers already hands out views into the memory-mapped file
and the staged path is cheap; measured on Granite 3.3 8B in bf16, anonymous
memory was about the same either way. Loading that same checkpoint as fp16,
the staged path held a full converted copy (16.9 GiB anonymous at peak) and
this path 2.7 GiB, against 1.7 GiB for the bf16 load, which converts
nothing; the difference is the tied embedding kept on the host and converted
weights waiting for the upload threads. Both fp16 loads took 10.5 s.
Uploading one weight at a time, the direct load took 32.5 s, because
``model.to`` uploads on several threads; the upload threads close that gap.
File-backed pages of the mmap stay resident until
the loader drops its references; those pages are reclaimable, unlike the
anonymous copies.

The read-ahead is ``HF_DEACTIVATE_ASYNC_LOAD``. Without it the workers
produce converted host copies faster than the card consumes them (15.8 GiB
anonymous on the same fp16 load), and the load is no faster. The variable is
process-wide, so a concurrent ordinary load in another thread would also run
without read-ahead for that window; that is a speed matter, not a
correctness one.

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
out of its way: since Transformers 5.23 a TP style is applied to the tensor
before ``set_param_for_module`` is reached, so the funnel this module patches
is not where sharding is decided, and ``load_model_common`` refuses to combine
``direct_to_device=True`` with ``tp_plan`` outright.

Status
------
Per-adapter opt-in via ``SPYRE_SUPPORTS_DIRECT_LOAD``. An adapter may only opt
in once its ``prepare_for_spyre`` is known to work on a model whose weights are
already on the device: the prep step is documented as operating on the CPU
model, and RoPE caches in particular must be built on the host (see
``prebuild_rope_cache``). ``SPYRE_DIRECT_HOST_MODULES`` is the escape hatch for
modules that must stay on the host through prep; they are transferred later by
the ordinary ``model.to(device)``. When that depends on the config, the adapter
provides ``spyre_direct_stays_on_host(model, target_name)`` instead -- e.g. the
attention projections ``pad_attention_heads`` rebuilds when the head dim is not
stick-aligned.
"""

import dataclasses
import os
import threading
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from typing import NamedTuple

import torch
import torch.nn as nn

__all__ = [
    "default_upload_threads",
    "spyre_direct_placement",
    "supports_direct_load",
]

# Swapping a module-level function is process-global, so concurrent loads in
# one process must not each install and restore their own copy. The lock guards
# install/restore and the depth count makes the last one out restore exactly
# once, so an inner block leaving never strands an outer one.
# ``_original_set_param_for_module`` is kept after the last block exits: a load
# on another thread may have looked the wrapper up just before, and its call
# still has to reach Transformers.
_patch_lock = threading.Lock()
_patch_depth = 0
_original_set_param_for_module = None
# Previous ``HF_DEACTIVATE_ASYNC_LOAD`` value, or ``_UNSET`` if it was absent.
# Restored when the last nested block exits. Transformers reads the variable
# once per load, at the start of ``convert_and_load_state_dict_in_model``.
_UNSET = object()
_ASYNC_LOAD_ENV = "HF_DEACTIVATE_ASYNC_LOAD"
_async_load_env_saved = _UNSET

# Bytes of host tensors handed to the upload threads and not yet swapped in,
# above which the funnel waits for one to finish before taking the next. Enough
# to keep every upload thread busy on an 8B model's largest projections (about
# 100 MiB each in fp16), and small next to the model itself.
_DEFAULT_MAX_IN_FLIGHT_BYTES = 1 << 30

# The installed wrapper is shared, but its *settings* are not: each thread
# loading a model has its own device and dtype (rank-local under torchrun).
# Keeping them in a thread-local stack means a concurrent load cannot inherit
# another thread's rank, and a thread that is not inside a direct-load block
# sees an empty stack and is passed straight through to Transformers.
_active = threading.local()


class _InFlight(NamedTuple):
    """One upload running on the block's pool, and where its result goes."""

    future: Future
    module: nn.Module
    name: str
    installed: torch.Tensor  # what ``set_param_for_module`` put in the module
    nbytes: int


@dataclasses.dataclass
class _Block:
    """One direct-load block: its settings and the uploads still in flight."""

    device: torch.device
    dtype: torch.dtype
    host_prefixes: tuple[str, ...]
    host_filter: Callable | None
    pool: ThreadPoolExecutor | None
    max_in_flight_bytes: int
    pending: list[_InFlight] = dataclasses.field(default_factory=list)
    in_flight_bytes: int = 0
    #: Tensors this block transferred (or handed to an upload thread). Zero
    #: after a load means the funnel was never reached.
    placed: int = 0


def _active_block() -> _Block | None:
    """The direct-load block this thread is inside, if any."""
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


def _host_filter(adapter_module):
    """The adapter's ``spyre_direct_stays_on_host(model, target_name)``, if any.

    For host residency that depends on the config, which a fixed
    ``SPYRE_DIRECT_HOST_MODULES`` cannot express.
    """
    return getattr(adapter_module, "spyre_direct_stays_on_host", None)


def _stays_on_host(target_name: str, host_prefixes: Sequence[str]) -> bool:
    """True when ``target_name`` belongs to a module kept on the host.

    Matches on dotted-path boundaries so ``"cls"`` does not also claim
    ``"classifier.weight"``.
    """
    for prefix in host_prefixes:
        if target_name == prefix or target_name.startswith(f"{prefix}."):
            return True
    return False


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
    weights land, so while this funnel is live the tied parameter is not yet
    the same object as its source. The two conditions below are what can be
    seen at placement time:

    1. the model declares the pair in ``_tied_weights_keys`` -- which it does
       whether or not tying is switched on, so this alone is not enough;
    2. ``tie_word_embeddings`` is set on the (text) config.

    Whether the tied parameter has been filled from the checkpoint does not
    matter. A checkpoint may carry its own ``lm_head.weight`` (``.bin``
    checkpoints store both keys): that key is kept on the host by the prefix
    check, and Transformers still ties the pair afterwards when the two hold
    the same values, so the source has to stay on the host all the same. If
    the values differ Transformers leaves them untied, and holding the source
    back then only means ``model.to`` transfers it later.
    """
    tied_keys = getattr(model, "_tied_weights_keys", None)
    if not isinstance(tied_keys, dict) or not _tying_is_active(model):
        # Some families declare a list of name patterns rather than pairs;
        # there is no source to resolve, so leave those to the prefix check.
        return False
    for tied_name, source_name in tied_keys.items():
        if source_name != target_name:
            continue
        if _stays_on_host(tied_name, host_prefixes):
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


def _should_place(target_name, param_value, hf_quantizer, host_prefixes):
    """Decide whether this wrapper owns the placement of one tensor.

    Declines, leaving Transformers' own behaviour untouched, when:

    * a quantizer owns it, since its packed representation is not ours to
      re-lay-out;
    * the value is not a plain tensor (``DTensor`` and friends);
    * it is not on the host already -- re-placing it would mean a device round
      trip, which for fp16 is also lossy (DLFLOAT16 on the device);
    * the adapter asked for that module to stay on the host.
    """
    if hf_quantizer is not None:
        return False
    if type(param_value) is not torch.Tensor and not isinstance(
        param_value, nn.Parameter
    ):
        return False
    if param_value.device.type != "cpu":
        return False
    return not _stays_on_host(target_name, host_prefixes)


def default_upload_threads(device) -> int:
    """Upload threads for a direct load onto ``device``: as many as ``model.to``.

    That is torch-spyre's count, which follows torch's intra-op threads, so
    ``OMP_NUM_THREADS=1`` uploads one weight at a time. 1 off Spyre, and for a
    Spyre device given without an index, since each thread has to be told its
    device.
    """
    device = torch.device(device)
    if device.type != "spyre" or device.index is None:
        return 1
    from torch_spyre.model_utils import _upload_thread_count

    return int(_upload_thread_count(None))


def _upload_pool(device: torch.device, upload_threads: int):
    """Threads for one block's uploads, or ``None`` to upload on the caller.

    On Spyre each thread is set up the way torch-spyre sets up ``model.to``'s:
    the device and a stream of its own, so the uploads do not queue behind one
    another on one stream.
    """
    if upload_threads <= 1:
        return None
    if device.type != "spyre":
        return ThreadPoolExecutor(
            max_workers=upload_threads, thread_name_prefix="spyre-direct-upload"
        )
    if device.index is None:
        raise ValueError(f"upload threads need a device index, got {device}")
    from torch_spyre.model_utils import _start_upload_thread

    return ThreadPoolExecutor(
        max_workers=upload_threads,
        thread_name_prefix="spyre-direct-upload",
        initializer=_start_upload_thread,
        initargs=(device.index,),
    )


def _installed_tensor(module, name):
    """The parameter or buffer ``set_param_for_module`` left at ``name``."""
    if name in module._parameters:
        return module._parameters[name]
    return module._buffers.get(name)


def _swap_in(item: _InFlight, device_tensor: torch.Tensor) -> None:
    """Put an uploaded tensor in place of the host one Transformers installed.

    As ``torch_spyre.model_utils._put_upload`` does for ``model.to``, a
    parameter keeps its Python object: whatever already holds it, including a
    tie Transformers makes after the weights load, sees the device tensor, and
    attributes such as ``_is_hf_initialized`` stay on it. Nothing changes if
    the attribute no longer holds the tensor that was installed.
    """
    module, name, installed = item.module, item.name, item.installed
    if module._buffers.get(name) is installed:
        module._buffers[name] = device_tensor
        return
    param = module._parameters.get(name)
    if param is not installed:
        return
    new = nn.Parameter(device_tensor, requires_grad=param.requires_grad)
    try:
        torch._C._swap_tensor_impl(param, new)
    except RuntimeError as e:
        warnings.warn(
            f"direct load: could not keep {name!r} as the same object ({e}); "
            "replacing it, so a holder outside the module keeps the host tensor",
            stacklevel=2,
        )
        new._is_hf_initialized = getattr(  # type: ignore[attr-defined]
            param, "_is_hf_initialized", False
        )
        module._parameters[name] = new


def _drain(block: _Block, limit: int) -> None:
    """Swap in finished uploads, waiting while over ``limit`` bytes are in flight.

    A ``limit`` below zero waits for every upload. An upload that failed raises
    here, on the loading thread.
    """
    while block.pending:
        done = [item for item in block.pending if item.future.done()]
        if not done:
            if block.in_flight_bytes <= limit:
                return
            wait([item.future for item in block.pending], return_when=FIRST_COMPLETED)
            continue
        for item in done:
            block.pending.remove(item)
            block.in_flight_bytes -= item.nbytes
            _swap_in(item, item.future.result())


@contextmanager
def spyre_direct_placement(
    *,
    device,
    dtype,
    host_modules=(),
    host_filter=None,
    upload_threads=1,
    max_in_flight_bytes=_DEFAULT_MAX_IN_FLIGHT_BYTES,
):
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
        host_filter: Optional ``(model, target_name) -> bool``; ``True`` keeps
            that parameter on the host. Consulted after ``host_modules``, for
            residency that depends on the model's config.
        upload_threads: Threads that upload weights while loading continues.
            1 (the default) uploads on the caller, one weight at a time;
            ``default_upload_threads(device)`` matches ``model.to``.
        max_in_flight_bytes: Host bytes handed to the upload threads and not
            yet swapped in, above which loading waits for an upload to finish.
            This bounds the host memory the threads add.

    Only active inside the ``with`` block, so a process that never loads a
    model pays nothing -- in particular ``transformers`` is imported here and
    not at ``import torch`` time.

    Safe to nest and to enter from several threads at once: the installed
    wrapper is shared, but the device, dtype, host modules and upload threads
    it uses belong to one block on one thread, and a thread outside any
    direct-load block is unaffected.

    Every upload is swapped in before the block exits. If the block exits with
    an exception, uploads not yet started are cancelled.

    Yields the block; its ``placed`` is how many tensors it transferred. Zero
    after a load means Transformers never called the funnel, for example
    because a new release installs weights some other way.

    Also sets ``HF_DEACTIVATE_ASYNC_LOAD=1`` for the duration, so Transformers
    does not read every checkpoint tensor into a host copy on worker threads
    before this funnel can transfer them. Restored (including "unset") when
    the last nested block exits.
    """
    global _patch_depth, _original_set_param_for_module, _async_load_env_saved

    from transformers import core_model_loading

    device = torch.device(device)
    block = _Block(
        device=device,
        dtype=dtype,
        host_prefixes=tuple(host_modules),
        host_filter=host_filter,
        pool=_upload_pool(device, upload_threads),
        max_in_flight_bytes=max_in_flight_bytes,
    )

    with _patch_lock:
        if _patch_depth == 0:
            _original_set_param_for_module = core_model_loading.set_param_for_module
            core_model_loading.set_param_for_module = _spyre_set_param_for_module
            previous = os.environ.get(_ASYNC_LOAD_ENV)
            _async_load_env_saved = _UNSET if previous is None else previous
            os.environ[_ASYNC_LOAD_ENV] = "1"
        _patch_depth += 1

    stack = getattr(_active, "stack", None)
    if stack is None:
        stack = _active.stack = []
    stack.append(block)

    try:
        yield block
        _drain(block, -1)
    finally:
        stack.pop()
        if block.pool is not None:
            # Waits for running uploads, which hold their host tensors, and
            # drops the rest; only after a failure is anything left here.
            block.pool.shutdown(wait=True, cancel_futures=True)
        with _patch_lock:
            _patch_depth -= 1
            if _patch_depth == 0:
                core_model_loading.set_param_for_module = _original_set_param_for_module
                saved = _async_load_env_saved
                _async_load_env_saved = _UNSET
                if saved is _UNSET:
                    os.environ.pop(_ASYNC_LOAD_ENV, None)
                else:
                    assert isinstance(saved, str)
                    os.environ[_ASYNC_LOAD_ENV] = saved


def _spyre_set_param_for_module(
    model,
    target_name,
    param_value,
    loading_info,
    hf_quantizer,
    *args,
    **kwargs,
):
    """Transfer the tensor to Spyre and have Transformers install it.

    Transformers keeps every other responsibility: ``set_param_for_module``
    still resolves the module, wraps parameters and records loading info. All
    this does is decide the tensor's device and layout.

    Without upload threads, the tensor is transferred first and the device
    tensor is installed. With them, the host tensor is installed, so that
    bookkeeping runs as it would on the host, and an upload thread transfers
    it while loading continues; ``_drain`` swaps the result in.
    """

    def install(value):
        return _original_set_param_for_module(
            model,
            target_name,
            value,
            loading_info,
            hf_quantizer,
            *args,
            **kwargs,
        )

    block = _active_block()
    if block is None:
        return install(param_value)
    if block.pool is not None:
        _drain(block, block.max_in_flight_bytes)
    if not (
        _should_place(
            target_name,
            param_value,
            hf_quantizer,
            block.host_prefixes,
        )
        and not _tied_to_host(model, target_name, block.host_prefixes)
        and not (
            block.host_filter is not None and block.host_filter(model, target_name)
        )
    ):
        return install(param_value)
    module, param_name = _resolve_target_module(model, target_name)
    if module is None:
        return install(param_value)

    if block.pool is None:
        placed = _placement_copy(
            module, param_name, param_value, device=block.device, dtype=block.dtype
        )
        block.placed += 1
        return install(placed)

    result = install(param_value)
    installed = _installed_tensor(module, param_name)
    if installed is None or installed.device.type != "cpu":
        # Not installed (a shape mismatch leaves the placeholder) or not ours.
        return result
    future = block.pool.submit(
        _placement_copy,
        module,
        param_name,
        param_value,
        device=block.device,
        dtype=block.dtype,
    )
    nbytes = param_value.numel() * param_value.element_size()
    block.pending.append(_InFlight(future, module, param_name, installed, nbytes))
    block.in_flight_bytes += nbytes
    block.placed += 1
    _drain(block, block.max_in_flight_bytes)
    return result


_spyre_set_param_for_module._spyre_direct_placement = True  # type: ignore[attr-defined]
