"""
Automatic module configuration generator using forward hooks.

This script automatically generates YAML configuration for all unique modules
in a model by:
1. Loading the model
2. Registering forward hooks on all modules
3. Running a forward pass to capture module inputs
4. Analyzing captured data to generate YAML config

Usage:
    python auto_generate_module_config.py --model_path ibm-granite/granite-3.3-8b-instruct --seq_len 128
"""

import argparse
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import yaml
from torch.utils._pytree import tree_flatten
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, StaticCache

logger = logging.getLogger(__name__)


# Get existing modules from PyTorch's module_db to avoid duplicates
try:
    from torch.testing._internal.common_modules import module_db

    # Extract just the class name from module_db names (e.g., "nn.Linear" -> "Linear")
    existing_modules = set()
    for m in module_db:
        # module_db names are like "nn.Linear", "nn.Conv2d", etc.
        if "." in m.name:
            class_name = m.name.split(".")[-1]
            existing_modules.add(class_name)
        else:
            existing_modules.add(m.name)
    logger.info(
        f"Found {len(existing_modules)} existing modules in PyTorch's module_db"
    )
except ImportError:
    existing_modules = set()
    logger.warning("could not import module_db, will not filter duplicates")


class PrettyDumper(yaml.SafeDumper):
    """Custom YAML dumper with consistent 2-space indentation."""

    def increase_indent(self, flow=False, indentless=False):
        """Ensure consistent indentation (no indentless sequences)."""
        return super().increase_indent(flow, False)

    def represent_data(self, data):
        """Override to handle shape lists specially."""
        # Check if this is a list that should be inline (shape values)
        if isinstance(data, list) and len(data) > 0:
            # Check if all elements are integers (shape lists are all ints)
            if all(isinstance(x, int) for x in data):
                # This is likely a shape list - use flow style
                return self.represent_sequence(
                    "tag:yaml.org,2002:seq", data, flow_style=True
                )

        # For everything else, use default representation
        return super().represent_data(data)


def _is_special_tensor(name: str) -> bool:
    """Check if tensor name indicates it should not be random."""
    return "position_embeddings" not in name.lower() and any(
        keyword in name.lower() for keyword in ["position", "mask", "ids"]
    )


# Extracted from the loaded config so a standalone module rebuilt from the YAML
# dispatches to the same attention path used at capture time. ``from_pretrained``
# leaves ``config._attn_implementation`` as ``None`` on some models, and a ``None``
# value makes ``AttentionInterface.get_interface`` emit the "standalone Module"
# warning and fall back to eager. Writing the resolved value keeps the generated
# config faithful to the runtime implementation.
DEFAULT_ATTN_IMPLEMENTATION = "sdpa"

# The dtype Spyre actually runs in. ``from_pretrained`` defaults to float32, but
# Spyre executes in bfloat16, so both the capture path (``load_model_only``) and
# the YAML emit path (``_tensor_info_to_spec``) default floating-point tensors to
# bfloat16. This keeps the generated config faithful to the runtime dtype
# regardless of the checkpoint's stored precision. Only floating-point dtypes are
# remapped; integer/bool tensors (ids, masks, positions) keep their own dtype.
DEFAULT_FLOAT_DTYPE = torch.bfloat16
_FLOAT_DTYPE_ALIASES = ("float16", "float32", "float64", "float", "half", "double")

# Special tensors (position/mask/ids -- see ``_is_special_tensor``) carry indices
# rather than activations, so they are forced to this integer dtype regardless of
# the dtype they were captured under. This makes their ``randint`` init consistent
# (randint on a floating-point dtype is meaningless).
DEFAULT_INT_DTYPE = torch.int64


def _resolve_attn_implementation(config: Any) -> str:
    """Return the attention implementation the model actually used.

    Prefers the concrete value set on the loaded config; falls back to
    ``DEFAULT_ATTN_IMPLEMENTATION`` (the ``from_pretrained`` default) when the
    config still reports ``None``, so the generated YAML never carries a null
    that would trigger the standalone-module warning.
    """
    impl = getattr(config, "_attn_implementation", None)
    if impl is None:
        return DEFAULT_ATTN_IMPLEMENTATION
    return impl


def _extract_config_kwargs(config: Any) -> Dict[str, Any]:
    """Extract the config parameters the framework needs to rebuild a module.

    ``_attn_implementation`` is resolved to a concrete implementation (never
    ``None``) so a module reconstructed from the YAML dispatches attention the
    same way it did during capture.
    """
    config_kwargs: Dict[str, Any] = {}
    for attr in [
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "max_position_embeddings",
    ]:
        if hasattr(config, attr):
            config_kwargs[attr] = getattr(config, attr)

    if hasattr(config, "_attn_implementation"):
        config_kwargs["_attn_implementation"] = _resolve_attn_implementation(config)

    return config_kwargs


def _spyre_config_kwargs(config: Any, model: Any) -> Dict[str, Any]:
    """``_extract_config_kwargs`` plus the head_dim the Spyre path actually runs.

    ``pad_attention_heads`` lifts head_dim to a Spyre stick boundary and rewrites
    the Q/K/V/O projections to that padded width, recording the result on
    ``model._spyre_head_dim``. Emitting that value means a module rebuilt from the
    generated YAML gets the PADDED projection shapes -- the ones the adapter
    actually runs -- rather than the checkpoint's unpadded ones.

    Only the shape is reproduced, not the padded weights: padding writes zeros
    into the new positions, and a module test builds fresh weights anyway. A model
    needing no padding (head_dim already >= 2 * BLOCK_SIZE) carries no
    ``_spyre_head_dim``, and its config value is already correct.
    """
    config_kwargs = _extract_config_kwargs(config)
    spyre_head_dim = getattr(model, "_spyre_head_dim", None)
    if spyre_head_dim is not None:
        config_kwargs["head_dim"] = spyre_head_dim
    return config_kwargs


# ---------------------------------------------------------------------------
# Spyre adapter capture (--loader spyre)
# ---------------------------------------------------------------------------

# The device the Spyre execution path runs on. ``hf_common.DEVICE`` is pinned to
# this; the loader below can patch it to "cpu" for an off-pod dry run.
SPYRE_DEVICE = "spyre"

# Spyre-introduced wrapper modules, i.e. the classes ``prepare_for_spyre()`` puts
# into the model tree in place of the HF ones. Named here rather than detected
# structurally because each needs different handling.
SPYRE_ATTENTION_TYPE = "StandardGQAAttention"
SPYRE_BLOCK_TYPE = "StandardGQABlock"

# Module types excluded from a --loader spyre YAML.
#
# StandardGQABlock: deliberately not emitted, to keep the test framework simple.
# Its ctor takes a live HF DecoderLayer *and* an is_res_mul flag, and it owns
# submodules (mlp, both norms, self_attn) that are already captured as standalone
# entries in their own right. Emitting it would need a second module-arg nesting
# level for no extra module-level coverage. The block's own arithmetic (residual
# order, residual_multiplier, norm placement) is covered end-to-end by
# tests/spyre/test_e2e_*.
#
# PrecomputedRotaryEmbedding / InvFreqShim: Spyre-internal RoPE plumbing wrapping
# the HF rotary embedding; the HF module itself is captured instead.
SPYRE_EXCLUDED_MODULE_TYPES = frozenset(
    {
        SPYRE_BLOCK_TYPE,
        "PrecomputedRotaryEmbedding",
        "InvFreqShim",
    }
)


def _ensure_hf_adapters_importable() -> None:
    """Make ``import hf_adapters`` work when run as a script from this directory.

    ``hf_adapters`` is not pip-installed on the Spyre pod (see CLAUDE.md), and
    running this file directly puts ``utils/module_discovery/`` on sys.path rather
    than the repo root. Every ``hf_adapters`` import in this module is lazy (inside
    a function) precisely so the HF loader path never needs this; the Spyre path
    calls it first. A pre-existing installed/PYTHONPATH copy wins -- this only adds
    a fallback.
    """
    import importlib.util
    import sys

    if importlib.util.find_spec("hf_adapters") is not None:
        return
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        logger.info("Adding repo root to sys.path for hf_adapters: %s", repo_root)
        sys.path.insert(0, repo_root)


def _snapshot_hf_attention_classes(model) -> Dict[str, Tuple[str, Any, int]]:
    """Record each decoder layer's HF attention class BEFORE prepare_for_spyre().

    ``StandardGQAAttention.__init__(attn)`` adopts an HF attention's projections
    and keeps none of its provenance -- no ``config``, no ``layer_idx``. Once
    ``prepare_standard_gqa_blocks`` has replaced ``layers[i]``, the original class
    is unrecoverable from the live tree, so the Spyre loader path must snapshot it
    while the plain HF model is still intact.

    Returns ``{layer_path: (attn_class_path, config, layer_idx)}`` keyed by the
    decoder layer's ``named_modules`` path, which survives the replacement (only
    the object at that path changes).
    """
    from hf_adapters.hf_common import get_backbone

    snapshot: Dict[str, Tuple[str, Any, int]] = {}
    backbone = get_backbone(model)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        logger.warning(
            "Model has no backbone .layers; no HF attention classes recorded. "
            "Spyre attention entries will not be rebuildable."
        )
        return snapshot

    # Resolve the backbone's own path so keys match named_modules() on `model`.
    backbone_path = ""
    for name, mod in model.named_modules():
        if mod is backbone:
            backbone_path = name
            break

    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        cls = type(attn)
        config = getattr(attn, "config", None)
        layer_idx = getattr(attn, "layer_idx", None)
        prefix = f"{backbone_path}.layers.{i}" if backbone_path else f"layers.{i}"
        snapshot[prefix] = (
            f"{cls.__module__}.{cls.__name__}",
            config,
            i if layer_idx is None else layer_idx,
        )
    return snapshot


def _extract_tensor_info(tensor: torch.Tensor, name: str) -> Dict[str, Any]:
    """Extract information from a single tensor."""
    return {
        "type": "tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "is_random": not _is_special_tensor(name),
        "requires_grad": tensor.requires_grad,
    }


# Scalar types recorded verbatim as a forward arg. A module's forward may take
# plain Python scalars alongside tensors -- e.g. the Spyre attention block's
# ``is_filling`` (bool) / ``token_index`` (int) / ``cache_position`` (int).
# ``bool`` precedes ``int`` only for clarity; ``isinstance`` covers both since
# bool subclasses int.
_SCALAR_ARG_TYPES = (bool, int, float, str)


def _extract_scalar_info(value: Any, name: str) -> Dict[str, Any] | None:
    """Describe a non-tensor scalar forward arg, or ``None`` if not a scalar.

    Tensor-free args carry no pytree leaves, so :func:`_process_pytree_structure`
    returns ``None`` for them. Dropping a *positional* arg is not merely a missing
    input: it shifts every later arg one slot left, so the replayed forward is
    called with the wrong arity and the wrong values. Recording the scalar keeps
    the positional sequence intact.

    ``None`` is recorded too (a genuine "pass None here" argument). It is checked
    before the scalar types because ``None`` is not an instance of any of them.
    """
    if value is None:
        return {"name": name, "type": "value", "value": None}
    if isinstance(value, _SCALAR_ARG_TYPES):
        # bool before int in the emitted type name for readability only; the YAML
        # carries the value itself, which round-trips with its Python type.
        return {"name": name, "type": "value", "value": value}
    return None


def _process_pytree_structure(value: Any, name: str) -> Dict[str, Any] | None:
    """
    Process a pytree structure (nested tensors/lists/tuples/dicts) and extract info.

    Uses PyTorch's tree_flatten to handle arbitrary nesting uniformly.
    """
    # Check if this is a tensor or contains tensors
    if isinstance(value, torch.Tensor):
        # Single tensor - simple case
        return {"name": name, **_extract_tensor_info(value, name)}

    # Use tree_flatten to extract all tensor leaves regardless of nesting.
    # We intentionally do not reconstruct the original structure since only
    # tensor metadata is needed for config generation.
    flat_values, _ = tree_flatten(value)

    # Extract info from all tensors in the flattened structure
    # Single source of truth: pytree handles all container types uniformly
    tensor_infos = []
    for item in flat_values:
        if isinstance(item, torch.Tensor):
            tensor_infos.append(_extract_tensor_info(item, name))

    # Post-process: enrich dict tensors with their keys
    if isinstance(value, dict) and tensor_infos:
        dict_keys = [k for k, v in value.items() if isinstance(v, torch.Tensor)]
        for i, key in enumerate(dict_keys):
            if i < len(tensor_infos):
                tensor_infos[i]["dict_key"] = key

    # If we found tensors, return with structure info
    if tensor_infos:
        # Determine container type from the original value
        if isinstance(value, tuple):
            container_type = "tuple"
        elif isinstance(value, list):
            container_type = "list"
        elif isinstance(value, dict):
            container_type = "dict"
        else:
            container_type = "pytree"

        return {
            "name": name,
            "type": container_type,
            "items": tensor_infos,
        }

    return None


def _resolve_layer_idx(module: Any) -> int | None:
    """Find a module's decoder layer index for indexing into the KV cache.

    In transformers >=5 the ``layer_idx`` lives on the attention submodule, not
    on the DecoderLayer itself, so we check the layer first (older layouts /
    other archs) and fall back to ``self_attn.layer_idx``. Returns ``None`` when
    neither exists (e.g. a norm/MLP module that never touches the KV cache).
    """
    layer_idx = getattr(module, "layer_idx", None)
    if layer_idx is not None:
        return layer_idx
    self_attn = getattr(module, "self_attn", None)
    if self_attn is not None:
        return getattr(self_attn, "layer_idx", None)
    return None


def _class_source_location(cls: type) -> Tuple[Optional[str], Optional[int]]:
    """Return (source file, first line of the class definition) for ``cls``.

    Resolved from the live class object while the hook still holds the module
    instance. ``module_path`` alone is not enough: a model loaded with
    ``trust_remote_code`` lives in a dynamically created module that cannot be
    re-imported by name later. Returns ``(None, None)`` when no source is
    retrievable (C extension, class synthesized at runtime).
    """
    try:
        source_file = inspect.getsourcefile(cls)
        _, lineno = inspect.getsourcelines(cls)
    except (OSError, TypeError):
        return None, None
    return source_file, lineno


def _shorten_source_path(path: str) -> str:
    """Trim an absolute source path down to something environment-independent.

    A ``site-packages``/``dist-packages`` install becomes ``<pkg>/...`` so the
    generated YAML does not hard-code the generating machine's venv layout.
    Paths outside a site install are returned unchanged.
    """
    parts = Path(path).parts
    for marker in ("site-packages", "dist-packages"):
        if marker in parts:
            return str(Path(*parts[parts.index(marker) + 1 :]))
    return path


def _get_transformers_ref() -> str:
    """Git ref used in generated transformers source URLs.

    Mirrors ``utils/model_ops/utils/torchop_yaml.py``: ``TRANSFORMERS_VERSION``
    overrides, otherwise the installed version becomes a ``vX.Y.Z`` release tag.
    A dev/editable install ("5.0.0.dev0") has no such tag, so it falls back to
    ``main`` rather than emitting a dead link.
    """
    version = os.getenv("TRANSFORMERS_VERSION")
    if version:
        return version
    try:
        import transformers
    except ImportError:
        return "main"
    return (
        "main" if "dev" in transformers.__version__ else f"v{transformers.__version__}"
    )


_TRANSFORMERS_BLOB_URL = "https://github.com/huggingface/transformers/blob"


def _source_reference(
    source_file: Optional[str], lineno: Optional[int]
) -> Optional[str]:
    """Render a captured source location as a human-followable reference.

    A file inside an installed ``transformers`` package becomes a GitHub blob
    URL pinned to the installed version, matching the scheme
    ``torchop_yaml._convert_transformers_path_to_url`` uses. Any other package
    (torch, vLLM, a trust_remote_code module) degrades to a venv-relative
    ``path:line``, since there is no single upstream repo to point at.
    """
    if not source_file:
        return None
    rel = _shorten_source_path(source_file)
    # rel differing from the input means the file came from a site install, so
    # a leading "transformers/" component is the installed transformers package
    # (and maps onto src/transformers/... in the upstream repo layout).
    if rel != source_file and rel.startswith("transformers/"):
        anchor = f"#L{lineno}" if lineno else ""
        return f"{_TRANSFORMERS_BLOB_URL}/{_get_transformers_ref()}/src/{rel}{anchor}"
    return f"{rel}:{lineno}" if lineno else rel


def _extract_cache_info(
    past_key_values: Any, name: str, layer_idx: int, config: Any = None
) -> Dict[str, Any] | None:
    """Snapshot one layer's populated K/V from a ``Cache`` for a decode step.

    A DecoderLayer receives ``past_key_values`` as a live
    :class:`~transformers.cache_utils.Cache`, not raw tensors. During prefill
    the layer's slot is empty (``keys is None``), so there is nothing to record
    and this returns ``None`` — the layer then runs its
    ``if past_key_values is not None: past_key_values.update(...)`` branch on
    freshly computed K/V only, which is the correct prefill behaviour. During
    decode the slot already holds ``past_len`` tokens; we snapshot that layer's
    ``keys``/``values`` so the module test can rebuild an equivalent Cache and
    exercise the same "attend over past + new token" path. Without this, the
    decode invocation would replay with ``past_key_values=None`` and silently
    degrade to a 1-token self-attention (the ``update`` branch never runs).

    Transformers >=5 stores per-layer K/V under ``.layers[i].keys/.values``
    (the older flat ``key_cache``/``value_cache`` lists are gone).

    Only :class:`~transformers.cache_utils.StaticCache` is recorded. A
    fixed-size StaticCache has a fully specified per-layer K/V shape that the
    test side can reconstruct deterministically; a growable ``DynamicCache``
    (the default when no cache is passed in) has no such fixed shape, so we warn
    and skip it rather than emit a cache the test cannot faithfully rebuild.
    Drive the generator with an explicit StaticCache to capture decode state.

    Args:
        past_key_values: The live cache passed to the DecoderLayer.
        name: The kwarg name (``"past_key_values"``).
        layer_idx: The layer whose K/V slot to snapshot.
        config: The model config the cache was built from. In transformers >=5
            a StaticCache no longer exposes ``.config``, but its ``__init__``
            requires one, so the test side needs ``config_path`` +
            ``config_kwargs`` to reconstruct it. Pass the DecoderLayer's config
            (e.g. ``module.self_attn.config``).

    Returns:
        A cache spec dict, or ``None`` when the cache is not a StaticCache, this
        layer's slot is empty (prefill), or it exposes no usable per-layer K/V.
    """
    cache_cls = type(past_key_values)
    if cache_cls.__name__ != "StaticCache":
        logger.warning(
            "past_key_values is a %s, not StaticCache; skipping cache capture. "
            "Drive the generator with an explicit StaticCache to record the "
            "decode KV state (see generate_gpt_oss_20b_config.py).",
            cache_cls.__name__,
        )
        return None

    layers = getattr(past_key_values, "layers", None)
    if layers is None or layer_idx >= len(layers):
        return None

    layer_cache = layers[layer_idx]
    keys = getattr(layer_cache, "keys", None)
    values = getattr(layer_cache, "values", None)

    # Empty slot -> prefill call; nothing populated to record.
    if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
        return None

    # StaticCache allocates the full max_cache_len up front, so keys/values are
    # [B, num_kv_heads, max_cache_len, head_dim] with only the first past_len
    # positions populated. Record just that populated slice so the shape means
    # "the real past" and the test side can prime a cache by a single update()
    # of past_len tokens (not the whole fixed allocation of mostly-zeros).
    #
    # Use the PER-LAYER length, not past_key_values.get_seq_length(): at a
    # decode step the whole-cache length reflects layers already updated this
    # pass, so for layer i>0 (whose slot hasn't been updated yet at pre-hook
    # time) it reads one token too long. layer_cache.get_seq_length() reports
    # just this layer's populated past, which is the same across layers.
    try:
        past_len = int(layer_cache.get_seq_length())
    except Exception:
        try:
            past_len = int(past_key_values.get_seq_length())
        except Exception:
            past_len = keys.shape[-2]
    if past_len <= 0:
        return None
    keys = keys[:, :, :past_len, :]
    values = values[:, :, :past_len, :]

    cache_info: Dict[str, Any] = {
        "name": name,
        "type": "cache",
        "cache_path": f"{cache_cls.__module__}.{cache_cls.__name__}",
        "layer_idx": layer_idx,
        # StaticCache.__init__ needs max_cache_len (the fixed allocation), which
        # is not derivable from the (sliced) K/V shape. Record it so the test
        # side rebuilds a cache of the same allocation before priming it.
        "max_cache_len": getattr(past_key_values, "max_cache_len", None),
        # keys/values carry real past tokens; the test rebuilds a cache of the
        # same seq length via update(). "key"/"value" are not special tensor
        # names, so they default to random init (see _is_special_tensor).
        "key": _extract_tensor_info(keys, f"{name}_key"),
        "value": _extract_tensor_info(values, f"{name}_value"),
    }

    # Snapshot the config so the test side can construct the concrete Cache with
    # matching dimensions (num_kv_heads, head_dim, ...). transformers >=5 no
    # longer exposes StaticCache.config, so we take the config passed in from
    # the DecoderLayer; fall back to the cache's own attribute for older builds.
    if config is None:
        config = getattr(past_key_values, "config", None)
    if config is not None:
        config_cls = type(config)
        config_kwargs = {}
        for attr in [
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_hidden_layers",
            "max_position_embeddings",
        ]:
            if hasattr(config, attr):
                config_kwargs[attr] = getattr(config, attr)
        # The module test rebuilds only a single decoder layer, so pin
        # num_hidden_layers to 1: the real model depth would size a KV
        # cache / layer stack the standalone module never populates.
        config_kwargs["num_hidden_layers"] = 1
        cache_info["config_path"] = f"{config_cls.__module__}.{config_cls.__name__}"
        cache_info["config_kwargs"] = config_kwargs

    return cache_info


class ModuleInfoCapture:
    """Captures module information during forward pass using hooks."""

    def __init__(self, spyre_attn_snapshot=None, spyre_model=None):
        self.module_data: Dict[str, Dict[str, Any]] = {}
        self.seen_module_configs: Set[str] = (
            set()
        )  # Track unique configs, not just types
        # Track model-level context (KV cache, execution mode)
        self.current_model_context: Dict[str, Any] = {}
        # Set only on the --loader spyre path: maps a decoder layer path to the HF
        # attention class/config it held before prepare_for_spyre() replaced it
        # (see _snapshot_hf_attention_classes). Left empty on the HF path, which
        # keeps capture_constructor_info's behaviour bit-identical there.
        self.spyre_attn_snapshot = spyre_attn_snapshot or {}
        # The prepared model, read for _spyre_head_dim (the padded head_dim the
        # adapter actually runs). None on the HF path.
        self.spyre_model = spyre_model

    def _resolve_spyre_attn_source(self, module_name: str):
        """Look up the pre-Spyre HF attention for a wrapper's ``named_modules`` path.

        ``module_name`` is the wrapper's path (e.g. ``model.layers.7.self_attn``)
        while the snapshot is keyed by the decoder layer's path
        (``model.layers.7``), so match on the longest recorded prefix rather than
        assuming a fixed suffix -- the attribute the wrapper sits under differs
        across adapters.
        """
        best = None
        for layer_path, entry in self.spyre_attn_snapshot.items():
            if module_name == layer_path or module_name.startswith(layer_path + "."):
                if best is None or len(layer_path) > len(best[0]):
                    best = (layer_path, entry)
        return None if best is None else best[1]

    def capture_constructor_info(
        self, module, module_name: str, module_type: str
    ) -> Dict[str, Any]:
        """
        Capture constructor information from an instantiated module.

        This inspects the module to infer what constructor args were used.
        For Transformers modules, we look for config objects and layer_idx.
        """
        constructor_args = []
        constructor_kwargs = {}

        # A Spyre attention wrapper adopts an HF attention's projections and keeps
        # no config/layer_idx of its own, so none of the heuristics below can
        # describe it (they would fall through to the 1-D-weight branch and emit a
        # wrong ctor). Emit a *module* arg instead: the test side rebuilds the
        # inner HF attention from its config, then wraps it -- exactly what
        # prepare_for_spyre did. Needs the OOT framework's InputArgModule.
        if module_type == SPYRE_ATTENTION_TYPE:
            entry = self._resolve_spyre_attn_source(module_name)
            if entry is not None:
                attn_path, attn_config, layer_idx = entry
                constructor_args.append(
                    {
                        "type": "module",
                        "module_path": attn_path,
                        "config_path": (
                            f"{type(attn_config).__module__}."
                            f"{type(attn_config).__name__}"
                        ),
                        # head_dim reflects _spyre_head_dim when the adapter padded
                        # it, so the rebuilt projections get the padded widths the
                        # adapter actually runs.
                        "config_kwargs": _spyre_config_kwargs(
                            attn_config, self.spyre_model
                        ),
                        "module_kwargs": {"layer_idx": layer_idx},
                    }
                )
                return {
                    "constructor_args": constructor_args,
                    "constructor_kwargs": constructor_kwargs,
                }
            logger.warning(
                "%s at %s: no pre-Spyre HF attention recorded; emitting no "
                "constructor args (this entry will not be rebuildable).",
                module_type,
                module_name,
            )

        # Special handling for decoder layers that don't expose config attribute
        # but require it as constructor arg (e.g., GraniteDecoderLayer)
        if "decoder" in module_type.lower() and "layer" in module_type.lower():
            # Try to get config from parent model or infer from module structure
            # For now, we'll look for self_attn or mlp submodules that might have config
            if hasattr(module, "self_attn") and hasattr(module.self_attn, "config"):
                config = module.self_attn.config
            elif hasattr(module, "mlp") and hasattr(module.mlp, "config"):
                config = module.mlp.config
            else:
                config = None

            if config is not None:
                config_class = type(config).__name__
                config_module = type(config).__module__

                # Extract key config parameters
                config_kwargs = _extract_config_kwargs(config)

                constructor_args.append(
                    {
                        "type": "config",
                        "config_path": f"{config_module}.{config_class}",
                        "config_kwargs": config_kwargs,
                    }
                )

                # Decoder layers typically need layer_idx as kwarg
                # Always add it for decoder layers, even if not found as attribute
                layer_idx_value = 0  # Default to 0
                if hasattr(module, "layer_idx") and module.layer_idx is not None:
                    layer_idx_value = module.layer_idx
                constructor_kwargs["layer_idx"] = {
                    "type": "int",
                    "value": layer_idx_value,
                }
        # Check if module has a config attribute (common in Transformers)
        elif hasattr(module, "config"):
            config = module.config
            config_class = type(config).__name__
            config_module = type(config).__module__

            # Extract key config parameters
            config_kwargs = _extract_config_kwargs(config)

            constructor_args.append(
                {
                    "type": "config",
                    "config_path": f"{config_module}.{config_class}",
                    "config_kwargs": config_kwargs,
                }
            )

            # Check for layer_idx (common in decoder layers with config)
            # Note: layer_idx can be 0, so check for attribute existence, not truthiness
            if hasattr(module, "layer_idx"):
                layer_idx_value = (
                    module.layer_idx if module.layer_idx is not None else 0
                )
                constructor_kwargs["layer_idx"] = {
                    "type": "int",
                    "value": layer_idx_value,
                }
        else:
            # No config - check for direct constructor parameters
            # RMSNorm: hidden_size or dim
            if hasattr(module, "weight") and hasattr(module.weight, "shape"):
                # Normalization layers typically have weight with shape (hidden_size,)
                if len(module.weight.shape) == 1:
                    hidden_size = module.weight.shape[0]
                    constructor_args.append({"type": "int", "value": hidden_size})
            elif hasattr(module, "normalized_shape"):
                # LayerNorm-style
                if isinstance(module.normalized_shape, tuple):
                    hidden_size = module.normalized_shape[0]
                else:
                    hidden_size = module.normalized_shape
                constructor_args.append({"type": "int", "value": hidden_size})

        return {
            "constructor_args": constructor_args,
            "constructor_kwargs": constructor_kwargs,
        }

    def create_model_hook(self):
        """Create a model-level hook to detect execution mode (prefill vs decode)..

        This hook runs BEFORE module-level hooks and sets context that module hooks can use.
        """

        def model_hook(model, args, kwargs):
            # Capture model-level context
            past_key_values = kwargs.get("past_key_values", None)
            attention_mask = kwargs.get("attention_mask", None)

            # Detect execution mode. A pre-allocated but empty cache (e.g. a
            # freshly constructed StaticCache passed into prefill) is not None,
            # so fall back to its sequence length to distinguish prefill from
            # decode.
            if past_key_values is None:
                mode = "prefill"
            elif (
                hasattr(past_key_values, "get_seq_length")
                and past_key_values.get_seq_length() == 0
            ):
                mode = "prefill"
            else:
                mode = "decode"

            # Store context for module hooks to access
            self.current_model_context = {
                "mode": mode,
                "attention_mask": attention_mask,
            }

        return model_hook

    def create_hook(self, module_name: str, module_type: str, module_instance):
        """Create a forward hook that captures module input information.

        This hook captures unique invocations of the module, deduplicating by input pattern.
        This allows testing with multiple input configurations (e.g., prefill + decode)
        without storing redundant identical invocations.
        """

        def hook(module, args, kwargs):
            # Capture constructor information to create unique config identifier
            constructor_info = self.capture_constructor_info(
                module, module_name, module_type
            )

            # Create a unique identifier based on module type + constructor args
            # This allows us to capture multiple variants of the same module type
            config_signature = self._create_config_signature(
                module_type, constructor_info
            )

            # Create unique module name for this variant
            unique_module_name = self._create_unique_module_name(
                module_type, constructor_info, config_signature
            )

            # Initialize module_info if this is the first invocation
            if unique_module_name not in self.module_data:
                self.seen_module_configs.add(config_signature)

                cls = module.__class__
                source_file, source_lineno = _class_source_location(cls)
                self.module_data[unique_module_name] = {
                    "name": unique_module_name,
                    "module_type": module_type,
                    "module_path": f"{cls.__module__}.{cls.__name__}",
                    "source_file": source_file,
                    "source_lineno": source_lineno,
                    "example_instance": module_name,
                    "constructor_args": constructor_info["constructor_args"],
                    "constructor_kwargs": constructor_info["constructor_kwargs"],
                    "invocations": [],  # List of unique invocations
                    "invocation_signatures": set(),  # Track seen invocation patterns
                }

            # Capture this invocation's inputs
            invocation_inputs = []

            # Analyze positional arguments using pytree.
            #
            # A positional arg MUST always yield an entry: these are replayed in
            # order, so silently skipping one (a tensor-free scalar, for which
            # _process_pytree_structure returns None) shifts every later arg into
            # the wrong slot and the rebuilt forward is called with the wrong
            # arity. Scalars are recorded as plain values; anything genuinely
            # undescribable is logged rather than dropped in silence.
            for i, arg in enumerate(args):
                input_info = _process_pytree_structure(arg, f"arg_{i}")
                if input_info is None:
                    input_info = _extract_scalar_info(arg, f"arg_{i}")
                if input_info is None:
                    logger.warning(
                        "%s: positional arg_%d of type %s is neither tensor-bearing "
                        "nor a scalar; the generated entry will be missing it and "
                        "every later positional arg will shift one slot left.",
                        module_type,
                        i,
                        type(arg).__name__,
                    )
                    continue
                invocation_inputs.append(input_info)

            # Analyze keyword arguments using pytree
            for key, value in kwargs.items():
                if key in ("past_key_values", "past_key_value"):
                    # A live Cache object can't go through the tensor-spec
                    # pytree path. For a decode step we snapshot this layer's
                    # populated K/V so the module test can rebuild an equivalent
                    # cache and drive the real "attend over past + new token"
                    # path; for prefill the slot is empty and this records
                    # nothing (equivalent to past_key_values=None).
                    layer_idx = _resolve_layer_idx(module)
                    if layer_idx is not None and value is not None:
                        layer_config = getattr(
                            getattr(module, "self_attn", None), "config", None
                        ) or getattr(module, "config", None)
                        cache_info = _extract_cache_info(
                            value, "past_key_values", layer_idx, config=layer_config
                        )
                        if cache_info is not None:
                            invocation_inputs.append(cache_info)
                    continue
                input_info = _process_pytree_structure(value, key)
                if input_info is None:
                    # Scalar kwargs matter for the same reason as scalar
                    # positionals (a forward may branch on them), though omitting
                    # one here only loses that kwarg -- it cannot misalign others.
                    input_info = _extract_scalar_info(value, key)
                if input_info:
                    invocation_inputs.append(input_info)

            # Create signature for this invocation to detect duplicates
            invocation_sig = self._create_invocation_signature(invocation_inputs)

            # Only add if this is a new unique invocation pattern
            if (
                invocation_sig
                not in self.module_data[unique_module_name]["invocation_signatures"]
            ):
                self.module_data[unique_module_name]["invocation_signatures"].add(
                    invocation_sig
                )
                self.module_data[unique_module_name]["invocations"].append(
                    invocation_inputs
                )

        return hook

    def _create_config_signature(
        self, module_type: str, constructor_info: Dict[str, Any]
    ) -> str:
        """Create a unique signature for a module configuration.

        This signature is used to detect duplicate configurations.
        layer_idx is EXCLUDED because we only need one representative layer.
        """
        # Build signature from constructor args
        sig_parts = [module_type]

        for arg in constructor_info.get("constructor_args", []):
            if arg["type"] == "int":
                sig_parts.append(f"int_{arg['value']}")
            elif arg["type"] == "config":
                sig_parts.append(f"config_{arg['config_path']}")
            else:
                sig_parts.append(f"{arg['type']}")

        # IMPORTANT: Exclude layer_idx from signature
        # We only need one representative layer, not all 40 decoder layers
        for key, kwarg in constructor_info.get("constructor_kwargs", {}).items():
            if key == "layer_idx":
                continue  # Skip layer_idx - treat all layers as same config
            if kwarg["type"] == "int":
                sig_parts.append(f"{key}_{kwarg['value']}")

        return "__".join(sig_parts)

    def _create_unique_module_name(
        self, module_type: str, constructor_info: Dict[str, Any], config_signature: str
    ) -> str:
        """Create a unique, human-readable name for a module variant.

        Names are based on the config signature (which excludes layer_idx),
        ensuring that modules with identical configs get the same name and
        their invocations are grouped together.

        Examples:
            MyRMSNorm with dim=4096 -> MyRMSNorm_4096
            MyRMSNorm with dim=2048 -> MyRMSNorm_2048
            GraniteDecoderLayer (all layers same config) -> GraniteDecoderLayer_layer0
        """
        # Check if there's a simple int arg (common for norm layers)
        args = constructor_info.get("constructor_args", [])
        if len(args) == 1 and args[0]["type"] == "int":
            return f"{module_type}_{args[0]['value']}"

        # For modules with layer_idx, use "layer0" as representative name
        # since all layers have the same config (layer_idx excluded from signature)
        kwargs = constructor_info.get("constructor_kwargs", {})
        if "layer_idx" in kwargs:
            # Use layer0 as the canonical name for all layers
            return f"{module_type}_layer0"

        # If no simple identifier, use a hash of the config signature
        # This ensures uniqueness while keeping names readable
        sig_hash = hashlib.sha256(config_signature.encode()).hexdigest()[:8]
        return f"{module_type}_{sig_hash}"

    def _create_invocation_signature(
        self, invocation_inputs: List[Dict[str, Any]]
    ) -> str:
        """Create a signature for an invocation based on input patterns.

        This signature captures the structure of inputs (shapes, dtypes, types)
        but not the actual values, allowing us to deduplicate identical invocations.

        Args:
            invocation_inputs: List of input info dicts from _process_pytree_structure

        Returns:
            A string signature representing this invocation pattern
        """

        def _extract_pattern(input_info: Dict[str, Any]) -> Dict[str, Any]:
            """Extract the pattern from an input, removing variable data.

            input_info structure from _process_pytree_structure:
            - Single tensor: {"name": "arg_0", "shape": [...], "dtype": ..., ...}
            - Container: {"name": "arg_0", "type": "list/tuple/dict/pytree", "items": [...]}
            """
            # A scalar: include the VALUE, not just the type. Prefill and decode
            # can differ only in these flags (is_filling / token_index /
            # cache_position on the Spyre attention path, where the tensor shapes
            # are otherwise identical), so ignoring the value would collapse the
            # two invocations into one and lose the decode pattern entirely.
            if input_info.get("type") == "value":
                return {"type": "value", "value": input_info.get("value")}
            # A KV cache: distinct pattern so prefill (no cache) and decode
            # (cache present) never collapse into one invocation signature.
            if input_info.get("type") == "cache":
                return {
                    "type": "cache",
                    "cache_path": input_info.get("cache_path"),
                    "key_shape": input_info.get("key", {}).get("shape"),
                    "value_shape": input_info.get("value", {}).get("shape"),
                }
            # Check if this is a container with items
            if "type" in input_info and "items" in input_info:
                # Container (list, tuple, dict, pytree)
                pattern = {
                    "type": input_info["type"],
                    "items": [
                        {
                            "shape": item.get("shape"),
                            "dtype": str(item.get("dtype")),
                            "init": item.get("init"),
                        }
                        for item in input_info["items"]
                    ],
                }
                return pattern
            elif "shape" in input_info:
                # Single tensor
                return {
                    "type": "tensor",
                    "shape": input_info.get("shape"),
                    "dtype": str(input_info.get("dtype")),
                    "init": input_info.get("init"),
                }
            else:
                # Unknown structure
                return {"type": "unknown"}

        # Build pattern for all inputs
        patterns = []
        for input_info in invocation_inputs:
            # input_info is already a dict with structure like:
            # {"name": "arg_0", "tensor": {...}} or {"name": "x", "type": "list", "items": [...]}
            # We want to extract the pattern from the whole input_info
            patterns.append(_extract_pattern(input_info))

        # Convert to JSON for consistent string representation
        pattern_str = json.dumps(patterns, sort_keys=True)
        return hashlib.sha256(pattern_str.encode()).hexdigest()

    def get_captured_modules(self) -> List[Dict[str, Any]]:
        """Return list of captured module information."""
        # Remove invocation_signatures before returning (internal tracking only)
        result = []
        for module_data in self.module_data.values():
            module_copy = module_data.copy()
            module_copy.pop("invocation_signatures", None)
            result.append(module_copy)
        return result


def get_all_custom_modules(
    model, excluded_types: frozenset = frozenset()
) -> List[Tuple[str, str, Any]]:
    """
    Get ALL custom module instances from the model (not just unique types).

    Args:
        model: The model to walk.
        excluded_types: Class names to skip entirely. Used by the Spyre loader
            path to drop wrapper modules not worth emitting as test entries (see
            SPYRE_EXCLUDED_MODULE_TYPES). Empty by default, so the HF path is
            unchanged. Excluding a container does NOT exclude its children: they
            appear in named_modules() in their own right and are still captured.

    Returns:
        List of (module_name, module_type, module_instance) tuples
    """
    custom_modules = []
    for name, module in model.named_modules():
        if name == "":  # Skip root
            continue

        module_type = type(module).__name__

        if module_type in excluded_types:
            continue

        # Skip if already in upstream module_db
        if module_type in existing_modules:
            continue

        # Keep ALL instances (not just first of each type)
        custom_modules.append((name, module_type, module))

    return custom_modules


def _convert_constructor_arg_to_sample_input(
    arg_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert constructor arg spec to sample_inputs_func format."""
    if arg_spec["type"] == "config":
        # Emit a structured config arg carrying the captured model dimensions so
        # the framework can rebuild the config (config_path + config_kwargs) with
        # the right shapes instead of library defaults. Resolved by
        # InputsEdits.build_cpu_args -> InputArgConfig in the OOT framework.
        return {
            "config_path": arg_spec["config_path"],
            "config_kwargs": arg_spec.get("config_kwargs", {}),
        }
    elif arg_spec["type"] == "module":
        # Resolved by the OOT framework's InputArgModule -> _build_inner_module:
        # import module_path, build the config, instantiate it, then hand the live
        # module to the outer wrapper's constructor.
        return {
            "module_path": arg_spec["module_path"],
            "config_path": arg_spec["config_path"],
            "config_kwargs": arg_spec.get("config_kwargs", {}),
            "module_kwargs": arg_spec.get("module_kwargs", {}),
        }
    elif arg_spec["type"] == "int":
        return {"value": arg_spec["value"]}
    elif arg_spec["type"] == "float":
        return {"value": arg_spec["value"]}
    elif arg_spec["type"] == "str":
        return {"value": arg_spec["value"]}
    elif arg_spec["type"] == "bool":
        return {"value": arg_spec["value"]}
    else:
        return {"value": None}


def _tensor_info_to_spec(tensor_info: Dict[str, Any], name: str) -> Dict[str, Any]:
    """
    Convert a single tensor info dict to sample_inputs tensor spec format.

    This function can be used with tree_map to transform entire structures.
    """
    dtype = tensor_info["dtype"]
    if not dtype.startswith("torch."):
        dtype = f"torch.{dtype}"

    # Default every floating-point tensor to bfloat16 (the dtype Spyre runs in),
    # regardless of the precision the checkpoint was captured in. A model loaded
    # in float32 would otherwise emit float32 specs; normalizing here guarantees
    # the "default is bfloat16" contract even when the capture path did not (or
    # could not) load the model in bfloat16. Integer/bool tensors are left alone.
    bare_dtype = dtype.replace("torch.", "")
    if bare_dtype in _FLOAT_DTYPE_ALIASES:
        dtype = str(DEFAULT_FLOAT_DTYPE)

    # Determine init strategy based on tensor characteristics
    is_random = tensor_info.get("is_random", True)
    init = "randn" if is_random else "zeros"
    init_args = {}

    # An integer tensor (e.g. token ids for an embedding, position ids, masks)
    # must not use randn -- torch.randn ("normal_kernel_cpu") is float-only and
    # raises NotImplementedError for integer dtypes. Use randint for any integer
    # dtype, and also for the name-based special tensors (position/mask/ids),
    # which may be captured under a generic name like "arg_0".
    is_int_dtype = any(t in dtype for t in ("int", "uint", "long", "short", "bool"))
    if is_int_dtype or _is_special_tensor(name):
        init = "randint"
        # A special tensor (position/mask/ids) holds indices, not activations,
        # so force it to an integer dtype. This keeps the randint init consistent
        # even when the tensor was captured under a floating-point dtype
        # tensor captured as bfloat16): randint on a float
        # dtype is meaningless, so it becomes torch.int64 here.
        if _is_special_tensor(name):
            dtype = str(DEFAULT_INT_DTYPE)
        # Use the smallest dimension of the tensor's own shape as the exclusive
        # upper bound (e.g. shape (64, 32, 128) -> high=32). This keeps generated
        # index/position values in range for that tensor rather than using a
        # fixed, possibly out-of-range constant. Guard against empty shapes and
        # zero/one-sized dims (randint needs high >= 1).
        shape = tensor_info.get("shape") or []
        high = min(shape) if shape else 1
        init_args = {"high": max(int(high), 1)}
    elif init in ("randn", "rand"):
        # Float random tensors use xavier init. xavier is undefined for <2-D
        # shapes (the OOT framework rejects it), so 1-D float tensors fall back
        # to randn.
        shape = tensor_info.get("shape") or []
        init = "xavier" if len(shape) >= 2 else "randn"

    tensor_spec = {
        "shape": tensor_info["shape"],
        "stride": None,  # Let PyTorch compute default stride
        "storage_offset": 0,
        "dtype": dtype,
        "device": "spyre",
        "init": init,
    }

    if init_args:
        tensor_spec["init_args"] = init_args

    return tensor_spec


def _convert_captured_input_to_sample_input(inp_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert captured input spec to sample_inputs_func format.

    Uses pytree utilities to handle single tensors and nested collections uniformly.
    The key insight: pytree lets us treat single tensors and collections the same way.
    """
    inp_name = inp_spec["name"]
    inp_type = inp_spec["type"]

    if inp_type == "value":
        # A plain Python scalar (or None) recorded verbatim. Resolved by the OOT
        # framework's InputArgValue, which passes it through unchanged.
        return {"value": inp_spec["value"]}

    if inp_type == "tensor":
        # Single tensor - wrap in standard format
        return {"tensor": _tensor_info_to_spec(inp_spec, inp_name)}

    elif inp_type in ("tuple", "list", "dict", "pytree"):
        # Collection of tensors - pytree handles all container types uniformly
        # Convert each tensor in the flattened structure
        tensor_list = [
            _tensor_info_to_spec(item, inp_name) for item in inp_spec.get("items", [])
        ]

        return {"tensor_list": tensor_list}

    elif inp_type == "cache":
        # A KV cache: emit cache_path + per-layer key/value tensor specs so the
        # test side can rebuild a concrete Cache and prime it via update(),
        # reproducing the decode path (attend over past + new token).
        cache_spec: Dict[str, Any] = {
            "cache_path": inp_spec["cache_path"],
            "layer_idx": inp_spec["layer_idx"],
            "key": _tensor_info_to_spec(inp_spec["key"], f"{inp_name}_key"),
            "value": _tensor_info_to_spec(inp_spec["value"], f"{inp_name}_value"),
        }
        if inp_spec.get("max_cache_len") is not None:
            cache_spec["max_cache_len"] = inp_spec["max_cache_len"]
        if "config_path" in inp_spec:
            cache_spec["config_path"] = inp_spec["config_path"]
            cache_spec["config_kwargs"] = inp_spec.get("config_kwargs", {})
        return {"cache": cache_spec}

    else:
        return {"value": None}


def _validate_cache_mask_consistency(
    invocation_inputs: List[Dict[str, Any]], module_name: str
) -> None:
    """Warn if a cached (decode) invocation lacks a mask that can cover the past.

    When an invocation carries a KV cache, the test side rebuilds a Cache primed
    with ``past_len`` tokens and drives a decode forward. That forward also needs
    an ``attention_mask`` whose key/value axis is at least ``past_len`` (the
    cache's populated length) — otherwise the mask and the cache disagree about
    how many past tokens exist and the replayed decode attends over the wrong
    span. This is a generation-time sanity check (logged, not fatal) so a
    malformed invocation is visible rather than silently emitted.

    K/V key shape is ``[B, num_kv_heads, head_dim, past_len]`` (past_len last).
    A 4-D ``attention_mask`` is ``[B, 1, q_len, kv_len]`` (kv_len last). We only
    require ``past_len <= kv_len`` since a fixed-length cache (e.g. StaticCache)
    reports its allocation, not its populated length, in the mask.
    """
    cache_spec = None
    mask_spec = None
    for inp in invocation_inputs:
        if inp.get("type") == "cache":
            cache_spec = inp
        elif inp.get("name") == "attention_mask":
            mask_spec = inp

    if cache_spec is None:
        return  # prefill invocation — nothing to check

    if mask_spec is None:
        logger.warning(
            "%s: decode invocation has a KV cache but no attention_mask; "
            "the replayed decode cannot mask the cached past correctly.",
            module_name,
        )
        return

    key_shape = cache_spec.get("key", {}).get("shape")
    mask_shape = mask_spec.get("shape")
    if not key_shape or not mask_shape:
        return
    past_len = key_shape[-1]
    kv_len = mask_shape[-1]
    if kv_len < past_len:
        logger.warning(
            "%s: attention_mask kv_len=%d < cached past_len=%d; mask cannot "
            "cover the cached past for the decode step.",
            module_name,
            kv_len,
            past_len,
        )


def _build_module_entry_dict(module_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a module entry dictionary for YAML generation.

    Args:
        module_info: Captured module information with multiple invocations

    Returns:
        Dictionary representing a module entry for YAML
    """
    # Build constructor_inputs
    constructor_args = []
    constructor_kwargs = {}

    for arg_spec in module_info.get("constructor_args", []):
        constructor_args.append(_convert_constructor_arg_to_sample_input(arg_spec))

    for key, kwarg_spec in module_info.get("constructor_kwargs", {}).items():
        if kwarg_spec["type"] == "int":
            constructor_kwargs[key] = kwarg_spec["value"]

    # Build forward_inputs from all invocations
    # NEW: Handle multiple invocations - each invocation becomes a separate input set
    invocations = module_info.get("invocations", [])

    if not invocations:
        # Fallback for old format (backward compatibility)
        invocations = [module_info.get("inputs", [])]

    # Process each invocation
    forward_inputs_list = []
    for invocation_inputs in invocations:
        forward_args = []
        forward_kwargs = {}

        for inp_spec in invocation_inputs:
            # Validate inp_spec has required fields
            if "name" not in inp_spec:
                logger.error(f"inp_spec missing 'name' field: {inp_spec}")
                continue  # Skip malformed entries

            inp_name = inp_spec["name"]
            converted = _convert_captured_input_to_sample_input(inp_spec)

            if inp_name.startswith("arg_"):
                forward_args.append(converted)
            else:
                forward_kwargs[inp_name] = converted

        # A decode invocation carries a KV cache; verify it also carries an
        # attention_mask whose key/value length can cover the cached past, so
        # the test side rebuilds a self-consistent (mask, cache) pair rather
        # than a decode step that silently attends over the wrong span.
        _validate_cache_mask_consistency(
            invocation_inputs, module_info.get("name", "<unknown>")
        )

        forward_inputs_list.append(
            {
                "args": forward_args if forward_args else [],
                "kwargs": forward_kwargs if forward_kwargs else {},
            }
        )

    forward_inputs = forward_inputs_list

    # Record where the class is defined so a reader of the generated YAML can
    # jump straight to the source. Appended to the free-text description rather
    # than emitted as its own key, so the entry stays within the shape the OOT
    # framework's include schema accepts. Absent for captures that carry no
    # source location (the vLLM generator builds module_info dicts by hand).
    description = f"Module: {module_info['module_path']}"
    location = _source_reference(
        module_info.get("source_file"), module_info.get("source_lineno")
    )
    if location:
        description = f"{description} (defined at {location})"

    # Build module entry
    entry = {
        "name": module_info["name"],
        "module_path": module_info["module_path"],
        "description": description,
        "constructor_inputs": {
            "args": constructor_args if constructor_args else [],
            "kwargs": constructor_kwargs if constructor_kwargs else {},
        },
        "forward_inputs": forward_inputs,
    }

    # Only the Spyre-loader capture path records a device-side parameter layout:
    # its modules were captured from a model already moved to Spyre by
    # move_model_to_spyre. HF-loader entries omit the key entirely so existing
    # configs regenerate byte-identically.
    if module_info.get("apply_device_layout"):
        entry["apply_device_layout"] = True

    return entry


def generate_unified_yaml_config(
    captured_modules: List[Dict[str, Any]], model_name: str
) -> str:
    """Generate unified YAML configuration using yaml.dump().

    This creates a single YAML file with edits.modules.include that contains:
    - Module name and path
    - constructor_inputs: Args/kwargs for module.__init__()
    - forward_inputs: Args/kwargs for module.forward()
    """
    # Build module entries
    module_entries = [_build_module_entry_dict(m) for m in captured_modules]

    # Build the complete configuration dictionary
    config = {
        "test_suite_config": {
            "files": [
                {
                    "path": "${TORCH_ROOT}/test/test_modules.py",
                    "unlisted_test_mode": "skip",
                    "tests": [
                        {
                            "names": ["*TestModule*::test_forward"],
                            "mode": "xfail",
                            "tags": [f"model__{model_name}"],
                            # Spyre's custom ops have no registered autograd
                            # formula, so upstream's test_forward (which builds
                            # modules with ordinary requires_grad=True
                            # parameters) must run under torch.no_grad() to
                            # avoid AOTAutograd tracing a backward graph at
                            # compile time.
                            "no_grad": True,
                            "edits": {"modules": {"include": module_entries}},
                        }
                    ],
                },
                {
                    "path": "${TORCH_DEVICE_ROOT}/tests/test_modules_custom.py",
                    "unlisted_test_mode": "skip",
                    "tests": [
                        {
                            "names": [
                                "*TestModuleCustom*::test_with_cpu",
                                "*TestModuleCustom*::test_eager_vs_compile",
                                "*TestModuleCustom*::test_layout_stride",
                            ],
                            "mode": "xfail",
                            "tags": [f"model__{model_name}", "custom_tests"],
                            # Same AOTAutograd/no_grad issue as test_forward
                            # above: these custom tests also build modules
                            # with requires_grad=True parameters and compile
                            # them for Spyre.
                            "no_grad": True,
                            "edits": {"modules": {"include": module_entries}},
                        }
                    ],
                },
            ],
            "global": {
                "supported_dtypes": [
                    {"name": "float16", "precision": {"atol": 0.005, "rtol": 0.005}},
                    {"name": "float32", "precision": {"atol": 0.001, "rtol": 0.001}},
                    {"name": "bfloat16", "precision": {"atol": 0.005, "rtol": 0.005}},
                ],
                "input_config": {"seed": 123},
            },
        }
    }

    # Generate YAML string with header comments and consistent 2-space indentation
    header = f"""# Auto-generated unified test configuration for {model_name}
# Generated by auto_generate_module_config.py
# Format compatible with PyTorch's test_modules.py (using edits.modules.include)

"""

    # Use custom Dumper with 2-space indentation for consistency
    yaml_str = header + yaml.dump(
        config,
        Dumper=PrettyDumper,
        default_flow_style=False,
        sort_keys=False,
        indent=2,
        width=float("inf"),  # Prevent line wrapping
    )
    return yaml_str


def run_prefill(model, inputs) -> Any:
    """Run a prefill forward pass under ``torch.no_grad()``.

    Wraps a single ``model(**inputs, use_cache=True)`` call so the same
    error-handling can be reused anywhere a prefill forward is needed.

    Args:
        model: The model to invoke.
        inputs: Mapping of forward kwargs (e.g. tokenizer output).

    Returns:
        The model outputs, or ``None`` if the forward pass raised.
    """
    logger.info(f"  Input shape: {inputs['input_ids'].shape}")
    try:
        with torch.no_grad():
            return model(**inputs, use_cache=True)
    except Exception as e:
        logger.exception(f"  ERROR during prefill: {e}")
        return None


def _build_decode_inputs(inputs, past_key_values) -> Dict[str, Any]:
    """Build the forward kwargs for a single decode step from prefill state."""
    batch_size = inputs["input_ids"].shape[0]
    # Single new token for decode
    next_token = torch.zeros((batch_size, 1), dtype=torch.long)
    return {
        "input_ids": next_token,  # Shape: [B, 1]
        "attention_mask": torch.cat(
            [
                inputs["attention_mask"],
                torch.ones((batch_size, 1), dtype=torch.long),
            ],
            dim=1,
        ),
        "past_key_values": past_key_values,  # Use cached KV
        "use_cache": True,
    }


def run_decode(model, inputs, prefill_outputs) -> Any:
    """Run a single decode forward pass using the KV cache from prefill.

    No-op (returns ``None``) when ``prefill_outputs`` carries no usable KV
    cache. Wraps the ``model(**decode_inputs)`` call with the same
    error-handling so it can be reused wherever a decode forward is needed.

    Args:
        model: The model to invoke.
        inputs: The original prefill forward kwargs (for shapes / masks).
        prefill_outputs: The outputs returned by :func:`run_prefill`.

    Returns:
        The decode outputs, or ``None`` if skipped or the forward raised.
    """
    if (
        prefill_outputs is None
        or not hasattr(prefill_outputs, "past_key_values")
        or prefill_outputs.past_key_values is None
    ):
        logger.info("\n  Skipping decode pass - no KV cache available")
        return None

    decode_inputs = _build_decode_inputs(inputs, prefill_outputs.past_key_values)
    logger.info(f"Decode input_ids shape: {decode_inputs['input_ids'].shape}")
    logger.info(f"Decode attention_mask shape: {decode_inputs['attention_mask'].shape}")
    logger.info(
        f"Decode past_key_values layers: {len(decode_inputs['past_key_values'])}"
    )
    try:
        with torch.no_grad():
            decode_outputs = model(**decode_inputs)
        logger.info(
            f"Decode complete. Output shape: "
            f"{decode_outputs.logits.shape if hasattr(decode_outputs, 'logits') else 'N/A'}"
        )
        return decode_outputs
    except Exception:
        logger.exception("ERROR during decode")
        return None


def register_capture_hooks(
    model,
    capture: "ModuleInfoCapture",
    excluded_types: frozenset = frozenset(),
) -> List[Any]:
    """Register the model-level + per-module capture hooks; return their handles.

    Split out of :func:`capture_module_invocations` so an alternative driver -- the
    Spyre loader path, which cannot call ``model(**inputs)`` -- reuses the exact
    same instrumentation instead of a divergent copy. The caller owns removal and
    must do it in a ``finally``.
    """
    all_custom_modules = get_all_custom_modules(model, excluded_types=excluded_types)
    logger.info(f"Found {len(all_custom_modules)} custom module instances")

    # This hook sets context that module-level hooks will read
    model_hook = capture.create_model_hook()
    handles = [model.register_forward_pre_hook(model_hook, with_kwargs=True)]

    # Register hooks on ALL custom module instances (not just unique types)
    for module_name, module_type, module_instance in all_custom_modules:
        hook = capture.create_hook(module_name, module_type, module_instance)
        handles.append(
            module_instance.register_forward_pre_hook(hook, with_kwargs=True)
        )
    return handles


def capture_module_invocations(model, capture: ModuleInfoCapture, inputs) -> None:
    """Register capture hooks, run prefill + decode, then remove the hooks.

    This drives the model through both execution modes so that
    ``capture`` observes every unique module invocation pattern. Hooks are
    always removed, even if a forward pass raises.

    Args:
        model: The model to instrument and run.
        capture: The :class:`ModuleInfoCapture` to populate.
        inputs: Forward kwargs for the prefill pass (e.g. tokenizer output).
    """
    handles = register_capture_hooks(model, capture)
    try:
        prefill_outputs = run_prefill(model, inputs)
        run_decode(model, inputs, prefill_outputs)
    finally:
        # Remove hooks even if a forward pass raised
        for handle in handles:
            handle.remove()


def load_model_only(
    model_path: str,
    model_cls=AutoModel,
    **from_pretrained_kwargs: Any,
):
    """Load an eval-mode model (no tokenizer).

    Split out from :func:`load_model_and_tokenizer` so callers whose tokenizer
    is not an ``AutoTokenizer`` (e.g. ``mistral_common``'s ``MistralTokenizer``,
    or a VLM processor) can still reuse the model-loading path.

    Args:
        model_path: HuggingFace model path or local directory.
        model_cls: The class to load with. Defaults to :class:`AutoModel`;
            pass :class:`AutoModelForCausalLM` for causal LMs whose bare
            backbone lacks ``past_key_values`` / logits (e.g. ``gpt_oss``), or
            an explicit architecture class such as
            ``Mistral3ForConditionalGeneration`` for VLMs.
        **from_pretrained_kwargs: Extra kwargs forwarded to
            ``from_pretrained`` (e.g. ``torch_dtype``, ``device_map``,
            ``quantization_config``, ``trust_remote_code``). ``torch_dtype``
            defaults to :data:`DEFAULT_FLOAT_DTYPE` (bfloat16, the dtype Spyre
            runs in) rather than ``from_pretrained``'s float32; pass it
            explicitly to override.

    Returns:
        The loaded, ``.eval()``-mode model.
    """
    # Capture in bfloat16 by default so the recorded floating-point tensors match
    # the dtype Spyre executes in. Callers may still override torch_dtype.
    from_pretrained_kwargs.setdefault("torch_dtype", DEFAULT_FLOAT_DTYPE)
    logger.info(f"Loading model: {model_path} via {model_cls.__name__}")
    return model_cls.from_pretrained(model_path, **from_pretrained_kwargs).eval()


def load_model_and_tokenizer(
    model_path: str,
    model_cls=AutoModel,
    **from_pretrained_kwargs: Any,
):
    """Load an eval-mode model and its ``AutoTokenizer``, fixing a missing pad token.

    Convenience wrapper around :func:`load_model_only` for the common case
    where the tokenizer is a standard HF ``AutoTokenizer``. See
    :func:`load_model_only` for the argument semantics.

    Returns:
        ``(model, tokenizer)``.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Fix missing pad_token for Mistral tokenizers
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model_only(model_path, model_cls=model_cls, **from_pretrained_kwargs)
    return model, tokenizer


def build_dummy_inputs(tokenizer, seq_len: int) -> Dict[str, Any]:
    """Tokenize placeholder text padded/truncated to ``seq_len``."""
    # Generate enough text to reach desired seq_len
    text = "This is a test input for capturing module information. " * (
        seq_len // 10 + 1
    )
    return tokenizer(
        text,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )


def load_spyre_model_and_tokenizer(
    model_path: str,
    dtype: torch.dtype = torch.float16,
    device: str = SPYRE_DEVICE,
):
    """Load via ``AutoSpyreModelForCausalLM``; return (model, tokenizer, module, snapshot).

    Four values, because the Spyre execution path needs more than the model:

    - ``module``: the resolved adapter module. Spyre runs
      ``module._run_forward(...)``, NOT ``model.forward``, so the driver needs it.
    - ``snapshot``: each layer's pre-Spyre HF attention class/config, recorded
      before ``prepare_for_spyre()`` discards that information (see
      :func:`_snapshot_hf_attention_classes`).

    ``dtype`` defaults to float16 to match ``AutoSpyreModelForCausalLM``, NOT to
    this module's bfloat16 :data:`DEFAULT_FLOAT_DTYPE`.

    ``device`` exists for off-pod dry runs: ``hf_common.DEVICE`` is a module
    constant pinned to "spyre", and the CPU test lane patches it in place. It must
    be patched BEFORE ``auto_spyre_model`` is imported, because
    ``move_model_to_spyre`` reads the module global at call time.
    """
    _ensure_hf_adapters_importable()

    if device != SPYRE_DEVICE:
        from hf_adapters import hf_common

        logger.info("Patching hf_common.DEVICE -> %s (off-pod dry run)", device)
        hf_common.DEVICE = device

    from hf_adapters.auto_spyre_model import (
        AutoSpyreModelForCausalLM,
        resolve_adapter_module,
    )
    from hf_adapters.hf_common import SpyreUnsupportedModelError

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve the adapter first, so an unsupported model fails before the (slow)
    # load + compile, and so the driver has _run_forward.
    adapter_module = resolve_adapter_module(model_path)
    if getattr(adapter_module, "_is_encoder_only", False):
        raise SpyreUnsupportedModelError(
            f"{model_path} is encoder-only; the Spyre loader path drives a "
            f"causal-LM generate() and cannot capture it. Use --loader hf."
        )

    # Snapshot the HF attention classes from a plain load first:
    # AutoSpyreModelForCausalLM.from_pretrained does load + prepare + move in one
    # call with no hook in between, and prepare_for_spyre() destroys the
    # provenance the Spyre attention entries need. Freed immediately to bound the
    # peak of holding two copies of the weights.
    probe = load_model_only(
        model_path, model_cls=AutoModelForCausalLM, torch_dtype=dtype
    )
    snapshot = _snapshot_hf_attention_classes(probe)
    del probe

    logger.info(
        "Loading %s via AutoSpyreModelForCausalLM (dtype=%s)", model_path, dtype
    )
    model = AutoSpyreModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    return model, tokenizer, adapter_module, snapshot


def run_spyre_capture_forward(
    model,
    tokenizer,
    adapter_module,
    seq_len: int,
    max_new_tokens: int = 3,
):
    """Drive the Spyre execution path so the capture hooks see every forward shape.

    Delegates to ``hf_common.generate`` rather than re-implementing the padded
    64-block loop: generate() already builds the block-padded input_ids, the KV
    cache list, the position_ids and the prefill/decode masks exactly as
    production does, so the captured shapes are the real ones.

    ``max_new_tokens=3`` is the minimum that reaches all THREE forward shapes
    generate() produces. ``tokens_in_block`` starts at ``BLOCK_SIZE - 1`` and
    advances ``(x + 1) % BLOCK_SIZE`` per token, so the loop runs::

        i=0  PREFILL    is_filling=False token_index=0  cache_position=0
        i=1  EXPANSION  is_filling=False token_index=0  cache_position=BLOCK_SIZE
        i=2  FILL       is_filling=True  token_index=1

    FILL matters disproportionately: ``is_filling`` selects a different branch in
    ``kv_cache_update`` (write one token at ``token_index`` vs. write the whole
    block), and torch.compile specializes on it, so FILL is a separately compiled
    binary -- and it is the shape almost every generated token actually uses.
    Stopping at 2 would leave that path untested.

    Going beyond 3 adds only more FILL invocations at higher ``token_index``
    values. Each is a distinct invocation signature (values, not just types, are
    compared) and therefore another compiled binary in the module test, so the
    default stops at the first one.

    ``seq_len`` is only a target: generate() block-pads the prompt to a multiple of
    BLOCK_SIZE, so the captured sequence length is the padded one. That is the
    correct thing to record -- it is what Spyre actually runs.

    Returns the generated text, or ``None`` if the forward raised.
    """
    ###from hf_adapters.hf_common import generate

    prompt = "This is a test input for capturing module information. " * (
        seq_len // 10 + 1
    )
    try:
        with torch.no_grad():
            """
            return generate(
                adapter_module._run_forward,
                model,
                tokenizer,
                [prompt],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            """
            return model.generate(
                tokenizer,
                [prompt],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                timing=True,
            )
    except Exception:
        logger.exception("ERROR during Spyre forward")
        return None


def generate_spyre_module_config(
    model_path: str,
    seq_len: int = 128,
    output: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
    device: str = SPYRE_DEVICE,
    max_new_tokens: int = 3,
    excluded_types: frozenset = SPYRE_EXCLUDED_MODULE_TYPES,
):
    """Capture a module-test YAML from an ``AutoSpyreModelForCausalLM`` model.

    The programmatic entry point for the Spyre loader path; :func:`main` is a thin
    CLI wrapper over it, so ``--loader spyre`` and a direct call share one
    implementation.

    Emitted entries name the Spyre wrapper as ``module_path`` with the HF module it
    wrapped as a nested module arg, and carry ``apply_device_layout: true`` so the
    module test allocates parameters with the adapter's device layout.

    NOTE: importing hf_adapters patches the RMSNorm class globally
    (``patch_rmsnorm``), so a single process must not also run the HF loader path.

    Returns:
        The written YAML path.
    """
    model, tokenizer, adapter_module, snapshot = load_spyre_model_and_tokenizer(
        model_path, dtype=dtype, device=device
    )

    capture = ModuleInfoCapture(spyre_attn_snapshot=snapshot, spyre_model=model)
    handles = register_capture_hooks(model, capture, excluded_types=excluded_types)
    try:
        run_spyre_capture_forward(
            model, tokenizer, adapter_module, seq_len, max_new_tokens
        )
    finally:
        for handle in handles:
            handle.remove()

    # Mark every captured entry so the emitted YAML requests the adapter's
    # device-side parameter layout (see _build_module_entry_dict). Set here rather
    # than per-hook because it is a property of the capture path, not of any one
    # module.
    for module_data in capture.module_data.values():
        module_data["apply_device_layout"] = True

    return write_module_config(capture, model_path, output, filename_suffix="_adapter")


def write_module_config(
    capture: ModuleInfoCapture,
    model_path: str,
    output: str = None,
    filename_suffix: str = "",
):
    """Generate the unified YAML config from captured modules and write it out."""
    # Extract model name from path (handle both local paths and HuggingFace paths)
    model_path_parts = model_path.rstrip("/").split("/")
    model_name = model_path_parts[
        -1
    ]  # e.g., "granite-3.3-8b-instruct" or "granite-3.0-2b-instruct"

    # For the YAML content, use underscores for the model_name field
    model_name_normalized = model_name.replace("-", "_").replace(".", "_")

    # Generate unified YAML config (new format)
    unified_yaml_content = generate_unified_yaml_config(
        capture.get_captured_modules(), model_name_normalized
    )

    # Determine output path
    if output:
        output_path = output
    else:
        # filename_suffix keeps the two loader paths from overwriting each other:
        # the HF path owns <model>_spyre.yaml, the Spyre-adapter path writes
        # <model>_spyre_adapter.yaml.
        output_path = (
            f"./tests/configs/module_tests/"
            f"{model_name_normalized}{filename_suffix}.yaml"
        )

    # Write unified YAML file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(unified_yaml_content)

    logger.info(f"\n✓ Generated unified configuration: {output_file}")

    # Print module summary
    captured_modules = capture.get_captured_modules()
    logger.info("\n  Module Summary:")
    logger.info(f"    Total modules captured: {len(captured_modules)}")
    for module_info in captured_modules:
        logger.info(f"      - {module_info['name']}")

    return output_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-generate module configuration YAML using forward hooks"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HuggingFace model path (e.g., ibm-granite/granite-3.3-8b-instruct)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Sequence length for forward pass (default: 128)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output YAML file path (default: ./tests/configs/<model>_spyre.yaml)",
    )
    parser.add_argument(
        "--loader",
        choices=["hf", "spyre"],
        default="hf",
        help="hf (default): AutoModel/AutoModelForCausalLM driven through a "
        "StaticCache prefill+decode. spyre: AutoSpyreModelForCausalLM (patched + "
        "compiled blocks) driven through hf_common.generate(). Mutually exclusive "
        "-- patch_rmsnorm rewrites the RMSNorm class globally, so one process "
        "does one loader.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default=None,
        help="Load dtype. Default: bfloat16 for --loader hf, float16 for "
        "--loader spyre (matching AutoSpyreModelForCausalLM). 'auto' consults the "
        "adapter registry's per-model dtype.",
    )
    parser.add_argument(
        "--device",
        default=SPYRE_DEVICE,
        help="--loader spyre only: patches hf_common.DEVICE. Use 'cpu' for an "
        "off-pod dry run (no torch_spyre required).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=3,
        help="--loader spyre only: decode steps to run (default 3 = PREFILL + "
        "EXPANSION + FILL, the minimum that exercises all three forward shapes; "
        "FILL is the is_filling=True path that nearly every generated token uses). "
        "Higher values only add more FILL invocations at successive token_index "
        "values, each a separately compiled binary in the module test.",
    )
    parser.add_argument(
        "--no_static_cache",
        action="store_true",
        help="--loader hf only. Disable the StaticCache used for the forward pass "
        "(default: enabled). "
        "When set, the model uses its default dynamic KV cache instead.",
    )
    parser.add_argument(
        "--max_cache_len",
        type=int,
        default=2048,
        help="--loader hf only. max_cache_len for the StaticCache (default: 2048). "
        "Ignored when --no_static_cache is set.",
    )
    return parser.parse_args()


def _resolve_dtype(name: Optional[str], model_path: str, loader: str) -> torch.dtype:
    """Resolve ``--dtype`` to a torch dtype, defaulting per loader.

    The two loaders have different natural defaults: the HF capture path records
    bfloat16 (:data:`DEFAULT_FLOAT_DTYPE`), while ``AutoSpyreModelForCausalLM``
    loads float16.
    """
    if name == "auto":
        _ensure_hf_adapters_importable()
        from hf_adapters.auto_spyre_model import torch_dtype_for_model_path

        return torch_dtype_for_model_path(model_path)
    if name is not None:
        return getattr(torch, name)
    return torch.float16 if loader == "spyre" else DEFAULT_FLOAT_DTYPE


def main():
    args = parse_args()
    dtype = _resolve_dtype(args.dtype, args.model_path, args.loader)

    if args.loader == "spyre":
        # These only shape the HF path's StaticCache; the Spyre path's KV caches
        # come from generate(). Warn rather than fail so a mixed command line is
        # visible instead of silently ignored.
        if args.no_static_cache:
            logger.warning("--no_static_cache is ignored with --loader spyre.")
        if args.max_cache_len != 2048:
            logger.warning("--max_cache_len is ignored with --loader spyre.")
        generate_spyre_module_config(
            args.model_path,
            seq_len=args.seq_len,
            output=args.output,
            dtype=dtype,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        return

    model, tokenizer = load_model_and_tokenizer(args.model_path, torch_dtype=dtype)
    inputs = build_dummy_inputs(tokenizer, args.seq_len)

    # Use a StaticCache by default; --no_static_cache falls back to the model's
    # default dynamic cache. The (empty) StaticCache is passed into the prefill
    # forward and reused for decode.
    if not args.no_static_cache:
        inputs["past_key_values"] = StaticCache(
            config=model.config, max_cache_len=args.max_cache_len
        )

    capture = ModuleInfoCapture()
    capture_module_invocations(model, capture, inputs)

    write_module_config(capture, args.model_path, args.output)


if __name__ == "__main__":
    main()
